"""Activation-space compilation probe for an out-of-core MoE.

The streamed model pays for expert *parameters*, but inference only observes
the much smaller function

    (hidden state, routed expert ids, router gates) -> MoE output.

This probe asks whether that function can be compiled into a resident,
router-conditioned surrogate.  It deliberately tests on held-out contiguous
tokens: fitting the captured activations is uninteresting; predicting the next
part of the sequence is the falsification gate.

The surrogate has four shared matrices and a tiny embedding per expert:

    y = Base(x) + Down(SwiGLU(x) * (1 + tanh(sum(g_e Emb[e]))))

At rank 128, quantizing one surrogate per MiniMax layer to 4-bit would take
about 0.33 GB for all 62 layers, versus 128.7 GB of expert weights.  This file
does not claim that the compression works -- it measures whether the held-out
error is low enough to justify building the full compiler.

Safety: only layers 0..L are built.  The default layer-1 probe materialises two
resident layers, not the full 128.7 GB model.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import psutil
from mlx.utils import tree_flatten

from neuron_tail_live import build_n_layers, capture


class ShadowMoE(nn.Module):
    """Small resident approximation conditioned by the target router."""

    def __init__(self, dims: int, rank: int, experts: int):
        super().__init__()
        self.base = nn.Linear(dims, dims, bias=False)
        self.gate = nn.Linear(dims, rank, bias=False)
        self.up = nn.Linear(dims, rank, bias=False)
        self.down = nn.Linear(rank, dims, bias=False)
        self.expert_code = nn.Embedding(experts, rank)

    def __call__(self, x, inds, gates):
        code = (self.expert_code(inds) * gates[..., None]).sum(axis=-2)
        hidden = nn.silu(self.gate(x)) * self.up(x)
        hidden = hidden * (1.0 + mx.tanh(code))
        return self.base(x) + self.down(hidden)


def parameter_count(model):
    return sum(x.size for _, x in tree_flatten(model.parameters()))


def relative_error(pred, target):
    p, y = pred.astype(mx.float32), target.astype(mx.float32)
    per_token = mx.linalg.norm(p - y, axis=-1) / (
        mx.linalg.norm(y, axis=-1) + 1e-20
    )
    aggregate = mx.linalg.norm(p - y) / (mx.linalg.norm(y) + 1e-20)
    cosine = mx.sum(p * y, axis=-1) / (
        mx.linalg.norm(p, axis=-1) * mx.linalg.norm(y, axis=-1) + 1e-20
    )
    ordered = mx.sort(per_token)
    p95 = ordered[min(int(0.95 * ordered.size), ordered.size - 1)]
    mx.eval(per_token, aggregate, cosine, p95)
    return {
        "relative": float(aggregate),
        "token_mean": float(mx.mean(per_token)),
        "token_p95": float(p95),
        "cosine": float(mx.mean(cosine)),
    }


def capture_dataset(snap: Path, layer: int, tokens: int, text_path: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snap))
    ids = mx.array(tok(text_path.read_text()[:100_000])["input_ids"][:tokens])
    if ids.size < tokens:
        raise ValueError(f"{text_path} has only {ids.size} tokens, need {tokens}")

    model, cfg = build_n_layers(snap, layer + 1)
    grabbed, block = capture(model, ids, layer)
    x = grabbed["x"][0].astype(mx.float32)
    inds = grabbed["inds"][0]
    gates = grabbed["gates"][0].astype(mx.float32)
    target = block(grabbed["x"])[0].astype(mx.float32)
    mx.eval(x, inds, gates, target)
    return x, inds, gates, target, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--split", choices=("contiguous", "random"), default="contiguous",
                    help="held-out suffix is the real generalization gate; random is a capacity control")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--text", default="eval/pride_prejudice.txt")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    if not 0.5 <= a.train_frac < 1.0:
        raise ValueError("--train-frac must be in [0.5, 1)")
    if a.tokens < 32 or a.rank < 1 or a.batch < 1:
        raise ValueError("need tokens >= 32, rank >= 1, and batch >= 1")

    need = 6 + 2.1 * (a.layer + 1)
    avail = psutil.virtual_memory().available / 1e9
    if avail <= need:
        raise MemoryError(
            f"only {avail:.1f} GB available; layer {a.layer} needs about {need:.1f}"
        )

    mx.random.seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.perf_counter()
    x, inds, gates, target, cfg = capture_dataset(
        Path(a.snap), a.layer, a.tokens, Path(a.text)
    )
    capture_s = time.perf_counter() - t0

    cut = int(a.tokens * a.train_frac)
    if a.split == "contiguous":
        train = (x[:cut], inds[:cut], gates[:cut], target[:cut])
        test = (x[cut:], inds[cut:], gates[cut:], target[cut:])
    else:
        order = np.random.permutation(a.tokens)
        train_idx = mx.array(order[:cut])
        test_idx = mx.array(order[cut:])
        train = tuple(a_[train_idx] for a_ in (x, inds, gates, target))
        test = tuple(a_[test_idx] for a_ in (x, inds, gates, target))
    dims = int(x.shape[-1])
    experts = int(cfg["num_local_experts"])
    shadow = ShadowMoE(dims, a.rank, experts)
    optimizer = optim.AdamW(a.lr, weight_decay=1e-4)

    def loss_fn(model, xb, ib, gb, yb):
        pred = model(xb, ib, gb).astype(mx.float32)
        return mx.mean((pred - yb) ** 2) / (mx.mean(yb ** 2) + 1e-20)

    loss_and_grad = nn.value_and_grad(shadow, loss_fn)
    order = np.arange(cut)
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        pick = np.random.choice(order, size=min(a.batch, cut), replace=False)
        pick = mx.array(pick)
        loss, grads = loss_and_grad(
            shadow,
            train[0][pick],
            train[1][pick],
            train[2][pick],
            train[3][pick],
        )
        optimizer.update(shadow, grads)
        mx.eval(shadow.parameters(), optimizer.state, loss)
        if step == 1 or step % 50 == 0 or step == a.steps:
            print(f"step {step:>4}/{a.steps}: normalized MSE {float(loss):.5f}",
                  flush=True)
    train_s = time.perf_counter() - t0

    shadow.eval()
    train_pred = shadow(*train[:3])
    test_pred = shadow(*test[:3])
    mx.eval(train_pred, test_pred)
    tr = relative_error(train_pred, train[3])
    te = relative_error(test_pred, test[3])

    params = parameter_count(shadow)
    total_layers = json.loads((Path(a.snap) / "config.json").read_text())["num_hidden_layers"]
    all_layers_q4_gb = params * total_layers * 0.5 / 1e9
    result = {
        "layer": a.layer,
        "tokens": a.tokens,
        "train_tokens": cut,
        "test_tokens": a.tokens - cut,
        "split": a.split,
        "rank": a.rank,
        "steps": a.steps,
        "parameters_per_layer": params,
        "estimated_all_layers_q4_gb": all_layers_q4_gb,
        "capture_seconds": capture_s,
        "train_seconds": train_s,
        "train": tr,
        "test": te,
    }
    out = Path("results") / (
        f"shadow_moe_L{a.layer}_r{a.rank}_n{a.tokens}.json"
    )
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"\nparameters/layer: {params/1e6:.2f} M")
    print(f"estimated 62-layer q4 footprint: {all_layers_q4_gb:.3f} GB")
    print(f"capture {capture_s:.1f}s, train {train_s:.1f}s")
    print(f"train relative {tr['relative']:.4f}, cosine {tr['cosine']:.4f}")
    print(f"test  relative {te['relative']:.4f}, cosine {te['cosine']:.4f}, "
          f"p95 {te['token_p95']:.4f}")
    print(f"wrote {out}")


def _self_check():
    mx.random.seed(0)
    model = ShadowMoE(16, 4, 8)
    x = mx.random.normal((3, 16))
    inds = mx.array([[0, 1], [2, 3], [4, 5]])
    gates = mx.full((3, 2), 0.5)
    y = model(x, inds, gates)
    mx.eval(y)
    assert y.shape == x.shape
    assert parameter_count(model) == 16 * 16 + 3 * 16 * 4 + 8 * 4
    same = relative_error(y, y)
    assert same["relative"] == 0.0
    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
