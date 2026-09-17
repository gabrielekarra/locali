import json

import numpy as np

_EPS = 1e-12


# ---------- metrics ----------


def _bin_indices(confidences, bins):
    # equal-width bins over [0, 1]; floor-based so confidence==1.0 lands in the last bin.
    idx = np.floor(confidences * bins).astype(int)
    return np.clip(idx, 0, bins - 1)


def ece(probs, labels, bins=15):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    n = probs.shape[0]
    if n == 0:
        return 0.0
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    idx = _bin_indices(confidence, bins)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / n) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def mce(probs, labels, bins=15):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    n = probs.shape[0]
    if n == 0:
        return 0.0
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    idx = _bin_indices(confidence, bins)
    worst = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        worst = max(worst, abs(confidence[mask].mean() - correct[mask].mean()))
    return float(worst)


def brier(probs, labels):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    n, k = probs.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs, labels, eps=_EPS):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    n = probs.shape[0]
    p_true = np.clip(probs[np.arange(n), labels], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def separation(probs, labels):
    # mean(confidence | correct) - mean(confidence | incorrect): the signal an
    # escalation cascade needs. Unlike ece/mce/brier/nll this is NaN, not 0,
    # when one of the two groups is empty (all-correct or all-wrong batch) -
    # the quantity is genuinely undefined there, not zero.
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    if not correct.any() or correct.all():
        return float("nan")
    return float(confidence[correct].mean() - confidence[~correct].mean())


def reliability_bins(probs, labels, bins=15):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    n = probs.shape[0]
    edges = np.linspace(0.0, 1.0, bins + 1)
    if n == 0:
        idx = np.array([], dtype=int)
        confidence = np.array([])
        correct = np.array([])
    else:
        confidence = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(np.float64)
        idx = _bin_indices(confidence, bins)
    out = []
    for b in range(bins):
        mask = idx == b
        count = int(mask.sum())
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": count,
                "confidence": float(confidence[mask].mean()) if count else None,
                "accuracy": float(correct[mask].mean()) if count else None,
            }
        )
    return out


# ---------- shared math helpers ----------


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _to_logits(probs, eps=_EPS):
    return np.log(np.clip(probs, eps, 1.0))


def _golden_section_minimize(f, lo, hi, tol=1e-6, max_iter=200):
    # f assumed unimodal (convex) on [lo, hi].
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _pava(values, weights):
    # Pool-adjacent-violators via a stack of (weighted_sum, weight, count) blocks.
    # Note: merging must build the new block in locals first, not via a[-2] += a.pop(),
    # since pop() shrinks the list before the -2 index is resolved and hits the wrong slot.
    val_sum = []
    wsum = []
    count = []
    for v, w in zip(values, weights):
        cur_sum, cur_w, cur_c = v * w, w, 1
        while val_sum and (val_sum[-1] / wsum[-1]) > (cur_sum / cur_w):
            cur_sum += val_sum.pop()
            cur_w += wsum.pop()
            cur_c += count.pop()
        val_sum.append(cur_sum)
        wsum.append(cur_w)
        count.append(cur_c)
    fitted = np.empty(sum(count), dtype=np.float64)
    pos = 0
    for s, w, c in zip(val_sum, wsum, count):
        fitted[pos:pos + c] = s / w
        pos += c
    return fitted


# ---------- calibrators ----------

_REGISTRY = {}


class Calibrator:
    kind = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.kind is not None:
            _REGISTRY[cls.kind] = cls

    def fit(self, probs, labels):
        raise NotImplementedError

    def transform(self, probs):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError

    def save(self, path):
        data = self.to_dict()
        data["kind"] = self.kind
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return _REGISTRY[data["kind"]].from_dict(data)


class Identity(Calibrator):
    kind = "identity"

    def fit(self, probs, labels):
        return self

    def transform(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        return probs / probs.sum(axis=1, keepdims=True)

    def to_dict(self):
        return {}

    @classmethod
    def from_dict(cls, data):
        return cls()


class TemperatureScaling(Calibrator):
    kind = "temperature_scaling"

    def __init__(self, T=1.0):
        self.T = float(T)

    def fit(self, probs, labels):
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels)
        logits = _to_logits(probs)

        def loss(log_t):
            scaled = logits / np.exp(log_t)
            return nll(_softmax(scaled), labels)

        # coarse grid to bracket the minimum, then a golden-section local refine.
        grid = np.linspace(np.log(1e-3), np.log(1e3), 61)
        losses = [loss(g) for g in grid]
        best = grid[int(np.argmin(losses))]
        step = grid[1] - grid[0]
        log_t = _golden_section_minimize(loss, best - step, best + step)
        self.T = float(np.exp(log_t))
        return self

    def transform(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        logits = _to_logits(probs)
        return _softmax(logits / self.T)

    def to_dict(self):
        return {"T": self.T}

    @classmethod
    def from_dict(cls, data):
        return cls(T=data["T"])


class VectorScaling(Calibrator):
    kind = "vector_scaling"

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

    def fit(self, probs, labels, lr=0.05, max_iter=5000, tol=1e-10, check_every=20):
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels)
        n, k = probs.shape
        x = _to_logits(probs)
        onehot = np.zeros((n, k))
        onehot[np.arange(n), labels] = 1.0

        a = np.ones(k)
        b = np.zeros(k)
        m_a = np.zeros(k)
        v_a = np.zeros(k)
        m_b = np.zeros(k)
        v_b = np.zeros(k)
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        prev_loss = np.inf

        for t in range(1, max_iter + 1):
            z = a * x + b
            q = _softmax(z)
            grad_z = (q - onehot) / n
            grad_a = np.einsum("ik,ik->k", grad_z, x)
            grad_b = grad_z.sum(axis=0)

            m_a = beta1 * m_a + (1 - beta1) * grad_a
            v_a = beta2 * v_a + (1 - beta2) * grad_a ** 2
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b ** 2

            mhat_a = m_a / (1 - beta1 ** t)
            vhat_a = v_a / (1 - beta2 ** t)
            mhat_b = m_b / (1 - beta1 ** t)
            vhat_b = v_b / (1 - beta2 ** t)

            a -= lr * mhat_a / (np.sqrt(vhat_a) + adam_eps)
            b -= lr * mhat_b / (np.sqrt(vhat_b) + adam_eps)

            if t % check_every == 0:
                loss = nll(q, labels)
                if abs(prev_loss - loss) < tol:
                    break
                prev_loss = loss

        self.a = a
        self.b = b
        return self

    def transform(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        x = _to_logits(probs)
        return _softmax(self.a * x + self.b)

    def to_dict(self):
        return {"a": self.a.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(
            a=np.array(data["a"], dtype=np.float64),
            b=np.array(data["b"], dtype=np.float64),
        )


class IsotonicBinary(Calibrator):
    kind = "isotonic_binary"

    def __init__(self, x_knots=None, y_knots=None):
        self.x_knots = x_knots
        self.y_knots = y_knots

    def fit(self, probs, labels):
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels)
        confidence = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(np.float64)

        order = np.argsort(confidence, kind="mergesort")
        x_sorted = confidence[order]
        y_sorted = correct[order]
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        sums = np.bincount(inverse, weights=y_sorted)
        means = sums / counts

        self.x_knots = unique_x
        self.y_knots = _pava(means, counts)
        return self

    def transform(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum(axis=1, keepdims=True)
        n, k = probs.shape
        top_idx = probs.argmax(axis=1)
        rows = np.arange(n)
        top_conf = probs[rows, top_idx]
        calibrated = np.clip(np.interp(top_conf, self.x_knots, self.y_knots), 0.0, 1.0)

        out = probs.copy()
        if k == 1:
            out[:, 0] = 1.0
            return out

        others_sum = 1.0 - top_conf
        remaining = 1.0 - calibrated
        out[rows, top_idx] = 0.0
        degenerate = others_sum <= _EPS
        safe_others = np.where(degenerate, 1.0, others_sum)
        out *= (remaining / safe_others)[:, None]
        if degenerate.any():
            out[degenerate] = (remaining[degenerate] / (k - 1))[:, None]
            out[rows[degenerate], top_idx[degenerate]] = 0.0
        out[rows, top_idx] = calibrated
        return out

    def to_dict(self):
        return {"x_knots": self.x_knots.tolist(), "y_knots": self.y_knots.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(
            x_knots=np.array(data["x_knots"], dtype=np.float64),
            y_knots=np.array(data["y_knots"], dtype=np.float64),
        )
