from chat import resolve_snapshot


def test_resolve_snapshot_autodetects_the_only_local_revision(tmp_path):
    model_root = tmp_path / "models--mlx-community--MiniMax-M2.5-4bit"
    snapshot = model_root / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    assert resolve_snapshot(None, model_root) == snapshot
