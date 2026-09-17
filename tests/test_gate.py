import numpy as np
import pytest

from gate import FrameGate, signature
from scene import Event, synthetic_sequence


def _flat(value, size=(64, 64)):
    return np.full(size, value, dtype=np.uint8)


def test_signature_is_normalised_and_gridded():
    sig = signature(_flat(128), grid=8)
    assert sig.shape == (8, 8)
    assert 0.0 <= sig.min() and sig.max() <= 1.0
    assert sig == pytest.approx(128 / 255, abs=1e-6)


def test_signature_is_resolution_independent():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    large = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)
    assert signature(small, grid=8) == pytest.approx(signature(large, grid=8), abs=1e-6)


def test_signature_rejects_a_frame_smaller_than_the_grid():
    with pytest.raises(ValueError):
        signature(_flat(10, size=(4, 4)), grid=16)


def test_first_frame_always_infers():
    g = FrameGate()
    v = g(_flat(100))
    assert v.infer and v.reason == "first"


def test_identical_frames_are_skipped():
    g = FrameGate(threshold=0.01, max_age=None)
    g(_flat(100))
    assert [g(_flat(100)).infer for _ in range(5)] == [False] * 5


def test_a_large_change_trips_the_gate():
    g = FrameGate(threshold=0.01, max_age=None)
    g(_flat(100))
    v = g(_flat(200))
    assert v.infer and v.reason == "changed"


def test_slow_drift_accumulates_against_a_fixed_reference():
    """The property the whole design rests on.

    Each step is far below the threshold, so a gate comparing consecutive
    frames would never fire. Against the last *inferred* frame the drift adds
    up and eventually trips.
    """
    g = FrameGate(threshold=0.05, max_age=None)
    g(_flat(100))
    step = 1  # 1/255 per frame, well under a 0.05 threshold
    fired = False
    for i in range(1, 60):
        v = g(_flat(100 + i * step))
        assert v.distance < 0.05 or v.infer
        if v.infer:
            fired = True
            assert v.reason == "changed"
            break
    assert fired, "drift never accumulated past the threshold"


def test_max_age_forces_inference_on_an_unchanging_stream():
    g = FrameGate(threshold=0.5, max_age=4)
    g(_flat(100))
    reasons = [g(_flat(100)).reason for _ in range(8)]
    assert "stale" in reasons
    assert reasons.count("stale") == 2  # frames 4 and 8


def test_skipped_frames_do_not_move_the_reference():
    g = FrameGate(threshold=0.05, max_age=None)
    g(_flat(100))
    g(_flat(105))  # skipped, must not become the reference
    assert g.check(_flat(105)).distance == pytest.approx(5 / 255, abs=1e-6)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        FrameGate(threshold=-0.1)
    with pytest.raises(ValueError):
        FrameGate(max_age=0)


def test_synthetic_sequence_has_labelled_events_and_drift():
    frames = list(synthetic_sequence(n_frames=60, events=(Event(20, 5),), seed=1))
    assert len(frames) == 60
    assert sum(is_event for _, is_event in frames) == 5
    first, last = signature(frames[0][0]), signature(frames[-1][0])
    assert np.abs(last - first).mean() > 0.01  # drift is present
    step = np.abs(signature(frames[1][0]) - first).mean()
    assert step < 0.005  # but invisible frame to frame


def test_gate_catches_every_event_at_the_documented_threshold():
    events = (Event(67, 8), Event(154, 5), Event(241, 12))
    frames = list(synthetic_sequence(n_frames=300, events=events))
    g = FrameGate(threshold=0.005, max_age=None)
    inferred = {i for i, (f, _) in enumerate(frames) if g(f).infer}
    for e in events:
        assert inferred & set(range(e.start, e.start + e.length)), f"missed {e}"
    assert len(inferred) / len(frames) < 0.15  # and still skips most frames
