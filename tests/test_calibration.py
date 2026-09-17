import numpy as np
import pytest

from calibration import (
    Calibrator,
    Identity,
    IsotonicBinary,
    TemperatureScaling,
    VectorScaling,
    _pava,
    brier,
    ece,
    mce,
    nll,
    reliability_bins,
    separation,
)


# ---------- hand-computed metric checks ----------
# probs/labels/expected values below are worked out by hand in the task notes, not
# derived by running this module; only pytest.approx tolerances are code-side.


def test_brier_hand_computed():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    # row0: (0.9-1)^2+(0.1-0)^2 = 0.02, row1: (0.2-0)^2+(0.8-1)^2 = 0.08, mean = 0.05
    assert brier(probs, labels) == pytest.approx(0.05, abs=1e-12)


def test_nll_hand_computed():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    # -(ln(0.9) + ln(0.8)) / 2, with ln(0.9)=-0.1053605156578264, ln(0.8)=-0.2231435513142101
    assert nll(probs, labels) == pytest.approx(0.16425203348601825, abs=1e-9)


def test_ece_mce_hand_computed():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    # both predictions correct; confidences 0.9 (bin 9) and 0.8 (bin 8) with bins=10
    # gaps: |0.9-1.0|=0.1, |0.8-1.0|=0.2, each bin weight 1/2 -> ece = 0.05+0.10 = 0.15
    assert ece(probs, labels, bins=10) == pytest.approx(0.15, abs=1e-12)
    assert mce(probs, labels, bins=10) == pytest.approx(0.2, abs=1e-12)


def test_ece_mce_hand_computed_with_wrong_prediction():
    probs = np.array([[0.6, 0.4]])
    labels = np.array([1])
    # predicted class 0, true class 1: wrong. confidence 0.6, accuracy 0 -> gap 0.6
    assert ece(probs, labels, bins=10) == pytest.approx(0.6, abs=1e-12)
    assert mce(probs, labels, bins=10) == pytest.approx(0.6, abs=1e-12)
    # brier: (0.6-0)^2+(0.4-1)^2 = 0.36+0.36 = 0.72
    assert brier(probs, labels) == pytest.approx(0.72, abs=1e-12)
    # nll: -ln(0.4) = 0.9162907318741551
    assert nll(probs, labels) == pytest.approx(0.9162907318741551, abs=1e-9)


def test_separation_hand_computed():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([0, 1, 1])
    # row0: pred 0, correct, conf 0.9. row1: pred 1, correct, conf 0.8.
    # row2: pred 0, label 1, wrong, conf 0.6.
    # mean(correct)=0.85, mean(wrong)=0.6, separation=0.25
    assert separation(probs, labels) == pytest.approx(0.25, abs=1e-12)


def test_separation_all_correct_is_nan():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    assert np.isnan(separation(probs, labels))


def test_separation_all_wrong_is_nan():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([1, 0])
    assert np.isnan(separation(probs, labels))


def test_separation_zero_when_no_signal():
    # correct and incorrect predictions share the same confidence -> no signal
    probs = np.array([[0.7, 0.3], [0.7, 0.3], [0.3, 0.7], [0.3, 0.7]])
    labels = np.array([0, 1, 1, 0])  # first pair correct, second pair wrong
    assert separation(probs, labels) == pytest.approx(0.0, abs=1e-12)


def test_reliability_bins_hand_computed():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    bins = reliability_bins(probs, labels, bins=10)
    assert len(bins) == 10
    bin8, bin9 = bins[8], bins[9]
    assert bin8["count"] == 1
    assert bin8["confidence"] == pytest.approx(0.8)
    assert bin8["accuracy"] == pytest.approx(1.0)
    assert bin9["count"] == 1
    assert bin9["confidence"] == pytest.approx(0.9)
    assert bin9["accuracy"] == pytest.approx(1.0)
    for i in range(8):
        assert bins[i]["count"] == 0
        assert bins[i]["confidence"] is None
        assert bins[i]["accuracy"] is None


# ---------- PAVA correctness ----------


def test_pava_textbook_example():
    # classic isotonic regression example: unconstrained [1,0,0,1] pools to [1/3,1/3,1/3,1]
    fitted = _pava(np.array([1.0, 0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0]))
    assert fitted == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0])


def test_pava_already_monotone_is_unchanged():
    fitted = _pava(np.array([0.1, 0.4, 0.4, 0.9]), np.array([1.0, 1.0, 1.0, 1.0]))
    assert fitted == pytest.approx([0.1, 0.4, 0.4, 0.9])


def test_pava_weighted_ties():
    # a heavier point should pull the pooled mean toward it
    fitted = _pava(np.array([1.0, 0.0]), np.array([1.0, 3.0]))
    # violation -> pool: mean = (1*1 + 0*3) / 4 = 0.25
    assert fitted == pytest.approx([0.25, 0.25])


def test_pava_monotone_output_on_random_input():
    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 1.0, size=50)
    weights = rng.uniform(0.5, 2.0, size=50)
    fitted = _pava(values, weights)
    assert np.all(np.diff(fitted) >= -1e-12)


# ---------- synthetic calibration fixtures ----------


def _calibration_fixture(gamma, seed=0, n=6000, k=3):
    rng = np.random.default_rng(seed)
    p_true = rng.dirichlet(alpha=[0.6] * k, size=n)
    labels = np.array([rng.choice(k, p=p_true[i]) for i in range(n)])
    sharpened = p_true ** gamma
    p_pred = sharpened / sharpened.sum(axis=1, keepdims=True)
    return p_pred, labels


def test_temperature_scaling_overconfident_data():
    probs, labels = _calibration_fixture(gamma=3.0)
    ece_before = ece(probs, labels)
    ts = TemperatureScaling().fit(probs, labels)
    calibrated = ts.transform(probs)
    ece_after = ece(calibrated, labels)
    assert ts.T > 1.0
    assert ece_after < ece_before * 0.5
    assert ece_after < 0.05


def test_temperature_scaling_underconfident_data():
    probs, labels = _calibration_fixture(gamma=0.3)
    ece_before = ece(probs, labels)
    ts = TemperatureScaling().fit(probs, labels)
    calibrated = ts.transform(probs)
    ece_after = ece(calibrated, labels)
    assert ts.T < 1.0
    assert ece_after < ece_before * 0.5
    assert ece_after < 0.05


def test_temperature_scaling_already_calibrated_data():
    probs, labels = _calibration_fixture(gamma=1.0)
    ece_before = ece(probs, labels)
    ts = TemperatureScaling().fit(probs, labels)
    calibrated = ts.transform(probs)
    ece_after = ece(calibrated, labels)
    assert 0.8 < ts.T < 1.25
    assert ece_after < ece_before + 0.02


def test_vector_scaling_improves_overconfident_ece():
    probs, labels = _calibration_fixture(gamma=3.0)
    ece_before = ece(probs, labels)
    vs = VectorScaling().fit(probs, labels)
    calibrated = vs.transform(probs)
    ece_after = ece(calibrated, labels)
    assert ece_after < ece_before * 0.5
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_vector_scaling_stable_on_already_calibrated_data():
    probs, labels = _calibration_fixture(gamma=1.0)
    ece_before = ece(probs, labels)
    vs = VectorScaling().fit(probs, labels)
    calibrated = vs.transform(probs)
    ece_after = ece(calibrated, labels)
    assert ece_after < ece_before + 0.02


def test_isotonic_binary_improves_overconfident_ece():
    probs, labels = _calibration_fixture(gamma=3.0)
    ece_before = ece(probs, labels)
    iso = IsotonicBinary().fit(probs, labels)
    calibrated = iso.transform(probs)
    ece_after = ece(calibrated, labels)
    assert ece_after < ece_before * 0.5
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.all(np.diff(iso.y_knots) >= -1e-12)


# ---------- round-trip save/load ----------


@pytest.mark.parametrize("cls", [Identity, TemperatureScaling, VectorScaling, IsotonicBinary])
def test_save_load_roundtrip(cls, tmp_path):
    rng = np.random.default_rng(5)
    probs = rng.dirichlet([1.0, 1.0, 1.0], size=80)
    labels = rng.integers(0, 3, size=80)
    calibrator = cls().fit(probs, labels)
    before = calibrator.transform(probs)

    path = tmp_path / f"{cls.__name__}.json"
    calibrator.save(path)
    loaded = Calibrator.load(path)
    assert isinstance(loaded, cls)
    after = loaded.transform(probs)
    assert np.array_equal(before, after)

    # dispatch also works when called on the concrete subclass, not just the base
    loaded_direct = cls.load(path)
    assert np.array_equal(before, loaded_direct.transform(probs))


# ---------- edge cases ----------


@pytest.mark.parametrize("cls", [Identity, TemperatureScaling, VectorScaling, IsotonicBinary])
def test_transform_rows_sum_to_one(cls):
    rng = np.random.default_rng(7)
    probs = rng.dirichlet([0.5, 0.5, 0.5, 0.5], size=300)
    labels = rng.integers(0, 4, size=300)
    calibrator = cls().fit(probs, labels)
    out = calibrator.transform(probs)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("cls", [Identity, TemperatureScaling, VectorScaling, IsotonicBinary])
def test_k_equals_2(cls):
    rng = np.random.default_rng(11)
    probs = rng.dirichlet([1.0, 1.0], size=100)
    labels = rng.integers(0, 2, size=100)
    calibrator = cls().fit(probs, labels)
    out = calibrator.transform(probs)
    assert out.shape == (100, 2)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("cls", [Identity, TemperatureScaling, VectorScaling, IsotonicBinary])
def test_n_equals_1(cls):
    probs = np.array([[0.7, 0.3]])
    labels = np.array([0])
    calibrator = cls().fit(probs, labels)
    out = calibrator.transform(probs)
    assert out.shape == (1, 2)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)
    assert not np.isnan(out).any()


@pytest.mark.parametrize("cls", [TemperatureScaling, VectorScaling, IsotonicBinary])
def test_class_never_appearing_in_labels(cls):
    rng = np.random.default_rng(13)
    probs = rng.dirichlet([1.0, 1.0, 1.0], size=200)
    labels = rng.integers(0, 2, size=200)  # class 2 never appears
    calibrator = cls().fit(probs, labels)
    out = calibrator.transform(probs)
    assert not np.isnan(out).any()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_ece_mce_reliability_bins_handle_empty_bins():
    rng = np.random.default_rng(17)
    # confidences clustered tightly, so most of 15 equal-width bins are empty
    probs = rng.dirichlet([50.0, 50.0], size=5)
    labels = rng.integers(0, 2, size=5)
    e = ece(probs, labels, bins=15)
    m = mce(probs, labels, bins=15)
    bins = reliability_bins(probs, labels, bins=15)
    assert np.isfinite(e)
    assert np.isfinite(m)
    assert len(bins) == 15
    empty = [b for b in bins if b["count"] == 0]
    assert len(empty) > 0
    for b in empty:
        assert b["confidence"] is None
        assert b["accuracy"] is None
    assert sum(b["count"] for b in bins) == 5


@pytest.mark.parametrize("cls", [TemperatureScaling, VectorScaling, IsotonicBinary])
def test_exact_zero_and_one_probabilities(cls):
    # last row: predicted probability for the true class is exactly 0.0
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 1, 0])
    assert np.isfinite(nll(probs, labels))
    assert np.isfinite(brier(probs, labels))
    calibrator = cls().fit(probs, labels)
    out = calibrator.transform(probs)
    assert not np.isnan(out).any()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_identity_is_a_no_op():
    probs = np.array([[0.2, 0.3, 0.5], [0.9, 0.05, 0.05]])
    labels = np.array([2, 0])
    calibrator = Identity().fit(probs, labels)
    out = calibrator.transform(probs)
    assert np.allclose(out, probs)
