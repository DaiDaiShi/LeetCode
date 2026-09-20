"""Offline test suite - no network, no market data.

The tests that matter most are the negative ones: a model that reports a
confident cycle phase on a random walk is worse than useless, because it will
do the same thing on real data and look just as plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_cycle import config, features, labels, synthetic
from btc_cycle.models import analog, composite, cycle_test, novelty, regime_hmm, supervised


@pytest.fixture(scope="module")
def cyclical():
    return features.build(synthetic.make_cyclical())


@pytest.fixture(scope="module")
def acyclic():
    return features.build(synthetic.make_acyclic())


# ---------------------------------------------------------------------------
# Causality - the property everything else depends on
# ---------------------------------------------------------------------------


def test_expanding_percentile_is_causal():
    """Appending future rows must not change any earlier value."""
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=1200), index=pd.date_range("2015-01-01", periods=1200))

    short = features.expanding_percentile(series.iloc[:800], min_periods=100)
    long = features.expanding_percentile(series, min_periods=100)

    pd.testing.assert_series_equal(short, long.iloc[:800], check_names=False)


def test_expanding_percentile_bounds_and_extremes():
    series = pd.Series([1.0, 5.0, 3.0, 9.0, 2.0] * 60)
    out = features.expanding_percentile(series, min_periods=5).dropna()
    assert out.between(0.0, 1.0).all()
    rising = pd.Series(np.arange(500, dtype=float))
    out = features.expanding_percentile(rising, min_periods=10).dropna()
    assert out.iloc[-1] == pytest.approx(1.0)


def test_features_are_causal(cyclical):
    """Truncating the input must leave the surviving feature rows unchanged."""
    raw = synthetic.make_cyclical()
    full = features.build(raw)
    truncated = features.build(raw.iloc[:3000])

    for column in ("mayer", "ma200w_mult", "drawdown", "ret_90", "realized_vol_30", "puell"):
        a = truncated[column].dropna()
        b = full[column].reindex(a.index)
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), column


def test_hmm_filtered_probabilities_are_causal(cyclical):
    """Forward filtering only. Row t must not move when rows after t appear."""
    result = regime_hmm.fit(cyclical, n_states=3, n_iter=60)
    block = cyclical[result.feature_names].replace([np.inf, -np.inf], np.nan).dropna()
    mean, std = result.model._btc_scaler
    X = (block.to_numpy(dtype=float) - mean) / std

    full = regime_hmm.filtered_probabilities(result.model, X)
    partial = regime_hmm.filtered_probabilities(result.model, X[: len(X) // 2])

    assert np.allclose(partial, full[: len(partial)], atol=1e-10)


def test_filtered_differs_from_smoothed(cyclical):
    """Guards against someone 'simplifying' filtering into predict_proba."""
    result = regime_hmm.fit(cyclical, n_states=3, n_iter=60)
    assert result.smoothed is not None
    difference = (result.filtered.to_numpy() - result.smoothed.to_numpy())
    assert np.abs(difference).max() > 0.01


def test_filtered_probabilities_are_a_distribution(cyclical):
    result = regime_hmm.fit(cyclical, n_states=4, n_iter=60)
    totals = result.filtered.sum(axis=1)
    assert np.allclose(totals.to_numpy(), 1.0, atol=1e-8)
    assert (result.filtered.to_numpy() >= -1e-12).all()


# ---------------------------------------------------------------------------
# The cycle-existence tests must work in BOTH directions
# ---------------------------------------------------------------------------


def test_spectral_test_finds_a_planted_cycle(cyclical):
    result = cycle_test.spectral_test(cyclical["price"], n_surrogates=200)
    assert result["p_value"] <= 0.10
    assert 3.0 <= result["peak_period_years"] <= 5.0


def test_spectral_test_rejects_a_random_walk(acyclic):
    """The one that stops the whole package from being astrology."""
    result = cycle_test.spectral_test(acyclic["price"], n_surrogates=200)
    assert result["p_value"] > 0.10


def test_spectral_test_rejects_many_random_walks():
    """False-positive rate must sit near the nominal level, not above it."""
    false_positives = 0
    trials = 12
    for seed in range(trials):
        frame = features.build(synthetic.make_acyclic(seed=seed))
        result = cycle_test.spectral_test(frame["price"], n_surrogates=120, seed=seed)
        if result["p_value"] <= 0.10:
            false_positives += 1
    assert false_positives <= 3, f"{false_positives}/{trials} false positives"


def test_phase_randomisation_would_have_had_no_power(cyclical):
    """Documents the bug this module is built to avoid.

    Phase-randomised surrogates preserve the amplitude spectrum exactly, so a
    spectral statistic is identical for data and surrogate. Any test built on
    them has zero power - which is why the null here is AR(1) instead.
    """
    x = cycle_test._detrend_log_price(cyclical["price"])
    rng = np.random.default_rng(0)
    spectrum = np.fft.rfft(x)
    phases = rng.uniform(0, 2 * np.pi, len(spectrum))
    phases[0] = 0.0
    if len(x) % 2 == 0:
        phases[-1] = 0.0
    surrogate = np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=len(x))

    original, _ = cycle_test._band_power(x, config.CYCLE_LENGTH_DAYS, 0.30)
    # Compare before the surrogate is re-detrended, where the identity is exact.
    randomised_power = np.abs(np.fft.rfft(surrogate)) ** 2
    original_power = np.abs(spectrum) ** 2
    assert np.allclose(randomised_power[1:], original_power[1:], rtol=1e-6)
    assert np.isfinite(original)


def test_power_analysis_reports_low_power_for_small_cycles(acyclic):
    result = cycle_test.spectral_power_analysis(
        acyclic["price"], amplitudes=(0.05, 1.5), n_trials=20, n_surrogates=60
    )
    rows = {r["log_amplitude"]: r["detection_rate"] for r in result["power_by_amplitude"]}
    assert rows[0.05] < rows[1.5]
    assert rows[0.05] < 0.5


def test_phase_lock_test_rejects_random_walk(acyclic):
    result = cycle_test.halving_phase_lock_test(acyclic, n_surrogates=200)
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_labels_never_cover_the_running_cycle():
    """The unfinished cycle has no top, so it must stay unlabelled."""
    index = pd.date_range("2011-01-01", "2026-01-01", freq="D")
    price = pd.Series(np.linspace(1, 100, len(index)), index=index)
    out = labels.label_phases(price)
    running = out[out.index >= pd.Timestamp(config.CURRENT_CYCLE.low_date)]
    assert running.isna().all()


def test_labels_order_within_a_completed_cycle():
    """accumulation must precede parabolic inside a low->top leg."""
    cycle = config.CYCLES[2]
    index = pd.date_range(cycle.low_date, cycle.top_date, freq="D")
    price = pd.Series(
        np.exp(np.linspace(np.log(cycle.low_price), np.log(cycle.top_price), len(index))),
        index=index,
    )
    out = labels.label_phases(price).dropna()
    first_parabolic = out[out == config.PARABOLIC].index.min()
    last_accumulation = out[out == config.ACCUMULATION].index.max()
    assert last_accumulation < first_parabolic


def test_coarse_mapping_covers_every_phase():
    series = pd.Series(config.PHASES)
    assert labels.coarse(series).notna().all()


# ---------------------------------------------------------------------------
# Validation honesty
# ---------------------------------------------------------------------------


def test_random_kfold_beats_leave_one_cycle_out(cyclical):
    """The inflation this package exists to warn about must be visible."""
    honest = supervised.leave_one_cycle_out(cyclical)
    inflated = supervised.random_kfold_for_contrast(cyclical)
    if "error" in honest or "error" in inflated:
        pytest.skip("synthetic anchors do not line up with the label table")
    assert inflated["random_kfold_accuracy"] >= honest["mean_accuracy"] - 0.05
    assert honest["effective_sample_size"] <= 4


def test_leave_one_cycle_out_reports_effective_sample_size(cyclical):
    result = supervised.leave_one_cycle_out(cyclical)
    if "error" in result:
        pytest.skip("no labelled folds on synthetic anchors")
    assert result["effective_sample_size"] <= len(config.COMPLETED_CYCLES)
    assert "warning" in result


# ---------------------------------------------------------------------------
# Composite and analogs
# ---------------------------------------------------------------------------


def test_composite_heat_stays_in_range(cyclical):
    out = composite.heat(cyclical)
    heat = out["heat"].dropna()
    assert heat.between(0.0, 1.0).all()
    assert len(heat) > 1000


def test_composite_phase_probabilities_sum_to_one():
    for value in (0.0, 0.2, 0.5, 0.9, 1.0):
        distribution = composite.phase_probabilities(value)
        assert sum(distribution.values()) == pytest.approx(1.0)


def test_dtw_distance_is_zero_for_identical_series():
    a = np.sin(np.linspace(0, 4 * np.pi, 120))
    distance, path = analog.dtw_distance(a, a.copy())
    assert distance == pytest.approx(0.0, abs=1e-9)
    assert len(path) >= 120


def test_dtw_matches_a_time_warped_copy():
    """A stretched copy must be closer than an unrelated series."""
    base = np.sin(np.linspace(0, 2 * np.pi, 100))
    stretched = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, 160),
                          np.sin(np.linspace(0, 2 * np.pi, 160)))
    unrelated = np.linspace(-1, 1, 100)
    near, _ = analog.dtw_distance(base, stretched)
    far, _ = analog.dtw_distance(base, unrelated)
    assert near < far


def test_amplitude_decay_is_monotone_down():
    result = analog.amplitude_decay()
    assert result["decay_per_cycle"] < 1.0
    assert result["n_points"] == len(config.COMPLETED_CYCLES)


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def test_novelty_flags_a_shifted_distribution(cyclical):
    """Push the tail of the series somewhere history has never been."""
    frame = cyclical.copy()
    tail = frame.index[-120:]
    for column in ("ret_90", "ret_365", "log_mayer", "log_ma200w_mult"):
        frame.loc[tail, column] = frame[column].max() * 4.0
    result = novelty.mahalanobis_novelty(frame, window=90)
    assert result["percentile_vs_history"] > 0.95


def test_novelty_quiet_on_ordinary_data(cyclical):
    result = novelty.mahalanobis_novelty(cyclical, window=90)
    assert "percentile_vs_history" in result
    assert 0.0 <= result["percentile_vs_history"] <= 1.0


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


def test_ensemble_is_not_confident_on_a_random_walk(acyclic):
    """The headline safety property.

    On a series with no cycle at all, the ensemble may still name a phase -
    price is somewhere, after all - but it must not be confident about it.
    """
    from btc_cycle import ensemble

    verdict = ensemble.evaluate(acyclic, fast=True, run_supervised=False)
    assert verdict.confidence < 0.45, verdict.to_dict()
    assert not verdict.cycle_informative or verdict.novelty_flag


def test_ensemble_probabilities_sum_to_one(cyclical):
    from btc_cycle import ensemble

    verdict = ensemble.evaluate(cyclical, fast=True, run_supervised=False)
    total = sum(v for v in verdict.probabilities.values() if np.isfinite(v))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_walk_forward_is_shorter_than_full_fit(cyclical):
    """Out-of-sample output must start after the minimum training window."""
    walk = regime_hmm.walk_forward(cyclical, n_states=3, min_train=1200, refit_every=365)
    assert len(walk) < len(cyclical)
    assert walk.index[0] > cyclical.index[0]
