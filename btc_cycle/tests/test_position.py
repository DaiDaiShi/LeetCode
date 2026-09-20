"""Tests for the position layer.

The important ones are about refusal: a sizing rule that cannot say "flat" or
"no short" is just a leveraged opinion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_cycle import config, ensemble, features, position, synthetic


@pytest.fixture(scope="module")
def cyclical():
    return features.build(synthetic.make_cyclical())


@pytest.fixture(scope="module")
def acyclic():
    return features.build(synthetic.make_acyclic())


# ---------------------------------------------------------------------------
# Base rates
# ---------------------------------------------------------------------------


def test_unconditional_drift_reports_its_own_uncertainty(cyclical):
    result = position.unconditional_drift(cyclical["price"])
    assert result["annual_vol"] > 0
    assert result["drift_standard_error"] > 0
    # SE must be sigma / sqrt(years), not sigma / sqrt(days).
    expected = result["annual_vol"] / np.sqrt(result["years"])
    assert result["drift_standard_error"] == pytest.approx(expected, rel=1e-6)


def test_historical_drift_uses_only_anchors():
    result = position.historical_drift_estimates()
    assert np.isfinite(result["low_to_low_median"])
    kinds = {row["kind"] for row in result["spans"]}
    assert "low-to-low" in kinds
    # Top-anchored spans must read higher than low-to-low ones.
    low = [r["annual_log_drift"] for r in result["spans"] if r["kind"] == "low-to-low"]
    top = [r["annual_log_drift"] for r in result["spans"] if r["kind"].startswith("low-to-top")]
    assert max(top) > max(low)


def test_bear_legs_are_all_negative():
    result = position.historical_bear_legs()
    assert len(result["legs"]) == len(config.COMPLETED_CYCLES)
    for leg in result["legs"]:
        assert leg["annual_log_return"] < 0
        assert -1.0 < leg["drawdown"] < 0


# ---------------------------------------------------------------------------
# The short gate
# ---------------------------------------------------------------------------


def test_hurdle_threshold_tightens_with_less_data():
    """Fewer independent observations must demand a more extreme estimate."""
    table = position.hurdle_table(effective_n=(3, 12))
    rows = {r["effective_n"]: r for r in table["by_effective_n"]}
    assert rows[3]["required_point_estimate"] < rows[12]["required_point_estimate"]
    assert rows[3]["standard_error"] > rows[12]["standard_error"]


def test_hurdle_breakeven_accounts_for_funding():
    """Shorts receive funding, so the break-even sits above zero."""
    table = position.hurdle_table(funding_annual=0.10, annual_cost=0.02)
    assert table["breakeven_conditional_return"] == pytest.approx(0.08)
    cheap = position.hurdle_table(funding_annual=0.0, annual_cost=0.02)
    assert cheap["breakeven_conditional_return"] < table["breakeven_conditional_return"]


def test_short_refused_when_estimate_is_noisy():
    """A negative point estimate swamped by its standard error must not trade."""
    stats = {config.BEAR: {"annualised_mean": -0.20, "bootstrap_se": 0.40}}
    price = pd.Series(
        np.exp(np.linspace(0, 3, 2000)),
        index=pd.date_range("2016-01-01", periods=2000, freq="D"),
    )
    hurdle = position.short_hurdle(price, stats)
    row = hurdle["by_phase"][config.BEAR]
    assert row["point_estimate_clears"]          # -20% is below break-even
    assert not row["short_justified"]            # but +2 SE is not
    exposure, _, rationale, refused = position.target_exposure(
        {config.BEAR: 1.0}, stats, vol=0.5, confidence=0.9, hurdle=hurdle
    )
    assert refused
    assert exposure == 0.0
    assert any("short suppressed" in line for line in rationale)


def test_short_allowed_when_evidence_is_overwhelming():
    """The gate must not be unconditional - a real bear leg has to pass."""
    stats = {config.BEAR: {"annualised_mean": -1.50, "bootstrap_se": 0.20}}
    price = pd.Series(
        np.exp(np.linspace(0, 3, 2000)),
        index=pd.date_range("2016-01-01", periods=2000, freq="D"),
    )
    hurdle = position.short_hurdle(price, stats)
    assert hurdle["by_phase"][config.BEAR]["short_justified"]
    exposure, _, _, refused = position.target_exposure(
        {config.BEAR: 1.0}, stats, vol=0.6, confidence=0.9, hurdle=hurdle
    )
    assert not refused
    assert exposure < 0
    assert exposure >= position.MAX_SHORT


def test_short_is_capped_however_strong_the_signal():
    stats = {config.BEAR: {"annualised_mean": -5.0, "bootstrap_se": 0.05}}
    price = pd.Series(
        np.exp(np.linspace(0, 3, 2000)),
        index=pd.date_range("2016-01-01", periods=2000, freq="D"),
    )
    hurdle = position.short_hurdle(price, stats)
    exposure, _, _, _ = position.target_exposure(
        {config.BEAR: 1.0}, stats, vol=0.3, confidence=1.0, hurdle=hurdle
    )
    assert exposure == pytest.approx(position.MAX_SHORT)


def test_no_hurdle_means_no_short():
    stats = {config.BEAR: {"annualised_mean": -2.0, "bootstrap_se": 0.01}}
    exposure, _, _, refused = position.target_exposure(
        {config.BEAR: 1.0}, stats, vol=0.5, confidence=1.0, hurdle=None
    )
    assert refused and exposure == 0.0


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_confidence_haircut_actually_bites():
    """Regression: clipping before the haircut, so low confidence reduces size.

    Applying the haircut to an unclipped Kelly number lets an oversized raw
    value survive a 90% cut and still hit the cap.
    """
    stats = {config.EARLY_BULL: {"annualised_mean": 0.60, "bootstrap_se": 0.10}}
    high, _, _, _ = position.target_exposure(
        {config.EARLY_BULL: 1.0}, stats, vol=0.15, confidence=0.5
    )
    low, _, _, _ = position.target_exposure(
        {config.EARLY_BULL: 1.0}, stats, vol=0.15, confidence=0.05
    )
    assert high == pytest.approx(position.MAX_LONG)
    assert low < high / 2


def test_exposure_respects_limits():
    stats = {config.PARABOLIC: {"annualised_mean": 3.0, "bootstrap_se": 0.1}}
    exposure, _, _, _ = position.target_exposure(
        {config.PARABOLIC: 1.0}, stats, vol=0.2, confidence=1.0
    )
    assert position.MAX_SHORT <= exposure <= position.MAX_LONG


def test_flat_when_no_phase_has_statistics():
    exposure, expected, rationale, _ = position.target_exposure(
        {config.EARLY_BULL: 1.0}, {}, vol=0.5, confidence=1.0
    )
    assert exposure == 0.0
    assert rationale


def test_describe_covers_the_range():
    assert "LONG" in position.describe(0.9)
    assert "LONG" in position.describe(0.4)
    assert "FLAT" in position.describe(0.0)
    assert "SHORT" in position.describe(-0.2)


# ---------------------------------------------------------------------------
# Causality and the strategy comparison
# ---------------------------------------------------------------------------


def test_phase_forward_returns_are_out_of_sample(cyclical):
    result = position.phase_forward_returns(cyclical, horizon=90, n_boot=60)
    assert "by_phase" in result
    for phase, stats in result["by_phase"].items():
        assert stats["effective_n"] <= stats["n_days"]
        assert np.isfinite(stats["annualised_mean"])


def test_backtest_compares_against_buy_and_hold(cyclical):
    result = position.backtest(cyclical, min_train=1200, refit_every=365)
    assert result["strategy"]["years"] == result["buy_and_hold"]["years"]
    assert np.isfinite(result["strategy"]["sharpe"])
    assert np.isfinite(result["buy_and_hold"]["total_multiple"])
    assert 0.0 <= result["average_exposure"] <= position.MAX_LONG


def test_backtest_without_shorts_never_goes_negative(cyclical):
    result = position.backtest(cyclical, allow_short=False, min_train=1200, refit_every=365)
    assert result["average_exposure"] >= 0.0


def test_recommend_is_flat_or_small_on_a_random_walk(acyclic):
    """No cycle, no conviction - and certainly no leverage."""
    verdict = ensemble.evaluate(acyclic, fast=True, run_supervised=False)
    advice = position.recommend(acyclic, verdict)
    assert abs(advice.target_exposure) <= 0.5, advice.to_dict()
    assert position.MAX_SHORT <= advice.target_exposure <= position.MAX_LONG
