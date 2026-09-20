"""Tests for the pattern edge tester.

The headline test is :func:`test_noise_profile_exposes_lookahead`. If it ever
fails, the audit has stopped separating hindsight from forecast and every
verdict this module produces is worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_cycle import patterns


def _random_walk(n=3000, drift=0.0009, vol=0.035, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))


@pytest.fixture(scope="module")
def walk():
    return _random_walk(seed=3)


# ---------------------------------------------------------------------------
# ZigZag primitive
# ---------------------------------------------------------------------------


def test_zigzag_alternates_highs_and_lows(walk):
    pivots = patterns.zigzag(walk, 0.15)
    assert len(pivots) > 5
    kinds = "".join(p.kind for p in pivots)
    assert "HH" not in kinds and "LL" not in kinds


def test_confirmation_always_follows_the_pivot(walk):
    """The lag is the latency every pivot pattern inherits; it is never zero."""
    pivots = patterns.zigzag(walk, 0.15)
    for pivot in pivots:
        assert pivot.confirm_index > pivot.index
        assert pivot.lag > 0


def test_confirmation_requires_the_full_retracement(walk):
    """A high is confirmed only once price is pct below it - by construction."""
    pct = 0.15
    pivots = patterns.zigzag(walk, pct)
    for pivot in pivots:
        confirming_price = walk[pivot.confirm_index]
        if pivot.kind == "H":
            assert confirming_price <= pivot.price * (1 - pct) + 1e-9
        else:
            assert confirming_price >= pivot.price * (1 + pct) - 1e-9


def test_zigzag_is_causal(walk):
    """Appending future bars must not move or relabel an earlier pivot."""
    short = patterns.zigzag(walk[:2000], 0.15)
    long = patterns.zigzag(walk, 0.15)
    # Every pivot confirmed before the truncation point must be identical.
    settled = [p for p in short if p.confirm_index < 1900]
    assert len(settled) > 3
    for pivot in settled:
        assert pivot in long


def test_tighter_threshold_finds_more_pivots(walk):
    assert len(patterns.zigzag(walk, 0.08)) > len(patterns.zigzag(walk, 0.25))


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def test_every_detector_signals_at_or_after_the_pattern(walk):
    """The API contract that makes look-ahead impossible to hide."""
    pivots = patterns.zigzag(walk, 0.15)
    for name, detector in patterns.DETECTORS.items():
        for match in detector(pivots, walk):
            assert match.signal_index >= match.pattern_end_index, name
            assert match.lag >= 0, name


def test_elliott_rules_reject_a_violating_sequence():
    """Hand-built: wave 4 overlaps wave 1, so rule 3 must reject it."""
    prices = [100, 150, 130, 200, 140, 260]     # 140 < 150 -> overlap
    pivots = [
        patterns.Pivot(i * 10, p, "L" if i % 2 == 0 else "H", i * 10 + 5)
        for i, p in enumerate(prices)
    ]
    assert patterns.elliott_impulse(pivots, np.array(prices, dtype=float)) == []


def test_elliott_rules_accept_a_clean_impulse():
    prices = [100, 150, 130, 220, 180, 260]     # no overlap, w3 longest
    pivots = [
        patterns.Pivot(i * 10, p, "L" if i % 2 == 0 else "H", i * 10 + 5)
        for i, p in enumerate(prices)
    ]
    found = patterns.elliott_impulse(pivots, np.array(prices, dtype=float))
    assert len(found) == 1
    assert found[0].signal_index == pivots[-1].confirm_index


def test_elliott_rejects_wave_two_full_retracement():
    prices = [100, 150, 95, 220, 180, 260]      # wave 2 below wave 1 start
    pivots = [
        patterns.Pivot(i * 10, p, "L" if i % 2 == 0 else "H", i * 10 + 5)
        for i, p in enumerate(prices)
    ]
    assert patterns.elliott_impulse(pivots, np.array(prices, dtype=float)) == []


def test_head_and_shoulders_signals_on_the_neckline_break():
    prices = np.array(
        [100] * 5 + [140] + [110] * 3 + [180] + [108] * 3 + [138] + [120, 105, 90],
        dtype=float,
    )
    pivots = [
        patterns.Pivot(5, 140.0, "H", 8), patterns.Pivot(8, 110.0, "L", 9),
        patterns.Pivot(9, 180.0, "H", 12), patterns.Pivot(12, 108.0, "L", 13),
        patterns.Pivot(13, 138.0, "H", 16),
    ]
    found = patterns.head_and_shoulders(pivots, prices)
    assert len(found) == 1
    neckline = found[0].detail["neckline"]
    assert prices[found[0].signal_index] < neckline


def test_double_top_requires_similar_highs():
    pivots = [
        patterns.Pivot(0, 100.0, "H", 3), patterns.Pivot(3, 70.0, "L", 5),
        patterns.Pivot(5, 160.0, "H", 8),       # 60% apart, not a double top
    ]
    prices = np.array([100, 90, 80, 70, 120, 160, 120, 90, 60], dtype=float)
    assert patterns.double_top(pivots, prices, tolerance=0.05) == []


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def test_noise_profile_exposes_lookahead():
    """The headline result, as a regression test.

    In a pure random walk every pivot-anchored pattern shows a large,
    significant "edge" measured from its own pivot, and none measured from the
    bar it could have been traded on. Nothing is being predicted in either
    case - the first number is the definition looking forward.
    """
    walk = _random_walk(n=3000, seed=11)
    profile = patterns.noise_profile(walk, "double_top", n_series=40, horizon=60)

    pivot_se = profile["from_pivot"]["standard_errors"]
    signal_se = profile["from_signal"]["standard_errors"]

    assert pivot_se > 5.0, f"pivot-anchored bias vanished ({pivot_se:.1f} se)"
    assert signal_se < 3.0, f"noise shows a causal edge ({signal_se:.1f} se)"
    assert pivot_se > signal_se * 3


def test_noise_profile_bias_points_the_pattern_s_way():
    """A bearish pattern biases down, a bullish one up - both spuriously."""
    walk = _random_walk(n=3000, seed=5)
    bear = patterns.noise_profile(walk, "double_top", n_series=30)
    bull = patterns.noise_profile(walk, "double_bottom", n_series=30)
    assert bear["from_pivot"]["delta_vs_baseline"] < 0
    assert bull["from_pivot"]["delta_vs_baseline"] > 0


def test_patterns_are_common_in_noise():
    walk = _random_walk(n=4000, seed=7)
    result = patterns.null_frequency(walk, "elliott_impulse", n_surrogates=40)
    assert result["series_with_at_least_one"] > 0.8
    assert result["null_mean"] > 1.0


def test_edge_test_flags_hindsight_only():
    walk = _random_walk(n=4000, seed=13)
    result = patterns.edge_test(walk, "double_top", n_surrogates=30)
    assert not result["tradeable"]
    assert "verdict" in result


def test_independent_subset_drops_overlaps():
    assert patterns._independent_subset([0, 10, 20, 100, 105, 200], 60) == [0, 100, 200]
    assert patterns._independent_subset([], 60) == []


def test_audit_reports_no_matches_gracefully():
    flat = np.full(500, 100.0)
    audit = patterns.lookahead_audit(flat, patterns.elliott_impulse)
    assert audit["n_matches"] == 0


def test_custom_detector_gets_the_same_treatment(walk):
    """A user-supplied detector must work through the same API."""
    def naive(pivots, prices):
        return [
            patterns.PatternMatch(name="naive", pivots=[p],
                                  signal_index=p.confirm_index, direction=-1)
            for p in pivots if p.kind == "H"
        ]

    result = patterns.edge_test(walk, naive, n_surrogates=20)
    assert "lookahead_audit" in result
    assert result["lookahead_audit"]["n_matches"] > 0
