"""Chart-pattern edge testing.

Does a chart pattern actually predict anything, or does it only look like it
does because the pattern is defined using the future?

That question has a definite answer and almost nobody computes it. This module
computes it for any pattern, through one architectural decision: **every
detector must declare a ``signal_index``** - the bar at which a live trader
would have known the pattern was there. Forward returns are measured from that
bar, never from the pattern's own pivot.

Why that matters, concretely. A ZigZag swing high is only confirmed once price
has fallen ``pct`` below it. So "price falls after a swing high" is true by
construction, in any series whatsoever. Measured from the pivot, a completed
Elliott impulse in a pure random walk "predicts" a 13.6% decline at 35.8
standard errors. Measured from the bar the label actually appears, the same
pattern in the same data gives +5.7%, 0.6 standard errors - indistinguishable
from the baseline. The entire apparent edge was the definition looking forward.

Two tests are run against every pattern:

* :func:`null_frequency` - how often does it appear in random walks matched to
  this series' own drift and volatility? A pattern that is just as common in
  noise carries no information by being present.
* :func:`lookahead_audit` - the pivot-versus-signal comparison above, which
  separates real edge from definitional hindsight.

Plug in your own detector and it gets the same treatment::

    from btc_cycle import patterns

    def my_pattern(pivots, prices):
        ...
        return [patterns.PatternMatch(name="mine", pivots=used,
                                      signal_index=when_you_would_know,
                                      direction=-1)]

    print(patterns.edge_test(prices, my_pattern))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

DEFAULT_ZIGZAG_PCT = 0.15
DEFAULT_HORIZON = 60


# ---------------------------------------------------------------------------
# Swing primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pivot:
    index: int          # bar where the extreme actually occurred
    price: float
    kind: str           # 'H' or 'L'
    confirm_index: int  # bar where a live trader could first have known

    @property
    def lag(self) -> int:
        return self.confirm_index - self.index


@dataclass
class PatternMatch:
    """One detected occurrence.

    ``signal_index`` is the whole point. It is the earliest bar at which the
    pattern was knowable in real time - the confirmation bar of the last pivot
    the detector relied on, or a later trigger such as a neckline break. A
    detector that sets it to a pivot index is lying about its own latency, and
    :func:`edge_test` will report an edge that cannot be traded.
    """

    name: str
    pivots: list
    signal_index: int
    direction: int = -1          # +1 expects price up, -1 expects price down
    detail: dict = field(default_factory=dict)

    @property
    def pattern_end_index(self) -> int:
        return max(p.index for p in self.pivots)

    @property
    def lag(self) -> int:
        return self.signal_index - self.pattern_end_index


def zigzag(prices: Sequence[float], pct: float = DEFAULT_ZIGZAG_PCT) -> list:
    """Swing points with the bar at which each was confirmed.

    A high is confirmed only once price trades ``pct`` below it; that gap is
    the latency every pivot-based pattern inherits and that hindsight charts
    hide.
    """
    values = np.asarray(prices, dtype=float)
    if len(values) < 2:
        return []

    pivots: list = []
    high_i = low_i = 0
    high_p = low_p = float(values[0])
    trend = 0

    for i in range(1, len(values)):
        price = float(values[i])
        if price > high_p:
            high_i, high_p = i, price
        if price < low_p:
            low_i, low_p = i, price

        if trend <= 0 and price >= low_p * (1.0 + pct):
            pivots.append(Pivot(low_i, low_p, "L", i))
            trend = 1
            high_i, high_p = i, price
        elif trend >= 0 and price <= high_p * (1.0 - pct):
            pivots.append(Pivot(high_i, high_p, "H", i))
            trend = -1
            low_i, low_p = i, price

    return pivots


def _kinds(pivots: Sequence[Pivot]) -> str:
    return "".join(p.kind for p in pivots)


def _first_cross_below(prices: np.ndarray, start: int, level: float) -> int | None:
    for i in range(start, len(prices)):
        if prices[i] < level:
            return i
    return None


def _first_cross_above(prices: np.ndarray, start: int, level: float) -> int | None:
    for i in range(start, len(prices)):
        if prices[i] > level:
            return i
    return None


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def elliott_impulse(pivots: Sequence[Pivot], prices: Sequence[float]) -> list:
    """Five-wave up impulse satisfying Elliott's three hard rules.

    Only three of Elliott's constraints are objective; everything else
    ("guidelines") may be violated at will, which is most of why the framework
    fits any history. The three:

    1. wave 2 does not retrace all of wave 1
    2. wave 3 is not the shortest of 1, 3, 5
    3. wave 4 does not overlap wave 1's territory

    In random walks roughly 10% of five-swing windows satisfy all three, and
    rules 1 and 2 alone pass 55% and 67% of the time - they barely constrain.

    Direction is -1: the framework's one operational claim is that a completed
    impulse is followed by a correction.
    """
    out = []
    for i in range(len(pivots) - 5):
        seq = list(pivots[i:i + 6])
        if _kinds(seq) != "LHLHLH":
            continue
        log_p = [np.log(p.price) for p in seq]
        wave1, wave3, wave5 = log_p[1] - log_p[0], log_p[3] - log_p[2], log_p[5] - log_p[4]

        rule1 = log_p[2] > log_p[0]
        rule2 = not (wave3 < wave1 and wave3 < wave5)
        rule3 = log_p[4] > log_p[1]
        if not (rule1 and rule2 and rule3):
            continue

        out.append(
            PatternMatch(
                name="elliott_impulse",
                pivots=seq,
                signal_index=seq[-1].confirm_index,
                direction=-1,
                detail={"wave1": wave1, "wave3": wave3, "wave5": wave5,
                        "extended_wave": int(np.argmax([wave1, wave3, wave5]) * 2 + 1)},
            )
        )
    return out


def head_and_shoulders(pivots: Sequence[Pivot], prices: Sequence[float],
                       shoulder_tolerance: float = 0.15) -> list:
    """Bearish head and shoulders, triggered on the neckline break.

    Unlike the Elliott detector this one can signal before its last pivot is
    confirmed - the neckline break is a price event, not a swing. That is the
    honest way to handle it, and it is why ``signal_index`` is computed rather
    than assumed.
    """
    values = np.asarray(prices, dtype=float)
    out = []
    for i in range(len(pivots) - 4):
        seq = list(pivots[i:i + 5])
        if _kinds(seq) != "HLHLH":
            continue
        left, trough1, head, trough2, right = seq
        if not (head.price > left.price and head.price > right.price):
            continue
        if abs(left.price - right.price) / max(left.price, right.price) > shoulder_tolerance:
            continue

        neckline = (trough1.price + trough2.price) / 2.0
        break_index = _first_cross_below(values, right.index, neckline)
        if break_index is None:
            continue

        out.append(
            PatternMatch(
                name="head_and_shoulders",
                pivots=seq,
                signal_index=max(break_index, right.index),
                direction=-1,
                detail={"neckline": neckline, "head": head.price,
                        "shoulder_asymmetry": abs(left.price - right.price) / head.price},
            )
        )
    return out


def double_top(pivots: Sequence[Pivot], prices: Sequence[float],
               tolerance: float = 0.05) -> list:
    """Two highs within ``tolerance``, triggered on the break of the trough."""
    values = np.asarray(prices, dtype=float)
    out = []
    for i in range(len(pivots) - 2):
        seq = list(pivots[i:i + 3])
        if _kinds(seq) != "HLH":
            continue
        first, trough, second = seq
        if abs(first.price - second.price) / max(first.price, second.price) > tolerance:
            continue
        break_index = _first_cross_below(values, second.index, trough.price)
        if break_index is None:
            continue
        out.append(
            PatternMatch(
                name="double_top", pivots=seq,
                signal_index=max(break_index, second.index), direction=-1,
                detail={"trough": trough.price,
                        "spread": abs(first.price - second.price) / first.price},
            )
        )
    return out


def double_bottom(pivots: Sequence[Pivot], prices: Sequence[float],
                  tolerance: float = 0.05) -> list:
    """Mirror of :func:`double_top`, triggered on the break of the peak."""
    values = np.asarray(prices, dtype=float)
    out = []
    for i in range(len(pivots) - 2):
        seq = list(pivots[i:i + 3])
        if _kinds(seq) != "LHL":
            continue
        first, peak, second = seq
        if abs(first.price - second.price) / max(first.price, second.price) > tolerance:
            continue
        break_index = _first_cross_above(values, second.index, peak.price)
        if break_index is None:
            continue
        out.append(
            PatternMatch(
                name="double_bottom", pivots=seq,
                signal_index=max(break_index, second.index), direction=+1,
                detail={"peak": peak.price,
                        "spread": abs(first.price - second.price) / first.price},
            )
        )
    return out


DETECTORS = {
    "elliott_impulse": elliott_impulse,
    "head_and_shoulders": head_and_shoulders,
    "double_top": double_top,
    "double_bottom": double_bottom,
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _independent_subset(indices: Sequence[int], horizon: int) -> list:
    """Greedily keep signals whose forward windows do not overlap.

    Overlapping windows are near-duplicates of one observation. Counting them
    all inflates any t-statistic by roughly the square root of the overlap, so
    the honest sample size is this subset.
    """
    keep, last = [], -10 ** 9
    for i in sorted(indices):
        if i - last >= horizon:
            keep.append(i)
            last = i
    return keep


def _forward_stats(log_prices: np.ndarray, indices: Sequence[int],
                   horizon: int, direction_sign: int = 1) -> dict:
    usable = [i for i in indices if i + horizon < len(log_prices)]
    if not usable:
        return {"n": 0}
    returns = np.array([log_prices[i + horizon] - log_prices[i] for i in usable])
    independent = _independent_subset(usable, horizon)
    indep_returns = np.array(
        [log_prices[i + horizon] - log_prices[i] for i in independent]
    )
    out = {
        "n": len(returns),
        "n_independent": len(indep_returns),
        "mean": float(returns.mean()),
        "median": float(np.median(returns)),
        "down_rate": float((returns < 0).mean()),
    }
    if len(indep_returns) > 1:
        out["independent_mean"] = float(indep_returns.mean())
        out["independent_se"] = float(indep_returns.std(ddof=1) / np.sqrt(len(indep_returns)))
    return out


def lookahead_audit(prices: Sequence[float], detector: Callable | str,
                    horizon: int = DEFAULT_HORIZON,
                    pct: float = DEFAULT_ZIGZAG_PCT,
                    baseline_step: int = 25) -> dict:
    """Measure the pattern twice: from its pivot, and from its signal bar.

    The gap between the two is the look-ahead bias the pattern carries. Only
    the second number is tradeable.
    """
    if isinstance(detector, str):
        if detector not in DETECTORS:
            raise ValueError(f"unknown detector {detector!r}; have {list(DETECTORS)}")
        detector = DETECTORS[detector]

    values = np.asarray(prices, dtype=float)
    log_prices = np.log(values)
    pivots = zigzag(values, pct)
    matches = detector(pivots, values)

    baseline_idx = list(range(0, len(values) - horizon, baseline_step))
    baseline = _forward_stats(log_prices, baseline_idx, horizon)

    if not matches:
        return {"n_matches": 0, "baseline": baseline,
                "note": "pattern never occurred in this series"}

    from_pivot = _forward_stats(
        log_prices, [m.pattern_end_index for m in matches], horizon
    )
    from_signal = _forward_stats(
        log_prices, [m.signal_index for m in matches], horizon
    )
    lags = np.array([m.lag for m in matches], dtype=float)

    def deviation(stats: dict) -> dict:
        if stats.get("n", 0) == 0 or "independent_se" not in stats:
            return {"delta": float("nan"), "standard_errors": float("nan")}
        delta = stats["independent_mean"] - baseline["mean"]
        se = stats["independent_se"]
        return {"delta": float(delta),
                "standard_errors": float(abs(delta) / se) if se > 0 else float("nan")}

    pivot_dev = deviation(from_pivot)
    signal_dev = deviation(from_signal)

    return {
        "n_matches": len(matches),
        "horizon": horizon,
        "baseline": baseline,
        "from_pivot": {**from_pivot, **pivot_dev},
        "from_signal": {**from_signal, **signal_dev},
        "median_lag_bars": float(np.median(lags)),
        "mean_lag_bars": float(lags.mean()),
        "lookahead_inflation": (
            float(pivot_dev["standard_errors"] - signal_dev["standard_errors"])
            if np.isfinite(pivot_dev["standard_errors"])
            and np.isfinite(signal_dev["standard_errors"])
            else float("nan")
        ),
        "reading": (
            "from_pivot is what a hindsight chart shows. from_signal is what "
            "you could have traded. If from_pivot is significant and "
            "from_signal is not, the pattern is a definition, not a forecast."
        ),
    }


def null_frequency(prices: Sequence[float], detector: Callable | str,
                   n_surrogates: int = 200, pct: float = DEFAULT_ZIGZAG_PCT,
                   seed: int = 0) -> dict:
    """How often does the pattern appear in random walks matched to this series?

    Matching drift and volatility matters: a pattern's frequency scales with
    volatility, so an unmatched null makes any pattern look either rare or
    ubiquitous for the wrong reason.
    """
    if isinstance(detector, str):
        if detector not in DETECTORS:
            raise ValueError(f"unknown detector {detector!r}; have {list(DETECTORS)}")
        detector = DETECTORS[detector]

    values = np.asarray(prices, dtype=float)
    daily = np.diff(np.log(values))
    drift, vol = float(daily.mean()), float(daily.std(ddof=1))
    n = len(values)

    observed = len(detector(zigzag(values, pct), values))

    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_surrogates):
        surrogate = np.exp(np.log(values[0]) + np.cumsum(rng.normal(drift, vol, n - 1)))
        surrogate = np.concatenate([[values[0]], surrogate])
        counts.append(len(detector(zigzag(surrogate, pct), surrogate)))
    counts = np.array(counts, dtype=float)

    return {
        "observed_count": observed,
        "null_mean": float(counts.mean()),
        "null_p95": float(np.percentile(counts, 95)),
        "p_value": float((counts >= observed).mean()),
        "series_with_at_least_one": float((counts >= 1).mean()),
        "matched_drift_annual": drift * 365.0,
        "matched_vol_annual": vol * np.sqrt(365.0),
        "reading": (
            "series_with_at_least_one near 1.0 means the pattern is a normal "
            "feature of noise. Finding one tells you nothing by itself."
        ),
    }


def edge_test(prices: Sequence[float], detector: Callable | str,
              horizon: int = DEFAULT_HORIZON, pct: float = DEFAULT_ZIGZAG_PCT,
              n_surrogates: int = 200, seed: int = 0) -> dict:
    """Both tests plus a verdict.

    A pattern passes only when its *causal* forward return departs from the
    baseline by more than two standard errors on non-overlapping observations.
    Frequency in noise is reported alongside, because a pattern that noise
    produces just as often carries no information from its mere presence.
    """
    if isinstance(detector, str):
        if detector not in DETECTORS:
            raise ValueError(f"unknown detector {detector!r}; have {list(DETECTORS)}")
        detector = DETECTORS[detector]

    audit = lookahead_audit(prices, detector, horizon=horizon, pct=pct)
    null = null_frequency(prices, detector, n_surrogates=n_surrogates, pct=pct, seed=seed)

    signal_se = audit.get("from_signal", {}).get("standard_errors", float("nan"))
    pivot_se = audit.get("from_pivot", {}).get("standard_errors", float("nan"))
    tradeable = bool(np.isfinite(signal_se) and signal_se >= 2.0)
    hindsight_only = bool(
        np.isfinite(pivot_se) and np.isfinite(signal_se)
        and pivot_se >= 2.0 and signal_se < 2.0
    )

    if audit.get("n_matches", 0) == 0:
        verdict = "pattern absent from this series"
    elif hindsight_only:
        verdict = (
            "HINDSIGHT ONLY - significant measured from the pivot, not from the "
            "bar you could have acted on. The edge is in the definition."
        )
    elif tradeable:
        verdict = "survives the causal test; check the null frequency before believing it"
    else:
        verdict = "no edge at either measurement point"

    return {
        "lookahead_audit": audit,
        "null_frequency": null,
        "tradeable": tradeable,
        "hindsight_only": hindsight_only,
        "verdict": verdict,
    }


def noise_profile(prices: Sequence[float], detector: Callable | str,
                  n_series: int = 100, horizon: int = DEFAULT_HORIZON,
                  pct: float = DEFAULT_ZIGZAG_PCT, seed: int = 0) -> dict:
    """Pool the audit across many matched random walks.

    One series yields only a handful of non-overlapping observations per
    pattern, which is too few to separate a real effect from sampling noise.
    Pooling across surrogates with the same drift and volatility gives the
    reference distribution: what this pattern looks like when there is
    definitely nothing to find.

    Run this before believing any result on the real series. If the real
    series' ``from_signal`` figure sits inside this distribution, the pattern
    is doing exactly what noise does.
    """
    if isinstance(detector, str):
        detector = DETECTORS[detector]

    values = np.asarray(prices, dtype=float)
    daily = np.diff(np.log(values))
    drift, vol = float(daily.mean()), float(daily.std(ddof=1))
    n = len(values)

    rng = np.random.default_rng(seed)
    pivot_returns: list = []
    signal_returns: list = []
    baseline_returns: list = []
    lags: list = []
    counts: list = []

    for _ in range(n_series):
        path = np.concatenate(
            [[values[0]], np.exp(np.log(values[0]) + np.cumsum(rng.normal(drift, vol, n - 1)))]
        )
        log_path = np.log(path)
        matches = detector(zigzag(path, pct), path)
        counts.append(len(matches))

        baseline_returns.extend(
            log_path[i + horizon] - log_path[i]
            for i in range(0, len(path) - horizon, horizon)   # non-overlapping
        )
        for index_list, sink in (
            ([m.pattern_end_index for m in matches], pivot_returns),
            ([m.signal_index for m in matches], signal_returns),
        ):
            for i in _independent_subset(
                [j for j in index_list if j + horizon < len(path)], horizon
            ):
                sink.append(log_path[i + horizon] - log_path[i])
        lags.extend(m.lag for m in matches)

    def summarise(sample: list, name: str) -> dict:
        arr = np.array(sample, dtype=float)
        if len(arr) < 2:
            return {"name": name, "n": len(arr)}
        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        return {
            "name": name, "n": len(arr), "mean": float(arr.mean()),
            "standard_error": se, "down_rate": float((arr < 0).mean()),
        }

    base = summarise(baseline_returns, "baseline")
    pivot = summarise(pivot_returns, "from_pivot")
    signal = summarise(signal_returns, "from_signal")

    for block in (pivot, signal):
        if "mean" in block and "mean" in base:
            delta = block["mean"] - base["mean"]
            block["delta_vs_baseline"] = float(delta)
            block["standard_errors"] = (
                float(abs(delta) / block["standard_error"])
                if block["standard_error"] > 0 else float("nan")
            )

    return {
        "n_series": n_series,
        "horizon": horizon,
        "matches_per_series": float(np.mean(counts)) if counts else 0.0,
        "series_with_at_least_one": float(np.mean([c >= 1 for c in counts])) if counts else 0.0,
        "median_lag_bars": float(np.median(lags)) if lags else float("nan"),
        "baseline": base,
        "from_pivot": pivot,
        "from_signal": signal,
        "reading": (
            "These are pure-noise results. from_pivot departing from baseline "
            "by many standard errors while from_signal does not is the "
            "signature of look-ahead in the pattern's own definition - it "
            "happens here with no signal present at all."
        ),
    }
