"""Cycle analogs via dynamic time warping.

Overlay the running cycle on the previous ones and ask how far along it looks.
This is the formal version of every "cycle overlay" chart on crypto Twitter,
with two differences: the alignment is warped rather than rigid (cycles have
been getting longer, so a fixed day-for-day overlay is wrong by construction),
and the output carries its dispersion.

Read the dispersion, not the mean. With three prior cycles the "forecast" is an
average of three paths whose amplitudes span 560x, 128x and 22x. The spread is
the finding; the central path is decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config


@dataclass
class AnalogMatch:
    cycle_index: int
    distance: float
    aligned_position: float   # 0..1 progress through the analog's low->top leg
    analog_days_to_top: int
    implied_days_to_top: int


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int | None = None) -> tuple:
    """Classic DTW with an optional Sakoe-Chiba band.

    Returns ``(distance, path)``; distance is normalised by path length so
    sequences of different lengths stay comparable.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf"), []
    if band is None:
        band = max(n, m)

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, int(i * m / n) - band)
        hi = min(m, int(i * m / n) + band)
        for j in range(lo, hi + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrack for the alignment path.
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    distance = cost[n, m] / max(len(path), 1)
    return float(distance), path


def _normalised_leg(price: pd.Series, start: pd.Timestamp,
                    end: pd.Timestamp | None) -> pd.Series:
    """Log price of one leg, rebased so the low is 0 - scale-free overlay."""
    mask = price.index >= start
    if end is not None:
        mask &= price.index <= end
    leg = price[mask].astype(float)
    if leg.empty:
        return leg
    return np.log(leg) - np.log(leg.iloc[0])


def _resample(series: pd.Series, n: int) -> np.ndarray:
    """Resample a leg to ``n`` points so DTW compares shapes, not lengths."""
    if len(series) < 2:
        return np.full(n, np.nan)
    source = np.linspace(0.0, 1.0, len(series))
    target = np.linspace(0.0, 1.0, n)
    return np.interp(target, source, series.to_numpy(dtype=float))


def match_current(price: pd.Series, resample_to: int = 200,
                  band_fraction: float = 0.25) -> list:
    """Match the running cycle against each completed cycle's low->top leg."""
    current = config.CURRENT_CYCLE
    current_leg = _normalised_leg(price, pd.Timestamp(current.low_date), None)
    if len(current_leg) < 60:
        raise ValueError("running cycle has too little data to match")

    current_days = len(current_leg)
    matches = []
    for cycle in config.COMPLETED_CYCLES:
        full = _normalised_leg(price, pd.Timestamp(cycle.low_date),
                               pd.Timestamp(cycle.top_date))
        if len(full) < 60:
            continue

        # Compare the running leg against every prefix of the analog, and take
        # the prefix that fits best. The best prefix length is the estimate of
        # "how far into that cycle we currently look".
        best = None
        for fraction in np.linspace(0.30, 1.00, 36):
            cut = max(int(len(full) * fraction), 30)
            prefix = full.iloc[:cut]
            a = _resample(current_leg, resample_to)
            b = _resample(prefix, resample_to)
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            distance, _ = dtw_distance(a, b, band=int(resample_to * band_fraction))
            if best is None or distance < best[0]:
                best = (distance, fraction, cut)

        if best is None:
            continue
        distance, fraction, cut = best
        analog_total = len(full)
        analog_days_to_top = analog_total - cut
        # Scale the analog's remaining days by how much slower/faster the
        # running cycle has travelled to reach the same shape.
        speed = current_days / max(cut, 1)
        matches.append(
            AnalogMatch(
                cycle_index=cycle.index,
                distance=distance,
                aligned_position=float(fraction),
                analog_days_to_top=int(analog_days_to_top),
                implied_days_to_top=int(round(analog_days_to_top * speed)),
            )
        )

    return sorted(matches, key=lambda m: m.distance)


def summarise(matches: list) -> dict:
    """Inverse-distance-weighted consensus, reported with its spread."""
    if not matches:
        return {}
    weights = np.array([1.0 / max(m.distance, 1e-6) for m in matches])
    weights = weights / weights.sum()
    positions = np.array([m.aligned_position for m in matches])
    days = np.array([m.implied_days_to_top for m in matches], dtype=float)

    return {
        "n_analogs": len(matches),
        "position_weighted": float((positions * weights).sum()),
        "position_min": float(positions.min()),
        "position_max": float(positions.max()),
        "days_to_top_weighted": float((days * weights).sum()),
        "days_to_top_min": float(days.min()),
        "days_to_top_max": float(days.max()),
        "best_analog_cycle": matches[0].cycle_index,
        "best_analog_distance": matches[0].distance,
    }


def amplitude_decay() -> dict:
    """Fit log(top multiple) against cycle index and extrapolate.

    Three points, one line - this is an illustration of the decay, not
    evidence for its functional form. Reported with the caveat attached.
    """
    cycles = [c for c in config.COMPLETED_CYCLES if c.top_multiple]
    if len(cycles) < 2:
        return {}
    x = np.array([c.index for c in cycles], dtype=float)
    y = np.log(np.array([c.top_multiple for c in cycles], dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    next_index = config.CURRENT_CYCLE.index
    predicted = float(np.exp(slope * next_index + intercept))
    residuals = y - (slope * x + intercept)
    return {
        "observed_multiples": {int(c.index): round(c.top_multiple, 1) for c in cycles},
        "decay_per_cycle": float(np.exp(slope)),
        "predicted_multiple": predicted,
        "predicted_top_price": predicted * config.CURRENT_CYCLE.low_price,
        "n_points": len(cycles),
        "residual_std_log": float(residuals.std(ddof=0)),
        "caveat": f"fit on {len(cycles)} points; treat as descriptive only",
    }


def halving_clock() -> dict:
    """Days from halving to top in each completed cycle, and the running count."""
    completed = [c for c in config.COMPLETED_CYCLES if c.halving_to_top_days]
    days = [c.halving_to_top_days for c in completed]
    today = pd.Timestamp.today().normalize().date()
    current = config.CURRENT_CYCLE
    elapsed = (today - current.halving_date).days
    return {
        "halving_to_top_days": {int(c.index): c.halving_to_top_days for c in completed},
        "mean_days": float(np.mean(days)) if days else float("nan"),
        "std_days": float(np.std(days, ddof=0)) if days else float("nan"),
        "current_days_since_halving": elapsed,
        "current_vs_mean": elapsed - float(np.mean(days)) if days else float("nan"),
    }
