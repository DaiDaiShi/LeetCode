"""Composite cycle-heat index.

The dumb-but-honest baseline: causally percentile-rank each late-cycle signal,
average them, read the phase off the band. Every fancier model in this package
has to beat this to justify itself - and on out-of-sample data most of them
do not.

Why percentile rather than z-score: MVRV, Mayer and Puell all have fat right
tails whose scale has shrunk every cycle. A z-score built on 2013 data calls
2021 "cold"; a rank does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, features

# Heat bands -> phase. Boundaries are round numbers on purpose: they are a
# reading aid, not a fitted parameter. Fitting them on 3 cycles would be
# fitting noise.
BANDS = [
    (0.00, 0.25, config.ACCUMULATION),
    (0.25, 0.55, config.EARLY_BULL),
    (0.55, 0.85, config.PARABOLIC),
    (0.85, 1.01, config.PARABOLIC),
]


def heat(frame: pd.DataFrame, signals: dict | None = None,
         min_signals: int = 3) -> pd.DataFrame:
    """Per-signal causal percentiles plus their average.

    Returns a frame with one column per available signal and a ``heat`` column.
    Rows where fewer than ``min_signals`` signals are available give NaN heat,
    so the early thin-history period does not produce confident nonsense.
    """
    signals = signals or config.COMPOSITE_SIGNALS
    names = features.available(frame, list(signals))
    if not names:
        raise ValueError("no composite signals available in frame")

    columns = {}
    for name in names:
        series = frame[name].astype(float)
        if signals[name] < 0:
            series = -series
        columns[name] = features.expanding_percentile(series)

    out = pd.DataFrame(columns, index=frame.index)
    counts = out.notna().sum(axis=1)
    out["heat"] = out[names].mean(axis=1).where(counts >= min_signals)
    out["n_signals"] = counts
    return out


def phase_from_heat(heat_series: pd.Series) -> pd.Series:
    """Map the heat index onto a phase label."""
    out = pd.Series(np.nan, index=heat_series.index, dtype=object)
    for low, high, phase in BANDS:
        mask = (heat_series >= low) & (heat_series < high)
        out.loc[mask] = phase
    return out


def phase_probabilities(heat_value: float) -> dict:
    """Soft phase reading from a single heat value.

    Triangular membership functions centred on the band midpoints. This is a
    presentation device - it expresses "0.62 is parabolic but not far from
    early-bull", nothing more. It is not a calibrated probability.
    """
    if heat_value is None or not np.isfinite(heat_value):
        return {phase: float("nan") for phase in config.PHASES}

    centres = {
        config.ACCUMULATION: 0.12,
        config.EARLY_BULL: 0.40,
        config.PARABOLIC: 0.78,
    }
    width = 0.32
    raw = {
        phase: max(0.0, 1.0 - abs(heat_value - centre) / width)
        for phase, centre in centres.items()
    }
    raw[config.BEAR] = 0.0  # heat alone cannot tell rising 0.5 from falling 0.5
    total = sum(raw.values())
    if total <= 0:
        return {phase: float("nan") for phase in config.PHASES}
    return {phase: value / total for phase, value in raw.items()}


def with_direction(frame: pd.DataFrame, heat_frame: pd.DataFrame,
                   lookback: int = 60) -> pd.DataFrame:
    """Add the direction of travel, which is what separates pre from post.

    A heat of 0.5 on the way up is 爆发前; the same 0.5 on the way down is
    爆发后. The composite index is symmetric and cannot see the difference, so
    slope and drawdown are carried alongside it.
    """
    out = heat_frame.copy()
    out["heat_slope"] = out["heat"].diff(lookback)
    out["drawdown"] = frame["drawdown"]
    out["rising"] = out["heat_slope"] > 0

    phase = phase_from_heat(out["heat"])
    falling_hot = (~out["rising"]) & (out["heat"] < 0.55) & (frame["drawdown"] < -0.25)
    phase = phase.mask(falling_hot, config.BEAR)
    out["phase"] = phase
    return out
