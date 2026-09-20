"""Hindsight phase labels, used for supervised training and for scoring.

Labels are deliberately non-causal - they are built from known cycle lows and
tops. That is fine for a *target*; it is fatal for a *feature*. Two rules keep
the two apart:

* the running cycle has no top in :mod:`btc_cycle.config`, so its later days
  come back as NaN and are never trained on;
* validation is leave-one-cycle-out, so a model is never scored on the same
  cycle whose top defined its labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def label_phases(price: pd.Series) -> pd.Series:
    """Phase label per day; NaN where the enclosing leg is not yet resolved.

    Within a low -> top leg the split is by *log-price progress*:
        progress = (log P - log P_low) / (log P_top - log P_low)
    which is scale-free and therefore comparable across cycles whose amplitudes
    differ by two orders of magnitude. From the top to the next low the label
    is ``bear``.
    """
    labels = pd.Series(np.nan, index=price.index, dtype=object)
    log_price = np.log(price.astype(float))

    for i, cycle in enumerate(config.CYCLES):
        low = pd.Timestamp(cycle.low_date)
        if cycle.top_date is None:
            continue  # running cycle: unresolved, stays NaN
        top = pd.Timestamp(cycle.top_date)

        leg = (price.index >= low) & (price.index <= top)
        if leg.any():
            log_low = np.log(cycle.low_price)
            log_top = np.log(cycle.top_price)
            progress = (log_price[leg] - log_low) / (log_top - log_low)
            progress = progress.clip(0.0, 1.0)
            phase = np.where(
                progress < config.ACCUMULATION_MAX_PROGRESS,
                config.ACCUMULATION,
                np.where(
                    progress < config.EARLY_BULL_MAX_PROGRESS,
                    config.EARLY_BULL,
                    config.PARABOLIC,
                ),
            )
            labels.loc[leg] = phase

        # top -> next cycle low is the bear leg
        next_low = (
            pd.Timestamp(config.CYCLES[i + 1].low_date)
            if i + 1 < len(config.CYCLES)
            else None
        )
        if next_low is not None:
            bear = (price.index > top) & (price.index < next_low)
            labels.loc[bear] = config.BEAR

    return labels


def cycle_id(index: pd.DatetimeIndex) -> pd.Series:
    """Which cycle each day belongs to (by cycle low), for grouped CV."""
    out = pd.Series(np.nan, index=index, dtype=float)
    for cycle in config.CYCLES:
        out.loc[index >= pd.Timestamp(cycle.low_date)] = float(cycle.index)
    return out


def forward_return(price: pd.Series, horizon: int = 90) -> pd.Series:
    """Forward log return over ``horizon`` days - the regression target."""
    return np.log(price.astype(float)).shift(-horizon) - np.log(price.astype(float))


def coarse(label: pd.Series) -> pd.Series:
    """Collapse four phases into the three buckets the question asks about."""
    mapping = {
        config.ACCUMULATION: "pre",       # 爆发前
        config.EARLY_BULL: "pre",
        config.PARABOLIC: "mid",          # 爆发中
        config.BEAR: "post",              # 爆发后
    }
    return label.map(mapping)
