"""Signal battery.

Every transform here is *causal*: the value at row ``t`` uses only rows
``<= t``. That rules out the single most common bug in published "cycle top
indicators" - normalising a signal over the whole sample, which leaks the
future top into every historical reading and makes any backtest look
clairvoyant.

The one deliberately non-causal thing in this package is the phase *label*
(see :mod:`btc_cycle.labels`). Labels are hindsight by construction; features
must not be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Causal normalisation
# ---------------------------------------------------------------------------


def expanding_percentile(series: pd.Series, min_periods: int = config.MIN_HISTORY_DAYS) -> pd.Series:
    """Rank of each value within the history up to and including that day.

    Returns values in [0, 1]. O(n log n) via an incrementally sorted list -
    a naive ``expanding().apply()`` is O(n^2) and takes minutes on 5k rows.
    """
    import bisect

    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    seen: list = []
    for i, value in enumerate(values):
        if np.isnan(value):
            if len(seen) >= min_periods:
                out[i] = np.nan
            continue
        position = bisect.bisect_left(seen, value)
        bisect.insort(seen, value)
        if len(seen) >= min_periods:
            # position among the len(seen) values including this one
            out[i] = position / max(len(seen) - 1, 1)
    return pd.Series(out, index=series.index, name=series.name)


def expanding_zscore(series: pd.Series, min_periods: int = config.MIN_HISTORY_DAYS) -> pd.Series:
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    return (series - mean) / std.replace(0.0, np.nan)


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or window // 2
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _running_cycle_low(price: pd.Series) -> pd.Series:
    """Price at the most recent *known* cycle low, causally.

    Uses the anchor table, but only anchors whose date has already passed at
    row ``t`` - so a backtest at 2019-06 sees the 2018-12 low and nothing later.
    """
    out = pd.Series(index=price.index, dtype=float)
    for cycle in config.CYCLES:
        stamp = pd.Timestamp(cycle.low_date)
        out.loc[out.index >= stamp] = cycle.low_price
    return out.ffill()


def build(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw price (+ optional on-chain/macro) frame into model features."""
    if "price" not in frame.columns:
        raise ValueError("input frame needs a 'price' column")

    price = frame["price"].astype(float)
    log_price = np.log(price)
    out = pd.DataFrame(index=frame.index)
    out["price"] = price
    out["log_price"] = log_price

    # --- Tier A: price-derived -------------------------------------------
    sma200 = price.rolling(200, min_periods=100).mean()
    sma1400 = price.rolling(1400, min_periods=400).mean()
    sma111 = price.rolling(111, min_periods=60).mean()
    sma350 = price.rolling(350, min_periods=180).mean()

    out["mayer"] = price / sma200
    out["ma200w_mult"] = price / sma1400
    out["pi_cycle_ratio"] = sma111 / (2.0 * sma350)
    out["log_mayer"] = np.log(out["mayer"])
    out["log_ma200w_mult"] = np.log(out["ma200w_mult"])

    running_ath = price.cummax()
    out["drawdown"] = price / running_ath - 1.0
    row_number = pd.Series(np.arange(len(price), dtype=float), index=price.index)
    last_ath_row = row_number.where(price >= running_ath).ffill()
    out["days_since_ath"] = row_number - last_ath_row

    out["ret_30"] = log_price.diff(30)
    out["ret_90"] = log_price.diff(90)
    out["ret_365"] = log_price.diff(365)
    daily = log_price.diff()
    out["realized_vol_30"] = daily.rolling(30, min_periods=15).std() * np.sqrt(365)

    cycle_low = _running_cycle_low(price)
    out["low_multiple"] = price / cycle_low

    # --- Halving clock ----------------------------------------------------
    halving_dates = [pd.Timestamp(h) for h in config.HALVINGS]
    last_halving = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for stamp in halving_dates:
        last_halving.loc[last_halving.index >= stamp] = stamp
    last_halving = last_halving.ffill()
    days_since = (frame.index.to_series() - last_halving).dt.days.astype(float)
    out["days_since_halving"] = days_since
    angle = 2.0 * np.pi * days_since / config.CYCLE_LENGTH_DAYS
    out["halving_sin"] = np.sin(angle)
    out["halving_cos"] = np.cos(angle)

    # --- Tier B: on-chain -------------------------------------------------
    if "mvrv" in frame.columns:
        out["mvrv"] = frame["mvrv"].astype(float)
    if "realized_price" in frame.columns:
        realized = frame["realized_price"].astype(float)
        out["realized_price_mult"] = price / realized.replace(0.0, np.nan)
    elif "mvrv" in frame.columns:
        # MVRV is market cap / realized cap, which equals price / realized price.
        out["realized_price_mult"] = frame["mvrv"].astype(float)
    if "miner_revenue" in frame.columns:
        revenue = frame["miner_revenue"].astype(float)
        out["puell"] = revenue / revenue.rolling(365, min_periods=200).mean()

    # --- Tier C: macro ----------------------------------------------------
    if "m2" in frame.columns:
        m2 = frame["m2"].astype(float)
        out["m2_yoy"] = m2.pct_change(365)
    if "dxy" in frame.columns:
        dxy = frame["dxy"].astype(float)
        out["dxy_yoy"] = dxy.pct_change(365)

    return out


def available(frame: pd.DataFrame, names: list[str]) -> list[str]:
    """Subset of ``names`` present and not entirely NaN."""
    return [n for n in names if n in frame.columns and frame[n].notna().any()]


def normalised(frame: pd.DataFrame, names: list[str], method: str = "percentile") -> pd.DataFrame:
    """Causally normalised copy of the named columns."""
    fn = expanding_percentile if method == "percentile" else expanding_zscore
    return pd.DataFrame({name: fn(frame[name]) for name in available(frame, names)},
                        index=frame.index)
