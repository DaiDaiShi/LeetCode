"""Cycle anchors, phase definitions and feature specs.

The anchor table below is the only piece of hand-entered market history in the
package. Dates are the widely-cited daily-close extremes; prices are rounded
daily closes and are used only for sanity checks, never for model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

ACCUMULATION = "accumulation"      # 底部吸筹
EARLY_BULL = "early_bull"          # 爆发前
PARABOLIC = "parabolic"            # 爆发中
BEAR = "bear"                      # 爆发后 / 下跌

PHASES = [ACCUMULATION, EARLY_BULL, PARABOLIC, BEAR]

PHASE_ZH = {
    ACCUMULATION: "底部吸筹",
    EARLY_BULL: "爆发前",
    PARABOLIC: "爆发中",
    BEAR: "爆发后/熊市",
}

# Log-price progress thresholds that split a low->top leg into three phases.
# progress = (log P - log P_low) / (log P_top - log P_low), clipped to [0, 1].
ACCUMULATION_MAX_PROGRESS = 0.25
EARLY_BULL_MAX_PROGRESS = 0.60


# ---------------------------------------------------------------------------
# Cycle anchors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cycle:
    index: int
    low_date: date
    low_price: float
    halving_date: date
    top_date: Optional[date] = None
    top_price: Optional[float] = None
    interim_top_date: Optional[date] = None
    note: str = ""

    @property
    def is_complete(self) -> bool:
        return self.top_date is not None

    @property
    def halving_to_top_days(self) -> Optional[int]:
        if self.top_date is None:
            return None
        return (self.top_date - self.halving_date).days

    @property
    def low_to_top_days(self) -> Optional[int]:
        if self.top_date is None:
            return None
        return (self.top_date - self.low_date).days

    @property
    def top_multiple(self) -> Optional[float]:
        if self.top_price is None:
            return None
        return self.top_price / self.low_price


HALVINGS = [
    date(2012, 11, 28),
    date(2016, 7, 9),
    date(2020, 5, 11),
    date(2024, 4, 19),
    # Projected; block-height driven, drifts by a few weeks.
    date(2028, 3, 26),
]

CYCLES = [
    Cycle(
        index=1,
        low_date=date(2011, 11, 18),
        low_price=2.05,
        halving_date=date(2012, 11, 28),
        top_date=date(2013, 12, 4),
        top_price=1163.0,
        note="Mt.Gox era, thin liquidity; treat amplitude as barely comparable.",
    ),
    Cycle(
        index=2,
        low_date=date(2015, 1, 14),
        low_price=152.0,
        halving_date=date(2016, 7, 9),
        top_date=date(2017, 12, 17),
        top_price=19650.0,
        note="Retail/ICO cycle.",
    ),
    Cycle(
        index=3,
        low_date=date(2018, 12, 15),
        low_price=3191.0,
        halving_date=date(2020, 5, 11),
        top_date=date(2021, 11, 10),
        top_price=68990.0,
        interim_top_date=date(2021, 4, 14),
        note="Double top: 2021-04-14 (~64.8k) and 2021-11-10 (~69k).",
    ),
    Cycle(
        index=4,
        low_date=date(2022, 11, 21),
        low_price=15476.0,
        halving_date=date(2024, 4, 19),
        top_date=None,
        top_price=None,
        note="ETF/institutional cycle. Top left unset on purpose - the running "
             "cycle must never be labelled from hindsight the model does not have.",
    ),
]

COMPLETED_CYCLES = [c for c in CYCLES if c.is_complete]
CURRENT_CYCLE = CYCLES[-1]


def halving_on_or_before(day: date) -> Optional[date]:
    prior = [h for h in HALVINGS if h <= day]
    return prior[-1] if prior else None


def cycle_containing(day: date) -> Optional[Cycle]:
    """Cycle whose low <= day, i.e. the leg the date belongs to."""
    prior = [c for c in CYCLES if c.low_date <= day]
    return prior[-1] if prior else None


def anchor_frame() -> pd.DataFrame:
    """Anchor table as a DataFrame, handy for reports and notebooks."""
    rows = []
    for c in CYCLES:
        rows.append(
            {
                "cycle": c.index,
                "low_date": c.low_date,
                "low_price": c.low_price,
                "halving_date": c.halving_date,
                "top_date": c.top_date,
                "top_price": c.top_price,
                "low_to_top_days": c.low_to_top_days,
                "halving_to_top_days": c.halving_to_top_days,
                "top_multiple": c.top_multiple,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

# Tier A features need nothing but a daily close series.
PRICE_FEATURES = [
    "mayer",              # P / SMA200
    "ma200w_mult",        # P / SMA(1400d)
    "drawdown",           # P / running ATH - 1  (<= 0)
    "ret_30",
    "ret_90",
    "ret_365",
    "realized_vol_30",
    "pi_cycle_ratio",     # SMA111 / (2 * SMA350); crosses 1 near historical tops
    "low_multiple",       # P / price at the cycle low
]

# Tier B needs on-chain data (Coin Metrics community endpoint is free).
ONCHAIN_FEATURES = [
    "mvrv",
    "realized_price_mult",
    "puell",
]

# Tier C is macro/liquidity (FRED + any global M2 export).
MACRO_FEATURES = [
    "m2_yoy",
    "dxy_yoy",
]

# Signals fed to the composite "cycle heat" index. Sign = +1 when a high raw
# value means a hot/late-cycle market.
COMPOSITE_SIGNALS = {
    "mayer": +1,
    "ma200w_mult": +1,
    "pi_cycle_ratio": +1,
    "ret_365": +1,
    "drawdown": +1,          # drawdown is negative; near 0 == at highs == hot
    "mvrv": +1,
    "realized_price_mult": +1,
    "puell": +1,
}

# Features used by the HMM. Kept small: 4 states x ~6 dims is already generous
# for ~4600 daily observations that contain only 3.5 independent cycles.
HMM_FEATURES = [
    "ret_90",
    "ret_365",
    "realized_vol_30",
    "drawdown",
    "log_mayer",
    "log_ma200w_mult",
]

MIN_HISTORY_DAYS = 400      # before this, causal stats are too thin to trust
DEFAULT_N_STATES = 4
CYCLE_LENGTH_DAYS = 1400    # ~4 years, used for harmonic phase terms
