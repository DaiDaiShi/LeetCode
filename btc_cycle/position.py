"""From a phase posterior to a position - and the arithmetic for refusing one.

The asymmetry this module is built around: Bitcoin's unconditional drift is
large and positive. A long is betting with that drift, a short against it. The
evidential bar for the two is therefore not the same, and any model that treats
``P(bear) = 0.6`` as the mirror image of ``P(bull) = 0.6`` will be short far too
often and lose money doing it.

:func:`short_hurdle` makes that precise. It computes how far a phase-conditional
return estimate has to sit below the unconditional drift before a short is
positive-EV after funding and costs, then compares that required shift against
the *sampling error of the estimate itself*. With roughly four independent
observations per phase, the standard error is usually the same size as the
shift being claimed - in which case the honest output is "no short signal is
resolvable here", not a short.

Everything is causal. Per-phase expected returns at day ``t`` are estimated only
from forward windows that had already closed by ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .models import regime_hmm

DAYS_PER_YEAR = 365.0

# Defaults are deliberately unfavourable to trading. Perp funding is quoted as
# what longs pay shorts, so a positive number is a headwind for longs and a
# tailwind for shorts - that is the one thing working in a short's favour, and
# it is nowhere near enough.
DEFAULT_FUNDING_ANNUAL = 0.10
DEFAULT_ANNUAL_COST = 0.02
DEFAULT_KELLY_FRACTION = 0.25

MAX_LONG = 1.00
MAX_SHORT = -0.25   # hard cap: a phase signal never justifies a large short


@dataclass
class PositionAdvice:
    as_of: str
    action: str
    target_exposure: float
    confidence: float
    expected_annual_return: float
    refused_short: bool = False
    rationale: list = field(default_factory=list)
    phase_stats: dict = field(default_factory=dict)
    hurdle: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "action": self.action,
            "target_exposure": self.target_exposure,
            "confidence": self.confidence,
            "expected_annual_return": self.expected_annual_return,
            "refused_short": self.refused_short,
            "rationale": self.rationale,
            "phase_stats": self.phase_stats,
            "hurdle": self.hurdle,
        }


# ---------------------------------------------------------------------------
# Base rates
# ---------------------------------------------------------------------------


def unconditional_drift(price: pd.Series, lookback_years: float | None = None) -> dict:
    """Annualised log drift and volatility - what a short has to overcome.

    The standard error matters as much as the point estimate: it is
    ``sigma / sqrt(years)``, and with Bitcoin's volatility even the
    *unconditional* drift is only a few standard errors from zero.
    """
    log_price = np.log(price.astype(float))
    daily = log_price.diff().dropna()
    if lookback_years:
        daily = daily.iloc[-int(lookback_years * DAYS_PER_YEAR):]
    if len(daily) < 100:
        return {"error": f"only {len(daily)} daily observations"}

    drift = float(daily.mean() * DAYS_PER_YEAR)
    vol = float(daily.std() * np.sqrt(DAYS_PER_YEAR))
    years = len(daily) / DAYS_PER_YEAR
    standard_error = vol / np.sqrt(max(years, 1e-9))

    return {
        "annual_log_drift": drift,
        "annual_vol": vol,
        "years": years,
        "drift_standard_error": standard_error,
        "drift_t_stat": drift / standard_error if standard_error > 0 else float("nan"),
        "sharpe": drift / vol if vol > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Phase-conditional returns, out of sample
# ---------------------------------------------------------------------------


def _block_bootstrap_phase_means(data: pd.DataFrame, block: int, n_boot: int,
                                 seed: int) -> dict:
    """Moving-block bootstrap standard errors for per-phase means.

    Plain i.i.d. bootstrap would be wrong here: forward windows overlap, so
    adjacent rows are nearly the same observation. Blocks the length of the
    horizon keep that dependence intact.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    if n <= block:
        return {}

    n_blocks = int(np.ceil(n / block))
    phases = sorted(data["phase"].unique())
    samples = {phase: [] for phase in phases}

    for _ in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        index = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        replicate = data.iloc[index]
        grouped = replicate.groupby("phase")["annualised"].mean()
        for phase in phases:
            samples[phase].append(float(grouped.get(phase, np.nan)))

    out = {}
    for phase, values in samples.items():
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            continue
        out[phase] = {
            "bootstrap_se": float(arr.std(ddof=1)),
            "ci_low": float(np.percentile(arr, 5)),
            "ci_high": float(np.percentile(arr, 95)),
        }
    return out


def phase_forward_returns(frame: pd.DataFrame, horizon: int = 90,
                          min_train: int = 1200, refit_every: int = 180,
                          n_boot: int = 400, seed: int = 0) -> dict:
    """Out-of-sample phase calls joined to realised forward returns.

    Uses :func:`regime_hmm.walk_forward`, so the phase label on day ``t`` came
    from a model that had not seen day ``t``. This is the table that decides
    whether a phase call is worth acting on at all.
    """
    walk = regime_hmm.walk_forward(frame, min_train=min_train, refit_every=refit_every)
    phases = regime_hmm.disambiguate(walk, frame)
    call = phases.idxmax(axis=1)

    log_price = np.log(frame["price"].astype(float))
    forward = (log_price.shift(-horizon) - log_price).reindex(call.index)

    data = pd.DataFrame({"phase": call, "forward": forward}).dropna()
    if data.empty:
        return {"error": "no out-of-sample phase calls with forward returns"}
    data["annualised"] = data["forward"] * (DAYS_PER_YEAR / horizon)

    grouped = data.groupby("phase")["annualised"]
    errors = _block_bootstrap_phase_means(data, block=horizon, n_boot=n_boot, seed=seed)

    stats = {}
    for phase, series in grouped:
        effective_n = len(series) / horizon
        entry = {
            "n_days": int(len(series)),
            "effective_n": round(effective_n, 1),
            "annualised_mean": float(series.mean()),
            "annualised_median": float(series.median()),
            "annualised_std": float(series.std()),
            "hit_rate_positive": float((series > 0).mean()),
        }
        entry.update(errors.get(phase, {}))
        stats[phase] = entry

    return {
        "horizon_days": horizon,
        "n_oos_days": int(len(data)),
        "by_phase": stats,
        "warning": (
            "effective_n is overlapping windows divided by the horizon. A phase "
            "with effective_n below ~5 has a mean you cannot distinguish from "
            "the unconditional drift."
        ),
    }


# ---------------------------------------------------------------------------
# The short hurdle
# ---------------------------------------------------------------------------


def short_hurdle(price: pd.Series, phase_stats: dict,
                 funding_annual: float = DEFAULT_FUNDING_ANNUAL,
                 annual_cost: float = DEFAULT_ANNUAL_COST) -> dict:
    """When, if ever, does the evidence justify a short?

    A short earns funding (longs pay shorts) and pays trading costs, so it is
    positive-EV when the conditional expected return satisfies::

        mu_conditional < funding_annual - annual_cost

    The gate applied here is stricter than the point estimate: the *upper* end
    of the conditional mean's confidence interval must also clear the
    break-even. That is what stops a noisy negative reading from becoming a
    position.
    """
    base = unconditional_drift(price)
    if "error" in base:
        return base

    breakeven = funding_annual - annual_cost
    required_shift = breakeven - base["annual_log_drift"]

    rows = {}
    for phase, stats in phase_stats.items():
        mean = stats.get("annualised_mean", float("nan"))
        standard_error = stats.get("bootstrap_se", float("nan"))
        upper = mean + 2.0 * standard_error if np.isfinite(standard_error) else float("nan")
        observed_shift = mean - base["annual_log_drift"]

        rows[phase] = {
            "annualised_mean": mean,
            "bootstrap_se": standard_error,
            "upper_2se": upper,
            "breakeven_for_short": breakeven,
            "shift_vs_unconditional": observed_shift,
            "required_shift": required_shift,
            "point_estimate_clears": bool(np.isfinite(mean) and mean < breakeven),
            "resolvable": bool(
                np.isfinite(standard_error)
                and standard_error > 0
                and abs(observed_shift) > 2.0 * standard_error
            ),
            "short_justified": bool(
                np.isfinite(upper) and upper < breakeven
            ),
        }

    return {
        "unconditional": base,
        "funding_annual": funding_annual,
        "annual_cost": annual_cost,
        "breakeven_conditional_return": breakeven,
        "required_shift_from_unconditional": required_shift,
        "by_phase": rows,
        "reading": (
            "required_shift is how far below the unconditional drift a phase's "
            "expected return must sit before shorting it beats sitting flat. "
            "Compare it against bootstrap_se: when the required shift is not "
            "several standard errors, no phase call can carry a short."
        ),
    }


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def target_exposure(phase_probabilities: dict, phase_stats: dict, vol: float,
                    confidence: float, hurdle: dict | None = None,
                    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
                    max_long: float = MAX_LONG,
                    max_short: float = MAX_SHORT) -> tuple:
    """Fractional-Kelly exposure from the phase posterior.

    ``f* = mu / sigma^2`` under log utility; a quarter-Kelly haircut covers
    parameter uncertainty, and the ensemble's own confidence scales it again.
    Shorts are gated on :func:`short_hurdle` rather than on the sign of ``mu``.
    """
    rationale: list = []

    expected = 0.0
    total_weight = 0.0
    for phase, probability in phase_probabilities.items():
        if not np.isfinite(probability) or probability <= 0:
            continue
        stats = phase_stats.get(phase)
        if not stats or not np.isfinite(stats.get("annualised_mean", np.nan)):
            continue
        expected += probability * stats["annualised_mean"]
        total_weight += probability
    if total_weight <= 0:
        return 0.0, float("nan"), ["no phase has a usable expected return; flat"], False
    expected /= total_weight

    if not np.isfinite(vol) or vol <= 0:
        return 0.0, expected, ["volatility estimate unusable; flat"], False

    kelly = expected / (vol ** 2)
    exposure = kelly_fraction * kelly
    rationale.append(
        f"posterior-weighted expected return {expected:+.1%}/yr, vol {vol:.0%} "
        f"-> full Kelly {kelly:+.2f}, quarter-Kelly {exposure:+.2f}"
    )

    # Clip to the position limits BEFORE the confidence haircut. Applying the
    # haircut to an unclipped Kelly lets a wildly oversized raw number survive
    # a 90% confidence cut and still hit the cap, which makes the haircut a
    # no-op exactly when it matters most.
    clipped = float(np.clip(exposure, max_short, max_long))
    if clipped != exposure:
        rationale.append(f"clipped to position limits: {exposure:+.2f} -> {clipped:+.2f}")
        exposure = clipped

    if np.isfinite(confidence):
        scale = max(min(confidence / 0.5, 1.0), 0.0)
        exposure *= scale
        rationale.append(
            f"scaled by ensemble confidence {confidence:.0%} (factor {scale:.2f})"
        )

    refused_short = False
    if exposure < 0:
        justified = False
        if hurdle:
            dominant = max(phase_probabilities, key=lambda p: phase_probabilities.get(p, 0))
            row = hurdle.get("by_phase", {}).get(dominant, {})
            justified = bool(row.get("short_justified"))
            if not justified:
                refused_short = True
                rationale.append(
                    f"short suppressed: {dominant} expected return "
                    f"{row.get('annualised_mean', float('nan')):+.1%} +/- "
                    f"{row.get('bootstrap_se', float('nan')):.1%} does not clear "
                    f"break-even {row.get('breakeven_for_short', float('nan')):+.1%} "
                    f"with 2 standard errors to spare"
                )
                exposure = 0.0
        else:
            refused_short = True
            exposure = 0.0
            rationale.append("short suppressed: no hurdle analysis available")

    exposure = float(np.clip(exposure, max_short, max_long))
    return exposure, expected, rationale, refused_short


def describe(exposure: float) -> str:
    if exposure >= 0.75:
        return "LONG (full size)"
    if exposure >= 0.25:
        return "LONG (reduced)"
    if exposure > -0.05:
        return "FLAT / minimal"
    return "SHORT (small)"


def recommend(frame: pd.DataFrame, verdict, horizon: int = 90,
              funding_annual: float = DEFAULT_FUNDING_ANNUAL,
              annual_cost: float = DEFAULT_ANNUAL_COST,
              kelly_fraction: float = DEFAULT_KELLY_FRACTION) -> PositionAdvice:
    """Full position read from an ensemble verdict."""
    returns = phase_forward_returns(frame, horizon=horizon)
    if "error" in returns:
        return PositionAdvice(
            as_of=verdict.as_of, action="FLAT / minimal", target_exposure=0.0,
            confidence=verdict.confidence, expected_annual_return=float("nan"),
            rationale=[f"no out-of-sample phase statistics: {returns['error']}"],
        )

    stats = returns["by_phase"]
    hurdle = short_hurdle(frame["price"], stats, funding_annual, annual_cost)

    vol = float(frame["realized_vol_30"].dropna().iloc[-1])
    exposure, expected, rationale, refused = target_exposure(
        verdict.probabilities, stats, vol, verdict.confidence, hurdle, kelly_fraction
    )

    if not verdict.cycle_informative:
        exposure *= 0.5
        rationale.append("halved: cycle-existence tests did not clear their nulls")
    if verdict.novelty_flag:
        exposure *= 0.5
        rationale.append("halved: current regime sits outside historical experience")

    return PositionAdvice(
        as_of=verdict.as_of,
        action=describe(exposure),
        target_exposure=round(float(exposure), 3),
        confidence=verdict.confidence,
        expected_annual_return=expected,
        refused_short=refused,
        rationale=rationale,
        phase_stats=stats,
        hurdle=hurdle,
    )


# ---------------------------------------------------------------------------
# Does any of this beat just holding?
# ---------------------------------------------------------------------------


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def backtest(frame: pd.DataFrame, horizon: int = 90,
             kelly_fraction: float = DEFAULT_KELLY_FRACTION,
             funding_annual: float = DEFAULT_FUNDING_ANNUAL,
             roundtrip_cost: float = 0.001,
             min_train: int = 1200, refit_every: int = 180,
             allow_short: bool = False) -> dict:
    """Walk-forward strategy versus buy-and-hold, after costs and funding.

    Per-phase expected returns are estimated causally: on day ``t`` only forward
    windows that had already closed by ``t`` contribute. The comparison that
    matters is against buy-and-hold, not against zero - beating cash is easy
    when the underlying compounds at 40%+.
    """
    walk = regime_hmm.walk_forward(frame, min_train=min_train, refit_every=refit_every)
    phases = regime_hmm.disambiguate(walk, frame)
    call = phases.idxmax(axis=1)

    log_price = np.log(frame["price"].astype(float))
    annualised = (log_price.shift(-horizon) - log_price) * (DAYS_PER_YEAR / horizon)

    # An observation starting at day i is only knowable at day i + horizon.
    observed = pd.DataFrame({"phase": call, "annualised": annualised.reindex(call.index)}).dropna()
    observed.index = observed.index + pd.Timedelta(days=horizon)

    expected_by_phase = {}
    for phase in observed["phase"].unique():
        series = observed.loc[observed["phase"] == phase, "annualised"]
        expected_by_phase[phase] = series.expanding(min_periods=horizon).mean()

    expected_frame = pd.DataFrame(expected_by_phase).reindex(frame.index).ffill()

    vol = frame["realized_vol_30"].reindex(frame.index)
    daily_return = log_price.diff()
    funding_daily = funding_annual / DAYS_PER_YEAR

    exposures = pd.Series(0.0, index=frame.index)
    for day in call.index:
        phase = call.loc[day]
        if phase not in expected_frame.columns:
            continue
        mu = expected_frame.at[day, phase]
        sigma = vol.get(day, np.nan)
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            continue
        exposure = kelly_fraction * mu / (sigma ** 2)
        low = MAX_SHORT if allow_short else 0.0
        exposures.at[day] = float(np.clip(exposure, low, MAX_LONG))

    active = exposures.loc[call.index[0]:] if len(call.index) else exposures
    aligned = active.reindex(frame.index).fillna(0.0)

    turnover = aligned.diff().abs().fillna(0.0)
    strategy_daily = (
        aligned.shift(1).fillna(0.0) * daily_return
        - turnover * roundtrip_cost
        - aligned.shift(1).fillna(0.0) * funding_daily
    )
    strategy_daily = strategy_daily.loc[active.index[0]:].dropna() if len(active) else strategy_daily.dropna()
    hold_daily = daily_return.reindex(strategy_daily.index).fillna(0.0)

    def summarise(daily: pd.Series, name: str) -> dict:
        equity = np.exp(daily.cumsum())
        years = len(daily) / DAYS_PER_YEAR
        total = float(equity.iloc[-1]) if len(equity) else float("nan")
        annual = float(daily.mean() * DAYS_PER_YEAR)
        sigma = float(daily.std() * np.sqrt(DAYS_PER_YEAR))
        return {
            "name": name,
            "years": round(years, 1),
            "total_multiple": total,
            "annual_log_return": annual,
            "annual_vol": sigma,
            "sharpe": annual / sigma if sigma > 0 else float("nan"),
            "max_drawdown": _max_drawdown(equity) if len(equity) else float("nan"),
        }

    strategy = summarise(strategy_daily, "phase strategy")
    hold = summarise(hold_daily, "buy and hold")

    return {
        "strategy": strategy,
        "buy_and_hold": hold,
        "beats_hold_on_total": strategy["total_multiple"] > hold["total_multiple"],
        "beats_hold_on_sharpe": strategy["sharpe"] > hold["sharpe"],
        "average_exposure": float(aligned.loc[strategy_daily.index].mean()),
        "allow_short": allow_short,
        "note": (
            "Costs and funding are charged. A strategy that wins on Sharpe but "
            "loses on total return is trading upside for smoothness - decide "
            "which you wanted before reading this as a win."
        ),
    }


# ---------------------------------------------------------------------------
# The part that needs no live data
# ---------------------------------------------------------------------------


def historical_drift_estimates() -> dict:
    """Annualised drift measured between cycle anchors.

    Needs no price series - only the low/top table in :mod:`btc_cycle.config`.
    Low-to-low spans are the honest ones: any span ending at a cycle top is
    anchored on the single highest print in four years and reads high.
    """
    rows = []
    for start in config.CYCLES:
        for end in config.CYCLES:
            if end.low_date <= start.low_date:
                continue
            years = (end.low_date - start.low_date).days / DAYS_PER_YEAR
            drift = float(np.log(end.low_price / start.low_price) / years)
            rows.append(
                {
                    "span": f"{start.low_date} -> {end.low_date}",
                    "kind": "low-to-low",
                    "years": round(years, 2),
                    "annual_log_drift": drift,
                }
            )
    for cycle in config.COMPLETED_CYCLES:
        years = cycle.low_to_top_days / DAYS_PER_YEAR
        rows.append(
            {
                "span": f"{cycle.low_date} -> {cycle.top_date}",
                "kind": "low-to-top (biased high)",
                "years": round(years, 2),
                "annual_log_drift": float(np.log(cycle.top_multiple) / years),
            }
        )

    low_to_low = [r["annual_log_drift"] for r in rows if r["kind"] == "low-to-low"]
    return {
        "spans": rows,
        "low_to_low_median": float(np.median(low_to_low)) if low_to_low else float("nan"),
        "low_to_low_min": float(np.min(low_to_low)) if low_to_low else float("nan"),
        "low_to_low_max": float(np.max(low_to_low)) if low_to_low else float("nan"),
    }


def hurdle_table(drift_annual: float = 0.40, vol_annual: float = 0.50,
                 funding_annual: float = DEFAULT_FUNDING_ANNUAL,
                 annual_cost: float = DEFAULT_ANNUAL_COST,
                 effective_n: tuple = (3, 5, 8, 12)) -> dict:
    """What a phase signal must be worth before a short becomes rational.

    Pure arithmetic on assumed parameters - no market data required. The
    output is the threshold conditional return, and the sampling error that
    threshold has to clear.
    """
    breakeven = funding_annual - annual_cost
    required_shift = breakeven - drift_annual

    rows = []
    for n in effective_n:
        standard_error = vol_annual / np.sqrt(n)
        # Gate: mean + 2 SE < breakeven  =>  mean < breakeven - 2 SE
        threshold = breakeven - 2.0 * standard_error
        rows.append(
            {
                "effective_n": n,
                "standard_error": standard_error,
                "required_point_estimate": threshold,
                "shift_from_unconditional": threshold - drift_annual,
                "shift_in_standard_errors": (drift_annual - threshold) / standard_error,
            }
        )

    return {
        "assumed_drift": drift_annual,
        "assumed_vol": vol_annual,
        "funding_annual": funding_annual,
        "annual_cost": annual_cost,
        "breakeven_conditional_return": breakeven,
        "required_shift_to_breakeven": required_shift,
        "by_effective_n": rows,
        "reading": (
            "required_point_estimate is how negative a phase's out-of-sample "
            "expected return must be before a short clears the gate. Compare it "
            "with what any phase has ever actually delivered."
        ),
    }


def historical_bear_legs() -> dict:
    """What the hindsight-labelled bear legs actually delivered.

    Computable from the anchor table alone. This is the other half of the short
    case: the hurdle is only interesting once you know whether any phase has
    ever cleared it.

    The catch is in the word *hindsight*. These are returns from the exact top
    to the exact low. A live phase model does not ring a bell at the top; it
    turns over some weeks later and turns back some weeks early, and the first
    weeks of a bear leg carry a disproportionate share of the decline. Treat
    these numbers as the ceiling on what a perfect detector could have earned,
    not as an expectation.
    """
    rows = []
    for i, cycle in enumerate(config.COMPLETED_CYCLES):
        following = config.CYCLES[i + 1] if i + 1 < len(config.CYCLES) else None
        if following is None:
            continue
        years = (following.low_date - cycle.top_date).days / DAYS_PER_YEAR
        total = float(np.log(following.low_price / cycle.top_price))
        rows.append(
            {
                "leg": f"{cycle.top_date} -> {following.low_date}",
                "years": round(years, 2),
                "drawdown": following.low_price / cycle.top_price - 1.0,
                "annual_log_return": total / years,
            }
        )

    annual = [r["annual_log_return"] for r in rows]
    return {
        "legs": rows,
        "median_annual_log_return": float(np.median(annual)) if annual else float("nan"),
        "caveat": (
            "top-to-low, labelled with hindsight. A live detector captures a "
            "fraction of this, and the fraction is the whole question."
        ),
    }


def short_case_summary(drift_annual: float = 0.40, vol_annual: float = 0.50,
                       funding_annual: float = DEFAULT_FUNDING_ANNUAL,
                       annual_cost: float = DEFAULT_ANNUAL_COST) -> dict:
    """The short case end to end, using no market data.

    Puts the hurdle and the historical bear legs side by side, then states what
    is left to establish - which is always the same thing: out-of-sample
    detection quality, and that needs live data.
    """
    hurdle = hurdle_table(drift_annual, vol_annual, funding_annual, annual_cost)
    bears = historical_bear_legs()

    # Threshold at a realistic effective sample size for one phase.
    row = next(r for r in hurdle["by_effective_n"] if r["effective_n"] == 8)
    threshold = row["required_point_estimate"]
    realised = bears["median_annual_log_return"]

    return {
        "hurdle": hurdle,
        "bear_legs": bears,
        "threshold_at_effective_n_8": threshold,
        "median_realised_bear_return": realised,
        "headroom": realised - threshold,
        "conclusion": (
            "Hindsight bear legs clear the hurdle with room to spare, so the "
            "short case does not fail on the arithmetic. It stands or falls "
            "entirely on whether an out-of-sample phase call identifies a bear "
            "leg early enough and cleanly enough to capture a usable fraction "
            "of it. That cannot be settled without live data."
            if np.isfinite(realised) and realised < threshold
            else "Check the inputs: realised bear returns did not clear the hurdle."
        ),
    }
