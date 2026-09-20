"""Combine the models into one phase posterior - and gate it on trust.

The combination rule is deliberately plain: an equally weighted average of the
models that reported, then a confidence haircut driven by how much the models
disagree and by whether the novelty detectors think the current market is
inside historical experience at all.

Learning the ensemble weights would be the natural next step and is the wrong
one here. Weights fitted on three cycles are three numbers fitted to three
observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, features, labels
from .models import analog, composite, cycle_test, novelty, regime_hmm, supervised


@dataclass
class PhaseVerdict:
    as_of: str
    probabilities: dict
    top_phase: str
    confidence: float
    agreement: float
    novelty_flag: bool
    cycle_informative: bool
    components: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "probabilities": self.probabilities,
            "top_phase": self.top_phase,
            "top_phase_zh": config.PHASE_ZH.get(self.top_phase, self.top_phase),
            "confidence": self.confidence,
            "agreement": self.agreement,
            "novelty_flag": self.novelty_flag,
            "cycle_informative": self.cycle_informative,
            "components": self.components,
            "notes": self.notes,
        }


def _normalise(distribution: dict) -> dict:
    values = np.array([distribution.get(p, 0.0) for p in config.PHASES], dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    total = values.sum()
    if total <= 0:
        return {p: float("nan") for p in config.PHASES}
    return {p: float(v / total) for p, v in zip(config.PHASES, values)}


def _entropy_agreement(distributions: list) -> float:
    """1 - normalised Jensen-Shannon spread. 1.0 = models agree exactly."""
    if len(distributions) < 2:
        return float("nan")
    matrix = np.array([[d.get(p, 0.0) for p in config.PHASES] for d in distributions])
    matrix = np.nan_to_num(matrix, nan=0.0)
    rows = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, rows, out=np.zeros_like(matrix), where=rows > 0)

    mean = matrix.mean(axis=0)

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    js = entropy(mean) - np.mean([entropy(row) for row in matrix])
    return float(max(0.0, 1.0 - js / np.log(len(config.PHASES))))


def evaluate(frame: pd.DataFrame, fast: bool = False,
             run_supervised: bool = True) -> PhaseVerdict:
    """Full read on the latest row of ``frame`` (output of :func:`features.build`)."""
    notes: list = []
    components: dict = {}
    distributions: list = []

    as_of = str(frame.index[-1].date())

    # --- 1. Composite heat ------------------------------------------------
    try:
        heat_frame = composite.heat(frame)
        directional = composite.with_direction(frame, heat_frame)
        latest_heat = float(directional["heat"].dropna().iloc[-1])
        heat_distribution = composite.phase_probabilities(latest_heat)
        if directional["drawdown"].iloc[-1] < -0.25 and not bool(directional["rising"].iloc[-1]):
            heat_distribution = {**{p: 0.0 for p in config.PHASES}, config.BEAR: 1.0}
            notes.append("composite forced to bear: heat falling from a >25% drawdown")
        components["composite"] = {
            "heat": latest_heat,
            "heat_percentile_note": "1.0 = hottest day in history to date",
            "n_signals": int(directional["n_signals"].iloc[-1]),
            "phase": directional["phase"].iloc[-1],
            "distribution": heat_distribution,
        }
        distributions.append(heat_distribution)
    except Exception as exc:
        notes.append(f"composite unavailable: {exc}")

    # --- 2. HMM -----------------------------------------------------------
    hmm_result = None
    try:
        hmm_result = regime_hmm.fit(frame)
        phases = regime_hmm.disambiguate(hmm_result.filtered, frame)
        latest = phases.dropna().iloc[-1].to_dict()
        state = int(np.argmax(hmm_result.filtered.iloc[-1].to_numpy()))
        components["hmm"] = {
            "distribution": latest,
            "dominant_state": state,
            "state_occupancy": hmm_result.state_summary["occupancy"].round(3).to_dict(),
            "expected_remaining_days_in_state": round(
                hmm_result.expected_remaining_days(state), 1
            ),
            "log_likelihood": hmm_result.log_likelihood,
            "caveat": "filtered (causal) probabilities; full-sample fit",
        }
        distributions.append(latest)
    except Exception as exc:
        notes.append(f"hmm unavailable: {exc}")

    # --- 3. Supervised ----------------------------------------------------
    if run_supervised:
        try:
            fitted = supervised.fit_current(frame)
            if "probabilities" in fitted:
                distribution = _normalise(fitted["probabilities"])
                components["supervised"] = {
                    "distribution": distribution,
                    "validation": supervised.leave_one_cycle_out(frame),
                }
                distributions.append(distribution)
        except Exception as exc:
            notes.append(f"supervised unavailable: {exc}")

    # --- 4. Analogs (context, not a vote) ---------------------------------
    try:
        matches = analog.match_current(frame["price"])
        components["analog"] = {
            "summary": analog.summarise(matches),
            "matches": [m.__dict__ for m in matches],
            "halving_clock": analog.halving_clock(),
            "amplitude_decay": analog.amplitude_decay(),
        }
        notes.append("analog block is context only and carries no vote (n=3 prior cycles)")
    except Exception as exc:
        notes.append(f"analog unavailable: {exc}")

    # --- 5. Is there a cycle at all? --------------------------------------
    cycle_informative = True
    try:
        spectral = cycle_test.spectral_test(frame["price"], n_surrogates=100 if fast else 500)
        predictive = cycle_test.predictive_test(frame)
        components["cycle_test"] = {"spectral": spectral, "predictive": predictive}
        spectral_ok = spectral.get("p_value", 1.0) <= 0.10
        predictive_ok = predictive.get("delta_r2", -1.0) > 0
        cycle_informative = bool(spectral_ok or predictive_ok)
        if not cycle_informative:
            notes.append(
                "NO-CYCLE branch: the ~4y peak does not beat an AR(1) red-noise "
                "null AND halving harmonics add no out-of-sample skill. Phase "
                "labels above describe where price sits, not where it is going. "
                "Check the power analysis before reading this as 'no cycle' "
                "rather than 'not enough data'."
            )
    except Exception as exc:
        notes.append(f"cycle tests unavailable: {exc}")

    # --- 6. Novelty -------------------------------------------------------
    novelty_flag = False
    try:
        mahalanobis = novelty.mahalanobis_novelty(frame)
        components["novelty"] = {"mahalanobis": mahalanobis}
        if hmm_result is not None:
            components["novelty"]["likelihood"] = novelty.likelihood_novelty(
                hmm_result, frame
            )
        novelty_flag = bool(mahalanobis.get("percentile_vs_history", 0.0) > 0.99)
        if novelty_flag:
            notes.append(
                "Novelty flag: the last 90 days sit outside the feature "
                "distribution of every prior cycle. Treat the phase call as "
                "extrapolation."
            )
    except Exception as exc:
        notes.append(f"novelty unavailable: {exc}")

    # --- Combine ----------------------------------------------------------
    if not distributions:
        raise RuntimeError("no model produced a phase distribution")

    stacked = np.array([[d.get(p, 0.0) for p in config.PHASES] for d in distributions])
    stacked = np.nan_to_num(stacked, nan=0.0)
    combined = _normalise(dict(zip(config.PHASES, stacked.mean(axis=0))))

    agreement = _entropy_agreement(distributions)
    top_phase = max(combined, key=lambda p: combined[p] if np.isfinite(combined[p]) else -1)

    confidence = float(combined[top_phase])
    if np.isfinite(agreement):
        confidence *= agreement
    if novelty_flag:
        confidence *= 0.5
    if not cycle_informative:
        confidence *= 0.5

    return PhaseVerdict(
        as_of=as_of,
        probabilities=combined,
        top_phase=top_phase,
        confidence=confidence,
        agreement=agreement,
        novelty_flag=novelty_flag,
        cycle_informative=cycle_informative,
        components=components,
        notes=notes,
    )


def backtest_phase_calls(frame: pd.DataFrame, horizon: int = 180) -> pd.DataFrame:
    """What did each phase call actually imply for forward returns?

    Uses out-of-sample walk-forward HMM states, joined to realised forward
    returns. This is the table to look at before believing any phase label:
    if 爆发中 and 爆发前 have overlapping forward-return distributions, the
    distinction is not paying for itself.
    """
    walk = regime_hmm.walk_forward(frame)
    phases = regime_hmm.disambiguate(walk, frame)
    call = phases.idxmax(axis=1)
    forward = labels.forward_return(frame["price"], horizon).reindex(call.index)

    data = pd.DataFrame({"phase": call, "forward_return": forward}).dropna()
    grouped = data.groupby("phase")["forward_return"]
    out = pd.DataFrame(
        {
            "n_days": grouped.size(),
            "mean_fwd_log_return": grouped.mean(),
            "median_fwd_log_return": grouped.median(),
            "std": grouped.std(),
            "hit_rate_positive": grouped.apply(lambda s: float((s > 0).mean())),
        }
    )
    out["effective_n"] = (out["n_days"] / horizon).round(1)
    return out
