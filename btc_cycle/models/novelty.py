"""Regime-break detection: "or is there no cycle any more?"

Every model in this package is conditioned on the past three cycles. If today's
market does not resemble anything in that history, the phase reading is an
extrapolation dressed as a classification, and the right answer to the user's
question is the third branch: no usable cycle structure.

Two detectors, deliberately different in kind:

* :func:`mahalanobis_novelty` - is today's feature vector far from every
  historical regime centroid?
* :func:`likelihood_novelty` - does a fitted HMM find the recent window
  improbable compared with how it scored history?

They disagree usefully. The first catches a market at unprecedented *levels*;
the second catches unprecedented *dynamics* at familiar levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def mahalanobis_novelty(frame: pd.DataFrame, feature_names: list | None = None,
                        reference_end: str | pd.Timestamp | None = None,
                        window: int = 90) -> dict:
    """Distance from the recent window to the historical feature distribution."""
    from scipy.stats import chi2

    feature_names = feature_names or config.HMM_FEATURES
    usable = [n for n in feature_names if n in frame.columns]
    block = frame[usable].replace([np.inf, -np.inf], np.nan).dropna()
    if len(block) < 500:
        return {"error": f"only {len(block)} usable rows"}

    if reference_end is None:
        # Reference = everything before the running cycle's low.
        reference_end = pd.Timestamp(config.CURRENT_CYCLE.low_date)
    reference = block[block.index < pd.Timestamp(reference_end)]
    if len(reference) < 300:
        reference = block.iloc[: int(len(block) * 0.7)]

    mean = reference.mean().to_numpy()
    covariance = np.cov(reference.to_numpy(dtype=float), rowvar=False)
    covariance += np.eye(len(usable)) * 1e-8
    inverse = np.linalg.pinv(covariance)

    def distance(row: np.ndarray) -> float:
        delta = row - mean
        return float(delta @ inverse @ delta)

    recent = block.iloc[-window:]
    distances = np.array([distance(r) for r in recent.to_numpy(dtype=float)])
    historical = np.array([distance(r) for r in reference.to_numpy(dtype=float)])

    degrees = len(usable)
    median_distance = float(np.median(distances))
    p_value = float(1.0 - chi2.cdf(median_distance, df=degrees))
    percentile = float((historical < median_distance).mean())

    return {
        "window_days": window,
        "n_features": degrees,
        "median_mahalanobis": median_distance,
        "chi2_p_value": p_value,
        "percentile_vs_history": percentile,
        "historical_median": float(np.median(historical)),
        "verdict": (
            "recent regime is outside historical experience"
            if percentile > 0.99
            else "recent regime is within historical experience"
        ),
    }


def likelihood_novelty(result, frame: pd.DataFrame, window: int = 90) -> dict:
    """Per-day HMM log-likelihood of the recent window vs its own history.

    Takes the :class:`~btc_cycle.models.regime_hmm.HMMResult` returned by
    ``fit``. A recent window in the bottom few percent of the historical
    likelihood distribution means the model is being asked to explain something
    it was never fitted on.
    """
    from .regime_hmm import _log_emission

    usable = result.feature_names
    block = frame[usable].replace([np.inf, -np.inf], np.nan).dropna()
    if len(block) < window + 300:
        return {"error": f"only {len(block)} usable rows"}

    scaler = getattr(result.model, "_btc_scaler", None)
    if scaler is None:
        mean, std = block.mean().to_numpy(), block.std().replace(0.0, 1.0).to_numpy()
    else:
        mean, std = scaler
    X = (block.to_numpy(dtype=float) - mean) / std

    log_b = _log_emission(result.model, X)
    # Marginal per-day likelihood under the stationary state distribution.
    transmat = np.asarray(result.model.transmat_)
    eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
    stationary = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    stationary = np.abs(stationary) / np.abs(stationary).sum()

    shift = log_b.max(axis=1, keepdims=True)
    per_day = np.log((np.exp(log_b - shift) * stationary).sum(axis=1)) + shift.ravel()
    series = pd.Series(per_day, index=block.index)

    recent = float(series.iloc[-window:].mean())
    historical = series.iloc[:-window]
    percentile = float((historical < recent).mean())

    return {
        "window_days": window,
        "recent_mean_loglik": recent,
        "historical_mean_loglik": float(historical.mean()),
        "historical_percentile": percentile,
        "verdict": (
            "recent data poorly explained by fitted regimes"
            if percentile < 0.05
            else "recent data consistent with fitted regimes"
        ),
    }


def structural_break_scan(frame: pd.DataFrame, column: str = "ret_90",
                          min_segment: int = 365) -> dict:
    """Single most likely variance/mean break point, by a CUSUM-style scan.

    Crude on purpose - one break, no multiple-testing correction. It answers
    "if something changed, roughly when?", not "did something change?".
    """
    series = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
    values = series.to_numpy(dtype=float)
    n = len(values)
    if n < 3 * min_segment:
        return {"error": f"only {n} rows"}

    best = None
    total_mean = values.mean()
    for cut in range(min_segment, n - min_segment, 15):
        left, right = values[:cut], values[cut:]
        pooled = (
            len(left) * (left.mean() - total_mean) ** 2
            + len(right) * (right.mean() - total_mean) ** 2
        )
        statistic = pooled / (values.var() + 1e-12)
        if best is None or statistic > best[0]:
            best = (statistic, cut)

    statistic, cut = best
    return {
        "column": column,
        "break_date": str(series.index[cut].date()),
        "statistic": float(statistic),
        "mean_before": float(values[:cut].mean()),
        "mean_after": float(values[cut:].mean()),
        "note": "descriptive scan; no multiple-testing correction applied",
    }
