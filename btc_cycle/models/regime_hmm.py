"""Gaussian hidden Markov model over cycle features.

This is the part that actually answers "which phase are we in, and how sure
can we be" as a probability rather than a vibe.

Two things here differ from the usual notebook version, and both matter more
than the model choice:

1. **Filtered, not smoothed, probabilities.** ``hmmlearn``'s ``predict_proba``
   returns P(state_t | *all* data, including t+1..T). On historical rows that
   is the future. Read that way, an HMM "calls" every top. :func:`filtered`
   runs the forward recursion only, so row t sees rows <= t and nothing else.

2. **Expanding-window refits.** A model fitted on the full sample has seen all
   four cycles before it labels 2015. :func:`walk_forward` refits periodically
   on data available at the time, which is the only version whose track record
   means anything.

State ordering is imposed after fitting (states come out of EM in arbitrary
order) by sorting on a severity score, so "state 0" is always the coldest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .. import config


@dataclass
class HMMResult:
    model: object
    feature_names: list
    state_order: np.ndarray                 # fitted index -> rank
    filtered: pd.DataFrame                  # causal P(state | data <= t)
    smoothed: Optional[pd.DataFrame] = None  # in-sample only, for inspection
    state_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    log_likelihood: float = float("nan")

    @property
    def transition_matrix(self) -> np.ndarray:
        raw = np.asarray(self.model.transmat_)
        order = self.state_order
        return raw[np.ix_(order, order)]

    def expected_remaining_days(self, state: int) -> float:
        """Mean residual time in a state, from its self-transition probability.

        Geometric duration: E[remaining] = 1 / (1 - p_ii). Memoryless, so it is
        a scale check ("weeks or quarters?"), not a countdown.
        """
        p_ii = float(self.transition_matrix[state, state])
        if p_ii >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - p_ii)


def _prepare(frame: pd.DataFrame, feature_names: list) -> tuple:
    usable = [n for n in feature_names if n in frame.columns]
    block = frame[usable].replace([np.inf, -np.inf], np.nan).dropna()
    return block, usable


def _log_emission(model, X: np.ndarray) -> np.ndarray:
    """log p(x_t | state) for each state, without touching hmmlearn internals."""
    from scipy.stats import multivariate_normal

    n_states = model.n_components
    means = np.asarray(model.means_)
    out = np.empty((X.shape[0], n_states))
    for state in range(n_states):
        cov = _covariance(model, state)
        out[:, state] = multivariate_normal.logpdf(
            X, mean=means[state], cov=cov, allow_singular=True
        )
    return out


def _covariance(model, state: int) -> np.ndarray:
    kind = model.covariance_type
    covars = np.asarray(model.covars_)
    if kind == "full":
        return covars[state]
    if kind == "diag":
        # hmmlearn already expands diag covars_ to (n, d, d); handle both shapes
        return covars[state] if covars.ndim == 3 else np.diag(covars[state])
    if kind == "spherical":
        dim = np.asarray(model.means_).shape[1]
        value = covars[state]
        return np.eye(dim) * (value if np.isscalar(value) else np.ravel(value)[0])
    if kind == "tied":
        return covars
    raise ValueError(f"unsupported covariance_type {kind!r}")


def filtered_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Forward recursion. Row t uses observations 0..t only."""
    log_b = _log_emission(model, X)
    n_obs, n_states = log_b.shape
    transmat = np.asarray(model.transmat_)
    startprob = np.asarray(model.startprob_)

    alpha = np.empty((n_obs, n_states))
    # Work in a shifted linear domain: subtract the row max before exponentiating
    # so a 6-dimensional Gaussian's tiny densities do not underflow to zero.
    b0 = np.exp(log_b[0] - log_b[0].max())
    current = startprob * b0
    total = current.sum()
    alpha[0] = current / total if total > 0 else np.full(n_states, 1.0 / n_states)

    for t in range(1, n_obs):
        b = np.exp(log_b[t] - log_b[t].max())
        current = (alpha[t - 1] @ transmat) * b
        total = current.sum()
        alpha[t] = current / total if total > 0 else alpha[t - 1]
    return alpha


def _order_states(model, X: np.ndarray, feature_names: list) -> np.ndarray:
    """Rank states coldest -> hottest so labels are stable across refits.

    Severity = standardised mean of the trend/valuation features, minus
    drawdown (a deep drawdown is cold). Purely a relabelling.
    """
    means = np.asarray(model.means_)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    z = (means - X.mean(axis=0)) / std

    weights = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        if name in ("ret_90", "ret_365", "log_mayer", "log_ma200w_mult", "mvrv"):
            weights[i] = 1.0
        elif name == "drawdown":
            weights[i] = 1.0   # drawdown is negative in bears -> lowers severity
        elif name == "realized_vol_30":
            weights[i] = 0.0   # vol is high at both ends; useless for ordering
    if weights.sum() == 0:
        weights[:] = 1.0

    severity = z @ weights
    return np.argsort(severity)


def fit(frame: pd.DataFrame, feature_names: list | None = None,
        n_states: int = config.DEFAULT_N_STATES, seed: int = 0,
        n_iter: int = 500, covariance_type: str = "full") -> HMMResult:
    """Fit an HMM on the whole frame and return causally filtered states."""
    from hmmlearn.hmm import GaussianHMM

    feature_names = feature_names or config.HMM_FEATURES
    block, usable = _prepare(frame, feature_names)
    if len(block) < 300:
        raise ValueError(f"only {len(block)} usable rows; need >= 300 to fit an HMM")

    X_raw = block.to_numpy(dtype=float)
    mean, std = X_raw.mean(axis=0), X_raw.std(axis=0)
    std[std == 0] = 1.0
    X = (X_raw - mean) / std

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=seed,
        tol=1e-4,
    )
    model.fit(X)

    order = _order_states(model, X, usable)
    inverse = np.argsort(order)  # fitted state -> rank

    alpha = filtered_probabilities(model, X)
    columns = [f"state_{i}" for i in range(n_states)]
    filt = pd.DataFrame(alpha[:, order], index=block.index, columns=columns)

    try:
        smooth = pd.DataFrame(model.predict_proba(X)[:, order], index=block.index,
                              columns=columns)
    except Exception:
        smooth = None

    summary = _summarise(block, alpha[:, order], usable)
    model._btc_scaler = (mean, std)  # remembered so walk_forward can reuse it

    return HMMResult(
        model=model,
        feature_names=usable,
        state_order=order,
        filtered=filt,
        smoothed=smooth,
        state_summary=summary,
        log_likelihood=float(model.score(X)),
    )


def _summarise(block: pd.DataFrame, probabilities: np.ndarray,
               names: list) -> pd.DataFrame:
    """Probability-weighted feature means per ordered state."""
    rows = []
    for state in range(probabilities.shape[1]):
        weights = probabilities[:, state]
        total = weights.sum()
        row = {"state": state, "occupancy": total / len(weights)}
        for name in names:
            values = block[name].to_numpy(dtype=float)
            row[name] = float((values * weights).sum() / total) if total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def walk_forward(frame: pd.DataFrame, feature_names: list | None = None,
                 n_states: int = config.DEFAULT_N_STATES,
                 min_train: int = 1200, refit_every: int = 180,
                 seed: int = 0) -> pd.DataFrame:
    """Out-of-sample filtered state probabilities.

    At each refit point the model is re-estimated on history only, then used to
    filter forward until the next refit. Nothing in row t depends on data after
    t - which is the whole point, and which is why the resulting track record
    looks so much worse than a full-sample fit.
    """
    from hmmlearn.hmm import GaussianHMM

    feature_names = feature_names or config.HMM_FEATURES
    block, usable = _prepare(frame, feature_names)
    if len(block) <= min_train:
        raise ValueError(f"{len(block)} usable rows, need more than min_train={min_train}")

    columns = [f"state_{i}" for i in range(n_states)]
    out = pd.DataFrame(np.nan, index=block.index, columns=columns)
    X_all = block.to_numpy(dtype=float)

    for start in range(min_train, len(block), refit_every):
        train = X_all[:start]
        mean, std = train.mean(axis=0), train.std(axis=0)
        std[std == 0] = 1.0

        model = GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=300, random_state=seed, tol=1e-4)
        try:
            model.fit((train - mean) / std)
        except Exception:
            continue

        order = _order_states(model, (train - mean) / std, usable)
        stop = min(start + refit_every, len(block))
        # Filter over history + the new window, then keep only the new window:
        # the forward recursion needs the earlier alphas to be correct, but
        # those earlier rows were already written by a previous (smaller) fit.
        window = (X_all[:stop] - mean) / std
        alpha = filtered_probabilities(model, window)
        out.iloc[start:stop] = alpha[start:stop][:, order]

    return out.dropna(how="all")


def phase_mapping(n_states: int = config.DEFAULT_N_STATES) -> dict:
    """Ordered state index -> phase name.

    With 4 states the coldest is the bear/accumulation floor and the hottest is
    the parabolic leg. The two middle states are the ambiguous ones, and which
    of them is 爆发前 versus 爆发后 depends on direction of travel, not on the
    state alone - see :func:`disambiguate`.
    """
    if n_states == 3:
        return {0: config.ACCUMULATION, 1: config.EARLY_BULL, 2: config.PARABOLIC}
    if n_states == 4:
        return {
            0: config.ACCUMULATION,
            1: config.EARLY_BULL,
            2: config.EARLY_BULL,
            3: config.PARABOLIC,
        }
    return {i: config.PHASES[min(i, len(config.PHASES) - 1)] for i in range(n_states)}


def disambiguate(probabilities: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Split the HMM's mid states into 爆发前 vs 爆发后 using direction.

    An HMM state is defined by the *distribution* of returns and valuation, so
    the same state covers the climb through a level and the slide back through
    it. Drawdown from the running ATH breaks the tie: still near the highs and
    it is pre-breakout, already 25%+ below and it is post.
    """
    mapping = phase_mapping(probabilities.shape[1])
    out = pd.DataFrame(0.0, index=probabilities.index, columns=config.PHASES)
    drawdown = frame["drawdown"].reindex(probabilities.index)
    momentum = frame["ret_90"].reindex(probabilities.index)

    post_like = ((drawdown < -0.25) & (momentum < 0)).fillna(False)

    for state in range(probabilities.shape[1]):
        phase = mapping[state]
        mass = probabilities.iloc[:, state]
        if phase in (config.EARLY_BULL, config.ACCUMULATION):
            out[config.BEAR] += mass.where(post_like, 0.0)
            out[phase] += mass.where(~post_like, 0.0)
        else:
            out[phase] += mass

    total = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(total, axis=0)
