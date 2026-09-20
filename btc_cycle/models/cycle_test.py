"""Does the four-year cycle still carry information?

"Which phase are we in" is only a meaningful question if the phases exist. The
tests here try hard to say no.

The trap they are built around: a persistent, trending, mean-reverting series
produces a fat low-frequency peak in its periodogram *even with no cycle at
all*. Red noise looks periodic. So a raw FFT peak near 1400 days is not
evidence - it has to beat a null that shares the series' own autocorrelation.

Four independent angles:

* :func:`spectral_test` - is the ~4y spectral peak bigger than phase-randomised
  surrogates of the same series produce?
* :func:`predictive_test` - do halving-clock harmonics beat a momentum-only
  baseline *out of sample*?
* :func:`rolling_explanatory_power` - is the halving clock's explanatory power
  decaying over time, and is liquidity picking it up?
* :func:`amplitude_signal_to_noise` - has cycle amplitude decayed into the
  noise band, so that even a real cycle is no longer tradeable?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


# ---------------------------------------------------------------------------
# 1. Spectral test against phase-randomised surrogates
# ---------------------------------------------------------------------------


def _detrend_log_price(price: pd.Series) -> np.ndarray:
    """Log price with its linear (in time) trend removed.

    Bitcoin's log price has a huge secular drift; leaving it in dumps all the
    power at the lowest frequency and drowns anything cyclical.
    """
    y = np.log(price.astype(float).to_numpy())
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    return y - (slope * t + intercept)


def _fit_ar1(x: np.ndarray) -> tuple:
    """Least-squares AR(1) fit, returning ``(phi, sigma)``."""
    a, b = x[:-1], x[1:]
    denominator = float(np.dot(a, a))
    phi = float(np.dot(a, b) / denominator) if denominator > 0 else 0.0
    phi = min(max(phi, -0.9999), 0.9999)
    sigma = float((b - phi * a).std(ddof=1))
    return phi, sigma


def _ar1_surrogate(n: int, phi: float, sigma: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Red-noise surrogate, detrended the same way the observed series was."""
    noise = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = noise[0] / np.sqrt(max(1.0 - phi ** 2, 1e-6))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    return x - (slope * t + intercept)


def _band_power(x: np.ndarray, target_days: int, tolerance: float,
                min_period: int = 90) -> tuple:
    """Peak power in the target band, as a share of sub-``min_period`` power."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), np.inf)
    band = (periods >= target_days * (1 - tolerance)) & (periods <= target_days * (1 + tolerance))
    if not band.any():
        return float("nan"), float("nan")

    power = np.abs(np.fft.rfft(x)) ** 2
    denominator = power[(periods >= min_period) & np.isfinite(periods)].sum()
    if denominator <= 0:
        return float("nan"), float("nan")
    share = float(power[band].max() / denominator)
    peak_period = float(periods[band][int(np.argmax(power[band]))])
    return share, peak_period


def spectral_test(price: pd.Series, target_days: int = config.CYCLE_LENGTH_DAYS,
                  tolerance: float = 0.30, n_surrogates: int = 500,
                  seed: int = 0) -> dict:
    """Is the ~4y spectral peak bigger than red noise alone would produce?

    The null is a fitted AR(1) process - same persistence, no cycle.

    Note on a tempting wrong choice: phase-randomised (AAFT) surrogates are the
    usual recipe for surrogate testing, but they preserve the amplitude
    spectrum *exactly*, so any statistic computed from that spectrum is
    identical for data and surrogate and the test has zero power. AR(1) is the
    right null for "is this peak just persistence?".
    """
    x = _detrend_log_price(price)
    n = len(x)
    if n < 3 * target_days:
        return {"error": f"need >= {3 * target_days} days, have {n}"}

    observed, peak_period = _band_power(x, target_days, tolerance)
    if not np.isfinite(observed):
        return {"error": "target period outside resolvable band"}

    phi, sigma = _fit_ar1(x)
    rng = np.random.default_rng(seed)
    null = np.array(
        [_band_power(_ar1_surrogate(n, phi, sigma, rng), target_days, tolerance)[0]
         for _ in range(n_surrogates)]
    )
    null = null[np.isfinite(null)]
    p_value = float((null >= observed).mean()) if len(null) else float("nan")

    return {
        "null_model": "AR(1) red noise",
        "ar1_phi": phi,
        "observed_band_power": observed,
        "null_mean": float(null.mean()) if len(null) else float("nan"),
        "null_p95": float(np.percentile(null, 95)) if len(null) else float("nan"),
        "p_value": p_value,
        "peak_period_days": peak_period,
        "peak_period_years": peak_period / 365.25,
        "n_surrogates": int(len(null)),
        "n_observations": n,
        "n_cycles_observed": n / target_days,
        "verdict": (
            "cycle peak not distinguishable from red noise"
            if not np.isfinite(p_value) or p_value > 0.10
            else "cycle peak survives the red-noise null"
        ),
    }


def spectral_power_analysis(price: pd.Series,
                            amplitudes: tuple = (0.2, 0.4, 0.8, 1.2),
                            target_days: int = config.CYCLE_LENGTH_DAYS,
                            tolerance: float = 0.30, n_trials: int = 60,
                            n_surrogates: int = 120, alpha: float = 0.10,
                            seed: int = 0) -> dict:
    """How big would a cycle have to be for this much data to detect it?

    Plants a sinusoid of known log-amplitude into AR(1) noise matched to the
    real series, then runs :func:`spectral_test`'s statistic and counts how
    often it clears ``alpha``.

    This is the number that decides whether "no cycle detected" means "no
    cycle" or "not enough data". With ~4 cycles of history the second reading
    is usually the right one, and the table below says so in numbers.
    """
    x = _detrend_log_price(price)
    n = len(x)
    phi, sigma = _fit_ar1(x)
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)

    # Shared null distribution: the statistic under no-cycle AR(1).
    null = np.array(
        [_band_power(_ar1_surrogate(n, phi, sigma, rng), target_days, tolerance)[0]
         for _ in range(n_surrogates)]
    )
    null = null[np.isfinite(null)]
    if not len(null):
        return {"error": "null distribution degenerate"}
    threshold = float(np.percentile(null, 100 * (1 - alpha)))

    rows = []
    for amplitude in amplitudes:
        detections = 0
        for _ in range(n_trials):
            noise = _ar1_surrogate(n, phi, sigma, rng)
            planted = noise + amplitude * np.sin(2 * np.pi * t / target_days + rng.uniform(0, 2 * np.pi))
            slope, intercept = np.polyfit(t, planted, 1)
            planted = planted - (slope * t + intercept)
            statistic, _ = _band_power(planted, target_days, tolerance)
            if np.isfinite(statistic) and statistic >= threshold:
                detections += 1
        rows.append(
            {
                "log_amplitude": amplitude,
                "peak_to_trough_multiple": float(np.exp(2 * amplitude)),
                "detection_rate": detections / n_trials,
            }
        )

    return {
        "n_observations": n,
        "n_cycles_observed": n / target_days,
        "alpha": alpha,
        "critical_value": threshold,
        "power_by_amplitude": rows,
        "note": (
            "detection_rate is statistical power. Anything below ~0.8 means a "
            "negative result is uninformative: the cycle could be there and "
            "this much history simply cannot see it."
        ),
    }


def halving_phase_lock_test(frame: pd.DataFrame, horizon: int = 90,
                            n_bins: int = 20, n_surrogates: int = 500,
                            seed: int = 0) -> dict:
    """Is the pattern phase-locked to the halving specifically?

    Folds forward returns onto the halving clock and asks whether the spread
    across phase bins beats a null that keeps the return series intact and only
    rotates the clock. Circular rotation preserves every autocorrelation in the
    data and destroys only the alignment to the halving - which is exactly the
    hypothesis under test.
    """
    if "days_since_halving" not in frame.columns or "log_price" not in frame.columns:
        return {"error": "missing days_since_halving/log_price"}

    target = (frame["log_price"].shift(-horizon) - frame["log_price"])
    data = pd.concat([frame["days_since_halving"], target.rename("target")], axis=1).dropna()
    if len(data) < 3 * config.CYCLE_LENGTH_DAYS:
        return {"error": f"{len(data)} usable rows"}

    phase = (data["days_since_halving"].to_numpy() % config.CYCLE_LENGTH_DAYS)
    bins = np.minimum((phase / config.CYCLE_LENGTH_DAYS * n_bins).astype(int), n_bins - 1)
    values = data["target"].to_numpy(dtype=float)

    def statistic(series: np.ndarray) -> float:
        means = np.array(
            [series[bins == b].mean() if (bins == b).any() else np.nan for b in range(n_bins)]
        )
        means = means[np.isfinite(means)]
        return float(means.var()) if len(means) > 1 else float("nan")

    observed = statistic(values)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_surrogates):
        shift = int(rng.integers(1, len(values)))
        null.append(statistic(np.roll(values, shift)))
    null = np.array([v for v in null if np.isfinite(v)])
    p_value = float((null >= observed).mean()) if len(null) else float("nan")

    bin_means = {
        int(b * config.CYCLE_LENGTH_DAYS / n_bins): float(values[bins == b].mean())
        for b in range(n_bins)
        if (bins == b).any()
    }

    return {
        "horizon_days": horizon,
        "n_bins": n_bins,
        "observed_between_bin_variance": observed,
        "null_mean": float(null.mean()) if len(null) else float("nan"),
        "p_value": p_value,
        "mean_forward_return_by_days_since_halving": bin_means,
        "verdict": (
            "forward returns are not phase-locked to the halving"
            if not np.isfinite(p_value) or p_value > 0.10
            else "forward returns are phase-locked to the halving"
        ),
    }


# ---------------------------------------------------------------------------
# 2. Out-of-sample predictive test
# ---------------------------------------------------------------------------


def predictive_test(frame: pd.DataFrame, horizon: int = 90,
                    min_train: int = 1200, step: int = 90) -> dict:
    """Do halving harmonics add out-of-sample skill over a momentum baseline?

    Baseline features: 90d and 365d log returns.
    Cycle features: baseline + sin/cos of the halving clock + days since halving.

    Skill is expanding-window OOS R^2. Overlapping horizons make the *level*
    of R^2 optimistic, but both models suffer identically, so the comparison
    is the readable part.
    """
    from sklearn.linear_model import Ridge

    needed = ["ret_90", "ret_365", "halving_sin", "halving_cos", "days_since_halving", "log_price"]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        return {"error": f"missing features {missing}"}

    data = frame[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    data["target"] = data["log_price"].shift(-horizon) - data["log_price"]
    data = data.dropna()
    if len(data) <= min_train + step:
        return {"error": f"{len(data)} usable rows, need > {min_train + step}"}

    baseline_cols = ["ret_90", "ret_365"]
    cycle_cols = baseline_cols + ["halving_sin", "halving_cos", "days_since_halving"]

    records = []
    for start in range(min_train, len(data) - step, step):
        # Purge the horizon: the last `horizon` training rows have targets that
        # overlap the test window. Leaving them in leaks the answer.
        train = data.iloc[: max(start - horizon, 1)]
        test = data.iloc[start : start + step]
        if len(train) < 200 or test.empty:
            continue

        row = {"date": test.index[0]}
        for name, cols in (("baseline", baseline_cols), ("cycle", cycle_cols)):
            mean = train[cols].mean()
            std = train[cols].std().replace(0.0, 1.0)
            model = Ridge(alpha=1.0).fit((train[cols] - mean) / std, train["target"])
            prediction = model.predict((test[cols] - mean) / std)
            row[f"{name}_sse"] = float(((test["target"] - prediction) ** 2).sum())
        row["naive_sse"] = float(((test["target"] - train["target"].mean()) ** 2).sum())
        records.append(row)

    if not records:
        return {"error": "no usable walk-forward windows"}

    results = pd.DataFrame(records)
    total_naive = results["naive_sse"].sum()

    def oos_r2(column: str) -> float:
        return float(1.0 - results[column].sum() / total_naive) if total_naive > 0 else float("nan")

    baseline_r2 = oos_r2("baseline_sse")
    cycle_r2 = oos_r2("cycle_sse")

    # Diebold-Mariano style sign test on per-window squared-error differences.
    diff = results["baseline_sse"] - results["cycle_sse"]
    wins = int((diff > 0).sum())
    from scipy.stats import binomtest

    p_value = float(binomtest(wins, len(diff), 0.5).pvalue)

    return {
        "horizon_days": horizon,
        "n_windows": len(results),
        "baseline_oos_r2": baseline_r2,
        "cycle_oos_r2": cycle_r2,
        "delta_r2": cycle_r2 - baseline_r2,
        "windows_cycle_wins": wins,
        "sign_test_p": p_value,
        "verdict": (
            "halving clock adds no out-of-sample skill"
            if cycle_r2 <= baseline_r2 or p_value > 0.10
            else "halving clock adds out-of-sample skill"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Is the halving clock being displaced by liquidity?
# ---------------------------------------------------------------------------


def rolling_explanatory_power(frame: pd.DataFrame, window: int = 730,
                              horizon: int = 90) -> pd.DataFrame:
    """Rolling R^2 of forward returns on the halving clock vs on liquidity.

    If the halving column trends to zero while the liquidity column does not,
    that is the quantitative form of "it stopped being a halving cycle and
    became a liquidity cycle".
    """
    target = frame["log_price"].shift(-horizon) - frame["log_price"]
    blocks = {"halving": ["halving_sin", "halving_cos"]}
    if "m2_yoy" in frame.columns:
        blocks["liquidity"] = ["m2_yoy"]
    if "dxy_yoy" in frame.columns:
        blocks.setdefault("liquidity", []).append("dxy_yoy")

    out = pd.DataFrame(index=frame.index, columns=list(blocks), dtype=float)
    for name, cols in blocks.items():
        cols = [c for c in cols if c in frame.columns]
        if not cols:
            continue
        data = pd.concat([frame[cols], target.rename("target")], axis=1).dropna()
        if len(data) < window:
            continue
        values = []
        index = []
        for end in range(window, len(data), 30):
            chunk = data.iloc[end - window : end]
            X = np.column_stack([np.ones(len(chunk))] + [chunk[c].to_numpy() for c in cols])
            y = chunk["target"].to_numpy()
            coefficients, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            fitted = X @ coefficients
            ss_res = float(((y - fitted) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            values.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
            index.append(chunk.index[-1])
        out.loc[index, name] = values

    return out.dropna(how="all")


# ---------------------------------------------------------------------------
# 4. Has the amplitude decayed into the noise?
# ---------------------------------------------------------------------------


def amplitude_signal_to_noise(price: pd.Series) -> dict:
    """Compare per-cycle amplitude against ordinary volatility.

    A cycle you cannot separate from normal volatility is not actionable even
    if it is statistically real.
    """
    daily = np.log(price.astype(float)).diff()
    annual_vol = float(daily.std() * np.sqrt(365))

    rows = []
    for cycle in config.COMPLETED_CYCLES:
        if not cycle.top_multiple or not cycle.low_to_top_days:
            continue
        years = cycle.low_to_top_days / 365.25
        amplitude = float(np.log(cycle.top_multiple))
        rows.append(
            {
                "cycle": cycle.index,
                "log_amplitude": amplitude,
                "years": years,
                "annualised_drift": amplitude / years,
                "snr_vs_vol": (amplitude / years) / annual_vol if annual_vol else np.nan,
            }
        )

    table = pd.DataFrame(rows)
    trend = np.nan
    if len(table) >= 2:
        slope, _ = np.polyfit(table["cycle"], table["log_amplitude"], 1)
        trend = float(slope)

    return {
        "annualised_vol": annual_vol,
        "per_cycle": table.to_dict("records"),
        "log_amplitude_trend_per_cycle": trend,
        "note": (
            "Each cycle's low->top drift has fallen relative to volatility. "
            "When annualised drift stops clearing ~1 sigma, phase calls stop "
            "paying for themselves after costs."
        ),
    }


def run_all(frame: pd.DataFrame, price: pd.Series, fast: bool = False) -> dict:
    """Every cycle-existence test, collected."""
    surrogates = 100 if fast else 500
    return {
        "spectral": spectral_test(price, n_surrogates=surrogates),
        "power": spectral_power_analysis(
            price, n_trials=25 if fast else 60, n_surrogates=60 if fast else 120
        ),
        "phase_lock": halving_phase_lock_test(frame, n_surrogates=surrogates),
        "predictive": predictive_test(frame),
        "amplitude": amplitude_signal_to_noise(price),
    }
