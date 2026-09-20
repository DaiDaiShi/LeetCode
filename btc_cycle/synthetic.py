"""Synthetic price generator.

Used by the test suite so the package can be verified with no network, and
useful on its own: it is the only way to check whether these models detect a
cycle that is *known* to be there, and - more importantly - whether they
hallucinate one in a series that has none.

:func:`make_cyclical` plants a real cycle. :func:`make_acyclic` produces a
random walk with matched drift and volatility and no cycle at all. Any model
that reports a confident phase on ``make_acyclic`` output is broken, and the
test suite asserts exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def make_cyclical(n_days: int = 4400, start: str = "2013-01-01",
                  cycle_days: int = config.CYCLE_LENGTH_DAYS,
                  amplitude: float = 1.4, drift: float = 0.0009,
                  vol: float = 0.035, decay: float = 0.75,
                  seed: int = 0) -> pd.DataFrame:
    """Log price = drift + decaying sinusoid + noise.

    ``decay`` shrinks the amplitude each cycle, mimicking the observed
    diminishing-returns pattern.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days, dtype=float)
    cycle_number = t / cycle_days
    envelope = amplitude * (decay ** cycle_number)
    seasonal = envelope * np.sin(2 * np.pi * t / cycle_days - np.pi / 2)
    noise = np.cumsum(rng.normal(0.0, vol, n_days)) * 0.25
    log_price = np.log(100.0) + drift * t + seasonal + noise

    index = pd.date_range(start=start, periods=n_days, freq="D")
    price = pd.Series(np.exp(log_price), index=index, name="price")
    return _decorate(price, rng)


def make_acyclic(n_days: int = 4400, start: str = "2013-01-01",
                 drift: float = 0.0009, vol: float = 0.035,
                 seed: int = 0) -> pd.DataFrame:
    """Geometric random walk: same drift and volatility, no cycle."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, vol, n_days)
    log_price = np.log(100.0) + np.cumsum(steps)
    index = pd.date_range(start=start, periods=n_days, freq="D")
    price = pd.Series(np.exp(log_price), index=index, name="price")
    return _decorate(price, rng)


def _decorate(price: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Attach plausible on-chain companions so the full pipeline can run."""
    frame = price.to_frame()
    sma = price.rolling(400, min_periods=1).mean()
    # A realized-price stand-in: slow-moving cost basis that trails spot.
    frame["realized_price"] = sma * (1.0 + rng.normal(0, 0.01, len(price)).cumsum() * 0.001)
    frame["mvrv"] = frame["price"] / frame["realized_price"]
    frame["miner_revenue"] = frame["price"] * rng.lognormal(0.0, 0.15, len(price)) * 900.0
    return frame
