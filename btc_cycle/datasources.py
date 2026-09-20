"""Data loading.

Three ways in, tried in order:

1. ``--csv`` / :func:`load_csv` - a file you exported yourself (Glassnode,
   CoinMetrics, TradingView, CryptoQuant). Always works, no network.
2. :func:`fetch_coinmetrics` - Coin Metrics' free community endpoint, which
   serves price, MVRV, realized cap and miner revenue with no API key.
3. :func:`fetch_blockchain_info` / :func:`fetch_bitstamp` - price-only
   fallbacks when the on-chain endpoint is unreachable.

Everything is cached under ``~/.cache/btc_cycle`` so repeated runs stay offline.

Any fetch can fail behind a corporate proxy or an egress policy. Failures raise
:class:`DataUnavailable` with the blocked host named, and the CLI degrades to
the price-only feature tier rather than silently inventing numbers.
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

CACHE_DIR = Path(os.environ.get("BTC_CYCLE_CACHE", Path.home() / ".cache" / "btc_cycle"))
CACHE_TTL_SECONDS = 12 * 3600

COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
BLOCKCHAIN_URL = "https://api.blockchain.info/charts/market-price"
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Coin Metrics community metrics used here. All are free-tier for BTC.
CM_METRICS = [
    "PriceUSD",        # daily close reference price
    "CapMVRVCur",      # MVRV ratio
    "CapRealUSD",      # realized cap
    "SplyCur",         # circulating supply
    "RevUSD",          # miner revenue, drives the Puell Multiple
]


class DataUnavailable(RuntimeError):
    """Raised when a remote source cannot be reached or returns nothing."""


@dataclass
class LoadResult:
    frame: pd.DataFrame
    source: str
    has_onchain: bool
    has_macro: bool
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.csv"


def _read_cache(name: str, ttl: Optional[float] = CACHE_TTL_SECONDS) -> Optional[pd.DataFrame]:
    path = _cache_path(name)
    if not path.exists():
        return None
    if ttl is not None and time.time() - path.stat().st_mtime > ttl:
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.set_index("date").sort_index()


def _write_cache(name: str, frame: pd.DataFrame) -> None:
    out = frame.copy()
    out.index.name = "date"
    out.to_csv(_cache_path(name))


def _get(url: str, params: dict, timeout: int = 30) -> "object":
    """GET returning parsed JSON, or raise DataUnavailable naming the host."""
    import requests  # imported lazily so offline tests need no network stack

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except Exception as exc:  # connection, proxy, DNS, TLS
        host = url.split("/")[2]
        raise DataUnavailable(f"{host} unreachable: {exc}") from exc
    if response.status_code != 200:
        host = url.split("/")[2]
        raise DataUnavailable(f"{host} returned HTTP {response.status_code}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise DataUnavailable(f"{url} returned non-JSON payload") from exc


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def fetch_coinmetrics(
    start: str = "2010-07-18",
    metrics: Iterable[str] = tuple(CM_METRICS),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily BTC price + on-chain metrics from the Coin Metrics community API."""
    if use_cache:
        cached = _read_cache("coinmetrics")
        if cached is not None:
            return cached

    rows: list = []
    params = {
        "assets": "btc",
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "page_size": 10000,
        "start_time": start,
    }
    url = COINMETRICS_URL
    for _ in range(40):  # hard page cap; ~5600 daily rows fits in one or two
        payload = _get(url, params)
        rows.extend(payload.get("data", []))
        next_url = payload.get("next_page_url")
        if not next_url:
            break
        url, params = next_url, {}

    if not rows:
        raise DataUnavailable("coinmetrics returned no rows")

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["time"]).dt.tz_localize(None).dt.normalize()
    frame = frame.drop(columns=[c for c in ("time", "asset") if c in frame])
    for col in frame.columns:
        if col != "date":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.set_index("date").sort_index()

    renamed = frame.rename(
        columns={
            "PriceUSD": "price",
            "CapMVRVCur": "mvrv",
            "CapRealUSD": "realized_cap",
            "SplyCur": "supply",
            "RevUSD": "miner_revenue",
        }
    )
    if "realized_cap" in renamed and "supply" in renamed:
        renamed["realized_price"] = renamed["realized_cap"] / renamed["supply"]
    _write_cache("coinmetrics", renamed)
    return renamed


def fetch_blockchain_info(use_cache: bool = True) -> pd.DataFrame:
    """Price-only fallback: blockchain.com market-price chart, daily since 2009."""
    if use_cache:
        cached = _read_cache("blockchain_price")
        if cached is not None:
            return cached

    payload = _get(BLOCKCHAIN_URL, {"timespan": "all", "format": "json", "sampled": "false"})
    values = payload.get("values") or []
    if not values:
        raise DataUnavailable("blockchain.info returned no values")
    frame = pd.DataFrame(values)
    frame["date"] = pd.to_datetime(frame["x"], unit="s").dt.normalize()
    frame = frame.rename(columns={"y": "price"})[["date", "price"]]
    frame = frame.set_index("date").sort_index()
    frame = frame[frame["price"] > 0]
    _write_cache("blockchain_price", frame)
    return frame


def fetch_bitstamp(use_cache: bool = True) -> pd.DataFrame:
    """Price-only fallback: Bitstamp daily OHLC, paginated back to 2011."""
    if use_cache:
        cached = _read_cache("bitstamp_price")
        if cached is not None:
            return cached

    import requests

    step, limit = 86400, 1000
    start = int(pd.Timestamp("2011-08-01").timestamp())
    now = int(time.time())
    chunks: list = []
    while start < now:
        payload = _get(
            BITSTAMP_URL,
            {"step": step, "limit": limit, "start": start},
        )
        ohlc = payload.get("data", {}).get("ohlc", [])
        if not ohlc:
            break
        chunks.extend(ohlc)
        start = int(ohlc[-1]["timestamp"]) + step
        time.sleep(0.4)  # be polite to a free endpoint

    if not chunks:
        raise DataUnavailable("bitstamp returned no candles")
    frame = pd.DataFrame(chunks)
    frame["date"] = pd.to_datetime(frame["timestamp"].astype(int), unit="s").dt.normalize()
    frame["price"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame[["date", "price"]].dropna().drop_duplicates("date")
    frame = frame.set_index("date").sort_index()
    _write_cache("bitstamp_price", frame)
    return frame


def fetch_fred(series_id: str, use_cache: bool = True) -> pd.Series:
    """One FRED series as a daily forward-filled Series (e.g. M2SL, DTWEXBGS)."""
    cache_name = f"fred_{series_id}"
    if use_cache:
        cached = _read_cache(cache_name)
        if cached is not None:
            return cached.iloc[:, 0]

    import requests

    try:
        response = requests.get(FRED_URL, params={"id": series_id}, timeout=30)
    except Exception as exc:
        raise DataUnavailable(f"fred.stlouisfed.org unreachable: {exc}") from exc
    if response.status_code != 200:
        raise DataUnavailable(f"fred.stlouisfed.org returned HTTP {response.status_code}")

    frame = pd.read_csv(io.StringIO(response.text))
    date_col, value_col = frame.columns[0], frame.columns[1]
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    series = frame.set_index(date_col)[value_col].dropna().sort_index()
    series.index.name = "date"
    series = series.rename(series_id)
    _write_cache(cache_name, series.to_frame())
    return series


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a user-exported CSV.

    Needs a date column (``date``/``time``/``Date``) and a ``price`` column.
    Optional columns picked up when present: ``mvrv``, ``realized_price``,
    ``miner_revenue``, ``m2``, ``dxy``.
    """
    frame = pd.read_csv(path)
    date_col = next(
        (c for c in ("date", "time", "Date", "Time", "timestamp") if c in frame.columns),
        None,
    )
    if date_col is None:
        raise ValueError(f"{path}: no date column found among {list(frame.columns)}")
    frame[date_col] = pd.to_datetime(frame[date_col], utc=True, errors="coerce")
    frame[date_col] = frame[date_col].dt.tz_localize(None).dt.normalize()

    lowered = {c: c.lower() for c in frame.columns}
    frame = frame.rename(columns=lowered)
    date_col = date_col.lower()
    if "price" not in frame.columns:
        for alt in ("close", "close_usd", "priceusd", "btc"):
            if alt in frame.columns:
                frame = frame.rename(columns={alt: "price"})
                break
    if "price" not in frame.columns:
        raise ValueError(f"{path}: no price/close column found")

    keep = [date_col, "price"] + [
        c
        for c in ("mvrv", "realized_price", "realized_cap", "supply", "miner_revenue", "m2", "dxy")
        if c in frame.columns
    ]
    frame = frame[keep].rename(columns={date_col: "date"}).dropna(subset=["date", "price"])
    frame = frame.drop_duplicates("date").set_index("date").sort_index()
    if "realized_price" not in frame and {"realized_cap", "supply"} <= set(frame.columns):
        frame["realized_price"] = frame["realized_cap"] / frame["supply"]
    return frame


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load(csv_path: Optional[str | Path] = None, with_macro: bool = True,
         use_cache: bool = True) -> LoadResult:
    """Best available dataset, degrading gracefully."""
    warnings: list = []

    if csv_path:
        frame = load_csv(csv_path)
        return LoadResult(
            frame=frame,
            source=f"csv:{csv_path}",
            has_onchain="mvrv" in frame.columns or "realized_price" in frame.columns,
            has_macro="m2" in frame.columns,
            warnings=warnings,
        )

    frame, source, has_onchain = None, "", False
    try:
        frame = fetch_coinmetrics(use_cache=use_cache)
        source, has_onchain = "coinmetrics", True
    except DataUnavailable as exc:
        warnings.append(f"coinmetrics unavailable ({exc}); falling back to price-only")
        for fetcher, name in ((fetch_blockchain_info, "blockchain.info"), (fetch_bitstamp, "bitstamp")):
            try:
                frame, source = fetcher(use_cache=use_cache), name
                break
            except DataUnavailable as inner:
                warnings.append(f"{name} unavailable ({inner})")

    if frame is None or frame.empty:
        raise DataUnavailable(
            "No price source reachable. Export a CSV (date,price[,mvrv,...]) "
            "and pass --csv, or unblock one of: "
            "community-api.coinmetrics.io, api.blockchain.info, www.bitstamp.net"
        )

    has_macro = False
    if with_macro:
        try:
            m2 = fetch_fred("M2SL", use_cache=use_cache)
            frame["m2"] = m2.reindex(frame.index, method="ffill")
            has_macro = True
        except DataUnavailable as exc:
            warnings.append(f"FRED M2 unavailable ({exc}); macro features disabled")
        try:
            dxy = fetch_fred("DTWEXBGS", use_cache=use_cache)
            frame["dxy"] = dxy.reindex(frame.index, method="ffill")
        except DataUnavailable as exc:
            warnings.append(f"FRED DXY unavailable ({exc})")

    # Daily grid, forward-filled: on-chain series occasionally skip a day.
    frame = frame.asfreq("D").ffill() if frame.index.inferred_freq is None else frame
    frame = frame[frame["price"] > 0]

    return LoadResult(frame=frame, source=source, has_onchain=has_onchain,
                      has_macro=has_macro, warnings=warnings)
