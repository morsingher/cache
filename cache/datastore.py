"""
Local data store module for loading financial data from the local database.

This module provides functions to load price data and macro time series from the
local `data/` directory, which is populated by `scripts/update_data.py`.
"""
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

# Paths
REPO_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_ROOT / "data"


def data_exists() -> bool:
    """Check if the local data store exists."""
    prices_path = DATA_DIR / "prices.parquet"
    return prices_path.exists()


def macro_series_exists() -> bool:
    """Check if macro time series data exists."""
    macro_path = DATA_DIR / "macro_series.parquet"
    return macro_path.exists()


@lru_cache(maxsize=1)
def _load_prices_cached(mtime: float) -> pd.DataFrame:
    """
    Load prices from parquet file with caching.
    
    The mtime parameter ensures the cache is invalidated when the file changes.
    """
    prices_path = DATA_DIR / "prices.parquet"
    return pd.read_parquet(prices_path)


@lru_cache(maxsize=1)
def _load_macro_series_cached(mtime: float) -> pd.DataFrame:
    """
    Load macro series from parquet file with caching.
    
    The mtime parameter ensures the cache is invalidated when the file changes.
    """
    macro_path = DATA_DIR / "macro_series.parquet"
    return pd.read_parquet(macro_path)


def load_prices() -> pd.DataFrame:
    """
    Load all price data from the local parquet file.
    
    Returns:
        DataFrame with DatetimeIndex and ticker columns.
        
    Raises:
        FileNotFoundError: If the local data store doesn't exist.
    """
    prices_path = DATA_DIR / "prices.parquet"
    
    if not prices_path.exists():
        raise FileNotFoundError(
            f"Local price data not found at {prices_path}. "
            "Run 'python scripts/update_data.py' to download data first."
        )
    
    # Use file modification time for cache invalidation
    mtime = prices_path.stat().st_mtime
    return _load_prices_cached(mtime).copy()


def get_prices_for_tickers(tickers: list[str]) -> pd.DataFrame:
    """
    Load prices for specific tickers from the local store.
    
    Args:
        tickers: List of ticker symbols.
        
    Returns:
        DataFrame with only the requested tickers (columns that exist).
    """
    prices = load_prices()
    available = [t for t in tickers if t in prices.columns]
    
    if not available:
        return pd.DataFrame()
    
    return prices[available]


def load_macro_series() -> pd.DataFrame:
    """
    Load macro time series from the local parquet file.
    
    Returns:
        DataFrame with DatetimeIndex and macro indicator columns:
        - ecb_dfr_pct: ECB deposit facility rate (%)
        - eu_10y_yield_pct: EU 10Y yield (German Bund as proxy) (%)
        - eu_cpi_idx: Euro Area HICP index
        - usd_eur: USD/EUR exchange rate
        - fed_rf_pct: Fed effective rate (%)
        - us_10y_yield_pct: US 10Y yield (%)
        - us_cpi_idx: US CPI index
        
    Raises:
        FileNotFoundError: If the macro data doesn't exist.
    """
    macro_path = DATA_DIR / "macro_series.parquet"
    
    if not macro_path.exists():
        raise FileNotFoundError(
            f"Local macro data not found at {macro_path}. "
            "Run 'python scripts/update_data.py --fred-api-key YOUR_KEY' to download."
        )
    
    # Use file modification time for cache invalidation
    mtime = macro_path.stat().st_mtime
    return _load_macro_series_cached(mtime).copy()


def load_macro_snapshot() -> dict | None:
    """
    Load macro snapshot from local JSON file.
    
    Returns:
        Dictionary with latest macro values and historical trends, or None if not available.
    """
    macro_path = DATA_DIR / "macro.json"
    
    if not macro_path.exists():
        return None
    
    with open(macro_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert asof string back to timestamp if present
    if data.get("asof"):
        data["asof"] = pd.Timestamp(data["asof"])
    
    return data


def get_macro_trends() -> dict[str, dict[str, float | None]] | None:
    """
    Get historical macro trends (3m/6m/12m ago values) from the snapshot.
    
    Returns:
        Dictionary with trend data for each macro indicator:
        {
            "ecb_dfr_pct": {"3m": value, "6m": value, "12m": value},
            "eu_10y_yield_pct": {...},
            ...
        }
        Returns None if not available.
    """
    snapshot = load_macro_snapshot()
    if snapshot is None:
        return None
    return snapshot.get("trends")


def compute_cpi_yoy_series(cpi_col: str = "eu_cpi_idx") -> pd.Series | None:
    """
    Compute YoY inflation from stored CPI index series.
    
    Args:
        cpi_col: Column name for CPI index ("eu_cpi_idx" or "us_cpi_idx")
        
    Returns:
        Series with YoY inflation (%), or None if not available.
    """
    if not macro_series_exists():
        return None
    
    try:
        macro_df = load_macro_series()
        if cpi_col not in macro_df.columns:
            return None
        
        cpi = macro_df[cpi_col].dropna()
        if len(cpi) < 13:
            return None
        
        # CPI is monthly, pct_change(12) gives YoY
        yoy = (cpi.pct_change(12) * 100.0).dropna()
        yoy.name = cpi_col.replace("_idx", "_yoy_pct")
        return yoy
    except Exception:
        return None


def get_metadata() -> dict:
    """
    Load metadata about the local data store.
    
    Returns:
        Dictionary with update timestamp, ticker list, and date ranges.
    """
    metadata_path = DATA_DIR / "metadata.json"
    
    if not metadata_path.exists():
        return {
            "updated_at": None,
            "tickers": [],
            "total_rows": 0,
            "date_range": {"start": None, "end": None},
            "ticker_ranges": {},
            "macro_available": False,
        }
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_data_freshness() -> str:
    """
    Get a human-readable string describing data freshness.
    
    Returns:
        String like "Updated 2 days ago" or "Data not available".
    """
    metadata = get_metadata()
    updated_at = metadata.get("updated_at")
    
    if not updated_at:
        return "Data not available"
    
    try:
        update_time = datetime.fromisoformat(updated_at)
        now = datetime.now(timezone.utc)
        
        # Handle both timezone-aware (new format) and naive (old format) timestamps
        if update_time.tzinfo is None:
            # Old naive timestamp - assume it was meant to be UTC
            update_time = update_time.replace(tzinfo=timezone.utc)
        
        delta = now - update_time
        total_seconds = delta.total_seconds()
        
        # Handle edge case where timestamp appears to be in the future
        # (can happen due to clock drift)
        if total_seconds < 0:
            return "Updated just now"
        
        if delta.days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                minutes = delta.seconds // 60
                if minutes == 0:
                    return "Updated just now"
                return f"Updated {minutes} minute{'s' if minutes != 1 else ''} ago"
            return f"Updated {hours} hour{'s' if hours != 1 else ''} ago"
        elif delta.days == 1:
            return "Updated yesterday"
        else:
            return f"Updated {delta.days} days ago"
    except Exception:
        return "Update time unknown"


def get_price_date_ranges() -> dict[str, tuple[str | None, str | None]]:
    """
    Get the date range for each ticker from metadata.
    
    Returns:
        Dictionary mapping ticker -> (start_date, end_date) as ISO strings.
    """
    metadata = get_metadata()
    ticker_ranges = metadata.get("ticker_ranges", {})
    
    result = {}
    for ticker, info in ticker_ranges.items():
        if isinstance(info, dict):
            result[ticker] = (info.get("start"), info.get("end"))
        else:
            result[ticker] = (None, None)
    
    return result


def get_available_tickers() -> list[str]:
    """
    Get list of tickers available in the local data store.
    
    Returns:
        List of ticker symbols.
    """
    metadata = get_metadata()
    return metadata.get("tickers", [])
