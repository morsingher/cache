#!/usr/bin/env python3
"""
Update script for local financial data store.

This script downloads all price data and macro time series from yfinance and FRED APIs,
and stores them in the local `data/` directory for offline use by the Streamlit app.

IMPORTANT: This script requires ALL tickers to be successfully downloaded.
If any ticker fails after all retries, the entire update is aborted to prevent
creating a partial/inconsistent database.

Usage:
    python scripts/update_data.py [--fred-api-key YOUR_KEY]

Environment variables:
    FRED_API_KEY: FRED API key (optional, for macro data)
"""
import argparse
import json
import os
import sys
import time
import random
from datetime import datetime, timezone
from pathlib import Path

# Add cache directory to path for imports
REPO_ROOT = Path(__file__).parent.parent.resolve()
CACHE_DIR = REPO_ROOT / "cache"
ASSETS_DIR = CACHE_DIR / "assets"
DATA_DIR = REPO_ROOT / "data"

if str(CACHE_DIR) not in sys.path:
    sys.path.insert(0, str(CACHE_DIR))
if str(ASSETS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSETS_DIR))

import numpy as np
import pandas as pd
import yfinance as yf

# Import synthesis functions for special assets
from dbmf_synth import reconstruct_european_history
from zprvx_synth import get_zprvx_series

# FRED series IDs for macro data time series
# Note: "EU" data uses Germany as proxy for rates/yields since ECB data is aggregate
FRED_SERIES = {
    "ecb_dfr_pct": "ECBDFR",           # ECB deposit facility rate (%)
    "eu_10y_yield_pct": "IRLTLT01DEM156N",  # Germany 10Y yield (proxy for EU)
    "eu_cpi_idx": "CP0000EZ19M086NEST",  # Euro Area HICP (goes back to 1996)
    "usd_eur": "DEXUSEU",              # USD/EUR exchange rate
    "fed_rf_pct": "EFFR",              # Fed effective rate (%)
    "us_10y_yield_pct": "DGS10",       # US 10Y yield (%)
    "us_cpi_idx": "CPIAUCSL",          # US CPI index (monthly)
}

# Retry configuration
MAX_RETRIES = 6
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 60.0  # seconds
JITTER = 0.2  # 20% jitter


def retry_with_backoff(func, *args, max_retries: int = MAX_RETRIES, operation_name: str = "operation", **kwargs):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to call
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        operation_name: Name for logging
        **kwargs: Keyword arguments for func
        
    Returns:
        Function result
        
    Raises:
        Exception: Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Check if this looks like a transient/retryable error
            is_retryable = any([
                "nonetype" in error_str,
                "subscriptable" in error_str,
                "connection" in error_str,
                "timeout" in error_str,
                "httperror" in error_str,
                "rate" in error_str,
                "429" in error_str,
                "500" in error_str,
                "502" in error_str,
                "503" in error_str,
                "empty" in error_str,
            ])
            
            if attempt < max_retries - 1 and is_retryable:
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                delay *= (1.0 + random.uniform(-JITTER, JITTER))
                print(f"    Attempt {attempt + 1}/{max_retries} failed for {operation_name}: {e}")
                print(f"    Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                # Last attempt or non-retryable error
                break
    
    raise last_exception


def load_assets_list() -> list[dict]:
    """Load the list of available assets from list.json."""
    assets_path = CACHE_DIR / "assets" / "list.json"
    with open(assets_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("Assets", [])


def download_single_ticker(ticker: str) -> pd.Series:
    """
    Download price data for a single ticker with retry logic.
    
    Returns:
        Series with Close prices
        
    Raises:
        ValueError: If no valid data after all retries
    """
    def _download():
        raw = yf.download(
            ticker,
            period="max",
            auto_adjust=True,
            ignore_tz=True,
            progress=False,
            threads=False,
        )
        
        if raw is None or raw.empty:
            raise ValueError(f"Empty data returned for {ticker}")
        
        # Extract Close prices
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            else:
                raise ValueError(f"No 'Close' column for {ticker}")
        elif "Close" in raw.columns:
            close = raw["Close"]
        else:
            raise ValueError(f"No 'Close' column for {ticker}")
        
        close = pd.to_numeric(close, errors="coerce").dropna()
        
        if close.empty:
            raise ValueError(f"No valid price data for {ticker}")
        
        close.name = ticker
        return close
    
    return retry_with_backoff(_download, operation_name=ticker)


def download_synthetic_dbmf() -> pd.Series:
    """
    Download and synthesize DBMF price series with retry logic.
    
    DBMF is reconstructed by stitching US DBMF data (converted to EUR via FX)
    with EU DBMFE.PA data for a longer history.
    
    Raises:
        ValueError: If synthesis fails after all retries
    """
    print("  Synthesizing DBMF (US ticker + EUR conversion + EU stitch)...")
    
    def _synthesize():
        df = reconstruct_european_history("DBMF", "DBMFE.PA", currency_pair="EURUSD=X")
        s = df["Combined_History"].rename("DBMF")
        s = s[~s.index.duplicated(keep="last")].sort_index()
        
        if s.empty:
            raise ValueError("DBMF synthesis returned empty series")
        
        return s
    
    s = retry_with_backoff(_synthesize, operation_name="DBMF synthesis")
    print(f"    DBMF: {len(s)} rows, {s.index.min().date()} to {s.index.max().date()}")
    return s


def download_synthetic_zprvx() -> pd.Series:
    """
    Download and synthesize ZPRVX price series with retry logic.
    
    ZPRVX = 70% ZPRV.DE (US Small Cap Value) + 30% ZPRX.DE (Europe Small Cap Value)
    
    Raises:
        ValueError: If synthesis fails after all retries
    """
    print("  Synthesizing ZPRVX (70% ZPRV.DE + 30% ZPRX.DE)...")
    
    def _synthesize():
        s = get_zprvx_series()
        
        if s.empty:
            raise ValueError("ZPRVX synthesis returned empty series")
        
        return s
    
    s = retry_with_backoff(_synthesize, operation_name="ZPRVX synthesis")
    print(f"    ZPRVX: {len(s)} rows, {s.index.min().date()} to {s.index.max().date()}")
    return s


def download_all_tickers(tickers: list[str]) -> pd.DataFrame:
    """
    Download price data for ALL tickers, with individual retries.
    
    This function ensures EVERY ticker is successfully downloaded.
    If any ticker fails after all retries, raises an error.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        DataFrame with all tickers as columns
        
    Raises:
        ValueError: If any ticker fails to download
    """
    synthetic_tickers = {"DBMF", "ZPRVX"}
    regular_tickers = [t for t in tickers if t not in synthetic_tickers]
    
    all_series = {}
    failed_tickers = []
    
    # Download regular tickers one by one with retry logic
    print(f"\n  Downloading {len(regular_tickers)} regular tickers...")
    for i, ticker in enumerate(regular_tickers):
        print(f"    [{i+1}/{len(regular_tickers)}] Downloading {ticker}...", end=" ")
        try:
            series = download_single_ticker(ticker)
            all_series[ticker] = series
            print(f"OK ({len(series)} rows)")
        except Exception as e:
            print(f"FAILED: {e}")
            failed_tickers.append((ticker, str(e)))
    
    # Download synthetic tickers
    if "DBMF" in tickers:
        try:
            all_series["DBMF"] = download_synthetic_dbmf()
        except Exception as e:
            print(f"    DBMF: FAILED - {e}")
            failed_tickers.append(("DBMF", str(e)))
    
    if "ZPRVX" in tickers:
        try:
            all_series["ZPRVX"] = download_synthetic_zprvx()
        except Exception as e:
            print(f"    ZPRVX: FAILED - {e}")
            failed_tickers.append(("ZPRVX", str(e)))
    
    # Check if any tickers failed
    if failed_tickers:
        print("\n" + "=" * 60)
        print("DOWNLOAD FAILED - The following tickers could not be downloaded:")
        print("=" * 60)
        for ticker, error in failed_tickers:
            print(f"  - {ticker}: {error}")
        print()
        print("The database was NOT updated to prevent partial/inconsistent data.")
        print("Please check your network connection and try again.")
        raise ValueError(
            f"Failed to download {len(failed_tickers)} ticker(s): "
            f"{', '.join(t for t, _ in failed_tickers)}"
        )
    
    # Verify we got all tickers
    missing = set(tickers) - set(all_series.keys())
    if missing:
        raise ValueError(f"Missing data for tickers: {sorted(missing)}")
    
    # Combine into DataFrame
    prices = pd.DataFrame(all_series)
    
    # Reorder columns to match original ticker order
    prices = prices[[t for t in tickers if t in prices.columns]]
    
    return prices


def update_prices() -> pd.DataFrame:
    """
    Download ALL price data (including synthetic assets) and save to parquet.
    
    This function will FAIL if any ticker cannot be downloaded after retries.
    No partial database will be created.
    """
    print("Loading asset list...")
    assets = load_assets_list()
    all_tickers = [str(a.get("Ticker", "")).strip() for a in assets if a.get("Ticker")]
    all_tickers = sorted(set(all_tickers))
    
    print(f"Requiring ALL {len(all_tickers)} tickers to be downloaded successfully.")
    print(f"Tickers: {', '.join(all_tickers)}")
    
    # Download all tickers with strict validation
    prices = download_all_tickers(all_tickers)
    
    # Final validation
    if prices.empty:
        raise ValueError("No price data downloaded")
    
    if len(prices.columns) != len(all_tickers):
        missing = set(all_tickers) - set(prices.columns)
        raise ValueError(f"Missing data for {len(missing)} ticker(s): {sorted(missing)}")
    
    # Validate each ticker has actual data
    empty_tickers = []
    for t in prices.columns:
        if prices[t].dropna().empty:
            empty_tickers.append(t)
    
    if empty_tickers:
        raise ValueError(f"Empty data for ticker(s): {sorted(empty_tickers)}")
    
    # Save to parquet
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices_path = DATA_DIR / "prices.parquet"
    prices.to_parquet(prices_path, engine="pyarrow")
    
    print(f"\n  Saved prices to {prices_path}")
    print(f"  Shape: {prices.shape[0]} rows x {prices.shape[1]} columns")
    print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    
    # Report on each ticker
    print("\n  Per-ticker summary:")
    for t in prices.columns:
        s = prices[t].dropna()
        print(f"    {t}: {s.index.min().date()} to {s.index.max().date()} ({len(s)} rows)")
    
    return prices


def _get_trailing_eps_with_retry(ticker_symbol: str, max_retries: int = 3) -> float | None:
    """
    Get trailing EPS for a ticker with multiple approaches and retries.
    
    Tries:
    1. ticker.info["trailingEps"]
    2. ticker.fast_info (if available)
    3. Compute from earnings_history (sum of last 4 quarters)
    4. Compute from trailingPE and price
    """
    import time as time_module
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Approach 1: Direct trailingEps from info
            info = ticker.info or {}
            trailing_eps = info.get("trailingEps")
            if trailing_eps is not None and trailing_eps > 0:
                return float(trailing_eps)
            
            # Approach 2: Compute from trailingPE and current price
            trailing_pe = info.get("trailingPE")
            current_price = info.get("regularMarketPrice") or info.get("previousClose")
            if trailing_pe is not None and trailing_pe > 0 and current_price is not None and current_price > 0:
                computed_eps = current_price / trailing_pe
                if computed_eps > 0:
                    return float(computed_eps)
            
            # Approach 3: Try to get from earnings history
            try:
                earnings_hist = ticker.get_earnings_history()
                if earnings_hist is not None and not earnings_hist.empty and "epsActual" in earnings_hist.columns:
                    # Get last 4 quarters of actual EPS
                    recent_eps = earnings_hist["epsActual"].dropna().tail(4)
                    if len(recent_eps) >= 4:
                        ttm_eps = recent_eps.sum()
                        if ttm_eps > 0:
                            return float(ttm_eps)
            except Exception:
                pass
            
            # Approach 4: Try fast_info
            try:
                fast_info = ticker.fast_info
                if hasattr(fast_info, "last_price"):
                    # Need PE to compute EPS from fast_info
                    pass  # fast_info doesn't have PE directly
            except Exception:
                pass
            
            # If we got here, this attempt failed - wait and retry
            if attempt < max_retries - 1:
                delay = BASE_DELAY * (2 ** attempt) * (1 + random.uniform(-JITTER, JITTER))
                time_module.sleep(delay)
                
        except Exception as e:
            if attempt < max_retries - 1:
                delay = BASE_DELAY * (2 ** attempt) * (1 + random.uniform(-JITTER, JITTER))
                time_module.sleep(delay)
    
    return None


def compute_global_earnings_yield_series(lookback_years: int = 3) -> pd.Series:
    """
    Compute global earnings yield time series from yfinance.
    
    Uses trailing P/E from ACWI (or fallback to URTH, SPY) to compute earnings yield (1/PE).
    For a time series, we approximate by using price history and a fixed trailing EPS.
    
    Returns:
        Daily series of earnings yield (%).
        
    Raises:
        ValueError: If unable to compute earnings yield from any source.
    """
    candidate_tickers = ["ACWI", "URTH", "SPY"]
    errors = []
    
    for ticker_symbol in candidate_tickers:
        print(f"    Trying {ticker_symbol}...")
        
        # Get trailing EPS with retry logic
        trailing_eps = _get_trailing_eps_with_retry(ticker_symbol, max_retries=3)
        
        if trailing_eps is None or trailing_eps <= 0:
            error_msg = f"{ticker_symbol}: Could not get valid trailing EPS"
            print(f"      {error_msg}")
            errors.append(error_msg)
            continue
        
        print(f"      Trailing EPS: {trailing_eps:.2f}")
        
        # Get price history with retry
        for attempt in range(MAX_RETRIES):
            try:
                lookback_start = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(start=lookback_start.strftime("%Y-%m-%d"), auto_adjust=True)
                
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    prices = hist["Close"].dropna()
                    if not prices.empty:
                        # Earnings yield = EPS / Price * 100 (approximate as trailing EPS is constant)
                        ey_series = (trailing_eps / prices) * 100.0
                        ey_series.name = "global_ey_pct"
                        
                        print(f"      Got {len(ey_series)} days of data")
                        print(f"      Latest EY: {ey_series.iloc[-1]:.2f}%")
                        
                        return ey_series
                
                error_msg = f"{ticker_symbol}: No valid price history"
                if attempt == MAX_RETRIES - 1:
                    print(f"      {error_msg}")
                    errors.append(error_msg)
                else:
                    import time as time_module
                    delay = BASE_DELAY * (2 ** attempt) * (1 + random.uniform(-JITTER, JITTER))
                    time_module.sleep(delay)
                    
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    error_msg = f"{ticker_symbol}: {str(e)}"
                    print(f"      Error: {error_msg}")
                    errors.append(error_msg)
                else:
                    import time as time_module
                    delay = BASE_DELAY * (2 ** attempt) * (1 + random.uniform(-JITTER, JITTER))
                    time_module.sleep(delay)
    
    # If we get here, all tickers failed
    raise ValueError(
        f"Failed to compute global earnings yield time series. "
        f"Tried: {', '.join(candidate_tickers)}. "
        f"Errors: {'; '.join(errors)}"
    )


def update_macro_series(fred_api_key: str | None = None) -> dict:
    """
    Download FRED macro data as time series and save to parquet.
    
    Also computes a snapshot with latest values and historical trends.
    """
    print("\nFetching FRED macro time series...")
    
    api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("  Warning: FRED_API_KEY not set. Skipping macro data.")
        return {}
    
    try:
        from fredapi import Fred
    except ImportError:
        print("  Warning: fredapi not installed. Skipping macro data.")
        return {}
    
    fred = Fred(api_key=api_key)
    
    # Download all series (last ~3 years for trends)
    obs_start = pd.Timestamp.now() - pd.DateOffset(months=40)
    
    series_data = {}
    for name, series_id in FRED_SERIES.items():
        print(f"  Fetching {series_id} ({name})...")
        try:
            s = fred.get_series(series_id, observation_start=obs_start)
            if s is not None and not s.empty:
                s = pd.to_numeric(s, errors="coerce").dropna()
                series_data[name] = s
                print(f"    Got {len(s)} observations")
            else:
                print(f"    No data returned")
        except Exception as e:
            print(f"    Failed: {e}")
    
    if not series_data:
        print("  Warning: No FRED data fetched.")
        return {}
    
    # Combine into a DataFrame
    macro_df = pd.DataFrame(series_data)
    macro_df = macro_df.sort_index()
    
    # Add global earnings yield time series (computed from yfinance)
    # This will raise ValueError if it fails - we want the entire update to fail
    print("\n  Computing global earnings yield time series...")
    global_ey_series = compute_global_earnings_yield_series()
    # Resample to monthly to match other series
    global_ey_monthly = global_ey_series.resample("ME").last().dropna()
    
    # Store the latest global EY value before merging (indices may not align perfectly)
    latest_global_ey = float(global_ey_monthly.iloc[-1]) if not global_ey_monthly.empty else None
    latest_global_ey_date = global_ey_monthly.index[-1] if not global_ey_monthly.empty else None
    print(f"    Latest Global EY: {latest_global_ey:.2f}% as of {latest_global_ey_date.date() if latest_global_ey_date else 'N/A'}")
    
    # Compute historical global EY trends directly from yfinance series (indices won't align with FRED)
    def get_ey_historical(months_ago: int) -> float | None:
        if global_ey_monthly.empty:
            return None
        target = global_ey_monthly.index.max() - pd.DateOffset(months=months_ago)
        sub = global_ey_monthly.loc[:target]
        return float(sub.iloc[-1]) if not sub.empty else None
    
    global_ey_trends = {
        "3m": get_ey_historical(3),
        "6m": get_ey_historical(6),
        "12m": get_ey_historical(12),
    }
    print(f"    Global EY trends: 3m={global_ey_trends['3m']:.2f}%, 6m={global_ey_trends['6m']:.2f}%, 12m={global_ey_trends['12m']:.2f}%" if all(v is not None for v in global_ey_trends.values()) else f"    Global EY trends: {global_ey_trends}")
    
    # Align with macro_df index - FRED has daily dates, yfinance has month-end dates
    # First, convert yfinance timezone-aware dates to timezone-naive to match FRED
    if global_ey_monthly.index.tz is not None:
        global_ey_monthly.index = global_ey_monthly.index.tz_convert("UTC").tz_localize(None)
    # Use reindex with forward-fill method to propagate values to daily dates
    global_ey_reindexed = global_ey_monthly.reindex(macro_df.index, method="ffill")
    macro_df["global_ey_pct"] = global_ey_reindexed
    non_null_count = macro_df["global_ey_pct"].notna().sum()
    print(f"    Added {len(global_ey_monthly)} months of global EY data ({non_null_count} daily values after reindex)")
    
    # Save time series to parquet
    macro_path = DATA_DIR / "macro_series.parquet"
    macro_df.to_parquet(macro_path, engine="pyarrow")
    print(f"\n  Saved macro series to {macro_path}")
    print(f"  Shape: {macro_df.shape[0]} rows x {macro_df.shape[1]} columns")
    
    # Compute snapshot with latest values, passing the known latest global EY and its trends
    snapshot = compute_macro_snapshot(macro_df, latest_global_ey=latest_global_ey, global_ey_trends=global_ey_trends)
    
    # Save snapshot to JSON for quick access
    snapshot_path = DATA_DIR / "macro.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"  Saved macro snapshot to {snapshot_path}")
    
    return snapshot


def compute_macro_snapshot(
    macro_df: pd.DataFrame,
    latest_global_ey: float | None = None,
    global_ey_trends: dict[str, float | None] | None = None,
) -> dict:
    """Compute snapshot with latest values and historical trends from macro time series.
    
    Args:
        macro_df: DataFrame with macro time series
        latest_global_ey: Pre-computed latest global EY value (to avoid index alignment issues)
        global_ey_trends: Pre-computed global EY trends (3m/6m/12m) from yfinance series
    """
    
    def get_latest(col: str) -> float | None:
        if col not in macro_df.columns:
            return None
        s = macro_df[col].dropna()
        return float(s.iloc[-1]) if not s.empty else None
    
    def get_historical(col: str, months_ago: int) -> float | None:
        if col not in macro_df.columns:
            return None
        s = macro_df[col].dropna()
        if s.empty:
            return None
        target = s.index.max() - pd.DateOffset(months=months_ago)
        sub = s.loc[:target]
        return float(sub.iloc[-1]) if not sub.empty else None
    
    def compute_yoy_inflation(cpi_col: str) -> float | None:
        """Compute YoY inflation from CPI index."""
        if cpi_col not in macro_df.columns:
            return None
        cpi = macro_df[cpi_col].dropna()
        if len(cpi) < 13:
            return None
        # CPI is monthly, so pct_change(12) gives YoY
        yoy = (cpi.pct_change(12) * 100.0).dropna()
        return float(yoy.iloc[-1]) if not yoy.empty else None
    
    asof = macro_df.index.max() if not macro_df.empty else None
    
    # Use pre-computed global EY if provided, otherwise try to get from DataFrame
    global_ey = latest_global_ey if latest_global_ey is not None else get_latest("global_ey_pct")
    global_ey_note = None
    if global_ey is None:
        global_ey_note = "Global EY unavailable - yfinance data may be stale"
    
    snapshot = {
        "asof": asof.isoformat() if asof else None,
        "ecb_dfr_pct": get_latest("ecb_dfr_pct"),
        "eu_10y_yield_pct": get_latest("eu_10y_yield_pct"),
        "eu_cpi_yoy_pct": compute_yoy_inflation("eu_cpi_idx"),
        "fed_rf_pct": get_latest("fed_rf_pct"),
        "us_10y_yield_pct": get_latest("us_10y_yield_pct"),
        "us_cpi_yoy_pct": compute_yoy_inflation("us_cpi_idx"),
        "usd_eur_spot": get_latest("usd_eur"),
        "usd_eur_3m_ago": get_historical("usd_eur", 3),
        "usd_eur_6m_ago": get_historical("usd_eur", 6),
        "usd_eur_12m_ago": get_historical("usd_eur", 12),
        "global_earnings_yield_est_pct": global_ey,
        "global_earnings_yield_note": global_ey_note,
        # Historical trends for LLM reports
        "trends": {
            "ecb_dfr_pct": {
                "3m": get_historical("ecb_dfr_pct", 3),
                "6m": get_historical("ecb_dfr_pct", 6),
                "12m": get_historical("ecb_dfr_pct", 12),
            },
            "eu_10y_yield_pct": {
                "3m": get_historical("eu_10y_yield_pct", 3),
                "6m": get_historical("eu_10y_yield_pct", 6),
                "12m": get_historical("eu_10y_yield_pct", 12),
            },
            "fed_rf_pct": {
                "3m": get_historical("fed_rf_pct", 3),
                "6m": get_historical("fed_rf_pct", 6),
                "12m": get_historical("fed_rf_pct", 12),
            },
            "us_10y_yield_pct": {
                "3m": get_historical("us_10y_yield_pct", 3),
                "6m": get_historical("us_10y_yield_pct", 6),
                "12m": get_historical("us_10y_yield_pct", 12),
            },
            "usd_eur": {
                "3m": get_historical("usd_eur", 3),
                "6m": get_historical("usd_eur", 6),
                "12m": get_historical("usd_eur", 12),
            },
            "global_ey_pct": global_ey_trends if global_ey_trends else {
                "3m": get_historical("global_ey_pct", 3),
                "6m": get_historical("global_ey_pct", 6),
                "12m": get_historical("global_ey_pct", 12),
            },
        },
    }
    
    return snapshot


def update_metadata(prices: pd.DataFrame, macro_available: bool) -> dict:
    """Write metadata about the data store."""
    print("\nWriting metadata...")
    
    # Compute date ranges for each ticker
    date_ranges = {}
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if not series.empty:
            date_ranges[ticker] = {
                "start": series.index.min().date().isoformat(),
                "end": series.index.max().date().isoformat(),
                "count": len(series),
            }
    
    metadata = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": list(prices.columns),
        "total_rows": len(prices),
        "date_range": {
            "start": prices.index.min().date().isoformat(),
            "end": prices.index.max().date().isoformat(),
        },
        "ticker_ranges": date_ranges,
        "macro_available": macro_available,
    }
    
    metadata_path = DATA_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata to {metadata_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Update local financial data store from yfinance and FRED APIs."
    )
    parser.add_argument(
        "--fred-api-key",
        type=str,
        default=None,
        help="FRED API key (or set FRED_API_KEY environment variable)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("FINANCIAL DATA UPDATE")
    print("=" * 60)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print()
    
    try:
        # Update prices (including synthetic assets)
        prices = update_prices()
        
        # Update macro time series
        macro_snapshot = update_macro_series(fred_api_key=args.fred_api_key)
        macro_available = bool(macro_snapshot)
        
        # Update metadata
        update_metadata(prices, macro_available)
        
        print()
        print("=" * 60)
        print("UPDATE COMPLETE")
        print("=" * 60)
        print(f"Data saved to: {DATA_DIR}")
        print("You can now run the Streamlit app without network dependencies.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
