import os
import json
import time
import random
import re
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any

# Assume cache module is in sys.path or available
try:
    from portfolio import Portfolio
except ImportError:
    # Fallback or error if not set up correctly
    import sys
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE_DIR = os.path.join(REPO_ROOT, "cache")
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from portfolio import Portfolio

# Import get_macro_snapshot safely
try:
    from rebalancing import get_macro_snapshot
except ImportError:
    import sys
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE_DIR = os.path.join(REPO_ROOT, "cache")
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from rebalancing import get_macro_snapshot

# Import datastore for local data loading
try:
    from datastore import (
        data_exists,
        load_prices,
        get_prices_for_tickers,
        get_price_date_ranges,
        get_data_freshness,
        load_macro_snapshot,
        load_macro_series,
        macro_series_exists,
        get_macro_trends,
    )
except ImportError:
    import sys
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE_DIR = os.path.join(REPO_ROOT, "cache")
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from datastore import (
        data_exists,
        load_prices,
        get_prices_for_tickers,
        get_price_date_ranges,
        get_data_freshness,
        load_macro_snapshot,
        load_macro_series,
        macro_series_exists,
        get_macro_trends,
    )

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")

def retry_on_rate_limit(fn, *args, max_retries: int = 5, initial_wait: float = 10.0, **kwargs):
    """
    Retry a function on rate limit errors with exponential backoff.
    (Kept for backward compatibility but rarely used now that we load from local store.)
    """
    last_error = None
    wait_time = initial_wait
    
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = (
                "rate" in error_str 
                or "too many requests" in error_str 
                or "429" in error_str
            )
            is_empty_data = "no valid price" in error_str
            
            if is_rate_limit or is_empty_data:
                last_error = e
                if attempt < max_retries - 1:
                    reason = "Rate limit hit" if is_rate_limit else "Empty data returned (possible rate limit)"
                    st.warning(
                        f"⏳ Yahoo Finance: {reason}. Waiting {wait_time:.0f}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    raise
            else:
                raise
    
    if last_error:
        raise last_error

@st.cache_data(ttl=3600, show_spinner=False)
def cached_download_prices(tickers_tuple: tuple[str, ...]) -> pd.DataFrame:
    """
    Load price data for tickers, preferring local data store.
    
    Falls back to yfinance API only if local data is not available.
    """
    # Prefer local data store
    if data_exists():
        prices = get_prices_for_tickers(list(tickers_tuple))
        if not prices.empty:
            return prices
    
    # Fallback to API (with retry support) if local data not available
    return retry_on_rate_limit(Portfolio.download_prices, list(tickers_tuple))


def store_prices_in_cache(prices: pd.DataFrame) -> None:
    """Store prices in session state cache for quick access."""
    if "prices_cache" not in st.session_state:
        st.session_state["prices_cache"] = {}
        
    if prices is None or prices.empty:
        return
    for t in prices.columns:
        s = pd.to_numeric(prices[t], errors="coerce").dropna()
        if not s.empty:
            st.session_state["prices_cache"][str(t)] = s.copy()


def get_prices_and_store(tickers_tuple: tuple[str, ...]) -> pd.DataFrame:
    """
    Get prices for tickers, using session state cache and local data store.
    
    Priority:
    1. Session state cache (fastest, for current session)
    2. Local parquet data store (fast, persistent)
    3. yfinance API fallback (slow, requires network)
    """
    if "prices_cache" not in st.session_state:
        st.session_state["prices_cache"] = {}
        
    tickers_norm = [str(t).strip() for t in tickers_tuple if str(t).strip()]
    if not tickers_norm:
        return pd.DataFrame()

    # Check what's missing from session cache
    missing = [t for t in tickers_norm if t not in st.session_state["prices_cache"]]
    
    if missing:
        # Try to load from local data store first
        if data_exists():
            local_prices = get_prices_for_tickers(missing)
            if not local_prices.empty:
                store_prices_in_cache(local_prices)
                # Update missing list
                missing = [t for t in missing if t not in local_prices.columns]
        
        # Fallback to API for any still-missing tickers
        if missing:
            prices_new = cached_download_prices(tuple(sorted(set(missing))))
            store_prices_in_cache(prices_new)
    
    data = {}
    available_tickers = []
    for t in tickers_norm:
        if t in st.session_state["prices_cache"]:
            data[t] = st.session_state["prices_cache"][t]
            available_tickers.append(t)
            
    if not data:
        return pd.DataFrame()
        
    prices = pd.concat(data, axis=1)
    prices = prices.reindex(columns=available_tickers)
    return prices

def get_cached_prices_only(tickers: list[str], *, source: str) -> pd.DataFrame:
    if "prices_cache" not in st.session_state:
        st.session_state["prices_cache"] = {}
        
    tickers_norm = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers_norm:
        return pd.DataFrame()
        
    missing = [t for t in tickers_norm if t not in st.session_state["prices_cache"]]
    if missing:
        raise ValueError(
            "Price data not cached yet for: "
            f"{missing}. Please run any analysis once to cache prices, then retry {source}."
        )
        
    data = {t: st.session_state["prices_cache"][t] for t in tickers_norm}
    prices = pd.concat(data, axis=1)
    prices = prices.reindex(columns=tickers_norm)
    return prices

@st.cache_data(ttl=3600, show_spinner=False)
def load_available_assets() -> list[dict[str, str]]:
    """Load available assets from list.json."""
    assets_path = os.path.join(CACHE_DIR, "assets", "list.json")
    try:
        with open(assets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("Assets", [])
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_asset_options() -> tuple[list[str], dict[str, tuple[str, str, str]], dict[str, str]]:
    assets = load_available_assets()
    assets = sorted(assets, key=lambda a: str(a.get("Name", "")).lower())
    options = []
    mapping = {}
    display_map = {}
    for asset in assets:
        name = asset.get("Name", "")
        ticker = asset.get("Ticker", "")
        short = asset.get("Short", name)
        if name and ticker:
            options.append(short)
            mapping[short] = (name, ticker, short)
            display_map[short] = f"{name} ({ticker})"
    return options, mapping, display_map

@st.cache_data(ttl=3600, show_spinner=False)
def get_short_name_map() -> dict[str, str]:
    assets = load_available_assets()
    return {asset.get("Ticker", ""): asset.get("Short", asset.get("Name", "")) for asset in assets if asset.get("Ticker")}

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_price_date_ranges(tickers: tuple[str, ...]) -> dict[str, tuple[str | None, str | None]]:
    """
    Get date ranges for tickers, preferring local metadata for speed.
    """
    tickers_list = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers_list:
        return {}
    tickers_list = sorted(set(tickers_list))
    out = {t: (None, None) for t in tickers_list}
    
    # Try to use local metadata first (instant)
    if data_exists():
        local_ranges = get_price_date_ranges()
        for t in tickers_list:
            if t in local_ranges:
                out[t] = local_ranges[t]
        # If all tickers found in local data, return early
        if all(out[t] != (None, None) for t in tickers_list):
            return out
    
    # Fallback: compute from actual prices (for any missing tickers)
    try:
        px = get_prices_and_store(tuple(sorted(tickers_list)))
    except Exception:
        return out
    if px is None or px.empty:
        return out
    for t in tickers_list:
        if out[t] != (None, None):
            continue  # Already have from metadata
        if t not in px.columns:
            continue
        s = pd.to_numeric(px[t], errors="coerce").dropna()
        if s.empty:
            continue
        try:
            start = str(pd.Timestamp(s.index.min()).date())
            end = str(pd.Timestamp(s.index.max()).date())
        except Exception:
            start, end = None, None
        out[t] = (start, end)
    return out

def render_asset_help_dropdown() -> None:
    assets = load_available_assets()
    assets = sorted(assets, key=lambda a: str(a.get("Name", "")).lower())
    if not assets:
        return

    tickers_all = [str(a.get("Ticker", "")).strip() for a in assets if str(a.get("Ticker", "")).strip()]
    # Data is preloaded at app startup (see app.py), so this should be instant from cache.
    date_ranges = cached_price_date_ranges(tuple(sorted(set(tickers_all))))

    with st.expander("💡 Need help choosing assets?", expanded=False):
        for asset in assets:
            name = asset.get("Name", "")
            ticker = asset.get("Ticker", "")
            description = asset.get("Description", "")
            link = asset.get("Link", "")
            
            if not name or not ticker:
                continue

            start_date, end_date = date_ranges.get(str(ticker).strip(), (None, None))
            sd = start_date or "—"
            ed = end_date or "—"
            availability_html = (
                f'<div style="color: #6b7280; font-size: 0.85em; margin-top: 0.15rem;">'
                f'Data available from <b>{sd}</b> to <b>{ed}</b>.'
                f"</div>"
            )
            
            if isinstance(link, list):
                desc_with_links = description
                embedded_tickers = re.findall(r'\(([A-Z0-9]+)\)', description)
                for i, embedded_ticker in enumerate(embedded_tickers):
                    if i < len(link):
                        clickable = f'(<a href="{link[i]}" target="_blank" style="color: #0066cc; text-decoration: underline;">{embedded_ticker}</a>)'
                        desc_with_links = desc_with_links.replace(f'({embedded_ticker})', clickable, 1)
                st.markdown(
                    f"**{name}** ({ticker}): {desc_with_links}{availability_html}",
                    unsafe_allow_html=True
                )
            elif link:
                ticker_link = f'<a href="{link}" target="_blank" style="color: #0066cc; text-decoration: underline;">{ticker}</a>'
                st.markdown(
                    f"**{name}** ({ticker_link}): {description}{availability_html}",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"**{name}** ({ticker}): {description}{availability_html}", unsafe_allow_html=True)

def presort_multiselect_state(*, key: str, sort_by: dict[str, str] | None = None) -> None:
    if key not in st.session_state:
        return
    v = st.session_state.get(key)
    if not isinstance(v, list):
        return
    def _k(x: str) -> str:
        if sort_by is None:
            return str(x).lower()
        return str(sort_by.get(str(x), str(x))).lower()

    sorted_vals = sorted([str(x) for x in v if x is not None], key=_k)
    if v != sorted_vals:
        st.session_state[key] = sorted_vals

def validate_portfolio_json_obj(obj: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate uploaded portfolio JSON.
    """
    errors: list[str] = []

    if not isinstance(obj, dict):
        return False, ["Top-level JSON must be an object."]

    if "Name" not in obj or not str(obj.get("Name", "")).strip():
        errors.append("Missing/empty top-level `Name`.")

    assets = obj.get("Assets")
    if not isinstance(assets, list) or not assets:
        errors.append("`Assets` must be a non-empty list.")
        return False, errors

    if "Value" in obj:
        try:
            v = float(obj.get("Value"))
            if not np.isfinite(v) or v <= 0:
                errors.append("Top-level `Value` must be > 0.")
        except Exception:
            errors.append("Top-level `Value` must be a number.")

    tickers: list[str] = []
    weights: list[float] = []
    targets: list[float] = []

    for i, a in enumerate(assets):
        if not isinstance(a, dict):
            errors.append(f"Assets[{i}] must be an object.")
            continue

        name = str(a.get("Name", "")).strip()
        ticker = str(a.get("Ticker", "")).strip()
        if not name:
            errors.append(f"Assets[{i}].Name is missing/empty.")
        if not ticker:
            errors.append(f"Assets[{i}].Ticker is missing/empty.")

        def _num(field: str) -> float | None:
            try:
                return float(a.get(field))
            except Exception:
                return None

        w = _num("Weight")
        t = _num("Target")

        if w is None:
            errors.append(f"Assets[{i}].Weight must be a number.")
        elif not np.isfinite(w) or w < 0:
            errors.append(f"Assets[{i}].Weight must be >= 0.")
        else:
            weights.append(float(w))

        if t is None:
            errors.append(f"Assets[{i}].Target must be a number.")
        elif not np.isfinite(t) or t <= 0:
            errors.append(f"Assets[{i}].Target must be > 0.")
        else:
            targets.append(float(t))

        if ticker:
            tickers.append(ticker)

    if errors:
        return False, errors

    if len(set(tickers)) != len(tickers):
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        errors.append(f"Duplicate tickers not allowed: {dupes}")

    tol = 0.25
    w_sum = float(sum(weights))
    t_sum = float(sum(targets))
    if abs(w_sum - 100.0) > tol:
        errors.append(f"`Weight` values must sum to 100 (±{tol}). Got {w_sum:.4f}.")
    if abs(t_sum - 100.0) > tol:
        errors.append(f"`Target` values must sum to 100 (±{tol}). Got {t_sum:.4f}.")
        
    known_assets = load_available_assets()
    known = {a.get("Ticker", "") for a in known_assets if a.get("Ticker")}
    unknown = sorted({t for t in tickers if t not in known})
    
    if unknown:
        pass

    return (len(errors) == 0), errors

def yf_has_data_for_tickers(tickers: tuple[str, ...]) -> dict[str, bool]:
    out = {t: False for t in tickers}
    if not tickers:
        return out
    try:
        px = retry_on_rate_limit(Portfolio.download_prices, list(tickers), period="6mo")
        if px is None or px.empty:
            return out
        for t in tickers:
            if t in px.columns and not px[t].dropna().empty:
                out[t] = True
        return out
    except Exception:
        return out

@st.cache_data(ttl=3600, show_spinner=False)
def cached_get_macro_snapshot(fred_api_key: str | None) -> Any:
    """
    Get macro snapshot, preferring local data store.

    Falls back to FRED API only if local data is not available.
    """
    # Try local data store first
    if data_exists():
        local_macro = load_macro_snapshot()
        if local_macro is not None:
            # Convert dict back to MacroSnapshot-like object for compatibility
            from rebalancing import MacroSnapshot
            return MacroSnapshot(
                asof=local_macro.get("asof"),
                ecb_dfr_pct=local_macro.get("ecb_dfr_pct"),
                eu_10y_yield_pct=local_macro.get("eu_10y_yield_pct"),
                eu_cpi_yoy_pct=local_macro.get("eu_cpi_yoy_pct"),
                fed_rf_pct=local_macro.get("fed_rf_pct"),
                us_10y_yield_pct=local_macro.get("us_10y_yield_pct"),
                us_cpi_yoy_pct=local_macro.get("us_cpi_yoy_pct"),
                global_earnings_yield_est_pct=local_macro.get("global_earnings_yield_est_pct"),
                global_earnings_yield_note=local_macro.get("global_earnings_yield_note"),
                usd_eur_spot=local_macro.get("usd_eur_spot"),
                usd_eur_3m_ago=local_macro.get("usd_eur_3m_ago"),
                usd_eur_6m_ago=local_macro.get("usd_eur_6m_ago"),
                usd_eur_12m_ago=local_macro.get("usd_eur_12m_ago"),
            )

    # Fallback to FRED API
    return get_macro_snapshot(fred_api_key=fred_api_key, debug=False)


def get_macro_trends() -> dict[str, dict[str, float | None]] | None:
    """
    Load macro trends from local data store.
    
    Returns a dict with keys like 'ecb_dfr_pct', 'eu_10y_yield_pct', etc.,
    each containing '3m', '6m', '12m' historical values.
    """
    if not data_exists():
        return None
    local_macro = load_macro_snapshot()
    if local_macro is None:
        return None
    return local_macro.get("trends")


def get_macro_chart_data(indicator_names: list[str]) -> list[dict[str, object]]:
    """
    Generate chart data for macro indicators from local macro series.
    
    Args:
        indicator_names: List of indicator display names like "ECB Overnight (%)", "DE 10Y Yield (%)", etc.
    
    Returns:
        List of dicts with Date, Indicator, Value for use in render_macro_chart.
    """
    if not macro_series_exists():
        return []
    
    try:
        macro_df = load_macro_series()
        if macro_df is None or macro_df.empty:
            return []
    except Exception:
        return []
    
    # Map display names to column names in macro_series.parquet
    # Note: Column names in parquet are: ecb_dfr_pct, eu_10y_yield_pct, eu_cpi_idx, 
    #       usd_eur, fed_rf_pct, us_10y_yield_pct, us_cpi_idx, global_ey_pct
    display_to_column = {
        "ECB Overnight (%)": "ecb_dfr_pct",
        "EU 10Y Yield (%)": "eu_10y_yield_pct",
        "FED Overnight (%)": "fed_rf_pct",
        "US 10Y Yield (%)": "us_10y_yield_pct",
        "USD/EUR": "usd_eur",
        "Global EY Est. (%)": "global_ey_pct",
    }
    
    # Special handling for YoY inflation (computed from CPI index)
    inflation_indicators = {
        "EU Inflation YoY (%)": "eu_cpi_idx",
        "US Inflation YoY (%)": "us_cpi_idx",
    }
    
    # Filter to last 12 months of data
    end_date = macro_df.index.max()
    start_date = end_date - pd.DateOffset(months=12)
    macro_12m = macro_df.loc[start_date:end_date].copy()
    
    # Compute YoY inflation from CPI indices if needed
    for display_name, cpi_col in inflation_indicators.items():
        if display_name in indicator_names and cpi_col in macro_df.columns:
            # Need full data for YoY calculation, then filter to 12m
            cpi_series = macro_df[cpi_col].dropna()
            if len(cpi_series) >= 13:
                yoy = (cpi_series.pct_change(12) * 100.0).dropna()
                yoy_12m = yoy.loc[start_date:end_date]
                # Store temporarily for extraction below
                col_name = f"{cpi_col}_yoy"
                macro_12m[col_name] = yoy_12m
                display_to_column[display_name] = col_name
    
    rows = []
    for display_name in indicator_names:
        col_name = display_to_column.get(display_name)
        if col_name and col_name in macro_12m.columns:
            series = macro_12m[col_name].dropna()
            for date, val in series.items():
                rows.append({
                    "Date": date,
                    "Indicator": display_name,
                    "Value": float(val),
                })
    
    return rows


def get_data_status() -> tuple[bool, str]:
    """
    Get the status of the local data store.
    
    Returns:
        Tuple of (data_available, freshness_string).
    """
    if not data_exists():
        return False, "Local data not available. Run 'python scripts/update_data.py' to download."
    return True, get_data_freshness()
