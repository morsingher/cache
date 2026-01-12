import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import random


class StepTimer:
    """Helper class for timing steps within a st.status() context."""
    
    def __init__(self, status_container):
        self.status = status_container
        self.start_time = time.time()
        self.step_start = None
        self.step_placeholder = None
    
    def step(self, label: str) -> None:
        """Start a new step, showing progress message."""
        # Complete previous step if any
        if self.step_placeholder is not None and self.step_start is not None:
            elapsed = time.time() - self.step_start
            # Update the placeholder with completion message
            # (placeholder can't be updated after new content, so we just start fresh)
        
        self.step_start = time.time()
        self.step_placeholder = st.empty()
        self.step_placeholder.write(f"{label}...")
    
    def done(self) -> None:
        """Mark current step as done with timing."""
        if self.step_placeholder is not None and self.step_start is not None:
            elapsed = time.time() - self.step_start
            self.step_placeholder.write(f"{self._get_current_label()} Done! ({elapsed:.2f}s)")
            self.step_placeholder = None
            self.step_start = None
    
    def _get_current_label(self) -> str:
        """Get the label from the current placeholder (fallback)."""
        return ""
    
    def total_time(self) -> float:
        """Get total elapsed time since timer creation."""
        return time.time() - self.start_time


def _timed_step(placeholder, label: str, start_time: float) -> None:
    """Update a placeholder with completed step message and timing."""
    elapsed = time.time() - start_time
    placeholder.write(f"{label} Done! ({elapsed:.2f}s)")


# --- Import backend exactly like the CLI does (modules live in ./cache) ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")
if CACHE_DIR not in sys.path:
    sys.path.insert(0, CACHE_DIR)

from portfolio import Portfolio  # noqa: E402
from comparison import run_comparison  # noqa: E402
from rebalancing import (  # noqa: E402
    compute_rebalancing_diagnostics,
    get_macro_snapshot,
    build_llm_rebalance_report,
    get_global_earnings_yield_series,
    get_us_earnings_yield_proxy_series_fred,
)
from whatif import (  # noqa: E402
    _apply_swap_from_stocks,
    _parse_tickers,
    _stocks_tickers,
    _target_weights_fraction,
    build_llm_whatif_report,
    diversification_scores,
    run_whatif,
)
from openrouter import (  # noqa: E402
    fetch_free_models,
    fetch_limits,
    chat_completion,
    chat_completion_messages,
    get_api_key,
    DEFAULT_FREE_MODELS,
)

try:
    from fredapi import Fred  # type: ignore
except Exception:  # pragma: no cover
    Fred = None


# --- Caching for expensive operations ---
# Cache Portfolio objects and price downloads to avoid re-fetching on every widget interaction.
# TTL of 1 hour ensures data is refreshed periodically but not on every rerun.

def _retry_on_rate_limit(fn, *args, max_retries: int = 5, initial_wait: float = 10.0, **kwargs):
    """
    Retry a function on rate limit errors with exponential backoff.
    Shows a warning toast to the user while waiting.
    
    Catches both explicit rate limit errors and "no valid price data" errors,
    which can occur when Yahoo Finance returns empty data due to rate limiting.
    """
    last_error = None
    wait_time = initial_wait
    
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit errors (various forms) OR empty data errors
            # Empty data errors can happen when YF is rate limited but doesn't raise an exception
            is_rate_limit = (
                "rate" in error_str 
                or "too many requests" in error_str 
                or "429" in error_str
            )
            is_empty_data = "no valid price" in error_str
            
            if is_rate_limit or is_empty_data:
                last_error = e
                if attempt < max_retries - 1:
                    # Show warning with countdown
                    reason = "Rate limit hit" if is_rate_limit else "Empty data returned (possible rate limit)"
                    st.warning(
                        f"⏳ Yahoo Finance: {reason}. Waiting {wait_time:.0f}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    raise
            else:
                # Not a retryable error, re-raise immediately
                raise
    
    # Should not reach here, but just in case
    if last_error:
        raise last_error


@st.cache_resource(ttl=3600, show_spinner="Loading portfolio (downloading price data, may retry on transient errors)...")
def _cached_load_portfolio(path: str) -> Portfolio:
    """
    Cache the entire Portfolio object (including downloaded prices).
    Uses cache_resource since Portfolio objects are not serializable.
    """
    return _retry_on_rate_limit(Portfolio.from_json, path)


@st.cache_data(ttl=3600, show_spinner="Downloading price data (may retry on transient errors)...")
def _cached_download_prices(tickers_tuple: tuple[str, ...]) -> pd.DataFrame:
    """Cached wrapper for Portfolio.download_prices with retry support."""
    return _retry_on_rate_limit(Portfolio.download_prices, list(tickers_tuple))


st.set_page_config(page_title="CACHE", page_icon="€", layout="centered")

# Register pastel color theme for Altair charts
@alt.theme.register("pastel", enable=True)
def _pastel_theme():
    return {
        "config": {
            "range": {
                "category": [
                    "#AEC6CF",  # Pastel Blue
                    "#FFB7C5",  # Pastel Pink
                    "#B39EB5",  # Pastel Purple
                    "#77DD77",  # Pastel Green
                    "#FDFD96",  # Pastel Yellow
                    "#FFB347",  # Pastel Orange
                    "#CFCFC4",  # Pastel Grey
                    "#F49AC2",  # Pastel Magenta
                    "#B19CD9",  # Pastel Lavender
                    "#FF6961",  # Pastel Red
                ]
            }
        }
    }

# alt.theme.register("pastel", _pastel_theme)
# alt.theme.enable("pastel")

# Minimal CSS for elements not controllable via theme alone
st.markdown(
    """
<style>
/* Secondary buttons */
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background-color: #ecebe3 !important;
}

/* Glide-data-grid (Streamlit dataframe) CSS custom properties */
:root {
    --gdg-bg-cell: #f4f3ed !important;
    --gdg-bg-header: #ecebe3 !important;
    --gdg-bg-header-has-focus: #e8e7dd !important;
    --gdg-bg-header-hovered: #e8e7dd !important;
    --gdg-accent-color: #bb5a38 !important;
    --gdg-accent-light: #ecebe3 !important;
    --gdg-bg-bubble: #ecebe3 !important;
    --gdg-bg-bubble-selected: #e8e7dd !important;
    --gdg-bg-search-result: #ecebe3 !important;
}

/* Additional table styling */
.stDataFrame,
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizableContainer"] {
    --gdg-bg-cell: #f4f3ed !important;
    --gdg-bg-header: #ecebe3 !important;
    --gdg-bg-header-has-focus: #e8e7dd !important;
    --gdg-bg-header-hovered: #e8e7dd !important;
}

/* NOTE: Do NOT globally center headings.
   We only center the custom app title in `_render_title()` via inline HTML styles.
   Leaving headings left-aligned fixes inconsistent section-header alignment. */

/* Reduce space between title (h1) and subtitle (h2) */
[data-testid="stHeading"] h1,
h1[data-testid="stHeading"] {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
[data-testid="stHeading"] h2 {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Style the custom title h1 */
h1[data-testid="stHeading"] {
    font-size: 52px !important;
    font-weight: 600 !important;
}

/* Hide anchor links on all headers */
[data-testid="stHeading"] a,
h1 a[href^="#"],
h2 a[href^="#"],
h3 a[href^="#"],
h4 a[href^="#"] {
    display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

def _rf_annual_controls(*, key_prefix: str, default_series: str = "ECBDFR") -> float:
    mode = st.radio(
        "Risk-free rate",
        options=["Federal Reserve Economic Data (FRED)", "Manual"],
        index=0,
        horizontal=True,
        key=f"{key_prefix}_rf_mode",
        help="Used for Sharpe/Sortino in analysis + comparisons/backtests.",
    )

    if mode == "Manual":
        rf = st.number_input(
            "Custom value (annual, decimal)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.005,
            key=f"{key_prefix}_rf_manual",
        )
        return float(rf)

    # series_id = st.text_input("FRED series id", value=default_series, key=f"{key_prefix}_rf_series")
    series_id = default_series
    if Fred is None:
        st.warning("fredapi not available; using 0%.")
        return 0.0

    api_key = st.secrets.get("FRED_API_KEY", "").strip()
    if not api_key:
        st.error("Set `FRED_API_KEY` in `.streamlit/secrets.toml` to use FRED data.")
        return 0.0

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_latest_fred_pct(_series_id: str, _api_key: str) -> float | None:
        # FRED can intermittently fail on cold starts; retry with backoff + jitter.
        fred = Fred(api_key=_api_key)
        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                s = fred.get_series(_series_id)
                if s is None:
                    raise RuntimeError("FRED returned None")
                s = s.dropna()
                if len(s) == 0:
                    raise RuntimeError("FRED returned empty series")
                return float(s.iloc[-1])
            except Exception as e:
                last_exc = e
                if attempt < 5:
                    delay = 0.7 * (2 ** attempt)
                    delay *= (1.0 + random.uniform(-0.15, 0.15))
                    time.sleep(delay)
        # Don't raise inside Streamlit cache: return None and show a friendly warning outside.
        _ = last_exc
        return None

    try:
        latest_pct = _cached_latest_fred_pct(series_id, api_key)
        if latest_pct is None:
            st.warning("FRED series returned no data; using 0%.")
            return 0.0
        st.caption(f"Latest: {latest_pct:.3f}%")
        return float(latest_pct / 100.0)
    except Exception as e:
        # Avoid confusing "(None)" messages; show details only if meaningful.
        msg = str(e).strip()
        details = f" ({msg})" if msg and msg.lower() != "none" else ""
        st.warning(f"FRED fetch failed; using 0%.{details}")
        return 0.0


def _backtest_controls(*, key_prefix: str, show_initial_amount: bool = True) -> tuple[str, float, float]:
    """
    Returns: (rebalance_frequency, initial_amount, rf_annual)
    """
    c1, c2 = st.columns([1, 1])
    with c1:
        rebalance_display = st.radio(
            "Rebalance frequency",
            options=["Monthly", "Quarterly", "Annually"],
            index=2,
            horizontal=True,
            key=f"{key_prefix}_reb_freq",
        )
        rebalance_frequency = str(rebalance_display).lower()
    with c2:
        initial_amount = 10_000.0
        if show_initial_amount:
            initial_amount = st.number_input(
                "Initial amount (EUR)",
                min_value=100.0,
                value=10_000.0,
                step=500.0,
                key=f"{key_prefix}_initial_amt",
            )

    rf_annual = _rf_annual_controls(key_prefix=key_prefix)

    return str(rebalance_frequency), float(initial_amount), float(rf_annual)


def _safe_temp_json(content: bytes) -> str:
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False)
    f.write(content)
    f.flush()
    f.close()
    return f.name


def _render_example_json_ui(*, key_prefix: str) -> None:
    """
    Show example portfolio JSONs (built-ins + a minimal template) to help users author their own files.
    """
    with st.expander("📄 Example JSON (click to view)", expanded=False):
        portfolios_dir = os.path.join(CACHE_DIR, "portfolios")
        example_paths: list[str] = []
        try:
            for fname in sorted(os.listdir(portfolios_dir)):
                if fname.lower().endswith(".json"):
                    example_paths.append(os.path.join(portfolios_dir, fname))
        except Exception:
            example_paths = []

        # Minimal template example: prefer explicit Stocks/Bonds if present, else first two by Name.
        assets_list = _load_available_assets()
        assets_by_name = sorted(assets_list, key=lambda a: str(a.get("Name", "")).lower())

        def _find_by_name(name: str) -> dict[str, str] | None:
            for a in assets_list:
                if str(a.get("Name", "")).strip().lower() == name.strip().lower():
                    return a
            return None

        a_stocks = _find_by_name("Stocks")
        a_bonds = _find_by_name("Bonds")
        if a_stocks and a_bonds:
            a1, a2 = a_stocks, a_bonds
        elif len(assets_by_name) >= 2:
            a1, a2 = assets_by_name[0], assets_by_name[1]
        else:
            a1, a2 = {"Name": "Stocks", "Ticker": "ACWE.MI"}, {"Name": "Bonds", "Ticker": "AGGH.MI"}

        # Include `Short` for guidance (parser ignores extra keys, but it's useful to users).
        minimal_assets = [
            {
                "Name": a1.get("Name", "Stocks"),
                "Ticker": a1.get("Ticker", "ACWE.MI"),
                "Short": a1.get("Short", a1.get("Name", "Stocks")),
                "Weight": 60.0,
                "Target": 60.0,
            },
            {
                "Name": a2.get("Name", "Bonds"),
                "Ticker": a2.get("Ticker", "AGGH.MI"),
                "Short": a2.get("Short", a2.get("Name", "Bonds")),
                "Weight": 40.0,
                "Target": 40.0,
            },
        ]
        minimal_example_obj = {"Name": "My Portfolio", "Assets": minimal_assets, "Value": 80_000.0}

        example_options = ["Minimal example (template)"] + [os.path.basename(p) for p in example_paths]
        selected_ex = st.selectbox("Choose an example", options=example_options, key=f"{key_prefix}_example_select")

        if selected_ex == "Minimal example (template)":
            txt = json.dumps(minimal_example_obj, indent=2)
            st.code(txt, language="json")
            st.download_button(
                "Download minimal example (.json)",
                data=txt.encode("utf-8"),
                file_name="portfolio_example_minimal.json",
                mime="application/json",
                key=f"{key_prefix}_download_example_min",
            )
            return

        match_path = None
        for pth in example_paths:
            if os.path.basename(pth) == selected_ex:
                match_path = pth
                break
        if not match_path:
            st.info("No example file selected.")
            return

        try:
            with open(match_path, "r", encoding="utf-8") as f:
                txt = f.read()
            st.code(txt, language="json")
            st.download_button(
                "Download this example (.json)",
                data=txt.encode("utf-8"),
                file_name=os.path.basename(match_path),
                mime="application/json",
                key=f"{key_prefix}_download_example_builtin",
            )
        except Exception as e:
            st.warning(f"Could not read example file: {e}")


def _load_available_assets() -> list[dict[str, str]]:
    """Load available assets from list.json."""
    assets_path = os.path.join(CACHE_DIR, "assets", "list.json")
    try:
        with open(assets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("Assets", [])
    except Exception:
        return []

def _asset_ticker_set() -> set[str]:
    return {a.get("Ticker", "") for a in _load_available_assets() if a.get("Ticker")}

@st.cache_data(ttl=3600, show_spinner=False)
def _yf_has_data_for_tickers(tickers: tuple[str, ...]) -> dict[str, bool]:
    """
    Best-effort: check whether yfinance can return *any* price data for the tickers.
    Uses a shorter period to avoid heavy downloads during validation.
    """
    out = {t: False for t in tickers}
    if not tickers:
        return out
    try:
        px = _retry_on_rate_limit(Portfolio.download_prices, list(tickers), period="6mo")
        if px is None or px.empty:
            return out
        for t in tickers:
            if t in px.columns and not px[t].dropna().empty:
                out[t] = True
        return out
    except Exception:
        return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _cached_price_date_ranges(tickers: tuple[str, ...]) -> dict[str, tuple[str | None, str | None]]:
    """
    Best-effort: get first/last available price dates per ticker.

    Notes:
    - Uses the same backend price downloader as the rest of the app (`Portfolio.download_prices`),
      so synthetic tickers like DBMF/ZPRVX are supported.
    - Cached to avoid recomputing on every rerun (and to avoid repeated Yahoo requests).
    """
    tickers_list = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers_list:
        return {}

    tickers_list = sorted(set(tickers_list))
    out: dict[str, tuple[str | None, str | None]] = {t: (None, None) for t in tickers_list}

    try:
        px = _retry_on_rate_limit(Portfolio.download_prices, list(tickers_list), period="max")
    except Exception:
        return out

    if px is None or px.empty:
        return out

    for t in tickers_list:
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

def _validate_portfolio_json_obj(obj: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate uploaded portfolio JSON before attempting to construct a Portfolio.

    Checks:
    - Template/schema shape
    - Numeric sanity (weights/targets > 0, sums ~ 100, Value > 0 if provided)
    - Ticker validity: either in `cache/assets/list.json` OR yfinance has data
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
        elif not np.isfinite(w) or w <= 0:
            errors.append(f"Assets[{i}].Weight must be > 0.")
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

    known = _asset_ticker_set()
    unknown = sorted({t for t in tickers if t not in known})
    if unknown:
        checks = _yf_has_data_for_tickers(tuple(unknown))
        missing = [t for t in unknown if not checks.get(t, False)]
        if missing:
            errors.append(
                "Unknown tickers must either be present in `cache/assets/list.json` or have price data on Yahoo Finance. "
                f"No data found for: {missing}"
            )

    return (len(errors) == 0), errors


def _get_asset_options() -> tuple[list[str], dict[str, tuple[str, str, str]], dict[str, str]]:
    """
    Get asset options for dropdown.
    Returns:
        - List of Short names (used as option values, shown in selected chips)
        - Dict mapping Short name to (Name, Ticker, Short)
        - Dict mapping Short name to display string "Name (Ticker)" for format_func
    """
    assets = _load_available_assets()
    # Sort by FULL name (not Short) for consistent UX.
    assets = sorted(assets, key=lambda a: str(a.get("Name", "")).lower())
    options = []
    mapping = {}
    display_map = {}  # Short -> "Name (Ticker)" for format_func
    for asset in assets:
        name = asset.get("Name", "")
        ticker = asset.get("Ticker", "")
        short = asset.get("Short", name)  # Fallback to Name if Short not present
        if name and ticker:
            options.append(short)
            mapping[short] = (name, ticker, short)
            display_map[short] = f"{name} ({ticker})"
    return options, mapping, display_map


def _get_short_name_map() -> dict[str, str]:
    """
    Build a mapping from ticker to Short name for all available assets.
    """
    assets = _load_available_assets()
    return {asset.get("Ticker", ""): asset.get("Short", asset.get("Name", "")) for asset in assets if asset.get("Ticker")}

def _presort_multiselect_state(*, key: str, sort_by: dict[str, str] | None = None) -> None:
    """
    Pre-sort a multiselect's stored state *before* the widget is instantiated.

    Streamlit raises if you try to modify `st.session_state[key]` *after* the widget with that key
    has been created in the same run.
    """
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


def _render_asset_help_dropdown() -> None:
    """Render a dropdown showing available assets with descriptions and clickable links."""
    assets = _load_available_assets()
    if not assets:
        return
    
    with st.expander("💡 Need help choosing assets?", expanded=False):
        # Compute once (cached) instead of per-asset.
        tickers_all = [str(a.get("Ticker", "")).strip() for a in assets if str(a.get("Ticker", "")).strip()]
        date_ranges = _cached_price_date_ranges(tuple(sorted(set(tickers_all))))
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
            
            # Handle Link being either a string or a list of strings
            # Use HTML for proper clickable links that open in new tab
            if isinstance(link, list):
                # Multiple links - this is a synthetic/composite asset
                # Make the embedded tickers in the description clickable instead
                # For ZPRVX: links are [ZPRV_link, ZPRX_link] and description mentions (ZPRV) and (ZPRX)
                desc_with_links = description
                # Extract ticker symbols from description that are in parentheses
                # and replace them with clickable links
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
                # Single link - make ticker clickable
                ticker_link = f'<a href="{link}" target="_blank" style="color: #0066cc; text-decoration: underline;">{ticker}</a>'
                st.markdown(
                    f"**{name}** ({ticker_link}): {description}{availability_html}",
                    unsafe_allow_html=True
                )
            else:
                # No link available
                st.markdown(f"**{name}** ({ticker}): {description}{availability_html}", unsafe_allow_html=True)


def _render_portfolio_preview(p: Portfolio) -> None:
    """Render a compact allocation preview for a portfolio using Short names."""
    target_weights_pct = getattr(p, "target_weights_pct", {})
    tickers = getattr(p, "tickers", [])
    name = getattr(p, "name", "Portfolio")
    short_map = _get_short_name_map()
    
    if not tickers or not target_weights_pct:
        return
    
    parts = []
    for ticker in tickers:
        short_name = short_map.get(ticker, ticker)
        weight = target_weights_pct.get(ticker, 0.0)
        parts.append(f"{short_name} ({weight:.1f}%)")
    
    allocation_str = ", ".join(parts)
    st.caption(f"**{name}:** {allocation_str}")


def _portfolio_json_from_manual(
    *,
    portfolio_name: str,
    df: pd.DataFrame,
    value_eur: float | None,
) -> dict[str, Any]:
    short_map = _get_short_name_map()
    assets: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        weight = round(float(row["Weight (%)"]), 2)
        # If Target (%) column exists, use it; otherwise use Weight (%) as target
        target = round(float(row["Target (%)"]), 2) if "Target (%)" in row.index else weight
        ticker = str(row["Ticker"]).strip()
        name = str(row["Asset Name"]).strip()
        short = short_map.get(ticker, name)  # Fallback to Name if Short not found
        assets.append(
            {
                "Name": name,
                "Ticker": ticker,
                "Short": short,
                "Weight": weight,
                "Target": target,
            }
        )
    obj: dict[str, Any] = {"Name": str(portfolio_name).strip() or "Portfolio", "Assets": assets}
    # Default to 100k EUR if value not specified
    obj["Value"] = round(float(value_eur), 2) if value_eur is not None else 100_000.0
    return obj


def _build_portfolio_from_manual(
    *,
    portfolio_name: str,
    df: pd.DataFrame,
    value_eur: float | None,
    normalize: bool,
) -> Portfolio:
    data = df.copy()
    data = data.replace({np.nan: None})

    # basic cleanup
    data["Asset Name"] = data["Asset Name"].astype(str).str.strip()
    data["Ticker"] = data["Ticker"].astype(str).str.strip()
    data["Weight (%)"] = pd.to_numeric(data["Weight (%)"], errors="coerce")
    
    # If Target (%) column doesn't exist, use Weight (%) as target
    has_target_col = "Target (%)" in data.columns
    if has_target_col:
        data["Target (%)"] = pd.to_numeric(data["Target (%)"], errors="coerce")
        data = data.dropna(subset=["Ticker", "Weight (%)", "Target (%)"], how="any")
    else:
        data = data.dropna(subset=["Ticker", "Weight (%)"], how="any")
        data["Target (%)"] = data["Weight (%)"]
    
    data = data[data["Ticker"].astype(str).str.len() > 0]
    if data.empty:
        raise ValueError("No valid rows. Provide at least one asset with Ticker/Weight.")

    # enforce unique tickers
    tickers = data["Ticker"].tolist()
    if len(set(tickers)) != len(tickers):
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        raise ValueError(f"Duplicate tickers not allowed: {dupes}")

    w_sum = float(np.nansum(data["Weight (%)"].to_numpy(dtype=float)))
    t_sum = float(np.nansum(data["Target (%)"].to_numpy(dtype=float)))
    if normalize:
        if w_sum <= 0 or t_sum <= 0:
            raise ValueError("Cannot normalize: weight sums must be > 0.")
        data["Weight (%)"] = data["Weight (%)"] * (100.0 / w_sum)
        data["Target (%)"] = data["Target (%)"] * (100.0 / t_sum)
        w_sum, t_sum = 100.0, 100.0

    tol = 0.25
    if abs(w_sum - 100.0) > tol:
        raise ValueError(f"Weights must sum to 100 (±{tol}). Got {w_sum:.4f}.")
    if has_target_col and abs(t_sum - 100.0) > tol:
        raise ValueError(f"Target weights must sum to 100 (±{tol}). Got {t_sum:.4f}.")

    assets = data["Asset Name"].tolist()
    weights = data["Weight (%)"].to_list()
    targets = data["Target (%)"].to_list()

    p = Portfolio(tickers=tickers, weights=weights, assets=assets)
    p.name = str(portfolio_name).strip() or "Portfolio"
    p.current_value_eur = float(value_eur) if value_eur is not None else None
    p.actual_weights_pct = {t: float(w) for t, w in zip(tickers, weights)}
    p.target_weights_pct = {t: float(w) for t, w in zip(tickers, targets)}
    return p


def portfolio_builder(
    *,
    key: str,
    title: str = "Portfolio",
    allow_value: bool = False,
) -> tuple[Portfolio | None, dict[str, Any] | None]:
    """
    Returns:
      - portfolio instance (or None if not ready)
      - portfolio json (dict) if built manually (else None)
    """
    if title:
        st.subheader(title)

    source = st.radio(
        "Create / load portfolio",
        options=["Manual", "Built-in JSON", "Upload JSON"],
        horizontal=True,
        key=f"{key}_source",
    )

    built_json_obj: dict[str, Any] | None = None

    if source == "Built-in JSON":
        portfolios_dir = os.path.join(CACHE_DIR, "portfolios")
        paths = []
        try:
            for fname in sorted(os.listdir(portfolios_dir)):
                if fname.lower().endswith(".json"):
                    paths.append(os.path.join(portfolios_dir, fname))
        except Exception:
            paths = []

        if not paths:
            st.error("No built-in portfolios found in `cache/portfolios/`.")
            return None, None

        path = st.selectbox("Select a portfolio JSON", options=paths, format_func=lambda p: os.path.basename(p), key=f"{key}_builtin")
        try:
            loaded_p = _cached_load_portfolio(path)
            _render_portfolio_preview(loaded_p)
            return loaded_p, None
        except Exception as e:
            st.error(str(e))
            return None, None

    if source == "Upload JSON":
        _render_example_json_ui(key_prefix=f"{key}_upload")

        up = st.file_uploader("Upload a portfolio JSON", type=["json"], key=f"{key}_upload")
        if up is None:
            return None, None
        try:
            raw = up.getvalue()
            # Validate uploaded JSON BEFORE constructing Portfolio (better error messages + avoids unnecessary downloads).
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                raise ValueError("Invalid JSON file (could not parse).")
            ok, errs = _validate_portfolio_json_obj(obj)
            if not ok:
                raise ValueError("Invalid portfolio JSON:\n- " + "\n- ".join(errs))

            tmp = _safe_temp_json(raw)
            loaded_p = Portfolio.from_json(tmp)
            _render_portfolio_preview(loaded_p)
            return loaded_p, None
        except Exception as e:
            st.error(str(e))
            return None, None

    # Manual
    col1, col2 = st.columns([2, 1])
    with col1:
        portfolio_name = st.text_input("Portfolio name", value="My Portfolio", key=f"{key}_name")
    with col2:
        value_eur = None
        if allow_value:
            value_eur = st.number_input("Current value (EUR)", min_value=0.0, value=80_000.0, step=1_000.0, key=f"{key}_value")

    # Load available assets for dropdown (options are Short names)
    asset_options, asset_mapping, asset_display_map = _get_asset_options()
    if not asset_options:
        st.error("No assets available. Check `cache/assets/list.json`.")
        return None, None

    # Show asset help dropdown
    _render_asset_help_dropdown()

    # Default selections: 60% Stocks, 40% Bonds (find by Short name)
    def _find_asset_by_short(opts: list[str], short_name: str) -> str:
        for opt in opts:
            if opt.lower() == short_name.lower():
                return opt
        return opts[0] if opts else ""
    
    default_stocks = _find_asset_by_short(asset_options, "Stocks")
    default_bonds = _find_asset_by_short(asset_options, "Bonds")
    # Fallback if not found
    if not default_stocks and asset_options:
        default_stocks = asset_options[0]
    if not default_bonds and len(asset_options) > 1:
        default_bonds = asset_options[1] if asset_options[1] != default_stocks else (asset_options[0] if asset_options[0] != default_stocks else "")
    
    # For rebalancing (allow_value=True), use table with both Weight and Target columns
    # For other sections, use slider-based UI with auto-normalization
    if allow_value:
        # === TABLE-BASED UI FOR REBALANCING ===
        default = pd.DataFrame(
            [
                {"Asset": default_stocks, "Weight (%)": 60.0, "Target (%)": 60.0},
                {"Asset": default_bonds, "Weight (%)": 40.0, "Target (%)": 40.0},
            ]
        )
        column_config = {
            "Asset": st.column_config.SelectboxColumn(
                "Asset",
                options=asset_options,
                required=True,
                help="Select an asset from the available list",
            ),
            "Weight (%)": st.column_config.NumberColumn(required=True, min_value=0.0, help="Current allocation"),
            "Target (%)": st.column_config.NumberColumn(required=True, min_value=0.0, help="Target allocation"),
        }
        
        edited_df = st.data_editor(
            default,
            num_rows="dynamic",
            width="stretch",
            key=f"{key}_editor",
            column_config=column_config,
        )
        
        df = pd.DataFrame(edited_df.to_dict("records"))
        
        # Parse asset selections to extract Name and Ticker
        asset_names = []
        tickers = []
        for x in df["Asset"].tolist():
            if pd.notna(x) and x and x in asset_mapping:
                name, ticker, _short = asset_mapping[x]
                asset_names.append(name)
                tickers.append(ticker)
            else:
                asset_names.append("")
                tickers.append("")
        df["Asset Name"] = asset_names
        df["Ticker"] = tickers
        
        # Calculate sums and check if normalization is needed
        df["Weight (%)"] = pd.to_numeric(df["Weight (%)"], errors="coerce")
        df["Target (%)"] = pd.to_numeric(df["Target (%)"], errors="coerce")
        w_sum = float(df["Weight (%)"].sum())
        t_sum = float(df["Target (%)"].sum())
        
        # Check if normalization is needed for Weight (%)
        if w_sum > 0 and abs(w_sum - 100.0) > 0.5:
            # Normalize and show warning
            raw_weights = df["Weight (%)"].tolist()
            df["Weight (%)"] = df["Weight (%)"] / w_sum * 100.0
            df["Weight (%)"] = df["Weight (%)"].round(2)
            
            # Asset column contains Short names directly
            change_lines = []
            for i, asset in enumerate(df["Asset"].tolist()):
                if pd.notna(asset) and asset:
                    change_lines.append(f"- {asset}: {raw_weights[i]:.0f}% → {df['Weight (%)'].iloc[i]:.1f}%")
            st.warning(
                f"⚠️ **Current weights** sum to **{w_sum:.0f}%** (not 100%). Normalizing automatically.\n\n"
                + "\n".join(change_lines)
            )
        elif w_sum > 0:
            df["Weight (%)"] = df["Weight (%)"].round(2)
        
        # Check if normalization is needed for Target (%)
        if t_sum > 0 and abs(t_sum - 100.0) > 0.5:
            # Normalize and show warning
            raw_targets = df["Target (%)"].tolist()
            df["Target (%)"] = df["Target (%)"] / t_sum * 100.0
            df["Target (%)"] = df["Target (%)"].round(2)
            
            # Asset column contains Short names directly
            change_lines = []
            for i, asset in enumerate(df["Asset"].tolist()):
                if pd.notna(asset) and asset:
                    change_lines.append(f"- {asset}: {raw_targets[i]:.0f}% → {df['Target (%)'].iloc[i]:.1f}%")
            st.warning(
                f"⚠️ **Target weights** sum to **{t_sum:.0f}%** (not 100%). Normalizing automatically.\n\n"
                + "\n".join(change_lines)
            )
        elif t_sum > 0:
            df["Target (%)"] = df["Target (%)"].round(2)
        
        # Show normalized allocation summary (Asset column contains Short names)
        if w_sum > 0 and t_sum > 0:
            w_parts = []
            t_parts = []
            for i, asset in enumerate(df["Asset"].tolist()):
                if pd.notna(asset) and asset:
                    w_parts.append(f"{asset}: {df['Weight (%)'].iloc[i]:.1f}%")
                    t_parts.append(f"{asset}: {df['Target (%)'].iloc[i]:.1f}%")
            st.caption(f"**Current:** {' · '.join(w_parts)}")
            st.caption(f"**Target:** {' · '.join(t_parts)}")
        
        normalize = True  # Always normalize in table mode now
    else:
        # === SLIDER-BASED UI FOR ANALYZE/COMPARE/WHATIF ===
        # Initialize session state for this portfolio builder
        weights_key = f"{key}_slider_weights"
        if weights_key not in st.session_state:
            st.session_state[weights_key] = {default_stocks: 60.0, default_bonds: 40.0}
        
        # Asset multiselect
        # Keep selection chips sorted by FULL name.
        _presort_multiselect_state(
            key=f"{key}_asset_select",
            sort_by={short: asset_mapping.get(short, (short, "", short))[0] for short in asset_options},
        )
        current_selection = list(st.session_state[weights_key].keys())
        # Ensure current selection only contains valid options
        current_selection = [a for a in current_selection if a in asset_options]
        if not current_selection:
            current_selection = [default_stocks, default_bonds]
        
        _asset_select_key = f"{key}_asset_select"
        if _asset_select_key in st.session_state:
            selected_assets = st.multiselect(
                "Select assets",
                options=asset_options,
                format_func=lambda x: asset_display_map.get(x, x),  # Show "Name (Ticker)" in dropdown
                key=_asset_select_key,
            )
        else:
            selected_assets = st.multiselect(
                "Select assets",
                options=asset_options,
                default=current_selection,
                format_func=lambda x: asset_display_map.get(x, x),  # Show "Name (Ticker)" in dropdown
                key=_asset_select_key,
            )
        
        if not selected_assets:
            st.warning("Please select at least one asset.")
            return None, None
        
        # Update weights for newly added assets (give them equal share of remaining)
        current_weights = st.session_state[weights_key]
        for asset in selected_assets:
            if asset not in current_weights:
                # New asset: give it a default weight
                current_weights[asset] = 10.0
        
        # Remove weights for deselected assets
        current_weights = {k: v for k, v in current_weights.items() if k in selected_assets}
        st.session_state[weights_key] = current_weights
        
        # Display sliders for each asset (selected_assets contains Short names)
        raw_weights: dict[str, float] = {}
        
        for asset in selected_assets:
            # asset is a Short name; get full Name for display
            if asset in asset_mapping:
                full_name, _ticker, _short = asset_mapping[asset]
            else:
                full_name = asset
            # Create a safe key from the asset name (remove special chars)
            safe_key = "".join(c if c.isalnum() else "_" for c in asset)
            
            col_name, col_slider = st.columns([2.5, 6.5])
            
            with col_name:
                st.markdown(f"**{full_name}**")
            
            with col_slider:
                # Use session state value as default
                default_val = float(current_weights.get(asset, 10.0))
                weight = st.slider(
                    f"Weight for {full_name}",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_val,
                    step=1.0,
                    key=f"{key}_slider_{safe_key}",
                    label_visibility="collapsed",
                )
                raw_weights[asset] = weight
        
        # Update session state with new weights
        st.session_state[weights_key] = raw_weights
        
        # Compute normalized weights
        total_raw = sum(raw_weights.values())
        if total_raw <= 0:
            st.warning("Total weight must be greater than 0.")
            return None, None
        
        # Normalize weights to sum to 100 (round to 2 decimals)
        normalized_weights = {k: round(v / total_raw * 100.0, 2) for k, v in raw_weights.items()}
        
        # Check if normalization was needed (sum != 100)
        if abs(total_raw - 100.0) > 0.5:
            # Show warning with before → after for each asset (use full Name)
            change_lines = []
            for asset in selected_assets:
                if asset in asset_mapping:
                    full_name, _ticker, _short = asset_mapping[asset]
                else:
                    full_name = asset
                change_lines.append(f"- {full_name}: {raw_weights[asset]:.0f}% → {normalized_weights[asset]:.1f}%")
            st.warning(
                f"⚠️ Weights sum to **{total_raw:.0f}%** (not 100%). Normalizing automatically.\n\n"
                + "\n".join(change_lines)
            )
        else:
            # Display compact allocation summary (no normalization needed, use full Name)
            alloc_parts = []
            for asset in selected_assets:
                if asset in asset_mapping:
                    full_name, _ticker, _short = asset_mapping[asset]
                else:
                    full_name = asset
                alloc_parts.append(f"{full_name}: **{normalized_weights[asset]:.1f}%**")
            st.caption(" · ".join(alloc_parts))
        
        rows = []
        for asset in selected_assets:
            if asset in asset_mapping:
                name, ticker, _short = asset_mapping[asset]
                rows.append({
                    "Asset": asset,
                    "Asset Name": name,
                    "Ticker": ticker,
                    "Weight (%)": normalized_weights[asset],
                })
        
        df = pd.DataFrame(rows)
        normalize = True  # Always normalize in slider mode

    try:
        built_json_obj = _portfolio_json_from_manual(portfolio_name=portfolio_name, df=df, value_eur=value_eur if allow_value else None)
        p = _build_portfolio_from_manual(
            portfolio_name=portfolio_name,
            df=df,
            value_eur=value_eur if allow_value else None,
            normalize=bool(normalize),
        )
    except Exception as e:
        st.error(str(e))
        return None, built_json_obj

    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            "Download JSON",
            data=json.dumps(built_json_obj, indent=2).encode("utf-8"),
            file_name=f"{(portfolio_name or 'portfolio').strip().replace(' ', '_')}.json",
            mime="application/json",
            key=f"{key}_download",
        )
    with c2:
        with st.expander("Preview JSON"):
            st.code(json.dumps(built_json_obj, indent=2), language="json")

    return p, built_json_obj


def _manual_portfolio_builder(
    *,
    key: str,
    title: str = "Manual Portfolio",
    show_json_download: bool = False,
) -> Portfolio | None:
    """
    Simplified portfolio builder that only allows manual entry (no JSON options).
    Uses slider-based UI with auto-normalization to 100%.
    """
    if title:
        st.subheader(title)

    portfolio_name = st.text_input("Portfolio name", value="My Portfolio", key=f"{key}_name")

    # Load available assets for dropdown (options are Short names)
    asset_options, asset_mapping, asset_display_map = _get_asset_options()
    if not asset_options:
        st.error("No assets available. Check `cache/assets/list.json`.")
        return None

    # Show asset help dropdown
    _render_asset_help_dropdown()

    # Default selections: 60% Stocks, 40% Bonds (find by Short name)
    def _find_asset_by_short(opts: list[str], short_name: str) -> str:
        for opt in opts:
            if opt.lower() == short_name.lower():
                return opt
        return opts[0] if opts else ""
    
    default_stocks = _find_asset_by_short(asset_options, "Stocks")
    default_bonds = _find_asset_by_short(asset_options, "Bonds")
    # Fallback if not found
    if not default_stocks and asset_options:
        default_stocks = asset_options[0]
    if not default_bonds and len(asset_options) > 1:
        default_bonds = asset_options[1] if asset_options[1] != default_stocks else (asset_options[0] if asset_options[0] != default_stocks else "")
    
    # === SLIDER-BASED UI ===
    # Initialize session state for this portfolio builder
    weights_key = f"{key}_slider_weights"
    if weights_key not in st.session_state:
        st.session_state[weights_key] = {default_stocks: 60.0, default_bonds: 40.0}
    
    # Asset multiselect
    current_selection = list(st.session_state[weights_key].keys())
    # Ensure current selection only contains valid options
    current_selection = [a for a in current_selection if a in asset_options]
    if not current_selection:
        current_selection = [default_stocks, default_bonds]
    
    _presort_multiselect_state(
        key=f"{key}_asset_select",
        sort_by={short: asset_mapping.get(short, (short, "", short))[0] for short in asset_options},
    )
    _asset_select_key = f"{key}_asset_select"
    if _asset_select_key in st.session_state:
        selected_assets = st.multiselect(
            "Select assets",
            options=asset_options,
            format_func=lambda x: asset_display_map.get(x, x),  # Show "Name (Ticker)" in dropdown
            key=_asset_select_key,
        )
    else:
        selected_assets = st.multiselect(
            "Select assets",
            options=asset_options,
            default=current_selection,
            format_func=lambda x: asset_display_map.get(x, x),  # Show "Name (Ticker)" in dropdown
            key=_asset_select_key,
        )
    
    if not selected_assets:
        st.warning("Please select at least one asset.")
        return None
    
    # Update weights for newly added assets
    current_weights = st.session_state[weights_key]
    for asset in selected_assets:
        if asset not in current_weights:
            current_weights[asset] = 10.0
    
    # Remove weights for deselected assets
    current_weights = {k: v for k, v in current_weights.items() if k in selected_assets}
    st.session_state[weights_key] = current_weights
    
    # Display sliders for each asset (selected_assets contains Short names)
    raw_weights: dict[str, float] = {}
    
    for asset in selected_assets:
        # asset is a Short name; get full Name for display
        if asset in asset_mapping:
            full_name, _ticker, _short = asset_mapping[asset]
        else:
            full_name = asset
        # Create a safe key from the asset name (remove special chars)
        safe_key = "".join(c if c.isalnum() else "_" for c in asset)
        
        col_name, col_slider = st.columns([2.5, 6.5])
        
        with col_name:
            st.markdown(f"**{full_name}**")
        
        with col_slider:
            default_val = float(current_weights.get(asset, 10.0))
            weight = st.slider(
                f"Weight for {full_name}",
                min_value=0.0,
                max_value=100.0,
                value=default_val,
                step=1.0,
                key=f"{key}_slider_{safe_key}",
                label_visibility="collapsed",
            )
            raw_weights[asset] = weight
    
    # Update session state
    st.session_state[weights_key] = raw_weights
    
    # Compute normalized weights
    total_raw = sum(raw_weights.values())
    if total_raw <= 0:
        st.warning("Total weight must be greater than 0.")
        return None
    
    # Normalize weights to sum to 100 (round to 2 decimals)
    normalized_weights = {k: round(v / total_raw * 100.0, 2) for k, v in raw_weights.items()}
    
    # Check if normalization was needed (sum != 100, use full Name)
    if abs(total_raw - 100.0) > 0.5:
        change_lines = []
        for asset in selected_assets:
            if asset in asset_mapping:
                full_name, _ticker, _short = asset_mapping[asset]
            else:
                full_name = asset
            change_lines.append(f"- {full_name}: {raw_weights[asset]:.0f}% → {normalized_weights[asset]:.1f}%")
        st.warning(
            f"⚠️ Weights sum to **{total_raw:.0f}%** (not 100%). Normalizing automatically.\n\n"
            + "\n".join(change_lines)
        )
    else:
        # Display compact allocation summary (no normalization needed, use full Name)
        alloc_parts = []
        for asset in selected_assets:
            if asset in asset_mapping:
                full_name, _ticker, _short = asset_mapping[asset]
            else:
                full_name = asset
            alloc_parts.append(f"{full_name}: **{normalized_weights[asset]:.1f}%**")
        st.caption(" · ".join(alloc_parts))
    
    rows = []
    for asset in selected_assets:
        if asset in asset_mapping:
            name, ticker, _short = asset_mapping[asset]
            rows.append({
                "Asset": asset,
                "Asset Name": name,
                "Ticker": ticker,
                "Weight (%)": normalized_weights[asset],
            })
    
    df = pd.DataFrame(rows)

    try:
        p = _build_portfolio_from_manual(
            portfolio_name=portfolio_name,
            df=df,
            value_eur=None,
            normalize=True,  # Always normalize in slider mode
        )
        
        # Show JSON download/preview if requested
        if show_json_download:
            built_json_obj = _portfolio_json_from_manual(portfolio_name=portfolio_name, df=df, value_eur=None)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.download_button(
                    "Download JSON",
                    data=json.dumps(built_json_obj, indent=2).encode("utf-8"),
                    file_name=f"{(portfolio_name or 'portfolio').strip().replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"{key}_download",
                )
            with c2:
                with st.expander("Preview JSON"):
                    st.code(json.dumps(built_json_obj, indent=2), language="json")
        
        return p
    except Exception as e:
        st.error(str(e))
        return None


def _rolling_correlation_to_stocks(p: Portfolio, window_days: int = 252) -> pd.DataFrame:
    """
    Computes rolling correlations (window_days) of every non-stock asset versus the portfolio's stock sleeve.
    If no stocks are in the portfolio, downloads external stocks data (ACWE.MI) for comparison.
    """
    # Default stocks ticker to use when portfolio has no stocks
    DEFAULT_STOCKS_TICKER = "ACWE.MI"
    
    prices = p._prices_df().dropna(how="all").sort_index()
    if prices.empty:
        return pd.DataFrame()

    assets_map: dict[str, str] = getattr(p, "assets", {})
    tickers: list[str] = list(getattr(p, "tickers", []))
    stock_tickers = [t for t in tickers if assets_map.get(t, "").strip().lower() == "stocks"]
    
    # If no stocks in portfolio, download external stocks data
    if not stock_tickers:
        try:
            stocks_prices = _cached_download_prices((DEFAULT_STOCKS_TICKER,))
            if stocks_prices.empty or DEFAULT_STOCKS_TICKER not in stocks_prices.columns:
                return pd.DataFrame()
            # Align external stocks with portfolio prices
            stocks_prices = stocks_prices[DEFAULT_STOCKS_TICKER].reindex(prices.index).ffill().bfill()
            stock_returns = stocks_prices.pct_change().dropna()
            # All portfolio assets should be correlated vs stocks
            non_stock_tickers = list(tickers)
        except Exception:
            return pd.DataFrame()
    else:
        returns = prices.pct_change().dropna(how="all")
        stock_returns = returns[stock_tickers].mean(axis=1).dropna()
        non_stock_tickers = [t for t in tickers if t not in stock_tickers]
    
    if stock_returns.empty:
        return pd.DataFrame()

    # Compute returns for portfolio assets
    returns = prices.pct_change().dropna(how="all")
    
    min_periods = min(window_days, 126)
    corr_series: list[pd.Series] = []
    for ticker in non_stock_tickers:
        if ticker not in returns.columns:
            continue
        series = returns[ticker].dropna()
        if series.empty:
            continue
        pair = pd.concat([stock_returns, series], axis=1).dropna(how="any")
        if pair.empty:
            continue
        corr = pair.iloc[:, 0].rolling(window=window_days, min_periods=min_periods).corr(pair.iloc[:, 1])
        short_map = _get_short_name_map()
        corr_series.append(corr.rename(short_map.get(ticker, ticker)))

    if not corr_series:
        return pd.DataFrame()
    return pd.concat(corr_series, axis=1).dropna(how="all")


def _drawdown_series(value: pd.Series) -> pd.Series:
    """
    Compute drawdown series from a value/equity series:
      dd(t) = value(t) / peak(t) - 1
    Returns a series with the same index (NaNs dropped).
    """
    if value is None or len(value) == 0:
        return pd.Series(dtype=float)
    v = pd.to_numeric(value, errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    peak = v.cummax()
    dd = (v / peak) - 1.0
    return dd


def _fmt_days(val: float | None) -> str:
    if val is None or not np.isfinite(float(val)):
        return "—"
    d = int(round(float(val)))
    return f"{d}d"


def _render_drawdown_chart_single(
    value_series: pd.Series,
    *,
    title: str = "Drawdown",
    height: int = 260,
) -> None:
    dd = _drawdown_series(value_series)
    if dd.empty:
        st.info("Drawdown unavailable.")
        return
    dd_pct = dd * 100.0
    df = dd_pct.to_frame(name="Drawdown (%)").reset_index().rename(columns={"index": "Date"})
    min_dd = float(dd_pct.min()) if not dd_pct.empty else -1.0
    # Pad the domain a bit for readability
    dom_min = min(-1.0, min_dd - 1.0)

    st.markdown(f"#### {title}")
    chart = (
        alt.Chart(df)
        .mark_area(color="#ef4444", opacity=0.28)
        .encode(
            x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y")),
            y=alt.Y("Drawdown (%):Q", title="Drawdown (%)", scale=alt.Scale(domain=[dom_min, 0.0])),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Drawdown (%):Q", title="Drawdown", format=".2f"),
            ],
        )
        .properties(height=height)
    )
    line = alt.Chart(df).mark_line(color="#ef4444", strokeWidth=1.8).encode(
        x="Date:T",
        y="Drawdown (%):Q",
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#bbbbbb", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((chart + line + zero).interactive(), width="stretch")

    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown(
            """
**Drawdown** measures how far the portfolio is below its prior peak at each point in time.

- **0%** means “at a new all-time high”.
- **-20%** means the portfolio is 20% below its previous peak.

**Longest drawdown period** is the longest stretch of time spent below the previous peak (i.e., “underwater”), measured in days.
            """
        )


def _render_drawdown_chart_multi(
    value_series_by_name: dict[str, pd.Series],
    *,
    title: str = "Drawdown",
    height: int = 320,
    portfolio_order: list[str] | None = None,
) -> None:
    rows: list[dict[str, object]] = []
    min_dd = 0.0
    if value_series_by_name is None:
        value_series_by_name = {}
    items: list[tuple[str, pd.Series]] = list(value_series_by_name.items())
    if portfolio_order:
        ordered: list[tuple[str, pd.Series]] = []
        seen: set[str] = set()
        for k in portfolio_order:
            if k in value_series_by_name:
                ordered.append((k, value_series_by_name[k]))
                seen.add(k)
        for k, v in items:
            if k not in seen:
                ordered.append((k, v))
        items = ordered

    for name, v in items:
        dd = _drawdown_series(v)
        if dd.empty:
            continue
        dd_pct = dd * 100.0
        if not dd_pct.empty:
            min_dd = float(min(min_dd, float(dd_pct.min())))
        for dt, val in dd_pct.items():
            rows.append({"Date": dt, "Portfolio": str(name), "Drawdown (%)": float(val)})

    if not rows:
        st.info("Drawdown unavailable.")
        return

    df = pd.DataFrame(rows)
    dom_min = min(-1.0, float(min_dd) - 1.0)

    st.markdown(f"#### {title}")
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y")),
            y=alt.Y("Drawdown (%):Q", title="Drawdown (%)", scale=alt.Scale(domain=[dom_min, 0.0])),
            color=alt.Color("Portfolio:N", title="Portfolio", sort=portfolio_order if portfolio_order else None),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Portfolio:N", title="Portfolio"),
                alt.Tooltip("Drawdown (%):Q", title="Drawdown", format=".2f"),
            ],
        )
        .properties(height=height)
        .interactive()
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#bbbbbb", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((chart + zero), width="stretch")

    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown(
            """
This compares **drawdowns** (peak-to-trough declines) across portfolios.

Lower (more negative) values mean deeper declines from previous highs. Portfolios that recover faster will show shorter “underwater” stretches.
            """
        )


def _compare_portfolios_streamlit(
    portfolios: list[tuple[str, Portfolio]],
    *,
    rf_annual: float,
    rebalance_frequency: str,
    initial_amount: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    if len(portfolios) < 2:
        raise ValueError("Need at least 2 portfolios to compare.")

    # Align each portfolio internally first
    for _, p in portfolios:
        p.adjust_dates(debug=False)

    # Common overlapping index across portfolios
    common_index = None
    for _, p in portfolios:
        px = p._prices_df().dropna(how="any").sort_index()
        idx = px.index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) < 3:
        raise ValueError("Portfolios do not have enough overlapping price history to compare.")
    common_index = common_index.sort_values()

    rows: list[dict[str, object]] = []
    value_series: dict[str, pd.Series] = {}

    for name, p in portfolios:
        if not hasattr(p, "target_weights_pct"):
            raise ValueError(f"Portfolio '{name}' is missing per-asset target weights.")

        wt_pct = [float(p.target_weights_pct[t]) for t in p.tickers]
        wt = Portfolio._normalize_weights_to_fraction(wt_pct)
        weights = {t: float(w) for t, w in zip(p.tickers, wt)}

        prices_df = p._prices_df().reindex(common_index).dropna(how="any")
        v = Portfolio.backtest_value_series(
            prices_df,
            weights,
            rebalance_frequency=str(rebalance_frequency),
            initial_value=float(initial_amount),
        )
        v = v.reindex(common_index).dropna()
        value_series[name] = v

        stats = Portfolio.backtest_stats(v, rf_annual=float(rf_annual))
        rows.append(
            {
                "Portfolio": name,
                "Total Return": float(stats.get("total_return", float("nan"))),
                "CAGR": float(stats.get("cagr", float("nan"))),
                "Vol (ann.)": float(stats.get("vol_annual", float("nan"))),
                "Sharpe": float(stats.get("sharpe", float("nan"))),
                "Sortino": float(stats.get("sortino", float("nan"))),
                "Max Drawdown": float(stats.get("max_drawdown", float("nan"))),
                "Ulcer Index": float(stats.get("ulcer_index", float("nan"))),
            }
        )

    df = pd.DataFrame(rows).set_index("Portfolio")
    return df, value_series


def _run_and_capture_stdout(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def _render_llm_query_ui(
    *,
    key_prefix: str,
    llm_prompt: str,
    title: str = "Ask an LLM",
) -> None:
    """
    Render a reusable interactive LLM chat UI component.
    
    The first message is the built prompt and the first response comes from the LLM.
    After that, the user can continue the conversation interactively.
    
    Includes:
    - API key check
    - Model selection (free models only)
    - Usage/limits display
    - Start chat button (for initial query)
    - Interactive chat interface with message history
    """
    if title:
        st.markdown(f"### {title}")
    
    # Check API key
    api_key = get_api_key()
    if not api_key:
        st.error(
            "**OPENROUTER_API_KEY** environment variable is not set. "
            "Please set it to use LLM features."
        )
        return
    
    # Session state keys for this chat instance
    chat_history_key = f"{key_prefix}_chat_history"
    chat_started_key = f"{key_prefix}_chat_started"
    chat_model_key = f"{key_prefix}_chat_model"
    total_tokens_key = f"{key_prefix}_total_tokens"
    chat_error_key = f"{key_prefix}_chat_error"  # Track last error separately
    
    # Fetch and display limits
    with st.expander("API Usage and Limits", expanded=False):
        limits = fetch_limits(api_key)
        if limits is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                if limits.credits_remaining is not None and limits.credit_limit is not None:
                    st.metric(
                        "Credits Remaining",
                        f"${limits.credits_remaining:.4f}",
                        delta=None,
                    )
                elif limits.credits_remaining is not None:
                    st.metric("Credits Remaining", f"${limits.credits_remaining:.4f}")
                else:
                    st.metric("Credits Remaining", "Unlimited (free tier)")
            with col2:
                if limits.usage_daily is not None:
                    st.metric("Daily Usage", f"${limits.usage_daily:.4f}")
                else:
                    st.metric("Daily Usage", "N/A")
            with col3:
                if limits.rate_limit_requests and limits.rate_limit_interval:
                    st.metric(
                        "Rate Limit",
                        f"{limits.rate_limit_requests} req/{limits.rate_limit_interval}",
                    )
                else:
                    st.metric("Rate Limit", "Default")
            
            st.caption(
                "**Free tier limits:** 20 requests/minute, 200 requests/day. "
                "Models with `:free` suffix have no token cost but may have usage limits."
            )
        else:
            st.warning("Could not fetch usage limits.")
            st.caption(
                "**Typical free tier limits:** 20 requests/minute, 200 requests/day."
            )
    
    # Model selection
    # Use cached free models list to avoid repeated API calls
    cache_key = f"{key_prefix}_free_models_cache"
    if cache_key not in st.session_state:
        with st.spinner("Loading available models..."):
            st.session_state[cache_key] = fetch_free_models(api_key)
    
    free_models = st.session_state[cache_key]
    
    # Find default model index
    default_model = "meta-llama/llama-3.3-70b-instruct:free"
    default_idx = 0
    if default_model in free_models:
        default_idx = free_models.index(default_model)
    
    # Disable model selection once chat has started
    chat_started = st.session_state.get(chat_started_key, False)
    
    selected_model = st.selectbox(
        "Select model (free tier only)",
        options=free_models,
        index=default_idx,
        key=f"{key_prefix}_model_select",
        help="All listed models are free to use. Some may have daily usage limits.",
        disabled=chat_started,
    )
    
    # Initialize advanced settings in session state with defaults if not present
    max_tokens_key = f"{key_prefix}_max_tokens"
    temperature_key = f"{key_prefix}_temperature"
    if max_tokens_key not in st.session_state:
        st.session_state[max_tokens_key] = 4000
    if temperature_key not in st.session_state:
        st.session_state[temperature_key] = 0.7
    
    # Advanced settings - use a container to prevent duplication issues
    advanced_container = st.container()
    with advanced_container:
        with st.expander("Advanced Settings", expanded=False):
            st.slider(
                "Max response tokens",
                min_value=500,
                max_value=8000,
                value=st.session_state[max_tokens_key],
                step=500,
                key=max_tokens_key,
                help="Maximum number of tokens in the LLM response. Higher = longer responses.",
                disabled=chat_started,
            )
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state[temperature_key],
                step=0.1,
                key=temperature_key,
                help="Higher = more creative, lower = more focused/deterministic.",
                disabled=chat_started,
            )
    
    # If chat hasn't started yet, show the "Start Chat" button
    if not chat_started:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Start Chat with LLM", type="primary", key=f"{key_prefix}_start_chat_btn"):
                # Initialize chat history with the built prompt as the first user message
                st.session_state[chat_history_key] = [
                    {"role": "user", "content": llm_prompt}
                ]
                st.session_state[chat_started_key] = True
                st.session_state[chat_model_key] = selected_model
                st.session_state[total_tokens_key] = 0
                st.rerun()
        with col2:
            # Show a hint about what will happen
            st.caption("Click to send the analysis prompt and start an interactive conversation.")
        return
    
    # System message to enforce formatting and conciseness
    SYSTEM_MESSAGE = {
        "role": "system",
        "content": (
            "You are a helpful financial assistant. Follow these rules strictly:\n"
            "1. NEVER use markdown headers (no #, ##, ###, etc.). Use plain text only.\n"
            "2. You may use bullet points, numbered lists, and bold/italic for emphasis.\n"
            "3. Be concise and to the point. Avoid verbosity while remaining complete and clear.\n"
            "4. Get straight to the actionable advice or answer."
        ),
    }
    
    # Chat has started - show the chat interface
    chat_history: list[dict[str, str]] = st.session_state.get(chat_history_key, [])
    model = st.session_state.get(chat_model_key, selected_model)
    total_tokens_used = st.session_state.get(total_tokens_key, 0)
    last_error: str | None = st.session_state.get(chat_error_key, None)
    
    # Check if we need to get the initial response (first user message sent, no assistant reply yet)
    # Also check that there's no pending error (user needs to retry or modify message first)
    needs_response = (
        len(chat_history) > 0 
        and chat_history[-1]["role"] == "user" 
        and last_error is None
    )
    
    # Display chat history FIRST (always visible)
    st.markdown("#### Conversation")
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(chat_history):
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                with st.chat_message("user"):
                    # For the first message (the built prompt), show a collapsed version
                    if i == 0:
                        with st.expander("📋 Analysis Prompt (click to expand)", expanded=False):
                            st.markdown(content)
                    else:
                        st.markdown(content)
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(content)
        
        # If there's a pending error, show it inside the chat with proper styling
        if last_error is not None:
            with st.chat_message("assistant"):
                st.error(f"**Request failed:** {last_error}")
                st.caption("You can retry the request or type a different message below.")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🔄 Retry", key=f"{key_prefix}_retry_btn", type="primary"):
                        # Clear the error and trigger a new request
                        st.session_state[chat_error_key] = None
                        st.rerun()
        
        # If waiting for response, show loading indicator INSIDE the chat
        elif needs_response:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get values from session state
                    max_tokens_val = st.session_state.get(max_tokens_key, 4000)
                    temperature_val = st.session_state.get(temperature_key, 0.7)
                    
                    # Build messages with system prompt
                    messages_to_send = [SYSTEM_MESSAGE] + chat_history
                    
                    response = chat_completion_messages(
                        messages=messages_to_send,
                        model=str(model),
                        api_key=api_key,
                        max_tokens=int(max_tokens_val),
                        temperature=float(temperature_val),
                    )
                    
                    if response.error:
                        # Store error separately - don't add to chat history
                        # This allows the user to retry without polluting the conversation
                        st.session_state[chat_error_key] = response.error
                    else:
                        # Clear any previous error
                        st.session_state[chat_error_key] = None
                        # Add assistant response to history
                        chat_history.append({"role": "assistant", "content": response.content})
                        st.session_state[chat_history_key] = chat_history
                        # Track total tokens
                        if response.total_tokens:
                            total_tokens_used += response.total_tokens
                            st.session_state[total_tokens_key] = total_tokens_used
                    
                    st.rerun()
    
    # Show token usage
    if total_tokens_used > 0:
        st.caption(f"Total tokens used in this conversation: {total_tokens_used:,}")
    
    # Chat input for follow-up messages
    user_input = st.chat_input("Type your follow-up question...", key=f"{key_prefix}_chat_input")
    
    if user_input:
        # Clear any pending error when user sends a new message
        if chat_error_key in st.session_state:
            del st.session_state[chat_error_key]
        # Add user message to history
        chat_history.append({"role": "user", "content": user_input})
        st.session_state[chat_history_key] = chat_history
        st.rerun()
    
    # Reset chat button
    st.markdown("")  # Spacing
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Reset Chat", key=f"{key_prefix}_reset_chat_btn"):
            # Clear chat-related session state (including error state)
            for key in [chat_history_key, chat_started_key, chat_model_key, total_tokens_key, chat_error_key]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col2:
        st.caption("Start a new conversation with the analysis prompt.")


def _render_title() -> None:
    # Use HTML to style the euro symbol with extra weight (font fallback makes it thinner)
    # st.markdown(
    #     '<h1 data-testid="stHeading" style="text-align: center;">CACH<span style="font-weight: 700;">€</span></h1>',
    #     unsafe_allow_html=True,
    # )
    st.markdown(
        """
        <h1 data-testid="stHeading" style="text-align: center;">
            CACH<span style="
                font-weight: 700; 
                display: inline-block; 
                transform: scaleX(1.3); 
                transform-origin: left;
                margin-right: 0.1em;
            ">€</span>
        </h1>
        """,
        unsafe_allow_html=True,
    )
    # Center ONLY this subtitle (other section headers should remain left-aligned).
    st.markdown(
        '<div style="text-align: center;"><h3 style="margin-top: 0;">Your financial assistant.</h3></div>',
        unsafe_allow_html=True,
    )


def _go(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


if "page" not in st.session_state:
    st.session_state["page"] = "home"

_render_title()

page = str(st.session_state.get("page", "home"))

if page != "home":
    c1, c2 = st.columns([1.5, 8.5])
    with c1:
        if st.button("← Back", type="secondary", key="nav_back", use_container_width=True):
            _go("home")
    st.markdown("")


if page == "home":
    # Center ONLY this home question (other section headers should remain left-aligned).
    st.markdown(
        '<div style="text-align: center;"><h3 style="margin-top: 0;">Hi, what do you need today?</h3></div>',
        unsafe_allow_html=True,
    )
    # st.markdown("")
    st.markdown("")  # Add spacing between question and buttons
    # Reserve space on both sides so the navigation buttons stay centered under the title.
    left, mid, right = st.columns([2.2, 3.6, 2.2])
    left.empty()
    right.empty()
    with mid:
        if st.button("Analyze a portfolio", type="primary", key="nav_analyze", use_container_width=True):
            _go("analyze")
        if st.button("Compare portfolios", type="primary", key="nav_compare", use_container_width=True):
            _go("compare")
        if st.button("Rebalance with new cash", type="primary", key="nav_rebalance", use_container_width=True):
            _go("rebalance")
        if st.button("What-if: add an asset", type="primary", key="nav_whatif", use_container_width=True):
            _go("whatif")
    
    # Global help section
    st.markdown("")
    with st.expander("ℹ️ What can this app do?", expanded=False):
        st.markdown("""
**Analyze a portfolio** — Evaluate a single portfolio's historical performance. See key metrics like CAGR, volatility, Sharpe ratio, and max drawdown. Visualize how your portfolio would have grown over time and understand correlations between assets.

**Compare portfolios** — Put multiple portfolios side-by-side to see which performed better historically. Useful for evaluating different allocation strategies (e.g., 60/40 vs 80/20) or comparing your portfolio against benchmarks.

**Rebalance with new cash** — Calculate how to allocate new money to bring your portfolio back to target weights without selling. Includes a macro dashboard (EU/DE + US snapshot + 12-month trend charts) and optional AI-assisted recommendations.

**What-if: add an asset** — Explore what would happen if you added a new asset to your portfolio. Analyze diversification benefits, risk-adjusted returns, and backtest the modified portfolio against your baseline.
        """)


elif page == "analyze":
    with st.expander("ℹ️ About this section", expanded=False):
        st.markdown("""
This section analyzes a single portfolio's historical performance using backtesting.

**What you'll get:**
- **Key metrics**: CAGR, volatility, Sharpe/Sortino ratios, max drawdown, and Ulcer Index
- **Value chart**: How your portfolio would have grown over your selected time period
- **Asset trajectories**: Individual performance of each asset in your portfolio
- **Correlation analysis**: Rolling 1-year correlation of each asset vs. stocks, helping you understand diversification

**How to use:** Create a portfolio using the sliders or load a pre-built one, adjust the backtest settings (rebalancing frequency, date range), then click "Run analysis".
        """)
    
    p, _ = portfolio_builder(key="analyze", title="Portfolio", allow_value=False)
    if p is not None:
        st.markdown("### Settings")
        rebalance_frequency, initial_amount, rf_annual = _backtest_controls(key_prefix="analyze")
        
        # Get available date range from portfolio data
        try:
            prices_raw = p._prices_df().dropna(how="any").sort_index()
            available_start = prices_raw.index.min().date() if not prices_raw.empty else None
            available_end = prices_raw.index.max().date() if not prices_raw.empty else None
        except Exception:
            available_start, available_end = None, None

        if available_start and available_end:
            # st.caption(f"Available data range: **{available_start}** to **{available_end}**")
            date_c1, date_c2 = st.columns(2)
            with date_c1:
                user_start = st.date_input(
                    "Start date",
                    value=available_start,
                    min_value=available_start,
                    max_value=available_end,
                    key="analyze_start_date",
                )
            with date_c2:
                user_end = st.date_input(
                    "End date",
                    value=available_end,
                    min_value=available_start,
                    max_value=available_end,
                    key="analyze_end_date",
                )
        else:
            st.warning("Could not determine available date range from portfolio data.")
            user_start, user_end = None, None

        y_scale = st.radio(
            "Y-axis scale (for value charts)",
            options=["Linear", "Logarithmic"],
            index=0,
            horizontal=True,
            key="analyze_y_scale",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            run = st.button("Run analysis", type="primary", key="analyze_run")
        with c2:
            show_chart = st.checkbox("Show charts", value=True, key="analyze_charts")

        # Run analysis and store in session state
        if run:
            with st.status("Running portfolio analysis...", expanded=True) as status:
                try:
                    total_start = time.time()
                    
                    step_ph = st.empty()
                    step_ph.write("Preparing data...")
                    step_start = time.time()
                    
                    p.adjust_dates(debug=False)
                    prices_full = p._prices_df().dropna(how="any").sort_index()

                    # Filter prices by user-selected date range BEFORE computing
                    if user_start and user_end:
                        prices_filtered = prices_full.loc[str(user_start):str(user_end)]
                    else:
                        prices_filtered = prices_full

                    if prices_filtered.empty:
                        raise ValueError("No data available in the selected date range.")

                    start_date = prices_filtered.index.min().date()
                    end_date = prices_filtered.index.max().date()

                    _timed_step(step_ph, "Preparing data...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Computing portfolio value...")
                    step_start = time.time()

                    # Get target weights
                    target_weights_pct = getattr(p, "target_weights_pct", {})
                    if not target_weights_pct:
                        raise ValueError("Portfolio is missing target weights.")

                    wt_pct = [float(target_weights_pct[t]) for t in p.tickers]
                    wt = Portfolio._normalize_weights_to_fraction(wt_pct)
                    weights = {t: float(w) for t, w in zip(p.tickers, wt)}

                    # Compute value series on filtered date range
                    value_series = Portfolio.backtest_value_series(
                        prices_filtered,
                        weights,
                        rebalance_frequency=str(rebalance_frequency),
                        initial_value=float(initial_amount),
                    ).dropna()

                    if value_series.empty:
                        raise ValueError("Portfolio value series is empty for the selected parameters.")

                    stats = Portfolio.backtest_stats(value_series, rf_annual=float(rf_annual))

                    _timed_step(step_ph, "Computing portfolio value...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Computing asset trajectories...")
                    step_start = time.time()

                    # Compute asset values on filtered range (use Short names for chart labels)
                    short_map = _get_short_name_map()
                    asset_values = float(initial_amount) * (prices_filtered / prices_filtered.iloc[0])
                    asset_values = asset_values.rename(columns={t: short_map.get(t, t) for t in asset_values.columns})

                    _timed_step(step_ph, "Computing asset trajectories...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Computing correlations...")
                    step_start = time.time()

                    # Compute correlations (uses external stocks data if no stocks in portfolio)
                    corr_df = _rolling_correlation_to_stocks(p)
                    if user_start and user_end and not corr_df.empty:
                        corr_df = corr_df.loc[str(user_start):str(user_end)]

                    # Build allocation data (use full names for pie chart legend)
                    labels = [p._label(t) for t in target_weights_pct.keys()]
                    sizes = [float(target_weights_pct[t]) for t in target_weights_pct.keys()]
                    legend_labels = [f"{label} ({weight:.1f}%)" for label, weight in zip(labels, sizes)]
                    alloc_df = pd.DataFrame({"Asset": labels, "Weight (%)": sizes, "Legend": legend_labels})

                    _timed_step(step_ph, "Computing correlations...", step_start)
                    
                    # Store everything in session state
                    st.session_state["analyze_results"] = {
                        "start_date": start_date,
                        "end_date": end_date,
                        "value_series": value_series,
                        "stats": stats,
                        "asset_values": asset_values,
                        "corr_df": corr_df,
                        "alloc_df": alloc_df,
                    }
                    
                    total_elapsed = time.time() - total_start
                    status.update(label=f"Analysis complete! ({total_elapsed:.2f}s)", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="Analysis failed", state="error", expanded=False)
                    st.error(str(e))
                    st.session_state.pop("analyze_results", None)

        # Display results from session state
        if "analyze_results" in st.session_state:
            results = st.session_state["analyze_results"]
            st.markdown("### Results")
            st.caption(f"Analysis period: {results['start_date']} → {results['end_date']}")

            stats = results["stats"]
            # 2×4 metrics grid
            m1, m2, m3, m4 = st.columns(4)
            total_ret = float(stats.get("total_return", float("nan")))
            m1.metric("Total Return", f"{total_ret*100:.2f}%" if np.isfinite(total_ret) else "—")
            m2.metric("CAGR", f"{stats['cagr']*100:.2f}%" if np.isfinite(stats["cagr"]) else "—")
            m3.metric("Vol (ann.)", f"{stats['vol_annual']*100:.2f}%" if np.isfinite(stats["vol_annual"]) else "—")
            max_dd = float(stats.get("max_drawdown", float("nan")))
            m4.metric("Max Drawdown", f"{max_dd*100:.2f}%" if np.isfinite(max_dd) else "—")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Sharpe", f"{stats['sharpe']:.2f}" if np.isfinite(stats["sharpe"]) else "—")
            m6.metric("Sortino", f"{stats['sortino']:.2f}" if np.isfinite(stats["sortino"]) else "—")
            ulcer = float(stats.get("ulcer_index", float("nan")))
            m7.metric("Ulcer Index", f"{ulcer:.2f}" if np.isfinite(ulcer) else "—")
            ldd = float(stats.get("longest_drawdown_days", float("nan")))
            # Fallback for older cached results (missing key) or NaN values.
            if not np.isfinite(ldd):
                try:
                    ldd = float(Portfolio.longest_drawdown_days(results["value_series"]))
                except Exception:
                    ldd = float("nan")
            m8.metric("Longest Drawdown", _fmt_days(ldd))

            with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                st.markdown("""
**Total Return** — Cumulative gain/loss over the entire period. A 50% total return means €10,000 became €15,000.

**CAGR (Compound Annual Growth Rate)** — The average annual return assuming profits are reinvested. A 10% CAGR means the portfolio grew by 10% per year on average.

**Volatility (annualized)** — A measure of how much returns fluctuate. Higher volatility means more uncertainty. Typically, 10-15% is moderate; above 20% is considered high.

**Sharpe Ratio** — Risk-adjusted return calculated as (Return - Risk-free rate) / Volatility. Higher is better: below 0.5 is poor; 0.5-1.0 is acceptable; above 1.0 is good; above 2.0 is excellent.

**Sortino Ratio** — Similar to Sharpe, but only penalizes *downside* volatility (losses), not upside movements. More relevant if you care primarily about avoiding losses rather than overall stability.

**Max Drawdown** — The largest peak-to-trough decline during the period. A -30% max drawdown means at some point the portfolio lost 30% from its previous high before recovering.

**Longest Drawdown** — The longest stretch of time the portfolio stayed below its previous peak (“underwater”), measured in days.

**Ulcer Index** — Measures downside volatility by computing the quadratic mean of percentage drawdowns from peak. Lower is better: below 5 is excellent, 5-10 is good, 10-15 is moderate, above 15 indicates significant drawdown stress.
                """)

            # Pie chart
            alloc_df = results["alloc_df"]
            st.markdown("#### Target allocation")
            pie = (
                alt.Chart(alloc_df)
                .mark_arc(innerRadius=50, stroke="#0e1117", strokeWidth=1.4)
                .encode(
                    theta=alt.Theta("Weight (%):Q", title="Weight (%)"),
                    color=alt.Color(
                        "Legend:N",
                        title="Allocation",
                        legend=alt.Legend(labelLimit=400),  # Allow longer legend labels
                    ),
                    tooltip=[
                        alt.Tooltip("Asset:N", title="Asset"),
                        alt.Tooltip("Weight (%):Q", title="Weight (%)", format=".1f"),
                    ],
                )
                .properties(width=280, height=280)
            )
            st.altair_chart(pie, width="stretch")

            if show_chart:
                y_scale_type = "log" if y_scale == "Logarithmic" else "linear"
                x_axis_format = alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y"))

                # Portfolio value chart
                value_series = results["value_series"]
                value_df = value_series.to_frame(name="Value")
                value_chart_df = value_df.reset_index().rename(columns={"index": "Date"})
                st.markdown("#### Portfolio value over time")
                chart_value = alt.Chart(value_chart_df).mark_line(color="#60a5fa", strokeWidth=2.4).encode(
                    x=x_axis_format,
                    y=alt.Y("Value:Q", title="Value (EUR)", scale=alt.Scale(type=y_scale_type)),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Date"),
                        alt.Tooltip("Value:Q", title="Value (EUR)", format=",.2f"),
                    ],
                ).properties(height=320).interactive()
                st.altair_chart(chart_value, width="stretch")

                # Asset value trajectories
                asset_values = results["asset_values"]
                if not asset_values.empty:
                    st.markdown("#### Asset value trajectories")
                    asset_labels = list(asset_values.columns)

                    selected_assets = st.multiselect(
                        "Select assets to display",
                        options=asset_labels,
                        default=asset_labels,
                        key="analyze_asset_selection",
                    )

                    if selected_assets:
                        asset_values_filtered = asset_values[selected_assets]
                        asset_long = (
                            asset_values_filtered.reset_index()
                            .rename(columns={"index": "Date"})
                            .melt(id_vars="Date", var_name="Asset", value_name="Value")
                        )
                        chart_assets = (
                            alt.Chart(asset_long)
                            .mark_line(strokeWidth=1.9)
                            .encode(
                                x=x_axis_format,
                                y=alt.Y("Value:Q", title="Value (EUR)", scale=alt.Scale(type=y_scale_type)),
                                color=alt.Color("Asset:N", title="Asset"),
                                tooltip=[
                                    alt.Tooltip("Date:T", title="Date"),
                                    alt.Tooltip("Asset:N", title="Asset"),
                                    alt.Tooltip("Value:Q", title="Value (EUR)", format=",.2f"),
                                ],
                            )
                            .properties(height=320)
                            .interactive()
                        )
                        st.altair_chart(chart_assets, width="stretch")
                    else:
                        st.info("Select at least one asset to display the chart.")
                else:
                    st.info("Asset price history unavailable; cannot render per-asset chart.")

                # Correlation chart
                corr_df = results["corr_df"]
                if corr_df.empty:
                    st.info("Not enough data to compute rolling correlations versus Stocks.")
                else:
                    st.markdown("#### Rolling 1Y correlation vs Stocks")
                    corr_labels = list(corr_df.columns)

                    selected_corr_assets = st.multiselect(
                        "Select assets to display",
                        options=corr_labels,
                        default=corr_labels,
                        key="analyze_corr_selection",
                    )

                    if selected_corr_assets:
                        corr_df_filtered = corr_df[selected_corr_assets]
                        corr_long = (
                            corr_df_filtered.reset_index()
                            .rename(columns={"index": "Date"})
                            .melt(id_vars="Date", var_name="Asset", value_name="Correlation")
                        )
                        corr_chart = (
                            alt.Chart(corr_long)
                            .mark_line(strokeWidth=1.8)
                            .encode(
                                x=x_axis_format,
                                y=alt.Y("Correlation:Q", title="Correlation", scale=alt.Scale(domain=[-1.05, 1.05])),
                                color=alt.Color("Asset:N", title="Asset"),
                                tooltip=[
                                    alt.Tooltip("Date:T", title="Date"),
                                    alt.Tooltip("Asset:N", title="Asset"),
                                    alt.Tooltip("Correlation:Q", title="Correlation", format=".2f"),
                                ],
                            )
                            .properties(height=320)
                        )
                        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#bbbbbb", strokeDash=[4, 4]).encode(
                            y="y"
                        )
                        st.altair_chart((corr_chart + zero_line).interactive(), width="stretch")
                    else:
                        st.info("Select at least one asset to display the chart.")
                    
                    with st.expander("ℹ️ What does this chart show?", expanded=False):
                        st.markdown("""
This chart shows how each non-stock asset's returns have moved relative to your stock allocation over rolling 1-year windows.

**Why correlate against Stocks?** Stocks are typically the core growth engine of a portfolio and the primary source of both returns and risk. The goal of diversification is to hold other assets that behave differently from stocks, especially during downturns, so they can cushion losses when stocks fall.

**Interpreting the values:**

**+1.0** — Moves perfectly in sync with stocks (no diversification benefit).

**0.0** — No relationship with stocks (good diversification).

**-1.0** — Moves opposite to stocks (strong hedge, rare in practice).

**Reference ranges:** Correlations below 0.3 provide meaningful diversification. Correlations above 0.6 suggest the asset moves largely with stocks and adds limited protection. Negative correlations are ideal during stock downturns but uncommon for most asset classes.

**Why "rolling"?** Correlations change over time, especially during market stress. A bond fund that normally has low correlation to stocks may suddenly become more correlated during a crisis. This chart helps you see how stable the diversification benefit has been historically.
                        """)

                # Drawdown chart (requested: after rolling correlation graph)
                _render_drawdown_chart_single(value_series, title="Drawdown")


elif page == "compare":
    with st.expander("ℹ️ About this section", expanded=False):
        st.markdown("""
This section compares multiple portfolios side-by-side using the same time period and settings.

**What you'll get:**
- **Allocation overview**: A table showing how each portfolio is allocated across assets
- **Performance comparison**: Key metrics (CAGR, volatility, Sharpe, max drawdown, etc.) for each portfolio
- **Value chart**: All portfolios on the same chart so you can visually compare growth trajectories

**How to use:** Select built-in portfolios, upload your own JSON files, or create manual portfolios using sliders. All portfolios will be compared over their common date range. Adjust settings like rebalancing frequency, then click "Run comparison".
        """)
    
    st.markdown("### Portfolios to compare")

    portfolios_dir = os.path.join(CACHE_DIR, "portfolios")
    builtin_paths = []
    try:
        for fname in sorted(os.listdir(portfolios_dir)):
            if fname.lower().endswith(".json"):
                builtin_paths.append(os.path.join(portfolios_dir, fname))
    except Exception:
        builtin_paths = []

    selected_builtin = st.multiselect(
        "Built-in portfolios",
        options=builtin_paths,
        format_func=lambda p: os.path.basename(p),
        default=builtin_paths[:2] if len(builtin_paths) >= 2 else [],
        key="compare_builtin",
    )
    
    # Show previews for selected built-in portfolios
    for path in selected_builtin:
        try:
            p_preview = _cached_load_portfolio(path)
            _render_portfolio_preview(p_preview)
        except Exception:
            pass

    uploaded = st.file_uploader("Upload additional portfolio JSONs", type=["json"], accept_multiple_files=True, key="compare_upload")
    
    # Show previews for uploaded portfolios
    if uploaded:
        for up in uploaded:
            try:
                raw = up.getvalue()
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    st.warning(f"Upload '{up.name}': invalid JSON (could not parse).")
                    continue
                ok, errs = _validate_portfolio_json_obj(obj)
                if not ok:
                    st.warning(f"Upload '{up.name}' rejected:\n- " + "\n- ".join(errs))
                    continue

                tmp = _safe_temp_json(raw)
                p_preview = Portfolio.from_json(tmp)
                _render_portfolio_preview(p_preview)
            except Exception:
                pass

    # Help users create valid JSONs by providing examples.
    _render_example_json_ui(key_prefix="compare")
    
    # Multiple manual portfolios
    # st.markdown("#### Manual portfolios")
    num_manual = st.number_input(
        "Number of manual portfolios to add",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        key="compare_num_manual",
    )
    
    manual_portfolios: list[Portfolio] = []
    for i in range(int(num_manual)):
        if i > 0:
            st.divider()
        mp = _manual_portfolio_builder(key=f"compare_manual_{i}", title="", show_json_download=True)
        if mp is not None:
            manual_portfolios.append(mp)

    # Collect all portfolios to determine available date range
    all_portfolios: list[tuple[str, Portfolio]] = []
    for path in selected_builtin:
        try:
            p_temp = _cached_load_portfolio(path)
            name = getattr(p_temp, "name", None) or os.path.basename(path)
            all_portfolios.append((str(name), p_temp))
        except Exception:
            pass
    if uploaded:
        for up in uploaded:
            try:
                raw = up.getvalue()
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                ok, errs = _validate_portfolio_json_obj(obj)
                if not ok:
                    continue

                tmp = _safe_temp_json(raw)
                p_temp = Portfolio.from_json(tmp)
                name = getattr(p_temp, "name", None) or up.name
                all_portfolios.append((str(name), p_temp))
            except Exception:
                pass
    for mp in manual_portfolios:
        all_portfolios.append((getattr(mp, "name", "Manual"), mp))

    # Settings section (after portfolios)
    st.markdown("### Settings")
    rebalance_frequency, initial_amount, rf_annual = _backtest_controls(key_prefix="compare")

    # Determine common date range across all selected portfolios
    available_start, available_end = None, None
    if len(all_portfolios) >= 2:
        try:
            common_index = None
            for _, p_temp in all_portfolios:
                p_temp.adjust_dates(debug=False)
                px = p_temp._prices_df().dropna(how="any").sort_index()
                idx = px.index
                common_index = idx if common_index is None else common_index.intersection(idx)
            if common_index is not None and len(common_index) >= 3:
                common_index = common_index.sort_values()
                available_start = common_index.min().date()
                available_end = common_index.max().date()
        except Exception:
            pass

    if available_start and available_end:
        # st.caption(f"Common data range: **{available_start}** to **{available_end}**")
        date_c1, date_c2 = st.columns(2)
        with date_c1:
            user_start = st.date_input(
                "Start date",
                value=available_start,
                min_value=available_start,
                max_value=available_end,
                key="compare_start_date",
            )
        with date_c2:
            user_end = st.date_input(
                "End date",
                value=available_end,
                min_value=available_start,
                max_value=available_end,
                key="compare_end_date",
            )
    else:
        if len(all_portfolios) < 2:
            st.info("Select at least 2 portfolios to see the available date range.")
        else:
            st.warning("Could not determine common date range from selected portfolios.")
        user_start, user_end = None, None

    y_scale = st.radio(
        "Y-axis scale",
        options=["Linear", "Logarithmic"],
        index=0,
        horizontal=True,
        key="compare_y_scale",
    )

    run = st.button("Run comparison", type="primary", key="compare_run")
    if run:
        with st.status("Comparing portfolios...", expanded=True) as status:
            try:
                total_start = time.time()
                
                if len(all_portfolios) < 2:
                    raise ValueError("Select/upload at least two portfolios.")

                step_ph = st.empty()
                step_ph.write("Aligning portfolio data...")
                step_start = time.time()
                
                # Align each portfolio internally first
                for _, p in all_portfolios:
                    p.adjust_dates(debug=False)

                # Common overlapping index across portfolios
                common_index = None
                for _, p in all_portfolios:
                    px = p._prices_df().dropna(how="any").sort_index()
                    idx = px.index
                    common_index = idx if common_index is None else common_index.intersection(idx)
                if common_index is None or len(common_index) < 3:
                    raise ValueError("Portfolios do not have enough overlapping price history to compare.")
                common_index = common_index.sort_values()

                # Apply user-selected date range
                if user_start and user_end:
                    common_index = common_index[(common_index >= str(user_start)) & (common_index <= str(user_end))]
                    if len(common_index) < 3:
                        raise ValueError("Not enough data in the selected date range.")

                start_date = common_index.min().date()
                end_date = common_index.max().date()

                _timed_step(step_ph, "Aligning portfolio data...", step_start)
                step_ph = st.empty()
                step_ph.write("Running backtests...")
                step_start = time.time()
                
                rows: list[dict[str, object]] = []
                value_series: dict[str, pd.Series] = {}

                for name, p in all_portfolios:
                    if not hasattr(p, "target_weights_pct"):
                        raise ValueError(f"Portfolio '{name}' is missing per-asset target weights.")

                    wt_pct = [float(p.target_weights_pct[t]) for t in p.tickers]
                    wt = Portfolio._normalize_weights_to_fraction(wt_pct)
                    weights = {t: float(w) for t, w in zip(p.tickers, wt)}

                    prices_df = p._prices_df().reindex(common_index).dropna(how="any")
                    v = Portfolio.backtest_value_series(
                        prices_df,
                        weights,
                        rebalance_frequency=str(rebalance_frequency),
                        initial_value=float(initial_amount),
                    )
                    v = v.reindex(common_index).dropna()
                    value_series[name] = v

                    stats = Portfolio.backtest_stats(v, rf_annual=float(rf_annual))
                    rows.append(
                        {
                            "Portfolio": name,
                            "Total Return": float(stats.get("total_return", float("nan"))),
                            "CAGR": float(stats.get("cagr", float("nan"))),
                            "Vol (ann.)": float(stats.get("vol_annual", float("nan"))),
                            "Sharpe": float(stats.get("sharpe", float("nan"))),
                            "Sortino": float(stats.get("sortino", float("nan"))),
                            "Max Drawdown": float(stats.get("max_drawdown", float("nan"))),
                            "Longest Drawdown": float(stats.get("longest_drawdown_days", float("nan"))),
                            "Ulcer Index": float(stats.get("ulcer_index", float("nan"))),
                        }
                    )

                df = pd.DataFrame(rows).set_index("Portfolio")
                
                _timed_step(step_ph, "Running backtests...", step_start)
                
                # Build allocation data for session state
                short_map = _get_short_name_map()
                all_asset_names: set[str] = set()
                portfolio_allocations: dict[str, dict[str, float]] = {}
                for name, p in all_portfolios:
                    target_weights = getattr(p, "target_weights_pct", {})
                    alloc: dict[str, float] = {}
                    for ticker in p.tickers:
                        short_name = short_map.get(ticker, ticker)
                        weight = float(target_weights.get(ticker, 0.0))
                        alloc[short_name] = alloc.get(short_name, 0.0) + weight
                        all_asset_names.add(short_name)
                    portfolio_allocations[name] = alloc

                sorted_asset_names = sorted(all_asset_names)
                alloc_rows = []
                for name, alloc in portfolio_allocations.items():
                    row = {"Portfolio": name}
                    for asset_name in sorted_asset_names:
                        weight = alloc.get(asset_name, 0.0)
                        row[asset_name] = f"{weight:.1f}%"
                    alloc_rows.append(row)
                alloc_df = pd.DataFrame(alloc_rows).set_index("Portfolio")
                
                # Format statistics table
                pretty = df.copy()
                pct_cols = ["Total Return", "CAGR", "Vol (ann.)", "Max Drawdown"]
                for c in pct_cols:
                    pretty[c] = (pretty[c].astype(float) * 100.0).round(2).astype(str) + "%"
                if "Longest Drawdown" in pretty.columns:
                    pretty["Longest Drawdown"] = pretty["Longest Drawdown"].apply(lambda x: _fmt_days(float(x)) if pd.notna(x) else "—")
                pretty["Ulcer Index"] = pretty["Ulcer Index"].astype(float).round(2).astype(str)
                pretty["Sharpe"] = pretty["Sharpe"].astype(float).round(2).astype(str)
                pretty["Sortino"] = pretty["Sortino"].astype(float).round(2).astype(str)
                
                # Store results in session state
                st.session_state["compare_results"] = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "alloc_df": alloc_df,
                    "stats_df": pretty,
                    "value_series": value_series,
                }
                
                total_elapsed = time.time() - total_start
                status.update(label=f"Comparison complete! ({total_elapsed:.2f}s)", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Comparison failed", state="error", expanded=False)
                st.error(str(e))
                st.session_state.pop("compare_results", None)
    
    # Display results from session state (outside the status block)
    if "compare_results" in st.session_state:
        results = st.session_state["compare_results"]

        # Backward-compatible: older cached runs won't have Longest Drawdown in stats_df.
        # Also, early buggy computations may have filled it with "—" everywhere.
        stats_df = results.get("stats_df")
        if isinstance(stats_df, pd.DataFrame):
            needs_ldd = ("Longest Drawdown" not in stats_df.columns) or (
                "Longest Drawdown" in stats_df.columns and stats_df["Longest Drawdown"].astype(str).replace("—", "").str.strip().eq("").all()
            )
            if needs_ldd:
                try:
                    vs = results.get("value_series") or {}
                    ldd_map = {k: _fmt_days(float(Portfolio.longest_drawdown_days(v))) for k, v in vs.items()}
                    stats_df = stats_df.copy()
                    stats_df["Longest Drawdown"] = [ldd_map.get(idx, "—") for idx in stats_df.index]
                    results["stats_df"] = stats_df
                except Exception:
                    pass
        
        st.markdown("### Allocation overview")
        st.dataframe(results["alloc_df"].T, width="stretch")

        st.markdown("### Results")
        st.caption(f"Comparison period: {results['start_date']} → {results['end_date']}")
        st.dataframe(results["stats_df"].T, width="stretch")

        with st.expander("ℹ️ What do these metrics mean?", expanded=False):
            st.markdown("""
**Total Return** — Cumulative gain/loss over the entire period. A 50% total return means €10,000 became €15,000.

**CAGR (Compound Annual Growth Rate)** — Average annual return assuming reinvestment. Useful for comparing portfolios across different time periods.

**Volatility (annualized)** — How much returns fluctuate year-to-year. Lower values indicate more stable returns.

**Sharpe Ratio** — Risk-adjusted return calculated as (Return - Risk-free) / Volatility. Higher is better; it measures return earned per unit of risk taken.

**Sortino Ratio** — Like Sharpe, but only penalizes downside volatility. More relevant if you're primarily concerned about losses rather than overall stability.

**Max Drawdown** — The worst peak-to-trough decline, showing the maximum pain an investor would have experienced during the period.

**Longest Drawdown** — The longest stretch of time the portfolio stayed below its previous peak (“underwater”), measured in days.

**Ulcer Index** — Measures downside volatility using the quadratic mean of percentage drawdowns. Lower is better: below 5 is excellent, 5-10 is good, 10-15 is moderate, above 15 indicates significant stress.

*A portfolio with higher CAGR but also higher Max Drawdown or Ulcer Index may not suit risk-averse investors.*
            """)

        # Build Altair chart
        st.markdown("#### Portfolio value over time")
        chart_data = []
        for name, v in results["value_series"].items():
            for date, value in v.items():
                chart_data.append({"Date": date, "Portfolio": name, "Value": float(value)})
        chart_df = pd.DataFrame(chart_data)

        y_scale_type = "log" if y_scale == "Logarithmic" else "linear"
        x_axis_format = alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y"))

        chart = (
            alt.Chart(chart_df)
            .mark_line(strokeWidth=2.0)
            .encode(
                x=x_axis_format,
                y=alt.Y("Value:Q", title="Value (EUR)", scale=alt.Scale(type=y_scale_type)),
                color=alt.Color("Portfolio:N", title="Portfolio"),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date"),
                    alt.Tooltip("Portfolio:N", title="Portfolio"),
                    alt.Tooltip("Value:Q", title="Value (EUR)", format=",.2f"),
                ],
            )
            .properties(height=400)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")

        # Drawdown comparison chart (requested)
        _render_drawdown_chart_multi(results["value_series"], title="Drawdown (comparison)")


elif page == "rebalance":
    with st.expander("ℹ️ About this section", expanded=False):
        st.markdown("""
This section helps you allocate new cash to your portfolio to move closer to your target weights—without selling any existing positions.

**What you'll get:**
- **Rebalancing actions**: How much to invest in each asset to minimize deviation from targets
- **Portfolio diagnostics**: Recent performance metrics, volatility, and correlations for each asset
- **Macro dashboard**: A 2×4 snapshot grid (EU/DE + US) and four 12-month trend charts (EU/DE, US, USD/EUR, earnings yield) for context
- **AI assistance**: Generate a prompt for an LLM to get personalized rebalancing advice

**How to use:** Enter your current portfolio with both current weights (what you have now) and target weights (what you want). Specify your current portfolio value and the new cash amount, then click "Compute rebalance".
        """)
    
    p, _ = portfolio_builder(key="rebalance", title="Portfolio", allow_value=True)
    if p is not None:
        st.markdown("### Settings")
        new_cash = st.number_input("New cash to allocate (EUR)", min_value=0.0, value=2_000.0, step=100.0, key="rebalance_cash")

        # Get FRED API key from secrets
        fred_api_key = st.secrets.get("FRED_API_KEY", "").strip()
        if not fred_api_key:
            st.error("FRED_API_KEY is not set. Please add it to `.streamlit/secrets.toml`.")

        run = st.button("Compute rebalance", type="primary", key="rebalance_run")
        if run:
            with st.status("Computing rebalance...", expanded=True) as status:
                try:
                    total_start = time.time()
                    
                    if not fred_api_key:
                        raise ValueError("FRED_API_KEY environment variable is required.")

                    step_ph = st.empty()
                    step_ph.write("Computing optimal allocation...")
                    step_start = time.time()
                    
                    table = p.rebalance(float(new_cash))
                    table_transposed = table.T
                    short_map = _get_short_name_map()
                    # Use tickers to look up Short names (table columns are labels, not tickers)
                    table_transposed.columns = [short_map.get(t, table_transposed.columns[i]) for i, t in enumerate(p.tickers)]

                    _timed_step(step_ph, "Computing optimal allocation...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Computing portfolio diagnostics...")
                    step_start = time.time()
                    
                    diag = None
                    diag_transposed = None
                    diag_error = None
                    try:
                        diag = compute_rebalancing_diagnostics(p)
                        diag_transposed = diag.T
                        # Use tickers to look up Short names (diag columns are labels, not tickers)
                        diag_transposed.columns = [short_map.get(t, diag_transposed.columns[i]) for i, t in enumerate(p.tickers)]
                    except Exception as e:
                        diag_error = str(e)

                    _timed_step(step_ph, "Computing portfolio diagnostics...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Fetching macro data from FRED...")
                    step_start = time.time()
                    
                    snap = get_macro_snapshot(fred_api_key=fred_api_key, debug=False)
                    
                    # Fetch macro chart data
                    macro_charts: dict[str, list[dict[str, object]]] = {"eu_de": [], "us": [], "fx": [], "earnings": []}
                    macro_trends: dict[str, dict[str, float | None]] = {}
                    if snap is not None and Fred is not None:
                        try:
                            fred = Fred(api_key=fred_api_key)
                            now = pd.Timestamp.now()
                            # Need >12m history for YoY inflation computation
                            obs_start = now - pd.DateOffset(months=26)

                            def _append_series(chart_key: str, *, label: str, series: pd.Series) -> None:
                                s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
                                if s.empty:
                                    return
                                for dt, v in s.items():
                                    macro_charts[chart_key].append({"Date": dt, "Indicator": label, "Value": float(v)})

                            def _yoy_from_cpi(cpi: pd.Series) -> pd.Series:
                                s = pd.to_numeric(cpi, errors="coerce").dropna().sort_index()
                                yoy = (s.pct_change(12) * 100.0).dropna()
                                return yoy

                            def _trend_vals(series: pd.Series) -> dict[str, float | None]:
                                s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
                                if s.empty:
                                    return {"3m": None, "6m": None, "12m": None}
                                offsets = {"3m": pd.DateOffset(months=3), "6m": pd.DateOffset(months=6), "12m": pd.DateOffset(years=1)}
                                out: dict[str, float | None] = {}
                                for k, off in offsets.items():
                                    target = now - off
                                    idx = s.index.get_indexer([target], method="nearest")[0]
                                    out[k] = float(s.iloc[idx]) if 0 <= idx < len(s) else None
                                return out

                            # --- EU/DE ---
                            s_ecb = fred.get_series("ECBDFR", observation_start=obs_start).dropna()
                            s_de10y = fred.get_series("IRLTLT01DEM156N", observation_start=obs_start).dropna()
                            s_de_cpi = fred.get_series("DEUCPIALLMINMEI", observation_start=obs_start).dropna()
                            s_de_infl = _yoy_from_cpi(s_de_cpi)
                            # keep last 12m for charts
                            one_year_ago = now - pd.DateOffset(years=1)
                            _append_series("eu_de", label="ECB Overnight (%)", series=s_ecb.loc[s_ecb.index >= one_year_ago])
                            _append_series("eu_de", label="DE 10Y Yield (%)", series=s_de10y.loc[s_de10y.index >= one_year_ago])
                            _append_series("eu_de", label="DE Inflation YoY (%)", series=s_de_infl.loc[s_de_infl.index >= one_year_ago])

                            macro_trends["ecb_deposit_rate_pct"] = _trend_vals(s_ecb)
                            macro_trends["de_10y_yield_pct"] = _trend_vals(s_de10y)
                            macro_trends["de_inflation_yoy_pct"] = _trend_vals(s_de_infl)

                            # --- US ---
                            s_fed = fred.get_series("EFFR", observation_start=obs_start).dropna()
                            s_us10y = fred.get_series("DGS10", observation_start=obs_start).dropna()
                            s_us_cpi = fred.get_series("CPIAUCSL", observation_start=obs_start).dropna()
                            s_us_infl = _yoy_from_cpi(s_us_cpi)
                            _append_series("us", label="FED Overnight (%)", series=s_fed.loc[s_fed.index >= one_year_ago])
                            _append_series("us", label="US 10Y Yield (%)", series=s_us10y.loc[s_us10y.index >= one_year_ago])
                            _append_series("us", label="US Inflation YoY (%)", series=s_us_infl.loc[s_us_infl.index >= one_year_ago])

                            macro_trends["fed_risk_free_pct"] = _trend_vals(s_fed)
                            macro_trends["us_10y_yield_pct"] = _trend_vals(s_us10y)
                            macro_trends["us_inflation_yoy_pct"] = _trend_vals(s_us_infl)

                            # --- FX (USD/EUR spot) ---
                            s_usd_eur = fred.get_series("DEXUSEU", observation_start=obs_start).dropna()
                            _append_series("fx", label="USD/EUR", series=s_usd_eur.loc[s_usd_eur.index >= one_year_ago])
                            macro_trends["usd_eur"] = _trend_vals(s_usd_eur)

                            # --- Earnings yield (est.) ---
                            try:
                                ecy = get_global_earnings_yield_series(debug=False, lookback_days=500)
                                if ecy is not None and not ecy.empty:
                                    ecy_12m = ecy.loc[ecy.index >= one_year_ago]
                                    _append_series("earnings", label="Global EY Est. (%)", series=ecy_12m)
                                    macro_trends["global_earnings_yield_est_pct"] = _trend_vals(ecy)
                                else:
                                    s_us_ey, _note = get_us_earnings_yield_proxy_series_fred(
                                        fred, debug=False, observation_start=obs_start
                                    )
                                    if s_us_ey is not None and not s_us_ey.empty:
                                        s_us_ey_12m = s_us_ey.loc[s_us_ey.index >= one_year_ago]
                                        _append_series("earnings", label="US EY Proxy (FRED, %)", series=s_us_ey_12m)
                                        macro_trends["global_earnings_yield_est_pct"] = _trend_vals(s_us_ey)
                            except Exception:
                                pass
                        except Exception:
                            # best-effort only
                            macro_charts = {"eu_de": [], "us": [], "fx": [], "earnings": []}
                            macro_trends = {}

                    # Generate LLM prompt
                    llm_prompt = None
                    current_value = getattr(p, "current_value_eur", None)
                    if current_value is not None and snap is not None:
                        try:
                            llm_prompt = build_llm_rebalance_report(
                                portfolio=p,
                                rebalance_table=table,
                                diagnostics_table=diag,
                                macro_snapshot=snap,
                                current_value=float(current_value),
                                new_cash=float(new_cash),
                                macro_trends=macro_trends if macro_trends else None,
                            )
                        except Exception:
                            pass

                    _timed_step(step_ph, "Fetching macro data from FRED...", step_start)
                    
                    # Store all results in session state
                    st.session_state["rebalance_results"] = {
                        "table_transposed": table_transposed,
                        "diag_transposed": diag_transposed,
                        "diag_error": diag_error,
                        "snap": snap,
                        "macro_charts": macro_charts,
                        "llm_prompt": llm_prompt,
                        "portfolio_name": getattr(p, "name", "Portfolio"),
                    }
                    # Clear any previous LLM response when recomputing
                    st.session_state.pop("rebalance_llm_response", None)
                    
                    total_elapsed = time.time() - total_start
                    status.update(label=f"Rebalancing complete! ({total_elapsed:.2f}s)", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="Rebalancing failed", state="error", expanded=False)
                    st.error(str(e))
                    st.session_state.pop("rebalance_results", None)

        # Display results from session state (persists across reruns)
        if "rebalance_results" in st.session_state:
            results = st.session_state["rebalance_results"]
            
            st.markdown("### Rebalancing actions")
            
            # Extract data for visualization
            table_t = results["table_transposed"]
            
            # Compute deviation from Current Weight - Target Weight
            deviation_data = None
            try:
                current_row = None
                target_row = None
                for row_name in table_t.index:
                    if "current" in row_name.lower() and "weight" in row_name.lower():
                        current_row = row_name
                    if "target" in row_name.lower() and "weight" in row_name.lower():
                        target_row = row_name
                
                if current_row and target_row:
                    current_vals = table_t.loc[current_row].astype(float)
                    target_vals = table_t.loc[target_row].astype(float)
                    deviations = current_vals - target_vals
                    deviation_data = pd.DataFrame({
                        "Asset": deviations.index,
                        "Deviation (%)": deviations.values
                    })
            except Exception:
                deviation_data = None
            
            # Show table first, then chart below
            st.dataframe(table_t.round(2), width="stretch")
            
            if deviation_data is not None and not deviation_data.empty:
                st.markdown("#### Deviation from target")
                deviation_chart = (
                    alt.Chart(deviation_data)
                    .mark_bar()
                    .encode(
                        y=alt.Y("Asset:N", title=None, sort="-x"),
                        x=alt.X("Deviation (%):Q", title="Deviation (%)", 
                                scale=alt.Scale(domain=[
                                    min(-5, deviation_data["Deviation (%)"].min() - 1),
                                    max(5, deviation_data["Deviation (%)"].max() + 1)
                                ])),
                        color=alt.condition(
                            alt.datum["Deviation (%)"] > 0,
                            alt.value("#ef4444"),  # Red for overweight
                            alt.value("#22c55e")   # Green for underweight
                        ),
                        tooltip=[
                            alt.Tooltip("Asset:N", title="Asset"),
                            alt.Tooltip("Deviation (%):Q", title="Deviation", format="+.2f"),
                        ],
                    )
                    .properties(height=max(150, len(deviation_data) * 30))
                )
                zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
                    color="#888888", strokeDash=[4, 4]
                ).encode(x="x:Q")
                st.altair_chart((deviation_chart + zero_line), width="stretch")
                st.caption("🔴 Overweight | 🟢 Underweight")
            
            with st.expander("ℹ️ How to read this", expanded=False):
                st.markdown("""
**Table rows:**

**Current Weight (%)** — Your current allocation to each asset based on current portfolio value.

**Target Weight (%)** — Your desired long-term allocation to each asset.

**Cash Allocation (EUR)** — The amount of new cash to invest in each asset to move toward target weights.

**New Weight (%)** — Your projected allocation after investing the new cash as recommended.

**Deviation chart:** Shows how far each asset deviates from its target weight. Red bars (positive) indicate overweight positions; green bars (negative) indicate underweight positions that should be prioritized for new cash.

The algorithm prioritizes underweight assets while ensuring no selling is required.
                """)

            if results["diag_transposed"] is not None:
                st.markdown("### Portfolio diagnostics")
                st.dataframe(results["diag_transposed"], width="stretch")
                
                with st.expander("ℹ️ What do these diagnostics mean?", expanded=False):
                    st.markdown("""
These diagnostics help you understand recent portfolio behavior over the last ~12 months.

**CAGR (12m, %)** — Compound Annual Growth Rate over the last 12 months. Shows how much each asset has returned recently.

**EWMA Price Distance % (3m/6m/12m)** — How far the current price is from its exponentially-weighted moving average over 3, 6, or 12 months. Positive values indicate price is above recent average (momentum); negative values suggest price is below average (potential value opportunity).

**EWMA Volatility % (Annualized)** — Recent annualized volatility using exponential weighting, which gives more weight to recent price movements. Higher values indicate more recent price swings.

**Z-Score (12m, on prices)** — How many standard deviations the current price is from its 12-month mean. Values above +2 suggest potentially overbought; below -2 suggest potentially oversold; near 0 means trading around historical average.

**Correlation to Stocks (12m, monthly)** — How closely each asset moved with your stock allocation over the last 12 months using monthly returns. Values range from -1 (moves opposite) to +1 (moves together). Lower or negative correlation provides better diversification.
                    """)
            elif results["diag_error"]:
                st.caption(f"Diagnostics unavailable: {results['diag_error']}")

            snap = results["snap"]
            if snap is not None:
                st.markdown("### Macro-economic overview")
                st.caption(f"Data as of: {snap.asof.date()}")
                
                # Helper to format values, showing "—" for None
                def _fmt_pct(val: float | None) -> str:
                    return f"{val:.2f}%" if val is not None else "—"
                
                def _fmt_fx(val: float | None) -> str:
                    return f"{val:.4f}" if val is not None else "—"
                
                # Check if any values are missing
                missing_data = any([
                    snap.ecb_dfr_pct is None,
                    snap.de_10y_yield_pct is None,
                    snap.de_cpi_yoy_pct is None,
                    snap.usd_eur_spot is None,
                    snap.fed_rf_pct is None,
                    snap.us_10y_yield_pct is None,
                    snap.us_cpi_yoy_pct is None,
                ])
                if missing_data:
                    st.warning("⚠️ Some FRED data failed to load. Try re-running the analysis.")

                # Layout as requested: 2 rows, 4 metrics each
                r1 = st.columns(4)
                r1[0].metric("ECB Overnight", _fmt_pct(snap.ecb_dfr_pct))
                r1[1].metric("DE 10Y Yield", _fmt_pct(snap.de_10y_yield_pct))
                r1[2].metric("DE Inflation YoY", _fmt_pct(snap.de_cpi_yoy_pct))
                r1[3].metric("USD/EUR spot", _fmt_fx(snap.usd_eur_spot))

                r2 = st.columns(4)
                r2[0].metric("FED Overnight", _fmt_pct(snap.fed_rf_pct))
                r2[1].metric("US 10Y Yield", _fmt_pct(snap.us_10y_yield_pct))
                r2[2].metric("US Inflation YoY", _fmt_pct(snap.us_cpi_yoy_pct))
                r2[3].metric("Global EY (est.)", _fmt_pct(snap.global_earnings_yield_est_pct))

                proxy_note = getattr(snap, "global_earnings_yield_note", None)
                if proxy_note:
                    st.info(proxy_note)
                
                with st.expander("ℹ️ What do these indicators mean?", expanded=False):
                    st.markdown("""
These indicators provide macro context (rates, inflation, FX, valuations) for rebalancing decisions.

**ECB Overnight** — Euro area policy rate; anchors short-term EUR rates and influences bond yields.

**DE 10Y Yield** — Long-term EUR “risk-free” proxy (Bund yield). Higher yields raise the opportunity cost of holding equities.

**DE Inflation YoY** — Proxy for Euro-area inflation pressure. Higher inflation can keep policy rates elevated.

**USD/EUR spot** — FX rate (USD per 1 EUR). Relevant if you hold USD assets unhedged.

**FED Overnight** — US overnight policy rate proxy; influences USD cash yields and discount rates.

**US 10Y Yield** — Key long-term USD rate. Higher yields can pressure equity valuations.

**US Inflation YoY** — US CPI year-over-year.

**Global Earnings Yield (est.)** — A simple valuation proxy for global equities (higher can imply “cheaper” equities vs bonds).

If the global estimate is unavailable (Yahoo fundamentals can be flaky on Streamlit Cloud), the app may fall back to a **US earnings-yield proxy from FRED**. When that happens, you'll see an explicit note in the UI.
                    """)


                charts = results.get("macro_charts") or {}
                if isinstance(charts, dict):
                    x_axis_format = alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y"))

                    def _render_macro_chart(
                        title: str,
                        data: list[dict[str, object]],
                        *,
                        y_title: str,
                        indicator_order: list[str] | None = None,
                        explainer_md: str | None = None,
                    ) -> None:
                        dfc = pd.DataFrame(data)
                        if dfc.empty:
                            st.info(f"{title}: unavailable.")
                            return
                        color = alt.Color("Indicator:N", title=None)
                        if indicator_order:
                            color = alt.Color("Indicator:N", title=None, sort=indicator_order)
                        chart = (
                            alt.Chart(dfc)
                            .mark_line(strokeWidth=2.0)
                            .encode(
                                x=x_axis_format,
                                y=alt.Y("Value:Q", title=y_title),
                                color=color,
                                tooltip=[
                                    alt.Tooltip("Date:T", title="Date"),
                                    alt.Tooltip("Indicator:N", title="Indicator"),
                                    alt.Tooltip("Value:Q", title="Value", format=".3f"),
                                ],
                            )
                            .properties(height=260)
                            .interactive()
                        )
                        st.markdown(f"##### {title}")
                        st.altair_chart(chart, width="stretch")
                        if explainer_md:
                            with st.expander("ℹ️ What does this chart show?", expanded=False):
                                st.markdown(explainer_md)

                    st.markdown("#### Last 12 months trend")
                    _render_macro_chart(
                        "EU / DE indicators",
                        charts.get("eu_de", []),
                        y_title="Percent (%)",
                        indicator_order=["ECB Overnight (%)", "DE 10Y Yield (%)", "DE Inflation YoY (%)"],
                        explainer_md=(
                            "- **ECB Overnight (%)**: euro area policy rate proxy (deposit facility rate).\n"
                            "- **DE 10Y Yield (%)**: long-term EUR rate proxy (Bund yield).\n"
                            "- **DE Inflation YoY (%)**: Germany CPI year-over-year.\n\n"
                            "Use this to see whether EUR policy/long rates and inflation have been trending up or down."
                        ),
                    )
                    _render_macro_chart(
                        "US indicators",
                        charts.get("us", []),
                        y_title="Percent (%)",
                        indicator_order=["FED Overnight (%)", "US 10Y Yield (%)", "US Inflation YoY (%)"],
                        explainer_md=(
                            "- **FED Overnight (%)**: US policy rate proxy (EFFR).\n"
                            "- **US 10Y Yield (%)**: long-term USD rate.\n"
                            "- **US Inflation YoY (%)**: US CPI year-over-year.\n\n"
                            "Use this to gauge the direction of US rates and inflation, which can affect global risk assets."
                        ),
                    )
                    _render_macro_chart(
                        "USD/EUR",
                        charts.get("fx", []),
                        y_title="USD per 1 EUR",
                        indicator_order=["USD/EUR"],
                        explainer_md=(
                            "**USD/EUR** is the amount of USD per 1 EUR. If you hold USD-denominated assets (unhedged), FX moves can materially impact EUR returns."
                        ),
                    )
                    _render_macro_chart(
                        "Global earnings yield (est.)",
                        charts.get("earnings", []),
                        y_title="Percent (%)",
                        indicator_order=["Global EY Est. (%)"],
                        explainer_md=(
                            "**Global EY (est.)** is a simple valuation proxy for global equities. "
                            "Higher values generally imply *cheaper* equities vs their own history (all else equal). "
                            "This is a best-effort estimate derived from ACWI price history + trailing EPS/PE snapshot."
                        ),
                    )
            else:
                st.warning("Could not fetch macro data. Check your FRED API key and try re-running the analysis.")

            # AI-Assisted Rebalancing section (combined prompt + query)
            if results["llm_prompt"]:
                st.markdown("### AI-Assisted Rebalancing")
                st.caption("Use the prompt below with your preferred LLM, or query one directly.")

                st.download_button(
                    "Download prompt (.md)",
                    data=results["llm_prompt"].encode("utf-8"),
                    file_name=f"rebalance_prompt_{results['portfolio_name'].replace(' ', '_')}.md",
                    mime="text/markdown",
                    key="rebalance_download_prompt",
                )
                with st.expander("View prompt", expanded=False):
                    st.markdown(results["llm_prompt"])

                # Inline LLM query UI (no separate header)
                _render_llm_query_ui(
                    key_prefix="rebalance",
                    llm_prompt=results["llm_prompt"],
                    title="",  # No title, already under AI-Assisted Rebalancing
                )


elif page == "whatif":
    with st.expander("ℹ️ About this section", expanded=False):
        st.markdown("""
This section explores what would happen if you added a new asset to your portfolio by swapping a portion of an existing position.

**What you'll get:**
- **Diversification analysis**: Correlations, volatility impact, and how well candidates diversify your portfolio
- **RRR analysis**: Return-to-Risk Ratio test based on portfolio theory—does the new asset clear the "no-harm" hurdle?
- **Backtest comparison**: Side-by-side performance of your baseline vs. each modified portfolio
- **AI assistance**: Generate a prompt for an LLM to help interpret the results

**How to use:** Create your base portfolio, select candidate assets to evaluate, choose which position to fund from and by how much, then click "Run what-if".
        """)
    
    p, _ = portfolio_builder(key="whatif", title="Base portfolio", allow_value=False)
    if p is not None:
        # Load predefined candidate assets
        whatif_assets_path = os.path.join(CACHE_DIR, "assets", "list.json")
        available_candidates: list[dict[str, str]] = []
        try:
            with open(whatif_assets_path, "r", encoding="utf-8") as f:
                whatif_data = json.load(f)
                available_candidates = whatif_data.get("Assets", [])
        except Exception as e:
            st.warning(f"Could not load candidate assets from list.json: {e}")

        # Get portfolio tickers to filter out already-included assets
        portfolio_tickers_set = set(getattr(p, "tickers", []))

        # Filter candidates: only show assets not already in the portfolio
        filtered_candidates = [
            asset for asset in available_candidates
            if asset.get("Ticker", "") not in portfolio_tickers_set
        ]

        # Initialize variables
        candidate_short_to_ticker: dict[str, str] = {}  # Short name -> ticker
        candidate_short_to_display: dict[str, str] = {}  # Short name -> "Name (Ticker)"
        selected_candidate_shorts: list[str] = []

        if not filtered_candidates:
            st.info("All predefined candidate assets are already in the portfolio.")
        else:
            # Build options using Short names as values, with display mapping
            candidate_short_to_name: dict[str, str] = {}  # Short name -> full Name
            for asset in filtered_candidates:
                short = asset.get("Short", asset["Name"])
                candidate_short_to_ticker[short] = asset["Ticker"]
                candidate_short_to_display[short] = f"{asset['Name']} ({asset['Ticker']})"
                candidate_short_to_name[short] = asset["Name"]
            # Sort options by FULL Name (case-insensitive)
            sorted_candidate_shorts = sorted(candidate_short_to_ticker.keys(), key=lambda s: str(candidate_short_to_name.get(s, s)).lower())
            # Keep selection chips sorted by FULL name (not short).
            _presort_multiselect_state(
                key="whatif_candidates",
                sort_by={short: str(candidate_short_to_name.get(short, short)) for short in sorted_candidate_shorts},
            )
            if "whatif_candidates" in st.session_state:
                selected_candidate_shorts = st.multiselect(
                    "Candidate assets to evaluate",
                    options=sorted_candidate_shorts,
                    format_func=lambda x: candidate_short_to_display.get(x, x),  # Show "Name (Ticker)" in dropdown
                    key="whatif_candidates",
                    help="Select one or more assets to analyze for potential inclusion in your portfolio.",
                )
            else:
                selected_candidate_shorts = st.multiselect(
                    "Candidate assets to evaluate",
                    options=sorted_candidate_shorts,
                    default=[],
                    format_func=lambda x: candidate_short_to_display.get(x, x),  # Show "Name (Ticker)" in dropdown
                    key="whatif_candidates",
                    help="Select one or more assets to analyze for potential inclusion in your portfolio.",
                )

        # Get available assets with their weights for the source selection
        assets_map = getattr(p, "assets", {})
        target_weights_pct = getattr(p, "target_weights_pct", {})
        source_options = []
        for ticker in p.tickers:
            label = assets_map.get(ticker, ticker)
            weight_pct = target_weights_pct.get(ticker, 0.0)
            source_options.append((ticker, f"{label} ({weight_pct:.1f}%)"))

        col1, col2 = st.columns(2)
        with col1:
            source_asset_idx = st.selectbox(
                "Fund new asset from",
                options=range(len(source_options)),
                format_func=lambda i: source_options[i][1],
                index=0,
                key="whatif_source_asset",
            )
            source_ticker = source_options[source_asset_idx][0] if source_options else None
            max_swap_pct = target_weights_pct.get(source_ticker, 0.0) if source_ticker else 25.0

        with col2:
            swap_pct = st.number_input(
                f"Allocation to new asset (%)",
                min_value=1.0,
                max_value=float(max_swap_pct),
                value=min(5.0, float(max_swap_pct)),
                step=1.0,
                key="whatif_swap_pct",
                help=f"Maximum: {max_swap_pct:.1f}% (current weight of selected source asset)",
            )
            swap_weight = swap_pct / 100.0

        # Settings section
        st.markdown("### Settings")
        rebalance_frequency, _, rf_annual = _backtest_controls(key_prefix="whatif", show_initial_amount=False)

        # Get available date range from portfolio data
        try:
            prices_raw = p._prices_df().dropna(how="any").sort_index()
            available_start = prices_raw.index.min().date() if not prices_raw.empty else None
            available_end = prices_raw.index.max().date() if not prices_raw.empty else None
        except Exception:
            available_start, available_end = None, None

        if available_start and available_end:
            # st.caption(f"Available data range: **{available_start}** to **{available_end}**")
            date_c1, date_c2 = st.columns(2)
            with date_c1:
                user_start = st.date_input(
                    "Start date",
                    value=available_start,
                    min_value=available_start,
                    max_value=available_end,
                    key="whatif_start_date",
                )
            with date_c2:
                user_end = st.date_input(
                    "End date",
                    value=available_end,
                    min_value=available_start,
                    max_value=available_end,
                    key="whatif_end_date",
                )
        else:
            st.warning("Could not determine available date range from portfolio data.")
            user_start, user_end = None, None

        y_scale = st.radio(
            "Y-axis scale (for value chart)",
            options=["Linear", "Logarithmic"],
            index=0,
            horizontal=True,
            key="whatif_y_scale",
        )

        run = st.button("Run what-if", type="primary", key="whatif_run")

        if run:
            with st.status("Running what-if analysis...", expanded=True) as status:
                try:
                    total_start = time.time()
                    
                    # Get selected candidates (tickers) from multiselect
                    if not filtered_candidates or not selected_candidate_shorts:
                        raise ValueError("No candidate assets selected. Please select at least one asset.")

                    # Extract tickers from Short names
                    candidates = [candidate_short_to_ticker[short] for short in selected_candidate_shorts]
                    # Build mapping from ticker to Short name for display
                    candidate_name_map = {candidate_short_to_ticker[short]: short for short in selected_candidate_shorts}

                    if source_ticker is None:
                        raise ValueError("No source asset selected.")

                    # Variables to store for LLM prompt
                    llm_scores_df = None
                    llm_rrr_df = None
                    llm_backtest_df = None

                    base_target_weights = _target_weights_fraction(p)

                    step_ph = st.empty()
                    step_ph.write("Downloading price data...")
                    step_start = time.time()
                    
                    tickers_universe = list(getattr(p, "tickers", [])) + [c for c in candidates if c not in getattr(p, "tickers", [])]
                    prices_universe = _cached_download_prices(tuple(sorted(tickers_universe)))

                    portfolio_tickers = list(getattr(p, "tickers", []))
                    candidate_cols = [c for c in candidates if c in prices_universe.columns]
                    cols_all = portfolio_tickers + candidate_cols
                    raw = prices_universe[cols_all].copy().sort_index()
                    firsts = raw.apply(lambda s: s.first_valid_index())
                    lasts = raw.apply(lambda s: s.last_valid_index())
                    if firsts.isna().any() or lasts.isna().any():
                        missing = list(raw.columns[firsts.isna() | lasts.isna()])
                        raise ValueError(f"Missing data for tickers: {missing}")
                    start = pd.Timestamp(max(firsts))
                    end = pd.Timestamp(min(lasts))
                    if start > end:
                        raise ValueError(f"No overlapping date range across all assets (start={start}, end={end}).")

                    analysis_start_str = str(start.date())
                    analysis_end_str = str(end.date())
                    raw = raw.loc[start:end].copy()

                    prices_portfolio = raw[portfolio_tickers].copy()
                    prices_candidates = raw[candidate_cols].copy()

                    px_p_me = Portfolio.resample_prices(Portfolio.fill_non_trading_days(prices_portfolio, freq="D"), freq="ME")
                    px_c_me = Portfolio.resample_prices(Portfolio.fill_non_trading_days(prices_candidates, freq="D"), freq="ME")
                    common_start_ts, _ = Portfolio.common_start_info(px_p_me, px_c_me)

                    rets_candidates = Portfolio.monthly_returns_from_prices(prices_candidates, return_method="pct", common_start=common_start_ts)
                    rets_portfolio = Portfolio.monthly_returns_from_prices(prices_portfolio, return_method="pct", common_start=common_start_ts)

                    _timed_step(step_ph, "Downloading price data...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Computing diversification scores...")
                    step_start = time.time()
                    
                    scores = diversification_scores(
                        rets_candidates,
                        rets_portfolio,
                        portfolio_weights=base_target_weights,
                        replace_from=source_ticker,
                        replace_weight=float(swap_weight),
                    )

                    # Diversification scores table
                    scores_transposed = None
                    if not scores.empty:
                        scores_display = scores.copy()
                        scores_display.index = [candidate_name_map.get(t, t) for t in scores_display.index]
                        pct_cols = ["delta_vol_if_swap", "delta_max_drawdown_if_swap", "cand_vol_ann"]
                        
                        # Transform max_abs_corr_offender: "TICKER (+0.XXX)" -> "Short Name"
                        short_map = _get_short_name_map()
                        def _offender_to_short(val: str) -> str:
                            if not val or not isinstance(val, str):
                                return "—"
                            # Extract ticker (everything before " (")
                            if " (" in val:
                                ticker = val.split(" (")[0].strip()
                            else:
                                ticker = val.strip()
                            # Look up short name
                            return short_map.get(ticker, ticker)
                        
                        if "max_abs_corr_offender" in scores_display.columns:
                            scores_display["max_abs_corr_offender"] = scores_display["max_abs_corr_offender"].apply(_offender_to_short)
                        
                        for col in scores_display.columns:
                            if col == "max_abs_corr_offender":
                                continue  # Already processed above
                            elif col == "n_months_used":
                                scores_display[col] = scores_display[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "—")
                            elif col in pct_cols:
                                scores_display[col] = scores_display[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
                            else:
                                scores_display[col] = scores_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

                        scores_display = scores_display.rename(columns={
                            "mean_abs_corr_to_assets": "Avg |Corr| to Assets",
                            "max_abs_corr_to_assets": "Max |Corr| to Assets",
                            "max_abs_corr_offender": "Highest Corr Asset",
                            "w_mean_abs_corr_to_assets": "Weighted Avg |Corr|",
                            "corr_to_portfolio": "Corr to Portfolio",
                            "delta_vol_if_swap": "Δ Volatility (swap)",
                            "delta_max_drawdown_if_swap": "Δ |Max Drawdown| (swap)",
                            "cand_vol_ann": "Candidate Vol (ann.)",
                            "n_months_used": "Months of Data",
                        })
                        scores_transposed = scores_display.T
                        llm_scores_df = scores_transposed

                    _timed_step(step_ph, "Computing diversification scores...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Running RRR analysis...")
                    step_start = time.time()
                    
                    # RRR Analysis (Portfolio Intuition)
                    rrr_transposed = None
                    rrr_error = None
                    try:
                        port_rets = rets_portfolio.copy()
                        w_series = pd.Series(base_target_weights, dtype=float)
                        w_series = w_series[w_series.index.isin(port_rets.columns)]
                        w_series = w_series / w_series.sum() if w_series.sum() != 0 else w_series
                        port_r = (port_rets[w_series.index] * w_series.values).sum(axis=1).dropna()

                        if not port_r.empty:
                            mu_p, vol_p = Portfolio.annualize_mean_std(port_r, periods_per_year=12)
                            rrr_p = Portfolio.return_to_risk_ratio(mu_p, vol_p)

                            rrr_rows = []
                            for c in rets_candidates.columns:
                                s = pd.to_numeric(rets_candidates[c], errors="coerce").dropna()
                                if s.empty:
                                    continue
                                aligned = pd.concat([port_r.rename("port"), s.rename("cand")], axis=1).dropna()
                                if aligned.shape[0] < 6:
                                    continue
                                rho = float(aligned["port"].corr(aligned["cand"]))
                                mu_a, vol_a = Portfolio.annualize_mean_std(aligned["cand"], periods_per_year=12)

                                if pd.notna(mu_a) and float(mu_a) > 0:
                                    rrr_a = Portfolio.return_to_risk_ratio(mu_a, vol_a)
                                    hurdle = float(rho) * float(rrr_p) if pd.notna(rho) and pd.notna(rrr_p) else float("nan")
                                    margin = float(rrr_a - hurdle) if pd.notna(rrr_a) and pd.notna(hurdle) else float("nan")
                                    rrr_pa = Portfolio.rrr_combination(mu_p=mu_p, vol_p=vol_p, mu_a=mu_a, vol_a=vol_a, rho=rho, w_a=float(swap_weight))
                                    delta_rrr = float(rrr_pa - rrr_p) if pd.notna(rrr_pa) and pd.notna(rrr_p) else float("nan")
                                    rrr_status = "PASS" if margin > 0 else "FAIL"
                                else:
                                    rrr_a = float("nan")
                                    hurdle = float("nan")
                                    margin = float("nan")
                                    rrr_pa = float("nan")
                                    delta_rrr = float("nan")
                                    rrr_status = "N/A (μ ≤ 0)"

                                rrr_rows.append({
                                    "Candidate": candidate_name_map.get(c, c),
                                    "Portfolio RRR": f"{rrr_p:.2f}" if pd.notna(rrr_p) else "—",
                                    "Asset RRR": f"{rrr_a:.2f}" if pd.notna(rrr_a) else "—",
                                    "Correlation (ρ)": f"{rho:.2f}" if pd.notna(rho) else "—",
                                    "Hurdle (ρ × RRR_p)": f"{hurdle:.2f}" if pd.notna(hurdle) else "—",
                                    "Margin (RRR_a - Hurdle)": f"{margin:.2f}" if pd.notna(margin) else "—",
                                    f"Combined RRR ({swap_pct:.0f}% swap)": f"{rrr_pa:.2f}" if pd.notna(rrr_pa) else "—",
                                    "Δ RRR vs Portfolio": f"{delta_rrr:.2f}" if pd.notna(delta_rrr) else "—",
                                    "Status": rrr_status,
                                })

                            if rrr_rows:
                                rrr_df = pd.DataFrame(rrr_rows).set_index("Candidate")
                                rrr_transposed = rrr_df.T
                                llm_rrr_df = rrr_transposed
                    except Exception as e:
                        rrr_error = str(e)

                    _timed_step(step_ph, "Running RRR analysis...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Running backtests...")
                    step_start = time.time()
                    
                    # Backtest comparison table
                    backtest_df = None
                    cand_weights: dict[str, dict[str, float]] = {}
                    for c in candidates:
                        if c not in prices_universe.columns:
                            continue
                        new_weights = base_target_weights.copy()
                        new_weights[source_ticker] = new_weights.get(source_ticker, 0.0) - swap_weight
                        new_weights[c] = swap_weight
                        cand_weights[c] = new_weights

                    if cand_weights:
                        universe = Portfolio.fill_non_trading_days(raw[portfolio_tickers + list(cand_weights.keys())], freq="D")
                        rows: list[dict[str, Any]] = []
                        value_series: dict[str, pd.Series] = {}
                        
                        px_base = universe[portfolio_tickers]
                        v_base = Portfolio.backtest_value_series(
                            px_base, base_target_weights, rebalance_frequency=str(rebalance_frequency), initial_value=1.0
                        )
                        s_base = Portfolio.backtest_stats(v_base, rf_annual=float(rf_annual))
                        rows.append({"Portfolio": "Baseline", **s_base})
                        value_series["Baseline"] = v_base

                        for c, w in cand_weights.items():
                            cols = portfolio_tickers + [c]
                            px = universe[cols]
                            v = Portfolio.backtest_value_series(px, w, rebalance_frequency=str(rebalance_frequency), initial_value=1.0)
                            s = Portfolio.backtest_stats(v, rf_annual=float(rf_annual))
                            cand_name = candidate_name_map.get(c, c)
                            portfolio_label = f"+ {cand_name}"
                            rows.append({"Portfolio": portfolio_label, **s})
                            value_series[portfolio_label] = v

                        df = pd.DataFrame(rows).set_index("Portfolio")
                        backtest_df = pd.DataFrame(index=df.index)
                        # Format all columns as strings to avoid Arrow serialization issues with mixed types
                        backtest_df["Total Return"] = (df["total_return"].astype(float) * 100.0).round(2).astype(str) + "%"
                        backtest_df["CAGR"] = (df["cagr"].astype(float) * 100.0).round(2).astype(str) + "%"
                        backtest_df["Vol (ann.)"] = (df["vol_annual"].astype(float) * 100.0).round(2).astype(str) + "%"
                        backtest_df["Sharpe"] = df["sharpe"].astype(float).round(2).astype(str)
                        backtest_df["Sortino"] = df["sortino"].astype(float).round(2).astype(str)
                        backtest_df["Max Drawdown"] = (df["max_drawdown"].astype(float) * 100.0).round(2).astype(str) + "%"
                        if "longest_drawdown_days" in df.columns:
                            backtest_df["Longest Drawdown"] = df["longest_drawdown_days"].apply(
                                lambda x: _fmt_days(float(x)) if pd.notna(x) else "—"
                            )
                        backtest_df["Ulcer Index"] = df["ulcer_index"].astype(float).round(2).astype(str)
                        llm_backtest_df = backtest_df

                    # Generate LLM prompt
                    llm_prompt = None
                    try:
                        candidate_names_list = [candidate_name_map.get(c, c) for c in candidates]
                        llm_prompt = build_llm_whatif_report(
                            portfolio=p,
                            candidates=candidates,
                            candidate_names=candidate_names_list,
                            source_ticker=source_ticker,
                            swap_weight=swap_weight,
                            diversification_df=llm_scores_df,
                            rrr_df=llm_rrr_df,
                            backtest_df=llm_backtest_df,
                            analysis_start=analysis_start_str,
                            analysis_end=analysis_end_str,
                        )
                    except Exception:
                        pass

                    # Store all results in session state
                    st.session_state["whatif_results"] = {
                        "analysis_start": analysis_start_str,
                        "analysis_end": analysis_end_str,
                        "scores_transposed": scores_transposed,
                        "rrr_transposed": rrr_transposed,
                        "rrr_error": rrr_error,
                        "backtest_df": backtest_df,
                        "value_series": value_series if cand_weights else {},
                        "llm_prompt": llm_prompt,
                        "portfolio_name": getattr(p, "name", "Portfolio"),
                        "swap_pct": swap_pct,
                    }
                    # Clear any previous LLM response when recomputing
                    st.session_state.pop("whatif_llm_response", None)
                    
                    _timed_step(step_ph, "Running backtests...", step_start)
                    
                    total_elapsed = time.time() - total_start
                    status.update(label=f"What-if analysis complete! ({total_elapsed:.2f}s)", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="What-if analysis failed", state="error", expanded=False)
                    st.error(str(e))
                    st.session_state.pop("whatif_results", None)

        # Display results from session state (persists across reruns)
        if "whatif_results" in st.session_state:
            results = st.session_state["whatif_results"]
            
            st.caption(f"Common analysis window: {results['analysis_start']} → {results['analysis_end']}")

            # Diversification scores table
            if results["scores_transposed"] is not None:
                st.markdown("### Diversification analysis")
                st.dataframe(results["scores_transposed"], width="stretch")
                
                with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                    st.markdown("""
This table shows how each candidate asset relates to your existing portfolio.

**Avg |Corr| to Assets** — Mean absolute correlation to each portfolio asset. Lower is better for diversification: below 0.3 is excellent, 0.3-0.5 is moderate, above 0.5 provides limited benefit.

**Max |Corr| to Assets** — Highest absolute correlation to any single portfolio asset. Lower is better; values above 0.7 indicate redundancy with an existing holding.

**Highest Corr Asset** — The portfolio asset with which the candidate is most correlated, helping identify potential overlaps.

**Weighted Avg |Corr|** — Correlation weighted by portfolio asset weights. Lower is better. More relevant than unweighted average because it accounts for position sizes.

**Corr to Portfolio** — Correlation to overall portfolio returns. Lower is better for diversification: below 0.3 indicates a strong diversifier, above 0.6 means it moves with the portfolio.

**Δ Volatility (swap)** — Change in portfolio volatility if the swap is made. Negative is better as it indicates reduced portfolio risk.

**Δ |Max Drawdown| (swap)** — Change in worst-case drawdown magnitude. Negative is better as it indicates smaller potential losses.

**Candidate Vol (ann.)** — The candidate's own annualized volatility. Note that high-volatility assets can still be good diversifiers if they're uncorrelated with your portfolio.

**Months of Data** — Analysis period length. More months of data generally means more reliable results.
                    """)

            # RRR Analysis
            st.markdown("### Return-to-Risk Ratio (RRR) analysis")

            if results["rrr_transposed"] is not None:
                st.dataframe(results["rrr_transposed"], width="stretch")
                
                with st.expander("ℹ️ What is RRR analysis?", expanded=False):
                    st.markdown("""
**RRR (Return-to-Risk Ratio)** analysis is based on the "Portfolio Intuition" paper (Kennedy, 2018).

**RRR** is calculated as Annualized Return / Annualized Volatility, similar to the Sharpe ratio but without subtracting the risk-free rate. **RRR_p** refers to your current portfolio's RRR, while **RRR_a** is the candidate asset's RRR. **rho** represents the correlation between candidate and portfolio returns.

**The "bare minimum no-harm" condition** states that for adding a new asset to not hurt the portfolio (as weight approaches 0), **RRR_a > rho × RRR_p** must hold. If the margin (RRR_a - rho × RRR_p) is positive, the asset passes this hurdle.

**Hurdle (ρ × RRR_p)** — The minimum RRR the asset must exceed to be beneficial.

**Margin (RRR_a - Hurdle)** — Positive means PASS (asset clears the hurdle); negative means FAIL.

**Combined RRR** — The portfolio's RRR after allocating the specified swap weight to the candidate. If Combined RRR > Portfolio RRR, adding the asset improves risk-adjusted returns.

**Δ RRR vs Portfolio** — The change in RRR. Positive values indicate improvement.

**Status** — PASS indicates the asset improves risk-adjusted returns; FAIL suggests it may hurt the portfolio.
                    """)
            elif results["rrr_error"]:
                st.caption(f"RRR analysis unavailable: {results['rrr_error']}")
            else:
                st.info("Could not compute RRR metrics (insufficient data or portfolio returns unavailable).")

            # Backtest comparison table (transposed: metrics as rows)
            if results["backtest_df"] is not None:
                st.markdown("### Backtest comparison")
                # Backward-compatible: older cached runs may not have Longest Drawdown column.
                bt_df = results["backtest_df"]
                if isinstance(bt_df, pd.DataFrame):
                    try:
                        if "Longest Drawdown" not in bt_df.columns:
                            vs = results.get("value_series") or {}
                            dd_map = {k: _fmt_days(float(Portfolio.longest_drawdown_days(v))) for k, v in vs.items()}
                            bt_df = bt_df.copy()
                            bt_df["Longest Drawdown"] = [dd_map.get(idx, "—") for idx in bt_df.index]
                            results["backtest_df"] = bt_df
                    except Exception:
                        pass
                st.dataframe(results["backtest_df"].T, width="stretch")
                
                with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                    st.markdown("""
This table compares historical performance between your baseline portfolio and portfolios with each candidate asset added.

**Total Return** — Cumulative gain/loss over the entire analysis period.

**CAGR** — Compound Annual Growth Rate, the average annual return assuming reinvestment. Higher is better for comparing across different time periods.

**Vol (ann.)** — Annualized standard deviation of returns. Lower generally indicates more stable returns, though this is a trade-off with potential returns.

**Sharpe** — Risk-adjusted return calculated as (Return - Risk-free) / Volatility. Higher is better: above 0.5 is decent, above 1.0 is good.

**Sortino** — Similar to Sharpe but only penalizes downside volatility. Higher is better, especially relevant for loss-averse investors.

**Max Drawdown** — The largest peak-to-trough decline during the period. Less negative values indicate smaller worst-case losses.

**Longest Drawdown** — The longest stretch of time the portfolio stayed below its previous peak (“underwater”), measured in days.

**Ulcer Index** — Measures downside volatility using the quadratic mean of percentage drawdowns. Lower is better: below 5 is excellent, 5-10 is good, above 10 indicates stress.

*Note: Past performance does not guarantee future results. Use backtests as one input among many considerations.*
                    """)

                # Portfolio value chart (like comparison section)
                value_series = results.get("value_series", {})
                if value_series:
                    st.markdown("#### Portfolio value over time")
                    # Build chart data with Baseline first for legend ordering
                    chart_data = []
                    # Process Baseline first, then other portfolios
                    portfolio_order = ["Baseline"] + [n for n in value_series.keys() if n != "Baseline"]
                    for name in portfolio_order:
                        if name in value_series:
                            for date, value in value_series[name].items():
                                chart_data.append({"Date": date, "Portfolio": name, "Value": float(value)})
                    chart_df = pd.DataFrame(chart_data)

                    y_scale_type = "log" if y_scale == "Logarithmic" else "linear"
                    chart = (
                        alt.Chart(chart_df)
                        .mark_line(strokeWidth=2.0)
                        .encode(
                            x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y")),
                            y=alt.Y("Value:Q", title="Value (normalized)", scale=alt.Scale(type=y_scale_type)),
                            color=alt.Color("Portfolio:N", title="Portfolio", sort=portfolio_order),
                            tooltip=[
                                alt.Tooltip("Date:T", title="Date"),
                                alt.Tooltip("Portfolio:N", title="Portfolio"),
                                alt.Tooltip("Value:Q", title="Value", format=",.4f"),
                            ],
                        )
                        .properties(height=400)
                        .interactive()
                    )
                    st.altair_chart(chart, width="stretch")

                    # Drawdown comparison chart (requested)
                    _render_drawdown_chart_multi(value_series, title="Drawdown (comparison)", portfolio_order=portfolio_order)

            # AI-Assisted Analysis section (combined prompt + query)
            if results["llm_prompt"]:
                st.markdown("### AI-Assisted Asset Selection")
                st.caption("Use the prompt below with your preferred LLM, or query one directly.")

                st.download_button(
                    "Download prompt (.md)",
                    data=results["llm_prompt"].encode("utf-8"),
                    file_name=f"whatif_prompt_{results['portfolio_name'].replace(' ', '_')}.md",
                    mime="text/markdown",
                    key="whatif_download_prompt",
                )
                with st.expander("View prompt", expanded=False):
                    st.markdown(results["llm_prompt"])

                # Inline LLM query UI (no separate header)
                _render_llm_query_ui(
                    key_prefix="whatif",
                    llm_prompt=results["llm_prompt"],
                    title="",  # No title, already under AI-Assisted Asset Selection
                )


