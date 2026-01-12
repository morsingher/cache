import logging
import os
import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from fredapi import Fred  # type: ignore
except Exception:
    Fred = None

logger = logging.getLogger(__name__)

_GLOBAL_ECY_CACHE: dict[str, tuple[float, float]] = {}
_GLOBAL_ECY_TTL_S: float = 6 * 3600.0  # 6 hours

_US_EY_PROXY_SERIES_IDS: list[str] = [
    # These series IDs may vary by FRED provider; we try in order.
    "SP500PE",  # S&P 500 PE Ratio (if available)
    "NASDAQ100", # Nasdaq 100 PE (rare)
    "PERATIO",
    "CAPE",
    "CAPE10",
]


def get_us_earnings_yield_proxy_series_fred(
    fred: "Fred",
    *,
    debug: bool,
    observation_start: Any | None = None,
) -> tuple[pd.Series | None, str | None]:
    """
    Best-effort US earnings yield proxy derived from a FRED PE-ratio series:
      earnings_yield_pct = 100 / PE

    Returns:
      (series, note) where note describes the proxy source, or (None, None).
    """
    # 1. Try standard PE series
    for sid in _US_EY_PROXY_SERIES_IDS:
        s_pe = _try_get_fred_series(fred, sid, debug=debug, observation_start=observation_start)
        if s_pe is None or s_pe.empty:
            continue
        pe = pd.to_numeric(s_pe, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        pe = pe[pe > 0]
        if pe.empty:
            continue
        ey = (100.0 / pe).replace([np.inf, -np.inf], np.nan).dropna()
        if ey.empty:
            continue
        ey.name = "global_earnings_yield_est_pct"
        note = f"Proxy used: US earnings yield via FRED series '{sid}' (100/PE)."
        return ey, note

    # 2. Fallback: US 10Y Yield as a rough proxy + fixed ERP estimate?
    # Using 10Y Yield directly is a lower bound, but better than nothing.
    # Adding a small ERP (e.g. 2.0%) brings it closer to typical earnings yields.
    # We use "DGS10" (Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity).
    try:
        s_10y = _try_get_fred_series(fred, "DGS10", debug=debug, observation_start=observation_start)
        if s_10y is not None and not s_10y.empty:
            # Assume 2.5% Equity Risk Premium (ERP)
            erp_estimate = 2.5
            ey = (s_10y + erp_estimate).dropna()
            if not ey.empty:
                ey.name = "global_earnings_yield_est_pct"
                note = f"Proxy used: US 10Y Yield (DGS10) + {erp_estimate}% estimated ERP (fallback)."
                return ey, note
    except Exception:
        pass

    return None, None

def _get_cached_global_earnings_yield_pct() -> float | None:
    ent = _GLOBAL_ECY_CACHE.get("acwi_ecy")
    if not ent:
        return None
    ts, val = ent
    if (time.time() - float(ts)) > float(_GLOBAL_ECY_TTL_S):
        return None
    return float(val)

def _set_cached_global_earnings_yield_pct(val: float) -> None:
    _GLOBAL_ECY_CACHE["acwi_ecy"] = (time.time(), float(val))

def _try_get_acwi_earnings_yield_est(debug: bool, max_retries: int = 3, base_delay: float = 1.0) -> float | None:
    """
    Best-effort global earnings yield estimate via yfinance (ACWI trailing PE).
    Fallback to URTH (MSCI World) or SPY (S&P 500) if ACWI info fails.
    """
    cached = _get_cached_global_earnings_yield_pct()
    if cached is not None:
        return cached

    # Candidate tickers to try for PE ratio (in order of preference)
    tickers = ["ACWI", "URTH", "SPY"]
    
    last_exc: Exception | None = None
    
    for ticker_symbol in tickers:
        for attempt in range(max_retries):
            try:
                # Use fast_info first if available (yfinance >= 0.2)
                # fast_info often has less rate-limiting than .info
                # However, fast_info doesn't natively expose PE.
                # So we stick to .info for now, but handle retries carefully.
                ticker = yf.Ticker(ticker_symbol)
                pe = ticker.info.get("trailingPE")
                
                if pe and float(pe) > 0:
                    ecy = (1.0 / float(pe)) * 100.0
                    _set_cached_global_earnings_yield_pct(ecy)
                    return float(ecy)
            except Exception as e:
                last_exc = e
            
            # Backoff before retry
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                delay *= (1.0 + random.uniform(-0.15, 0.15))
                time.sleep(delay)
        
        # If we failed all retries for this ticker, try the next ticker

    if last_exc is not None:
        logger.warning(f"[macro] Failed to fetch earnings-yield estimate (tried {tickers}): {last_exc}")
        if debug:
            print(f"[macro] Failed to fetch earnings-yield estimate (tried {tickers}): {last_exc}")
    return None


@dataclass(frozen=True)
class MacroSnapshot:
    asof: pd.Timestamp
    ecb_dfr_pct: float | None
    de_10y_yield_pct: float | None
    de_cpi_yoy_pct: float | None
    fed_rf_pct: float | None
    us_10y_yield_pct: float | None
    us_cpi_yoy_pct: float | None
    global_earnings_yield_est_pct: float | None
    global_earnings_yield_note: str | None
    usd_eur_spot: float | None
    usd_eur_3m_ago: float | None
    usd_eur_6m_ago: float | None
    usd_eur_12m_ago: float | None


_GLOBAL_ECY_SERIES_CACHE: dict[str, tuple[float, pd.Series]] = {}
_GLOBAL_ECY_SERIES_TTL_S: float = 6 * 3600.0  # 6 hours

def get_global_earnings_yield_series(*, debug: bool = False, lookback_days: int = 500) -> pd.Series | None:
    """
    Best-effort *time series* for "global earnings yield (est.)".

    We approximate earnings yield as (trailing EPS / price) and treat trailing EPS as constant
    over the lookback window (reasonable for a simple 12m trend chart).
    
    Tries ACWI -> URTH -> SPY.
    """
    cache_key = f"acwi_ecy_series_{int(lookback_days)}"
    cached = _GLOBAL_ECY_SERIES_CACHE.get(cache_key)
    if cached and (time.time() - float(cached[0])) <= float(_GLOBAL_ECY_SERIES_TTL_S):
        return cached[1].copy()

    # Tickers to try
    candidate_tickers = ["ACWI", "URTH", "SPY"]

    for ticker_symbol in candidate_tickers:
        # 1. Fetch price history
        try:
            px = yf.download(ticker_symbol, period="max", auto_adjust=True, progress=False)
            if px is None or px.empty:
                continue
            if isinstance(px.columns, pd.MultiIndex):
                px = px["Close"]
                if isinstance(px, pd.DataFrame):
                    px = px.iloc[:, 0]
            elif "Close" in px.columns:
                px = px["Close"]
            else:
                px = px.iloc[:, 0] if len(px.columns) else None
            
            if px is None:
                continue
            
            px = pd.to_numeric(px, errors="coerce").dropna().sort_index()
            if px.empty:
                continue
                
            cutoff = px.index.max() - pd.Timedelta(days=int(lookback_days))
            px = px.loc[px.index >= cutoff]
            if px.empty:
                continue
        except Exception as e:
            if debug:
                print(f"[macro] Failed to download {ticker_symbol} prices: {e}")
            continue

        # 2. Get trailing EPS (fallback to trailing PE)
        trailing_eps = None
        trailing_pe = None
        
        # Retry logic for info fetch
        for attempt in range(3):
            try:
                info = yf.Ticker(ticker_symbol).info
                trailing_eps = info.get("trailingEps")
                trailing_pe = info.get("trailingPE")
                if (trailing_eps and float(trailing_eps) > 0) or (trailing_pe and float(trailing_pe) > 0):
                    break
            except Exception:
                if attempt < 2:
                    time.sleep(1.0 + random.random())

        eps = None
        try:
            if trailing_eps is not None and float(trailing_eps) > 0:
                eps = float(trailing_eps)
            elif trailing_pe is not None and float(trailing_pe) > 0:
                # EPS ≈ Price / PE (use latest price as anchor)
                eps = float(px.iloc[-1]) / float(trailing_pe)
        except Exception:
            eps = None

        if eps is None or eps <= 0:
            if debug:
                print(f"[macro] Could not determine EPS for {ticker_symbol}")
            continue

        # If we got here, we have prices and EPS. Calculate series.
        ecy = (float(eps) / px) * 100.0
        ecy = ecy.replace([np.inf, -np.inf], np.nan).dropna()
        if ecy.empty:
            continue
        
        ecy.name = "global_earnings_yield_est_pct"
        _GLOBAL_ECY_SERIES_CACHE[cache_key] = (time.time(), ecy.copy())
        return ecy

    return None


def _try_get_fred_series(
    fred: "Fred",
    series_id: str,
    debug: bool,
    observation_start: Any | None = None,
    max_retries: int = 6,
    base_delay: float = 0.5,
) -> pd.Series | None:
    """
    Fetch a FRED series with retry logic and exponential backoff.
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            if observation_start is not None:
                s = fred.get_series(series_id, observation_start=observation_start)
            else:
                s = fred.get_series(series_id)
            if isinstance(s, pd.Series):
                return s.dropna()
            return None
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            # Check if this is a retryable error
            retryable = any([
                "Connection" in error_msg,
                "Timeout" in error_msg,
                "HTTPError" in error_msg,
                "404" in error_msg,
                "500" in error_msg,
                "502" in error_msg,
                "503" in error_msg,
                "URL" in error_msg,
                "urlopen" in error_msg.lower(),
            ])
            
            if attempt < max_retries - 1 and retryable:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"FRED fetch failed for {series_id} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                break
    
    if debug and last_exception:
        print(f"[macro] Failed to fetch FRED series {series_id}: {last_exception}")
    return None


def get_macro_snapshot(fred_api_key: str | None = None, debug: bool = False) -> MacroSnapshot | None:
    """
    Minimal macro snapshot for rebalancing context.

    Uses FRED if available and API key is provided (arg or env var FRED_API_KEY).
    Series used (best-effort):
      - ECB deposit facility rate (ECBDFR) (%)
      - Germany 10Y yield (IRLTLT01DEM156N) (%)
      - Germany CPI (DEUCPIALLMINMEI) (index, monthly) -> YoY inflation (%)
      - USD/EUR spot (DEXUSEU) (USD per 1 EUR)
      - Fed risk-free proxy (EFFR) (%)
      - US 10Y yield (DGS10) (%)
      - US CPI (CPIAUCSL) (index, monthly) -> YoY inflation (%)
      - Global earnings yield estimate (best-effort, derived from ACWI)
    """
    if Fred is None:
        if debug:
            print("[macro] fredapi not available; skipping macro overlay.")
        return None

    api_key = (fred_api_key or os.environ.get("FRED_API_KEY", "")).strip()
    if not api_key:
        if debug:
            print("[macro] Missing FRED API key; skipping macro overlay.")
        return None

    fred = Fred(api_key=api_key)

    # Pull ~2y to compute YoY inflation reliably and to support 12m trend charts elsewhere.
    obs_start = pd.Timestamp.now() - pd.DateOffset(months=26)
    s_ecb_dfr = _try_get_fred_series(fred, "ECBDFR", debug=debug, observation_start=obs_start)  # %
    s_de_10y = _try_get_fred_series(fred, "IRLTLT01DEM156N", debug=debug, observation_start=obs_start)  # %
    s_de_cpi = _try_get_fred_series(fred, "DEUCPIALLMINMEI", debug=debug, observation_start=obs_start)  # index
    s_usd_eur = _try_get_fred_series(fred, "DEXUSEU", debug=debug, observation_start=obs_start)  # USD per 1 EUR
    s_fed_rf = _try_get_fred_series(fred, "EFFR", debug=debug, observation_start=obs_start)  # %
    s_us_10y = _try_get_fred_series(fred, "DGS10", debug=debug, observation_start=obs_start)  # %
    s_us_cpi = _try_get_fred_series(fred, "CPIAUCSL", debug=debug, observation_start=obs_start)  # index

    cols: dict[str, pd.Series] = {}
    if s_ecb_dfr is not None:
        cols["ecb_dfr_pct"] = pd.to_numeric(s_ecb_dfr, errors="coerce")
    if s_de_10y is not None:
        cols["de_10y_yield_pct"] = pd.to_numeric(s_de_10y, errors="coerce")
    if s_de_cpi is not None:
        cols["de_cpi_idx"] = pd.to_numeric(s_de_cpi, errors="coerce")
    if s_usd_eur is not None:
        cols["usd_eur"] = pd.to_numeric(s_usd_eur, errors="coerce")
    if s_fed_rf is not None:
        cols["fed_rf_pct"] = pd.to_numeric(s_fed_rf, errors="coerce")
    if s_us_10y is not None:
        cols["us_10y_yield_pct"] = pd.to_numeric(s_us_10y, errors="coerce")
    if s_us_cpi is not None:
        cols["us_cpi_idx"] = pd.to_numeric(s_us_cpi, errors="coerce")

    if not cols:
        return None

    df = pd.DataFrame(cols).sort_index().ffill().dropna(how="all")
    if df.empty:
        return None

    asof = pd.Timestamp(df.index.max())

    ecb_dfr = float(df["ecb_dfr_pct"].iloc[-1]) if "ecb_dfr_pct" in df.columns else None
    de_10y = float(df["de_10y_yield_pct"].iloc[-1]) if "de_10y_yield_pct" in df.columns else None
    fed_rf = float(df["fed_rf_pct"].iloc[-1]) if "fed_rf_pct" in df.columns else None
    us_10y = float(df["us_10y_yield_pct"].iloc[-1]) if "us_10y_yield_pct" in df.columns else None

    de_cpi_yoy = None
    # IMPORTANT: compute CPI YoY on the original CPI series, not on the merged/ffilled dataframe.
    # Otherwise pct_change(12) can accidentally mean "12 days" if the index became daily.
    if s_de_cpi is not None:
        cpi = pd.to_numeric(s_de_cpi, errors="coerce").dropna()
        if cpi.shape[0] >= 13:
            infl_yoy = cpi.pct_change(12) * 100.0  # monthly YoY (%)
            infl_yoy = infl_yoy.dropna()
            if not infl_yoy.empty:
                de_cpi_yoy = float(infl_yoy.iloc[-1])

    us_cpi_yoy = None
    if s_us_cpi is not None:
        cpi = pd.to_numeric(s_us_cpi, errors="coerce").dropna()
        if cpi.shape[0] >= 13:
            infl_yoy = (cpi.pct_change(12) * 100.0).dropna()
            if not infl_yoy.empty:
                us_cpi_yoy = float(infl_yoy.iloc[-1])

    usd_eur_spot = float(df["usd_eur"].iloc[-1]) if "usd_eur" in df.columns else None

    usd_eur_3m = usd_eur_6m = usd_eur_12m = None
    if "usd_eur" in df.columns:
        usd_eur_daily = df["usd_eur"].asfreq("D").ffill().dropna()
        if not usd_eur_daily.empty:
            end = usd_eur_daily.index.max()

            def _asof(months: int) -> float | None:
                target = end - pd.DateOffset(months=months)
                sub = usd_eur_daily.loc[:target]
                if sub.empty:
                    return None
                return float(sub.iloc[-1])

            usd_eur_3m = _asof(3)
            usd_eur_6m = _asof(6)
            usd_eur_12m = _asof(12)

    # Global earnings yield estimate (best-effort; non-critical).
    global_note: str | None = None
    global_ecy = _try_get_acwi_earnings_yield_est(debug=debug)
    # If we can build a series, prefer its latest point for consistency with the trend chart.
    try:
        s_ecy = get_global_earnings_yield_series(debug=debug, lookback_days=500)
        if s_ecy is not None and not s_ecy.empty:
            global_ecy = float(s_ecy.iloc[-1])
    except Exception as e:
        logger.warning(f"[macro] get_global_earnings_yield_series failed in snapshot: {e}")

    # If Yahoo-based global estimate is unavailable, fall back to a US proxy via FRED (and note it).
    if global_ecy is None:
        s_us_ey, note = get_us_earnings_yield_proxy_series_fred(
            fred, debug=debug, observation_start=obs_start
        )
        if s_us_ey is not None and not s_us_ey.empty:
            global_ecy = float(s_us_ey.iloc[-1])
            global_note = note

    return MacroSnapshot(
        asof=asof,
        ecb_dfr_pct=ecb_dfr,
        de_10y_yield_pct=de_10y,
        de_cpi_yoy_pct=de_cpi_yoy,
        fed_rf_pct=fed_rf,
        us_10y_yield_pct=us_10y,
        us_cpi_yoy_pct=us_cpi_yoy,
        global_earnings_yield_est_pct=global_ecy,
        global_earnings_yield_note=global_note,
        usd_eur_spot=usd_eur_spot,
        usd_eur_3m_ago=usd_eur_3m,
        usd_eur_6m_ago=usd_eur_6m,
        usd_eur_12m_ago=usd_eur_12m,
    )


def print_macro_overview(fred_api_key: str | None = None, debug: bool = False) -> None:
    snap = get_macro_snapshot(fred_api_key=fred_api_key, debug=debug)
    if snap is None:
        print("MACRO OVERVIEW: unavailable (missing FRED_API_KEY or fetch failed)")
        return

    print(f"MACRO OVERVIEW (as of: {snap.asof.date().isoformat()})\n")
    if snap.ecb_dfr_pct is not None:
        print(f"ECB Deposit Facility Rate: {snap.ecb_dfr_pct:.2f}%")
    if snap.de_10y_yield_pct is not None:
        print(f"DE 10Y Yield: {snap.de_10y_yield_pct:.2f}%")
    if snap.de_cpi_yoy_pct is not None:
        print(f"DE CPI YoY: {snap.de_cpi_yoy_pct:.2f}%")
    if snap.usd_eur_spot is not None:
        print(f"USD/EUR: {snap.usd_eur_spot:.4f}")

    if snap.fed_rf_pct is not None:
        print(f"Fed risk-free rate (EFFR): {snap.fed_rf_pct:.2f}%")
    if snap.us_10y_yield_pct is not None:
        print(f"US 10Y Yield: {snap.us_10y_yield_pct:.2f}%")
    if snap.us_cpi_yoy_pct is not None:
        print(f"US CPI YoY: {snap.us_cpi_yoy_pct:.2f}%")
    if snap.global_earnings_yield_est_pct is not None:
        print(f"Global Earnings Yield (est.): {snap.global_earnings_yield_est_pct:.2f}%")
        if snap.global_earnings_yield_note:
            print(f"  Note: {snap.global_earnings_yield_note}")

    if snap.usd_eur_spot is not None:
        def _fmt_change(past: float | None) -> str:
            if past is None:
                return ""
            change = (snap.usd_eur_spot / past - 1.0) * 100.0
            return f" ({change:+.2f}%)"
        if snap.usd_eur_3m_ago is not None:
            print(f"USD/EUR (3m ago): {snap.usd_eur_3m_ago:.4f}{_fmt_change(snap.usd_eur_3m_ago)}")
        if snap.usd_eur_6m_ago is not None:
            print(f"USD/EUR (6m ago): {snap.usd_eur_6m_ago:.4f}{_fmt_change(snap.usd_eur_6m_ago)}")
        if snap.usd_eur_12m_ago is not None:
            print(f"USD/EUR (12m ago): {snap.usd_eur_12m_ago:.4f}{_fmt_change(snap.usd_eur_12m_ago)}")


def ewma_volatility_lambda(log_returns_series: pd.Series, lam: float = 0.94) -> float:
    """
    RiskMetrics-style EWMA volatility (non-annualized, same periodicity as returns).
    Uses: var_t = lam*var_{t-1} + (1-lam)*r_t^2
    """
    r = log_returns_series.dropna().to_numpy(dtype=float)
    if r.size < 2:
        return float("nan")
    var = float(np.nanvar(r, ddof=1))
    for x in r:
        var = lam * var + (1.0 - lam) * (x ** 2)
    return float(np.sqrt(var))


def compute_rebalancing_diagnostics(
    portfolio: Any,
    lookback_days: int = 365,
    trading_days_per_year: int = 252,
    ewma_lambda: float = 0.94,
    ewma_spans_trading_days: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Build a rebalancing diagnostics table using ~last year of data.

    Includes:
      - Correlation to Stocks over 3/6/12 months (daily log returns)
      - EWMA price distance % over 3m/6m/12m (trading-day spans)
      - EWMA volatility % (annualized)
      - Z-score (12m) on prices

    Expects portfolio to be cache.portfolio.Portfolio and to have:
      - tickers, assets (ticker->asset_name), display_labels or _label()
      - prices (downloaded yfinance Close, DataFrame/Series)
    """
    if ewma_spans_trading_days is None:
        ewma_spans_trading_days = {"3m": 63, "6m": 126, "12m": 252}

    # Get prices DataFrame with ticker columns (no forced common start; we use per-series dropna)
    prices = portfolio._prices_df() if hasattr(portfolio, "_prices_df") else None
    if prices is None:
        raise ValueError("Portfolio object does not expose prices in a usable form.")
    prices = prices.copy().sort_index()
    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("No prices available to compute rebalancing diagnostics.")

    end = prices.index.max()
    start = end - pd.Timedelta(days=int(lookback_days))
    prices_lookback = prices.loc[prices.index >= start].copy()
    if prices_lookback.shape[0] < max(60, int(trading_days_per_year // 4)):
        raise ValueError("Not enough lookback history to compute rebalancing diagnostics.")

    # Daily log returns over the lookback window
    log_returns = np.log(prices_lookback).diff()

    # Identify stocks benchmark from asset names
    # Default stocks ticker to use when portfolio has no stocks
    DEFAULT_STOCKS_TICKER = "ACWE.MI"
    
    assets_map = getattr(portfolio, "assets", {})
    tickers = list(prices_lookback.columns)
    stock_tickers = [t for t in tickers if str(assets_map.get(t, "")).lower() == "stocks"]
    stocks_lr = None
    external_stocks_prices = None
    
    if stock_tickers:
        stocks_lr = log_returns[stock_tickers].mean(axis=1)
    else:
        # Download external stocks data for correlation computation
        try:
            # Prefer the Portfolio yfinance wrapper (retry logic) if available.
            try:
                from portfolio import Portfolio  # local import to avoid circulars
                px = Portfolio.download_prices([DEFAULT_STOCKS_TICKER], period="max", auto_adjust=True, ignore_tz=True, progress=False)
                if not px.empty and DEFAULT_STOCKS_TICKER in px.columns:
                    external_stocks_prices = px[DEFAULT_STOCKS_TICKER].dropna()
                else:
                    external_stocks_prices = None
            except Exception:
                external_stocks_prices = None
        except Exception:
            external_stocks_prices = None

    # Current price (last available for each ticker within lookback)
    current_price = pd.Series(
        {t: float(prices_lookback[t].dropna().iloc[-1]) if prices_lookback[t].dropna().size else np.nan for t in tickers}
    )

    # EWMA prices from the last ~12m window (more rows is fine; we cap for stability)
    prices_td = prices_lookback.tail(max(400, int(ewma_spans_trading_days.get("12m", 252)) + 50))
    ewma_price = {k: prices_td.ewm(span=int(span), adjust=False).mean().iloc[-1] for k, span in ewma_spans_trading_days.items()}
    ewma_price_df = pd.DataFrame(ewma_price)

    # EWMA vol on last trading_days_per_year returns
    lr_12m = log_returns.tail(int(trading_days_per_year))
    ewma_vol = pd.Series({t: ewma_volatility_lambda(lr_12m[t], lam=float(ewma_lambda)) for t in tickers}, dtype=float)

    # Z-score on last 12m prices (using available observations)
    zscore = pd.Series(index=tickers, dtype=float)
    for t in tickers:
        s = prices_lookback[t].dropna()
        if s.size < 2:
            zscore[t] = np.nan
            continue
        mu = float(s.mean())
        sd = float(s.std(ddof=1))
        zscore[t] = (float(s.iloc[-1]) - mu) / sd if sd > 0 else np.nan

    # 12m correlation vs stocks on MONTHLY log returns (more stable than daily).
    corr_12m_monthly = pd.Series(index=tickers, dtype=float)
    stocks_bench_m = None
    use_external_stocks = False
    
    if stock_tickers:
        # Use portfolio's stocks for benchmark
        monthly_prices = prices_lookback.resample("ME").last().dropna(how="any")
        if monthly_prices.shape[0] >= 13:
            r_m = np.log(monthly_prices / monthly_prices.shift(1)).dropna(how="any")
            stocks_bench_m = r_m[stock_tickers].mean(axis=1)
    elif external_stocks_prices is not None:
        # Use external stocks for benchmark
        use_external_stocks = True
        try:
            # Get external stocks prices aligned with portfolio date range
            ext_prices = external_stocks_prices.reindex(prices_lookback.index).ffill().bfill()
            monthly_ext = ext_prices.resample("ME").last().dropna()
            monthly_prices = prices_lookback.resample("ME").last().dropna(how="any")
            if monthly_ext.shape[0] >= 13 and monthly_prices.shape[0] >= 13:
                r_m = np.log(monthly_prices / monthly_prices.shift(1)).dropna(how="any")
                r_ext = np.log(monthly_ext / monthly_ext.shift(1)).dropna()
                stocks_bench_m = r_ext.reindex(r_m.index)
        except Exception:
            stocks_bench_m = None
    
    if stocks_bench_m is not None:
        # last 12 monthly observations
        r_m = r_m.tail(12)
        stocks_bench_m = stocks_bench_m.reindex(r_m.index)
        for t in tickers:
            if not use_external_stocks and str(assets_map.get(t, "")).lower() == "stocks":
                corr_12m_monthly[t] = 1.0
                continue
            if t not in r_m.columns:
                continue
            pair = pd.DataFrame({"Stocks": stocks_bench_m, "X": r_m[t]}).dropna()
            corr_12m_monthly[t] = float(pair["Stocks"].corr(pair["X"])) if pair.shape[0] >= 2 else np.nan

    # 12m CAGR per asset (using actual available dates within the last ~365 days)
    cagr_12m = pd.Series(index=tickers, dtype=float)
    cagr_start = end - pd.Timedelta(days=365)
    for t in tickers:
        s = prices.loc[prices.index >= cagr_start, t].dropna()
        if s.size < 2:
            cagr_12m[t] = np.nan
            continue
        years = (s.index[-1] - s.index[0]).days / 365.25
        cagr_12m[t] = float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan

    # Build table (index by display label)
    label = (lambda t: portfolio._label(t)) if hasattr(portfolio, "_label") else (lambda t: t)
    idx = [label(t) for t in tickers]

    # Helper to format percentage values with % symbol and 2 decimal places
    def _fmt_pct(val: float) -> str:
        return f"{val:.2f}%" if pd.notna(val) else "—"
    
    def _fmt_num(val: float) -> str:
        return f"{val:.2f}" if pd.notna(val) else "—"
    
    table = pd.DataFrame(index=idx)
    table["CAGR 12m"] = [_fmt_pct(float(cagr_12m[t]) * 100.0) if pd.notna(cagr_12m[t]) else "—" for t in tickers]
    table["EWMA Price Dist 3m"] = [_fmt_pct(((current_price[t] / float(ewma_price_df.loc[t, "3m"])) - 1.0) * 100.0) if pd.notna(current_price[t]) and pd.notna(ewma_price_df.loc[t, "3m"]) else "—" for t in tickers]
    table["EWMA Price Dist 6m"] = [_fmt_pct(((current_price[t] / float(ewma_price_df.loc[t, "6m"])) - 1.0) * 100.0) if pd.notna(current_price[t]) and pd.notna(ewma_price_df.loc[t, "6m"]) else "—" for t in tickers]
    table["EWMA Price Dist 12m"] = [_fmt_pct(((current_price[t] / float(ewma_price_df.loc[t, "12m"])) - 1.0) * 100.0) if pd.notna(current_price[t]) and pd.notna(ewma_price_df.loc[t, "12m"]) else "—" for t in tickers]
    table["EWMA Vol (ann)"] = [_fmt_pct(float(ewma_vol[t]) * np.sqrt(float(trading_days_per_year)) * 100.0) if pd.notna(ewma_vol[t]) else "—" for t in tickers]
    table["Z-Score 12m"] = [_fmt_num(float(zscore[t])) for t in tickers]
    # Add correlation to stocks if computed (from portfolio stocks or external stocks)
    if stocks_bench_m is not None:
        table["Corr vs Stocks 12m"] = [_fmt_num(float(corr_12m_monthly[t])) for t in tickers]

    table.index.name = None
    return table


def _safe_filename(s: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    out = "".join(c if c in keep else "_" for c in str(s).strip())
    return out or "portfolio"


def build_llm_rebalance_report(
    portfolio: Any,
    rebalance_table: pd.DataFrame,
    diagnostics_table: pd.DataFrame | None,
    macro_snapshot: MacroSnapshot | None,
    current_value: float,
    new_cash: float,
    macro_trends: dict[str, dict[str, float | None]] | None = None,
) -> str:
    """
    Generate an LLM prompt in the same markdown structure used in dashboard.py,
    including (a) system prompt, (b) macro snapshot, (c) diagnostics, (d) baseline allocation,
    and (e) final question.
    """
    portfolio_name = getattr(portfolio, "name", "Portfolio")

    system_prompt = (
        "You are a careful financial analyst and portfolio assistant. "
        "Your job is to help the user decide how to allocate new cash into their portfolio. "
        "Use the provided macro snapshot and portfolio diagnostics, but do not hallucinate data. "
        "Ask clarifying questions if needed, and explicitly separate facts from assumptions. "
        "Consider practical constraints (simplicity, diversification, risk tolerance, time horizon, "
        "rebalancing costs/taxes, and liquidity). Provide a concrete recommended allocation of the "
        "new cash across the portfolio assets, with brief reasoning and optional alternatives.\n\n"
        "The user is based in Italy and the portfolio is in EUR, intended as a permanent, long-term investment."
    )

    macro_lines: list[str] = []
    if macro_snapshot is None:
        macro_lines.append("Macro snapshot: unavailable (missing `FRED_API_KEY` or data fetch failed).")
    else:
        macro_lines.append(f"As-of: {macro_snapshot.asof.date().isoformat()}")
        macro_lines.append("")
        macro_lines.append("EU / DE (levels):")
        if macro_snapshot.ecb_dfr_pct is not None:
            macro_lines.append(f"  ECB deposit rate: {macro_snapshot.ecb_dfr_pct:.2f}%")
        if macro_snapshot.de_10y_yield_pct is not None:
            macro_lines.append(f"  DE 10Y yield: {macro_snapshot.de_10y_yield_pct:.2f}%")
        if macro_snapshot.de_cpi_yoy_pct is not None:
            macro_lines.append(f"  DE inflation YoY: {macro_snapshot.de_cpi_yoy_pct:.2f}%")
        if macro_snapshot.usd_eur_spot is not None:
            macro_lines.append(f"  USD/EUR spot: {macro_snapshot.usd_eur_spot:.4f}")

        macro_lines.append("")
        macro_lines.append("US (levels):")
        if macro_snapshot.fed_rf_pct is not None:
            macro_lines.append(f"  Fed risk-free rate (EFFR): {macro_snapshot.fed_rf_pct:.2f}%")
        if macro_snapshot.us_10y_yield_pct is not None:
            macro_lines.append(f"  US 10Y yield: {macro_snapshot.us_10y_yield_pct:.2f}%")
        if macro_snapshot.us_cpi_yoy_pct is not None:
            macro_lines.append(f"  US inflation YoY: {macro_snapshot.us_cpi_yoy_pct:.2f}%")
        if macro_snapshot.global_earnings_yield_est_pct is not None:
            macro_lines.append(f"  Global earnings yield (est.): {macro_snapshot.global_earnings_yield_est_pct:.2f}%")
            if getattr(macro_snapshot, "global_earnings_yield_note", None):
                macro_lines.append(f"    Note: {macro_snapshot.global_earnings_yield_note}")

        # Add historical trends if available
        if macro_trends:
            macro_lines.append("")  # blank line
            macro_lines.append("Historical trends (3m / 6m / 12m ago):")

            def _fmt_trend(key: str, *, label: str, fmt: str) -> None:
                d = macro_trends.get(key)
                if not isinstance(d, dict):
                    return
                vals: list[str] = []
                for p in ["3m", "6m", "12m"]:
                    v = d.get(p)
                    if v is None:
                        vals.append("N/A")
                    else:
                        vals.append(fmt.format(v))
                macro_lines.append(f"  {label}: {vals[0]} / {vals[1]} / {vals[2]}")

            _fmt_trend("ecb_deposit_rate_pct", label="ECB deposit rate", fmt="{:.2f}%")
            _fmt_trend("de_10y_yield_pct", label="DE 10Y yield", fmt="{:.2f}%")
            _fmt_trend("de_inflation_yoy_pct", label="DE inflation YoY", fmt="{:.2f}%")
            _fmt_trend("usd_eur", label="USD/EUR", fmt="{:.4f}")
            _fmt_trend("fed_risk_free_pct", label="Fed risk-free (EFFR)", fmt="{:.2f}%")
            _fmt_trend("us_10y_yield_pct", label="US 10Y yield", fmt="{:.2f}%")
            _fmt_trend("us_inflation_yoy_pct", label="US inflation YoY", fmt="{:.2f}%")
            _fmt_trend("global_earnings_yield_est_pct", label="Global earnings yield (est.)", fmt="{:.2f}%")

    diagnostics_explain = (
        "Diagnostics table notes:\n"
        "- CAGR 12m: annualized growth rate over ~the last 12 months of prices.\n"
        "- EWMA Price Dist 3m/6m/12m: (current price / EWMA price) - 1, using spans 63/126/252 trading days.\n"
        "- EWMA Vol (ann): RiskMetrics EWMA volatility on daily log returns (lambda=0.94), annualized with sqrt(252).\n"
        "- Z-Score 12m: (current price - mean) / std over the lookback price window.\n"
        "- Corr vs Stocks 12m: correlation of monthly log returns vs the 'Stocks' benchmark.\n"
    )

    baseline_section = []
    baseline_section.append("## Baseline (mathematically optimal) cash allocation\n\n")
    baseline_section.append(
        "The following allocation is the **baseline** suggestion produced by the app. "
        "It is the *mathematically optimal* allocation of the new cash **under a pure rebalancing objective**: "
        "use only non-negative buys (no selling) and distribute exactly the new cash amount such that the "
        "resulting portfolio weights move as close as possible to the target weights (in a least-squares sense). "
        "This is implemented via a projection onto the simplex (sum of buys equals new cash, buys are non-negative).\n\n"
        "**However**, this baseline ignores forward-looking considerations (expected returns, valuations, regime/macro signals, "
        "risk constraints, and idiosyncratic opportunities). Use it as a reference point, and propose deviations only when "
        "you can justify the trade-off.\n\n"
    )
    baseline_section.append("```\n" + rebalance_table.round(4).T.to_string() + "\n```\n\n")

    question = (
        "Considering the macro snapshot and the portfolio diagnostics below, "
        "how would you allocate the new cash into the user's portfolio?\n\n"
        "Provide a recommended EUR allocation per asset (summing exactly to the new cash amount), "
        "and briefly justify the decision. If you would deviate from pure target-rebalancing, "
        "explain why and what risks/assumptions drive the deviation."
    )

    report: list[str] = []
    report.append("## System prompt\n\n")
    report.append(system_prompt + "\n\n")
    report.append("## Macro-economic snapshot\n\n")
    report.append("```\n" + "\n".join(macro_lines) + "\n```\n\n")
    report.append("## Portfolio diagnostics (assets as columns)\n\n")
    report.append(diagnostics_explain + "\n")
    if diagnostics_table is None:
        report.append("Diagnostics: unavailable.\n\n")
    else:
        report.append("```\n" + diagnostics_table.round(4).T.to_string() + "\n```\n\n")
    report.append("## Cash inputs\n\n")
    report.append(
        "```\n"
        f"Portfolio name: {portfolio_name}\n"
        f"Current portfolio value (EUR): {float(current_value):,.2f}\n"
        f"New cash to allocate (EUR): {float(new_cash):,.2f}\n"
        "```\n\n"
    )
    report.append("".join(baseline_section))
    report.append("## Question\n\n")
    report.append(question + "\n")
    return "".join(report)


def write_llm_report(text: str, portfolio_name: str) -> str:
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    reports_dir = os.path.abspath(reports_dir)
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"rebalance_prompt_{_safe_filename(portfolio_name)}_{ts}.md"
    out = os.path.join(reports_dir, fname)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


def run_rebalancing(portfolio: Any, new_cash: float, fred_api_key: str | None = None, debug: bool = False) -> None:
    """
    Helper for cache/run.py. Expects `portfolio` to be an instance of cache.portfolio.Portfolio
    with `current_value_eur` and `rebalance(new_cash)` implemented.
    """
    cv = getattr(portfolio, "current_value_eur", None)
    if cv is None:
        raise ValueError("Portfolio is missing current_value_eur; ensure JSON includes top-level 'Value'.")

    print(f"REBALANCING ({float(cv):,.2f} EUR + {float(new_cash):,.2f} EUR)\n")

    table = portfolio.rebalance(float(new_cash))
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(4).T.to_string())
    print()

    print_macro_overview(fred_api_key=fred_api_key, debug=debug)
    print()

    # Diagnostics table (last ~12 months)
    diag = None
    try:
        diag = compute_rebalancing_diagnostics(portfolio)
        # Merge weights from rebalance table (same index orientation: assets as columns there)
        print("PORTFOLIO DIAGNOSTICS (last ~12 months)\n")
        with pd.option_context("display.width", 220, "display.max_columns", None):
            print(diag.round(4).T.to_string())
        print()
    except Exception as e:
        if debug:
            raise
        print(f"PORTFOLIO DIAGNOSTICS: unavailable ({e})\n")

    # LLM prompt/report
    try:
        macro = get_macro_snapshot(fred_api_key=fred_api_key, debug=debug)
        cv = float(cv)
        prompt = build_llm_rebalance_report(
            portfolio=portfolio,
            rebalance_table=table,
            diagnostics_table=diag,
            macro_snapshot=macro,
            current_value=cv,
            new_cash=float(new_cash),
        )
        out_path = write_llm_report(prompt, portfolio_name=getattr(portfolio, "name", "Portfolio"))
        print(f"LLM prompt written to: {out_path}\n")
    except Exception as e:
        if debug:
            raise
        print(f"LLM prompt: unavailable ({e})\n")


