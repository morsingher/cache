import os
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


@dataclass(frozen=True)
class MacroSnapshot:
    asof: pd.Timestamp
    ecb_dfr_pct: float | None
    usd_10y_tips_yield_pct: float | None
    de_10y_yield_pct: float | None
    de_cpi_yoy_pct: float | None
    de_10y_real_yield_proxy_pct: float | None
    global_earnings_yield_est_pct: float | None
    eurusd_spot: float | None
    eurusd_3m_ago: float | None
    eurusd_6m_ago: float | None
    eurusd_12m_ago: float | None


def _try_get_fred_series(fred: "Fred", series_id: str, debug: bool) -> pd.Series | None:
    try:
        s = fred.get_series(series_id)
        if isinstance(s, pd.Series):
            return s.dropna()
        return None
    except Exception as e:
        if debug:
            print(f"[macro] Failed to fetch FRED series {series_id}: {e}")
        return None


def get_macro_snapshot(fred_api_key: str | None = None, debug: bool = False) -> MacroSnapshot | None:
    """
    Minimal macro snapshot for rebalancing context.

    Uses FRED if available and API key is provided (arg or env var FRED_API_KEY).
    Series used (best-effort):
      - ECB deposit facility rate (ECBDFR) (%)
      - US 10Y TIPS real yield (DFII10) (%)
      - Germany 10Y yield (IRLTLT01DEM156N) (%)
      - Germany CPI (DEUCPIALLMINMEI) (index, monthly) -> YoY inflation (%)
      - EUR/USD spot (DEXUSEU)
      - Global earnings yield estimate from ACWI trailing P/E (best-effort)
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

    s_ecb_dfr = _try_get_fred_series(fred, "ECBDFR", debug=debug)  # %
    s_us_tips_10y = _try_get_fred_series(fred, "DFII10", debug=debug)  # %
    s_de_10y = _try_get_fred_series(fred, "IRLTLT01DEM156N", debug=debug)  # %
    s_de_cpi = _try_get_fred_series(fred, "DEUCPIALLMINMEI", debug=debug)  # index
    s_eurusd = _try_get_fred_series(fred, "DEXUSEU", debug=debug)  # EUR/USD

    cols: dict[str, pd.Series] = {}
    if s_ecb_dfr is not None:
        cols["ecb_dfr_pct"] = pd.to_numeric(s_ecb_dfr, errors="coerce")
    if s_us_tips_10y is not None:
        cols["usd_10y_tips_yield_pct"] = pd.to_numeric(s_us_tips_10y, errors="coerce")
    if s_de_10y is not None:
        cols["de_10y_yield_pct"] = pd.to_numeric(s_de_10y, errors="coerce")
    if s_de_cpi is not None:
        cols["de_cpi_idx"] = pd.to_numeric(s_de_cpi, errors="coerce")
    if s_eurusd is not None:
        cols["eurusd"] = pd.to_numeric(s_eurusd, errors="coerce")

    if not cols:
        return None

    df = pd.DataFrame(cols).sort_index().ffill().dropna(how="all")
    if df.empty:
        return None

    asof = pd.Timestamp(df.index.max())

    ecb_dfr = float(df["ecb_dfr_pct"].iloc[-1]) if "ecb_dfr_pct" in df.columns else None
    us_tips_10y = float(df["usd_10y_tips_yield_pct"].iloc[-1]) if "usd_10y_tips_yield_pct" in df.columns else None
    de_10y = float(df["de_10y_yield_pct"].iloc[-1]) if "de_10y_yield_pct" in df.columns else None

    de_cpi_yoy = None
    de_real_proxy = None
    # IMPORTANT: compute CPI YoY on the original CPI series, not on the merged/ffilled dataframe.
    # Otherwise pct_change(12) can accidentally mean "12 days" if the index became daily.
    if s_de_cpi is not None and s_de_10y is not None:
        cpi = pd.to_numeric(s_de_cpi, errors="coerce").dropna()
        nom = pd.to_numeric(s_de_10y, errors="coerce").dropna()
        if cpi.shape[0] >= 13 and nom.shape[0] >= 2:
            infl_yoy = cpi.pct_change(12) * 100.0  # monthly YoY (%)
            infl_yoy = infl_yoy.dropna()
            # Align on time and forward-fill monthly inflation to match nominal yield dates.
            aligned = pd.DataFrame({"nom": nom, "infl_yoy": infl_yoy}).sort_index().ffill().dropna()
            if not aligned.empty:
                de_cpi_yoy = float(aligned["infl_yoy"].iloc[-1])
                de_real_proxy = float(aligned["nom"].iloc[-1] - aligned["infl_yoy"].iloc[-1])

    eurusd_spot = float(df["eurusd"].iloc[-1]) if "eurusd" in df.columns else None

    eurusd_3m = eurusd_6m = eurusd_12m = None
    if "eurusd" in df.columns:
        eurusd_daily = df["eurusd"].asfreq("D").ffill().dropna()
        if not eurusd_daily.empty:
            end = eurusd_daily.index.max()

            def _asof(months: int) -> float | None:
                target = end - pd.DateOffset(months=months)
                sub = eurusd_daily.loc[:target]
                if sub.empty:
                    return None
                return float(sub.iloc[-1])

            eurusd_3m = _asof(3)
            eurusd_6m = _asof(6)
            eurusd_12m = _asof(12)

    # Global earnings yield estimate from ACWI trailing P/E (best-effort).
    global_ecy = None
    try:
        pe = yf.Ticker("ACWI").info.get("trailingPE")
        if pe and float(pe) > 0:
            global_ecy = (1.0 / float(pe)) * 100.0
    except Exception as e:
        if debug:
            print(f"[macro] Failed to fetch ACWI trailingPE for earnings yield estimate: {e}")

    return MacroSnapshot(
        asof=asof,
        ecb_dfr_pct=ecb_dfr,
        usd_10y_tips_yield_pct=us_tips_10y,
        de_10y_yield_pct=de_10y,
        de_cpi_yoy_pct=de_cpi_yoy,
        de_10y_real_yield_proxy_pct=de_real_proxy,
        global_earnings_yield_est_pct=global_ecy,
        eurusd_spot=eurusd_spot,
        eurusd_3m_ago=eurusd_3m,
        eurusd_6m_ago=eurusd_6m,
        eurusd_12m_ago=eurusd_12m,
    )


def print_macro_overview(fred_api_key: str | None = None, debug: bool = False) -> None:
    snap = get_macro_snapshot(fred_api_key=fred_api_key, debug=debug)
    if snap is None:
        print("MACRO OVERVIEW: unavailable (missing FRED_API_KEY or fetch failed)")
        return

    print(f"MACRO OVERVIEW (as of: {snap.asof.date().isoformat()})\n")
    if snap.global_earnings_yield_est_pct is not None:
        print(f"Global Earnings Yield (est., ACWI 1/PE): {snap.global_earnings_yield_est_pct:.2f}%")
    if snap.usd_10y_tips_yield_pct is not None:
        print(f"US 10Y TIPS Yield: {snap.usd_10y_tips_yield_pct:.2f}%")
    if snap.ecb_dfr_pct is not None:
        print(f"ECB Deposit Facility Rate: {snap.ecb_dfr_pct:.2f}%")
    if snap.de_10y_yield_pct is not None:
        print(f"DE 10Y Yield: {snap.de_10y_yield_pct:.2f}%")
    if snap.de_cpi_yoy_pct is not None:
        print(f"DE CPI YoY: {snap.de_cpi_yoy_pct:.2f}%")
    if snap.de_10y_real_yield_proxy_pct is not None:
        print(f"DE 10Y Real Yield Proxy (nominal - CPI YoY): {snap.de_10y_real_yield_proxy_pct:.2f}%")
    if snap.eurusd_spot is not None:
        print(f"EUR/USD: {snap.eurusd_spot:.4f}")
        def _fmt_change(past: float | None) -> str:
            if past is None:
                return ""
            change = (snap.eurusd_spot / past - 1.0) * 100.0
            return f" ({change:+.2f}%)"
        if snap.eurusd_3m_ago is not None:
            print(f"EUR/USD (3m ago): {snap.eurusd_3m_ago:.4f}{_fmt_change(snap.eurusd_3m_ago)}")
        if snap.eurusd_6m_ago is not None:
            print(f"EUR/USD (6m ago): {snap.eurusd_6m_ago:.4f}{_fmt_change(snap.eurusd_6m_ago)}")
        if snap.eurusd_12m_ago is not None:
            print(f"EUR/USD (12m ago): {snap.eurusd_12m_ago:.4f}{_fmt_change(snap.eurusd_12m_ago)}")


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
            external_stocks_prices = yf.download(
                DEFAULT_STOCKS_TICKER,
                period="max",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(external_stocks_prices.columns, pd.MultiIndex):
                external_stocks_prices = external_stocks_prices["Close"]
                if isinstance(external_stocks_prices, pd.DataFrame):
                    external_stocks_prices = external_stocks_prices.iloc[:, 0]
            elif "Close" in external_stocks_prices.columns:
                external_stocks_prices = external_stocks_prices["Close"]
            else:
                external_stocks_prices = external_stocks_prices.iloc[:, 0] if len(external_stocks_prices.columns) > 0 else None
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
        if macro_snapshot.global_earnings_yield_est_pct is not None:
            macro_lines.append(f"Global earnings yield (est., ACWI 1/PE): {macro_snapshot.global_earnings_yield_est_pct:.2f}%")
        if macro_snapshot.usd_10y_tips_yield_pct is not None:
            macro_lines.append(f"US 10Y TIPS yield: {macro_snapshot.usd_10y_tips_yield_pct:.2f}%")
        if macro_snapshot.ecb_dfr_pct is not None:
            macro_lines.append(f"ECB deposit facility rate: {macro_snapshot.ecb_dfr_pct:.2f}%")
        if macro_snapshot.de_10y_yield_pct is not None:
            macro_lines.append(f"DE 10Y yield: {macro_snapshot.de_10y_yield_pct:.2f}%")
        if macro_snapshot.de_cpi_yoy_pct is not None:
            macro_lines.append(f"DE CPI YoY: {macro_snapshot.de_cpi_yoy_pct:.2f}%")
        if macro_snapshot.de_10y_real_yield_proxy_pct is not None:
            macro_lines.append(f"DE 10Y real yield proxy (nominal - CPI YoY): {macro_snapshot.de_10y_real_yield_proxy_pct:.2f}%")
        if macro_snapshot.eurusd_spot is not None:
            macro_lines.append(f"EUR/USD spot: {macro_snapshot.eurusd_spot:.4f}")

        # Add historical trends if available
        if macro_trends:
            macro_lines.append("")  # blank line
            macro_lines.append("Historical trends (3m / 6m / 12m ago):")
            if "eurusd" in macro_trends:
                eurusd = macro_trends["eurusd"]
                vals = []
                for period in ["3m", "6m", "12m"]:
                    v = eurusd.get(period)
                    vals.append(f"{v:.4f}" if v is not None else "N/A")
                macro_lines.append(f"  EUR/USD: {vals[0]} / {vals[1]} / {vals[2]}")
            if "ecb_rate" in macro_trends:
                ecb = macro_trends["ecb_rate"]
                vals = []
                for period in ["3m", "6m", "12m"]:
                    v = ecb.get(period)
                    vals.append(f"{v:.2f}%" if v is not None else "N/A")
                macro_lines.append(f"  ECB rate: {vals[0]} / {vals[1]} / {vals[2]}")

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


