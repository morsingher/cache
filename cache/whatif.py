from typing import Any
from datetime import datetime
import os

import numpy as _np
import pandas as pd

from portfolio import Portfolio


def _parse_tickers(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for v in values:
        parts = [p.strip() for p in str(v).split(",")]
        out.extend([p for p in parts if p])
    # de-dupe, stable
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
    s = float(sum(float(x) for x in w.values()))
    if s <= 0:
        raise ValueError("Invalid weights: sum must be > 0.")
    return {k: float(v) / s for k, v in w.items()}


def _target_weights_fraction(portfolio: Any) -> dict[str, float]:
    if not hasattr(portfolio, "target_weights_pct"):
        raise ValueError("Portfolio is missing target weights; ensure JSON includes per-asset 'Target'.")
    tickers = list(getattr(portfolio, "tickers", []))
    wt_pct = {t: float(portfolio.target_weights_pct[t]) for t in tickers}
    # Use Portfolio normalization to accept percent-like input robustly.
    w_frac = Portfolio._normalize_weights_to_fraction([w / 100.0 for w in wt_pct.values()])
    return {t: float(w) for t, w in zip(tickers, w_frac)}


def _stocks_tickers(portfolio: Any) -> list[str]:
    tickers = list(getattr(portfolio, "tickers", []))
    assets = getattr(portfolio, "assets", {})
    return [t for t in tickers if str(assets.get(t, "")).strip().lower() == "stocks"]


def _apply_swap_from_stocks(
    *,
    base_target_weights: dict[str, float],
    stock_tickers: list[str],
    add_ticker: str,
    swap_weight: float,
) -> dict[str, float]:
    if add_ticker in base_target_weights:
        raise ValueError(f"'{add_ticker}' is already in the portfolio; refusing to treat it as a new asset.")
    if not stock_tickers:
        raise ValueError("No tickers classified as 'Stocks' in the portfolio JSON (asset Name).")
    if swap_weight <= 0 or swap_weight >= 1:
        raise ValueError("--swap-weight must be a fraction in (0, 1).")

    stocks_total = float(sum(base_target_weights.get(t, 0.0) for t in stock_tickers))
    if stocks_total <= 0:
        raise ValueError("Total target weight of Stocks is 0; cannot fund a swap from stocks.")
    if swap_weight > stocks_total + 1e-12:
        raise ValueError(
            f"Cannot fund {swap_weight*100:.2f}% from Stocks: stocks target weight is only {stocks_total*100:.2f}%."
        )

    new_w = dict(base_target_weights)
    # Reduce stocks proportionally to preserve internal stocks composition.
    for t in stock_tickers:
        share = float(base_target_weights[t]) / stocks_total
        new_w[t] = float(new_w[t]) - float(swap_weight) * share
    new_w[add_ticker] = float(swap_weight)
    # Numerical safety: clip tiny negatives (floating point) and re-normalize.
    eps = 1e-12
    for k, v in list(new_w.items()):
        if float(v) < -eps:
            raise ValueError("Weights must be non-negative and sum to > 0.")
        if float(v) < 0:
            new_w[k] = 0.0
    total = float(sum(new_w.values()))
    if total <= 0:
        raise ValueError("Weights must be non-negative and sum to > 0.")
    return {k: float(v) / total for k, v in new_w.items()}


def _weighted_portfolio_returns(
    rets: pd.DataFrame,
    weights: dict[str, float],
    *,
    require_all_assets: bool = True,
) -> pd.Series:
    """
    Weighted portfolio return series (assumes SIMPLE returns, i.e. pct returns).
    """
    if rets.empty or not weights:
        return pd.Series(dtype=float, name="W_Portfolio")
    w = pd.Series(weights, dtype=float)
    w = w[w != 0.0]
    cols = [c for c in w.index if c in rets.columns]
    if not cols:
        return pd.Series(dtype=float, name="W_Portfolio")
    sub = rets[cols].copy()
    if require_all_assets:
        sub = sub.dropna(how="any")
    ww = w.loc[cols]
    ww = ww / ww.sum()
    return (sub * ww.values).sum(axis=1).rename("W_Portfolio")


def _print_rrr_intuition_metric(
    *,
    prices_portfolio: pd.DataFrame,
    prices_candidates: pd.DataFrame,
    portfolio_target_weights: dict[str, float],
    swap_weight: float,
) -> None:
    """
    Additional evaluation step based on "Portfolio Intuition" (Kennedy, 2018).

    Computes (on monthly simple returns):
      - Portfolio RRR_p and asset RRR_a (annualized mean / annualized vol)
      - Correlation rho between candidate and portfolio monthly returns
      - "Bare-minimum no-harm" condition as w_a -> 0:
            RRR_a > rho * RRR_p
        We report the margin: RRR_a - rho*RRR_p (positive => passes the inequality).
      - Exact-weight combined RRR_pa using the paper's formula at w_a = swap_weight:
            RRR_pa = (w_p*r_p + w_a*r_a)/sqrt(w_p^2*σ_p^2 + w_a^2*σ_a^2 + 2*w_p*w_a*σ_p*σ_a*rho)
        and compare it to RRR_p.
    """
    # Monthly simple returns (full available window; we intentionally do NOT apply a lookback here)
    px_p_me = Portfolio.resample_prices(Portfolio.fill_non_trading_days(prices_portfolio, freq="D"), freq="ME")
    px_c_me = Portfolio.resample_prices(Portfolio.fill_non_trading_days(prices_candidates, freq="D"), freq="ME")
    common_start, _ = Portfolio.common_start_info(px_p_me, px_c_me)

    rets_assets = Portfolio.monthly_returns_from_prices(prices_portfolio, return_method="pct", common_start=common_start)
    rets_cands = Portfolio.monthly_returns_from_prices(prices_candidates, return_method="pct", common_start=common_start)

    port_r = _weighted_portfolio_returns(rets_assets, portfolio_target_weights, require_all_assets=True).dropna()
    if port_r.empty:
        print("RRR / CORRELATION HURDLE: unavailable (portfolio monthly return series empty)\n")
        return

    mu_p, vol_p = Portfolio.annualize_mean_std(port_r, periods_per_year=12)
    if pd.notna(mu_p) and float(mu_p) <= 0:
        print("RRR / CORRELATION HURDLE (from Portfolio Intuition, Kennedy 2018)")
        print(
            "N/A: the portfolio’s estimated annualized mean return over the selected lookback is <= 0.\n"
            "In this regime the inequality can behave unintuitively (as noted by the paper).\n"
            "Please ignore this metric for decision-making and rely on the backtest table instead.\n"
        )
        return
    rrr_p = Portfolio.return_to_risk_ratio(mu_p, vol_p)

    rows: list[dict[str, float | str]] = []
    for c in rets_cands.columns:
        s = pd.to_numeric(rets_cands[c], errors="coerce").dropna()
        if s.empty:
            continue
        aligned = pd.concat([port_r, s.rename("cand")], axis=1).dropna()
        if aligned.shape[0] < 6:
            continue
        rho = float(aligned["W_Portfolio"].corr(aligned["cand"]))
        mu_a, vol_a = Portfolio.annualize_mean_std(aligned["cand"], periods_per_year=12)
        if pd.notna(mu_a) and float(mu_a) <= 0:
            rows.append(
                {
                    "Candidate": str(c),
                    "RRR_p": float(rrr_p),
                    "RRR_a": float("nan"),
                    "rho(p,a)": float(rho),
                    "RRR_a - rho*RRR_p": float("nan"),
                    "RRR_pa(w)": float("nan"),
                    "RRR_pa(w) - RRR_p": float("nan"),
                    "n_months": float(aligned.shape[0]),
                    "Status": "N/A (mu_a<=0)",
                }
            )
            continue

        rrr_a = Portfolio.return_to_risk_ratio(mu_a, vol_a)

        hurdle = float(rho) * float(rrr_p) if pd.notna(rho) and pd.notna(rrr_p) else float("nan")
        margin = float(rrr_a - hurdle) if pd.notna(rrr_a) and pd.notna(hurdle) else float("nan")

        rrr_pa = Portfolio.rrr_combination(mu_p=mu_p, vol_p=vol_p, mu_a=mu_a, vol_a=vol_a, rho=rho, w_a=float(swap_weight))
        delta_rrr_pa = float(rrr_pa - rrr_p) if pd.notna(rrr_pa) and pd.notna(rrr_p) else float("nan")

        rows.append(
            {
                "Candidate": str(c),
                "RRR_p": float(rrr_p),
                "RRR_a": float(rrr_a),
                "rho(p,a)": float(rho),
                "RRR_a - rho*RRR_p": float(margin),
                "RRR_pa(w)": float(rrr_pa),
                "RRR_pa(w) - RRR_p": float(delta_rrr_pa),
                "n_months": float(aligned.shape[0]),
                "Status": "OK",
            }
        )

    if not rows:
        print("RRR / CORRELATION HURDLE: unavailable (not enough overlapping monthly data)\n")
        return

    df = pd.DataFrame(rows).set_index("Candidate")
    df = df.sort_values("RRR_pa(w) - RRR_p", ascending=False)

    print("RRR / CORRELATION HURDLE (from Portfolio Intuition, Kennedy 2018)")
    print("- Uses monthly simple returns (annualized) over the full common backtest window")
    print(f"- w (new asset weight) = {float(swap_weight)*100:.2f}%  (w_p = {100.0 - float(swap_weight)*100:.2f}%)")
    print("- Bare-minimum filter (w->0): passes if RRR_a - rho*RRR_p > 0")
    print("- If a candidate’s estimated mean return is <= 0, it is labeled N/A and this metric should be ignored for that row.")
    print()
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(df.round(4).to_string())
    print()

    # PASS/FAIL explainer
    # - "Bare-minimum" test from the paper (limit as w->0)
    # - "At-w" test using the paper's exact combined-RRR formula at the chosen w
    eps = 1e-12
    ok = df["Status"].astype(str) == "OK"
    na = df.index[~ok].tolist()

    pass_bare = df.index[ok & (df["RRR_a - rho*RRR_p"].astype(float) > eps)].tolist()
    fail_bare = df.index[ok & (df["RRR_a - rho*RRR_p"].astype(float) <= eps)].tolist()
    pass_w = df.index[ok & (df["RRR_pa(w) - RRR_p"].astype(float) > eps)].tolist()
    fail_w = df.index[ok & (df["RRR_pa(w) - RRR_p"].astype(float) <= eps)].tolist()

    def _fmt_list(xs: list[str]) -> str:
        return ", ".join(xs) if xs else "(none)"

    print("PASS/FAIL SUMMARY")
    print(f"- Bare-minimum (w→0) PASS: {_fmt_list(pass_bare)}")
    print(f"- Bare-minimum (w→0) FAIL: {_fmt_list(fail_bare)}")
    print(f"- At w={float(swap_weight)*100:.2f}% PASS (RRR_pa(w) > RRR_p): {_fmt_list(pass_w)}")
    print(f"- At w={float(swap_weight)*100:.2f}% FAIL (RRR_pa(w) ≤ RRR_p): {_fmt_list(fail_w)}")
    if na:
        print(f"- N/A (ignore this metric): {_fmt_list(na)}")
    print(
        "\nNotes:\n"
        "- RRR here is return/volatility (not Sharpe; no risk-free subtraction).\n"
        "- RRR_pa(w) is the combined portfolio’s RRR after allocating weight w to the candidate.\n"
        "- As requested: if the estimated mean return is <= 0, we label the row N/A and you should ignore this metric."
    )
    print()


def _equal_weight_portfolio_returns(rets: pd.DataFrame, *, min_assets: int = 2) -> pd.Series:
    if rets.empty:
        return pd.Series(dtype=float)
    counts = rets.notna().sum(axis=1)
    port = rets.mean(axis=1, skipna=True)
    port[counts < min_assets] = float("nan")
    return port.rename("EQW_Portfolio")


def _max_drawdown_from_simple_returns(simple_rets: pd.Series) -> float:
    if simple_rets is None or simple_rets.empty:
        return float("nan")
    r = pd.to_numeric(simple_rets, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def diversification_scores(
    candidate_rets: pd.DataFrame,
    portfolio_asset_rets: pd.DataFrame,
    *,
    portfolio_weights: dict[str, float] | None = None,
    replace_from: str | None = None,
    replace_weight: float = 0.05,
) -> pd.DataFrame:
    """
    Ported from commodities.py: rank candidates by diversification vs existing portfolio assets.

    Uses SIMPLE monthly returns because swap vol and drawdown are computed on an arithmetic portfolio series.
    """
    if candidate_rets.empty or portfolio_asset_rets.empty:
        return pd.DataFrame()

    c = candidate_rets.dropna(how="all")
    p = portfolio_asset_rets.dropna(how="all")

    idx = c.index.union(p.index)
    c = c.reindex(idx)
    p = p.reindex(idx)

    if portfolio_weights:
        port = _weighted_portfolio_returns(p, portfolio_weights, require_all_assets=True)
        port_name = "W_Portfolio"
    else:
        port = _equal_weight_portfolio_returns(p)
        port_name = "EQW_Portfolio"

    swap_from = p[replace_from] if replace_from and replace_from in p.columns else None

    rows = []
    for cand in c.columns:
        # Correlations vs each current holding (pairwise on overlapping data)
        corr_by_asset: dict[str, float] = {}
        for a in p.columns:
            pair = pd.concat([c[cand], p[a]], axis=1).dropna()
            if len(pair) >= 6:
                corr_by_asset[a] = float(pair[cand].corr(pair[a]))

        if not corr_by_asset:
            mean_abs = _np.nan
            max_abs = _np.nan
            max_abs_asset = ""
            max_abs_corr = _np.nan
        else:
            corr_arr = _np.array(list(corr_by_asset.values()), dtype=float)
            mean_abs = float(_np.nanmean(_np.abs(corr_arr)))
            # find the asset with max |corr|
            max_abs_asset = max(corr_by_asset.keys(), key=lambda k: abs(float(corr_by_asset[k])))
            max_abs_corr = float(corr_by_asset[max_abs_asset])
            max_abs = float(abs(max_abs_corr))

        if portfolio_weights:
            w = pd.Series(portfolio_weights, dtype=float)
            w = w[w.index.isin(p.columns)]
            w = w / w.sum() if w.sum() != 0 else w
            w_abs = []
            w_w = []
            for a, wa in w.items():
                if a in corr_by_asset and not _np.isnan(corr_by_asset[a]):
                    w_abs.append(abs(corr_by_asset[a]) * wa)
                    w_w.append(wa)
            w_mean_abs = float(_np.sum(w_abs) / _np.sum(w_w)) if w_w and _np.sum(w_w) > 0 else _np.nan
        else:
            w_mean_abs = _np.nan

        pairp = pd.concat([c[cand], port], axis=1).dropna()
        corr_port = pairp[cand].corr(pairp[port_name]) if len(pairp) >= 6 else _np.nan

        base = port.dropna()
        if swap_from is not None:
            comb = pd.concat([base, swap_from, c[cand]], axis=1).dropna()
        else:
            comb = pd.concat([base, c[cand]], axis=1).dropna()

        if len(comb) >= 6 and swap_from is not None:
            base_al = comb[port_name]
            from_al = comb[replace_from]
            cand_al = comb[cand]
            new_port = base_al - float(replace_weight) * from_al + float(replace_weight) * cand_al
            base_vol = float(base_al.std(ddof=1))
            new_vol = float(new_port.std(ddof=1))
            delta_vol = new_vol - base_vol
            base_mdd = _max_drawdown_from_simple_returns(base_al)
            new_mdd = _max_drawdown_from_simple_returns(new_port)
            # Use delta of absolute values: negative = improvement (lower drawdown)
            delta_mdd = float(abs(new_mdd) - abs(base_mdd)) if not (_np.isnan(new_mdd) or _np.isnan(base_mdd)) else _np.nan
        else:
            delta_vol = _np.nan
            delta_mdd = _np.nan

        cand_vol_ann = float(c[cand].dropna().std(ddof=1) * _np.sqrt(12)) if c[cand].dropna().shape[0] >= 6 else _np.nan

        offending = ""
        if max_abs_asset:
            offending = f"{max_abs_asset} ({max_abs_corr:+.3f})"

        rows.append(
            {
                "Candidate": cand,
                "mean_abs_corr_to_assets": mean_abs,
                "max_abs_corr_to_assets": max_abs,
                "max_abs_corr_offender": offending,
                "w_mean_abs_corr_to_assets": w_mean_abs,
                "corr_to_portfolio": float(corr_port) if corr_port is not None else _np.nan,
                "delta_vol_if_swap": float(delta_vol) if delta_vol is not None else _np.nan,
                "delta_max_drawdown_if_swap": float(delta_mdd) if delta_mdd is not None else _np.nan,
                "cand_vol_ann": cand_vol_ann,
                "n_months_used": int(pairp.shape[0]) if pairp is not None else 0,
            }
        )

    df = pd.DataFrame(rows).set_index("Candidate")
    primary = "w_mean_abs_corr_to_assets" if df["w_mean_abs_corr_to_assets"].notna().any() else "mean_abs_corr_to_assets"
    df = df.sort_values([primary, "delta_vol_if_swap", "delta_max_drawdown_if_swap"], ascending=[True, True, True])
    return df


def _print_diversification_ranking(
    *,
    prices_portfolio: pd.DataFrame,
    prices_candidates: pd.DataFrame,
    portfolio_target_weights: dict[str, float],
    replace_from: str | None,
    swap_weight: float
) -> pd.DataFrame:
    # Common start on month-end index (fair comparison for correlations)
    px_p_me = Portfolio.resample_prices(Portfolio.fill_non_trading_days(prices_portfolio, freq="D"), freq="ME")
    px_c_me = Portfolio.resample_prices(Portfolio.fill_non_trading_days(prices_candidates, freq="D"), freq="ME")
    common_start, starts_df = Portfolio.common_start_info(px_p_me, px_c_me)
    if common_start is not None and not starts_df.empty:
        culprits = starts_df[starts_df["Start"] == common_start].index.tolist()
        culprits_str = ", ".join(culprits) if culprits else "(unknown)"
        print(f"Shared start date (month-end index): {common_start.date()}  [forced by: {culprits_str}]")
    else:
        print("Shared start date (month-end index): unavailable (insufficient data).")

    # SIMPLE monthly returns (pct) for swap-vol and drawdown logic inside diversification_scores
    rets_candidates = Portfolio.monthly_returns_from_prices(prices_candidates, return_method="pct", common_start=common_start)
    rets_portfolio = Portfolio.monthly_returns_from_prices(prices_portfolio, return_method="pct", common_start=common_start)

    scores = diversification_scores(
        rets_candidates,
        rets_portfolio,
        portfolio_weights=portfolio_target_weights,
        replace_from=replace_from,
        replace_weight=float(swap_weight),
    )

    if scores.empty:
        print("\nDiversification ranking: unavailable (no score rows computed).\n")
        return scores

    cols = [
        "w_mean_abs_corr_to_assets",
        "max_abs_corr_to_assets",
        "max_abs_corr_offender",
        "corr_to_portfolio",
        "delta_vol_if_swap",
        "delta_max_drawdown_if_swap",
        "cand_vol_ann",
    ]
    pretty_names = {
        "w_mean_abs_corr_to_assets": "Mean |corr| vs current holdings (weighted)",
        "max_abs_corr_to_assets": "Max |corr| vs any current holding",
        "max_abs_corr_offender": "Most correlated holding (ticker, signed corr)",
        "corr_to_portfolio": "Correlation vs current portfolio (weighted)",
        "delta_vol_if_swap": "Δ monthly volatility (swap 5% Stocks → new asset)",
        "delta_max_drawdown_if_swap": "Δ max drawdown (swap 5% Stocks → new asset)",
        "cand_vol_ann": "Candidate annualized volatility",
    }
    to_show = scores[[c for c in cols if c in scores.columns]].copy().rename(columns=pretty_names)

    pct_cols = {
        "Δ monthly volatility (swap 5% Stocks → new asset)",
        "Δ max drawdown (swap 5% Stocks → new asset)",
        "Candidate annualized volatility",
    }
    corr_cols = {
        "Mean |corr| vs current holdings (weighted)",
        "Max |corr| vs any current holding",
        "Correlation vs current portfolio (weighted)",
    }

    def _fmt(metric: str, v) -> str:
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return ""
        if metric in pct_cols:
            return f"{float(v):+.4%}"
        if metric in corr_cols:
            return f"{float(v):.4f}"
        return f"{float(v):.4f}"

    formatted = to_show.T.copy()
    formatted = formatted.apply(lambda row: row.map(lambda v: _fmt(row.name, v)), axis=1)

    print("\nDIVERSIFICATION RANKING (lower mean |corr| is better)")
    if replace_from is None:
        print("Note: swap-impact columns (Δvol, Δ max drawdown) may be blank because Stocks is split across multiple tickers.")
    print()
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(formatted.to_string())
    print(
        "\nNote on drawdown: Δ max drawdown = (new − base).\n"
        "Max drawdown is negative, so a POSITIVE Δ means the drawdown became LESS severe (improved).\n"
        "A NEGATIVE Δ means it became more severe (worsened)."
    )
    print(f"\nRecommended (rank #1): {scores.index[0]}\n")
    return scores


def _print_backtest_table(
    *,
    prices_universe: pd.DataFrame,
    portfolio_tickers: list[str],
    candidates: list[str],
    base_weights: dict[str, float],
    cand_weights: dict[str, dict[str, float]],
    rf_annual: float,
    rebalance_frequency: str,
) -> None:
    # Single fair window across ALL assets involved in any candidate portfolio.
    #
    # IMPORTANT: compute start/end on the RAW price panel (no forward-fill),
    # otherwise a forward-filled series could appear to have data past its true last observation.
    raw = prices_universe.copy().sort_index()
    firsts = raw.apply(lambda s: s.first_valid_index())
    lasts = raw.apply(lambda s: s.last_valid_index())
    if firsts.isna().any() or lasts.isna().any():
        missing = list(raw.columns[firsts.isna() | lasts.isna()])
        raise ValueError(f"Missing data for tickers (cannot define common backtest range): {missing}")
    start = pd.Timestamp(max(firsts))
    end = pd.Timestamp(min(lasts))
    if start > end:
        raise ValueError(f"No overlapping date range across all assets (start={start}, end={end}).")

    print(f"Common backtest window (all assets): {start.date()} -> {end.date()}")

    # Now fill to a daily calendar INSIDE the strict common window only.
    universe = Portfolio.fill_non_trading_days(raw.loc[start:end], freq="D")

    rows: list[dict] = []

    # Baseline
    px_base = universe[portfolio_tickers]
    v_base = Portfolio.backtest_value_series(px_base, base_weights, rebalance_frequency=rebalance_frequency, initial_value=1.0)
    stats = Portfolio.backtest_stats(v_base, rf_annual=float(rf_annual))
    rows.append({"Portfolio": "Baseline (Target weights)", **stats})

    # Candidates
    for c in candidates:
        w = cand_weights[c]
        cols = portfolio_tickers + [c]
        px = universe[cols]
        v = Portfolio.backtest_value_series(px, w, rebalance_frequency=rebalance_frequency, initial_value=1.0)
        s = Portfolio.backtest_stats(v, rf_annual=float(rf_annual))
        rows.append({"Portfolio": f"Target + {c} (funded from Stocks)", **s})

    df = pd.DataFrame(rows).set_index("Portfolio")

    out = pd.DataFrame(index=df.index)
    out["Total Return (%)"] = df["total_return"].astype(float) * 100.0
    out["CAGR (%)"] = df["cagr"].astype(float) * 100.0
    out["Volatility (%)"] = df["vol_annual"].astype(float) * 100.0
    out["Max Drawdown (%)"] = df["max_drawdown"].astype(float) * 100.0
    if "longest_drawdown_days" in df.columns:
        out["Longest Drawdown (days)"] = df["longest_drawdown_days"].astype(float)
    out["Sharpe"] = df["sharpe"].astype(float)
    out["Sortino"] = df["sortino"].astype(float)

    print("\nBACKTEST RESULTS (rebalance-to-target; stats from daily LOG returns)\n")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        # percentages to 2dp; ratios to 2dp
        to_print = out.copy()
        to_print[["Total Return (%)", "CAGR (%)", "Volatility (%)", "Max Drawdown (%)"]] = to_print[
            ["Total Return (%)", "CAGR (%)", "Volatility (%)", "Max Drawdown (%)"]
        ].round(2)
        if "Longest Drawdown (days)" in to_print.columns:
            # Keep as integer days where possible.
            to_print["Longest Drawdown (days)"] = to_print["Longest Drawdown (days)"].apply(
                lambda x: int(round(float(x))) if pd.notna(x) and np.isfinite(float(x)) else np.nan
            )
        to_print[["Sharpe", "Sortino"]] = to_print[["Sharpe", "Sortino"]].round(2)
        print(to_print.to_string())
    print()


def run_whatif(
    portfolio: Any,
    add_asset_args: list[str] | None,
    *,
    swap_weight: float = 0.05,
    rf_annual: float = 0.0,
    rebalance_frequency: str = "annually",
) -> None:
    candidates = _parse_tickers(add_asset_args)
    if not candidates:
        raise ValueError("--add-asset provided but no valid tickers parsed.")

    base_target_weights = _target_weights_fraction(portfolio)
    stock_tickers = _stocks_tickers(portfolio)

    print("WHAT-IF ADD-ASSET ANALYSIS\n")
    print(
        f"Default funding rule: remove {float(swap_weight)*100:.2f}% from the Stocks allocation "
        f"and allocate {float(swap_weight)*100:.2f}% to the new asset."
    )
    if stock_tickers:
        print(f"Stocks tickers (funding source): {', '.join(stock_tickers)}")
    print()

    # Fetch prices for portfolio tickers + candidates (single download, fair window).
    tickers_universe = list(getattr(portfolio, "tickers", [])) + [c for c in candidates if c not in getattr(portfolio, "tickers", [])]
    prices_universe = Portfolio.download_prices(tickers_universe)

    portfolio_tickers = list(getattr(portfolio, "tickers", []))
    candidate_cols = [c for c in candidates if c in prices_universe.columns]

    # Use the SAME window everywhere: the strict common overlap across portfolio + candidates (same as backtest).
    cols_all = portfolio_tickers + candidate_cols
    raw = prices_universe[cols_all].copy().sort_index()
    firsts = raw.apply(lambda s: s.first_valid_index())
    lasts = raw.apply(lambda s: s.last_valid_index())
    if firsts.isna().any() or lasts.isna().any():
        missing = list(raw.columns[firsts.isna() | lasts.isna()])
        raise ValueError(f"Missing data for tickers (cannot define common analysis range): {missing}")
    start = pd.Timestamp(max(firsts))
    end = pd.Timestamp(min(lasts))
    if start > end:
        raise ValueError(f"No overlapping date range across all assets (start={start}, end={end}).")

    print(f"Common analysis/backtest window (all assets): {start.date()} -> {end.date()}\n")

    raw = raw.loc[start:end].copy()
    prices_portfolio = raw[portfolio_tickers].copy()
    prices_candidates = raw[candidate_cols].copy()

    # Diversification ranking (table across all candidates at once)
    replace_from = stock_tickers[0] if len(stock_tickers) == 1 else None
    _print_diversification_ranking(
        prices_portfolio=prices_portfolio,
        prices_candidates=prices_candidates,
        portfolio_target_weights=base_target_weights,
        replace_from=replace_from,
        swap_weight=float(swap_weight),
    )

    # Additional "Portfolio Intuition" metric (RRR vs correlation hurdle)
    _print_rrr_intuition_metric(
        prices_portfolio=prices_portfolio,
        prices_candidates=prices_candidates,
        portfolio_target_weights=base_target_weights,
        swap_weight=float(swap_weight)
    )

    # Backtests: baseline vs each candidate (rebalance-to-target)
    cand_weights: dict[str, dict[str, float]] = {}
    for c in candidates:
        if c not in prices_universe.columns:
            print(f"Skipping {c}: no price data column downloaded.")
            continue
        cand_weights[c] = _apply_swap_from_stocks(
            base_target_weights=base_target_weights,
            stock_tickers=stock_tickers,
            add_ticker=c,
            swap_weight=float(swap_weight),
        )

    if cand_weights:
        cols = portfolio_tickers + list(cand_weights.keys())
        _print_backtest_table(
            prices_universe=raw[cols],
            portfolio_tickers=portfolio_tickers,
            candidates=list(cand_weights.keys()),
            base_weights=base_target_weights,
            cand_weights=cand_weights,
            rf_annual=float(rf_annual),
            rebalance_frequency=rebalance_frequency,
        )
    else:
        print("No candidates were backtested (no valid candidate weights).\n")


def _safe_filename(s: str) -> str:
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    out = "".join(c if c in keep else "_" for c in str(s).strip())
    return out or "portfolio"


def _df_to_csv_block(df: pd.DataFrame, *, index_label: str | None = None) -> str:
    table = df.copy()
    if index_label:
        table.index = table.index.astype(str)
        table.index.name = str(index_label)
    return "```csv\n" + table.to_csv(index=True) + "```\n\n"


def build_llm_whatif_report(
    portfolio: Any,
    candidates: list[str],
    candidate_names: list[str],
    source_ticker: str,
    swap_weight: float,
    diversification_df: pd.DataFrame,
    rrr_df: pd.DataFrame | None,
    backtest_df: pd.DataFrame,
    analysis_start: str,
    analysis_end: str,
    user_preferences: str | None = None,
) -> str:
    """
    Generate an LLM prompt for what-if analysis, structured as:
    (a) system prompt, (b) portfolio overview, (c) diversification analysis with explanations,
    (d) RRR analysis with explanations, (e) backtest comparison, and (f) final question.

    Args:
        candidates: List of candidate tickers
        candidate_names: List of candidate asset names (same order as candidates)
    """
    portfolio_name = getattr(portfolio, "name", "Portfolio")
    assets_map = getattr(portfolio, "assets", {})
    target_weights_pct = getattr(portfolio, "target_weights_pct", {})
    tickers = list(getattr(portfolio, "tickers", []))

    # Build ticker-to-name mapping for candidates
    cand_name_map = {t: n for t, n in zip(candidates, candidate_names)}

    # System prompt
    system_prompt = (
        "You are a careful financial analyst and portfolio assistant. "
        "Your job is to help the user decide whether to add new assets to their portfolio and, if so, which ones. "
        "Use the provided diversification analysis, return-to-risk (RRR) metrics, and backtest results to inform your judgment. "
        "Do not hallucinate data. Ask clarifying questions if needed, and explicitly separate facts from assumptions.\n\n"
        "Consider practical constraints: diversification benefits, correlation to existing holdings, volatility impact, "
        "historical performance, risk tolerance, time horizon, rebalancing costs/taxes, and liquidity. "
        "Provide a concrete ranking of the candidate assets from most to least beneficial, with brief reasoning.\n\n"
        "The user is based in Italy and the portfolio is in EUR, intended as a permanent, long-term investment."
    )

    # Portfolio overview section
    source_label = assets_map.get(source_ticker, source_ticker)
    source_weight_pct = target_weights_pct.get(source_ticker, 0.0)

    overview_lines = [
        f"Portfolio name: {portfolio_name}",
        f"Analysis period: {analysis_start} to {analysis_end}",
        "",
        "Current portfolio allocation:",
    ]
    for t in tickers:
        label = assets_map.get(t, t)
        weight = target_weights_pct.get(t, 0.0)
        overview_lines.append(f"  - {label} ({t}): {weight:.1f}%")

    overview_lines.append("")
    # List candidates with their names and tickers
    cand_display = [f"{cand_name_map.get(c, c)} ({c})" for c in candidates]
    overview_lines.append(f"Candidate assets to evaluate: {', '.join(cand_display)}")
    overview_lines.append(f"Funding source: {source_label} ({source_ticker}), currently at {source_weight_pct:.1f}%")
    overview_lines.append(f"Swap weight: {swap_weight*100:.1f}% (from {source_label} to each candidate)")

    # Diversification analysis explanation
    diversification_explain = (
        "## Diversification Analysis\n\n"
        "This table shows how each candidate asset relates to your existing portfolio in terms of correlation and risk impact.\n\n"
        "**Column explanations:**\n"
        "- **Avg |Corr| to Assets**: Mean absolute correlation to each portfolio asset. *Lower is better* (< 0.3 = excellent diversifier, 0.3-0.5 = moderate, > 0.5 = limited diversification).\n"
        "- **Max |Corr| to Assets**: Highest absolute correlation to any single portfolio asset. *Lower is better* (watch for values > 0.7, indicating redundancy).\n"
        "- **Highest Corr Asset**: The portfolio asset with which the candidate is most correlated.\n"
        "- **Weighted Avg |Corr|**: Correlation weighted by portfolio asset weights. *Lower is better*. More relevant than unweighted average.\n"
        "- **Corr to Portfolio**: Correlation to the overall portfolio returns. *Lower is better* for diversification (< 0.3 = strong diversifier, > 0.6 = moves with portfolio).\n"
        "- **Δ Volatility (swap)**: Change in portfolio volatility if the swap is made. *Negative is better* (volatility reduction).\n"
        "- **Δ |Max Drawdown| (swap)**: Change in absolute max drawdown. *Negative is better* (smaller drawdowns).\n"
        "- **Candidate Vol (ann.)**: The candidate's own annualized volatility. For reference; high-vol assets can still be good diversifiers if uncorrelated.\n"
        "- **Months of Data**: Number of months used for analysis. More data = more reliable.\n"
    )

    # RRR analysis explanation
    rrr_explain = (
        "## Return-to-Risk Ratio (RRR) Analysis\n\n"
        "RRR analysis is based on the 'Portfolio Intuition' framework (Kennedy, 2018). "
        "It helps determine whether adding an asset improves the portfolio's risk-adjusted returns.\n\n"
        "**Key concepts:**\n"
        "- **RRR** = Annualized Return / Annualized Volatility (like Sharpe ratio, but without subtracting risk-free rate).\n"
        "- **Portfolio RRR**: The RRR of your current portfolio.\n"
        "- **Asset RRR**: The RRR of the candidate asset.\n"
        "- **Correlation (ρ)**: Correlation between the candidate and portfolio returns.\n\n"
        "**The 'bare minimum no-harm' condition:**\n"
        "For adding a new asset to not hurt the portfolio (as allocation → 0): **RRR_asset > ρ × RRR_portfolio**\n\n"
        "**Column explanations:**\n"
        "- **Hurdle (ρ × RRR_p)**: The minimum RRR the asset must exceed to be beneficial.\n"
        "- **Margin (RRR_a - Hurdle)**: *Positive = PASS* (asset clears the hurdle), *Negative = FAIL*.\n"
        "- **Combined RRR (X% swap)**: Portfolio RRR after making the swap. *Higher than Portfolio RRR is better*.\n"
        "- **Δ RRR vs Portfolio**: Change in RRR. *Positive is better*.\n"
        "- **Status**: PASS = asset improves risk-adjusted returns, FAIL = asset may hurt the portfolio.\n"
    )

    # Backtest explanation
    backtest_explain = (
        "## Backtest Comparison\n\n"
        "Historical performance comparison between the baseline portfolio and portfolios with each candidate added.\n\n"
        "**Column explanations:**\n"
        "- **Total Return (%)**: Cumulative return over the entire period.\n"
        "- **CAGR (%)**: Compound Annual Growth Rate. *Higher is better*.\n"
        "- **Volatility (%)**: Annualized standard deviation of returns. *Lower is generally better* (unless seeking higher returns).\n"
        "- **Max Drawdown (%)**: Largest peak-to-trough decline. *Lower (less negative) is better*.\n"
        "- **Sharpe**: Risk-adjusted return (excess return / volatility). *Higher is better* (> 0.5 = decent, > 1.0 = good).\n"
        "- **Sortino**: Like Sharpe but only penalizes downside volatility. *Higher is better*.\n\n"
        "**Note**: Past performance does not guarantee future results. Use backtests as one input among many.\n"
    )

    # Final question
    question = (
        "## Question\n\n"
        "Considering the diversification analysis, RRR metrics, and historical backtests provided above, "
        "please rank the candidate assets from **most beneficial to least beneficial** for adding to this portfolio.\n\n"
        "For each candidate, provide:\n"
        "1. A ranking position (1 = most beneficial)\n"
        "2. A brief justification (single sentence) covering:\n"
        "   - Diversification benefit (correlation impact)\n"
        "   - Risk-adjusted return potential (RRR analysis)\n"
        "   - Historical performance evidence\n"
        "   - Any concerns or trade-offs\n\n"
        "Finally, provide an overall recommendation: Should the user add any of these assets? "
        "If yes, which one(s) and at what allocation? If not, explain why the current portfolio may already be well-positioned."
    )

    # Build the report
    report: list[str] = []
    report.append("## System Prompt\n\n")
    report.append(system_prompt + "\n\n")

    report.append("## Portfolio Overview\n\n")
    report.append("```\n" + "\n".join(overview_lines) + "\n```\n\n")
    if user_preferences and str(user_preferences).strip():
        report.append("## User preferences\n\n")
        report.append("```\n" + str(user_preferences).strip() + "\n```\n\n")

    report.append(diversification_explain)
    if diversification_df is not None and not diversification_df.empty:
        report.append(
            "\nThe table below is CSV with a header row. "
            "The first column is the row label (Metric).\n\n"
        )
        report.append(_df_to_csv_block(diversification_df, index_label="Metric"))
    else:
        report.append("\nDiversification data: unavailable.\n\n")

    report.append(rrr_explain)
    if rrr_df is not None and not rrr_df.empty:
        report.append(
            "\nThe table below is CSV with a header row. "
            "The first column is the row label (Metric).\n\n"
        )
        report.append(_df_to_csv_block(rrr_df, index_label="Metric"))
    else:
        report.append("\nRRR data: unavailable.\n\n")

    report.append(backtest_explain)
    if backtest_df is not None and not backtest_df.empty:
        report.append(
            "\nThe table below is CSV with a header row. "
            "The first column is the row label (Portfolio).\n\n"
        )
        report.append(_df_to_csv_block(backtest_df, index_label="Portfolio"))
    else:
        report.append("\nBacktest data: unavailable.\n\n")

    report.append(question)

    return "".join(report)


def write_llm_whatif_report(text: str, portfolio_name: str) -> str:
    """Write the LLM what-if report to disk and return the path."""
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    reports_dir = os.path.abspath(reports_dir)
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"whatif_prompt_{_safe_filename(portfolio_name)}_{ts}.md"
    out = os.path.join(reports_dir, fname)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


