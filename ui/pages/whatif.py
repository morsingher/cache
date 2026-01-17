import time
import random
import pandas as pd
import json
import os
import streamlit as st
import altair as alt
import numpy as np

try:
    from whatif import (
        _target_weights_fraction,
        diversification_scores,
        build_llm_whatif_report,
    )
except ImportError:
    import sys
    import os
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR = os.path.join(REPO_ROOT, "cache")
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from whatif import (
        _target_weights_fraction,
        diversification_scores,
        build_llm_whatif_report,
    )

try:
    from portfolio import Portfolio
except ImportError:
    from portfolio import Portfolio

from ui.portfolio_builder import render_portfolio_builder
from ui.assets import (
    get_prices_and_store,
    load_available_assets,
    get_short_name_map,
)
from ui.components import timed_step, render_llm_query_ui, backtest_controls
from ui.charts import (
    render_portfolio_value_chart, 
    render_drawdown_chart,
    render_rolling_return_chart, 
    render_rolling_volatility_chart
)

def _fmt_days(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "nan"
    if pd.isna(v):
        return "nan"
    return f"{int(round(v))}d"

def render():
    with st.expander("ℹ️ About this section", expanded=False):
        st.markdown("""
This section explores what would happen if you added a new asset to your portfolio by swapping a portion of an existing position.
What you'll get:

- **Diversification Analysis**: Correlations, volatility impact, and how well candidates diversify your portfolio.
- **RRR analysis**: Return-to-Risk Ratio test based on modern portfolio theory from [Bridge Alternatives](https://www.bridgealternatives.com/insights/portfolio-intuition).
- **Backtest Comparison**: Side-by-side performance of your baseline portfolio against each modified portfolio.
- **Portfolio Value**: A chart showing the historical growth of your baseline portfolio against each modified version.
- **Drawdown Comparison**: A chart showing the drawdown of your baseline portfolio against each modified version.
- **AI Assistant**: A detailed prompt containing all the previous information that you can either copy and paste into your preferred LLM, or use directly within the app through the [OpenRouter API](https://openrouter.ai/).
In this case, only free tier models are available, so take the responses with a grain of salt. The goal is to help you reason about the trade-offs and make the best decision.

How to use: define your baseline portfolio by either manually creating it, selecting a built-in one or uploading your own, select candidate assets to evaluate, choose which position to fund from and by how much, then click "Run what-if".
        """)
    
    p, _ = render_portfolio_builder(key="whatif", title="Portfolio", allow_value=False)
    
    if p is not None:
        # Load predefined candidate assets
        available_candidates = load_available_assets()
        
        # Get portfolio tickers to filter out already-included assets
        portfolio_tickers_set = set(getattr(p, "tickers", []))
        
        # Build a unique key for this portfolio (source mode + tickers + name)
        portfolio_source = st.session_state.get("whatif_source", "")
        portfolio_name = getattr(p, "name", "")
        portfolio_identity_key = (portfolio_source, portfolio_name, tuple(sorted(portfolio_tickers_set)))

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
            
            # Detect portfolio change (source, name, or tickers) and reset candidates if needed
            prev_portfolio_key = st.session_state.get("whatif_portfolio_identity_key")
            if prev_portfolio_key != portfolio_identity_key:
                # Portfolio changed - clear old selection and pick new random defaults
                st.session_state.pop("whatif_candidates", None)
                st.session_state["whatif_portfolio_identity_key"] = portfolio_identity_key
            
            # Initialize with 3 random candidates if no selection exists
            if "whatif_candidates" not in st.session_state:
                num_defaults = min(3, len(sorted_candidate_shorts))
                random_defaults = random.sample(sorted_candidate_shorts, num_defaults)
                st.session_state["whatif_candidates"] = random_defaults
            
            selected_candidate_shorts = st.multiselect(
                "Candidate assets to evaluate",
                options=sorted_candidate_shorts,
                format_func=lambda x: candidate_short_to_display.get(x, x),
                key="whatif_candidates",
                help="Select one or more assets to analyze for potential inclusion in your portfolio.",
            )
            
            # Show warning if no candidates selected
            if not selected_candidate_shorts:
                st.error("Please select at least one candidate asset to evaluate.")

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
            # We need to map index to option
            if not source_options:
                st.warning("Portfolio has no assets.")
                return

            source_asset_idx = st.selectbox(
                "Fund new asset from",
                options=range(len(source_options)),
                format_func=lambda i: source_options[i][1],
                index=0,
                key="whatif_source_asset",
            )
            source_ticker = source_options[source_asset_idx][0]
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

        st.markdown("### Settings")
        rebalance_frequency, _, rf_annual = backtest_controls(key_prefix="whatif", show_initial_amount=False)

        # Date range selection logic (similar to analyze)
        try:
            prices_raw = p._prices_df().dropna(how="any").sort_index()
            available_start = prices_raw.index.min().date() if not prices_raw.empty else None
            available_end = prices_raw.index.max().date() if not prices_raw.empty else None
        except Exception:
            available_start, available_end = None, None

        if available_start and available_end:
            date_c1, date_c2 = st.columns(2)
            with date_c1:
                user_start = st.date_input("Start date", value=available_start, min_value=available_start, max_value=available_end, key="whatif_start_date")
            with date_c2:
                user_end = st.date_input("End date", value=available_end, min_value=available_start, max_value=available_end, key="whatif_end_date")
        else:
            st.warning("Could not determine available date range.")
            user_start, user_end = None, None

        y_scale = st.radio("Y-axis scale (for value chart)", options=["Linear", "Logarithmic"], index=0, horizontal=True, key="whatif_y_scale")

        user_prefs = st.text_area(
            "User preferences (optional)",
            placeholder="Example: I prefer assets with more/less volatility.",
            key="whatif_user_prefs",
            help="Add any constraints or preferences to include in the LLM prompt.",
        )

        run = st.button("Run what-if", type="primary", key="whatif_run")

        if run:
            with st.status("Running what-if analysis...", expanded=True) as status:
                try:
                    total_start = time.time()
                    
                    if not filtered_candidates or not selected_candidate_shorts:
                        raise ValueError("No candidate assets selected.")

                    candidates = [candidate_short_to_ticker[short] for short in selected_candidate_shorts]
                    candidate_name_map = {candidate_short_to_ticker[short]: short for short in selected_candidate_shorts}

                    base_target_weights = _target_weights_fraction(p)

                    step_ph = st.empty()
                    step_ph.write("Downloading price data...")
                    step_start = time.time()
                    
                    tickers_universe = list(getattr(p, "tickers", [])) + [c for c in candidates if c not in getattr(p, "tickers", [])]
                    prices_universe = get_prices_and_store(tuple(sorted(tickers_universe)))

                    portfolio_tickers = list(getattr(p, "tickers", []))
                    candidate_cols = [c for c in candidates if c in prices_universe.columns]
                    cols_all = portfolio_tickers + candidate_cols
                    raw = prices_universe[cols_all].copy().sort_index()
                    
                    # Determine overlapping range
                    firsts = raw.apply(lambda s: s.first_valid_index())
                    lasts = raw.apply(lambda s: s.last_valid_index())
                    if firsts.isna().any() or lasts.isna().any():
                        missing = list(raw.columns[firsts.isna() | lasts.isna()])
                        raise ValueError(f"Missing data for tickers: {missing}")
                    
                    start = pd.Timestamp(max(firsts))
                    end = pd.Timestamp(min(lasts))
                    
                    if user_start and user_end:
                        # Apply user range but constrained by data availability
                        req_start = pd.Timestamp(user_start)
                        req_end = pd.Timestamp(user_end)
                        start = max(start, req_start)
                        end = min(end, req_end)
                    
                    if start > end:
                        raise ValueError(f"No overlapping date range (start={start}, end={end}).")

                    analysis_start_str = str(start.date())
                    analysis_end_str = str(end.date())
                    raw = raw.loc[start:end].copy()

                    prices_portfolio = raw[portfolio_tickers].copy()
                    prices_candidates = raw[candidate_cols].copy()

                    # Resample for monthly returns
                    px_p_me = Portfolio.resample_prices(Portfolio.fill_non_trading_days(prices_portfolio, freq="D"), freq="ME")
                    px_c_me = Portfolio.resample_prices(Portfolio.fill_non_trading_days(prices_candidates, freq="D"), freq="ME")
                    common_start_ts, _ = Portfolio.common_start_info(px_p_me, px_c_me)

                    rets_candidates = Portfolio.monthly_returns_from_prices(prices_candidates, return_method="pct", common_start=common_start_ts)
                    rets_portfolio = Portfolio.monthly_returns_from_prices(prices_portfolio, return_method="pct", common_start=common_start_ts)

                    timed_step(step_ph, "Downloading price data...", step_start)
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

                    scores_transposed = None
                    llm_scores_df = None
                    candidate_order = [candidate_name_map.get(c, c) for c in candidates]
                    candidate_order = sorted(dict.fromkeys(candidate_order))
                    if not scores.empty:
                        scores_display = scores.copy()
                        scores_display.index = [candidate_name_map.get(t, t) for t in scores_display.index]
                        
                        # Formatting helpers
                        short_map = get_short_name_map()
                        def _offender_to_short(val: str) -> str:
                            if not val or not isinstance(val, str): return "—"
                            if " (" in val: ticker = val.split(" (")[0].strip()
                            else: ticker = val.strip()
                            return short_map.get(ticker, ticker)
                        
                        if "max_abs_corr_offender" in scores_display.columns:
                            scores_display["max_abs_corr_offender"] = scores_display["max_abs_corr_offender"].apply(_offender_to_short)
                        
                        scores_display = scores_display.rename(columns={
                            "mean_abs_corr_to_assets": "Avg |Corr| to Assets",
                            "max_abs_corr_to_assets": "Max |Corr| to Assets",
                            "max_abs_corr_offender": "Highest Corr Asset",
                            "w_mean_abs_corr_to_assets": "Weighted Avg |Corr|",
                            "corr_to_portfolio": "Corr to Portfolio",
                            "delta_vol_if_swap": "Δ Volatility (%)",
                            "delta_max_drawdown_if_swap": "Δ |Max Drawdown| (%)",
                            "cand_vol_ann": "Candidate Vol (ann.)",
                            "n_months_used": "Months of Data",
                        })
                        scores_transposed = scores_display.T
                        # Ensure stable candidate ordering across tables
                        scores_transposed = scores_transposed.reindex(columns=[c for c in candidate_order if c in scores_transposed.columns])
                        llm_scores_df = scores_transposed

                    timed_step(step_ph, "Computing diversification scores...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Running RRR analysis...")
                    step_start = time.time()
                    
                    # RRR Analysis
                    rrr_transposed = None
                    llm_rrr_df = None
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
                                if s.empty: continue
                                aligned = pd.concat([port_r.rename("port"), s.rename("cand")], axis=1).dropna()
                                if aligned.shape[0] < 6: continue
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
                                rrr_transposed = rrr_transposed.reindex(columns=[c for c in candidate_order if c in rrr_transposed.columns])
                                llm_rrr_df = rrr_transposed
                    except Exception as e:
                        rrr_error = str(e)

                    timed_step(step_ph, "Running RRR analysis...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Running backtests...")
                    step_start = time.time()
                    
                    # Backtest comparison
                    backtest_df = None
                    llm_backtest_df = None
                    cand_weights: dict[str, dict[str, float]] = {}
                    for c in candidates:
                        if c not in prices_universe.columns: continue
                        new_weights = base_target_weights.copy()
                        new_weights[source_ticker] = new_weights.get(source_ticker, 0.0) - swap_weight
                        new_weights[c] = swap_weight
                        cand_weights[c] = new_weights

                    value_series_dict = {}
                    if cand_weights:
                        universe = Portfolio.fill_non_trading_days(raw[portfolio_tickers + list(cand_weights.keys())], freq="D")
                        rows = []
                        
                        px_base = universe[portfolio_tickers]
                        v_base = Portfolio.backtest_value_series(px_base, base_target_weights, rebalance_frequency=str(rebalance_frequency), initial_value=1.0)
                        s_base = Portfolio.backtest_stats(v_base, rf_annual=float(rf_annual))
                        rows.append({"Portfolio": "Baseline", **s_base})
                        value_series_dict["Baseline"] = v_base

                        for c, w in cand_weights.items():
                            cols = portfolio_tickers + [c]
                            px = universe[cols]
                            v = Portfolio.backtest_value_series(px, w, rebalance_frequency=str(rebalance_frequency), initial_value=1.0)
                            s = Portfolio.backtest_stats(v, rf_annual=float(rf_annual))
                            cand_name = candidate_name_map.get(c, c)
                            portfolio_label = f"+ {cand_name}"
                            rows.append({"Portfolio": portfolio_label, **s})
                            value_series_dict[portfolio_label] = v

                        backtest_results_df = pd.DataFrame(rows).set_index("Portfolio")
                        ordered_index = ["Baseline"] + [f"+ {c}" for c in candidate_order if f"+ {c}" in backtest_results_df.index]
                        backtest_results_df = backtest_results_df.reindex(index=ordered_index)
                        # Format for LLM
                        llm_backtest_df = backtest_results_df.copy()
                        # Clean up numeric columns for display later if needed, but LLM needs raw or formatted string
                        backtest_df = backtest_results_df # Store raw for now

                        # Align diversification deltas with backtest metrics (same methodology/window).
                        if scores_transposed is not None and "Baseline" in backtest_df.index:
                            base_vol = float(backtest_df.loc["Baseline", "vol_annual"])
                            base_mdd = float(backtest_df.loc["Baseline", "max_drawdown"])
                            for cand_name in candidate_order:
                                label = f"+ {cand_name}"
                                if label not in backtest_df.index:
                                    continue
                                cand_vol = float(backtest_df.loc[label, "vol_annual"])
                                cand_mdd = float(backtest_df.loc[label, "max_drawdown"])
                                delta_vol = cand_vol - base_vol
                                delta_mdd = abs(cand_mdd) - abs(base_mdd)
                                if "Δ Volatility (%)" in scores_transposed.index and cand_name in scores_transposed.columns:
                                    scores_transposed.loc["Δ Volatility (%)", cand_name] = delta_vol
                                if "Δ |Max Drawdown| (%)" in scores_transposed.index and cand_name in scores_transposed.columns:
                                    scores_transposed.loc["Δ |Max Drawdown| (%)", cand_name] = delta_mdd
                            llm_scores_df = scores_transposed

                    timed_step(step_ph, "Running backtests...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Generating LLM prompt...")
                    step_start = time.time()

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
                            user_preferences=user_prefs,
                        )
                    except Exception:
                        pass

                    st.session_state["whatif_results"] = {
                        "analysis_start": analysis_start_str,
                        "analysis_end": analysis_end_str,
                        "scores_transposed": scores_transposed,
                        "rrr_transposed": rrr_transposed,
                        "rrr_error": rrr_error,
                        "backtest_df": backtest_df,
                        "value_series": value_series_dict,
                        "llm_prompt": llm_prompt,
                        "portfolio_name": getattr(p, "name", "Portfolio"),
                        "swap_pct": swap_pct,
                    }
                    st.session_state.pop("whatif_llm_response", None)
                    
                    total_elapsed = time.time() - total_start
                    status.update(label=f"What-if analysis complete! ({total_elapsed:.2f}s)", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="What-if analysis failed", state="error", expanded=False)
                    st.error(str(e))
                    st.session_state.pop("whatif_results", None)

        if "whatif_results" in st.session_state:
            results = st.session_state["whatif_results"]
            
            # Show data freshness with analysis period
            from ui.assets import get_data_status
            data_available, freshness = get_data_status()
            freshness_note = f" | Data: {freshness}" if data_available else ""
            st.caption(f"Common analysis window: {results['analysis_start']} → {results['analysis_end']}{freshness_note}")

            if results["scores_transposed"] is not None:
                st.markdown("### Diversification Analysis")
                # Round numeric values to 2 decimal places
                scores_display = results["scores_transposed"].copy()
                pct_rows = {
                    "Δ Volatility (%)",
                    "Δ |Max Drawdown| (%)",
                    "Candidate Vol (ann.)",
                }
                def _fmt_cell(x, row_name):
                    if isinstance(x, (int, float)) and pd.notna(x):
                        if row_name in pct_rows:
                            return f"{float(x) * 100:.2f}%"
                        return f"{float(x):.2f}"
                    if isinstance(x, float) and x == int(x):
                        return str(int(x))
                    return str(x)

                scores_display = scores_display.apply(lambda row: row.map(lambda x: _fmt_cell(x, row.name)), axis=1)
                st.dataframe(scores_display, width="stretch")
                
                with st.expander("ℹ️ How to read this?", expanded=False):
                    st.markdown("""
This table shows how well each candidate asset would diversify your portfolio. Lower correlation values generally indicate better diversification potential.

**Avg |Corr| to Assets** — Average absolute correlation between the candidate and each existing portfolio asset. Lower values mean the candidate moves more independently from your current holdings.

**Max |Corr| to Assets** — Highest absolute correlation to any single portfolio asset. Identifies if the candidate is highly correlated with at least one position.

**Highest Corr Asset** — Which portfolio asset the candidate is most correlated with. Useful to identify potential redundancy.

**Weighted Avg |Corr|** — Average correlation weighted by portfolio asset weights. Accounts for how much each correlated asset matters in your allocation.

**Corr to Portfolio** — Correlation to overall portfolio returns. Lower values mean the candidate provides more diversification at the portfolio level.

**Δ Volatility (%)** — Projected change in portfolio volatility if you add this candidate. Negative values indicate the swap would reduce overall portfolio risk.

**Δ |Max Drawdown| (%)** — Projected change in maximum drawdown magnitude. Negative values mean smaller worst-case losses.

**Candidate Vol (ann.)** — The candidate's own annualized volatility. Higher values indicate more price swings.

**Months of Data** — How many months of overlapping data were used for the analysis. More data generally means more reliable estimates.
                    """)

            st.markdown("### Return-to-Risk Ratio (RRR) Analysis")
            if results["rrr_transposed"] is not None:
                st.dataframe(results["rrr_transposed"].astype(str), width="stretch")
                with st.expander("ℹ️ How to read this?", expanded=False):
                    st.markdown("""
This analysis uses the Return-to-Risk Ratio (RRR = Annualized Return / Annualized Volatility) to determine if adding a candidate asset would improve your portfolio's risk-adjusted performance.
Note: RRR here is computed from monthly simple returns and then annualized (12 periods/year).

**Portfolio RRR** — Your current portfolio's return-to-risk ratio. This is the benchmark to beat.

**Asset RRR** — The candidate asset's own return-to-risk ratio. Higher is better, but this alone doesn't determine if adding it helps your portfolio.

**Correlation (ρ)** — Correlation between the candidate and your portfolio returns. Lower correlation means more diversification benefit.

**Hurdle (ρ × RRR_p)** — The minimum RRR the candidate needs to clear to potentially improve your portfolio. It equals the portfolio RRR scaled by correlation. Lower correlation = lower hurdle = easier to pass.

**Margin (RRR_a - Hurdle)** — How much the candidate's RRR exceeds the hurdle. Positive margin suggests the asset could add value. Larger margins indicate stronger candidates.

**Combined RRR (X% swap)** — Projected portfolio RRR after swapping the specified percentage. This is the key metric—higher than Portfolio RRR means improvement.

**Δ RRR vs Portfolio** — Direct comparison: Combined RRR minus Portfolio RRR. Positive values indicate the swap would improve risk-adjusted returns.

**Status** — PASS means the candidate clears the hurdle test (positive margin). FAIL means it doesn't. N/A indicates insufficient data or negative expected returns.

*Note: PASS doesn't guarantee improvement—it's a necessary but not sufficient condition. Always consider the Combined RRR and Δ RRR for the complete picture.*
                    
All the credits to [Bridge Alternatives](https://www.bridgealternatives.com/insights/portfolio-intuition).
                    """)
            elif results["rrr_error"]:
                st.caption(f"RRR analysis unavailable: {results['rrr_error']}")

            if results["backtest_df"] is not None:
                st.markdown("### Backtest Comparison")
                bt_df = results["backtest_df"].copy()
                
                # Format percentage columns
                pct_cols = ["total_return", "cagr", "vol_annual", "max_drawdown"]
                for c in pct_cols:
                    if c in bt_df.columns:
                        bt_df[c] = bt_df[c].apply(lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "—")
                
                # Format other numeric columns to 2 decimals
                for c in ["sharpe", "sortino", "ulcer_index", "max_gain"]:
                    if c in bt_df.columns:
                        bt_df[c] = bt_df[c].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "—")
                
                bt_df = bt_df.rename(columns={
                    "total_return": "Total Return (%)", 
                    "cagr": "CAGR (%)", 
                    "vol_annual": "Vol (ann.) (%)", 
                    "max_drawdown": "Max Drawdown (%)",
                    "sharpe": "Sharpe", "sortino": "Sortino", "ulcer_index": "Ulcer Index",
                    "max_gain": "Max Gain (%)"
                })
                
                if "longest_drawdown_days" in bt_df.columns:
                    bt_df["Longest Drawdown"] = bt_df["longest_drawdown_days"].apply(lambda x: _fmt_days(x))
                    bt_df = bt_df.drop(columns=["longest_drawdown_days"])

                st.dataframe(bt_df.T, width="stretch")
                
                with st.expander("ℹ️ How to read this?", expanded=False):
                    st.markdown("""
This table shows historical backtest results comparing your baseline portfolio against modified versions that include each candidate asset. The "Baseline" column shows your original portfolio; other columns show what would have happened if you had swapped in each candidate.

**Total Return (%)** — Overall percentage gain from start to end of the analysis period.

**CAGR (%)** — Compound Annual Growth Rate. The smoothed annual return that would produce the same total return.

**Vol (ann.) (%)** — Annualized volatility (standard deviation of returns). Lower values indicate smoother performance.
Note: Volatility, Sharpe, and Sortino are computed from daily log returns and annualized using 252 trading days.

**Sharpe** — Risk-adjusted return: (Return - Risk-Free Rate) / Volatility. Higher is better.

**Sortino** — Like Sharpe, but only penalizes downside volatility. Higher is better.

**Max Drawdown (%)** — Largest peak-to-trough decline. Shows worst-case loss if you bought at the peak.

**Longest Drawdown** — How long (in days) until the portfolio recovered from its worst drawdown.

**Ulcer Index** — Measures both depth and duration of drawdowns. Lower is better.

*Note: Past performance does not guarantee future results. Backtests show what would have happened historically, not what will happen.*
                    """)

                value_series = results.get("value_series", {})
                if value_series:
                    st.markdown("#### Portfolio Value")
                    
                    # Ensure Baseline is first in the order
                    portfolio_order = ["Baseline"] + [k for k in value_series.keys() if k != "Baseline"]
                    
                    chart_data = []
                    for name in portfolio_order:
                        if name in value_series:
                            v = value_series[name]
                            for date, val in v.items():
                                chart_data.append({"Date": date, "Portfolio": name, "Value": float(val)})
                    
                    chart_df = pd.DataFrame(chart_data)
                    render_portfolio_value_chart(chart_df, y_scale=y_scale, portfolio_order=portfolio_order)
                    
                    with st.expander("ℹ️ How to read this?", expanded=False):
                        st.markdown("""
This chart compares the historical growth of your baseline portfolio against each modified version. All portfolios start at €1 for easy comparison.
The **Baseline** line shows your original portfolio. Other lines show how the portfolio would have performed with each candidate asset swapped in. Lines that end higher performed better over the period; lines that are smoother had less volatility along the way.
                        """)
                    
                    render_drawdown_chart(value_series, title="Drawdown Comparison", portfolio_order=portfolio_order)

                    render_rolling_return_chart(
                        value_series,
                        portfolio_order=portfolio_order
                    )
                    render_rolling_volatility_chart(
                        value_series,
                        portfolio_order=portfolio_order
                    )

            if results["llm_prompt"]:
                st.markdown("### AI Assistant")
                st.caption("Use the prompt below with your preferred LLM, or query one directly.")
                with st.expander("View/copy prompt", expanded=False):
                    # Note: Don't use a fixed key here - it would cause Streamlit to show
                    # the cached session state value instead of the updated prompt
                    st.text_area(
                        "Prompt (copyable)",
                        value=results["llm_prompt"],
                        height=300,
                        help="Select all and copy (Ctrl/Cmd+C).",
                    )

                render_llm_query_ui(
                    key_prefix="whatif",
                    llm_prompt=results["llm_prompt"],
                    title="", 
                )
