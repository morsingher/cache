import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

try:
    from portfolio import Portfolio
except ImportError:
    import sys
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR = os.path.join(REPO_ROOT, "cache")
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from portfolio import Portfolio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")

from ui.portfolio_builder import (
    cached_load_portfolio,
    cached_portfolio_name,
    render_portfolio_preview,
    render_example_json_ui,
    portfolio_from_json_obj_with_cache,
    render_portfolio_builder
)
from ui.components import backtest_controls, timed_step, fmt_days
from ui.assets import validate_portfolio_json_obj, get_short_name_map
from ui.charts import (
    render_portfolio_value_chart, 
    render_drawdown_chart,
    render_rolling_return_chart,
    render_rolling_volatility_chart
)

def render():
    with st.expander("ℹ️ About this section", expanded=False):
        st.markdown("""
This section compares multiple portfolios side-by-side using the same time period and settings.
What you'll get:

- **Allocation Overview**: A table showing how each portfolio is allocated across assets.
- **Key Metrics**: Key metrics for each portfolio, such as total return, CAGR, annualized volatility, Sharpe/Sortino ratios, ulcer index, max drawdown and longest drawdown period.
- **Portfolios Value**: All portfolios on the same chart so you can visually compare growth trajectories
- **Drawdown Comparison**: A chart showing the portfolios' drawdowns over time, in order to compare their risk profiles.

How to use: Select built-in portfolios, upload your own JSON files, or create manual portfolios using sliders. All portfolios will be compared over their common date range. Adjust settings, then click "Run comparison".

**Disclaimer:** This feature has been implemented for convenience, but there are [much](https://backtes.to/) [better](https://testfol.io/) [alternatives](https://www.portfoliovisualizer.com/) if portfolios comparison
is your only goal. CACH€ shines in AI-assisted portfolio rebalancing and what-if scenarios.
        """)
    
    st.markdown("### Portfolios")

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
        format_func=cached_portfolio_name,
        default=builtin_paths[:2] if len(builtin_paths) >= 2 else [],
        key="compare_builtin",
    )
    
    # Show previews for selected built-in portfolios
    for path in selected_builtin:
        try:
            p_preview = cached_load_portfolio(path)
            render_portfolio_preview(p_preview)
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
                ok, errs = validate_portfolio_json_obj(obj)
                if not ok:
                    st.warning(f"Upload '{up.name}' rejected:\n- " + "\n- ".join(errs))
                    continue

                p_preview = portfolio_from_json_obj_with_cache(obj, source=up.name)
                render_portfolio_preview(p_preview)
            except Exception:
                pass

    # Help users create valid JSONs by providing examples.
    render_example_json_ui(key_prefix="compare")
    
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
        # Use render_portfolio_builder but restricted to manual mode
        mp, _ = render_portfolio_builder(
            key=f"compare_manual_{i}", 
            title="", 
            modes=["Manual"], 
            default_mode="Manual"
        )
        if mp is not None:
            manual_portfolios.append(mp)

    # Collect all portfolios to determine available date range
    all_portfolios: list[tuple[str, Portfolio]] = []
    for path in selected_builtin:
        try:
            p_temp = cached_load_portfolio(path)
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
                ok, errs = validate_portfolio_json_obj(obj)
                if not ok:
                    continue

                p_temp = portfolio_from_json_obj_with_cache(obj, source=up.name)
                name = getattr(p_temp, "name", None) or up.name
                all_portfolios.append((str(name), p_temp))
            except Exception:
                pass
    for mp in manual_portfolios:
        all_portfolios.append((getattr(mp, "name", "Manual"), mp))

    # Settings section (after portfolios)
    st.markdown("### Settings")
    rebalance_frequency, initial_amount, rf_annual = backtest_controls(key_prefix="compare")

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

                timed_step(step_ph, "Aligning portfolio data...", step_start)
                step_ph = st.empty()
                step_ph.write("Running backtests...")
                step_start = time.time()
                
                rows: list[dict[str, object]] = []
                value_series: dict[str, pd.Series] = {}
                allocations: dict[str, dict[str, float]] = {}  # portfolio_name -> {short_name: weight_pct}
                short_map = get_short_name_map()

                for name, p in all_portfolios:
                    if not hasattr(p, "target_weights_pct"):
                        raise ValueError(f"Portfolio '{name}' is missing per-asset target weights.")

                    wt_pct = [float(p.target_weights_pct[t]) for t in p.tickers]
                    wt = Portfolio._normalize_weights_to_fraction(wt_pct)
                    weights = {t: float(w) for t, w in zip(p.tickers, wt)}
                    
                    # Collect allocation using short names
                    alloc = {}
                    for ticker in p.tickers:
                        short_name = short_map.get(ticker, ticker)
                        weight_pct = float(p.target_weights_pct.get(ticker, 0.0))
                        alloc[short_name] = weight_pct
                    allocations[name] = alloc

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

                df_metrics = pd.DataFrame(rows).set_index("Portfolio")
                
                timed_step(step_ph, "Running backtests...", step_start)
                
                # Store results in session state
                st.session_state["compare_results"] = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "metrics": df_metrics,
                    "value_series": value_series,
                    "allocations": allocations,
                }
                
                total_elapsed = time.time() - total_start
                status.update(label=f"Comparison complete! ({total_elapsed:.2f}s)", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Comparison failed", state="error", expanded=False)
                st.error(str(e))
                st.session_state.pop("compare_results", None)

    # Display results
    if "compare_results" in st.session_state:
        results = st.session_state["compare_results"]
        
        # Show data freshness with comparison period
        from ui.assets import get_data_status
        data_available, freshness = get_data_status()
        freshness_note = f" | Data: {freshness}" if data_available else ""
        st.caption(f"Comparison period: {results['start_date']} → {results['end_date']}{freshness_note}")

        # Allocation Overview
        st.markdown("### Allocation Overview")
        allocations = results.get("allocations", {})
        if allocations:
            # Build a DataFrame with assets as rows and portfolios as columns
            all_assets = set()
            for alloc in allocations.values():
                all_assets.update(alloc.keys())
            all_assets = sorted(all_assets)
            
            alloc_data = {}
            for portfolio_name, alloc in allocations.items():
                alloc_data[portfolio_name] = {asset: f"{alloc.get(asset, 0.0):.1f}%" for asset in all_assets}
            
            alloc_df = pd.DataFrame(alloc_data, index=all_assets)
            # Replace 0.0% with "—" for cleaner display
            alloc_df = alloc_df.replace("0.0%", "—")
            st.dataframe(alloc_df, width="stretch")
            
            with st.expander("ℹ️ How to read this?", expanded=False):
                st.markdown("""
This table shows the target allocation of each portfolio across all assets. Each column represents a portfolio, and each row represents an asset. The percentage shown is the target weight for that asset in that portfolio.

A dash (—) indicates the portfolio does not hold that asset.
                """)

        # Key Metrics
        st.markdown("### Key Metrics")
        
        # Format DataFrame for display - TRANSPOSED (metrics as rows, portfolios as columns)
        df_display = results["metrics"].copy()
        
        # Format percentage columns
        pct_cols = ["Total Return", "CAGR", "Vol (ann.)", "Max Drawdown"]
        for c in pct_cols:
            if c in df_display.columns:
                df_display[c] = df_display[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
        
        # Format other numeric columns
        for c in ["Sharpe", "Sortino", "Ulcer Index"]:
            if c in df_display.columns:
                df_display[c] = df_display[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        
        if "Longest Drawdown" in df_display.columns:
            df_display["Longest Drawdown"] = df_display["Longest Drawdown"].apply(lambda x: fmt_days(x))

        # Transpose: metrics as rows, portfolios as columns
        st.dataframe(df_display.T, width="stretch")
        
        with st.expander("ℹ️ What do these metrics mean?", expanded=False):
            st.markdown(
                """
**Total Return** — The overall percentage gain (or loss) from start to end of the period.

**CAGR** — Compound Annual Growth Rate. The smoothed annual return that would produce the same total return if compounded each year.

**Vol (ann.)** — Annualized standard deviation of returns. Higher values indicate larger price swings.

**Sharpe** — Risk-adjusted return: (Return - Risk-Free Rate) / Volatility. Higher is better; above 1.0 is generally considered good.

**Max Drawdown** — The largest peak-to-trough decline during the period. Shows worst-case loss if you bought at the peak.

**Longest Drawdown** — How long (in days) the portfolio took to recover from its worst drawdown.

**Ulcer Index** — Measures both depth and duration of drawdowns. Lower is better; it penalizes prolonged declines more than brief ones.

**Sortino** — Like Sharpe, but only penalizes downside volatility. Higher is better; useful when returns are asymmetric.
                """
            )

        st.markdown("#### Portfolios Value")
        
        value_series = results["value_series"]
        chart_data = []
        for name, v in value_series.items():
            for date, val in v.items():
                chart_data.append({"Date": date, "Portfolio": name, "Value": float(val)})
        
        chart_df = pd.DataFrame(chart_data)
        
        # Sort order by final value
        final_values = {name: v.iloc[-1] for name, v in value_series.items()}
        sorted_portfolios = sorted(final_values.keys(), key=lambda k: final_values[k], reverse=True)
        
        render_portfolio_value_chart(
            chart_df,
            portfolio_order=sorted_portfolios,
            y_scale=y_scale
        )
        
        with st.expander("ℹ️ What does this chart show?", expanded=False):
            st.markdown(
                """
This shows the **growth of each portfolio's value** over time, starting from a normalized value. Each line represents how your investment would have grown (or shrunk) by each date, allowing direct comparison of portfolio performance over the common period.
                """
            )

        render_drawdown_chart(
            value_series,
            title="Drawdown Comparison",
            portfolio_order=sorted_portfolios
        )

        render_rolling_return_chart(
            value_series,
            portfolio_order=sorted_portfolios
        )

        render_rolling_volatility_chart(
            value_series,
            portfolio_order=sorted_portfolios
        )
