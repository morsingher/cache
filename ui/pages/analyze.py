import time
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.dates as mdates

try:
    from portfolio import Portfolio
except ImportError:
    import sys
    import os
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR = os.path.join(REPO_ROOT, "cache")
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from portfolio import Portfolio

from ui.portfolio_builder import render_portfolio_builder
from ui.components import backtest_controls, timed_step
from ui.charts import (
    render_portfolio_value_chart, 
    render_drawdown_chart, 
    render_rolling_return_chart, 
    render_rolling_volatility_chart
)
from ui.assets import get_short_name_map, get_data_status, load_available_assets

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
This section analyzes a single portfolio's historical performance using backtesting.
What you'll get:
- **Key Metrics**: Total return, CAGR, volatility, Sharpe/Sortino ratios, ulcer index, max drawdown, longest drawdown period.
- **Portfolio Value**: How your portfolio would have grown over your selected time period.
- **Asset Trajectories**: Individual performance of each asset in your portfolio, with the possibility of showing specific assets only.
- **Rolling 12M Correlation vs Stocks**: Rolling 1-year correlation of each asset vs. stocks, helping you understand diversification.
- **Portfolio Drawdown**: A chart showing the portfolio's drawdown over time, helping you understand the risk of your portfolio.

How to use: create a portfolio using the sliders, use a default one or load a pre-built one, adjust the backtest settings, then click "Run analysis".

**Disclaimer:** This feature has been implemented for convenience, but there are [much](https://backtes.to/) [better](https://testfol.io/) [alternatives](https://www.portfoliovisualizer.com/) if portfolio analysis
is your only goal. CACH€ shines in AI-assisted portfolio rebalancing and what-if scenarios.
        """)
    
    p, _ = render_portfolio_builder(key="analyze", title="Portfolio", allow_value=False)
    if p is not None:
        st.markdown("### Settings")
        rebalance_frequency, initial_amount, rf_annual = backtest_controls(key_prefix="analyze")
        
        # Get available date range from portfolio data
        try:
            prices_raw = p._prices_df().dropna(how="any").sort_index()
            available_start = prices_raw.index.min().date() if not prices_raw.empty else None
            available_end = prices_raw.index.max().date() if not prices_raw.empty else None
        except Exception:
            available_start, available_end = None, None

        if available_start and available_end:
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

                    if user_start and user_end:
                        prices_filtered = prices_full.loc[str(user_start):str(user_end)]
                    else:
                        prices_filtered = prices_full

                    if prices_filtered.empty:
                        raise ValueError("No data available in the selected date range.")

                    timed_step(step_ph, "Preparing data...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Computing portfolio value...")
                    step_start = time.time()

                    target_weights_pct = getattr(p, "target_weights_pct", {})
                    if not target_weights_pct:
                        raise ValueError("Portfolio is missing target weights.")

                    wt_pct = [float(target_weights_pct[t]) for t in p.tickers]
                    wt = Portfolio._normalize_weights_to_fraction(wt_pct)
                    weights = {t: float(w) for t, w in zip(p.tickers, wt)}

                    value_series = Portfolio.backtest_value_series(
                        prices_filtered,
                        weights,
                        rebalance_frequency=str(rebalance_frequency),
                        initial_value=float(initial_amount),
                    ).dropna()

                    if value_series.empty:
                        raise ValueError("Portfolio value series is empty.")

                    # Individual asset series (for trajectories)
                    base_price = prices_filtered.iloc[0]
                    asset_growth = prices_filtered.divide(base_price) * float(initial_amount)

                    timed_step(step_ph, "Computing portfolio value...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Calculating statistics...")
                    step_start = time.time()

                    stats = Portfolio.backtest_stats(
                        value_series,
                        rf_annual=float(rf_annual),
                    )

                    timed_step(step_ph, "Calculating statistics...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Computing correlations...")
                    step_start = time.time()

                    rolling_corr = p.rolling_corr_vs_stocks(window_months=12, debug=False)
                    # Filter rolling corr to date range
                    if not rolling_corr.empty:
                        rolling_corr = rolling_corr.loc[str(user_start):str(user_end)]

                    timed_step(step_ph, "Computing correlations...", step_start)
                    
                    st.session_state["analyze_results"] = {
                        "p_name": getattr(p, "name", "Portfolio"),
                        "value_series": value_series,
                        "asset_growth": asset_growth,
                        "stats": stats,
                        "rolling_corr": rolling_corr,
                        "rebalance_frequency": rebalance_frequency,
                        "initial_amount": initial_amount,
                        "rf_annual": rf_annual,
                        "start_date": str(value_series.index[0].date()),
                        "end_date": str(value_series.index[-1].date()),
                    }
                    
                    total_elapsed = time.time() - total_start
                    status.update(label=f"Analysis complete! ({total_elapsed:.2f}s)", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="Analysis failed", state="error", expanded=False)
                    st.error(str(e))
                    st.session_state.pop("analyze_results", None)

        if "analyze_results" in st.session_state:
            res = st.session_state["analyze_results"]
            
            st.markdown(f"### Key Metrics")
            # Show data freshness and analysis period
            data_available, freshness = get_data_status()
            freshness_note = f" | Data: {freshness}" if data_available else ""
            st.caption(f"Period: {res['start_date']} to {res['end_date']} | Rebalance: {res['rebalance_frequency']} | RF: {res['rf_annual']:.1%}{freshness_note}")

            stats = res["stats"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{stats['total_return']*100:.2f}%")
            col2.metric("CAGR", f"{stats['cagr']*100:.2f}%")
            col3.metric("Volatility (ann.)", f"{stats['vol_annual']*100:.2f}%")
            col4.metric("Sharpe Ratio", f"{stats['sharpe']:.2f}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Max Drawdown", f"{stats['max_drawdown']*100:.2f}%")
            col6.metric("Longest Drawdown", _fmt_days(stats.get("longest_drawdown_days")))
            col7.metric("Ulcer Index", f"{stats['ulcer_index']:.2f}")
            col8.metric("Sortino Ratio", f"{stats['sortino']:.2f}")

            with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                st.markdown(
                    """
**Total Return** — The overall percentage gain (or loss) from start to end of the period.

**CAGR** — Compound Annual Growth Rate. The smoothed annual return that would produce the same total return if compounded each year.

**Volatility (ann.)** — Annualized standard deviation of returns. Higher values indicate larger price swings.

**Sharpe Ratio** — Risk-adjusted return: (Return - Risk-Free Rate) / Volatility. Higher is better; above 1.0 is generally considered good.

**Max Drawdown** — The largest peak-to-trough decline during the period. Shows worst-case loss if you bought at the peak.

**Longest Drawdown** — How long (in days) the portfolio took to recover from its worst drawdown.

**Ulcer Index** — Measures both depth and duration of drawdowns. Lower is better; it penalizes prolonged declines more than brief ones.

**Sortino Ratio** — Like Sharpe, but only penalizes downside volatility. Higher is better; useful when returns are asymmetric.

Note: Volatility, Sharpe, and Sortino here are computed from daily log returns and annualized using 252 trading days.
                    """
                )

            if show_chart:
                # Use consistent blue color for portfolio across all charts
                PORTFOLIO_COLOR = "#4e79a7"
                
                st.markdown("#### Portfolio Value")
                
                # Single portfolio - no legend needed, just plot the line
                v_series = res["value_series"]
                value_df = pd.DataFrame({"Date": v_series.index, "Value": v_series.values})
                
                value_chart = (
                    alt.Chart(value_df)
                    .mark_line(strokeWidth=2.0, color=PORTFOLIO_COLOR)
                    .encode(
                        x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y")),
                        y=alt.Y("Value:Q", title="Value (EUR)", scale=alt.Scale(type="log" if y_scale == "Logarithmic" else "linear")),
                        tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Value:Q", title="Value", format=",.2f")]
                    )
                    .properties(height=400)
                    .interactive()
                )
                st.altair_chart(value_chart, width="stretch")
                
                with st.expander("ℹ️ What does this chart show?", expanded=False):
                    st.markdown(
                        f"""
This shows the **growth of your portfolio value** over time, starting from €{res['initial_amount']:,.0f}. The line represents how your investment would have grown (or shrunk) by each date, assuming all dividends are reinvested and the portfolio is rebalanced at the specified frequency.
                        """
                    )

                # Trajectories - use short names, NO portfolio line
                st.markdown("#### Asset Trajectories")
                asset_growth = res["asset_growth"]
                short_map = get_short_name_map()
                
                # Build ticker->short name mapping for display
                ticker_to_short = {t: short_map.get(t, t) for t in asset_growth.columns}
                
                # Create fixed color mapping for assets (same colors in trajectories and correlation)
                # Use tableau10 palette colors
                TABLEAU10_COLORS = [
                    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"
                ]
                all_short_names = sorted(set(ticker_to_short.values()))
                asset_color_map = {name: TABLEAU10_COLORS[i % len(TABLEAU10_COLORS)] for i, name in enumerate(all_short_names)}
                
                # Multiselect using short names
                all_assets = list(asset_growth.columns)
                selected_assets = st.multiselect(
                    "Show assets:", 
                    options=all_assets, 
                    default=all_assets, 
                    key="analyze_traj_select",
                    format_func=lambda t: ticker_to_short.get(t, t)
                )
                
                if selected_assets:
                    traj_data = []
                    for asset in selected_assets:
                        s = asset_growth[asset]
                        short_name = ticker_to_short.get(asset, asset)
                        for date, val in s.items():
                            traj_data.append({"Date": date, "Asset": short_name, "Value": float(val)})
                    
                    traj_df = pd.DataFrame(traj_data)
                    
                    # Build color scale with fixed domain and range
                    selected_short_names = sorted(set(ticker_to_short[a] for a in selected_assets))
                    color_domain = selected_short_names
                    color_range = [asset_color_map[name] for name in selected_short_names]
                    
                    traj_chart = (
                        alt.Chart(traj_df)
                        .mark_line(strokeWidth=2.0)
                        .encode(
                            x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y")),
                            y=alt.Y("Value:Q", title="Value (EUR)", scale=alt.Scale(type="log" if y_scale == "Logarithmic" else "linear")),
                            color=alt.Color("Asset:N", scale=alt.Scale(domain=color_domain, range=color_range), title=None),
                            tooltip=["Date:T", "Asset:N", alt.Tooltip("Value:Q", format=",.2f")]
                        )
                        .properties(height=400)
                        .interactive()
                    )
                    st.altair_chart(traj_chart, width="stretch")
                    
                    with st.expander("ℹ️ What does this chart show?", expanded=False):
                        st.markdown(
                            f"""
This shows the **individual growth trajectories** of each asset in your portfolio, starting from €{res['initial_amount']:,.0f}. It helps you see which assets drove performance and how they moved relative to each other over time.
                            """
                        )

                # Rolling Correlation - use short names with same fixed colors as trajectories
                rolling = res["rolling_corr"]
                if not rolling.empty:
                    st.markdown("#### Rolling 12M Correlation vs Stocks")
                    
                    # Build reverse mapping: full_name -> short_name and ticker -> short_name
                    # The rolling columns are display_labels which could be "Full Name" or "Full Name (TICKER)"
                    full_to_short = {}
                    assets = load_available_assets()
                    for asset in assets:
                        ticker = asset.get("Ticker", "")
                        full_name = asset.get("Name", "")
                        short_name = asset.get("Short", full_name)
                        if ticker:
                            full_to_short[ticker] = short_name
                            full_to_short[full_name] = short_name
                            # Also handle "Name (Ticker)" format
                            full_to_short[f"{full_name} ({ticker})"] = short_name
                    
                    roll_data = []
                    for col in rolling.columns:
                        # Try to map column to short name
                        short_name = full_to_short.get(col, col)
                        for date, val in rolling[col].items():
                            if pd.isna(val):
                                continue
                            roll_data.append({"Date": date, "Asset": short_name, "Correlation": float(val)})
                    
                    roll_df = pd.DataFrame(roll_data)
                    
                    # Use same fixed color mapping as trajectories
                    roll_short_names = sorted(roll_df["Asset"].unique())
                    roll_color_domain = roll_short_names
                    roll_color_range = [asset_color_map.get(name, "#888888") for name in roll_short_names]
                    
                    roll_chart = (
                        alt.Chart(roll_df)
                        .mark_line(strokeWidth=3.0)
                        .encode(
                            x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y")),
                            y=alt.Y("Correlation:Q", title="Correlation"),
                            color=alt.Color("Asset:N", scale=alt.Scale(domain=roll_color_domain, range=roll_color_range), title=None),
                            tooltip=["Date:T", "Asset:N", alt.Tooltip("Correlation:Q", format=".2f")]
                        )
                        .properties(height=300)
                        .interactive()
                    )
                    st.altair_chart(roll_chart, width="stretch")
                    
                    with st.expander("ℹ️ What does this chart show?", expanded=False):
                        st.markdown(
                            """
This shows **rolling 12-month correlation** of each non-stock asset vs. stocks. Values range from -1 (moves opposite) to +1 (moves together). Lower or negative correlation provides better diversification during stock downturns.
                            """
                        )

                # Drawdown Chart - single portfolio, same color as value chart
                st.markdown("#### Portfolio Drawdown")
                dd = (v_series / v_series.cummax() - 1.0) * 100.0
                dd_df = pd.DataFrame({"Date": dd.index, "Drawdown (%)": dd.values})
                
                dd_chart = (
                    alt.Chart(dd_df)
                    .mark_line(strokeWidth=2.0, color=PORTFOLIO_COLOR)
                    .encode(
                        x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y")),
                        y=alt.Y("Drawdown (%):Q", title="Drawdown (%)", scale=alt.Scale(domain=[float(dd.min()) - 1, 0])),
                        tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Drawdown (%):Q", title="Drawdown", format=".2f")]
                    )
                    .properties(height=300)
                    .interactive()
                )
                zero_line = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#bbbbbb", strokeDash=[4, 4]).encode(y="y:Q")
                st.altair_chart((dd_chart + zero_line), width="stretch")
                
                with st.expander("ℹ️ What does this chart show?", expanded=False):
                    st.markdown(
                        """
This shows **drawdowns** (peak-to-trough declines) of your portfolio over time. Lower (more negative) values mean deeper declines from previous highs. The portfolio recovers when the line returns to 0%.
                        """
                    )

                # Rolling Returns & Volatility
                render_rolling_return_chart(
                    {"Portfolio": v_series},
                    portfolio_order=["Portfolio"],
                    color=PORTFOLIO_COLOR,
                    explainer_md="""
This shows the **rolling 12-month return** for your portfolio. Each point on the line represents the total return over the preceding 12 months.
                    """
                )
                render_rolling_volatility_chart(
                    {"Portfolio": v_series},
                    portfolio_order=["Portfolio"],
                    color=PORTFOLIO_COLOR,
                    explainer_md="""
This shows the **rolling 12-month annualized volatility** for your portfolio. Higher values indicate periods where the portfolio experienced larger daily price swings.
                    """
                )
