import time
import pandas as pd
import streamlit as st
import altair as alt

try:
    from rebalancing import (
        compute_rebalancing_diagnostics,
        build_llm_withdraw_report,
    )
except ImportError:
    import sys
    import os
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR = os.path.join(REPO_ROOT, "cache")
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from rebalancing import (
        compute_rebalancing_diagnostics,
        build_llm_withdraw_report,
    )

from ui.portfolio_builder import render_portfolio_builder
from ui.assets import cached_get_macro_snapshot, get_macro_chart_data, get_macro_trends, get_short_name_map, load_available_assets
from ui.components import timed_step, render_llm_query_ui
from ui.charts import render_deviation_chart, render_macro_chart

def render():
    with st.expander("ℹ️ About this section", expanded=False):
        st.markdown("""
This section helps you withdraw cash from your portfolio by **selling** existing assets.
It calculates the optimal sell allocation to bring your portfolio closer to target weights while raising the requested cash.
What you'll get:

- **Optimal Sell Plan**: Exact amounts to sell from each asset that minimizes the deviation from the target weights while raising the requested cash.
- **Portfolio Diagnostics**: Valuation and trend metrics (Z-score, EWMA distance) to help you decide if you want to deviate from the target by exploiting market opportunities.
- **Macro-Economic Overview**: Key macro indicators such as central banks rates, inflation, FX rates and associated trends, in order to provide more context. For example, you might want to avoid buying bonds during QE periods or you might want to overweight stocks after dot-com or GFC-like events.
- **AI Assistant**: A detailed prompt containing all the previous information that you can either copy and paste into your preferred LLM, or use directly within the app through the [OpenRouter API](https://openrouter.ai/).
In this case, only free tier models are available, so take the responses with a grain of salt. The goal is to help you reason about the trade-offs and make the best decision.

How to use: define your current weights and target weights (using the table, a built-in portfolio or upload your own), input the withdrawal amount and optional preferences, then click "Run withdrawal analysis".

**Tax considerations**: When withdrawing, consider which assets have unrealized capital gains. In Italy, capital gains are taxed at 26%, so selling assets with large gains triggers significant taxes. Add any tax or transaction cost constraint to the user preferences box for AI-assisted reasoning.

References for tactical allocation models:
* Cliff Asness, Antti Ilmanen, Thomas Maloney - [**Market Timing: Sin a Little**](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Market-Timing-Sin-a-Little.pdf)
* Mebane T. Faber - [**A Quantitative Approach to Tactical Asset Allocation**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461)
* Adam Butler, Mike Philbrick, Rodrigo Gordillo, David Varadi - [**Adaptive Asset Allocation: A Primer**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2328254)
* Diana Barro, Elio Canestrelli, Fabio Lanza - [**Volatility vs. Downside Risk: Optimally Protecting Against Drawdowns and Maintaining Portfolio Performance**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2521007)
        """)
    
    # Portfolio builder in "Table Mode" (allow_value=True)
    p, _ = render_portfolio_builder(key="withdraw", title="Portfolio & Targets", allow_value=True)
    
    if p is not None:
        current_value = getattr(p, "current_value_eur", 0.0) or 0.0
        max_withdraw = max(100.0, current_value - 100.0)  # Leave at least 100 EUR
        
        st.markdown("### Settings")
        withdraw_amount = st.number_input(
            "Cash to withdraw (EUR)",
            min_value=100.0,
            max_value=max_withdraw,
            value=min(10_000.0, max_withdraw),
            step=100.0,
            key="withdraw_amount",
            help=f"Maximum withdrawal: {max_withdraw:,.2f} EUR (must leave at least 100 EUR in portfolio)",
        )
        
        user_prefs = st.text_area(
            "User preferences (optional)",
            placeholder="Example 1: Asset X has a large unrealized gain, I want to avoid selling it.\nExample 2: Asset Y is at a loss, I want to sell it to harvest the tax loss.\nExample 3: I need to minimize the tax impact of this withdrawal.",
            key="withdraw_user_prefs",
            help="Add any constraints or preferences to include in the LLM prompt (e.g., assets to avoid selling due to capital gains).",
        )

        run = st.button("Run withdrawal analysis", type="primary", key="withdraw_run")
        
        if run:
            with st.status("Running withdrawal analysis...", expanded=True) as status:
                try:
                    total_start = time.time()
                    
                    step_ph = st.empty()
                    step_ph.write("Preparing data...")
                    step_start = time.time()
                    
                    # 1. Math optimal withdrawal
                    df_withdraw = p.withdraw(withdraw_amount=float(withdraw_amount))
                    
                    timed_step(step_ph, "Computing sell allocation...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Fetching macro data...")
                    step_start = time.time()
                    
                    # 2. Macro data
                    fred_api_key = st.secrets.get("FRED_API_KEY", "")
                    snap = cached_get_macro_snapshot(fred_api_key=fred_api_key)
                    
                    timed_step(step_ph, "Fetching macro data...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Computing diagnostics...")
                    step_start = time.time()
                    
                    # 3. Diagnostics
                    diag = compute_rebalancing_diagnostics(p)
                    diag_transposed = diag.T if not diag.empty else None
                    diag_error = None
                    if diag.empty:
                        diag_error = "Insufficient history."
                    
                    timed_step(step_ph, "Computing diagnostics...", step_start)
                    step_ph = st.empty()
                    step_ph.write("Generating LLM prompt...")
                    step_start = time.time()
                    
                    # 4. LLM Prompt
                    macro_trends = get_macro_trends()
                    llm_prompt = build_llm_withdraw_report(
                        portfolio=p,
                        withdraw_table=df_withdraw,
                        diagnostics_table=diag_transposed,
                        macro_snapshot=snap,
                        current_value=float(current_value),
                        withdraw_amount=float(withdraw_amount),
                        macro_trends=macro_trends,
                        user_preferences=user_prefs,
                    )
                    
                    # Prepare macro charts data from local macro series
                    macro_charts = {
                        "eu": get_macro_chart_data(["ECB Overnight (%)", "EU 10Y Yield (%)", "EU Inflation YoY (%)"]),
                        "us": get_macro_chart_data(["FED Overnight (%)", "US 10Y Yield (%)", "US Inflation YoY (%)"]),
                        "fx": get_macro_chart_data(["USD/EUR"]),
                        "earnings": get_macro_chart_data(["Global EY Est. (%)"]),
                    }

                    st.session_state["withdraw_results"] = {
                        "df_withdraw": df_withdraw,
                        "snap": snap,
                        "diag_transposed": diag_transposed,
                        "diag_error": diag_error,
                        "llm_prompt": llm_prompt,
                        "macro_charts": macro_charts,
                    }
                    st.session_state.pop("withdraw_llm_response", None)
                    
                    total_elapsed = time.time() - total_start
                    status.update(label=f"Withdrawal analysis complete! ({total_elapsed:.2f}s)", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="Withdrawal analysis failed", state="error", expanded=False)
                    st.error(str(e))
                    st.session_state.pop("withdraw_results", None)

        if "withdraw_results" in st.session_state:
            results = st.session_state["withdraw_results"]
            
            # Show data freshness
            from ui.assets import get_data_status
            data_available, freshness = get_data_status()
            if data_available:
                st.caption(f"Data: {freshness}")
            
            st.markdown("### Optimal Sell Plan")
            df_display = results["df_withdraw"].copy()
            
            # Map index (full names/tickers) to short names
            short_map = get_short_name_map()
            # Build reverse mapping from full name or "Name (Ticker)" format to short name
            assets = load_available_assets()
            label_to_short = {}
            for asset in assets:
                ticker = asset.get("Ticker", "")
                full_name = asset.get("Name", "")
                short_name = asset.get("Short", full_name)
                if ticker:
                    label_to_short[ticker] = short_name
                    label_to_short[full_name] = short_name
                    label_to_short[f"{full_name} ({ticker})"] = short_name
            
            # Apply short name mapping to index
            df_display.index = [label_to_short.get(idx, idx) for idx in df_display.index]
            
            # Formatting
            for col in ["Current Weight (%)", "Target Weight (%)", "New Weight (%)"]:
                df_display[col] = df_display[col].map("{:.2f}".format)
            df_display["Sell (EUR)"] = df_display["Sell (EUR)"].map("{:,.2f}".format)
            
            st.dataframe(df_display.astype(str), width="stretch")
            
            with st.expander("ℹ️ How to read this?", expanded=False):
                st.markdown("""
**Current Weight (%)** — Your current allocation to each asset based on current portfolio value.

**Target Weight (%)** — Your desired long-term allocation to each asset.

**Sell (EUR)** — The amount to sell from each asset to raise the requested cash while moving toward target weights.

**New Weight (%)** — Your projected allocation after selling as recommended.

The algorithm prioritizes selling overweight assets while ensuring no buying is required.
                """)
            
            # Deviation chart
            df_withdraw = results["df_withdraw"]
            wc = pd.to_numeric(df_withdraw["Current Weight (%)"], errors="coerce")
            wt = pd.to_numeric(df_withdraw["Target Weight (%)"], errors="coerce")
            deviation = wc - wt
            # Use short names for the chart
            short_names = [label_to_short.get(idx, idx) for idx in df_withdraw.index]
            deviation_data = pd.DataFrame({
                "Asset": short_names,
                "Deviation (%)": deviation.values
            })
            
            render_deviation_chart(deviation_data)
            
            with st.expander("ℹ️ How to read this?", expanded=False):
                st.markdown("""
Shows how far each asset deviates from its target weight. Red bars (positive) indicate overweight positions that should be prioritized for selling; green bars (negative) indicate underweight positions to avoid selling if possible. If you don't see any bars, it means all assets are already at their target weights.
                """)

            if results["diag_transposed"] is not None:
                st.markdown("### Portfolio Diagnostics")
                # Map column names (tickers/full names) to short names
                diag_display = results["diag_transposed"].copy()
                diag_display.columns = [label_to_short.get(col, col) for col in diag_display.columns]
                st.dataframe(diag_display.astype(str), width="stretch")
                
                with st.expander("ℹ️ What do these diagnostics mean?", expanded=False):
                    st.markdown("""
These diagnostics help you understand recent portfolio behavior over the last ~12 months.

**CAGR (12m, %)** — Compound Annual Growth Rate over the last 12 months. Shows how much each asset has returned recently. High CAGR may indicate large unrealized gains (potential tax liability if sold).

**EWMA Price Distance % (3m/6m/12m)** — How far the current price is from its exponentially-weighted moving average over 3, 6, or 12 months. Positive values indicate price is above recent average (momentum); negative values suggest price is below average (potential value opportunity or candidate for tax-loss selling).

**EWMA Volatility % (Annualized)** — Recent annualized volatility using exponential weighting, which gives more weight to recent price movements. Higher values indicate more recent price swings.

**Z-Score (12m, on prices)** — How many standard deviations the current price is from its 12-month mean. Values above +2 suggest potentially overbought (consider selling); below -2 suggest potentially oversold (avoid selling if possible).

**Correlation to Stocks (12m, monthly)** — How closely each asset moved with your stock allocation over the last 12 months using monthly returns. Values range from -1 (moves opposite) to +1 (moves together). 
                    """)
            elif results["diag_error"]:
                st.caption(f"Diagnostics unavailable: {results['diag_error']}")

            snap = results["snap"]
            if snap is not None:
                st.markdown("### Macro-Economic Overview")
                st.caption(f"Data as of: {snap.asof.date()}")
                
                def _fmt_pct(val: float | None) -> str:
                    return f"{val:.2f}%" if val is not None else "—"
                
                def _fmt_fx(val: float | None) -> str:
                    return f"{val:.4f}" if val is not None else "—"
                
                r1 = st.columns(4)
                r1[0].metric("ECB Overnight", _fmt_pct(snap.ecb_dfr_pct))
                r1[1].metric("EU 10Y Yield", _fmt_pct(snap.eu_10y_yield_pct))
                r1[2].metric("EU Inflation YoY", _fmt_pct(snap.eu_cpi_yoy_pct))
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
These indicators provide macro context (rates, inflation, FX, valuations) for withdrawal decisions.

**ECB Overnight** — Euro area policy rate; anchors short-term EUR rates and influences bond yields.

**EU 10Y Yield** — Long-term EUR "risk-free" proxy (German Bund yield). Higher yields raise the opportunity cost of holding equities.

**EU Inflation YoY** — Euro Area HICP year-over-year. Higher inflation can keep policy rates elevated.

**USD/EUR spot** — FX rate (USD per 1 EUR). Relevant if you hold USD assets unhedged.

**FED Overnight** — US overnight policy rate proxy; influences USD cash yields and discount rates.

**US 10Y Yield** — Key long-term USD rate. Higher yields can pressure equity valuations.

**US Inflation YoY** — US CPI year-over-year.

**Global Earnings Yield (est.)** — A simple valuation proxy for global equities (higher can imply "cheaper" equities vs bonds).

If the global estimate is unavailable (Yahoo fundamentals can be flaky on Streamlit Cloud), the app may fall back to a **US earnings-yield proxy from FRED**. When that happens, you'll see an explicit note in the UI.
                    """)

                charts = results.get("macro_charts", {})
                if charts:
                    st.markdown("#### Last 12M Trend")
                    render_macro_chart(
                        "EU Data",
                        charts.get("eu", []),
                        y_title="Percent (%)",
                        indicator_order=["ECB Overnight (%)", "EU 10Y Yield (%)", "EU Inflation YoY (%)"],
                        explainer_md=(
                            "- **ECB Overnight (%)**: Euro area policy rate proxy (deposit facility rate).\n"
                            "- **EU 10Y Yield (%)**: Long-term EUR rate proxy (German Bund yield).\n"
                            "- **EU Inflation YoY (%)**: Euro Area HICP year-over-year.\n\n"
                            "Use this to see whether EUR policy/long rates and inflation have been trending up or down."
                        ),
                    )
                    render_macro_chart(
                        "US Data",
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
                    render_macro_chart(
                        "USD/EUR",
                        charts.get("fx", []),
                        y_title="USD per 1 EUR",
                        indicator_order=["USD/EUR"],
                        explainer_md=(
                            "**USD/EUR** is the amount of USD per 1 EUR. If you hold USD-denominated assets (unhedged), FX moves can materially impact EUR returns."
                        ),
                    )
                    render_macro_chart(
                        "Global Earnings Yield (est.)",
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
                    key_prefix="withdraw",
                    llm_prompt=results["llm_prompt"],
                    title="", 
                )
