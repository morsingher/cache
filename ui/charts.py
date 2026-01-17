import altair as alt
import pandas as pd
import streamlit as st
import numpy as np

def _drawdown_series(value: pd.Series) -> pd.Series:
    if value is None or value.empty:
        return pd.Series(dtype=float)
    peak = value.cummax()
    return (value / peak) - 1.0

def render_portfolio_value_chart(
    chart_df: pd.DataFrame,
    *,
    portfolio_order: list[str] | None = None,
    y_scale: str = "Linear",
    height: int = 400
) -> None:
    y_scale_type = "log" if y_scale == "Logarithmic" else "linear"
    
    color_kwargs = {"title": None}  # No legend title, just show the labels
    if portfolio_order:
        color_kwargs["sort"] = portfolio_order

    chart = (
        alt.Chart(chart_df)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y")),
            y=alt.Y("Value:Q", title="Value (normalized)", scale=alt.Scale(type=y_scale_type)),
            color=alt.Color("Portfolio:N", **color_kwargs),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Portfolio:N", title="Name"),
                alt.Tooltip("Value:Q", title="Value", format=",.4f"),
            ],
        )
        .properties(height=height)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")

def render_drawdown_chart(
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

    if title:
        st.markdown(f"#### {title}")
        
    color_kwargs = {"title": None}  # No legend title, just show the labels
    if portfolio_order:
        color_kwargs["sort"] = portfolio_order

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X(
                "Date:T",
                title="Date",
                axis=alt.Axis(format="%m/%Y", labelPadding=6, titlePadding=10),
            ),
            y=alt.Y("Drawdown (%):Q", title="Drawdown (%)", scale=alt.Scale(domain=[dom_min, 0.0])),
            color=alt.Color("Portfolio:N", **color_kwargs),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Portfolio:N", title="Name"),
                alt.Tooltip("Drawdown (%):Q", title="Drawdown", format=".2f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#bbbbbb", strokeDash=[4, 4]).encode(y="y:Q")
    layered = (
        alt.layer(chart, zero)
        .properties(height=height, padding={"bottom": 20})
        .interactive()
    )
    st.altair_chart(layered, width="stretch")

    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown(
            """
This compares **drawdowns** (peak-to-trough declines) across portfolios. Lower (more negative) values mean deeper declines from previous highs. Portfolios that recover faster will show shorter “underwater” stretches.
            """
        )

def render_rolling_return_chart(
    value_series_by_name: dict[str, pd.Series],
    *,
    window_days: int = 252,
    title: str = "Rolling 12M Returns",
    height: int = 320,
    portfolio_order: list[str] | None = None,
    color: str | None = None,
    explainer_md: str | None = None,
) -> None:
    rows: list[dict[str, object]] = []
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
        # pct_change(252) approximates 12-month return for daily data
        rr = v.pct_change(periods=window_days)
        if rr.empty:
            continue
        rr_pct = rr * 100.0
        # Drop initial NaNs
        rr_pct = rr_pct.dropna()
        for dt, val in rr_pct.items():
            rows.append({"Date": dt, "Portfolio": str(name), "Return (%)": float(val)})

    if not rows:
        st.info("Rolling returns unavailable (insufficient history).")
        return

    df = pd.DataFrame(rows)

    if title:
        st.markdown(f"#### {title}")

    if color:
        chart = alt.Chart(df).mark_line(strokeWidth=2.0, color=color)
    else:
        chart = alt.Chart(df).mark_line(strokeWidth=2.0)

    encode_kwargs = {
        "x": alt.X(
            "Date:T",
            title="Date",
            axis=alt.Axis(format="%m/%Y", labelPadding=6, titlePadding=10),
        ),
        "y": alt.Y("Return (%):Q", title="Rolling Return (%)"),
        "tooltip": [
            alt.Tooltip("Date:T", title="Date"),
            alt.Tooltip("Portfolio:N", title="Name"),
            alt.Tooltip("Return (%):Q", title="Return", format=".2f"),
        ],
    }

    if not color:
        color_kwargs = {"title": None}
        if portfolio_order:
            color_kwargs["sort"] = portfolio_order
        encode_kwargs["color"] = alt.Color("Portfolio:N", **color_kwargs)

    chart = chart.encode(**encode_kwargs)

    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#bbbbbb", strokeDash=[4, 4]).encode(y="y:Q")
    layered = (
        alt.layer(chart, zero)
        .properties(height=height, padding={"bottom": 20})
        .interactive()
    )
    st.altair_chart(layered, width="stretch")

    default_explainer = """
This shows the **rolling 12-month return** for each portfolio. Each point on the line represents the total return over the preceding 12 months.
    """
    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown(explainer_md if explainer_md else default_explainer)

def render_rolling_volatility_chart(
    value_series_by_name: dict[str, pd.Series],
    *,
    window_days: int = 252,
    title: str = "Rolling 12M Volatility",
    height: int = 320,
    portfolio_order: list[str] | None = None,
    color: str | None = None,
    explainer_md: str | None = None,
) -> None:
    rows: list[dict[str, object]] = []
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
        # Rolling annualized volatility
        r = np.log(v / v.shift(1))
        vol = r.rolling(window=window_days).std() * np.sqrt(252)
        if vol.empty:
            continue
        vol_pct = vol * 100.0
        vol_pct = vol_pct.dropna()
        for dt, val in vol_pct.items():
            rows.append({"Date": dt, "Portfolio": str(name), "Volatility (%)": float(val)})

    if not rows:
        st.info("Rolling volatility unavailable (insufficient history).")
        return

    df = pd.DataFrame(rows)

    if title:
        st.markdown(f"#### {title}")

    if color:
        chart = alt.Chart(df).mark_line(strokeWidth=3.0, color=color)
    else:
        chart = alt.Chart(df).mark_line(strokeWidth=3.0)

    encode_kwargs = {
        "x": alt.X(
            "Date:T",
            title="Date",
            axis=alt.Axis(format="%m/%Y", labelPadding=6, titlePadding=10),
        ),
        "y": alt.Y("Volatility (%):Q", title="Rolling Volatility (%)"),
        "tooltip": [
            alt.Tooltip("Date:T", title="Date"),
            alt.Tooltip("Portfolio:N", title="Name"),
            alt.Tooltip("Volatility (%):Q", title="Volatility", format=".2f"),
        ],
    }

    if not color:
        color_kwargs = {"title": None}
        if portfolio_order:
            color_kwargs["sort"] = portfolio_order
        encode_kwargs["color"] = alt.Color("Portfolio:N", **color_kwargs)

    chart = chart.encode(**encode_kwargs)

    st.altair_chart(chart.properties(height=height, padding={"bottom": 20}).interactive(), width="stretch")

    default_explainer = """
This shows the **rolling 12-month annualized volatility** for each portfolio. Higher values indicate periods where the portfolio experienced larger daily price swings.
    """
    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown(explainer_md if explainer_md else default_explainer)

def render_deviation_chart(deviation_data: pd.DataFrame) -> None:
    st.markdown("#### Deviation from Target")
    
    # Sort by deviation descending for better readability
    sorted_assets = deviation_data.sort_values("Deviation (%)", ascending=True)["Asset"].tolist()
    
    deviation_chart = (
        alt.Chart(deviation_data)
        .mark_bar()
        .encode(
            y=alt.Y("Asset:N", title=None, sort=sorted_assets, 
                    axis=alt.Axis(labelLimit=0)),  # Ensure all labels are shown
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
        .properties(height=max(200, len(deviation_data) * 35))
    )
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="#888888", strokeDash=[4, 4]
    ).encode(x="x:Q")
    st.altair_chart((deviation_chart + zero_line), width="stretch")
    st.caption("🔴 Overweight | 🟢 Underweight")

def render_macro_chart(
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
        
    color_kwargs = {"title": None}
    if indicator_order:
        color_kwargs["sort"] = indicator_order
        
    chart = (
        alt.Chart(dfc)
        .mark_line(strokeWidth=3.0)
        .encode(
            x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%m/%Y")),
            y=alt.Y("Value:Q", title=y_title),
            color=alt.Color("Indicator:N", **color_kwargs),
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
