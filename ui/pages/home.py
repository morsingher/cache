import streamlit as st

def render():
    # Apply larger styling for homepage elements only
    st.markdown("""
        <style>
        /* Larger buttons on homepage only */
        [data-testid="stVerticalBlock"] [data-testid="stButton"] button {
            padding: 1rem 1.5rem !important;
            min-height: 3.5rem !important;
        }
        [data-testid="stVerticalBlock"] [data-testid="stButton"] button p {
            font-size: 1.3rem !important;
            font-weight: 500 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Reserve space on both sides so the navigation buttons stay centered under the title.
    left, mid, right = st.columns([2.2, 3.6, 2.2])
    left.empty()
    right.empty()
    with mid:
        # REBRANDING: Commented out analyze and compare - keeping code for future use
        # if st.button("Analyze a Portfolio", type="primary", key="nav_analyze", use_container_width=True):
        #     st.session_state["page"] = "analyze"
        #     st.rerun()
        # if st.button("Compare Portfolios", type="primary", key="nav_compare", use_container_width=True):
        #     st.session_state["page"] = "compare"
        #     st.rerun()
        if st.button("Deploy Cash", type="primary", key="nav_rebalance", use_container_width=True):
            st.session_state["page"] = "rebalance"
            st.rerun()
        if st.button("Withdraw Cash", type="primary", key="nav_withdraw", use_container_width=True):
            st.session_state["page"] = "withdraw"
            st.rerun()
        if st.button("Add a New Asset", type="primary", key="nav_whatif", use_container_width=True):
            st.session_state["page"] = "whatif"
            st.rerun()
    
    # Global help section
    st.markdown("")
    with st.expander("ℹ️ FAQs", expanded=False):
        st.markdown("""
**What can this app do?**

This app can help you manage your portfolio in 3 ways:
- Deploy cash into it, by considering deviations from target weights, macroeconomic conditions, trends, transaction costs and tax optimizations. An AI will assist you in the process.
- Withdraw cash from it, by considering deviations from target weights, macroeconomic conditions, trends, transaction costs and tax optimizations. An AI will assist you in the process.
- Evaluate the impact of adding new assets in terms of diversification and risk-adjusted returns. An AI will assist you in the process.

**Who is this app intended for?**

This app is intended for long-term, passive investors with EUR-denominated portfolios built with ETFs. 
If you are interested in active management, short-term operations or individual securities, this app is not for you.

**Which assets are available and what is the data source?**

This app supports a wide range of ETFs covering virtually any existing asset class. A complete list with additional information is available within each section.
Historical prices are downloaded from Yahoo Finance using their [free API](https://ranaroussi.github.io/yfinance/reference/index.html), while macroeconomic data is downloaded from [FRED](https://fred.stlouisfed.org/).

**How does the AI assist you?**

In order to optimize multi-objective problems such as rebalancing and what-if scenarios, this app uses a combination of statistical techniques and LLM-assisted reasoning.
In practice, a very detailed prompt is built from analysis results and user preferences, which can be either manually copied and pasted by the user into
their preferred LLM, or used directly within the app through the [OpenRouter API](https://openrouter.ai/).
In this case, only free tier models are available, so take the responses with a grain of salt.

**Why is it called CACH€?**

The name is an intended pun on the word "cache", which is pronounced as "cash", but has a widespread use in computer science.

**I found a bug / I have a feature request / I want to contribute**

Please [open an issue](https://github.com/morsingher/cache/issues) on the GitHub repository or fill this [Google Form](https://forms.gle/ATuhP6zS3LgqMy9k9).
""")
