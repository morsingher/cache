import os
import sys
import base64
import streamlit as st

# --- Setup Paths ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")
if CACHE_DIR not in sys.path:
    sys.path.insert(0, CACHE_DIR)

# --- Imports ---
from ui.styles import apply_custom_css
# REBRANDING: Commented out analyze and compare - keeping code for future use
from ui.pages import home, rebalance, withdraw, whatif
# from ui.pages import analyze, compare
from ui.assets import cached_price_date_ranges, load_available_assets, get_data_status

# --- Configuration ---
st.set_page_config(page_title="CACH€", page_icon="€", layout="centered")
apply_custom_css()

# --- Preload Assets ---
# With local data store, this is now instant (no API calls).
# Falls back to API if local data not available.
@st.cache_data(ttl=3600, show_spinner=False)
def _preload_all_assets() -> bool:
    """Preload all asset price data at app startup (instant with local data store)."""
    assets = load_available_assets()
    tickers = [str(a.get("Ticker", "")).strip() for a in assets if str(a.get("Ticker", "")).strip()]
    if tickers:
        cached_price_date_ranges(tuple(sorted(set(tickers))))
    return True

# --- Navigation State ---
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# Handle query params for navigation
if "go" in st.query_params:
    go = st.query_params["go"]
    if go == "home":
        # Clear results when going home
        keys_to_clear = [
                    # REBRANDING: Commented out analyze and compare - keeping code for future use
                    # "analyze_results",
                    # "compare_results",
                    "rebalance_results",
                    "withdraw_results",
                    "whatif_results",
                    "rebalance_chat_history",
                    "rebalance_chat_started",
                    "withdraw_chat_history",
                    "withdraw_chat_started",
                    "whatif_chat_history",
                    "whatif_chat_started",
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.session_state["page"] = "home"
    st.query_params.clear()

def render_header():
    # Load logo as base64
    logo_path = os.path.join(REPO_ROOT, "logo.png")
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    
    # Custom CSS for header styling
    st.markdown(
        """
        <style>
        .cache-logo {
            width: 250px !important;
            max-width: 250px !important;
            height: auto !important;
            margin-bottom: 0.5rem;
        }
        .cache-subtitle {
            font-size: 1.8rem !important;
            margin-top: 0 !important;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Center-aligned logo with home link
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="?go=home" target="_self">
                <img src="data:image/png;base64,{logo_base64}" class="cache-logo" style="width: 250px; max-width: 250px;" alt="CACH€">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Center-aligned subtitle
    st.markdown(
        '<h3 class="cache-subtitle">Your Rebalancing Assistant</h3>',
        unsafe_allow_html=True,
    )
    # Only show warning if local data is not available (not freshness)
    data_available, freshness = get_data_status()
    if not data_available:
        st.warning(freshness)

def render_navigation_back():
    if st.session_state["page"] != "home":
        c1, c2 = st.columns([1.5, 8.5])
        with c1:
            if st.button("← Back", type="secondary", key="nav_back", use_container_width=True):
                st.session_state["page"] = "home"
                # Clear analysis results on back
                keys_to_clear = [
                    # REBRANDING: Commented out analyze and compare - keeping code for future use
                    # "analyze_results", 
                    # "compare_results", 
                    "rebalance_results",
                    "withdraw_results",
                    "whatif_results"
                ]
                for k in keys_to_clear:
                    st.session_state.pop(k, None)
                st.rerun()
        st.markdown("")

# --- Main App Logic ---
render_header()
_preload_all_assets()
render_navigation_back()

page = st.session_state["page"]

if page == "home":
    home.render()
# REBRANDING: Commented out analyze and compare - keeping code for future use
# elif page == "analyze":
#     analyze.render()
# elif page == "compare":
#     compare.render()
elif page == "rebalance":
    rebalance.render()
elif page == "withdraw":
    withdraw.render()
elif page == "whatif":
    whatif.render()
else:
    st.error(f"Unknown page: {page}")
