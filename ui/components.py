import json
import time
import random
import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

try:
    from fredapi import Fred
except ImportError:
    Fred = None


def fmt_days(x: float) -> str:
    """
    Format a number of days as a human-readable string (e.g., "123d").
    
    Args:
        x: Number of days (can be float or NaN).
        
    Returns:
        Formatted string like "123d" or "nan" if invalid.
    """
    try:
        v = float(x)
    except Exception:
        return "nan"
    if pd.isna(v):
        return "nan"
    return f"{int(round(v))}d"

# Imports for LLM
try:
    from openrouter import fetch_free_models, chat_completion_messages, get_api_key
except ImportError:
    import sys
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE_DIR = os.path.join(REPO_ROOT, "cache")
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from openrouter import fetch_free_models, chat_completion_messages, get_api_key

class StepTimer:
    """Helper class for timing steps within a st.status() context."""
    
    def __init__(self, status_container):
        self.status = status_container
        self.start_time = time.time()
        self.step_start = None
        self.step_placeholder = None
    
    def step(self, label: str) -> None:
        """Start a new step, showing progress message."""
        if self.step_placeholder is not None and self.step_start is not None:
            pass
        
        self.step_start = time.time()
        self.step_placeholder = st.empty()
        self.step_placeholder.write(f"{label}...")
    
    def done(self) -> None:
        """Mark current step as done with timing."""
        if self.step_placeholder is not None and self.step_start is not None:
            elapsed = time.time() - self.step_start
            self.step_placeholder.write(f"{self._get_current_label()} Done! ({elapsed:.2f}s)")
            self.step_placeholder = None
            self.step_start = None
    
    def _get_current_label(self) -> str:
        return ""
    
    def total_time(self) -> float:
        return time.time() - self.start_time


def timed_step(placeholder, label: str, start_time: float) -> None:
    """Update a placeholder with completed step message and timing."""
    elapsed = time.time() - start_time
    placeholder.write(f"{label} Done! ({elapsed:.2f}s)")


def render_copy_button(prompt: str, *, key: str, label: str = "Copy prompt") -> None:
    html = f"""
    <div>
      <style>
        #copy-btn-{key} {{
          padding: 0.4rem 0.85rem;
          border-radius: 0.5rem;
          border: 1px solid #d6d3c7;
          background: #ecebe3;
          color: #2f2f2f;
          font-weight: 600;
          font-size: 0.95rem;
          cursor: pointer;
          transition: background-color 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
        }}
        #copy-btn-{key}:hover {{
          background: #e5e3da;
          border-color: #cfcab7;
        }}
        #copy-btn-{key}:active {{
          background: #ddd9cc;
          border-color: #c4bea9;
        }}
      </style>
      <button id="copy-btn-{key}">
        {label}
      </button>
      <span id="copy-status-{key}" style="margin-left: 0.5rem; color: #6b7280; font-size: 0.9em;"></span>
    </div>
    <script>
      const text = {json.dumps(prompt)};
      const btn = document.getElementById("copy-btn-{key}");
      const status = document.getElementById("copy-status-{key}");
      btn.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(text);
        }} catch (e) {{
          const ta = document.createElement("textarea");
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
        }}
        status.textContent = "Copied";
        setTimeout(() => status.textContent = "", 1500);
      }});
    </script>
    """
    components.html(html, height=40)

@st.cache_data(ttl=3600, show_spinner="Fetching FRED data (cached)...")
def _cached_latest_fred_pct(_series_id: str, _api_key: str) -> float | None:
    if Fred is None:
        return None
    try:
        fred = Fred(api_key=_api_key)
    except Exception:
        return None
        
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            s = fred.get_series(_series_id)
            if s is None:
                raise RuntimeError("FRED returned None")
            s = s.dropna()
            if len(s) == 0:
                raise RuntimeError("FRED returned empty series")
            return float(s.iloc[-1])
        except Exception as e:
            last_exc = e
            if attempt < 5:
                delay = 0.7 * (2 ** attempt)
                delay *= (1.0 + random.uniform(-0.15, 0.15))
                time.sleep(delay)
    return None

def rf_annual_controls(*, key_prefix: str) -> float:
    """Risk-free rate control, defaults to ECB overnight rate from local database."""
    from ui.assets import data_exists
    from cache.datastore import load_macro_snapshot
    
    mode = st.radio(
        "Risk-free rate",
        options=["ECB Overnight (default)", "Manual"],
        index=0,
        horizontal=True,
        key=f"{key_prefix}_rf_mode",
        help="Return without risk in EU, default is ECB overnight rate.",
    )

    if mode == "Manual":
        rf = st.number_input(
            "Custom value (annual, decimal)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.005,
            key=f"{key_prefix}_rf_manual",
        )
        return float(rf)

    # Try to load from local database first (fast, no network)
    if data_exists():
        local_macro = load_macro_snapshot()
        if local_macro is not None:
            ecb_rate = local_macro.get("ecb_dfr_pct")
            if ecb_rate is not None:
                st.caption(f"ECB overnight: {ecb_rate:.2f}%")
                return float(ecb_rate / 100.0)
    
    # Fallback to FRED if local data unavailable
    if Fred is None:
        st.warning("Local data unavailable and fredapi not installed; using 0%.")
        return 0.0

    api_key = st.secrets.get("FRED_API_KEY", "").strip()
    if not api_key:
        st.warning("Local data unavailable. Set `FRED_API_KEY` to fetch from FRED.")
        return 0.0

    try:
        latest_pct = _cached_latest_fred_pct("ECBDFR", api_key)
        if latest_pct is None:
            st.warning("FRED series returned no data; using 0%.")
            return 0.0
        st.caption(f"ECB overnight (FRED): {latest_pct:.3f}%")
        return float(latest_pct / 100.0)
    except Exception as e:
        msg = str(e).strip()
        details = f" ({msg})" if msg and msg.lower() != "none" else ""
        st.warning(f"FRED fetch failed; using 0%.{details}")
        return 0.0

def backtest_controls(*, key_prefix: str, show_initial_amount: bool = True) -> tuple[str, float, float]:
    """
    Returns: (rebalance_frequency, initial_amount, rf_annual)
    """
    c1, c2 = st.columns([1, 1])
    with c1:
        rebalance_display = st.radio(
            "Rebalance frequency",
            options=["Monthly", "Quarterly", "Annually"],
            index=2,
            horizontal=True,
            key=f"{key_prefix}_reb_freq",
        )
        rebalance_frequency = str(rebalance_display).lower()
    with c2:
        initial_amount = 10_000.0
        if show_initial_amount:
            initial_amount = st.number_input(
                "Initial amount (EUR)",
                min_value=100.0,
                value=10_000.0,
                step=500.0,
                key=f"{key_prefix}_initial_amt",
            )

    rf_annual = rf_annual_controls(key_prefix=key_prefix)

    return str(rebalance_frequency), float(initial_amount), float(rf_annual)

def render_llm_query_ui(
    *,
    key_prefix: str,
    llm_prompt: str,
    title: str = "Ask an LLM",
) -> None:
    """
    Render a reusable interactive LLM chat UI component.
    """
    if title:
        st.markdown(f"### {title}")
    
    # Check API key
    api_key = get_api_key()
    if not api_key:
        st.error(
            "**OPENROUTER_API_KEY** environment variable is not set. "
            "Please set it to use LLM features."
        )
        return
    
    # Session state keys for this chat instance
    chat_history_key = f"{key_prefix}_chat_history"
    chat_started_key = f"{key_prefix}_chat_started"
    chat_model_key = f"{key_prefix}_chat_model"
    chat_error_key = f"{key_prefix}_chat_error"
    
    cache_key = f"{key_prefix}_free_models_cache"
    if cache_key not in st.session_state:
        with st.spinner("Loading available models..."):
            st.session_state[cache_key] = fetch_free_models(api_key)
    
    free_models = st.session_state[cache_key]
    
    default_model = "meta-llama/llama-3.3-70b-instruct:free"
    default_idx = 0
    if default_model in free_models:
        default_idx = free_models.index(default_model)
    
    chat_started = st.session_state.get(chat_started_key, False)
    
    selected_model = st.selectbox(
        "Select model (free tier only)",
        options=free_models,
        index=default_idx,
        key=f"{key_prefix}_model_select",
        help="All listed models are free to use. Some may have daily usage limits.",
        disabled=chat_started,
    )
    
    if not chat_started:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Start Chat", type="primary", key=f"{key_prefix}_start_chat_btn"):
                st.session_state[chat_history_key] = [
                    {"role": "user", "content": llm_prompt}
                ]
                st.session_state[chat_started_key] = True
                st.session_state[chat_model_key] = selected_model
                st.rerun()
        with col2:
            st.caption("Click to start an interactive conversation with an AI assistant. Responses may take a few seconds.")
        return
    
    SYSTEM_MESSAGE = {
        "role": "system",
        "content": (
            "You are a helpful financial assistant. Follow these rules strictly:\n"
            "1. NEVER use markdown headers (no #, ##, ###, etc.). Use plain text only.\n"
            "2. You may use bullet points, numbered lists, and bold/italic for emphasis.\n"
            "3. Be concise and to the point. Avoid verbosity while remaining complete and clear.\n"
            "4. Get straight to the actionable advice or answer."
        ),
    }
    
    chat_history: list[dict[str, str]] = st.session_state.get(chat_history_key, [])
    model = st.session_state.get(chat_model_key, selected_model)
    last_error: str | None = st.session_state.get(chat_error_key, None)
    
    needs_response = (
        len(chat_history) > 0 
        and chat_history[-1]["role"] == "user" 
        and last_error is None
    )
    
    st.markdown("#### Conversation")
    
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(chat_history):
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                with st.chat_message("user"):
                    if i == 0:
                        with st.expander("📋 Analysis Prompt (click to expand)", expanded=False):
                            st.markdown(content)
                    else:
                        st.markdown(content)
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(content)
        
        if last_error is not None:
            with st.chat_message("assistant"):
                st.error(f"**Request failed:** {last_error}")
                st.caption("You can retry the request or type a different message below.")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🔄 Retry", key=f"{key_prefix}_retry_btn", type="primary"):
                        st.session_state[chat_error_key] = None
                        st.rerun()
        
        elif needs_response:
            with st.chat_message("assistant"):
                with st.spinner("Thinking (may take a few seconds)..."):
                    messages_to_send = [SYSTEM_MESSAGE] + chat_history
                    
                    response = chat_completion_messages(
                        messages=messages_to_send,
                        model=str(model),
                        api_key=api_key,
                        max_tokens=4000,
                        temperature=0.7,
                    )
                    
                    if response.error:
                        st.session_state[chat_error_key] = response.error
                    else:
                        st.session_state[chat_error_key] = None
                        chat_history.append({"role": "assistant", "content": response.content})
                        st.session_state[chat_history_key] = chat_history
                    st.rerun()
    
    user_input = st.chat_input("Type your follow-up question...", key=f"{key_prefix}_chat_input")
    
    if user_input:
        if chat_error_key in st.session_state:
            del st.session_state[chat_error_key]
        chat_history.append({"role": "user", "content": user_input})
        st.session_state[chat_history_key] = chat_history
        st.rerun()
    
    st.markdown("")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Reset Chat", key=f"{key_prefix}_reset_chat_btn"):
            for key in [chat_history_key, chat_started_key, chat_model_key, chat_error_key]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col2:
        st.caption("Start a new conversation with the analysis prompt.")
