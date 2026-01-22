import os
import json
import pandas as pd
import streamlit as st
import numpy as np
from typing import Any, Tuple, Optional

# Adjust imports based on new structure
try:
    from ui.assets import (
        get_asset_options,
        get_short_name_map,
        presort_multiselect_state,
        render_asset_help_dropdown,
        validate_portfolio_json_obj,
        get_prices_and_store,
        retry_on_rate_limit,
        load_available_assets,
    )
except ImportError:
    # Fallback during development
    from .assets import (
        get_asset_options,
        get_short_name_map,
        presort_multiselect_state,
        render_asset_help_dropdown,
        validate_portfolio_json_obj,
        get_prices_and_store,
        retry_on_rate_limit,
        load_available_assets,
    )

try:
    from portfolio import Portfolio
except ImportError:
    import sys
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE_DIR = os.path.join(REPO_ROOT, "cache")
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from portfolio import Portfolio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")


@st.cache_data(ttl=3600, show_spinner=False)
def load_available_portfolios() -> list[dict[str, str]]:
    """Load available built-in portfolios with their metadata."""
    portfolios_dir = os.path.join(CACHE_DIR, "portfolios")
    portfolios = []
    try:
        for fname in sorted(os.listdir(portfolios_dir)):
            if fname.lower().endswith(".json"):
                path = os.path.join(portfolios_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    portfolios.append({
                        "Name": obj.get("Name", fname.replace(".json", "")),
                        "Description": obj.get("Description", ""),
                        "Link": obj.get("Link", ""),
                        "Path": path,
                    })
                except Exception:
                    portfolios.append({
                        "Name": fname.replace(".json", ""),
                        "Description": "",
                        "Link": "",
                        "Path": path,
                    })
    except Exception:
        pass
    return portfolios


def render_portfolio_help_dropdown() -> None:
    """Render a help dropdown explaining built-in portfolios."""
    portfolios = load_available_portfolios()
    if not portfolios:
        return

    # Only show if at least one portfolio has a description
    if not any(p.get("Description") for p in portfolios):
        return

    with st.expander("💡 Need help choosing a portfolio?", expanded=False):
        for portfolio in portfolios:
            name = portfolio.get("Name", "")
            description = portfolio.get("Description", "")
            link = portfolio.get("Link", "")
            
            if not name:
                continue
            
            # Skip portfolios without descriptions
            if not description:
                st.markdown(f"**{name}**: *No description available.*")
                continue

            if link:
                name_link = f'<a href="{link}" target="_blank" style="color: #0066cc; text-decoration: underline;">{name}</a>'
                st.markdown(
                    f"**{name_link}**: {description}",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"**{name}**: {description}")


@st.cache_resource(ttl=3600, show_spinner="Loading portfolio (cached prices)...")
def cached_load_portfolio(path: str) -> Portfolio:
    """
    Cache the entire Portfolio object (including downloaded prices).
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return retry_on_rate_limit(portfolio_from_json_obj_with_cache, obj, source=path)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_portfolio_name(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        name = str(obj.get("Name", "")).strip()
        return name if name else os.path.basename(path)
    except Exception:
        return os.path.basename(path)

def portfolio_from_json_obj_with_cache(obj: dict[str, Any], *, source: str) -> Portfolio:
    assets_list = obj.get("Assets")
    if not isinstance(assets_list, list) or not assets_list:
        raise ValueError(f"Invalid portfolio json: missing/non-list 'Assets' in {source}")
    tickers = [str(a.get("Ticker", "")).strip() for a in assets_list if str(a.get("Ticker", "")).strip()]
    prices = get_prices_and_store(tuple(sorted(set(tickers)))) if tickers else pd.DataFrame()
    return Portfolio.from_dict(obj, source=source, prices=prices)

def _build_portfolio_from_manual(
    *,
    portfolio_name: str,
    df: pd.DataFrame,
    value_eur: float | None,
    normalize: bool,
) -> Portfolio:
    data = df.copy()
    data = data.replace({np.nan: None})

    data["Asset Name"] = data["Asset Name"].astype(str).str.strip()
    data["Ticker"] = data["Ticker"].astype(str).str.strip()
    data["Weight (%)"] = pd.to_numeric(data["Weight (%)"], errors="coerce")
    
    has_target_col = "Target (%)" in data.columns
    if has_target_col:
        data["Target (%)"] = pd.to_numeric(data["Target (%)"], errors="coerce")
        data = data.dropna(subset=["Ticker", "Weight (%)", "Target (%)"], how="any")
    else:
        data = data.dropna(subset=["Ticker", "Weight (%)"], how="any")
        data["Target (%)"] = data["Weight (%)"]
    
    data = data[data["Ticker"].astype(str).str.len() > 0]
    if data.empty:
        raise ValueError("No valid rows. Provide at least one asset with Ticker/Weight.")

    tickers = data["Ticker"].tolist()
    if len(set(tickers)) != len(tickers):
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        raise ValueError(f"Duplicate tickers not allowed: {dupes}")

    w_sum = float(np.nansum(data["Weight (%)"].to_numpy(dtype=float)))
    t_sum = float(np.nansum(data["Target (%)"].to_numpy(dtype=float)))
    if normalize:
        if w_sum <= 0 or t_sum <= 0:
            raise ValueError("Cannot normalize: weight sums must be > 0.")
        data["Weight (%)"] = data["Weight (%)"] * (100.0 / w_sum)
        data["Target (%)"] = data["Target (%)"] * (100.0 / t_sum)
        w_sum, t_sum = 100.0, 100.0

    tol = 0.25
    if abs(w_sum - 100.0) > tol:
        raise ValueError(f"Weights must sum to 100 (±{tol}). Got {w_sum:.4f}.")
    if has_target_col and abs(t_sum - 100.0) > tol:
        raise ValueError(f"Target weights must sum to 100 (±{tol}). Got {t_sum:.4f}.")

    assets = data["Asset Name"].tolist()
    weights = data["Weight (%)"].to_list()
    targets = data["Target (%)"].to_list()

    prices = get_prices_and_store(tuple(sorted(set(tickers)))) if tickers else pd.DataFrame()
    p = Portfolio(tickers=tickers, weights=weights, assets=assets, prices=prices)
    p.name = str(portfolio_name).strip() or "Portfolio"
    p.current_value_eur = float(value_eur) if value_eur is not None else None
    p.actual_weights_pct = {t: float(w) for t, w in zip(tickers, weights)}
    p.target_weights_pct = {t: float(w) for t, w in zip(tickers, targets)}
    return p

def _portfolio_json_from_manual(
    *,
    portfolio_name: str,
    df: pd.DataFrame,
    value_eur: float | None,
) -> dict[str, Any]:
    short_map = get_short_name_map()
    assets: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        weight = round(float(row["Weight (%)"]), 2)
        target = round(float(row["Target (%)"]), 2) if "Target (%)" in row.index else weight
        ticker = str(row["Ticker"]).strip()
        name = str(row["Asset Name"]).strip()
        short = short_map.get(ticker, name)
        assets.append(
            {
                "Name": name,
                "Ticker": ticker,
                "Short": short,
                "Weight": weight,
                "Target": target,
            }
        )
    obj: dict[str, Any] = {
        "Name": str(portfolio_name).strip() or "Portfolio",
        "Description": "",
        "Link": "",
        "Assets": assets,
    }
    obj["Value"] = round(float(value_eur), 2) if value_eur is not None else 100_000.0
    return obj

def render_example_json_ui(*, key_prefix: str, show_dropdown: bool = True) -> None:
    with st.expander("📄 Example JSON (click to view)", expanded=False):
        portfolios_dir = os.path.join(CACHE_DIR, "portfolios")
        example_paths: list[str] = []
        try:
            for fname in sorted(os.listdir(portfolios_dir)):
                if fname.lower().endswith(".json"):
                    example_paths.append(os.path.join(portfolios_dir, fname))
        except Exception:
            example_paths = []

        assets_list = load_available_assets()
        assets_by_name = sorted(assets_list, key=lambda a: str(a.get("Name", "")).lower())

        def _find_by_name(name: str) -> dict[str, str] | None:
            for a in assets_list:
                if str(a.get("Name", "")).strip().lower() == name.strip().lower():
                    return a
            return None

        a_stocks = _find_by_name("Stocks")
        a_bonds = _find_by_name("Bonds")
        if a_stocks and a_bonds:
            a1, a2 = a_stocks, a_bonds
        elif len(assets_by_name) >= 2:
            a1, a2 = assets_by_name[0], assets_by_name[1]
        else:
            a1, a2 = {"Name": "Stocks", "Ticker": "ACWE.MI"}, {"Name": "Bonds", "Ticker": "AGGH.MI"}

        minimal_assets = [
            {
                "Name": a1.get("Name", "Stocks"),
                "Ticker": a1.get("Ticker", "ACWE.MI"),
                "Short": a1.get("Short", a1.get("Name", "Stocks")),
                "Weight": 60.0,
                "Target": 60.0,
            },
            {
                "Name": a2.get("Name", "Bonds"),
                "Ticker": a2.get("Ticker", "AGGH.MI"),
                "Short": a2.get("Short", a2.get("Name", "Bonds")),
                "Weight": 40.0,
                "Target": 40.0,
            },
        ]
        minimal_example_obj = {
            "Name": "My Portfolio",
            "Description": "",
            "Link": "",
            "Assets": minimal_assets,
            "Value": 100_000.0,
        }

        if not show_dropdown:
            txt = json.dumps(minimal_example_obj, indent=2)
            st.code(txt, language="json")
            st.download_button(
                "Download (.json)",
                data=txt.encode("utf-8"),
                file_name="portfolio_example_minimal.json",
                mime="application/json",
                key=f"{key_prefix}_download_example_min",
            )
            return

        example_options = ["Minimal example (template)"] + [os.path.basename(p) for p in example_paths]
        selected_ex = st.selectbox("Choose an example", options=example_options, key=f"{key_prefix}_example_select")

        if selected_ex == "Minimal example (template)":
            txt = json.dumps(minimal_example_obj, indent=2)
            st.code(txt, language="json")
            st.download_button(
                "Download minimal example (.json)",
                data=txt.encode("utf-8"),
                file_name="portfolio_example_minimal.json",
                mime="application/json",
                key=f"{key_prefix}_download_example_min",
            )
            return

        match_path = None
        for pth in example_paths:
            if os.path.basename(pth) == selected_ex:
                match_path = pth
                break
        if not match_path:
            st.info("No example file selected.")
            return

        try:
            with open(match_path, "r", encoding="utf-8") as f:
                txt = f.read()
            st.code(txt, language="json")
            st.download_button(
                "Download this example (.json)",
                data=txt.encode("utf-8"),
                file_name=os.path.basename(match_path),
                mime="application/json",
                key=f"{key_prefix}_download_example_builtin",
            )
        except Exception as e:
            st.warning(f"Could not read example file: {e}")

def render_portfolio_preview(p: Portfolio, show_current: bool = False) -> None:
    """Render a compact allocation preview for a portfolio using Short names.
    
    Args:
        p: Portfolio object
        show_current: If True, show both current and target weights (for rebalancing)
    """
    target_weights_pct = getattr(p, "target_weights_pct", {})
    actual_weights_pct = getattr(p, "actual_weights_pct", {})
    tickers = getattr(p, "tickers", [])
    name = getattr(p, "name", "Portfolio")
    short_map = get_short_name_map()
    
    if not tickers or not target_weights_pct:
        return
    
    if show_current and actual_weights_pct:
        # Show both current and target weights
        current_parts = []
        target_parts = []
        for ticker in tickers:
            short_name = short_map.get(ticker, ticker)
            current_w = actual_weights_pct.get(ticker, 0.0)
            target_w = target_weights_pct.get(ticker, 0.0)
            current_parts.append(f"{short_name} ({current_w:.1f}%)")
            target_parts.append(f"{short_name} ({target_w:.1f}%)")
        
        st.caption(f"**{name}**")
        st.caption(f"Current: {', '.join(current_parts)}")
        st.caption(f"Target: {', '.join(target_parts)}")
    else:
        # Just show target weights
        parts = []
        for ticker in tickers:
            short_name = short_map.get(ticker, ticker)
            weight = target_weights_pct.get(ticker, 0.0)
            parts.append(f"{short_name} ({weight:.1f}%)")
        
        allocation_str = ", ".join(parts)
        st.caption(f"**{name}:** {allocation_str}")

# Try to use experimental_fragment if available, otherwise standard
try:
    from streamlit import fragment
except ImportError:
    try:
        from streamlit import experimental_fragment as fragment
    except ImportError:
        # No-op decorator
        def fragment(func):
            return func

@fragment
def render_portfolio_builder(
    *,
    key: str,
    title: str = "Portfolio",
    modes: list[str] = ["Manual", "Built-in", "Upload"],
    allow_value: bool = False,
    default_mode: str = "Manual",
) -> Tuple[Optional[Portfolio], Optional[dict[str, Any]]]:
    """
    Unified Portfolio Builder component.
    """
    if title:
        st.subheader(title)

    if len(modes) > 1:
        source = st.radio(
            "Create / load portfolio",
            options=modes,
            index=modes.index(default_mode) if default_mode in modes else 0,
            horizontal=True,
            key=f"{key}_source",
        )
    else:
        source = modes[0]

    built_json_obj: dict[str, Any] | None = None

    if source == "Built-in":
        portfolios_dir = os.path.join(CACHE_DIR, "portfolios")
        paths = []
        try:
            for fname in sorted(os.listdir(portfolios_dir)):
                if fname.lower().endswith(".json"):
                    paths.append(os.path.join(portfolios_dir, fname))
        except Exception:
            paths = []

        if not paths:
            st.error("No built-in portfolios found in `cache/portfolios/`.")
            return None, None

        render_portfolio_help_dropdown()

        path = st.selectbox(
            "Select a built-in portfolio",
            options=paths,
            format_func=cached_portfolio_name,
            key=f"{key}_builtin",
        )
        try:
            loaded_p = cached_load_portfolio(path)
            # In rebalancing section, show both current and target weights
            render_portfolio_preview(loaded_p, show_current=(key == "rebalance"))
            return loaded_p, None
        except Exception as e:
            st.error(str(e))
            return None, None

    if source == "Upload":
        render_example_json_ui(key_prefix=f"{key}_upload", show_dropdown=False)

        up = st.file_uploader("Upload a portfolio JSON", type=["json"], key=f"{key}_upload")
        if up is None:
            return None, None
        try:
            raw = up.getvalue()
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                raise ValueError("Invalid JSON file (could not parse).")
            ok, errs = validate_portfolio_json_obj(obj)
            if not ok:
                raise ValueError("Invalid portfolio JSON:\n- " + "\n- ".join(errs))

            loaded_p = portfolio_from_json_obj_with_cache(obj, source=up.name)
            # In rebalancing section, show both current and target weights
            render_portfolio_preview(loaded_p, show_current=(key == "rebalance"))
            return loaded_p, None
        except Exception as e:
            st.error(str(e))
            return None, None

    # Manual Mode
    col1, col2 = st.columns([2, 1])
    with col1:
        portfolio_name = st.text_input("Portfolio name", value="My Portfolio", key=f"{key}_name")
    with col2:
        value_eur = None
        if allow_value:
            value_eur = st.number_input("Current value (EUR)", min_value=0.0, value=100_000.0, step=1_000.0, key=f"{key}_value")

    asset_options, asset_mapping, asset_display_map = get_asset_options()
    if not asset_options:
        st.error("No assets available. Check `cache/assets/list.json`.")
        return None, None

    render_asset_help_dropdown()

    def _find_asset_by_short(opts: list[str], short_name: str) -> str:
        for opt in opts:
            if opt.lower() == short_name.lower():
                return opt
        return opts[0] if opts else ""
    
    default_stocks = _find_asset_by_short(asset_options, "Stocks")
    default_bonds = _find_asset_by_short(asset_options, "Bonds")
    if not default_stocks and asset_options:
        default_stocks = asset_options[0]
    if not default_bonds and len(asset_options) > 1:
        default_bonds = asset_options[1] if asset_options[1] != default_stocks else (asset_options[0] if asset_options[0] != default_stocks else "")
    
    if allow_value:
        # Table-based UI
        default = pd.DataFrame(
            [
                {"Asset": default_stocks, "Weight (%)": 65.0, "Target (%)": 60.0},
                {"Asset": default_bonds, "Weight (%)": 35.0, "Target (%)": 40.0},
            ]
        )
        column_config = {
            "Asset": st.column_config.SelectboxColumn(
                "Asset",
                options=asset_options,
                required=True,
                help="Select an asset from the available list",
            ),
            "Weight (%)": st.column_config.NumberColumn("Current Weight (%)", required=True, min_value=0.0, help="Current allocation percentage"),
            "Target (%)": st.column_config.NumberColumn("Target Weight (%)", required=True, min_value=0.0, help="Target allocation percentage"),
        }
        
        edited_df = st.data_editor(
            default,
            num_rows="dynamic",
            width="stretch",
            key=f"{key}_editor",
            column_config=column_config,
        )
        
        df = pd.DataFrame(edited_df.to_dict("records"))
        
        asset_names = []
        tickers = []
        for x in df["Asset"].tolist():
            if pd.notna(x) and x and x in asset_mapping:
                name, ticker, _short = asset_mapping[x]
                asset_names.append(name)
                tickers.append(ticker)
            else:
                asset_names.append("")
                tickers.append("")
        df["Asset Name"] = asset_names
        df["Ticker"] = tickers
        
        df["Weight (%)"] = pd.to_numeric(df["Weight (%)"], errors="coerce")
        df["Target (%)"] = pd.to_numeric(df["Target (%)"], errors="coerce")
        w_sum = float(df["Weight (%)"].sum())
        t_sum = float(df["Target (%)"].sum())
        
        if w_sum > 0 and abs(w_sum - 100.0) > 0.5:
            df["Weight (%)"] = df["Weight (%)"] / w_sum * 100.0
            df["Weight (%)"] = df["Weight (%)"].round(2)
            st.warning(f"⚠️ **Current weights** sum to **{w_sum:.0f}%**. Normalized automatically.")
        
        if t_sum > 0 and abs(t_sum - 100.0) > 0.5:
            df["Target (%)"] = df["Target (%)"] / t_sum * 100.0
            df["Target (%)"] = df["Target (%)"].round(2)
            st.warning(f"⚠️ **Target weights** sum to **{t_sum:.0f}%**. Normalized automatically.")
        
        normalize = True

    else:
        # Slider-based UI
        weights_key = f"{key}_slider_weights"
        order_key = f"{key}_asset_order"  # Track order separately
        
        if weights_key not in st.session_state:
            st.session_state[weights_key] = {default_stocks: 60.0, default_bonds: 40.0}
        if order_key not in st.session_state:
            st.session_state[order_key] = [default_stocks, default_bonds]
        
        # Note: We intentionally don't call presort_multiselect_state here
        # to preserve the user's selection order in the dropdown chips
        current_selection = list(st.session_state[weights_key].keys())
        current_selection = [a for a in current_selection if a in asset_options]
        if not current_selection:
            current_selection = [default_stocks, default_bonds]
        
        # Fix for session state warning: don't provide default if key is already in state
        ms_key = f"{key}_asset_select"
        ms_default = current_selection if ms_key not in st.session_state else None

        selected_assets = st.multiselect(
            "Select assets",
            options=asset_options,
            default=ms_default,
            format_func=lambda x: asset_display_map.get(x, x),
            key=ms_key,
        )
        
        if not selected_assets:
            st.warning("Please select at least one asset.")
            return None, None
        
        # Maintain stable order: keep existing order, append new assets at end
        prev_order = st.session_state.get(order_key, [])
        stable_order = [a for a in prev_order if a in selected_assets]
        for a in selected_assets:
            if a not in stable_order:
                stable_order.append(a)
        st.session_state[order_key] = stable_order
        
        current_weights = st.session_state[weights_key]
        for asset in selected_assets:
            if asset not in current_weights:
                current_weights[asset] = 10.0
        
        current_weights = {k: v for k, v in current_weights.items() if k in selected_assets}
        st.session_state[weights_key] = current_weights
        
        raw_weights: dict[str, float] = {}
        
        # Use stable_order instead of selected_assets for consistent ordering
        for asset in stable_order:
            if asset in asset_mapping:
                full_name, _ticker, _short = asset_mapping[asset]
            else:
                full_name = asset
            safe_key = "".join(c if c.isalnum() else "_" for c in asset)
            
            col_name, col_slider = st.columns([2.5, 6.5])
            with col_name:
                st.markdown(f"**{full_name}**")
            with col_slider:
                default_val = float(current_weights.get(asset, 10.0))
                weight = st.slider(
                    f"Weight for {full_name}",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_val,
                    step=1.0,
                    key=f"{key}_slider_{safe_key}",
                    label_visibility="collapsed",
                )
                raw_weights[asset] = weight
        
        st.session_state[weights_key] = raw_weights
        
        total_raw = sum(raw_weights.values())
        if total_raw <= 0:
            st.warning("Total weight must be greater than 0.")
            return None, None
        
        normalized_weights = {k: round(v / total_raw * 100.0, 2) for k, v in raw_weights.items()}
        
        if abs(total_raw - 100.0) > 0.5:
             st.caption(f"Weights sum to {total_raw:.0f}%. Normalizing to 100%.")
        else:
             alloc_parts = []
             for asset in stable_order:
                if asset in asset_mapping:
                    full_name, _ticker, _short = asset_mapping[asset]
                else:
                    full_name = asset
                alloc_parts.append(f"{full_name}: **{normalized_weights[asset]:.1f}%**")
             st.caption(" · ".join(alloc_parts))

        rows = []
        for asset in stable_order:
            if asset in asset_mapping:
                name, ticker, _short = asset_mapping[asset]
                rows.append({
                    "Asset": asset,
                    "Asset Name": name,
                    "Ticker": ticker,
                    "Weight (%)": normalized_weights[asset],
                })
        
        df = pd.DataFrame(rows)
        normalize = True

    try:
        built_json_obj = _portfolio_json_from_manual(portfolio_name=portfolio_name, df=df, value_eur=value_eur if allow_value else None)
        p = _build_portfolio_from_manual(
            portfolio_name=portfolio_name,
            df=df,
            value_eur=value_eur if allow_value else None,
            normalize=bool(normalize),
        )
    except Exception as e:
        st.error(str(e))
        return None, built_json_obj

    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            "Download JSON",
            data=json.dumps(built_json_obj, indent=2).encode("utf-8"),
            file_name=f"{(portfolio_name or 'portfolio').strip().replace(' ', '_')}.json",
            mime="application/json",
            key=f"{key}_download",
        )
    with c2:
        with st.expander("Preview JSON"):
            st.code(json.dumps(built_json_obj, indent=2), language="json")

    return p, built_json_obj
