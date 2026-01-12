import os

import numpy as np
import pandas as pd

from portfolio import Portfolio


def _flatten_comma_separated(items: list[str] | None) -> list[str]:
    if not items:
        return []
    out: list[str] = []
    for x in items:
        if x is None:
            continue
        parts = [p.strip() for p in str(x).split(",")]
        out.extend([p for p in parts if p])
    return out


def _fmt_pct(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "nan"
    if not np.isfinite(v):
        return "nan"
    return f"{v*100.0:.2f}%"


def _fmt_num(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "nan"
    if not np.isfinite(v):
        return "nan"
    return f"{v:.3f}"


def _fmt_days(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "nan"
    if not np.isfinite(v):
        return "nan"
    return f"{int(round(v))}d"


def run_comparison(args, *, rf_annual: float):
    """
    Compare multiple portfolio JSONs passed via args.compare.

    - Uses TARGET weights from JSON ('Target' per asset), not current weights.
    - Builds each portfolio equity curve from the SAME initial cash amount and rebalance frequency.
    - Aligns all portfolios to a common overlapping date index (intersection) for a fair comparison.
    """
    paths = _flatten_comma_separated(getattr(args, "compare", None))
    if len(paths) < 2:
        raise ValueError("--compare requires at least 2 portfolio JSON paths.")

    portfolios: list[tuple[str, str, Portfolio]] = []  # (display_name, path, portfolio)
    for path in paths:
        p = Portfolio.from_json(path)
        p.adjust_dates(debug=False)  # align within-portfolio tickers to their common window
        name = getattr(p, "name", None) or os.path.basename(path)
        portfolios.append((name, path, p))

    # Disambiguate duplicate display names by appending the filename.
    name_counts: dict[str, int] = {}
    for n, _, _ in portfolios:
        name_counts[n] = name_counts.get(n, 0) + 1
    if any(c > 1 for c in name_counts.values()):
        portfolios = [
            (f"{n} ({os.path.basename(path)})" if name_counts.get(n, 0) > 1 else n, path, p)
            for (n, path, p) in portfolios
        ]

    # Common overlapping index across all portfolios (intersection of each portfolio's aligned price index).
    common_index = None
    for _, _, p in portfolios:
        px = p._prices_df().dropna(how="any").sort_index()
        idx = px.index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) < 3:
        raise ValueError("Portfolios do not have enough overlapping price history to compare.")
    common_index = common_index.sort_values()

    # Build target-weight value series and stats for each portfolio on the shared index.
    rows: list[dict[str, object]] = []
    value_series: dict[str, pd.Series] = {}
    for name, _, p in portfolios:
        if not hasattr(p, "target_weights_pct"):
            raise ValueError(f"Portfolio '{name}' is missing per-asset 'Target' weights in its JSON.")

        wt_pct = [float(p.target_weights_pct[t]) for t in p.tickers]
        wt = Portfolio._normalize_weights_to_fraction(wt_pct)
        weights = {t: float(w) for t, w in zip(p.tickers, wt)}

        prices_df = p._prices_df().reindex(common_index).dropna(how="any")
        if len(prices_df) < 3:
            raise ValueError(f"Portfolio '{name}' has insufficient overlapping data after alignment.")

        v = Portfolio.backtest_value_series(
            prices_df,
            weights,
            rebalance_frequency=str(getattr(args, "rebalance_frequency", "annually")),
            initial_value=float(getattr(args, "initial_amount", 10_000.0)),
        )
        v = v.reindex(common_index).dropna()
        if len(v) < 3:
            raise ValueError(f"Portfolio '{name}' produced an empty value series after alignment.")

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
                "Max Gain": float(stats.get("max_gain", float("nan"))),
            }
        )

    df = pd.DataFrame(rows).set_index("Portfolio")
    print("COMPARISON (target weights, daily log returns stats; annualized where applicable):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        pct_cols = {"Total Return", "CAGR", "Vol (ann.)", "Max Drawdown", "Max Gain"}
        formatters = {c: _fmt_pct for c in df.columns if c in pct_cols}
        for c in df.columns:
            if c not in formatters:
                formatters[c] = _fmt_num
        if "Longest Drawdown" in df.columns:
            formatters["Longest Drawdown"] = _fmt_days
        print(df.to_string(formatters=formatters))
    print()

    if not bool(getattr(args, "plot", False)):
        print("Plotting disabled (pass --plot to enable).\n")
        return

    # Shared plot of portfolio equity curves.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6.5))
    for name, v in value_series.items():
        ax.plot(v.index, v.values, linewidth=2.0, alpha=0.9, label=name)

    start = pd.Timestamp(common_index[0]).date().isoformat()
    end = pd.Timestamp(common_index[-1]).date().isoformat()
    initial_amount = float(getattr(args, "initial_amount", 10_000.0))
    rebalance_frequency = str(getattr(args, "rebalance_frequency", "annually")).lower()
    ax.set_title(
        f"Portfolio comparison (target weights) — X={initial_amount:.2f} EUR — {start} to {end} "
        f"[rebalance: {rebalance_frequency}]"
    )
    ax.set_ylabel("Value (EUR)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()


