import argparse
import os

import pandas as pd
from fredapi import Fred

from comparison import run_comparison
from portfolio import Portfolio
from rebalancing import run_rebalancing
from whatif import run_whatif

def print_welcome_message():
    msg = "Welcome to CACHE"
    num_stars = len(msg) + 4
    print(f"\n{'*' * num_stars}\n{'* ' + msg + ' *'}\n{'*' * num_stars}\n")

def _get_rf_annual(args) -> float:
    rf_annual = float(args.rf) if args.rf is not None else 0.0
    if args.rf is None and args.rf_fred_series:
        api_key = args.fred_api_key or os.environ.get("FRED_API_KEY")
        if api_key:
            fred = Fred(api_key=api_key)
            series = fred.get_series(args.rf_fred_series).dropna()
            if not series.empty:
                latest = float(series.iloc[-1])  # usually in percent for rate/yield series
                rf_annual = latest / 100.0
                print(f"Risk-free rate (FRED {args.rf_fred_series}): {latest:.3f}%")
        else:
            print("Risk-free rate: using 0% (set FRED_API_KEY env var or pass --fred-api-key)")
    return float(rf_annual)

def main(args):

    print_welcome_message()

    rf_annual = None
    if args.analyze or args.add_asset or args.compare:
        print("RISK-FREE RATE\n")
        rf_annual = _get_rf_annual(args)
        print()

    if args.compare:
        run_comparison(args, rf_annual=float(rf_annual or 0.0))
        return

    if not args.portfolio:
        raise ValueError("You must pass --portfolio (or use --compare).")

    portfolio = Portfolio.from_json(args.portfolio)
    print(f"Loaded {portfolio.name} from {args.portfolio}")
    start_date, end_date = portfolio.adjust_dates(debug=False)
    print(f"Date window: {start_date} -> {end_date}")
    print()
    print("PORTFOLIO:")
    print("-" * 10)
    print(portfolio)
    print()

    if args.analyze:

        print("ANALYSIS\n")
        if rf_annual is None:
            rf_annual = 0.0

        stats = portfolio.compute_stats(
            rf_annual=rf_annual,
            rebalance_frequency=args.rebalance_frequency,
            debug=False,
        )
        print("STATISTICS (daily log returns, annualized):")
        print(f"- CAGR:    {stats['cagr']*100:.2f}%")
        print(f"- Vol:     {stats['vol_annual']*100:.2f}%")
        print(f"- Sharpe:  {stats['sharpe']:.2f} (rf={rf_annual*100:.2f}%)")
        print(f"- Sortino: {stats['sortino']:.2f} (MAR=rf)")
        print()

        # Correlations (monthly)
        corr = portfolio.correlation_matrix_monthly(debug=False)
        print("CORRELATIONS (monthly log returns):")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(corr.round(3))
        print()

        if args.plot:
            portfolio.plot(
                X=args.initial_amount,
                rf_annual=rf_annual,
                rebalance_frequency=args.rebalance_frequency,
            )
        else:
            print("Plotting disabled (pass --plot to enable).\n")

    if args.add_asset:
        run_whatif(
            portfolio,
            args.add_asset,
            swap_weight=float(args.swap_weight),
            rf_annual=float(rf_annual or 0.0),
            rebalance_frequency=str(args.rebalance_frequency),
        )

    if args.rebalance is not None:
        api_key = args.fred_api_key or os.environ.get("FRED_API_KEY")
        run_rebalancing(portfolio, float(args.rebalance), fred_api_key=api_key, debug=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, default=None, required=False, help="Path to a portfolio JSON file")
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Compare two or more portfolio JSON files (target weights). Accepts comma-separated values too.",
    )
    parser.add_argument("--analyze", action="store_true", help="Print statistics and correlations")
    parser.add_argument("--rebalance", type=float, default=None, help="New cash (EUR) to allocate optimally towards target weights")
    parser.add_argument("--initial-amount", type=float, default=10_000.0, help="Initial cash amount to invest for plotting")
    parser.add_argument("--plot", action="store_true", help="Enable plotting (disabled by default)")
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        default="annually",
        choices=["monthly", "quarterly", "annually"],
        help="Rebalance the analysis backtest to current weights at this frequency",
    )
    parser.add_argument(
        "--add-asset",
        nargs="+",
        default=None,
        help="One or more tickers to evaluate as a new asset (each independently). Accepts comma-separated values too.",
    )
    parser.add_argument(
        "--swap-weight",
        type=float,
        default=0.05,
        help="Fraction of portfolio to shift from Stocks target allocation into the new asset (default: 0.05 = 5%).",
    )
    parser.add_argument("--rf", type=float, default=None, help="Annual risk-free rate as decimal (e.g. 0.03 for 3%)")
    parser.add_argument("--rf-fred-series", type=str, default="ECBDFR", help="FRED series id for risk-free proxy (value assumed in %)")
    parser.add_argument("--fred-api-key", type=str, default=None, help="FRED API key (or set env var FRED_API_KEY)")
    args = parser.parse_args()

    main(args)