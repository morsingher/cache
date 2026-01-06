import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def reconstruct_zprvx_history(
    ticker1: str = "ZPRV.DE",
    ticker2: str = "ZPRX.DE",
    weight1: float = 0.70,
    weight2: float = 0.30,
) -> pd.DataFrame:
    """
    Reconstruct a synthetic price history for ZPRVX as a weighted combination
    of two underlying ETFs.

    ZPRVX = 70% ZPRV.DE (US Small Cap Value) + 30% ZPRX.DE (Europe Small Cap Value)

    The synthetic price is computed as a static-weight portfolio starting at 100:
        synthetic_t = 100 * (w1 * ZPRV_t/ZPRV_0 + w2 * ZPRX_t/ZPRX_0)

    Returns:
        DataFrame with columns:
            - ticker1: Close prices for ZPRV.DE
            - ticker2: Close prices for ZPRX.DE
            - Combined_History: Synthetic ZPRVX prices
    """
    # 1. Fetch Data
    tickers = [ticker1, ticker2]
    raw_data = yf.download(tickers, period="max", auto_adjust=True, progress=False)

    # 2. Robust Column Flattening (handle all yfinance return structures)
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Check if 'Close' is a top-level key
        if "Close" in raw_data.columns.get_level_values(0):
            df = raw_data["Close"].copy()
        else:
            # Fallback: Try to extract Close if it's in the second level
            try:
                df = raw_data.xs("Close", axis=1, level=1, drop_level=True).copy()
            except KeyError:
                df = raw_data.copy()
    else:
        if "Close" in raw_data.columns:
            df = raw_data["Close"].copy()
        else:
            df = raw_data.copy()

    # FORCE columns to be simple strings (removing any MultiIndex levels or names)
    df.columns = [str(c[1]) if isinstance(c, tuple) else str(c) for c in df.columns]
    df.columns = df.columns.str.strip()

    # Ensure index is datetime and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 3. Validate we have both tickers
    if ticker1 not in df.columns:
        raise ValueError(f"Ticker {ticker1} not found in downloaded data.")
    if ticker2 not in df.columns:
        raise ValueError(f"Ticker {ticker2} not found in downloaded data.")

    # 4. Find common date range where both have data
    first_valid_1 = df[ticker1].first_valid_index()
    first_valid_2 = df[ticker2].first_valid_index()
    last_valid_1 = df[ticker1].last_valid_index()
    last_valid_2 = df[ticker2].last_valid_index()

    if first_valid_1 is None or first_valid_2 is None:
        raise ValueError("No valid data found for one or both tickers.")

    common_start = max(first_valid_1, first_valid_2)
    common_end = min(last_valid_1, last_valid_2)

    if common_start > common_end:
        raise ValueError(
            f"No overlapping date range for {ticker1} and {ticker2}."
        )

    # Slice to common range
    df = df.loc[common_start:common_end].copy()

    # Forward fill any gaps (non-trading days)
    df = df.ffill()

    # Drop rows with any NaN (shouldn't happen after ffill, but be safe)
    df = df.dropna()

    if df.empty:
        raise ValueError("No valid data after cleaning.")

    # 5. Compute synthetic price as weighted portfolio
    # Starting value = 100
    p1_0 = float(df[ticker1].iloc[0])
    p2_0 = float(df[ticker2].iloc[0])

    if p1_0 <= 0 or p2_0 <= 0:
        raise ValueError("Invalid starting prices (must be positive).")

    # Normalized returns: P_t / P_0
    norm1 = df[ticker1] / p1_0
    norm2 = df[ticker2] / p2_0

    # Weighted combination starting at 100
    df["Combined_History"] = 100.0 * (weight1 * norm1 + weight2 * norm2)

    return df


def get_zprvx_series() -> pd.Series:
    """
    Convenience function to get just the synthetic ZPRVX price series.

    Returns:
        pd.Series: Synthetic ZPRVX prices indexed by date, named 'ZPRVX'.
    """
    df = reconstruct_zprvx_history()
    s = df["Combined_History"].rename("ZPRVX")
    # Remove any duplicate indices
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


if __name__ == "__main__":
    try:
        print("Fetching and reconstructing ZPRVX (70% ZPRV.DE + 30% ZPRX.DE)...")
        df = reconstruct_zprvx_history()
        print("Success!")
        print(f"\nDate range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Total days: {len(df)}")
        print(f"\nLast 5 rows:")
        print(df[["ZPRV.DE", "ZPRX.DE", "Combined_History"]].tail())

        # --- Visualization ---
        plt.figure(figsize=(12, 6))

        # Normalize all to start at 100 for comparison
        p1_0 = df["ZPRV.DE"].iloc[0]
        p2_0 = df["ZPRX.DE"].iloc[0]

        plt.plot(
            df.index,
            100 * df["ZPRV.DE"] / p1_0,
            label="ZPRV.DE (US SCV)",
            color="blue",
            alpha=0.7,
            linewidth=1.5,
        )
        plt.plot(
            df.index,
            100 * df["ZPRX.DE"] / p2_0,
            label="ZPRX.DE (EU SCV)",
            color="orange",
            alpha=0.7,
            linewidth=1.5,
        )
        plt.plot(
            df.index,
            df["Combined_History"],
            label="ZPRVX Synthetic (70/30)",
            color="green",
            linewidth=2.5,
        )
        plt.title("ZPRVX Synthetic: 70% ZPRV.DE + 30% ZPRX.DE")
        plt.ylabel("Normalized Price (start=100)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

