import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def reconstruct_european_history(us_ticker, eu_ticker, currency_pair="EURUSD=X"):
    
    # 1. Fetch Data
    tickers = [us_ticker, eu_ticker, currency_pair]
    # Download full data
    raw_data = yf.download(tickers, period="max", auto_adjust=True, progress=False)
    
    # 2. Robust Column Flattening
    # This block handles all variations of yfinance return structures (MultiIndex or Flat)
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Check if 'Close' is a top-level key (standard in recent yfinance)
        if 'Close' in raw_data.columns.get_level_values(0):
            df = raw_data['Close'].copy()
        else:
            # Fallback: Try to extract Close if it's in the second level
            try:
                df = raw_data.xs('Close', axis=1, level=1, drop_level=True).copy()
            except KeyError:
                # Last resort: just take the whole thing if structure is weird
                df = raw_data.copy()
    else:
        # If it's already flat (e.g. single ticker download sometimes), just ensure we have Close
        if 'Close' in raw_data.columns:
            df = raw_data['Close'].copy()
        else:
            df = raw_data.copy()

    # FORCE columns to be simple strings (removing any remaining MultiIndex levels or names)
    # This explicitly resolves the "2 levels on left, 1 on right" error.
    df.columns = [str(c[1]) if isinstance(c, tuple) else str(c) for c in df.columns]
    
    # Clean up column names to ensure they match our inputs (remove accidental whitespace)
    df.columns = df.columns.str.strip()
    
    # Ensure index is datetime and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 3. Clean Data
    # Drop rows where US data is missing (market closed)
    df = df.dropna(subset=[us_ticker]).copy()
    
    # Forward fill missing FX rates
    if currency_pair in df.columns:
        df[currency_pair] = df[currency_pair].ffill()
    else:
        raise ValueError(f"Currency pair {currency_pair} not found in downloaded data.")

    # 4. Calculate Unscaled Synthetic EUR Price
    df['Synthetic_Raw'] = df[us_ticker] / df[currency_pair]
    
    # 5. Calculate Scaling Factor (The "Stitch")
    first_eu_date = df[eu_ticker].first_valid_index()
    
    if first_eu_date is None:
        raise ValueError("No European data found to stitch against.")
        
    real_price_at_link = df.loc[first_eu_date, eu_ticker]
    synth_price_at_link = df.loc[first_eu_date, 'Synthetic_Raw']
    
    scaling_factor = real_price_at_link / synth_price_at_link
    
    # 6. Apply Scaling
    df['Synthetic_History'] = df['Synthetic_Raw'] * scaling_factor
    
    # 7. Combine
    # Explicitly casting to Series ensures merge compatibility
    real_series = df[eu_ticker]
    synth_series = df['Synthetic_History']
    
    df['Combined_History'] = real_series.combine_first(synth_series)
    
    return df

if __name__ == "__main__":
    try:
        us_etf = "DBMF"
        eu_etf = "DBMFE.PA"
        
        print("Fetching and reconstructing...")
        df = reconstruct_european_history(us_etf, eu_etf)
        print("Success!")
        print(df[['Combined_History']].tail())

        # --- Visualization ---
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Synthetic_History'], label='Synthetic', color='gray', linestyle='--', alpha=0.6)
        plt.plot(df.index, df[eu_etf], label='Actual', color='#0052cc', linewidth=2)
        plt.title(f"Reconstructed: {eu_etf}")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")