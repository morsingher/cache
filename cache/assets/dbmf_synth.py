import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def reconstruct_european_history(us_ticker, eu_ticker, currency_pair="EURUSD=X"):
    
    # 1. Fetch Data
    tickers = [us_ticker, eu_ticker, currency_pair]
    data = yf.download(tickers, period="max", auto_adjust=True, progress=False)['Close']
    
    # 2. Clean Data (Drop rows where US data is missing)
    # We only care about days where the US market was open
    df = data.dropna(subset=[us_ticker]).copy()
    
    # Forward fill missing FX rates (e.g., US market open but FX data gap)
    df[currency_pair] = df[currency_pair].ffill()

    # 3. Calculate Unscaled Synthetic EUR Price
    # Formula: USD_Price / (USD_per_EUR) = EUR_Price
    df['Synthetic_Raw'] = df[us_ticker] / df[currency_pair]
    
    # 4. Calculate Scaling Factor (The "Stitch")
    # Find the first date where we have REAL European data
    first_eu_date = df[eu_ticker].first_valid_index()
    
    if first_eu_date is None:
        raise ValueError("No European data found to stitch against.")
        
    # Get the prices on that specific "Link Date"
    real_price_at_link = df.loc[first_eu_date, eu_ticker]
    synth_price_at_link = df.loc[first_eu_date, 'Synthetic_Raw']
    
    # Scaling Factor: How much do we need to multiply the synthetic data 
    # to make it match the real data on day 1?
    scaling_factor = real_price_at_link / synth_price_at_link
    
    # 5. Apply Scaling to create final Synthetic History
    df['Synthetic_History'] = df['Synthetic_Raw'] * scaling_factor
    
    # 6. Combine: Use Real data where available, otherwise use Synthetic
    # We create a new column 'Combined'
    df['Combined_History'] = df[eu_ticker].combine_first(df['Synthetic_History'])
    
    return df

if __name__ == "__main__":
    # --- Execution / Demo Plot ---
    try:
        # tickers
        us_etf = "DBMF"       # The Proxy (US History)
        eu_etf = "DBMFE.PA"   # The Target (European History)

        df = reconstruct_european_history(us_etf, eu_etf)

        # --- Visualization ---
        plt.figure(figsize=(12, 6))

        # Plot the Synthetic part (older data)
        plt.plot(
            df.index,
            df['Synthetic_History'],
            label='Synthetic Backfill (USD converted to EUR)',
            color='gray',
            linestyle='--',
            alpha=0.6
        )

        # Plot the Real part (newer data)
        # We re-plot the EU ticker on top to show the seamless join
        plt.plot(df.index, df[eu_etf], label=f'Actual {eu_etf}', color='#0052cc', linewidth=2)

        plt.title(f"Reconstructed History: {eu_etf} (via {us_etf})", fontsize=14)
        plt.ylabel("Price (€)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")