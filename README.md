# CACH€ - Financial Portfolio Assistant

A Streamlit application for portfolio analysis, comparison, rebalancing, withdrawal planning, and what-if scenarios. Built for long-term, passive EUR-denominated investors using ETFs.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### 📊 Analyze a Portfolio
Backtest a single portfolio over a selected date range with comprehensive metrics:
- **Key Metrics**: Total return, CAGR, volatility, Sharpe/Sortino ratios, Ulcer Index, max drawdown, longest drawdown period
- **Portfolio Value Chart**: Historical growth visualization with linear/logarithmic scale
- **Asset Trajectories**: Individual performance of each asset in your portfolio
- **Rolling 12M Correlation vs Stocks**: Understand diversification over time
- **Drawdown Chart**: Visualize portfolio drawdowns and recovery periods
- **Rolling Returns & Volatility**: 12-month rolling metrics

### 📈 Compare Portfolios
Compare multiple portfolios side-by-side:
- Select from built-in portfolios, upload custom JSONs, or create manually
- **Allocation Overview**: Visual comparison of asset allocations
- **Key Metrics Table**: Side-by-side performance statistics
- **Combined Charts**: Portfolio value, drawdown, rolling returns, and volatility

### 💰 Rebalance with New Cash
Compute optimal allocation of new funds to approach target weights **without selling** (tax-efficient):
- **Optimal Buy Plan**: Exact EUR amounts to invest in each asset
- **Portfolio Diagnostics**: Valuation metrics (Z-score, EWMA distance, 12m CAGR)
- **Macro Dashboard**: ECB/FED rates, inflation, yields, FX, earnings yield estimates
- **AI Assistant**: LLM-assisted recommendations via OpenRouter (free tier)

### 💸 Withdraw Cash
Plan optimal withdrawals by **selling** assets while minimizing deviation from targets:
- **Optimal Sell Plan**: Exact EUR amounts to sell per asset
- **Same trend + macro diagnostics and AI assistance as rebalancing**

### 🔮 What-if: Add an Asset
Evaluate the impact of adding new assets to your portfolio:
- **Diversification Analysis**: Correlations, volatility impact, weighted metrics
- **RRR Analysis**: Return-to-Risk Ratio test based on [Portfolio Intuition (Kennedy 2018)](https://www.bridgealternatives.com/insights/portfolio-intuition)
- **Backtest Comparison**: Side-by-side performance vs baseline
- **AI Assistant**: LLM-assisted recommendations

## Quick Start

The app is publicly deployed on the [Streamlit Cloud](https://cache-finance.streamlit.app/). For local development, follow the steps below.

### 1. Clone and Install

```bash
git clone https://github.com/morsingher/cache.git
cd cache
pip install -r requirements.txt
```

### 2. Download Data

The app uses a local data store for fast, offline access. Download the latest data:

```bash
# Basic (prices only)
python scripts/update_data.py

# With macro data (requires free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html)
python scripts/update_data.py --fred-api-key YOUR_KEY
```

### 3. Configure API Keys (Optional)

Create `.streamlit/secrets.toml` for additional features:

```toml
# Required for macro data refresh via API
FRED_API_KEY = "your-fred-api-key"

# Required for AI Assistant feature
OPENROUTER_API_KEY = "your-openrouter-api-key"
```

### 4. Run the App

```bash
streamlit run app.py
```

## Available Assets

The app supports 29 ETFs and cryptocurrencies covering major asset classes:

| Category | Assets |
|----------|--------|
| **Equities** | Global Stocks (ACWI), Multifactor, Min Vol, Momentum, Value, Quality, Small Cap, SCV, High Dividend |
| **Fixed Income** | Global Bonds, EUR Bonds, EUR Inflation-Linked, EUR High Yield, Short/Mid/Long Duration |
| **Alternatives** | Gold, Silver, Managed Futures (DBMF), Commodity Beta/Carry/Mix, Real Estate |
| **Crypto** | Bitcoin (BTC), Ethereum (ETH), Solana (SOL), XRP |
| **Cash** | EUR Money Market (XEON) |

See the full list with descriptions in the app under "Need help choosing assets?"

## Built-in Portfolios

| Portfolio | Description |
|-----------|-------------|
| **60/40** | Classic 60% stocks, 40% bonds allocation |
| **All Weather** | Ray Dalio-inspired risk parity approach |
| **Golden Butterfly** | Balanced allocation with gold and small-cap value |
| **Model** | Diversified portfolio with alternatives (inspired by Cockroach/Italian Leather Sofa) |
| **Permanent** | Harry Browne's permanent portfolio concept |

## Portfolio JSON Format

Create custom portfolios using this JSON structure:

```json
{
  "Name": "My Portfolio",
  "Description": "Optional description",
  "Link": "https://optional-reference-link.com",
  "Assets": [
    {"Name": "Stocks", "Ticker": "ACWE.MI", "Short": "Stocks", "Weight": 60.0, "Target": 60.0},
    {"Name": "Bonds", "Ticker": "AGGH.MI", "Short": "Bonds", "Weight": 40.0, "Target": 40.0}
  ],
  "Value": 100000.0
}
```

- **Weight**: Current allocation (%) - used for rebalancing calculations
- **Target**: Target allocation (%) - your desired long-term weights
- **Value**: Current portfolio value in EUR (required for rebalancing/withdrawal)

## FAQs

**Why Streamlit?**

Streamlit is just a convenient way to provide nice, modern frontend to a Python backend. Also, it offers free hosting which is a nice feature. If the app gains a solid user base, I might consider switching to a more robust infrastructure in the future.

**What is the data source? Why is it cached locally?**

Historical prices are from [Yahoo Finance](https://github.com/ranaroussi/yfinance), while acroeconomic data is from [FRED](https://fred.stlouisfed.org/). Data is stored locally for fast, offline access. Free API calls to yfinance and FRED can frequently fail, especially when coming from crowded, shared servers like Streamlit. This is not a big deal, since it ensures smooth UX and the intended user should not care about the latest data anyway. That said, I am well aware premium data APIs exist and I might consider switching to those in the future. Notably, this would also enable further diversification analysis (across sectors, geographies, companies) and significantly extended time frames.

**How does the AI assistant work?**

The app builds detailed prompts from analysis results that can be either copied and pasted into your preferred LLM, or used directly via [OpenRouter API](https://openrouter.ai/) (free tier models). The goal is to use AI-assisted reasoning to help the user take the best decision, especially when combining multiple signals (optimal allocation, trends, macro indicators) is inherently ill-posed. Frontier models are quite pricey, therefore I restrict the usage to free models, for now. I might consider removing this limitation in the future and allow the user insert his own API key.

**Why is it called CACH€?**

The name is an intended pun on the word "cache", which is pronounced as "cash", but has a widespread use in computer science.

## Project Structure

```
finance/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── cache/
│   ├── portfolio.py       # Core Portfolio class with backtesting
│   ├── rebalancing.py     # Rebalancing logic and macro data
│   ├── whatif.py          # What-if analysis functions
│   ├── comparison.py      # CLI comparison utilities
│   ├── datastore.py       # Local data store interface
│   ├── openrouter.py      # LLM API integration
│   ├── assets/            # Asset definitions and synthetic series
│   └── portfolios/        # Built-in portfolio JSONs
├── ui/
│   ├── pages/             # Streamlit page modules
│   ├── components.py      # Reusable UI components
│   ├── portfolio_builder.py # Portfolio creation UI
│   ├── charts.py          # Altair chart helpers
│   ├── assets.py          # Asset loading and caching
│   └── styles.py          # Custom CSS styles
├── data/                  # Local data store (gitignored)
├── scripts/
│   └── update_data.py     # Data download script
└── static/                # Custom fonts
```

## Acknowledgements

Built with [Streamlit](https://streamlit.io/), [Altair](https://altair-viz.github.io/), [yfinance](https://github.com/ranaroussi/yfinance), and [FRED](https://fred.stlouisfed.org/). Kudos to my vibe-coding teammates Claude 4.5 Opus High, Gemini 3 Pro and GPT 5.2 Codex.

### References

- Cliff Asness, Antti Ilmanen, Thomas Maloney - [Market Timing: Sin a Little](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Market-Timing-Sin-a-Little.pdf)
- Mebane T. Faber - [A Quantitative Approach to Tactical Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461)
- Adam Butler et al. - [Adaptive Asset Allocation: A Primer](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2328254)
- Diana Barro et al. - [Volatility vs. Downside Risk](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2521007)
- Chris Kennedy - [Portfolio Intuition](https://www.bridgealternatives.com/insights/portfolio-intuition)
- Nicola Protasoni - [The Italian Leather Sofa](https://theitalianleathersofa.com/)
- Mutiny Funds - [The Cockroach Approach](https://mutinyfund.com/wp-content/uploads/2024/04/The-Cockroach-Approach.pdf)

## License

MIT License - see [LICENSE](LICENSE) for details.
