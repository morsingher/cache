# CACHE - Financial Portfolio Assistant

A Streamlit application for portfolio analysis, comparison, rebalancing, and what-if scenarios.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Add your API keys to `.streamlit/secrets.toml`:
   ```toml
   OPENROUTER_API_KEY = "your-key-here"
   FRED_API_KEY = "your-key-here"
   ```
3. Run the app: `streamlit run app.py`

## Sections

### Analyze a portfolio
Backtest a single portfolio over a selected date range. View performance metrics (CAGR, volatility, Sharpe, Sortino, max drawdown), allocation breakdown, value trajectories, and rolling correlations.

### Compare portfolios
Compare multiple portfolios side-by-side. Select from built-in portfolios, upload custom JSONs, or create one manually. View allocation differences and performance statistics on a common date range.

### Rebalance with new cash
Given your current portfolio value and new cash to invest, compute the optimal allocation to bring the portfolio back to target weights. Includes macro-economic context from FRED and optional LLM-assisted recommendations.

### What-if: add an asset
Evaluate the impact of adding new assets to your portfolio. Analyzes diversification scores, return-to-risk ratios (RRR), and backtests the modified portfolio against the baseline. It also includes optional LLM-assisted recommendations.

## Acknowledgements

Vibe-coded with Claude Opus-4.5-High. RRR analysis inspired from [Bridge Alternatives](https://www.bridgealternatives.com/insights/portfolio-intuition).
