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
Backtest a single portfolio over a selected date range. View performance metrics (Total Return, CAGR, volatility, Sharpe, Sortino, max drawdown, longest drawdown period), allocation breakdown, value trajectories, rolling correlations, and a drawdown chart.

### Compare portfolios
Compare multiple portfolios side-by-side. Select from built-in portfolios, upload custom JSONs, or create one manually. View allocation differences and performance statistics (including longest drawdown) on a common date range, plus drawdown and value charts for visual comparison.

### Rebalance with new cash
Given your current portfolio value and new cash to invest, compute the optimal allocation to bring the portfolio back to target weights.

Includes a **macro dashboard** (best-effort, via FRED + an earnings-yield estimate) with:
- A **2×4** snapshot grid:
  - **EU/DE**: ECB deposit rate, DE 10Y yield, DE inflation YoY, USD/EUR spot
  - **US**: Fed risk-free rate (EFFR), US 10Y yield, US inflation YoY, global earnings yield (est.)
- **Four 12-month trend charts**: EU/DE, US, USD/EUR, and global earnings yield (est.)

Also supports optional LLM-assisted recommendations.

### What-if: add an asset
Evaluate the impact of adding new assets to your portfolio. Analyzes diversification scores, return-to-risk ratios (RRR), and backtests the modified portfolio against the baseline (including drawdown comparison + longest drawdown). It also includes optional LLM-assisted recommendations.

## Acknowledgements

Vibe-coded with Claude Opus-4.5-High and GPT 5.2 xHigh. RRR analysis inspired from [Bridge Alternatives](https://www.bridgealternatives.com/insights/portfolio-intuition).
