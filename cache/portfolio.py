import time
import logging

import yfinance as yf
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import json

# Configure logging for retry attempts
logger = logging.getLogger(__name__)


def _retry_yf_download(
    tickers: list[str],
    *,
    period: str = "max",
    auto_adjust: bool = True,
    ignore_tz: bool = True,
    progress: bool = False,
    threads: bool = True,
    max_retries: int = 4,
    base_delay: float = 1.0,
) -> pd.DataFrame | None:
    """
    Wrapper around yf.download with retry logic and exponential backoff.
    
    Args:
        tickers: List of ticker symbols to download.
        period: Data period (e.g., "max", "1y").
        auto_adjust: Whether to auto-adjust prices.
        ignore_tz: Whether to ignore timezone.
        progress: Whether to show download progress.
        threads: Whether to use multi-threading.
        max_retries: Maximum number of retry attempts (default: 4).
        base_delay: Base delay in seconds between retries (default: 1.0).
    
    Returns:
        DataFrame with downloaded data, or None if all retries failed.
    
    Raises:
        Exception: Re-raises the last exception if all retries fail.
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers,
                period=period,
                auto_adjust=auto_adjust,
                ignore_tz=ignore_tz,
                progress=progress,
                threads=threads,
            )
            
            # Check if the download actually returned valid data
            if raw is None or raw.empty:
                raise ValueError(f"yfinance returned empty data for tickers: {tickers}")
            
            # For single ticker, check if we got actual price data
            if len(tickers) == 1:
                if isinstance(raw.columns, pd.MultiIndex):
                    # Check if the Close data exists and has values
                    if "Close" not in raw.columns.get_level_values(0):
                        raise ValueError(f"No 'Close' data returned for {tickers[0]}")
                    close_data = raw["Close"]
                    if close_data.dropna().empty:
                        raise ValueError(f"Empty 'Close' data for {tickers[0]}")
                else:
                    if "Close" not in raw.columns and raw.dropna().empty:
                        raise ValueError(f"Empty data for {tickers[0]}")
            else:
                # Multiple tickers: check if Close data exists
                if "Close" in raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else "Close" in raw.columns:
                    close_data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
                    if close_data.dropna(how="all").empty:
                        raise ValueError(f"Empty 'Close' data for tickers: {tickers}")
            
            return raw
            
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            # Check if this is a retryable error
            retryable = any([
                "NoneType" in error_msg,
                "subscriptable" in error_msg,
                "Connection" in error_msg,
                "Timeout" in error_msg,
                "HTTPError" in error_msg,
                "Empty" in error_msg,
                "404" in error_msg,
                "500" in error_msg,
                "502" in error_msg,
                "503" in error_msg,
            ])
            
            if attempt < max_retries - 1 and retryable:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"yfinance download failed for {tickers} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                # Non-retryable error or last attempt
                break
    
    # All retries exhausted or non-retryable error
    if last_exception:
        raise last_exception
    
    return None

class Portfolio:
    def __init__(self, tickers: list[str], weights: list[float], assets: list[str] | None = None):
        self.tickers = tickers
        self.weights_input = weights
        if len(self.tickers) != len(self.weights_input):
            raise ValueError(
                f"tickers count ({len(self.tickers)}) must match weights count ({len(self.weights_input)})"
            )
        if assets is None:
            assets = ["unknown"] * len(self.tickers)
        if len(assets) != len(self.tickers):
            raise ValueError(
                f"tickers count ({len(self.tickers)}) must match assets count ({len(assets)})"
            )
        self.assets: dict[str, str] = {t: str(a) for t, a in zip(self.tickers, assets)}
        # Display labels: prefer asset name; disambiguate duplicates with "(TICKER)".
        counts: dict[str, int] = {}
        for a in self.assets.values():
            counts[a] = counts.get(a, 0) + 1
        self.display_labels: dict[str, str] = {}
        for t in self.tickers:
            a = self.assets.get(t, "unknown")
            self.display_labels[t] = f"{a} ({t})" if counts.get(a, 0) > 1 else a

        # Prices (Close) for all tickers. Includes DBMF stitched EUR history if requested.
        prices = Portfolio.download_prices(self.tickers, period="max", auto_adjust=True, ignore_tz=True, progress=False)
        # Keep legacy shape compatibility: Portfolio historically stored a yfinance-like "data" object
        # and exposed close prices as self.data["Close"].
        # Here we store just the Close panel under a top-level "Close" key.
        self.data = pd.concat({"Close": prices}, axis=1)
        self.prices = self.data["Close"]
        # Build a stable ticker->last_price mapping; yfinance may reorder columns.
        if isinstance(self.prices, pd.Series):
            # Single ticker: prices is a Series indexed by date.
            s = self.prices.dropna()
            if s.empty:
                raise ValueError("No valid price data found for the requested ticker(s).")
            self.last_prices = {self.tickers[0]: float(s.iloc[-1])}
        else:
            # Multiple tickers: prices is a DataFrame with ticker columns.
            # Do NOT use iloc[-1] directly: last row can contain NaNs for some tickers.
            self.last_prices = {}
            for t in self.tickers:
                s = self.prices[t].dropna()
                if s.empty:
                    raise ValueError(f"No valid price data found for ticker '{t}'.")
                self.last_prices[t] = float(s.iloc[-1])

        # Store weights (supports either fractions that sum to ~1, or percentages that sum to ~100).
        raw_sum = float(sum(self.weights_input))
        if raw_sum <= 0:
            raise ValueError("Weights must sum to a positive number.")
        if any(w < 0 for w in self.weights_input):
            raise ValueError("Weights must be non-negative.")

        # Heuristic: if sum is clearly > 1, interpret as percentages.
        if raw_sum > 1.5:
            frac = [float(w) / 100.0 for w in self.weights_input]
        else:
            frac = [float(w) for w in self.weights_input]

        frac_sum = float(sum(frac))
        if frac_sum <= 0:
            raise ValueError("Weights must sum to a positive number after normalization.")

        self.weights: dict[str, float] = {t: w / frac_sum for t, w in zip(self.tickers, frac)}
        self.weights_pct: dict[str, float] = {t: w * 100.0 for t, w in self.weights.items()}

    # -----------------------------
    # Shared utilities (used by what-if, analysis, etc.)
    # -----------------------------

    @staticmethod
    def download_prices(
        tickers: list[str],
        *,
        period: str = "max",
        auto_adjust: bool = True,
        ignore_tz: bool = True,
        progress: bool = False,
        threads: bool = True,
    ) -> pd.DataFrame:
        """
        Download Close/AdjClose-equivalent prices for tickers as a DataFrame (columns=tickers).

        Notes:
        - Uses yfinance bulk download.
        - Special-case: if "DBMF" is requested, we stitch a long EUR history
          (EU listing + synthetic backfill via FX) to avoid a short history.
        """
        tickers = list(tickers)
        if not tickers:
            return pd.DataFrame()

        # Optional DBMF stitched series (kept here so we can eventually delete commodities.py).
        dbmf_series = None
        if "DBMF" in tickers:
            try:
                # Support both invocation styles:
                # - `python cache/run.py` (sys.path contains .../cache) -> import via "assets"
                # - `python -m cache.run` (sys.path contains repo root) -> import via "cache.assets"
                try:
                    from assets.dbmf_synth import reconstruct_european_history  # type: ignore
                except Exception:
                    from cache.assets.dbmf_synth import reconstruct_european_history  # type: ignore

                us_ticker = "DBMF"
                eu_ticker = "DBMFE.PA"
                fx_ticker = "EURUSD=X"
                df = reconstruct_european_history(us_ticker, eu_ticker, currency_pair=fx_ticker)
                s = df["Combined_History"].rename(us_ticker)
                s = s[~s.index.duplicated(keep="last")].sort_index()
                dbmf_series = s
            except Exception:
                # If stitching fails, fall back to normal yf download.
                dbmf_series = None

        # Optional ZPRVX synthetic series (70% ZPRV.DE + 30% ZPRX.DE).
        zprvx_series = None
        if "ZPRVX" in tickers:
            try:
                try:
                    from assets.zprvx_synth import get_zprvx_series  # type: ignore
                except Exception:
                    from cache.assets.zprvx_synth import get_zprvx_series  # type: ignore

                zprvx_series = get_zprvx_series()
            except Exception:
                # If synthesis fails, fall back to normal yf download (will likely fail).
                zprvx_series = None

        # Build list of tickers to download from yfinance (exclude synthetic ones)
        tickers_to_download = tickers.copy()
        if dbmf_series is not None and "DBMF" in tickers_to_download:
            tickers_to_download = [t for t in tickers_to_download if t != "DBMF"]
        if zprvx_series is not None and "ZPRVX" in tickers_to_download:
            tickers_to_download = [t for t in tickers_to_download if t != "ZPRVX"]

        raw = None
        if tickers_to_download:
            raw = _retry_yf_download(
                tickers_to_download,
                period=period,
                auto_adjust=auto_adjust,
                ignore_tz=ignore_tz,
                progress=progress,
                threads=threads,
                max_retries=4,
                base_delay=1.0,
            )

        if not tickers_to_download:
            prices = pd.DataFrame()
        elif len(tickers_to_download) > 1:
            prices = raw["Close"]
        else:
            # Single ticker case: yfinance may return a MultiIndex DataFrame.
            # We need to flatten it to a single-level column structure before joining.
            only = tickers_to_download[0]
            if isinstance(raw.columns, pd.MultiIndex):
                # Extract the Close prices - may be Series or DataFrame depending on yfinance version
                close_data = raw["Close"]
                if isinstance(close_data, pd.Series):
                    prices = close_data.to_frame(name=only)
                else:
                    # It's a DataFrame, flatten columns and rename
                    prices = close_data.copy()
                    prices.columns = [only]
            else:
                prices = raw[["Close"]].rename(columns={"Close": only})

        # Ensure prices has a flat column structure before joining synthetic series
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(-1)

        if dbmf_series is not None:
            prices = prices.join(dbmf_series, how="outer")

        if zprvx_series is not None:
            prices = prices.join(zprvx_series, how="outer")

        # Keep columns in requested order (and drop extras).
        prices = prices.reindex(columns=tickers)
        return prices

    @staticmethod
    def fill_non_trading_days(prices: pd.DataFrame, *, freq: str = "D") -> pd.DataFrame:
        """
        Reindex to a full calendar (default daily) and forward-fill prices.

        Important:
        - We DO NOT fill values before the first valid observation of each column.
        - This creates 0-returns on non-trading days for instruments that don't trade daily.
        """
        if prices is None or prices.empty:
            return pd.DataFrame() if prices is None else prices.copy()

        px = prices.sort_index()
        start = px.index.min()
        end = px.index.max()
        if pd.isna(start) or pd.isna(end):
            return px.copy()

        full_idx = pd.date_range(start=start.normalize(), end=end.normalize(), freq=freq)
        filled = px.reindex(full_idx).ffill()

        # prevent forward-filling "into the past" before each series begins
        first_valid = px.apply(lambda s: s.first_valid_index())
        for col, first in first_valid.items():
            if first is None:
                continue
            filled.loc[filled.index < pd.Timestamp(first).normalize(), col] = np.nan

        return filled

    @staticmethod
    def resample_prices(prices: pd.DataFrame, *, freq: str = "ME") -> pd.DataFrame:
        """
        Resample prices to a lower frequency (default month-end).

        Notes:
        - We assume prices are already on a daily calendar (see fill_non_trading_days).
        - Using .last() on a forward-filled daily calendar approximates "month-end close"
          even for instruments that don't trade every day.
        """
        if prices is None or prices.empty:
            return pd.DataFrame() if prices is None else prices.copy()
        px = prices.sort_index()
        if not isinstance(px.index, pd.DatetimeIndex):
            raise TypeError("prices index must be a DatetimeIndex to resample")
        return px.resample(freq).last()

    @staticmethod
    def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        """
        Compute returns from a price DataFrame (frequency is whatever the index represents).

        - method="log": log-returns (recommended for correlations)
        - method="pct": simple percentage returns
        """
        if prices is None or prices.empty:
            return pd.DataFrame() if prices is None else prices.copy()
        px = prices.sort_index()
        if method == "log":
            px_num = px.apply(pd.to_numeric, errors="coerce")
            return np.log(px_num).diff()
        if method == "pct":
            return px.pct_change()
        raise ValueError(f"Unknown return method: {method}")

    @staticmethod
    def common_start_info(*price_dfs: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.DataFrame]:
        """
        Compute the shared (most recent) start date across multiple price DataFrames.

        Returns:
          - common_start: the max of each column's first valid timestamp
          - starts: DataFrame indexed by column name with a 'Start' column
        """
        if not price_dfs:
            return None, pd.DataFrame(columns=["Start"])
        combined = pd.concat([df for df in price_dfs if df is not None and not df.empty], axis=1)
        if combined.empty:
            return None, pd.DataFrame(columns=["Start"])
        starts = combined.apply(lambda s: s.first_valid_index())
        starts = starts.dropna()
        starts_df = pd.DataFrame({"Start": starts}).sort_values("Start", ascending=False)
        if starts_df.empty:
            return None, starts_df
        common_start = starts_df["Start"].max()
        return pd.Timestamp(common_start), starts_df

    @staticmethod
    def monthly_returns_from_prices(
        prices: pd.DataFrame,
        *,
        return_method: str = "log",
        common_start: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Convert a (possibly irregular) price panel to month-end returns:
          daily calendar (ffill) -> month-end prices -> returns
        """
        if prices is None or prices.empty:
            return pd.DataFrame()
        px_d = Portfolio.fill_non_trading_days(prices, freq="D")
        px_me = Portfolio.resample_prices(px_d, freq="ME")
        if common_start is not None:
            px_me = px_me.loc[px_me.index >= pd.Timestamp(common_start)]
        rets = Portfolio.compute_returns(px_me, method=return_method)
        return rets.dropna(how="all")

    @staticmethod
    def backtest_value_series(
        prices: pd.DataFrame,
        weights: dict[str, float],
        *,
        rebalance_frequency: str = "annually",
        initial_value: float = 1.0,
    ) -> pd.Series:
        """
        Rebalance-to-target backtest on a provided daily price panel.

        Uses buy-and-hold between rebalances, and rebalances to fixed weights at
        month/quarter/year-end.
        """
        if prices is None or prices.empty:
            return pd.Series(dtype=float, name="portfolio_value")
        if not weights:
            return pd.Series(dtype=float, name="portfolio_value")

        cols = list(weights.keys())
        px = prices[cols].copy().sort_index()
        px = px.dropna(how="any")
        if px.empty:
            return pd.Series(dtype=float, name="portfolio_value")

        w = pd.Series(weights, dtype=float)
        if (w < 0).any() or float(w.sum()) <= 0:
            raise ValueError("Weights must be non-negative and sum to > 0.")
        w = w / float(w.sum())
        wv = w.reindex(cols).to_numpy(dtype=float)

        freq = str(rebalance_frequency).strip().lower()
        freq_map = {"monthly": "ME", "quarterly": "QE", "annually": "YE", "annual": "YE", "yearly": "YE"}
        if freq not in freq_map:
            raise ValueError("rebalance_frequency must be one of: monthly, quarterly, annually")

        rebalance_dates = set(px.resample(freq_map[freq]).last().index)
        if px.index[0] in rebalance_dates:
            rebalance_dates.remove(px.index[0])

        p0 = px.iloc[0].to_numpy(dtype=float)
        shares = (float(initial_value) * wv) / p0
        values: list[float] = []
        for dt, row in px.iterrows():
            p = row.to_numpy(dtype=float)
            v = float(np.dot(shares, p))
            values.append(v)
            if dt in rebalance_dates:
                shares = (v * wv) / p
        return pd.Series(values, index=px.index, name="portfolio_value")

    @staticmethod
    def max_drawdown(value: pd.Series) -> float:
        """
        Max drawdown computed from a value/equity series.
        Returns a negative number (e.g. -0.25 for -25%).
        """
        if value is None or value.empty:
            return float("nan")
        v = pd.to_numeric(value, errors="coerce").dropna()
        if v.empty:
            return float("nan")
        peak = v.cummax()
        dd = v / peak - 1.0
        return float(dd.min())

    @staticmethod
    def max_gain(value: pd.Series) -> float:
        """
        Max gain from the initial value, computed from a value/equity series.
        Returns a non-negative number (e.g. 0.40 for +40%).
        """
        if value is None or value.empty:
            return float("nan")
        v = pd.to_numeric(value, errors="coerce").dropna()
        if v.empty:
            return float("nan")
        base = float(v.iloc[0])
        if not np.isfinite(base) or base <= 0:
            return float("nan")
        gain = v / base - 1.0
        return float(gain.max())

    @staticmethod
    def ulcer_index(value: pd.Series) -> float:
        """
        Ulcer Index: measures downside volatility by computing the quadratic mean
        of percentage drawdowns from peak.
        
        UI = sqrt(mean(drawdown^2))
        
        Lower is better. Values typically range from 0 (no drawdowns) to 20+ (severe).
        """
        if value is None or value.empty:
            return float("nan")
        v = pd.to_numeric(value, errors="coerce").dropna()
        if v.empty or len(v) < 2:
            return float("nan")
        
        # Compute running maximum (peak)
        peak = v.cummax()
        # Percentage drawdown from peak (as positive values for squaring)
        drawdown_pct = ((peak - v) / peak) * 100.0
        # Ulcer Index = sqrt(mean(drawdown^2))
        ui = float(np.sqrt((drawdown_pct ** 2).mean()))
        return ui

    @staticmethod
    def backtest_stats(
        value: pd.Series,
        *,
        rf_annual: float = 0.0,
        mar_annual: float | None = None,
        trading_days_per_year: int = 252,
    ) -> dict[str, float]:
        """
        Compute stats from a value series using DAILY LOG RETURNS for vol/Sharpe/Sortino,
        plus total return, max drawdown, and max gain computed from the value series.
        """
        if mar_annual is None:
            mar_annual = rf_annual
        if value is None or value.empty or len(value) < 3:
            return {
                "total_return": float("nan"),
                "cagr": float("nan"),
                "vol_annual": float("nan"),
                "sharpe": float("nan"),
                "sortino": float("nan"),
                "max_drawdown": float("nan"),
                "ulcer_index": float("nan"),
            }
        v = pd.to_numeric(value, errors="coerce").dropna()
        if len(v) < 3:
            return {
                "total_return": float("nan"),
                "cagr": float("nan"),
                "vol_annual": float("nan"),
                "sharpe": float("nan"),
                "sortino": float("nan"),
                "max_drawdown": float("nan"),
                "ulcer_index": float("nan"),
            }

        total_return = float(v.iloc[-1] / v.iloc[0] - 1.0)
        years = (v.index[-1] - v.index[0]).days / 365.25
        cagr = float((v.iloc[-1] / v.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

        r = np.log(v / v.shift(1)).dropna()
        vol_annual = float(r.std(ddof=1) * np.sqrt(float(trading_days_per_year))) if not r.empty else float("nan")

        rf_cc_daily = float(np.log1p(float(rf_annual)) / float(trading_days_per_year))
        excess = r - rf_cc_daily
        sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(float(trading_days_per_year))) if r.std(ddof=1) > 0 else float("nan")

        mar_cc_daily = float(np.log1p(float(mar_annual)) / float(trading_days_per_year))
        downside = np.minimum(0.0, r - mar_cc_daily)
        downside_dev_annual = float(np.sqrt((downside**2).mean()) * np.sqrt(float(trading_days_per_year)))
        sortino = float(excess.mean() * float(trading_days_per_year) / downside_dev_annual) if downside_dev_annual > 0 else float("nan")

        mdd = Portfolio.max_drawdown(v)
        mg = Portfolio.max_gain(v)
        ui = Portfolio.ulcer_index(v)

        return {
            "total_return": total_return,
            "cagr": cagr,
            "vol_annual": vol_annual,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": mdd,
            "max_gain": mg,
            "ulcer_index": ui,
        }

    @staticmethod
    def annualize_mean_std(returns: pd.Series, *, periods_per_year: int) -> tuple[float, float]:
        """
        Annualize mean and standard deviation from a periodic return series.

        Uses the common approximation:
          mu_annual  = mean(r) * periods_per_year
          vol_annual = std(r)  * sqrt(periods_per_year)
        """
        r = pd.to_numeric(returns, errors="coerce").dropna()
        if r.empty:
            return float("nan"), float("nan")
        mu = float(r.mean()) * float(periods_per_year)
        vol = float(r.std(ddof=1)) * float(np.sqrt(float(periods_per_year)))
        return mu, vol

    @staticmethod
    def return_to_risk_ratio(mu: float, vol: float) -> float:
        """
        Return-to-risk ratio (RRR) = mu / vol (Sharpe-like but without rf subtraction).
        """
        if not np.isfinite(mu) or not np.isfinite(vol) or vol <= 0:
            return float("nan")
        return float(mu / vol)

    @staticmethod
    def rrr_combination(
        *,
        mu_p: float,
        vol_p: float,
        mu_a: float,
        vol_a: float,
        rho: float,
        w_a: float,
    ) -> float:
        """
        Paper formula (Portfolio Intuition, Kennedy 2018):

          RRR_pa = (w_p * r_p + w_a * r_a) / sqrt(w_p^2 * σ_p^2 + w_a^2 * σ_a^2 + 2*w_p*w_a*σ_p*σ_a*ρ)

        Here we interpret:
          r_p, r_a  as annualized mean returns (mu_p, mu_a)
          σ_p, σ_a  as annualized volatilities (vol_p, vol_a)
          ρ         as correlation of return series (same periodicity used to estimate vols)
          w_a       as weight of the new asset post-allocation; w_p = 1 - w_a
        """
        w_a = float(w_a)
        if w_a <= 0 or w_a >= 1:
            return float("nan")
        w_p = 1.0 - w_a
        if not all(np.isfinite(x) for x in [mu_p, vol_p, mu_a, vol_a, rho]):
            return float("nan")
        if vol_p <= 0 or vol_a <= 0:
            return float("nan")
        num = w_p * float(mu_p) + w_a * float(mu_a)
        den_sq = (w_p**2) * (float(vol_p) ** 2) + (w_a**2) * (float(vol_a) ** 2) + 2.0 * w_p * w_a * float(vol_p) * float(vol_a) * float(rho)
        if den_sq <= 0:
            return float("nan")
        return float(num / np.sqrt(den_sq))

    @classmethod
    def from_json(cls, path: str) -> "Portfolio":
        """
        Load a portfolio from a JSON file.

        Expected schema (extended):
        {
          "Name": "Model Portfolio",
          "Assets": [
            {"Name": "Stocks", "Ticker": "ACWE.MI", "Weight": 70.0, "Target": 70.0},
            ...
          ],
          "Value": 80200.0
        }
        """
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        name = str(obj.get("Name", "Portfolio"))
        assets_list = obj.get("Assets")
        if not isinstance(assets_list, list) or not assets_list:
            raise ValueError(f"Invalid portfolio json: missing/non-list 'Assets' in {path}")

        tickers: list[str] = []
        weights: list[float] = []  # current/actual weights
        targets: list[float] = []  # target weights
        assets: list[str] = []     # asset class names
        for i, a in enumerate(assets_list):
            if not isinstance(a, dict):
                raise ValueError(f"Invalid asset entry at index {i}: expected object, got {type(a)}")
            asset_name = a.get("Name")
            ticker = a.get("Ticker")
            weight = a.get("Weight")
            target = a.get("Target")
            if asset_name is None or ticker is None or weight is None or target is None:
                raise ValueError(f"Invalid asset entry at index {i}: requires Name/Ticker/Weight/Target")
            assets.append(str(asset_name))
            tickers.append(str(ticker))
            weights.append(float(weight))
            targets.append(float(target))

        # Validate that the JSON percentages are well-formed.
        # We keep a small tolerance because weights are often rounded to 1-2 decimals.
        w_sum = float(sum(weights))
        t_sum = float(sum(targets))
        tol = 0.25  # percentage points
        if abs(w_sum - 100.0) > tol:
            raise ValueError(
                f"Invalid portfolio json in {path}: current weights ('Weight') must sum to 100; got {w_sum:.4f}"
            )
        if abs(t_sum - 100.0) > tol:
            raise ValueError(
                f"Invalid portfolio json in {path}: target weights ('Target') must sum to 100; got {t_sum:.4f}"
            )

        p = cls(tickers, weights, assets=assets)
        p.name = name
        p.current_value_eur = float(obj.get("Value")) if obj.get("Value") is not None else None
        p.actual_weights_pct = {t: float(w) for t, w in zip(tickers, weights)}
        p.target_weights_pct = {t: float(w) for t, w in zip(tickers, targets)}
        return p

    @staticmethod
    def _normalize_weights_to_fraction(weights: list[float]) -> np.ndarray:
        """
        Normalize weights to fractions that sum to 1.
        Accepts either percent-like weights summing to ~100 or fractions summing to ~1.
        """
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative.")
        total = float(w.sum())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        # If it looks like percent weights, convert to fraction.
        if total > 1.5:
            w = w / 100.0
            total = float(w.sum())
        return w / total

    @staticmethod
    def _allocate_new_cash(current_value: float, new_cash: float, wc: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mathematically optimal cash allocation (no selling):
        project the desired changes onto the simplex {d >= 0, sum(d) = new_cash}.
        Equivalent to dashboard.allocate_new_cash().
        """
        X = float(current_value)
        Y = float(new_cash)
        a = X * wc
        b = (X + Y) * wt - a

        u = b.copy()
        u_sorted = np.sort(u)[::-1]
        css = np.cumsum(u_sorted)
        idx = np.nonzero(u_sorted - (css - Y) / (np.arange(1, u.size + 1)) > 0)[0]
        if idx.size == 0:
            theta = (css[0] - Y) / 1.0
        else:
            rho = int(idx[-1])
            theta = (css[rho] - Y) / float(rho + 1)
        delta = np.maximum(0.0, b - theta)
        if float(delta.sum()) > 0:
            delta *= Y / float(delta.sum())
        new_amounts = a + delta
        new_weights = new_amounts / (X + Y)
        return delta, new_amounts, new_weights

    def rebalance(self, new_cash: float) -> pd.DataFrame:
        """
        Compute the cash allocation across assets to move from current weights to target weights,
        using a no-selling, simplex-projection allocation (optimal under a pure rebalancing objective).

        Requires:
          - self.current_value_eur (from JSON 'Value')
          - self.actual_weights_pct, self.target_weights_pct (from JSON 'Weight'/'Target')
        """
        if getattr(self, "current_value_eur", None) is None:
            raise ValueError("Portfolio JSON is missing top-level 'Value' (current portfolio value in EUR).")
        if new_cash is None or float(new_cash) <= 0:
            raise ValueError("new_cash must be > 0.")
        if not hasattr(self, "actual_weights_pct") or not hasattr(self, "target_weights_pct"):
            raise ValueError("Portfolio JSON must include per-asset 'Weight' and 'Target'.")

        current_value = float(self.current_value_eur)
        labels = [self._label(t) for t in self.tickers]
        wc_pct = [float(self.actual_weights_pct[t]) for t in self.tickers]
        wt_pct = [float(self.target_weights_pct[t]) for t in self.tickers]

        wc = self._normalize_weights_to_fraction(wc_pct)
        wt = self._normalize_weights_to_fraction(wt_pct)

        delta, _, new_w = self._allocate_new_cash(current_value, float(new_cash), wc=wc, wt=wt)

        out = pd.DataFrame(
            {
                "Current Weight (%)": np.asarray(wc) * 100.0,
                "Target Weight (%)": np.asarray(wt) * 100.0,
                "Cash Allocation (EUR)": delta,
                "New Weight (%)": np.asarray(new_w) * 100.0,
            },
            index=labels,
        )
        out.index.name = None
        return out

    def _prices_df(self) -> pd.DataFrame:
        # Normalize internal prices to a DataFrame with columns=self.tickers.
        if isinstance(self.prices, pd.Series):
            return self.prices.to_frame(name=self.tickers[0])
        return self.prices[self.tickers].copy()

    def _label(self, ticker: str) -> str:
        return self.display_labels.get(ticker, ticker)

    def _monthly_log_returns(self, debug: bool = False) -> pd.DataFrame:
        """
        Monthly log returns over the adjusted date window.
        (In practice, correlations are typically computed on returns, not raw prices.)
        """
        self.adjust_dates(debug=debug)
        prices_df = self._prices_df().dropna(how="any")
        # Pandas prefers explicit month-end frequency.
        monthly_prices = prices_df.resample("ME").last().dropna(how="any")
        if len(monthly_prices) < 2:
            raise ValueError("Not enough monthly data points to compute correlations.")
        return np.log(monthly_prices / monthly_prices.shift(1)).dropna(how="any")

    def correlation_matrix_monthly(self, debug: bool = False) -> pd.DataFrame:
        """
        Correlation matrix across all assets over the adjusted date window,
        using monthly log returns.
        """
        r_m = self._monthly_log_returns(debug=debug)
        corr = r_m.corr()
        label_map = {t: self._label(t) for t in corr.columns}
        corr = corr.rename(index=label_map, columns=label_map)
        # Avoid printing "Ticker" twice when displaying the matrix.
        corr.index.name = None
        corr.columns.name = None
        return corr

    def rolling_corr_vs_stocks(self, window_months: int = 12, debug: bool = False) -> pd.DataFrame:
        """
        Rolling correlation (monthly, window=window_months) between each asset and a 'stocks' benchmark.
        Stocks benchmark is built as an equal-weighted average of monthly log returns of tickers
        whose asset class is 'stocks' (case-insensitive).
        
        If no stocks are in the portfolio, downloads external stocks data (ACWE.MI) for comparison.

        Returns:
          DataFrame indexed by month-end dates, columns=tickers (non-stocks only by default).
        """
        # Default stocks ticker to use when portfolio has no stocks
        DEFAULT_STOCKS_TICKER = "ACWE.MI"
        
        r_m = self._monthly_log_returns(debug=debug)
        stock_tickers = [t for t in self.tickers if self.assets.get(t, "").lower() == "stocks"]
        
        if not stock_tickers:
            # No stocks in portfolio - download external stocks data
            try:
                stocks_prices = Portfolio.download_prices([DEFAULT_STOCKS_TICKER])
                if stocks_prices.empty or DEFAULT_STOCKS_TICKER not in stocks_prices.columns:
                    return pd.DataFrame()
                # Compute monthly returns for external stocks
                stocks_d = Portfolio.fill_non_trading_days(stocks_prices, freq="D")
                stocks_me = Portfolio.resample_prices(stocks_d, freq="ME")
                stocks_returns = Portfolio.compute_returns(stocks_me, method="log")
                stocks_benchmark = stocks_returns[DEFAULT_STOCKS_TICKER].dropna()
                # Align with portfolio returns index
                stocks_benchmark = stocks_benchmark.reindex(r_m.index)
                # All portfolio assets should be correlated vs stocks
                non_stock_tickers = list(self.tickers)
            except Exception:
                return pd.DataFrame()
        else:
            stocks_benchmark = r_m[stock_tickers].mean(axis=1)
            non_stock_tickers = [t for t in self.tickers if t not in stock_tickers]

        out: dict[str, pd.Series] = {}
        for t in non_stock_tickers:
            if t not in r_m.columns:
                continue
            out[self._label(t)] = r_m[t].rolling(window_months).corr(stocks_benchmark)

        return pd.DataFrame(out)

    def _print_ticker_date_ranges(self) -> None:
        # Prints first/last valid dates for each ticker's Close series.
        if isinstance(self.prices, pd.Series):
            t = self.tickers[0] if self.tickers else "<unknown>"
            start = self.prices.first_valid_index()
            end = self.prices.last_valid_index()
            start_s = pd.Timestamp(start).date().isoformat() if start is not None else "None"
            end_s = pd.Timestamp(end).date().isoformat() if end is not None else "None"
            print(f"[debug] {t}: start={start_s} end={end_s}")
            return

        for t in self.tickers:
            s = self.prices[t]
            start = s.first_valid_index()
            end = s.last_valid_index()
            start_s = pd.Timestamp(start).date().isoformat() if start is not None else "None"
            end_s = pd.Timestamp(end).date().isoformat() if end is not None else "None"
            print(f"[debug] {t}: start={start_s} end={end_s}")

    def __str__(self) -> str:
        lines: list[str] = []
        for ticker, asset in self.assets.items():
            last_price = self.last_prices[ticker]
            weight = self.weights_pct[ticker]
            lines.append(
                f"{weight:6.2f}% {asset} ({ticker}): Last Price = {last_price:.4f} EUR"
            )
        lines.append("-" * 10)
        lines.append("Total: 100.00%")
        return "\n".join(lines)

    # Computes shared start/end date where all tickers have data and slices data accordingly.
    def adjust_dates(self, debug: bool = True) -> tuple[date, date]:
        """
        Compute the date range where *all* tickers have valid (non-NaN) prices.

        Side effects:
          - stores the result in self.start_date / self.end_date (pd.Timestamp)
          - slices self.data and self.prices to that common window
          - drops any remaining NaNs within the window

        Returns:
          (date, date): (common_start, common_end) as YYYY-MM-DD
        """

        if debug:
            self._print_ticker_date_ranges()

        if isinstance(self.prices, pd.Series):
            start = self.prices.first_valid_index()
            end = self.prices.last_valid_index()
            if start is None:
                raise ValueError("No valid price data found for the requested ticker(s).")
            if end is None:
                raise ValueError("No valid price data found for the requested ticker(s).")
            self.start_date = pd.Timestamp(start)
            self.end_date = pd.Timestamp(end)
            self.prices = self.prices.loc[self.start_date : self.end_date].dropna()
            self.data = self.data.loc[self.start_date : self.end_date].copy()
            return (self.start_date.date(), self.end_date.date())

        # DataFrame case (multiple tickers): common start is latest first-valid; common end is earliest last-valid.
        missing_cols = [t for t in self.tickers if t not in self.prices.columns]
        if missing_cols:
            raise KeyError(f"Missing ticker columns in downloaded data: {missing_cols}")

        first_valid_dates: list[pd.Timestamp] = []
        last_valid_dates: list[pd.Timestamp] = []
        for t in self.tickers:
            s = self.prices[t]
            d0 = s.first_valid_index()
            d1 = s.last_valid_index()
            if d0 is None:
                raise ValueError(f"No valid price data found for ticker '{t}'.")
            if d1 is None:
                raise ValueError(f"No valid price data found for ticker '{t}'.")
            first_valid_dates.append(pd.Timestamp(d0))
            last_valid_dates.append(pd.Timestamp(d1))

        self.start_date = max(first_valid_dates)
        self.end_date = min(last_valid_dates)
        if self.end_date < self.start_date:
            raise ValueError(
                f"No overlapping date range across tickers (start={self.start_date.date()}, end={self.end_date.date()})."
            )

        # Slice to common window; ensure all tickers are present and aligned.
        self.prices = self.prices.loc[self.start_date : self.end_date, self.tickers].copy()
        self.prices = self.prices.dropna(how="any")
        self.data = self.data.loc[self.start_date : self.end_date].copy()
        return (self.start_date.date(), self.end_date.date())

    def plot(
        self,
        X: float = 10_000.0,
        rf_annual: float = 0.0,
        mar_annual: float | None = None,
        rebalance_frequency: str = "annually",
    ):
        """
        Plot the value of a cash amount X invested from the common start date to now.
        - Portfolio value is computed using current portfolio weights (from holdings) and
          rebalanced implicitly at each point in time:
            V(t) = X * sum_i w_i * (P_i(t) / P_i(t0))
        - Individual assets are plotted as "all-in" series (X invested in each asset) using
          lighter/thinner lines to emphasize the portfolio.
        Also annotates max drawdown and max gain (from initial value) for the portfolio series.
        """
        self.plot_value(
            X=float(X),
            rf_annual=float(rf_annual),
            mar_annual=mar_annual,
            rebalance_frequency=rebalance_frequency,
            show=True,
        )

    def _portfolio_value_series(
        self,
        X: float = 1.0,
        rebalance_frequency: str = "annually",
        debug: bool = False,
    ) -> pd.Series:
        """
        Compute the portfolio value series from the common window using stored weights and periodic rebalancing.

        Rebalancing:
          - "monthly": rebalance at month-end
          - "quarterly": rebalance at quarter-end
          - "annually": rebalance at year-end

        Within each period, holdings are buy-and-hold; at rebalance dates, holdings are reset to the fixed weights.
        """
        self.adjust_dates(debug=debug)

        if isinstance(self.prices, pd.Series):
            prices_df = self.prices.to_frame(name=self.tickers[0])
            tickers = [self.tickers[0]]
        else:
            prices_df = self.prices[self.tickers].copy()
            tickers = list(self.tickers)

        prices_df = prices_df.dropna(how="any").sort_index()
        if prices_df.empty:
            raise ValueError("No overlapping price history after aligning to common date window.")

        # Weights vector aligned to price columns
        w = np.asarray([float(self.weights[t]) for t in tickers], dtype=float)
        w_sum = float(w.sum())
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Invalid weights for portfolio value series.")
        w = w / w_sum

        freq = str(rebalance_frequency).strip().lower()
        freq_map = {"monthly": "ME", "quarterly": "QE", "annually": "YE", "annual": "YE", "yearly": "YE"}
        if freq not in freq_map:
            raise ValueError("rebalance_frequency must be one of: monthly, quarterly, annually")

        rebalance_dates = set(prices_df.resample(freq_map[freq]).last().index)
        # Avoid an immediate rebalance on the first day.
        if prices_df.index[0] in rebalance_dates:
            rebalance_dates.remove(prices_df.index[0])

        p0 = prices_df.iloc[0].to_numpy(dtype=float)
        if np.any(~np.isfinite(p0)) or np.any(p0 <= 0):
            raise ValueError("Invalid starting prices for portfolio simulation.")

        shares = (float(X) * w) / p0
        values: list[float] = []
        for dt, row in prices_df.iterrows():
            p = row.to_numpy(dtype=float)
            v = float(np.dot(shares, p))
            values.append(v)
            if dt in rebalance_dates:
                shares = (v * w) / p

        return pd.Series(values, index=prices_df.index, name="portfolio_value")

    def compute_stats(
        self,
        rf_annual: float = 0.0,
        mar_annual: float | None = None,
        rebalance_frequency: str = "annually",
        debug: bool = False,
    ) -> dict[str, float]:
        """
        Compute common performance statistics using daily log returns (standard in finance literature).

        - Returns used: daily log returns of the portfolio value series.
        - Annualization: 252 trading days/year.
        - CAGR: computed from start/end value over calendar time (365.25 days/year).
        - Sharpe: uses continuously-compounded risk-free rate: r_f,cc_daily = log(1+rf_annual)/252
        - Sortino: downside deviation computed vs MAR (minimum acceptable return). If mar_annual is None, uses rf_annual.

        Returns a dict with:
          - cagr
          - vol_annual
          - sharpe
          - sortino
        """
        if mar_annual is None:
            mar_annual = rf_annual

        v = self._portfolio_value_series(X=1.0, rebalance_frequency=rebalance_frequency, debug=debug)
        if len(v) < 3:
            raise ValueError("Not enough data points to compute statistics.")

        # Daily log returns
        r = np.log(v / v.shift(1)).dropna()
        if r.empty:
            raise ValueError("No returns available to compute statistics.")

        # CAGR over calendar time
        start_ts = v.index[0]
        end_ts = v.index[-1]
        years = (end_ts - start_ts).days / 365.25
        if years <= 0:
            raise ValueError("Invalid date range for CAGR computation.")
        cagr = float((v.iloc[-1] / v.iloc[0]) ** (1.0 / years) - 1.0)

        # Annualized volatility from daily log returns
        vol_annual = float(r.std(ddof=1) * np.sqrt(252.0))

        # Sharpe (daily cc excess returns)
        rf_cc_daily = float(np.log1p(rf_annual) / 252.0)
        excess = r - rf_cc_daily
        sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(252.0)) if r.std(ddof=1) > 0 else float("nan")

        # Sortino
        mar_cc_daily = float(np.log1p(mar_annual) / 252.0)
        downside = np.minimum(0.0, r - mar_cc_daily)
        downside_dev_annual = float(np.sqrt((downside**2).mean()) * np.sqrt(252.0))
        sortino = float(excess.mean() * 252.0 / downside_dev_annual) if downside_dev_annual > 0 else float("nan")

        return {
            "cagr": cagr,
            "vol_annual": vol_annual,
            "sharpe": sharpe,
            "sortino": sortino,
        }

    def plot_value(
        self,
        X: float = 10_000.0,
        rf_annual: float = 0.0,
        mar_annual: float | None = None,
        rebalance_frequency: str = "annually",
        show: bool = True,
        save_path: str | None = None,
        debug: bool = False,
    ):
        # Ensure prices are aligned to the common window.
        self.adjust_dates(debug=debug)

        # Normalize to a DataFrame with columns=self.tickers
        if isinstance(self.prices, pd.Series):
            prices_df = self.prices.to_frame(name=self.tickers[0])
            tickers = [self.tickers[0]]
        else:
            prices_df = self.prices[self.tickers].copy()
            tickers = list(self.tickers)

        # Drop any remaining NaNs after common window alignment (e.g., partial days)
        prices_df = prices_df.dropna(how="any")
        if prices_df.empty:
            raise ValueError("No overlapping price history after aligning to common date window.")

        # Weights (fractions) provided by the user (normalized in __init__)
        weights = {t: float(self.weights[t]) for t in tickers}

        # Growth index per asset and portfolio
        base = prices_df.iloc[0]
        growth = prices_df.divide(base)  # P(t)/P(t0)

        # Individual assets: X invested entirely in each asset (for comparison)
        asset_values = float(X) * growth

        # Portfolio value series with periodic rebalancing
        portfolio_value = self._portfolio_value_series(
            X=float(X),
            rebalance_frequency=rebalance_frequency,
            debug=False,
        )
        # Ensure alignment with the price index for plotting
        portfolio_value = portfolio_value.reindex(asset_values.index).dropna()
        if portfolio_value.empty:
            raise ValueError("Portfolio value series is empty after alignment.")

        # Max gain from initial and max drawdown from peak (portfolio)
        running_max = portfolio_value.cummax()
        drawdown = (portfolio_value / running_max) - 1.0
        max_drawdown = float(drawdown.min())
        dd_date = drawdown.idxmin()

        gain = (portfolio_value / float(X)) - 1.0
        max_gain = float(gain.max())
        gain_date = gain.idxmax()

        # Plot (two panels): value on top, rolling correlations below
        fig, (ax, ax_corr) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(12, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )

        # Individual assets (lighter + thinner)
        for t in tickers:
            ax.plot(
                asset_values.index,
                asset_values[t],
                linewidth=1.0,
                alpha=0.5,
                label=self._label(t),
            )

        # Portfolio (darker + thicker)
        ax.plot(
            portfolio_value.index,
            portfolio_value.values,
            color="black",
            linewidth=2.8,
            label="Portfolio",
            zorder=5,
        )

        # Markers for max drawdown and max gain (portfolio)
        ax.scatter([dd_date], [portfolio_value.loc[dd_date]], color="red", s=30, zorder=6)
        ax.scatter([gain_date], [portfolio_value.loc[gain_date]], color="green", s=30, zorder=6)
        # Start marker
        start_date = portfolio_value.index[0]
        start_val = float(portfolio_value.iloc[0])
        ax.scatter([start_date], [start_val], color="blue", s=28, zorder=6)

        # Annotations
        dd_text = f"Max drawdown: {max_drawdown*100:.2f}%"
        gain_text = f"Max gain: {max_gain*100:.2f}%"
        ax.annotate(
            dd_text,
            xy=(mdates.date2num(dd_date), float(portfolio_value.loc[dd_date])),
            xytext=(-12, -20),
            textcoords="offset points",
            color="red",
            fontsize=10,
            ha="right",
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
        )
        ax.annotate(
            gain_text,
            xy=(mdates.date2num(gain_date), float(portfolio_value.loc[gain_date])),
            xytext=(-12, 12),
            textcoords="offset points",
            color="green",
            fontsize=10,
            ha="right",
            arrowprops=dict(arrowstyle="->", color="green", lw=1),
        )

        # Y axis limits: focus on portfolio range (pad a bit below max drawdown point and above max gain point)
        dd_val = float(portfolio_value.loc[dd_date])
        gain_val = float(portfolio_value.loc[gain_date])
        # Ensure the initial investment point is included in the visible range.
        lo = min(dd_val, gain_val, start_val)
        hi = max(dd_val, gain_val, start_val)
        span = hi - lo
        pad = 0.20 * span if span > 0 else 1.0
        y_min = max(0.0, lo - pad)
        y_max = hi + pad
        ax.set_ylim(y_min, y_max)

        # Stats box (portfolio)
        try:
            stats = self.compute_stats(
                rf_annual=float(rf_annual),
                mar_annual=mar_annual,
                rebalance_frequency=rebalance_frequency,
                debug=False,
            )
            stats_text = (
                f"CAGR:    {stats['cagr']*100:.2f}%\n"
                f"Vol:     {stats['vol_annual']*100:.2f}%\n"
                f"Sharpe:  {stats['sharpe']:.2f}\n"
                f"Sortino: {stats['sortino']:.2f}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="none"),
            )
        except Exception:
            # If stats fail for any reason, keep the plot usable.
            pass

        # Labels / title
        start_d = getattr(self, "start_date", prices_df.index[0]).date().isoformat()
        end_d = getattr(self, "end_date", prices_df.index[-1]).date().isoformat()
        ax.set_title(
            f"Portfolio value (X={float(X):.2f} EUR) from {start_d} to {end_d} "
            f"[rebalance: {str(rebalance_frequency).lower()}]"
        )
        ax.set_ylabel("Value (EUR)")
        ax.grid(True, alpha=0.25)

        # Legend: keep portfolio first
        handles, labels = ax.get_legend_handles_labels()
        if "Portfolio" in labels:
            i = labels.index("Portfolio")
            handles = [handles[i]] + handles[:i] + handles[i+1:]
            labels = [labels[i]] + labels[:i] + labels[i+1:]
        # Legend: place outside so it doesn't overlap the stats box / data.
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
        )

        # Leave room on the right for the external legend.
        # Rolling 12M correlation vs stocks (monthly)
        try:
            rolling = self.rolling_corr_vs_stocks(window_months=12, debug=False)
            if rolling.empty:
                raise ValueError("No non-stocks assets to correlate vs stocks.")
            for col in rolling.columns:
                ax_corr.plot(rolling.index, rolling[col], linewidth=1.4, alpha=0.85, label=col)
            ax_corr.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
            ax_corr.set_ylim(-1.0, 1.0)
            ax_corr.set_ylabel("Corr")
            ax_corr.set_title("Rolling 12M correlation vs stocks (monthly)")
            ax_corr.grid(True, alpha=0.25)
            ax_corr.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
                fontsize=9,
            )
        except Exception as e:
            ax_corr.axis("off")
            ax_corr.text(
                0.01,
                0.5,
                f"Rolling correlation unavailable: {e}",
                transform=ax_corr.transAxes,
                va="center",
                ha="left",
                fontsize=10,
            )

        fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))

        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)