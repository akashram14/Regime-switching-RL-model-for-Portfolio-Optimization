"""
Data pipeline & feature engineering.

Fetches S&P 500 sector indices, VIX, and S&P 500 index from Yahoo Finance.
Computes log returns, rolling volatilities (vol20, vol60, vol20/vol60),
and standardizes volatility features using expanding windows (no lookahead).

Reference: Section 4.2 (States) and 5.1 (Data & Features) of the paper.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, Dict, Optional

# S&P 500 sector ETFs (proxies for sector indices)
SECTOR_TICKERS = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

MARKET_TICKERS = {
    "^GSPC": "S&P 500",
    "^VIX": "VIX",
}


def download_data(
    start: str = "2006-01-01",
    end: str = "2022-01-01",
    sector_tickers: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close prices for sector ETFs, S&P 500, and VIX.
    
    Returns:
        sector_prices: DataFrame of sector ETF adjusted close prices
        sp500_prices: Series/DataFrame of S&P 500 close prices
        vix_prices: Series/DataFrame of VIX close values
    """
    if sector_tickers is None:
        sector_tickers = SECTOR_TICKERS

    tickers = list(sector_tickers.keys()) + ["^GSPC", "^VIX"]
    
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    
    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw
    
    sector_prices = close[list(sector_tickers.keys())].copy()
    sp500_prices = close["^GSPC"].copy()
    vix_prices = close["^VIX"].copy()
    
    # Forward fill then drop any remaining NaN rows at the start
    sector_prices = sector_prices.ffill().dropna()
    sp500_prices = sp500_prices.ffill().dropna()
    vix_prices = vix_prices.ffill().dropna()
    
    # Align all DataFrames to common dates
    common_idx = sector_prices.index.intersection(sp500_prices.index).intersection(vix_prices.index)
    sector_prices = sector_prices.loc[common_idx]
    sp500_prices = sp500_prices.loc[common_idx]
    vix_prices = vix_prices.loc[common_idx]
    
    print(f"Data range: {common_idx[0].date()} to {common_idx[-1].date()}, {len(common_idx)} trading days")
    return sector_prices, sp500_prices, vix_prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns: r_t = log(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def compute_volatility_features(
    sp500_prices: pd.Series,
    vix_prices: pd.Series,
) -> pd.DataFrame:
    """
    Compute the three market volatility features from the paper:
    1. vol20: 20-day rolling std of S&P 500 log returns
    2. vol20/vol60: ratio of 20-day to 60-day rolling volatility
    3. VIX: VIX index value
    
    All features are standardized using expanding window statistics
    (mean and std) to prevent information leakage.
    
    Reference: Section 5.1 of the paper.
    """
    sp500_log_ret = compute_log_returns(sp500_prices)
    
    vol20 = sp500_log_ret.rolling(window=20).std()
    vol60 = sp500_log_ret.rolling(window=60).std()
    vol_ratio = vol20 / vol60
    
    vol_features = pd.DataFrame({
        "vol20": vol20,
        "vol20_vol60_ratio": vol_ratio,
        "vix": vix_prices,
    })
    
    # Standardize with expanding window (no lookahead bias)
    expanding_mean = vol_features.expanding(min_periods=60).mean()
    expanding_std = vol_features.expanding(min_periods=60).std()
    
    vol_features_standardized = (vol_features - expanding_mean) / expanding_std
    vol_features_standardized.columns = [c + "_std" for c in vol_features.columns]
    
    return vol_features_standardized


def build_dataset(
    start: str = "2006-01-01",
    end: str = "2022-01-01",
    lookback: int = 60,
    sector_tickers: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Build the complete dataset for the portfolio environment.
    
    Returns a dict containing:
        - sector_prices: aligned sector prices
        - sector_log_returns: log returns for each sector
        - vol_features: standardized volatility features (vol20, vol20/vol60, VIX)
        - sp500_prices: S&P 500 prices
        - tickers: list of sector ticker symbols
        - ticker_names: dict mapping tickers to names
        - lookback: lookback window T
    """
    if sector_tickers is None:
        sector_tickers = SECTOR_TICKERS
    
    sector_prices, sp500_prices, vix_prices = download_data(
        start=start, end=end, sector_tickers=sector_tickers
    )
    
    sector_log_returns = compute_log_returns(sector_prices)
    vol_features = compute_volatility_features(sp500_prices, vix_prices)
    
    # Align everything and drop initial NaN rows from rolling windows
    common_idx = (
        sector_log_returns.dropna().index
        .intersection(vol_features.dropna().index)
    )
    
    # Need at least `lookback` days of history
    common_idx = common_idx[lookback:]
    
    dataset = {
        "sector_prices": sector_prices.loc[common_idx],
        "sector_log_returns": sector_log_returns.loc[common_idx],
        "vol_features": vol_features.loc[common_idx],
        "sp500_prices": sp500_prices.loc[common_idx],
        "all_log_returns": sector_log_returns,  # full history for lookback
        "all_vol_features": vol_features,  # full history for lookback
        "all_sector_prices": sector_prices,  # full history for lookback
        "tickers": list(sector_tickers.keys()),
        "ticker_names": sector_tickers,
        "lookback": lookback,
        "dates": common_idx,
    }
    
    print(f"Dataset built: {len(common_idx)} usable trading days, {len(sector_tickers)} assets")
    return dataset


def split_by_date(
    dataset: Dict,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[Dict, Dict, Dict]:
    """
    Split dataset into train/val/test by date ranges.
    Each split contains the indices into the dataset dates array.
    """
    dates = dataset["dates"]
    
    def get_mask(start, end):
        return (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    
    train_mask = get_mask(train_start, train_end)
    val_mask = get_mask(val_start, val_end)
    test_mask = get_mask(test_start, test_end)
    
    return (
        {"dates": dates[train_mask], "mask": train_mask},
        {"dates": dates[val_mask], "mask": val_mask},
        {"dates": dates[test_mask], "mask": test_mask},
    )


def generate_sliding_windows(
    dataset: Dict,
    train_years: int = 5,
    val_years: int = 1,
    test_years: int = 1,
    start_year: int = 2006,
    end_year: int = 2022,
) -> list:
    """
    Generate sliding window splits as described in Section 5.2.
    Each window: [train_years train | val_years val | test_years test]
    Windows shift by 1 year.
    
    Returns list of (train_split, val_split, test_split) tuples.
    """
    windows = []
    total_window = train_years + val_years + test_years
    
    for year in range(start_year, end_year - total_window + 2):
        train_start = f"{year}-01-01"
        train_end = f"{year + train_years}-01-01"
        val_start = train_end
        val_end = f"{year + train_years + val_years}-01-01"
        test_start = val_end
        test_end = f"{year + total_window}-01-01"
        
        train_split, val_split, test_split = split_by_date(
            dataset, train_start, train_end, val_start, val_end, test_start, test_end
        )
        
        # Only add if all splits have data
        if len(train_split["dates"]) > 0 and len(val_split["dates"]) > 0 and len(test_split["dates"]) > 0:
            windows.append({
                "train": train_split,
                "val": val_split,
                "test": test_split,
                "label": f"Train [{year}-{year+train_years}) | Val {year+train_years} | Test {year+total_window-1}",
            })
    
    print(f"Generated {len(windows)} sliding windows")
    for w in windows:
        print(f"  {w['label']}: train={len(w['train']['dates'])}, val={len(w['val']['dates'])}, test={len(w['test']['dates'])}")
    
    return windows
