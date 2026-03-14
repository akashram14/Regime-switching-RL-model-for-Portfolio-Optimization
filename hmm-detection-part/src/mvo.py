"""
Mean-Variance Optimization (MVO) baseline strategy.

Uses a 60-day rolling lookback to estimate asset means and covariances,
applies Ledoit-Wolf shrinkage, then optimizes for maximum Sharpe ratio
using PyPortfolioOpt.

Reference: Section 5.3 of Sood et al. (2023)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pypfopt import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return


class MVOStrategy:
    """
    Mean-Variance Optimization strategy that maximizes the Sharpe ratio.
    
    At each timestep, uses a rolling lookback window of asset returns to:
    1. Estimate expected returns (sample mean)
    2. Estimate covariance matrix (Ledoit-Wolf shrinkage)
    3. Solve for max Sharpe ratio portfolio weights
    
    Falls back to equal-weight if optimization fails.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        risk_free_rate: float = 0.0,
    ):
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
    
    def get_weights(
        self,
        prices_window: pd.DataFrame,
    ) -> np.ndarray:
        """
        Compute optimal portfolio weights for max Sharpe ratio.
        
        Args:
            prices_window: DataFrame of asset prices over the lookback window.
                          Shape: (lookback, n_assets)
        
        Returns:
            weights: Array of portfolio weights (n_assets,), sums to <= 1.
                     Remainder goes to cash.
        """
        n_assets = prices_window.shape[1]
        
        try:
            # Expected returns (annualized from daily data)
            mu = mean_historical_return(prices_window, frequency=252)
            
            # Covariance matrix with Ledoit-Wolf shrinkage
            S = CovarianceShrinkage(prices_window, frequency=252).ledoit_wolf()
            
            # Ensure positive semi-definite
            eigenvalues, eigenvectors = np.linalg.eigh(S.values)
            eigenvalues = np.maximum(eigenvalues, 0)
            S_psd = pd.DataFrame(
                eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T,
                index=S.index,
                columns=S.columns,
            )
            
            # Add small regularization for numerical stability
            S_psd += np.eye(n_assets) * 1e-8
            
            # Optimize for max Sharpe
            ef = EfficientFrontier(mu, S_psd, weight_bounds=(0, 1))
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned_weights = ef.clean_weights()
            
            weights = np.array([cleaned_weights[col] for col in prices_window.columns])
            
            # Ensure weights sum to <= 1
            total = weights.sum()
            if total > 1.0:
                weights = weights / total
            
            return weights
            
        except Exception as e:
            # Fallback to equal weight if optimization fails
            weights = np.ones(n_assets) / n_assets
            return weights
    
    def run_backtest(
        self,
        dataset: Dict,
        test_dates: pd.DatetimeIndex,
        initial_cash: float = 100_000.0,
    ) -> Dict:
        """
        Run MVO backtest over test dates.
        
        For each day in test_dates:
        1. Get the lookback window of prices before this date
        2. Compute optimal weights via MVO
        3. Rebalance portfolio (whole shares)
        4. Record portfolio value and weights
        
        Returns:
            Dictionary with portfolio values, returns, weights history, etc.
        """
        all_prices = dataset["all_sector_prices"]
        tickers = dataset["tickers"]
        n_assets = len(tickers)
        
        # Initialize portfolio
        portfolio_value = initial_cash
        shares = np.zeros(n_assets)
        cash = initial_cash
        
        # Tracking
        history = {
            "dates": [],
            "portfolio_values": [],
            "daily_returns": [],
            "weights": [],
            "cash_weight": [],
        }
        
        for i, date in enumerate(test_dates):
            # Get lookback window of prices
            all_dates = all_prices.index
            date_pos = all_dates.get_loc(date)
            
            if date_pos < self.lookback:
                # Not enough history, stay in cash
                history["dates"].append(date)
                history["portfolio_values"].append(portfolio_value)
                history["daily_returns"].append(0.0)
                history["weights"].append(np.zeros(n_assets))
                history["cash_weight"].append(1.0)
                continue
            
            prices_window = all_prices.iloc[date_pos - self.lookback:date_pos]
            current_prices = all_prices.loc[date].values.astype(np.float64)
            
            # Calculate current portfolio value
            old_portfolio_value = np.sum(shares * current_prices) + cash
            
            # Get MVO weights
            target_weights = self.get_weights(prices_window)
            
            # Rebalance to whole shares
            target_values = target_weights * old_portfolio_value
            new_shares = np.floor(target_values / (current_prices + 1e-10))
            shares_cost = np.sum(new_shares * current_prices)
            new_cash = old_portfolio_value - shares_cost
            
            shares = new_shares
            cash = new_cash
            
            # If not the last day, compute return based on next day's prices
            if i < len(test_dates) - 1:
                next_date = test_dates[i + 1]
                next_prices = all_prices.loc[next_date].values.astype(np.float64)
                new_portfolio_value = np.sum(shares * next_prices) + cash
            else:
                new_portfolio_value = np.sum(shares * current_prices) + cash
            
            daily_return = (new_portfolio_value - old_portfolio_value) / (old_portfolio_value + 1e-10)
            
            # Actual weights after rebalancing
            actual_weights = (shares * current_prices) / (old_portfolio_value + 1e-10)
            cash_w = cash / (old_portfolio_value + 1e-10)
            
            history["dates"].append(date)
            history["portfolio_values"].append(new_portfolio_value)
            history["daily_returns"].append(daily_return)
            history["weights"].append(actual_weights.copy())
            history["cash_weight"].append(cash_w)
            
            portfolio_value = new_portfolio_value
        
        history["dates"] = pd.DatetimeIndex(history["dates"])
        history["portfolio_values"] = np.array(history["portfolio_values"])
        history["daily_returns"] = np.array(history["daily_returns"])
        history["weights"] = np.array(history["weights"])
        history["cash_weight"] = np.array(history["cash_weight"])
        
        return history
