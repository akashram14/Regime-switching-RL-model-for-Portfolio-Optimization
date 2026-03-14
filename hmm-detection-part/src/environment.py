"""
Portfolio Trading Environment (Gymnasium compatible).

Implements the market replay environment from Section 4.5 of the paper:
- State: [(n+1) × T] matrix with portfolio weights, log returns, and vol features
- Action: portfolio weight vector via softmax (long-only, sum=1)
- Reward: Differential Sharpe Ratio (Section 4.3)
- Handles whole-share rebalancing with cash remainder

Reference: Sections 4.1-4.5 of Sood et al. (2023)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class DifferentialSharpe:
    """
    Differential Sharpe Ratio reward calculator.
    
    The Differential Sharpe Ratio D_t is the incremental impact of the
    return R_t on the Sharpe ratio, computed using exponential moving averages.
    
    D_t = (B_{t-1} * ΔA_t - 0.5 * A_{t-1} * ΔB_t) / (B_{t-1} - A_{t-1}^2)^{3/2}
    
    where:
        A_t = A_{t-1} + η * ΔA_t     (EMA of returns)
        B_t = B_{t-1} + η * ΔB_t     (EMA of squared returns)
        ΔA_t = R_t - A_{t-1}
        ΔB_t = R_t^2 - B_{t-1}
    
    η ≈ 1/252 (one trading year)
    
    Reference: Section 4.3 of the paper; Moody et al. (1998)
    """
    
    def __init__(self, eta: float = 1.0 / 252.0):
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns
    
    def reset(self):
        self.A = 0.0
        self.B = 0.0
    
    def __call__(self, portfolio_return: float) -> float:
        """
        Compute the Differential Sharpe Ratio for a given portfolio return.
        
        Args:
            portfolio_return: The portfolio's simple return at this timestep.
            
        Returns:
            D_t: The differential Sharpe ratio reward.
        """
        R_t = portfolio_return
        
        delta_A = R_t - self.A
        delta_B = R_t ** 2 - self.B
        
        denominator = self.B - self.A ** 2
        
        if denominator > 1e-12:
            D_t = (self.B * delta_A - 0.5 * self.A * delta_B) / (denominator ** 1.5)
        else:
            # Early timesteps: reward is simply the return
            D_t = R_t
        
        # Update EMAs
        self.A = self.A + self.eta * delta_A
        self.B = self.B + self.eta * delta_B
        
        return D_t


class PortfolioEnv(gym.Env):
    """
    Portfolio trading environment with market replay.
    
    At each timestep t:
    1. Calculate current portfolio value from held shares + cash
    2. Agent outputs desired portfolio weights
    3. Environment rebalances (whole shares only, remainder -> cash)
    4. Day shifts, new prices arrive
    5. Environment computes new portfolio value and reward
    
    State matrix S_t is [(n+1) × T]:
        Row 0..n-1: [current_weight_i, r_{i,t-1}, ..., r_{i,t-T+1}] for each asset
        Row n:      [cash_weight, vol20_std, vol20_vol60_ratio_std, vix_std]
        (padded with zeros if T > 3 for the last row)
    
    Reference: Sections 4.1, 4.2, 4.5
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        dataset: Dict,
        date_indices: np.ndarray,
        initial_cash: float = 100_000.0,
        lookback: int = 60,
        reward_type: str = "differential_sharpe",
        eta: float = 1.0 / 252.0,
    ):
        """
        Args:
            dataset: Output from data.build_dataset()
            date_indices: Array of dates for this environment instance (e.g., training dates)
            initial_cash: Starting portfolio value in dollars
            lookback: Number of past days T for the state (default 60)
            reward_type: "differential_sharpe" or "log_return"
            eta: Smoothing parameter for Differential Sharpe (1/252)
        """
        super().__init__()
        
        self.dataset = dataset
        self.dates = date_indices
        self.initial_cash = initial_cash
        self.lookback = lookback
        self.reward_type = reward_type
        
        # Asset info
        self.tickers = dataset["tickers"]
        self.n_assets = len(self.tickers)
        
        # Data arrays (full history for lookback access)
        self.all_log_returns = dataset["all_log_returns"]
        self.all_vol_features = dataset["all_vol_features"]
        self.all_prices = dataset["all_sector_prices"]
        
        # Reward calculator
        self.diff_sharpe = DifferentialSharpe(eta=eta)
        
        # Action space: n_assets + 1 continuous values, softmax'd to portfolio weights
        # Finite bounds required by SB3/Gymnasium; softmax is shift-invariant so this is fine
        self.action_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_assets + 1,),  # +1 for cash
            dtype=np.float32,
        )
        
        # Observation space: (n_assets + 1) × lookback matrix, flattened
        self.obs_shape = ((self.n_assets + 1), self.lookback)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_shape[0] * self.obs_shape[1],),
            dtype=np.float32,
        )
        
        # Internal state
        self._current_step = 0
        self._portfolio_value = initial_cash
        self._shares = np.zeros(self.n_assets)
        self._cash = initial_cash
        self._weights = np.zeros(self.n_assets + 1)
        self._weights[-1] = 1.0  # start all cash
    
    def _get_prices_at(self, date) -> np.ndarray:
        """Get sector prices at a given date."""
        return self.all_prices.loc[date].values.astype(np.float64)
    
    def _get_observation(self) -> np.ndarray:
        """
        Build the state matrix S_t.
        
        Shape: (n_assets + 1) × lookback
        - Rows 0..n_assets-1: [current_weight, log_returns over lookback-1 days]
        - Row n_assets: [cash_weight, vol_features (3 values), zeros padding]
        """
        current_date = self.dates[self._current_step]
        
        # Find the lookback window in the full date index
        all_dates = self.all_log_returns.index
        current_pos = all_dates.get_loc(current_date)
        
        # We need lookback-1 days of returns before current_date
        # (first column is the weight)
        lookback_start = max(0, current_pos - self.lookback + 1)
        lookback_dates = all_dates[lookback_start:current_pos + 1]
        
        state = np.zeros(self.obs_shape, dtype=np.float32)
        
        # For each asset: [weight, log_returns...]
        for i, ticker in enumerate(self.tickers):
            state[i, 0] = self._weights[i]
            returns = self.all_log_returns[ticker].loc[lookback_dates].values
            n_ret = min(len(returns), self.lookback - 1)
            # Most recent returns first (after weight)
            state[i, 1:1 + n_ret] = returns[-n_ret:][::-1] if n_ret > 0 else 0.0
        
        # Last row: [cash_weight, vol20_std, vol20/vol60_std, vix_std, ...]
        state[self.n_assets, 0] = self._weights[-1]  # cash weight
        
        vol_cols = self.all_vol_features.columns
        vol_vals = self.all_vol_features.loc[current_date].values
        n_vol = min(len(vol_vals), self.lookback - 1)
        state[self.n_assets, 1:1 + n_vol] = vol_vals[:n_vol]
        
        return state.flatten()
    
    def _rebalance(self, target_weights: np.ndarray):
        """
        Rebalance portfolio to target weights using whole shares.
        
        Args:
            target_weights: Array of length (n_assets + 1), sums to 1.
                            Last element is cash weight.
        """
        current_date = self.dates[self._current_step]
        prices = self._get_prices_at(current_date)
        
        # Calculate target dollar amounts
        target_values = target_weights[:self.n_assets] * self._portfolio_value
        
        # Convert to whole shares (floor)
        new_shares = np.floor(target_values / (prices + 1e-10))
        
        # Actual value allocated to shares
        shares_value = np.sum(new_shares * prices)
        
        # Remaining goes to cash
        self._shares = new_shares
        self._cash = self._portfolio_value - shares_value
        
        # Recompute actual weights after rounding
        shares_values = self._shares * prices
        total = np.sum(shares_values) + self._cash
        
        if total > 0:
            self._weights[:self.n_assets] = shares_values / total
            self._weights[-1] = self._cash / total
        else:
            self._weights = np.zeros(self.n_assets + 1)
            self._weights[-1] = 1.0
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self._current_step = 0
        self._portfolio_value = self.initial_cash
        self._shares = np.zeros(self.n_assets)
        self._cash = self.initial_cash
        self._weights = np.zeros(self.n_assets + 1)
        self._weights[-1] = 1.0  # start all cash
        
        self.diff_sharpe.reset()
        
        obs = self._get_observation()
        info = {"portfolio_value": self._portfolio_value, "weights": self._weights.copy()}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one trading day.
        
        1. Apply softmax to action to get target weights
        2. Rebalance portfolio at current prices
        3. Advance to next day
        4. Calculate new portfolio value and reward
        """
        # 1. Convert action to portfolio weights via softmax
        action = np.array(action, dtype=np.float64)
        exp_a = np.exp(action - np.max(action))  # numerical stability
        target_weights = exp_a / exp_a.sum()
        
        # 2. Rebalance at current prices
        self._rebalance(target_weights)
        
        old_portfolio_value = self._portfolio_value
        
        # 3. Advance to next day
        self._current_step += 1
        terminated = self._current_step >= len(self.dates) - 1
        
        # 4. Calculate new portfolio value at new prices
        if not terminated:
            new_date = self.dates[self._current_step]
            new_prices = self._get_prices_at(new_date)
            self._portfolio_value = np.sum(self._shares * new_prices) + self._cash
            
            # Update weights
            shares_values = self._shares * new_prices
            total = self._portfolio_value
            if total > 0:
                self._weights[:self.n_assets] = shares_values / total
                self._weights[-1] = self._cash / total
        
        # 5. Calculate reward
        portfolio_return = (self._portfolio_value - old_portfolio_value) / (old_portfolio_value + 1e-10)
        
        if self.reward_type == "differential_sharpe":
            reward = self.diff_sharpe(portfolio_return)
        elif self.reward_type == "log_return":
            reward = np.log(1 + portfolio_return + 1e-10)
        else:
            reward = portfolio_return
        
        # Build observation
        obs = self._get_observation() if not terminated else np.zeros(self.obs_shape[0] * self.obs_shape[1], dtype=np.float32)
        
        info = {
            "portfolio_value": self._portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": self._weights.copy(),
            "date": self.dates[self._current_step] if not terminated else self.dates[-1],
        }
        
        return obs, float(reward), terminated, False, info
    
    def render(self):
        pass


def make_env(
    dataset: Dict,
    dates: pd.DatetimeIndex,
    initial_cash: float = 100_000.0,
    lookback: int = 60,
    reward_type: str = "differential_sharpe",
    seed: int = 0,
):
    """Factory function for creating vectorized environments."""
    def _init():
        env = PortfolioEnv(
            dataset=dataset,
            date_indices=dates,
            initial_cash=initial_cash,
            lookback=lookback,
            reward_type=reward_type,
        )
        env.reset(seed=seed)
        return env
    return _init
