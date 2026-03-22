import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Reward calculators (stateful, reset each episode)
# ---------------------------------------------------------------------------

class DifferentialSharpe:
    """Differential Sharpe Ratio reward (Moody & Saffell, 1998)."""

    def __init__(self, eta: float = 1.0 / 252.0):
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns

    def reset(self):
        self.A = 0.0
        self.B = 0.0

    def __call__(self, R_t: float) -> float:
        delta_A = R_t - self.A
        delta_B = R_t ** 2 - self.B
        denom = self.B - self.A ** 2
        if denom > 1e-12:
            D_t = (self.B * delta_A - 0.5 * self.A * delta_B) / (denom ** 1.5)
        else:
            D_t = R_t
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B
        return float(D_t)


class DifferentialSortino:
    """Differential Sortino Ratio — penalises only downside deviation."""

    def __init__(self, eta: float = 1.0 / 252.0):
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.D = 0.0  # EMA of squared downside returns

    def reset(self):
        self.A = 0.0
        self.D = 0.0

    def __call__(self, R_t: float) -> float:
        delta_A = R_t - self.A
        downside_sq = min(R_t, 0.0) ** 2
        delta_D = downside_sq - self.D
        if self.D > 1e-12:
            D_t = (self.D * delta_A - 0.5 * self.A * delta_D) / (self.D ** 1.5)
        else:
            D_t = R_t
        self.A += self.eta * delta_A
        self.D += self.eta * delta_D
        return float(D_t)


class RiskPenaltyReward:
    """Risk-averse reward for high-volatility regimes — penalises variance."""

    def __init__(self, eta: float = 1.0 / 252.0, risk_aversion: float = 5.0):
        self.eta = eta
        self.risk_aversion = risk_aversion
        self.var_ema = 0.0

    def reset(self):
        self.var_ema = 0.0

    def __call__(self, R_t: float) -> float:
        penalty = self.risk_aversion * R_t ** 2
        if R_t < 0:
            penalty += self.risk_aversion * abs(R_t)
        self.var_ema += self.eta * (R_t ** 2 - self.var_ema)
        return float(R_t - penalty)


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}
    REGIME_COLS = ["p_low_vol", "p_med_vol", "p_high_vol"]

    def __init__(
        self,
        features_path: str = "features.csv",
        returns_path: str = "returns.csv",
        regimes_path: str = "data/regimes.csv",
        use_regimes: bool = True,
        transaction_cost_coef: float = 0.001,
        reward_type: str = "default",
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        super().__init__()

        self.assets = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
        self.n_assets = len(self.assets)
        self.transaction_cost_coef = transaction_cost_coef
        self.use_regimes = use_regimes
        self.reward_type = reward_type

        # Stateful reward calculator (some types are stateless → None)
        _reward_map = {
            "differential_sharpe": DifferentialSharpe,
            "sortino": DifferentialSortino,
            "risk_parity": RiskPenaltyReward,
        }
        self._reward_calc = _reward_map[reward_type]() if reward_type in _reward_map else None

        features_df = pd.read_csv(features_path)
        returns_df = pd.read_csv(returns_path)

        if (
            "Date" not in features_df.columns
            or "Date" not in returns_df.columns
        ):
            raise ValueError("features.csv and returns.csv must include a 'Date' column.")

        features_df["Date"] = pd.to_datetime(features_df["Date"])
        returns_df["Date"] = pd.to_datetime(returns_df["Date"])

        # Keep only overlapping dates and sort.
        merged = pd.merge(features_df, returns_df, on="Date", how="inner", suffixes=("_feat", "_retfile"))

        if self.use_regimes:
            regimes_df = pd.read_csv(regimes_path)
            if "Date" not in regimes_df.columns:
                raise ValueError("regimes.csv must include a 'Date' column.")
            regimes_df["Date"] = pd.to_datetime(regimes_df["Date"])

            missing_regime_cols = [c for c in self.REGIME_COLS if c not in regimes_df.columns]
            if missing_regime_cols:
                raise ValueError(f"Missing regime columns in regimes.csv: {missing_regime_cols}")

            merged = pd.merge(merged, regimes_df[["Date", *self.REGIME_COLS]], on="Date", how="left")

        merged = merged.sort_values("Date").reset_index(drop=True)

        if start_date is not None:
            merged = merged[merged["Date"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            merged = merged[merged["Date"] <= pd.to_datetime(end_date)]
        merged = merged.reset_index(drop=True)

        if merged.empty:
            raise ValueError("No overlapping dates found between features.csv and returns.csv.")
        self.dates = merged["Date"].copy().reset_index(drop=True)

        # Daily asset log returns from returns.csv in fixed asset order.
        missing_ret_cols = [a for a in self.assets if a not in merged.columns]
        if missing_ret_cols:
            raise ValueError(f"Missing return columns in returns.csv: {missing_ret_cols}")
        self.returns = merged[self.assets].astype(np.float32).to_numpy()

        # Rolling volatility feature columns from features.csv in fixed asset order.
        vol_cols = []
        for asset in self.assets:
            asset_vol_cols = [c for c in merged.columns if c.startswith(f"{asset}_vol")]
            asset_vol_cols = sorted(asset_vol_cols)
            vol_cols.extend(asset_vol_cols)
        if not vol_cols:
            raise ValueError("No rolling volatility columns found (expected columns like '<ASSET>_vol20').")

        vol_df = merged[vol_cols].copy()
        vol_df = vol_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        self.vol_features = vol_df.astype(np.float32).to_numpy()
        self.n_vol_features = self.vol_features.shape[1]

        if self.use_regimes:
            regime_df = merged[self.REGIME_COLS].copy()
            regime_df = regime_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
            self.regime_features = regime_df.astype(np.float32).to_numpy()
            self.n_regime_features = self.regime_features.shape[1]
        else:
            self.regime_features = None
            self.n_regime_features = 0

        self.n_steps = len(merged)
        if self.n_steps < 2:
            raise ValueError("Dataset must have at least 2 rows to provide last-day return observations.")

        obs_dim = self.n_assets + self.n_vol_features + self.n_assets + self.n_regime_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Raw action logits; converted to valid portfolio weights via softmax.
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.current_step = 1
        self.prev_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def _get_observation(self, step_idx: int) -> np.ndarray:
        last_day_returns = self.returns[step_idx - 1]
        vol = self.vol_features[step_idx - 1]
        obs_parts = [last_day_returns, vol, self.prev_weights]
        if self.use_regimes:
            obs_parts.append(self.regime_features[step_idx - 1])
        obs = np.concatenate(obs_parts).astype(np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 1
        self.prev_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        if self._reward_calc is not None:
            self._reward_calc.reset()
        obs = self._get_observation(self.current_step)
        info = {"weights": self.prev_weights.copy()}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.n_assets)
        new_weights = self._softmax(action).astype(np.float32)

        asset_returns = self.returns[self.current_step]
        portfolio_return = float(np.dot(new_weights, asset_returns))

        transaction_cost = float(np.sum(np.abs(new_weights - self.prev_weights)))
        tc_penalty = self.transaction_cost_coef * transaction_cost

        # Compute reward based on reward_type
        if self._reward_calc is not None:
            base_reward = self._reward_calc(portfolio_return)
        else:
            base_reward = portfolio_return

        if self.reward_type == "risk_parity":
            reward = base_reward - 2.0 * tc_penalty   # extra turnover penalty
        else:
            reward = base_reward - tc_penalty

        self.prev_weights = new_weights
        self.current_step += 1

        terminated = self.current_step >= self.n_steps
        truncated = False

        obs_step = self.current_step if not terminated else self.n_steps - 1
        obs = self._get_observation(obs_step)

        info = {
            "portfolio_return": portfolio_return,
            "transaction_cost": transaction_cost,
            "weights": new_weights.copy(),
        }

        return obs, float(reward), terminated, truncated, info
