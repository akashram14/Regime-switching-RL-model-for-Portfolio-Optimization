"""
Train all agents on the 11-ETF environment (Sood et al. / FinPlan baseline).

Trains:
  1. Baseline PPO          — Differential Sharpe reward (paper reproduction)
  2. Low-vol regime agent  — Sortino reward
  3. Med-vol regime agent  — Differential Sharpe reward
  4. High-vol regime agent — Risk Parity reward

Uses the paper's PPO hyperparameters (Table 1):
  n_steps=756, batch_size=1260, n_epochs=16, gamma=0.9, gae_lambda=0.9, etc.

Usage:
    cd hmm-detection-part
    python train_all.py                           # 500 K steps (fast test)
    python train_all.py --timesteps 7500000       # full paper config
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from src.data import build_dataset, split_by_date
from src.environment import PortfolioEnv, make_env

# Drop XLC (2018) and XLRE (2015) — too new for long backtests
SECTOR_TICKERS = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_schedule(initial_lr: float, final_lr: float):
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


class ProgressCallback(BaseCallback):
    def __init__(self, log_every: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.log_every:
            self._last = self.num_timesteps
            print(f"    {self.num_timesteps:,} timesteps")
        return True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CACHE_PATH = Path("data/dataset_cache.pkl")


def get_dataset() -> dict:
    """Build (or load cached) 11-ETF dataset."""
    if CACHE_PATH.exists():
        print(f"Loading cached dataset from {CACHE_PATH} ...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("Building dataset (downloading from yfinance) ...")
    dataset = build_dataset(start="2006-01-01", end="2022-01-01", lookback=60,
                            sector_tickers=SECTOR_TICKERS)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Cached dataset to {CACHE_PATH}")
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

REGIME_CONFIGS = {
    "baseline": {
        "reward_type": "differential_sharpe",
        "description": "Paper baseline — Differential Sharpe",
    },
    "regime_low_vol": {
        "reward_type": "sortino",
        "description": "Low-vol agent  — Sortino reward",
    },
    "regime_med_vol": {
        "reward_type": "differential_sharpe",
        "description": "Med-vol agent  — Differential Sharpe",
    },
    "regime_high_vol": {
        "reward_type": "risk_parity",
        "description": "High-vol agent — Risk Parity reward",
    },
}


def create_and_train(
    name: str,
    reward_type: str,
    dataset: dict,
    train_dates: pd.DatetimeIndex,
    total_timesteps: int,
    model_dir: Path,
    n_envs: int = 4,
    seed: int = 42,
):
    """Create a PPO agent with the paper's hyperparameters and train it."""
    lookback = dataset["lookback"]

    env_fns = [
        make_env(dataset, train_dates, lookback=lookback,
                 reward_type=reward_type, seed=seed + i)
        for i in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(3e-4, 1e-5),
        n_steps=756,
        batch_size=min(1260, 756 * n_envs),  # paper: 1260
        n_epochs=16,
        gamma=0.9,
        gae_lambda=0.9,
        clip_range=0.25,
        ent_coef=0.0,
        verbose=0,
        seed=seed,
        device="auto",
    )

    print(f"\n{'='*60}")
    print(f"  Training: {name}  ({reward_type})")
    print(f"  Assets  : {len(dataset['tickers'])} sector ETFs")
    print(f"  Obs dim : {vec_env.observation_space.shape}")
    print(f"  Steps   : {total_timesteps:,}")
    print(f"{'='*60}")

    try:
        model.learn(total_timesteps=total_timesteps,
                    callback=ProgressCallback(), progress_bar=True)
    except ImportError:
        model.learn(total_timesteps=total_timesteps,
                    callback=ProgressCallback(), progress_bar=False)

    save_path = model_dir / f"ppo_{name}"
    model.save(str(save_path))
    print(f"  ✓ Saved: {save_path}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train baseline + regime PPO agents")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Training timesteps per agent (paper uses 7 500 000)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel training environments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset()

    # Split: Train 2007-2017, Test 2018-2022
    dates = dataset["dates"]
    train_dates = dates[(dates >= "2007-01-01") & (dates < "2017-01-01")]
    print(f"Training dates: {train_dates[0].date()} → {train_dates[-1].date()} "
          f"({len(train_dates)} days)")

    for name, cfg in REGIME_CONFIGS.items():
        create_and_train(
            name=name,
            reward_type=cfg["reward_type"],
            dataset=dataset,
            train_dates=train_dates,
            total_timesteps=args.timesteps,
            model_dir=model_dir,
            n_envs=args.n_envs,
            seed=args.seed,
        )

    print(f"\n{'='*60}")
    print("  All agents trained!")
    print(f"{'='*60}")
    for name, cfg in REGIME_CONFIGS.items():
        print(f"  models/ppo_{name}.zip  ←  {cfg['description']}")


if __name__ == "__main__":
    main()
