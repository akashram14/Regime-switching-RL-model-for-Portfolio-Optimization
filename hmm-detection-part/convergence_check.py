"""
Convergence check: train baseline PPO at multiple checkpoints and evaluate.
Tests whether 500K steps is sufficient vs the paper's 7.5M.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data import build_dataset, split_by_date
from src.environment import PortfolioEnv, make_env
from src.evaluation import compute_metrics
from src.hmm import RegimeDetector, compute_hmm_features

SECTOR_TICKERS = {
    "XLB": "Materials", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Technology", "XLP": "Consumer Staples",
    "XLU": "Utilities", "XLV": "Health Care", "XLY": "Consumer Discretionary",
}

CACHE_PATH = Path("data/dataset_cache.pkl")

def get_dataset():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    dataset = build_dataset(start="2006-01-01", end="2022-01-01", lookback=60,
                            sector_tickers=SECTOR_TICKERS)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def linear_schedule(initial_lr, final_lr):
    def schedule(progress_remaining):
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


def evaluate_model(model, dataset, test_dates):
    """Run model on test env and compute Sharpe."""
    env = PortfolioEnv(dataset=dataset, date_indices=test_dates)
    obs, _ = env.reset()
    daily_returns = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        daily_returns.append(info["portfolio_return"])
    m = compute_metrics(np.array(daily_returns), trading_days=252)
    return m


def main():
    dataset = get_dataset()
    dates = dataset["dates"]
    train_dates = dates[(dates >= "2007-01-01") & (dates < "2017-01-01")]
    test_dates = dates[(dates >= "2018-01-01") & (dates < "2022-01-01")]

    lookback = dataset["lookback"]
    n_envs = 4
    seed = 42

    env_fns = [
        make_env(dataset, train_dates, lookback=lookback,
                 reward_type="differential_sharpe", seed=seed + i)
        for i in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(3e-4, 1e-5),
        n_steps=756,
        batch_size=min(1260, 756 * n_envs),
        n_epochs=16,
        gamma=0.9,
        gae_lambda=0.9,
        clip_range=0.25,
        ent_coef=0.0,
        verbose=0,
        seed=seed,
        device="auto",
    )

    # Checkpoints to evaluate
    checkpoints = [50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
    total_trained = 0

    print(f"{'Timesteps':>12s} {'Sharpe':>8s} {'Ann.Ret':>10s} {'Ann.Vol':>10s} {'MaxDD':>10s} {'Sortino':>8s} {'Calmar':>8s}")
    print("-" * 72)

    for ckpt in checkpoints:
        steps_to_train = ckpt - total_trained
        if steps_to_train <= 0:
            continue
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        total_trained = ckpt

        m = evaluate_model(model, dataset, test_dates)
        print(f"{ckpt:>12,d} {m['sharpe_ratio']:>8.3f} {m['annual_return']:>9.2%} "
              f"{m['annual_volatility']:>9.2%} {m['max_drawdown']:>9.2%} "
              f"{m['sortino_ratio']:>8.3f} {m['calmar_ratio']:>8.3f}")

    print("\nPaper baseline target: Sharpe ~0.83, Ann.Ret ~14%, MaxDD ~-29%")


if __name__ == "__main__":
    main()
