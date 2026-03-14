"""
PPO Agent wrapper using Stable-Baselines3.

Configures the PPO agent with hyperparameters from Table 1 of the paper:
- Learning rate: 3e-4 annealed to 1e-5
- n_steps: 756 per environment
- batch_size: 1260
- n_epochs: 16
- gamma: 0.9
- gae_lambda: 0.9
- clip_range: 0.25
- n_envs: 10 (vectorized SubprocVecEnv)

Reference: Section 5.2 of Sood et al. (2023)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from src.environment import make_env, PortfolioEnv


def linear_schedule(initial_lr: float, final_lr: float) -> Callable:
    """
    Linear learning rate schedule: anneals from initial_lr to final_lr.
    
    SB3 calls this with progress_remaining ∈ [1.0, 0.0].
    """
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


class ValidationCallback(BaseCallback):
    """
    Callback that evaluates the agent on validation data periodically.
    Records mean episode reward for model selection.
    """
    
    def __init__(
        self,
        val_env: PortfolioEnv,
        eval_freq: int = 50_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.best_model_path = None
        self.eval_results = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward = self._evaluate()
            self.eval_results.append({
                "timestep": self.num_timesteps,
                "mean_reward": mean_reward,
            })
            if self.verbose > 0:
                print(f"  [Eval @ {self.num_timesteps}] Mean reward: {mean_reward:.6f}")
        return True
    
    def _evaluate(self, n_eval_episodes: int = 1) -> float:
        """Run agent on validation environment and return mean reward."""
        total_rewards = []
        
        for _ in range(n_eval_episodes):
            obs, _ = self.val_env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.val_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)


def create_ppo_agent(
    dataset: Dict,
    train_dates: pd.DatetimeIndex,
    n_envs: int = 10,
    initial_cash: float = 100_000.0,
    lookback: int = 60,
    total_timesteps: int = 7_500_000,
    seed: int = 42,
    initial_lr: float = 3e-4,
    final_lr: float = 1e-5,
    pretrained_model: Optional[PPO] = None,
    verbose: int = 1,
) -> PPO:
    """
    Create a PPO agent with the paper's hyperparameters.
    
    Args:
        dataset: Output from data.build_dataset()
        train_dates: DatetimeIndex of training dates
        n_envs: Number of vectorized environments (default 10)
        initial_cash: Starting portfolio value
        lookback: Lookback window T for states
        total_timesteps: Total training timesteps
        seed: Random seed
        initial_lr: Initial learning rate
        final_lr: Final learning rate (annealed)
        pretrained_model: Optional pretrained model to warm-start from
        verbose: Verbosity level
    
    Returns:
        PPO agent (untrained - call .learn() to train)
    """
    # Create vectorized environments
    env_fns = [
        make_env(dataset, train_dates, initial_cash, lookback, seed=seed + i)
        for i in range(n_envs)
    ]
    
    # Use DummyVecEnv for simplicity (SubprocVecEnv can have pickling issues)
    vec_env = DummyVecEnv(env_fns)
    
    # PPO hyperparameters from Table 1
    n_steps = 756  # Rollout buffer: ~3 years of trading days per env
    batch_size = 1260  # ~5 years of trading days
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(initial_lr, final_lr),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=16,
        gamma=0.9,
        gae_lambda=0.9,
        clip_range=0.25,
        ent_coef=0.0,
        verbose=verbose,
        seed=seed,
        device="auto",
    )
    
    # Warm-start from pretrained model if provided
    if pretrained_model is not None:
        model.set_parameters(pretrained_model.get_parameters())
    
    return model


def train_agent(
    model: PPO,
    total_timesteps: int = 7_500_000,
    val_env: Optional[PortfolioEnv] = None,
    eval_freq: int = 100_000,
    verbose: int = 1,
) -> PPO:
    """
    Train the PPO agent.
    
    Args:
        model: PPO agent
        total_timesteps: Total training timesteps
        val_env: Optional validation environment for periodic evaluation
        eval_freq: How often to evaluate (in timesteps)
        verbose: Verbosity level
    
    Returns:
        Trained PPO agent
    """
    callbacks = []
    if val_env is not None:
        callbacks.append(ValidationCallback(val_env, eval_freq=eval_freq, verbose=verbose))
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=True,
        )
    except ImportError:
        # Fallback if tqdm/rich not installed
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=False,
        )
    
    return model


def evaluate_agent(
    model: PPO,
    dataset: Dict,
    eval_dates: pd.DatetimeIndex,
    initial_cash: float = 100_000.0,
    lookback: int = 60,
) -> Dict:
    """
    Evaluate a trained PPO agent in deterministic mode.
    
    Returns:
        Dictionary with portfolio values, returns, weights history, etc.
    """
    env = PortfolioEnv(
        dataset=dataset,
        date_indices=eval_dates,
        initial_cash=initial_cash,
        lookback=lookback,
        reward_type="differential_sharpe",
    )
    
    obs, _ = env.reset()
    
    history = {
        "dates": [],
        "portfolio_values": [],
        "daily_returns": [],
        "weights": [],
        "rewards": [],
    }
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        history["dates"].append(info["date"])
        history["portfolio_values"].append(info["portfolio_value"])
        history["daily_returns"].append(info["portfolio_return"])
        history["weights"].append(info["weights"])
        history["rewards"].append(reward)
        
        done = terminated or truncated
    
    history["dates"] = pd.DatetimeIndex(history["dates"])
    history["portfolio_values"] = np.array(history["portfolio_values"])
    history["daily_returns"] = np.array(history["daily_returns"])
    history["weights"] = np.array(history["weights"])
    history["rewards"] = np.array(history["rewards"])
    
    return history
