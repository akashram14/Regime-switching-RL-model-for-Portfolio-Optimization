"""
Train regime-specialised PPO agents (Mixture-of-Experts architecture).

Trains 3 PPO agents, each with a different reward function:
  - Low-volatility agent  : Sortino reward  (asymmetric upside capture)
  - Med-volatility agent  : Differential Sharpe reward
  - High-volatility agent : Risk Parity reward  (tail-risk minimisation)

All agents see the same 20-dim observation space (no regime features).
Regime routing is handled externally at inference time by the HMM.

Usage:
    python train_regime_agents.py
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env_portfolio import PortfolioEnv


class ProgressCallback(BaseCallback):
    def __init__(self, log_every: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_every:
            self._last_log = self.num_timesteps
            print(f"  Progress: {self.num_timesteps} timesteps")
        return True


REGIME_AGENTS = {
    "low_vol": {
        "reward_type": "sortino",
        "description": "Sortino reward (asymmetric upside capture)",
    },
    "med_vol": {
        "reward_type": "differential_sharpe",
        "description": "Differential Sharpe reward",
    },
    "high_vol": {
        "reward_type": "risk_parity",
        "description": "Risk Parity reward (tail-risk minimisation)",
    },
}


def train_regime_agent(
    *,
    name: str,
    reward_type: str,
    start_date: str,
    end_date: str,
    total_timesteps: int,
    model_dir: Path,
) -> None:
    env = PortfolioEnv(
        features_path="features.csv",
        returns_path="returns.csv",
        use_regimes=False,  # routing is external
        reward_type=reward_type,
        start_date=start_date,
        end_date=end_date,
    )

    obs, _ = env.reset()
    print(
        f"\n{'='*60}\n"
        f"  Training: {name}  ({reward_type} reward)\n"
        f"  Observation dim : {obs.shape[0]}\n"
        f"  Training period : {start_date} \u2192 {end_date}\n"
        f"  Timesteps       : {total_timesteps}\n"
        f"{'='*60}"
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=ProgressCallback(log_every=10_000),
    )

    save_path = model_dir / f"ppo_regime_{name}"
    model.save(str(save_path))
    print(f"  \u2713 Saved: {save_path}")


def main() -> None:
    start_date = "2008-01-01"
    end_date = "2018-12-31"
    total_timesteps = 100_000
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in REGIME_AGENTS.items():
        train_regime_agent(
            name=name,
            reward_type=cfg["reward_type"],
            start_date=start_date,
            end_date=end_date,
            total_timesteps=total_timesteps,
            model_dir=model_dir,
        )

    print(f"\n{'='*60}")
    print("  All regime agents trained successfully!")
    print(f"{'='*60}")
    for name, cfg in REGIME_AGENTS.items():
        print(f"  ppo_regime_{name}.zip  \u2190  {cfg['description']}")


if __name__ == "__main__":
    main()
