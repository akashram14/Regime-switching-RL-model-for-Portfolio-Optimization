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
            print(f"Training progress: {self.num_timesteps} timesteps")
        return True


def train_variant(
    *,
    name: str,
    use_regimes: bool,
    start_date: str,
    end_date: str,
    total_timesteps: int,
    model_dir: Path,
) -> None:
    env = PortfolioEnv(
        features_path="features.csv",
        returns_path="returns.csv",
        use_regimes=use_regimes,
        start_date=start_date,
        end_date=end_date,
    )

    # Quick API sanity check before training.
    obs, _ = env.reset()
    action = env.action_space.sample()
    _ = env.step(action)
    print(
        f"{name} environment ready. Observation shape: {obs.shape}, "
        f"obs_dim: {env.observation_space.shape[0]}, "
        f"Action shape: {env.action_space.shape}, "
        f"Train period: {start_date} to {end_date}"
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

    print(f"Starting {name} PPO training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=ProgressCallback(log_every=10_000))

    save_path = model_dir / f"ppo_{name}"
    model.save(str(save_path))
    print(f"{name} training complete. Model saved to: {save_path}")


def main() -> None:
    start_date = "2008-01-01"
    end_date = "2018-12-31"
    total_timesteps = 100_000
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    train_variant(
        name="baseline",
        use_regimes=False,
        start_date=start_date,
        end_date=end_date,
        total_timesteps=total_timesteps,
        model_dir=model_dir,
    )
    train_variant(
        name="regime_aware",
        use_regimes=True,
        start_date=start_date,
        end_date=end_date,
        total_timesteps=total_timesteps,
        model_dir=model_dir,
    )


if __name__ == "__main__":
    main()
