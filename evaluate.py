import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env_portfolio import PortfolioEnv


def compute_metrics(daily_returns: np.ndarray, portfolio_values: np.ndarray) -> dict:
    eps = 1e-12
    mean_ret = float(np.mean(daily_returns))
    std_ret = float(np.std(daily_returns))
    sharpe = (np.sqrt(252.0) * mean_ret / (std_ret + eps)) if len(daily_returns) > 1 else 0.0

    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = portfolio_values / (running_max + eps) - 1.0
    max_drawdown = float(np.min(drawdowns))

    n_days = len(daily_returns)
    if n_days > 0 and portfolio_values[0] > 0:
        total_return = portfolio_values[-1] / portfolio_values[0]
        annualized_return = float(total_return ** (252.0 / n_days) - 1.0)
        total_return_pct = float(total_return - 1.0)
    else:
        annualized_return = 0.0
        total_return_pct = 0.0

    return {
        "total_return": total_return_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "annualized_return": annualized_return,
    }


def evaluate_model(
    *,
    label: str,
    model_path: str,
    use_regimes: bool,
    test_start: str,
    test_end: str,
    results_dir: Path,
) -> dict:
    env = PortfolioEnv(
        features_path="features.csv",
        returns_path="returns.csv",
        use_regimes=use_regimes,
        start_date=test_start,
        end_date=test_end,
    )

    # SB3 stores models as .zip; load works with or without extension.
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    terminated = False
    truncated = False

    portfolio_value = 1.0
    dates = []
    values = [portfolio_value]
    daily_returns = []
    weights_records = []

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        day_return = float(info["portfolio_return"])
        portfolio_value *= (1.0 + day_return)

        step_idx = env.current_step - 1
        step_date = env.dates.iloc[step_idx]

        dates.append(step_date)
        values.append(portfolio_value)
        daily_returns.append(day_return)

        w = info["weights"]
        weights_records.append(
            {
                "Date": step_date,
                "SPY_weight": float(w[0]),
                "QQQ_weight": float(w[1]),
                "IWM_weight": float(w[2]),
                "TLT_weight": float(w[3]),
                "GLD_weight": float(w[4]),
            }
        )

    metrics = compute_metrics(np.asarray(daily_returns, dtype=np.float64), np.asarray(values, dtype=np.float64))
    slug = label.lower().replace("-", "_").replace(" ", "_")
    metrics_path = results_dir / f"metrics_{slug}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    equity_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "portfolio_value": values[1:],
            "daily_return": daily_returns,
        }
    )
    weights_df = pd.DataFrame(weights_records)
    if not weights_df.empty:
        weights_df["Date"] = pd.to_datetime(weights_df["Date"])
        equity_df = equity_df.merge(weights_df, on="Date", how="left")

    equity_csv_path = results_dir / f"equity_curve_{slug}.csv"
    equity_df.to_csv(equity_csv_path, index=False)

    plt.figure(figsize=(10, 5))
    cumulative_return = np.asarray(values[1:], dtype=np.float64) - 1.0
    plt.plot(pd.to_datetime(dates), cumulative_return, label=label)
    plt.title(f"{label} Cumulative Return (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = results_dir / f"ppo_equity_curve_{slug}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"## {label} Results")
    print(f"Total Return: {metrics['total_return']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
    print("")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved equity curve: {equity_csv_path}")
    print(f"Saved plot: {plot_path}")
    print("")

    return metrics


def main() -> None:
    test_start = "2019-01-01"
    test_end = "2024-12-31"

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluate_model(
        label="Baseline PPO",
        model_path="models/ppo_baseline",
        use_regimes=False,
        test_start=test_start,
        test_end=test_end,
        results_dir=results_dir,
    )
    evaluate_model(
        label="Regime-Aware PPO",
        model_path="models/ppo_regime_aware",
        use_regimes=True,
        test_start=test_start,
        test_end=test_end,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
