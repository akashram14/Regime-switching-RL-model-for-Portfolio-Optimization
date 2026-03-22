"""
Evaluate regime-switching portfolio strategy.

Compares four strategies on the test period:
  1. Baseline PPO           (no regime information)
  2. Regime-Aware PPO       (regime probabilities as extra features)
  3. Regime-Switching PPO   (3 specialised agents routed by HMM)
  4. Equal-weight benchmark

Produces:
  - Comprehensive metrics table (Sharpe, Sortino, Calmar, VaR, ...)
  - Equity-curve comparison plot with drawdowns and regime overlay
  - Per-strategy equity CSVs and a combined JSON metrics file

Usage:
    python evaluate_regime_switching.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env_portfolio import PortfolioEnv


# ---------------------------------------------------------------------------
# Advanced metrics
# ---------------------------------------------------------------------------

def compute_advanced_metrics(
    daily_returns: np.ndarray,
    portfolio_values: np.ndarray,
) -> dict:
    eps = 1e-12
    returns = np.asarray(daily_returns, dtype=np.float64)
    values = np.asarray(portfolio_values, dtype=np.float64)

    if len(returns) == 0:
        return {k: 0.0 for k in [
            "total_return", "annualized_return", "annual_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "daily_var_95",
        ]}

    n_days = len(returns)
    total_return = values[-1] / (values[0] + eps) - 1.0
    annualized_return = (1 + total_return) ** (252.0 / max(n_days, 1)) - 1.0
    annual_vol = np.std(returns, ddof=1) * np.sqrt(252)

    # Sharpe
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    sharpe = np.sqrt(252) * mean_ret / (std_ret + eps) if std_ret > eps else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else eps
    sortino = np.sqrt(252) * mean_ret / (downside_std + eps)

    # Drawdown
    running_max = np.maximum.accumulate(values)
    drawdowns = values / (running_max + eps) - 1.0
    max_dd = float(np.min(drawdowns))

    # Calmar
    calmar = annualized_return / abs(max_dd) if abs(max_dd) > eps else 0.0

    # VaR 95 %
    var_95 = float(np.percentile(returns, 5))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "daily_var_95": var_95,
    }


# ---------------------------------------------------------------------------
# Single-agent evaluation helper
# ---------------------------------------------------------------------------

def _run_agent(env: PortfolioEnv, model: PPO) -> dict:
    obs, _ = env.reset()
    portfolio_value = 1.0
    dates, values, daily_returns, weights_list = [], [1.0], [], []

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        ret = float(info["portfolio_return"])
        portfolio_value *= 1.0 + ret

        step_date = env.dates.iloc[env.current_step - 1]
        dates.append(step_date)
        values.append(portfolio_value)
        daily_returns.append(ret)
        weights_list.append(info["weights"].copy())

    return {
        "dates": dates,
        "values": np.array(values),
        "daily_returns": np.array(daily_returns),
        "weights": np.array(weights_list),
    }


def evaluate_single_agent(
    model_path: str,
    use_regimes: bool,
    test_start: str,
    test_end: str,
) -> dict:
    env = PortfolioEnv(
        features_path="features.csv",
        returns_path="returns.csv",
        use_regimes=use_regimes,
        reward_type="default",
        start_date=test_start,
        end_date=test_end,
    )
    model = PPO.load(model_path, env=env)
    return _run_agent(env, model)


# ---------------------------------------------------------------------------
# Regime-switching evaluation (the core contribution)
# ---------------------------------------------------------------------------

def evaluate_regime_switching(test_start: str, test_end: str) -> dict:
    """Route each step to a regime-specialised agent via precomputed HMM."""

    # Pre-load regime probabilities
    regimes_df = pd.read_csv("data/regimes.csv", parse_dates=["Date"])
    regime_cols = ["p_low_vol", "p_med_vol", "p_high_vol"]
    regimes_df = regimes_df.set_index("Date")
    dominant = regimes_df[regime_cols].values.argmax(axis=1)
    regime_series = pd.Series(dominant, index=regimes_df.index, dtype=int)

    regime_names = ["low_vol", "med_vol", "high_vol"]

    # Environment — no regime features (agents don't see them)
    env = PortfolioEnv(
        features_path="features.csv",
        returns_path="returns.csv",
        use_regimes=False,
        reward_type="default",
        start_date=test_start,
        end_date=test_end,
    )

    # Load specialised agents
    agents = {}
    for name in regime_names:
        agents[name] = PPO.load(f"models/ppo_regime_{name}", env=env)

    # Precompute regime for every env date  (robust to minor date mismatches)
    env_dates_ts = pd.to_datetime(env.dates)
    regime_for_step = np.ones(len(env_dates_ts), dtype=int)  # default = med
    for i, d in enumerate(env_dates_ts):
        ts = pd.Timestamp(d)
        if ts in regime_series.index:
            regime_for_step[i] = regime_series.loc[ts]
        else:
            idx = regime_series.index.get_indexer([ts], method="nearest")[0]
            regime_for_step[i] = int(regime_series.iloc[idx])

    # Run evaluation
    obs, _ = env.reset()
    portfolio_value = 1.0
    dates, values, daily_returns = [], [1.0], []
    weights_list, regime_choices = [], []

    terminated = truncated = False
    while not (terminated or truncated):
        step_idx = env.current_step
        regime_idx = int(regime_for_step[step_idx])
        regime_name = regime_names[regime_idx]

        action, _ = agents[regime_name].predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        ret = float(info["portfolio_return"])
        portfolio_value *= 1.0 + ret

        step_date = env.dates.iloc[env.current_step - 1]
        dates.append(step_date)
        values.append(portfolio_value)
        daily_returns.append(ret)
        weights_list.append(info["weights"].copy())
        regime_choices.append(regime_name)

    return {
        "dates": dates,
        "values": np.array(values),
        "daily_returns": np.array(daily_returns),
        "weights": np.array(weights_list),
        "regimes": regime_choices,
    }


# ---------------------------------------------------------------------------
# Equal-weight benchmark
# ---------------------------------------------------------------------------

def evaluate_equal_weight(test_start: str, test_end: str) -> dict:
    env = PortfolioEnv(
        features_path="features.csv",
        returns_path="returns.csv",
        use_regimes=False,
        reward_type="default",
        start_date=test_start,
        end_date=test_end,
    )

    obs, _ = env.reset()
    portfolio_value = 1.0
    dates, values, daily_returns = [], [1.0], []

    terminated = truncated = False
    while not (terminated or truncated):
        action = np.zeros(env.n_assets, dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)

        ret = float(info["portfolio_return"])
        portfolio_value *= 1.0 + ret

        step_date = env.dates.iloc[env.current_step - 1]
        dates.append(step_date)
        values.append(portfolio_value)
        daily_returns.append(ret)

    return {
        "dates": dates,
        "values": np.array(values),
        "daily_returns": np.array(daily_returns),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results: dict, results_dir: Path) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 14),
        gridspec_kw={"height_ratios": [3, 2, 1]},
    )

    colors = {
        "Baseline PPO": "#1f77b4",
        "Regime-Aware PPO": "#ff7f0e",
        "Regime-Switching PPO": "#2ca02c",
        "Equal Weight": "#d62728",
    }

    # Equity curve
    ax = axes[0]
    for label, data in results.items():
        ax.plot(
            pd.to_datetime(data["dates"]),
            data["values"][1:],
            label=label,
            color=colors.get(label, "gray"),
            lw=1.5,
        )
    ax.set_ylabel("Portfolio Value (starting $1)")
    ax.set_title("Strategy Comparison \u2014 Test Period")
    ax.legend()
    ax.grid(alpha=0.3)

    # Drawdown
    ax = axes[1]
    for label, data in results.items():
        vals = data["values"][1:]
        running_max = np.maximum.accumulate(vals)
        dd = vals / running_max - 1
        ax.fill_between(
            pd.to_datetime(data["dates"]),
            dd, 0,
            alpha=0.3,
            color=colors.get(label, "gray"),
            label=label,
        )
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)

    # Regime overlay
    ax = axes[2]
    rs_data = results.get("Regime-Switching PPO")
    if rs_data and "regimes" in rs_data:
        regime_color = {"low_vol": "green", "med_vol": "gold", "high_vol": "red"}
        regime_num = {"low_vol": 0, "med_vol": 1, "high_vol": 2}
        ax.scatter(
            pd.to_datetime(rs_data["dates"]),
            [regime_num[r] for r in rs_data["regimes"]],
            c=[regime_color[r] for r in rs_data["regimes"]],
            s=3, alpha=0.6,
        )
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Low Vol", "Med Vol", "High Vol"])
        ax.set_ylabel("Active Regime")
        ax.set_title("HMM Regime Routing")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = results_dir / "regime_switching_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


# ---------------------------------------------------------------------------
# Formatted table
# ---------------------------------------------------------------------------

def print_metrics_table(all_metrics: dict) -> None:
    headers = list(all_metrics.keys())
    metric_keys = list(next(iter(all_metrics.values())).keys())

    col_w = max(len(h) for h in headers) + 2
    col_w = max(col_w, 18)

    print(f"\n{'='*(22 + col_w * len(headers))}")
    print("  STRATEGY COMPARISON \u2014 TEST PERIOD")
    print(f"{'='*(22 + col_w * len(headers))}")

    print(f"{'Metric':<22}", end="")
    for h in headers:
        print(f"{h:>{col_w}}", end="")
    print()
    print("-" * (22 + col_w * len(headers)))

    pct_keys = {
        "total_return", "annualized_return", "annual_volatility",
        "max_drawdown", "daily_var_95",
    }
    for key in metric_keys:
        print(f"{key:<22}", end="")
        for h in headers:
            val = all_metrics[h][key]
            if key in pct_keys:
                print(f"{val:>{col_w-1}.2%} ", end="")
            else:
                print(f"{val:>{col_w}.4f}", end="")
        print()

    print(f"{'='*(22 + col_w * len(headers))}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    test_start = "2019-01-01"
    test_end = "2024-12-31"

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    all_metrics: dict[str, dict] = {}

    # 1. Baseline PPO
    print("Evaluating Baseline PPO ...")
    r = evaluate_single_agent(
        "models/ppo_baseline", use_regimes=False,
        test_start=test_start, test_end=test_end,
    )
    results["Baseline PPO"] = r
    all_metrics["Baseline PPO"] = compute_advanced_metrics(r["daily_returns"], r["values"])

    # 2. Regime-Aware PPO (single agent with regime features)
    print("Evaluating Regime-Aware PPO ...")
    r = evaluate_single_agent(
        "models/ppo_regime_aware", use_regimes=True,
        test_start=test_start, test_end=test_end,
    )
    results["Regime-Aware PPO"] = r
    all_metrics["Regime-Aware PPO"] = compute_advanced_metrics(r["daily_returns"], r["values"])

    # 3. Regime-Switching PPO (3 agents + HMM routing)
    print("Evaluating Regime-Switching PPO ...")
    r = evaluate_regime_switching(test_start, test_end)
    results["Regime-Switching PPO"] = r
    all_metrics["Regime-Switching PPO"] = compute_advanced_metrics(r["daily_returns"], r["values"])

    # 4. Equal-weight benchmark
    print("Evaluating Equal Weight benchmark ...")
    r = evaluate_equal_weight(test_start, test_end)
    results["Equal Weight"] = r
    all_metrics["Equal Weight"] = compute_advanced_metrics(r["daily_returns"], r["values"])

    # Print comparison table
    print_metrics_table(all_metrics)

    # Save metrics JSON
    serialisable = {
        k: {mk: float(mv) for mk, mv in v.items()}
        for k, v in all_metrics.items()
    }
    metrics_path = results_dir / "metrics_all_strategies.json"
    with open(metrics_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Save equity CSVs
    for label, data in results.items():
        slug = label.lower().replace("-", "_").replace(" ", "_")
        df = pd.DataFrame({
            "Date": pd.to_datetime(data["dates"]),
            "portfolio_value": data["values"][1:],
            "daily_return": data["daily_returns"],
        })
        csv_path = results_dir / f"equity_curve_{slug}.csv"
        df.to_csv(csv_path, index=False)

    # Plot comparison
    plot_comparison(results, results_dir)

    # Regime distribution summary
    rs = results.get("Regime-Switching PPO")
    if rs and "regimes" in rs:
        counts = pd.Series(rs["regimes"]).value_counts()
        print("\nRegime routing distribution:")
        for regime, count in counts.items():
            pct = count / len(rs["regimes"]) * 100
            print(f"  {regime}: {count} days ({pct:.1f}%)")


if __name__ == "__main__":
    main()
