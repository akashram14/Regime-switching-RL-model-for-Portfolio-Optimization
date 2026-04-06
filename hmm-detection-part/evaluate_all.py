"""
Evaluate all strategies on the 11-ETF environment.

Compares:
  1. Baseline PPO       (paper reproduction — single agent, Differential Sharpe)
  2. MVO                (Max-Sharpe, Ledoit-Wolf shrinkage)
  3. Regime-Switching   (3 specialised agents routed by HMM)
  4. Equal-Weight       (1/N benchmark)

Produces:
  - Formatted metrics table (Sharpe, Sortino, Calmar, VaR, turnover, ...)
  - Equity-curve comparison plot with drawdowns and regime overlay
  - JSON metrics & CSV equity curves in results/

Usage:
    cd hmm-detection-part
    python evaluate_all.py
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data import build_dataset
from src.environment import PortfolioEnv
from src.evaluation import compute_metrics, compute_turnover
from src.hmm import RegimeDetector, compute_hmm_features
from src.mvo import MVOStrategy

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
# Dataset (same cache as train_all.py)
# ---------------------------------------------------------------------------

CACHE_PATH = Path("data/dataset_cache.pkl")


def get_dataset() -> dict:
    if CACHE_PATH.exists():
        print(f"Loading cached dataset from {CACHE_PATH} ...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    print("Building dataset ...")
    dataset = build_dataset(start="2006-01-01", end="2022-01-01", lookback=60,
                            sector_tickers=SECTOR_TICKERS)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


# ---------------------------------------------------------------------------
# Agent evaluation helpers
# ---------------------------------------------------------------------------

def run_agent_on_env(env: PortfolioEnv, model: PPO) -> dict:
    """Run a single PPO model on an environment, return history dict."""
    obs, _ = env.reset()
    pv = env.initial_cash
    dates, values, daily_returns, weights_list = [], [pv], [], []

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        dates.append(info["date"])
        values.append(info["portfolio_value"])
        ret = info["portfolio_return"]
        daily_returns.append(ret)
        weights_list.append(info["weights"].copy())

    return {
        "dates": pd.DatetimeIndex(dates),
        "values": np.array(values),
        "daily_returns": np.array(daily_returns),
        "weights": np.array(weights_list),
    }


def evaluate_baseline_ppo(dataset: dict, test_dates) -> dict:
    print("  Running Baseline PPO ...")
    env = PortfolioEnv(dataset=dataset, date_indices=test_dates)
    model = PPO.load("models/ppo_baseline", env=env)
    return run_agent_on_env(env, model)


def evaluate_mvo(dataset: dict, test_dates) -> dict:
    print("  Running MVO ...")
    mvo = MVOStrategy(lookback=60)
    history = mvo.run_backtest(dataset, test_dates, initial_cash=100_000.0)
    # Wrap in same format
    vals = np.concatenate([[100_000.0], history["portfolio_values"]])
    return {
        "dates": history["dates"],
        "values": vals,
        "daily_returns": history["daily_returns"],
        "weights": history["weights"],
    }


def evaluate_equal_weight(dataset: dict, test_dates) -> dict:
    print("  Running Equal-Weight ...")
    env = PortfolioEnv(dataset=dataset, date_indices=test_dates)
    obs, _ = env.reset()
    pv = env.initial_cash
    dates, values, daily_returns = [], [pv], []

    terminated = truncated = False
    while not (terminated or truncated):
        # Equal logits → softmax → equal weights
        action = np.zeros(env.n_assets + 1, dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        dates.append(info["date"])
        values.append(info["portfolio_value"])
        daily_returns.append(info["portfolio_return"])

    return {
        "dates": pd.DatetimeIndex(dates),
        "values": np.array(values),
        "daily_returns": np.array(daily_returns),
    }


# ---------------------------------------------------------------------------
# Regime-switching evaluation (core contribution)
# ---------------------------------------------------------------------------

def evaluate_regime_switching(dataset: dict, test_dates) -> dict:
    print("  Running Regime-Switching PPO ...")

    # --- HMM setup (save/load for reproducibility) ---
    hmm_path = Path("models/hmm_detector.pkl")
    sp500_prices = dataset["sp500_prices"]
    features = compute_hmm_features(sp500_prices)

    if hmm_path.exists():
        import pickle as _pkl
        print("    Loading saved HMM from", hmm_path)
        with open(hmm_path, "rb") as f:
            detector = _pkl.load(f)
    else:
        train_features = features.loc[:"2016-12-31"]
        detector = RegimeDetector(n_regimes=3, random_state=42)
        detector.fit(train_features)
        import pickle as _pkl
        with open(hmm_path, "wb") as f:
            _pkl.dump(detector, f)
        print("    Saved HMM to", hmm_path)

    # Predict regimes for all available dates
    all_regimes = detector.predict(features)   # Series: date → 0/1/2

    regime_names = ["regime_low_vol", "regime_med_vol", "regime_high_vol"]

    # Load the 3 specialised agents (all share the same obs/action space)
    env = PortfolioEnv(dataset=dataset, date_indices=test_dates)
    agents = {}
    for rn in regime_names:
        agents[rn] = PPO.load(f"models/ppo_{rn}", env=env)

    # Pre-build regime lookup for test dates
    def get_regime(date):
        ts = pd.Timestamp(date)
        if ts in all_regimes.index:
            return int(all_regimes.loc[ts])
        idx = all_regimes.index.get_indexer([ts], method="nearest")[0]
        return int(all_regimes.iloc[idx])

    # Run episode
    obs, _ = env.reset()
    pv = env.initial_cash
    dates, values, daily_returns = [], [pv], []
    weights_list, regime_choices = [], []

    terminated = truncated = False
    while not (terminated or truncated):
        current_date = env.dates[env._current_step]
        regime_idx = get_regime(current_date)
        regime_name = regime_names[regime_idx]

        action, _ = agents[regime_name].predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        dates.append(info["date"])
        values.append(info["portfolio_value"])
        daily_returns.append(info["portfolio_return"])
        weights_list.append(info["weights"].copy())
        regime_choices.append(regime_name)

    return {
        "dates": pd.DatetimeIndex(dates),
        "values": np.array(values),
        "daily_returns": np.array(daily_returns),
        "weights": np.array(weights_list),
        "regimes": regime_choices,
    }


# ---------------------------------------------------------------------------
# Metrics & output
# ---------------------------------------------------------------------------

def build_metrics(history: dict) -> dict:
    m = compute_metrics(
        daily_returns=history["daily_returns"],
        risk_free_rate=0.0,
        trading_days=252,
    )
    if "weights" in history and len(history["weights"]) > 1:
        to = compute_turnover(history["weights"])
        m["mean_turnover"] = float(np.mean(to))
    return m


def print_metrics_table(all_metrics: dict):
    headers = list(all_metrics.keys())
    metric_keys = list(next(iter(all_metrics.values())).keys())

    col_w = max(max(len(h) for h in headers) + 2, 20)
    line_w = 26 + col_w * len(headers)

    print(f"\n{'='*line_w}")
    print("  STRATEGY COMPARISON — 11 Sector ETFs  (Test Period)")
    print(f"{'='*line_w}")
    print(f"{'Metric':<26}", end="")
    for h in headers:
        print(f"{h:>{col_w}}", end="")
    print()
    print("-" * line_w)

    pct_keys = {"annual_return", "cumulative_return", "annual_volatility",
                "max_drawdown", "daily_var_95"}
    for key in metric_keys:
        print(f"{key:<26}", end="")
        for h in headers:
            val = all_metrics[h].get(key, float("nan"))
            if key in pct_keys:
                print(f"{val:>{col_w-1}.2%} ", end="")
            else:
                print(f"{val:>{col_w}.4f}", end="")
        print()
    print(f"{'='*line_w}\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results: dict, results_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(15, 14),
                             gridspec_kw={"height_ratios": [3, 2, 1]})

    colors = {
        "Baseline PPO": "#1f77b4",
        "MVO": "#9467bd",
        "Regime-Switching PPO": "#2ca02c",
        "Equal Weight": "#d62728",
    }

    # --- Equity curves ---
    ax = axes[0]
    for label, data in results.items():
        vals = data["values"][1:]
        normalised = vals / vals[0]
        ax.plot(data["dates"], normalised, label=label,
                color=colors.get(label, "gray"), lw=1.5)
    ax.set_ylabel("Normalised Portfolio Value")
    ax.set_title("Strategy Comparison — 11 Sector ETFs (Test Period)")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Drawdowns ---
    ax = axes[1]
    for label, data in results.items():
        vals = data["values"][1:]
        rmax = np.maximum.accumulate(vals)
        dd = vals / rmax - 1
        ax.fill_between(data["dates"], dd, 0, alpha=0.3,
                        color=colors.get(label, "gray"), label=label)
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Regime overlay ---
    ax = axes[2]
    rs = results.get("Regime-Switching PPO")
    if rs and "regimes" in rs:
        cmap = {"regime_low_vol": "green", "regime_med_vol": "gold",
                "regime_high_vol": "red"}
        nmap = {"regime_low_vol": 0, "regime_med_vol": 1, "regime_high_vol": 2}
        ax.scatter(rs["dates"], [nmap[r] for r in rs["regimes"]],
                   c=[cmap[r] for r in rs["regimes"]], s=3, alpha=0.6)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Low Vol", "Med Vol", "High Vol"])
        ax.set_ylabel("Active Regime")
        ax.set_title("HMM Regime Routing")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = results_dir / "comparison_11etf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset()

    # Test period: 2018-2022
    dates = dataset["dates"]
    test_dates = dates[(dates >= "2018-01-01") & (dates < "2022-01-01")]
    print(f"Test dates: {test_dates[0].date()} → {test_dates[-1].date()} "
          f"({len(test_dates)} days)")

    # Run all strategies
    results = {}
    results["Baseline PPO"]         = evaluate_baseline_ppo(dataset, test_dates)
    results["MVO"]                  = evaluate_mvo(dataset, test_dates)
    results["Regime-Switching PPO"] = evaluate_regime_switching(dataset, test_dates)
    results["Equal Weight"]         = evaluate_equal_weight(dataset, test_dates)

    # Compute metrics
    all_metrics = {label: build_metrics(data) for label, data in results.items()}
    print_metrics_table(all_metrics)

    # Regime distribution
    rs = results["Regime-Switching PPO"]
    if "regimes" in rs:
        counts = pd.Series(rs["regimes"]).value_counts()
        print("Regime routing distribution:")
        for regime, count in counts.items():
            pct = count / len(rs["regimes"]) * 100
            rname = regime.replace("regime_", "").replace("_", " ").title()
            print(f"  {rname}: {count} days ({pct:.1f}%)")
        print()

    # Save JSON metrics
    serialisable = {k: {mk: float(mv) for mk, mv in v.items()} for k, v in all_metrics.items()}
    metrics_path = results_dir / "metrics_11etf.json"
    with open(metrics_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Save equity CSVs
    for label, data in results.items():
        slug = label.lower().replace("-", "_").replace(" ", "_")
        df = pd.DataFrame({
            "Date": data["dates"],
            "portfolio_value": data["values"][1:],
            "daily_return": data["daily_returns"],
        })
        df.to_csv(results_dir / f"equity_{slug}_11etf.csv", index=False)

    # Plot
    plot_comparison(results, results_dir)


if __name__ == "__main__":
    main()
