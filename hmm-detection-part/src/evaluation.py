"""
Backtest evaluation and metrics.

Computes performance statistics for comparing DRL vs MVO strategies:
- Annual return, cumulative returns
- Annual volatility
- Sharpe ratio, Sortino ratio, Calmar ratio
- Max drawdown
- Portfolio turnover (Δpw)

Reference: Section 5.4 and Table 2 of Sood et al. (2023)
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt


def compute_metrics(
    daily_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> Dict:
    """
    Compute performance metrics from daily returns.
    
    Args:
        daily_returns: Array of daily portfolio returns
        risk_free_rate: Annual risk-free rate (default 0)
        trading_days: Number of trading days per year
    
    Returns:
        Dictionary of performance metrics
    """
    returns = daily_returns[~np.isnan(daily_returns)]
    
    if len(returns) == 0:
        return {k: 0.0 for k in [
            "annual_return", "cumulative_return", "annual_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "omega_ratio", "skew", "kurtosis",
            "tail_ratio", "daily_var_95",
        ]}
    
    # Cumulative returns
    cum_returns = np.cumprod(1 + returns)
    total_return = cum_returns[-1] - 1
    
    # Annual return (geometric)
    n_years = len(returns) / trading_days
    annual_return = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
    
    # Annual volatility
    annual_vol = np.std(returns, ddof=1) * np.sqrt(trading_days)
    
    # Sharpe ratio
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess_returns = returns - daily_rf
    if np.std(excess_returns, ddof=1) > 0:
        sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(trading_days)
    else:
        sharpe = 0.0
    
    # Sortino ratio (downside deviation only)
    downside_returns = returns[returns < daily_rf] - daily_rf
    if len(downside_returns) > 0 and np.std(downside_returns, ddof=1) > 0:
        sortino = np.mean(excess_returns) / np.std(downside_returns, ddof=1) * np.sqrt(trading_days)
    else:
        sortino = 0.0
    
    # Max drawdown
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1
    max_dd = np.min(drawdowns)
    
    # Calmar ratio
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
    
    # Omega ratio
    threshold = daily_rf
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    omega = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else np.inf
    
    # Skew and Kurtosis
    skew = float(pd.Series(returns).skew())
    kurtosis = float(pd.Series(returns).kurtosis())
    
    # Tail ratio (95th percentile / abs(5th percentile))
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    tail_ratio = abs(p95 / p5) if abs(p5) > 1e-10 else np.inf
    
    # Daily Value at Risk (95%)
    daily_var = np.percentile(returns, 5)
    
    return {
        "annual_return": annual_return,
        "cumulative_return": total_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "omega_ratio": omega,
        "skew": skew,
        "kurtosis": kurtosis,
        "tail_ratio": tail_ratio,
        "daily_var_95": daily_var,
    }


def compute_turnover(weights_history: np.ndarray) -> np.ndarray:
    """
    Compute portfolio turnover Δpw (absolute weight changes).
    
    Δpw_t = sum(|w_{t} - w_{t-1}|) for non-cash assets
    Δpw ∈ [0.0, 2.0]
    
    Reference: Section 6 of the paper.
    """
    if len(weights_history) < 2:
        return np.array([0.0])
    
    # Exclude cash column (last column)
    asset_weights = weights_history[:, :-1] if weights_history.shape[1] > 1 else weights_history
    
    diffs = np.abs(np.diff(asset_weights, axis=0))
    turnover = np.sum(diffs, axis=1)
    
    return turnover


def print_metrics_table(
    drl_metrics: Dict,
    mvo_metrics: Dict,
    title: str = "Backtest Results",
):
    """Print a formatted comparison table of metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'DRL':>15} {'MVO':>15}")
    print(f"{'-'*55}")
    
    for key in drl_metrics:
        drl_val = drl_metrics[key]
        mvo_val = mvo_metrics[key]
        print(f"{key:<25} {drl_val:>15.4f} {mvo_val:>15.4f}")
    
    print(f"{'='*60}\n")


def plot_backtest_comparison(
    drl_history: Dict,
    mvo_history: Dict,
    title: str = "DRL vs MVO Portfolio Comparison",
    save_path: str = None,
):
    """
    Plot comparison of DRL and MVO portfolio values and drawdowns.
    
    Similar to Figure 2 in the paper.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Portfolio Value
    ax = axes[0]
    ax.plot(drl_history["dates"], drl_history["portfolio_values"],
            label="DRL (PPO)", color="blue", lw=1.5)
    ax.plot(mvo_history["dates"], mvo_history["portfolio_values"],
            label="MVO", color="orange", lw=1.5)
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Cumulative Returns
    ax = axes[1]
    drl_cum = np.cumprod(1 + drl_history["daily_returns"]) - 1
    mvo_cum = np.cumprod(1 + mvo_history["daily_returns"]) - 1
    ax.plot(drl_history["dates"], drl_cum, label="DRL", color="blue", lw=1.5)
    ax.plot(mvo_history["dates"], mvo_cum, label="MVO", color="orange", lw=1.5)
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Drawdown
    ax = axes[2]
    for hist, label, color in [
        (drl_history, "DRL", "blue"),
        (mvo_history, "MVO", "orange"),
    ]:
        cum = np.cumprod(1 + hist["daily_returns"])
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        ax.fill_between(hist["dates"], dd, 0, alpha=0.3, color=color, label=label)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_weight_evolution(
    history: Dict,
    ticker_names: Dict,
    tickers: List[str],
    title: str = "Portfolio Weight Evolution",
    save_path: str = None,
):
    """Plot stacked area chart of portfolio weight evolution."""
    weights = history["weights"]
    dates = history["dates"]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    labels = [ticker_names.get(t, t) for t in tickers] + ["Cash"]
    
    ax.stackplot(
        dates,
        *[weights[:, i] for i in range(weights.shape[1])],
        labels=labels,
        alpha=0.8,
    )
    
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


def plot_returns_distribution(
    drl_returns: np.ndarray,
    mvo_returns: np.ndarray,
    title: str = "Distribution of Daily Returns",
    save_path: str = None,
):
    """Plot histogram comparison of daily return distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, returns, label, color in [
        (axes[0], drl_returns, "DRL (PPO)", "blue"),
        (axes[1], mvo_returns, "MVO", "orange"),
    ]:
        ax.hist(returns, bins=50, alpha=0.7, color=color, edgecolor="black", lw=0.5)
        ax.axvline(np.mean(returns), color="red", linestyle="--", label=f"Mean: {np.mean(returns):.4f}")
        ax.set_title(f"{label} Daily Returns")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()
