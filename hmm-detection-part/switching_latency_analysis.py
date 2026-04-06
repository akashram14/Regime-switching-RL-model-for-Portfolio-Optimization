"""
Switching Latency Analysis for HMM Regime Detection.

Quantifies:
  1. Detection delay — how many days after a "true" volatility shift does
     the HMM change its dominant regime?
  2. Transition cost — what portfolio return is lost/gained during the
     delay window compared to an oracle that switches instantly?
  3. Probability ramp — how quickly do regime probabilities shift at each
     transition boundary?

Produces:
  - Summary table of all regime transitions in the test period
  - Per-transition latency and cost analysis
  - Aggregate latency statistics
  - Saves results to results/switching_latency.json

Usage:
    cd hmm-detection-part
    python switching_latency_analysis.py
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
from src.hmm import RegimeDetector, compute_hmm_features

SECTOR_TICKERS = {
    "XLB": "Materials", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Technology", "XLP": "Consumer Staples",
    "XLU": "Utilities", "XLV": "Health Care", "XLY": "Consumer Discretionary",
}

CACHE_PATH = Path("data/dataset_cache.pkl")
REGIME_NAMES = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}


def get_dataset() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    dataset = build_dataset(start="2006-01-01", end="2022-01-01", lookback=60,
                            sector_tickers=SECTOR_TICKERS)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def find_transitions(regime_series: pd.Series) -> list:
    """Find all regime transition points."""
    transitions = []
    for i in range(1, len(regime_series)):
        if regime_series.iloc[i] != regime_series.iloc[i - 1]:
            transitions.append({
                "date": regime_series.index[i],
                "from_regime": int(regime_series.iloc[i - 1]),
                "to_regime": int(regime_series.iloc[i]),
            })
    return transitions


def compute_probability_ramp(
    proba_df: pd.DataFrame, transition_date: pd.Timestamp,
    to_regime: int, window: int = 10,
) -> dict:
    """
    Measure how quickly regime probability ramps up around a transition.

    Looks at a window before and after the transition and reports:
      - Days before transition where target regime prob first exceeded 0.10, 0.25, 0.50
      - Probability trajectory in the window
    """
    col_map = {0: "p_low_vol", 1: "p_med_vol", 2: "p_high_vol"}
    target_col = col_map[to_regime]

    idx = proba_df.index.get_loc(transition_date)
    start = max(0, idx - window)
    end = min(len(proba_df), idx + window + 1)

    window_df = proba_df.iloc[start:end]
    target_probs = window_df[target_col]

    # Find when target prob first exceeded thresholds (before transition)
    pre_transition = proba_df.iloc[max(0, idx - window):idx + 1][target_col]
    thresholds = {}
    for thresh in [0.10, 0.25, 0.50]:
        exceeded = pre_transition[pre_transition >= thresh]
        if len(exceeded) > 0:
            first_date = exceeded.index[0]
            days_before = (transition_date - first_date).days
            thresholds[f"days_before_p{int(thresh*100)}"] = int(days_before)
        else:
            thresholds[f"days_before_p{int(thresh*100)}"] = None

    return {
        "thresholds": thresholds,
        "prob_at_switch": float(proba_df.loc[transition_date, target_col]),
        "prob_1d_before": float(proba_df.iloc[idx - 1][target_col]) if idx > 0 else None,
        "prob_trajectory": {
            str(d.date()): float(v)
            for d, v in target_probs.items()
        },
    }


def compute_transition_cost(
    dataset: dict, test_dates, proba_df: pd.DataFrame,
    transition_date: pd.Timestamp, from_regime: int, to_regime: int,
    lookback_days: int = 5,
) -> dict:
    """
    Compute the cost of delayed switching around a transition.

    Compares returns of:
      1. The FROM agent (what was active before the switch)
      2. The TO agent (what becomes active after)
      3. The regime-switching system (which switches on transition_date)

    Over a window of lookback_days before and after the transition.
    """
    regime_names_map = ["regime_low_vol", "regime_med_vol", "regime_high_vol"]

    # Get dates around the transition
    all_test = pd.DatetimeIndex(test_dates)
    idx = all_test.get_indexer([transition_date], method="nearest")[0]
    start = max(0, idx - lookback_days)
    end = min(len(all_test), idx + lookback_days + 1)
    window_dates = all_test[start:end]

    # Load equity curves (already computed)
    results_dir = Path("results")
    try:
        rs_equity = pd.read_csv(results_dir / "equity_regime_switching_ppo_11etf.csv")
        bl_equity = pd.read_csv(results_dir / "equity_baseline_ppo_11etf.csv")
        rs_equity["Date"] = pd.to_datetime(rs_equity["Date"])
        bl_equity["Date"] = pd.to_datetime(bl_equity["Date"])

        # Returns around transition
        mask = rs_equity["Date"].isin(window_dates)
        rs_returns = rs_equity.loc[mask, "daily_return"].values
        bl_returns = bl_equity.loc[mask, "daily_return"].values

        # Split into pre-transition and post-transition
        trans_idx_in_window = lookback_days  # middle of window
        if len(rs_returns) > trans_idx_in_window:
            pre_rs = rs_returns[:trans_idx_in_window]
            post_rs = rs_returns[trans_idx_in_window:]
            pre_bl = bl_returns[:trans_idx_in_window]
            post_bl = bl_returns[trans_idx_in_window:]
        else:
            pre_rs = rs_returns
            post_rs = np.array([])
            pre_bl = bl_returns
            post_bl = np.array([])

        return {
            "window_return_rs": float(np.prod(1 + rs_returns) - 1) if len(rs_returns) > 0 else 0.0,
            "window_return_baseline": float(np.prod(1 + bl_returns) - 1) if len(bl_returns) > 0 else 0.0,
            "pre_switch_return_rs": float(np.prod(1 + pre_rs) - 1) if len(pre_rs) > 0 else 0.0,
            "post_switch_return_rs": float(np.prod(1 + post_rs) - 1) if len(post_rs) > 0 else 0.0,
            "advantage": float(
                (np.prod(1 + rs_returns) - 1) - (np.prod(1 + bl_returns) - 1)
            ) if len(rs_returns) > 0 else 0.0,
        }
    except FileNotFoundError:
        return {"error": "Equity CSVs not found — run evaluate_all.py first"}


def compute_realized_vol(returns: np.ndarray, window: int = 10) -> pd.Series:
    """Compute rolling realized volatility to detect true vol shifts."""
    s = pd.Series(returns)
    return s.rolling(window).std() * np.sqrt(252)


def main():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data --
    dataset = get_dataset()
    sp500_prices = dataset["sp500_prices"]
    features = compute_hmm_features(sp500_prices)
    train_features = features.loc[:"2016-12-31"]

    # -- Fit HMM --
    detector = RegimeDetector(n_regimes=3, random_state=42)
    detector.fit(train_features)

    # -- Regime prediction --
    all_regimes = detector.predict(features)
    all_proba = detector.predict_proba(features)

    # -- Test period --
    dates = dataset["dates"]
    test_dates = dates[(dates >= "2018-01-01") & (dates < "2022-01-01")]
    test_regimes = all_regimes.loc[test_dates[0]:test_dates[-1]]
    test_proba = all_proba.loc[test_dates[0]:test_dates[-1]]

    # -- Compute realized vol for "ground truth" reference --
    sp500_returns = np.log(sp500_prices / sp500_prices.shift(1)).dropna()
    realized_vol = sp500_returns.rolling(10).std() * np.sqrt(252)

    # -- Find transitions --
    transitions = find_transitions(test_regimes)
    print(f"Found {len(transitions)} regime transitions in test period "
          f"({test_dates[0].date()} to {test_dates[-1].date()})")
    print()

    # -- Analyze each transition --
    transition_results = []
    for i, t in enumerate(transitions):
        from_r = t["from_regime"]
        to_r = t["to_regime"]
        date = t["date"]

        # Probability ramp analysis
        ramp = compute_probability_ramp(test_proba, date, to_r, window=10)

        # Transition cost
        cost = compute_transition_cost(
            dataset, test_dates, test_proba, date, from_r, to_r,
            lookback_days=5,
        )

        # Realized vol at transition
        rv = float(realized_vol.loc[date]) if date in realized_vol.index else None

        # How long does new regime persist?
        future_regimes = test_regimes.loc[date:]
        changes = future_regimes[future_regimes != to_r]
        if len(changes) > 0:
            persist_days = (changes.index[0] - date).days
        else:
            persist_days = (test_regimes.index[-1] - date).days

        result = {
            "transition_id": i + 1,
            "date": str(date.date()),
            "from": REGIME_NAMES[from_r],
            "to": REGIME_NAMES[to_r],
            "realized_vol_annualized": rv,
            "prob_at_switch": ramp["prob_at_switch"],
            "prob_1d_before": ramp["prob_1d_before"],
            "detection_lag": ramp["thresholds"],
            "persistence_days": persist_days,
            "transition_cost": cost,
        }
        transition_results.append(result)

        # Print summary
        lag_str = ""
        for k, v in ramp["thresholds"].items():
            if v is not None:
                lag_str += f"  {k}: {v}d"
        print(f"  #{i+1}  {date.date()}  {REGIME_NAMES[from_r]:>8} → {REGIME_NAMES[to_r]:<8}"
              f"  p={ramp['prob_at_switch']:.3f}  persist={persist_days}d{lag_str}")

    # -- Aggregate statistics --
    print(f"\n{'='*70}")
    print("AGGREGATE SWITCHING LATENCY STATISTICS")
    print(f"{'='*70}")

    # Classify transitions by type
    escalations = [t for t in transition_results if
                   list(REGIME_NAMES.keys())[list(REGIME_NAMES.values()).index(t["to"])]
                   > list(REGIME_NAMES.keys())[list(REGIME_NAMES.values()).index(t["from"])]]
    de_escalations = [t for t in transition_results if
                      list(REGIME_NAMES.keys())[list(REGIME_NAMES.values()).index(t["to"])]
                      < list(REGIME_NAMES.keys())[list(REGIME_NAMES.values()).index(t["from"])]]

    def avg_lag(results_list, key):
        vals = [t["detection_lag"].get(key) for t in results_list
                if t["detection_lag"].get(key) is not None]
        return np.mean(vals) if vals else None

    def avg_persist(results_list):
        return np.mean([t["persistence_days"] for t in results_list])

    def avg_advantage(results_list):
        vals = [t["transition_cost"].get("advantage", 0.0) for t in results_list
                if "advantage" in t["transition_cost"]]
        return np.mean(vals) if vals else None

    print(f"\nTotal transitions:    {len(transition_results)}")
    print(f"  Escalations (→ higher vol):  {len(escalations)}")
    print(f"  De-escalations (→ lower vol): {len(de_escalations)}")

    print(f"\nAvg persistence (escalation):    {avg_persist(escalations):.0f} days")
    print(f"Avg persistence (de-escalation): {avg_persist(de_escalations):.0f} days")

    for label, subset in [("Escalations", escalations), ("De-escalations", de_escalations)]:
        print(f"\n{label}:")
        for key_name in ["days_before_p10", "days_before_p25", "days_before_p50"]:
            val = avg_lag(subset, key_name)
            if val is not None:
                print(f"  Avg {key_name}: {val:.1f} days before hard switch")

    adv_esc = avg_advantage(escalations)
    adv_de = avg_advantage(de_escalations)
    print(f"\nAvg RS vs Baseline advantage around transitions:")
    if adv_esc is not None:
        print(f"  Escalations:    {adv_esc:+.4f} ({adv_esc*100:+.2f}%)")
    if adv_de is not None:
        print(f"  De-escalations: {adv_de:+.4f} ({adv_de*100:+.2f}%)")

    # -- Save JSON --
    output = {
        "n_transitions": len(transition_results),
        "n_escalations": len(escalations),
        "n_de_escalations": len(de_escalations),
        "avg_persistence_escalation_days": float(avg_persist(escalations)),
        "avg_persistence_de_escalation_days": float(avg_persist(de_escalations)),
        "transitions": transition_results,
    }
    out_path = results_dir / "switching_latency.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # -- Plot: probability ramps around key transitions --
    # Focus on transitions involving High Vol (most interesting)
    high_transitions = [t for t in transitions
                        if t["from_regime"] == 2 or t["to_regime"] == 2]

    if high_transitions:
        n_plots = min(len(high_transitions), 6)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.5 * n_plots), sharex=False)
        if n_plots == 1:
            axes = [axes]

        for ax, t in zip(axes, high_transitions[:n_plots]):
            date = t["date"]
            idx = test_proba.index.get_loc(date)
            start = max(0, idx - 15)
            end = min(len(test_proba), idx + 15)
            window = test_proba.iloc[start:end]

            ax.plot(window.index, window["p_low_vol"], "g-", label="p(Low Vol)", lw=1.5)
            ax.plot(window.index, window["p_med_vol"], color="gold", ls="-", label="p(Med Vol)", lw=1.5)
            ax.plot(window.index, window["p_high_vol"], "r-", label="p(High Vol)", lw=1.5)
            ax.axvline(date, color="black", ls="--", alpha=0.7, label="Switch date")
            ax.set_ylabel("Probability")
            ax.set_title(
                f"{date.date()}: {REGIME_NAMES[t['from_regime']]} → {REGIME_NAMES[t['to_regime']]}",
                fontsize=11,
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)

        plt.suptitle("HMM Regime Probability Ramps Around High-Vol Transitions", fontsize=13, y=1.01)
        plt.tight_layout()
        path = results_dir / "switching_latency_ramps.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
