"""
Per-regime case study analysis.

Computes:
1. Per-regime performance metrics for each strategy
2. COVID-19 crash case study (Feb-Apr 2020)
3. Regime transition analysis
4. BIC model selection for HMM
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.data import build_dataset
from src.hmm import RegimeDetector, compute_hmm_features
from src.evaluation import compute_metrics

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

CACHE_PATH = Path("data/dataset_cache.pkl")
RESULTS_DIR = Path("results")


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


def load_equity_curves():
    """Load all equity curve CSVs."""
    strategies = {
        "Baseline PPO": "equity_baseline_ppo_11etf.csv",
        "MVO": "equity_mvo_11etf.csv",
        "Regime-Switching PPO": "equity_regime_switching_ppo_11etf.csv",
        "Equal Weight": "equity_equal_weight_11etf.csv",
    }
    curves = {}
    for label, fname in strategies.items():
        df = pd.read_csv(RESULTS_DIR / fname, parse_dates=["Date"])
        df = df.set_index("Date")
        curves[label] = df
    return curves


def get_regimes_for_test(dataset):
    """Load saved HMM (or fit and save), predict regimes for all dates."""
    hmm_path = Path("models/hmm_detector.pkl")
    sp500_prices = dataset["sp500_prices"]
    features = compute_hmm_features(sp500_prices)

    if hmm_path.exists():
        print("  Loading saved HMM from", hmm_path)
        with open(hmm_path, "rb") as f:
            detector = pickle.load(f)
    else:
        train_features = features.loc[:"2016-12-31"]
        detector = RegimeDetector(n_regimes=3, random_state=42)
        detector.fit(train_features)
        with open(hmm_path, "wb") as f:
            pickle.dump(detector, f)
        print("  Saved HMM to", hmm_path)

    all_regimes = detector.predict(features)
    return all_regimes, detector, features


def per_regime_metrics(curves, regimes):
    """Compute metrics for each strategy broken down by regime."""
    regime_names = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}
    results = {}

    for label, df in curves.items():
        results[label] = {}
        for regime_id, regime_name in regime_names.items():
            # Find dates where this regime was active
            regime_dates = regimes[regimes == regime_id].index
            # Intersect with this strategy's dates
            common = df.index.intersection(regime_dates)
            if len(common) < 5:
                continue
            regime_returns = df.loc[common, "daily_return"].values
            m = compute_metrics(regime_returns, trading_days=252)
            m["n_days"] = len(common)
            results[label][regime_name] = m

    return results


def covid_case_study(curves, regimes):
    """Analyze performance during COVID crash (Feb 19 - Mar 23, 2020)
    and recovery (Mar 23 - Jun 30, 2020)."""
    crash_start = pd.Timestamp("2020-02-19")
    crash_end = pd.Timestamp("2020-03-23")
    recovery_end = pd.Timestamp("2020-06-30")

    results = {}
    for label, df in curves.items():
        # Crash period
        crash_mask = (df.index >= crash_start) & (df.index <= crash_end)
        crash_returns = df.loc[crash_mask, "daily_return"].values

        # Recovery period
        recovery_mask = (df.index > crash_end) & (df.index <= recovery_end)
        recovery_returns = df.loc[recovery_mask, "daily_return"].values

        if len(crash_returns) < 2:
            continue

        crash_cum = np.prod(1 + crash_returns) - 1
        crash_vol = np.std(crash_returns, ddof=1) * np.sqrt(252)
        crash_dd = np.min(np.cumprod(1 + crash_returns) /
                          np.maximum.accumulate(np.cumprod(1 + crash_returns)) - 1)

        recovery_cum = np.prod(1 + recovery_returns) - 1 if len(recovery_returns) > 0 else 0

        # Regime distribution during crash
        crash_dates = df.index[crash_mask]
        regime_during_crash = []
        for d in crash_dates:
            if d in regimes.index:
                regime_during_crash.append(regimes.loc[d])
            else:
                idx = regimes.index.get_indexer([d], method="nearest")[0]
                regime_during_crash.append(regimes.iloc[idx])

        regime_counts = pd.Series(regime_during_crash).value_counts()

        results[label] = {
            "crash_return": crash_cum,
            "crash_volatility": crash_vol,
            "crash_max_dd": crash_dd,
            "crash_n_days": len(crash_returns),
            "recovery_return": recovery_cum,
            "recovery_n_days": len(recovery_returns),
        }

    return results


def bic_model_selection(features_train):
    """Compute BIC for K=2,3,4,5 states."""
    X = features_train.values
    results = {}
    for k in [2, 3, 4, 5]:
        model = GaussianHMM(
            n_components=k,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        model.fit(X)
        n_params = k * k + k * 3 + k * 6 - 1  # transition + means + full cov
        log_likelihood = model.score(X) * len(X)
        bic = -2 * log_likelihood + n_params * np.log(len(X))
        aic = -2 * log_likelihood + 2 * n_params
        results[k] = {
            "log_likelihood": log_likelihood,
            "bic": bic,
            "aic": aic,
            "n_params": n_params,
        }
    return results


def main():
    print("Loading dataset...")
    dataset = get_dataset()

    print("Loading equity curves...")
    curves = load_equity_curves()

    print("Fitting HMM and predicting regimes...")
    regimes, detector, features = get_regimes_for_test(dataset)

    # Test period regimes
    test_regimes = regimes[(regimes.index >= "2018-01-01") & (regimes.index < "2022-01-01")]

    # 1. Per-regime metrics
    print("\n" + "="*70)
    print("  PER-REGIME PERFORMANCE METRICS")
    print("="*70)
    pr_metrics = per_regime_metrics(curves, regimes)

    for regime_name in ["Low Vol", "Med Vol", "High Vol"]:
        print(f"\n--- {regime_name} Regime ---")
        for strategy in ["Baseline PPO", "MVO", "Regime-Switching PPO", "Equal Weight"]:
            if regime_name in pr_metrics.get(strategy, {}):
                m = pr_metrics[strategy][regime_name]
                print(f"  {strategy:25s}  Sharpe={m['sharpe_ratio']:+.3f}  "
                      f"Ann.Ret={m['annual_return']:+.2%}  "
                      f"Ann.Vol={m['annual_volatility']:.2%}  "
                      f"MaxDD={m['max_drawdown']:.2%}  "
                      f"Sortino={m['sortino_ratio']:+.3f}  "
                      f"Days={m['n_days']}")

    # 2. COVID case study
    print("\n" + "="*70)
    print("  COVID-19 CASE STUDY")
    print("="*70)
    covid = covid_case_study(curves, regimes)
    print(f"\n{'Strategy':25s} {'Crash Return':>14s} {'Crash Vol':>12s} {'Crash MaxDD':>13s} {'Recovery Ret':>14s}")
    print("-"*80)
    for label in ["Baseline PPO", "MVO", "Regime-Switching PPO", "Equal Weight"]:
        if label in covid:
            c = covid[label]
            print(f"{label:25s} {c['crash_return']:>13.2%} {c['crash_volatility']:>11.2%} "
                  f"{c['crash_max_dd']:>12.2%} {c['recovery_return']:>13.2%}")

    # Regime distribution during test
    print(f"\n--- Regime Distribution (2018-2022 test period) ---")
    regime_names = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}
    for rid, rname in regime_names.items():
        count = (test_regimes == rid).sum()
        pct = count / len(test_regimes) * 100
        print(f"  {rname}: {count} days ({pct:.1f}%)")

    # 3. BIC model selection
    print("\n" + "="*70)
    print("  BIC MODEL SELECTION")
    print("="*70)
    train_features = features.loc[:"2016-12-31"]
    bic_results = bic_model_selection(train_features)
    print(f"\n{'K':>4s} {'Log-Lik':>14s} {'BIC':>14s} {'AIC':>14s} {'Params':>8s}")
    print("-"*58)
    for k, r in sorted(bic_results.items()):
        marker = " <-- best" if r['bic'] == min(v['bic'] for v in bic_results.values()) else ""
        print(f"{k:>4d} {r['log_likelihood']:>14.1f} {r['bic']:>14.1f} {r['aic']:>14.1f} {r['n_params']:>8d}{marker}")

    # 4. Transition matrix
    print("\n" + "="*70)
    print("  HMM TRANSITION MATRIX")
    print("="*70)
    trans = detector.model.transmat_
    # Reorder to match sorted regime labels
    order = detector._regime_order
    sorted_trans = trans[np.ix_(order, order)]
    print(f"\n{'':>12s} {'→ Low Vol':>12s} {'→ Med Vol':>12s} {'→ High Vol':>12s}")
    for i, rname in enumerate(["Low Vol", "Med Vol", "High Vol"]):
        print(f"{rname:>12s}", end="")
        for j in range(3):
            print(f"{sorted_trans[i, j]:>12.4f}", end="")
        print()

    # 5. Per-regime S&P 500 characteristics
    print("\n" + "="*70)
    print("  PER-REGIME S&P 500 CHARACTERISTICS (training data)")
    print("="*70)
    train_regimes = regimes.loc[:"2016-12-31"]
    sp500_returns = np.log(dataset["sp500_prices"] / dataset["sp500_prices"].shift(1)).dropna()
    for rid, rname in regime_names.items():
        rdates = train_regimes[train_regimes == rid].index
        common = sp500_returns.index.intersection(rdates)
        rets = sp500_returns.loc[common]
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        print(f"  {rname}: Ann.Ret={ann_ret:+.2%}, Ann.Vol={ann_vol:.2%}, "
              f"Sharpe={sharpe:.2f}, Days={len(common)}")

    # Save all results as JSON
    output = {
        "per_regime_metrics": {},
        "covid_case_study": covid,
        "bic_model_selection": {str(k): v for k, v in bic_results.items()},
        "regime_distribution": {
            regime_names[rid]: int((test_regimes == rid).sum())
            for rid in regime_names
        },
    }
    for strategy, regimes_data in pr_metrics.items():
        output["per_regime_metrics"][strategy] = {}
        for rname, m in regimes_data.items():
            output["per_regime_metrics"][strategy][rname] = {
                k: float(v) for k, v in m.items()
            }

    out_path = RESULTS_DIR / "regime_case_study.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved all results to {out_path}")


if __name__ == "__main__":
    main()
