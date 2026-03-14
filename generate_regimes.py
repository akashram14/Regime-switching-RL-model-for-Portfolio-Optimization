from pathlib import Path
import sys

import pandas as pd


TRAIN_START = "2008-01-01"
TRAIN_END = "2018-12-31"
ROOT = Path(__file__).resolve().parent
HMM_SRC = ROOT / "hmm-detection-part" / "src"
PRICES_PATH = ROOT / "prices.csv"
RETURNS_PATH = ROOT / "returns.csv"
OUTPUT_PATH = ROOT / "data" / "regimes.csv"


def load_hmm_module():
    if not HMM_SRC.exists():
        raise FileNotFoundError(f"HMM source directory not found: {HMM_SRC}")

    sys.path.insert(0, str(HMM_SRC))
    try:
        from hmm import RegimeDetector, compute_hmm_features
    except ImportError as exc:
        raise ImportError(
            "Failed to import RegimeDetector and compute_hmm_features from "
            f"{HMM_SRC / 'hmm.py'}"
        ) from exc

    return RegimeDetector, compute_hmm_features


def load_rl_dates() -> pd.DatetimeIndex:
    returns_df = pd.read_csv(RETURNS_PATH, parse_dates=["Date"])
    if "Date" not in returns_df.columns:
        raise ValueError("returns.csv must include a 'Date' column.")

    dates = pd.DatetimeIndex(returns_df["Date"]).sort_values().unique()
    return pd.DatetimeIndex(dates)


def load_spy_prices() -> pd.Series:
    prices_df = pd.read_csv(PRICES_PATH, parse_dates=["Date"])
    required_cols = {"Date", "SPY"}
    missing_cols = required_cols.difference(prices_df.columns)
    if missing_cols:
        raise ValueError(f"prices.csv is missing required columns: {sorted(missing_cols)}")

    prices_df = prices_df.loc[:, ["Date", "SPY"]].dropna(subset=["SPY"])
    prices_df = prices_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return pd.Series(prices_df["SPY"].values, index=prices_df["Date"], name="SPY")


def main() -> None:
    RegimeDetector, compute_hmm_features = load_hmm_module()

    rl_dates = load_rl_dates()
    spy_prices = load_spy_prices()

    hmm_features = compute_hmm_features(spy_prices)
    train_features = hmm_features.loc[TRAIN_START:TRAIN_END]
    if train_features.empty:
        raise ValueError("No HMM training features found in the requested training period.")

    detector = RegimeDetector(n_regimes=3, random_state=42)
    detector.fit(train_features)

    regime_proba = detector.predict_proba(hmm_features)
    aligned = regime_proba.reindex(rl_dates).sort_index()
    aligned = aligned.ffill().bfill()
    aligned.index.name = "Date"

    output_df = aligned.reset_index()
    output_df["Date"] = pd.to_datetime(output_df["Date"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Detected regimes: {detector.n_regimes}")
    print(f"Output rows: {len(output_df)}")
    print(
        "Date range: "
        f"{output_df['Date'].min().date()} to {output_df['Date'].max().date()}"
    )
    print(output_df.head(10).to_string(index=False))
    print(f"Saved regimes to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
