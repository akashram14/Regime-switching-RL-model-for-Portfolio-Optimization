from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
START_DATE = "2008-01-01"
END_DATE = "2024-12-31"
OUTPUT_DIR = Path(".")


def download_adjusted_close(
    tickers: list[str], start: str, end_inclusive: str
) -> pd.DataFrame:
    # yfinance uses an exclusive end date, so add one day to include END_DATE.
    end_exclusive = (pd.to_datetime(end_inclusive) + pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
    )

    if "Adj Close" not in raw.columns.get_level_values(0):
        raise ValueError("Adjusted close prices were not returned by yfinance.")

    prices = raw["Adj Close"].copy().sort_index()
    prices = prices.dropna(how="all")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna(how="all")
    return returns


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    vol20 = returns.rolling(window=20).std().add_suffix("_vol20")
    vol60 = returns.rolling(window=60).std().add_suffix("_vol60")

    features = pd.concat(
        [
            returns.add_suffix("_ret"),
            vol20,
            vol60,
        ],
        axis=1,
    )
    return features


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc["2008-01-01":"2018-12-31"].copy()
    test = df.loc["2019-01-01":"2024-12-31"].copy()
    return train, test


def plot_cumulative_returns(returns: pd.DataFrame, output_dir: Path) -> None:
    cumulative = np.exp(returns.cumsum()) - 1

    plt.figure(figsize=(12, 7))
    for ticker in returns.columns:
        plt.plot(cumulative.index, cumulative[ticker], label=ticker, linewidth=1.3)

    plt.title("Cumulative Returns (Log-Return Compounded)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_returns.png", dpi=150)
    plt.show()


def main() -> None:
    prices = download_adjusted_close(TICKERS, START_DATE, END_DATE)
    returns = compute_log_returns(prices)
    features = build_features(returns)

    train_features, test_features = split_train_test(features)

    prices.to_csv(OUTPUT_DIR / "prices.csv", index=True)
    returns.to_csv(OUTPUT_DIR / "returns.csv", index=True)
    features.to_csv(OUTPUT_DIR / "features.csv", index=True)

    plot_cumulative_returns(returns, OUTPUT_DIR)

    print("Saved:")
    print(f"- {OUTPUT_DIR / 'prices.csv'}")
    print(f"- {OUTPUT_DIR / 'returns.csv'}")
    print(f"- {OUTPUT_DIR / 'features.csv'}")
    print(f"- {OUTPUT_DIR / 'cumulative_returns.png'}")
    print("")
    print("Feature split:")
    print(
        f"- Train (2008-01-01 to 2018-12-31): {train_features.shape[0]} rows, "
        f"{train_features.shape[1]} columns"
    )
    print(
        f"- Test  (2019-01-01 to 2024-12-31): {test_features.shape[0]} rows, "
        f"{test_features.shape[1]} columns"
    )


if __name__ == "__main__":
    main()
