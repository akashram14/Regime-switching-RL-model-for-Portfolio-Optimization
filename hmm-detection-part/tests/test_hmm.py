"""Quick smoke test for src/hmm.py"""
import numpy as np
import pandas as pd
from src.hmm import RegimeDetector, compute_hmm_features

np.random.seed(42)
dates = pd.date_range("2010-01-01", periods=500, freq="B")

prices_list = [100.0]
for i in range(499):
    if i < 200:
        ret = np.random.normal(0.0005, 0.008)
    elif i < 350:
        ret = np.random.normal(-0.001, 0.025)
    else:
        ret = np.random.normal(0.0003, 0.012)
    prices_list.append(prices_list[-1] * np.exp(ret))

prices = pd.Series(prices_list, index=dates, name="price")

features = compute_hmm_features(prices)
print(f"Features shape: {features.shape}")
print(f"Columns: {list(features.columns)}")

detector = RegimeDetector(n_regimes=3, random_state=42)
detector.fit(features)
print(f"Fit OK, log-likelihood: {detector.score(features):.2f}")

regimes = detector.predict(features)
print(f"Regime counts: {dict(regimes.value_counts().sort_index())}")

print(f"\nTransition matrix:")
print(detector.transition_matrix.round(3).to_string())

print(f"\nStationary dist: {dict(detector.stationary_distribution.round(3))}")

r = detector.predict_single(features.iloc[-1].values)
print(f"Latest regime: {r}")

proba = detector.predict_proba(features.tail(3))
print(f"\nPosterior probs (last 3 days):")
print(proba.round(4).to_string())

print("\nAll tests passed!")
