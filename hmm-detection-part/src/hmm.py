"""
HMM-based Market Regime Detection.

Uses a Gaussian Hidden Markov Model to classify market conditions into
3 regimes (low/medium/high volatility) based on observation features:
  - Daily log returns
  - 20-day rolling volatility
  - Moving averages (20-day, 60-day, and their ratio)

This module is independent of the RL agents. It provides:
  1. RegimeDetector class — fit HMM, predict regimes, get transition matrix
  2. Feature engineering — compute observation vectors for the HMM
  3. Clean interface — regime_label = detector.predict(features_today)

Architecture (from your diagram):
  YFinance data → [log returns, rolling vol, moving avg] → HMM → regime (0,1,2)
  regime → routes to RL Agent (low vol) | RL Agent (med vol) | RL Agent (high vol)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from hmmlearn.hmm import GaussianHMM


# ---------------------------------------------------------------------------
# Feature Engineering for HMM observations
# ---------------------------------------------------------------------------

def compute_hmm_features(
    prices: pd.Series,
    vol_window: int = 20,
    ma_short: int = 20,
    ma_long: int = 60,
) -> pd.DataFrame:
    """
    Compute the observation feature vectors for the HMM.
    
    Features:
        1. daily_log_return: log(P_t / P_{t-1})
        2. rolling_vol: 20-day rolling std of log returns
        3. ma_ratio: MA(20) / MA(60)  — short-term vs long-term trend
    
    Args:
        prices: Price series (e.g., S&P 500 close prices)
        vol_window: Rolling window for volatility (default 20)
        ma_short: Short moving average window (default 20)
        ma_long: Long moving average window (default 60)
    
    Returns:
        DataFrame with columns: [daily_log_return, rolling_vol, ma_ratio]
        Rows with NaN (from rolling windows) are dropped.
    """
    log_returns = np.log(prices / prices.shift(1))
    rolling_vol = log_returns.rolling(window=vol_window).std()
    
    ma_short_vals = prices.rolling(window=ma_short).mean()
    ma_long_vals = prices.rolling(window=ma_long).mean()
    ma_ratio = ma_short_vals / ma_long_vals
    
    features = pd.DataFrame({
        "daily_log_return": log_returns,
        "rolling_vol": rolling_vol,
        "ma_ratio": ma_ratio,
    }, index=prices.index)
    
    features = features.dropna()
    return features


# ---------------------------------------------------------------------------
# Regime Detector (Gaussian HMM)
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Gaussian HMM-based market regime detector.
    
    Fits a 3-state Gaussian HMM on observation features and classifies
    each timestep into a regime:
        0 = Low volatility  (calm/bull market)
        1 = Medium volatility (transition/normal)
        2 = High volatility (crisis/bear market)
    
    After fitting, regimes are automatically sorted so that state 0 has
    the lowest mean volatility and state 2 has the highest.
    
    Usage:
        detector = RegimeDetector(n_regimes=3)
        detector.fit(train_features)
        regimes = detector.predict(test_features)
        current_regime = detector.predict_single(today_features)
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
        tol: float = 1e-4,
    ):
        """
        Args:
            n_regimes: Number of hidden states (default 3: low/med/high vol)
            covariance_type: "full", "diag", "spherical", or "tied"
            n_iter: Max EM iterations for fitting
            random_state: Random seed for reproducibility
            tol: Convergence threshold for EM
        """
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol,
        )
        self._is_fitted = False
        self._regime_order = None  # mapping from HMM states to sorted regimes
    
    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        """
        Fit the HMM on training features.
        
        After fitting, states are reordered so that:
            regime 0 = lowest mean rolling_vol
            regime 2 = highest mean rolling_vol
        
        Args:
            features: DataFrame from compute_hmm_features()
                      Columns: [daily_log_return, rolling_vol, ma_ratio]
        
        Returns:
            self (for chaining)
        """
        X = features.values
        self.model.fit(X)
        self._is_fitted = True
        
        # Sort regimes by mean rolling volatility (column index 1)
        # so regime 0 = low vol, regime 2 = high vol
        mean_vols = self.model.means_[:, 1]  # rolling_vol column
        self._regime_order = np.argsort(mean_vols)  # [lowest, mid, highest]
        
        self._feature_columns = list(features.columns)
        return self
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for a sequence of observations.
        
        Args:
            features: DataFrame from compute_hmm_features()
        
        Returns:
            Series of regime labels (0=low vol, 1=med vol, 2=high vol)
            indexed the same as the input features.
        """
        assert self._is_fitted, "Must call fit() before predict()"
        
        X = features.values
        raw_states = self.model.predict(X)
        
        # Remap to sorted order
        sorted_states = np.zeros_like(raw_states)
        for new_label, old_label in enumerate(self._regime_order):
            sorted_states[raw_states == old_label] = new_label
        
        return pd.Series(sorted_states, index=features.index, name="regime")
    
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Get posterior probabilities for each regime at each timestep.
        
        Returns:
            DataFrame with columns [p_low, p_med, p_high], indexed like features.
        """
        assert self._is_fitted, "Must call fit() before predict_proba()"
        
        X = features.values
        raw_proba = self.model.predict_proba(X)
        
        # Reorder columns to match sorted regime labels
        sorted_proba = raw_proba[:, self._regime_order]
        
        return pd.DataFrame(
            sorted_proba,
            index=features.index,
            columns=["p_low_vol", "p_med_vol", "p_high_vol"],
        )
    
    def predict_single(self, feature_vector: np.ndarray) -> int:
        """
        Predict regime for a single observation (for online/live use).
        
        This is the key interface for the RL agent routing:
            regime = detector.predict_single(today_features)
            agent = agents[regime]  # select the right RL agent
            action = agent.predict(state)
        
        Args:
            feature_vector: 1D array of shape (n_features,)
        
        Returns:
            Regime label (0, 1, or 2)
        """
        assert self._is_fitted, "Must call fit() before predict_single()"
        
        X = feature_vector.reshape(1, -1)
        raw_state = self.model.predict(X)[0]
        
        # Map to sorted label
        for new_label, old_label in enumerate(self._regime_order):
            if raw_state == old_label:
                return int(new_label)
        return int(raw_state)
    
    @property
    def transition_matrix(self) -> pd.DataFrame:
        """
        Get the transition probability matrix (sorted by regime).
        
        T[i,j] = P(regime_j at t+1 | regime_i at t)
        
        Returns:
            DataFrame with regime labels as both index and columns.
        """
        assert self._is_fitted, "Must call fit() before accessing transition_matrix"
        
        raw_T = self.model.transmat_
        
        # Reorder rows and columns
        sorted_T = raw_T[self._regime_order][:, self._regime_order]
        
        labels = ["Low Vol", "Med Vol", "High Vol"]
        return pd.DataFrame(sorted_T, index=labels, columns=labels)
    
    @property
    def regime_means(self) -> pd.DataFrame:
        """
        Get the mean observation vector for each regime (sorted).
        
        Returns:
            DataFrame with regime labels as index, feature names as columns.
        """
        assert self._is_fitted, "Must call fit() before accessing regime_means"
        
        sorted_means = self.model.means_[self._regime_order]
        labels = ["Low Vol", "Med Vol", "High Vol"]
        return pd.DataFrame(
            sorted_means,
            index=labels,
            columns=self._feature_columns,
        )
    
    @property
    def regime_covariances(self) -> List[pd.DataFrame]:
        """Get covariance matrices for each regime (sorted)."""
        assert self._is_fitted, "Must call fit() before accessing regime_covariances"
        
        covs = []
        for new_label in range(self.n_regimes):
            old_label = self._regime_order[new_label]
            cov = self.model.covars_[old_label]
            covs.append(pd.DataFrame(
                cov,
                index=self._feature_columns,
                columns=self._feature_columns,
            ))
        return covs
    
    @property
    def stationary_distribution(self) -> pd.Series:
        """
        Compute the stationary distribution of the Markov chain.
        This tells you the long-run fraction of time spent in each regime.
        """
        assert self._is_fitted, "Must call fit()"
        
        T = self.model.transmat_[self._regime_order][:, self._regime_order]
        
        # Solve pi @ T = pi, sum(pi) = 1
        n = T.shape[0]
        A = np.vstack([T.T - np.eye(n), np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        
        labels = ["Low Vol", "Med Vol", "High Vol"]
        return pd.Series(pi, index=labels, name="stationary_prob")
    
    def score(self, features: pd.DataFrame) -> float:
        """
        Log-likelihood of the observation sequence under the fitted model.
        Higher is better. Useful for model selection (e.g., choosing n_regimes).
        """
        assert self._is_fitted, "Must call fit() before score()"
        return self.model.score(features.values)
    
    def get_regime_periods(self, regimes: pd.Series) -> Dict[int, List[Tuple]]:
        """
        Extract contiguous periods for each regime.
        
        Returns:
            Dict mapping regime_label -> list of (start_date, end_date) tuples.
        """
        periods = {i: [] for i in range(self.n_regimes)}
        
        current_regime = regimes.iloc[0]
        start_date = regimes.index[0]
        
        for date, regime in regimes.items():
            if regime != current_regime:
                periods[current_regime].append((start_date, prev_date))
                current_regime = regime
                start_date = date
            prev_date = date
        
        # Close last period
        periods[current_regime].append((start_date, regimes.index[-1]))
        
        return periods
