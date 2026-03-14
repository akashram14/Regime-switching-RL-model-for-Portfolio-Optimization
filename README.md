# Regime-Switching RL Model for Portfolio Optimization

This repository contains a reinforcement learning portfolio allocation system and an HMM-based market regime detection pipeline integrated through regime probability features.

## Project Structure

- `prepare_etf_dataset.py`: downloads ETF price data and builds `prices.csv`, `returns.csv`, and `features.csv`
- `generate_regimes.py`: fits the HMM on `SPY` and exports aligned regime probabilities to `data/regimes.csv`
- `env_portfolio.py`: Gymnasium portfolio environment with optional regime features via `use_regimes`
- `train_ppo.py`: trains two PPO agents with identical hyperparameters
  - baseline: no regime features
  - regime-aware: includes HMM regime probabilities
- `evaluate.py`: evaluates both trained models on matching observation spaces
- `hmm-detection-part/`: teammate HMM/regime detection codebase kept in-repo

## Models

Training produces:

- `models/ppo_baseline.zip`
- `models/ppo_regime_aware.zip`

## Evaluation Outputs

Evaluation writes separate result files for each model under `results/`, including:

- metrics JSON files
- equity curve CSV files
- cumulative return plots

## Environment Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt hmmlearn numpy pandas scikit-learn stable-baselines3 gymnasium
```

## Typical Workflow

1. Build or refresh the ETF dataset:

```powershell
.\.venv\Scripts\python.exe prepare_etf_dataset.py
```

2. Generate HMM regime probabilities:

```powershell
.\.venv\Scripts\python.exe generate_regimes.py
```

3. Train both PPO agents:

```powershell
.\.venv\Scripts\python.exe train_ppo.py
```

4. Evaluate both models:

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

## Current Observation Schemas

- Baseline PPO: 20 features
- Regime-aware PPO: 23 features

## Notes

- The RL environment supports toggling regime features with `use_regimes=True/False`.
- The HMM bridge uses `SPY` as the market proxy and exports `p_low_vol`, `p_med_vol`, and `p_high_vol`.
