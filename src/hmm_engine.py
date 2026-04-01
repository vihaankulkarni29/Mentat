"""HMM training, persistence, and regime decoding for Mentat."""

from __future__ import annotations

import os
import pickle
from typing import TypedDict

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]


class DecodedRegime(TypedDict):
    current_state: int
    state_probs: np.ndarray
    posteriors: np.ndarray
    transition_matrix: np.ndarray
    regime_series: np.ndarray
    window_index: pd.Index


def train_hmm(
    feature_df: pd.DataFrame,
    observation_cols: list[str],
    n_states: int,
    model_path: str,
) -> tuple[GaussianHMM, StandardScaler]:
    """Train Gaussian HMM and persist model + scaler."""
    X = feature_df[observation_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        tol=1e-4,
        random_state=42,
    )
    model.fit(X_scaled)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    return model, scaler


def load_hmm(model_path: str) -> tuple[GaussianHMM, StandardScaler]:
    """Load persisted HMM and scaler."""
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"]


def decode_regime(
    feature_df: pd.DataFrame,
    model: GaussianHMM,
    scaler: StandardScaler,
    observation_cols: list[str],
    rolling_window: int,
) -> DecodedRegime:
    """Decode current regime and state probabilities with rolling Viterbi window."""
    window = feature_df[observation_cols].tail(rolling_window)
    X = window.values
    X_scaled = scaler.transform(X)

    regime_series = model.predict(X_scaled)
    current_state = int(regime_series[-1])

    posteriors = model.predict_proba(X_scaled)
    state_probs = posteriors[-1]

    return {
        "current_state": current_state,
        "state_probs": state_probs,
        "posteriors": posteriors,
        "transition_matrix": model.transmat_,
        "regime_series": regime_series,
        "window_index": window.index,
    }


def _state_profiles(
    feature_df: pd.DataFrame,
    model: GaussianHMM,
    scaler: StandardScaler,
    observation_cols: list[str],
) -> dict[int, dict[str, float]]:
    """Build per-state empirical profiles from decoded full-history observations."""
    X = feature_df[observation_cols].values
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)

    profiled = feature_df.copy()
    profiled["_state"] = states

    profiles: dict[int, dict[str, float]] = {}
    for state in sorted(profiled["_state"].unique()):
        s = profiled[profiled["_state"] == state]
        returns = s["log_ret_1d"].dropna()

        autocorr = 0.0
        if len(returns) > 2:
            ac = returns.autocorr(lag=1)
            autocorr = 0.0 if pd.isna(ac) else float(ac)

        profiles[int(state)] = {
            "mean_ret": float(returns.mean()) if not returns.empty else 0.0,
            "vol": float(s["rvol_10d"].mean()) if "rvol_10d" in s else 0.0,
            "abs_mean_ret": float(abs(returns.mean())) if not returns.empty else 0.0,
            "rsi_mean": float(s["rsi"].mean()) if "rsi" in s else 50.0,
            "autocorr": autocorr,
        }

    return profiles


def label_states(
    feature_df: pd.DataFrame,
    model: GaussianHMM,
    scaler: StandardScaler,
    observation_cols: list[str],
) -> dict[int, str]:
    """Phase 1.1 labeling by multi-metric signature instead of return-only sorting."""
    profiles = _state_profiles(feature_df, model, scaler, observation_cols)
    states = list(profiles.keys())
    if len(states) < 4:
        return {state: f"REGIME_{i}" for i, state in enumerate(states)}

    labels: dict[int, str] = {}

    # 1) Crash/Crisis: worst average returns.
    crash_state = min(states, key=lambda s: profiles[s]["mean_ret"])
    labels[crash_state] = "CRASH/CRISIS"

    remaining = [s for s in states if s != crash_state]

    # 2) Low-vol trending: good returns, lower vol, RSI bias above neutral.
    trending_state = max(
        remaining,
        key=lambda s: (
            profiles[s]["mean_ret"]
            - 0.5 * profiles[s]["vol"]
            + 0.001 * (profiles[s]["rsi_mean"] - 50)
        ),
    )
    labels[trending_state] = "LOW-VOL TRENDING"

    remaining = [s for s in remaining if s != trending_state]

    # 3) High-vol ranging: high vol with low directional drift.
    ranging_state = max(
        remaining,
        key=lambda s: profiles[s]["vol"] - 2.0 * profiles[s]["abs_mean_ret"],
    )
    labels[ranging_state] = "HIGH-VOL RANGING"

    remaining = [s for s in remaining if s != ranging_state]

    # 4) Mean-reverting: residual state, usually with weaker trend persistence.
    for state in remaining:
        labels[state] = "MEAN-REVERTING"

    return labels
