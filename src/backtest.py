"""Phase 1.2 regime validation utilities."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]

from src import config
from src.hmm_engine import decode_regime, train_hmm


def walk_forward_backtest(
    feature_df: pd.DataFrame,
    observation_cols: list[str],
    n_states: int,
    train_size: int = 504,
    test_size: int = 21,
) -> pd.DataFrame:
    """Rolling train/test evaluation for regime-conditioned out-of-sample returns."""
    results: list[dict] = []
    total = len(feature_df)

    temp_model_path = os.path.join(config.MODEL_DIR, "_wf_hmm_tmp.pkl")

    for start in range(0, total - train_size - test_size, test_size):
        train_df = feature_df.iloc[start : start + train_size]
        test_df = feature_df.iloc[start + train_size : start + train_size + test_size].copy()

        # Forward return from prediction day to next day.
        test_df["fwd_ret_1d"] = test_df["log_ret_1d"].shift(-1)

        try:
            model, scaler, labels, _quality = train_hmm(
                feature_df=train_df,
                observation_cols=observation_cols,
                n_states=n_states,
                model_path=temp_model_path,
            )
        except ValueError:
            continue

        decoded = decode_regime(
            feature_df=pd.concat([train_df.tail(60), test_df]),
            model=model,
            scaler=scaler,
            observation_cols=observation_cols,
            rolling_window=60 + len(test_df),
        )

        test_regimes = decoded["regime_series"][-len(test_df) :]

        for i, state in enumerate(test_regimes):
            fwd_ret = test_df["fwd_ret_1d"].iloc[i]
            if pd.isna(fwd_ret):
                continue

            results.append(
                {
                    "date": test_df.index[i],
                    "state": int(state),
                    "label": labels.get(int(state), f"S{int(state)}"),
                    "fwd_ret": float(fwd_ret),
                }
            )

    return pd.DataFrame(results)


def regime_return_heatmap(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize forward-return behavior per inferred regime."""
    if backtest_df.empty:
        return pd.DataFrame(columns=["mean_ret", "std_ret", "n_days", "hit_rate", "sharpe_proxy"])

    summary = (
        backtest_df.groupby("label")["fwd_ret"]
        .agg(["mean", "std", "count", lambda x: (x > 0).mean()])
        .round(4)
    )
    summary.columns = ["mean_ret", "std_ret", "n_days", "hit_rate"]
    summary["sharpe_proxy"] = (summary["mean_ret"] / summary["std_ret"] * np.sqrt(252)).replace(
        [np.inf, -np.inf], np.nan
    )
    summary["sharpe_proxy"] = summary["sharpe_proxy"].round(2)
    return summary.sort_values("mean_ret", ascending=False)


def bic_model_selection(
    feature_df: pd.DataFrame,
    observation_cols: list[str],
    n_states_range: Tuple[int, int] = (2, 6),
) -> dict[int, float]:
    """Estimate BIC across candidate state counts (lower is better)."""
    X = feature_df[observation_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(X_scaled)

    bic_scores: dict[int, float] = {}
    low, high = n_states_range

    for k in range(low, high + 1):
        model = GaussianHMM(
            n_components=k,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )
        model.fit(X_scaled)
        log_lik = float(model.score(X_scaled)) * n
        n_features = len(observation_cols)

        # Approximate parameter count for full-cov Gaussian HMM.
        n_params = (k - 1) + (k * (k - 1)) + (k * n_features) + (k * n_features * (n_features + 1) / 2)
        bic = -2 * log_lik + n_params * np.log(n)
        bic_scores[k] = round(float(bic), 1)

    return bic_scores
