"""Risk metrics and outlier detection for Mentat."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR."""
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    return float(np.percentile(cleaned, (1 - confidence) * 100))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall)."""
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    var = compute_var(cleaned, confidence)
    tail = cleaned[cleaned <= var]
    if tail.empty:
        return var
    return float(tail.mean())


def compute_sharpe(returns: pd.Series, risk_free: float = 0.065) -> float:
    """Annualized Sharpe ratio."""
    cleaned = returns.dropna()
    if cleaned.std() == 0 or cleaned.empty:
        return 0.0
    excess = cleaned - risk_free / 252
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """Beta to benchmark returns."""
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0

    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    denom = cov[1, 1]
    if denom == 0:
        return 0.0
    return float(cov[0, 1] / denom)


def detect_outliers(feature_df: pd.DataFrame, z_threshold: float) -> dict:
    """Z-score outlier flags for key features."""
    if feature_df.empty:
        return {}

    today = feature_df.iloc[-1]
    flags: dict[str, dict[str, float]] = {}

    for col in ["log_ret_1d", "vol_zscore", "rvol_10d"]:
        series = feature_df[col].dropna()
        if len(series) < 20 or series.std() == 0:
            continue
        z = (today[col] - series.mean()) / series.std()
        if abs(z) > z_threshold:
            flags[col] = {"z_score": round(float(z), 2), "value": round(float(today[col]), 4)}

    return flags


def regime_risk_metrics(
    returns: pd.Series,
    regime_series: np.ndarray,
    current_state: int,
    confidence: float,
) -> dict:
    """Regime-conditioned risk metrics with fallback to full series."""
    cleaned = returns.dropna()
    if cleaned.empty:
        return {
            "regime_var_95": 0.0,
            "regime_cvar_95": 0.0,
            "regime_sharpe": 0.0,
            "regime_n_days": 0,
        }

    tail_rets = cleaned.tail(len(regime_series))
    mask = np.array(regime_series) == current_state
    regime_rets = tail_rets[mask]

    if len(regime_rets) < 10:
        regime_rets = cleaned

    return {
        "regime_var_95": round(compute_var(regime_rets, confidence), 4),
        "regime_cvar_95": round(compute_cvar(regime_rets, confidence), 4),
        "regime_sharpe": round(compute_sharpe(regime_rets), 2),
        "regime_n_days": int(len(regime_rets)),
    }
