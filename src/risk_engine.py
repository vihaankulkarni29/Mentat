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


def _robust_zscore(series: pd.Series, value: float) -> float:
    """MAD-based z-score fallback when classic std z-score is unstable."""
    cleaned = series.dropna()
    if len(cleaned) < 10:
        return 0.0

    median = float(cleaned.median())
    mad = float((cleaned - median).abs().median())
    if mad == 0:
        return 0.0

    # 0.6745 rescales MAD to std-equivalent under normal assumptions.
    return float(0.6745 * (value - median) / mad)


def detect_outliers(feature_df: pd.DataFrame, z_threshold: float) -> dict:
    """Multi-horizon outlier engine to surface asymmetric growth/dislocation moves."""
    if feature_df.empty:
        return {}

    today = feature_df.iloc[-1]
    flags: dict[str, dict[str, float | str]] = {}
    total_score = 0.0

    feature_weights = {
        "log_ret_1d": 1.4,
        "vol_zscore": 1.0,
        "rvol_10d": 0.9,
    }

    for col in ["log_ret_1d", "vol_zscore", "rvol_10d"]:
        if col not in feature_df.columns or pd.isna(today.get(col)):
            continue

        value = float(today[col])
        weight = feature_weights.get(col, 1.0)

        # Full-history context (classic z-score)
        series = feature_df[col].dropna()
        if len(series) >= 20 and series.std() > 0:
            z = (value - float(series.mean())) / float(series.std())
            if abs(z) > z_threshold:
                flags[col] = {
                    "z_score": round(float(z), 2),
                    "value": round(value, 4),
                    "context": "vs full history",
                }
                total_score += max(abs(float(z)) - z_threshold, 0.0) * weight

        # Robust median/MAD context
        z_robust = _robust_zscore(series, value)
        if abs(z_robust) > z_threshold:
            flags[f"{col}_robust"] = {
                "z_score": round(float(z_robust), 2),
                "value": round(value, 4),
                "context": "vs robust median/MAD",
            }
            total_score += max(abs(float(z_robust)) - z_threshold, 0.0) * (0.7 * weight)

        # Multi-horizon short-term shock contexts
        for window in (5, 20, 60):
            recent = feature_df[col].tail(window).dropna()
            if len(recent) < max(5, window // 3) or recent.std() == 0:
                continue
            z_recent = (value - float(recent.mean())) / float(recent.std())
            if abs(z_recent) > z_threshold:
                key = f"{col}_{window}d"
                flags[key] = {
                    "z_score": round(float(z_recent), 2),
                    "value": round(value, 4),
                    "context": f"vs {window}d rolling",
                }
                horizon_multiplier = 1.15 if window == 5 else (1.0 if window == 20 else 0.85)
                total_score += max(abs(float(z_recent)) - z_threshold, 0.0) * weight * horizon_multiplier

    # Capture percentile tail event for one-day returns.
    if "log_ret_1d" in feature_df.columns and not pd.isna(today.get("log_ret_1d")):
        ret_series = feature_df["log_ret_1d"].dropna()
        if len(ret_series) >= 30:
            pct_rank = float((ret_series <= float(today["log_ret_1d"])).mean())
            if pct_rank <= 0.03 or pct_rank >= 0.97:
                direction = "UPSIDE" if pct_rank >= 0.97 else "DOWNSIDE"
                flags["log_ret_1d_tail"] = {
                    "percentile": round(pct_rank, 3),
                    "value": round(float(today["log_ret_1d"]), 4),
                    "context": f"{direction} tail event",
                }
                total_score += 1.25

    if not flags:
        return {}

    # Composite metadata for ranking opportunities in brief/dashboard.
    outlier_score = round(float(total_score), 2)
    if outlier_score >= 6:
        severity = "EXTREME"
    elif outlier_score >= 3:
        severity = "HIGH"
    else:
        severity = "MODERATE"

    # Directional hint based on return and volume pressure.
    ret = float(today.get("log_ret_1d", 0.0))
    rv = float(today.get("rvol_10d", 1.0))
    if ret > 0 and rv > 1.2:
        opportunity_bias = "BULLISH_BREAKOUT"
    elif ret < 0 and rv > 1.2:
        opportunity_bias = "BEARISH_BREAKDOWN"
    else:
        opportunity_bias = "DISLOCATION"

    flags["outlier_meta"] = {
        "score": outlier_score,
        "severity": severity,
        "bias": opportunity_bias,
    }
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
