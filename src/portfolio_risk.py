"""Mentat Phase 2.2 — Portfolio-level risk, sizing, and drawdown monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Position sizing ────────────────────────────────────────────────────────────

REGIME_SIZING = {
    "LOW-VOL TRENDING":  1.0,   # full size
    "HIGH-VOL RANGING":  0.5,   # half size
    "MEAN-REVERTING":    0.6,   # moderate
    "CRASH/CRISIS":      0.2,   # minimal / survival
    "UNCERTAIN":         0.3,   # reduced until signal clarifies
}


def regime_position_size(
    regime: str,
    base_capital: float,
    var_95: float,
    max_risk_per_trade: float = 0.02,
) -> dict[str, float]:
    """
    Combine regime sizing multiplier with VaR-based position sizing.

    Formula: position = (max_risk_per_trade * capital) / abs(VaR_95)
    Then apply regime multiplier to scale down in adverse regimes.

    max_risk_per_trade = 2% of capital at risk per position (configurable).
    """
    regime_mult  = REGIME_SIZING.get(regime, 0.3)
    var_amount   = abs(var_95) if var_95 != 0 else 0.02

    # Pure VaR-based size: how many rupees to put at risk
    raw_size     = (max_risk_per_trade * base_capital) / var_amount

    # Apply regime multiplier
    sized        = raw_size * regime_mult

    # Hard cap: never more than 20% of capital in a single name
    capped       = min(sized, base_capital * 0.20)

    return {
        "raw_size_inr":     round(raw_size, 0),
        "regime_size_inr":  round(sized, 0),
        "capped_size_inr":  round(capped, 0),
        "regime_mult":      regime_mult,
        "pct_of_capital":   round(capped / base_capital, 4),
    }


# ── Correlation matrix ─────────────────────────────────────────────────────────

def compute_regime_correlation(
    returns_dict: dict[str, pd.Series],
    regime_map: dict[str, str],
    target_regime: str,
    min_days: int = 15,
) -> pd.DataFrame:
    """
    Compute correlation matrix ONLY for days when each stock was in the target regime.

    This tells you: when CRASH/CRISIS hits, which stocks move together?
    A portfolio full of high-crisis-correlation names offers no diversification.
    """
    # Align all return series
    aligned = pd.DataFrame(returns_dict).dropna()

    # For each stock, mask to days it was in the target regime
    # Simple approximation: use all dates if we don't have per-day regime history
    # In Phase 2.3 you'll replace this with the full regime tape
    filtered = aligned  # placeholder — replace with regime-masked returns

    if len(filtered) < min_days:
        return pd.DataFrame()

    corr = filtered.corr()
    return corr.round(3)


def portfolio_var(
    returns_dict: dict[str, pd.Series],
    weights: dict[str, float],
    confidence: float = 0.95,
) -> dict[str, float]:
    """
    Portfolio-level VaR accounting for correlation.
    Far more accurate than summing individual VaRs (which ignores diversification).
    """
    tickers = list(weights.keys())
    aligned = pd.DataFrame({t: returns_dict[t] for t in tickers if t in returns_dict}).dropna()

    if aligned.empty or len(aligned) < 20:
        return {"portfolio_var_95": 0.0, "portfolio_cvar_95": 0.0}

    w = np.array([weights[t] for t in tickers if t in aligned.columns])
    w = w / w.sum()  # normalise

    port_rets = aligned.values @ w
    var  = float(np.percentile(port_rets, (1 - confidence) * 100))
    tail = port_rets[port_rets <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var

    return {
        "portfolio_var_95":  round(var, 4),
        "portfolio_cvar_95": round(cvar, 4),
        "annualised_vol":    round(float(port_rets.std() * np.sqrt(252)), 4),
        "sharpe":            round(
            float((port_rets.mean() - 0.065/252) / port_rets.std() * np.sqrt(252)), 2
        ) if port_rets.std() > 0 else 0.0,
    }


# ── Drawdown monitor ──────────────────────────────────────────────────────────

def drawdown_analysis(returns: pd.Series) -> dict[str, float]:
    """
    Full drawdown profile for a return series.
    Max drawdown, current drawdown, time underwater — all in one call.
    """
    cumulative = (1 + returns.dropna()).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max

    max_dd      = float(drawdown.min())
    current_dd  = float(drawdown.iloc[-1])

    # Time underwater = consecutive days below previous high
    underwater = (drawdown < 0).astype(int)
    streaks     = underwater * (underwater.groupby((underwater != underwater.shift()).cumsum()).cumcount() + 1)
    max_streak  = int(streaks.max()) if len(streaks) > 0 else 0

    return {
        "max_drawdown":       round(max_dd, 4),
        "current_drawdown":   round(current_dd, 4),
        "max_days_underwater": max_streak,
        "recovery_needed":    round(-current_dd / (1 + current_dd), 4) if current_dd < 0 else 0.0,
    }


# ── Portfolio summary ─────────────────────────────────────────────────────────

def build_portfolio_summary(
    scan_df: pd.DataFrame,
    returns_dict: dict[str, pd.Series],
    base_capital: float = 1_000_000,
    holdings: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Master portfolio risk table. One row per holding with:
    - Regime and confidence
    - Suggested position size
    - Individual drawdown
    - Regime-adjusted action

    holdings = {ticker: current_value_inr} — your actual positions.
    If None, uses equal-weight suggestion.
    """
    rows = []
    for _, stock in scan_df.iterrows():
        ticker  = stock["ticker"]
        regime  = stock["regime"]
        conf    = stock["confidence"]

        if ticker not in returns_dict:
            continue

        rets = returns_dict[ticker]
        dd   = drawdown_analysis(rets)
        var  = float(np.percentile(rets.dropna(), 5))

        sizing = regime_position_size(
            regime=regime,
            base_capital=base_capital,
            var_95=var,
        )

        current_val = (holdings or {}).get(ticker, 0)
        suggested   = sizing["capped_size_inr"]
        delta       = suggested - current_val

        action = _size_action(regime, current_val, suggested, dd["current_drawdown"])

        rows.append({
            "ticker":           ticker,
            "sector":           stock.get("sector", ""),
            "regime":           regime,
            "confidence":       conf,
            "regime_mult":      sizing["regime_mult"],
            "suggested_inr":    suggested,
            "current_inr":      current_val,
            "delta_inr":        round(delta, 0),
            "action":           action,
            "var_95":           round(var, 4),
            "max_dd":           dd["max_drawdown"],
            "current_dd":       dd["current_drawdown"],
            "days_underwater":  dd["max_days_underwater"],
        })

    return pd.DataFrame(rows).sort_values("regime_mult", ascending=False)


def _size_action(
    regime: str,
    current: float,
    suggested: float,
    current_dd: float,
) -> str:
    if regime == "CRASH/CRISIS":
        if current > 0:
            return "REDUCE — crisis regime, preserve capital"
        return "AVOID — crisis regime"
    if regime == "LOW-VOL TRENDING":
        if current == 0:
            return "CONSIDER ENTRY — trending, low vol"
        if current < suggested * 0.7:
            return "SIZE UP — regime supports larger position"
        return "HOLD — sized appropriately"
    if regime in ("HIGH-VOL RANGING", "UNCERTAIN"):
        if current > suggested:
            return "TRIM — reduce to regime-appropriate size"
        return "HOLD SMALL — wait for regime clarity"
    if regime == "MEAN-REVERTING":
        if current_dd < -0.10:
            return "MONITOR — deep drawdown in mean-reverting regime"
        return "HOLD — monitor for reversal signal"
    return "HOLD"
