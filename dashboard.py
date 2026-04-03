"""Mentat live radar dashboard.

Displays:
- Current active seed signals
- Current HMM state of ^NSEMDCP50 benchmark
- Simulated portfolio value path from seed backtest trades (base INR 7,500)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.data_ingestion import build_feature_matrix, fetch_stock_data, fetch_vix
from src.hmm_engine import decode_regime, load_hmm, passes_sniper_gate, train_hmm


st.set_page_config(page_title="Mentat Radar Dashboard", layout="wide")


def _latest_seed_trades() -> Path | None:
    base = Path("analysis/validation")
    files = sorted(base.glob("seed_backtest_trades_*.csv"))
    return files[-1] if files else None


def _portfolio_curve(trades: pd.DataFrame, starting_capital: float = 7500.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"date": [pd.Timestamp.today().normalize()], "portfolio": [starting_capital]})

    t = trades.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"], errors="coerce")
    t = t.dropna(subset=["exit_date"]).sort_values("exit_date")

    # Equal capital deployment per closed trade in this simulation.
    value = starting_capital
    rows = []
    for _, row in t.iterrows():
        ret = float(row.get("ret", 0.0))
        value *= 1.0 + ret
        rows.append({"date": row["exit_date"], "portfolio": value})

    if not rows:
        return pd.DataFrame({"date": [pd.Timestamp.today().normalize()], "portfolio": [starting_capital]})
    return pd.DataFrame(rows)


def _seed_active_signals() -> pd.DataFrame:
    excluded = set(config.MENTAT_EXCLUDED)
    tickers = [t for t in config.MENTAT_SEED_UNIVERSE if t not in excluded]

    vix = fetch_vix(config.LOOKBACK_YEARS)
    signals: list[dict[str, float | str]] = []

    for ticker in tickers:
        df = fetch_stock_data(ticker, config.LOOKBACK_YEARS)
        if df.empty:
            continue

        feat = build_feature_matrix(df, vix)
        if feat.empty or len(feat) < max(config.ROLLING_WINDOW, 120):
            continue

        obs_cols = ["log_ret_1d", "log_ret_20d", "rvol_20d", "rsi", "vol_zscore", "hurst_30d"]
        if any(c not in feat.columns for c in obs_cols):
            continue

        model_path = Path(config.MODEL_DIR) / f"{ticker}_hmm.pkl"
        try:
            if model_path.exists():
                model, scaler, labels, _quality = load_hmm(str(model_path))
            else:
                model, scaler, labels, _quality = train_hmm(
                    feature_df=feat,
                    observation_cols=obs_cols,
                    n_states=4,
                    model_path=str(model_path),
                )
        except Exception:
            continue

        decoded = decode_regime(
            feature_df=feat,
            model=model,
            scaler=scaler,
            observation_cols=obs_cols,
            rolling_window=min(config.ROLLING_WINDOW, len(feat)),
        )

        state = int(decoded["current_state"])
        regime = labels.get(state, f"REGIME_{state}")
        latest = feat.iloc[-1]
        vol_now = float(pd.to_numeric(df["Volume"].iloc[-1], errors="coerce"))
        vol_5 = float(pd.to_numeric(df["Volume"].rolling(5).mean().iloc[-1], errors="coerce"))
        vol_gate = pd.notna(vol_now) and pd.notna(vol_5) and vol_5 > 0 and vol_now > vol_5

        if passes_sniper_gate(
            regime_label=regime,
            rsi=float(latest["rsi"]),
            hurst=float(latest["hurst_30d"]),
            rsi_min=40,
            rsi_max=65,
            hurst_min=0.55,
        ) and vol_gate:
            signals.append(
                {
                    "ticker": ticker,
                    "regime": regime,
                    "rsi": round(float(latest["rsi"]), 2),
                    "hurst_30d": round(float(latest["hurst_30d"]), 3),
                    "vol_ratio_5d": round(vol_now / vol_5, 2),
                }
            )

    return pd.DataFrame(signals)


def _benchmark_state() -> tuple[str, float]:
    ticker = "^NSEMDCP50"
    df = fetch_stock_data(ticker, config.LOOKBACK_YEARS)
    if df.empty:
        return "N/A", 0.0

    vix = fetch_vix(config.LOOKBACK_YEARS)
    feat = build_feature_matrix(df, vix)
    if feat.empty or len(feat) < 120:
        return "N/A", 0.0

    obs_cols = ["log_ret_1d", "log_ret_20d", "rvol_20d", "rsi", "vol_zscore", "hurst_30d"]
    if any(c not in feat.columns for c in obs_cols):
        return "N/A", 0.0

    model_path = Path(config.MODEL_DIR) / "_benchmark_nsemdcp50_hmm.pkl"
    try:
        model, scaler, labels, _quality = train_hmm(
            feature_df=feat,
            observation_cols=obs_cols,
            n_states=3,
            model_path=str(model_path),
        )
        decoded = decode_regime(
            feature_df=feat,
            model=model,
            scaler=scaler,
            observation_cols=obs_cols,
            rolling_window=min(config.ROLLING_WINDOW, len(feat)),
        )
        state = int(decoded["current_state"])
        conf = float(decoded["state_probs"][state]) if state < len(decoded["state_probs"]) else float(max(decoded["state_probs"]))
        return labels.get(state, f"REGIME_{state}"), conf
    except Exception:
        return "N/A", 0.0


st.title("Mentat Live Radar")

trades_path = _latest_seed_trades()
trades_df = pd.read_csv(trades_path) if trades_path else pd.DataFrame()
portfolio_df = _portfolio_curve(trades_df, starting_capital=7500.0)
current_portfolio = float(portfolio_df["portfolio"].iloc[-1])

bench_state, bench_conf = _benchmark_state()
signals_df = _seed_active_signals()

c1, c2, c3 = st.columns(3)
c1.metric("Simulated Portfolio (INR)", f"{current_portfolio:,.2f}")
c2.metric("Benchmark State (^NSEMDCP50)", bench_state)
c3.metric("Benchmark State Confidence", f"{bench_conf:.0%}")

st.subheader("Current Active Signals")
if signals_df.empty:
    st.info("No active Hurst-qualified sniper signals right now.")
else:
    st.dataframe(signals_df, use_container_width=True, hide_index=True)

st.subheader("Portfolio Path (Base INR 7,500)")
fig = px.line(portfolio_df, x="date", y="portfolio", markers=True)
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

st.caption(f"Trades source: {trades_path if trades_path else 'No seed_backtest_trades file found'}")
