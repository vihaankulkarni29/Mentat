"""Mentat Phase 2.1 — NSE universe regime scanner."""

from __future__ import annotations

import os
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from src import config
from src.data_ingestion import build_feature_matrix, fetch_stock_data, fetch_vix
from src.hmm_engine import decode_regime, load_hmm, train_hmm

# Liquid NSE universe — expand this over time
NSE_UNIVERSE = [
    # Large cap
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "BAJAJ-AUTO.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    # Mid cap additions
    "PIDILITIND.NS", "GODREJCP.NS", "MCDOWELL-N.NS", "VOLTAS.NS", "PERSISTENT.NS",
    "COFORGE.NS", "LTIM.NS", "MPHASIS.NS", "DIXON.NS", "TRENT.NS",
]

SECTOR_MAP = {
    "RELIANCE.NS":    "Energy/Telecom",
    "TCS.NS":         "IT",
    "HDFCBANK.NS":    "Banking",
    "INFY.NS":        "IT",
    "ICICIBANK.NS":   "Banking",
    "HINDUNILVR.NS":  "FMCG",
    "SBIN.NS":        "Banking",
    "BHARTIARTL.NS":  "Telecom",
    "KOTAKBANK.NS":   "Banking",
    "LT.NS":          "Infrastructure",
    "AXISBANK.NS":    "Banking",
    "ASIANPAINT.NS":  "Paints",
    "MARUTI.NS":      "Auto",
    "SUNPHARMA.NS":   "Pharma",
    "TITAN.NS":       "Consumer",
    "BAJAJ-AUTO.NS":  "Auto",
    "WIPRO.NS":       "IT",
    "HCLTECH.NS":     "IT",
    "ULTRACEMCO.NS":  "Cement",
    "NESTLEIND.NS":   "FMCG",
    "PIDILITIND.NS":  "Chemicals",
    "GODREJCP.NS":    "FMCG",
    "MCDOWELL-N.NS":  "Consumer",
    "VOLTAS.NS":      "Consumer Durables",
    "PERSISTENT.NS":  "IT",
    "COFORGE.NS":     "IT",
    "LTIM.NS":        "IT",
    "MPHASIS.NS":     "IT",
    "DIXON.NS":       "Consumer Durables",
    "TRENT.NS":       "Retail",
}


def _scan_single_ticker(ticker: str, vix: pd.Series) -> dict | None:
    """Scan one ticker — designed to run in a thread pool."""
    try:
        df = fetch_stock_data(ticker, config.LOOKBACK_YEARS)
        if df.empty:
            return None

        feat = build_feature_matrix(df, vix)
        if feat.empty or len(feat) < config.ROLLING_WINDOW + 20:
            return None

        model_path = os.path.join(config.MODEL_DIR, f"{ticker}_hmm.pkl")

        if os.path.exists(model_path):
            model, scaler, state_labels, model_quality = load_hmm(model_path)
            # Auto-retrain if feature schema changed
            expected = len(config.OBSERVATION_COLS)
            actual = getattr(scaler, "n_features_in_", expected)
            if actual != expected:
                model, scaler, state_labels, model_quality = train_hmm(
                    feat, config.OBSERVATION_COLS, config.N_STATES, model_path
                )
        else:
            model, scaler, state_labels, model_quality = train_hmm(
                feat, config.OBSERVATION_COLS, config.N_STATES, model_path
            )

        decoded = decode_regime(
            feat, model, scaler, config.OBSERVATION_COLS, config.ROLLING_WINDOW
        )

        current_state = decoded["current_state"]
        state_probs   = decoded["state_probs"]
        max_prob      = float(np.max(state_probs))
        base_label    = state_labels.get(current_state, f"REGIME_{current_state}")
        regime_label  = "UNCERTAIN" if max_prob < config.UNCERTAIN_CONFIDENCE_THRESHOLD else base_label

        # Persistence
        consecutive = 0
        for s in reversed(decoded["regime_series"].tolist()):
            if int(s) == current_state:
                consecutive += 1
            else:
                break

        # Transition risk: probability of leaving current state
        trans_away = 1.0 - float(decoded["transition_matrix"][current_state, current_state])

        return {
            "ticker":        ticker,
            "sector":        SECTOR_MAP.get(ticker, "Unknown"),
            "regime":        regime_label,
            "confidence":    round(max_prob, 3),
            "persistence_d": consecutive,
            "trans_risk":    round(trans_away, 3),
            "min_persist":   model_quality.get("min_persistence", 0.0) if model_quality else 0.0,
        }

    except Exception as e:
        print(f"  [ERR] {ticker}: {e}")
        return None


def run_universe_scan(
    tickers: list[str] | None = None,
    max_workers: int = 6,
) -> pd.DataFrame:
    """
    Scan the full NSE universe in parallel.
    Returns a DataFrame sorted by regime severity and transition risk.
    """
    universe = tickers or NSE_UNIVERSE
    vix = fetch_vix(config.LOOKBACK_YEARS)

    print(f"\n[UNIVERSE SCAN] {len(universe)} tickers | {date.today()}")

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scan_single_ticker, t, vix): t for t in universe}
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)
                print(
                    f"  {result['ticker']:20s} "
                    f"{result['regime']:22s} "
                    f"conf={result['confidence']:.0%} "
                    f"persist={result['persistence_d']}d"
                )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Regime severity order for sorting
    severity = {
        "CRASH/CRISIS":     0,
        "HIGH-VOL RANGING": 1,
        "UNCERTAIN":        2,
        "MEAN-REVERTING":   3,
        "LOW-VOL TRENDING": 4,
    }
    df["_severity"] = df["regime"].map(lambda r: severity.get(str(r), 2))
    df = df.sort_values(["_severity", "trans_risk"], ascending=[True, False])
    df = df.drop(columns=["_severity"])

    return df


def build_sector_regime_map(scan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate regime distribution by sector.
    Tells you: is IT sector in crisis? Is Banking mixed?
    This is the market-wide intelligence layer.
    """
    if scan_df.empty:
        return pd.DataFrame()

    sector_summary = []
    for sector, grp in scan_df.groupby("sector"):
        regime_counts = grp["regime"].value_counts().to_dict()
        dominant      = grp["regime"].value_counts().index[0]
        crisis_pct    = regime_counts.get("CRASH/CRISIS", 0) / len(grp)
        trending_pct  = regime_counts.get("LOW-VOL TRENDING", 0) / len(grp)
        avg_trans     = grp["trans_risk"].mean()

        sector_summary.append({
            "sector":         sector,
            "n_stocks":       len(grp),
            "dominant":       dominant,
            "crisis_pct":     round(crisis_pct, 2),
            "trending_pct":   round(trending_pct, 2),
            "avg_trans_risk": round(avg_trans, 3),
            "regime_mix":     str(regime_counts),
        })

    return pd.DataFrame(sector_summary).sort_values("crisis_pct", ascending=False)


def save_universe_scan(scan_df: pd.DataFrame, sector_df: pd.DataFrame) -> str:
    """Persist universe scan and sector map to analysis directory."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    today = date.today().isoformat()

    scan_path   = os.path.join(config.OUTPUT_DIR, f"universe_scan_{today}.csv")
    sector_path = os.path.join(config.OUTPUT_DIR, f"sector_regime_{today}.csv")

    scan_df.to_csv(scan_path, index=False)
    sector_df.to_csv(sector_path, index=False)

    print(f"[OK] Universe scan saved: {scan_path}")
    print(f"[OK] Sector map saved:    {sector_path}")
    return scan_path
