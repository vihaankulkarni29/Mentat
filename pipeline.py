"""Mentat Phase 1 orchestrator: data -> features -> HMM -> risk -> report."""

from __future__ import annotations

import argparse
import os

import numpy as np

from src import config
from src.data_ingestion import build_feature_matrix, fetch_market_returns, fetch_stock_data, fetch_vix
from src.hmm_engine import decode_regime, label_states, load_hmm, train_hmm
from src.report import build_report, save_report, send_email
from src.risk_engine import compute_beta, detect_outliers, regime_risk_metrics


def run_pipeline(retrain: bool = False, tickers: list[str] | None = None) -> dict[str, dict]:
    """Run full Mentat daily pipeline for all configured tickers."""
    print("\n" + "=" * 72)
    print("MENTAT PHASE 1 - HMM REGIME PIPELINE")
    print("=" * 72)

    vix = fetch_vix(config.LOOKBACK_YEARS)
    market_rets = fetch_market_returns(config.MARKET_BENCHMARK, config.LOOKBACK_YEARS)
    results: dict[str, dict] = {}

    effective_tickers = tickers if tickers else config.TICKERS

    for ticker in effective_tickers:
        print(f"[RUN] {ticker}")
        df = fetch_stock_data(ticker, config.LOOKBACK_YEARS)
        if df.empty:
            print(f"[WARN] No data for {ticker}, skipping")
            continue

        feat = build_feature_matrix(df, vix)
        if feat.empty:
            print(f"[WARN] Feature matrix empty for {ticker}, skipping")
            continue

        model_path = os.path.join(config.MODEL_DIR, f"{ticker}_hmm.pkl")
        if retrain or not os.path.exists(model_path):
            model, scaler = train_hmm(
                feature_df=feat,
                observation_cols=config.OBSERVATION_COLS,
                n_states=config.N_STATES,
                model_path=model_path,
            )
        else:
            model, scaler = load_hmm(model_path)

        decoded = decode_regime(
            feature_df=feat,
            model=model,
            scaler=scaler,
            observation_cols=config.OBSERVATION_COLS,
            rolling_window=config.ROLLING_WINDOW,
        )
        state_labels = label_states(
            feature_df=feat,
            model=model,
            scaler=scaler,
            observation_cols=config.OBSERVATION_COLS,
        )
        current_state = decoded["current_state"]
        state_probs = decoded["state_probs"]
        posteriors = decoded["posteriors"]
        transition_matrix = decoded["transition_matrix"]
        regime_series = decoded["regime_series"]
        base_regime_label = state_labels.get(current_state, f"REGIME_{current_state}")

        max_prob = float(np.max(state_probs))
        if max_prob < config.UNCERTAIN_CONFIDENCE_THRESHOLD:
            regime_label = "UNCERTAIN / TRANSITION"
        else:
            regime_label = base_regime_label

        history_n = min(config.HISTORY_DAYS, len(regime_series))
        history_dates = decoded["window_index"][-history_n:]
        history_states = regime_series[-history_n:]
        history_post = posteriors[-history_n:]

        regime_history = []
        for i in range(history_n):
            state_id = int(history_states[i])
            state_label = state_labels.get(state_id, f"REGIME_{state_id}")
            conf = float(np.max(history_post[i]))
            if conf < config.UNCERTAIN_CONFIDENCE_THRESHOLD:
                state_label = "UNCERTAIN / TRANSITION"

            regime_history.append(
                {
                    "date": str(history_dates[i].date()),
                    "state": state_id,
                    "regime": state_label,
                    "confidence": round(conf, 4),
                }
            )

        returns = feat["log_ret_1d"]
        risk = regime_risk_metrics(
            returns=returns,
            regime_series=regime_series,
            current_state=current_state,
            confidence=config.VAR_CONFIDENCE,
        )
        aligned_market = market_rets.reindex(returns.index)
        risk["beta"] = round(compute_beta(returns.tail(60), aligned_market.tail(60)), 2)

        outliers = detect_outliers(feat, config.VOL_OUTLIER_Z)

        trans_away = 1 - transition_matrix[current_state, current_state]
        if trans_away > 0.3:
            outliers["regime_shift_risk"] = {"prob_leaving_state": round(float(trans_away), 2)}

        feat.to_csv(os.path.join(config.OUTPUT_DIR, f"{ticker}_feature_matrix.csv"))

        results[ticker] = {
            "regime_label": regime_label,
            "base_regime_label": base_regime_label,
            "regime_confidence": round(max_prob, 4),
            "state_probs": state_probs.tolist(),
            "risk": risk,
            "outliers": outliers,
            "regime_history": regime_history,
        }

    report = build_report(results)
    print("\n" + report)
    report_path = save_report(report)
    print(f"\n[OK] Report saved: {report_path}")

    if config.SEND_EMAIL:
        send_email(report)
        print("[OK] Report emailed")

    print("=" * 72 + "\n")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mentat Phase 1 - HMM Regime Pipeline")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain HMM models before decoding today's regime",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    run_pipeline(retrain=args.retrain)
