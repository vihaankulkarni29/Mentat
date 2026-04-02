"""Mentat Phase 1 orchestrator: data -> features -> HMM -> risk -> report."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src import config
from src.backtest import bic_model_selection, regime_return_heatmap, walk_forward_backtest
from src.brief import build_morning_brief
from src.data_ingestion import build_feature_matrix, fetch_market_returns, fetch_stock_data, fetch_vix
from src.hmm_engine import decode_regime, label_states, load_hmm, train_hmm
from src.intelligence import run_intelligence_layer
from src.report import build_report, save_report, send_email
from src.risk_engine import compute_beta, detect_outliers, regime_risk_metrics


def compute_regime_persistence(regime_series: np.ndarray, current_state: int) -> dict[str, int | str]:
    """Count consecutive days in current regime and map to maturity bucket."""
    consecutive = 0
    for state in reversed(regime_series.tolist()):
        if int(state) == int(current_state):
            consecutive += 1
        else:
            break

    if consecutive <= 3:
        maturity = "FRESH (0-3d)"
    elif consecutive <= 10:
        maturity = "ESTABLISHING (4-10d)"
    elif consecutive <= 30:
        maturity = "ESTABLISHED (11-30d)"
    else:
        maturity = "MATURE (30d+)"

    return {
        "days_in_regime": consecutive,
        "regime_maturity": maturity,
    }


def run_pipeline(
    retrain: bool = False,
    tickers: list[str] | None = None,
    validate: bool = False,
) -> dict[str, dict]:
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

        # Phase 1.2 validation is intentionally run on a shorter recent window to keep feedback fast.
        validation_feat = feat.tail(max(config.WF_TRAIN_SIZE + config.WF_TEST_SIZE * 3, 756)) if validate else feat

        model_path = os.path.join(config.MODEL_DIR, f"{ticker}_hmm.pkl")
        if retrain or not os.path.exists(model_path):
            model, scaler, state_labels, model_quality = train_hmm(
                feature_df=validation_feat,
                observation_cols=config.OBSERVATION_COLS,
                n_states=config.N_STATES,
                model_path=model_path,
            )
        else:
            model, scaler, state_labels, model_quality = load_hmm(model_path)
            expected_features = len(config.OBSERVATION_COLS)
            actual_features = getattr(scaler, "n_features_in_", expected_features)
            if actual_features != expected_features:
                print(
                    f"[WARN] Model feature count mismatch for {ticker}: "
                    f"artifact expects {actual_features}, current pipeline uses {expected_features}. Retraining."
                )
                model, scaler, state_labels, model_quality = train_hmm(
                    feature_df=validation_feat,
                    observation_cols=config.OBSERVATION_COLS,
                    n_states=config.N_STATES,
                    model_path=model_path,
                )

            # Backward compatibility for older model artifacts missing labels.
            if not state_labels:
                state_labels = label_states(
                    feature_df=feat,
                    model=model,
                    scaler=scaler,
                    observation_cols=config.OBSERVATION_COLS,
                )

        decoded = decode_regime(
            feature_df=validation_feat,
            model=model,
            scaler=scaler,
            observation_cols=config.OBSERVATION_COLS,
            rolling_window=config.ROLLING_WINDOW,
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

        persistence = compute_regime_persistence(regime_series, current_state)

        returns = validation_feat["log_ret_1d"]
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

        validation_feat.to_csv(os.path.join(config.OUTPUT_DIR, f"{ticker}_feature_matrix.csv"))

        validation_summary = None
        if validate:
            os.makedirs(config.VALIDATION_DIR, exist_ok=True)

            bic = bic_model_selection(
                feature_df=validation_feat,
                observation_cols=config.OBSERVATION_COLS,
                n_states_range=(config.BIC_STATE_MIN, config.BIC_STATE_MAX),
            )
            bt_df = walk_forward_backtest(
                feature_df=validation_feat,
                observation_cols=config.OBSERVATION_COLS,
                n_states=config.N_STATES,
                train_size=config.WF_TRAIN_SIZE,
                test_size=config.WF_TEST_SIZE,
            )
            heatmap = regime_return_heatmap(bt_df)

            pd.Series(bic, name="bic").to_csv(
                os.path.join(config.VALIDATION_DIR, f"{ticker}_bic_scores.csv"), header=True
            )
            bt_df.to_csv(os.path.join(config.VALIDATION_DIR, f"{ticker}_walk_forward.csv"), index=False)
            heatmap.to_csv(os.path.join(config.VALIDATION_DIR, f"{ticker}_regime_heatmap.csv"))

            if not heatmap.empty:
                best_state = heatmap.index[0]
                bic_min_state = min(bic.items(), key=lambda item: item[1])[0]
                validation_summary = {
                    "best_regime": str(best_state),
                    "best_mean_ret": float(heatmap.iloc[0]["mean_ret"]),
                    "best_hit_rate": float(heatmap.iloc[0]["hit_rate"]),
                    "bic_min_state": int(bic_min_state),
                    "bic_scores": bic,
                }
            else:
                bic_min_state = min(bic.items(), key=lambda item: item[1])[0]
                validation_summary = {
                    "best_regime": "N/A",
                    "best_mean_ret": 0.0,
                    "best_hit_rate": 0.0,
                    "bic_min_state": int(bic_min_state),
                    "bic_scores": bic,
                }

        results[ticker] = {
            "regime_label": regime_label,
            "base_regime_label": base_regime_label,
            "regime_confidence": round(max_prob, 4),
            "state_probs": state_probs.tolist(),
            "state_labels": {str(k): v for k, v in state_labels.items()},
            "risk": risk,
            "outliers": outliers,
            "regime_history": regime_history,
            "persistence": persistence,
            "model_quality": model_quality,
            "validation": validation_summary,
        }

    report = build_report(results)
    print("\n" + report)
    report_path = save_report(report)
    print(f"\n[OK] Report saved: {report_path}")

    # Phase 3: Intelligence layer (news sentiment)
    print("\n[INTEL] Running news sentiment analysis...")
    sentiment_data = run_intelligence_layer(effective_tickers)

    # Build morning brief combining regime + outliers + sentiment
    morning_brief = build_morning_brief(results, sentiment_data)
    print("\n" + morning_brief)

    # Save morning brief
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    today = pd.Timestamp.now().date().isoformat()
    brief_path = os.path.join(config.REPORT_DIR, f"mentat_brief_{today}.txt")
    with open(brief_path, "w") as f:
        f.write(morning_brief)
    print(f"\n[OK] Morning brief saved: {brief_path}")

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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run Phase 1.2 validation (walk-forward, regime heatmap, BIC) and save outputs",
    )
    parser.add_argument(
        "--tickers",
        default="",
        help="Optional comma-separated tickers override (e.g., RELIANCE.NS,TCS.NS)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Run Phase 2.1 NSE universe scan",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    os.makedirs(config.VALIDATION_DIR, exist_ok=True)

    if args.scan:
        from src.universe import run_universe_scan, build_sector_regime_map, save_universe_scan
        scan_df   = run_universe_scan()
        sector_df = build_sector_regime_map(scan_df)

        print("\n=== SECTOR REGIME MAP ===")
        print(sector_df.to_string(index=False))
        print("\n=== UNIVERSE SCAN ===")
        print(scan_df.to_string(index=False))

        save_universe_scan(scan_df, sector_df)
    else:
        selected_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
        run_pipeline(retrain=args.retrain, tickers=selected_tickers, validate=args.validate)
