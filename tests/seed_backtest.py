"""Mentat seed-universe HMM backtest (train 2022-2023, test 2024).

Architecture:
- Universe: Mentat seed universe (information-asymmetry sectors)
- Train: 2022-01-01 -> 2023-12-31
- Test:  2024-01-01 -> 2024-12-31
- Entry: MEAN-REVERTING -> LOW-VOL TRENDING + RSI gate + volume > 5-day average
- Exit: leaves LOW-VOL TRENDING OR RSI > 70 OR stop-loss <= -8%

Outputs:
- analysis/validation/seed_backtest_trades_YYYY-MM-DD.csv
- analysis/validation/seed_backtest_equity_YYYY-MM-DD.csv
- analysis/validation/seed_backtest_summary_YYYY-MM-DD.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_ingestion import build_feature_matrix
from src.hmm_engine import train_hmm


@dataclass
class Trade:
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    ret: float
    reason: str
    hold_days: int


def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, timeout=20)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().copy()


def _download_vix(start: str, end: str) -> pd.Series:
    vix = _download_ohlcv(config.VIX_TICKER, start, end)
    if vix.empty:
        return pd.Series(dtype=float, name="VIX")
    return vix["Close"].rename("VIX")


def _daily_sharpe(returns: pd.Series) -> float:
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    std = float(cleaned.std())
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float((float(cleaned.mean()) / std) * np.sqrt(252))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _load_benchmark_returns(start: str, end: str) -> tuple[pd.Series, str]:
    for bench in config.MENTAT_BENCHMARKS:
        df = _download_ohlcv(bench, start, end)
        if df.empty:
            continue
        rets = df["Close"].pct_change().dropna()
        if not rets.empty:
            return rets, bench
    return pd.Series(dtype=float), "N/A"


def _regime_for_series(
    feature_df: pd.DataFrame,
    observation_cols: list[str],
    n_states: int,
    model_path: str,
) -> tuple[pd.Series, dict[int, str], int]:
    model, scaler, labels, _quality = train_hmm(
        feature_df=feature_df,
        observation_cols=observation_cols,
        n_states=n_states,
        model_path=model_path,
    )

    X = feature_df[observation_cols].values
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)

    mapped = []
    for state in states:
        mapped.append(labels.get(int(state), f"REGIME_{int(state)}"))

    regime_series = pd.Series(mapped, index=feature_df.index, name="regime")
    return regime_series, labels, n_states


def run_seed_backtest(
    tickers: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    output_dir: str,
    rsi_min: float,
    rsi_max: float,
    retrain_days: int,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # Include buffer before train window for stable feature initialization.
    data_start = "2021-01-01"
    data_end = (pd.Timestamp(test_end) + pd.Timedelta(days=2)).date().isoformat()

    print("=" * 72)
    print("MENTAT SEED-UNIVERSE BACKTEST")
    print("=" * 72)
    print(f"Train: {train_start} -> {train_end}")
    print(f"Test:  {test_start} -> {test_end}")
    print(f"Universe size: {len(tickers)}")
    print(f"Entry RSI band: {rsi_min} -> {rsi_max}")
    print("Volume filter: current volume > 5-day average volume")
    print(f"Walk-forward retraining cadence: every {retrain_days} trading days")

    vix = _download_vix(data_start, data_end)

    trades: list[Trade] = []
    daily_book: dict[pd.Timestamp, float] = {}

    for ticker in tickers:
        print(f"[RUN] {ticker}")
        ohlcv = _download_ohlcv(ticker, data_start, data_end)
        if ohlcv.empty:
            print("  [WARN] no data")
            continue

        feat = build_feature_matrix(ohlcv, vix)
        feat = feat.loc[feat.index.intersection(ohlcv.index)].copy()
        if len(feat) < 300:
            print("  [WARN] not enough feature rows")
            continue

        train_feat = feat.loc[(feat.index >= train_start) & (feat.index <= train_end)].copy()
        test_feat = feat.loc[(feat.index >= test_start) & (feat.index <= test_end)].copy()

        if len(train_feat) < 200 or len(test_feat) < 20:
            print("  [WARN] insufficient train/test split")
            continue

        # Use momentum + volatility + RSI + volume pressure.
        obs_cols = ["log_ret_1d", "log_ret_20d", "rvol_20d", "rsi", "vol_zscore"]
        if any(col not in feat.columns for col in obs_cols):
            print("  [WARN] required features unavailable")
            continue

        model_path = os.path.join(config.MODEL_DIR, f"_seed_bt_{ticker}.pkl")

        model = None
        scaler = None
        labels: dict[int, str] = {}
        n_used = 0
        last_retrain_i = -1

        test_idx = test_feat.index
        in_position = False
        entry_price = 0.0
        entry_date: pd.Timestamp | None = None

        for i in range(2, len(test_idx)):
            day = test_idx[i]
            prev_day = test_idx[i - 1]

            # Walk-forward retrain: initial train and then periodic adaptation.
            if model is None or (i - last_retrain_i) >= retrain_days:
                adaptive_train = feat.loc[(feat.index >= train_start) & (feat.index <= prev_day)].copy()
                if len(adaptive_train) < 200:
                    continue

                try:
                    model, scaler, labels, _quality = train_hmm(
                        feature_df=adaptive_train,
                        observation_cols=obs_cols,
                        n_states=4,
                        model_path=model_path,
                    )
                    n_used = 4
                except Exception:
                    try:
                        model, scaler, labels, _quality = train_hmm(
                            feature_df=adaptive_train,
                            observation_cols=obs_cols,
                            n_states=3,
                            model_path=model_path,
                        )
                        n_used = 3
                        print("  [WARN] 4-state fit unstable; using 3-state fallback")
                    except Exception as exc:
                        print(f"  [WARN] walk-forward train failed: {exc}")
                        continue

                last_retrain_i = i

            if model is None or scaler is None:
                continue

            hist_prev = feat.loc[(feat.index >= train_start) & (feat.index <= prev_day)].copy()
            hist_today = feat.loc[(feat.index >= train_start) & (feat.index <= day)].copy()
            if len(hist_prev) < 50 or len(hist_today) < 50:
                continue

            X_prev = scaler.transform(hist_prev[obs_cols].values)
            X_today = scaler.transform(hist_today[obs_cols].values)
            st_prev = int(model.predict(X_prev)[-1])
            st_today = int(model.predict(X_today)[-1])

            regime_prev = labels.get(st_prev, f"REGIME_{st_prev}")
            regime_today = labels.get(st_today, f"REGIME_{st_today}")

            row_today = test_feat.loc[day]

            close_today = float(pd.to_numeric(ohlcv.loc[day, "Close"], errors="coerce"))
            if not np.isfinite(close_today) or close_today <= 0:
                continue

            if not in_position:
                # In 3-state fallback, use HIGH-VOL RANGING as the mean-reversion proxy.
                mr_proxy = "MEAN-REVERTING" if n_used >= 4 else "HIGH-VOL RANGING"
                shifted = regime_prev == mr_proxy and regime_today == "LOW-VOL TRENDING"
                rsi_gate = rsi_min <= float(row_today["rsi"]) <= rsi_max

                vol_now = float(pd.to_numeric(ohlcv.loc[day, "Volume"], errors="coerce"))
                vol_5dma = float(pd.to_numeric(ohlcv["Volume"].rolling(5).mean().loc[day], errors="coerce"))
                vol_gate = np.isfinite(vol_now) and np.isfinite(vol_5dma) and vol_5dma > 0 and vol_now > vol_5dma

                if shifted and rsi_gate and vol_gate:
                    in_position = True
                    entry_price = close_today
                    entry_date = day
                continue

            # Manage active trade.
            ret_now = close_today / entry_price - 1.0
            exit_regime = regime_today != "LOW-VOL TRENDING"
            exit_rsi = float(row_today["rsi"]) > 70
            stop_hit = ret_now <= -0.08

            if exit_regime or exit_rsi or stop_hit:
                reason = "REGIME_EXIT" if exit_regime else ("RSI_OVERHEAT" if exit_rsi else "STOP_LOSS")
                hold_days = int((day - entry_date).days) if entry_date is not None else 0
                trade = Trade(
                    ticker=ticker,
                    entry_date=entry_date.date().isoformat() if entry_date is not None else day.date().isoformat(),
                    exit_date=day.date().isoformat(),
                    entry_price=round(entry_price, 2),
                    exit_price=round(close_today, 2),
                    ret=round(ret_now, 6),
                    reason=reason,
                    hold_days=hold_days,
                )
                trades.append(trade)

                exit_ts = pd.Timestamp(day)
                daily_book[exit_ts] = daily_book.get(exit_ts, 0.0) + float(ret_now)

                in_position = False
                entry_price = 0.0
                entry_date = None

    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    if trades_df.empty:
        ret_by_day = pd.Series([0.0], index=[pd.Timestamp(test_start)], dtype=float)
        eq = (1.0 + ret_by_day).cumprod()
    else:
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        trades_df = trades_df.sort_values("exit_date").reset_index(drop=True)

        # Build equity from realized trade returns on exit dates.
        ret_by_day = trades_df.groupby("exit_date")["ret"].mean().sort_index()
        eq = (1.0 + ret_by_day).cumprod()

    bench_rets, bench_name = _load_benchmark_returns(test_start, data_end)
    if not bench_rets.empty:
        bench_slice = bench_rets.loc[
            (bench_rets.index >= pd.Timestamp(test_start))
            & (bench_rets.index <= pd.Timestamp(test_end))
        ]
        bench_eq = (1.0 + bench_slice).cumprod() if not bench_slice.empty else pd.Series(dtype=float)
    else:
        bench_eq = pd.Series(dtype=float)

    strategy_total = float(eq.iloc[-1] - 1.0)
    benchmark_total = float(bench_eq.iloc[-1] - 1.0) if not bench_eq.empty else 0.0

    win_rate = float((trades_df["ret"] > 0).mean()) if not trades_df.empty else 0.0
    sharpe = _daily_sharpe(ret_by_day)
    max_dd = _max_drawdown(eq)

    summary = {
        "train_window": f"{train_start} -> {train_end}",
        "test_window": f"{test_start} -> {test_end}",
        "universe_n": len(tickers),
        "rsi_band": f"{rsi_min}-{rsi_max}",
        "retrain_days": retrain_days,
        "trades": int(len(trades_df)),
        "strategy_total_return": round(strategy_total, 4),
        "benchmark_name": bench_name,
        "benchmark_total_return": round(benchmark_total, 4),
        "alpha_vs_benchmark": round(strategy_total - benchmark_total, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
    }

    stamp = date.today().isoformat()
    trades_path = os.path.join(output_dir, f"seed_backtest_trades_{stamp}.csv")
    equity_path = os.path.join(output_dir, f"seed_backtest_equity_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"seed_backtest_summary_{stamp}.txt")

    trades_df.to_csv(trades_path, index=False)

    equity_df = pd.DataFrame({
        "date": eq.index,
        "strategy_equity": eq.values,
    })
    if not bench_eq.empty:
        aligned = bench_eq.reindex(eq.index).ffill()
        equity_df["benchmark_equity"] = aligned.values
    equity_df.to_csv(equity_path, index=False)

    lines = [
        "=" * 72,
        "MENTAT SEED BACKTEST SUMMARY",
        "=" * 72,
        f"Train:                {summary['train_window']}",
        f"Test:                 {summary['test_window']}",
        f"Universe size:        {summary['universe_n']}",
        f"RSI band:             {summary['rsi_band']}",
        f"Retrain cadence:      {summary['retrain_days']} trading days",
        "Volume gate:          current volume > 5-day average volume",
        f"Trades:               {summary['trades']}",
        "",
        f"Strategy return:      {summary['strategy_total_return']:.2%}",
        f"Benchmark ({summary['benchmark_name']}): {summary['benchmark_total_return']:.2%}",
        f"Alpha vs benchmark:   {summary['alpha_vs_benchmark']:.2%}",
        f"Sharpe ratio:         {summary['sharpe']}",
        f"Max drawdown:         {summary['max_drawdown']:.2%}",
        f"Win rate:             {summary['win_rate']:.2%}",
        "",
        f"Trades CSV:           {trades_path}",
        f"Equity CSV:           {equity_path}",
        f"Summary TXT:          {summary_path}",
        "=" * 72,
    ]

    report = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    return {
        "summary": summary,
        "trades_path": trades_path,
        "equity_path": equity_path,
        "summary_path": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mentat seed-universe HMM backtest")
    parser.add_argument("--tickers", default="", help="Optional comma-separated ticker override")
    parser.add_argument("--train-start", default="2022-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument("--rsi-min", type=float, default=40.0)
    parser.add_argument("--rsi-max", type=float, default=65.0)
    parser.add_argument("--retrain-days", type=int, default=5, help="Walk-forward retrain frequency in trading days")
    parser.add_argument("--output-dir", default="analysis/validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tickers.strip():
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        excluded = set(config.MENTAT_EXCLUDED)
        tickers = [t for t in config.MENTAT_SEED_UNIVERSE if t not in excluded]

    run_seed_backtest(
        tickers=tickers,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        output_dir=args.output_dir,
        rsi_min=float(args.rsi_min),
        rsi_max=float(args.rsi_max),
        retrain_days=max(1, int(args.retrain_days)),
    )


if __name__ == "__main__":
    main()
