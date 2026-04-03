"""Mentat blind walk-forward backtest (2024 crucible)."""

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
from src.hmm_engine import decode_regime, train_hmm
from src.risk_engine import regime_risk_metrics


@dataclass
class FrictionConfig:
    transaction_cost: float = 0.0005  # 0.05%
    slippage: float = 0.0010  # 0.10% per side


def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        timeout=20,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna().copy()


def _download_vix(start: str, end: str) -> pd.Series:
    vix = _download_ohlcv(config.VIX_TICKER, start=start, end=end)
    if vix.empty:
        return pd.Series(dtype=float, name="VIX")
    return vix["Close"].rename("VIX")


def _base_position_from_regime(regime_label: str, allow_short_crisis: bool) -> float:
    regime_map = {
        "LOW-VOL TRENDING": 0.20,
        "MEAN-REVERTING": 0.10,
        "HIGH-VOL RANGING": 0.05,
        "UNCERTAIN / TRANSITION": 0.03,
        "CRASH/CRISIS": -0.10 if allow_short_crisis else 0.00,
    }
    return regime_map.get(regime_label, 0.0)


def _kelly_fraction(trade_returns: pd.Series) -> float:
    """Estimate Kelly fraction from historical realized trade returns."""
    if len(trade_returns) < 20:
        return 0.5

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    if len(losses) == 0:
        return 1.0

    p = len(wins) / len(trade_returns)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = abs(float(losses.mean()))

    if avg_loss == 0:
        return 1.0

    b = avg_win / avg_loss if avg_loss > 0 else 0.0
    if b <= 0:
        return 0.0

    raw_kelly = p - (1 - p) / b
    return float(np.clip(raw_kelly, 0.0, 1.0))


def _compute_cagr(equity_curve: pd.Series) -> float:
    if equity_curve.empty or equity_curve.iloc[0] <= 0:
        return 0.0

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_days = len(equity_curve)
    if n_days < 2:
        return 0.0

    return float(total_return ** (252 / n_days) - 1)


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def _profit_factor(pnl_series: pd.Series) -> float:
    gross_profit = float(pnl_series[pnl_series > 0].sum())
    gross_loss = abs(float(pnl_series[pnl_series < 0].sum()))
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def run_blind_backtest(
    tickers: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    friction: FrictionConfig,
    allow_short_crisis: bool,
    output_dir: str,
    max_days: int | None = None,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print("=" * 72)
    print("MENTAT BLIND WALK-FORWARD BACKTEST")
    print("=" * 72)
    print(f"Train window: {train_start} -> {train_end}")
    print(f"Test window:  {test_start} -> {test_end}")
    print(
        f"Friction: tx_cost={friction.transaction_cost:.4%}, "
        f"slippage_per_side={friction.slippage:.4%}"
    )
    print(f"Universe: {', '.join(tickers)}")

    full_start = train_start
    full_end = (pd.Timestamp(test_end) + pd.Timedelta(days=2)).date().isoformat()

    vix = _download_vix(start=full_start, end=full_end)

    ohlcv_by_ticker: dict[str, pd.DataFrame] = {}
    feat_by_ticker: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        ohlcv = _download_ohlcv(ticker, start=full_start, end=full_end)
        if ohlcv.empty:
            print(f"[WARN] {ticker}: no data, skipping")
            continue

        feat = build_feature_matrix(ohlcv, vix)
        if feat.empty:
            print(f"[WARN] {ticker}: no feature matrix, skipping")
            continue

        # Only keep features where we have same-day open/close for trade execution.
        feat = feat.loc[feat.index.intersection(ohlcv.index)]
        if feat.empty:
            print(f"[WARN] {ticker}: no aligned feature/price rows, skipping")
            continue

        ohlcv_by_ticker[ticker] = ohlcv
        feat_by_ticker[ticker] = feat

    if not feat_by_ticker:
        raise RuntimeError("No valid ticker data available for blind backtest")

    # Build common test dates in the blind test period.
    all_test_dates = pd.DatetimeIndex([])
    for feat in feat_by_ticker.values():
        dates = feat.loc[(feat.index >= test_start) & (feat.index <= test_end)].index
        all_test_dates = all_test_dates.union(pd.DatetimeIndex(dates))

    all_test_dates = all_test_dates.sort_values()
    if max_days is not None:
        all_test_dates = all_test_dates[:max_days]

    trades: list[dict[str, Any]] = []

    for i, decision_day in enumerate(all_test_dates, 1):
        print(f"[DAY {i}/{len(all_test_dates)}] Decision date: {decision_day.date()}")

        day_portfolio_ret = 0.0
        day_active_positions = 0

        for ticker in tickers:
            if ticker not in feat_by_ticker:
                continue

            feat = feat_by_ticker[ticker]
            ohlcv = ohlcv_by_ticker[ticker]

            if decision_day not in feat.index:
                continue

            current_idx_raw = feat.index.get_indexer(pd.DatetimeIndex([decision_day]))[0]
            if not isinstance(current_idx_raw, (int, np.integer)):
                continue
            current_idx = int(current_idx_raw)
            if current_idx <= 0:
                continue

            prev_day = feat.index[current_idx - 1]

            # Strict blind mask: model sees only data up to previous trading day.
            train_feat = feat.loc[(feat.index >= train_start) & (feat.index <= prev_day)].copy()

            if len(train_feat) < max(120, config.ROLLING_WINDOW + 5):
                continue

            temp_model_path = os.path.join(config.MODEL_DIR, f"_blind_tmp_{ticker}.pkl")

            try:
                model, scaler, labels, _quality = train_hmm(
                    feature_df=train_feat,
                    observation_cols=config.OBSERVATION_COLS,
                    n_states=config.N_STATES,
                    model_path=temp_model_path,
                )
            except Exception as exc:
                print(f"  [WARN] {ticker}: train failed ({exc})")
                continue

            decoded = decode_regime(
                feature_df=train_feat,
                model=model,
                scaler=scaler,
                observation_cols=config.OBSERVATION_COLS,
                rolling_window=min(config.ROLLING_WINDOW, len(train_feat)),
            )

            state = int(decoded["current_state"])
            state_probs = decoded["state_probs"]
            regime_conf = float(np.max(state_probs))
            regime_label = labels.get(state, f"REGIME_{state}")
            if regime_conf < config.UNCERTAIN_CONFIDENCE_THRESHOLD:
                regime_label = "UNCERTAIN / TRANSITION"

            # Risk estimate from in-regime historical returns up to the decision day.
            risk = regime_risk_metrics(
                returns=train_feat["log_ret_1d"],
                regime_series=decoded["regime_series"],
                current_state=state,
                confidence=config.VAR_CONFIDENCE,
            )

            base_position = _base_position_from_regime(regime_label, allow_short_crisis)

            # Kelly scaling from past realized trades for this ticker only.
            hist_rets = pd.Series(
                [t["net_trade_return"] for t in trades if t["ticker"] == ticker],
                dtype=float,
            )
            kelly = _kelly_fraction(hist_rets)
            # Half-Kelly, clipped for stability.
            kelly_scale = float(np.clip(0.5 * kelly + 0.5, 0.25, 1.0))
            position = base_position * kelly_scale

            if position == 0:
                continue

            if decision_day not in ohlcv.index:
                continue

            open_next = float(pd.to_numeric(ohlcv.at[decision_day, "Open"], errors="coerce"))
            close_next = float(pd.to_numeric(ohlcv.at[decision_day, "Close"], errors="coerce"))
            if open_next <= 0 or close_next <= 0:
                continue

            # Execute from same-day open to same-day close using the pre-open prediction.
            underlying_ret = (close_next / open_next) - 1.0
            gross_trade_return = position * underlying_ret

            # Round-trip friction: entry + exit.
            friction_penalty = abs(position) * (2 * friction.slippage + friction.transaction_cost)
            net_trade_return = gross_trade_return - friction_penalty

            # VaR/CVaR breach checks on underlying return (model risk telemetry).
            var95 = float(risk.get("regime_var_95", 0.0))
            cvar95 = float(risk.get("regime_cvar_95", 0.0))
            var_breach = int(underlying_ret < var95)
            cvar_breach = int(underlying_ret < cvar95)

            day_portfolio_ret += net_trade_return
            day_active_positions += 1

            trades.append(
                {
                    "decision_day": decision_day.date().isoformat(),
                    "execution_day": decision_day.date().isoformat(),
                    "ticker": ticker,
                    "regime": regime_label,
                    "regime_confidence": round(regime_conf, 4),
                    "state": state,
                    "kelly_fraction": round(kelly, 4),
                    "kelly_scale": round(kelly_scale, 4),
                    "base_position": round(base_position, 4),
                    "position": round(position, 4),
                    "underlying_ret": round(underlying_ret, 6),
                    "gross_trade_return": round(gross_trade_return, 6),
                    "friction_penalty": round(friction_penalty, 6),
                    "net_trade_return": round(net_trade_return, 6),
                    "var95": round(var95, 6),
                    "cvar95": round(cvar95, 6),
                    "var_breach": var_breach,
                    "cvar_breach": cvar_breach,
                }
            )

        if day_active_positions == 0:
            trades.append(
                {
                    "decision_day": decision_day.date().isoformat(),
                    "execution_day": pd.NaT,
                    "ticker": "CASH",
                    "regime": "NO_SIGNAL",
                    "regime_confidence": 0.0,
                    "state": -1,
                    "kelly_fraction": 0.0,
                    "kelly_scale": 0.0,
                    "base_position": 0.0,
                    "position": 0.0,
                    "underlying_ret": 0.0,
                    "gross_trade_return": 0.0,
                    "friction_penalty": 0.0,
                    "net_trade_return": 0.0,
                    "var95": 0.0,
                    "cvar95": 0.0,
                    "var_breach": 0,
                    "cvar_breach": 0,
                }
            )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        raise RuntimeError("No trades were generated in blind backtest")

    executed = trades_df[trades_df["ticker"] != "CASH"].copy()

    # Aggregate to a daily portfolio return timeline.
    daily_ret = (
        executed.groupby("execution_day")["net_trade_return"].sum().sort_index()
        if not executed.empty
        else pd.Series(dtype=float)
    )

    if daily_ret.empty:
        # all-cash fallback
        daily_ret = pd.Series([0.0], index=[test_start], name="net_trade_return")

    equity = (1.0 + daily_ret).cumprod()

    # Benchmark over the same execution window (open-to-close to match strategy horizon).
    benchmark_df = _download_ohlcv(config.MARKET_BENCHMARK, start=test_start, end=full_end)
    if benchmark_df.empty:
        benchmark_cagr = 0.0
        benchmark_total_ret = 0.0
    else:
        benchmark_rets = (benchmark_df["Close"] / benchmark_df["Open"] - 1.0).dropna()
        if not daily_ret.empty:
            bench_slice = benchmark_rets.loc[
                (benchmark_rets.index >= pd.Timestamp(daily_ret.index.min()))
                & (benchmark_rets.index <= pd.Timestamp(daily_ret.index.max()))
            ]
        else:
            bench_slice = benchmark_rets.loc[benchmark_rets.index >= pd.Timestamp(test_start)]
        if bench_slice.empty:
            benchmark_cagr = 0.0
            benchmark_total_ret = 0.0
        else:
            bench_equity = (1.0 + bench_slice).cumprod()
            benchmark_cagr = _compute_cagr(bench_equity)
            benchmark_total_ret = float(bench_equity.iloc[-1] - 1.0)

    strategy_cagr = _compute_cagr(equity)
    strategy_total_ret = float(equity.iloc[-1] - 1.0)
    alpha_vs_benchmark = strategy_total_ret - benchmark_total_ret

    max_dd = _compute_max_drawdown(equity)

    winning = executed[executed["net_trade_return"] > 0]
    total_executed = len(executed)
    hit_rate = float(len(winning) / total_executed) if total_executed > 0 else 0.0
    profit_factor = _profit_factor(executed["net_trade_return"] if total_executed > 0 else pd.Series(dtype=float))

    var_breach_rate = (
        float(executed["var_breach"].mean()) if total_executed > 0 else 0.0
    )
    cvar_breach_rate = (
        float(executed["cvar_breach"].mean()) if total_executed > 0 else 0.0
    )

    summary = {
        "train_window": f"{train_start} -> {train_end}",
        "test_window": f"{test_start} -> {test_end}",
        "n_trades": int(total_executed),
        "strategy_total_return": round(strategy_total_ret, 4),
        "strategy_cagr": round(strategy_cagr, 4),
        "benchmark_total_return": round(benchmark_total_ret, 4),
        "benchmark_cagr": round(benchmark_cagr, 4),
        "alpha_vs_benchmark": round(alpha_vs_benchmark, 4),
        "max_drawdown": round(max_dd, 4),
        "hit_rate": round(hit_rate, 4),
        "profit_factor": round(float(profit_factor), 4) if np.isfinite(profit_factor) else "inf",
        "var_breach_rate": round(var_breach_rate, 4),
        "cvar_breach_rate": round(cvar_breach_rate, 4),
        "tx_cost": friction.transaction_cost,
        "slippage": friction.slippage,
    }

    stamp = date.today().isoformat()
    trades_path = os.path.join(output_dir, f"blind_backtest_trades_{stamp}.csv")
    daily_path = os.path.join(output_dir, f"blind_backtest_daily_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"blind_backtest_summary_{stamp}.txt")

    trades_df.to_csv(trades_path, index=False)
    pd.DataFrame(
        {
            "execution_day": daily_ret.index,
            "daily_return": daily_ret.values,
            "equity": equity.values,
        }
    ).to_csv(daily_path, index=False)

    lines = [
        "=" * 72,
        "MENTAT BLIND BACKTEST SUMMARY",
        "=" * 72,
        f"Train: {summary['train_window']}",
        f"Test:  {summary['test_window']}",
        "",
        f"Trades executed:      {summary['n_trades']}",
        f"Strategy total return:{summary['strategy_total_return']:.2%}",
        f"Benchmark return:     {summary['benchmark_total_return']:.2%}",
        f"Alpha vs benchmark:   {summary['alpha_vs_benchmark']:.2%}",
        f"Strategy CAGR:        {summary['strategy_cagr']:.2%}",
        f"Benchmark CAGR:       {summary['benchmark_cagr']:.2%}",
        "",
        f"Max Drawdown:         {summary['max_drawdown']:.2%}",
        f"Hit Rate:             {summary['hit_rate']:.2%}",
        f"Profit Factor:        {summary['profit_factor']}",
        f"VaR Breach Rate:      {summary['var_breach_rate']:.2%}",
        f"CVaR Breach Rate:     {summary['cvar_breach_rate']:.2%}",
        "",
        f"Transaction cost:     {summary['tx_cost']:.4%}",
        f"Slippage (per side):  {summary['slippage']:.4%}",
        "=" * 72,
        f"Trades CSV: {trades_path}",
        f"Daily CSV:  {daily_path}",
        f"Summary:    {summary_path}",
    ]

    report_text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)

    return {
        "summary": summary,
        "trades_path": trades_path,
        "daily_path": daily_path,
        "summary_path": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mentat blind walk-forward backtest")
    parser.add_argument("--tickers", default=",".join(config.TICKERS), help="Comma-separated ticker list")
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument("--tx-cost", type=float, default=0.0005, help="One-way transaction cost")
    parser.add_argument("--slippage", type=float, default=0.0010, help="Per-side slippage")
    parser.add_argument("--allow-short-crisis", action="store_true")
    parser.add_argument("--output-dir", default="analysis/validation")
    parser.add_argument("--max-days", type=int, default=None, help="Optional debug cap on number of test days")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    friction = FrictionConfig(
        transaction_cost=float(args.tx_cost),
        slippage=float(args.slippage),
    )

    run_blind_backtest(
        tickers=tickers,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        friction=friction,
        allow_short_crisis=bool(args.allow_short_crisis),
        output_dir=args.output_dir,
        max_days=args.max_days,
    )


if __name__ == "__main__":
    main()
