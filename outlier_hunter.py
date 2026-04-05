"""Institutional VCP breakout screener for NSE symbols.

Scans a provided ticker universe and identifies names passing all gates:
1) Trend structure (price > SMA50 > SMA200)
2) Tight historical Bollinger Band Width coil
3) Volume shock (>3x 20-day average volume)
4) Breakout candle (close > open and close > upper Bollinger Band)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf  # type: ignore[import-not-found]
from tqdm import tqdm  # type: ignore[import-not-found]


DEFAULT_OUTPUT = Path("vcp_targets.csv")

# Replace this with your full Nifty Smallcap 250 universe when available.
DEFAULT_TICKERS = [
    "IEX",
    "CDSL",
    "BSE",
    "SONATSOFTW",
    "CYIENT",
    "KAYNES",
    "IRFC",
    "RVNL",
    "RAILTEL",
    "CLEAN",
    "DEEPAKNTR",
    "PIIND",
    "COFORGE",
    "PERSISTENT",
    "DIXON",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE VCP outlier screener")
    parser.add_argument("--csv", default="", help="Optional CSV path containing ticker symbols")
    parser.add_argument(
        "--symbol-col",
        default="",
        help="Optional CSV column name for symbols (default: first column)",
    )
    parser.add_argument(
        "--scan-window-days",
        type=int,
        default=150,
        help="Recent daily bars used for VCP logic (default: 150)",
    )
    parser.add_argument(
        "--download-days",
        type=int,
        default=320,
        help="Download horizon; needs >200 for SMA200 stability (default: 320)",
    )
    parser.add_argument(
        "--as-of-date",
        default="",
        help="Optional YYYY-MM-DD date to simulate prior session as 'today'",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV for passing tickers")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if not symbol:
        return ""
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def load_tickers(csv_path: str, symbol_col: str) -> list[str]:
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Ticker CSV not found: {path}")

        raw = pd.read_csv(path)
        if raw.empty:
            return []

        if symbol_col:
            if symbol_col not in raw.columns:
                raise ValueError(f"Column '{symbol_col}' not found in {path}")
            series = raw[symbol_col]
        else:
            series = raw.iloc[:, 0]

        tickers = [normalize_symbol(v) for v in series.dropna().astype(str).tolist()]
    else:
        tickers = [normalize_symbol(v) for v in DEFAULT_TICKERS]

    # Stable unique order
    deduped: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        if not t or t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped


def download_history(symbol: str, download_days: int) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=f"{download_days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("%s download failed: %s", symbol, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        logging.warning("%s returned no data", symbol)
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(df.columns):
        logging.warning("%s missing required OHLCV columns", symbol)
        return pd.DataFrame()

    return df.dropna().copy()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    px = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce")

    feat = df.copy()
    feat["sma_20"] = px.rolling(20, min_periods=20).mean()
    feat["sma_50"] = px.rolling(50, min_periods=50).mean()
    feat["sma_200"] = px.rolling(200, min_periods=200).mean()

    std_20 = px.rolling(20, min_periods=20).std()
    feat["bb_upper"] = feat["sma_20"] + 2 * std_20
    feat["bb_lower"] = feat["sma_20"] - 2 * std_20
    feat["bbw"] = (feat["bb_upper"] - feat["bb_lower"]) / feat["sma_20"]

    feat["vol_avg_20"] = vol.rolling(20, min_periods=20).mean()
    feat["pct_change"] = px.pct_change()
    return feat


def evaluate_vcp(symbol: str, feat: pd.DataFrame, scan_window_days: int) -> dict | None:
    if len(feat) < max(200, scan_window_days):
        logging.info("%s skipped: insufficient bars (%d)", symbol, len(feat))
        return None

    window = feat.tail(scan_window_days).copy()
    latest = window.iloc[-1]

    bbw_120 = window["bbw"].tail(120).dropna()
    if bbw_120.empty:
        return None

    cond1 = (
        latest["Close"] > latest["sma_50"]
        and latest["sma_50"] > latest["sma_200"]
    )

    cond2 = latest["bbw"] < (bbw_120.min() * 1.2)

    vol_mult = float(latest["Volume"] / latest["vol_avg_20"]) if latest["vol_avg_20"] > 0 else 0.0
    cond3 = vol_mult > 3.0

    cond4 = latest["Close"] > latest["Open"] and latest["Close"] > latest["bb_upper"]

    if not (cond1 and cond2 and cond3 and cond4):
        return None

    return {
        "ticker": symbol,
        "date": str(window.index[-1].date()),
        "close": round(float(latest["Close"]), 2),
        "pct_change": round(float(latest["pct_change"] * 100), 2),
        "volume_multiplier": round(vol_mult, 2),
        "bbw": round(float(latest["bbw"]), 4),
    }


def render_summary(rows: Iterable[dict]) -> None:
    rows = list(rows)
    print("\n" + "=" * 78)
    print("NSE VCP OUTLIER HUNTER")
    print("=" * 78)
    if not rows:
        print("No tickers passed all VCP breakout gates today.")
        return

    out = pd.DataFrame(rows).sort_values("volume_multiplier", ascending=False).reset_index(drop=True)
    print(out[["ticker", "close", "pct_change", "volume_multiplier", "bbw"]].to_string(index=False))


def main() -> None:
    configure_logging()
    args = parse_args()

    tickers = load_tickers(args.csv, args.symbol_col)
    if not tickers:
        raise RuntimeError("No tickers loaded for screening")

    as_of_ts = pd.Timestamp(args.as_of_date) if args.as_of_date else None
    if as_of_ts is not None:
        logging.info("Time-machine mode active: evaluating as of %s", as_of_ts.date())

    logging.info("Scanning %d tickers", len(tickers))

    passed: list[dict] = []
    failures = 0

    for symbol in tqdm(tickers, desc="Scanning NSE universe", unit="ticker"):
        raw = download_history(symbol, args.download_days)
        if raw.empty:
            failures += 1
            continue

        feat = compute_features(raw)
        if as_of_ts is not None:
            # Simulate yesterday by removing rows after the selected date.
            feat = feat.loc[feat.index <= as_of_ts].copy()
            if feat.empty:
                continue

        row = evaluate_vcp(symbol, feat, args.scan_window_days)
        if row is not None:
            passed.append(row)

    render_summary(passed)

    out_path = Path(args.output)
    pd.DataFrame(passed).to_csv(out_path, index=False)

    print("\n" + "-" * 78)
    print(f"Passed tickers: {len(passed)}")
    print(f"Failed downloads: {failures}")
    print(f"Saved: {out_path}")
    print("-" * 78)


if __name__ == "__main__":
    main()
