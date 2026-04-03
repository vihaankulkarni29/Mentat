"""Mentat Stage 1 radar scanner for Nifty 500 cross-sectional momentum.

This script finds the strongest stocks in the Nifty 500 universe using simple
momentum math:
- load Nifty 500 constituents automatically
- pull the last 6 months of data with yfinance
- compute 90-trading-day ROC
- require price above the 200-day moving average when enough history exists
- print the top 10 gems

Usage:
    python scanner.py
    python scanner.py --top 10 --save-chart
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf  # type: ignore[import-not-found]

from src import config


NIFTY_500_SOURCE = "https://en.wikipedia.org/wiki/NIFTY_500"
NIFTY_500_API = "https://www.nseindia.com/api/equity-stockIndices"
NSE_HOME = "https://www.nseindia.com/"
OUTPUT_DIR = Path("analysis/radar")


@dataclass(frozen=True)
class RadarRow:
    symbol: str
    company: str
    industry: str
    latest_close: float
    roc_90: float
    ma_200: float | None
    above_200dma: bool
    bars: int


def load_nifty500_symbols() -> list[dict[str, str]]:
    """Load Nifty 500 constituents from NSE, with Wikipedia as fallback.

    Returns a list of dictionaries containing symbol, company, and industry.
    """
    try:
        return _load_nifty500_from_nse()
    except Exception as exc:
        try:
            response = httpx.get(
                NIFTY_500_API,
                params={
                    "action": "parse",
                    "page": "NIFTY_500",
                    "prop": "wikitext",
                    "format": "json",
                },
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            wikitext = response.json()["parse"]["wikitext"]["*"]
            return _parse_nifty500_wikitext(wikitext)
        except Exception as fallback_exc:
            raise RuntimeError(f"Could not load Nifty 500 constituents: {exc}; fallback failed: {fallback_exc}") from fallback_exc


def _load_nifty500_from_nse() -> list[dict[str, str]]:
    """Load Nifty 500 constituents from the NSE index API."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/market-data/live-equity-market",
        "X-Requested-With": "XMLHttpRequest",
    }

    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client:
        client.get(NSE_HOME)
        response = client.get(NIFTY_500_API, params={"index": "NIFTY 500"})
        response.raise_for_status()
        payload = response.json()

    data = payload.get("data", [])
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for entry in data:
        symbol = str(entry.get("symbol", "")).strip()
        company = str(entry.get("meta", {}).get("companyName", entry.get("companyName", ""))).strip()
        industry = str(entry.get("meta", {}).get("industry", entry.get("industry", ""))).strip()
        series = entry.get("series", [])

        if not symbol or symbol in seen:
            continue
        if isinstance(series, list) and "EQ" not in series and symbol != "NIFTY 500":
            continue
        if symbol == "NIFTY 500":
            continue

        seen.add(symbol)
        rows.append({"symbol": symbol, "company": company, "industry": industry})

    rows.sort(key=lambda item: item["symbol"])
    if len(rows) < 400:
        raise RuntimeError(f"Parsed too few Nifty 500 symbols from NSE: {len(rows)}")

    return rows


def _clean_wiki_text(text: str) -> str:
    """Remove common wikitext markup from a cell."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"''+", "", text)
    text = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", text)
    return text.strip()


def _parse_nifty500_wikitext(wikitext: str) -> list[dict[str, str]]:
    """Parse the Nifty 500 constituents table from MediaWiki wikitext."""
    marker = "{| class=\"wikitable sortable mw-collapsible\""
    start = wikitext.find(marker)
    if start == -1:
        raise RuntimeError("Could not locate Nifty 500 constituents table in wikitext")

    tail = wikitext[start:]
    end_marker = "\n== Other Notable Indices =="
    end = tail.find(end_marker)
    if end == -1:
        end = tail.find("\n== Index Methodology ==")
    if end == -1:
        end = len(tail)

    table = tail[:end]
    rows: list[dict[str, str]] = []
    current: list[str] = []

    for raw_line in table.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("{|") or line.startswith("|+") or line.startswith("!"):
            continue
        if line == "|-":
            if len(current) >= 6:
                rows.append(
                    {
                        "symbol": _clean_wiki_text(current[3]),
                        "company": _clean_wiki_text(current[1]),
                        "industry": _clean_wiki_text(current[2]),
                    }
                )
            current = []
            continue
        if line == "|}":
            break
        if line.startswith("|"):
            current.append(line[1:].strip())

    if len(current) >= 6:
        rows.append(
            {
                "symbol": _clean_wiki_text(current[3]),
                "company": _clean_wiki_text(current[1]),
                "industry": _clean_wiki_text(current[2]),
            }
        )

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        symbol = row["symbol"].strip()
        if not symbol or symbol.lower() == "nan" or symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(row)

    if len(deduped) < 400:
        raise RuntimeError(f"Parsed too few Nifty 500 symbols: {len(deduped)}")

    return deduped


def download_history(symbol: str) -> pd.DataFrame:
    """Download 6 months of adjusted OHLCV history."""
    ticker = f"{symbol}.NS"
    df = yf.download(
        ticker,
        period="6mo",
        auto_adjust=True,
        progress=False,
        timeout=20,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna().copy()


def score_symbol(meta: dict[str, str]) -> RadarRow | None:
    """Compute ROC and 200DMA filter for a single symbol."""
    symbol = meta["symbol"]
    df = download_history(symbol)
    if df.empty or "Close" not in df.columns:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 30:
        return None

    latest_close = float(close.iloc[-1])
    roc_90 = float(close.pct_change(90).iloc[-1]) if len(close) > 90 else float(close.iloc[-1] / close.iloc[0] - 1.0)

    ma_200 = None
    above_200dma = False
    if len(close) >= 200:
        ma_200 = float(close.rolling(200).mean().iloc[-1])
        above_200dma = latest_close > ma_200 if pd.notna(ma_200) else False

    return RadarRow(
        symbol=symbol,
        company=meta["company"],
        industry=meta["industry"],
        latest_close=latest_close,
        roc_90=roc_90,
        ma_200=ma_200,
        above_200dma=above_200dma,
        bars=len(close),
    )


def run_radar(max_workers: int = 10) -> pd.DataFrame:
    """Scan Nifty 500 and rank by 90-day ROC."""
    universe = load_nifty500_symbols()
    print(f"[RADAR] Loaded {len(universe)} Nifty 500 constituents")

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(score_symbol, meta): meta for meta in universe}
        for future in as_completed(futures):
            row = future.result()
            if row is None:
                continue
            records.append(
                {
                    "symbol": row.symbol,
                    "company": row.company,
                    "industry": row.industry,
                    "latest_close": row.latest_close,
                    "roc_90": row.roc_90,
                    "ma_200": row.ma_200,
                    "above_200dma": row.above_200dma,
                    "bars": row.bars,
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["roc_90", "above_200dma", "bars"], ascending=[False, False, False]).reset_index(drop=True)
    return df


def save_top10_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Save a simple horizontal bar chart for the top 10 radar names."""
    top = df.head(10).copy()
    if top.empty:
        return

    top = top.sort_values("roc_90", ascending=True)
    colors = ["#2e8b57" if flag else "#d97706" for flag in top["above_200dma"].tolist()]

    plt.figure(figsize=(11, 7))
    bars = plt.barh(top["symbol"], top["roc_90"] * 100, color=colors)
    plt.axvline(0, color="white", linewidth=0.8)
    plt.title("Mentat Stage 1 Radar: Top 10 Nifty 500 Gems by 90D ROC")
    plt.xlabel("90-day ROC (%)")
    plt.ylabel("Ticker")
    plt.grid(axis="x", alpha=0.2)

    for bar, roc, above in zip(bars, top["roc_90"], top["above_200dma"]):
        label = f"{roc * 100:.1f}% | {'>200DMA' if above else '<200DMA'}"
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {label}", va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mentat Stage 1 Nifty 500 radar scanner")
    parser.add_argument("--top", type=int, default=10, help="Number of top gems to print")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent download workers")
    parser.add_argument("--save-chart", action="store_true", help="Save a top-10 ROC bar chart")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_radar(max_workers=args.workers)
    if df.empty:
        raise RuntimeError("Radar scan returned no results")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().date().isoformat()
    csv_path = OUTPUT_DIR / f"nifty500_radar_{stamp}.csv"
    df.to_csv(csv_path, index=False)

    top = df.head(args.top).copy()
    print("=" * 80)
    print("MENTAT STAGE 1 - NIFTY 500 RADAR")
    print("=" * 80)
    print(f"Universe scanned: {len(df)} symbols")
    print(f"Output CSV: {csv_path}")
    print("")
    print(top[["symbol", "company", "industry", "roc_90", "above_200dma"]].to_string(index=False, formatters={"roc_90": lambda x: f"{x:.2%}"}))

    if args.save_chart:
        chart_path = OUTPUT_DIR / f"nifty500_radar_top10_{stamp}.png"
        save_top10_chart(df, chart_path)
        print(f"\nChart saved: {chart_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()