"""Parameter sweep for Mentat seed backtest.

Grid:
- Hurst min: 0.50, 0.52, 0.54
- Retrain days: 5, 20
- RSI: fixed at 40-65

Outputs:
- analysis/validation/seed_param_sweep_YYYY-MM-DD.csv
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKTEST = ROOT / "tests" / "seed_backtest.py"
OUT_BASE = ROOT / "analysis" / "validation" / "sweeps"

HURST_VALUES = [0.50, 0.52, 0.54]
RETRAIN_VALUES = [5, 20]
RSI_MIN = 40
RSI_MAX = 65


def parse_summary(summary_path: Path) -> dict[str, float]:
    text = summary_path.read_text(encoding="utf-8", errors="replace")

    def extract(pattern: str, default: float = 0.0) -> float:
        m = re.search(pattern, text)
        if not m:
            return default
        return float(m.group(1))

    trades = extract(r"Trades:\s+(\d+)")
    strat = extract(r"Strategy return:\s+(-?\d+\.\d+)%")
    bench = extract(r"Benchmark \([^)]*\):\s+(-?\d+\.\d+)%")
    alpha = extract(r"Alpha vs benchmark:\s+(-?\d+\.\d+)%")
    sharpe = extract(r"Sharpe ratio:\s+(-?\d+\.\d+)")
    mdd = extract(r"Max drawdown:\s+(-?\d+\.\d+)%")
    win = extract(r"Win rate:\s+(-?\d+\.\d+)%")

    return {
        "trades": trades,
        "strategy_return_pct": strat,
        "benchmark_return_pct": bench,
        "alpha_pct": alpha,
        "sharpe": sharpe,
        "max_drawdown_pct": mdd,
        "win_rate_pct": win,
    }


def run_one(hurst_min: float, retrain_days: int) -> dict[str, float | int | str]:
    tag = f"h{hurst_min:.2f}_r{retrain_days}"
    out_dir = OUT_BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(BACKTEST),
        "--retrain-days",
        str(retrain_days),
        "--rsi-min",
        str(RSI_MIN),
        "--rsi-max",
        str(RSI_MAX),
        "--hurst-min",
        str(hurst_min),
        "--output-dir",
        str(out_dir),
    ]

    print(f"[SWEEP] Running {tag}")
    completed = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    stamp = date.today().isoformat()
    summary_path = out_dir / f"seed_backtest_summary_{stamp}.txt"

    row: dict[str, float | int | str] = {
        "hurst_min": hurst_min,
        "retrain_days": retrain_days,
        "rsi_min": RSI_MIN,
        "rsi_max": RSI_MAX,
        "status": "ok" if completed.returncode == 0 and summary_path.exists() else "failed",
        "summary_path": str(summary_path),
    }

    if row["status"] == "ok":
        row.update(parse_summary(summary_path))
    else:
        row.update(
            {
                "trades": 0,
                "strategy_return_pct": 0.0,
                "benchmark_return_pct": 0.0,
                "alpha_pct": 0.0,
                "sharpe": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
            }
        )

    return row


def main() -> None:
    rows: list[dict[str, float | int | str]] = []

    for retrain_days in RETRAIN_VALUES:
        for hurst_min in HURST_VALUES:
            rows.append(run_one(hurst_min=hurst_min, retrain_days=retrain_days))

    df = pd.DataFrame(rows)
    df = df.sort_values(["sharpe", "win_rate_pct", "alpha_pct"], ascending=[False, False, False])

    out_file = ROOT / "analysis" / "validation" / f"seed_param_sweep_{date.today().isoformat()}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print("\n=== SWEEP RESULTS (sorted) ===")
    show_cols = [
        "hurst_min",
        "retrain_days",
        "trades",
        "win_rate_pct",
        "max_drawdown_pct",
        "sharpe",
        "alpha_pct",
        "status",
    ]
    print(df[show_cols].to_string(index=False))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
