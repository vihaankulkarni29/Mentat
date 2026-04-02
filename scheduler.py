"""Mentat scheduler for daily inference, weekly retraining, and universe scanning."""

from apscheduler.schedulers.blocking import BlockingScheduler
import pytz

from pipeline import run_pipeline

scheduler = BlockingScheduler(timezone=pytz.timezone("Asia/Kolkata"))


@scheduler.scheduled_job("cron", day_of_week="mon-fri", hour=22, minute=30)
def overnight_intelligence() -> None:
    """Phase 3: Overnight news collection and intelligence analysis."""
    print("Running overnight intelligence collection...")
    from src import config
    from src.intelligence import run_intelligence_layer
    sentiment_data = run_intelligence_layer(config.TICKERS)
    print(f"[INTEL] Overnight collection complete: {len(sentiment_data['signals'])} signals")


@scheduler.scheduled_job("cron", day_of_week="mon-fri", hour=9, minute=30)
def daily_job() -> None:
    print("Running daily Mentat pipeline with morning brief...")
    run_pipeline(retrain=False)


@scheduler.scheduled_job("cron", day_of_week="sun", hour=8, minute=0)
def weekly_retrain() -> None:
    print("Retraining Mentat HMM models...")
    run_pipeline(retrain=True)


@scheduler.scheduled_job("cron", day_of_week="sun", hour=10, minute=0)
def weekly_universe_scan() -> None:
    """Phase 2.1: Weekly NSE universe regime scan."""
    print("Running weekly NSE universe scan...")
    from src.universe import run_universe_scan, build_sector_regime_map, save_universe_scan
    scan_df   = run_universe_scan()
    sector_df = build_sector_regime_map(scan_df)
    save_universe_scan(scan_df, sector_df)
    print("[OK] Universe scan complete")


if __name__ == "__main__":
    scheduler.start()
