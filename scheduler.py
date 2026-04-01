"""Mentat scheduler for daily inference and weekly retraining."""

from apscheduler.schedulers.blocking import BlockingScheduler
import pytz

from pipeline import run_pipeline

scheduler = BlockingScheduler(timezone=pytz.timezone("Asia/Kolkata"))


@scheduler.scheduled_job("cron", day_of_week="mon-fri", hour=9, minute=30)
def daily_job() -> None:
    print("Running daily Mentat pipeline...")
    run_pipeline(retrain=False)


@scheduler.scheduled_job("cron", day_of_week="sun", hour=8, minute=0)
def weekly_retrain() -> None:
    print("Retraining Mentat HMM models...")
    run_pipeline(retrain=True)


if __name__ == "__main__":
    scheduler.start()
