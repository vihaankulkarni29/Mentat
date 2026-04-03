"""
Mentat Live App — continuous dashboard with background scheduler.
Run: streamlit run live.py
"""

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

GROQ_KEY_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))

from pipeline import run_pipeline
from src import config
from src.intelligence import run_intelligence_layer
from src.universe import run_universe_scan, build_sector_regime_map, save_universe_scan

# Page config
st.set_page_config(
    page_title="Mentat Live — Stock Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dune-inspired styling
st.markdown("""
<style>
:root {
  --bg: #1f120a;
  --panel: #2b190d;
  --panel-soft: #3a2413;
  --sand: #d2b48c;
  --spice: #e07a2f;
  --mint: #a9d6b6;
  --alert: #d95d39;
}

.stApp {
  background: linear-gradient(180deg, #1b1008 0%, #241408 40%, #2d1808 100%);
  color: var(--sand);
}

.stMetric {
    background-color: var(--panel);
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid var(--spice);
}

.mentat-card {
  background: var(--panel);
  border: 1px solid #5a351b;
  border-radius: 12px;
  padding: 1.2rem;
  margin-bottom: 0.8rem;
}

.mentat-title {
  color: var(--spice);
  font-weight: 700;
  font-size: 1.1rem;
}

.mentat-sub {
  color: var(--sand);
  font-size: 0.9rem;
  font-weight: 500;
}

.status-live {
  color: var(--mint);
  font-weight: 600;
}

.status-alert {
  color: var(--alert);
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SCHEDULER INITIALIZATION (runs once, in background)
# ============================================================================

@st.cache_resource
def init_scheduler():
    """Initialize APScheduler to run in background (non-blocking)."""
    scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Kolkata"))
    
    @scheduler.scheduled_job("cron", day_of_week="mon-fri", hour=22, minute=30)
    def overnight_intelligence():
        print("[SCHEDULER] 22:30 — Running overnight intelligence collection...")
        try:
            sentiment_data = run_intelligence_layer(config.TICKERS)
            print(f"[SCHEDULER] Overnight intelligence complete: {len(sentiment_data.get('signals', []))} signals found")
        except Exception as e:
            print(f"[SCHEDULER] Overnight intelligence error: {e}")
    
    @scheduler.scheduled_job("cron", day_of_week="mon-fri", hour=9, minute=30)
    def daily_job():
        print("[SCHEDULER] 09:30 — Running daily Mentat pipeline with morning brief...")
        try:
            run_pipeline(retrain=False)
            print("[SCHEDULER] Daily pipeline complete")
        except Exception as e:
            print(f"[SCHEDULER] Daily pipeline error: {e}")
    
    @scheduler.scheduled_job("cron", day_of_week="sun", hour=8, minute=0)
    def weekly_retrain():
        print("[SCHEDULER] 08:00 (Sunday) — Retraining Mentat HMM models...")
        try:
            run_pipeline(retrain=True)
            print("[SCHEDULER] Weekly retrain complete")
        except Exception as e:
            print(f"[SCHEDULER] Weekly retrain error: {e}")
    
    @scheduler.scheduled_job("cron", day_of_week="sun", hour=10, minute=0)
    def weekly_universe_scan():
        print("[SCHEDULER] 10:00 (Sunday) — Running weekly NSE universe scan...")
        try:
            scan_df = run_universe_scan()
            sector_df = build_sector_regime_map(scan_df)
            save_universe_scan(scan_df, sector_df)
            print("[SCHEDULER] Universe scan complete")
        except Exception as e:
            print(f"[SCHEDULER] Universe scan error: {e}")
    
    scheduler.start()
    print("[INIT] Scheduler started successfully in background")
    return scheduler


def load_latest_brief():
    """Load the latest morning brief from file."""
    brief_dir = Path("analysis/mentat_reports")
    if not brief_dir.exists():
        return None
    
    brief_files = sorted(brief_dir.glob("mentat_brief_*.txt"))
    if not brief_files:
        return None
    
    latest = brief_files[-1]
    return _read_text_with_fallback(latest), latest.name


def load_latest_report():
    """Load the latest daily report from file."""
    report_dir = Path("analysis/reports")
    if not report_dir.exists():
        return None
    
    report_files = sorted(report_dir.glob("daily_report_*.txt"))
    if not report_files:
        return None
    
    latest = report_files[-1]
    return _read_text_with_fallback(latest), latest.name


def _read_text_with_fallback(file_path: Path) -> str:
    """Read text safely from files that may have mixed encodings."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # Last resort: keep app alive and show best-effort content.
    return file_path.read_text(encoding="utf-8", errors="replace")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize scheduler (runs once, cached)
    scheduler = init_scheduler()
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("## 🧠 Mentat Live — Intelligence Dashboard")
    with col2:
        st.metric("Status", "🟢 LIVE", delta="Scheduler Running", delta_color="off")
    with col3:
        st.metric("Time (IST)", datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%H:%M:%S"))
    
    st.divider()
    
    # ========================================================================
    # SIDEBAR: MANUAL CONTROLS
    # ========================================================================
    
    with st.sidebar:
        st.markdown("### ⚙️ Manual Controls")
        
        if st.button("🚀 Run Pipeline Now (Daily + Brief)", use_container_width=True):
            st.info("Running daily pipeline with morning brief...")
            try:
                run_pipeline(retrain=False)
                st.success("✅ Pipeline executed successfully")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Pipeline error: {e}")
        
        if st.button("🔄 Retrain Models Now (Weekly)", use_container_width=True):
            st.info("Retraining HMM models...")
            try:
                run_pipeline(retrain=True)
                st.success("✅ Retrain complete")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Retrain error: {e}")
        
        if st.button("🌍 Scan Universe Now (30 Tickers)", use_container_width=True):
            st.info("Scanning 30 NSE tickers...")
            try:
                scan_df = run_universe_scan()
                sector_df = build_sector_regime_map(scan_df)
                save_universe_scan(scan_df, sector_df)
                st.success("✅ Universe scan complete")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Universe scan error: {e}")
        
        if st.button("📰 Update Intelligence Now (News + Groq)", use_container_width=True):
            st.info("Collecting news and analyzing sentiment...")
            try:
                sentiment_data = run_intelligence_layer(config.TICKERS)
                st.success(f"✅ Intelligence updated: {len(sentiment_data.get('signals', []))} signals found")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Intelligence error: {e}")
        
        st.divider()
        
        st.markdown("### ⏰ Scheduled Jobs")
        st.markdown("""
        - **22:30 Mon-Fri** — Overnight Intelligence
        - **09:30 Mon-Fri** — Morning Brief + Daily Pipeline
        - **08:00 Sunday** — Weekly Retrain
        - **10:00 Sunday** — Universe Scan
        
        All times in IST (Asia/Kolkata)
        """)
        
        st.divider()
        st.markdown("### 🔑 Configuration")
        if GROQ_KEY_AVAILABLE:
            st.write("**Groq API Key**: YES (from environment)")
        else:
            st.write("**Groq API Key**: NO (set GROQ_API_KEY to enable sentiment)")
        st.write(f"**Tickers**: {len(config.TICKERS)} tracked")
    
    # ========================================================================
    # MAIN CONTENT: MORNING BRIEF
    # ========================================================================
    
    tab_brief, tab_report, tab_refresh = st.tabs(["📋 Morning Brief", "📊 Daily Report", "🔄 Status"])
    
    with tab_brief:
        st.markdown("### Morning Brief")
        brief_data = load_latest_brief()
        if brief_data:
            brief_text, brief_file = brief_data
            st.markdown(f"**{brief_file}**")
            st.text(brief_text)
        else:
            st.info("No morning brief available yet. Click 'Run Pipeline Now' to generate one.")
    
    with tab_report:
        st.markdown("### Daily Report")
        report_data = load_latest_report()
        if report_data:
            report_text, report_file = report_data
            st.markdown(f"**{report_file}**")
            st.text(report_text)
        else:
            st.info("No daily report available yet. Click 'Run Pipeline Now' to generate one.")
    
    with tab_refresh:
        st.markdown("### Live Status")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Brief Age", "Check sidebar for latest")
        with col_b:
            st.metric("Report Age", "Check sidebar for latest")
        with col_c:
            st.metric("Next Job", "See sidebar schedule")
        
        st.info("""
        **Scheduler Running in Background**
        
        The scheduler is continuously running. Scheduled jobs will execute at their designated times:
        - Overnight: 22:30 (news collection)
        - Morning: 09:30 (daily brief)
        - Weekly: Sunday 8:00 (retrain) + 10:00 (universe)
        
        Click manual controls to run jobs on-demand.
        """)


if __name__ == "__main__":
    main()
