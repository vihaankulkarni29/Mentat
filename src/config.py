"""Mentat Phase 1 configuration."""

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
LOOKBACK_YEARS = 3
ROLLING_WINDOW = 60
N_STATES = 4
VAR_CONFIDENCE = 0.95
VOL_OUTLIER_Z = 2.5
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.55
HISTORY_DAYS = 5

# Observation features used by HMM (Layer 2 -> Layer 3)
OBSERVATION_COLS = ["log_ret_1d", "rvol_10d", "rsi", "vol_zscore"]

# Directories
MODEL_DIR = "./models"
OUTPUT_DIR = "./analysis/pipeline"
REPORT_DIR = "./analysis/mentat_reports"

# Market proxy for beta
MARKET_BENCHMARK = "^NSEI"
VIX_TICKER = "^VIX"

# Email settings (optional)
SEND_EMAIL = False
REPORT_EMAIL = ""
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_PASSWORD_ENV = "MENTAT_SMTP_APP_PASSWORD"
