"""Mentat Phase 1 configuration."""

# Stage 2 default universe for day-to-day runs.
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# Stage 1/Backtest seed universe focused on information-asymmetry pockets.
# Intentional bias: defense, railways, renewables, specialty chemicals,
# healthcare, electronics manufacturing, and mid-cap IT.
MENTAT_SEED_UNIVERSE = [
	# Defense / Aerospace
	"HAL.NS", "BEL.NS", "BDL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "DATAPATTNS.NS",
	# Railways / Rail Infra
	"IRFC.NS", "RVNL.NS", "IRCON.NS", "RAILTEL.NS", "TITAGARH.NS", "BEML.NS",
	# Renewables / Power Equipment
	"SUZLON.NS", "INOXWIND.NS", "KPIGREEN.NS", "WAAREEENER.NS", "NTPCGREEN.NS", "PREMIERENE.NS",
	# Specialty Chemicals
	"DEEPAKNTR.NS", "NAVINFLUOR.NS", "SRF.NS", "PIIND.NS", "AARTIIND.NS", "CLEAN.NS",
	# Healthcare / Pharma / Diagnostics
	"LALPATHLAB.NS", "METROPOLIS.NS", "DRREDDY.NS", "TORNTPHARM.NS", "LAURUSLABS.NS", "KIMS.NS",
	# Consumer electronics / EMS
	"DIXON.NS", "KAYNES.NS", "SYRMA.NS", "AMBER.NS", "PGEL.NS", "AVALON.NS",
	# Mid-cap IT / Digital engineering
	"PERSISTENT.NS", "COFORGE.NS", "LTIM.NS", "MPHASIS.NS", "CYIENT.NS", "SONATSOFTW.NS",
]

# Explicitly excluded heavyweight/saturated names for this seed framework.
MENTAT_EXCLUDED = [
	# Large-cap IT
	"TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
	# Telecom and oil & gas majors
	"BHARTIARTL.NS", "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS",
	# PSU banks (saturated/consensus heavy)
	"SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS",
]

# Small/mid benchmark proxies for seed-universe backtests (fallback to NIFTY50 if unavailable).
MENTAT_BENCHMARKS = ["^NSEMDCP50", "^CNXSMCAP", "^NSEI"]
LOOKBACK_YEARS = 3
ROLLING_WINDOW = 60
N_STATES = 3
VAR_CONFIDENCE = 0.95
VOL_OUTLIER_Z = 2.5
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.55
HISTORY_DAYS = 5

# Observation features used by HMM (Layer 2 -> Layer 3)
OBSERVATION_COLS = ["log_ret_1d", "rvol_10d", "rsi", "vol_zscore", "vix"]

# Phase 1.2 validation settings
WF_TRAIN_SIZE = 504
WF_TEST_SIZE = 21
BIC_STATE_MIN = 2
BIC_STATE_MAX = 6
VALIDATION_DIR = "./analysis/validation"

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
