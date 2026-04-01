# Mentat (Phase 1)

Mentat is a Hidden Markov Model (HMM) regime engine for daily stock decision support.

Core idea:

- Market regime is hidden.
- We observe symptoms: returns, volatility, RSI, volume anomaly.
- HMM infers the most likely current regime and transition risk.

## Layered Architecture

1. Data ingestion: OHLCV + VIX + market benchmark
2. Feature engineering: log returns, realized volatility, RSI, volume z-score
3. HMM core: train/load Gaussian HMM, decode current regime with Viterbi
4. Risk and outliers: regime VaR/CVaR/Sharpe, beta, z-score alerts
5. Reporting: beginner-friendly daily report + optional email

## Project Structure

```
Stock Market Analysis/
├── pipeline.py
├── scheduler.py
├── requirements.txt
├── README.md
├── models/
├── analysis/
│   ├── pipeline/
│   └── mentat_reports/
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_ingestion.py
    ├── hmm_engine.py
    ├── risk_engine.py
    └── report.py
```

## Install

```bash
pip install -r requirements.txt
```

## Configure

Edit [src/config.py](src/config.py):

- `TICKERS`
- `LOOKBACK_YEARS`
- `ROLLING_WINDOW`
- `N_STATES`
- `VOL_OUTLIER_Z`
- `SEND_EMAIL` and `REPORT_EMAIL` (optional)

## Run Mentat

Daily inference (load existing models):

```bash
python pipeline.py
```

Force retraining:

```bash
python pipeline.py --retrain
```

## Run Streamlit App

```bash
streamlit run app.py
```

The app provides a beginner-first dashboard with:

1. Regime label and confidence
2. Risk metrics in card format
3. Outlier alerts
4. Last 5 sessions regime history
5. Mentat guidance statements

## Schedule Automation

Run scheduler (Mon-Fri daily run + Sunday retrain):

```bash
python scheduler.py
```

## Output Files

- Feature matrices: `analysis/pipeline/<TICKER>_feature_matrix.csv`
- Persisted models: `models/<TICKER>_hmm.pkl`
- Daily report text: `analysis/mentat_reports/mentat_report_YYYY-MM-DD.txt`

## Beginner Decision View

Each ticker report gives:

1. Regime label
2. Confidence split across states
3. Risk snapshot (VaR, CVaR, Sharpe, Beta)
4. Outlier flags (returns, volume, realized vol)
5. Action hint in plain language

Use it as a decision support dashboard, not as an auto-trading instruction.

## Model Notes

- HMM type: Gaussian HMM with full covariance
- Training: Baum-Welch (EM)
- Decoding: Viterbi over rolling window
- Suggested cadence: decode daily, retrain weekly

## Phase 1.1 Calibration

Phase 1.1 adds three practical upgrades:

1. Multi-metric state labeling using returns + realized vol + RSI context
2. Uncertainty gate: outputs `UNCERTAIN / TRANSITION` when posterior confidence is low
3. Rolling regime tape: last 5 sessions shown in reports and app

## Disclaimer

For educational/research use only. Not financial advice.
