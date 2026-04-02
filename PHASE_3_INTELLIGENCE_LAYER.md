# Phase 3: OpenClaude Intelligence Layer

## Overview

Phase 3 adds a **news intelligence pipeline** that runs overnight, scraping market news and using Groq AI to identify stock-specific catalysts. The morning brief then combines regime data (Phase 1), portfolio risk (Phase 2) with sentiment insights to give you a complete decision picture.

**Schedule:**
- **10:30 PM**: OpenClaude Bot collects news and scores sentiment (Monday–Friday)
- **8:00 AM**: Morning brief delivered with market mood, top 5 outliers, catalysts

## Three Parallel News Streams

### Stream 1: MoneyControl Headlines
- 30 latest market/stock news articles
- Public business news, earnings releases, analyst notes

### Stream 2: Economic Times Markets
- 30 latest ET Markets headlines
- Earnings, M&A, regulatory news, commentary

### Stream 3: BSE Announcements
- Real-time corporate announcements via BSE public API
- Order wins, results, board decisions, insider activity
- **Highest signal source** — direct from exchange

## Groq API Sentiment Analysis

Each argument is sent to Groq's **mixtral-8x7b** model with a focused prompt:

```
"Does this news create a directional catalyst for any NSE-listed stock 
in the next 5 trading days? If yes: ticker, direction, confidence 1-10."
```

**Returns:**
```json
[
  {
    "ticker": "TCS.NS",
    "direction": "BULLISH",
    "confidence": 8,
    "reason": "Strong Q4 guidance on cloud exports",
    "headline_ref": "TCS Q4 cloud services"
  },
  ...
]
```

**Key:** Only **direct catalysts** are scored — earnings surprises, order wins, regulatory decisions, promoter activity. Vague macro news is ignored unless it directly impacts a sector.

## Morning Brief Synthesis

The `src/brief.py` combines three signals into one conviction score per stock:

```
Conviction = (regime_score × confidence) + (outlier_boost × 0.5) + sentiment_boost - sentiment_drag
```

### Regime Score
- LOW-VOL TRENDING: +3
- MEAN-REVERTING: +1
- HIGH-VOL RANGING: 0
- CRASH/CRISIS: -2
- UNCERTAIN: -1

### Outlier Boost
- +1 for each type of volatility outlier (max +2)

### Sentiment Boost
- Bullish signals add (confidence/10) to score
- Bearish signals subtract (confidence/10)

**Result:** Top 5 non-crisis stocks ranked by conviction, **each with its supporting sentiment reason**.

## Sample Morning Brief Output

```
==================================================
MENTAT MORNING BRIEF
Thursday, 02 April 2026
==================================================

Good morning, Master Vihaan.

-- MARKET INTELLIGENCE ---------------------
Overall mood: DEFENSIVE — majority in crisis regime.
Mentat assessment: Adverse conditions detected. Preservation takes priority.
Stocks scanned: 30 | In crisis: 18 | Trending: 2

-- TOP OUTLIERS TODAY ----------------------

  1. NESTLEIND.NS
     Regime: HIGH-VOL RANGING | Confidence: 100% | In regime: 19 days
     Sharpe: 0.23 | VaR 95%: -1.87%
     Conviction score: 4.2/10
     Outlier flags: vol_zscore_recent
     News [BULLISH 7/10]: Strong q1 margin expansion guidance

  2. TITAN.NS
     Regime: HIGH-VOL RANGING | Confidence: 99.7% | In regime: 8 days
     Sharpe: -0.45 | VaR 95%: -2.34%
     Conviction score: 3.1/10
     News [BULLISH 5/10]: Festive season jewelry demand picking up

  3. RELIANCE.NS
     Regime: HIGH-VOL RANGING | Confidence: 100% | In regime: 21 days
     Sharpe: -1.60 | VaR 95%: -2.61%
     Conviction score: 0.0/10

-- CRISIS / AVOID TODAY ---------------------
  TCS.NS: CRASH/CRISIS | 10d | VaR -4.08%
  INFY.NS: CRASH/CRISIS | 10d | VaR -4.20%
  HDFCBANK.NS: CRASH/CRISIS | 10d | VaR -3.44%

-- NEWS INTELLIGENCE ------------------------
  Articles processed: 76
  Bullish catalysts:  3
  Bearish catalysts:  2
  [BULLISH] NESTLEIND.NS: q1 margin expansion confirmed in guidance
  [BULLISH] TITAN.NS: festive demand accelerating ahead of q2
  [BEARISH] INFY.NS: offshore hiring slowdown cited by management
  [BEARISH] SBIN.NS: npl ratios rising in q1 results

==================================================
The calculations are complete. The decision is yours.
Mentat signs off -- Thursday, 02 April 2026
==================================================
```

## Files & Modules

### `src/intelligence.py`
**Functions:**
- `scrape_moneycontrol_headlines()` → list of {headline, url, source}
- `scrape_economic_times_headlines()` → list of {headline, url, source}
- `scrape_bse_announcements(tickers)` → list of {headline, company, scrip, source, category}
- `analyse_sentiment_batch(articles, known_tickers)` → list of Groq sentiment signals
- `run_intelligence_layer(known_tickers)` → {signals, by_ticker, n_articles, date}

### `src/brief.py`
**Functions:**
- `build_morning_brief(pipeline_results, sentiment_data)` → formatted text report

**Scores each stock by:**
1. Regime state (crisis -2 to trending +3)
2. Outlier count (max +2)
3. News sentiment (BULLISH/BEARISH ±n/10)

**Outputs:**
- Market mood assessment
- Top 5 non-crisis setups with conviction scores
- Crisis watchlist
- News catalyst summary (top 4 signals)

## Integration with Pipeline & Scheduler

### In `pipeline.py`
After building the daily report, the pipeline now:
1. Calls `run_intelligence_layer(tickers)` to fetch and score news
2. Builds `build_morning_brief(results, sentiment_data)`
3. Saves brief to `analysis/mentat_reports/mentat_brief_YYYY-MM-DD.txt`

### In `scheduler.py`
**New overnight job (22:30, Mon–Fri):**
```python
@scheduler.scheduled_job("cron", day_of_week="mon-fri", hour=22, minute=30)
def overnight_intelligence() -> None:
    """Phase 3: Overnight news collection and intelligence analysis."""
    sentiment_data = run_intelligence_layer(config.TICKERS)
    print(f"[INTEL] Overnight: {len(sentiment_data['signals'])} signals")
```

**Daily morning job (09:30, Mon–Fri):**
```
run_pipeline(retrain=False)  # includes sentiment + brief
```

## Groq API Setup

Set your Groq API key as an environment variable:
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_YOUR_KEY_HERE"

# Or add to .env / system environment variables
```

If `GROQ_API_KEY` is not set, the intelligence layer **gracefully skips sentiment analysis** and the morning brief still builds with regime + outlier scores.

## How to Use

### Run the full pipeline with morning brief
```bash
python pipeline.py --tickers RELIANCE.NS,TCS.NS,INFY.NS
```

### Run universe scan
```bash
python pipeline.py --scan
```

### Scheduler (automated)
```bash
python scheduler.py
```

**Execution timeline:**
- **22:30**: Overnight news collection
- **08:00**: Weekly retrain (Sundays)
- **09:30**: Daily morning briefing (Mon–Fri)
- **10:00**: Weekly universe scan (Sundays)

## What This Gives You

**Before Phase 3:** You saw regime + risk + outliers. Good, but incomplete.

**After Phase 3:** Every morning you get:
- Which regime each stock is in
- Which ones are flashing outlier signals
- **Which ones have positive/negative catalysts from yesterday's news**
- One conviction score (0-10) that tells you how much to trust each setup
- Action hints per stock (avoid, hold, size up, etc.)

**Example decision:**
- TCS is in CRASH/CRISIS (regime score -2)
- No outlier flags (boost 0)
- But MoneyControl reports "management notes strong services export pipeline" (BULLISH +0.6)
- Final conviction: -2 + 0.6 = **-1.4/10** → Still avoid despite the news because regime dominates

## Next Steps (Phase 3.1+)

1. **Real-time alerts**: Slack/email when a high-conviction (7+) catalyst hits
2. **Backtested sentiment**: Compare news sentiment against actual returns to weight its importance
3. **Sector-level sentiment**: Aggregate catalyst scores by sector for rotation signals
4. **Event calendar**: Earnings announcements, RBI decisions, dividend dates
5. **Insider tracking**: Track promoter buying/selling as a confidence boost

---

**Phase 3 Status: ✅ COMPLETE. OpenClaude Bot live. morning brief integrated. News intelligence working.**
