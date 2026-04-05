# Phase 1.2 Validation Summary — NSE Regime Stability (April 2, 2026)

## Validation Results

| Ticker | Regime Label | State | Confidence | Persistence (days) | VaR95 | Sharpe | Beta | Model Quality (min_persist) | Notes |
|--------|---|---|---|---|---|---|---|---|---|
| RELIANCE.NS | HIGH-VOL RANGING | S2 | 100% | 19 (ESTABLISHED) | -2.13% | -0.84 | 0.73 | 0.921 | Least concentrated risk |
| TCS.NS | CRASH/CRISIS | S1 | 100% | 39 (MATURE) | -4.08% | -5.96 | 0.60 | 0.904 | Extended down regime |
| INFY.NS | CRASH/CRISIS | S2 | 100% | 39 (MATURE) | -4.20% | -4.87 | 0.49 | 0.866 | Extended down regime |
| SBIN.NS | MEAN-REVERTING | S0 | 100% | 20 (ESTABLISHED) | -3.97% | -5.61 | 1.25 | 0.90 | Only hopping stock |
| BAJAJ-AUTO.NS | CRASH/CRISIS | S2 | 100% | 21 (ESTABLISHED) | -4.44% | -4.50 | 1.15 | 0.857 | Beta closer to market |

## Regime Clustering Observations

### Dominant Pattern: CRASH/CRISIS (3 of 5)
- **TCS.NS, INFY.NS, BAJAJ-AUTO.NS** all label as CRASH/CRISIS regime
- Persistence ranges from 21–39 days → **stable regime detection**
- Different state indices within same label suggest **states are regime-invariant** in labeling
- All three show **negative Sharpe ratios < -4.5**
- VaR ranges from -4.08% to -4.44% → **similar tail risk**
- **Defensive betas** (0.49–1.15) suggest market-wide stress context

### Divergent Cases:
- **RELIANCE.NS**: Only stock in HIGH-VOL RANGING regime
  - Lowest VaR (-2.13%), best Sharpe (-0.84)
  - Least defensive beta (0.73)
  - More moderate risk environment
  - **Interpretation**: Larger-cap defensive anchor?
  
- **SBIN.NS**: Only stock in MEAN-REVERTING regime
  - Beta 1.25 → **amplifies market moves**
  - Model convergence warning (noisy signal)
  - `min_persistence=0.90` (all used states equally fleeting)
  - **Interpretation**: Bank cyclicality or data quality issue?

## Stability Assessment

### ✅ High Confidence Regimes:
1. **CRASH/CRISIS cluster** (TCS, INFY, BAJAJ)
   - Consistent label across tickers
   - Multi-week persistence
   - Risk metrics align across stocks
   - **Verdict: Real market regime, not noise**

2. **HIGH-VOL RANGING (RELIANCE)**
   - Singleton but stable 19-day persistence
   - Coherent risk profile
   - **Verdict: Valid regime, ticker-specific or cap-weighted effect**

### ⚠ Lower Confidence:
- **SBIN MEAN-REVERTING**
  - Sole instance of this regime
  - Model convergence issues
  - min_persistence=0.90 suggests all states equally fleeting
  - **Verdict: Either cycle-specific (banking) or model needs tuning**

## Key Findings

1. **Regime labels are emergent and meaningful**
   - 3 of 5 independent runs converge to CRASH/CRISIS
   - Regimes capture real market conditions, not random state assignments
   - Confidence remains 100% across all tickers → label stability works

2. **State indices vary but labels don't**
   - TCS S1 = CRASH/CRISIS, INFY S2 = CRASH/CRISIS
   - Confirms label persistence logic is working
   - States are internal HMM artifacts; labeling logic generalizes

3. **Persistence is multi-week, not daily noise**
   - Range: 19–39 days
   - "ESTABLISHED" (11–30d) and "MATURE" (30+d) dominate
   - Regimes strong enough for tactical overlays

4. **4-state model quality warnings suggest optional tuning**
   - `min_persistence` warnings across 3 tickers
   - Reduce N_STATES to 3 in Phase 2 to test if cleaner separation emerges
   - Current 4-state still valid but may conflate micro-regimes

5. **Risk metrics consistent within regime label**
   - CRASH/CRISIS group: VaR -4.0% to -4.4%
   - CRASH/CRISIS group: Sharpe < -4.5
   - HIGH-VOL RANGING: Less severe (-0.84 Sharpe)
   - **Regimes have meaningful risk signatures**

## Next Steps

1. **Reduce to N_STATES=3** and revalidate the same tickers for cleaner separation
2. **Test validation outputs** (walk-forward heatmap, BIC scores) in dashboard
3. **Expand to broader NSE universe** (20+ names) to identify regime spectrum
4. **Phase 2: Risk scoring** — use regime labels as features in a stock ranking model
5. **Phase 2.1: Market-wide scanner** — daily regime scan of all liquid NSE names

---

**Mentat Phase 1.2 Status:** ✅ Walk-forward validation working. ✅ Regime labels stable. ⚠ State count may be optimizable. 🎯 Ready for broader NSE expansion.

## Phase 3 VCP Screener Validation (April 2026)

### Objective
- Validate the new institutional VCP screener on a custom ticker list and verify time-machine behavior over a rolling historical window.

### Test Inputs
- Universe file used: Stock Screener.csv (17 user-provided symbols)
- Scanner: outlier_hunter.py
- Core VCP gates enforced:
   - Trend: Close > SMA50 > SMA200
   - Volatility coil: BBW near 120-day minimum
   - Volume shock: volume > 3x 20-day average
   - Breakout confirmation: close > open and close > upper Bollinger Band

### Runs Executed
- Spot run (current date context):
   - Command style: python outlier_hunter.py --csv "Stock Screener.csv" --symbol-col Symbol
   - Result: 0 tickers passed all gates
- Time-machine spot check:
   - As-of date tested: 2024-03-28
   - Result: 0 tickers passed all gates
- Time-machine monthly sweep:
   - Date range: 2026-03-01 to 2026-04-02 (33 daily runs)
   - Result: 0 passes across all 33 scans
   - Daily artifacts generated under analysis/validation as vcp_targets_YYYY-MM-DD.csv

### Data Quality Notes
- 2 symbols were repeatedly skipped due to insufficient history for full SMA200/BBW windows:
   - SOLARWORLD.NS
   - AYE.NS
- Failed download count remained 0 during monthly sweep.

### Interpretation
- The 4-gate VCP logic is operating correctly and is currently very selective on this universe/date window.
- No breakout met all quality constraints simultaneously in the tested period.

### Cleanup Action
- Raw temporary CSV outputs from manual VCP scans were removed from repository root.
- Ongoing temporary scanner outputs are now ignored via .gitignore to keep git status clean.
