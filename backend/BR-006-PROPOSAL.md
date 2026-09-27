# BR-006: Prospective Paper Validation Pilot

**Experiment ID:** `BR-006`  
**Preregistration Date:** `2026-09-27`  
**Status:** **`LOCKED & FROZEN — FINAL DESIGN LOCK APPROVED`**  
**Parent Studies:** `BR-005` (Predictive Cross-Sectional Ranking, `PREDICTIVELY_VALIDATED`), `BR-005.1` (Macro Exposure Overlay, `NOT_VALIDATED` under full Sharpe hurdle but Gates 1, 3, 4 passed)  
**Governance:** Strictly **`RESEARCH_ONLY`** (Zero production changes, `EntryZoneEstimator` / BR-003.3 on strict HOLD, zero live trading)

---

## 1. Research Context & Motivation

Through the quantitative research progression from BR-001 through BR-005.1, Binocrypt has firmly established two empirical pillars:

1. **Relative Cross-Sectional Alpha is Validated (BR-005):**  
   Model 3 LightGBM Pairwise LambdaRank demonstrated statistically significant, monotonic out-of-sample ranking capacity on the primary 14-day horizon ($IC = +0.0656, t_{HAC} = +2.32$, 95% circular block bootstrap CI strictly positive, full quintile monotonicity $Q_1 \to Q_5$, positive $IC$ in 4/4 walk-forward folds).
2. **Causal Macro Exposure Overlay Mitigates Broad Market Drawdown (BR-005.1):**  
   Overlaying a causal BTC EMA50 trend filter ($E_1$) slashed maximum drawdown from **54.16% to 21.19%**, elevated the Calmar ratio from **0.26 to 1.42**, and increased Net Sharpe from **+0.53 to +0.89** while preserving an active alpha spread of **+2.27% per period** (62.5% active win rate).

### The Necessity of the Prospective Paper Validation Pilot:
No matter how strictly a historical backtest purges and embargoes walk-forward folds, historical simulation always carries an inescapable hazard of subtle retrospective conditioning and selection bias.

The ultimate scientific test of an algorithmic trading architecture is **Prospective Out-of-Sample Paper Validation**:
> *Does the completely frozen ranking and exposure architecture maintain its cross-sectional predictive edge and risk-adjusted economic performance on genuinely unseen, forward-arriving observations generated in real time, with zero parameter modifications or post-hoc tuning?*

---

## 2. Completely Frozen Architecture & Evaluation Window

All parameters, feature definitions, universe constituents, and operational rules are **strictly frozen**:

### A. Evaluation Window & Checkpoint Structure
1. **Interim Checkpoint (6 Bi-Weekly Periods / ~84 Calendar Days):**  
   Six 14D periods are the minimum interim checkpoint to verify logging fidelity, execution cost alignment, and absence of catastrophic failure. **This checkpoint produces an interim audit report only.** The model will **not** be promoted, retuned, or terminated because of interim performance.
2. **Final Evaluation Window (12 Consecutive Bi-Weekly Periods / ~168 Calendar Days):**  
   A full window of 12 consecutive 14-day rebalance periods is frozen as the mandatory duration required for any final production-validation determination.

### B. Frozen Model Parameters & Features (Zero Retraining / Tuning)
1. **Ranking Engine (Model 3 LambdaRank):**  
   - Objective: `lambdarank`
   - Canonical Hyperparameters: `n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_samples=30, reg_alpha=0.5, reg_lambda=1.0, random_state=42`
   - Model weights trained on historical data up to 2026-09-25 23:59:59 UTC, serialized to disk, and cryptographically hashed (`br006_frozen_lambdarank.joblib`).
   - **Zero retraining, recalibration, threshold changes, feature changes, or overlay changes using BR-006 observations.**
2. **Feature Set (45 Causal Features):**  
   Strictly identical `FEATURE_COLS` computed point-in-time from daily OHLCV bars.
3. **Universe (`SBRU_V1`):**  
   The frozen 88 Binance Spot USDT pairs. No asset additions or replacements.
4. **Treatment of Unavailable or Delisted Symbols:**  
   If an asset in `SBRU_V1` becomes delisted, suspended, or halts trading, its target portfolio weight is liquidated to cash at the last valid close price (or 0.0% return if frozen). **No future-information-based replacement symbols may be introduced.**
5. **Macro Exposure Overlay ($E_1$ BTC EMA50):**  
   $$w_{macro}(t) = \begin{cases} 1.0 & \text{if } \text{Close}_{BTC, t} \ge \text{EMA}_{50, BTC, t} \\ 0.0 & \text{if } \text{Close}_{BTC, t} < \text{EMA}_{50, BTC, t} \text{ (100\% USDT Cash)} \end{cases}$$
6. **Rebalancing Horizon & Portfolio Construction:**  
   - Non-overlapping 14-day holding periods ($H = 14\text{D}$).
   - Top quintile $Q_5$ (top ~17 assets) equally weighted by $w_{macro}(t) / |Q_5|$.
   - Unallocated capital $(1 - w_{macro}(t))$ held in USDT earning strictly **0.0% cash yield**.
7. **Execution Timing Discipline:**  
   - Signals formed strictly at Close of day $t$ (23:59:59 UTC).
   - Prospective paper portfolio executes at `Open[t+1]` (00:00:00 UTC).
8. **Transaction Friction & Execution Cost Accounting:**  
   - **Modeled Friction Baseline:** 0.40% round-trip fee on spot equity turnover; 0.20% one-way fee on capital moving into or out of cash.
   - **Realized Execution Cost Tracking:** Realized execution cost will be tracked using Binance Spot standard fees (0.10% taker/maker per side) plus bid-ask half-spread, compared against the modeled 0.40% baseline **without assuming favorable fills**.

---

## 3. Reference Benchmarks & Prospective Validation Gates

### Historical Reference Standards (Preserved Without Weakening)
All prospective results will be benchmarked against the historical out-of-sample metrics established in BR-005 and BR-005.1:
- Historical Mean Rank $IC$: $+0.0656$ ($t_{HAC} = +2.32$)
- Historical Active Gross Alpha Spread: $+2.27\%$ per 14-day period
- Historical Active Win Rate vs $EW_{matched}$: $62.5\%$
- Historical Calmar Ratio: $1.42$
- Historical Net Max Drawdown: $21.19\%$
- Historical Net Sharpe: $+0.89$

The historical validation standards are preserved and will **not** be weakened simply because the prospective sample is smaller.

### Preregistered Prospective Validation Gates

Across the prospective pilot, the architecture will be evaluated against five gates:

### Gate P1: Prospective Predictive Power Gate
- Mean Rank Information Coefficient ($IC$) across prospective periods must be strictly positive:
  $$\text{Mean Prospective } IC > 0.00$$
- The average gross return spread between top and bottom quintiles must be positive:
  $$Q_5 - Q_1 > 0.0\%$$

### Gate P2: Macro Exposure Protection Gate
- In any prospective period where $w_{macro}(t) = 0.0$ (cash allocation), portfolio drawdown must not exceed transaction exit friction ($MaxDD_{cash} \le 0.50\%$).
- Overall prospective maximum drawdown must not exceed 25.0%:
  $$MaxDD_{prospective} \le 25.0\%$$

### Gate P3: Alpha Non-Degradation Gate (Exposure-Matched)
- **Cumulative Excess Return:** Cumulative net return of $Q_5$ must exceed the Exposure-Matched Equal-Weight Benchmark ($EW_{matched}$):
  $$\text{Cumulative Net Return}(Q_5) - \text{Cumulative Net Return}(EW_{matched}) > 0.0\%$$
- **Active Period Spread:** During active equity windows ($w_{macro} > 0$), gross alpha spread must remain positive:
  $$Q_5 - EW_{matched} > 0.0\%$$
- **Active Win Rate:** Active period win rate vs $EW_{matched}$ must be $\ge 50.0\%$.

### Gate P4: Implementation & Cost Fidelity Gate
- Realized transaction friction and portfolio turnover must match modeled projections within a tolerance band of $\pm 10\%$, confirming that the 0.40% round-trip assumption accurately reflects non-favorable execution.

### Gate P5: Protocol Integrity Gate
- Verification that all snapshot files were committed to git prior to forward price realization, with zero model retraining or parameter changes.

---

## 4. Prospective Logging Discipline & Workflow

At each bi-weekly boundary $t$:
1. **Run Snapshot Collector:** Execute `python backend/app/research/br006/collect_snapshot.py --date YYYY-MM-DD`.
2. **Log Point-in-Time Snapshot:** Serialize to `backend/app/research/br006/snapshots/snapshot_YYYYMMDD.json`:
   - Feature vector for all 88 assets as of Close $t$.
   - Emitted ranking scores and top quintile ($Q_5$) basket.
   - Causal BTC EMA50 value and resulting $w_{macro}(t)$.
   - Target portfolio weight vector (including cash allocation).
3. **Commit Snapshot to Git:** Immediately commit the snapshot file with a git timestamp prior to the arrival of prices for $t+1 \to t+1+14$.
4. **Evaluate Completed Periods:** When a 14-day holding period concludes, execute the evaluator to record realized returns against $EW_{matched}$ and $EW_{unhedged}$.

---

## 5. Model Governance

- Model status remains strictly **`RESEARCH_ONLY`** throughout BR-006.
- Zero live trading, zero production routing, and zero order placement.
- `EntryZoneEstimator` (BR-003.3) remains on strict **HOLD**.
