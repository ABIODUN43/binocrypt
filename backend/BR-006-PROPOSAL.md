# BR-006 Research Preregistration Proposal: Prospective Paper Validation of Frozen LambdaRank + BTC EMA50 Architecture

**Experiment ID:** `BR-006`  
**Preregistration Date:** `2026-09-27`  
**Status:** **`PROPOSED / PREREGISTERED — PENDING USER REVIEW & LOCK`**  
**Parent Studies:** `BR-005` (Predictive Cross-Sectional Ranking, `PREDICTIVELY_VALIDATED`), `BR-005.1` (Macro Exposure Overlay, `NOT_VALIDATED` under full Sharpe hurdle but Gates 1, 3, 4 passed)  
**Governance:** Strictly **`RESEARCH_ONLY`** (Zero production changes, `EntryZoneEstimator` / BR-003.3 on strict HOLD, no execution until locked)

---

## 1. Research Context & Motivation

Through BR-001 to BR-005.1, the quantitative research journey has established two empirical pillars:

1. **Relative Selection Alpha Exists (BR-005):**  
   Model 3 LightGBM Pairwise LambdaRank demonstrated statistically significant, monotonic out-of-sample ranking capacity on the primary 14-day horizon ($IC = +0.0656, t_{HAC} = +2.32$, full quintile monotonicity $Q_1 \to Q_5$, positive $IC$ in 4/4 walk-forward folds).
2. **Macro Exposure Control Mitigates Drawdown (BR-005.1):**  
   Overlaying a causal BTC EMA50 trend filter ($E_1$) cut maximum drawdown from **54.16% to 21.19%**, elevated the Calmar ratio from **0.26 to 1.42**, and increased Net Sharpe from **+0.53 to +0.89** while preserving an active alpha spread of **+2.27% per period** (62.5% active win rate).

### The Necessity of Prospective Validation:
No matter how strictly a historical backtest purges and embargoes walk-forward folds, backtested performance carries an inescapable hazard of subtle retrospective conditioning and selection bias.

The ultimate scientific test of an algorithmic trading architecture is **Prospective Out-of-Sample Paper Validation**:
> *Does the completely frozen ranking and exposure architecture maintain its cross-sectional predictive edge and risk-adjusted economic performance on genuinely unseen, forward-arriving observations generated in real time, with zero parameter modifications or post-hoc tuning?*

---

## 2. Completely Frozen Architecture (Zero Modifications)

Under BR-006, all parameters, feature definitions, universe constituents, and operational rules are **strictly frozen**:

1. **Ranking Engine (Model 3 LambdaRank):**  
   - Objective: `lambdarank`
   - Hyperparameters: `n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_samples=30, reg_alpha=0.5, reg_lambda=1.0, random_state=42`
   - Model weights trained on historical data up to 2026-09-25 23:59:59 UTC, serialized to disk, and cryptographically hashed. Zero retraining during the prospective test.
2. **Feature Set (45 Causal Features):**  
   Strictly identical `FEATURE_COLS` computed point-in-time from daily OHLCV bars.
3. **Universe (`SBRU_V1`):**  
   The frozen 88 Binance Spot USDT pairs. No asset additions or deletions.
4. **Macro Exposure Overlay ($E_1$ BTC EMA50):**  
   $$w_{macro}(t) = \begin{cases} 1.0 & \text{if } \text{Close}_{BTC, t} \ge \text{EMA}_{50, BTC, t} \\ 0.0 & \text{if } \text{Close}_{BTC, t} < \text{EMA}_{50, BTC, t} \text{ (100\% USDT Cash)} \end{cases}$$
5. **Rebalancing Horizon & Portfolio Construction:**  
   - Non-overlapping 14-day holding periods ($H = 14\text{D}$).
   - Top quintile $Q_5$ (top ~17 assets) equally weighted by $w_{macro}(t) / |Q_5|$.
   - Unallocated capital $(1 - w_{macro}(t))$ held in USDT earning 0.0% yield.
6. **Execution Timing Discipline:**  
   - Signals formed strictly at the close of day $t$ (23:59:59 UTC).
   - Prospective paper portfolio executes at `Open[t+1]` (00:00:00 UTC).
7. **Transaction Friction Assumptions:**  
   - 0.40% round-trip fee on spot equity turnover.
   - 0.20% one-way fee on capital moving into or out of cash.

---

## 3. Prospective Validation Protocol & Safeguards

To ensure unassailable prospective integrity:

### A. Pre-Execution Cryptographic Commitment
Prior to logging any forward observation:
- The trained model binary `br006_frozen_lambdarank.joblib` is saved to disk and its SHA-256 hash is recorded in the locked version of this proposal.
- A static configuration file `backend/app/research/br006/config.json` stores all frozen metadata.

### B. Forward-Only Logging Procedure
At each 14-day rebalance interval:
1. **Timestamped Signal Snapshot:** At Close of day $t$, log the feature vector, the emitted scores for all 88 assets, the BTC EMA50 status, $w_{macro}(t)$, and the resulting target basket weights.
2. **Immutable Commit:** The signal snapshot file `snapshot_YYYYMMDD.json` is committed to git before forward prices unfold.
3. **Execution at Next Open:** Target weights are recorded as executing at `Open[t+1]`.
4. **Forward Outcome Evaluation:** Only after the 14-day holding window completes ($t+1+14$), actual realized returns and portfolio equity are evaluated against benchmarks.

---

## 4. Preregistered Prospective Validation Gates

The prospective paper test will run across a minimum of **6 prospective rebalance periods (84 calendar days)** and evaluate:

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
- During active equity windows ($w_{macro} > 0$), the portfolio must achieve a positive excess return over the exposure-matched equal-weight benchmark:
  $$Q_5 - EW_{matched} > 0.0\%$$
- Active batting average vs $EW_{matched}$ must be $\ge 50.0\%$.

### Gate P4: Implementation Fidelity Gate
- Realized transaction friction and portfolio turnover must match modeled projections within a tolerance band of $\pm 10\%$.

---

## 5. Governance & Constraints

- **Strict Research Only:** All logging and validation reside strictly in `backend/app/research/br006/`.
- **Zero Production Changes:** No production routes, databases, live exchange API keys, or order execution pipelines are modified.
- **EntryZoneEstimator (BR-003.3):** Remains on **strict HOLD**.
- **No Implementation Until Locked:** No prospective script or data collector will be run until the user has formally reviewed and locked this proposal.
