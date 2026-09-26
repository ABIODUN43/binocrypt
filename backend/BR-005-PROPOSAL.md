# BR-005 Research Proposal: Cross-Sectional Opportunity Ranking & Friction-Adjusted Selection

**Proposal Identifier:** `BR-005-PROPOSAL`  
**Date:** `2026-09-26`  
**Status:** `LOCKED & FROZEN` (Preregistered — Phase 1 implementation begins only from this frozen specification)  
**Parent Studies:** `BR-002` (Depression Dynamics), `BR-003` (Level-Reach), `BR-004` (Probability Calibration Audit)  
**Strategic Focus:** Transitioning from uncalibrated absolute multi-week probabilities to relative cross-sectional opportunity ranking and friction-adjusted economic selection  

---

## 1. Research Context & Motivation

The progression of empirical research in Binocrypt has firmly established:
1. **Unconditional Rules Fail (BR-002.1A):** Simple floor/dip-buying heuristics have zero statistical edge over the broader market.
2. **Relative Ranking Signal Exists (BR-002.1B/C, BR-003.1, BR-004):** Across all experiments, causal feature sets consistently exhibited strong ranking and discrimination capacity ($AUC \approx 0.61\text{--}0.69$, Top-vs-Bottom quintile spreads $> +25\%$).
3. **Absolute Multi-Week Calibration Fails (BR-003, BR-004):** Cycle-level macro regime shifts cause multi-week test base rates to drift by $\pm 10\text{--}20\%$, preventing static and dynamic models from beating expanding climatology ($BSS_{B1} \le 0$).

**Core Strategic Pivot:**  
Rather than spending further research cycles trying to force machine learning models to emit calibrated absolute probabilities over multi-week horizons, **BR-005 directly tests the remaining strongest hypothesis: relative opportunity ranking across assets**. 

Critically, this investigation is decoupled from any premature assumptions about exact entry prices ($L^*$) or execution timing. We first establish whether point-in-time features can reliably rank cross-sectional forward returns, and only then evaluate whether a disciplined top-quantile portfolio survives real-world friction.

---

## 2. Decoupled Research Questions & Formal Hypotheses

To ensure strict separation between statistical prediction and commercial simulation, the research is split into two distinct, decoupled inquiries:

### Phase 1 Research Question (Predictive Ranking):
> *Can point-in-time causal features rank cryptocurrency assets within the daily Binance Spot cross-section such that higher-ranked assets exhibit statistically significant, monotonic out-of-sample forward returns over a 14-day horizon?*

### Phase 2 Research Question (Economic Portfolio Simulation):
> *If cross-sectional predictive ranking is established in Phase 1, does a long-only top-quantile selection strategy rebalanced at horizon $H$ generate superior risk-adjusted economic performance (Sharpe ratio $\ge 1.00$, excess return over universe benchmark) after 0.40% round-trip transaction costs?*

---

## 3. Preserved Controls & Research Discipline

1. **Horizon Hierarchy:**  
   - **Primary Horizon:** **14-Day ($H = 14\text{D}$)**.  
   - **Secondary Horizons:** 7-Day ($H = 7\text{D}$) and 30-Day ($H = 30\text{D}$).  
   - **Strict Gating Constraint:** Secondary horizons are reported strictly for cross-horizon robustness and **cannot substitute** for a failed primary 14D predictive gate.
2. **Explicit Forward Return Definition:**  
   The target variable is the forward arithmetic return:
   $$R_{i, t, H} = \frac{\text{Close}_{i, t+H}}{\text{Close}_{i, t}} - 1$$
   All feature conditioning and model score emission for day $t$ use **only** information available through time $t$ (daily close).
3. **Frozen 45 Causal Features:** Strictly identical feature set (`FEATURE_COLS`) computed exclusively from historical OHLCV data prior to or at time $t$. No new macro variables or retrospective indicators.
4. **Universe:** `SBRU_V1` (88 consistently-listed Binance Spot USDT pairs). Direct Binance Spot (`api.binance.com`) only. No third-party or fallback data.
5. **Expanding Walk-Forward Protocol:**  
   - Minimum training: 180 calendar days.
   - Step size: 90 calendar days.
   - Purge window: 90 calendar days ($H_{max}$).
   - Embargo window: 45 calendar days.
   - 4 Expanding Folds over 1,095 calendar days (identical to BR-003 and BR-004).
6. **Strict Model Governance:**  
   - All models remain tagged as **`RESEARCH_ONLY`**.
   - Zero modifications to the production trading engine, database schemas, or routing.

---

## 4. Candidate Models & Approved Baselines (Explicit Score Direction)

To ensure unambiguous ranking semantics, the direction of every ranking score is explicitly preregistered:

### Approved Baselines:
- **Baseline 0 (Random Ranking):**  
  $$s_{i,t} \sim \text{Uniform}(0, 1)$$  
  Expected $IC = 0.00$. Serves as the null hypothesis.
- **Baseline 1 (Simple Momentum Rank):**  
  $$s_{i,t} = \text{mom}_{i, t, 14d} = \frac{\text{Close}_{i, t}}{\text{Close}_{i, t-14}} - 1$$  
  *Direction:* Higher momentum $\implies$ Higher predicted rank (anticipates cross-sectional trend continuation).
- **Baseline 2 (Simple Reversal / Dip-Buying Rank):**  
  $$s_{i,t} = -\text{dd\_from\_90d\_high}_{i, t}$$  
  *Direction:* Deepest drawdown $\implies$ Higher predicted rank (anticipates cross-sectional mean-reversion / floor rebound).

### Candidate Machine Learning Models:
- **Model 1 (Cross-Sectional Ridge Regression):**  
  Linear $L_2$-regularized model trained on daily $z$-scored causal features to predict forward return $R_{i, t, H}$.
- **Model 2 (Non-Linear LightGBM Regressor):**  
  Constrained gradient boosted trees (100 estimators, max depth 3, num leaves 7, regularized) predicting forward return $R_{i, t, H}$.
- **Model 3 (LightGBM Pairwise LambdaRank):**  
  Direct optimization of cross-sectional ranking order.  
  *Pairwise Preference Definition:* For any day $t$ in the **training fold only**, asset $i$ is preferred over asset $j$ ($i \succ j$) if and only if $R_{i, t, H} > R_{j, t, H}$. Preference pairs are derived **strictly from training-period forward returns**. Zero lookahead into test periods.

---

## 5. Phased Evaluation Protocol & Pre-Registered Gates

```text
Phase 1: Cross-Sectional Predictive Ranking Analysis (H = 14D)
                     ↓
        Did Phase 1 Pass All 3 Predictive Gates?
        ├── YES → Proceed to Phase 2 (Friction-Adjusted Portfolio Simulation)
        └── NO  → STOP (Do not simulate portfolio; declare NOT_VALIDATED)
```

---

### Phase 1: Predictive Ranking Evaluation (Metrics & Gate)

#### 1. Daily Spearman Rank Correlation (Rank $IC$):
For each day $t$ in the out-of-sample test window, compute the Spearman rank correlation between the predicted model scores $\hat{s}_{i,t}$ and the realized forward returns $R_{i, t, H}$ across all active universe assets:
$$IC_t = \text{SpearmanCorr}(\hat{s}_t, R_{t, H})$$

#### 2. Statistical Inference (HAC / Newey-West & Block Bootstrap):
Because forward returns over $H = 14\text{D}$ overlap across consecutive days, daily $IC_t$ observations exhibit serial autocorrelation.
- **Primary Statistical Test:** Heteroskedasticity and Autocorrelation Consistent (HAC) / Newey-West variance estimator with lag bandwidth $L = H + 1 = 15$ days:
  $$SE_{HAC} = \sqrt{\frac{1}{T} \left( \hat{\gamma}_0 + 2 \sum_{l=1}^{L} \left(1 - \frac{l}{L+1}\right) \hat{\gamma}_l \right)}$$
  $$t_{HAC} = \frac{\overline{IC}}{SE_{HAC}}$$
- **Robustness Check:** Circular block bootstrap (1,000 resamples, block length = 14 days) reporting empirical 95% confidence intervals for $\overline{IC}$.

#### 3. Full Monotonic Quintile Gradients:
Assets on each day $t$ are partitioned into 5 cross-sectional quintiles ($Q_1$ lowest 20% to $Q_5$ highest 20%). We track the mean realized forward return $\mu(Q_k)$ across all test days.
- **Strict Full Monotonicity Requirement:**
  $$\mu(Q_1) < \mu(Q_2) < \mu(Q_3) < \mu(Q_4) < \mu(Q_5)$$
- **Top-Minus-Bottom Spread:**
  $$\Delta_{Q5-Q1} = \mu(Q_5) - \mu(Q_1) > 0$$

#### Pre-Registered Phase 1 Predictive Gate:
A candidate model passes Phase 1 if and only if it jointly satisfies all three criteria on the **primary 14D horizon**:
1. **Predictive Strength:** Out-of-sample Mean Rank $IC \ge +0.030$ with Newey-West $t_{HAC} \ge 2.00$ (and 95% block bootstrap CI strictly above zero).
2. **Strict Full Monotonicity:** Realized mean returns satisfy $\mu(Q_1) < \mu(Q_2) < \mu(Q_3) < \mu(Q_4) < \mu(Q_5)$ with spread $\Delta_{Q5-Q1} > 0$.
3. **Temporal Consistency:** Mean Rank $IC > 0.00$ in at least **3 out of 4** walk-forward folds.

*If no model passes Phase 1 on the primary 14D horizon, BR-005 is formally declared `NOT_VALIDATED`, and Phase 2 will NOT be executed.*

---

### Phase 2: Preregistered Friction-Adjusted Portfolio Simulation (Gated)

*Phase 2 will be executed ONLY if a candidate model passes the Phase 1 Predictive Gate.*

#### Simulation Specifications:
1. **Portfolio Structures:**
   - **Top Decile ($D_{10}$):** Top 10% of ranked assets (~9 symbols).
   - **Top Quintile ($Q_5$):** Top 20% of ranked assets (~18 symbols).
   - **Benchmark ($EW$):** Equal-weighted portfolio of all active universe symbols.
2. **Rebalancing Schedule:**
   - Primary 14D Horizon: Bi-weekly rebalance (every 14 days).
   - Secondary 7D: Weekly rebalance.
   - Secondary 30D: Monthly rebalance.
3. **Capital Allocation & Position Sizing:**
   - Equal weight per selected asset ($1 / K$).
   - Long-only, no leverage, full capital reinvestment.
4. **Mandatory Friction Modeling:**
   - Fixed **0.40% round-trip transaction fee** (0.20% entry taker/maker fee + slippage, 0.20% exit fee + slippage) applied to all portfolio turnover:
     $$\text{Fee}_t = \text{Turnover}_t \times 0.40\%$$
5. **Economic Metrics:**
   - Compound Annual Growth Rate ($CAGR_{net}$).
   - Annualized Net Volatility ($\sigma_{ann}$).
   - Net Sharpe Ratio ($\text{Sharpe}_{net}$, assuming 0% risk-free rate).
   - Net Sortino Ratio ($\text{Sortino}_{net}$).
   - Maximum Drawdown ($MaxDD$) and Calmar Ratio.
   - Average Annual Turnover.

#### Pre-Registered Phase 2 Economic Gate:
A strategy is declared **`ECONOMICALLY_VALIDATED`** if and only if:
- **Net Sharpe Ratio $\ge 1.00$** after 0.40% round-trip friction.
- **Excess Return over Benchmark:** $CAGR_{net}(Q_5) > CAGR_{net}(EW)$.
- **Risk Preservation:** $MaxDD(Q_5) \le MaxDD(EW) + 5.0\%$.

---

## 6. Implementation Guardrails & Execution Status

- **Status:** This document is the locked and frozen preregistration specification.
- **Decoupled Architecture:** Phase 1 and Phase 2 will be executed in sequence. Phase 2 execution is strictly gated on Phase 1 passing.
- **Zero Production Tampering:** The production engine, schemas, and routes remain completely untouched. All outputs are written to research artifacts.
