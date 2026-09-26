# BR-005 Research Proposal: Cross-Sectional Opportunity Ranking & Friction-Adjusted Selection

**Proposal Identifier:** `BR-005-PROPOSAL`  
**Date:** `2026-09-26`  
**Status:** `PROPOSED` (Preregistered — No code, models, or simulations implemented until approved)  
**Parent Studies:** `BR-002` (Depression Dynamics), `BR-003` (Level-Reach), `BR-004` (Probability Calibration Audit)  
**Strategic Focus:** Transitioning from uncalibrated absolute multi-week probabilities to relative cross-sectional ranking and friction-adjusted economic selection  

---

## 1. Research Context & Motivation

The progression of empirical research in Binocrypt has firmly established:
1. **Unconditional Rules Fail (BR-002.1A):** Simple floor/dip-buying heuristics have zero statistical edge over the broader market.
2. **Relative Ranking Signal Exists (BR-002.1B/C, BR-003.1, BR-004):** Across all experiments, causal feature sets consistently exhibited strong ranking and discrimination capacity ($AUC \approx 0.61\text{--}0.69$, Top-vs-Bottom quintile spreads $> +25\%$).
3. **Absolute Multi-Week Calibration Fails (BR-003, BR-004):** Cycle-level macro regime shifts cause multi-week test base rates to drift by $\pm 10\text{--}20\%$, preventing static and dynamic models from beating expanding climatology ($BSS_{B1} \le 0$).

**Core Pivot:**  
Rather than spending further research cycles trying to force machine learning models to emit calibrated absolute probabilities over multi-week horizons, **BR-005 tests whether the validated relative ranking signal is economically meaningful**. Specifically, we evaluate whether causal features can identify which assets will outperform the cross-sectional median, and whether a long-only top-quantile selection strategy generates risk-adjusted excess returns after real-world transaction friction.

---

## 2. Research Questions & Hypotheses

### Primary Research Question:
> *Can point-in-time causal features rank cryptocurrency assets within the daily Binance Spot cross-section such that higher-ranked assets exhibit superior out-of-sample risk-adjusted forward outcomes across 7D, 14D, and 30D horizons?*

### Formal Hypothesis 1 ($H_1$ — Cross-Sectional Predictive Ranking):
> *A causal ranking model trained on point-in-time features produces a positive, statistically significant daily cross-sectional Spearman rank correlation (Mean Rank $IC > 0.03, t\text{-statistic} > 2.0$) and a strictly monotonic out-of-sample forward return gradient across quintiles.*

### Formal Hypothesis 2 ($H_2$ — Friction-Adjusted Economic Excess Return):
> *A long-only top-quantile portfolio rebalanced at horizon $H$ achieves an annualized Sharpe ratio $\ge 1.00$ and beats the equal-weighted universe benchmark after 0.40% round-trip transaction costs.*

---

## 3. Preserved Controls & Research Discipline

1. **Frozen 45 Causal Features:** Strictly identical feature set (`FEATURE_COLS`) computed exclusively from historical OHLCV data prior to time $t$. No new macro variables or retrospective indicators.
2. **Universe:** `SBRU_V1` (88 consistently-listed Binance Spot USDT pairs). Direct Binance Spot (`api.binance.com`) only.
3. **Expanding Walk-Forward Protocol:**  
   - Minimum training: 180 calendar days.
   - Step size: 90 calendar days.
   - Purge window: 90 calendar days ($H_{max}$).
   - Embargo window: 45 calendar days.
   - 4 Expanding Folds over 1,095 calendar days (identical to BR-003 and BR-004).
4. **Strict Model Governance:**  
   - All models remain tagged as **`RESEARCH_ONLY`**.
   - Zero modifications to the production trading engine, database schemas, or routing.

---

## 4. Candidate Models & Approved Baselines

### Baselines:
- **Baseline 0 (Random Ranking):** Uniform random cross-sectional ranking (expected $IC = 0.00$).
- **Baseline 1 (Simple Momentum Rank):** Rank based on trailing 14D/30D arithmetic momentum (`mom_14d`, `mom_30d`).
- **Baseline 2 (Simple Mean-Reversion / Dip-Buying Rank):** Rank based on 90D drawdown (`dd_from_90d_high`, testing whether buying the deepest dips provides a cross-sectional edge).

### Candidate Ranking Models:
- **Model 1 (Cross-Sectional Ridge Regression):** Linear $L_2$-regularized regression trained on cross-sectionally standardized features ($z$-scored daily) to predict forward return.
- **Model 2 (Non-Linear LightGBM Regressor):** Constrained gradient boosted trees (100 trees, depth 3, regularized) predicting forward return.
- **Model 3 (LightGBM Pairwise LambdaRank):** Direct optimization of NDCG / pairwise ranking order across the daily cross-section.

---

## 5. Phased Evaluation Protocol & Pre-Registered Gates

To prevent data mining and premature backtesting, BR-005 enforces a strict **two-phase gating architecture**:

```text
Phase 1: Cross-Sectional Predictive Ranking Analysis
                     ↓
        Did Phase 1 Pass Predictive Gate?
        ├── YES → Proceed to Phase 2 (Portfolio Simulation)
        └── NO  → STOP (Do not simulate portfolio; declare NOT_VALIDATED)
```

### Phase 1: Predictive Ranking Evaluation (Metrics & Gate)

#### Evaluated Metrics:
1. **Daily Spearman Rank Correlation (Rank $IC$):**  
   $$IC_t = \text{SpearmanCorr}(\hat{s}_t, y_{t,H})$$
   where $\hat{s}_t$ is the cross-sectional score vector and $y_{t,H}$ is the realized forward return vector over horizon $H \in [7\text{D}, 14\text{D}, 30\text{D}]$.
2. **Information Ratio of $IC$ ($IR_{IC}$):**  
   $$IR_{IC} = \frac{\overline{IC}}{\sigma(IC)} \times \sqrt{\frac{365}{H}}$$
3. **$t$-statistic of $IC$:**  
   $$t = \frac{\overline{IC}}{\sigma(IC) / \sqrt{N_{days}}}$$
4. **Quantile Spread & Monotonicity:**  
   Daily partitioning into 5 cross-sectional quintiles ($Q_1$ lowest to $Q_5$ highest).  
   - Monotonic return gradient: $\mu(Q_5) > \mu(Q_4) > \mu(Q_3) > \mu(Q_2) > \mu(Q_1)$.
   - Top-minus-Bottom spread: $\Delta_{Q5-Q1} = \mu(Q_5) - \mu(Q_1)$.

#### Pre-Registered Phase 1 Predictive Gate:
A model passes Phase 1 if and only if:
- **Mean Rank $IC \ge +0.030$** with **$t\text{-statistic} \ge 2.00$** across the pooled out-of-sample period.
- **Monotonicity:** $\mu(Q_5) > \mu(Q_3) > \mu(Q_1)$ with spread $\Delta_{Q5-Q1} > 0$.
- **Temporal Consistency:** Mean Rank $IC > 0.00$ in at least **3 out of 4** walk-forward folds.

---

### Phase 2: Preregistered Friction-Adjusted Portfolio Simulation (Gated)

*Phase 2 will be executed ONLY if a candidate model passes the Phase 1 Predictive Gate.*

#### Simulation Specifications:
1. **Portfolio Structures:**
   - **Top Decile ($D_{10}$):** Top 10% of ranked assets (~9 symbols).
   - **Top Quintile ($Q_5$):** Top 20% of ranked assets (~18 symbols).
   - **Benchmark ($EW$):** Equal-weighted portfolio of all active universe symbols.
2. **Rebalancing Schedule:**
   - $H = 7\text{D}$: Weekly rebalance.
   - $H = 14\text{D}$: Bi-weekly rebalance.
   - $H = 30\text{D}$: Monthly rebalance.
3. **Capital Allocation & Position Sizing:**
   - Equal weight per selected asset ($1 / K$).
   - Full capital reinvestment (no leverage, long-only).
4. **Mandatory Friction Modeling:**
   - **Transaction Cost:** Fixed **0.40% round-trip fee** (0.20% entry taker/maker fee + slippage, 0.20% exit fee + slippage) applied to all turnover.
   - Turnover explicitly tracked and penalized: $\text{Friction}_t = \text{Turnover}_t \times 0.40\%$.
5. **Economic Performance Metrics:**
   - Net Compound Annual Growth Rate ($CAGR_{net}$).
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

## 6. Implementation Guardrails

- **Proposal Status:** Strictly non-executable proposal until user review.
- **Code Freeze:** No files in `backend/app/research/br005/` or scripts will be written until this proposal is reviewed and approved.
- **Decoupled from BR-003.3:** BR-003.3 remains on hold. BR-005 operates as an independent, relative-value investigation.
