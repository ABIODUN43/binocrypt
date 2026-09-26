# BR-004 Research Proposal: Dynamic Regime-Conditioned & Online Probability Calibration

**Proposal Identifier:** `BR-004-PROPOSAL`  
**Date:** `2026-09-26`  
**Status:** `PROPOSED` (Awaiting User Review — No code or models implemented)  
**Parent Studies:** `BR-002` (Depression Dynamics), `BR-003` (Forward Path & Level-Reach Studies)  
**Target Application:** Multi-week probability calibration ($h \ge 14\text{D}$) for downstream path modeling  

---

## 1. Context & Motivation

In previous research:
1. **BR-002.1A/B/C** established that unconditional depressed-state buying has zero statistical edge, but causal features provide significant ranking discrimination ($AUC \approx 0.62\text{--}0.68$). However, static probability models suffered base-rate collapse across market cycles, yielding $BSS \le 0$.
2. **BR-003.1 & BR-003.2** tested forward level-reach ($P(\min Low \le L)$) and first-passage ($P(\tau_L < \tau_U)$). While short horizons ($h \le 3\text{D}$) successfully passed predictive gates ($BSS = +0.0413, ECE = 0.0408$), multi-week horizons ($h \ge 14\text{D}$) failed calibration ($BSS = -0.0771$).
3. **BR-003-CALIBRATION-AUDIT** proved that the long-horizon calibration failure is not an implementation flaw:
   - Walk-forward splits strictly preserved temporal isolation.
   - Post-hoc static calibrators (Platt, Isotonic) trained on historical folds locked in stale base rates and exacerbated test miscalibration during macro regime shifts.
   - Test cohort arrival rates swung by $\pm 10\text{--}20\%$ across expanding folds, driving large $\text{Reliability}$ penalties in the Murphy Brier decomposition.

**Core Scientific Question:**  
Can dynamic macro regime conditioning or online base-rate tracking resolve the non-stationary calibration penalty and produce validated, calibrated multi-week probabilities ($BSS > 0 \land ECE < 0.10$)?

---

## 2. Formal Research Hypotheses

### Hypothesis 1 ($H_1$): Dynamic Macro Regime Conditioning
> *Conditioning the probability calibrator hierarchically on point-in-time macro regime states (Bitcoin trend, cross-sectional breadth, and volatility state) will absorb cycle-level base-rate shifts, reducing the out-of-sample calibration error ($ECE < 0.10$) and generating positive Brier Skill Score ($BSS > 0$) relative to both fold climatology ($B_1$) and static regime climatology ($B_2$).*

### Hypothesis 2 ($H_2$): Online / Recurrent Base-Rate Tracking
> *Updating the calibrator's base-rate prior $\pi_t$ dynamically via an online causal recursion (e.g. exponential moving average or Bayesian online filter on closed-horizon realizations) will eliminate stale base-rate bias without lookahead, producing $BSS > 0$ relative to expanding climatology ($B_1$) and rolling climatology ($B_3$).*

---

## 3. Approved Benchmark Baselines

To prove that improvements stem from predictive adaptation rather than statistical artifacts, candidate models must beat four pre-registered causal baselines:

1. **Baseline 0 ($B_0$ — Cumulative Expanding Climatology):**  
   Naive base rate of all observations from dataset inception up to fold train end: $p_{B0} = \bar{y}_{hist}$.
2. **Baseline 1 ($B_1$ — Immediate Fold Climatology):**  
   Naive base rate of the expanding training fold: $p_{B1} = \bar{y}_{train}$.
3. **Baseline 2 ($B_2$ — Binary Regime Climatology):**  
   Training fold base rate conditioned on Bitcoin trend:  
   $$p_{B2} = P(y=1 \mid \text{BTC} > \text{EMA}_{50})_{train} \text{ if } \text{BTC}_t > \text{EMA}_{50,t} \text{ else } P(y=1 \mid \text{BTC} \le \text{EMA}_{50})_{train}$$
4. **Baseline 3 ($B_3$ — Online Rolling Climatology):**  
   A non-ML benchmark that tracks the trailing $W$-day empirical base rate of realized events ($t_i + h \le t$) with $W \in [30\text{D}, 60\text{D}, 90\text{D}]$. This tests whether machine learning adds value beyond a simple moving-average base-rate tracker.

---

## 4. Candidate Adaptation & Calibration Architectures

### Architecture A: Multi-State Regime-Conditioned Calibrator
- **Concept:** Segment the calibration mapping by macro market state:
  $$\text{logit}(P(Y=1 \mid z_t, R_t)) = a_{R_t} \cdot z_t + b_{R_t}$$
  where $R_t \in \{\text{Bull}, \text{Bear}, \text{Sideways/HighVol}\}$ is determined causal at time $t$ via $\text{BTC}/\text{EMA}_{50}$, 20-day realized volatility, and market breadth.
- **Hypothesis Tested:** Macro state conditioning absorbs the base-rate shift while preserving intra-regime feature discrimination.

### Architecture B: Bayesian Online Log-Odds Updating
- **Concept:** Separate discrimination from base-rate tracking. LightGBM outputs margin $z_t = \log \frac{p_t}{1 - p_t}$. The operational probability is:
  $$P_t = \sigma(z_t + \delta_t)$$
  where $\delta_t$ is an online base-rate correction updated recursively on closed-horizon realizations ($t_i + h \le t$):
  $$\delta_t = \delta_{t-1} + \eta \cdot (y_{t-h} - \hat{p}_{t-h})$$
- **Hypothesis Tested:** Continuous online tracking prevents stale training-set priors from contaminating future test predictions.

### Architecture C: Rolling-Window Walk-Forward Training
- **Concept:** Replace the expanding walk-forward window with a rolling 365-day training window (with identical 90D purge and 45D embargo).
- **Hypothesis Tested:** Expiring data older than 1 year removes obsolete cycle regimes from tree splits and calibrator parameters.

---

## 5. Strict Leakage, Causality, and Data Controls

1. **Horizon Closure Constraint (Strict Online Causality):**  
   For any online updating mechanism operating at calendar day $t$, the algorithm may **only** observe outcomes from events whose forecast horizon $h$ has fully closed:
   $$\text{Observable outcomes at day } t = \{y_i : t_i + h \le t\}$$
   Outcomes from events initiated at $t - k$ (where $k < h$) remain strictly unobserved.
2. **Purge and Embargo Protocols:**  
   - Walk-forward model training retains the mandatory **90-day purge** ($H_{max}$) and **45-day embargo** between training end and test start.
3. **Data Scope:**  
   - Universe: `SBRU_V1` (88 Binance Spot USDT pairs).
   - Exchange: Direct Binance Spot (`api.binance.com`) only. No third-party or fallback data.
4. **Frozen Features and Targets:**  
   - Primary target benchmark: BR-003.1 ARB Proxy ($L=-8\%, h=14\text{D}$).
   - Secondary targets: $L=-8\%, h=30\text{D}$ and BR-003.2 ARB Diagnostic ($L=-6.77\%, U=+5.05\%, 30\text{D}$).
   - Feature set: Frozen 45 causal features (`FEATURE_COLS`).

---

## 6. Walk-Forward Validation Protocol

- **Timeline:** 1,095 calendar days (3 calendar years).
- **Folds:** 4 expanding walk-forward folds (identical to BR-003.1/BR-003.2 for direct comparability).
- **Split Parameters:** Min train = 180 days, step = 90 days, purge = 90 days, embargo = 45 days.
- **Evaluation Windows:**
  - Fold 0: Test Aug 2024 – Nov 2024
  - Fold 1: Test Mar 2025 – Jun 2025
  - Fold 2: Test Oct 2025 – Jan 2026
  - Fold 3: Test Jun 2026 – Sep 2026

---

## 7. Predictive Gates & Pre-Registered Decision Rules

A candidate calibration model will be declared **`VALIDATED`** if and only if it jointly satisfies all four pre-registered criteria on the primary target ($L=-8\%, h=14\text{D}$):

| Gate ID | Metric | Threshold | Target / Condition |
| :---: | :--- | :---: | :--- |
| **Gate 1** | Brier Skill Score vs $B_1$ | **$BSS_{B1} > 0.00$** | Pooled out-of-sample test predictions beat fold climatology. |
| **Gate 2** | Brier Skill Score vs $B_3$ | **$BSS_{B3} > 0.00$** | Pooled predictions beat online rolling climatology (ML adds value beyond dynamic base rate). |
| **Gate 3** | Expected Calibration Error | **$ECE < 0.10$** | Evaluated across 10 probability bins on pooled out-of-sample test predictions. |
| **Gate 4** | Temporal Fold Consistency | **$\ge 3 \text{ of } 4$ Folds** | $BSS_{B1} > 0.00$ and $ECE < 0.10$ in at least 3 individual test folds (preventing single-fold anomalies). |

### Governance Outcomes:
- **If ALL Gates Pass:** The model is certified for downstream consumption, and we may proceed to propose and implement the operational entry-zone estimator (BR-003.3).
- **If ANY Gate Fails:** The model remains tagged as **`RESEARCH_ONLY`**, is rejected from the production registry, and multi-week operational entry-zone modeling is formally declared unvalidated.

---

## 8. Explicit Stop Gate & Next Steps

This proposal is strictly an experimental blueprint. In adherence to research protocol:
- **No production code has been modified.**
- **No models have been trained or registered.**
- Execution will begin **only** after user review, refinement, and explicit approval.
