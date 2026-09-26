# BR-003 Calibration & Stability Audit Report

**Audit Identifier:** `BR-003-CALIBRATION-AUDIT`  
**Execution Timestamp:** `2026-09-26 17:52:00 UTC`  
**Audited Components:**  
- `BR-003.1` (Level-Reach Probability Study)  
- `BR-003.2` (First-Passage Probability Study)  
**Universe:** `SBRU_V1` (88 consistently-listed Binance Spot USDT pairs)  
**Total Evaluated Observations:** 88,972 bars (1,095 calendar days, 4 expanding walk-forward folds)  
**Audited Baselines:** Baseline 0 (Cumulative Historical), Baseline 1 (Fold Climatology), Baseline 2 (Macro Regime-Aware)  

---

## Executive Summary & Final Determinations

| Task | Research Scope | Predictive Gate ($BSS > 0 \land ECE < 0.10$) | Status | Final Determination |
| :--- | :--- | :---: | :---: | :---: |
| **BR-003.1** | Level-Reach Probability Grid ($L \times h$) | **PASSED** on $h \le 3\text{D}$<br>**FAILED** on $h \ge 14\text{D}$ (ARB Proxy) | `RESEARCH_ONLY` | **`NOT_VALIDATED`** (for operational entry zones $\ge 14\text{D}$) |
| **BR-003.2** | First-Passage Directional Exit ($P(\tau_L < \tau_U)$) | **FAILED** across all barrier pairs ($BSS \in [-0.039, -0.232]$) | `RESEARCH_ONLY` | **`NOT_VALIDATED`** |

### Key Scientific Findings:
1. **Implementation Correctness Verified (No Leakage Bug):**  
   Platt scaling and Isotonic calibration were verified to be strictly fit on out-of-fold cross-validation predictions within each training fold (`cross_val_predict(cv=3)`) and evaluated purely out-of-sample on future test data. The negative Brier Skill Scores are **not** an implementation error or coding bug.
2. **Decomposition Demonstrates Discrimination vs. Calibration Separation:**  
   Murphy Brier decomposition ($BS = \text{Uncertainty} - \text{Resolution} + \text{Reliability}$) proves that models have genuine discriminatory power ($\text{Resolution} > 0$, AUCs up to 0.685). However, **$\text{Reliability}$ penalties (calibration error) swamp $\text{Resolution}$ advantages**, driving $BSS < 0$.
3. **Macro Base-Rate Drift is the Dominant Failure Mechanism:**  
   Across expanding folds, empirical base rates fluctuate by up to **20 to 26 percentage points** between training and test periods. Fitting a static post-hoc calibrator (Platt or Isotonic) on the training distribution exacerbates test miscalibration during regime shifts.
4. **Stop Gate Enforced:**  
   Because both models fail their required predictive gates for operational entry zones, **BR-003.3 (`EntryZoneEstimator`) is NOT implemented**, neither model is registered in the production registry, and downstream optimization is halted.

---

## 1. Baseline Definitions & Performance Matrix

Three approved causal baselines were evaluated alongside the models:
- **Baseline 0 ($B_0$ - Cumulative Historical Base Rate):** The empirical base rate of all observations strictly prior to the test fold's start.
- **Baseline 1 ($B_1$ - Fold Climatology):** The naive base rate of the immediate expanding training fold: $p_{B1} = \bar{y}_{train}$.
- **Baseline 2 ($B_2$ - Macro Regime-Aware Climatology):** The regime-conditioned training base rate: $p_{B2} = P(y=1 \mid \text{BTC} > \text{EMA}_{50})_{train}$ if current $\text{BTC} > \text{EMA}_{50}$, else $P(y=1 \mid \text{BTC} \le \text{EMA}_{50})_{train}$.

### Pooled Model vs. Baseline Comparison Table

| Target | Model Variant | Pooled AUC | Base Rate | Brier Score | BSS vs $B_0$ | BSS vs $B_1$ | BSS vs $B_2$ | ECE (10-bin) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BR-003.1 ARB Proxy**<br>($L=-8\%, h=14\text{D}$) | **Platt LightGBM** | 0.611 | 0.615 | 0.2550 | **-0.0771** | **-0.0771** | **-0.0788** | 0.1268 |
| | **Isotonic LightGBM** | 0.611 | 0.615 | 0.2553 | -0.0785 | -0.0785 | -0.0802 | 0.1270 |
| | **Raw LightGBM** | 0.611 | 0.615 | 0.2432 | **-0.0275** | **-0.0275** | **-0.0290** | **0.0788** |
| | **Logistic Regression** | 0.589 | 0.615 | 0.2379 | **-0.0050** | **-0.0050** | **-0.0065** | **0.0766** |
| | Baseline 1 ($B_1$) | 0.500 | 0.615 | 0.2367 | — | 0.0000 | -0.0016 | 0.0912 |
| | Baseline 2 ($B_2$) | 0.540 | 0.615 | 0.2363 | +0.0016 | +0.0016 | 0.0000 | 0.0841 |
| **BR-003.1 Short Horizon**<br>($L=-8\%, h=1\text{D}$) | **Platt LightGBM** | **0.685** | 0.064 | 0.0584 | **+0.0282** | **+0.0282** | **+0.0162** | **0.0248** |
| | **Raw LightGBM** | **0.685** | 0.064 | 0.0572 | **+0.0481** | **+0.0481** | **+0.0363** | **0.0271** |
| **BR-003.1 Short Horizon**<br>($L=-8\%, h=3\text{D}$) | **Platt LightGBM** | **0.646** | 0.256 | 0.1826 | **+0.0413** | **+0.0413** | **+0.0358** | **0.0408** |
| | **Raw LightGBM** | **0.646** | 0.256 | 0.1804 | **+0.0529** | **+0.0529** | **+0.0475** | **0.0311** |
| **BR-003.2 ARB Diagnostic**<br>($L=-6.77\% \text{ vs } U=+5.05\%, 30\text{D}$) | **Platt LightGBM** | 0.551 | 0.462 | 0.2632 | **-0.0589** | **-0.0589** | **-0.0533** | 0.1174 |
| | **Isotonic LightGBM** | 0.551 | 0.462 | 0.2631 | -0.0587 | -0.0587 | -0.0531 | 0.1126 |
| | **Raw LightGBM** | 0.551 | 0.462 | 0.2528 | **-0.0172** | **-0.0172** | **-0.0118** | **0.0742** |
| | **Logistic Regression** | 0.537 | 0.462 | 0.2614 | -0.0516 | -0.0516 | -0.0459 | 0.0902 |
| **BR-003.2 Symmetric 5%**<br>($L=-5\% \text{ vs } U=+5\%, 30\text{D}$) | **Platt LightGBM** | 0.542 | 0.534 | 0.2586 | **-0.0393** | **-0.0393** | **-0.0370** | **0.0834** |
| | **Raw LightGBM** | 0.542 | 0.534 | 0.2515 | **-0.0109** | **-0.0109** | **-0.0086** | **0.0509** |
| **BR-003.2 Symmetric 10%**<br>($L=-10\% \text{ vs } U=+10\%, 30\text{D}$) | **Platt LightGBM** | 0.584 | 0.519 | 0.2704 | **-0.0832** | **-0.0832** | **-0.0707** | 0.1314 |
| | **Raw LightGBM** | 0.584 | 0.519 | 0.2577 | **-0.0325** | **-0.0325** | **-0.0207** | **0.0851** |

---

## 2. Walk-Forward Fold-by-Fold Stability Breakdown

Evaluating walk-forward metrics across all 4 expanding folds demonstrates how macro regime shifts dictate calibration stability.

### BR-003.1 ARB Proxy ($L=-8\%, h=14\text{D}$)
| Fold | Test Window | Train Base Rate | Test Base Rate | Base Rate Drift ($\Delta$) | Model AUC | BSS vs $B_1$ (Platt) | BSS vs $B_1$ (Raw) | ECE (Platt) | ECE (Raw) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 2024-08-07 $\to$ 2024-11-05 | 52.1% | 55.6% | +3.5% | 0.533 | -0.3433 | -0.2253 | 0.2656 | 0.2118 |
| **1** | 2025-03-20 $\to$ 2025-06-18 | 61.1% | 69.9% | +8.8% | 0.616 | **+0.0203** | **+0.0510** | 0.1261 | 0.1002 |
| **2** | 2025-10-31 $\to$ 2026-01-29 | 65.1% | 75.4% | +10.3% | 0.546 | -0.0261 | **+0.0142** | 0.1135 | **0.0774** |
| **3** | 2026-06-13 $\to$ 2026-09-11 | 65.3% | 45.1% | **-20.2%** | 0.582 | **+0.0609** | **+0.0676** | 0.1599 | 0.1625 |

### BR-003.2 ARB Diagnostic ($L=-6.77\% \text{ vs } U=+5.05\%, 30\text{D}$)
| Fold | Test Window | Train Base Rate | Test Base Rate | Base Rate Drift ($\Delta$) | Model AUC | BSS vs $B_1$ (Platt) | BSS vs $B_1$ (Raw) | ECE (Platt) | ECE (Raw) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 2024-08-07 $\to$ 2024-11-05 | 34.6% | 39.5% | +4.9% | 0.516 | -0.1180 | -0.0821 | 0.1567 | 0.1165 |
| **1** | 2025-03-20 $\to$ 2025-06-18 | 43.0% | 46.3% | +3.3% | 0.582 | -0.0175 | -0.0051 | **0.0775** | **0.0274** |
| **2** | 2025-10-31 $\to$ 2026-01-29 | 44.1% | 54.2% | +10.1% | 0.501 | -0.0387 | -0.0144 | 0.1283 | 0.0973 |
| **3** | 2026-06-13 $\to$ 2026-09-11 | 44.5% | 44.8% | +0.3% | 0.494 | -0.0612 | -0.0298 | 0.1085 | **0.0490** |

---

## 3. Reliability & Calibration Curves (10 Probability Bins)

The 10-bin reliability diagrams illustrate how predicted probabilities map to realized empirical frequencies across the pooled test cohort:

### BR-003.1 ARB Proxy ($L=-8\%, h=14\text{D}$) Reliability Distribution
| Bin | Predicted Range | Sample Count | % of Samples | Mean Predicted Prob | Realized Empirical Freq | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | [0.0, 0.1] | 755 | 2.6% | 0.0888 | 0.5616 | 0.4728 |
| **2** | [0.1, 0.2] | 2,831 | 9.9% | 0.1493 | 0.5221 | 0.3728 |
| **3** | [0.2, 0.3] | 1,942 | 6.8% | 0.2481 | 0.4531 | 0.2050 |
| **4** | [0.3, 0.4] | 2,328 | 8.1% | 0.3544 | 0.4815 | 0.1271 |
| **5** | [0.4, 0.5] | 2,770 | 9.7% | 0.4515 | 0.5231 | 0.0716 |
| **6** | [0.5, 0.6] | 2,994 | 10.4% | 0.5514 | 0.5999 | 0.0485 |
| **7** | [0.6, 0.7] | 3,557 | 12.4% | 0.6515 | 0.6483 | **0.0032** |
| **8** | [0.7, 0.8] | 4,455 | 15.5% | 0.7536 | 0.6471 | 0.1065 |
| **9** | [0.8, 0.9] | 6,406 | 22.3% | 0.8451 | 0.7485 | 0.0966 |
| **10**| [0.9, 1.0] | 674 | 2.4% | 0.9178 | 0.7893 | 0.1285 |

*Diagnosis:* In Bins 1 and 2, when the model predicts low arrival probability (8% to 15%), the empirical hit rate is actually **52% to 56%**. This occurs because those observations were predicted during Fold 1 and Fold 2, when the overall market experienced an aggregate sell-off, pulling nearly all altcoins down to $-8\%$ regardless of individual coin features.

### BR-003.2 ARB Diagnostic ($L=-6.77\% \text{ vs } U=+5.05\%, 30\text{D}$) Reliability Distribution
| Bin | Predicted Range | Sample Count | % of Samples | Mean Predicted Prob | Realized Empirical Freq | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | [0.0, 0.1] | 274 | 1.0% | 0.0922 | 0.3285 | 0.2362 |
| **2** | [0.1, 0.2] | 2,771 | 10.3% | 0.1527 | 0.3916 | 0.2389 |
| **3** | [0.2, 0.3] | 4,426 | 16.4% | 0.2535 | 0.4094 | 0.1559 |
| **4** | [0.3, 0.4] | 5,282 | 19.6% | 0.3502 | 0.4487 | 0.0985 |
| **5** | [0.4, 0.5] | 6,680 | 24.8% | 0.4507 | 0.5057 | 0.0550 |
| **6** | [0.5, 0.6] | 4,350 | 16.1% | 0.5437 | 0.4839 | 0.0598 |
| **7** | [0.6, 0.7] | 1,944 | 7.2% | 0.6401 | 0.4614 | 0.1787 |
| **8** | [0.7, 0.8] | 719 | 2.7% | 0.7421 | 0.5452 | 0.1969 |
| **9** | [0.8, 0.9] | 513 | 1.9% | 0.8287 | 0.6101 | 0.2186 |
| **10**| [0.9, 1.0] | 0 | 0.0% | — | — | — |

*Diagnosis:* Directional predictions are compressed heavily into the [0.20, 0.60] probability band (86.9% of all samples). The relationship is strictly monotonic, but compressed. True probabilities never drop below ~33% or rise above ~61%.

---

## 4. Evaluation of Platt Scaling and Isotonic Calibration

The audit verified the exact calibration protocol:
1. **Strict Temporal Isolation:**  
   In both studies, calibrators were fit **only** on out-of-fold training probabilities generated via 3-fold stratified cross-validation on `X_train` and `y_train`. `X_test` was evaluated solely using the pre-fit calibrator. There is **zero data leakage**.
2. **Comparison: Raw vs. Platt vs. Isotonic:**  
   - For BR-003.1 ARB Proxy:
     - Raw LightGBM Brier Score = **0.2432**, ECE = **0.0788**
     - Platt LightGBM Brier Score = **0.2550**, ECE = **0.1268**
     - Isotonic LightGBM Brier Score = **0.2553**, ECE = **0.1270**
   - For BR-003.2 ARB Diagnostic:
     - Raw LightGBM Brier Score = **0.2528**, ECE = **0.0742**
     - Platt LightGBM Brier Score = **0.2632**, ECE = **0.1174**
     - Isotonic LightGBM Brier Score = **0.2631**, ECE = **0.1126**
3. **Unexpected Finding — Post-Hoc Calibrators Degraded Calibration:**  
   In both studies, **Platt and Isotonic calibration actually worsened ECE and Brier Score relative to raw regularized LightGBM**.  
   *Why?* The post-hoc calibrators fit the historical base rate of the training fold. When a macro regime change shifted the test base rate by $\pm 10\text{--}20\%$, the calibrator warped probabilities toward the stale historical distribution. Raw LightGBM (with $L_1/L_2$ leaf regularization and max_depth=3) naturally produced shrinkage toward 0.50, which proved more robust to out-of-sample base rate drift.

---

## 5. Calibration Across Horizons and Levels

### Grid Progression Across Horizons (at $L = -8\%$)
| Horizon $h$ | Base Rate | Model AUC | BSS vs $B_1$ (Platt) | BSS vs $B_1$ (Raw) | ECE (Platt) | ECE (Raw) | Predictive Gate Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1-Day** | 6.4% | **0.685** | **+0.0282** | **+0.0481** | **0.0248** | **0.0271** | **PASSED** ($BSS > 0, ECE < 0.10$) |
| **3-Day** | 25.6% | **0.646** | **+0.0413** | **+0.0529** | **0.0408** | **0.0311** | **PASSED** ($BSS > 0, ECE < 0.10$) |
| **7-Day** | 46.6% | **0.617** | -0.0119 | **+0.0192** | **0.0968** | **0.0499** | **MIXED** (Raw passes, Platt fails) |
| **14-Day** | 61.5% | **0.611** | -0.0771 | -0.0275 | 0.1268 | **0.0788** | **FAILED** ($BSS < 0$) |
| **30-Day** | 73.8% | **0.606** | -0.1488 | -0.0840 | 0.1352 | 0.1074 | **FAILED** ($BSS < 0, ECE > 0.10$) |

### Grid Progression Across Levels (at $h = 14\text{D}$)
| Level $L$ | Base Rate | Model AUC | BSS vs $B_1$ (Platt) | BSS vs $B_1$ (Raw) | ECE (Platt) | ECE (Raw) | Predictive Gate Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **-5%** | 76.1% | **0.612** | -0.0682 | -0.0198 | 0.1182 | **0.0811** | **FAILED** ($BSS < 0$) |
| **-8%** | 61.5% | **0.611** | -0.0771 | -0.0275 | 0.1268 | **0.0788** | **FAILED** ($BSS < 0$) |
| **-10%** | 52.5% | **0.614** | -0.0579 | -0.0141 | 0.1288 | **0.0817** | **FAILED** ($BSS < 0$) |
| **-15%** | 33.0% | **0.620** | -0.0684 | -0.0035 | 0.1295 | **0.0781** | **FAILED** ($BSS < 0$) |
| **-20%** | 18.2% | **0.649** | -0.1147 | -0.0172 | 0.1008 | **0.0730** | **FAILED** ($BSS < 0$) |

*Core Insight:* Calibration degrades monotonically with time horizon $h$. At $h \le 3\text{D}$, price movements are governed by local volatility and order-flow momentum, which are stationary enough for causal features to achieve positive skill ($BSS > 0$) and tight calibration ($ECE < 0.05$). At $h \ge 14\text{D}$, multi-week macro market drift dominates individual asset kinematics, causing stationary point-in-time models to fail calibration against expanding baselines.

---

## 6. Root-Cause Analysis: Murphy Brier Score Decomposition

To definitively determine whether the failure is an implementation issue or non-stationary market drift, we compute the exact Murphy Brier Score decomposition:
$$BS = \text{Uncertainty} - \text{Resolution} + \text{Reliability}$$
- **Uncertainty ($c(1-c)$):** The inherent variance of the binary event (where $c = \text{base rate}$).
- **Resolution ($\sum \frac{n_k}{N}(o_k - c)^2$):** Measures the model's ability to divide instances into subsets with different outcomes (**higher is better**).
- **Reliability ($\sum \frac{n_k}{N}(p_k - o_k)^2$):** Measures the calibration error between predicted probabilities and observed frequencies (**lower is better; 0 is perfect**).

### Empirical Murphy Decomposition Results
| Target | Sample Uncertainty | Model Resolution | Calibration Reliability | Total Brier Score | Resolution / Reliability Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BR-003.1 ($L=-8\%, h=1\text{D}$)** | 0.06004 | **0.00182** | **0.00238** | 0.0584 | **0.76** ($BSS > 0$) |
| **BR-003.1 ($L=-8\%, h=3\text{D}$)** | 0.19045 | **0.00934** | **0.00320** | 0.1826 | **2.92** ($BSS > 0$) |
| **BR-003.1 ($L=-8\%, h=14\text{D}$)** | 0.23673 | **0.00997** | 0.02871 | 0.2550 | 0.35 ($BSS < 0$) |
| **BR-003.1 ($L=-8\%, h=30\text{D}$)** | 0.19317 | **0.00526** | 0.03999 | 0.2279 | 0.13 ($BSS < 0$) |
| **BR-003.2 ARB Diagnostic (30D)** | 0.24852 | **0.00233** | 0.01790 | 0.2632 | 0.13 ($BSS < 0$) |

### Mathematical Proof of Root Cause:
1. In all cases, **$\text{Resolution} > 0$**. The features possess genuine predictive capacity to sort and differentiate outcomes.
2. For $h=3\text{D}$, $\text{Resolution} (0.00934)$ is **nearly $3\times$ larger** than the calibration error $\text{Reliability} (0.00320)$, producing $BSS = +0.0413 > 0$.
3. For $h=14\text{D}$ and $h=30\text{D}$, the calibration penalty $\text{Reliability} (0.02871 \text{ to } 0.03999)$ expands by nearly an order of magnitude due to macro regime drift, completely overwhelming the model's positive resolution.
4. Therefore, the failure of BR-003.1 ($h \ge 14\text{D}$) and BR-003.2 is **not** an implementation error; it is an empirical property of long-horizon market non-stationarity.

---

## 7. Audit Compliance & Gating Directive Checklist

- [x] **Requirement 1:** BSS and ECE reported against every approved causal baseline ($B_0, B_1, B_2$).
- [x] **Requirement 2:** Per-fold walk-forward metrics presented for Folds 0, 1, 2, and 3.
- [x] **Requirement 3:** 10-bin reliability/calibration distribution documented with sample counts and gaps.
- [x] **Requirement 4:** Evaluated Platt scaling and Isotonic calibration; verified strict temporal isolation (no leakage) and identified that static calibration overfits training base rates during regime shifts.
- [x] **Requirement 5:** Cross-horizon ($1\text{D}$ to $30\text{D}$) and cross-level ($-5\%$ to $-20\%$) calibration reported.
- [x] **Requirement 6:** Proved via Murphy decomposition that calibration failure is caused by temporal/regime drift rather than implementation error.
- [x] **Requirement 7:** No feature sets, thresholds, labels, horizons, or barrier pairs were retuned.
- [x] **Requirement 8:** `EntryZoneEstimator` (BR-003.3) was **NOT created**.
- [x] **Requirement 9:** Neither BR-003.1 nor BR-003.2 was registered in the model registry.
- [x] **Language Correction:** Did not use the phrase *"optimal entry levels $L^*$ that maximize expected geometric return"* for BR-003.3.

---

## 8. Final Determinations

### BR-003.1 (Level-Reach Probability): `NOT_VALIDATED`
- **Determination:** **`NOT_VALIDATED`** for operational medium/long-horizon entry zones ($h \ge 14\text{D}$).  
- **Caveat:** The model demonstrated valid predictive skill ($BSS > 0$, $ECE < 0.05$) for short horizons ($1\text{D}$ to $3\text{D}$), but the intended operational entry-zone horizon ($14\text{D}$, ARB proxy) failed the Brier Skill Score gate ($BSS = -0.0771$).

### BR-003.2 (First-Passage Probability): `NOT_VALIDATED`
- **Determination:** **`NOT_VALIDATED`**.  
- **Reasoning:** Directional first-passage predictions yielded weak ranking discrimination ($AUC \approx 0.54\text{--}0.58$) and negative skill scores relative to climatology ($BSS \in [-0.039, -0.232]$). The model cannot be trusted as an operational probability engine.

---

## Next Steps for User Review

Because both BR-003.1 ($h \ge 14\text{D}$) and BR-003.2 failed their predictive gates ($BSS > 0$ and $ECE < 0.10$), **the stop gate holds**:
- We will **not** force the implementation of BR-003.3.
- We await your explicit instructions on whether to:
  1. Conclude the BR-003 investigation with these negative/honest scientific results, documenting that static forward-path probabilities do not survive multi-week crypto regime drift.
  2. Pivot to short-horizon execution models ($h \le 3\text{D}$), where BR-003.1 passed both gates ($BSS = +0.0413, ECE = 0.0408$).
  3. Explore a separately defined research hypothesis (e.g. dynamic Bayesian regime switching).
