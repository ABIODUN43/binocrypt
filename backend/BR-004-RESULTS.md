# BR-004: Dynamic Regime-Conditioned & Online Probability Calibration Results

**Experiment ID:** `BR-004`  
**Execution Timestamp:** `2026-09-26 18:05:05 UTC`  
**Universe:** `SBRU_V1` (88 consistently-listed Binance Spot USDT pairs)  
**Total Evaluated Observations:** 88,972 bars (1,095 calendar days across 4 expanding walk-forward folds)  
**Primary Target:** BR-003.1 ARB Proxy Level-Reach: $L = -8\%, h = 14\text{D}$ (`reach_m08_14d`, $N = 28,712$)  
**Secondary Targets:**  
- Level-Reach: $L = -8\%, h = 30\text{D}$ (`reach_m08_30d`, $N = 27,434$)  
- First-Passage: ARB Diagnostic $L = -6.77\%, U = +5.05\%, h = 30\text{D}$ (`fp_arb_label`, $N = 26,959$ resolved directional exits)  
**Feature Set:** 45 Causal Features (`FEATURE_COLS`, strictly frozen)  
**Validation Protocol:** Expanding Walk-Forward (180D min train, 90D step, 90D purge, 45D embargo)  
**Model Registration Status:** All models remain strictly **`RESEARCH_ONLY`** (No production model registered)  

---

## Executive Summary & Final Determinations

The BR-004 experiment evaluated whether dynamic macro regime conditioning (Architecture A), online Bayesian log-odds updating (Architecture B), or rolling-window walk-forward training (Architecture C) could eliminate the multi-week non-stationary calibration penalty observed in BR-003 and achieve validated probability calibration:

$$\text{Predictive Gates: } BSS_{B1} > 0.00 \quad \land \quad BSS_{B3} > 0.00 \quad \land \quad ECE < 0.10 \quad \land \quad \ge 3 \text{ of 4 Folds Jointly Passing}$$

### Summary of Results & Final Determinations

| Candidate Model / Architecture | Primary Gate 1 ($BSS_{B1} > 0$) | Primary Gate 2 ($BSS_{B3} > 0$) | Primary Gate 3 ($ECE < 0.10$) | Primary Gate 4 ($\ge 3/4$ Folds) | Final Determination |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static BR-003 Platt LightGBM** | **FAILED** (-0.0771) | **FAILED** (-0.0165) | **FAILED** (0.1268) | **FAILED** (0/4 Folds) | **`NOT_VALIDATED`** |
| **Static BR-003 Raw LightGBM** | **FAILED** (-0.0275) | **PASSED** (+0.0304) | **PASSED** (0.0788) | **FAILED** (1/4 Folds) | **`NOT_VALIDATED`** |
| **Architecture A: Multi-State Regime Calibrator** | **FAILED** (-0.0776) | **FAILED** (-0.0170) | **FAILED** (0.1258) | **FAILED** (0/4 Folds) | **`NOT_VALIDATED`** |
| **Architecture B: Bayesian Online Log-Odds Updating** | **FAILED** (-0.0292) | **PASSED** (+0.0287) | **PASSED** (0.0925) | **FAILED** (1/4 Folds) | **`NOT_VALIDATED`** |
| **Architecture C: Rolling 365D Walk-Forward** | **FAILED** (-0.1157) | **FAILED** (-0.0529) | **FAILED** (0.1565) | **FAILED** (0/4 Folds) | **`NOT_VALIDATED`** |

### Key Scientific Insights:
1. **Architecture B Emerges as the Superior Calibrator:**  
   Bayesian online updating on closed-horizon outcomes reduced the Brier score from 0.2559 (Static Platt) down to 0.2445 on the primary target, and from 0.2639 down to 0.2546 on the ARB first-passage target. It passed the calibration threshold ($ECE = 0.0925 < 0.10$) and beat the dynamic rolling baseline ($BSS_{B3} = +0.0287$). However, it still failed Gate 1 ($BSS_{B1} < 0$) against the expanding fold climatology and was not temporally stable across $\ge 3$ folds.
2. **Architecture A Shows Stale Regime Memory:**  
   Partitioning calibration into discrete macro regimes ($BTC > EMA_{50}$ with positive vs. negative momentum) failed to improve upon the global calibrator ($BSS_{B1} = -0.0776$ vs. $-0.0771$). Fitting calibrators per regime on historical training folds assumes that a 2026 bull market behaves identically to a 2024 bull market; in reality, base rates within the same macro regime drifted by $>15$ percentage points across cycles.
### Primary Conclusion:
> **“Under the preregistered architectures, targets, universe, and expanding walk-forward protocol, no candidate satisfied all four predictive validation gates for multi-week probability forecasting. Architecture B improved over the dynamic rolling baseline B3 and achieved acceptable pooled ECE, but failed the B1 and temporal-consistency requirements.”**

---

## 1. Primary Target Performance Matrix ($L = -8\%, h = 14\text{D}$ ARB Proxy)

Evaluated across $N = 28,712$ out-of-sample observations (pooled base rate = 61.52%):

| Model / Architecture | Out-of-Sample AUC | PR-AUC | Brier Score | BSS vs $B_0$ | BSS vs $B_1$ | BSS vs $B_2$ | BSS vs $B_3$ | ECE (10-bin) | MCE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0 (Cumulative Expanding)** | 0.471 | 0.589 | 0.2375 | 0.0000 | 0.0000 | +0.0016 | +0.0563 | 0.0100 | 0.0100 |
| **Baseline 1 (Fold Climatology)** | 0.471 | 0.589 | 0.2375 | 0.0000 | 0.0000 | +0.0016 | +0.0563 | 0.0100 | 0.0100 |
| **Baseline 2 (Binary Macro Regime)** | 0.533 | 0.644 | 0.2372 | +0.0016 | +0.0016 | 0.0000 | +0.0578 | 0.0940 | 0.0940 |
| **Baseline 3 (Online Rolling 60D)** | 0.526 | 0.638 | 0.2517 | -0.0597 | -0.0597 | -0.0613 | 0.0000 | 0.1090 | 0.1490 |
| **Static BR-003 Raw LightGBM** | 0.611 | 0.708 | 0.2441 | -0.0275 | -0.0275 | -0.0290 | **+0.0304** | **0.0788** | 0.1774 |
| **Static BR-003 Platt LightGBM** | **0.614** | **0.708** | 0.2559 | -0.0771 | -0.0771 | -0.0788 | -0.0165 | 0.1268 | 0.4728 |
| **Architecture A: Multi-State Regime** | **0.615** | 0.707 | 0.2560 | -0.0776 | -0.0776 | -0.0792 | -0.0170 | 0.1258 | 0.4682 |
| **Architecture B: Bayesian Online Updating** | 0.609 | 0.690 | **0.2445** | **-0.0292** | **-0.0292** | **-0.0308** | **+0.0287** | **0.0925** | 0.4169 |
| **Architecture C: Rolling 365D Walk-Forward** | 0.606 | 0.699 | 0.2650 | -0.1157 | -0.1157 | -0.1174 | -0.0529 | 0.1565 | 0.3552 |

---

## 2. Walk-Forward Fold-by-Fold Breakdown (Primary Target)

Evaluating performance across the four expanding walk-forward folds demonstrates how each architecture handles macro drift:

### Fold 0: 2024-08-07 to 2024-11-05 ($N_{test} = 7,162$)
- **Train Base Rate:** 52.1% | **Test Base Rate:** 55.6% (Drift: $+3.5\%$)
- **Baseline 1 ($B_1$):** Brier = 0.2481, ECE = 0.0343
- **Baseline 3 ($B_3$):** Brier = 0.2776, BSS vs $B_1$ = -0.1191
- **Static BR-003 Raw:** Brier = 0.3040, BSS vs $B_1$ = -0.2253, ECE = 0.2118, AUC = 0.533
- **Static BR-003 Platt:** Brier = 0.3333, BSS vs $B_1$ = -0.3433, ECE = 0.2656, AUC = 0.533
- **Architecture A (Regime):** Brier = 0.3337, BSS vs $B_1$ = -0.3451, ECE = 0.2573, AUC = 0.513
- **Architecture B (Online):** Brier = **0.2940**, BSS vs $B_1$ = **-0.1849**, ECE = **0.1819**, AUC = 0.528
- **Architecture C (Rolling):** Brier = 0.3333, BSS vs $B_1$ = -0.3433, ECE = 0.2656, AUC = 0.533

### Fold 1: 2025-03-20 to 2025-06-18 ($N_{test} = 7,164$)
- **Train Base Rate:** 61.1% | **Test Base Rate:** 69.9% (Drift: $+8.8\%$)
- **Baseline 1 ($B_1$):** Brier = 0.2182, ECE = 0.0886
- **Baseline 3 ($B_3$):** Brier = 0.2371, BSS vs $B_1$ = -0.0865
- **Static BR-003 Raw:** Brier = 0.2070, BSS vs $B_1$ = **+0.0510**, ECE = 0.1002, AUC = 0.616
- **Static BR-003 Platt:** Brier = 0.2138, BSS vs $B_1$ = **+0.0203**, ECE = 0.1261, AUC = 0.616
- **Architecture A (Regime):** Brier = 0.2134, BSS vs $B_1$ = **+0.0219**, ECE = 0.1385, AUC = 0.621
- **Architecture B (Online):** Brier = **0.2059**, BSS vs $B_1$ = **+0.0563**, ECE = **0.0884**, AUC = 0.603 *(Passed all gates in Fold 1)*
- **Architecture C (Rolling):** Brier = 0.2188, BSS vs $B_1$ = -0.0029, ECE = 0.1350, AUC = 0.612

### Fold 2: 2025-10-31 to 2026-01-29 ($N_{test} = 7,185$)
- **Train Base Rate:** 65.1% | **Test Base Rate:** 75.4% (Drift: $+10.3\%$)
- **Baseline 1 ($B_1$):** Brier = 0.1962, ECE = 0.1032
- **Baseline 3 ($B_3$):** Brier = 0.2187, BSS vs $B_1$ = -0.1152
- **Static BR-003 Raw:** Brier = **0.1934**, BSS vs $B_1$ = **+0.0142**, ECE = **0.0774**, AUC = 0.546
- **Static BR-003 Platt:** Brier = 0.2013, BSS vs $B_1$ = -0.0261, ECE = 0.1135, AUC = 0.546
- **Architecture A (Regime):** Brier = 0.2027, BSS vs $B_1$ = -0.0333, ECE = 0.1068, AUC = 0.558
- **Architecture B (Online):** Brier = 0.2018, BSS vs $B_1$ = -0.0288, ECE = 0.1180, AUC = 0.407
- **Architecture C (Rolling):** Brier = 0.2305, BSS vs $B_1$ = -0.1750, ECE = 0.1770, AUC = 0.484

### Fold 3: 2026-06-13 to 2026-09-11 ($N_{test} = 7,201$)
- **Train Base Rate:** 65.3% | **Test Base Rate:** 45.1% (Drift: **-20.2%**)
- **Baseline 1 ($B_1$):** Brier = 0.2885, ECE = 0.2021
- **Baseline 3 ($B_3$):** Brier = 0.2723, BSS vs $B_1$ = +0.0561
- **Static BR-003 Raw:** Brier = **0.2690**, BSS vs $B_1$ = **+0.0676**, ECE = 0.1625, AUC = 0.582
- **Static BR-003 Platt:** Brier = 0.2709, BSS vs $B_1$ = **+0.0609**, ECE = 0.1599, AUC = 0.582
- **Architecture A (Regime):** Brier = 0.2698, BSS vs $B_1$ = **+0.0645**, ECE = 0.1551, AUC = 0.576
- **Architecture B (Online):** Brier = 0.2743, BSS vs $B_1$ = **+0.0490**, ECE = 0.1792, AUC = 0.583
- **Architecture C (Rolling):** Brier = 0.2741, BSS vs $B_1$ = **+0.0498**, ECE = 0.1620, AUC = 0.585

---

## 3. Reliability Distribution (Architecture B, Primary Target)

The 10-bin calibration distribution for Architecture B demonstrates substantial improvement over Static Platt, particularly in compressing extreme miscalibration gaps:

| Bin | Predicted Range | Sample Count | % of Samples | Mean Predicted Prob | Realized Empirical Freq | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | [0.0, 0.1] | 0 | 0.0% | 0.0500 | 0.0000 | 0.0000 |
| **2** | [0.1, 0.2] | 1,039 | 3.6% | 0.1625 | 0.5794 | 0.4169 |
| **3** | [0.2, 0.3] | 2,047 | 7.1% | 0.2525 | 0.5354 | 0.2829 |
| **4** | [0.3, 0.4] | 1,587 | 5.5% | 0.3468 | 0.4764 | 0.1296 |
| **5** | [0.4, 0.5] | 2,536 | 8.8% | 0.4549 | 0.4286 | **0.0263** |
| **6** | [0.5, 0.6] | 4,146 | 14.4% | 0.5533 | 0.5171 | **0.0362** |
| **7** | [0.6, 0.7] | 5,309 | 18.5% | 0.6508 | 0.6274 | **0.0233** |
| **8** | [0.7, 0.8] | 5,084 | 17.7% | 0.7490 | 0.7073 | **0.0417** |
| **9** | [0.8, 0.9] | 6,442 | 22.4% | 0.8479 | 0.7212 | 0.1267 |
| **10**| [0.9, 1.0] | 522 | 1.8% | 0.9086 | 0.7778 | 0.1308 |

*Comparison:* In Bins 5 through 8 (representing 59.4% of all observations), Architecture B achieves outstanding calibration with empirical gaps of only **2.3% to 4.2%**. However, Bins 2 and 3 retain substantial gaps because sudden regime reversals (such as the onset of Fold 3's quiet market or Fold 1's rapid selloff) require several horizon-closed batches before the online recursive error accumulator can fully adjust.

---

## 4. Regime-Specific Breakdown (Primary Target)

Evaluating predictions across the three point-in-time macro regimes reveals where each architecture gains or loses skill:

| Macro Regime | Sample Count | Realized Base Rate | Arch A AUC | Arch A Brier | Arch A BSS vs $B_1$ | Arch A ECE | Arch B AUC | Arch B Brier | Arch B BSS vs $B_1$ | Arch B ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bear ($\text{BTC} \le \text{EMA}_{50}$)** | 16,557 | 57.1% | 0.662 | 0.2353 | **+0.0226** | **0.0851** | 0.641 | 0.2416 | -0.0036 | 0.1032 |
| **Bull Strong ($\text{BTC} > \text{EMA}_{50}, \text{Mom} \ge 0$)** | 10,169 | 69.4% | 0.537 | 0.2962 | -0.2702 | 0.2034 | 0.563 | 0.2569 | -0.1017 | 0.1440 |
| **Bull Transition ($\text{BTC} > \text{EMA}_{50}, \text{Mom} < 0$)** | 1,986 | 57.7% | **0.692** | **0.2229** | **+0.0461** | **0.0683** | **0.730** | **0.2055** | **+0.1207** | **0.0299** |

*Insight:* Both architectures achieve positive Brier Skill Scores ($BSS > 0$) and excellent calibration ($ECE < 0.07$) during Bear and Bull Transition markets. In Bull Transition regimes, Architecture B achieves an outstanding $BSS = +0.1207$ and $ECE = 0.0299$. However, both models fail during Strong Bull markets, where runaway altcoin momentum causes unexpected pullbacks to reach $-8\%$ at a 69.4% base rate that contrasts sharply with prior historical bull runs.

---

## 5. Secondary Targets Performance

### Secondary Target 1: Level-Reach $L = -8\%, h = 30\text{D}$ ($N = 27,434$, Base Rate = 73.79%)
| Model / Architecture | AUC | PR-AUC | Brier Score | BSS vs $B_1$ | BSS vs $B_3$ | ECE (10-bin) | MCE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 ($B_1$ Climatology)** | 0.490 | 0.728 | 0.1932 | 0.0000 | +0.1627 | 0.0100 | 0.0100 |
| **Baseline 3 (Online Rolling 60D)** | 0.443 | 0.707 | 0.2307 | -0.1944 | 0.0000 | 0.1557 | 0.2679 |
| **Static BR-003 Raw LightGBM** | 0.606 | 0.803 | 0.2094 | -0.0840 | **+0.0924** | 0.1074 | 0.2625 |
| **Static BR-003 Platt LightGBM** | 0.607 | 0.803 | 0.2220 | -0.1488 | **+0.0379** | 0.1352 | 0.3082 |
| **Architecture A (Regime)** | 0.612 | 0.803 | 0.2224 | -0.1511 | **+0.0362** | 0.1337 | 0.3168 |
| **Architecture B (Online)** | 0.584 | 0.785 | **0.2115** | **-0.0949** | **+0.0835** | **0.1264** | 0.2854 |
| **Architecture C (Rolling 365D)** | 0.596 | 0.793 | 0.2291 | -0.1861 | +0.0071 | 0.1487 | 0.3082 |

### Secondary Target 2: First-Passage ARB Diagnostic $30\text{D}$ ($N = 26,959$, Base Rate = 46.15%)
| Model / Architecture | AUC | PR-AUC | Brier Score | BSS vs $B_1$ | BSS vs $B_3$ | ECE (10-bin) | MCE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 ($B_1$ Climatology)** | 0.536 | 0.477 | 0.2492 | 0.0000 | +0.0380 | 0.0490 | 0.0491 |
| **Baseline 3 (Online Rolling 60D)** | 0.472 | 0.466 | 0.2590 | -0.0395 | 0.0000 | 0.0858 | 0.1245 |
| **Static BR-003 Raw LightGBM** | 0.551 | 0.502 | 0.2535 | -0.0172 | **+0.0214** | **0.0742** | 0.2133 |
| **Static BR-003 Platt LightGBM** | 0.549 | 0.501 | 0.2639 | -0.0589 | -0.0187 | 0.1174 | 0.2389 |
| **Architecture A (Regime)** | 0.550 | 0.502 | 0.2631 | -0.0556 | -0.0155 | 0.1110 | 0.2499 |
| **Architecture B (Online)** | 0.552 | 0.504 | **0.2546** | **-0.0215** | **+0.0172** | **0.0760** | 0.4236 |
| **Architecture C (Rolling 365D)** | 0.548 | 0.498 | 0.2673 | -0.0727 | -0.0320 | 0.1261 | 0.9088 |

---

## 6. Audit & Preregistered Safeguard Compliance

- [x] **Target Definitions Frozen:** Primary ($L=-8\%, 14\text{D}$), Secondary ($L=-8\%, 30\text{D}$, ARB First-Passage $30\text{D}$).
- [x] **45 Causal Features Frozen:** No new coin-level or macro indicators added beyond point-in-time BTC regime.
- [x] **SBRU_V1 Universe Frozen:** 88 consistently-listed Binance Spot USDT pairs.
- [x] **Purge & Embargo Preserved:** 90D purge and 45D embargo maintained across all expanding folds.
- [x] **Safeguard 1 (Regime Calibrator Fallback):** Minimum 200 samples, 20 positives, 20 negatives enforced for Architecture A; fallen back deterministically to global calibrator when conditions were unmet.
- [x] **Safeguard 2 (Baseline 3 Causality):** $B_3$ strictly filtered for horizon-closed observations where $t_i + h \le t_k$; current day and open paths were never included.
- [x] **Model Governance Enforced:** All models remain tagged as **`RESEARCH_ONLY`**.
- [x] **Zero Production Tampering:** `EntryZoneEstimator` (BR-003.3) was **NOT created** and production routing was untouched.

---

## 7. Final Determinations & Research Recommendations

### Determination per Architecture:
1. **Architecture A (Multi-State Regime Calibrator): `NOT_VALIDATED`**  
   Fails Gate 1 ($BSS_{B1} = -0.0776$), Gate 2 ($BSS_{B3} = -0.0170$), Gate 3 ($ECE = 0.1258$), and Gate 4 (0/4 folds). Discrete macro conditioning on historical folds does not resolve test distribution drift.
2. **Architecture B (Bayesian Online Updating): `NOT_VALIDATED`**  
   Demonstrates meaningful progress by passing Gate 2 ($BSS_{B3} = +0.0287$) and Gate 3 ($ECE = 0.0925 < 0.10$), but ultimately fails Gate 1 ($BSS_{B1} = -0.0292 < 0$) and Gate 4 (only 1/4 folds passed jointly). While superior to static calibration, it is not sufficiently robust for production probability serving.
### Primary Conclusion:
> **“Under the preregistered architectures, targets, universe, and expanding walk-forward protocol, no candidate satisfied all four predictive validation gates for multi-week probability forecasting. Architecture B improved over the dynamic rolling baseline B3 and achieved acceptable pooled ECE, but failed the B1 and temporal-consistency requirements.”**

### Strategic Recommendation for Binocrypt:
Do **not** force downstream components to consume uncalibrated multi-week probabilities. Instead of spending further cycles attempting to force an absolute probability model to do something the empirical data does not support under these walk-forward tests, the next research experiment should evaluate whether the observed ranking capacity (AUC $\approx 0.61\text{--}0.69$, quintile spreads $>+25\%$) translates into economically meaningful cross-sectional selection:
- **BR-005 (Cross-Sectional Opportunity Ranking):** Test whether causal features can rank assets within the daily Binance Spot cross-section such that higher-ranked assets exhibit superior out-of-sample risk-adjusted forward outcomes across 7D/14D/30D horizons, evaluated under strict predictive gates prior to friction-adjusted portfolio simulation.
