# BR-007 Research Proposal: Swing Reversal & Bottom-Formation Intelligence

**Proposal Identifier:** `BR-007-PROPOSAL`  
**Date:** `2026-10-20`  
**Status:** `PREREGISTERED & FROZEN FOR REVIEW` (Research Proposal — Zero Code Modifications Permitted Prior to Formal Approval)  
**Parent Studies:** `BR-002` (Depression Dynamics), `BR-003` (Level-Reach), `BR-004` (Calibration Audit), `BR-005` (LambdaRank Cross-Sectional Ranking), `BR-006` (Prospective Validation Pilot)  
**Research Focus:** Causal detection of downside exhaustion, bottom-formation stabilization, calibrated recovery estimation, and confirmation-gated swing entry under strict execution friction  

---

## 1. Executive Summary & Motivation

### 1.1 The Practical Problem
A recurring challenge in cryptocurrency spot trading—vividly illustrated by recent market behavior in declining assets such as **ARB/USDT**—is distinguishing between:
1. **A terminal falling knife** (a continuing selloff that cascades through multiple support levels with high momentum and liquidity voids);
2. **A recoverable swing bottom** (an asset experiencing seller exhaustion, volume absorption, structural base formation, and eventual mean-reverting recovery).

Traders frequently attempt to "pick the exact bottom" or rely on simplistic technical indicators (e.g., RSI $< 30$ or moving-average touches). In **BR-002.1A**, Binocrypt proved empirically that unconditional dip-buying heuristics exhibit **zero statistical edge** ($IC \approx 0.00$) and suffer devastating tail risk. Furthermore, the **Binocrypt Forecast Intelligence Audit** revealed that naive volatility cones centered on live spot prices float continuously downward during severe selloffs, offering no true bottom-prediction capability.

### 1.2 The BR-007 Strategic Objective
BR-007 investigates whether **point-in-time causal features** can:
1. Detect developing **downside exhaustion** during severe asset declines;
2. Differentiate continuing liquidations from stabilizing **base-formation** patterns;
3. Emit **calibrated probability estimates** that an asset will achieve a predefined upside recovery threshold ($U$) before hitting a protective downside failure barrier ($D$) within a fixed swing horizon ($H$);
4. Evaluate whether requiring **structural confirmation** justifies the cost of entering above the absolute lowest print;
5. Determine whether a risk-controlled swing trading policy achieves **positive net expectancy** after realistic fees, bid-ask spread, and execution friction.

### 1.3 Core Scientific Precaution: No "Exact Bottom" Fallacy
BR-007 explicitly rejects the claim that any quantitative system can predict the exact final tick or low of a selloff in real time. The proposed architecture separates **exhaustion detection** from **confirmation of recovery**, explicitly requiring structural market proof before proposing an actionable entry.

---

## 2. Formal Research Questions & Hypotheses

To prevent data dredging and post-hoc rationalization, BR-007 preregisters five decoupled research questions:

### Research Question 1 (Downside Exhaustion Discrimination):
> *Can strictly causal, point-in-time features distinguish a continuing selloff from an exhaustion state where immediate downside momentum has statistically decelerated?*
* **Null Hypothesis ($H_{0,1}$):** Downside forward return distribution conditional on exhaustion detection is identical to the unconditional selloff distribution ($AUC \le 0.50$, Kolmogorov-Smirnov $p \ge 0.05$).
* **Alternative Hypothesis ($H_{A,1}$):** Exhaustion-flagged observations exhibit statistically significant reduction in forward downside tail risk ($AUC \ge 0.60, p < 0.01$).

### Research Question 2 (Calibrated Event-Based Recovery Estimation):
> *Can a probabilistic classifier estimate the likelihood that price reaches a predefined upside recovery barrier ($+U$) before hitting an additional downside failure barrier ($-D$) over swing horizon $H$ with superior calibration to climatological base rates?*
* **Null Hypothesis ($H_{0,2}$):** The model's Brier Skill Score relative to unconditional expanding base rates is non-positive ($BSS_{B0} \le 0.00$).
* **Alternative Hypothesis ($H_{A,2}$):** The model demonstrates true probability forecasting skill ($BSS_{B0} > 0.05$, Expected Calibration Error $ECE \le 0.08$).

### Research Question 3 (Value of Confirmation vs. Bottom-Fishing):
> *Does waiting for causal structural confirmation (e.g., higher-low formation, swing resistance reclaim) yield superior risk-adjusted swing outcomes compared to unconfirmed limit orders placed at potential-bottom exhaustion prints?*
* **Null Hypothesis ($H_{0,3}$):** Unconfirmed entries at exhaustion prints have identical or superior Sharpe/expectancy compared to confirmed entries after accounting for execution price difference.
* **Alternative Hypothesis ($H_{A,3}$):** Confirmation filters out a sufficient fraction of catastrophic continuation trades to deliver higher net Sharpe ratio and lower Maximum Adverse Excursion (MAE), even though entry price is higher.

### Research Question 4 (Cross-Asset & Regime Robustness):
> *Does the exhaustion/recovery signal maintain positive skill across diverse asset volatility profiles, macro Bitcoin trend regimes, and liquidity environments, or is it concentrated in bull-market pullbacks?*
* **Null Hypothesis ($H_{0,4}$):** Predictive skill is statistically indistinguishable from zero during BTC bearish macro regimes (BTC Close $<$ 50-day EMA).
* **Alternative Hypothesis ($H_{A,4}$):** Recovery predictions maintain statistically significant discrimination ($AUC \ge 0.58$) across both bullish and bearish macro regimes.

### Research Question 5 (Economic Friction Survival):
> *Does an end-to-end swing trading policy based on confirmed recovery signals generate positive net expectancy and a Sharpe ratio $\ge 1.00$ after 0.40% round-trip execution costs?*
* **Null Hypothesis ($H_{0,5}$):** Net Sharpe ratio after friction $< 1.00$ or net expectancy $\le 0.00\%$.
* **Alternative Hypothesis ($H_{A,5}$):** Net Sharpe ratio $\ge 1.00$, Calmar ratio $\ge 0.70$, and net expectancy $> 0.50\%$ per trade.

---

## 3. Finite State Machine: Decoupling Detection from Confirmation

To prevent ambiguous semantics, BR-007 structures asset trajectory into six mutually exclusive, causally determined states:

```
                      ┌──────────────────────┐
                      │      DECLINING       │ (Active downtrend: fresh lower lows)
                      └──────────┬───────────┘
                                 │ Exhaustion Criteria Met (Momentum decel, vol absorption)
                                 ▼
                      ┌──────────────────────┐
                      │ DOWNSIDE_EXHAUSTION_ │
                      │        WATCH         │
                      └──────────┬───────────┘
                                 │ Stabilization (Causal swing low held for N bars)
                                 ▼
                      ┌──────────────────────┐
                      │     BASE_FORMING     │ (Range boundary established; volatility compression)
                      └────┬────────────┬────┘
      Higher Low / Breakout│            │ Failure (Fresh low broken: Close < Swing Low)
                           ▼            ▼
             ┌───────────────────┐  ┌───────────────────┐
             │REVERSAL_CONFIRMED │  │  FAILED_REVERSAL  │──► [Returns to DECLINING]
             └─────────┬─────────┘  └───────────────────┘
                       │ Price reaches +U before -D
                       ▼
             ┌───────────────────┐
             │     RECOVERY      │ (Target Achieved)
             └───────────────────┘
```

### 3.1 Formal State Definitions & Transition Rules

All transition criteria are strictly causal and evaluated using only closed bars at time $t$:

1. **`DECLINING`:**
   * *Definition:* Asset is in an active selloff.
   * *Causal Entry Condition:* $P_t < \text{EMA}_{50}(t)$ and $P_t$ has made a new 20-bar low within the past 5 bars.
2. **`DOWNSIDE_EXHAUSTION_WATCH`:**
   * *Definition:* Downside momentum is decelerating, but price has not yet established a floor.
   * *Causal Entry Condition:* 
     1. Asset is in `DECLINING` state;
     2. Normalized Drawdown from 20-bar high exceeds $3.0 \times \text{ATR}_{14}(t)$;
     3. Downside deceleration: $|\Delta P_{t, t-3}| < |\Delta P_{t-3, t-6}|$ and RSI(14) slope is positive over the last 3 bars;
     4. Volume absorption: Volume in the lowest quartile of the bar exceeds the 20-bar volume average.
3. **`BASE_FORMING`:**
   * *Definition:* A provisional swing low has formed and held without making fresh lows for a minimum stabilization period.
   * *Causal Entry Condition:*
     1. Asset was in `DOWNSIDE_EXHAUSTION_WATCH`;
     2. A causal pivot low $L_{\text{pivot}}$ occurred at bar $t - k$ ($3 \le k \le 12$);
     3. For all bars $\tau \in [t - k + 1, t]$, $\text{Low}_\tau \ge L_{\text{pivot}}$;
     4. Normalized trading range over the last $k$ bars has compressed ($\frac{\text{High}_{k} - \text{Low}_{k}}{\text{ATR}_{14}} \le 2.0$).
4. **`REVERSAL_CONFIRMED`:**
   * *Definition:* Price produces structural confirmation of upward transition. **Only this state permits trade generation.**
   * *Causal Entry Condition:*
     1. Asset was in `BASE_FORMING`;
     2. Price achieves one of two pre-specified confirmation rules:
        * *Rule A (Structural Higher Low Reclaim):* A secondary swing low forms where $L_{\text{secondary}} > L_{\text{pivot}}$, and price closes above the intermediate swing high.
        * *Rule B (Moving Average Reclaim):* Price closes above 1-hour $\text{EMA}_{20}$ with volume $> 1.25\times$ 20-bar average.
5. **`FAILED_REVERSAL`:**
   * *Definition:* The candidate base failed; downward liquidation resumed.
   * *Causal Entry Condition:*
     * If in `BASE_FORMING` or `REVERSAL_CONFIRMED`, $\text{Close}_t < L_{\text{pivot}}$.
     * State resets immediately to `DECLINING`.
6. **`RECOVERY`:**
   * *Definition:* Upside swing target ($+U$) achieved within horizon $H$.

---

## 4. Market-Data Infrastructure Audit & Timeframe Specification

### 4.1 Empirical Audit of Current Market-Data Infrastructure
A rigorous code inspection of the existing Binocrypt repository reveals:
* **Research Parquet Store (`backend/app/research/data/`):**
  * Downloads and maintains **only 1D (daily) klines** for the `SBRU_V1` 88-symbol universe ([`downloader.py#L49-L55`](file:///backend/app/research/data/downloader.py)).
  * Does **not** store or paginate intraday klines (4H, 1H, 15M, 5M).
* **Live Runtime API (`backend/app/services/market_data.py`):**
  * Implements dynamic REST fetching for `1h` and `4h` intervals with a limit of 100 bars ([`market_data.py#L233-L250`](file:///backend/app/services/market_data.py)).
  * Can route requests to Binance Spot (`api.binance.com`), Binance.US, and Bybit Spot.
* **15M and 5M Candlesticks:**
  * **Completely absent from both historical storage and runtime services.** There is zero caching, persistence, or backtesting support for 15M/5M bars in the current codebase.

### 4.2 Preregistered Timeframe Architecture for BR-007
Based on the empirical infrastructure audit, BR-007 establishes the following explicit hierarchy:

| Hierarchy Level | Timeframe | Role in BR-007 | Data Sourcing Protocol |
| :--- | :---: | :--- | :--- |
| **Macro Regime Context** | **1D** | Market trend, broad BTC regime, multi-day oversold context | Sourced from existing `SBRU_V1` 1D Parquet shards. |
| **Intermediate Setup** | **4H** | Multi-day swing support, volume profile, higher-timeframe exhaustion | To be added to research downloader (paginated 4H Parquet download). |
| **Primary Execution & Signal** | **1H** | Exhaustion detection, base formation, pivot tracking, confirmation | Primary bar for event-based labels and feature formation. |
| **Intraday Confirmation (Optional)**| **15M** | Fine-grained bar-level confirmation | **Explicit Constraint:** If 15M Parquet data is not downloaded, 1H bar close must be used as the definitive confirmation level. No phantom assumptions about 5M/15M data. |

---

## 5. Causal Feature Architecture (Signal Families)

Every candidate feature must be **strictly causal** and computable at the close of bar $t$. Retrospective peak/trough algorithms (such as ZigZag or future lookahead pivot identification) are strictly prohibited.

### Family 1: Downside Momentum Deceleration
1. `accel_3_6`: Ratio of recent price change to preceding change:
   $$\text{accel\_3\_6}_t = \frac{P_t - P_{t-3}}{P_{t-3} - P_{t-6} + \epsilon}$$
2. `atr_velocity`: 3-bar rate of change in True Range normalized by 20-bar ATR:
   $$\text{atr\_velocity}_t = \frac{\text{ATR}_{3}(t) - \text{ATR}_{14}(t)}{\text{ATR}_{14}(t)}$$
3. `lower_wick_ratio`: Buyer absorption at lows:
   $$\text{lower\_wick\_ratio}_t = \frac{\min(\text{Open}_t, \text{Close}_t) - \text{Low}_t}{\text{High}_t - \text{Low}_t + \epsilon}$$

### Family 2: ATR-Normalized Drawdown & Distance Metrics
4. `dd_norm_24h`: Drawdown from highest high of past 24 bars, normalized by $\text{ATR}_{14}$:
   $$\text{dd\_norm\_24h}_t = \frac{\max_{\tau \in [t-24, t]}(\text{High}_\tau) - P_t}{\text{ATR}_{14}(t)}$$
5. `dd_norm_72h`: Normalized drawdown from 72-bar (3-day) high.
6. `dist_to_causal_pivot`: Distance from most recent causal swing low ($L_{\text{pivot}}$):
   $$\text{dist\_to\_pivot}_t = \frac{P_t - L_{\text{pivot}}}{L_{\text{pivot}}}$$
   *(Where $L_{\text{pivot}}$ is confirmed only when a low is followed by at least 2 higher lows).*

### Family 3: Causal Momentum Divergence
7. `rsi_14`: Classical 14-period Relative Strength Index.
8. `causal_rsi_divergence`: Point-in-time regular bullish divergence:
   $$\text{rsi\_div}_t = \begin{cases} 1 & \text{if } P_t < P_{\text{prior\_pivot}} \text{ and } \text{RSI}_t > \text{RSI}_{\text{prior\_pivot}} \\ 0 & \text{otherwise} \end{cases}$$
   *(Both price pivot and RSI pivot must be established causally at least 2 bars prior).*
9. `macd_hist_slope`: 3-bar slope of the MACD histogram during negative readings.

### Family 4: Structural Base & Higher-Low Formation
10. `bars_since_lowest_low`: Number of bars elapsed since the minimum low of the trailing 48 bars.
11. `higher_low_count`: Count of consecutive local causal pivots satisfying $L_k > L_{k-1}$ over the past 30 bars.
12. `range_compression_ratio`: Ratio of current 6-bar range to trailing 24-bar range:
    $$\text{compression}_t = \frac{\max_{6}(\text{High}) - \min_{6}(\text{Low})}{\max_{24}(\text{High}) - \min_{24}(\text{Low}) + \epsilon}$$

### Family 5: Short-Term Resistance Reclaim
13. `ema20_dist`: Percentage distance from 1-hour $\text{EMA}_{20}$:
    $$\text{ema20\_dist}_t = \frac{P_t - \text{EMA}_{20}(t)}{\text{EMA}_{20}(t)}$$
14. `reclaim_ema20_bars`: Number of bars since price crossed from below to above $\text{EMA}_{20}$.
15. `reclaim_swing_high`: Binary indicator that $P_t$ has closed above the highest high of the preceding 5 bars.

### Family 6: Volume Absorption & Liquidity
16. `volume_exhaustion_ratio`: Ratio of volume on down-bars versus up-bars over the last 12 bars:
    $$\text{vol\_exhaustion}_t = \frac{\sum_{\tau=t-11}^t V_\tau \cdot \mathbb{I}_{\text{Close}_\tau < \text{Open}_\tau}}{\sum_{\tau=t-11}^t V_\tau \cdot \mathbb{I}_{\text{Close}_\tau \ge \text{Open}_\tau} + \epsilon}$$
17. `volume_zscore_at_low`: Volume z-score on the bar that established the recent pivot low.
18. `on_balance_volume_slope`: 5-bar linear regression slope of OBV.

### Family 7: Market Context & Relative Strength
19. `btc_trend_regime`: Macro regime filter ($\mathbb{I}_{\text{BTC Close} \ge \text{BTC EMA}_{50}}$).
20. `asset_rs_vs_btc`: 24-hour relative performance versus BTC:
    $$\text{rs\_vs\_btc}_t = \left(\frac{P_{i,t}}{P_{i,t-24}}\right) - \left(\frac{\text{BTC}_t}{\text{BTC}_{t-24}}\right)$$
21. `alt_breadth_above_ema20`: Fraction of the `SBRU_V1` universe trading above their respective 1-hour $\text{EMA}_{20}$.

---

## 6. Formal Event-Based Target Definition (Triple-Barrier Framework)

Rather than predicting ambiguous continuous price levels or uncalibrated multi-week endpoints, BR-007 defines an **event-based triple-barrier target** evaluated over a forward holding period from decision bar $t$.

### 6.1 Barrier Parameters
Let entry price be $P_{\text{entry}} = \text{Close}_t$ (or next-bar open $\text{Open}_{t+1}$ under causal execution).
Three barriers are evaluated simultaneously:
1. **Upper Barrier (Upside Recovery Target $+U$):**
   $$\text{Barrier}_{\text{upper}} = P_{\text{entry}} \times (1 + U)$$
2. **Lower Barrier (Downside Failure Stop $-D$):**
   $$\text{Barrier}_{\text{lower}} = P_{\text{entry}} \times (1 - D)$$
3. **Horizontal Barrier (Time Horizon $H$):**
   Barriers expire at bar $t + H$.

### 6.2 Pre-Registered Barrier Combinations
BR-007 freezes **three explicit barrier configurations** across the 1-hour resolution:

| Configuration | Upside Target ($+U$) | Downside Stop ($-D$) | Horizon ($H$) | Target Payoff Ratio ($U / D$) | Target Character |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Config A (Primary Swing)** | **$+6.0\%$** | **$-3.0\%$** | **72 Hours (3 Days)** | **2.00 : 1** | Standard swing recovery |
| **Config B (Quick Bounce)** | **$+3.5\%$** | **$-2.0\%$** | **24 Hours (1 Day)** | **1.75 : 1** | Immediate relief rally |
| **Config C (Extended Base)** | **$+10.0\%$** | **$-4.5\%$** | **168 Hours (7 Days)**| **2.22 : 1** | Multi-day structural turn |

### 6.3 Event Labeling Logic ($Y_{i, t}$)
For every observation $t$ where an asset enters `REVERSAL_CONFIRMED` (or `DOWNSIDE_EXHAUSTION_WATCH` for Phase 1 testing), future 1-hour bars $\tau \in [t+1, t+H]$ are inspected:

$$Y_{i, t} = \begin{cases} 
1 & \text{if } \text{High}_\tau \ge \text{Barrier}_{\text{upper}} \text{ before } \text{Low}_\tau \le \text{Barrier}_{\text{lower}} \\
0 & \text{if } \text{Low}_\tau \le \text{Barrier}_{\text{lower}} \text{ before } \text{High}_\tau \ge \text{Barrier}_{\text{upper}} \\
0 & \text{if neither barrier is touched before bar } t + H \text{ (Time Expiration)}
\end{cases}$$

### 6.4 Ambiguous Intrabar Resolution & Worst-Case Discipline
If within a single 1-hour bar $\tau$, **both** the upper and lower barriers are crossed ($\text{High}_\tau \ge \text{Barrier}_{\text{upper}}$ AND $\text{Low}_\tau \le \text{Barrier}_{\text{lower}}$):
* **Strict Worst-Case Assumption:** The lower barrier (stop-loss) is assumed to have been touched **first**.
* $$Y_{i, t} = 0$$
* This guarantees that reported performance is conservative and cannot be artificially inflated by favorable intrabar ordering assumptions.

---

## 7. Model Architecture & Calibration Protocol

### 7.1 Approved Baselines
To evaluate whether machine learning adds genuine predictive alpha:
1. **Baseline B0 (Unconditional Climatology):**
   Emits the historical expanding event rate from the training fold:
   $$\hat{p}_{B0} = \frac{1}{N_{\text{train}}} \sum_{j=1}^{N_{\text{train}}} Y_j$$
2. **Baseline B1 (Naive Oversold Bounce Rule):**
   Heuristic trigger based on classical indicators:
   $$\text{Trigger if } \text{RSI}_{14} < 30 \text{ and } P_t < \text{Lower Bollinger Band}$$
3. **Baseline B2 (Causal Moving Average Reclaim Rule):**
   Fixed heuristic rule: enter when $P_t$ closes above $\text{EMA}_{20}$ after being below for $\ge 12$ bars, with fixed $U/D$ targets.

### 7.2 Candidate Classification Models
1. **Model M1: Calibrated Regularized Logistic Regression ($L_2$ ElasticNet)**  
   Linear log-odds model with monotonic shrinkage.
2. **Model M2: Constrained LightGBM Classifier with Probability Calibration**  
   Tree depth $\le 3$, max leaves $\le 7$, min child samples $\ge 50$, regularized $L_1/L_2$.
   * **Probability Calibration Layer:** Raw model margins are calibrated out-of-sample using **Isotonic Regression** (if training samples $\ge 1,000$) or **Platt Sigmoid Scaling** fitted strictly on validation folds.

### 7.3 Calibration & Probability Quality Metrics
Models must be evaluated as probabilistic forecasts, not merely ranking scores:
* **Brier Score ($BS$):** Mean squared error of probability forecasts:
  $$BS = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - Y_i)^2$$
* **Brier Skill Score ($BSS_{B0}$):** Relative skill over the unconditional base rate:
  $$BSS_{B0} = 1 - \frac{BS_{\text{model}}}{BS_{B0}}$$
* **Expected Calibration Error ($ECE$):** Weighted difference between predicted confidence and empirical accuracy across 10 decile bins:
  $$ECE = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
* **Reliability Diagrams:** Visual inspection of empirical win rate versus predicted probability deciles.

---

## 8. Expanding Walk-Forward Evaluation Protocol

### 8.1 Chronological Splits & Leakage Prevention
To guarantee strict out-of-sample validity and prevent temporal lookahead:
* **Validation Structure:** 4 Chronological Expanding Folds.
* **Minimum Initial Training:** 180 calendar days.
* **Fold Test Duration:** 90 calendar days per fold.
* **Purge Window:** 7 calendar days ($168\text{h} = H_{\max}$) immediately preceding the test window to eliminate overlapping label dependency.
* **Embargo Window:** 3 calendar days immediately following the test fold.

```
Fold 1: [--- Train: 180 Days ---][Purge 7D][--- Test 1: 90 Days ---][Embargo 3D]
Fold 2: [------- Train: 270 Days -------][Purge 7D][--- Test 2: 90 Days ---][Embargo 3D]
Fold 3: [----------- Train: 360 Days -----------][Purge 7D][--- Test 3: 90 Days ---][Embargo 3D]
Fold 4: [--------------- Train: 450 Days ---------------][Purge 7D][--- Test 4: 90 Days ---]
```

### 8.2 Path-Dependent Execution Metrics
For every evaluated trade entry:
* **Maximum Adverse Excursion (MAE):** Maximum unrealized percentage loss experienced before target, stop, or horizon:
  $$\text{MAE} = \min_{\tau \in [t+1, t_{\text{exit}}]} \left( \frac{\text{Low}_\tau - P_{\text{entry}}}{P_{\text{entry}}} \right)$$
* **Maximum Favorable Excursion (MFE):** Maximum unrealized percentage gain experienced during trade:
  $$\text{MFE} = \max_{\tau \in [t+1, t_{\text{exit}}]} \left( \frac{\text{High}_\tau - P_{\text{entry}}}{P_{\text{entry}}} \right)$$
* **Time to Resolution ($T_{\text{res}}$):** Median number of hours from entry to barrier resolution.

---

## 9. Realistic Friction & Economic Gates

### 9.1 Transaction Friction Specification
All economic evaluations must incorporate:
* **Exchange Fee:** $0.10\%$ spot taker fee per leg ($0.20\%$ round-trip).
* **Execution Slippage:** $0.05\%$ per leg based on Binance Spot order-book dynamics ($0.10\%$ round-trip).
* **Buffer for Bid-Ask Spread:** $0.10\%$ round-trip.
* **Total Modeled Friction:** **$0.40\%$ round-trip** deducted from every simulated trade return.

### 9.2 Pre-Registered Validation Gates

A candidate model must pass **all four sequential gates** to be deemed scientifically validated:

```
┌─────────────────────────────────────────────────────────────┐
│ GATE 1: Statistical Predictive Discrimination Gate           │
│ - Primary Target Config A (6% / 3% / 72h):                  │
│   Out-of-sample AUC-ROC >= 0.62 (p < 0.01 under HAC)        │
│ - Precision-Recall AUC (PR-AUC) > Unconditional Base Rate   │
│ - Top-Decile Win Rate - Bottom-Decile Win Rate >= 20.0%     │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE 2: Probability Calibration & Brier Skill Gate          │
│ - Brier Skill Score over expanding base rate: BSS_B0 > 0.05 │
│ - Expected Calibration Error: ECE <= 0.08 (8.0%)            │
│ - Monotonicity: Empirical win rates strictly monotonic      │
│   across grouped probability terciles                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE 3: Economic Expectancy Gate (Post-Friction)            │
│ - Net Sharpe Ratio >= 1.00 (annualized from swing trades)   │
│ - Net Profit Factor >= 1.35                                 │
│ - Net Expectancy >= +0.60% per executed trade after 0.40%   │
│   round-trip fees and slippage                              │
│ - Calmar Ratio >= 0.70                                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ GATE 4: Regime & Temporal Robustness Gate                   │
│ - Positive net PnL in >= 3 of the 4 walk-forward folds      │
│ - Maximum Peak-to-Trough Strategy Drawdown <= 20.0%         │
│ - Positive net expectancy in both BTC Bull and BTC Bear     │
│   macro regimes (BTC Close >= EMA50 vs < EMA50)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Research Governance, Safety & Boundary Rules

1. **Strict Research-Only Status:**  
   All code, models, and data artifacts created under BR-007 are tagged `RESEARCH_ONLY`. No live trading endpoints or real-money accounts may execute BR-007 signals.
2. **Absolute Non-Interference with BR-006:**  
   The prospective paper-validation pilot for the frozen LightGBM LambdaRank model (`BR-006`) is live. Its frozen model artifact (`frozen_model.joblib`), registered SHA-256 hash, configuration, Snapshot 0, and bi-weekly observation schedule must remain **completely untouched and unmodified**.
3. **No Modification of Live Trading Engine:**  
   Production routing, `PaperTrader`, `SetupEngine`, and live database schemas will not be modified during BR-007 research.
4. **No "Guaranteed Bottom" Language:**  
   All output, logs, reports, and UI proposals must refer strictly to *"Probabilistic Downside Exhaustion"* and *"Confirmed Swing Reversal Setup"*. Language claiming "the bottom is in" or "final low reached" is strictly prohibited.

---

## 11. Implementation Roadmap (Post-Approval Only)

Upon formal user approval of this research proposal, implementation will proceed through the following phased steps:

```
[Phase 1: Data Infrastructure Extension]
  - Add paginated 1H and 4H Parquet downloaders for SBRU_V1 universe
  - Verify zero missing bars and strict timestamp alignment
       │
       ▼
[Phase 2: Causal Feature Extraction & Target Labeling]
  - Implement 21 point-in-time features without lookahead
  - Generate event-based triple-barrier labels (Configs A, B, C)
  - Verify worst-case intrabar collision logic
       │
       ▼
[Phase 3: Walk-Forward Model Training & Calibration]
  - Execute 4-fold expanding walk-forward splits
  - Train M1 (ElasticNet) and M2 (LightGBM) with Isotonic/Platt calibration
  - Calculate Brier scores, ECE, AUC, and reliability plots
       │
       ▼
[Phase 4: Confirmation Value Analysis & Economic Simulation]
  - Compare unconfirmed exhaustion entries vs. confirmed entries
  - Apply 0.40% friction model and evaluate Gates 1–4
  - Produce comprehensive results report: BR-007-RESULTS.md
```

---
*Proposal Preregistered & Submitted by: Antigravity Autonomous Quantitative Research System*
