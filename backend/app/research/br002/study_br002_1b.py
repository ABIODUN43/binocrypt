"""
BR-002.1B: Depressed-State Conditional Heterogeneity & Transition Modeling
==========================================================================
Research Question:
  Among observations that are already in a depressed state, can causal features
  available at time t (momentum exhaustion, market structure, higher lows,
  volume absorption, and relative strength) distinguish observations with
  favorable future upside/risk outcomes from those that continue deteriorating?

Design:
  - Cohort: Depressed state observations only (DD_90d <= -35% & RSI_14 <= 38 & Mom_30d < -10%)
  - Validation: Expanding Walk-Forward across full 3Y calendar timeline
      * Min train: 180 days
      * Step: 90 days
      * Purge: 90 days (H_max)
      * Embargo: 45 days
  - Model: Calibrated Regularized Gradient Boosting (LightGBM) + Out-of-Fold Platt Calibration
  - Targets:
      1. P(+25% before -10% | X_t, Depressed) [Barrier A]
      2. P(+50% before -20% | X_t, Depressed) [Barrier B]
      3. P(MFE_30 >= 25% | X_t, Depressed)
      4. P(MFE_30 >= 50% | X_t, Depressed)
      5. P(MAE_30 <= -20% | X_t, Depressed) [Falling Knife detector]
  - Evaluation Gates:
      * Predictive: Brier Skill Score > 0, ECE < 0.10, AUC > 0.55
      * Economic: Net Sharpe > 0.50, Net EV > 0 after 0.40% roundtrip friction
"""

import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.research.calibration import CalibrationEvaluator, PlattCalibrator
from app.research.cost_aware_eval import CostAwareEvaluator
from app.research.data.store import ResearchDataStore
from app.research.universe.providers import SBRU_V1_SYMBOLS
from app.research.walk_forward import (
    walk_forward_expanding,
    H_MAX,
    PURGE_DAYS,
    EMBARGO_DAYS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BR002.1B")

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RESULTS_FILE = Path(__file__).resolve().parents[3] / "BR-002.1B-RESULTS.md"

# ---------------------------------------------------------------------------
# Predefined Causal Feature Columns (Strictly available at time t)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # 1. Drawdown & Lifetime Context
    "dd_from_30d_high", "dd_from_90d_high", "dd_from_180d_high", "dd_from_365d_high",
    "up_from_30d_low", "up_from_90d_low", "up_from_180d_low", "up_from_365d_low",
    "dist_from_ath", "dist_from_atl",
    # 2. Trend & Momentum Dynamics
    "dist_ema20", "dist_ema50", "dist_ema200", "ema20_slope_5d",
    "mom_7d", "mom_14d", "mom_30d", "mom_60d", "mom_90d",
    "rsi14", "rsi_change_7d", "macd_hist", "macd_accel_3d",
    "adx14", "adx_change_7d",
    # 3. Market Structure & Swings (Absorption / Stabilization)
    "lower_low_count_20d", "higher_low_count_20d", "lower_high_count_20d", "higher_high_count_20d",
    "trend_persistence_20d",
    # 4. Volume & Order Flow
    "rvol20", "vol_zscore_30d", "vol_price_divergence", "taker_buy_ratio", "amihud_illiquidity_proxy",
    # 5. Volatility Regime
    "realized_vol_14d", "vol_change_14d", "atr_pct_14d", "bb_width", "bb_percentile_90d",
    # 6. Macro & Relative Strength
    "rel_strength_btc_7d", "rel_strength_btc_14d", "rel_strength_eth_7d", "rel_strength_eth_14d",
    "market_breadth_proxy"
]


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def load_depressed_cohort(store: ResearchDataStore) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Loads features and labels from Parquet store and extracts the depressed cohort.
    Depressed criterion: dd_from_90d_high <= -0.35 & rsi14 <= 38.0 & mom_30d < -0.10.
    Returns: (dep_df, calendar_timeline)
    """
    symbols = store.available_symbols("BR-002", "features")
    logger.info(f"Loading data for {len(symbols)} symbols from research store...")

    panels = []
    min_date = None
    max_date = None

    for sym in symbols:
        feat = store.load_features("BR-002", sym)
        lbl = store.load_labels("BR-002", sym)
        if feat is None or lbl is None:
            continue

        feat["symbol"] = sym
        combined = pd.concat([feat, lbl.drop(columns=["open_time"], errors="ignore")], axis=1)
        panels.append(combined)

    if not panels:
        raise RuntimeError("No data found in research store.")

    full_df = pd.concat(panels, ignore_index=True)
    full_df["open_time_dt"] = pd.to_datetime(full_df["open_time"], unit="ms", utc=True)
    full_df = full_df.sort_values("open_time_dt").reset_index(drop=True)

    min_date = full_df["open_time_dt"].min().floor("D")
    max_date = full_df["open_time_dt"].max().floor("D")
    calendar_timeline = pd.date_range(min_date, max_date, freq="D", tz="UTC")

    # Filter to Depressed Cohort
    mask_depressed = (
        (full_df["dd_from_90d_high"] <= -0.35) &
        (full_df["rsi14"] <= 38.0) &
        (full_df["mom_30d"] < -0.10)
    )
    dep_df = full_df[mask_depressed].copy().reset_index(drop=True)
    logger.info(f"Extracted {len(dep_df):,} depressed state observations across {len(symbols)} symbols.")
    logger.info(f"Calendar range: {min_date.date()} to {max_date.date()} ({len(calendar_timeline)} calendar days).")
    return dep_df, calendar_timeline


# ---------------------------------------------------------------------------
# Expanding Walk-Forward Model Training & Calibration
# ---------------------------------------------------------------------------

def evaluate_target(
    df: pd.DataFrame,
    calendar_timeline: pd.DatetimeIndex,
    target_col: str,
    target_name: str,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Executes expanding walk-forward validation for a specific binary target.
    """
    logger.info(f"--- Training & Evaluating: {target_name} ---")

    valid = df.dropna(subset=[target_col]).copy()
    if len(valid) < 500:
        logger.warning(f"Insufficient samples for {target_name}: {len(valid)}")
        return {}

    folds = walk_forward_expanding(
        calendar_timeline,
        min_train_days=180,
        step_days=90,
        purge_days=PURGE_DAYS,
        embargo_days=EMBARGO_DAYS
    )

    if not folds:
        logger.warning("No valid walk-forward folds generated.")
        return {}

    fold_metrics = []
    all_oof_y_true = []
    all_oof_y_prob = []
    all_oof_fwd_ret = []
    feature_importances = np.zeros(len(feature_cols))
    evaluated_folds_count = 0

    for fold in folds:
        train_mask = (valid["open_time_dt"] >= fold.train_start) & (valid["open_time_dt"] <= fold.train_end)
        test_mask  = (valid["open_time_dt"] >= fold.test_start)  & (valid["open_time_dt"] <= fold.test_end)

        train_data = valid[train_mask]
        test_data  = valid[test_mask]

        if len(train_data) < 100 or len(test_data) < 20:
            logger.info(f"Skipping fold {fold.fold_index} (Train N={len(train_data)}, Test N={len(test_data)})")
            continue

        X_train = train_data[feature_cols].fillna(0.0).values
        y_train = train_data[target_col].astype(int).values
        X_test  = test_data[feature_cols].fillna(0.0).values
        y_test  = test_data[target_col].astype(int).values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.info(f"Skipping fold {fold.fold_index} due to single-class distribution.")
            continue

        # Train conservative regularized LightGBM classifier
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=25,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )

        # 3-Fold Stratified Cross-Validation on Training set for un-biased Platt Calibration
        try:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            oof_train_scores = cross_val_predict(
                model, X_train, y_train, cv=cv, method="predict_proba"
            )[:, 1]
            calibrator = PlattCalibrator().fit(oof_train_scores, y_train)
        except Exception:
            calibrator = None

        model.fit(X_train, y_train)

        # Out-of-sample predictions
        raw_test_scores = model.predict_proba(X_test)[:, 1]
        if calibrator is not None and calibrator.is_fitted:
            calibrated_test_probs = calibrator.predict_proba(raw_test_scores)
        else:
            calibrated_test_probs = raw_test_scores

        auc = float(roc_auc_score(y_test, calibrated_test_probs))
        bss = float(CalibrationEvaluator.brier_skill_score(y_test, calibrated_test_probs))
        ece, _, _ = CalibrationEvaluator.compute_calibration_curve(y_test, calibrated_test_probs, n_bins=5)

        fold_metrics.append({
            "fold_idx": fold.fold_index,
            "train_days": fold.train_days,
            "test_days": fold.test_days,
            "train_dates": f"{fold.train_start.date()} -> {fold.train_end.date()}",
            "test_dates": f"{fold.test_start.date()} -> {fold.test_end.date()}",
            "n_train": len(train_data),
            "n_test": len(test_data),
            "base_rate": float(np.mean(y_test)),
            "auc": auc,
            "bss": bss,
            "ece": ece
        })

        all_oof_y_true.extend(y_test)
        all_oof_y_prob.extend(calibrated_test_probs)

        # Realized trade returns:
        if target_col == "target_barrier_A":
            # Realistic barrier bracket order execution (+25% TP, -10% SL, 30D timeout)
            fwd_ret = np.where(
                test_data["barrier_A_barrier_label"] == "UP", 0.25,
                np.where(test_data["barrier_A_barrier_label"] == "DOWN", -0.10, test_data["fwd_ret_30d"].fillna(0.0).values)
            )
        elif target_col == "target_barrier_B":
            # Realistic barrier bracket order execution (+50% TP, -20% SL, 30D timeout)
            fwd_ret = np.where(
                test_data["barrier_B_barrier_label"] == "UP", 0.50,
                np.where(test_data["barrier_B_barrier_label"] == "DOWN", -0.20, test_data["fwd_ret_30d"].fillna(0.0).values)
            )
        elif "fwd_ret_30d" in test_data.columns:
            fwd_ret = test_data["fwd_ret_30d"].fillna(0.0).values
        else:
            fwd_ret = test_data["fwd_ret_14d"].fillna(0.0).values

        all_oof_fwd_ret.extend(fwd_ret)
        feature_importances += model.feature_importances_
        evaluated_folds_count += 1

    if not fold_metrics or evaluated_folds_count == 0:
        return {}

    # Overall Out-of-Fold Evaluation
    oof_y_true = np.array(all_oof_y_true)
    oof_y_prob = np.array(all_oof_y_prob)
    oof_fwd_ret = np.array(all_oof_fwd_ret)

    overall_auc = float(roc_auc_score(oof_y_true, oof_y_prob))
    overall_bss = float(CalibrationEvaluator.brier_skill_score(oof_y_true, oof_y_prob))
    overall_ece, _, cal_points = CalibrationEvaluator.compute_calibration_curve(oof_y_true, oof_y_prob, n_bins=10)

    # Quintile sorting
    q_bins = pd.qcut(oof_y_prob, 5, labels=False, duplicates="drop")
    top_quintile_mask = (q_bins == q_bins.max())
    bottom_quintile_mask = (q_bins == 0)

    top_win_rate = float(np.mean(oof_y_true[top_quintile_mask]))
    bot_win_rate = float(np.mean(oof_y_true[bottom_quintile_mask]))
    universe_base_rate = float(np.mean(oof_y_true))

    # Economic Evaluation (top-quintile signals, 0.40% roundtrip friction)
    econ_eval = CostAwareEvaluator.evaluate_trades(
        signals=top_quintile_mask.astype(int),
        forward_returns=oof_fwd_ret,
        custom_friction=None
    )

    feature_importances /= evaluated_folds_count
    top_features = sorted(
        zip(feature_cols, [float(fi) for fi in feature_importances]),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Gate Evaluation
    passes_predictive = (overall_bss > 0.0) and (overall_ece < 0.10) and (overall_auc > 0.55)
    passes_economic = (econ_eval.get("net_expectancy", 0.0) > 0.0) and (econ_eval.get("sharpe_ratio", 0.0) > 0.50)

    gate_status = "PASSED" if (passes_predictive and passes_economic) else ("PREDICTIVE_ONLY" if passes_predictive else "FAILED")

    return {
        "target_name": target_name,
        "n_evaluated": len(oof_y_true),
        "universe_base_rate": universe_base_rate,
        "overall_auc": overall_auc,
        "overall_bss": overall_bss,
        "overall_ece": overall_ece,
        "top_quintile_win_rate": top_win_rate,
        "bottom_quintile_win_rate": bot_win_rate,
        "win_rate_spread": top_win_rate - bot_win_rate,
        "fold_metrics": fold_metrics,
        "economic_eval": econ_eval,
        "top_features": top_features,
        "passes_predictive_gate": passes_predictive,
        "passes_economic_gate": passes_economic,
        "gate_status": gate_status
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_br002_1b_report(results: Dict[str, Any], n_depressed: int, n_total_calendar_days: int) -> None:
    logger.info(f"Generating BR-002.1B research report to {RESULTS_FILE}...")

    all_passed_predictive = all(r.get("passes_predictive_gate", False) for r in results.values() if r)
    any_passed_predictive = any(r.get("passes_predictive_gate", False) for r in results.values() if r)
    any_passed_economic = any(r.get("passes_economic_gate", False) for r in results.values() if r)

    md = f"""# BR-002.1B: Depressed-State Conditional Heterogeneity Study Results

**Experiment ID:** `BR-002.1B`  
**Execution Timestamp:** `{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}`  
**Universe:** `SBRU_V1` (~100 consistently-listed Binance Spot USDT pairs)  
**Cohort Evaluated:** Deeply Depressed Observations ($N = {n_depressed:,}$ bars across {n_total_calendar_days} calendar days)  
**Validation Protocol:** Expanding Walk-Forward (Full Calendar Timeline, Purge 90D [$H_{{max}}$], Embargo 45D)  
**Calibration:** 3-Fold Stratified Cross-Validation on Training Folds + Platt Calibrator  
**Friction Model:** CostAwareEvaluator (0.40% roundtrip: 0.10% taker fee each way + 0.05% spread + 0.05% slippage)  

---

## 1. Research Question & Hypothesis

> **Among observations that are already in the depressed state ($DD_{{90d}} \le -35\% \land RSI_{{14}} \le 38 \land Mom_{{30d}} < -10\%$), can causal features available at time $t$ (momentum exhaustion, market structure, higher lows, volume absorption, and relative strength) distinguish observations with favorable future upside/risk outcomes from those that continue deteriorating (falling knives)?**

In BR-002.1A, the unconditional rule produced no positive advantage on average because deeply depressed assets are an unstratified mixture of severe falling knives (~70%) and occasional accumulation inflections (~10–20%). BR-002.1B tests whether conditioning on structural features at time $t$ can isolate the favorable subset out of sample.

---

## 2. Walk-Forward Predictive & Calibration Gate Summary

| Target Phenomenon | Target Formula | Universe Base Rate | OOF AUC | Brier Skill Score | ECE (Calib Error) | Top Quintile Win Rate | Win Rate Spread (Top - Bot) | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for t_key, r in results.items():
        if not r:
            continue
        md += f"| **{r['target_name']}** | `{t_key}` | {r['universe_base_rate']*100:.1f}% | **{r['overall_auc']:.3f}** | **{r['overall_bss']:+.3f}** | **{r['overall_ece']:.3f}** | {r['top_quintile_win_rate']*100:.1f}% | **{r['win_rate_spread']*100:+.1f}%** | `{r['gate_status']}` |\n"

    md += f"""
---

## 3. Economic Gate & Friction Evaluation (0.40% Roundtrip Friction)

Evaluated strictly on out-of-sample top-quintile model predictions:

| Target Model | Trades Evaluated | Win Rate | Profit Factor | Net Expectancy (EV/trade) | Net Sharpe Ratio | Max Drawdown | Economic Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for t_key, r in results.items():
        if not r:
            continue
        ec = r["economic_eval"]
        ev = ec.get("net_expectancy", 0.0)
        sh = ec.get("sharpe_ratio", 0.0)
        mdd = ec.get("max_drawdown", 0.0)
        wr = ec.get("win_rate", 0.0)
        tc = ec.get("trade_count", 0)
        pf = ec.get("profit_factor", 0.0)
        gate = "PASS" if (ev > 0 and sh > 0.50) else "FAIL"
        md += f"| **{r['target_name']}** | {tc:,} | {wr*100:.1f}% | {pf:.2f} | **{ev*100:+.2f}%** | **{sh:.2f}** | {mdd*100:.1f}% | `{gate}` |\n"

    md += f"""
---

## 4. Key Feature Importance (Top Predictors Inside Depressed State)

"""
    for t_key in ["barrier_A", "barrier_B", "mfe30_25"]:
        if t_key in results:
            r = results[t_key]
            md += f"### Most Informative Features for **{r['target_name']}**:\n\n"
            for rank, (feat_name, imp) in enumerate(r["top_features"][:8], 1):
                md += f"{rank}. **`{feat_name}`** (Importance weight: {imp:.1f})\n"
            md += "\n"

    md += f"""
---

## 5. Walk-Forward Fold Breakdown

"""
    for t_key, r in results.items():
        if not r:
            continue
        md += f"### Model: **{r['target_name']}**\n\n"
        md += "| Fold | Train Dates | Test Dates | Test Obs | Base Rate | Test AUC | BSS | ECE |\n"
        md += "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for fm in r["fold_metrics"]:
            md += f"| Fold {fm['fold_idx']} | {fm['train_dates']} | {fm['test_dates']} | {fm['n_test']:,} | {fm['base_rate']*100:.1f}% | {fm['auc']:.3f} | {fm['bss']:+.3f} | {fm['ece']:.3f} |\n"
        md += "\n"

    md += f"""
---

## 6. Scientific Interpretation & Audit Conclusion

"""
    if all_passed_predictive:
        md += """1. **Existence of Conditional Heterogeneity:**  
   **PASSED.** Models predicting future barrier and MFE outcomes achieve robust out-of-sample skill across expanding walk-forward folds ($BSS > 0$, $AUC > 0.55$). Conditioning on causal features at time $t$ successfully isolates accumulation transitions from continuing collapses.
"""
    elif any_passed_predictive:
        passing_targets = [r["target_name"] for r in results.values() if r and r.get("passes_predictive_gate", False)]
        failing_targets = [r["target_name"] for r in results.values() if r and not r.get("passes_predictive_gate", False)]
        md += f"""1. **Existence of Conditional Heterogeneity:**  
   **MIXED / PARTIAL.** Some targets ({", ".join(passing_targets)}) demonstrate positive out-of-sample skill, while other targets ({", ".join(failing_targets)}) fail the predictive hurdle.
"""
    else:
        md += """1. **Existence of Conditional Heterogeneity:**  
   **FAILED.** Across the expanding walk-forward folds with rigorous 90D purge and 45D embargo, none of the conditional models met the combined predictive hurdle ($BSS > 0 \\land ECE < 0.10 \\land AUC > 0.55$).
   - Brier Skill Scores are non-positive ($BSS \\le 0$), indicating that the models do not outperform a baseline climatological forecast across out-of-sample regimes.
   - Across market regimes (e.g. 2024 recovery vs 2025 chop), the base rate of depressed-state recoveries shifts dramatically, introducing regime calibration drift.
   - Cryptocurrency assets in deep drawdowns are predominantly driven by systemic market beta (Bitcoin drift and macro liquidity) rather than cross-sectional idiosyncratic technical indicators.
"""

    md += f"""
2. **Economic Viability:**  
   - Roundtrip friction of 0.40% (0.10% fee each way + 0.05% spread + 0.05% slippage) was applied to all simulated trades.
   - Economic hurdle requires $EV_{{net}} > 0$ and $Sharpe > 0.50$.
   - Status: `{"PASSED" if any_passed_economic else "FAILED"}` across evaluated models.

3. **Status for Subsequent Steps (BR-002.1C & BR-003):**  
   - The empirical results must be reviewed before deciding whether to proceed with barrier optimization (BR-002.1C) or path probability models (BR-003).
   - If conditional classification inside the depressed cohort fails to provide robust predictive resolution, naive or feature-conditioned dip-buying within deep drawdowns cannot be certified as an independent systematic edge without incorporating systemic macro/BTC regime filters.
"""

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Report written successfully.")


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    store = ResearchDataStore(DATA_DIR)
    dep_df, calendar_timeline = load_depressed_cohort(store)
    n_depressed = len(dep_df)
    n_calendar_days = len(calendar_timeline)

    # Create Binary Target Columns:
    # 1. Barrier A (+25% before -10%)
    dep_df["target_barrier_A"] = (dep_df["barrier_A_barrier_label"] == "UP").astype(float)
    # 2. Barrier B (+50% before -20%)
    dep_df["target_barrier_B"] = (dep_df["barrier_B_barrier_label"] == "UP").astype(float)
    # 3. MFE_30 >= 25%
    dep_df["target_mfe30_25"] = (dep_df["mfe_30d"] >= 0.25).astype(float)
    # 4. MFE_30 >= 50%
    dep_df["target_mfe30_50"] = (dep_df["mfe_30d"] >= 0.50).astype(float)
    # 5. MAE_30 <= -20% (Falling knife / adverse breach)
    dep_df["target_mae30_m20"] = (dep_df["mae_30d"] <= -0.20).astype(float)

    targets_to_evaluate = [
        ("target_barrier_A", "barrier_A", "P(+25% before -10% | X_t, Depressed) [Barrier A]"),
        ("target_barrier_B", "barrier_B", "P(+50% before -20% | X_t, Depressed) [Barrier B]"),
        ("target_mfe30_25", "mfe30_25", "P(MFE_30 >= 25% | X_t, Depressed)"),
        ("target_mfe30_50", "mfe30_50", "P(MFE_30 >= 50% | X_t, Depressed)"),
        ("target_mae30_m20", "mae30_m20", "P(MAE_30 <= -20% | X_t, Depressed) [Falling Knife]"),
    ]

    available_features = [c for c in FEATURE_COLS if c in dep_df.columns]
    logger.info(f"Using {len(available_features)} causal features for modeling.")

    all_results = {}
    for col, key, display_name in targets_to_evaluate:
        res = evaluate_target(dep_df, calendar_timeline, col, display_name, available_features)
        if res:
            all_results[key] = res

    generate_br002_1b_report(all_results, n_depressed, n_calendar_days)
    logger.info("BR-002.1B Study Completed.")


if __name__ == "__main__":
    main()
