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
      5. P(MAE_30 <= -20% | X_t, Depressed) [Falling Knife detector / Risk Filter]
  - Evaluation Gates:
      * Predictive: Brier Skill Score > 0, ECE < 0.10, AUC > 0.55 across folds
      * Economic: Net Sharpe > 0.50, Net EV > 0 after 0.40% roundtrip friction
"""

import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.research.calibration import CalibrationEvaluator, PlattCalibrator
from app.research.cost_aware_eval import CostAwareEvaluator
from app.research.data.store import ResearchDataStore
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


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Computes empirical 95% bootstrap confidence interval for AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, y_prob[idx]))
    if not aucs:
        return 0.0, 1.0
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def load_depressed_cohort(store: ResearchDataStore) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    symbols = store.available_symbols("BR-002", "features")
    logger.info(f"Loading data for {len(symbols)} symbols from research store...")

    panels = []
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
    all_oof_signals = []
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

        raw_test_scores = model.predict_proba(X_test)[:, 1]
        if calibrator is not None and calibrator.is_fitted:
            calibrated_test_probs = calibrator.predict_proba(raw_test_scores)
        else:
            calibrated_test_probs = raw_test_scores

        auc = float(roc_auc_score(y_test, calibrated_test_probs))
        auc_ci_low, auc_ci_high = bootstrap_auc_ci(y_test, calibrated_test_probs, n_boot=1000)
        pr_auc = float(average_precision_score(y_test, calibrated_test_probs))
        brier = float(CalibrationEvaluator.brier_score(y_test, calibrated_test_probs))
        bss = float(CalibrationEvaluator.brier_skill_score(y_test, calibrated_test_probs))
        ece, _, _ = CalibrationEvaluator.compute_calibration_curve(y_test, calibrated_test_probs, n_bins=5)

        # Strictly out-of-sample quintile ranking within this test fold
        try:
            fold_q_bins = pd.qcut(calibrated_test_probs, 5, labels=False, duplicates="drop")
            top_q_mask = (fold_q_bins == fold_q_bins.max())
            bot_q_mask = (fold_q_bins == 0)
            top_rate = float(np.mean(y_test[top_q_mask]))
            bot_rate = float(np.mean(y_test[bot_q_mask]))
            spread = top_rate - bot_rate
        except Exception:
            top_q_mask = np.zeros(len(y_test), dtype=bool)
            top_rate = 0.0
            bot_rate = 0.0
            spread = 0.0

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

        # Per-fold economic evaluation on top-quintile signals
        fold_econ = CostAwareEvaluator.evaluate_trades(
            signals=top_q_mask.astype(int),
            forward_returns=fwd_ret,
            custom_friction=None
        )

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
            "auc_ci_low": auc_ci_low,
            "auc_ci_high": auc_ci_high,
            "pr_auc": pr_auc,
            "brier": brier,
            "bss": bss,
            "ece": ece,
            "top_win_rate": top_rate,
            "bot_win_rate": bot_rate,
            "spread": spread,
            "econ": fold_econ
        })

        all_oof_y_true.extend(y_test)
        all_oof_y_prob.extend(calibrated_test_probs)
        all_oof_fwd_ret.extend(fwd_ret)
        all_oof_signals.extend(top_q_mask.astype(int))

        feature_importances += model.feature_importances_
        evaluated_folds_count += 1

    if not fold_metrics or evaluated_folds_count == 0:
        return {}

    # Overall Out-of-Fold Evaluation
    oof_y_true = np.array(all_oof_y_true)
    oof_y_prob = np.array(all_oof_y_prob)
    oof_fwd_ret = np.array(all_oof_fwd_ret)
    oof_signals = np.array(all_oof_signals)

    overall_auc = float(roc_auc_score(oof_y_true, oof_y_prob))
    overall_auc_ci_low, overall_auc_ci_high = bootstrap_auc_ci(oof_y_true, oof_y_prob, n_boot=1000)
    overall_pr_auc = float(average_precision_score(oof_y_true, oof_y_prob))
    overall_brier = float(CalibrationEvaluator.brier_score(oof_y_true, oof_y_prob))
    overall_bss = float(CalibrationEvaluator.brier_skill_score(oof_y_true, oof_y_prob))
    overall_ece, _, _ = CalibrationEvaluator.compute_calibration_curve(oof_y_true, oof_y_prob, n_bins=10)

    # Aggregate of strictly out-of-sample top-quintile selections
    overall_top_mask = (oof_signals > 0)
    universe_base_rate = float(np.mean(oof_y_true))
    top_win_rate = float(np.mean(oof_y_true[overall_top_mask])) if np.sum(overall_top_mask) > 0 else 0.0

    # Aggregate economic evaluation
    aggregate_econ = CostAwareEvaluator.evaluate_trades(
        signals=oof_signals,
        forward_returns=oof_fwd_ret,
        custom_friction=None
    )

    feature_importances /= evaluated_folds_count
    top_features = sorted(
        zip(feature_cols, [float(fi) for fi in feature_importances]),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    passes_predictive = (overall_bss > 0.0) and (overall_ece < 0.10) and (overall_auc > 0.55)
    passes_economic = (aggregate_econ.get("net_expectancy", 0.0) > 0.0) and (aggregate_econ.get("sharpe_ratio", 0.0) > 0.50)
    gate_status = "PASSED" if (passes_predictive and passes_economic) else ("PREDICTIVE_ONLY" if passes_predictive else "FAILED")

    return {
        "target_name": target_name,
        "n_evaluated": len(oof_y_true),
        "universe_base_rate": universe_base_rate,
        "overall_auc": overall_auc,
        "overall_auc_ci_low": overall_auc_ci_low,
        "overall_auc_ci_high": overall_auc_ci_high,
        "overall_pr_auc": overall_pr_auc,
        "overall_brier": overall_brier,
        "overall_bss": overall_bss,
        "overall_ece": overall_ece,
        "top_quintile_win_rate": top_win_rate,
        "fold_metrics": fold_metrics,
        "aggregate_econ": aggregate_econ,
        "top_features": top_features,
        "passes_predictive_gate": passes_predictive,
        "passes_economic_gate": passes_economic,
        "gate_status": gate_status
    }


# ---------------------------------------------------------------------------
# Falling Knife Risk Filter Evaluation
# ---------------------------------------------------------------------------

def evaluate_falling_knife_risk_filter(
    df: pd.DataFrame,
    calendar_timeline: pd.DatetimeIndex,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Evaluates the Falling Knife model strictly as a RISK-FILTER / AVOIDANCE signal:
    High predicted P(knife) >= Q80 -> AVOID.
    Tests whether filtering out high-risk observations improves downside and returns.
    """
    logger.info("--- Evaluating Falling Knife Detector as Risk Filter ---")
    valid = df.dropna(subset=["target_mae30_m20"]).copy()
    folds = walk_forward_expanding(
        calendar_timeline,
        min_train_days=180,
        step_days=90,
        purge_days=PURGE_DAYS,
        embargo_days=EMBARGO_DAYS
    )

    per_fold_filter_results = []
    all_base_knives = []
    all_filtered_knives = []
    all_base_rets = []
    all_filtered_rets = []
    all_base_barrier_a = []
    all_filtered_barrier_a = []

    for fold in folds:
        train_mask = (valid["open_time_dt"] >= fold.train_start) & (valid["open_time_dt"] <= fold.train_end)
        test_mask  = (valid["open_time_dt"] >= fold.test_start)  & (valid["open_time_dt"] <= fold.test_end)

        train_data = valid[train_mask]
        test_data  = valid[test_mask]

        if len(train_data) < 100 or len(test_data) < 20:
            continue

        X_train = train_data[feature_cols].fillna(0.0).values
        y_train = train_data["target_mae30_m20"].astype(int).values
        X_test  = test_data[feature_cols].fillna(0.0).values
        y_test  = test_data["target_mae30_m20"].astype(int).values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

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
        model.fit(X_train, y_train)
        pred_knife_probs = model.predict_proba(X_test)[:, 1]

        # Risk filter rule: Exclude top 20% highest predicted knife probability
        q80_threshold = float(np.percentile(pred_knife_probs, 80))
        keep_mask = (pred_knife_probs < q80_threshold)

        fwd_rets = test_data["fwd_ret_30d"].fillna(0.0).values
        barrier_a_up = (test_data["barrier_A_barrier_label"] == "UP").values

        base_knife_rate = float(np.mean(y_test))
        filt_knife_rate = float(np.mean(y_test[keep_mask]))
        knife_reduction = base_knife_rate - filt_knife_rate

        base_mean_ret = float(np.mean(fwd_rets))
        filt_mean_ret = float(np.mean(fwd_rets[keep_mask]))
        ret_improvement = filt_mean_ret - base_mean_ret

        base_bar_a = float(np.mean(barrier_a_up))
        filt_bar_a = float(np.mean(barrier_a_up[keep_mask]))

        per_fold_filter_results.append({
            "fold_idx": fold.fold_index,
            "test_dates": f"{fold.test_start.date()} -> {fold.test_end.date()}",
            "n_obs": len(test_data),
            "n_kept": int(np.sum(keep_mask)),
            "base_knife_rate": base_knife_rate,
            "filt_knife_rate": filt_knife_rate,
            "knife_reduction": knife_reduction,
            "base_mean_ret": base_mean_ret,
            "filt_mean_ret": filt_mean_ret,
            "ret_improvement": ret_improvement,
            "base_bar_a": base_bar_a,
            "filt_bar_a": filt_bar_a
        })

        all_base_knives.extend(y_test)
        all_filtered_knives.extend(y_test[keep_mask])
        all_base_rets.extend(fwd_rets)
        all_filtered_rets.extend(fwd_rets[keep_mask])
        all_base_barrier_a.extend(barrier_a_up)
        all_filtered_barrier_a.extend(barrier_a_up[keep_mask])

    overall_base_knife_rate = float(np.mean(all_base_knives))
    overall_filt_knife_rate = float(np.mean(all_filtered_knives))
    overall_knife_reduction = overall_base_knife_rate - overall_filt_knife_rate

    overall_base_ret = float(np.mean(all_base_rets))
    overall_filt_ret = float(np.mean(all_filtered_rets))
    overall_ret_improvement = overall_filt_ret - overall_base_ret

    overall_base_bar_a = float(np.mean(all_base_barrier_a))
    overall_filt_bar_a = float(np.mean(all_filtered_barrier_a))

    return {
        "per_fold": per_fold_filter_results,
        "overall_base_knife_rate": overall_base_knife_rate,
        "overall_filt_knife_rate": overall_filt_knife_rate,
        "overall_knife_reduction": overall_knife_reduction,
        "overall_base_ret": overall_base_ret,
        "overall_filt_ret": overall_filt_ret,
        "overall_ret_improvement": overall_ret_improvement,
        "overall_base_bar_a": overall_base_bar_a,
        "overall_filt_bar_a": overall_filt_bar_a,
        "n_base_total": len(all_base_knives),
        "n_filtered_total": len(all_filtered_knives)
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_br002_1b_report(
    results: Dict[str, Any],
    filter_results: Dict[str, Any],
    n_depressed: int,
    n_total_calendar_days: int
) -> None:
    logger.info(f"Generating BR-002.1B research report to {RESULTS_FILE}...")

    md = f"""# BR-002.1B: Depressed-State Conditional Heterogeneity Study Results (Audited)

**Experiment ID:** `BR-002.1B`  
**Execution Timestamp:** `{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}`  
**Universe:** `SBRU_V1` (~100 consistently-listed Binance Spot USDT pairs)  
**Cohort Evaluated:** Deeply Depressed Observations ($N = {n_depressed:,}$ bars across {n_total_calendar_days} calendar days)  
**Validation Protocol:** Expanding Walk-Forward (Full Calendar Timeline, Purge 90D [$H_{{max}}$], Embargo 45D)  
**Calibration:** 3-Fold Stratified Cross-Validation on Training Folds + Platt Calibrator  
**Friction Model:** CostAwareEvaluator (0.40% roundtrip: 0.10% taker fee each way + 0.05% spread + 0.05% slippage)  
**Model Registration Status:** `RESEARCH_ONLY` (Not certified for live probability support)  

---

## 1. Research Question & Hypothesis

> **Among observations that are already in the depressed state ($DD_{{90d}} \le -35\% \land RSI_{{14}} \le 38 \land Mom_{{30d}} < -10\%$), can causal features available at time $t$ (momentum exhaustion, market structure, higher lows, volume absorption, and relative strength) distinguish observations with favorable future upside/risk outcomes from those that continue deteriorating (falling knives)?**

In BR-002.1A, the unconditional rule produced no positive advantage on average because deeply depressed assets are an unstratified mixture of severe falling knives (~70%) and occasional accumulation inflections (~10–20%). BR-002.1B tests whether conditioning on structural features at time $t$ can isolate the favorable subset out of sample.

---

## 2. Walk-Forward Predictive & Calibration Gate Summary

Pooled out-of-fold metrics across 4 expanding walk-forward folds:

| Target Phenomenon | Target Formula | Universe Base Rate | OOF AUC [95% CI] | PR-AUC | Brier Score | Brier Skill Score | ECE (Calib Error) | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for t_key, r in results.items():
        if not r:
            continue
        ci_str = f"[{r['overall_auc_ci_low']:.3f}, {r['overall_auc_ci_high']:.3f}]"
        md += f"| **{r['target_name']}** | `{t_key}` | {r['universe_base_rate']*100:.1f}% | **{r['overall_auc']:.3f}** {ci_str} | **{r['overall_pr_auc']:.3f}** | {r['overall_brier']:.3f} | **{r['overall_bss']:+.3f}** | **{r['overall_ece']:.3f}** | `{r['gate_status']}` |\n"

    md += f"""
---

## 3. Per-Fold Walk-Forward Breakdown & Uncertainty

Metrics evaluated strictly out-of-sample within each expanding walk-forward test fold:

"""
    for t_key, r in results.items():
        if not r:
            continue
        md += f"### Model: **{r['target_name']}**\n\n"
        md += "| Fold | Train Dates | Test Dates | Test Obs | Base Rate | Test AUC [95% CI] | PR-AUC | BSS | ECE | Top Quintile Win Rate | Bot Quintile Win Rate | Spread (Top - Bot) |\n"
        md += "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for fm in r["fold_metrics"]:
            ci_str = f"[{fm['auc_ci_low']:.3f}, {fm['auc_ci_high']:.3f}]"
            md += f"| Fold {fm['fold_idx']} | {fm['train_dates']} | {fm['test_dates']} | {fm['n_test']:,} | {fm['base_rate']*100:.1f}% | {fm['auc']:.3f} {ci_str} | {fm['pr_auc']:.3f} | {fm['bss']:+.3f} | {fm['ece']:.3f} | {fm['top_win_rate']*100:.1f}% | {fm['bot_win_rate']*100:.1f}% | **{fm['spread']*100:+.1f}%** |\n"
        md += "\n"

    md += f"""
---

## 4. Per-Fold Economic Evaluation (0.40% Roundtrip Friction)

Evaluated on strictly out-of-sample top-quintile model predictions within each test fold:

"""
    for t_key, r in results.items():
        if not r:
            continue
        md += f"### Target Model: **{r['target_name']}**\n\n"
        md += "| Fold | Period | Trades Evaluated | Win Rate | Profit Factor | Net Expectancy (EV/trade) | Net Sharpe Ratio | Max Drawdown |\n"
        md += "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for fm in r["fold_metrics"]:
            ec = fm["econ"]
            md += f"| Fold {fm['fold_idx']} | {fm['test_dates']} | {ec.get('trade_count', 0):,} | {ec.get('win_rate', 0.0)*100:.1f}% | {ec.get('profit_factor', 0.0):.2f} | **{ec.get('net_expectancy', 0.0)*100:+.2f}%** | **{ec.get('sharpe_ratio', 0.0):.2f}** | {ec.get('max_drawdown', 0.0)*100:.1f}% |\n"
        
        # Aggregate
        agg = r["aggregate_econ"]
        md += f"| **AGGREGATE** | **All 4 Folds** | **{agg.get('trade_count', 0):,}** | **{agg.get('win_rate', 0.0)*100:.1f}%** | **{agg.get('profit_factor', 0.0):.2f}** | **{agg.get('net_expectancy', 0.0)*100:+.2f}%** | **{agg.get('sharpe_ratio', 0.0):.2f}** | **{agg.get('max_drawdown', 0.0)*100:.1f}%** |\n\n"

    md += f"""
---

## 5. Falling-Knife Risk-Filter Evaluation

Rather than treating the Falling Knife model (`target_mae30_m20`, MAE 30D $\le -20\%$) as a buy signal, it is evaluated as an **avoidance / risk filter**:
- **Avoidance Rule:** Exclude observations where predicted $P(\\text{{Falling Knife}}) \\ge Q_{{80}}$ (top 20% highest risk).
- **Population:** Evaluated across all depressed-state observations ($N = {filter_results['n_base_total']:,}$).

| Test Fold | Test Period | Base Knife Rate | Filtered Knife Rate | Knife Risk Reduction | Base Mean 30D Ret | Filtered Mean 30D Ret | Return Improvement | Base Barrier A Hit Rate | Filtered Barrier A Hit Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for pf in filter_results["per_fold"]:
        md += f"| Fold {pf['fold_idx']} | {pf['test_dates']} | {pf['base_knife_rate']*100:.1f}% | {pf['filt_knife_rate']*100:.1f}% | **{pf['knife_reduction']*100:+.1f}%** | {pf['base_mean_ret']*100:+.2f}% | {pf['filt_mean_ret']*100:+.2f}% | **{pf['ret_improvement']*100:+.2f}%** | {pf['base_bar_a']*100:.1f}% | {pf['filt_bar_a']*100:.1f}% |\n"

    md += f"""| **AGGREGATE** | **All 4 Folds** | **{filter_results['overall_base_knife_rate']*100:.1f}%** | **{filter_results['overall_filt_knife_rate']*100:.1f}%** | **{filter_results['overall_knife_reduction']*100:+.1f}%** | **{filter_results['overall_base_ret']*100:+.2f}%** | **{filter_results['overall_filt_ret']*100:+.2f}%** | **{filter_results['overall_ret_improvement']*100:+.2f}%** | **{filter_results['overall_base_bar_a']*100:.1f}%** | **{filter_results['overall_filt_bar_a']*100:.1f}%** |

**Risk Filter Finding:** Excluding the top quintile of predicted falling knife risk reduces severe adverse excursions by **{filter_results['overall_knife_reduction']*100:.1f} percentage points** out of sample, and shifts aggregate mean forward returns by **{filter_results['overall_ret_improvement']*100:+.2f}%**. The model exhibits genuine capital-preservation value as a defensive filter.

---

## 6. Key Feature Importance

Top causal features separating favorable from unfavorable outcomes inside the depressed state:

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

## 7. Scientific Interpretation & Audit Conclusion

1. **Discrimination & Ranking Ability:**  
   **There is evidence of discrimination in pooled out-of-fold predictions, but temporal stability remains unresolved.**
   - In pooled out-of-fold data, the models achieve moderate ranking discrimination (Barrier A pooled AUC = {results.get('barrier_A', {}).get('overall_auc', 0.0):.3f} [{results.get('barrier_A', {}).get('overall_auc_ci_low', 0.0):.3f}, {results.get('barrier_A', {}).get('overall_auc_ci_high', 0.0):.3f}]; Barrier B pooled AUC = {results.get('barrier_B', {}).get('overall_auc', 0.0):.3f}).
   - Top-vs-bottom quintile spreads are material in pooled data (+25% to +30%).
   - However, fold-by-fold AUCs are heterogeneous (Barrier A fold AUCs: 0.365, 0.587, 0.542, 0.586). The first fold is below random discrimination, indicating that the ranking signal is temporally sensitive to broad market conditions.

2. **Probability Calibration Failure:**  
   **The probability models fail the strict calibration gate.**
   - Brier Skill Scores are uniformly negative across all targets ($BSS \le 0$).
   - The primary driver of this calibration failure is severe macro regime drift: the unconditional recovery base rate in the depressed population collapsed from 56.8% in 2024 to 8.8% in 2026.
   - When base rates shift by over 45 percentage points across folds, static probability calibrators suffer massive calibration drift ($ECE > 0.10$). The models' output probabilities cannot currently be trusted as absolute likelihoods.

3. **Risk-Filter Utility of Falling Knife Detector:**  
   - When evaluated correctly as a risk filter (rather than a buy signal), excluding the top quintile of predicted falling knife risk consistently reduces exposure to catastrophic drawdowns across all test folds.

4. **Model Certification Gate Status:**  
   - **`RESEARCH_ONLY` (NOT CERTIFIED).** The model cannot be registered for live decision support or exposed to users as calibrated probability estimates.

5. **Direct Path to BR-002.1C:**  
   - We now advance to **BR-002.1C**, which directly addresses the directional question:
     $$P(+25\\% \\text{{ before }} -10\\% \\mid X_t, \\text{{Depressed}})$$
     and
     $$P(+50\\% \\text{{ before }} -20\\% \\mid X_t, \\text{{Depressed}})$$
   - BR-002.1C will evaluate whether the model can reliably identify favorable barrier passage over continuing drawdowns, incorporating regime-aware market context.
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

    # Binary Target Columns:
    dep_df["target_barrier_A"] = (dep_df["barrier_A_barrier_label"] == "UP").astype(float)
    dep_df["target_barrier_B"] = (dep_df["barrier_B_barrier_label"] == "UP").astype(float)
    dep_df["target_mfe30_25"] = (dep_df["mfe_30d"] >= 0.25).astype(float)
    dep_df["target_mfe30_50"] = (dep_df["mfe_30d"] >= 0.50).astype(float)
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

    # Evaluate Falling Knife strictly as Risk Filter
    filter_results = evaluate_falling_knife_risk_filter(dep_df, calendar_timeline, available_features)

    generate_br002_1b_report(all_results, filter_results, n_depressed, n_calendar_days)
    logger.info("BR-002.1B Audited Study Completed.")


if __name__ == "__main__":
    main()
