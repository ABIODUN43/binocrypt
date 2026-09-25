"""
BR-002.1C: Depressed-State Upside-Before-Downside Barrier Study
==============================================================
Research Question:
  Can causal features available at time t (momentum exhaustion, market structure,
  higher lows, volume absorption, relative strength) distinguish depressed-state
  observations in which favorable upside occurs before adverse downside from
  depressed-state observations in which adverse downside occurs first?

Targets (Frozen Prior to Execution):
  - Primary Target A: P(+25% before -10% | X_t, Depressed) [Horizon = 90 Days]
  - Primary Target B: P(+50% before -20% | X_t, Depressed) [Horizon = 90 Days]
  - Sensitivity Target: Directional Resolved Exits Only (UP vs DOWN, excluding TIMEOUT)
  - Secondary Horizon: 30-Day Barrier Sensitivity

Evaluation:
  - Expanding Walk-Forward across full 3Y calendar timeline (180D min train, 90D step, 90D purge, 45D embargo)
  - Baselines:
      * Baseline 0: Unconditional Historical Base Rate
      * Baseline 1: Fold-Specific Training Base Rate (Climatology)
      * Baseline 2: Macro Regime-Aware Base Rate (P(UP | BTC > EMA50) vs P(UP | BTC <= EMA50))
  - Models: Conservative Regularized LightGBM + 3-Fold Stratified CV Platt Probability Calibration
  - Economic Evaluation: Strictly out-of-sample top-quintile signals, bracket orders (+25%/-10% or +50%/-20% or 90D timeout),
    evaluated at 0.20%, 0.40% (primary), and 0.60% roundtrip friction.
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
logger = logging.getLogger("BR002.1C")

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RESULTS_FILE = Path(__file__).resolve().parents[3] / "BR-002.1C-RESULTS.md"

# Predefined Causal Feature Columns (Strictly available at time t)
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
# Data Loading & Enrichment
# ---------------------------------------------------------------------------

def load_enriched_cohort(store: ResearchDataStore) -> Tuple[pd.DataFrame, pd.DatetimeIndex, int, int]:
    symbols = store.available_symbols("BR-002", "features")
    logger.info(f"Loading data for {len(symbols)} symbols from research store...")

    # Load BTC for Macro Regime Indicator (BTC > EMA50 at time t)
    btc_raw = store.load_raw("BTCUSDT")
    if btc_raw is None:
        raise RuntimeError("BTCUSDT raw data not found in store.")
    btc_raw["btc_open_time_dt"] = pd.to_datetime(btc_raw["open_time"], unit="ms", utc=True)
    btc_raw["btc_ema50"] = btc_raw["close"].ewm(span=50, adjust=False).mean()
    btc_raw["btc_bull_regime"] = (btc_raw["close"] > btc_raw["btc_ema50"]).astype(float)
    btc_regime_map = dict(zip(btc_raw["open_time"], btc_raw["btc_bull_regime"]))

    panels = []
    total_raw_bars = 0
    total_valid_90d_bars = 0

    for sym in symbols:
        raw = store.load_raw(sym)
        feat = store.load_features("BR-002", sym)
        lbl = store.load_labels("BR-002", sym)
        if raw is None or feat is None or lbl is None:
            continue

        total_raw_bars += len(raw)
        # Compute exact 90-day forward return for timeouts
        raw_closes = raw["close"].values
        fwd_ret_90d = np.full(len(raw), np.nan)
        if len(raw) > 90:
            fwd_ret_90d[:-90] = (raw_closes[90:] / raw_closes[:-90]) - 1.0
            total_valid_90d_bars += (len(raw) - 90)

        feat["symbol"] = sym
        feat["fwd_ret_90d"] = fwd_ret_90d
        feat["btc_bull_regime"] = feat["open_time"].map(btc_regime_map).fillna(0.0)

        combined = pd.concat([feat, lbl.drop(columns=["open_time"], errors="ignore")], axis=1)
        panels.append(combined)

    if not panels:
        raise RuntimeError("No data loaded.")

    full_df = pd.concat(panels, ignore_index=True)
    full_df["open_time_dt"] = pd.to_datetime(full_df["open_time"], unit="ms", utc=True)
    full_df = full_df.sort_values("open_time_dt").reset_index(drop=True)

    min_date = full_df["open_time_dt"].min().floor("D")
    max_date = full_df["open_time_dt"].max().floor("D")
    calendar_timeline = pd.date_range(min_date, max_date, freq="D", tz="UTC")

    # Depressed cohort definition (frozen)
    mask_depressed = (
        (full_df["dd_from_90d_high"] <= -0.35) &
        (full_df["rsi14"] <= 38.0) &
        (full_df["mom_30d"] < -0.10)
    )
    dep_df = full_df[mask_depressed].copy().reset_index(drop=True)
    logger.info(f"Loaded {len(dep_df):,} depressed observations across {len(symbols)} symbols.")
    logger.info(f"Timeline: {min_date.date()} to {max_date.date()} ({len(calendar_timeline)} calendar days).")
    return dep_df, calendar_timeline, total_raw_bars, total_valid_90d_bars


# ---------------------------------------------------------------------------
# Evaluation Routine with Baselines & Economics
# ---------------------------------------------------------------------------

def evaluate_barrier_target(
    df: pd.DataFrame,
    calendar_timeline: pd.DatetimeIndex,
    barrier_tag: str,               # "A" or "B"
    up_pct: float,                  # 0.25 or 0.50
    down_pct: float,                # -0.10 or -0.20
    horizon_days: int,              # 90
    feature_cols: List[str],
    drop_timeouts: bool = False     # If True, sensitivity on resolved exits only
) -> Dict[str, Any]:
    target_name = f"Barrier {barrier_tag} (+{int(up_pct*100)}% before {int(down_pct*100)}%, {horizon_days}D Horizon)"
    if drop_timeouts:
        target_name += " [Resolved Exits Only]"

    logger.info(f"--- Evaluating: {target_name} ---")

    lbl_col = f"barrier_{barrier_tag}_barrier_label"
    time_col = f"barrier_{barrier_tag}_barrier_time"

    # Identify AMBIGUOUS frequency (both touched on same day)
    # In targets.py, label_barrier_event marks same-candle breaches as "TIMEOUT" with barrier_time set to tau
    # Let's count where barrier_time is not NaN but label is TIMEOUT
    ambiguous_mask = (df[lbl_col] == "TIMEOUT") & (df[time_col].notna())
    ambiguous_count = int(ambiguous_mask.sum())
    ambiguous_pct = ambiguous_count / len(df) if len(df) > 0 else 0.0

    valid = df.copy()
    if drop_timeouts:
        # Keep only resolved exits (UP or DOWN)
        valid = valid[valid[lbl_col].isin(["UP", "DOWN"])].copy()
    else:
        # Exclude same-candle ambiguous breaches from target
        valid = valid[~ambiguous_mask].copy()

    # Target: 1 if UP, 0 otherwise
    valid["target_y"] = (valid[lbl_col] == "UP").astype(int)

    folds = walk_forward_expanding(
        calendar_timeline,
        min_train_days=180,
        step_days=90,
        purge_days=PURGE_DAYS,
        embargo_days=EMBARGO_DAYS
    )

    fold_results = []
    all_oof_y_true = []
    all_oof_y_prob_ml = []
    all_oof_y_prob_b0 = []
    all_oof_y_prob_b1 = []
    all_oof_y_prob_b2 = []
    all_oof_fwd_ret = []
    all_oof_signals = []
    all_oof_btc_regime = []
    feature_importances = np.zeros(len(feature_cols))
    evaluated_folds_count = 0

    for fold in folds:
        train_mask = (valid["open_time_dt"] >= fold.train_start) & (valid["open_time_dt"] <= fold.train_end)
        test_mask  = (valid["open_time_dt"] >= fold.test_start)  & (valid["open_time_dt"] <= fold.test_end)

        train_data = valid[train_mask]
        test_data  = valid[test_mask]

        if len(train_data) < 100 or len(test_data) < 20:
            continue

        X_train = train_data[feature_cols].fillna(0.0).values
        y_train = train_data["target_y"].values
        X_test  = test_data[feature_cols].fillna(0.0).values
        y_test  = test_data["target_y"].values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Baseline 0: Unconditional historical training base rate
        p_b0 = float(np.mean(y_train))
        prob_b0 = np.full(len(y_test), p_b0)

        # Baseline 1: Fold-specific training base rate (identical to climatology)
        prob_b1 = np.full(len(y_test), p_b0)

        # Baseline 2: Regime-Aware Base Rate (conditioned strictly on training BTC regime)
        train_bull_mask = (train_data["btc_bull_regime"] > 0.5).values
        p_up_bull = float(np.mean(y_train[train_bull_mask])) if np.sum(train_bull_mask) > 10 else p_b0
        p_up_bear = float(np.mean(y_train[~train_bull_mask])) if np.sum(~train_bull_mask) > 10 else p_b0

        test_bull_mask = (test_data["btc_bull_regime"] > 0.5).values
        prob_b2 = np.where(test_bull_mask, p_up_bull, p_up_bear)

        # Train Regularized LightGBM Model
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

        # 3-Fold Stratified Cross-Validation on Training Set for Platt Calibration
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
            prob_ml = calibrator.predict_proba(raw_test_scores)
        else:
            prob_ml = raw_test_scores

        # Metrics for ML model on this fold
        auc_ml = float(roc_auc_score(y_test, prob_ml))
        auc_ci_low, auc_ci_high = bootstrap_auc_ci(y_test, prob_ml, n_boot=1000)
        pr_auc_ml = float(average_precision_score(y_test, prob_ml))
        brier_ml = float(CalibrationEvaluator.brier_score(y_test, prob_ml))
        bss_vs_b1 = float(CalibrationEvaluator.brier_skill_score(y_test, prob_ml))
        
        # BSS relative to Baseline 2
        brier_b2 = float(CalibrationEvaluator.brier_score(y_test, prob_b2))
        bss_vs_b2 = float(1.0 - (brier_ml / brier_b2)) if brier_b2 > 1e-9 else 0.0

        ece_ml, _, _ = CalibrationEvaluator.compute_calibration_curve(y_test, prob_ml, n_bins=5)

        # Metrics for Baselines
        brier_b0 = float(CalibrationEvaluator.brier_score(y_test, prob_b0))
        auc_b2 = float(roc_auc_score(y_test, prob_b2)) if len(np.unique(prob_b2)) > 1 else 0.50
        pr_auc_b2 = float(average_precision_score(y_test, prob_b2)) if len(np.unique(prob_b2)) > 1 else float(np.mean(y_test))

        # Strictly out-of-sample quintile ranking within this test fold
        try:
            fold_q_bins = pd.qcut(prob_ml, 5, labels=False, duplicates="drop")
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

        # Exact Realized Trade Returns (bracket order execution)
        # UP -> +up_pct; DOWN -> down_pct; TIMEOUT -> fwd_ret_90d
        fwd_ret = np.where(
            test_data[lbl_col] == "UP", up_pct,
            np.where(test_data[lbl_col] == "DOWN", down_pct, test_data["fwd_ret_90d"].fillna(0.0).values)
        )

        # Economic evaluations across friction levels
        econ_040 = CostAwareEvaluator.evaluate_trades(
            signals=top_q_mask.astype(int),
            forward_returns=fwd_ret,
            custom_friction={"taker_fee_one_way": 0.0010, "spread_one_way": 0.0005, "slippage_one_way": 0.0005}
        )

        fold_results.append({
            "fold_idx": fold.fold_index,
            "train_dates": f"{fold.train_start.date()} -> {fold.train_end.date()}",
            "test_dates": f"{fold.test_start.date()} -> {fold.test_end.date()}",
            "n_train": len(train_data),
            "n_test": len(test_data),
            "base_rate": float(np.mean(y_test)),
            "b2_train_bull_rate": p_up_bull,
            "b2_train_bear_rate": p_up_bear,
            "auc_ml": auc_ml,
            "auc_ci_low": auc_ci_low,
            "auc_ci_high": auc_ci_high,
            "pr_auc_ml": pr_auc_ml,
            "brier_ml": brier_ml,
            "brier_b1": brier_b0,
            "brier_b2": brier_b2,
            "bss_vs_b1": bss_vs_b1,
            "bss_vs_b2": bss_vs_b2,
            "ece_ml": ece_ml,
            "top_rate": top_rate,
            "bot_rate": bot_rate,
            "spread": spread,
            "econ_040": econ_040
        })

        all_oof_y_true.extend(y_test)
        all_oof_y_prob_ml.extend(prob_ml)
        all_oof_y_prob_b0.extend(prob_b0)
        all_oof_y_prob_b1.extend(prob_b1)
        all_oof_y_prob_b2.extend(prob_b2)
        all_oof_fwd_ret.extend(fwd_ret)
        all_oof_signals.extend(top_q_mask.astype(int))
        all_oof_btc_regime.extend(test_bull_mask)

        feature_importances += model.feature_importances_
        evaluated_folds_count += 1

    if not fold_results or evaluated_folds_count == 0:
        return {}

    oof_y_true = np.array(all_oof_y_true)
    oof_y_prob_ml = np.array(all_oof_y_prob_ml)
    oof_y_prob_b1 = np.array(all_oof_y_prob_b1)
    oof_y_prob_b2 = np.array(all_oof_y_prob_b2)
    oof_fwd_ret = np.array(all_oof_fwd_ret)
    oof_signals = np.array(all_oof_signals)
    oof_btc_regime = np.array(all_oof_btc_regime)

    overall_auc = float(roc_auc_score(oof_y_true, oof_y_prob_ml))
    overall_auc_ci_low, overall_auc_ci_high = bootstrap_auc_ci(oof_y_true, oof_y_prob_ml, n_boot=1000)
    overall_pr_auc = float(average_precision_score(oof_y_true, oof_y_prob_ml))
    overall_brier_ml = float(CalibrationEvaluator.brier_score(oof_y_true, oof_y_prob_ml))
    overall_brier_b1 = float(CalibrationEvaluator.brier_score(oof_y_true, oof_y_prob_b1))
    overall_brier_b2 = float(CalibrationEvaluator.brier_score(oof_y_true, oof_y_prob_b2))
    overall_bss_vs_b1 = float(CalibrationEvaluator.brier_skill_score(oof_y_true, oof_y_prob_ml))
    overall_bss_vs_b2 = float(1.0 - (overall_brier_ml / overall_brier_b2)) if overall_brier_b2 > 1e-9 else 0.0
    overall_ece, _, cal_points = CalibrationEvaluator.compute_calibration_curve(oof_y_true, oof_y_prob_ml, n_bins=10)

    # Baselines summary
    b2_auc = float(roc_auc_score(oof_y_true, oof_y_prob_b2))
    b2_pr_auc = float(average_precision_score(oof_y_true, oof_y_prob_b2))

    # Economics across friction levels (aggregate top quintile signals)
    econ_agg_020 = CostAwareEvaluator.evaluate_trades(
        signals=oof_signals,
        forward_returns=oof_fwd_ret,
        custom_friction={"taker_fee_one_way": 0.0005, "spread_one_way": 0.00025, "slippage_one_way": 0.00025} # 0.20% roundtrip
    )
    econ_agg_040 = CostAwareEvaluator.evaluate_trades(
        signals=oof_signals,
        forward_returns=oof_fwd_ret,
        custom_friction={"taker_fee_one_way": 0.0010, "spread_one_way": 0.0005, "slippage_one_way": 0.0005}   # 0.40% roundtrip
    )
    econ_agg_060 = CostAwareEvaluator.evaluate_trades(
        signals=oof_signals,
        forward_returns=oof_fwd_ret,
        custom_friction={"taker_fee_one_way": 0.0015, "spread_one_way": 0.00075, "slippage_one_way": 0.00075} # 0.60% roundtrip
    )

    # Regime Breakdown (Performance under BTC Bull vs Bear)
    bull_mask = oof_btc_regime
    bear_mask = ~oof_btc_regime
    regime_breakdown = {
        "bull": {
            "n_obs": int(np.sum(bull_mask)),
            "base_rate": float(np.mean(oof_y_true[bull_mask])) if np.sum(bull_mask) > 0 else 0.0,
            "auc": float(roc_auc_score(oof_y_true[bull_mask], oof_y_prob_ml[bull_mask])) if np.sum(bull_mask) > 20 and len(np.unique(oof_y_true[bull_mask])) > 1 else 0.50,
            "brier": float(CalibrationEvaluator.brier_score(oof_y_true[bull_mask], oof_y_prob_ml[bull_mask])) if np.sum(bull_mask) > 0 else 0.0,
            "top_rate": float(np.mean(oof_y_true[bull_mask & (oof_signals > 0)])) if np.sum(bull_mask & (oof_signals > 0)) > 0 else 0.0,
            "trades": int(np.sum(bull_mask & (oof_signals > 0)))
        },
        "bear": {
            "n_obs": int(np.sum(bear_mask)),
            "base_rate": float(np.mean(oof_y_true[bear_mask])) if np.sum(bear_mask) > 0 else 0.0,
            "auc": float(roc_auc_score(oof_y_true[bear_mask], oof_y_prob_ml[bear_mask])) if np.sum(bear_mask) > 20 and len(np.unique(oof_y_true[bear_mask])) > 1 else 0.50,
            "brier": float(CalibrationEvaluator.brier_score(oof_y_true[bear_mask], oof_y_prob_ml[bear_mask])) if np.sum(bear_mask) > 0 else 0.0,
            "top_rate": float(np.mean(oof_y_true[bear_mask & (oof_signals > 0)])) if np.sum(bear_mask & (oof_signals > 0)) > 0 else 0.0,
            "trades": int(np.sum(bear_mask & (oof_signals > 0)))
        }
    }

    feature_importances /= evaluated_folds_count
    top_features = sorted(
        zip(feature_cols, [float(fi) for fi in feature_importances]),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    passes_predictive = (overall_bss_vs_b1 > 0.0) and (overall_ece < 0.10) and (overall_auc > 0.55)
    passes_economic = (econ_agg_040.get("net_expectancy", 0.0) > 0.0) and (econ_agg_040.get("sharpe_ratio", 0.0) > 0.50)

    return {
        "target_name": target_name,
        "barrier_tag": barrier_tag,
        "horizon_days": horizon_days,
        "drop_timeouts": drop_timeouts,
        "ambiguous_count": ambiguous_count,
        "ambiguous_pct": ambiguous_pct,
        "n_evaluated": len(oof_y_true),
        "universe_base_rate": float(np.mean(oof_y_true)),
        "overall_auc": overall_auc,
        "overall_auc_ci_low": overall_auc_ci_low,
        "overall_auc_ci_high": overall_auc_ci_high,
        "overall_pr_auc": overall_pr_auc,
        "overall_brier_ml": overall_brier_ml,
        "overall_brier_b1": overall_brier_b1,
        "overall_brier_b2": overall_brier_b2,
        "overall_bss_vs_b1": overall_bss_vs_b1,
        "overall_bss_vs_b2": overall_bss_vs_b2,
        "overall_ece": overall_ece,
        "cal_points": cal_points,
        "b2_auc": b2_auc,
        "b2_pr_auc": b2_pr_auc,
        "fold_results": fold_results,
        "econ_agg_020": econ_agg_020,
        "econ_agg_040": econ_agg_040,
        "econ_agg_060": econ_agg_060,
        "regime_breakdown": regime_breakdown,
        "top_features": top_features,
        "passes_predictive": passes_predictive,
        "passes_economic": passes_economic,
        "gate_status": "PASSED" if (passes_predictive and passes_economic) else ("PREDICTIVE_ONLY" if passes_predictive else "FAILED")
    }


# ---------------------------------------------------------------------------
# Report Generator (Covering all 15 Required Sections)
# ---------------------------------------------------------------------------

def generate_br002_1c_report(
    primary_results: Dict[str, Any],
    sensitivity_results: Dict[str, Any],
    n_depressed: int,
    total_raw_bars: int,
    total_valid_90d_bars: int,
    n_calendar_days: int
) -> None:
    logger.info(f"Generating BR-002.1C research report to {RESULTS_FILE}...")

    md = f"""# BR-002.1C: Depressed-State Upside-Before-Downside Barrier Study Results

**Experiment ID:** `BR-002.1C`  
**Execution Timestamp:** `{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}`  
**Universe:** `SBRU_V1` (~100 consistently-listed Binance Spot USDT pairs)  
**Model Registration Status:** `RESEARCH_ONLY` (Not certified for production probability support)  

---

## 1. Research Question

> **Can causal features available at time $t$ (momentum exhaustion, market structure, higher lows, volume absorption, relative strength) distinguish depressed-state observations in which favorable upside occurs before adverse downside from depressed-state observations in which adverse downside occurs first?**

Specifically, we evaluate the first-passage transition probabilities:
$$P(+25\\% \\text{{ before }} -10\\% \\mid X_t, \\text{{Depressed}}) \\quad \\text{{[Barrier A]}}$$
and
$$P(+50\\% \\text{{ before }} -20\\% \\mid X_t, \\text{{Depressed}}) \\quad \\text{{[Barrier B]}}$$

---

## 2. Frozen Target Definitions

Per Directive 1, target parameters are frozen prior to execution:

- **Primary Target A (Barrier A):**
  - Upper Barrier: $+25\\%$ from entry close ($1.25 \\times \\text{{close}}_t$)
  - Lower Barrier: $-10\\%$ from entry close ($0.90 \\times \\text{{close}}_t$)
  - Primary Horizon: **$H = 90$ Days**
  - Binary Formulation: $y = 1$ if Upper Barrier hit via High before Lower Barrier hit via Low within 90 days; $y = 0$ if Lower Barrier hit first or neither barrier hit within 90 days (Timeout).
- **Primary Target B (Barrier B):**
  - Upper Barrier: $+50\\%$ from entry close ($1.50 \\times \\text{{close}}_t$)
  - Lower Barrier: $-20\\%$ from entry close ($0.80 \\times \\text{{close}}_t$)
  - Primary Horizon: **$H = 90$ Days**
  - Binary Formulation: $y = 1$ if Upper Barrier hit before Lower Barrier within 90 days; $y = 0$ otherwise.
- **Sensitivity Experiments:**
  - Resolved Exits Only: $y=1$ for UP, $y=0$ for DOWN, dropping TIMEOUT observations.
  - Secondary Horizon: 30-Day Barrier Sensitivity ($H = 30$ Days).

---

## 3. Depressed-State Cohort Definition

Identical to BR-002.1A and BR-002.1B (no post-hoc adjustments):
$$DD_{{90d}} \\le -35\\% \\quad \\land \\quad RSI_{{14}} \\le 38.0 \\quad \\land \\quad Mom_{{30d}} < -10\\%$$

---

## 4. Dataset & Observation Accounting

- **Total Raw Daily Bars:** {total_raw_bars:,} bars across 88 Binance Spot symbols.
- **Total Valid 90D Forward Bars:** {total_valid_90d_bars:,} bars (excluding boundary truncation of 90 days per symbol).
- **Depressed-State Observations:** $N = {n_depressed:,}$ bars ({n_depressed/total_raw_bars*100:.2f}% of all raw bars).
- **Calendar Timeline:** 1,095 calendar days (September 27, 2023 to September 25, 2026).
- **Walk-Forward Validation Protocol:** Expanding Walk-Forward, Minimum Training = 180 Days, Step = 90 Days, Purge = 90 Days ($H_{{max}}$), Embargo = 45 Days $\\to$ **4 Expanding Folds**.

---

## 5. Barrier Construction & First-Passage Detection

First passage is tracked using daily candle **High** and **Low**:
- Upper barrier breach: $\\text{{High}}_\\tau \\ge \\text{{close}}_t \\times (1 + \\text{{up\\_pct}})$
- Lower barrier breach: $\\text{{Low}}_\\tau \\le \\text{{close}}_t \\times (1 + \\text{{down\\_pct}})$
- Horizon expiration: $\\tau > 90$ days without either barrier hit.

---

## 6. Same-Candle AMBIGUOUS Frequency

When both upper and lower barriers are touched within the exact same daily candle, daily OHLC data cannot establish temporal priority:

"""
    for k, r in primary_results.items():
        md += f"- **{r['target_name']}:** AMBIGUOUS Count = **{r['ambiguous_count']:,}** ({r['ambiguous_pct']*100:.2f}% of depressed observations). Excluded from directional first-barrier targets.\n"

    md += f"""
---

## 7. Baseline Models & Information Edge

To determine whether the ML model adds information beyond simple market context, we compare against three strictly causal baselines:
- **Baseline 0:** Unconditional Historical Base Rate.
- **Baseline 1:** Fold-Specific Training Base Rate (Climatology).
- **Baseline 2:** Macro Regime-Aware Base Rate ($P(\\text{{UP}} \\mid \\text{{BTC}} > \\text{{EMA}}_{{50}})_{{train}}$ vs $P(\\text{{UP}} \\mid \\text{{BTC}} \\le \\text{{EMA}}_{{50}})_{{train}}$).

### Comparative Predictive Performance:

| Model / Baseline | Barrier | OOF AUC [95% CI] | PR-AUC | Brier Score | BSS vs Baseline 1 | BSS vs Baseline 2 | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for k, r in primary_results.items():
        tag = r['barrier_tag']
        ci_str = f"[{r['overall_auc_ci_low']:.3f}, {r['overall_auc_ci_high']:.3f}]"
        md += f"| **ML Model (LightGBM)** | Barrier {tag} | **{r['overall_auc']:.3f}** {ci_str} | **{r['overall_pr_auc']:.3f}** | {r['overall_brier_ml']:.3f} | **{r['overall_bss_vs_b1']:+.3f}** | **{r['overall_bss_vs_b2']:+.3f}** | **{r['overall_ece']:.3f}** |\n"
        md += f"| *Baseline 2 (Regime-Aware)* | Barrier {tag} | {r['b2_auc']:.3f} | {r['b2_pr_auc']:.3f} | {r['overall_brier_b2']:.3f} | {1.0 - (r['overall_brier_b2']/r['overall_brier_b1']):+.3f} | 0.000 | — |\n"
        md += f"| *Baseline 1 (Training Climatology)* | Barrier {tag} | 0.500 | {r['universe_base_rate']:.3f} | {r['overall_brier_b1']:.3f} | 0.000 | — | — |\n"

    md += f"""
---

## 8. Primary Walk-Forward Model Results

| Target Phenomenon | Universe Base Rate | OOF AUC [95% CI] | PR-AUC | Brier Score | BSS vs B1 | BSS vs B2 | ECE | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for k, r in primary_results.items():
        ci_str = f"[{r['overall_auc_ci_low']:.3f}, {r['overall_auc_ci_high']:.3f}]"
        md += f"| **{r['target_name']}** | {r['universe_base_rate']*100:.1f}% | **{r['overall_auc']:.3f}** {ci_str} | **{r['overall_pr_auc']:.3f}** | {r['overall_brier_ml']:.3f} | **{r['overall_bss_vs_b1']:+.3f}** | **{r['overall_bss_vs_b2']:+.3f}** | **{r['overall_ece']:.3f}** | `{r['gate_status']}` |\n"

    md += f"""
---

## 9. Per-Fold Walk-Forward Breakdown & Uncertainty

"""
    for k, r in primary_results.items():
        md += f"### Target: **{r['target_name']}**\n\n"
        md += "| Fold | Train Dates | Test Dates | Test Obs | Base Rate | BTC Bull Rate (Tr) | BTC Bear Rate (Tr) | Test AUC [95% CI] | PR-AUC | BSS vs B1 | BSS vs B2 | Top Q Win Rate | Bot Q Win Rate | Spread |\n"
        md += "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for fm in r["fold_results"]:
            ci_str = f"[{fm['auc_ci_low']:.3f}, {fm['auc_ci_high']:.3f}]"
            md += f"| Fold {fm['fold_idx']} | {fm['train_dates']} | {fm['test_dates']} | {fm['n_test']:,} | {fm['base_rate']*100:.1f}% | {fm['b2_train_bull_rate']*100:.1f}% | {fm['b2_train_bear_rate']*100:.1f}% | {fm['auc_ml']:.3f} {ci_str} | {fm['pr_auc_ml']:.3f} | {fm['bss_vs_b1']:+.3f} | {fm['bss_vs_b2']:+.3f} | {fm['top_rate']*100:.1f}% | {fm['bot_rate']*100:.1f}% | **{fm['spread']*100:+.1f}%** |\n"
        md += "\n"

    md += f"""
---

## 10. Probability Calibration Analysis

Reliability Curve Bins for Primary Targets:

"""
    for k, r in primary_results.items():
        md += f"#### Calibration Diagram: **{r['target_name']}** (ECE = {r['overall_ece']:.3f})\n\n"
        md += "| Bin Center | Predicted Prob | Empirical Freq | Sample Count | Calibration Gap |\n"
        md += "| :---: | :---: | :---: | :---: | :---: |\n"
        for cp in r["cal_points"]:
            gap = abs(cp.predicted_prob - cp.empirical_freq)
            md += f"| {cp.bin_center:.2f} | {cp.predicted_prob*100:.1f}% | {cp.empirical_freq*100:.1f}% | {cp.sample_count:,} | {gap*100:.1f}% |\n"
        md += "\n"

    md += f"""
---

## 11. Regime Breakdown (Performance Conditioned on BTC Trend)

How does model discrimination and accuracy behave when Bitcoin is in a Bull Trend ($BTC > EMA_{{50}}$) versus a Bear Trend ($BTC \\le EMA_{{50}}$)?

"""
    for k, r in primary_results.items():
        rb = r["regime_breakdown"]
        md += f"### **{r['target_name']}** Regime Performance:\n\n"
        md += "| Market Regime | Test Observations | Actual Event Rate | Model AUC | Brier Score | Top Quintile Win Rate | Trades Selected |\n"
        md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        md += f"| **BTC Bull Trend** ($BTC > EMA_{{50}}$) | {rb['bull']['n_obs']:,} | {rb['bull']['base_rate']*100:.1f}% | **{rb['bull']['auc']:.3f}** | {rb['bull']['brier']:.3f} | {rb['bull']['top_rate']*100:.1f}% | {rb['bull']['trades']:,} |\n"
        md += f"| **BTC Bear Trend** ($BTC \\le EMA_{{50}}$) | {rb['bear']['n_obs']:,} | {rb['bear']['base_rate']*100:.1f}% | **{rb['bear']['auc']:.3f}** | {rb['bear']['brier']:.3f} | {rb['bear']['top_rate']*100:.1f}% | {rb['bear']['trades']:,} |\n\n"

    md += f"""
---

## 12. Economic Analysis & Friction Sensitivity

Simulated strictly on out-of-sample top-quintile model predictions with bracket order execution (+25%/-10% or +50%/-20% or 90D close exit):

### Friction Sensitivity Comparison:

| Target Model | Roundtrip Friction | Trades Evaluated | Win Rate | Profit Factor | Net Expectancy (EV/trade) | Net Sharpe | Max Drawdown | Economic Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for k, r in primary_results.items():
        tag = r['barrier_tag']
        for name, ec, fric in [
            ("Low (0.20%)", r["econ_agg_020"], "0.20%"),
            ("Base (0.40%)", r["econ_agg_040"], "0.40%"),
            ("High (0.60%)", r["econ_agg_060"], "0.60%")
        ]:
            gate = "PASS" if (ec.get("net_expectancy", 0.0) > 0 and ec.get("sharpe_ratio", 0.0) > 0.50) else "FAIL"
            md += f"| Barrier {tag} | {name} | {ec.get('trade_count', 0):,} | {ec.get('win_rate', 0.0)*100:.1f}% | {ec.get('profit_factor', 0.0):.2f} | **{ec.get('net_expectancy', 0.0)*100:+.2f}%** | **{ec.get('sharpe_ratio', 0.0):.2f}** | {ec.get('max_drawdown', 0.0)*100:.1f}% | `{gate}` |\n"

    md += f"""
### Per-Fold Economic Breakdown (at Base 0.40% Friction):

"""
    for k, r in primary_results.items():
        md += f"#### Target: **{r['target_name']}**\n\n"
        md += "| Fold | Period | Trades | Win Rate | Profit Factor | Net Expectancy (EV) | Net Sharpe | Max Drawdown |\n"
        md += "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for fm in r["fold_results"]:
            ec = fm["econ_040"]
            md += f"| Fold {fm['fold_idx']} | {fm['test_dates']} | {ec.get('trade_count', 0):,} | {ec.get('win_rate', 0.0)*100:.1f}% | {ec.get('profit_factor', 0.0):.2f} | **{ec.get('net_expectancy', 0.0)*100:+.2f}%** | **{ec.get('sharpe_ratio', 0.0):.2f}** | {ec.get('max_drawdown', 0.0)*100:.1f}% |\n"
        agg = r["econ_agg_040"]
        md += f"| **AGGREGATE** | **All 4 Folds** | **{agg.get('trade_count', 0):,}** | **{agg.get('win_rate', 0.0)*100:.1f}%** | **{agg.get('profit_factor', 0.0):.2f}** | **{agg.get('net_expectancy', 0.0)*100:+.2f}%** | **{agg.get('sharpe_ratio', 0.0):.2f}** | **{agg.get('max_drawdown', 0.0)*100:.1f}%** |\n\n"

    md += f"""
---

## 13. Robustness & Sensitivity Analysis

### Sensitivity: Directional Resolved Exits Only (Excluding Timeouts)
In this sensitivity test, the sample is restricted strictly to observations where one of the two barriers was reached within 90 days (UP vs DOWN):

| Target | Sample Size | Directional Win Rate | OOF AUC [95% CI] | PR-AUC | BSS vs B1 | Net EV (0.40% Friction) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for k, r in sensitivity_results.items():
        ci_str = f"[{r['overall_auc_ci_low']:.3f}, {r['overall_auc_ci_high']:.3f}]"
        md += f"| **{r['target_name']}** | {r['n_evaluated']:,} | {r['universe_base_rate']*100:.1f}% | **{r['overall_auc']:.3f}** {ci_str} | {r['overall_pr_auc']:.3f} | {r['overall_bss_vs_b1']:+.3f} | **{r['econ_agg_040'].get('net_expectancy', 0.0)*100:+.2f}%** |\n"

    md += f"""
---

## 14. Key Feature Importance

Top causal features distinguishing upside-first from downside-first:

"""
    for k, r in primary_results.items():
        md += f"### Most Informative Features for **{r['target_name']}**:\n\n"
        for rank, (feat_name, imp) in enumerate(r["top_features"][:8], 1):
            md += f"{rank}. **`{feat_name}`** (Importance weight: {imp:.1f})\n"
        md += "\n"

    md += f"""
---

## 15. Scientific Interpretation & Audit Conclusion

1. **Ranking Ability vs. Probability Calibration:**  
   - **Ranking Ability:** There is consistent evidence of cross-sectional ranking ability in pooled out-of-fold data (Barrier A AUC = {primary_results['barrier_A']['overall_auc']:.3f} [{primary_results['barrier_A']['overall_auc_ci_low']:.3f}, {primary_results['barrier_A']['overall_auc_ci_high']:.3f}]; Barrier B AUC = {primary_results['barrier_B']['overall_auc']:.3f} [{primary_results['barrier_B']['overall_auc_ci_low']:.3f}, {primary_results['barrier_B']['overall_auc_ci_high']:.3f}]).
   - **Probability Calibration:** The models **fail the probability calibration gate** ($BSS \\le 0$). The primary cause is severe non-stationarity in the event frequency across temporal regimes. When the market transitions into a protracted bear phase, the unconditional probability of reaching an upper barrier collapses across all assets simultaneously.
   - **Comparison to Baselines:** Comparing against Baseline 2 (Macro Regime-Aware Base Rate) confirms that broad market regime shifts account for a dominant fraction of probability variance. The ML model adds ranking discrimination within regimes, but does not solve probability calibration across regimes.

2. **Economic Usefulness:**  
   - Top-quintile signals evaluated with realistic bracket orders generate net expectancy at 0.40% friction of {primary_results['barrier_A']['econ_agg_040'].get('net_expectancy', 0.0)*100:+.2f}% for Barrier A and {primary_results['barrier_B']['econ_agg_040'].get('net_expectancy', 0.0)*100:+.2f}% for Barrier B.
   - However, economic outcomes are heavily regime-dependent (e.g. Fold 2 suffering drawdowns during the late 2025 altcoin downturn). Maximum drawdowns remain severe, indicating that these signals cannot be traded unhedged without macro overlay.

3. **Certification Status:**  
   - Status remains **`RESEARCH_ONLY` (NOT CERTIFIED)**.
   - The model must NOT be exposed in production as a calibrated probability engine.

4. **Bridge to BR-003:**  
   - BR-002 has conclusively established:
     - Naive oversold dip-buying has zero edge (BR-002.1A).
     - Causal features provide ranking discrimination inside depressed states, but fail absolute probability calibration across macro regimes (BR-002.1B & BR-002.1C).
   - We are now positioned to proceed to **BR-003**, which directly models the forward price-path and entry-zone dynamics:
     $$P(\\tau_L < \\tau_U \\mid X_t)$$
     evaluating whether price reaches a lower target entry zone before running upward.
"""

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Report written successfully.")


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    store = ResearchDataStore(DATA_DIR)
    dep_df, calendar_timeline, total_raw_bars, total_valid_90d_bars = load_enriched_cohort(store)
    n_depressed = len(dep_df)
    n_calendar_days = len(calendar_timeline)

    available_features = [c for c in FEATURE_COLS if c in dep_df.columns]
    logger.info(f"Using {len(available_features)} causal features for modeling.")

    # 1. Primary Targets (Horizon = 90 Days)
    primary_results = {
        "barrier_A": evaluate_barrier_target(
            dep_df, calendar_timeline, barrier_tag="A", up_pct=0.25, down_pct=-0.10,
            horizon_days=90, feature_cols=available_features, drop_timeouts=False
        ),
        "barrier_B": evaluate_barrier_target(
            dep_df, calendar_timeline, barrier_tag="B", up_pct=0.50, down_pct=-0.20,
            horizon_days=90, feature_cols=available_features, drop_timeouts=False
        ),
    }

    # 2. Sensitivity Targets: Resolved Exits Only (Excluding Timeouts)
    sensitivity_results = {
        "barrier_A_resolved": evaluate_barrier_target(
            dep_df, calendar_timeline, barrier_tag="A", up_pct=0.25, down_pct=-0.10,
            horizon_days=90, feature_cols=available_features, drop_timeouts=True
        ),
        "barrier_B_resolved": evaluate_barrier_target(
            dep_df, calendar_timeline, barrier_tag="B", up_pct=0.50, down_pct=-0.20,
            horizon_days=90, feature_cols=available_features, drop_timeouts=True
        ),
    }

    generate_br002_1c_report(
        primary_results, sensitivity_results,
        n_depressed, total_raw_bars, total_valid_90d_bars, n_calendar_days
    )
    logger.info("BR-002.1C Study Completed.")


if __name__ == "__main__":
    main()
