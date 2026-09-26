"""
BR-003 Calibration & Stability Audit Script
===========================================
Executes a rigorous, targeted calibration and stability audit for:
  - BR-003.1 (Level-Reach Probability)
  - BR-003.2 (First-Passage Probability)

Requirements checked:
  1. BSS and ECE against every approved causal baseline (B0, B1, B2).
  2. Per-fold walk-forward metrics (Folds 1, 2, 3, 4).
  3. 10-bin reliability/calibration distribution.
  4. Strict evaluation of Platt scaling and Isotonic calibration (leakage check, functional form).
  5. Cross-horizon and cross-barrier calibration breakdown.
  6. Regime drift vs implementation error analysis (Murphy Brier decomposition: Uncertainty, Reliability, Resolution).
"""

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

backend_dir = Path(__file__).resolve().parents[3]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.research.calibration import CalibrationEvaluator, PlattCalibrator
from app.research.data.store import ResearchDataStore
from app.research.targets import TargetGenerator
from app.research.walk_forward import (
    walk_forward_expanding,
    PURGE_DAYS,
    EMBARGO_DAYS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BR003_AUDIT")

DATA_DIR = backend_dir / "data"

FEATURE_COLS = [
    "dd_from_30d_high", "dd_from_90d_high", "dd_from_180d_high", "dd_from_365d_high",
    "up_from_30d_low", "up_from_90d_low", "up_from_180d_low", "up_from_365d_low",
    "dist_from_ath", "dist_from_atl",
    "dist_ema20", "dist_ema50", "dist_ema200", "ema20_slope_5d",
    "mom_7d", "mom_14d", "mom_30d", "mom_60d", "mom_90d",
    "rsi14", "rsi_change_7d", "macd_hist", "macd_accel_3d",
    "adx14", "adx_change_7d",
    "lower_low_count_20d", "higher_low_count_20d", "lower_high_count_20d", "higher_high_count_20d",
    "trend_persistence_20d",
    "rvol20", "vol_zscore_30d", "vol_price_divergence", "taker_buy_ratio", "amihud_illiquidity_proxy",
    "realized_vol_14d", "vol_change_14d", "atr_pct_14d", "bb_width", "bb_percentile_90d",
    "rel_strength_btc_7d", "rel_strength_btc_14d", "rel_strength_eth_7d", "rel_strength_eth_14d",
    "market_breadth_proxy"
]

def murphy_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Decomposes Brier Score into:
      BS = Uncertainty - Resolution + Reliability
    where:
      Uncertainty = c * (1 - c), with c = base rate of sample
      Reliability = sum (n_k / N) * (p_k - o_k)^2  [Calibration error, lower is better]
      Resolution  = sum (n_k / N) * (o_k - c)^2    [Sorting/discrimination skill, higher is better]
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y = y_true[valid]
    p = np.clip(y_prob[valid], 0.0, 1.0)
    N = len(y)
    if N == 0:
        return {"bs": 0.0, "uncertainty": 0.0, "reliability": 0.0, "resolution": 0.0}

    c = float(np.mean(y))
    uncertainty = c * (1.0 - c)
    bs = float(np.mean((p - y) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= bin_edges[i]) & (p <= bin_edges[i+1])
        else:
            mask = (p >= bin_edges[i]) & (p < bin_edges[i+1])

        n_k = int(np.sum(mask))
        if n_k > 0:
            p_k = float(np.mean(p[mask]))
            o_k = float(np.mean(y[mask]))
            weight = n_k / N
            reliability += weight * ((p_k - o_k) ** 2)
            resolution += weight * ((o_k - c) ** 2)

    return {
        "bs": bs,
        "base_rate": c,
        "uncertainty": uncertainty,
        "reliability": reliability,
        "resolution": resolution
    }

def compute_bin_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    valid = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y = y_true[valid]
    p = np.clip(y_prob[valid], 0.0, 1.0)
    N = len(y)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i+1]
        if i == n_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        n_k = int(np.sum(mask))
        if n_k > 0:
            p_mean = float(np.mean(p[mask]))
            o_mean = float(np.mean(y[mask]))
            gap = abs(p_mean - o_mean)
        else:
            p_mean = (low + high) / 2.0
            o_mean = 0.0
            gap = 0.0
        rows.append({
            "bin": i + 1,
            "range": f"[{low:.1f}, {high:.1f}]",
            "count": n_k,
            "pct_samples": round((n_k / N) * 100, 2) if N > 0 else 0.0,
            "pred_prob": round(p_mean, 4),
            "obs_freq": round(o_mean, 4),
            "gap": round(gap, 4)
        })
    return rows

def load_data(store: ResearchDataStore) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    symbols = store.available_symbols("BR-002", "features")
    btc_raw = store.load_raw("BTCUSDT")
    btc_raw["btc_ema50"] = btc_raw["close"].ewm(span=50, adjust=False).mean()
    btc_raw["btc_bull_regime"] = (btc_raw["close"] > btc_raw["btc_ema50"]).astype(float)
    btc_regime_map = dict(zip(btc_raw["open_time"], btc_raw["btc_bull_regime"]))

    panels = []
    for sym in symbols:
        raw = store.load_raw(sym)
        feat = store.load_features("BR-002", sym)
        if raw is None or feat is None:
            continue
        feat["symbol"] = sym
        feat["btc_bull_regime"] = feat["open_time"].map(btc_regime_map).fillna(0.0)

        # Precompute BR-003.1 targets
        lows = raw["low"].values
        closes = raw["close"].values
        for L in [-0.05, -0.08, -0.10, -0.15, -0.20]:
            tag_l = f"m{int(abs(L)*100):02d}"
            for h in [1, 3, 7, 14, 30]:
                feat[f"reach_{tag_l}_{h}d"] = TargetGenerator.label_level_reach(lows, closes, L, h)

        # Precompute ARB diagnostic and key BR-003.2 targets
        fp_arb = TargetGenerator.label_first_passage(raw["high"].values, lows, closes, -0.0677, +0.0505, 30)
        feat["fp_arb_label"] = fp_arb["first_passage_label"]

        fp_sym5 = TargetGenerator.label_first_passage(raw["high"].values, lows, closes, -0.05, +0.05, 30)
        feat["fp_sym5_label"] = fp_sym5["first_passage_label"]

        fp_sym10 = TargetGenerator.label_first_passage(raw["high"].values, lows, closes, -0.10, +0.10, 30)
        feat["fp_sym10_label"] = fp_sym10["first_passage_label"]

        fp_asym10_25 = TargetGenerator.label_first_passage(raw["high"].values, lows, closes, -0.10, +0.25, 30)
        feat["fp_asym10_25_label"] = fp_asym10_25["first_passage_label"]

        panels.append(feat)

    full_df = pd.concat(panels, ignore_index=True)
    full_df["open_time_dt"] = pd.to_datetime(full_df["open_time"], unit="ms", utc=True)
    full_df = full_df.sort_values("open_time_dt").reset_index(drop=True)

    min_date = full_df["open_time_dt"].min().floor("D")
    max_date = full_df["open_time_dt"].max().floor("D")
    calendar_timeline = pd.date_range(min_date, max_date, freq="D", tz="UTC")
    return full_df, calendar_timeline

def audit_target(
    df: pd.DataFrame,
    calendar_timeline: pd.DatetimeIndex,
    target_col: str,
    target_name: str
) -> Dict[str, Any]:
    valid = df.dropna(subset=[target_col]).copy()
    folds = walk_forward_expanding(
        calendar_timeline,
        min_train_days=180,
        step_days=90,
        purge_days=PURGE_DAYS,
        embargo_days=EMBARGO_DAYS
    )

    fold_metrics = []
    pooled_data = {
        "y": [],
        "p_raw": [],
        "p_platt": [],
        "p_iso": [],
        "p_lr": [],
        "p_b0": [],
        "p_b1": [],
        "p_b2": []
    }

    # Expanding historical base rate accumulator for B0
    for fold in folds:
        train_mask = (valid["open_time_dt"] >= fold.train_start) & (valid["open_time_dt"] <= fold.train_end)
        test_mask  = (valid["open_time_dt"] >= fold.test_start)  & (valid["open_time_dt"] <= fold.test_end)

        train_data = valid[train_mask]
        test_data  = valid[test_mask]

        if len(train_data) < 200 or len(test_data) < 50:
            continue

        X_train = train_data[FEATURE_COLS].fillna(0.0).values
        y_train = train_data[target_col].astype(int).values
        X_test  = test_data[FEATURE_COLS].fillna(0.0).values
        y_test  = test_data[target_col].astype(int).values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Baseline 0: Expanding cumulative historical base rate from dataset start to train_end
        cum_train_mask = (valid["open_time_dt"] <= fold.train_end)
        p_b0 = float(np.mean(valid.loc[cum_train_mask, target_col].astype(int)))
        prob_b0 = np.full(len(y_test), p_b0)

        # Baseline 1: Fold Climatology (mean of training fold)
        p_b1 = float(np.mean(y_train))
        prob_b1 = np.full(len(y_test), p_b1)

        # Baseline 2: Macro Regime-Aware Base Rate
        tr_bull_mask = (train_data["btc_bull_regime"] > 0.5).values
        p_up_bull = float(np.mean(y_train[tr_bull_mask])) if np.sum(tr_bull_mask) > 10 else p_b1
        p_up_bear = float(np.mean(y_train[~tr_bull_mask])) if np.sum(~tr_bull_mask) > 10 else p_b1
        te_bull_mask = (test_data["btc_bull_regime"] > 0.5).values
        prob_b2 = np.where(te_bull_mask, p_up_bull, p_up_bear)

        # Model: Regularized LightGBM
        lgb = LGBMClassifier(
            n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
            reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbose=-1
        )

        # Cross-validation inside training fold ONLY for calibration
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_train_scores = cross_val_predict(lgb, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

        # 1. Platt Calibrator (strictly fit on train fold OOF)
        platt = PlattCalibrator().fit(oof_train_scores, y_train)

        # 2. Isotonic Calibrator (strictly fit on train fold OOF)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_train_scores, y_train)

        # Train primary model on full training fold
        lgb.fit(X_train, y_train)
        p_raw = lgb.predict_proba(X_test)[:, 1]
        p_platt = platt.predict_proba(p_raw) if platt.is_fitted else p_raw
        p_iso = iso.predict(p_raw)

        # Model: Logistic Regression (L2 + Scaler)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=200, solver="lbfgs"))
        lr.fit(X_train, y_train)
        p_lr = lr.predict_proba(X_test)[:, 1]

        # Compute Brier scores for fold
        bs_b0 = float(CalibrationEvaluator.brier_score(y_test, prob_b0))
        bs_b1 = float(CalibrationEvaluator.brier_score(y_test, prob_b1))
        bs_b2 = float(CalibrationEvaluator.brier_score(y_test, prob_b2))
        bs_raw = float(CalibrationEvaluator.brier_score(y_test, p_raw))
        bs_platt = float(CalibrationEvaluator.brier_score(y_test, p_platt))
        bs_iso = float(CalibrationEvaluator.brier_score(y_test, p_iso))
        bs_lr = float(CalibrationEvaluator.brier_score(y_test, p_lr))

        # ECEs for fold (10 bins)
        ece_raw, _, _ = CalibrationEvaluator.compute_calibration_curve(y_test, p_raw, n_bins=10)
        ece_platt, _, _ = CalibrationEvaluator.compute_calibration_curve(y_test, p_platt, n_bins=10)
        ece_iso, _, _ = CalibrationEvaluator.compute_calibration_curve(y_test, p_iso, n_bins=10)
        ece_lr, _, _ = CalibrationEvaluator.compute_calibration_curve(y_test, p_lr, n_bins=10)

        # BSS relative to B0, B1, B2 for fold
        def bss(bs_m, bs_ref):
            return float(1.0 - (bs_m / bs_ref)) if bs_ref > 1e-9 else 0.0

        auc_raw = float(roc_auc_score(y_test, p_raw))
        test_base_rate = float(np.mean(y_test))
        base_rate_drift = test_base_rate - p_b1

        # Murphy decomposition on Platt predictions
        murphy_fold = murphy_decomposition(y_test, p_platt, n_bins=10)

        fold_metrics.append({
            "fold_idx": fold.fold_index,
            "train_dates": f"{fold.train_start.date()} -> {fold.train_end.date()}",
            "test_dates": f"{fold.test_start.date()} -> {fold.test_end.date()}",
            "n_train": len(train_data),
            "n_test": len(test_data),
            "train_base_rate": p_b1,
            "test_base_rate": test_base_rate,
            "base_rate_drift": base_rate_drift,
            "auc": auc_raw,
            # Brier scores
            "bs_b0": bs_b0,
            "bs_b1": bs_b1,
            "bs_b2": bs_b2,
            "bs_raw": bs_raw,
            "bs_platt": bs_platt,
            "bs_iso": bs_iso,
            "bs_lr": bs_lr,
            # BSS vs B1
            "bss_raw_vs_b1": bss(bs_raw, bs_b1),
            "bss_platt_vs_b1": bss(bs_platt, bs_b1),
            "bss_iso_vs_b1": bss(bs_iso, bs_b1),
            "bss_lr_vs_b1": bss(bs_lr, bs_b1),
            # BSS vs B2
            "bss_platt_vs_b2": bss(bs_platt, bs_b2),
            # BSS vs B0
            "bss_platt_vs_b0": bss(bs_platt, bs_b0),
            # ECE
            "ece_raw": ece_raw,
            "ece_platt": ece_platt,
            "ece_iso": ece_iso,
            "ece_lr": ece_lr,
            # Murphy components
            "uncertainty": murphy_fold["uncertainty"],
            "reliability": murphy_fold["reliability"],
            "resolution": murphy_fold["resolution"]
        })

        pooled_data["y"].extend(y_test)
        pooled_data["p_raw"].extend(p_raw)
        pooled_data["p_platt"].extend(p_platt)
        pooled_data["p_iso"].extend(p_iso)
        pooled_data["p_lr"].extend(p_lr)
        pooled_data["p_b0"].extend(prob_b0)
        pooled_data["p_b1"].extend(prob_b1)
        pooled_data["p_b2"].extend(prob_b2)

    # Compute pooled metrics
    y_all = np.array(pooled_data["y"])
    p_raw_all = np.array(pooled_data["p_raw"])
    p_platt_all = np.array(pooled_data["p_platt"])
    p_iso_all = np.array(pooled_data["p_iso"])
    p_lr_all = np.array(pooled_data["p_lr"])
    p_b0_all = np.array(pooled_data["p_b0"])
    p_b1_all = np.array(pooled_data["p_b1"])
    p_b2_all = np.array(pooled_data["p_b2"])

    def bss_pooled(m_prob, ref_prob):
        bs_m = float(np.mean((m_prob - y_all) ** 2))
        bs_ref = float(np.mean((ref_prob - y_all) ** 2))
        return float(1.0 - (bs_m / bs_ref)) if bs_ref > 1e-9 else 0.0

    pooled_bs_platt = float(np.mean((p_platt_all - y_all) ** 2))
    pooled_bs_b0 = float(np.mean((p_b0_all - y_all) ** 2))
    pooled_bs_b1 = float(np.mean((p_b1_all - y_all) ** 2))
    pooled_bs_b2 = float(np.mean((p_b2_all - y_all) ** 2))

    pooled_ece_platt, _, _ = CalibrationEvaluator.compute_calibration_curve(y_all, p_platt_all, n_bins=10)
    pooled_ece_raw, _, _ = CalibrationEvaluator.compute_calibration_curve(y_all, p_raw_all, n_bins=10)
    pooled_ece_iso, _, _ = CalibrationEvaluator.compute_calibration_curve(y_all, p_iso_all, n_bins=10)
    pooled_ece_lr, _, _ = CalibrationEvaluator.compute_calibration_curve(y_all, p_lr_all, n_bins=10)

    bin_table_platt = compute_bin_table(y_all, p_platt_all, n_bins=10)
    murphy_pooled = murphy_decomposition(y_all, p_platt_all, n_bins=10)

    return {
        "target_name": target_name,
        "n_samples": len(y_all),
        "pooled_base_rate": float(np.mean(y_all)),
        "pooled_auc": float(roc_auc_score(y_all, p_raw_all)),
        "fold_metrics": fold_metrics,
        "pooled_summary": {
            "bs_platt": pooled_bs_platt,
            "bs_b0": pooled_bs_b0,
            "bs_b1": pooled_bs_b1,
            "bs_b2": pooled_bs_b2,
            "bss_platt_vs_b0": bss_pooled(p_platt_all, p_b0_all),
            "bss_platt_vs_b1": bss_pooled(p_platt_all, p_b1_all),
            "bss_platt_vs_b2": bss_pooled(p_platt_all, p_b2_all),
            "bss_raw_vs_b1": bss_pooled(p_raw_all, p_b1_all),
            "bss_iso_vs_b1": bss_pooled(p_iso_all, p_b1_all),
            "bss_lr_vs_b1": bss_pooled(p_lr_all, p_b1_all),
            "ece_platt": pooled_ece_platt,
            "ece_raw": pooled_ece_raw,
            "ece_iso": pooled_ece_iso,
            "ece_lr": pooled_ece_lr,
            "murphy": murphy_pooled
        },
        "bin_table_platt": bin_table_platt
    }

def run_full_audit():
    store = ResearchDataStore(DATA_DIR)
    logger.info("Loading dataset and calculating audit targets...")
    df, calendar_timeline = load_data(store)
    logger.info(f"Loaded {len(df):,} observations.")

    # Target sets to audit
    reach_targets = [
        ("reach_m08_14d", "BR-003.1 ARB Proxy: L=-8%, h=14D"),
        ("reach_m08_1d", "BR-003.1 L=-8%, h=1D"),
        ("reach_m08_3d", "BR-003.1 L=-8%, h=3D"),
        ("reach_m08_7d", "BR-003.1 L=-8%, h=7D"),
        ("reach_m08_30d", "BR-003.1 L=-8%, h=30D"),
        ("reach_m05_14d", "BR-003.1 L=-5%, h=14D"),
        ("reach_m10_14d", "BR-003.1 L=-10%, h=14D"),
        ("reach_m15_14d", "BR-003.1 L=-15%, h=14D"),
        ("reach_m20_14d", "BR-003.1 L=-20%, h=14D")
    ]

    reach_results = []
    for col, name in reach_targets:
        logger.info(f"Auditing reach target: {name}")
        res = audit_target(df, calendar_timeline, col, name)
        reach_results.append(res)

    # First passage targets
    fp_targets = [
        ("fp_arb_label", "BR-003.2 ARB Diagnostic: L=-6.77% vs U=+5.05% (30D)"),
        ("fp_sym5_label", "BR-003.2 Symmetric: L=-5% vs U=+5% (30D)"),
        ("fp_sym10_label", "BR-003.2 Symmetric: L=-10% vs U=+10% (30D)"),
        ("fp_asym10_25_label", "BR-003.2 Asymmetric: L=-10% vs U=+25% (30D)")
    ]

    fp_results = []
    for col, name in fp_targets:
        logger.info(f"Auditing first-passage target: {name}")
        # Filter for resolved directional exits only
        resolved_mask = (df[col] == TargetGenerator.FP_LOWER_FIRST) | (df[col] == TargetGenerator.FP_UPPER_FIRST)
        valid_fp = df[resolved_mask].copy()
        valid_fp["target_y"] = (valid_fp[col] == TargetGenerator.FP_LOWER_FIRST).astype(int)
        res = audit_target(valid_fp, calendar_timeline, "target_y", name)
        fp_results.append(res)

    return reach_results, fp_results

if __name__ == "__main__":
    import json
    reach_res, fp_res = run_full_audit()
    
    # Save raw audit json for artifact generation
    out_path = backend_dir / "audit_br003_raw.json"
    with open(out_path, "w") as f:
        json.dump({"reach_results": reach_res, "fp_results": fp_res}, f, indent=2, default=str)
    print(f"Audit completed successfully. Results saved to {out_path}")
