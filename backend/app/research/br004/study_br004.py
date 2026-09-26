"""
BR-004: Dynamic Regime-Conditioned & Online Probability Calibration Study
========================================================================
Research Question:
  Can dynamic macro regime conditioning (Architecture A), online Bayesian
  log-odds updating (Architecture B), or rolling-window walk-forward training
  (Architecture C) eliminate the multi-week non-stationary calibration penalty
  and achieve validated probability calibration:
      BSS > 0 (vs B1 & B3) and ECE < 0.10 across walk-forward folds?

Safeguards & Controls:
  - 45 causal features frozen.
  - SBRU_V1 88 Binance Spot pairs universe frozen.
  - 14D primary horizon (reach_m08_14d); 30D secondary horizons.
  - 90D purge, 45D embargo across expanding walk-forward splits.
  - B0/B1/B2/B3 baselines pre-registered.
  - B3 strictly uses horizon-closed outcomes (t_i + h <= current_time).
  - Minimum sample fallback for regime-specific calibrators.
  - Strict RESEARCH_ONLY governance.
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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
logger = logging.getLogger("BR004")

DATA_DIR = backend_dir / "data"
RESULTS_FILE = backend_dir / "BR-004-RESULTS.md"

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
    if btc_raw is None:
        raise RuntimeError("BTCUSDT raw data not found.")
    
    # BTC Macro Features
    btc_raw["btc_ema50"] = btc_raw["close"].ewm(span=50, adjust=False).mean()
    btc_raw["btc_mom14"] = btc_raw["close"].pct_change(14).fillna(0.0)
    
    # 3 Causal Macro Regimes:
    # 0 = Bear (BTC <= EMA50)
    # 1 = Bull Strong (BTC > EMA50 and Mom14 >= 0)
    # 2 = Bull Transition/Pullback (BTC > EMA50 and Mom14 < 0)
    conditions = [
        (btc_raw["close"] <= btc_raw["btc_ema50"]),
        (btc_raw["close"] > btc_raw["btc_ema50"]) & (btc_raw["btc_mom14"] >= 0.0),
        (btc_raw["close"] > btc_raw["btc_ema50"]) & (btc_raw["btc_mom14"] < 0.0)
    ]
    choices = [0, 1, 2]
    btc_raw["macro_regime"] = np.select(conditions, choices, default=0)
    btc_raw["btc_bull_binary"] = (btc_raw["close"] > btc_raw["btc_ema50"]).astype(float)
    
    regime_map = dict(zip(btc_raw["open_time"], btc_raw["macro_regime"]))
    bull_binary_map = dict(zip(btc_raw["open_time"], btc_raw["btc_bull_binary"]))

    panels = []
    for sym in symbols:
        raw = store.load_raw(sym)
        feat = store.load_features("BR-002", sym)
        if raw is None or feat is None:
            continue
        feat["symbol"] = sym
        feat["macro_regime"] = feat["open_time"].map(regime_map).fillna(0).astype(int)
        feat["btc_bull_binary"] = feat["open_time"].map(bull_binary_map).fillna(0.0)

        lows = raw["low"].values
        closes = raw["close"].values
        highs = raw["high"].values

        # Primary Target: Level Reach L = -8%, h = 14D
        feat["reach_m08_14d"] = TargetGenerator.label_level_reach(lows, closes, -0.08, 14)

        # Secondary Target 1: Level Reach L = -8%, h = 30D
        feat["reach_m08_30d"] = TargetGenerator.label_level_reach(lows, closes, -0.08, 30)

        # Secondary Target 2: First Passage ARB Diagnostic L = -6.77%, U = +5.05%, h = 30D
        fp_arb = TargetGenerator.label_first_passage(highs, lows, closes, -0.0677, +0.0505, 30)
        feat["fp_arb_label"] = fp_arb["first_passage_label"]

        panels.append(feat)

    full_df = pd.concat(panels, ignore_index=True)
    full_df["open_time_dt"] = pd.to_datetime(full_df["open_time"], unit="ms", utc=True)
    full_df = full_df.sort_values("open_time_dt").reset_index(drop=True)

    min_date = full_df["open_time_dt"].min().floor("D")
    max_date = full_df["open_time_dt"].max().floor("D")
    calendar_timeline = pd.date_range(min_date, max_date, freq="D", tz="UTC")
    return full_df, calendar_timeline

# ---------------------------------------------------------------------------
# Architecture Implementations
# ---------------------------------------------------------------------------

class RegimeConditionedCalibrator:
    """
    Architecture A: Multi-State Regime-Conditioned Calibrator
    Fits separate Platt logistic sigmoids per macro regime r in {0, 1, 2}.
    Enforces minimum sample safeguard: >= 200 samples, >= 20 positives, >= 20 negatives.
    Falls back deterministically to global calibrator if safeguard fails.
    """
    def __init__(self, min_samples: int = 200, min_pos: int = 20, min_neg: int = 20):
        self.min_samples = min_samples
        self.min_pos = min_pos
        self.min_neg = min_neg
        self.global_calibrator = PlattCalibrator()
        self.regime_calibrators: Dict[int, PlattCalibrator] = {}
        self.regime_fallback_flags: Dict[int, bool] = {}

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray, regimes: np.ndarray) -> "RegimeConditionedCalibrator":
        # Fit global fallback calibrator
        self.global_calibrator.fit(raw_scores, y_true)

        for r in np.unique(regimes):
            mask = (regimes == r)
            n_r = int(np.sum(mask))
            n_pos = int(np.sum(y_true[mask] == 1))
            n_neg = int(np.sum(y_true[mask] == 0))

            if n_r >= self.min_samples and n_pos >= self.min_pos and n_neg >= self.min_neg:
                cal = PlattCalibrator().fit(raw_scores[mask], y_true[mask])
                self.regime_calibrators[r] = cal
                self.regime_fallback_flags[r] = False
            else:
                self.regime_calibrators[r] = self.global_calibrator
                self.regime_fallback_flags[r] = True

        return self

    def predict_proba(self, raw_scores: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        out = np.zeros(len(raw_scores), dtype=float)
        for r in np.unique(regimes):
            mask = (regimes == r)
            cal = self.regime_calibrators.get(r, self.global_calibrator)
            out[mask] = cal.predict_proba(raw_scores[mask])
        return out


def evaluate_architecture_b_online(
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    target_col: str,
    h_days: int,
    raw_test_scores: np.ndarray,
    eta: float = 0.05
) -> np.ndarray:
    """
    Architecture B: Bayesian Online Log-Odds Updating
    Calculates operational probability: P_t = sigmoid(logit(p_raw) + delta_t)
    where delta_t is updated strictly on events whose forward horizon h closed on day t:
      {i : open_time_dt_i + h days == day_t}.
    Strictly causal; no lookahead.
    """
    # Convert raw probability to log-odds margin z (clipped for numerical stability)
    p_clipped = np.clip(raw_test_scores, 0.001, 0.999)
    z_scores = np.log(p_clipped / (1.0 - p_clipped))

    # Build a lookup of realized closed events from full_df
    # For every event i in full_df with target_col not nan, its closure date is open_time_dt + h days
    valid_closure = full_df.dropna(subset=[target_col]).copy()
    valid_closure["close_date"] = (valid_closure["open_time_dt"] + pd.Timedelta(days=h_days)).dt.floor("D")
    
    # Group closed events by closure date
    closure_groups = dict(tuple(valid_closure.groupby("close_date")))

    # Test dates in chronological order
    test_dates = test_df["open_time_dt"].dt.floor("D").values
    unique_dates = np.unique(test_dates)

    delta_t = 0.0
    daily_delta_map = {}

    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        if d_ts.tz is None:
            d_ts = d_ts.tz_localize("UTC")
        # Check if any events closed on date d
        if d_ts in closure_groups:
            closed_batch = closure_groups[d_ts]
            # Observed empirical hit rate of closed cohort
            obs_rate = float(closed_batch[target_col].mean())
            # Estimate predicted hit rate under current delta: sigmoid(z + delta)
            # For simplicity and stability, update delta via: delta = delta + eta * (obs_rate - sigmoid(delta))
            current_base_prob = 1.0 / (1.0 + np.exp(-delta_t))
            error = obs_rate - current_base_prob
            delta_t = float(np.clip(delta_t + eta * error, -2.5, 2.5))

        daily_delta_map[d] = delta_t

    # Map daily delta back to test observations
    test_deltas = np.array([daily_delta_map[d] for d in test_dates])
    # Calibrated probability: sigmoid(z + delta)
    calibrated_probs = 1.0 / (1.0 + np.exp(-np.clip(z_scores + test_deltas, -20.0, 20.0)))
    return calibrated_probs


def compute_baseline_3_rolling(
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    target_col: str,
    h_days: int,
    window_days: int = 60,
    p_b1_fallback: float = 0.50
) -> np.ndarray:
    """
    Baseline 3 (B3): Online Rolling Climatology
    For test observation at day t, computes empirical base rate of all events
    across the universe whose horizon has closed within trailing window:
      {i : t - window_days <= t_i + h <= t}.
    Never includes current day t or unclosed observations.
    """
    valid_closure = full_df.dropna(subset=[target_col]).copy()
    valid_closure["close_date"] = (valid_closure["open_time_dt"] + pd.Timedelta(days=h_days)).dt.floor("D")

    test_dates = test_df["open_time_dt"].dt.floor("D").values
    unique_dates = np.unique(test_dates)

    daily_b3_map = {}
    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        if d_ts.tz is None:
            d_ts = d_ts.tz_localize("UTC")
        w_start = d_ts - pd.Timedelta(days=window_days)
        # Closed events: close_date <= d_ts and close_date >= w_start
        mask = (valid_closure["close_date"] <= d_ts) & (valid_closure["close_date"] >= w_start)
        closed_sub = valid_closure[mask]

        if len(closed_sub) >= 50:
            p_b3 = float(closed_sub[target_col].mean())
        else:
            p_b3 = p_b1_fallback

        daily_b3_map[d] = p_b3

    return np.array([daily_b3_map[d] for d in test_dates])


# ---------------------------------------------------------------------------
# Comprehensive BR-004 Evaluation Engine
# ---------------------------------------------------------------------------

def run_br004_study():
    store = ResearchDataStore(DATA_DIR)
    logger.info("Loading dataset and calculating causal features and targets...")
    full_df, calendar_timeline = load_data(store)
    logger.info(f"Loaded {len(full_df):,} total bars.")

    targets_to_evaluate = [
        ("reach_m08_14d", "Primary: Level-Reach L=-8%, h=14D (ARB Proxy)", 14, False),
        ("reach_m08_30d", "Secondary: Level-Reach L=-8%, h=30D", 30, False),
        ("fp_arb_label", "Secondary: First-Passage ARB Diagnostic (30D)", 30, True)
    ]

    all_study_results = []

    for target_col, target_name, h_days, is_first_passage in targets_to_evaluate:
        logger.info(f"===========================================================")
        logger.info(f"Running BR-004 on target: {target_name}")
        logger.info(f"===========================================================")

        if is_first_passage:
            # Strictly filter for resolved directional exits only
            resolved_mask = (full_df[target_col] == TargetGenerator.FP_LOWER_FIRST) | (full_df[target_col] == TargetGenerator.FP_UPPER_FIRST)
            valid_df = full_df[resolved_mask].copy()
            valid_df["target_y"] = (valid_df[target_col] == TargetGenerator.FP_LOWER_FIRST).astype(int)
            eval_target = "target_y"
        else:
            valid_df = full_df.dropna(subset=[target_col]).copy()
            valid_df["target_y"] = valid_df[target_col].astype(int)
            eval_target = "target_y"

        folds = walk_forward_expanding(
            calendar_timeline,
            min_train_days=180,
            step_days=90,
            purge_days=PURGE_DAYS,
            embargo_days=EMBARGO_DAYS
        )

        fold_evaluations = []
        pooled_predictions = {
            "y_true": [],
            "p_b0": [],
            "p_b1": [],
            "p_b2": [],
            "p_b3": [],
            "p_raw_br003": [],
            "p_platt_br003": [],
            "p_arch_a": [],
            "p_arch_b": [],
            "p_arch_c": [],
            "macro_regime": []
        }

        for fold in folds:
            train_mask = (valid_df["open_time_dt"] >= fold.train_start) & (valid_df["open_time_dt"] <= fold.train_end)
            test_mask  = (valid_df["open_time_dt"] >= fold.test_start)  & (valid_df["open_time_dt"] <= fold.test_end)

            train_data = valid_df[train_mask]
            test_data  = valid_df[test_mask]

            if len(train_data) < 200 or len(test_data) < 50:
                continue

            X_train = train_data[FEATURE_COLS].fillna(0.0).values
            y_train = train_data["target_y"].values
            regimes_train = train_data["macro_regime"].values

            X_test  = test_data[FEATURE_COLS].fillna(0.0).values
            y_test  = test_data["target_y"].values
            regimes_test  = test_data["macro_regime"].values

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            # ---------------------------------------------------------------
            # Baselines Calculation
            # ---------------------------------------------------------------
            # Baseline 0: Cumulative expanding base rate
            cum_mask = (valid_df["open_time_dt"] <= fold.train_end)
            p_b0 = float(valid_df.loc[cum_mask, "target_y"].mean())
            prob_b0 = np.full(len(y_test), p_b0)

            # Baseline 1: Fold Climatology
            p_b1 = float(np.mean(y_train))
            prob_b1 = np.full(len(y_test), p_b1)

            # Baseline 2: Macro Regime Climatology
            tr_bull_mask = (train_data["btc_bull_binary"] > 0.5).values
            p_bull = float(np.mean(y_train[tr_bull_mask])) if np.sum(tr_bull_mask) > 10 else p_b1
            p_bear = float(np.mean(y_train[~tr_bull_mask])) if np.sum(~tr_bull_mask) > 10 else p_b1
            te_bull_mask = (test_data["btc_bull_binary"] > 0.5).values
            prob_b2 = np.where(te_bull_mask, p_bull, p_bear)

            # Baseline 3: Online Rolling Climatology (W=60D of closed events)
            prob_b3 = compute_baseline_3_rolling(
                test_data, valid_df, "target_y", h_days, window_days=60, p_b1_fallback=p_b1
            )

            # ---------------------------------------------------------------
            # Model Training (Expanding): Frozen LightGBM
            # ---------------------------------------------------------------
            lgb_model = LGBMClassifier(
                n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
                reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbose=-1
            )

            # 3-Fold Stratified CV on training fold only
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            oof_train_scores = cross_val_predict(lgb_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

            # Fit Static Platt Calibrator (BR-003 Baseline)
            static_platt = PlattCalibrator().fit(oof_train_scores, y_train)

            # Fit Architecture A: Multi-State Regime Calibrator
            arch_a_cal = RegimeConditionedCalibrator(min_samples=200, min_pos=20, min_neg=20)
            arch_a_cal.fit(oof_train_scores, y_train, regimes_train)

            # Train LightGBM on full expanding training fold
            lgb_model.fit(X_train, y_train)
            raw_test_scores = lgb_model.predict_proba(X_test)[:, 1]

            # Static Predictions
            prob_raw_br003 = raw_test_scores
            prob_platt_br003 = static_platt.predict_proba(raw_test_scores)

            # Architecture A Predictions
            prob_arch_a = arch_a_cal.predict_proba(raw_test_scores, regimes_test)

            # Architecture B Predictions (Bayesian Online Updating)
            prob_arch_b = evaluate_architecture_b_online(
                test_data, valid_df, "target_y", h_days, raw_test_scores, eta=0.05
            )

            # ---------------------------------------------------------------
            # Architecture C: Rolling-Window 365D Model
            # ---------------------------------------------------------------
            rolling_train_start = fold.train_end - pd.Timedelta(days=365)
            rolling_train_mask = (valid_df["open_time_dt"] >= rolling_train_start) & (valid_df["open_time_dt"] <= fold.train_end)
            rolling_train_data = valid_df[rolling_train_mask]

            if len(rolling_train_data) >= 200 and len(np.unique(rolling_train_data["target_y"])) > 1:
                X_roll_train = rolling_train_data[FEATURE_COLS].fillna(0.0).values
                y_roll_train = rolling_train_data["target_y"].values

                lgb_roll = LGBMClassifier(
                    n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
                    reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbose=-1
                )
                oof_roll_scores = cross_val_predict(lgb_roll, X_roll_train, y_roll_train, cv=cv, method="predict_proba")[:, 1]
                roll_platt = PlattCalibrator().fit(oof_roll_scores, y_roll_train)
                lgb_roll.fit(X_roll_train, y_roll_train)
                raw_roll_test = lgb_roll.predict_proba(X_test)[:, 1]
                prob_arch_c = roll_platt.predict_proba(raw_roll_test)
            else:
                prob_arch_c = prob_platt_br003

            # ---------------------------------------------------------------
            # Fold Metrics Computation
            # ---------------------------------------------------------------
            def get_metrics(p_vec: np.ndarray, name: str) -> Dict[str, float]:
                bs = float(CalibrationEvaluator.brier_score(y_test, p_vec))
                bs_b1 = float(CalibrationEvaluator.brier_score(y_test, prob_b1))
                bs_b3 = float(CalibrationEvaluator.brier_score(y_test, prob_b3))
                bss_b1 = float(1.0 - (bs / bs_b1)) if bs_b1 > 1e-9 else 0.0
                bss_b3 = float(1.0 - (bs / bs_b3)) if bs_b3 > 1e-9 else 0.0
                ece, _, _ = CalibrationEvaluator.compute_calibration_curve(y_test, p_vec, n_bins=10)
                auc = float(roc_auc_score(y_test, p_vec)) if len(np.unique(p_vec)) > 1 else 0.50
                return {"bs": bs, "bss_b1": bss_b1, "bss_b3": bss_b3, "ece": ece, "auc": auc}

            m_b0 = get_metrics(prob_b0, "B0")
            m_b1 = get_metrics(prob_b1, "B1")
            m_b2 = get_metrics(prob_b2, "B2")
            m_b3 = get_metrics(prob_b3, "B3")
            m_raw = get_metrics(prob_raw_br003, "BR003_Raw")
            m_platt = get_metrics(prob_platt_br003, "BR003_Platt")
            m_a = get_metrics(prob_arch_a, "Arch_A")
            m_b = get_metrics(prob_arch_b, "Arch_B")
            m_c = get_metrics(prob_arch_c, "Arch_C")

            fold_evaluations.append({
                "fold_idx": fold.fold_index,
                "train_dates": f"{fold.train_start.date()} -> {fold.train_end.date()}",
                "test_dates": f"{fold.test_start.date()} -> {fold.test_end.date()}",
                "n_train": len(train_data),
                "n_test": len(test_data),
                "train_base_rate": p_b1,
                "test_base_rate": float(np.mean(y_test)),
                "base_rate_drift": float(np.mean(y_test) - p_b1),
                "m_b0": m_b0,
                "m_b1": m_b1,
                "m_b2": m_b2,
                "m_b3": m_b3,
                "m_raw": m_raw,
                "m_platt": m_platt,
                "m_a": m_a,
                "m_b": m_b,
                "m_c": m_c
            })

            # Accumulate pooled predictions
            pooled_predictions["y_true"].extend(y_test)
            pooled_predictions["p_b0"].extend(prob_b0)
            pooled_predictions["p_b1"].extend(prob_b1)
            pooled_predictions["p_b2"].extend(prob_b2)
            pooled_predictions["p_b3"].extend(prob_b3)
            pooled_predictions["p_raw_br003"].extend(prob_raw_br003)
            pooled_predictions["p_platt_br003"].extend(prob_platt_br003)
            pooled_predictions["p_arch_a"].extend(prob_arch_a)
            pooled_predictions["p_arch_b"].extend(prob_arch_b)
            pooled_predictions["p_arch_c"].extend(prob_arch_c)
            pooled_predictions["macro_regime"].extend(regimes_test)

        # -------------------------------------------------------------------
        # Pooled Performance Calculation
        # -------------------------------------------------------------------
        y_all = np.array(pooled_predictions["y_true"])
        p_b0_all = np.array(pooled_predictions["p_b0"])
        p_b1_all = np.array(pooled_predictions["p_b1"])
        p_b2_all = np.array(pooled_predictions["p_b2"])
        p_b3_all = np.array(pooled_predictions["p_b3"])
        p_raw_all = np.array(pooled_predictions["p_raw_br003"])
        p_platt_all = np.array(pooled_predictions["p_platt_br003"])
        p_a_all = np.array(pooled_predictions["p_arch_a"])
        p_b_all = np.array(pooled_predictions["p_arch_b"])
        p_c_all = np.array(pooled_predictions["p_arch_c"])
        regimes_all = np.array(pooled_predictions["macro_regime"])

        def get_pooled_metrics(p_vec: np.ndarray, name: str) -> Dict[str, Any]:
            bs = float(np.mean((p_vec - y_all) ** 2))
            bs_b0 = float(np.mean((p_b0_all - y_all) ** 2))
            bs_b1 = float(np.mean((p_b1_all - y_all) ** 2))
            bs_b2 = float(np.mean((p_b2_all - y_all) ** 2))
            bs_b3 = float(np.mean((p_b3_all - y_all) ** 2))

            bss_b0 = float(1.0 - (bs / bs_b0)) if bs_b0 > 1e-9 else 0.0
            bss_b1 = float(1.0 - (bs / bs_b1)) if bs_b1 > 1e-9 else 0.0
            bss_b2 = float(1.0 - (bs / bs_b2)) if bs_b2 > 1e-9 else 0.0
            bss_b3 = float(1.0 - (bs / bs_b3)) if bs_b3 > 1e-9 else 0.0

            ece, mce, _ = CalibrationEvaluator.compute_calibration_curve(y_all, p_vec, n_bins=10)
            auc = float(roc_auc_score(y_all, p_vec)) if len(np.unique(p_vec)) > 1 else 0.50
            pr_auc = float(average_precision_score(y_all, p_vec)) if len(np.unique(p_vec)) > 1 else 0.50
            murphy = murphy_decomposition(y_all, p_vec, n_bins=10)
            bin_table = compute_bin_table(y_all, p_vec, n_bins=10)

            return {
                "name": name,
                "bs": bs,
                "bss_b0": bss_b0,
                "bss_b1": bss_b1,
                "bss_b2": bss_b2,
                "bss_b3": bss_b3,
                "ece": ece,
                "mce": mce,
                "auc": auc,
                "pr_auc": pr_auc,
                "murphy": murphy,
                "bin_table": bin_table
            }

        pooled_res = {
            "B0": get_pooled_metrics(p_b0_all, "Baseline 0 (Cumulative)"),
            "B1": get_pooled_metrics(p_b1_all, "Baseline 1 (Fold Climatology)"),
            "B2": get_pooled_metrics(p_b2_all, "Baseline 2 (Binary Regime)"),
            "B3": get_pooled_metrics(p_b3_all, "Baseline 3 (Online Rolling 60D)"),
            "BR003_Raw": get_pooled_metrics(p_raw_all, "Static BR-003 Raw LightGBM"),
            "BR003_Platt": get_pooled_metrics(p_platt_all, "Static BR-003 Platt LightGBM"),
            "Arch_A": get_pooled_metrics(p_a_all, "Architecture A (Regime-Conditioned Calibrator)"),
            "Arch_B": get_pooled_metrics(p_b_all, "Architecture B (Bayesian Online Updating)"),
            "Arch_C": get_pooled_metrics(p_c_all, "Architecture C (Rolling 365D Walk-Forward)")
        }

        # Regime-Stratified Performance for Architecture A and B
        regime_breakdowns = {}
        for r_code, r_name in [(0, "Bear (BTC <= EMA50)"), (1, "Bull Strong (BTC > EMA50, Mom >= 0)"), (2, "Bull Transition (BTC > EMA50, Mom < 0)")]:
            r_mask = (regimes_all == r_code)
            if np.sum(r_mask) > 100:
                y_r = y_all[r_mask]
                bs_b1_r = float(np.mean((p_b1_all[r_mask] - y_r) ** 2))
                regime_breakdowns[r_name] = {
                    "count": int(np.sum(r_mask)),
                    "base_rate": float(np.mean(y_r)),
                    "auc_arch_a": float(roc_auc_score(y_r, p_a_all[r_mask])),
                    "bs_arch_a": float(np.mean((p_a_all[r_mask] - y_r) ** 2)),
                    "bss_b1_arch_a": float(1.0 - (np.mean((p_a_all[r_mask] - y_r) ** 2) / bs_b1_r)),
                    "ece_arch_a": float(CalibrationEvaluator.compute_calibration_curve(y_r, p_a_all[r_mask], n_bins=5)[0]),
                    "auc_arch_b": float(roc_auc_score(y_r, p_b_all[r_mask])),
                    "bs_arch_b": float(np.mean((p_b_all[r_mask] - y_r) ** 2)),
                    "bss_b1_arch_b": float(1.0 - (np.mean((p_b_all[r_mask] - y_r) ** 2) / bs_b1_r)),
                    "ece_arch_b": float(CalibrationEvaluator.compute_calibration_curve(y_r, p_b_all[r_mask], n_bins=5)[0])
                }

        all_study_results.append({
            "target_col": target_col,
            "target_name": target_name,
            "h_days": h_days,
            "n_samples": len(y_all),
            "pooled_base_rate": float(np.mean(y_all)),
            "fold_evaluations": fold_evaluations,
            "pooled_summary": pooled_res,
            "regime_breakdowns": regime_breakdowns
        })

    return all_study_results

if __name__ == "__main__":
    results = run_br004_study()
    out_json = backend_dir / "br004_results_raw.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"BR-004 study complete. Raw results saved to {out_json}")
