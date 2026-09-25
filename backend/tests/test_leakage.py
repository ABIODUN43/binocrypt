"""
Leakage & Causality Test Suite
==============================
Enforces strict quantitative separation:
  - Features cannot know the future (Group A)
  - Labels must describe the future (Group B)
  - Train and test splits must maintain temporal isolation with purge & embargo (Group C)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from app.research.features import FeatureExtractor
from app.research.targets import TargetGenerator
from app.research.labels.causal_hmm import CausalHMMRegimeLabeller, HMMMode
from app.research.walk_forward import (
    walk_forward_expanding,
    walk_forward_rolling,
    PURGE_DAYS,
    EMBARGO_DAYS,
)
from app.research.universe.providers import SBRUProvider, SBRU_V1_LISTED_FROM


def _generate_synthetic_klines(n: int = 150, base_price: float = 100.0, seed: int = 42):
    """
    Generate synthetic klines in Binance format:
    [open_time, open, high, low, close, volume, close_time, quote_volume, trade_count, taker_buy_base, taker_buy_quote, ignore]
    """
    np.random.seed(seed)
    start_ts = 1672531200000  # 2023-01-01 00:00:00 UTC
    day_ms = 86400000
    
    klines = []
    price = base_price
    for i in range(n):
        open_time = start_ts + i * day_ms
        close_time = open_time + day_ms - 1
        ret = np.random.normal(0.0005, 0.02)
        close_p = max(1.0, price * (1.0 + ret))
        high_p = max(price, close_p) * (1.0 + abs(np.random.normal(0.005, 0.005)))
        low_p = min(price, close_p) * (1.0 - abs(np.random.normal(0.005, 0.005)))
        vol = float(np.random.uniform(1000, 5000))
        quote_vol = vol * close_p
        trade_count = int(np.random.randint(500, 2000))
        taker_base = vol * float(np.random.uniform(0.4, 0.6))
        taker_quote = taker_base * close_p
        
        klines.append([
            open_time, price, high_p, low_p, close_p, vol,
            close_time, quote_vol, trade_count, taker_base, taker_quote, "0"
        ])
        price = close_p
    return klines


# =====================================================================
# Group A — Feature Causality Tests
# Features at prior rows <= t MUST NOT change when future data is altered
# =====================================================================

def test_features_ignore_future_close():
    klines = _generate_synthetic_klines(n=100)
    feat_orig = FeatureExtractor.extract_features(klines)
    
    # Append a future candle with a 10x close shock
    future_candle = list(klines[-1])
    future_candle[0] += 86400000
    future_candle[6] += 86400000
    future_candle[4] = future_candle[4] * 10.0  # extreme close
    
    extended_klines = klines + [future_candle]
    feat_extended = FeatureExtractor.extract_features(extended_klines)
    
    numeric_cols = [c for c in feat_orig.columns if c not in ["timestamp", "close"]]
    for col in numeric_cols:
        orig_vals = feat_orig[col].values
        ext_vals = feat_extended[col].iloc[:len(klines)].values
        np.testing.assert_allclose(
            orig_vals, ext_vals, rtol=1e-5, atol=1e-5,
            err_msg=f"Feature {col} changed when future close was modified!"
        )


def test_features_ignore_future_high():
    klines = _generate_synthetic_klines(n=100)
    feat_orig = FeatureExtractor.extract_features(klines)
    
    # Append future candle with 50x spike high
    future_candle = list(klines[-1])
    future_candle[0] += 86400000
    future_candle[6] += 86400000
    future_candle[2] = future_candle[2] * 50.0  # huge spike high
    
    extended_klines = klines + [future_candle]
    feat_extended = FeatureExtractor.extract_features(extended_klines)
    
    numeric_cols = [c for c in feat_orig.columns if c not in ["timestamp", "close"]]
    for col in numeric_cols:
        orig_vals = feat_orig[col].values
        ext_vals = feat_extended[col].iloc[:len(klines)].values
        np.testing.assert_allclose(
            orig_vals, ext_vals, rtol=1e-5, atol=1e-5,
            err_msg=f"Feature {col} leaked future high!"
        )


def test_features_ignore_future_low():
    klines = _generate_synthetic_klines(n=100)
    feat_orig = FeatureExtractor.extract_features(klines)
    
    # Append future candle with near-zero flash crash low
    future_candle = list(klines[-1])
    future_candle[0] += 86400000
    future_candle[6] += 86400000
    future_candle[3] = 0.0001
    
    extended_klines = klines + [future_candle]
    feat_extended = FeatureExtractor.extract_features(extended_klines)
    
    numeric_cols = [c for c in feat_orig.columns if c not in ["timestamp", "close"]]
    for col in numeric_cols:
        orig_vals = feat_orig[col].values
        ext_vals = feat_extended[col].iloc[:len(klines)].values
        np.testing.assert_allclose(
            orig_vals, ext_vals, rtol=1e-5, atol=1e-5,
            err_msg=f"Feature {col} leaked future low!"
        )


def test_features_ignore_future_volume():
    klines = _generate_synthetic_klines(n=100)
    feat_orig = FeatureExtractor.extract_features(klines)
    
    # Append future candle with 1000x volume
    future_candle = list(klines[-1])
    future_candle[0] += 86400000
    future_candle[6] += 86400000
    future_candle[5] = future_candle[5] * 1000.0
    
    extended_klines = klines + [future_candle]
    feat_extended = FeatureExtractor.extract_features(extended_klines)
    
    numeric_cols = [c for c in feat_orig.columns if c not in ["timestamp", "close"]]
    for col in numeric_cols:
        orig_vals = feat_orig[col].values
        ext_vals = feat_extended[col].iloc[:len(klines)].values
        np.testing.assert_allclose(
            orig_vals, ext_vals, rtol=1e-5, atol=1e-5,
            err_msg=f"Feature {col} leaked future volume!"
        )


def test_ath_causal():
    klines = _generate_synthetic_klines(n=80)
    feat_orig = FeatureExtractor.extract_features(klines)
    
    # Append candle with 1000x ATH
    future_candle = list(klines[-1])
    future_candle[0] += 86400000
    future_candle[2] = 999999.0
    
    extended = klines + [future_candle]
    feat_ext = FeatureExtractor.extract_features(extended)
    
    # Prior ath_to_date and dist_from_ath must remain exactly identical
    np.testing.assert_allclose(
        feat_orig["ath_to_date"].values,
        feat_ext["ath_to_date"].iloc[:len(klines)].values,
        rtol=1e-6, err_msg="ath_to_date leaked future ATH!"
    )
    np.testing.assert_allclose(
        feat_orig["dist_from_ath"].values,
        feat_ext["dist_from_ath"].iloc[:len(klines)].values,
        rtol=1e-6, err_msg="dist_from_ath leaked future ATH!"
    )


def test_atl_causal():
    klines = _generate_synthetic_klines(n=80)
    feat_orig = FeatureExtractor.extract_features(klines)
    
    # Append candle with all-time low near 0
    future_candle = list(klines[-1])
    future_candle[0] += 86400000
    future_candle[3] = 0.000001
    
    extended = klines + [future_candle]
    feat_ext = FeatureExtractor.extract_features(extended)
    
    np.testing.assert_allclose(
        feat_orig["atl_to_date"].values,
        feat_ext["atl_to_date"].iloc[:len(klines)].values,
        rtol=1e-6, err_msg="atl_to_date leaked future ATL!"
    )
    np.testing.assert_allclose(
        feat_orig["dist_from_atl"].values,
        feat_ext["dist_from_atl"].iloc[:len(klines)].values,
        rtol=1e-6, err_msg="dist_from_atl leaked future ATL!"
    )


def test_hmm_causal():
    """Verify that causal HMM regime labeller in FORWARD_FILTER mode does not retroactively change past states."""
    np.random.seed(42)
    log_rets = np.random.normal(0, 0.02, 120)
    
    labeller = CausalHMMRegimeLabeller(mode=HMMMode.FORWARD_FILTER, min_train_len=60, random_state=42)
    states_before = labeller.label_causal(log_rets, train_end_idx=80)
    
    # Extend series with high volatility future
    extended_rets = np.concatenate([log_rets, np.random.normal(0, 0.10, 30)])
    states_after = labeller.label_causal(extended_rets, train_end_idx=80)
    
    np.testing.assert_array_equal(
        states_before,
        states_after[:len(log_rets)],
        err_msg="HMM states in prior window changed when future observations were appended!"
    )


def test_rolling_normalization_causal():
    """Verify rolling volatility and rolling drawdown strictly use observations <= t."""
    klines = _generate_synthetic_klines(n=90)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count", "taker_base", "taker_quote", "ignore"
    ])
    feat = FeatureExtractor.extract_features(klines)
    
    # Recompute dd_from_30d_high manually for row 50
    sub_high = df["high"].iloc[21:51] # 30 periods ending at index 50
    manual_30_max = sub_high.max()
    expected_dd = (df["close"].iloc[50] - manual_30_max) / (manual_30_max + 1e-9)
    actual_dd = feat["dd_from_30d_high"].iloc[50]
    
    np.testing.assert_allclose(actual_dd, expected_dd, rtol=1e-5)


# =====================================================================
# Group B — Label Correctness Tests
# Labels MUST describe the future and react appropriately to future path
# =====================================================================

def test_mfe_changes_within_horizon():
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])
    highs = closes * 1.01
    horizon = 3
    
    mfe_orig = TargetGenerator.generate_mfe(highs, closes, horizon=horizon)
    
    # Increase future high at t=2 (within horizon of t=0, which spans t+1..t+3 => indices 1,2,3)
    highs_modified = highs.copy()
    highs_modified[2] = 200.0 # 2x spike
    
    mfe_mod = TargetGenerator.generate_mfe(highs_modified, closes, horizon=horizon)
    
    assert mfe_mod[0] > mfe_orig[0], "MFE at t=0 should increase when High at t=2 is increased!"
    assert np.isclose(mfe_mod[0], (200.0 / 100.0) - 1.0)


def test_mfe_unchanged_beyond_horizon():
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])
    highs = closes * 1.01
    horizon = 3
    
    mfe_orig = TargetGenerator.generate_mfe(highs, closes, horizon=horizon)
    
    # Modify high at t=5 (outside horizon of t=0, since horizon covers 1, 2, 3)
    highs_modified = highs.copy()
    highs_modified[5] = 999.0
    
    mfe_mod = TargetGenerator.generate_mfe(highs_modified, closes, horizon=horizon)
    
    assert np.isclose(mfe_mod[0], mfe_orig[0]), "MFE at t=0 must NOT change when High beyond horizon is modified!"


def test_mae_changes_within_horizon():
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])
    lows = closes * 0.99
    horizon = 3
    
    mae_orig = TargetGenerator.generate_mae(lows, closes, horizon=horizon)
    
    # Modify low at t=2 (inside horizon of t=0)
    lows_modified = lows.copy()
    lows_modified[2] = 50.0 # 50% crash
    
    mae_mod = TargetGenerator.generate_mae(lows_modified, closes, horizon=horizon)
    
    assert mae_mod[0] < mae_orig[0], "MAE at t=0 should drop when Low at t=2 crashes!"
    assert np.isclose(mae_mod[0], (50.0 / 100.0) - 1.0)


def test_mae_unchanged_beyond_horizon():
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])
    lows = closes * 0.99
    horizon = 3
    
    mae_orig = TargetGenerator.generate_mae(lows, closes, horizon=horizon)
    
    # Modify low at t=6 (outside horizon of t=0)
    lows_modified = lows.copy()
    lows_modified[6] = 1.0
    
    mae_mod = TargetGenerator.generate_mae(lows_modified, closes, horizon=horizon)
    
    assert np.isclose(mae_mod[0], mae_orig[0]), "MAE at t=0 must NOT change when Low outside horizon is modified!"


def test_level_reach_label_correct():
    closes = np.array([100.0, 98.0, 96.0, 94.0, 93.0, 92.0])
    lows   = np.array([ 99.0, 97.0, 94.5, 93.5, 92.5, 91.0])
    # -5% level from 100 is 95.0. Low at index 2 reaches 94.5 <= 95.0
    reached = TargetGenerator.label_level_reach(lows, closes, lower_level_pct=-0.05, horizon=3)
    assert reached[0] == 1.0
    
    # -10% level from 100 is 90.0. Min low in horizon [indices 1,2,3] is 93.5 > 90.0 => 0.0
    reached_10 = TargetGenerator.label_level_reach(lows, closes, lower_level_pct=-0.10, horizon=3)
    assert reached_10[0] == 0.0


def test_first_passage_label_correct():
    closes = np.array([100.0, 101.0, 102.0, 103.0, 100.0])
    # Case 1: Lower barrier reached first
    highs_1 = np.array([101.0, 102.0, 103.0, 104.0, 101.0])
    lows_1  = np.array([ 99.0,  94.0,  98.0,  99.0,  98.0]) # bar 1 drops to 94 (<= 95)
    fp1 = TargetGenerator.label_first_passage(highs_1, lows_1, closes, lower_pct=-0.05, upper_pct=0.05, horizon=3)
    assert fp1["first_passage_label"][0] == TargetGenerator.FP_LOWER_FIRST
    assert fp1["time_to_lower"][0] == 1.0

    # Case 2: Upper barrier reached first
    highs_2 = np.array([101.0, 106.0, 102.0, 103.0, 101.0]) # bar 1 hits 106 (>= 105)
    lows_2  = np.array([ 99.0,  98.0,  94.0,  99.0,  98.0])
    fp2 = TargetGenerator.label_first_passage(highs_2, lows_2, closes, lower_pct=-0.05, upper_pct=0.05, horizon=3)
    assert fp2["first_passage_label"][0] == TargetGenerator.FP_UPPER_FIRST
    assert fp2["time_to_upper"][0] == 1.0

    # Case 3: AMBIGUOUS — both barriers breached within the exact same candle
    highs_3 = np.array([101.0, 107.0, 102.0, 103.0, 101.0]) # bar 1 hits 107 (>= 105)
    lows_3  = np.array([ 99.0,  93.0,  98.0,  99.0,  98.0]) # bar 1 also drops to 93 (<= 95)
    fp3 = TargetGenerator.label_first_passage(highs_3, lows_3, closes, lower_pct=-0.05, upper_pct=0.05, horizon=3)
    assert fp3["first_passage_label"][0] == TargetGenerator.FP_AMBIGUOUS
    assert fp3["ambiguous_flag"][0] == True


# =====================================================================
# Group C — Temporal Validation Tests
# Isolates train/test datasets and prevents structural leakages
# =====================================================================

def test_no_target_columns_in_features():
    klines = _generate_synthetic_klines(n=60)
    feat = FeatureExtractor.extract_features(klines)
    
    forbidden_tokens = ["fwd_", "mfe", "mae", "barrier", "target", "label", "first_passage"]
    for col in feat.columns:
        for tok in forbidden_tokens:
            assert tok not in col.lower(), f"Forbidden target/label token '{tok}' found in feature column '{col}'!"


def test_walk_forward_no_future():
    # 500 calendar days of dates
    dates = pd.date_range("2022-01-01", periods=500, freq="D")
    folds = walk_forward_expanding(dates, min_train_days=180, step_days=90, purge_days=90, embargo_days=45)
    
    assert len(folds) > 0, "Walk forward generated 0 folds!"
    for fold in folds:
        assert fold.train_start < fold.train_end
        assert fold.train_end < fold.test_start
        assert fold.test_start < fold.test_end
        # True expanding check: train_start is always the origin
        assert fold.train_start == dates[0], f"Fold {fold.fold_index} train_start does not start at origin!"


def test_purge_embargo():
    dates = pd.date_range("2022-01-01", periods=500, freq="D")
    folds = walk_forward_expanding(dates, min_train_days=180, step_days=90, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS)
    
    min_required_gap_days = PURGE_DAYS + EMBARGO_DAYS # 90 + 45 = 135
    for fold in folds:
        gap = (fold.test_start - fold.train_end).days
        assert gap >= min_required_gap_days, f"Fold {fold.fold_index} gap {gap} < required {min_required_gap_days}!"


def test_universe_sbru_no_future_listings():
    sbru = SBRUProvider()
    # Before listing date, eligible_symbols should be empty
    early_date = date(2021, 6, 1)
    eligible_early = sbru.eligible_symbols(early_date)
    assert len(eligible_early) == 0, "SBRU returned eligible symbols before SBRU_V1_LISTED_FROM!"
    
    # After listing date, eligible_symbols contains static universe
    valid_date = date(2023, 1, 1)
    eligible_valid = sbru.eligible_symbols(valid_date)
    assert len(eligible_valid) >= 50
