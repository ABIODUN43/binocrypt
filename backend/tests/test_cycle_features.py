"""
Unit tests for extended BR-002 cycle features.
"""

import numpy as np
import pandas as pd
import pytest
from app.research.features import FeatureExtractor

def _generate_synthetic_klines(n: int = 120, base_price: float = 50.0):
    np.random.seed(123)
    start_ts = 1672531200000
    day_ms = 86400000
    klines = []
    price = base_price
    for i in range(n):
        open_time = start_ts + i * day_ms
        close_time = open_time + day_ms - 1
        ret = np.random.normal(0.0002, 0.015)
        close_p = max(0.1, price * (1.0 + ret))
        high_p = max(price, close_p) * 1.01
        low_p = min(price, close_p) * 0.99
        vol = float(np.random.uniform(500, 2000))
        quote_vol = vol * close_p
        trade_count = 1000
        taker_base = vol * 0.5
        taker_quote = taker_base * close_p
        klines.append([
            open_time, price, high_p, low_p, close_p, vol,
            close_time, quote_vol, trade_count, taker_base, taker_quote, "0"
        ])
        price = close_p
    return klines

def test_feature_groups_present():
    klines = _generate_synthetic_klines(n=100)
    feat = FeatureExtractor.extract_features(klines)
    
    expected_new_cols = [
        "dd_from_180d_high", "dd_from_365d_high",
        "up_from_30d_low", "up_from_90d_low", "up_from_180d_low", "up_from_365d_low",
        "ath_to_date", "atl_to_date", "dist_from_ath", "dist_from_atl",
        "lower_low_count_20d", "higher_high_count_20d", "lower_high_count_20d", "higher_low_count_20d",
        "trend_persistence_20d",
        "mom_30d", "mom_60d", "mom_90d", "rsi_change_7d", "macd_accel_3d",
        "adx14", "adx_change_7d",
        "vol_zscore_30d", "taker_buy_ratio",
        "vol_change_14d", "bb_percentile_90d"
    ]
    for col in expected_new_cols:
        assert col in feat.columns, f"Expected column {col} missing from extracted features!"

def test_dist_from_ath_bounds():
    klines = _generate_synthetic_klines(n=60)
    feat = FeatureExtractor.extract_features(klines)
    # dist_from_ath should always be <= 0.0 because close <= max(high)
    assert (feat["dist_from_ath"] <= 1e-6).all(), "dist_from_ath exceeded 0!"
    # dist_from_atl should always be >= 0.0 because close >= min(low)
    assert (feat["dist_from_atl"] >= -1e-6).all(), "dist_from_atl went below 0!"

def test_get_feature_subset():
    klines = _generate_synthetic_klines(n=60)
    feat = FeatureExtractor.extract_features(klines)
    subset = FeatureExtractor.get_feature_subset(feat, ["drawdown_rolling", "structure"])
    assert "dd_from_30d_high" in subset.columns
    assert "lower_low_count_20d" in subset.columns
    assert "rsi14" not in subset.columns
