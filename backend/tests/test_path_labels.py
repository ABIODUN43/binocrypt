"""
Unit tests for BR-002/BR-003 MFE/MAE, Barrier, Level-Reach and First-Passage labels.
"""

import numpy as np
import pytest
from app.research.targets import TargetGenerator

def test_mfe_mae_all_horizons():
    n = 50
    closes = np.linspace(100, 150, n)
    highs = closes * 1.02
    lows = closes * 0.98
    
    res = TargetGenerator.generate_mfe_mae_all_horizons(highs, lows, closes, horizons=[7, 14, 30])
    for h in [7, 14, 30]:
        assert f"mfe_{h}d" in res
        assert f"mae_{h}d" in res
        assert len(res[f"mfe_{h}d"]) == n
        # Last h elements must be NaN
        assert np.isnan(res[f"mfe_{h}d"][-h:]).all()

def test_barrier_event_generation():
    closes = np.array([100.0, 105.0, 110.0, 126.0, 130.0]) # Hits +25% at index 3
    highs = closes * 1.01
    lows = closes * 0.99
    
    # Barrier A (+25% vs -10%)
    res = TargetGenerator.label_barrier_event(highs, lows, closes, up_pct=0.25, down_pct=-0.10, horizon=4)
    assert res["barrier_label"][0] == "UP"
    assert res["barrier_time"][0] == 3.0

def test_all_path_labels():
    closes = np.array([100.0, 99.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0])
    highs = closes * 1.01
    lows = closes * 0.99
    
    path_labels = TargetGenerator.generate_all_path_labels(
        highs, lows, closes,
        level_pairs=[(-0.05, 0.05)],
        horizon=5
    )
    assert "lvl_reach_L5U5" in path_labels
    assert "fp_label_L5U5" in path_labels
    assert "fp_ambiguous_L5U5" in path_labels
    # At index 0, price is 100, at index 2 low is 94*0.99 = 93.06 <= 95 -> L reached
    assert path_labels["lvl_reach_L5U5"][0] == 1.0
    assert path_labels["fp_label_L5U5"][0] == TargetGenerator.FP_LOWER_FIRST
