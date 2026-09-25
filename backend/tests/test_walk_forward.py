"""
Unit tests for Walk-Forward expanding and rolling validation engines.
"""

import pytest
import pandas as pd
from app.research.walk_forward import (
    walk_forward_expanding,
    walk_forward_rolling,
    get_fold_data,
    H_MAX,
    PURGE_DAYS,
    EMBARGO_DAYS,
)

def test_walk_forward_expanding_structure():
    # 1200 calendar days provides >= 4 expanding folds with 135-day total gap (90 purge + 45 embargo)
    dates = pd.date_range("2021-01-01", periods=1200, freq="D")
    folds = walk_forward_expanding(dates, min_train_days=180, step_days=90, purge_days=90, embargo_days=45)
    
    assert len(folds) >= 4
    for i, fold in enumerate(folds):
        assert fold.protocol == "expanding"
        assert fold.train_start == dates[0]
        # Training window should expand over successive folds
        if i > 0:
            prev_fold = folds[i - 1]
            assert fold.train_end > prev_fold.train_end
            assert fold.test_start > prev_fold.test_start
            assert fold.test_end > prev_fold.test_end

def test_walk_forward_rolling_structure():
    dates = pd.date_range("2021-01-01", periods=1200, freq="D")
    train_win = 180
    folds = walk_forward_rolling(dates, train_window_days=train_win, step_days=90, purge_days=90, embargo_days=45)
    
    assert len(folds) >= 3
    for fold in folds:
        assert fold.protocol == "rolling"
        assert fold.train_days <= train_win + 5

def test_get_fold_data_split():
    dates = pd.date_range("2022-01-01", periods=600, freq="D")
    df = pd.DataFrame({
        "open_time_dt": dates,
        "val": range(len(dates))
    })
    
    folds = walk_forward_expanding(dates, min_train_days=180, step_days=90, purge_days=90, embargo_days=45)
    assert len(folds) > 0
    fold = folds[0]
    
    train_df, test_df = get_fold_data(df, fold, date_col="open_time_dt")
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert train_df["open_time_dt"].max() <= fold.train_end
    assert test_df["open_time_dt"].min() >= fold.test_start
    assert test_df["open_time_dt"].max() <= fold.test_end
    # Ensure no temporal overlap
    assert train_df["open_time_dt"].max() < test_df["open_time_dt"].min()
