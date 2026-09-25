"""
Walk-Forward Validation Utilities
===================================
Two protocols are provided:
  - expanding  (primary research protocol)
  - rolling    (sensitivity experiment only)

Temporal parameters are derived from H_max = 90D:
  PURGE_DAYS   = 90  (no training label within H_max of any test observation)
  EMBARGO_DAYS = 45  (gap between train_end and test_start, beyond purge)

Expanding Walk-Forward (primary)
---------------------------------
Training always starts at dates[0] and GROWS at each fold.
Test window advances by STEP_DAYS.

  Fold 1: TRAIN [0 → T1]       │ purge+embargo │ TEST [T1+135 → T1+225]
  Fold 2: TRAIN [0 → T1+225]   │ purge+embargo │ TEST [T1+360 → T1+450]
  Fold 3: TRAIN [0 → T1+450]   │ purge+embargo │ TEST [T1+585 → T1+675]

Rolling Walk-Forward (sensitivity only)
-----------------------------------------
Training window is fixed width (TRAIN_WINDOW_DAYS) and slides forward.
Results labelled "rolling_sensitivity" in reports, not used as primary evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (all derived from H_max = 90D)
# ---------------------------------------------------------------------------
H_MAX           = 90    # maximum forward label horizon (days)
PURGE_DAYS      = 90    # = H_max: no training label within this of any test obs
EMBARGO_DAYS    = 45    # = H_max/2: gap between train_end and test_start
MIN_TRAIN_DAYS  = 180   # minimum training period before first fold
STEP_DAYS       = 90    # test window size / fold advance step


@dataclass
class WalkForwardFold:
    fold_index: int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    protocol:    str           # "expanding" | "rolling"
    train_days:  int
    test_days:   int

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_index} [{self.protocol}] "
            f"TRAIN {self.train_start.date()} → {self.train_end.date()} "
            f"({self.train_days}d)  "
            f"TEST {self.test_start.date()} → {self.test_end.date()} "
            f"({self.test_days}d)"
        )


def walk_forward_expanding(
    dates: pd.DatetimeIndex,
    min_train_days: int  = MIN_TRAIN_DAYS,
    step_days: int       = STEP_DAYS,
    purge_days: int      = PURGE_DAYS,
    embargo_days: int    = EMBARGO_DAYS,
) -> List[WalkForwardFold]:
    """
    True expanding walk-forward.

    Training starts at dates[0] on every fold and grows by `step_days` each time.
    The total gap between train_end and test_start is always
    >= purge_days + embargo_days = 135D.

    Parameters
    ----------
    dates : sorted DatetimeIndex of all available observation dates
    min_train_days : minimum number of training days before first fold
    step_days : how far the test window advances per fold (also test width)
    purge_days : temporal buffer (= H_max) between train labels and test obs
    embargo_days : additional gap to prevent label-boundary contamination

    Returns
    -------
    List of WalkForwardFold (may be empty if data is too short)
    """
    if len(dates) == 0:
        return []

    folds: List[WalkForwardFold] = []
    origin    = dates[0]
    train_end = origin + pd.Timedelta(days=min_train_days)
    gap       = pd.Timedelta(days=purge_days + embargo_days)
    step      = pd.Timedelta(days=step_days)

    fold_idx = 0
    while True:
        test_start = train_end + gap
        test_end   = test_start + step

        if test_end > dates[-1]:
            break

        # Collect actual date indices within each slice
        train_mask = (dates >= origin) & (dates <= train_end)
        test_mask  = (dates >= test_start) & (dates <= test_end)

        n_train = int(train_mask.sum())
        n_test  = int(test_mask.sum())

        if n_train >= min_train_days and n_test > 0:
            folds.append(WalkForwardFold(
                fold_index  = fold_idx,
                train_start = origin,
                train_end   = train_end,
                test_start  = test_start,
                test_end    = test_end,
                protocol    = "expanding",
                train_days  = n_train,
                test_days   = n_test,
            ))
            fold_idx += 1

        train_end = test_end  # expand training window for next fold

    logger.info(
        f"[WalkForward] Expanding: {len(folds)} folds, "
        f"purge={purge_days}D, embargo={embargo_days}D, step={step_days}D"
    )
    return folds


def walk_forward_rolling(
    dates: pd.DatetimeIndex,
    train_window_days: int = MIN_TRAIN_DAYS,
    step_days: int         = STEP_DAYS,
    purge_days: int        = PURGE_DAYS,
    embargo_days: int      = EMBARGO_DAYS,
) -> List[WalkForwardFold]:
    """
    Rolling walk-forward (SENSITIVITY EXPERIMENT ONLY).

    Training window is fixed at `train_window_days` and slides forward.
    Primary results must use expanding walk-forward.
    Results from this function should be clearly labelled
    'rolling_sensitivity' in any research report.
    """
    if len(dates) == 0:
        return []

    folds: List[WalkForwardFold] = []
    gap   = pd.Timedelta(days=purge_days + embargo_days)
    step  = pd.Timedelta(days=step_days)
    win   = pd.Timedelta(days=train_window_days)

    train_start = dates[0]
    fold_idx    = 0

    while True:
        train_end  = train_start + win
        test_start = train_end + gap
        test_end   = test_start + step

        if test_end > dates[-1]:
            break

        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask  = (dates >= test_start) & (dates <= test_end)

        n_train = int(train_mask.sum())
        n_test  = int(test_mask.sum())

        if n_train > 0 and n_test > 0:
            folds.append(WalkForwardFold(
                fold_index  = fold_idx,
                train_start = train_start,
                train_end   = train_end,
                test_start  = test_start,
                test_end    = test_end,
                protocol    = "rolling",
                train_days  = n_train,
                test_days   = n_test,
            ))
            fold_idx += 1
        train_start = train_start + step  # slide window forward by step_days

    logger.info(
        f"[WalkForward] Rolling: {len(folds)} folds, "
        f"window={train_window_days}D, purge={purge_days}D, embargo={embargo_days}D"
    )
    return folds


def get_fold_data(
    df: pd.DataFrame,
    fold: WalkForwardFold,
    date_col: str = "open_time_dt",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into (train, test) according to a WalkForwardFold.

    `date_col` must be a datetime column in `df`.
    Returns (train_df, test_df).
    """
    train_mask = (df[date_col] >= fold.train_start) & (df[date_col] <= fold.train_end)
    test_mask  = (df[date_col] >= fold.test_start)  & (df[date_col] <= fold.test_end)
    return df[train_mask].copy(), df[test_mask].copy()
