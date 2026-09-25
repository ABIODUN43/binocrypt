import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

class TargetGenerator:
    """
    Constructs and evaluates candidate Bear-to-Recovery transition targets (Section 5 & 5.1).
    Enforces strict point-in-time boundaries: target labels are future outcomes used only
    for supervised evaluation, never leaked into feature generation.
    """
    
    @staticmethod
    def generate_forward_returns(closes: np.ndarray, horizon: int) -> np.ndarray:
        """Calculates forward horizon return (P_{t+h} / P_t - 1)."""
        n = len(closes)
        fwd = np.full(n, np.nan)
        for t in range(n - horizon):
            fwd[t] = (closes[t + horizon] - closes[t]) / (closes[t] + 1e-9)
        return fwd

    @classmethod
    def label_R_A(cls, closes: np.ndarray, horizon: int = 14, threshold: float = 0.05) -> np.ndarray:
        """
        Label R-A: Simple economic benchmark. Forward 14D return > threshold (e.g. +5%).
        """
        fwd = cls.generate_forward_returns(closes, horizon)
        return np.where(np.isnan(fwd), np.nan, (fwd >= threshold).astype(float))

    @classmethod
    def label_R_B(cls, closes: np.ndarray, horizon: int = 14) -> np.ndarray:
        """
        Label R-B: Positive forward return + structural trend improvement (Price > EMA20).
        """
        fwd = cls.generate_forward_returns(closes, horizon)
        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values
        trend_improved = np.zeros(len(closes))
        for t in range(len(closes) - horizon):
            if closes[t + horizon] > ema20[t + horizon] and fwd[t] > 0:
                trend_improved[t] = 1.0
            else:
                trend_improved[t] = 0.0
        return np.where(np.isnan(fwd), np.nan, trend_improved)

    @classmethod
    def label_R_C(cls, closes: np.ndarray, lookback: int = 30, horizon: int = 14, fraction: float = 0.382) -> np.ndarray:
        """
        Label R-C: Recovery by fraction (38.2% Fib) from local rolling drawdown.
        """
        n = len(closes)
        labels = np.full(n, np.nan)
        fwd = cls.generate_forward_returns(closes, horizon)

        for t in range(lookback, n - horizon):
            window = closes[t - lookback:t + 1]
            local_high = np.max(window)
            local_low = np.min(window)
            drawdown = local_high - local_low
            if drawdown > 0:
                required_rebound = local_low + fraction * drawdown
                if closes[t + horizon] >= required_rebound:
                    labels[t] = 1.0
                else:
                    labels[t] = 0.0
            else:
                labels[t] = 0.0
        return labels

    @classmethod
    def label_R_D(cls, hmm_states: np.ndarray, horizon: int = 14) -> np.ndarray:
        """
        Label R-D: Latent-regime definition. Transitions into State R (2) from State B (0) within horizon.
        """
        n = len(hmm_states)
        labels = np.full(n, 0.0)
        for t in range(n - horizon):
            window = hmm_states[t + 1:t + horizon + 1]
            if 2 in window: # State 2 is Recovery
                labels[t] = 1.0
        return labels

    @classmethod
    def label_R_E(cls, closes: np.ndarray, hmm_states: np.ndarray, horizon: int = 14) -> np.ndarray:
        """
        Label R-E: Composite regime + economic outcome.
        Requires both forward return > 5% AND transition into Recovery regime.
        """
        r_a = cls.label_R_A(closes, horizon, threshold=0.05)
        r_d = cls.label_R_D(hmm_states, horizon)
        composite = (r_a == 1.0) & (r_d == 1.0)
        return np.where(np.isnan(r_a), np.nan, composite.astype(float))

    @classmethod
    def generate_all_horizon_targets(cls, closes: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates standard horizon targets: Y_7, Y_14, Y_30.
        """
        return {
            "Y_7": cls.label_R_A(closes, horizon=7, threshold=0.03),
            "Y_14": cls.label_R_A(closes, horizon=14, threshold=0.05),
            "Y_30": cls.label_R_A(closes, horizon=30, threshold=0.10),
            "R_E_14": cls.label_R_B(closes, horizon=14)
        }

    # =========================================================================
    # BR-002 Labels — MFE/MAE using High/Low (not closes)
    # =========================================================================

    @staticmethod
    def generate_mfe(
        highs: np.ndarray,
        closes: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Maximum Favorable Excursion using future candle HIGHS.

        MFE_{t,h} = max(High[t+1 : t+h]) / Close[t] - 1

        Uses candle highs (not closes) because a price level can be reached
        intrabar even if the close is lower — wicks matter.

        Returns NaN for the last `horizon` rows where labels cannot be formed.
        """
        n = len(closes)
        mfe = np.full(n, np.nan, dtype=np.float64)
        for t in range(n - horizon):
            future_highs = highs[t + 1 : t + horizon + 1]
            mfe[t] = (np.max(future_highs) / (closes[t] + 1e-9)) - 1.0
        return mfe

    @staticmethod
    def generate_mae(
        lows: np.ndarray,
        closes: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Maximum Adverse Excursion using future candle LOWS.

        MAE_{t,h} = min(Low[t+1 : t+h]) / Close[t] - 1

        Uses candle lows (not closes) because a price can breach a level
        intrabar even if the close is above it — wicks matter.

        Returns NaN for the last `horizon` rows.
        """
        n = len(closes)
        mae = np.full(n, np.nan, dtype=np.float64)
        for t in range(n - horizon):
            future_lows = lows[t + 1 : t + horizon + 1]
            mae[t] = (np.min(future_lows) / (closes[t] + 1e-9)) - 1.0
        return mae

    @classmethod
    def generate_mfe_mae_all_horizons(
        cls,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        horizons: list = None,
    ) -> dict:
        """
        Generates MFE and MAE for all standard horizons.
        Returns dict of column_name -> np.ndarray.
        """
        if horizons is None:
            horizons = [7, 14, 30]
        out = {}
        for h in horizons:
            out[f"mfe_{h}d"] = cls.generate_mfe(highs, closes, h)
            out[f"mae_{h}d"] = cls.generate_mae(lows, closes, h)
        return out

    # =========================================================================
    # BR-002 Labels — Barrier Events (A/B/C) using High/Low paths
    # =========================================================================

    @staticmethod
    def label_barrier_event(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        up_pct: float,
        down_pct: float,
        horizon: int,
    ) -> dict:
        """
        For each observation t, determines whether the upper or lower barrier
        is reached first within the horizon, using bar-by-bar High/Low.

        Labels:
          "UP"      — upper barrier (close[t] * (1 + up_pct)) reached via High first
          "DOWN"    — lower barrier (close[t] * (1 + down_pct)) reached via Low first
          "TIMEOUT" — neither barrier reached within horizon

        Parameters
        ----------
        up_pct   : positive float, e.g. 0.25 for +25%
        down_pct : negative float, e.g. -0.10 for -10%

        Returns dict of np.ndarray columns:
          barrier_label   : str array
          barrier_time    : float (days to event, NaN for TIMEOUT)
          barrier_mfe     : float (MFE at event time)
          barrier_mae     : float (MAE at event time)
        """
        n = len(closes)
        labels = np.full(n, "TIMEOUT", dtype=object)
        times  = np.full(n, np.nan)
        mfes   = np.full(n, np.nan)
        maes   = np.full(n, np.nan)

        for t in range(n - horizon):
            entry = closes[t]
            upper = entry * (1.0 + up_pct)
            lower = entry * (1.0 + down_pct)
            max_high = entry
            min_low  = entry

            for tau in range(1, horizon + 1):
                i = t + tau
                h = highs[i]
                l = lows[i]
                max_high = max(max_high, h)
                min_low  = min(min_low,  l)

                if h >= upper and l <= lower:
                    # Both breached in the same bar — AMBIGUOUS for directional
                    # but we still record event time and extremes
                    labels[t] = "TIMEOUT"  # excluded from UP/DOWN training
                    times[t]  = float(tau)
                    mfes[t]   = (max_high / entry) - 1.0
                    maes[t]   = (min_low  / entry) - 1.0
                    break
                elif h >= upper:
                    labels[t] = "UP"
                    times[t]  = float(tau)
                    mfes[t]   = (max_high / entry) - 1.0
                    maes[t]   = (min_low  / entry) - 1.0
                    break
                elif l <= lower:
                    labels[t] = "DOWN"
                    times[t]  = float(tau)
                    mfes[t]   = (max_high / entry) - 1.0
                    maes[t]   = (min_low  / entry) - 1.0
                    break
            else:
                # Horizon expired
                mfes[t] = (max(highs[t + 1 : t + horizon + 1]) / entry) - 1.0
                maes[t] = (min(lows[t + 1  : t + horizon + 1]) / entry) - 1.0

        return {
            "barrier_label": labels,
            "barrier_time":  times,
            "barrier_mfe":   mfes,
            "barrier_mae":   maes,
        }

    @classmethod
    def generate_all_barrier_labels(
        cls,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        horizon: int = 90,
    ) -> dict:
        """
        Generates Barrier A (+25% before -10%), B (+50% before -20%),
        C (+100% before -30%) labels for BR-002.1C.
        """
        out = {}
        for tag, up, dn in [("A", 0.25, -0.10), ("B", 0.50, -0.20), ("C", 1.00, -0.30)]:
            result = cls.label_barrier_event(highs, lows, closes, up, dn, horizon)
            for k, v in result.items():
                out[f"barrier_{tag}_{k}"] = v
        return out

    # =========================================================================
    # BR-003 Labels — Level-Reach and First-Passage using High/Low
    # =========================================================================

    @staticmethod
    def label_level_reach(
        lows: np.ndarray,
        closes: np.ndarray,
        lower_level_pct: float,
        horizon: int,
    ) -> np.ndarray:
        """
        P(min Low[t+1:t+h] <= L | X_t) — empirical frequency label.

        lower_level_pct: negative, e.g. -0.05 for 5% below close[t].
        Uses candle lows — a wick to the level counts.
        Returns 1 if the level is reached (via any Low), else 0.
        """
        n = len(closes)
        reached = np.full(n, np.nan)
        for t in range(n - horizon):
            level = closes[t] * (1.0 + lower_level_pct)
            future_lows = lows[t + 1 : t + horizon + 1]
            reached[t] = 1.0 if np.min(future_lows) <= level else 0.0
        return reached

    # First-passage label constants
    FP_LOWER_FIRST = 1
    FP_UPPER_FIRST = -1
    FP_AMBIGUOUS   = 0
    FP_TIMEOUT     = -2

    @staticmethod
    def label_first_passage(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        lower_pct: float,
        upper_pct: float,
        horizon: int,
    ) -> dict:
        """
        First-barrier-touched label for BR-003.2.

        P(tau_L < tau_U | X_t) — which barrier is reached first?

        Uses bar-by-bar High and Low to simulate the price path.

        Labels:
          FP_LOWER_FIRST  (1)  — lower barrier touched first via Low
          FP_UPPER_FIRST (-1)  — upper barrier touched first via High
          FP_AMBIGUOUS    (0)  — both touched within the same candle
                                  (intrabar ordering unknown; EXCLUDED from
                                   directional binary training)
          FP_TIMEOUT     (-2)  — horizon expired, neither barrier reached

        Parameters
        ----------
        lower_pct : negative float, e.g. -0.05 for -5% below current close
        upper_pct : positive float, e.g.  0.05 for +5% above current close

        Returns dict of np.ndarray:
          first_passage_label  : int array
          time_to_lower        : float (days, NaN if not reached)
          time_to_upper        : float (days, NaN if not reached)
          ambiguous_flag       : bool array
        """
        n = len(closes)
        fp_label    = np.full(n, TargetGenerator.FP_TIMEOUT, dtype=np.int8)
        t_lower     = np.full(n, np.nan)
        t_upper     = np.full(n, np.nan)
        ambiguous   = np.zeros(n, dtype=bool)

        for t in range(n - horizon):
            entry = closes[t]
            lower = entry * (1.0 + lower_pct)
            upper = entry * (1.0 + upper_pct)

            for tau in range(1, horizon + 1):
                i = t + tau
                low_breach  = lows[i]  <= lower
                high_breach = highs[i] >= upper

                if low_breach and high_breach:
                    # Both barriers breached within the same candle.
                    # Intrabar ordering is UNKNOWABLE from OHLC data.
                    fp_label[t]  = TargetGenerator.FP_AMBIGUOUS
                    t_lower[t]   = float(tau)
                    t_upper[t]   = float(tau)
                    ambiguous[t] = True
                    break
                elif low_breach:
                    fp_label[t] = TargetGenerator.FP_LOWER_FIRST
                    t_lower[t]  = float(tau)
                    break
                elif high_breach:
                    fp_label[t] = TargetGenerator.FP_UPPER_FIRST
                    t_upper[t]  = float(tau)
                    break
            # If loop completes without break: FP_TIMEOUT (default)

        return {
            "first_passage_label": fp_label,
            "time_to_lower":       t_lower,
            "time_to_upper":       t_upper,
            "ambiguous_flag":      ambiguous,
        }

    @classmethod
    def generate_all_path_labels(
        cls,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        level_pairs: list = None,
        horizon: int = 30,
    ) -> dict:
        """
        Generates level-reach and first-passage labels for all candidate level pairs.

        level_pairs: list of (lower_pct, upper_pct) tuples.
        Default: all combinations of lower in {-5%,-8%,-10%,-15%,-20%}
                 with upper in {+3%,+5%,+8%,+10%}.
        """
        if level_pairs is None:
            lowers = [-0.05, -0.08, -0.10, -0.15, -0.20]
            uppers = [0.03,  0.05,  0.08,  0.10]
            level_pairs = [(lo, up) for lo in lowers for up in uppers]

        out = {}
        for lower_pct, upper_pct in level_pairs:
            tag = f"L{abs(int(lower_pct*100))}U{int(upper_pct*100)}"
            # Level reach label
            out[f"lvl_reach_{tag}"] = cls.label_level_reach(
                lows, closes, lower_pct, horizon
            )
            # First-passage labels
            fp = cls.label_first_passage(
                highs, lows, closes, lower_pct, upper_pct, horizon
            )
            out[f"fp_label_{tag}"]    = fp["first_passage_label"]
            out[f"fp_t_lower_{tag}"]  = fp["time_to_lower"]
            out[f"fp_t_upper_{tag}"]  = fp["time_to_upper"]
            out[f"fp_ambiguous_{tag}"] = fp["ambiguous_flag"].astype(np.int8)

        return out


target_generator = TargetGenerator()
