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

target_generator = TargetGenerator()
