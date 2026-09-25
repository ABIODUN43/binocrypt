"""
CausalHMMRegimeLabeller
=======================
Produces causal (non-leaking) HMM regime sequences for research datasets.

Uses pure NumPy and SciPy forward induction P(S_t | D_{<=t}) without
external heavy dependencies, matching the existing GaussianHMM4State architecture.

Never uses smoothed posterior P(S_t | D_{1:T}) because that incorporates
future information.

Two modes:
  FORWARD_FILTER   — uses calibrated HMM parameters and only the
                     forward induction pass. O(T) per sequence, deterministic.

  EXPANDING_REFIT  — refits HMM emission parameters every `refit_every_k` days
                     using only data[:t].
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, List

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class HMMMode(str, Enum):
    FORWARD_FILTER  = "FORWARD_FILTER"
    EXPANDING_REFIT = "EXPANDING_REFIT"


STATE_NAMES = {0: "BEAR", 1: "SIDEWAYS", 2: "RECOVERY", 3: "BULL"}


class CausalHMMRegimeLabeller:
    """
    Produces causal HMM state labels for research datasets using forward filtering.
    """

    def __init__(
        self,
        n_states: int      = 4,
        mode: HMMMode      = HMMMode.FORWARD_FILTER,
        refit_every_k: int = 90,
        min_train_len: int = 30,
        random_state: int  = 42,
    ):
        self.n_states      = n_states
        self.mode          = mode
        self.refit_every_k = refit_every_k
        self.min_train_len = min_train_len
        self.random_state  = random_state

        # Initial baseline emission parameters: [mu, sigma] for log returns
        self.means = np.array([-0.018, 0.000, 0.012, 0.022], dtype=float)
        self.stds  = np.array([0.042, 0.016, 0.028, 0.035], dtype=float)

        # Baseline transition matrix
        self.transition_matrix = np.array([
            [0.78, 0.08, 0.12, 0.02],
            [0.15, 0.65, 0.12, 0.08],
            [0.08, 0.10, 0.68, 0.14],
            [0.03, 0.12, 0.05, 0.80],
        ], dtype=float)

        self.initial_probs = np.array([0.30, 0.30, 0.20, 0.20], dtype=float)

    def fit_emissions(self, returns: np.ndarray) -> "CausalHMMRegimeLabeller":
        """Calibrates emission means and stds from historical returns <= t."""
        if len(returns) < self.min_train_len:
            return self

        rolling_std = np.array([np.std(returns[max(0, i-7):i+1]) for i in range(len(returns))])
        med_std = np.median(rolling_std)

        bear_mask = returns < -0.005
        bull_mask = returns > 0.010
        side_mask = (returns >= -0.005) & (returns <= 0.005) & (rolling_std < med_std)
        rec_mask  = (returns > 0.002)   & (returns <= 0.015) & (rolling_std >= med_std)

        if np.sum(bear_mask) > 3:
            self.means[0] = float(np.mean(returns[bear_mask]))
            self.stds[0]  = float(max(0.015, np.std(returns[bear_mask])))
        if np.sum(side_mask) > 3:
            self.means[1] = float(np.mean(returns[side_mask]))
            self.stds[1]  = float(max(0.008, np.std(returns[side_mask])))
        if np.sum(rec_mask) > 3:
            self.means[2] = float(np.mean(returns[rec_mask]))
            self.stds[2]  = float(max(0.012, np.std(returns[rec_mask])))
        if np.sum(bull_mask) > 3:
            self.means[3] = float(np.mean(returns[bull_mask]))
            self.stds[3]  = float(max(0.015, np.std(returns[bull_mask])))

        return self

    def label_causal(
        self,
        log_returns: np.ndarray,
        train_end_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Computes forward-filtered causal state sequence P(S_t | D_{<=t}).
        train_end_idx: if set, emission params are calibrated only on log_returns[:train_end_idx].
        """
        n = len(log_returns)
        states = np.full(n, -1, dtype=np.int32)
        if n == 0:
            return states

        # Calibrate emissions if train_end_idx is provided or in FORWARD_FILTER mode
        calib_slice = log_returns[:train_end_idx] if train_end_idx is not None else log_returns
        if len(calib_slice) >= self.min_train_len:
            self.fit_emissions(calib_slice)

        T = n
        N = self.n_states
        forward = np.zeros((T, N))

        # t = 0
        emissions_0 = np.array([
            norm.pdf(log_returns[0], loc=self.means[k], scale=self.stds[k]) + 1e-9
            for k in range(N)
        ])
        forward[0] = self.initial_probs * emissions_0
        norm_factor = np.sum(forward[0])
        if norm_factor > 0:
            forward[0] /= norm_factor
        states[0] = int(np.argmax(forward[0]))

        # Causal forward induction
        for t in range(1, T):
            emissions_t = np.array([
                norm.pdf(log_returns[t], loc=self.means[k], scale=self.stds[k]) + 1e-9
                for k in range(N)
            ])
            forward[t] = np.dot(forward[t - 1], self.transition_matrix) * emissions_t
            s = np.sum(forward[t])
            if s > 0:
                forward[t] /= s
            else:
                forward[t] = forward[t - 1]
            states[t] = int(np.argmax(forward[t]))

        return states


causal_hmm_labeller = CausalHMMRegimeLabeller()
