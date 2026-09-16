import math
import numpy as np
from scipy.stats import norm
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from ..models.schemas import (
    HMMStatePosteriors,
    HMMTransitionMatrix,
    TransitionTimingPoint,
    HMMAnalysisResponse
)

class HMMPosteriorResult:
    def __init__(self, probs: np.ndarray, states: List[str]):
        self.probs = probs
        self.state_probabilities = {states[i]: float(probs[i]) for i in range(len(states))}
        self.predicted_state = states[int(np.argmax(probs))]

class GaussianHMM4State:
    """
    4-State Latent Gaussian Hidden Markov Model for Cryptocurrency Regime Modeling:
    - State 0 (Bear): Persistent negative drift, elevated variance
    - State 1 (Sideways): Near-zero drift, compressed variance
    - State 2 (Recovery): Positive inflection drift, moderate variance
    - State 3 (Bull): Sustained positive drift, robust variance
    """
    STATES = ["Bear", "Sideways", "Recovery", "Bull"]

    def __init__(self):
        # Initial state emission parameters: [mu, sigma] for log returns
        self.means = np.array([-0.018, 0.000, 0.012, 0.022], dtype=float)
        self.stds = np.array([0.042, 0.016, 0.028, 0.035], dtype=float)

        # Baseline state transition matrix A_ij = P(S_{t+1}=j | S_t=i)
        self.transition_matrix = np.array([
            [0.78, 0.08, 0.12, 0.02], # From Bear
            [0.15, 0.65, 0.12, 0.08], # From Sideways
            [0.08, 0.10, 0.68, 0.14], # From Recovery
            [0.03, 0.12, 0.05, 0.80], # From Bull
        ], dtype=float)

        # Initial stationary distribution
        self.initial_probs = np.array([0.30, 0.30, 0.20, 0.20], dtype=float)

    def fit_from_klines(self, klines: List[List[Any]]) -> "GaussianHMM4State":
        """
        Calibrates HMM emission parameters using historical klines.
        """
        if not klines or len(klines) < 30:
            return self

        closes = np.array([float(k[4]) for k in klines], dtype=float)
        returns = np.diff(np.log(closes + 1e-9))

        if len(returns) < 25:
            return self

        rolling_std = np.array([np.std(returns[max(0, i-7):i+1]) for i in range(len(returns))])
        
        bear_mask = returns < -0.005
        bull_mask = returns > 0.010
        side_mask = (returns >= -0.005) & (returns <= 0.005) & (rolling_std < np.median(rolling_std))
        rec_mask = (returns > 0.002) & (returns <= 0.015) & (rolling_std >= np.median(rolling_std))

        if np.sum(bear_mask) > 3:
            self.means[0] = np.mean(returns[bear_mask])
            self.stds[0] = max(0.015, np.std(returns[bear_mask]))
        if np.sum(side_mask) > 3:
            self.means[1] = np.mean(returns[side_mask])
            self.stds[1] = max(0.008, np.std(returns[side_mask]))
        if np.sum(rec_mask) > 3:
            self.means[2] = np.mean(returns[rec_mask])
            self.stds[2] = max(0.012, np.std(returns[rec_mask]))
        if np.sum(bull_mask) > 3:
            self.means[3] = np.mean(returns[bull_mask])
            self.stds[3] = max(0.015, np.std(returns[bull_mask]))

        return self

    def compute_all_posteriors(self, klines: List[List[Any]]) -> np.ndarray:
        """
        Computes forward state posteriors P(S_t = k | X_{1..t}) for all t in 0..T-1.
        Returns array of shape (T, 4).
        """
        if not klines or len(klines) < 2:
            return np.ones((len(klines), 4)) * 0.25

        closes = np.array([float(k[4]) for k in klines], dtype=float)
        returns = np.diff(np.log(closes + 1e-9))

        T = len(returns)
        N = 4
        forward = np.zeros((T, N))

        # t = 0
        emissions_0 = np.array([norm.pdf(returns[0], loc=self.means[k], scale=self.stds[k]) + 1e-9 for k in range(N)])
        forward[0] = self.initial_probs * emissions_0
        norm_factor = np.sum(forward[0])
        if norm_factor > 0:
            forward[0] /= norm_factor

        # Forward induction
        for t in range(1, T):
            emissions_t = np.array([norm.pdf(returns[t], loc=self.means[k], scale=self.stds[k]) + 1e-9 for k in range(N)])
            forward[t] = np.dot(forward[t - 1], self.transition_matrix) * emissions_t
            s = np.sum(forward[t])
            if s > 0:
                forward[t] /= s
            else:
                forward[t] = forward[t - 1]

        # Prepend initial state for t=0 candle
        res = np.vstack([self.initial_probs.reshape(1, 4), forward])
        return res

    def compute_posteriors(self, klines: List[List[Any]]) -> List[HMMPosteriorResult]:
        """Returns sequence of HMMPosteriorResult objects for each kline timestamp."""
        probs_matrix = self.compute_all_posteriors(klines)
        return [HMMPosteriorResult(probs_matrix[i], self.STATES) for i in range(len(probs_matrix))]

    def compute_forward_posteriors(self, klines: List[List[Any]]) -> np.ndarray:
        """Returns the final state posterior vector P(S_T | X_{1..T})."""
        all_post = self.compute_all_posteriors(klines)
        return all_post[-1]

    def compute_transition_timing_mass(self, current_posteriors: np.ndarray, max_horizon: int = 30) -> List[TransitionTimingPoint]:
        """
        Computes daily transition probability mass distribution:
        q_h = P(T_BR = h | X_t)
        """
        points = []
        rec_idx = 2
        p = current_posteriors.copy()
        cumulative = 0.0

        for h in range(1, max_horizon + 1):
            p_next = np.dot(p, self.transition_matrix)
            mass = float(p_next[rec_idx] * (1.0 - cumulative))
            decay = math.exp(-0.02 * (h - 1))
            mass = max(0.005, min(0.35, mass * decay))
            cumulative = min(0.95, cumulative + mass)
            
            points.append(TransitionTimingPoint(
                day=h,
                probability_mass=round(mass * 100.0, 1),
                cumulative_prob=round(cumulative * 100.0, 1)
            ))
            p = p_next

        return points

    def estimate_modal_transition_window(self, dominant_state: str = "BEAR", max_horizon: int = 30) -> str:
        idx = 0
        for i, s in enumerate(self.STATES):
            if s.upper() == dominant_state.upper():
                idx = i
                break
        init_v = np.zeros(4)
        init_v[idx] = 1.0
        timing_points = self.compute_transition_timing_mass(init_v, max_horizon=max_horizon)
        best_point = max(timing_points, key=lambda x: x.probability_mass)
        peak_day = best_point.day
        window_start = max(1, peak_day - 2)
        window_end = peak_day + 3
        return f"Day {window_start} - Day {window_end} (Peak: Day {peak_day} at {best_point.probability_mass}%)"

    def analyze(self, klines: List[List[Any]], current_price: float, symbol: str) -> HMMAnalysisResponse:
        """
        Produces complete quantitative HMM research analysis for an asset.
        """
        self.fit_from_klines(klines)
        posteriors = self.compute_forward_posteriors(klines)
        timing_points = self.compute_transition_timing_mass(posteriors, max_horizon=30)

        p_bear = round(float(posteriors[0]) * 100.0, 1)
        p_side = round(float(posteriors[1]) * 100.0, 1)
        p_rec = round(float(posteriors[2]) * 100.0, 1)
        p_bull = round(float(posteriors[3]) * 100.0, 1)
        
        diff = round(100.0 - (p_bear + p_side + p_rec + p_bull), 1)
        p_rec = round(p_rec + diff, 1)

        states = ["Bear", "Sideways", "Recovery", "Bull"]
        dom_idx = int(np.argmax(posteriors))
        dom_state = states[dom_idx].upper()

        best_point = max(timing_points, key=lambda x: x.probability_mass)
        peak_day = best_point.day
        window_start = max(1, peak_day - 2)
        window_end = peak_day + 3
        modal_window = f"Day {window_start} - Day {window_end} (Peak: Day {peak_day} at {best_point.probability_mass}%)"

        matrix_list = [
            [round(float(val), 3) for val in row]
            for row in self.transition_matrix
        ]

        return HMMAnalysisResponse(
            symbol=symbol,
            current_price=current_price,
            state_posteriors=HMMStatePosteriors(
                bear=p_bear,
                sideways=p_side,
                recovery=p_rec,
                bull=p_bull,
                dominant_state=dom_state
            ),
            transition_matrix=HMMTransitionMatrix(
                states=states,
                matrix=matrix_list
            ),
            timing_distribution=timing_points,
            modal_window=modal_window,
            recovery_readiness_pct=round(p_rec + p_bull * 0.5, 1),
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        )

hmm_engine = GaussianHMM4State()
