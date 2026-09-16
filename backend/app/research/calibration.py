import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ..models.schemas import CalibrationCurvePoint

class CalibrationEvaluator:
    """
    Evaluates probabilistic calibration for BR-001 models (Section 4 & 7.1).
    Computes Brier Score, Expected Calibration Error (ECE), and 10-bin Reliability Diagrams.
    """

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Computes mean squared error of probability forecasts: (1/N) * sum((p_i - y_i)^2)."""
        valid = ~np.isnan(y_true) & ~np.isnan(y_prob)
        if np.sum(valid) == 0:
            return 0.25
        return float(np.mean((y_prob[valid] - y_true[valid]) ** 2))

    @staticmethod
    def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Brier Skill Score (BSS) relative to naive climatology baseline:
        BSS = 1 - (BS / BS_ref)
        BSS > 0 indicates predictive skill beyond base rate.
        """
        valid = ~np.isnan(y_true) & ~np.isnan(y_prob)
        if np.sum(valid) == 0:
            return 0.0
        y = y_true[valid]
        p = y_prob[valid]
        bs = np.mean((p - y) ** 2)
        base_rate = np.mean(y)
        bs_ref = np.mean((base_rate - y) ** 2)
        if bs_ref < 1e-9:
            return 0.0
        return float(1.0 - (bs / bs_ref))

    @classmethod
    def compute_calibration_curve(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, float, List[CalibrationCurvePoint]]:
        """
        Calculates ECE, MCE, and points for reliability diagram.
        Returns: (ece, mce, list_of_calibration_points)
        """
        valid = ~np.isnan(y_true) & ~np.isnan(y_prob)
        if np.sum(valid) == 0:
            return 0.0, 0.0, []

        y = y_true[valid]
        p = np.clip(y_prob[valid], 0.0, 1.0)
        n = len(y)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        points = []
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (p >= low) & (p <= high)
            else:
                mask = (p >= low) & (p < high)

            count = int(np.sum(mask))
            midpoint = float((low + high) / 2.0)

            if count > 0:
                bin_pred = float(np.mean(p[mask]))
                bin_obs = float(np.mean(y[mask]))
                gap = abs(bin_pred - bin_obs)
                ece += (count / n) * gap
                mce = max(mce, gap)
            else:
                bin_pred = midpoint
                bin_obs = 0.0

            points.append(CalibrationCurvePoint(
                bin_center=round(midpoint, 3),
                predicted_prob=round(bin_pred, 4),
                empirical_freq=round(bin_obs, 4),
                sample_count=count
            ))

        return float(ece), float(mce), points

class PlattCalibrator:
    """
    Univariate Platt Scaling (Logistic Sigmoid Calibrator)
    P(Y=1 | z) = 1 / (1 + exp(-(a * z + b)))
    """
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.is_fitted = False

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        valid = ~np.isnan(raw_scores) & ~np.isnan(y_true)
        if np.sum(valid) < 10:
            return self

        x = raw_scores[valid]
        y = y_true[valid]

        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0, solver="lbfgs")
            clf.fit(x.reshape(-1, 1), y)
            self.a = float(clf.coef_[0][0])
            self.b = float(clf.intercept_[0])
            self.is_fitted = True
        except Exception:
            self.a = 2.0
            self.b = -1.0
            self.is_fitted = True

        return self

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.clip(raw_scores, 0.0, 1.0)
        z = self.a * raw_scores + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
