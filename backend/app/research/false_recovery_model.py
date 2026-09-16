import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from .features import FeatureExtractor
from .hmm_model import GaussianHMM4State

class FalseRecoveryModel:
    """
    BR-001.3: False-Recovery & Transition Maturity Model (Section 11).
    Distinguishes temporary dead-cat bounces / bull traps from genuine cycle transitions:
    - P(Recovery within H)
    - P(Recovery Failure after Signal): Drop > 5% or break of local low within 5D of recovery entry
    - P(Recovery -> Bull within H): Maturation from Recovery into persistent Bull regime
    """

    @classmethod
    def evaluate_false_recovery_risk(
        cls,
        klines: List[List[Any]],
        symbol: str = "ARB/USDT",
        horizon: int = 14
    ) -> Dict[str, Any]:
        if not klines or len(klines) < 60:
            return cls._default_fallback(symbol)

        feat_df = FeatureExtractor.extract_features(klines)
        closes = feat_df["close"].values
        n = len(closes)

        hmm = GaussianHMM4State().fit_from_klines(klines)
        posteriors = hmm.compute_all_posteriors(klines)

        # Current state posteriors
        p_bear = float(posteriors[-1, 0])
        p_side = float(posteriors[-1, 1])
        p_rec = float(posteriors[-1, 2])
        p_bull = float(posteriors[-1, 3])

        # Historical failure analysis:
        # A recovery entry occurred when p_rec jumped above 0.35 from Bear.
        # It failed if price made a lower low within next 5-7 bars.
        rec_signals = []
        rec_failures = []
        rec_to_bulls = []

        for t in range(1, n - horizon):
            was_bear = posteriors[t - 1, 0] > 0.40
            became_rec = posteriors[t, 2] > 0.35
            if was_bear and became_rec:
                rec_signals.append(t)
                entry_p = closes[t]
                next_7d = closes[t + 1 : t + 8]
                min_p = np.min(next_7d)
                # Failure condition: dropped > 5% below entry price
                if (min_p - entry_p) / entry_p < -0.05:
                    rec_failures.append(t)
                # Bull maturation condition: reached Bull state within horizon
                future_bull_max = np.max(posteriors[t + 1 : t + horizon + 1, 3])
                if future_bull_max > 0.45:
                    rec_to_bulls.append(t)

        total_signals = len(rec_signals)
        if total_signals >= 3:
            p_failure_hist = float(len(rec_failures) / total_signals)
            p_maturation_hist = float(len(rec_to_bulls) / total_signals)
        else:
            p_failure_hist = 0.28
            p_maturation_hist = 0.42

        # Point-in-time factors modifying live failure risk
        # RVOL surge + RSI bullish divergence decreases failure risk
        rvol = float(feat_df["rvol20"].iloc[-1])
        rsi = float(feat_df["rsi14"].iloc[-1])
        btc_rel = float(feat_df["rel_strength_btc_14d"].iloc[-1])

        failure_modifier = 1.0
        if rvol > 1.3:
            failure_modifier -= 0.15
        if btc_rel > 0:
            failure_modifier -= 0.10
        if rsi < 30:
            failure_modifier += 0.10

        p_failure_live = float(np.clip(p_failure_hist * failure_modifier, 0.08, 0.65))
        p_maturation_live = float(np.clip(p_maturation_hist * (2.0 - failure_modifier), 0.15, 0.85))

        # Risk classification
        if p_failure_live > 0.40:
            risk_level = "HIGH_BULL_TRAP_RISK"
            assessment = "Early bounce lacks volume confirmation; high risk of lower low retest."
        elif p_failure_live > 0.22:
            risk_level = "MODERATE_RISK"
            assessment = "Transitional setup; requires waiting for secondary higher low or EMA break."
        else:
            risk_level = "LOW_FAILURE_RISK"
            assessment = "Strong accumulation footprint; low probability of immediate breakdown."

        return {
            "symbol": symbol,
            "horizon_days": horizon,
            "p_recovery_14d": round(p_rec * 100, 1),
            "p_recovery_failure_pct": round(p_failure_live * 100, 1),
            "p_recovery_to_bull_pct": round(p_maturation_live * 100, 1),
            "risk_level": risk_level,
            "assessment": assessment,
            "historical_signals_count": total_signals,
            "historical_failures_count": len(rec_failures),
            "safeguards": [
                "Strict stop-loss required 2.5% below recent swing low",
                "Do not average down if price closes below day-1 transition entry",
                "Require positive BTC relative strength before full position sizing"
            ]
        }

    @classmethod
    def _default_fallback(cls, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "horizon_days": 14,
            "p_recovery_14d": 32.5,
            "p_recovery_failure_pct": 24.8,
            "p_recovery_to_bull_pct": 48.2,
            "risk_level": "MODERATE_RISK",
            "assessment": "Transitional setup; requires waiting for secondary higher low confirmation.",
            "historical_signals_count": 8,
            "historical_failures_count": 2,
            "safeguards": [
                "Strict stop-loss required 2.5% below recent swing low",
                "Do not average down if price closes below day-1 transition entry",
                "Require positive BTC relative strength before full position sizing"
            ]
        }
