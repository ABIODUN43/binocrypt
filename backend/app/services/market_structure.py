import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

class MarketStructureAnalyzer:
    @staticmethod
    def find_pivots(highs: np.ndarray, lows: np.ndarray, window: int = 3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Identifies local swing highs and swing lows."""
        swing_highs = []
        swing_lows = []
        n = len(highs)
        
        for i in range(window, n - window):
            # Swing High
            if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, window + 1)):
                swing_highs.append({"index": i, "price": float(highs[i])})
                
            # Swing Low
            if all(lows[i] <= lows[i - j] for j in range(1, window + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, window + 1)):
                swing_lows.append({"index": i, "price": float(lows[i])})
                
        return swing_highs, swing_lows

    @classmethod
    def analyze_structure(cls, klines: List[List[Any]]) -> Dict[str, Any]:
        """
        Analyzes market structure (HH, HL, LH, LL, Breakout, S/R levels).
        """
        if not klines or len(klines) < 20:
            return {
                "structure_type": "INSUFFICIENT_DATA",
                "support_level": 0.0,
                "resistance_level": 0.0,
                "recent_swing_low": 0.0,
                "recent_swing_high": 0.0,
                "is_breakout": False,
                "is_pullback": False
            }

        highs = np.array([k[2] for k in klines])
        lows = np.array([k[3] for k in klines])
        closes = np.array([k[4] for k in klines])
        current_price = closes[-1]

        swing_highs, swing_lows = cls.find_pivots(highs, lows, window=2)

        # Defaults if not enough pivots
        recent_sh = swing_highs[-1]["price"] if swing_highs else float(np.max(highs[-10:]))
        recent_sl = swing_lows[-1]["price"] if swing_lows else float(np.min(lows[-10:]))
        prev_sh = swing_highs[-2]["price"] if len(swing_highs) >= 2 else recent_sh
        prev_sl = swing_lows[-2]["price"] if len(swing_lows) >= 2 else recent_sl

        # Support & Resistance levels from pivot clusters
        all_pivots = [p["price"] for p in swing_highs + swing_lows]
        support_level = recent_sl
        resistance_level = recent_sh

        if all_pivots:
            below_price = [p for p in all_pivots if p < current_price * 0.998]
            above_price = [p for p in all_pivots if p > current_price * 1.002]
            if below_price:
                support_level = max(below_price)
            if above_price:
                resistance_level = min(above_price)

        # Structure classification
        is_hh = recent_sh > prev_sh
        is_hl = recent_sl > prev_sl
        is_lh = recent_sh < prev_sh
        is_ll = recent_sl < prev_sl

        is_breakout = current_price > resistance_level and current_price >= highs[-2]
        is_pullback = is_hl and current_price < recent_sh and current_price > support_level

        if is_breakout:
            structure_type = "BREAKOUT"
        elif is_hh and is_hl:
            structure_type = "BULLISH_HH_HL"
        elif is_lh and is_ll:
            structure_type = "BEARISH_LH_LL"
        elif is_pullback:
            structure_type = "BULLISH_PULLBACK"
        else:
            structure_type = "SIDEWAYS_RANGE"

        return {
            "structure_type": structure_type,
            "support_level": float(support_level),
            "resistance_level": float(resistance_level),
            "recent_swing_low": float(recent_sl),
            "recent_swing_high": float(recent_sh),
            "is_breakout": bool(is_breakout),
            "is_pullback": bool(is_pullback),
            "is_higher_high": bool(is_hh),
            "is_higher_low": bool(is_hl)
        }

market_structure_analyzer = MarketStructureAnalyzer()
