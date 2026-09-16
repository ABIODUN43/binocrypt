from typing import Dict, Any, List, Optional
from datetime import datetime
from ..models.schemas import TradeSetup, OpportunityScore

class SetupEngine:
    @staticmethod
    def generate_setup(
        score: OpportunityScore,
        features_1h: Dict[str, Any],
        structure_1h: Dict[str, Any],
        features_4h: Optional[Dict[str, Any]] = None
    ) -> TradeSetup:
        """
        Calculates exact mathematical entry zone, stop loss, and targets.
        """
        price = features_1h["price"]
        atr = features_1h["atr_14"]
        recent_sl = structure_1h.get("recent_swing_low", 0.0)
        support = structure_1h.get("support_level", 0.0)
        resistance = structure_1h.get("resistance_level", 0.0)
        stype = structure_1h.get("structure_type", "SIDEWAYS_RANGE")

        # Determine Entry Zone
        entry_zone_min = round(price * 0.997, 4)
        entry_zone_max = round(price * 1.002, 4)

        # Stop Loss Placement (Below swing low or 1.5 * ATR)
        stop_candidates = []
        if recent_sl > 0 and recent_sl < price:
            stop_candidates.append(recent_sl * 0.996) # slight buffer below pivot
        if support > 0 and support < price:
            stop_candidates.append(support * 0.995)
        stop_candidates.append(price - (1.5 * atr))

        # Pick the most sensible stop (between 1.5% and 5% distance)
        valid_stops = [s for s in stop_candidates if 0.015 <= ((price - s) / price) <= 0.065]
        if valid_stops:
            stop_loss = max(valid_stops) # Tightest safe valid stop
        else:
            stop_loss = price * 0.975 # default 2.5% stop

        stop_loss = round(stop_loss, 4)
        risk_distance = price - stop_loss
        stop_distance_pct = round((risk_distance / price) * 100.0, 2)

        # Targets
        # TP1: 1.8x risk
        tp1 = round(price + (risk_distance * 1.8), 4)
        # TP2: 3.0x risk
        tp2 = round(price + (risk_distance * 3.0), 4)
        rr_ratio = round((tp1 - price) / (risk_distance + 1e-9), 2)

        # Setup type name
        if stype == "BREAKOUT":
            setup_name = "Breakout & Volume Expansion"
        elif stype in ["BULLISH_HH_HL", "BULLISH_PULLBACK"]:
            setup_name = "Trend Continuation & Pullback"
        else:
            setup_name = "Key Level Bounce"

        # Signal State
        if score.total_score >= 78 and rr_ratio >= 1.7:
            signal_state = "READY"
        elif score.total_score >= 68:
            signal_state = "SETUP_FORMING"
        elif score.total_score >= 58:
            signal_state = "WATCH"
        else:
            signal_state = "NO_TRADE"

        invalidation = f"1H candle close below ${stop_loss} ({stop_distance_pct}% from entry)"

        rationale = []
        if score.trend_score >= 14:
            rationale.append("EMA 20/50/200 aligned in bullish stacking with positive slope")
        if score.momentum_score >= 14:
            rationale.append(f"RSI ({features_1h['rsi_14']:.1f}) in prime momentum zone with positive MACD")
        if score.volume_score >= 14:
            rationale.append(f"Volume surge {features_1h['rvol_20']:.1f}x above 20-period moving average")
        if stype in ["BREAKOUT", "BULLISH_HH_HL"]:
            rationale.append(f"Market structure confirmed: {stype.replace('_', ' ')}")
        rationale.append(f"Favorable Risk/Reward profile (1 : {rr_ratio}) with tight structural invalidation")

        return TradeSetup(
            symbol=score.symbol,
            direction="LONG",
            signal_state=signal_state,
            score=score.total_score,
            grade=score.grade,
            entry_zone_min=entry_zone_min,
            entry_zone_max=entry_zone_max,
            current_price=round(price, 4),
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            risk_reward=rr_ratio,
            stop_distance_pct=stop_distance_pct,
            invalidation=invalidation,
            setup_name=setup_name,
            confidence_pct=int(score.win_probability * 100),
            rationale_bullets=rationale,
            created_at=datetime.utcnow()
        )

setup_engine = SetupEngine()
