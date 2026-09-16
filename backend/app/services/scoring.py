from typing import Dict, Any, Optional
from ..models.schemas import OpportunityScore

class OpportunityScoringEngine:
    @staticmethod
    def score_candidate(
        symbol: str,
        ticker: Dict[str, Any],
        features_1h: Dict[str, Any],
        features_4h: Optional[Dict[str, Any]],
        structure_1h: Dict[str, Any],
        regime: str,
        estimated_rr: float = 2.0
    ) -> OpportunityScore:
        """
        Calculates multi-factor transparent score (0-100) for a coin.
        """
        price = features_1h["price"]
        volume_24h = float(ticker.get("volume_24h_usd", 0.0))
        change_24h = float(ticker.get("change_24h", 0.0))
        spread_pct = float(ticker.get("spread_pct", 0.05))

        # 1. Trend Score (Max 20)
        t_score = 0.0
        if price > features_1h["ema_20"]: t_score += 5.0
        if features_1h["ema_20"] > features_1h["ema_50"]: t_score += 6.0
        if features_1h["ema_50"] > features_1h["ema_200"]: t_score += 4.0
        if features_1h["ema_slope_20"] > 0.05: t_score += 3.0
        if features_1h["adx_14"] > 22.0 and features_1h["plus_di"] > features_1h["minus_di"]: t_score += 2.0
        trend_score = min(20.0, t_score)

        # 2. Momentum Score (Max 20)
        m_score = 0.0
        rsi = features_1h["rsi_14"]
        if 52.0 <= rsi <= 68.0:
            m_score += 8.0
        elif 45.0 <= rsi < 52.0 or 68.0 < rsi <= 74.0:
            m_score += 5.0
        elif rsi > 75.0: # Extreme overbought
            m_score += 2.0

        if features_1h["macd_line"] > features_1h["macd_signal"]:
            m_score += 6.0
        if features_1h["macd_hist"] > 0:
            m_score += 6.0
        momentum_score = min(20.0, m_score)

        # 3. Volume Score (Max 20)
        v_score = 0.0
        rvol = features_1h["rvol_20"]
        if rvol >= 2.0:
            v_score += 8.0
        elif rvol >= 1.3:
            v_score += 6.0
        elif rvol >= 1.0:
            v_score += 4.0

        if features_1h["volume_accel"] > 0:
            v_score += 6.0
        if features_1h["obv"] > 0:
            v_score += 6.0
        volume_score = min(20.0, v_score)

        # 4. Structure Score (Max 20)
        s_score = 0.0
        stype = structure_1h.get("structure_type", "")
        if stype == "BREAKOUT":
            s_score += 18.0
        elif stype == "BULLISH_HH_HL":
            s_score += 16.0
        elif stype == "BULLISH_PULLBACK":
            s_score += 14.0
        elif stype == "SIDEWAYS_RANGE":
            s_score += 8.0
        else:
            s_score += 4.0

        if structure_1h.get("is_higher_low", False):
            s_score += 2.0
        structure_score = min(20.0, s_score)

        # 5. Liquidity & Spread Score (Max 10)
        l_score = 0.0
        if spread_pct <= 0.05:
            l_score += 5.0
        elif spread_pct <= 0.10:
            l_score += 4.0
        elif spread_pct <= 0.18:
            l_score += 2.0

        if volume_24h >= 50_000_000:
            l_score += 5.0
        elif volume_24h >= 15_000_000:
            l_score += 4.0
        else:
            l_score += 2.0
        liquidity_score = min(10.0, l_score)

        # 6. Volatility Score (Max 10)
        vol_score = 0.0
        atr_pct = (features_1h["atr_14"] / (price + 1e-9)) * 100.0
        if 1.2 <= atr_pct <= 4.0:
            vol_score += 5.0
        elif 0.8 <= atr_pct < 1.2 or 4.0 < atr_pct <= 6.0:
            vol_score += 3.0
        
        bb_w = features_1h["bb_width"]
        if bb_w > 0.04:  # healthy band expansion
            vol_score += 5.0
        else:
            vol_score += 3.0
        volatility_score = min(10.0, vol_score)

        # 7. Regime Alignment Score (Max 10)
        r_score = 0.0
        if regime == "BULL_TREND":
            r_score = 10.0
        elif regime == "BREAKOUT_ENVIRONMENT":
            r_score = 8.0
        elif regime == "SIDEWAYS":
            r_score = 5.0
        elif regime == "HIGH_VOLATILITY":
            r_score = 3.0
        else:
            r_score = 0.0
        regime_score = r_score

        # 8. Risk/Reward Score (Max 10)
        if estimated_rr >= 3.0:
            rr_score = 10.0
        elif estimated_rr >= 2.5:
            rr_score = 8.5
        elif estimated_rr >= 2.0:
            rr_score = 7.0
        elif estimated_rr >= 1.5:
            rr_score = 5.0
        else:
            rr_score = 2.0

        # Total Raw: Max 120 -> Normalized to 100
        raw_total = (trend_score + momentum_score + volume_score + 
                     structure_score + liquidity_score + volatility_score + 
                     regime_score + rr_score)
        total_normalized = round((raw_total / 120.0) * 100.0, 1)

        # Classification Grade
        if total_normalized >= 88.0:
            grade = "EXCELLENT"
        elif total_normalized >= 78.0:
            grade = "STRONG"
        elif total_normalized >= 68.0:
            grade = "GOOD"
        elif total_normalized >= 58.0:
            grade = "WATCH"
        else:
            grade = "IGNORE"

        # Win Probability Heuristic: base 45% + confluence
        prob = 0.45
        if trend_score >= 15: prob += 0.08
        if momentum_score >= 15: prob += 0.08
        if volume_score >= 15: prob += 0.06
        if structure_score >= 15: prob += 0.08
        if regime in ["BULL_TREND", "BREAKOUT_ENVIRONMENT"]: prob += 0.05
        prob = min(0.82, max(0.35, prob))

        return OpportunityScore(
            symbol=symbol,
            price=price,
            change_24h=change_24h,
            volume_24h=volume_24h,
            spread_pct=round(spread_pct, 4),
            trend_score=round(trend_score, 1),
            momentum_score=round(momentum_score, 1),
            volume_score=round(volume_score, 1),
            structure_score=round(structure_score, 1),
            liquidity_score=round(liquidity_score, 1),
            volatility_score=round(volatility_score, 1),
            regime_score=round(regime_score, 1),
            rr_score=round(rr_score, 1),
            total_score=total_normalized,
            grade=grade,
            win_probability=round(prob, 2)
        )

scoring_engine = OpportunityScoringEngine()
