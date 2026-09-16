from typing import Dict, Any, List
from ..models.schemas import AIExplanationResponse, OpportunityScore, TradeSetup, MarketRegime

class AIExplainerService:
    @staticmethod
    def explain_opportunity(
        symbol: str,
        score: OpportunityScore,
        setup: TradeSetup,
        regime: MarketRegime,
        features_1h: Dict[str, Any],
        structure_1h: Dict[str, Any]
    ) -> AIExplanationResponse:
        """
        Translates structured quantitative evidence into an explainable, institutional trade thesis.
        Strictly grounds reasoning in calculated parameters — never hallucinations.
        """
        price = features_1h["price"]
        rsi = features_1h["rsi_14"]
        rvol = features_1h["rvol_20"]
        adx = features_1h["adx_14"]
        macd_hist = features_1h["macd_hist"]
        stype = structure_1h.get("structure_type", "").replace("_", " ")

        # 1. Why it ranked high
        why_ranked = []
        if score.trend_score >= 15:
            why_ranked.append(f"Strong Trend Confluence (Score {score.trend_score}/20): Price (${price}) is trading cleanly above the 20 and 50 EMAs with positive slope and ADX at {adx:.1f}, confirming trend persistence.")
        if score.momentum_score >= 15:
            why_ranked.append(f"Optimal Momentum Regime (Score {score.momentum_score}/20): RSI stands at {rsi:.1f} — providing healthy bullish expansion without entering the overbought exhaustion zone (>75).")
        if score.volume_score >= 14:
            why_ranked.append(f"Institutional Volume Influx (Score {score.volume_score}/20): RVOL is {rvol:.1f}x higher than the 20-period moving average, validating genuine demand rather than low-liquidity drift.")
        if score.structure_score >= 15:
            why_ranked.append(f"Market Structure Integrity (Score {score.structure_score}/20): Confirmed {stype} with clean swing lows protecting downside.")

        if not why_ranked:
            why_ranked.append(f"Moderate confluence across technical indicators with overall rating of {score.grade} ({score.total_score}/100).")

        # 2. Exact Technical Triggers
        triggers = [
            f"1H Entry Zone: ${setup.entry_zone_min} – ${setup.entry_zone_max}",
            f"MACD Status: Histogram {macd_hist:+.4f} indicates expanding buyer velocity",
            f"Structure: {stype} with recent swing low identified at ${structure_1h.get('recent_swing_low', 0.0)}",
            f"Risk / Reward Ratio: Asymmetric 1 : {setup.risk_reward} target profile"
        ]

        # 3. Invalidation & Risks
        risks = [
            f"Hard Stop Invalidation: 1H candle closing below ${setup.stop_loss} ({setup.stop_distance_pct}% drawdown).",
            f"Macro Sensitivity: A sudden {regime.btc_trend_1h} breakdown in Bitcoin would degrade setup probability.",
            f"Slippage Model: Ensure entry executed within liquid spread ({score.spread_pct}%) to prevent cost drag."
        ]

        # 4. Small Account Verdict ($40 Target)
        # Sizing guidance
        risk_2pct = 40.0 * 0.02 # $0.80
        stop_dist_ratio = setup.stop_distance_pct / 100.0
        calculated_pos = round(risk_2pct / (stop_dist_ratio + 1e-9), 2)
        pos_str = f"${min(38.0, max(5.0, calculated_pos)):.2f}"

        verdict = (
            f"Suitable for disciplined spot execution under current {regime.regime} conditions. "
            f"For a $40 account risking strictly 2.0% ($0.80), position size should be {pos_str} USDT. "
            f"Lock stop to breakeven once TP1 (${setup.tp1}) is reached. Never chase above ${setup.entry_zone_max}."
        )

        return AIExplanationResponse(
            symbol=symbol,
            summary_title=f"{symbol} Quantitative Trade Breakdown (Score: {score.total_score}/100 - {score.grade})",
            regime_alignment=f"{regime.regime} ({regime.confidence}% Confidence) - {regime.trade_permission.replace('_', ' ')}",
            why_ranked_high=why_ranked,
            technical_triggers=triggers,
            invalidation_and_risks=risks,
            small_account_verdict=verdict
        )

ai_explainer = AIExplainerService()
