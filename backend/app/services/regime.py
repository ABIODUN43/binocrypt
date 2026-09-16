import time
import asyncio
from typing import Dict, Any, List, Optional
from .market_data import market_data
from .features import feature_engine
from ..models.schemas import MarketRegime

class MarketRegimeEngine:
    def __init__(self):
        self.last_regime: Optional[MarketRegime] = None
        self.last_calculated: float = 0
        self.cache_ttl: float = 45.0  # 45 seconds cache

    async def detect_regime(self) -> MarketRegime:
        """
        Determines market regime using BTC, ETH, market breadth, and volatility.
        """
        now = time.time()
        if self.last_regime and (now - self.last_calculated < self.cache_ttl):
            return self.last_regime

        # Fetch BTC & ETH concurrently
        btc_4h_task = market_data.get_klines("BTCUSDT", interval="4h", limit=60)
        btc_1h_task = market_data.get_klines("BTCUSDT", interval="1h", limit=60)
        eth_4h_task = market_data.get_klines("ETHUSDT", interval="4h", limit=60)
        eth_1h_task = market_data.get_klines("ETHUSDT", interval="1h", limit=60)
        btc_ticker_task = market_data.get_ticker("BTCUSDT")

        btc_4h, btc_1h, eth_4h, eth_1h, btc_ticker = await asyncio.gather(
            btc_4h_task, btc_1h_task, eth_4h_task, eth_1h_task, btc_ticker_task
        )

        if not btc_ticker:
            btc_ticker = market_data.ticker_map.get("BTCUSDT")

        if btc_ticker:
            btc_price = float(btc_ticker.get("lastPrice", 0.0))
            btc_change_24h = float(btc_ticker.get("priceChangePercent", 0.0))
        elif btc_1h:
            btc_price = float(btc_1h[-1][4])
            btc_change_24h = 0.0
        else:
            btc_price = 0.0
            btc_change_24h = 0.0

        btc_feat_4h = feature_engine.calculate_indicators(btc_4h) if btc_4h else None
        btc_feat_1h = feature_engine.calculate_indicators(btc_1h) if btc_1h else None
        eth_feat_4h = feature_engine.calculate_indicators(eth_4h) if eth_4h else None
        eth_feat_1h = feature_engine.calculate_indicators(eth_1h) if eth_1h else None

        # Determine individual trends
        def get_trend(feat):
            if not feat:
                return "NEUTRAL"
            price = feat["price"]
            ema20 = feat["ema_20"]
            ema50 = feat["ema_50"]
            slope = feat["ema_slope_20"]
            if price > ema20 and ema20 > ema50 and slope > 0.1:
                return "STRONG_BULLISH"
            elif price > ema50:
                return "BULLISH"
            elif price < ema20 and ema20 < ema50 and slope < -0.1:
                return "STRONG_BEARISH"
            elif price < ema50:
                return "BEARISH"
            return "SIDEWAYS"

        btc_trend_4h = get_trend(btc_feat_4h)
        btc_trend_1h = get_trend(btc_feat_1h)
        eth_trend_4h = get_trend(eth_feat_4h)
        eth_trend_1h = get_trend(eth_feat_1h)

        # Volatility assessment (BTC ATR / Price)
        vol_pct = (btc_feat_1h["atr_14"] / (btc_price + 1e-9)) * 100 if (btc_feat_1h and btc_price > 0) else 1.5
        if vol_pct > 3.5:
            vol_state = "EXTREME"
        elif vol_pct > 2.2:
            vol_state = "HIGH"
        elif vol_pct < 0.8:
            vol_state = "LOW"
        else:
            vol_state = "NORMAL"

        # Market breadth estimation across key bellwethers in parallel
        bellwethers = ["SOLUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "NEARUSDT", "LINKUSDT", "DOGEUSDT", "SUIUSDT"]
        bell_tasks = [market_data.get_klines(s, interval="1h", limit=55) for s in bellwethers]
        bell_results = await asyncio.gather(*bell_tasks)

        above_ema50_count = 0
        total_tested = 0
        for kl in bell_results:
            if kl:
                f = feature_engine.calculate_indicators(kl)
                if f:
                    total_tested += 1
                    if f["price"] > f["ema_50"]:
                        above_ema50_count += 1

        breadth_pct = (above_ema50_count / total_tested * 100.0) if total_tested > 0 else 50.0

        # Regime classification synthesis
        bull_points = 0
        bear_points = 0

        if "BULLISH" in btc_trend_4h: bull_points += 3
        if "BEARISH" in btc_trend_4h: bear_points += 3
        if "BULLISH" in btc_trend_1h: bull_points += 2
        if "BEARISH" in btc_trend_1h: bear_points += 2
        if "BULLISH" in eth_trend_4h: bull_points += 2
        if "BEARISH" in eth_trend_4h: bear_points += 2
        if breadth_pct > 65: bull_points += 3
        elif breadth_pct < 35: bear_points += 3

        if vol_state == "EXTREME" and btc_change_24h < -5.0:
            regime = "PANIC"
            confidence = 88
            trade_perm = "NO_TRADE"
            summary = "Market capitulation / extreme panic selling. High risk of slippage and whipsaws. All long entries blocked."
        elif vol_state in ["HIGH", "EXTREME"] and abs(bull_points - bear_points) <= 2:
            regime = "HIGH_VOLATILITY"
            confidence = 78
            trade_perm = "DEFENSIVE_CASH"
            summary = "Elevated macro volatility with erratic direction. Strict capital preservation enforced."
        elif bull_points >= 7 and breadth_pct >= 60:
            regime = "BULL_TREND"
            confidence = min(95, 60 + bull_points * 4)
            trade_perm = "AGGRESSIVE_LONGS"
            summary = "Strong bullish macro trend across BTC, ETH, and broad altcoin universe. High probability for spot trend setups."
        elif bear_points >= 7 and breadth_pct <= 35:
            regime = "BEAR_TREND"
            confidence = min(95, 60 + bear_points * 4)
            trade_perm = "NO_TRADE"
            summary = "Bearish macro structure. Altcoins facing systemic downward pressure. Cash defense active."
        elif vol_state == "LOW" and 40 <= breadth_pct <= 60:
            regime = "SIDEWAYS"
            confidence = 72
            trade_perm = "SELECTIVE_LONGS"
            summary = "Range-bound / consolidating market. Momentum plays carry lower expectancy. Only high-confluence support bounces permitted."
        else:
            regime = "BREAKOUT_ENVIRONMENT"
            confidence = 70
            trade_perm = "SELECTIVE_LONGS"
            summary = "Mixed conditions transitioning into breakout expansion. Selective opportunities with tight stops."

        res = MarketRegime(
            regime=regime,
            confidence=confidence,
            btc_price=btc_price,
            btc_change_24h=btc_change_24h,
            btc_trend_4h=btc_trend_4h,
            btc_trend_1h=btc_trend_1h,
            eth_trend_4h=eth_trend_4h,
            eth_trend_1h=eth_trend_1h,
            market_breadth_pct=round(breadth_pct, 1),
            volatility_state=vol_state,
            trade_permission=trade_perm,
            summary=summary
        )
        self.last_regime = res
        self.last_calculated = now
        return res

regime_engine = MarketRegimeEngine()
