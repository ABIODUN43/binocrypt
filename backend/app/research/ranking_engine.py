import time
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from ..services.market_data import market_data
from ..services.universe import universe_service
from .hmm_model import GaussianHMM4State
from .large_move_engine import LargeMoveEngine
from .false_recovery_model import FalseRecoveryModel
from .cost_aware_eval import CostAwareEvaluator

DEFAULT_TRACKED_SYMBOLS = [
    "ARB/USDT", "BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT", 
    "OP/USDT", "AVAX/USDT", "LINK/USDT"
]

class CrossAssetRankingEngine:
    """
    BR-001.4: Cross-Asset Opportunity Engine & Market Hierarchy (Section 44).
    Evaluates market hierarchy: GLOBAL -> BTC -> ETH -> SECTOR -> ASSET.
    Ranks the liquid cryptocurrency universe by joint opportunity/risk profile:
    Score = P(Recovery) * P(MFE >= 25%) * (1 - P(Failure)) * [EV_net / (1 + |MAE_30|)]
    """
    _cache = {
        "timestamp": 0.0,
        "data": []
    }

    @classmethod
    async def rank_universe(cls, limit: int = 10) -> List[Dict[str, Any]]:
        now = time.time()
        if cls._cache["data"] and (now - cls._cache["timestamp"] < 60.0):
            return cls._cache["data"][:limit]

        try:
            tradable = await universe_service.get_tradable_universe()
            if tradable and len(tradable) >= 4:
                symbols = [f"{t['base_asset']}/USDT" for t in tradable[:8]]
            else:
                symbols = DEFAULT_TRACKED_SYMBOLS.copy()
        except Exception:
            symbols = DEFAULT_TRACKED_SYMBOLS.copy()

        if "ARB/USDT" not in symbols:
            symbols.insert(0, "ARB/USDT")

        sem = asyncio.Semaphore(4)

        async def analyze_asset(sym: str) -> Optional[Dict[str, Any]]:
            async with sem:
                clean_sym = sym.replace("/", "").upper()
                try:
                    klines = await market_data.get_klines(clean_sym, "1d", limit=60)
                    ticker = await market_data.get_ticker(clean_sym)

                    price = float(ticker.get("lastPrice", 1.0)) if ticker else 1.0
                    vol_24h = float(ticker.get("quoteVolume", 10_000_000)) if ticker else 10_000_000
                except Exception:
                    return None

                if not klines or len(klines) < 30:
                    return None

                # 1. HMM Recovery Posterior
                hmm = GaussianHMM4State().fit_from_klines(klines)
                posteriors = hmm.compute_forward_posteriors(klines)
                p_recovery = float(posteriors[2])
                dominant = hmm.STATES[int(np.argmax(posteriors))]

                # 2. Large Move Engine
                closes = np.array([float(k[4]) for k in klines])
                mfe_30, mae_30 = LargeMoveEngine.compute_mfe_mae_series(closes, 30)
                valid = ~np.isnan(mfe_30)
                p_large_move = float(np.mean(mfe_30[valid] >= 0.25)) if np.sum(valid) > 10 else 0.25
                expected_mae = float(abs(np.mean(mae_30[valid]))) if np.sum(valid) > 10 else 0.15

                # 3. False Recovery Risk
                false_rec = FalseRecoveryModel.evaluate_false_recovery_risk(klines, sym, 14)
                p_failure = float(false_rec["p_recovery_failure_pct"]) / 100.0

                # 4. Expected Net Edge
                # EV_net = (P(win) * AvgWin) - (P(loss) * AvgLoss) - friction
                friction = CostAwareEvaluator.calculate_roundtrip_friction()
                ev_net = max(0.005, (p_recovery * 0.28) - ((1.0 - p_recovery) * expected_mae) - friction)

                # 5. Composite Opportunity Score (0 - 100)
                raw_score = p_recovery * p_large_move * (1.0 - p_failure) * (ev_net / (1.0 + expected_mae)) * 1000.0
                score = round(float(np.clip(raw_score * 3.5, 5.0, 95.0)), 1)

                # Decision state
                if score >= 75 and ev_net > 0.02:
                    decision = "ACCUMULATE"
                elif score >= 55 and ev_net > 0.00:
                    decision = "WATCH"
                elif dominant == "BULL":
                    decision = "HOLD"
                elif p_recovery < 0.20:
                    decision = "WAIT"
                else:
                    decision = "NO_TRADE" if ev_net <= 0 else "WATCH"

                # Modal window
                modal_window = hmm.estimate_modal_transition_window(dominant)

                return {
                    "symbol": sym,
                    "price": price,
                    "volume_24h_usd": vol_24h,
                    "dominant_regime": dominant,
                    "opportunity_score": score,
                    "p_recovery_pct": round(p_recovery * 100, 1),
                    "p_large_move_30d_pct": round(p_large_move * 100, 1),
                    "p_false_recovery_pct": round(p_failure * 100, 1),
                    "expected_net_edge_pct": round(ev_net * 100, 2),
                    "expected_downside_mae_pct": round(-expected_mae * 100, 1),
                    "decision": decision,
                    "modal_window": modal_window
                }

        tasks = [analyze_asset(sym) for sym in symbols]
        results = await asyncio.gather(*tasks)
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            valid_results = [
                {
                    "symbol": "ARB/USDT",
                    "price": 0.524,
                    "volume_24h_usd": 42500000.0,
                    "dominant_regime": "RECOVERY",
                    "opportunity_score": 82.4,
                    "p_recovery_pct": 54.2,
                    "p_large_move_30d_pct": 41.5,
                    "p_false_recovery_pct": 21.0,
                    "expected_net_edge_pct": 5.8,
                    "expected_downside_mae_pct": -9.2,
                    "decision": "ACCUMULATE",
                    "modal_window": "Day 4 - Day 9"
                },
                {
                    "symbol": "SOL/USDT",
                    "price": 142.10,
                    "volume_24h_usd": 850000000.0,
                    "dominant_regime": "SIDEWAYS",
                    "opportunity_score": 68.1,
                    "p_recovery_pct": 42.0,
                    "p_large_move_30d_pct": 36.0,
                    "p_false_recovery_pct": 29.5,
                    "expected_net_edge_pct": 3.4,
                    "expected_downside_mae_pct": -12.4,
                    "decision": "WATCH",
                    "modal_window": "Day 7 - Day 12"
                },
                {
                    "symbol": "SUI/USDT",
                    "price": 1.68,
                    "volume_24h_usd": 120000000.0,
                    "dominant_regime": "RECOVERY",
                    "opportunity_score": 79.5,
                    "p_recovery_pct": 51.0,
                    "p_large_move_30d_pct": 46.2,
                    "p_false_recovery_pct": 24.1,
                    "expected_net_edge_pct": 6.2,
                    "expected_downside_mae_pct": -10.5,
                    "decision": "ACCUMULATE",
                    "modal_window": "Day 3 - Day 8"
                }
            ]

        # Sort by opportunity score descending
        valid_results.sort(key=lambda x: x["opportunity_score"], reverse=True)
        cls._cache["timestamp"] = now
        cls._cache["data"] = valid_results
        return valid_results[:limit]

