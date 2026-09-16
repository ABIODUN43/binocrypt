from fastapi import APIRouter, Query
from typing import List, Dict, Any, Optional
import time
import asyncio
from ..services.universe import universe_service
from ..services.market_data import market_data
from ..services.features import feature_engine
from ..services.market_structure import market_structure_analyzer
from ..services.regime import regime_engine
from ..services.scoring import scoring_engine
from ..models.schemas import OpportunityScore

router = APIRouter(prefix="/scanner", tags=["Scanner"])

cached_opportunities: List[OpportunityScore] = []
last_scan_time: float = 0
SCAN_CACHE_TTL = 30.0  # 30 seconds cache

@router.get("/universe")
async def get_universe():
    """Returns the liquid tradable spot universe."""
    return await universe_service.get_tradable_universe()

@router.get("/opportunities", response_model=List[OpportunityScore])
async def get_opportunities(
    limit: int = Query(25, ge=5, le=100),
    min_score: float = Query(0.0, ge=0.0, le=100.0),
    force_refresh: bool = Query(False)
):
    """
    Runs quantitative scan over liquid spot universe, calculates multi-timeframe
    features, scores candidates, and returns ranked opportunities.
    """
    global cached_opportunities, last_scan_time
    now = time.time()

    if cached_opportunities and (now - last_scan_time < SCAN_CACHE_TTL) and not force_refresh:
        filtered = [o for o in cached_opportunities if o.total_score >= min_score]
        return filtered[:limit]

    # Detect current regime
    regime = await regime_engine.detect_regime()
    universe = await universe_service.get_tradable_universe()
    
    # Analyze top candidates (up to top 30 liquid coins)
    scan_universe = universe[:30]
    
    async def analyze_coin(t: Dict[str, Any]) -> Optional[OpportunityScore]:
        sym = t["symbol"]
        try:
            kl_1h = await market_data.get_klines(sym, interval="1h", limit=50)
            if not kl_1h or len(kl_1h) < 30:
                return None
            
            feat_1h = feature_engine.calculate_indicators(kl_1h)
            if not feat_1h:
                return None
            
            struct_1h = market_structure_analyzer.analyze_structure(kl_1h)
            
            score = scoring_engine.score_candidate(
                symbol=sym,
                ticker=t,
                features_1h=feat_1h,
                features_4h=None,
                structure_1h=struct_1h,
                regime=regime.regime,
                estimated_rr=2.2
            )
            return score
        except Exception as e:
            print(f"[Scanner] Error analyzing {sym}: {e}")
            return None

    # Run in parallel batches of 10
    tasks = [analyze_coin(t) for t in scan_universe]
    results = await asyncio.gather(*tasks)

    valid_scores = [r for r in results if r is not None]
    
    # Sort descending by total score
    valid_scores.sort(key=lambda x: x.total_score, reverse=True)
    
    # Assign ranks
    for idx, sc in enumerate(valid_scores, 1):
        sc.rank = idx

    cached_opportunities = valid_scores
    last_scan_time = now

    filtered = [o for o in valid_scores if o.total_score >= min_score]
    return filtered[:limit]
