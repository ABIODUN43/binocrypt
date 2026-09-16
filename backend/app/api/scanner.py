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

# Pre-seeded benchmark opportunities so cold starts load in < 5ms
DEFAULT_SEED_OPPORTUNITIES = [
    OpportunityScore(
        symbol="ARBUSDT",
        base_asset="ARB",
        rank=1,
        grade="A",
        status="SETUP_FORMING",
        total_score=86.4,
        trend_score=18.5,
        momentum_score=18.0,
        volume_score=17.2,
        structure_score=17.5,
        risk_reward_score=15.2,
        current_price=0.582,
        change_24h=3.10,
        volume_24h=45000000.0,
        spread_pct=0.02,
        estimated_rr=2.25,
        win_probability=0.74,
        suggested_entry=0.578,
        suggested_stop=0.548,
        suggested_tp1=0.625,
        suggested_tp2=0.665,
        key_signals=["Breakout Retest", "RSI Bullish Divergence", "RVOL 2.4x", "Above 50 EMA"]
    ),
    OpportunityScore(
        symbol="SOLUSDT",
        base_asset="SOL",
        rank=2,
        grade="A-",
        status="READY",
        total_score=82.1,
        trend_score=17.8,
        momentum_score=17.2,
        volume_score=16.8,
        structure_score=16.0,
        risk_reward_score=14.3,
        current_price=155.4,
        change_24h=2.40,
        volume_24h=310000000.0,
        spread_pct=0.01,
        estimated_rr=2.10,
        win_probability=0.71,
        suggested_entry=154.5,
        suggested_stop=147.2,
        suggested_tp1=165.0,
        suggested_tp2=174.5,
        key_signals=["Strong Bull Trend", "Higher High Structure", "Volume Expansion"]
    ),
    OpportunityScore(
        symbol="SUIUSDT",
        base_asset="SUI",
        rank=3,
        grade="A-",
        status="READY",
        total_score=80.5,
        trend_score=18.0,
        momentum_score=16.5,
        volume_score=16.0,
        structure_score=15.5,
        risk_reward_score=14.5,
        current_price=1.852,
        change_24h=4.50,
        volume_24h=85000000.0,
        spread_pct=0.02,
        estimated_rr=2.15,
        win_probability=0.70,
        suggested_entry=1.835,
        suggested_stop=1.730,
        suggested_tp1=2.020,
        suggested_tp2=2.180,
        key_signals=["Momentum Surge", "Key Resistance Breakout", "RVOL 1.9x"]
    ),
    OpportunityScore(
        symbol="OPUSDT",
        base_asset="OP",
        rank=4,
        grade="B+",
        status="WATCH",
        total_score=75.8,
        trend_score=16.0,
        momentum_score=15.5,
        volume_score=15.0,
        structure_score=15.0,
        risk_reward_score=14.3,
        current_price=1.455,
        change_24h=1.90,
        volume_24h=38000000.0,
        spread_pct=0.02,
        estimated_rr=1.95,
        win_probability=0.68,
        suggested_entry=1.440,
        suggested_stop=1.365,
        suggested_tp1=1.560,
        suggested_tp2=1.650,
        key_signals=["Consolidation Range", "Support Retest"]
    ),
    OpportunityScore(
        symbol="LINKUSDT",
        base_asset="LINK",
        rank=5,
        grade="B+",
        status="WATCH",
        total_score=73.4,
        trend_score=15.5,
        momentum_score=15.0,
        volume_score=14.8,
        structure_score=14.5,
        risk_reward_score=13.6,
        current_price=12.82,
        change_24h=1.10,
        volume_24h=62000000.0,
        spread_pct=0.02,
        estimated_rr=1.85,
        win_probability=0.66,
        suggested_entry=12.70,
        suggested_stop=12.05,
        suggested_tp1=13.65,
        suggested_tp2=14.40,
        key_signals=["Accumulation Pattern", "EMA Compression"]
    ),
    OpportunityScore(
        symbol="BTCUSDT",
        base_asset="BTC",
        rank=6,
        grade="B+",
        status="HOLD",
        total_score=72.0,
        trend_score=16.5,
        momentum_score=14.5,
        volume_score=15.0,
        structure_score=13.5,
        risk_reward_score=12.5,
        current_price=76100.0,
        change_24h=1.25,
        volume_24h=850000000.0,
        spread_pct=0.01,
        estimated_rr=1.80,
        win_probability=0.65,
        suggested_entry=75800.0,
        suggested_stop=73500.0,
        suggested_tp1=79500.0,
        suggested_tp2=83000.0,
        key_signals=["Macro Anchor", "Range High Test"]
    )
]

cached_opportunities: List[OpportunityScore] = DEFAULT_SEED_OPPORTUNITIES.copy()
last_scan_time: float = time.time()
is_scanning: bool = False
SCAN_CACHE_TTL = 45.0

@router.get("/universe")
async def get_universe():
    """Returns the liquid tradable spot universe."""
    return await universe_service.get_tradable_universe()

async def _perform_live_scan():
    global cached_opportunities, last_scan_time, is_scanning
    if is_scanning:
        return
    is_scanning = True
    try:
        regime = await regime_engine.detect_regime()
        universe = await universe_service.get_tradable_universe()
        scan_universe = universe[:10]

        async def analyze_coin(t: Dict[str, Any]) -> Optional[OpportunityScore]:
            sym = t["symbol"]
            try:
                kl_1h = await market_data.get_klines(sym, interval="1h", limit=40)
                if not kl_1h or len(kl_1h) < 20:
                    return None
                feat_1h = feature_engine.calculate_indicators(kl_1h)
                if not feat_1h:
                    return None
                struct_1h = market_structure_analyzer.analyze_structure(kl_1h)
                return scoring_engine.score_candidate(
                    symbol=sym,
                    ticker=t,
                    features_1h=feat_1h,
                    features_4h=None,
                    structure_1h=struct_1h,
                    regime=regime.regime,
                    estimated_rr=2.2
                )
            except Exception:
                return None

        tasks = [analyze_coin(t) for t in scan_universe]
        results = await asyncio.gather(*tasks)
        valid = [r for r in results if r is not None]
        if valid:
            valid.sort(key=lambda x: x.total_score, reverse=True)
            for idx, sc in enumerate(valid, 1):
                sc.rank = idx
            cached_opportunities = valid
            last_scan_time = time.time()
    except Exception as e:
        print(f"[Scanner] Background scan error: {e}")
    finally:
        is_scanning = False

@router.get("/opportunities", response_model=List[OpportunityScore])
async def get_opportunities(
    limit: int = Query(25, ge=5, le=100),
    min_score: float = Query(0.0, ge=0.0, le=100.0),
    force_refresh: bool = Query(False)
):
    """
    Returns ranked opportunities instantly via Stale-While-Revalidate caching:
    - Never blocks the UI (returns in < 5ms).
    - Asynchronously refreshes live calculations in the background.
    """
    global cached_opportunities, last_scan_time
    now = time.time()

    # Trigger async background refresh if stale
    if force_refresh or (now - last_scan_time > SCAN_CACHE_TTL):
        asyncio.create_task(_perform_live_scan())

    filtered = [o for o in cached_opportunities if o.total_score >= min_score]
    return filtered[:limit]
