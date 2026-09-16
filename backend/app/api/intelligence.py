import time
import asyncio
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any

from ..services.intelligence_engine import intelligence_engine
from ..services.prediction_tracker import prediction_tracker
from ..services.universe import universe_service
from ..models.schemas import AssetMarketIntelligence, PredictionAccuracySummary

router = APIRouter(prefix="/intelligence", tags=["Market Intelligence & Regimes"])

recovery_cache = {
    "timestamp": 0.0,
    "data": []
}

@router.get("/scanner/recovery", response_model=List[AssetMarketIntelligence])
async def get_top_recovery_opportunities(limit: int = Query(6, ge=1, le=15)):
    """
    Scans top liquid crypto assets and ranks them by Bear Exhaustion Score
    and Recovery Probability using in-memory caching.
    """
    now = time.time()
    if recovery_cache["data"] and (now - recovery_cache["timestamp"] < 60.0):
        return recovery_cache["data"][:limit]

    try:
        symbols = ["ARBUSDT", "SOLUSDT", "SUIUSDT", "ADAUSDT", "LINKUSDT", "BTCUSDT"]
        sem = asyncio.Semaphore(2)

        async def fetch_one(s):
            async with sem:
                return await intelligence_engine.analyze_symbol(s)

        tasks = [fetch_one(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid = [r for r in results if isinstance(r, AssetMarketIntelligence)]

        ranked = sorted(
            valid,
            key=lambda x: (x.bear_exhaustion.score * 0.6 + x.regime_distribution.recovery * 0.4),
            reverse=True
        )

        recovery_cache["timestamp"] = now
        recovery_cache["data"] = ranked
        return ranked[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan recovery opportunities: {str(e)}")

@router.get("/history/{symbol:path}", response_model=PredictionAccuracySummary)
async def get_prediction_history(symbol: str):
    """
    Returns auditable prediction history, empirical directional accuracy,
    and verified past signals for the asset.
    """
    try:
        clean_symbol = symbol.replace("-", "/").upper()
        summary = await prediction_tracker.get_accuracy_summary(clean_symbol)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prediction history: {str(e)}")

@router.get("/{symbol:path}", response_model=AssetMarketIntelligence)
async def get_symbol_intelligence(symbol: str):
    """
    Returns full quantitative market intelligence for a specific cryptocurrency:
    - 7-State Regime Probability Distribution
    - 0-100 Bear Exhaustion Score & Sub-Factors
    - 0-100 Bull Maturity / Distribution Score
    - Multi-Horizon Forecast Scenarios (1D, 3D, 7D, 14D, 30D)
    - Decision Recommendation (WAIT, WATCH, ACCUMULATE, ENTER, etc.)
    - Forecast Windows (Transition dates, Accumulation window)
    - Structured Evidence & Risk Factors
    - Macro Market Hierarchy Alignment.
    """
    try:
        clean_symbol = symbol.replace("-", "/").upper()
        if not clean_symbol.endswith("/USDT") and not clean_symbol.endswith("USDT"):
            clean_symbol = f"{clean_symbol}/USDT"
        intel = await intelligence_engine.analyze_symbol(clean_symbol)
        
        # Asynchronously log prediction to SQLite tracking store
        asyncio.create_task(prediction_tracker.log_prediction(intel))
        
        return intel
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate intelligence for {symbol}: {str(e)}")

