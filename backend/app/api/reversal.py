from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from ..services.reversal_engine import reversal_engine

router = APIRouter(prefix="/reversal", tags=["reversal-intelligence"])

@router.get("/scanner")
async def get_reversal_scanner():
    """
    BR-007 Universe Reversal & Bottom-Formation Scanner.
    Scans the liquid crypto universe for active exhaustion, base-forming, and confirmed reversal states.
    """
    try:
        items = await reversal_engine.get_reversal_scanner()
        return {
            "success": True,
            "count": len(items),
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan universe for bottom formations: {str(e)}")

@router.get("/{symbol}")
async def get_symbol_reversal_intelligence(symbol: str):
    """
    BR-007 Deep Bottom-Formation Intelligence for an individual asset.
    Returns:
    - Finite State Machine state (DECLINING, DOWNSIDE_EXHAUSTION_WATCH, BASE_FORMING, REVERSAL_CONFIRMED)
    - Potential-Bottom Zone (Estimated floor, upper envelope, invalidation stop)
    - Triple-Barrier Event Targets (+6%/-3%/72h, etc.)
    - Causal 21-feature radar metrics and execution setup
    """
    try:
        data = await reversal_engine.get_reversal_intelligence(symbol)
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute bottom intelligence for {symbol}: {str(e)}")
