from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from ..services.reversal_engine import reversal_engine
from ..services.signal_guard import signal_guard

router = APIRouter(prefix="/reversal", tags=["reversal-intelligence"])

class ExecuteReversalSignalRequest(BaseModel):
    symbol: str
    signal_id: Optional[str] = None
    client_order_id: Optional[str] = None

@router.post("/execute")
async def execute_reversal_signal(req: ExecuteReversalSignalRequest):
    """
    BR-008: Atomic Reversal Signal Execution Endpoint.
    Validates signal lifecycle, feed freshness, target proximity, and prevents duplicate execution.
    """
    try:
        res = await signal_guard.execute_signal(
            symbol=req.symbol,
            signal_id=req.signal_id,
            client_order_id=req.client_order_id
        )
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("error", "Execution rejected."))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

@router.get("/signals/{symbol}")
async def get_symbol_signals(symbol: str):
    """Fetch persistent signal lifecycle records for an asset."""
    try:
        clean_sym = symbol.upper().replace("/", "").replace("-", "")
        sig_id = await signal_guard._get_latest_actionable_signal_id(clean_sym)
        sig = await signal_guard._get_signal_record(sig_id) if sig_id else None
        return {"success": True, "signal": sig}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch signals for {symbol}: {str(e)}")

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
