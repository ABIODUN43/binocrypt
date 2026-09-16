from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from ..models.schemas import PaperAccountSummary, PaperPosition, PaperOrderRequest
from ..services.paper_trader import paper_trader

router = APIRouter(prefix="/paper", tags=["Paper Trading"])

@router.get("/account", response_model=PaperAccountSummary)
async def get_paper_account():
    """Returns live paper account metrics and $40 experiment progress."""
    return await paper_trader.get_account_summary()

@router.get("/positions", response_model=List[PaperPosition])
async def get_open_positions():
    """Returns all open paper positions with live unrealized PnL."""
    return await paper_trader.get_open_positions()

@router.get("/history", response_model=List[PaperPosition])
async def get_trade_history():
    """Returns historical closed paper trades."""
    return await paper_trader.get_trade_history()

@router.post("/order")
async def place_order(req: PaperOrderRequest):
    """Executes a risk-controlled paper trade order."""
    res = await paper_trader.place_order(req)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Order failed"))
    return res

@router.post("/close/{position_id}")
async def close_position(position_id: int):
    """Manually closes an open paper trade."""
    res = await paper_trader.close_position(position_id, exit_reason="MANUAL_CLOSE")
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Failed to close position"))
    return res

@router.post("/reset")
async def reset_paper_account(capital: float = 40.0):
    """Resets paper account back to initial capital (default $40.00)."""
    await paper_trader.reset_account(capital)
    return {"success": True, "message": f"Paper account successfully reset to ${capital:.2f}"}
