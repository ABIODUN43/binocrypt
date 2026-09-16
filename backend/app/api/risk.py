from fastapi import APIRouter
from ..models.schemas import RiskSizingRequest, RiskSizingResponse
from ..services.risk_engine import risk_engine
from ..core.config import settings

router = APIRouter(prefix="/risk", tags=["Risk Engine"])

@router.post("/calculate", response_model=RiskSizingResponse)
async def calculate_risk(req: RiskSizingRequest):
    """Calculates position sizing strictly from stop loss distance for small accounts."""
    return risk_engine.calculate_position_size(
        account_equity=req.account_equity,
        risk_pct=req.risk_pct,
        entry_price=req.entry_price,
        stop_loss=req.stop_loss
    )

@router.get("/rules")
async def get_risk_rules():
    """Returns system risk parameters and small account limits."""
    return {
        "starting_capital": settings.DEFAULT_CAPITAL,
        "max_risk_per_trade_pct": settings.MAX_RISK_PER_TRADE_PCT,
        "max_risk_per_trade_usd": settings.DEFAULT_CAPITAL * (settings.MAX_RISK_PER_TRADE_PCT / 100.0),
        "max_open_positions": settings.MAX_OPEN_POSITIONS,
        "max_daily_drawdown_pct": settings.MAX_DAILY_DRAWDOWN_PCT,
        "max_total_drawdown_pct": settings.MAX_TOTAL_DRAWDOWN_PCT,
        "min_order_notional_usd": settings.MIN_ORDER_NOTIONAL,
        "formula": "Position Size = Risk Amount ($) / ((Entry - Stop) / Entry)",
        "objective": "Capital preservation and positive expectancy on small accounts without leverage."
    }
