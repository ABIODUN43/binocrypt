from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from ..services.market_data import market_data
from ..services.features import feature_engine
from ..services.market_structure import market_structure_analyzer
from ..services.regime import regime_engine
from ..services.scoring import scoring_engine
from ..services.setup_engine import setup_engine
from ..models.schemas import TradeSetup

router = APIRouter(prefix="/setups", tags=["Setups"])

@router.get("/active", response_model=List[TradeSetup])
async def get_active_setups(limit: int = Query(5, ge=1, le=20)):
    """Returns top active setups across the liquid market."""
    from .scanner import get_opportunities
    opportunities = await get_opportunities(limit=15)
    
    setups = []
    for opp in opportunities:
        if opp.total_score < 60.0:
            continue
        try:
            kl_1h = await market_data.get_klines(opp.symbol, interval="1h", limit=50)
            feat_1h = feature_engine.calculate_indicators(kl_1h)
            struct_1h = market_structure_analyzer.analyze_structure(kl_1h)
            if feat_1h:
                setup = setup_engine.generate_setup(opp, feat_1h, struct_1h)
                setups.append(setup)
        except Exception as e:
            print(f"[Setups] Error generating setup for {opp.symbol}: {e}")

    # Prioritize READY or SETUP_FORMING
    setups.sort(key=lambda s: (s.signal_state == "READY", s.score), reverse=True)
    return setups[:limit]

@router.get("/{symbol}", response_model=TradeSetup)
async def get_symbol_setup(symbol: str):
    """Calculates live trade setup for a specific crypto asset instantly."""
    from .scanner import cached_opportunities
    sym = symbol.upper()
    if not sym.endswith("USDT") and not sym.endswith("BTC"):
        sym = f"{sym}USDT"

    # Fast check: If available in scanner cache, reuse the scored opportunity
    matching_opp = next((o for o in cached_opportunities if o.symbol == sym), None)

    kl_1h = await market_data.get_klines(sym, interval="1h", limit=50)
    if not kl_1h or len(kl_1h) < 15:
        raise HTTPException(status_code=400, detail=f"Insufficient historical data for {sym}")

    feat_1h = feature_engine.calculate_indicators(kl_1h)
    struct_1h = market_structure_analyzer.analyze_structure(kl_1h)

    if matching_opp:
        setup = setup_engine.generate_setup(matching_opp, feat_1h, struct_1h)
        return setup

    # Fallback if not yet in scanner cache
    ticker = await market_data.get_ticker(sym)
    if not ticker:
        ticker = market_data.ticker_map.get(sym)

    if not ticker:
        ticker = {
            "symbol": sym,
            "quoteVolume": 25000000.0,
            "priceChangePercent": 0.0,
            "lastPrice": kl_1h[-1][4]
        }

    ticker_dict = {
        "symbol": sym,
        "volume_24h_usd": float(ticker.get("quoteVolume", 25000000.0)),
        "change_24h": float(ticker.get("priceChangePercent", 0.0)),
        "spread_pct": 0.05
    }

    if regime_engine.last_regime:
        regime_str = regime_engine.last_regime.regime
    else:
        regime_obj = await regime_engine.detect_regime()
        regime_str = regime_obj.regime

    score = scoring_engine.score_candidate(
        symbol=sym,
        ticker=ticker_dict,
        features_1h=feat_1h,
        features_4h=None,
        structure_1h=struct_1h,
        regime=regime_str,
        estimated_rr=2.2
    )

    setup = setup_engine.generate_setup(score, feat_1h, struct_1h)
    return setup
