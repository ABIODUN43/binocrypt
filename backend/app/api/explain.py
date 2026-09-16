from fastapi import APIRouter, HTTPException
from ..services.market_data import market_data
from ..services.features import feature_engine
from ..services.market_structure import market_structure_analyzer
from ..services.regime import regime_engine
from ..services.scoring import scoring_engine
from ..services.setup_engine import setup_engine
from ..services.ai_explainer import ai_explainer
from ..models.schemas import AIExplanationResponse

router = APIRouter(prefix="/explain", tags=["AI Quantitative Explainer"])

@router.get("/{symbol}", response_model=AIExplanationResponse)
async def explain_symbol(symbol: str):
    """
    Synthesizes mathematical indicator features and market structure into
    transparent, grounded explanations without hallucinations.
    """
    sym = symbol.upper()
    ticker = await market_data.get_ticker(sym)
    if not ticker:
        ticker = market_data.ticker_map.get(sym)

    kl_1h = await market_data.get_klines(sym, interval="1h", limit=50)
    if not kl_1h or len(kl_1h) < 30:
        raise HTTPException(status_code=400, detail=f"Insufficient candle history for {sym}")

    if not ticker:
        ticker = {
            "symbol": sym,
            "quoteVolume": 25000000.0,
            "priceChangePercent": 0.0,
            "lastPrice": kl_1h[-1][4]
        }

    feat_1h = feature_engine.calculate_indicators(kl_1h)
    struct_1h = market_structure_analyzer.analyze_structure(kl_1h)
    regime = await regime_engine.detect_regime()

    ticker_dict = {
        "symbol": sym,
        "volume_24h_usd": float(ticker.get("quoteVolume", 0.0)),
        "change_24h": float(ticker.get("priceChangePercent", 0.0)),
        "spread_pct": 0.05
    }

    score = scoring_engine.score_candidate(
        symbol=sym,
        ticker=ticker_dict,
        features_1h=feat_1h,
        features_4h=None,
        structure_1h=struct_1h,
        regime=regime.regime,
        estimated_rr=2.2
    )

    setup = setup_engine.generate_setup(score, feat_1h, struct_1h)

    explanation = ai_explainer.explain_opportunity(
        symbol=sym,
        score=score,
        setup=setup,
        regime=regime,
        features_1h=feat_1h,
        structure_1h=struct_1h
    )

    return explanation
