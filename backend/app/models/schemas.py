from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Candle(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class IndicatorFeatures(BaseModel):
    symbol: str
    timeframe: str
    price: float
    ema_20: float
    ema_50: float
    ema_200: float
    ema_slope_20: float
    adx_14: float
    plus_di: float
    minus_di: float
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    atr_14: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    rvol_20: float
    volume_accel: float
    obv: float
    structure: str  # e.g., 'BULLISH_HH_HL', 'SIDEWAYS_RANGE', 'BEARISH_LH_LL', 'BREAKOUT'
    support_level: float
    resistance_level: float

class MarketRegime(BaseModel):
    regime: str  # BULL_TREND, BEAR_TREND, SIDEWAYS, HIGH_VOLATILITY, BREAKOUT_ENVIRONMENT, PANIC
    confidence: int  # 0 - 100
    btc_price: float
    btc_change_24h: float
    btc_trend_4h: str
    btc_trend_1h: str
    eth_trend_4h: str
    eth_trend_1h: str
    market_breadth_pct: float  # % coins above 50 EMA
    volatility_state: str      # LOW, NORMAL, HIGH, EXTREME
    trade_permission: str      # AGGRESSIVE_LONGS, SELECTIVE_LONGS, DEFENSIVE_CASH, NO_TRADE
    summary: str

class OpportunityScore(BaseModel):
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    spread_pct: float
    trend_score: float        # max 20
    momentum_score: float     # max 20
    volume_score: float       # max 20
    structure_score: float    # max 20
    liquidity_score: float    # max 10
    volatility_score: float   # max 10
    regime_score: float       # max 10
    rr_score: float           # max 10
    total_score: float        # max 100 (normalized)
    grade: str                # EXCELLENT, STRONG, GOOD, WATCH, IGNORE
    win_probability: float    # e.g., 0.72 (72%)
    rank: int = 0

class TradeSetup(BaseModel):
    symbol: str
    direction: str = "LONG"
    signal_state: str  # NO_TRADE, WATCH, SETUP_FORMING, READY, ENTERED, TP1_HIT, TP2_HIT, STOPPED, INVALIDATED
    score: float
    grade: str
    entry_zone_min: float
    entry_zone_max: float
    current_price: float
    stop_loss: float
    tp1: float
    tp2: float
    risk_reward: float
    stop_distance_pct: float
    invalidation: str
    setup_name: str
    confidence_pct: int
    rationale_bullets: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RiskSizingRequest(BaseModel):
    account_equity: float = 40.0
    risk_pct: float = 2.0
    entry_price: float
    stop_loss: float

class RiskSizingResponse(BaseModel):
    account_equity: float
    risk_pct: float
    risk_amount_usd: float
    stop_distance_pct: float
    recommended_position_usd: float
    recommended_quantity: float
    is_valid: bool
    warning_or_notes: str

class PaperOrderRequest(BaseModel):
    symbol: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    custom_notional_usd: Optional[float] = None

class PaperPosition(BaseModel):
    id: int
    symbol: str
    side: str
    status: str
    entry_price: float
    current_price: float
    quantity: float
    notional_usd: float
    stop_loss: float
    tp1: float
    tp2: float
    tp1_hit: bool
    risk_amount: float
    exit_price: Optional[float]
    pnl_usd: float
    pnl_pct: float
    fees_paid: float
    exit_reason: Optional[str]
    opened_at: str
    closed_at: Optional[str]

class PaperAccountSummary(BaseModel):
    starting_equity: float
    current_equity: float
    peak_equity: float
    available_cash: float
    daily_start_equity: float
    total_pnl_usd: float
    total_return_pct: float
    daily_pnl_usd: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    open_positions_count: int
    is_circuit_breaker_active: bool
    experiment_milestone_pct: float # progress towards $150–$200

class BacktestRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    limit_bars: int = 500
    risk_pct: float = 2.0
    min_score: float = 70.0
    rr_target: float = 2.0

class BacktestTrade(BaseModel):
    entry_time: str
    exit_time: str
    symbol: str
    entry_price: float
    exit_price: float
    stop_loss: float
    tp1: float
    tp2: float
    exit_reason: str # TP1, TP2, STOP_LOSS, TIMEOUT
    pnl_usd: float
    pnl_pct: float
    fee_usd: float
    net_pnl_usd: float
    cumulative_equity: float

class BacktestResult(BaseModel):
    symbol: str
    timeframe: str
    total_bars: int
    start_time: str
    end_time: str
    starting_capital: float
    final_equity: float
    net_profit_usd: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    average_trade_pct: float
    trades: List[BacktestTrade]

class AIExplanationResponse(BaseModel):
    symbol: str
    summary_title: str
    regime_alignment: str
    why_ranked_high: List[str]
    technical_triggers: List[str]
    invalidation_and_risks: List[str]
    small_account_verdict: str

# ------------------------------------------------------------------------------
# MARKET INTELLIGENCE & REGIME FORECASTING SCHEMAS
# ------------------------------------------------------------------------------

class RegimeProbabilities(BaseModel):
    strong_bull: float = Field(..., description="Probability of STRONG_BULL regime (%)")
    bull: float = Field(..., description="Probability of BULL regime (%)")
    recovery: float = Field(..., description="Probability of RECOVERY regime (%)")
    neutral: float = Field(..., description="Probability of NEUTRAL regime (%)")
    distribution: float = Field(..., description="Probability of DISTRIBUTION regime (%)")
    bear: float = Field(..., description="Probability of BEAR regime (%)")
    strong_bear: float = Field(..., description="Probability of STRONG_BEAR regime (%)")
    primary_regime: str = Field(..., description="The highest-probability regime")

class BearExhaustionScore(BaseModel):
    score: float = Field(..., description="Composite Bear Exhaustion Score (0-100)")
    status: str = Field(..., description="Qualitative exhaustion classification")
    trend_deterioration: float = Field(..., description="Subscore 0-100: waning downward trend momentum")
    momentum_recovery: float = Field(..., description="Subscore 0-100: positive RSI/MACD divergence")
    volume_accumulation: float = Field(..., description="Subscore 0-100: selling dry-up & green accumulation surges")
    market_context: float = Field(..., description="Subscore 0-100: BTC/ETH recovery and universe breadth")
    volatility_transition: float = Field(..., description="Subscore 0-100: ATR compression & volatility stabilization")
    btc_confirmation: float = Field(..., description="Subscore 0-100: BTC trend structure support")
    reasons: List[str] = []

class BullMaturityScore(BaseModel):
    score: float = Field(..., description="Composite Bull Maturity / Distribution Score (0-100)")
    status: str = Field(..., description="Maturity / Overextension classification")
    overextension_score: float = Field(..., description="Subscore 0-100: distance above long-term EMAs")
    distribution_volume: float = Field(..., description="Subscore 0-100: selling into rallies")
    momentum_divergence: float = Field(..., description="Subscore 0-100: bearish divergence on momentum")
    risk_reward_deterioration: float = Field(..., description="Subscore 0-100: asymmetric downside risk")
    reasons: List[str] = []

class HorizonForecast(BaseModel):
    horizon: str = Field(..., description="Forecast horizon e.g. 1D, 3D, 7D, 14D, 30D")
    direction: str = Field(..., description="Expected direction: BULLISH, NEUTRAL, BEARISH")
    bull_prob: float = Field(..., description="Estimated probability of positive return (%)")
    bear_prob: float = Field(..., description="Estimated probability of negative return (%)")
    recovery_prob: float = Field(..., description="Estimated probability of recovery/transition (%)")
    expected_return_min: float = Field(..., description="Expected return range minimum (%)")
    expected_return_max: float = Field(..., description="Expected return range maximum (%)")
    bear_case_price: float = Field(..., description="Probabilistic bear scenario price")
    base_case_price: float = Field(..., description="Probabilistic base scenario price")
    bull_case_price: float = Field(..., description="Probabilistic bull scenario price")
    uncertainty_pct: float = Field(..., description="Model uncertainty percentage (%)")
    confidence_pct: float = Field(..., description="Forecast confidence level (%)")

class DecisionTriggerItem(BaseModel):
    label: str
    satisfied: bool
    detail: str

class MarketHierarchyContext(BaseModel):
    global_regime: str
    btc_regime: str
    eth_regime: str
    sector: str = "Layer 1 / Ecosystem"
    sector_strength: str = "MODERATE"
    asset_alignment: str

class AssetMarketIntelligence(BaseModel):
    symbol: str
    price: float
    change_24h: float
    regime_distribution: RegimeProbabilities
    bear_exhaustion: BearExhaustionScore
    bull_maturity: BullMaturityScore
    decision: str = Field(..., description="WAIT, WATCH, ACCUMULATE, ENTER, HOLD, REDUCE, TAKE_PROFIT, EXIT, AVOID")
    decision_badge: str
    decision_summary: str
    triggers_checklist: List[DecisionTriggerItem]
    transition_window: str = Field(..., description="Model forecast window e.g. 'Oct 11 – Oct 17'")
    accumulation_window: str = Field(..., description="Potential accumulation window e.g. 'Oct 13 – Oct 18'")
    multi_horizon_forecasts: List[HorizonForecast]
    hierarchy: MarketHierarchyContext
    evidence_bullets: List[str]
    risk_warnings: List[str]
    model_agreement_pct: float
    prediction_confidence_pct: float
    historical_directional_accuracy_pct: float
    historical_recovery_accuracy_pct: float
    model_version: str = "RegimeEnsemble v2.4"
    timestamp: str

class PredictionRecord(BaseModel):
    id: int
    symbol: str
    generated_at: str
    horizon_days: int
    target_date: str
    predicted_regime: str
    predicted_direction: str
    recovery_prob: float
    bear_exhaustion_score: float
    decision: str
    entry_price: float
    bear_case_price: float
    base_case_price: float
    bull_case_price: float
    actual_exit_price: Optional[float] = None
    actual_return_pct: Optional[float] = None
    outcome_evaluated: bool = False
    is_correct: Optional[bool] = None

class PredictionAccuracySummary(BaseModel):
    total_predictions: int
    evaluated_predictions: int
    accuracy_7d: float
    accuracy_14d: float
    accuracy_30d: float
    regime_accuracy: float
    win_rate: float
    recent_signals: List[PredictionRecord]

# ------------------------------------------------------------------------------
# BR-001 QUANTITATIVE RESEARCH BLUEPRINT SCHEMAS
# ------------------------------------------------------------------------------

class HMMStatePosteriors(BaseModel):
    bear: float = Field(..., description="State B: Persistent negative regime (%)")
    sideways: float = Field(..., description="State S: Neutral / consolidating regime (%)")
    recovery: float = Field(..., description="State R: Bear pressure weakening & positive regime forming (%)")
    bull: float = Field(..., description="State U: Persistent positive expansion (%)")
    dominant_state: str

class HMMTransitionMatrix(BaseModel):
    states: List[str] = ["Bear", "Sideways", "Recovery", "Bull"]
    matrix: List[List[float]] = Field(..., description="4x4 transition matrix P(S_{t+1}=j | S_t=i)")

class TransitionTimingPoint(BaseModel):
    day: int
    probability_mass: float
    cumulative_prob: float

class CalibrationCurvePoint(BaseModel):
    bin_center: float
    predicted_prob: float
    empirical_freq: float
    sample_count: int

class ResearchExperimentRecord(BaseModel):
    experiment_id: str
    name: str
    hypothesis: str
    features_used: List[str]
    model_type: str
    target_label: str
    brier_score: float
    roc_auc: float
    pr_auc: float
    calibration_error: float
    gross_return_pct: float
    net_return_pct: float
    net_sharpe: float
    max_drawdown_pct: float
    status: str
    decision: str
    rejection_reason: Optional[str] = None
    notes: str

class ResearchGraveyardEntry(BaseModel):
    id: int
    experiment_id: str
    hypothesis: str
    rejection_reason: str
    flaw_type: str
    archived_at: str
    lessons_learned: str

class HMMAnalysisResponse(BaseModel):
    symbol: str
    current_price: float
    state_posteriors: HMMStatePosteriors
    transition_matrix: HMMTransitionMatrix
    timing_distribution: List[TransitionTimingPoint]
    modal_window: str
    recovery_readiness_pct: float
    timestamp: str
