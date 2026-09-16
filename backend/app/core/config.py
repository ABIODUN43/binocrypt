from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True)
    
    PROJECT_NAME: str = "Binocrypt Quantitative Terminal"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # Binance Public API
    BINANCE_REST_BASE: str = "https://api.binance.com/api/v3"
    
    # Capital & Risk Defaults (The $40 Experiment)
    DEFAULT_CAPITAL: float = 40.00
    MAX_RISK_PER_TRADE_PCT: float = 2.0  # 1.5% - 2.0% ($0.60 - $0.80)
    MAX_OPEN_POSITIONS: int = 2          # Conservative for $40 capital
    MAX_DAILY_DRAWDOWN_PCT: float = 5.0  # 5% circuit breaker
    MAX_TOTAL_DRAWDOWN_PCT: float = 10.0 # 10% portfolio halt
    MIN_ORDER_NOTIONAL: float = 5.0      # Binance spot minimum ~$5 USDT
    
    # Universe Filters
    MIN_24H_VOLUME_USDT: float = 5_000_000.0  # Min $5M 24h volume
    MAX_SPREAD_PCT: float = 0.20              # Max 0.20% spread
    
    # Backtest Defaults
    DEFAULT_FEE_PCT: float = 0.10             # 0.10% spot taker fee
    DEFAULT_SLIPPAGE_PCT: float = 0.05        # 0.05% slippage model
    
    # DB
    DB_PATH: str = "binocrypt.db"

settings = Settings()
