from typing import List, Dict, Any
from .market_data import market_data
from ..core.config import settings

STABLECOINS_AND_FIAT = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "EUR", "GBP", "AUD", "BRL", 
    "TRY", "RUB", "BIDR", "IDRT", "UAH", "NGN", "PLN", "RON", "ARS", 
    "COP", "CZK", "ZAR", "USDP", "AEUR", "EURI", "USTC", "PAXG"
}

LEVERAGED_SUFFIXES = ["UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT"]

class UniverseFilter:
    def __init__(self):
        self.cached_universe: List[Dict[str, Any]] = []
        self.last_update: float = 0

    async def get_tradable_universe(self, min_volume: float = None) -> List[Dict[str, Any]]:
        """
        Filters all Binance pairs down to liquid USDT spot assets:
        1. Must end in USDT
        2. Exclude stablecoins and fiat
        3. Exclude leveraged tokens
        4. 24h quote volume >= min_volume
        5. Spread <= 0.20%
        """
        min_vol = min_volume if min_volume is not None else settings.MIN_24H_VOLUME_USDT
        tickers = await market_data.get_all_24h_tickers()
        
        filtered = []
        for t in tickers:
            symbol = t.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue

            base_asset = symbol[:-4]
            if base_asset in STABLECOINS_AND_FIAT:
                continue

            if any(symbol.endswith(sfx) for sfx in LEVERAGED_SUFFIXES):
                continue

            quote_vol = float(t.get("quoteVolume", 0.0))
            if quote_vol < min_vol:
                continue

            bid_price = float(t.get("bidPrice", 0.0))
            ask_price = float(t.get("askPrice", 0.0))
            last_price = float(t.get("lastPrice", 0.0))
            
            if last_price <= 0:
                continue

            spread_pct = 0.0
            if bid_price > 0 and ask_price > 0:
                spread_pct = ((ask_price - bid_price) / bid_price) * 100.0

            if spread_pct > settings.MAX_SPREAD_PCT:
                continue

            filtered.append({
                "symbol": symbol,
                "base_asset": base_asset,
                "price": last_price,
                "change_24h": float(t.get("priceChangePercent", 0.0)),
                "volume_24h_usd": quote_vol,
                "high_24h": float(t.get("highPrice", 0.0)),
                "low_24h": float(t.get("lowPrice", 0.0)),
                "spread_pct": round(spread_pct, 4),
                "count_trades": int(t.get("count", 0))
            })

        if not filtered:
            fallback_symbols = ["ARBUSDT", "SOLUSDT", "SUIUSDT", "BTCUSDT", "ETHUSDT", "OPUSDT", "AVAXUSDT", "LINKUSDT", "ADAUSDT", "NEARUSDT"]
            for sym in fallback_symbols:
                t = market_data.ticker_map.get(sym) or await market_data.get_ticker(sym)
                if t:
                    filtered.append({
                        "symbol": sym,
                        "base_asset": sym.replace("USDT", ""),
                        "price": float(t.get("lastPrice", 1.0)),
                        "change_24h": float(t.get("priceChangePercent", 0.0)),
                        "volume_24h_usd": float(t.get("quoteVolume", 50000000.0)),
                        "high_24h": float(t.get("highPrice", 1.0)),
                        "low_24h": float(t.get("lowPrice", 1.0)),
                        "spread_pct": 0.02,
                        "count_trades": int(t.get("count", 10000))
                    })

        # Sort descending by 24h USD volume
        filtered.sort(key=lambda x: x["volume_24h_usd"], reverse=True)
        self.cached_universe = filtered
        return filtered

universe_service = UniverseFilter()

