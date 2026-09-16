import httpx
import asyncio
import time
from typing import List, Dict, Any, Optional
from ..core.config import settings

class MarketDataService:
    def __init__(self):
        self.base_url = settings.BINANCE_REST_BASE
        self.ticker_cache: Dict[str, Any] = {}
        self.ticker_map: Dict[str, Dict[str, Any]] = {}  # In-memory fast symbol lookup
        self.ticker_cache_time: float = 0
        self.kline_cache: Dict[str, Any] = {}
        self.kline_cache_expiry: float = 30.0  # 30 seconds cache for klines
        self.sem = asyncio.Semaphore(6)        # Concurrency throttle to prevent Windows socket exhaustion
        self._client: Optional[httpx.AsyncClient] = None

    def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def get_all_24h_tickers(self) -> List[Dict[str, Any]]:
        """Fetch 24h ticker statistics for all pairs with 30s cache."""
        now = time.time()
        if self.ticker_cache and (now - self.ticker_cache_time < 30.0):
            return self.ticker_cache.get("tickers", [])

        for attempt in range(3):
            try:
                async with self.sem:
                    client = self.get_client()
                    url = f"{self.base_url}/ticker/24hr"
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Populate in-memory fast symbol map
                    for t in data:
                        sym = t.get("symbol")
                        if sym:
                            self.ticker_map[sym] = t

                    self.ticker_cache = {"tickers": data}
                    self.ticker_cache_time = now
                    return data
            except Exception as e:
                if attempt == 2:
                    print(f"[MarketData] Error fetching 24h tickers (attempt {attempt+1}): {e}")
                    if "tickers" in self.ticker_cache:
                        return self.ticker_cache["tickers"]
                    return list(self.ticker_map.values())
                await asyncio.sleep(0.3 * (attempt + 1))

        return list(self.ticker_map.values())

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch 24h ticker for a specific symbol with cache fallback."""
        sym = symbol.upper()
        
        # Check if already in in-memory ticker_map
        cached = self.ticker_map.get(sym)
        if cached:
            # If cached within 30s, return immediately
            if time.time() - self.ticker_cache_time < 30.0:
                return cached

        # Attempt fresh fetch
        for attempt in range(3):
            try:
                async with self.sem:
                    client = self.get_client()
                    url = f"{self.base_url}/ticker/24hr?symbol={sym}"
                    res = await client.get(url)
                    res.raise_for_status()
                    data = res.json()
                    self.ticker_map[sym] = data
                    return data
            except Exception as e:
                if attempt == 2:
                    print(f"[MarketData] Failed to fetch ticker for {sym}: {e}")
                    if cached:
                        return cached
                    # Fallback to general ticker cache search
                    if "tickers" in self.ticker_cache:
                        for t in self.ticker_cache["tickers"]:
                            if t.get("symbol") == sym:
                                self.ticker_map[sym] = t
                                return t
                await asyncio.sleep(0.2 * (attempt + 1))

        return cached

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[List[Any]]:
        """
        Fetch OHLCV klines for symbol with retry and caching.
        """
        sym = symbol.upper()
        cache_key = f"{sym}_{interval}_{limit}"
        now = time.time()
        if cache_key in self.kline_cache:
            entry = self.kline_cache[cache_key]
            if now - entry["time"] < self.kline_cache_expiry:
                return entry["data"]

        for attempt in range(3):
            try:
                async with self.sem:
                    client = self.get_client()
                    url = f"{self.base_url}/klines"
                    params = {
                        "symbol": sym,
                        "interval": interval,
                        "limit": limit
                    }
                    res = await client.get(url, params=params)
                    res.raise_for_status()
                    raw_klines = res.json()
                    
                    formatted = []
                    for k in raw_klines:
                        formatted.append([
                            int(k[0]),
                            float(k[1]),
                            float(k[2]),
                            float(k[3]),
                            float(k[4]),
                            float(k[5])
                        ])
                    
                    self.kline_cache[cache_key] = {"time": now, "data": formatted}
                    return formatted
            except Exception as e:
                if attempt == 2:
                    print(f"[MarketData] Error fetching klines for {sym} {interval}: {e}")
                    if cache_key in self.kline_cache:
                        return self.kline_cache[cache_key]["data"]
                await asyncio.sleep(0.2 * (attempt + 1))

        if cache_key in self.kline_cache:
            return self.kline_cache[cache_key]["data"]
        return []

    async def get_orderbook_depth(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """Fetch top order book depth to estimate spread and liquidity."""
        sym = symbol.upper()
        try:
            async with self.sem:
                client = self.get_client()
                url = f"{self.base_url}/depth"
                params = {"symbol": sym, "limit": limit}
                res = await client.get(url, params=params)
                res.raise_for_status()
                data = res.json()
                
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                
                if bids and asks:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    spread = best_ask - best_bid
                    spread_pct = (spread / best_bid) * 100.0 if best_bid > 0 else 0.0
                    bid_depth_usd = sum(float(b[0]) * float(b[1]) for b in bids)
                    ask_depth_usd = sum(float(a[0]) * float(a[1]) for a in asks)
                    return {
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread_usd": spread,
                        "spread_pct": spread_pct,
                        "bid_depth_usd": bid_depth_usd,
                        "ask_depth_usd": ask_depth_usd
                    }
        except Exception as e:
            pass
            
        return {
            "best_bid": 0.0, "best_ask": 0.0, "spread_usd": 0.0,
            "spread_pct": 0.05, "bid_depth_usd": 100000.0, "ask_depth_usd": 100000.0
        }

market_data = MarketDataService()
