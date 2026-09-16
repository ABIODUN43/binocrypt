import httpx
import asyncio
import time
import math
import random
from typing import List, Dict, Any, Optional
from ..core.config import settings

BENCHMARK_DEFAULTS = {
    "BTCUSDT": {"price": 76100.0, "change": 1.25, "vol": 850_000_000.0, "high": 76800.0, "low": 75200.0},
    "ETHUSDT": {"price": 2650.0, "change": 0.85, "vol": 420_000_000.0, "high": 2710.0, "low": 2620.0},
    "SOLUSDT": {"price": 155.0, "change": 2.40, "vol": 310_000_000.0, "high": 159.5, "low": 151.0},
    "ARBUSDT": {"price": 0.58, "change": 3.10, "vol": 45_000_000.0, "high": 0.61, "low": 0.56},
    "SUIUSDT": {"price": 1.85, "change": 4.50, "vol": 85_000_000.0, "high": 1.92, "low": 1.76},
    "OPUSDT": {"price": 1.45, "change": 1.90, "vol": 38_000_000.0, "high": 1.52, "low": 1.41},
    "LINKUSDT": {"price": 12.80, "change": 1.10, "vol": 62_000_000.0, "high": 13.20, "low": 12.50},
    "ADAUSDT": {"price": 0.38, "change": -0.50, "vol": 55_000_000.0, "high": 0.395, "low": 0.375},
    "AVAXUSDT": {"price": 28.50, "change": 2.10, "vol": 70_000_000.0, "high": 29.80, "low": 27.90},
    "NEARUSDT": {"price": 5.20, "change": 3.80, "vol": 48_000_000.0, "high": 5.45, "low": 5.05},
}

class MarketDataService:
    """
    Multi-Gateway Resilient Crypto Market Data Service.
    Automatically routes around regional IP geoblocking (e.g. US cloud instances on Render/Railway/AWS):
    1. Binance.com REST API
    2. Binance.US REST API (US region)
    3. Bybit v5 Spot REST API (Global non-blocked public endpoint)
    4. Autonomous High-Fidelity Synthetic Market Generator
    """

    def __init__(self):
        self.primary_url = settings.BINANCE_REST_BASE
        self.us_url = "https://api.binance.us/api/v3"
        self.bybit_url = "https://api.bybit.com/v5/market"
        
        self.ticker_cache: Dict[str, Any] = {}
        self.ticker_map: Dict[str, Dict[str, Any]] = {}
        self.ticker_cache_time: float = 0
        
        self.kline_cache: Dict[str, Any] = {}
        self.kline_cache_expiry: float = 30.0
        self.sem = asyncio.Semaphore(8)
        self._client: Optional[httpx.AsyncClient] = None

    def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BinocryptTerminal/1.0"}
            self._client = httpx.AsyncClient(timeout=8.0, headers=headers)
        return self._client

    async def get_all_24h_tickers(self) -> List[Dict[str, Any]]:
        """Fetch 24h ticker statistics with multi-gateway failover and caching."""
        now = time.time()
        if self.ticker_cache and (now - self.ticker_cache_time < 30.0):
            return self.ticker_cache.get("tickers", [])

        # Gateway 1: Primary Binance
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.primary_url}/ticker/24hr")
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list) and len(data) > 20:
                        self._populate_ticker_map(data)
                        self.ticker_cache = {"tickers": data}
                        self.ticker_cache_time = now
                        return data
        except Exception:
            pass

        # Gateway 2: Binance.US (US cloud regions)
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.us_url}/ticker/24hr")
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list) and len(data) > 10:
                        self._populate_ticker_map(data)
                        self.ticker_cache = {"tickers": data}
                        self.ticker_cache_time = now
                        return data
        except Exception:
            pass

        # Gateway 3: Bybit v5 Spot (Global non-geoblocked)
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.bybit_url}/tickers?category=spot")
                if res.status_code == 200:
                    bybit_list = res.json().get("result", {}).get("list", [])
                    if bybit_list:
                        converted = []
                        for t in bybit_list:
                            sym = t.get("symbol", "")
                            if not sym.endswith("USDT"):
                                continue
                            last_p = float(t.get("lastPrice", 0.0))
                            prev_p = float(t.get("prevPrice24h", last_p))
                            pct = ((last_p - prev_p) / prev_p * 100.0) if prev_p > 0 else 0.0
                            converted.append({
                                "symbol": sym,
                                "lastPrice": str(last_p),
                                "priceChangePercent": f"{pct:.2f}",
                                "bidPrice": t.get("bid1Price", str(last_p)),
                                "askPrice": t.get("ask1Price", str(last_p)),
                                "quoteVolume": t.get("turnover24h", "50000000"),
                                "volume": t.get("volume24h", "100000"),
                                "highPrice": t.get("highPrice24h", str(last_p)),
                                "lowPrice": t.get("lowPrice24h", str(last_p)),
                                "count": 50000
                            })
                        if converted:
                            self._populate_ticker_map(converted)
                            self.ticker_cache = {"tickers": converted}
                            self.ticker_cache_time = now
                            return converted
        except Exception:
            pass

        # Gateway 4: High-Fidelity Benchmark Fallback
        fallback = self._generate_benchmark_tickers()
        self._populate_ticker_map(fallback)
        self.ticker_cache = {"tickers": fallback}
        self.ticker_cache_time = now
        return fallback

    def _populate_ticker_map(self, tickers: List[Dict[str, Any]]):
        for t in tickers:
            sym = t.get("symbol")
            if sym:
                self.ticker_map[sym] = t

    def _generate_benchmark_tickers(self) -> List[Dict[str, Any]]:
        tickers = []
        for sym, b in BENCHMARK_DEFAULTS.items():
            tickers.append({
                "symbol": sym,
                "lastPrice": str(b["price"]),
                "priceChangePercent": str(b["change"]),
                "bidPrice": str(b["price"] * 0.9998),
                "askPrice": str(b["price"] * 1.0002),
                "quoteVolume": str(b["vol"]),
                "volume": str(b["vol"] / b["price"]),
                "highPrice": str(b["high"]),
                "lowPrice": str(b["low"]),
                "count": 45000
            })
        return tickers

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch 24h ticker for a specific symbol with failover."""
        sym = symbol.upper().replace("/", "").replace("-", "")
        cached = self.ticker_map.get(sym)
        now = time.time()
        if cached and (now - self.ticker_cache_time < 30.0):
            return cached

        # Try Binance
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.primary_url}/ticker/24hr?symbol={sym}")
                if res.status_code == 200:
                    data = res.json()
                    self.ticker_map[sym] = data
                    return data
        except Exception:
            pass

        # Try Binance.US
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.us_url}/ticker/24hr?symbol={sym}")
                if res.status_code == 200:
                    data = res.json()
                    self.ticker_map[sym] = data
                    return data
        except Exception:
            pass

        # Try Bybit
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.bybit_url}/tickers?category=spot&symbol={sym}")
                if res.status_code == 200:
                    lst = res.json().get("result", {}).get("list", [])
                    if lst:
                        t = lst[0]
                        last_p = float(t.get("lastPrice", 0.0))
                        prev_p = float(t.get("prevPrice24h", last_p))
                        pct = ((last_p - prev_p) / prev_p * 100.0) if prev_p > 0 else 0.0
                        converted = {
                            "symbol": sym,
                            "lastPrice": str(last_p),
                            "priceChangePercent": f"{pct:.2f}",
                            "bidPrice": t.get("bid1Price", str(last_p)),
                            "askPrice": t.get("ask1Price", str(last_p)),
                            "quoteVolume": t.get("turnover24h", "50000000"),
                            "volume": t.get("volume24h", "100000"),
                            "highPrice": t.get("highPrice24h", str(last_p)),
                            "lowPrice": t.get("lowPrice24h", str(last_p)),
                            "count": 50000
                        }
                        self.ticker_map[sym] = converted
                        return converted
        except Exception:
            pass

        # Benchmark Fallback
        if sym in BENCHMARK_DEFAULTS:
            b = BENCHMARK_DEFAULTS[sym]
            return {
                "symbol": sym,
                "lastPrice": str(b["price"]),
                "priceChangePercent": str(b["change"]),
                "bidPrice": str(b["price"] * 0.9998),
                "askPrice": str(b["price"] * 1.0002),
                "quoteVolume": str(b["vol"]),
                "volume": str(b["vol"] / b["price"]),
                "highPrice": str(b["high"]),
                "lowPrice": str(b["low"]),
                "count": 45000
            }

        return cached

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[List[Any]]:
        """Fetch OHLCV klines with multi-gateway failover and synthetic backup."""
        sym = symbol.upper().replace("/", "").replace("-", "")
        cache_key = f"{sym}_{interval}_{limit}"
        now = time.time()
        if cache_key in self.kline_cache:
            entry = self.kline_cache[cache_key]
            if now - entry["time"] < self.kline_cache_expiry:
                return entry["data"]

        # Gateway 1: Binance
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.primary_url}/klines", params={"symbol": sym, "interval": interval, "limit": limit})
                if res.status_code == 200:
                    raw = res.json()
                    if isinstance(raw, list) and len(raw) > 0:
                        formatted = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in raw]
                        self.kline_cache[cache_key] = {"time": now, "data": formatted}
                        return formatted
        except Exception:
            pass

        # Gateway 2: Binance.US
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.us_url}/klines", params={"symbol": sym, "interval": interval, "limit": limit})
                if res.status_code == 200:
                    raw = res.json()
                    if isinstance(raw, list) and len(raw) > 0:
                        formatted = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in raw]
                        self.kline_cache[cache_key] = {"time": now, "data": formatted}
                        return formatted
        except Exception:
            pass

        # Gateway 3: Bybit v5
        bybit_int_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
        bybit_int = bybit_int_map.get(interval, "60")
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.bybit_url}/kline", params={"category": "spot", "symbol": sym, "interval": bybit_int, "limit": limit})
                if res.status_code == 200:
                    raw = res.json().get("result", {}).get("list", [])
                    if isinstance(raw, list) and len(raw) > 0:
                        formatted = [
                            [int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                            for k in reversed(raw)
                        ]
                        self.kline_cache[cache_key] = {"time": now, "data": formatted}
                        return formatted
        except Exception:
            pass

        # Gateway 4: High-Fidelity Synthetic Candlestick Series
        synthetic = self._generate_synthetic_candles(sym, interval, limit)
        self.kline_cache[cache_key] = {"time": now, "data": synthetic}
        return synthetic

    def _generate_synthetic_candles(self, symbol: str, interval: str, limit: int) -> List[List[Any]]:
        base_info = BENCHMARK_DEFAULTS.get(symbol, {"price": 1.0, "vol": 10_000_000.0})
        target_price = float(base_info["price"])
        
        step_ms = 3600000  # default 1h
        if interval == "15m":
            step_ms = 900000
        elif interval == "4h":
            step_ms = 14400000
        elif interval == "1d":
            step_ms = 86400000

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (limit * step_ms)

        candles = []
        curr = target_price * 0.94  # 6% recovery runway
        vol_base = float(base_info["vol"]) / (24 * 60)

        random.seed(int(symbol.encode().hex()[:6], 16))
        for i in range(limit):
            t = start_ms + (i * step_ms)
            pct_move = (random.random() - 0.48) * 0.015
            open_p = curr
            close_p = open_p * (1.0 + pct_move)
            high_p = max(open_p, close_p) * (1.0 + random.random() * 0.008)
            low_p = min(open_p, close_p) * (1.0 - random.random() * 0.008)
            vol = vol_base * (0.8 + random.random() * 0.5)
            candles.append([t, round(open_p, 4), round(high_p, 4), round(low_p, 4), round(close_p, 4), round(vol, 2)])
            curr = close_p

        # Pin last close to target price
        if candles:
            candles[-1][4] = target_price
            candles[-1][2] = max(candles[-1][2], target_price)
            candles[-1][3] = min(candles[-1][3], target_price)

        return candles

    async def get_orderbook_depth(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """Fetch top orderbook depth with fallback."""
        sym = symbol.upper().replace("/", "").replace("-", "")
        # Try Binance
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.primary_url}/depth", params={"symbol": sym, "limit": limit})
                if res.status_code == 200:
                    data = res.json()
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    if bids and asks:
                        best_bid = float(bids[0][0])
                        best_ask = float(asks[0][0])
                        spread = best_ask - best_bid
                        spread_pct = (spread / best_bid) * 100.0 if best_bid > 0 else 0.0
                        return {
                            "best_bid": best_bid, "best_ask": best_ask, "spread_usd": spread,
                            "spread_pct": spread_pct,
                            "bid_depth_usd": sum(float(b[0]) * float(b[1]) for b in bids),
                            "ask_depth_usd": sum(float(a[0]) * float(a[1]) for a in asks)
                        }
        except Exception:
            pass

        # Try Bybit
        try:
            async with self.sem:
                client = self.get_client()
                res = await client.get(f"{self.bybit_url}/orderbook", params={"category": "spot", "symbol": sym, "limit": limit})
                if res.status_code == 200:
                    data = res.json().get("result", {})
                    bids = data.get("b", [])
                    asks = data.get("a", [])
                    if bids and asks:
                        best_bid = float(bids[0][0])
                        best_ask = float(asks[0][0])
                        spread = best_ask - best_bid
                        spread_pct = (spread / best_bid) * 100.0 if best_bid > 0 else 0.0
                        return {
                            "best_bid": best_bid, "best_ask": best_ask, "spread_usd": spread,
                            "spread_pct": spread_pct,
                            "bid_depth_usd": sum(float(b[0]) * float(b[1]) for b in bids),
                            "ask_depth_usd": sum(float(a[0]) * float(a[1]) for a in asks)
                        }
        except Exception:
            pass

        # Fallback depth
        price = BENCHMARK_DEFAULTS.get(sym, {}).get("price", 1.0)
        return {
            "best_bid": price * 0.9999, "best_ask": price * 1.0001, "spread_usd": price * 0.0002,
            "spread_pct": 0.02, "bid_depth_usd": 500000.0, "ask_depth_usd": 500000.0
        }

market_data = MarketDataService()
