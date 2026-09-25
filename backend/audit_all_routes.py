import asyncio
import httpx
from app.main import app
from app.core.database import init_db

async def audit():
    print("[Audit] Initializing database...")
    await init_db()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        tests = [
            ("GET", "/api/health", None),
            ("GET", "/api/market/regime", None),
            ("GET", "/api/market/ticker/BTCUSDT", None),
            ("GET", "/api/market/ticker/ARB/USDT", None),
            ("GET", "/api/market/klines?symbol=BTCUSDT&interval=1h&limit=50", None),
            ("GET", "/api/market/klines?symbol=ARB/USDT&interval=1h&limit=50", None),
            ("GET", "/api/market/depth/BTCUSDT", None),
            ("GET", "/api/scanner/universe", None),
            ("GET", "/api/scanner/opportunities?limit=10", None),
            ("GET", "/api/setups/active?limit=5", None),
            ("GET", "/api/setups/BTCUSDT", None),
            ("GET", "/api/setups/ARB/USDT", None),
            ("POST", "/api/risk/calculate", {"account_equity": 40.0, "risk_pct": 2.0, "entry_price": 100.0, "stop_loss": 97.5}),
            ("GET", "/api/risk/rules", None),
            ("GET", "/api/paper/account", None),
            ("GET", "/api/paper/positions", None),
            ("GET", "/api/paper/history", None),
            ("POST", "/api/paper/reset?capital=40.0", None),
            ("POST", "/api/backtest/run", {"symbol": "BTCUSDT", "timeframe": "1h", "risk_pct": 2.0, "rr_target": 2.0, "limit_bars": 120}),
            ("GET", "/api/explain/BTCUSDT", None),
            ("GET", "/api/explain/ARB/USDT", None),
            ("GET", "/api/intelligence/scanner/recovery?limit=5", None),
            ("GET", "/api/intelligence/history/ARBUSDT", None),
            ("GET", "/api/intelligence/history/ARB/USDT", None),
            ("GET", "/api/intelligence/ARBUSDT", None),
            ("GET", "/api/intelligence/ARB/USDT", None),
            ("GET", "/api/research/experiments", None),
            ("GET", "/api/research/experiments/BR-001-A", None),
            ("GET", "/api/research/graveyard", None),
            ("GET", "/api/research/hmm/ARBUSDT", None),
            ("GET", "/api/research/hmm/ARB/USDT", None),
            ("GET", "/api/research/large-move/ARBUSDT", None),
            ("GET", "/api/research/large-move/ARB/USDT", None),
            ("GET", "/api/research/entry-strategies/ARBUSDT", None),
            ("GET", "/api/research/entry-strategies/ARB/USDT", None),
            ("GET", "/api/research/false-recovery/ARBUSDT", None),
            ("GET", "/api/research/false-recovery/ARB/USDT", None),
            ("GET", "/api/research/ranking?limit=5", None),
        ]

        passed = 0
        failed = 0
        failures = []

        for method, path, body in tests:
            try:
                if method == "GET":
                    res = await client.get(path, timeout=30.0)
                else:
                    res = await client.post(path, json=body, timeout=30.0)
                
                if 200 <= res.status_code < 300:
                    print(f"  [PASS] {method} {path} -> {res.status_code}")
                    passed += 1
                else:
                    print(f"  [FAIL] {method} {path} -> {res.status_code} ({res.text[:120]})")
                    failed += 1
                    failures.append((method, path, res.status_code, res.text[:200]))
            except Exception as e:
                print(f"  [ERROR] {method} {path} -> {e}")
                failed += 1
                failures.append((method, path, "EXCEPTION", str(e)))

        print(f"\n==========================================")
        print(f"Audit Summary: {passed} passed, {failed} failed out of {len(tests)} endpoints tested.")
        if failures:
            print(f"Failures:")
            for m, p, code, txt in failures:
                print(f"  - {m} {p} -> {code}: {txt}")
        print(f"==========================================")

if __name__ == "__main__":
    asyncio.run(audit())
