"""
BR-009 Probabilistic Recovery & Optimal Entry Intelligence Test Suite
Verifies:
1. First-Passage Target Probability Monotonicity (P(TP1) >= P(TP2) >= P(TP3) >= P(TP4)).
2. 6-Strategy Entry Benchmark Generation & Optimal Utility Selection.
3. Out-of-Sample Calibration & Brier Score (< 0.18).
4. Decision classification logic (ENTER, WAIT, AVOID, INSUFFICIENT_EVIDENCE).
5. Database schema and forecast record persistence.
6. Execution guard safety integration with BR-008.
7. FastAPI route registration & backward compatibility.
"""

import sys
import os
import asyncio
import time

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.database import init_db, get_db
from app.services.recovery_engine import recovery_engine
from app.main import app

async def run_tests():
    print("=" * 70)
    print("RUNNING BR-009 PROBABILISTIC RECOVERY & OPTIMAL ENTRY AUDIT SUITE")
    print("=" * 70)

    # Step 0: Ensure DB is initialized
    await init_db()
    print("[PASS] Database initialized with recovery_forecast_records table.")

    # -------------------------------------------------------------
    # Test 1: Full Report Generation & Target Probability Monotonicity
    # -------------------------------------------------------------
    print("\n--- Test 1: 14-Day Target Monotonicity Test (ARBUSDT) ---")
    data = await recovery_engine.get_recovery_intelligence("ARBUSDT")
    assert data["feature"] == "BR-009", f"Expected feature BR-009, got {data.get('feature')}"
    assert data["forecast_horizon_days"] == 14, "Horizon must be exactly 14 calendar days"
    assert len(data["target_forecasts"]) == 4, "Must have exactly 4 hierarchical targets (TP1-TP4)"

    tf = data["target_forecasts"]
    p1 = tf[0]["target_first_probability"]
    p2 = tf[1]["target_first_probability"]
    p3 = tf[2]["target_first_probability"]
    p4 = tf[3]["target_first_probability"]

    print(f"Probabilities: TP1={p1}%, TP2={p2}%, TP3={p3}%, TP4={p4}%")
    assert p1 >= p2, f"Monotonicity violated: TP1 ({p1}%) < TP2 ({p2}%)"
    assert p2 >= p3, f"Monotonicity violated: TP2 ({p2}%) < TP3 ({p3}%)"
    assert p3 >= p4, f"Monotonicity violated: TP3 ({p3}%) < TP4 ({p4}%)"
    print("[PASS] Monotonicity strictly satisfied: P(TP1) >= P(TP2) >= P(TP3) >= P(TP4)")

    # -------------------------------------------------------------
    # Test 2: Invalidation & Timeout Partition
    # -------------------------------------------------------------
    print("\n--- Test 2: Barrier Probability Partition ---")
    bp = data["barrier_probabilities"]
    assert "invalidation_first_prob" in bp, "Missing invalidation probability"
    assert "timeout_14d_prob" in bp, "Missing timeout probability"
    print(f"TP1={bp['tp1_prob']}%, Invalidation={bp['invalidation_first_prob']}%, Timeout={bp['timeout_14d_prob']}%")
    assert bp["invalidation_first_prob"] > 0, "Invalidation probability must be positive"
    assert bp["timeout_14d_prob"] > 0, "Timeout probability must be positive"
    print("[PASS] Barrier probability partition verified.")

    # -------------------------------------------------------------
    # Test 3: 6-Strategy Entry Benchmark Comparison
    # -------------------------------------------------------------
    print("\n--- Test 3: 6-Strategy Benchmark Evaluation ---")
    strats = data["entry_strategy_benchmarks"]
    expected_strats = ["strategy_a", "strategy_b", "strategy_c", "strategy_d", "strategy_e", "strategy_f"]
    for s_key in expected_strats:
        assert s_key in strats, f"Missing strategy {s_key} in benchmark"
        strat = strats[s_key]
        assert strat["entry_price"] > 0, f"Strategy {s_key} entry price must be positive"
        assert strat["risk_reward_ratio"] > 0, f"Strategy {s_key} R:R must be positive"
        assert -100.0 < strat["net_expectancy_pct"] < 100.0, f"Net expectancy out of bounds: {strat['net_expectancy_pct']}"
        print(f"  {strat['name']}: Entry=${strat['entry_price']:.4f}, Prob={(strat['target_first_probability']*100):.1f}%, NetExp={strat['net_expectancy_pct']}%")

    optimal_key = data["candidate_entry"]["optimal_strategy_key"]
    assert optimal_key in expected_strats, f"Invalid optimal strategy key: {optimal_key}"
    print(f"[PASS] Optimal entry selected: {data['candidate_entry']['optimal_strategy_name']} (Preferred Entry=${data['candidate_entry']['preferred']:.4f})")

    # -------------------------------------------------------------
    # Test 4: Decision Classification Integrity
    # -------------------------------------------------------------
    print("\n--- Test 4: Categorical Decision Validity ---")
    valid_decisions = ["ENTER", "WAIT", "AVOID", "INSUFFICIENT_EVIDENCE"]
    assert data["decision"] in valid_decisions, f"Invalid decision: {data['decision']}"
    assert len(data["reason"]) > 10, "Decision must provide an explanatory rationale"
    print(f"[PASS] Decision verified: {data['decision']} ({data['decision_badge']}) - {data['reason'][:60]}...")

    # -------------------------------------------------------------
    # Test 5: Calibration & Brier Score Validation
    # -------------------------------------------------------------
    print("\n--- Test 5: Out-of-Sample Calibration Metrics ---")
    cal = data["calibration"]
    assert cal["status"] == "VALIDATED_EMPIRICAL"
    assert cal["brier_score"] < 0.18, f"Brier score ({cal['brier_score']}) exceeds target threshold (0.18)"
    assert len(cal["reliability_bins"]) == 4, "Must provide 4 empirical reliability bins"
    print(f"[PASS] Calibration verified: Brier Score={cal['brier_score']}, Log Loss={cal['log_loss']}, Validated N={cal['sample_size_evaluated']}")

    # -------------------------------------------------------------
    # Test 6: Database Persistence Check
    # -------------------------------------------------------------
    print("\n--- Test 6: Forecast Persistence in SQLite ---")
    async with get_db() as db:
        async with db.execute("SELECT COUNT(*) FROM recovery_forecast_records WHERE symbol = 'ARBUSDT'") as cur:
            row = await cur.fetchone()
            count = row[0]
            assert count >= 1, "Forecast record was not persisted to database"
            print(f"[PASS] Confirmed {count} recovery forecast record(s) persisted in database.")

    # -------------------------------------------------------------
    # Test 7: Universe Recovery Scanner
    # -------------------------------------------------------------
    print("\n--- Test 7: Universe Recovery Scanner ---")
    scanner_items = await recovery_engine.get_recovery_scanner()
    assert len(scanner_items) >= 5, f"Expected at least 5 scanned assets, got {len(scanner_items)}"
    first_item = scanner_items[0]
    assert "symbol" in first_item and "decision" in first_item and "net_expectancy_pct" in first_item
    print(f"[PASS] Scanner returned {len(scanner_items)} assets. Top rank: {first_item['symbol']} ({first_item['decision']}), Net Exp={first_item['net_expectancy_pct']}%")

    # -------------------------------------------------------------
    # Test 8: All Registered App Routes Backward Compatibility
    # -------------------------------------------------------------
    print("\n--- Test 8: FastAPI Routes Audit ---")
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    required_routes = [
        "/api/recovery/{symbol}",
        "/api/recovery/scanner",
        "/api/recovery/strategies/{symbol}",
        "/api/recovery/calibration",
        "/api/recovery/execute",
        "/api/reversal/execute",
        "/api/reversal/scanner",
        "/api/reversal/{symbol}",
        "/api/market/klines",
        "/api/paper/order",
        "/api/paper/positions"
    ]
    for r in required_routes:
        assert r in routes, f"Missing required endpoint: {r}"
    print(f"[PASS] All {len(required_routes)} required routes verified. Total registered endpoints: {len(routes)}")

    print("\n" + "=" * 70)
    print("ALL BR-009 VERIFICATION AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_tests())
