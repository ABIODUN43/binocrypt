import sys
import os
import asyncio
import time

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.database import init_db, get_db
from app.services.signal_guard import signal_guard
from app.services.reversal_engine import reversal_engine
from app.main import app

async def run_tests():
    print("=" * 70)
    print("RUNNING BR-008 SIGNAL LIFECYCLE & EXECUTION SAFETY AUDIT SUITE")
    print("=" * 70)

    # Step 0: Ensure DB is initialized
    await init_db()
    print("[PASS] Database initialized with signal_lifecycle_records & paper_positions migration.")

    # -------------------------------------------------------------
    # Test 1: Signal Lifecycle State Transitions
    # -------------------------------------------------------------
    print("\n--- Test 1: Signal Lifecycle State Evaluation ---")
    
    # Base setup template modeled after user's ARB example
    symbol = "ARBUSDT"
    candle_time = int(time.time()) - 300 # 5 min ago
    feed_meta_fresh = {
        "source": "BINANCE",
        "feed_status": "LIVE_HEALTHY",
        "age_seconds": 2.5,
        "latest_candle_time": candle_time
    }
    
    bottom_zone = {
        "potential_bottom_low": 0.1654,
        "potential_bottom_high": 0.1662,
        "invalidation_price": 0.1646,
        "dist_to_floor_pct": 0.3,
        "dist_to_invalidation_pct": 0.8
    }
    
    triple_barriers = {
        "config_a": {
            "target_price": 0.1668,
            "stop_price": 0.1646,
            "risk_reward_ratio": 1.75,
            "win_probability_pct": 62.0
        }
    }
    
    raw_setup = {
        "status": "READY_TO_EXECUTE",
        "action_recommendation": "Confirmed structural reversal.",
        "entry_zone_min": 0.1649,
        "entry_zone_max": 0.1659,
        "stop_loss_price": 0.1646,
        "tp1_price": 0.1668,
        "tp2_price": 0.1678,
        "risk_reward_ratio": 1.75,
        "can_execute": True
    }

    # Case 1A: Perfect Entry within zone [0.1649 - 0.1659]
    eval_1a = await signal_guard.evaluate_signal_lifecycle(
        symbol=symbol,
        analytical_state="REVERSAL_CONFIRMED",
        bottom_zone=bottom_zone,
        raw_execution_setup=raw_setup,
        triple_barriers=triple_barriers,
        feed_meta=feed_meta_fresh,
        current_price=0.1655
    )
    assert eval_1a["lifecycle_state"] == "READY_TO_EXECUTE", f"Expected READY_TO_EXECUTE, got {eval_1a['lifecycle_state']}"
    assert eval_1a["can_execute"] is True, "Expected can_execute to be True in entry zone"
    print(f"[PASS] 1A (In Entry Zone): State={eval_1a['lifecycle_state']}, CanExecute={eval_1a['can_execute']}")

    # Case 1B: Exact User ARB Anomaly: Price ran to $0.16874, above TP1 ($0.1668)
    eval_1b = await signal_guard.evaluate_signal_lifecycle(
        symbol=symbol,
        analytical_state="REVERSAL_CONFIRMED",
        bottom_zone=bottom_zone,
        raw_execution_setup=raw_setup,
        triple_barriers=triple_barriers,
        feed_meta=feed_meta_fresh,
        current_price=0.16874
    )
    assert eval_1b["lifecycle_state"] == "TARGET_REACHED_BEFORE_ENTRY", f"Expected TARGET_REACHED_BEFORE_ENTRY, got {eval_1b['lifecycle_state']}"
    assert eval_1b["can_execute"] is False, "Expected can_execute to be False after target reached"
    assert "reached or exceeded TP1 target" in eval_1b["eligibility_reason"], f"Unexpected reason: {eval_1b['eligibility_reason']}"
    print(f"[PASS] 1B (ARB Anomaly Reached TP1): State={eval_1b['lifecycle_state']}, CanExecute={eval_1b['can_execute']}, Reason={eval_1b['eligibility_reason']}")

    # Case 1C: Price Runaway Above Entry Zone (e.g. 0.1664 > max 0.1659, but below TP1)
    eval_1c = await signal_guard.evaluate_signal_lifecycle(
        symbol=symbol,
        analytical_state="REVERSAL_CONFIRMED",
        bottom_zone=bottom_zone,
        raw_execution_setup=raw_setup,
        triple_barriers=triple_barriers,
        feed_meta=feed_meta_fresh,
        current_price=0.1664
    )
    assert eval_1c["lifecycle_state"] == "MISSED_ENTRY", f"Expected MISSED_ENTRY, got {eval_1c['lifecycle_state']}"
    assert eval_1c["can_execute"] is False, "Expected can_execute to be False on runaway"
    print(f"[PASS] 1C (Runaway Above Zone): State={eval_1c['lifecycle_state']}, CanExecute={eval_1c['can_execute']}")

    # Case 1D: Structural Stop Breach Below Invalidation (e.g. 0.1640 < 0.1646)
    eval_1d = await signal_guard.evaluate_signal_lifecycle(
        symbol=symbol,
        analytical_state="REVERSAL_CONFIRMED",
        bottom_zone=bottom_zone,
        raw_execution_setup=raw_setup,
        triple_barriers=triple_barriers,
        feed_meta=feed_meta_fresh,
        current_price=0.1640
    )
    assert eval_1d["lifecycle_state"] == "INVALIDATED", f"Expected INVALIDATED, got {eval_1d['lifecycle_state']}"
    assert eval_1d["can_execute"] is False, "Expected can_execute to be False on invalidation"
    print(f"[PASS] 1D (Structural Invalidation): State={eval_1d['lifecycle_state']}, CanExecute={eval_1d['can_execute']}")

    # Case 1E: Stale Market Data Feed (> 180s)
    feed_meta_stale = {
        "source": "BINANCE",
        "feed_status": "STALE",
        "age_seconds": 240.0,
        "latest_candle_time": candle_time - 240
    }
    eval_1e = await signal_guard.evaluate_signal_lifecycle(
        symbol=symbol,
        analytical_state="REVERSAL_CONFIRMED",
        bottom_zone=bottom_zone,
        raw_execution_setup=raw_setup,
        triple_barriers=triple_barriers,
        feed_meta=feed_meta_stale,
        current_price=0.1655
    )
    assert eval_1e["lifecycle_state"] == "STALE_MARKET_DATA", f"Expected STALE_MARKET_DATA, got {eval_1e['lifecycle_state']}"
    assert eval_1e["can_execute"] is False, "Expected can_execute to be False on stale feed"
    print(f"[PASS] 1E (Stale Feed Protection): State={eval_1e['lifecycle_state']}, CanExecute={eval_1e['can_execute']}")

    # -------------------------------------------------------------
    # Test 2: Idempotency & Duplicate Execution Prevention
    # -------------------------------------------------------------
    print("\n--- Test 2: Atomic Execution & Duplicate Rejection ---")
    
    # Get current live ticker price so pre-execution revalidation passes
    from app.services.market_data import market_data
    t = await market_data.get_ticker("ARBUSDT")
    live_p = float(t.get("lastPrice", 0.58)) if t else 0.58
    
    test_bottom_zone = {
        "potential_bottom_low": round(live_p * 0.98, 4),
        "potential_bottom_high": round(live_p * 1.01, 4),
        "invalidation_price": round(live_p * 0.97, 4),
        "dist_to_floor_pct": 2.0,
        "dist_to_invalidation_pct": 3.0
    }
    test_raw_setup = {
        "status": "READY_TO_EXECUTE",
        "action_recommendation": "Confirmed structural reversal.",
        "entry_zone_min": round(live_p * 0.995, 4),
        "entry_zone_max": round(live_p * 1.005, 4),
        "stop_loss_price": round(live_p * 0.97, 4),
        "tp1_price": round(live_p * 1.05, 4),
        "tp2_price": round(live_p * 1.08, 4),
        "risk_reward_ratio": 1.75,
        "can_execute": True
    }
    test_feed_meta = {
        "source": "BINANCE",
        "feed_status": "LIVE_HEALTHY",
        "age_seconds": 1.0,
        "latest_candle_time": int(time.time() * 1000)
    }

    fresh_eval = await signal_guard.evaluate_signal_lifecycle(
        symbol="ARBUSDT",
        analytical_state="REVERSAL_CONFIRMED",
        bottom_zone=test_bottom_zone,
        raw_execution_setup=test_raw_setup,
        triple_barriers=triple_barriers,
        feed_meta=test_feed_meta,
        current_price=live_p
    )
    sig_id = fresh_eval["signal_id"]
    assert sig_id is not None, "Signal ID should have been generated"
    assert fresh_eval["can_execute"] is True, f"Signal should be actionable, got {fresh_eval}"

    # Execution Attempt 1: Should Succeed
    exec_1 = await signal_guard.execute_signal(
        symbol="ARBUSDT",
        signal_id=sig_id
    )
    assert exec_1["success"] is True, f"Execution 1 failed: {exec_1}"
    assert exec_1["status"] == "POSITION_OPEN", f"Execution 1 not POSITION_OPEN: {exec_1}"
    print(f"[PASS] Execution 1 succeeded: Fill Price=${exec_1['entry_price']}, Notional=${exec_1['notional_usd']}")

    # Execution Attempt 2: Same Signal ID -> MUST BE REJECTED (Idempotency)
    exec_2 = await signal_guard.execute_signal(
        symbol="ARBUSDT",
        signal_id=sig_id
    )
    assert exec_2["success"] is False, f"Execution 2 should have been rejected but got: {exec_2}"
    assert "already been executed" in exec_2["error"], f"Expected already executed error, got: {exec_2['error']}"
    print(f"[PASS] Execution 2 rejected cleanly: {exec_2['error']}")

    # -------------------------------------------------------------
    # Test 3: Reversal Engine Integration
    # -------------------------------------------------------------
    print("\n--- Test 3: Reversal Engine Live Intelligence Integration ---")
    data = await reversal_engine.get_reversal_intelligence("ARBUSDT")
    assert "lifecycle_guard" in data, "lifecycle_guard missing from get_reversal_intelligence"
    assert "data_provenance" in data, "data_provenance missing from get_reversal_intelligence"
    assert "signal_id" in data["execution_setup"], "signal_id missing from execution_setup"
    assert "live_rr" in data["execution_setup"], "live_rr missing from execution_setup"
    print(f"[PASS] Reversal Intelligence Engine payload verified: State={data['state']}, Lifecycle={data['lifecycle_guard']['lifecycle_state']}, FeedSource={data['data_provenance']['source']}")

    # -------------------------------------------------------------
    # Test 4: All FastAPI Routes Verification
    # -------------------------------------------------------------
    print("\n--- Test 4: Route Backward Compatibility Check ---")
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    required_routes = [
        "/api/reversal/execute",
        "/api/reversal/signals/{symbol}",
        "/api/reversal/scanner",
        "/api/reversal/{symbol}",
        "/api/market/klines",
        "/api/paper/order",
        "/api/paper/positions",
        "/api/paper/account",
        "/api/paper/history"
    ]
    for rr in required_routes:
        assert rr in routes, f"Missing required route: {rr}"
    print(f"[PASS] All required routes verified. Total registered routes in app: {len(routes)}")

    print("\n" + "=" * 70)
    print("ALL BR-008 VERIFICATION AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_tests())
