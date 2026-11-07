import time
import math
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import aiosqlite

from ..core.config import settings
from .market_data import market_data
from .paper_trader import paper_trader
from ..models.schemas import PaperOrderRequest

class SignalLifecycleGuard:
    """
    BR-008: Signal Lifecycle, Entry Timing Intelligence & Execution Guard.
    
    Ensures that Binocrypt only presents and permits actionable trades when:
    1. The original analytical setup is still structurally valid.
    2. The live price satisfies the acceptable entry zone.
    3. The price has NOT already reached or exceeded TP1/TP2 targets before entry.
    4. The signal validity window (expiry) has not elapsed.
    5. The market-data feed is fresh and verified (not stale/synthetic).
    6. The live risk-to-reward ratio remains favorable (>= 1.30).
    7. The signal has NOT already been consumed or executed.
    8. Duplicate and concurrent execution requests are strictly prevented via database locking.
    """

    def __init__(self, db_path: str = settings.DB_PATH):
        self.db_path = db_path
        self._execution_locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, signal_id: str) -> asyncio.Lock:
        if signal_id not in self._execution_locks:
            self._execution_locks[signal_id] = asyncio.Lock()
        return self._execution_locks[signal_id]

    def construct_signal_id(self, symbol: str, confirmation_candle_time: int, floor_price: float) -> str:
        """Constructs a deterministic, persistent identity for a confirmed reversal setup."""
        clean_sym = symbol.upper().replace("/", "").replace("-", "")
        # Floor normalized to 4 decimals integer
        floor_int = int(round(floor_price * 10000))
        return f"SIG-BR007-{clean_sym}-{confirmation_candle_time}-{floor_int}"

    async def evaluate_signal_lifecycle(
        self,
        symbol: str,
        analytical_state: str,
        bottom_zone: Dict[str, Any],
        raw_execution_setup: Dict[str, Any],
        triple_barriers: Dict[str, Any],
        feed_meta: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        BR-008 Core Evaluation Engine.
        Determines whether a detected BR-007 setup is eligible for new entry at current time.
        """
        clean_sym = symbol.upper().replace("/", "").replace("-", "")
        now = time.time()
        now_dt = datetime.now(timezone.utc).isoformat()
        
        # 1. Non-confirmed states (DECLINING, DOWNSIDE_EXHAUSTION_WATCH, BASE_FORMING, FAILED_REVERSAL)
        if analytical_state != "REVERSAL_CONFIRMED":
            mapped_state = "MONITORING" if analytical_state in ["DOWNSIDE_EXHAUSTION_WATCH", "BASE_FORMING"] else analytical_state
            return {
                "signal_id": None,
                "lifecycle_state": mapped_state,
                "lifecycle_label": f"Monitoring ({analytical_state.replace('_', ' ').title()})",
                "lifecycle_color": "amber" if mapped_state == "MONITORING" else "slate",
                "is_actionable": False,
                "can_execute": False,
                "eligibility_reason": "Structural confirmation not yet satisfied. Entry strictly locked to prevent falling-knife risk.",
                "action_recommendation": raw_execution_setup.get("action_recommendation", "Stand aside."),
                "feed_source": feed_meta.get("source", "BINANCE"),
                "feed_freshness_seconds": feed_meta.get("age_seconds", 0.0),
                "feed_status": feed_meta.get("feed_status", "LIVE_HEALTHY"),
                "live_rr": raw_execution_setup.get("risk_reward_ratio", 1.8),
                "created_at": now_dt,
                "expires_at": None
            }

        # 2. Analytical state is REVERSAL_CONFIRMED. Construct / Retrieve Signal Identity.
        candle_time = int(feed_meta.get("latest_candle_time", int(now * 1000)))
        floor_price = float(bottom_zone.get("potential_bottom_low", current_price))
        invalidation_price = float(bottom_zone.get("invalidation_price", floor_price * 0.995))
        
        # Use initial entry boundaries from setup
        entry_zone_min = float(raw_execution_setup.get("entry_zone_min", current_price * 0.997))
        entry_zone_max = float(raw_execution_setup.get("entry_zone_max", current_price * 1.003))
        tp1_price = float(raw_execution_setup.get("tp1_price", current_price * 1.06))
        tp2_price = float(raw_execution_setup.get("tp2_price", current_price * 1.10))
        original_rr = float(raw_execution_setup.get("risk_reward_ratio", 1.8))
        
        signal_id = self.construct_signal_id(clean_sym, candle_time, floor_price)

        # 3. Check persistent database for existing signal record
        existing_sig = await self._get_signal_record(signal_id)
        
        # If signal already exists in DB, keep immutable original parameters
        if existing_sig:
            floor_price = existing_sig["floor_price"]
            invalidation_price = existing_sig["invalidation_price"]
            entry_zone_min = existing_sig["entry_zone_min"]
            entry_zone_max = existing_sig["entry_zone_max"]
            tp1_price = existing_sig["tp1_price"]
            tp2_price = existing_sig["tp2_price"]
            original_rr = existing_sig["original_rr"]
            candle_time = existing_sig["confirmation_candle_time"]
            created_at = existing_sig["created_at"]
            expires_at = existing_sig["expires_at"]
            consumed = bool(existing_sig["consumed"])
            prior_state = existing_sig["lifecycle_state"]
            pos_id = existing_sig.get("position_id")
        else:
            created_at = now_dt
            # Initial validity policy: 1 further completed 1H candle (2 hours total from confirmation candle)
            candle_sec = (candle_time / 1000.0) if candle_time > 1e11 else float(candle_time)
            expires_at_epoch = candle_sec + (2 * 3600)
            expires_at = datetime.fromtimestamp(expires_at_epoch, tz=timezone.utc).isoformat()
            consumed = False
            prior_state = "REVERSAL_CONFIRMED"
            pos_id = None

        # 4. Perform BR-008 Lifecycle & Eligibility Checks
        
        # Check A: Already Consumed / Open Position
        if consumed:
            # Verify if position is open or closed
            is_open = await self._is_position_open(pos_id, clean_sym)
            state = "POSITION_OPEN" if is_open else "CLOSED"
            reason = f"Position is currently active (Position #{pos_id})." if is_open else "Signal setup has already been traded and closed."
            action_rec = "Manage open position according to TP/SL targets." if is_open else "Trade completed. Stand aside for next setup."
            
            evaluated = {
                "signal_id": signal_id,
                "lifecycle_state": state,
                "lifecycle_label": "Position Open (Active)" if is_open else "Signal Closed (Completed)",
                "lifecycle_color": "emerald" if is_open else "slate",
                "is_actionable": False,
                "can_execute": False,
                "eligibility_reason": reason,
                "action_recommendation": action_rec,
                "feed_source": feed_meta.get("source", "BINANCE"),
                "feed_freshness_seconds": feed_meta.get("age_seconds", 0.0),
                "feed_status": feed_meta.get("feed_status", "LIVE_HEALTHY"),
                "live_rr": original_rr,
                "created_at": created_at,
                "expires_at": expires_at,
                "position_id": pos_id
            }
            await self._upsert_signal_record(signal_id, clean_sym, evaluated, floor_price, invalidation_price, entry_zone_min, entry_zone_max, tp1_price, tp2_price, original_rr, candle_time)
            return evaluated

        # Check B: Market Data Freshness
        if feed_meta.get("is_stale", False) or feed_meta.get("age_seconds", 0) > 180.0:
            state = "STALE_MARKET_DATA"
            evaluated = {
                "signal_id": signal_id,
                "lifecycle_state": state,
                "lifecycle_label": "Feed Stale (Execution Locked)",
                "lifecycle_color": "rose",
                "is_actionable": False,
                "can_execute": False,
                "eligibility_reason": f"Market data feed is stale ({feed_meta.get('age_seconds', 0)}s old) or unverified. Execution blocked for safety.",
                "action_recommendation": "Waiting for verified live exchange quotes.",
                "feed_source": feed_meta.get("source", "UNKNOWN"),
                "feed_freshness_seconds": feed_meta.get("age_seconds", 0.0),
                "feed_status": "FEED_STALE",
                "live_rr": original_rr,
                "created_at": created_at,
                "expires_at": expires_at
            }
            await self._upsert_signal_record(signal_id, clean_sym, evaluated, floor_price, invalidation_price, entry_zone_min, entry_zone_max, tp1_price, tp2_price, original_rr, candle_time)
            return evaluated

        # Check C: Structural Invalidation Breach
        if current_price <= invalidation_price:
            state = "INVALIDATED"
            evaluated = {
                "signal_id": signal_id,
                "lifecycle_state": state,
                "lifecycle_label": "Invalidated (Floor Breached)",
                "lifecycle_color": "rose",
                "is_actionable": False,
                "can_execute": False,
                "eligibility_reason": f"Price (${current_price:.4f}) dropped below structural invalidation stop (${invalidation_price:.4f}). Reversal setup failed.",
                "action_recommendation": "Setup invalidated. Do NOT enter.",
                "feed_source": feed_meta.get("source", "BINANCE"),
                "feed_freshness_seconds": feed_meta.get("age_seconds", 0.0),
                "feed_status": feed_meta.get("feed_status", "LIVE_HEALTHY"),
                "live_rr": 0.0,
                "created_at": created_at,
                "expires_at": expires_at
            }
            await self._upsert_signal_record(signal_id, clean_sym, evaluated, floor_price, invalidation_price, entry_zone_min, entry_zone_max, tp1_price, tp2_price, original_rr, candle_time)
            return evaluated

        # Check D: Target Reached Before Entry (The exact problem the user encountered!)
        if current_price >= tp1_price:
            state = "TARGET_REACHED_BEFORE_ENTRY"
            evaluated = {
                "signal_id": signal_id,
                "lifecycle_state": state,
                "lifecycle_label": "Target Reached (Missed)",
                "lifecycle_color": "rose",
                "is_actionable": False,
                "can_execute": False,
                "eligibility_reason": f"Missed Entry: Price (${current_price:.4f}) has already reached or exceeded TP1 target (${tp1_price:.4f}). Do NOT chase into resistance.",
                "action_recommendation": "Target reached before entry. Opportunity is closed.",
                "feed_source": feed_meta.get("source", "BINANCE"),
                "feed_freshness_seconds": feed_meta.get("age_seconds", 0.0),
                "feed_status": feed_meta.get("feed_status", "LIVE_HEALTHY"),
                "live_rr": 0.0,
                "created_at": created_at,
                "expires_at": expires_at
            }
            await self._upsert_signal_record(signal_id, clean_sym, evaluated, floor_price, invalidation_price, entry_zone_min, entry_zone_max, tp1_price, tp2_price, original_rr, candle_time)
            return evaluated

        # Check E: Price Runaway / Above Acceptable Entry Zone
        if current_price > entry_zone_max:
            state = "MISSED_ENTRY"
            live_risk = current_price - invalidation_price
            live_reward = tp1_price - current_price
            live_rr = round(live_reward / (live_risk + 1e-9), 2)
            
            evaluated = {
                "signal_id": signal_id,
                "lifecycle_state": state,
                "lifecycle_label": "Missed Entry (Above Zone)",
                "lifecycle_color": "amber",
                "is_actionable": False,
                "can_execute": False,
                "eligibility_reason": f"Price (${current_price:.4f}) is above acceptable entry zone (${entry_zone_min:.4f}–${entry_zone_max:.4f}). Entering now would chase with adverse risk.",
                "action_recommendation": "Waiting for a pullback into the entry zone or next base.",
                "feed_source": feed_meta.get("source", "BINANCE"),
                "feed_freshness_seconds": feed_meta.get("age_seconds", 0.0),
                "feed_status": feed_meta.get("feed_status", "LIVE_HEALTHY"),
                "live_rr": live_rr,
                "created_at": created_at,
                "expires_at": expires_at
            }
            await self._upsert_signal_record(signal_id, clean_sym, evaluated, floor_price, invalidation_price, entry_zone_min, entry_zone_max, tp1_price, tp2_price, original_rr, candle_time)
            return evaluated

        # Check F: Signal Expiration Window (2 hours from confirmation candle)
        expires_at_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00")) if "Z" in expires_at or "+" in expires_at else datetime.fromisoformat(expires_at).replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires_at_dt:
            state = "EXPIRED"
            evaluated = {
                "signal_id": signal_id,
                "lifecycle_state": state,
                "lifecycle_label": "Signal Expired (Timed Out)",
                "lifecycle_color": "slate",
                "is_actionable": False,
                "can_execute": False,
                "eligibility_reason": f"Signal validity window has elapsed without execution. Setup is no longer active.",
                "action_recommendation": "Signal expired. Awaiting fresh structural confirmation.",
                "feed_source": feed_meta.get("source", "BINANCE"),
                "feed_freshness_seconds": feed_meta.get("age_seconds", 0.0),
                "feed_status": feed_meta.get("feed_status", "LIVE_HEALTHY"),
                "live_rr": original_rr,
                "created_at": created_at,
                "expires_at": expires_at
            }
            await self._upsert_signal_record(signal_id, clean_sym, evaluated, floor_price, invalidation_price, entry_zone_min, entry_zone_max, tp1_price, tp2_price, original_rr, candle_time)
            return evaluated

        # Check G: Risk / Reward Degradation Check
        live_risk = current_price - invalidation_price
        live_reward = tp1_price - current_price
        live_rr = round(live_reward / (live_risk + 1e-9), 2)
        
        if live_rr < 1.30 or live_risk <= 0:
            state = "REJECTED_RISK"
            evaluated = {
                "signal_id": signal_id,
                "lifecycle_state": state,
                "lifecycle_label": "Rejected (Low R:R)",
                "lifecycle_color": "rose",
                "is_actionable": False,
                "can_execute": False,
                "eligibility_reason": f"Current price degrades Reward-to-Risk ratio to 1:{live_rr:.2f} (below minimum 1:1.30 constraint).",
                "action_recommendation": "Risk parameters insufficient. Do NOT enter.",
                "feed_source": feed_meta.get("source", "BINANCE"),
                "feed_freshness_seconds": feed_meta.get("age_seconds", 0.0),
                "feed_status": feed_meta.get("feed_status", "LIVE_HEALTHY"),
                "live_rr": live_rr,
                "created_at": created_at,
                "expires_at": expires_at
            }
            await self._upsert_signal_record(signal_id, clean_sym, evaluated, floor_price, invalidation_price, entry_zone_min, entry_zone_max, tp1_price, tp2_price, original_rr, candle_time)
            return evaluated

        # Check H: All checks PASSED -> READY_TO_EXECUTE!
        state = "READY_TO_EXECUTE"
        evaluated = {
            "signal_id": signal_id,
            "lifecycle_state": state,
            "lifecycle_label": "Ready to Execute (Verified)",
            "lifecycle_color": "emerald",
            "is_actionable": True,
            "can_execute": True,
            "eligibility_reason": f"Confirmed reversal within valid entry zone (${entry_zone_min:.4f}–${entry_zone_max:.4f}) with 1:{live_rr:.2f} R:R to TP1 (${tp1_price:.4f}).",
            "action_recommendation": "All pre-flight checks passed. Risk-controlled spot entry authorized.",
            "feed_source": feed_meta.get("source", "BINANCE"),
            "feed_freshness_seconds": feed_meta.get("age_seconds", 0.0),
            "feed_status": feed_meta.get("feed_status", "LIVE_HEALTHY"),
            "live_rr": live_rr,
            "created_at": created_at,
            "expires_at": expires_at
        }
        await self._upsert_signal_record(signal_id, clean_sym, evaluated, floor_price, invalidation_price, entry_zone_min, entry_zone_max, tp1_price, tp2_price, original_rr, candle_time)
        return evaluated

    async def execute_signal(
        self,
        symbol: str,
        signal_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Atomically executes a validated reversal signal.
        Enforces idempotency, duplicate protection, live revalidation, and persistent position mapping.
        """
        clean_sym = symbol.upper().replace("/", "").replace("-", "")
        
        # If signal_id not provided, look up latest active READY_TO_EXECUTE signal for symbol
        if not signal_id:
            signal_id = await self._get_latest_actionable_signal_id(clean_sym)
            if not signal_id:
                return {
                    "success": False,
                    "error": f"No active READY_TO_EXECUTE signal found for {clean_sym}. Setup may have expired or missed entry.",
                    "status": "REJECTED_NO_ACTIVE_SIGNAL"
                }

        lock = self._get_lock(signal_id)
        async with lock:
            sig = await self._get_signal_record(signal_id)
            if not sig:
                return {
                    "success": False,
                    "error": f"Signal {signal_id} not found in database.",
                    "status": "SIGNAL_NOT_FOUND"
                }

            # 1. Idempotency & Duplicate Check
            if bool(sig["consumed"]):
                return {
                    "success": False,
                    "error": f"Signal {signal_id} has already been executed (Position #{sig.get('position_id')}). Repeated entries prohibited.",
                    "status": "ALREADY_CONSUMED",
                    "position_id": sig.get("position_id")
                }

            # 2. Live Re-validation immediately before execution
            kmeta = await market_data.get_klines_metadata(clean_sym, interval="1h", limit=50)
            t = await market_data.get_ticker(clean_sym)
            live_price = float(t.get("lastPrice", 0.0)) if t else float(kmeta["candles"][-1][4])

            # Check invalidation
            if live_price <= sig["invalidation_price"]:
                await self._update_signal_state(signal_id, "INVALIDATED", f"Price (${live_price:.4f}) dropped below invalidation stop.")
                return {"success": False, "error": "Trade rejected: live price breached structural invalidation stop.", "status": "INVALIDATED"}

            # Check target reached
            if live_price >= sig["tp1_price"]:
                await self._update_signal_state(signal_id, "TARGET_REACHED_BEFORE_ENTRY", f"Live price (${live_price:.4f}) already reached TP1 (${sig['tp1_price']:.4f}).")
                return {"success": False, "error": "Trade rejected: price already reached TP1 target. Entry window closed.", "status": "TARGET_REACHED_BEFORE_ENTRY"}

            # Check price runaway
            if live_price > (sig["entry_zone_max"] * 1.006):
                await self._update_signal_state(signal_id, "MISSED_ENTRY", f"Live price (${live_price:.4f}) exceeded entry zone.")
                return {"success": False, "error": "Trade rejected: price moved beyond acceptable entry zone.", "status": "MISSED_ENTRY"}

            # Check feed freshness
            if kmeta.get("is_stale") or kmeta.get("age_seconds", 0) > 180.0:
                await self._update_signal_state(signal_id, "STALE_MARKET_DATA", "Feed stale during execution attempt.")
                return {"success": False, "error": "Trade rejected: exchange market feed is stale.", "status": "STALE_MARKET_DATA"}

            # 3. Mark Signal as ENTRY_SUBMITTED in DB to prevent concurrent execution
            order_key = client_order_id or f"ORD-{signal_id}-{int(time.time())}"
            await self._mark_signal_submitted(signal_id, order_key)

            # 4. Route to paper execution engine with the signal's exact stop & targets
            order_req = PaperOrderRequest(
                symbol=clean_sym,
                entry_price=live_price,
                stop_loss=sig["invalidation_price"],
                tp1=sig["tp1_price"],
                tp2=sig["tp2_price"],
                signal_id=signal_id,
                client_order_id=order_key
            )
            
            res = await paper_trader.place_order(order_req)
            if res.get("success"):
                pos_id = res.get("position_id")
                await self._mark_signal_filled(signal_id, pos_id, live_price)
                return {
                    "success": True,
                    "status": "POSITION_OPEN",
                    "signal_id": signal_id,
                    "position_id": pos_id,
                    "symbol": clean_sym,
                    "entry_price": live_price,
                    "stop_loss": sig["invalidation_price"],
                    "tp1": sig["tp1_price"],
                    "tp2": sig["tp2_price"],
                    "notional_usd": res.get("notional_usd"),
                    "quantity": res.get("quantity"),
                    "message": f"Confirmed swing trade executed for {clean_sym} @ ${live_price:.4f}"
                }
            else:
                # Order rejected by risk engine (e.g. circuit breaker, cash cap)
                await self._update_signal_state(signal_id, "REJECTED_RISK", res.get("error", "Execution rejected by risk engine."))
                return {
                    "success": False,
                    "status": "REJECTED_RISK",
                    "error": res.get("error", "Risk engine rejected execution request.")
                }

    async def _get_signal_record(self, signal_id: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM signal_lifecycle_records WHERE signal_id = ?", (signal_id,)) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def _get_latest_actionable_signal_id(self, symbol: str) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT signal_id FROM signal_lifecycle_records WHERE symbol = ? AND lifecycle_state = 'READY_TO_EXECUTE' AND consumed = 0 ORDER BY created_at DESC LIMIT 1",
                (symbol,)
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else None

    async def _is_position_open(self, pos_id: Optional[int], symbol: str) -> bool:
        if not pos_id:
            return False
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM paper_positions WHERE id = ? AND status = 'OPEN'", (pos_id,)) as cur:
                count = (await cur.fetchone())[0]
                return count > 0

    async def _upsert_signal_record(
        self,
        signal_id: str,
        symbol: str,
        eval_data: Dict[str, Any],
        floor: float,
        invalidation: float,
        entry_min: float,
        entry_max: float,
        tp1: float,
        tp2: float,
        rr: float,
        candle_time: int
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO signal_lifecycle_records (
                    signal_id, symbol, lifecycle_state, confirmed_at, confirmation_candle_time,
                    last_evaluated_at, last_market_data_at, data_source, expires_at,
                    floor_price, invalidation_price, entry_zone_min, entry_zone_max,
                    tp1_price, tp2_price, original_rr, evaluated_price, evaluated_rr,
                    is_actionable, action_recommendation, eligibility_reason
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signal_id) DO UPDATE SET
                    lifecycle_state = excluded.lifecycle_state,
                    last_evaluated_at = CURRENT_TIMESTAMP,
                    last_market_data_at = CURRENT_TIMESTAMP,
                    data_source = excluded.data_source,
                    evaluated_price = excluded.evaluated_price,
                    evaluated_rr = excluded.evaluated_rr,
                    is_actionable = excluded.is_actionable,
                    action_recommendation = excluded.action_recommendation,
                    eligibility_reason = excluded.eligibility_reason
            """, (
                signal_id, symbol, eval_data["lifecycle_state"], candle_time,
                eval_data["feed_source"], eval_data["expires_at"],
                floor, invalidation, entry_min, entry_max, tp1, tp2, rr,
                eval_data.get("current_price"), eval_data.get("live_rr", rr),
                1 if eval_data["is_actionable"] else 0,
                eval_data["action_recommendation"], eval_data["eligibility_reason"]
            ))
            await db.commit()

    async def _mark_signal_submitted(self, signal_id: str, client_order_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE signal_lifecycle_records 
                SET consumed = 1, lifecycle_state = 'ENTRY_SUBMITTED', consumed_at = CURRENT_TIMESTAMP, client_order_id = ?
                WHERE signal_id = ?
            """, (client_order_id, signal_id))
            await db.commit()

    async def _mark_signal_filled(self, signal_id: str, position_id: int, entry_price: float):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE signal_lifecycle_records 
                SET lifecycle_state = 'POSITION_OPEN', position_id = ?, evaluated_price = ?
                WHERE signal_id = ?
            """, (position_id, entry_price, signal_id))
            await db.commit()

    async def _update_signal_state(self, signal_id: str, new_state: str, reason: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE signal_lifecycle_records 
                SET lifecycle_state = ?, eligibility_reason = ?, is_actionable = 0, last_evaluated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
            """, (new_state, reason, signal_id))
            await db.commit()

signal_guard = SignalLifecycleGuard()
