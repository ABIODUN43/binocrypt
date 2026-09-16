import aiosqlite
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..core.config import settings
from ..models.schemas import PaperAccountSummary, PaperPosition, PaperOrderRequest
from .market_data import market_data
from .risk_engine import risk_engine

class PaperTradingService:
    def __init__(self):
        self.db_path = settings.DB_PATH

    async def get_account_summary(self) -> PaperAccountSummary:
        """Calculates live equity, PnL, win rate, and $40 milestone progress."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Fetch account record
            async with db.execute("SELECT * FROM paper_account WHERE id = 1") as cursor:
                acc = await cursor.fetchone()
                if not acc:
                    starting = settings.DEFAULT_CAPITAL
                    curr_eq = starting
                    cash = starting
                    daily_start = starting
                    is_circuit = 0
                else:
                    starting = acc["starting_equity"]
                    curr_eq = acc["current_equity"]
                    cash = acc["available_cash"]
                    daily_start = acc["daily_start_equity"]
                    is_circuit = acc["is_circuit_breaker_active"]

            # Fetch open positions and compute live unrealized PnL
            async with db.execute("SELECT * FROM paper_positions WHERE status = 'OPEN'") as cursor:
                open_rows = await cursor.fetchall()
            
            unrealized_pnl = 0.0
            open_count = len(open_rows)

            for pos in open_rows:
                sym = pos["symbol"]
                t = await market_data.get_ticker(sym)
                if t:
                    live_price = float(t.get("lastPrice", pos["current_price"]))
                    qty = pos["quantity"]
                    entry = pos["entry_price"]
                    pos_pnl = (live_price - entry) * qty
                    unrealized_pnl += pos_pnl
                    
                    # Update row with latest price
                    await db.execute("""
                        UPDATE paper_positions 
                        SET current_price = ?, pnl_usd = ?, pnl_pct = ?
                        WHERE id = ?
                    """, (live_price, pos_pnl, ((live_price - entry)/entry)*100.0, pos["id"]))

            await db.commit()

            # Live equity = available cash + notional of open positions + unrealized PnL
            invested_cash = sum(pos["notional_usd"] for pos in open_rows)
            live_equity = round(cash + invested_cash + unrealized_pnl, 2)
            total_pnl_usd = round(live_equity - starting, 2)
            total_return_pct = round((total_pnl_usd / starting) * 100.0, 2)

            daily_pnl_usd = round(live_equity - daily_start, 2)
            daily_pnl_pct = round((daily_pnl_usd / daily_start) * 100.0, 2)

            # Historical closed trade stats
            async with db.execute("SELECT * FROM paper_positions WHERE status = 'CLOSED'") as cursor:
                closed_rows = await cursor.fetchall()

            total_trades = len(closed_rows)
            winning_trades = sum(1 for r in closed_rows if r["pnl_usd"] > 0)
            win_rate = round((winning_trades / total_trades * 100.0), 1) if total_trades > 0 else 0.0

            # Drawdown calculation
            peak = max(acc["peak_equity"] if acc else starting, live_equity)
            dd_pct = round(((peak - live_equity) / peak) * 100.0, 2) if peak > 0 else 0.0

            # Update peak if new high
            if acc and live_equity > acc["peak_equity"]:
                await db.execute("UPDATE paper_account SET peak_equity = ?, current_equity = ? WHERE id = 1", (live_equity, live_equity))
                await db.commit()

            # $40 -> $150–$200 milestone progress
            # 0% at $40, 100% at $175 (midpoint of $150–$200)
            target_goal = 175.0
            milestone_pct = max(0.0, min(100.0, round(((live_equity - starting) / (target_goal - starting)) * 100.0, 1)))

            return PaperAccountSummary(
                starting_equity=starting,
                current_equity=live_equity,
                peak_equity=peak,
                available_cash=round(cash, 2),
                daily_start_equity=daily_start,
                total_pnl_usd=total_pnl_usd,
                total_return_pct=total_return_pct,
                daily_pnl_usd=daily_pnl_usd,
                daily_pnl_pct=daily_pnl_pct,
                max_drawdown_pct=dd_pct,
                win_rate_pct=win_rate,
                total_trades=total_trades,
                open_positions_count=open_count,
                is_circuit_breaker_active=bool(is_circuit or dd_pct >= settings.MAX_DAILY_DRAWDOWN_PCT),
                experiment_milestone_pct=milestone_pct
            )

    async def get_open_positions(self) -> List[PaperPosition]:
        """Fetch all currently open paper positions with live metrics."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM paper_positions WHERE status = 'OPEN' ORDER BY opened_at DESC") as cursor:
                rows = await cursor.fetchall()
                
            positions = []
            for r in rows:
                positions.append(PaperPosition(
                    id=r["id"],
                    symbol=r["symbol"],
                    side=r["side"],
                    status=r["status"],
                    entry_price=r["entry_price"],
                    current_price=r["current_price"],
                    quantity=r["quantity"],
                    notional_usd=r["notional_usd"],
                    stop_loss=r["stop_loss"],
                    tp1=r["tp1"],
                    tp2=r["tp2"],
                    tp1_hit=bool(r["tp1_hit"]),
                    risk_amount=r["risk_amount"],
                    exit_price=r["exit_price"],
                    pnl_usd=r["pnl_usd"],
                    pnl_pct=r["pnl_pct"],
                    fees_paid=r["fees_paid"],
                    exit_reason=r["exit_reason"],
                    opened_at=r["opened_at"],
                    closed_at=r["closed_at"]
                ))
            return positions

    async def get_trade_history(self) -> List[PaperPosition]:
        """Fetch closed paper trading history."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM paper_positions WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT 50") as cursor:
                rows = await cursor.fetchall()

            history = []
            for r in rows:
                history.append(PaperPosition(
                    id=r["id"],
                    symbol=r["symbol"],
                    side=r["side"],
                    status=r["status"],
                    entry_price=r["entry_price"],
                    current_price=r["current_price"],
                    quantity=r["quantity"],
                    notional_usd=r["notional_usd"],
                    stop_loss=r["stop_loss"],
                    tp1=r["tp1"],
                    tp2=r["tp2"],
                    tp1_hit=bool(r["tp1_hit"]),
                    risk_amount=r["risk_amount"],
                    exit_price=r["exit_price"],
                    pnl_usd=r["pnl_usd"],
                    pnl_pct=r["pnl_pct"],
                    fees_paid=r["fees_paid"],
                    exit_reason=r["exit_reason"],
                    opened_at=r["opened_at"],
                    closed_at=r["closed_at"]
                ))
            return history

    async def place_order(self, req: PaperOrderRequest) -> Dict[str, Any]:
        """Validates risk rules and opens a paper spot position."""
        acc = await self.get_account_summary()
        
        # 1. Check Circuit Breaker
        if acc.is_circuit_breaker_active:
            return {"success": False, "error": "Daily risk limit or drawdown circuit breaker active. New trades prohibited."}

        # 2. Check Open Positions Cap (Max 2 for $40 account)
        if acc.open_positions_count >= settings.MAX_OPEN_POSITIONS:
            return {"success": False, "error": f"Max concurrent positions reached ({settings.MAX_OPEN_POSITIONS}). Protect account capital."}

        # 3. Determine entry price (prefer provided price, fallback to live ticker or memory cache)
        entry_price = req.entry_price
        if not entry_price or entry_price <= 0:
            t = await market_data.get_ticker(req.symbol)
            if t:
                entry_price = float(t.get("lastPrice", 0.0))

        if not entry_price or entry_price <= 0:
            cached_t = market_data.ticker_map.get(req.symbol.upper())
            if cached_t:
                entry_price = float(cached_t.get("lastPrice", 0.0))

        if not entry_price or entry_price <= 0:
            return {"success": False, "error": f"Could not determine price for {req.symbol}. Please enter entry price."}

        # Set or validate Stop Loss & Targets
        stop_loss = req.stop_loss or (entry_price * 0.975) # default 2.5% stop
        if stop_loss >= entry_price:
            return {"success": False, "error": "Stop loss must be lower than entry price for spot long."}

        risk_dist = entry_price - stop_loss
        tp1 = req.tp1 or round(entry_price + (risk_dist * 1.8), 4)
        tp2 = req.tp2 or round(entry_price + (risk_dist * 3.0), 4)

        # 4. Calculate Sizing via Risk Engine
        if req.custom_notional_usd:
            notional = min(acc.available_cash, req.custom_notional_usd)
        else:
            sizing = risk_engine.calculate_position_size(
                account_equity=acc.current_equity,
                risk_pct=settings.MAX_RISK_PER_TRADE_PCT,
                entry_price=entry_price,
                stop_loss=stop_loss
            )
            notional = min(acc.available_cash, sizing.recommended_position_usd)

        if notional < settings.MIN_ORDER_NOTIONAL:
            return {"success": False, "error": f"Order size (${notional:.2f}) below exchange minimum (${settings.MIN_ORDER_NOTIONAL:.2f} USDT)."}

        if notional > acc.available_cash:
            return {"success": False, "error": f"Insufficient available cash (${acc.available_cash:.2f}) for order (${notional:.2f})."}

        quantity = round(notional / entry_price, 6)
        entry_fee = round(notional * (settings.DEFAULT_FEE_PCT / 100.0), 4)
        risk_amt = round(notional * ((entry_price - stop_loss) / entry_price), 2)

        async with aiosqlite.connect(self.db_path) as db:
            # Deduct cash
            new_cash = acc.available_cash - notional - entry_fee
            await db.execute("UPDATE paper_account SET available_cash = ? WHERE id = 1", (new_cash,))
            
            # Insert position
            cursor = await db.execute("""
                INSERT INTO paper_positions 
                (symbol, side, status, entry_price, current_price, quantity, notional_usd, 
                 stop_loss, tp1, tp2, risk_amount, fees_paid, opened_at)
                VALUES (?, 'BUY', 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (req.symbol.upper(), entry_price, entry_price, quantity, notional, stop_loss, tp1, tp2, risk_amt, entry_fee))
            
            pos_id = cursor.lastrowid
            await db.commit()

        return {
            "success": True,
            "position_id": pos_id,
            "symbol": req.symbol.upper(),
            "entry_price": entry_price,
            "quantity": quantity,
            "notional_usd": notional,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "risk_amount": risk_amt,
            "fee_paid": entry_fee
        }

    async def close_position(self, position_id: int, exit_reason: str = "MANUAL_CLOSE") -> Dict[str, Any]:
        """Closes an open paper position and updates cash & statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM paper_positions WHERE id = ? AND status = 'OPEN'", (position_id,)) as cursor:
                pos = await cursor.fetchone()
                if not pos:
                    return {"success": False, "error": "Position not found or already closed."}

            t = await market_data.get_ticker(pos["symbol"])
            exit_price = float(t.get("lastPrice", pos["current_price"])) if t else pos["current_price"]
            qty = pos["quantity"]
            entry = pos["entry_price"]
            
            exit_notional = exit_price * qty
            exit_fee = exit_notional * (settings.DEFAULT_FEE_PCT / 100.0)
            gross_pnl = (exit_price - entry) * qty
            net_pnl = gross_pnl - pos["fees_paid"] - exit_fee
            pnl_pct = ((exit_price - entry) / entry) * 100.0

            # Update account cash
            async with db.execute("SELECT available_cash, current_equity FROM paper_account WHERE id = 1") as cursor:
                acc = await cursor.fetchone()
                cur_cash = acc["available_cash"]
                new_cash = round(cur_cash + exit_notional - exit_fee, 2)
                new_equity = round(acc["current_equity"] + net_pnl, 2)

            await db.execute("""
                UPDATE paper_account 
                SET available_cash = ?, current_equity = ?
                WHERE id = 1
            """, (new_cash, new_equity))

            # Mark position as CLOSED
            await db.execute("""
                UPDATE paper_positions 
                SET status = 'CLOSED', current_price = ?, exit_price = ?, 
                    pnl_usd = ?, pnl_pct = ?, fees_paid = fees_paid + ?, 
                    exit_reason = ?, closed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (exit_price, exit_price, round(net_pnl, 2), round(pnl_pct, 2), exit_fee, exit_reason, position_id))

            await db.commit()

        return {
            "success": True,
            "position_id": position_id,
            "exit_price": exit_price,
            "net_pnl_usd": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "exit_reason": exit_reason
        }

    async def check_triggers(self):
        """Scans open positions and triggers TP1, TP2, or SL if price criteria met."""
        positions = await self.get_open_positions()
        for p in positions:
            t = await market_data.get_ticker(p.symbol)
            if not t:
                continue
            curr = float(t.get("lastPrice", p.current_price))
            
            # 1. Check Stop Loss
            if curr <= p.stop_loss:
                await self.close_position(p.id, exit_reason="STOP_LOSS")
                continue

            # 2. Check TP2
            if curr >= p.tp2:
                await self.close_position(p.id, exit_reason="TP2_HIT")
                continue

            # 3. Check TP1 (secure breakeven stop)
            if not p.tp1_hit and curr >= p.tp1:
                # Move stop to breakeven (entry price)
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        UPDATE paper_positions 
                        SET tp1_hit = 1, stop_loss = ? 
                        WHERE id = ?
                    """, (p.entry_price, p.id))
                    await db.commit()

    async def reset_account(self, capital: float = 40.0):
        """Resets paper account to starting state ($40.00)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM paper_positions")
            await db.execute("DELETE FROM equity_snapshots")
            await db.execute("""
                UPDATE paper_account 
                SET starting_equity = ?, current_equity = ?, peak_equity = ?, 
                    available_cash = ?, daily_start_equity = ?, is_circuit_breaker_active = 0
                WHERE id = 1
            """, (capital, capital, capital, capital, capital))
            
            await db.execute("""
                INSERT INTO equity_snapshots (equity, cash, pnl_usd, return_pct, drawdown_pct)
                VALUES (?, ?, 0.0, 0.0, 0.0)
            """, (capital, capital))
            await db.commit()

paper_trader = PaperTradingService()
