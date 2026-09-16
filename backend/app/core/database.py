import aiosqlite
import os
from contextlib import asynccontextmanager
from .config import settings

async def init_db():
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY,
                starting_equity REAL NOT NULL,
                current_equity REAL NOT NULL,
                peak_equity REAL NOT NULL,
                available_cash REAL NOT NULL,
                daily_start_equity REAL NOT NULL,
                is_circuit_breaker_active INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL DEFAULT 'BUY',
                status TEXT NOT NULL DEFAULT 'OPEN', -- OPEN, CLOSED
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                quantity REAL NOT NULL,
                notional_usd REAL NOT NULL,
                stop_loss REAL NOT NULL,
                tp1 REAL NOT NULL,
                tp2 REAL NOT NULL,
                tp1_hit INTEGER DEFAULT 0,
                risk_amount REAL NOT NULL,
                exit_price REAL,
                pnl_usd REAL DEFAULT 0.0,
                pnl_pct REAL DEFAULT 0.0,
                fees_paid REAL DEFAULT 0.0,
                exit_reason TEXT,
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                pnl_usd REAL NOT NULL,
                return_pct REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS backtest_cache (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                trades_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                target_date TEXT NOT NULL,
                predicted_regime TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                recovery_prob REAL NOT NULL,
                bear_exhaustion_score REAL NOT NULL,
                decision TEXT NOT NULL,
                entry_price REAL NOT NULL,
                bear_case_price REAL NOT NULL,
                base_case_price REAL NOT NULL,
                bull_case_price REAL NOT NULL,
                actual_exit_price REAL,
                actual_return_pct REAL,
                outcome_evaluated INTEGER DEFAULT 0,
                is_correct INTEGER,
                model_version TEXT DEFAULT 'RegimeEnsemble v2.4'
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_pred_symbol ON prediction_records(symbol)")

        await db.execute("""
            CREATE TABLE IF NOT EXISTS research_experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                universe TEXT NOT NULL,
                model_type TEXT NOT NULL,
                target_label TEXT NOT NULL,
                features_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                status TEXT NOT NULL,
                decision TEXT NOT NULL,
                rejection_reason TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS research_graveyard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                rejection_reason TEXT NOT NULL,
                flaw_type TEXT NOT NULL,
                lessons_learned TEXT NOT NULL,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Check if account row exists, if not initialize with $40 default
        async with db.execute("SELECT COUNT(*) FROM paper_account") as cursor:
            count = (await cursor.fetchone())[0]
            if count == 0:
                await db.execute("""
                    INSERT INTO paper_account (id, starting_equity, current_equity, peak_equity, available_cash, daily_start_equity)
                    VALUES (1, ?, ?, ?, ?, ?)
                """, (settings.DEFAULT_CAPITAL, settings.DEFAULT_CAPITAL, settings.DEFAULT_CAPITAL, settings.DEFAULT_CAPITAL, settings.DEFAULT_CAPITAL))
                
                await db.execute("""
                    INSERT INTO equity_snapshots (equity, cash, pnl_usd, return_pct, drawdown_pct)
                    VALUES (?, ?, 0.0, 0.0, 0.0)
                """, (settings.DEFAULT_CAPITAL, settings.DEFAULT_CAPITAL))

        await db.commit()

@asynccontextmanager
async def get_db():
    db = await aiosqlite.connect(settings.DB_PATH)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()
