import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timezone
from ..models.schemas import BacktestResult, BacktestTrade
from ..core.config import settings
from .features import feature_engine
from .market_structure import market_structure_analyzer

class QuantitativeBacktester:
    @staticmethod
    def run_backtest(
        klines: List[List[Any]],
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        initial_capital: float = 40.0,
        risk_pct: float = 2.0,
        fee_pct: float = 0.10,
        slippage_pct: float = 0.05,
        target_rr: float = 2.0
    ) -> BacktestResult:
        """
        Executes event-driven bar-by-bar backtest over historical klines:
        klines: [[timestamp, open, high, low, close, volume], ...]
        """
        if len(klines) < 60:
            raise ValueError(f"Insufficient klines ({len(klines)}) to run backtest. Need at least 60.")

        capital = initial_capital
        peak_equity = initial_capital
        max_drawdown = 0.0
        trades: List[BacktestTrade] = []
        
        in_position = False
        entry_price = 0.0
        entry_time = ""
        position_size_usd = 0.0
        quantity = 0.0
        stop_loss = 0.0
        tp1 = 0.0
        tp2 = 0.0
        
        total_bars = len(klines)

        # Bar-by-bar iteration starting after warm-up period of 35 bars
        for i in range(35, total_bars - 1):
            current_bar = klines[i]
            next_bar = klines[i + 1]
            
            bar_time = datetime.fromtimestamp(current_bar[0] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
            next_bar_time = datetime.fromtimestamp(next_bar[0] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
            
            # 1. Manage existing position against next_bar (High, Low)
            if in_position:
                n_high = next_bar[2]
                n_low = next_bar[3]
                
                # Check Stop Loss first (conservative execution)
                if n_low <= stop_loss:
                    exit_price = stop_loss * (1 - slippage_pct / 100.0)
                    gross_pnl = (exit_price - entry_price) * quantity
                    fee = (entry_price * quantity * fee_pct / 100.0) + (exit_price * quantity * fee_pct / 100.0)
                    net_pnl = gross_pnl - fee
                    capital += net_pnl
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    
                    trades.append(BacktestTrade(
                        entry_time=entry_time,
                        exit_time=next_bar_time,
                        symbol=symbol,
                        entry_price=round(entry_price, 4),
                        exit_price=round(exit_price, 4),
                        stop_loss=round(stop_loss, 4),
                        tp1=round(tp1, 4),
                        tp2=round(tp2, 4),
                        exit_reason="STOP_LOSS",
                        pnl_usd=round(gross_pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        fee_usd=round(fee, 3),
                        net_pnl_usd=round(net_pnl, 2),
                        cumulative_equity=round(capital, 2)
                    ))
                    in_position = False
                    
                # Check TP2 / TP1
                elif n_high >= tp1:
                    # Target achieved
                    exit_price = tp1 * (1 - slippage_pct / 100.0)
                    gross_pnl = (exit_price - entry_price) * quantity
                    fee = (entry_price * quantity * fee_pct / 100.0) + (exit_price * quantity * fee_pct / 100.0)
                    net_pnl = gross_pnl - fee
                    capital += net_pnl
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    
                    trades.append(BacktestTrade(
                        entry_time=entry_time,
                        exit_time=next_bar_time,
                        symbol=symbol,
                        entry_price=round(entry_price, 4),
                        exit_price=round(exit_price, 4),
                        stop_loss=round(stop_loss, 4),
                        tp1=round(tp1, 4),
                        tp2=round(tp2, 4),
                        exit_reason="TAKE_PROFIT",
                        pnl_usd=round(gross_pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        fee_usd=round(fee, 3),
                        net_pnl_usd=round(net_pnl, 2),
                        cumulative_equity=round(capital, 2)
                    ))
                    in_position = False

                # Update drawdown tracking
                if capital > peak_equity:
                    peak_equity = capital
                dd = ((peak_equity - capital) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                if dd > max_drawdown:
                    max_drawdown = dd

                continue

            # 2. If not in position, evaluate signal using ONLY historical data up to bar i
            sub_klines = klines[:i + 1]
            feat = feature_engine.calculate_indicators(sub_klines)
            if not feat:
                continue

            struct = market_structure_analyzer.analyze_structure(sub_klines)
            
            c_price = feat["price"]
            ema20 = feat["ema_20"]
            ema50 = feat["ema_50"]
            rsi = feat["rsi_14"]
            rvol = feat["rvol_20"]
            macd_hist = feat["macd_hist"]
            stype = struct.get("structure_type", "")

            # Confluence Entry Rule:
            # 1. Price > EMA 20 and EMA 20 > EMA 50
            # 2. 50 <= RSI <= 72 (momentum without extreme exhaustion)
            # 3. MACD histogram positive or crossing up
            # 4. RVOL >= 1.15
            # 5. Structure is BREAKOUT or BULLISH_HH_HL or PULLBACK
            is_entry_signal = (
                c_price > ema20 and 
                ema20 > ema50 and 
                50.0 <= rsi <= 72.0 and 
                macd_hist > -0.05 and 
                rvol >= 1.15 and 
                stype in ["BREAKOUT", "BULLISH_HH_HL", "BULLISH_PULLBACK"]
            )

            if is_entry_signal and capital > 5.0:
                # Enter on next bar open with slippage
                entry_price = next_bar[1] * (1 + slippage_pct / 100.0)
                entry_time = next_bar_time
                
                # Stop loss placement: swing low or 1.5 * ATR
                sl_cand = struct.get("recent_swing_low", 0.0)
                atr = feat["atr_14"]
                if sl_cand > 0 and 0.015 <= ((entry_price - sl_cand) / entry_price) <= 0.05:
                    stop_loss = sl_cand * 0.997
                else:
                    stop_loss = entry_price - (1.5 * atr)

                stop_dist = entry_price - stop_loss
                if stop_dist <= 0 or (stop_dist / entry_price) > 0.06:
                    stop_loss = entry_price * 0.975 # fallback 2.5% stop
                    stop_dist = entry_price - stop_loss

                # Targets
                tp1 = entry_price + (stop_dist * target_rr)
                tp2 = entry_price + (stop_dist * (target_rr + 1.2))

                # Position sizing for small account
                risk_amt = capital * (risk_pct / 100.0)
                raw_size = risk_amt / ((entry_price - stop_loss) / entry_price)
                
                # Cap at current capital or minimum $5 order
                position_size_usd = min(capital * 0.95, max(settings.MIN_ORDER_NOTIONAL, raw_size))
                quantity = position_size_usd / entry_price
                in_position = True

        # Wrap up metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.net_pnl_usd > 0)
        losing_trades = sum(1 for t in trades if t.net_pnl_usd <= 0)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_wins = sum(t.net_pnl_usd for t in trades if t.net_pnl_usd > 0)
        gross_losses = abs(sum(t.net_pnl_usd for t in trades if t.net_pnl_usd < 0))
        profit_factor = round(gross_wins / (gross_losses + 1e-9), 2) if gross_losses > 0 else (9.9 if gross_wins > 0 else 0.0)

        net_profit = capital - initial_capital
        total_return_pct = (net_profit / initial_capital) * 100.0
        avg_trade_pct = np.mean([t.pnl_pct for t in trades]) if trades else 0.0

        # Sharpe ratio estimate from trade returns
        trade_returns = [t.pnl_pct for t in trades]
        if len(trade_returns) >= 3 and np.std(trade_returns) > 0:
            sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(len(trade_returns))
        else:
            sharpe = 0.0

        start_date = datetime.fromtimestamp(klines[0][0] / 1000, timezone.utc).strftime("%Y-%m-%d")
        end_date = datetime.fromtimestamp(klines[-1][0] / 1000, timezone.utc).strftime("%Y-%m-%d")

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=total_bars,
            start_time=start_date,
            end_time=end_date,
            starting_capital=initial_capital,
            final_equity=round(capital, 2),
            net_profit_usd=round(net_profit, 2),
            total_return_pct=round(total_return_pct, 2),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=round(win_rate, 1),
            profit_factor=profit_factor,
            max_drawdown_pct=round(max_drawdown, 2),
            sharpe_ratio=round(float(sharpe), 2),
            average_trade_pct=round(float(avg_trade_pct), 2),
            trades=trades
        )

backtester = QuantitativeBacktester()
