import React, { useState } from 'react';
import { 
  DollarSign, 
  TrendingUp, 
  TrendingDown, 
  ShieldCheck, 
  ShieldAlert, 
  XCircle, 
  RotateCcw, 
  Clock, 
  CheckCircle, 
  AlertTriangle 
} from 'lucide-react';

export default function PaperPortfolioView({ 
  account, 
  positions, 
  history, 
  onClosePosition, 
  onResetAccount 
}) {
  const [isResetConfirmOpen, setIsResetConfirmOpen] = useState(false);

  if (!account) return null;

  return (
    <div className="space-y-6">
      
      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
        
        {/* Starting Capital */}
        <div className="terminal-card p-3.5">
          <div className="text-[10px] uppercase font-mono text-gray-400">Starting Equity</div>
          <div className="text-base font-bold font-mono-nums text-white mt-1">
            ${account.starting_equity.toFixed(2)}
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">$40 Experiment Base</div>
        </div>

        {/* Current Equity */}
        <div className="terminal-card p-3.5 border-blue-800/40">
          <div className="text-[10px] uppercase font-mono text-gray-400">Current Equity</div>
          <div className="text-lg font-bold font-mono-nums text-blue-400 mt-0.5">
            ${account.current_equity.toFixed(2)}
          </div>
          <div className="text-[10px] text-gray-400 font-mono-nums">
            Cash: ${account.available_cash.toFixed(2)}
          </div>
        </div>

        {/* Total Return */}
        <div className="terminal-card p-3.5">
          <div className="text-[10px] uppercase font-mono text-gray-400">Total Net P/L</div>
          <div className={`text-base font-bold font-mono-nums mt-1 ${account.total_pnl_usd >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {account.total_pnl_usd >= 0 ? '+' : ''}${account.total_pnl_usd.toFixed(2)}
          </div>
          <div className={`text-[10px] font-mono-nums ${account.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {account.total_return_pct >= 0 ? '+' : ''}{account.total_return_pct.toFixed(2)}%
          </div>
        </div>

        {/* Win Rate */}
        <div className="terminal-card p-3.5">
          <div className="text-[10px] uppercase font-mono text-gray-400">Win Rate</div>
          <div className="text-base font-bold font-mono-nums text-emerald-400 mt-1">
            {account.win_rate_pct.toFixed(1)}%
          </div>
          <div className="text-[10px] text-gray-500 font-mono-nums">
            {account.total_trades} Closed Trades
          </div>
        </div>

        {/* Max Drawdown */}
        <div className="terminal-card p-3.5">
          <div className="text-[10px] uppercase font-mono text-gray-400">Max Drawdown</div>
          <div className="text-base font-bold font-mono-nums text-rose-400 mt-1">
            -{account.max_drawdown_pct.toFixed(1)}%
          </div>
          <div className="text-[10px] text-gray-500 font-mono-nums">
            Peak: ${account.peak_equity.toFixed(2)}
          </div>
        </div>

        {/* Risk Circuit Breaker */}
        <div className="terminal-card p-3.5">
          <div className="text-[10px] uppercase font-mono text-gray-400">Circuit Breaker</div>
          <div className="flex items-center gap-1.5 mt-1">
            {account.is_circuit_breaker_active ? (
              <span className="flex items-center gap-1 text-xs font-bold text-rose-400">
                <ShieldAlert className="w-4 h-4" /> TRIPPED
              </span>
            ) : (
              <span className="flex items-center gap-1 text-xs font-bold text-emerald-400">
                <ShieldCheck className="w-4 h-4" /> ACTIVE &amp; SAFE
              </span>
            )}
          </div>
          <div className="text-[10px] text-gray-500">Max 5% Daily Loss</div>
        </div>

      </div>

      {/* Active Positions Table */}
      <div className="terminal-card overflow-hidden">
        <div className="p-4 border-b border-[#1e2638] flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Open Positions ({positions.length} / 2 Max)
            </h3>
            <p className="text-xs text-gray-400">Live positions automatically monitored for TP1, TP2, and SL</p>
          </div>

          <button
            onClick={() => setIsResetConfirmOpen(true)}
            className="px-2.5 py-1 rounded bg-[#0b0e14] hover:bg-rose-950/40 text-gray-400 hover:text-rose-300 border border-[#1e2638] hover:border-rose-800/40 text-xs flex items-center gap-1.5 transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Account ($40)</span>
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#1e2638] bg-[#0e121a] text-[11px] font-mono text-gray-400 uppercase tracking-wider">
                <th className="py-2.5 px-4">Asset</th>
                <th className="py-2.5 px-3">Entry Price</th>
                <th className="py-2.5 px-3">Live Price</th>
                <th className="py-2.5 px-3">Position Size</th>
                <th className="py-2.5 px-3">Stop Loss</th>
                <th className="py-2.5 px-3">TP1 / TP2</th>
                <th className="py-2.5 px-3">Unrealized P/L</th>
                <th className="py-2.5 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2638]/60 font-mono-nums">
              {positions.length === 0 ? (
                <tr>
                  <td colSpan="8" className="py-8 text-center text-gray-500 font-sans">
                    No active positions. Find an opportunity from the scanner to open a paper trade.
                  </td>
                </tr>
              ) : (
                positions.map((pos) => (
                  <tr key={pos.id} className="hover:bg-[#182030]/60 transition">
                    <td className="py-3 px-4 font-bold text-white font-sans">
                      {pos.symbol}
                      <span className="ml-1.5 text-[10px] text-emerald-400 font-mono">SPOT</span>
                    </td>
                    <td className="py-3 px-3 text-gray-200">
                      ${pos.entry_price < 1 ? pos.entry_price.toFixed(4) : pos.entry_price.toFixed(2)}
                    </td>
                    <td className="py-3 px-3 text-white font-semibold">
                      ${pos.current_price < 1 ? pos.current_price.toFixed(4) : pos.current_price.toFixed(2)}
                    </td>
                    <td className="py-3 px-3 text-gray-300">
                      ${pos.notional_usd.toFixed(2)}
                      <span className="text-[10px] text-gray-500 block">({pos.quantity} qty)</span>
                    </td>
                    <td className="py-3 px-3 text-rose-400 font-semibold">
                      ${pos.stop_loss < 1 ? pos.stop_loss.toFixed(4) : pos.stop_loss.toFixed(2)}
                      {pos.tp1_hit && <span className="text-[10px] text-blue-400 block font-sans">Breakeven Secured</span>}
                    </td>
                    <td className="py-3 px-3 text-gray-300">
                      <div className="text-emerald-400 font-medium">TP1: ${pos.tp1}</div>
                      <div className="text-cyan-400 text-[10px]">TP2: ${pos.tp2}</div>
                    </td>
                    <td className="py-3 px-3 font-semibold">
                      <div className={pos.pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"}>
                        {pos.pnl_usd >= 0 ? "+" : ""}${pos.pnl_usd.toFixed(2)}
                      </div>
                      <div className={`text-[10px] ${pos.pnl_pct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                        {pos.pnl_pct >= 0 ? "+" : ""}{pos.pnl_pct.toFixed(2)}%
                      </div>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => onClosePosition(pos.id)}
                        className="px-2.5 py-1 rounded bg-rose-600/20 hover:bg-rose-600/30 text-rose-400 border border-rose-500/30 text-xs font-semibold transition"
                      >
                        Close
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Closed Trades History Table */}
      <div className="terminal-card overflow-hidden">
        <div className="p-4 border-b border-[#1e2638]">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            Trade Execution Journal &amp; History
          </h3>
          <p className="text-xs text-gray-400">Audited fills with fee and slippage accounting</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#1e2638] bg-[#0e121a] text-[11px] font-mono text-gray-400 uppercase tracking-wider">
                <th className="py-2.5 px-4">Closed Time</th>
                <th className="py-2.5 px-3">Asset</th>
                <th className="py-2.5 px-3">Entry Price</th>
                <th className="py-2.5 px-3">Exit Price</th>
                <th className="py-2.5 px-3">Exit Reason</th>
                <th className="py-2.5 px-3">Fees Paid</th>
                <th className="py-2.5 px-4 text-right">Net P/L</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2638]/60 font-mono-nums">
              {history.length === 0 ? (
                <tr>
                  <td colSpan="7" className="py-6 text-center text-gray-500 font-sans">
                    No closed trades recorded yet.
                  </td>
                </tr>
              ) : (
                history.map((t) => (
                  <tr key={t.id} className="hover:bg-[#182030]/60 transition">
                    <td className="py-2.5 px-4 text-gray-400 font-sans text-[11px]">
                      {t.closed_at ? new Date(t.closed_at).toLocaleTimeString() : 'N/A'}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-white font-sans">
                      {t.symbol}
                    </td>
                    <td className="py-2.5 px-3 text-gray-300">
                      ${t.entry_price < 1 ? t.entry_price.toFixed(4) : t.entry_price.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-gray-300">
                      ${t.exit_price ? (t.exit_price < 1 ? t.exit_price.toFixed(4) : t.exit_price.toFixed(2)) : 'N/A'}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                        t.exit_reason === 'TP1_HIT' || t.exit_reason === 'TP2_HIT'
                          ? 'bg-emerald-950 text-emerald-400 border border-emerald-800/50'
                          : t.exit_reason === 'STOP_LOSS'
                          ? 'bg-rose-950 text-rose-400 border border-rose-800/50'
                          : 'bg-gray-800 text-gray-300'
                      }`}>
                        {t.exit_reason}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-gray-500">
                      -${t.fees_paid.toFixed(3)}
                    </td>
                    <td className="py-2.5 px-4 text-right font-semibold">
                      <span className={t.pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"}>
                        {t.pnl_usd >= 0 ? "+" : ""}${t.pnl_usd.toFixed(2)} ({t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct.toFixed(2)}%)
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Reset Confirmation Modal */}
      {isResetConfirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="terminal-card max-w-sm w-full p-5 bg-[#121722] border-rose-900/50">
            <h4 className="text-sm font-bold text-white mb-2">Reset Paper Portfolio?</h4>
            <p className="text-xs text-gray-400 mb-4">
              This will clear all active positions, closed trade history, and reset virtual equity back to $40.00.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setIsResetConfirmOpen(false)}
                className="px-3 py-1.5 rounded bg-gray-800 text-gray-300 text-xs"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  onResetAccount(40.0);
                  setIsResetConfirmOpen(false);
                }}
                className="px-3 py-1.5 rounded bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold"
              >
                Confirm Reset to $40.00
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
