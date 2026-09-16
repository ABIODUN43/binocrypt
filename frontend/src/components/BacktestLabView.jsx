import React, { useState } from 'react';
import { api } from '../services/api';
import { Play, FlaskConical, TrendingUp, AlertCircle, BarChart3, RefreshCw } from 'lucide-react';

export default function BacktestLabView() {
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');
  const [limitBars, setLimitBars] = useState(250);
  const [riskPct, setRiskPct] = useState(2.0);
  const [targetRR, setTargetRR] = useState(2.0);
  
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleRunBacktest = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.runBacktest({
        symbol: symbol.toUpperCase(),
        timeframe: timeframe,
        limit_bars: parseInt(limitBars),
        risk_pct: parseFloat(riskPct),
        rr_target: parseFloat(targetRR)
      });
      setResult(res);
    } catch (err) {
      setError(err.message || 'Backtest failed to execute');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      
      {/* Parameters Form */}
      <div className="terminal-card p-5">
        <div className="flex items-center justify-between border-b border-[#1e2638] pb-3 mb-4">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-blue-400" />
              <span>Quantitative Strategy Backtesting Engine</span>
            </h3>
            <p className="text-xs text-gray-400">
              Evaluates spot strategy against real Binance historical candles with 0.1% fee &amp; 0.05% slippage
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs">
          
          <div>
            <label className="block text-gray-400 font-mono mb-1">PAIR</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="w-full bg-[#0b0e14] border border-[#1e2638] rounded p-2 text-white font-mono font-bold focus:outline-none focus:border-blue-500"
            />
          </div>

          <div>
            <label className="block text-gray-400 font-mono mb-1">TIMEFRAME</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="w-full bg-[#0b0e14] border border-[#1e2638] rounded p-2 text-white font-mono font-semibold focus:outline-none focus:border-blue-500"
            >
              <option value="15m">15M (Entry Timing)</option>
              <option value="1h">1H (Momentum)</option>
              <option value="4h">4H (Macro Trend)</option>
            </select>
          </div>

          <div>
            <label className="block text-gray-400 font-mono mb-1">CANDLE LOOKBACK</label>
            <select
              value={limitBars}
              onChange={(e) => setLimitBars(e.target.value)}
              className="w-full bg-[#0b0e14] border border-[#1e2638] rounded p-2 text-white font-mono font-semibold focus:outline-none focus:border-blue-500"
            >
              <option value="100">100 Candles (~4 days 1H)</option>
              <option value="250">250 Candles (~10 days 1H)</option>
              <option value="500">500 Candles (~20 days 1H)</option>
            </select>
          </div>

          <div>
            <label className="block text-gray-400 font-mono mb-1">RISK / TRADE (%)</label>
            <input
              type="number"
              step="0.5"
              value={riskPct}
              onChange={(e) => setRiskPct(e.target.value)}
              className="w-full bg-[#0b0e14] border border-[#1e2638] rounded p-2 text-white font-mono font-semibold focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={handleRunBacktest}
              disabled={isLoading}
              className="w-full py-2 rounded bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs shadow-lg shadow-blue-600/30 flex items-center justify-center gap-2 transition disabled:opacity-50"
            >
              {isLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              <span>{isLoading ? 'Simulating...' : 'Run Simulation'}</span>
            </button>
          </div>

        </div>

        {error && (
          <div className="mt-3 p-3 bg-rose-950/40 border border-rose-800/50 rounded flex items-center gap-2 text-xs text-rose-300">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Results Section */}
      {result && (
        <div className="space-y-6">
          
          {/* KPI Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
            
            <div className="terminal-card p-3">
              <div className="text-[10px] uppercase font-mono text-gray-400">Final Equity</div>
              <div className="text-base font-bold font-mono-nums text-white mt-1">
                ${result.final_equity.toFixed(2)}
              </div>
              <div className="text-[10px] text-gray-500 font-mono-nums">Start: ${result.starting_capital.toFixed(2)}</div>
            </div>

            <div className="terminal-card p-3">
              <div className="text-[10px] uppercase font-mono text-gray-400">Net Return</div>
              <div className={`text-base font-bold font-mono-nums mt-1 ${result.net_profit_usd >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {result.net_profit_usd >= 0 ? '+' : ''}${result.net_profit_usd.toFixed(2)}
              </div>
              <div className={`text-[10px] font-mono-nums ${result.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {result.total_return_pct >= 0 ? '+' : ''}{result.total_return_pct.toFixed(2)}%
              </div>
            </div>

            <div className="terminal-card p-3">
              <div className="text-[10px] uppercase font-mono text-gray-400">Win Rate</div>
              <div className="text-base font-bold font-mono-nums text-emerald-400 mt-1">
                {result.win_rate_pct.toFixed(1)}%
              </div>
              <div className="text-[10px] text-gray-500 font-mono-nums">
                {result.winning_trades}W / {result.losing_trades}L
              </div>
            </div>

            <div className="terminal-card p-3">
              <div className="text-[10px] uppercase font-mono text-gray-400">Profit Factor</div>
              <div className="text-base font-bold font-mono-nums text-cyan-400 mt-1">
                {result.profit_factor.toFixed(2)}
              </div>
              <div className="text-[10px] text-gray-500 font-mono-nums">Gross W / L</div>
            </div>

            <div className="terminal-card p-3">
              <div className="text-[10px] uppercase font-mono text-gray-400">Max Drawdown</div>
              <div className="text-base font-bold font-mono-nums text-rose-400 mt-1">
                -{result.max_drawdown_pct.toFixed(1)}%
              </div>
              <div className="text-[10px] text-gray-500 font-mono-nums">Peak-to-Trough</div>
            </div>

            <div className="terminal-card p-3">
              <div className="text-[10px] uppercase font-mono text-gray-400">Sharpe Ratio</div>
              <div className="text-base font-bold font-mono-nums text-white mt-1">
                {result.sharpe_ratio.toFixed(2)}
              </div>
              <div className="text-[10px] text-gray-500 font-mono-nums">Risk-adjusted</div>
            </div>

            <div className="terminal-card p-3">
              <div className="text-[10px] uppercase font-mono text-gray-400">Total Trades</div>
              <div className="text-base font-bold font-mono-nums text-white mt-1">
                {result.total_trades}
              </div>
              <div className="text-[10px] text-gray-500 font-mono-nums">{result.total_bars} bars</div>
            </div>

          </div>

          {/* Equity Curve Visualization */}
          {result.trades.length > 0 && (
            <div className="terminal-card p-4">
              <div className="text-xs font-mono font-bold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <BarChart3 className="w-4 h-4 text-blue-400" />
                <span>Simulated Equity Curve ($40 Starting)</span>
              </div>
              
              <div className="h-44 w-full">
                <svg viewBox="0 0 700 150" className="w-full h-full">
                  {(() => {
                    const equities = [result.starting_capital, ...result.trades.map(t => t.cumulative_equity)];
                    const minE = Math.min(...equities) * 0.95;
                    const maxE = Math.max(...equities) * 1.05;
                    const range = maxE - minE || 1;
                    
                    const points = equities.map((eq, i) => {
                      const x = (i / (equities.length - 1)) * 680 + 10;
                      const y = 140 - ((eq - minE) / range) * 130;
                      return `${x},${y}`;
                    }).join(' ');

                    const isProfitable = result.final_equity >= result.starting_capital;
                    const strokeColor = isProfitable ? "#10b981" : "#ef4444";

                    return (
                      <g>
                        {/* Baseline */}
                        <line
                          x1="10"
                          y1={140 - ((result.starting_capital - minE) / range) * 130}
                          x2="690"
                          y2={140 - ((result.starting_capital - minE) / range) * 130}
                          stroke="#334155"
                          strokeDasharray="4,4"
                        />
                        {/* Curve */}
                        <polyline
                          fill="none"
                          stroke={strokeColor}
                          strokeWidth="2.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          points={points}
                        />
                      </g>
                    );
                  })()}
                </svg>
              </div>
            </div>
          )}

          {/* Trade Log Table */}
          <div className="terminal-card overflow-hidden">
            <div className="p-4 border-b border-[#1e2638]">
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                Simulated Trade Ledger ({result.trades.length} Fills)
              </h4>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#1e2638] bg-[#0e121a] text-[10px] font-mono text-gray-400 uppercase">
                    <th className="py-2 px-3">Entry Time</th>
                    <th className="py-2 px-3">Exit Time</th>
                    <th className="py-2 px-3">Entry</th>
                    <th className="py-2 px-3">Exit</th>
                    <th className="py-2 px-3">Reason</th>
                    <th className="py-2 px-3">Fees</th>
                    <th className="py-2 px-3">Net P/L ($)</th>
                    <th className="py-2 px-3 text-right">Cumulative Equity</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1e2638]/60 font-mono-nums">
                  {result.trades.map((t, idx) => (
                    <tr key={idx} className="hover:bg-[#182030]/60 transition">
                      <td className="py-2 px-3 text-gray-400 text-[11px] font-sans">{t.entry_time}</td>
                      <td className="py-2 px-3 text-gray-400 text-[11px] font-sans">{t.exit_time}</td>
                      <td className="py-2 px-3 text-gray-200">${t.entry_price}</td>
                      <td className="py-2 px-3 text-gray-200">${t.exit_price}</td>
                      <td className="py-2 px-3">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          t.exit_reason === 'TAKE_PROFIT'
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-800/40'
                            : 'bg-rose-950 text-rose-400 border border-rose-800/40'
                        }`}>
                          {t.exit_reason}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-gray-500">-${t.fee_usd.toFixed(3)}</td>
                      <td className={`py-2 px-3 font-semibold ${t.net_pnl_usd >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {t.net_pnl_usd >= 0 ? '+' : ''}${t.net_pnl_usd.toFixed(2)} ({t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct.toFixed(1)}%)
                      </td>
                      <td className="py-2 px-3 text-right font-bold text-white">
                        ${t.cumulative_equity.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}

    </div>
  );
}
