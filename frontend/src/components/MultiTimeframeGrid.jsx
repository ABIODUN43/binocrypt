import React from 'react';
import { Layers, Activity, TrendingUp, Compass, Volume2, Shield } from 'lucide-react';

export default function MultiTimeframeGrid({ setup, score }) {
  if (!setup) return null;

  return (
    <div className="space-y-4">
      {/* 3-Timeframe Confluence Cards (4H, 1H, 15M) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        
        {/* 4H Macro Card */}
        <div className="terminal-card p-3.5 bg-[#0e121a]">
          <div className="flex items-center justify-between border-b border-[#1e2638] pb-2 mb-2.5">
            <span className="text-xs font-mono font-bold text-blue-400">4H TIMEFRAME</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-blue-950 text-blue-300">Macro Structure</span>
          </div>
          <div className="space-y-1.5 text-xs text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-400">Macro Trend:</span>
              <span className="font-semibold text-emerald-400">BULLISH EXPANSION</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">EMA Alignment:</span>
              <span className="font-mono text-gray-200">20 &gt; 50 &gt; 200</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Key Support:</span>
              <span className="font-mono font-semibold text-gray-200">${setup.stop_loss}</span>
            </div>
          </div>
        </div>

        {/* 1H Momentum & Continuation Card */}
        <div className="terminal-card p-3.5 bg-[#0e121a]">
          <div className="flex items-center justify-between border-b border-[#1e2638] pb-2 mb-2.5">
            <span className="text-xs font-mono font-bold text-purple-400">1H TIMEFRAME</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-purple-950 text-purple-300">Momentum Engine</span>
          </div>
          <div className="space-y-1.5 text-xs text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-400">Setup Pattern:</span>
              <span className="font-semibold text-purple-300">{setup.setup_name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Structure Type:</span>
              <span className="font-mono text-emerald-400">Higher Highs &amp; Lows</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">R:R Asymmetry:</span>
              <span className="font-mono font-bold text-emerald-400">1 : {setup.risk_reward}</span>
            </div>
          </div>
        </div>

        {/* 15M Entry Timing Card */}
        <div className="terminal-card p-3.5 bg-[#0e121a]">
          <div className="flex items-center justify-between border-b border-[#1e2638] pb-2 mb-2.5">
            <span className="text-xs font-mono font-bold text-emerald-400">15M TIMEFRAME</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300">Execution Timing</span>
          </div>
          <div className="space-y-1.5 text-xs text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-400">Entry Trigger:</span>
              <span className="font-semibold text-white">${setup.entry_zone_min} - ${setup.entry_zone_max}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max Invalidation:</span>
              <span className="font-mono text-rose-400">-{setup.stop_distance_pct}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Signal State:</span>
              <span className="font-mono font-bold text-emerald-400">{setup.signal_state}</span>
            </div>
          </div>
        </div>

      </div>

      {/* Quantitative Indicator Radar Matrix */}
      {score && (
        <div className="terminal-card p-4">
          <div className="text-xs font-mono font-bold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-blue-400" />
            <span>Opportunity Score Breakdown (Total: {score.total_score}/100)</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            
            <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
              <div className="flex justify-between text-[11px] text-gray-400">
                <span>Trend Stacking</span>
                <span className="font-mono font-bold text-white">{score.trend_score}/20</span>
              </div>
              <div className="w-full bg-gray-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-blue-500 h-full" style={{ width: `${(score.trend_score/20)*100}%` }} />
              </div>
            </div>

            <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
              <div className="flex justify-between text-[11px] text-gray-400">
                <span>Momentum (RSI/MACD)</span>
                <span className="font-mono font-bold text-white">{score.momentum_score}/20</span>
              </div>
              <div className="w-full bg-gray-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-purple-500 h-full" style={{ width: `${(score.momentum_score/20)*100}%` }} />
              </div>
            </div>

            <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
              <div className="flex justify-between text-[11px] text-gray-400">
                <span>Volume &amp; RVOL</span>
                <span className="font-mono font-bold text-white">{score.volume_score}/20</span>
              </div>
              <div className="w-full bg-gray-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-emerald-500 h-full" style={{ width: `${(score.volume_score/20)*100}%` }} />
              </div>
            </div>

            <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
              <div className="flex justify-between text-[11px] text-gray-400">
                <span>Market Structure</span>
                <span className="font-mono font-bold text-white">{score.structure_score}/20</span>
              </div>
              <div className="w-full bg-gray-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-cyan-500 h-full" style={{ width: `${(score.structure_score/20)*100}%` }} />
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
