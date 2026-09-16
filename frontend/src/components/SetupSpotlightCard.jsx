import React from 'react';
import { 
  Crosshair, 
  ShieldAlert, 
  Target, 
  ArrowUpRight, 
  CheckCircle, 
  Sparkles, 
  Layers, 
  DollarSign, 
  TrendingUp 
} from 'lucide-react';

export default function SetupSpotlightCard({ 
  setup, 
  onExecutePaperTrade, 
  onOpenExplain, 
  account,
  opportunities = [],
  onSelectSymbol
}) {
  if (!setup) {
    return (
      <div className="terminal-card p-6 flex flex-col items-center justify-center text-center">
        <Crosshair className="w-8 h-8 text-gray-600 mb-2" />
        <div className="text-sm font-semibold text-gray-400">Select an asset from the scanner</div>
        <p className="text-xs text-gray-500 max-w-xs mt-1">
          Binocrypt will compute precise mathematical entry, stop loss, and multi-tier targets.
        </p>
      </div>
    );
  }

  // Calculate small account ($40) sizing details
  const equity = account ? account.current_equity : 40.0;
  const riskAmount = equity * 0.02; // 2% risk = $0.80 on $40
  const stopDistRatio = (setup.stop_distance_pct / 100.0) || 0.03;
  const rawPositionUsd = riskAmount / stopDistRatio;
  const recommendedUsd = Math.min(equity * 0.95, Math.max(5.0, rawPositionUsd));
  const estimatedTokens = (recommendedUsd / setup.current_price).toFixed(4);

  return (
    <div className="terminal-card p-5 relative overflow-hidden border-blue-900/40">
      {/* Background watermark */}
      <div className="absolute right-3 top-3 opacity-5 pointer-events-none">
        <Crosshair className="w-32 h-32 text-blue-400" />
      </div>

      {/* Quick Coin Switcher Bar */}
      {opportunities.length > 0 && (
        <div className="flex items-center gap-1.5 overflow-x-auto pb-2.5 mb-3 border-b border-[#1e2638]">
          <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider shrink-0 mr-1">Active Setups:</span>
          {opportunities.slice(0, 7).map((opp, idx) => {
            const isCurrent = setup.symbol === opp.symbol;
            return (
              <button
                key={opp.symbol}
                onClick={() => onSelectSymbol && onSelectSymbol(opp.symbol)}
                className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold transition shrink-0 flex items-center gap-1 ${
                  isCurrent
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30 ring-1 ring-blue-400'
                    : 'bg-[#0b0e14] text-gray-400 hover:text-gray-100 hover:bg-[#182030] border border-[#1e2638]'
                }`}
              >
                <span>#{idx + 1} {opp.symbol.replace('USDT', '')}</span>
                <span className={`text-[9px] px-1 py-0.2 rounded font-normal ${
                  isCurrent ? 'bg-blue-700 text-white' : 'bg-gray-800 text-gray-400'
                }`}>
                  {opp.total_score}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between gap-3 border-b border-[#1e2638] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold text-white tracking-wider">{setup.symbol}</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-950 text-emerald-400 border border-emerald-500/40">
              SPOT {setup.direction}
            </span>
            <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
              setup.signal_state === 'READY' ? 'bg-emerald-900/50 text-emerald-300 border border-emerald-500/50' :
              setup.signal_state === 'SETUP_FORMING' ? 'bg-blue-900/50 text-blue-300 border border-blue-500/50' :
              'bg-gray-800 text-gray-400'
            }`}>
              {setup.signal_state}
            </span>
          </div>
          <div className="text-xs text-gray-400 mt-1 flex items-center gap-2">
            <span>{setup.setup_name}</span>
            <span>•</span>
            <span className="text-emerald-400 font-mono font-medium">Confidence: {setup.confidence_pct}%</span>
          </div>
        </div>

        <div className="text-right">
          <div className="text-[10px] uppercase text-gray-400 font-semibold tracking-wider">SCORE</div>
          <div className="text-xl font-mono font-bold text-white leading-tight">
            {setup.score}<span className="text-xs text-gray-500 font-normal">/100</span>
          </div>
        </div>
      </div>

      {/* Levels Matrix */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 my-4">
        {/* Entry Zone */}
        <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
          <div className="flex items-center justify-between text-[10px] text-gray-400 font-mono">
            <span>ENTRY ZONE</span>
            <ArrowUpRight className="w-3 h-3 text-blue-400" />
          </div>
          <div className="font-mono-nums font-bold text-sm text-blue-400 mt-1">
            ${setup.current_price < 1 ? setup.entry_zone_min.toFixed(4) : setup.entry_zone_min.toFixed(2)}
          </div>
          <div className="text-[10px] text-gray-500 font-mono">
            to ${setup.current_price < 1 ? setup.entry_zone_max.toFixed(4) : setup.entry_zone_max.toFixed(2)}
          </div>
        </div>

        {/* Stop Loss */}
        <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
          <div className="flex items-center justify-between text-[10px] text-gray-400 font-mono">
            <span>STOP LOSS</span>
            <ShieldAlert className="w-3 h-3 text-rose-400" />
          </div>
          <div className="font-mono-nums font-bold text-sm text-rose-400 mt-1">
            ${setup.current_price < 1 ? setup.stop_loss.toFixed(4) : setup.stop_loss.toFixed(2)}
          </div>
          <div className="text-[10px] text-rose-500/80 font-mono">
            -{setup.stop_distance_pct}% from entry
          </div>
        </div>

        {/* Target 1 */}
        <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
          <div className="flex items-center justify-between text-[10px] text-gray-400 font-mono">
            <span>TARGET 1</span>
            <Target className="w-3 h-3 text-emerald-400" />
          </div>
          <div className="font-mono-nums font-bold text-sm text-emerald-400 mt-1">
            ${setup.current_price < 1 ? setup.tp1.toFixed(4) : setup.tp1.toFixed(2)}
          </div>
          <div className="text-[10px] text-emerald-500/80 font-mono">
            1 : {setup.risk_reward} R:R (Move SL to BE)
          </div>
        </div>

        {/* Target 2 */}
        <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
          <div className="flex items-center justify-between text-[10px] text-gray-400 font-mono">
            <span>TARGET 2</span>
            <Target className="w-3 h-3 text-cyan-400" />
          </div>
          <div className="font-mono-nums font-bold text-sm text-cyan-400 mt-1">
            ${setup.current_price < 1 ? setup.tp2.toFixed(4) : setup.tp2.toFixed(2)}
          </div>
          <div className="text-[10px] text-cyan-500/80 font-mono">
            1 : {(setup.risk_reward * 1.6).toFixed(1)} Runner
          </div>
        </div>
      </div>

      {/* Rationale Bullets */}
      <div className="space-y-1 text-xs text-gray-300 mb-4 bg-[#0e121a] p-3 rounded border border-[#1e2638]/80">
        <div className="text-[10px] uppercase font-mono font-semibold text-gray-400 mb-1.5 flex items-center gap-1.5">
          <Layers className="w-3 h-3 text-blue-400" />
          <span>Quantitative Confluence Rationale</span>
        </div>
        {setup.rationale_bullets.map((bullet, idx) => (
          <div key={idx} className="flex items-start gap-2">
            <span className="text-emerald-400 font-bold">•</span>
            <span className="leading-snug">{bullet}</span>
          </div>
        ))}
        <div className="text-[11px] text-rose-400/90 pt-1 border-t border-gray-800/80 mt-2 font-mono">
          <span className="font-semibold uppercase">Invalidation:</span> {setup.invalidation}
        </div>
      </div>

      {/* Small Account Risk Guidance Box */}
      <div className="bg-blue-950/20 border border-blue-800/40 p-3 rounded flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="text-[11px] font-semibold text-blue-300 flex items-center gap-1.5">
            <DollarSign className="w-3.5 h-3.5 text-blue-400" />
            <span>$40 Account Sizing (2.0% Risk = ${riskAmount.toFixed(2)})</span>
          </div>
          <p className="text-[11px] text-gray-400 mt-0.5">
            Recommended position: <strong className="text-white font-mono">${recommendedUsd.toFixed(2)} USDT</strong> (~{estimatedTokens} tokens)
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onOpenExplain(setup.symbol)}
            className="px-3 py-1.5 rounded bg-purple-900/40 hover:bg-purple-900/60 text-purple-300 border border-purple-700/50 text-xs font-semibold flex items-center gap-1.5 transition"
          >
            <Sparkles className="w-3.5 h-3.5 text-purple-400" />
            <span>AI Breakdown</span>
          </button>

          <button
            onClick={() => onExecutePaperTrade(setup)}
            className="px-4 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold shadow-lg shadow-emerald-600/30 flex items-center gap-1.5 transition"
          >
            <CheckCircle className="w-3.5 h-3.5" />
            <span>Place Paper Trade</span>
          </button>
        </div>
      </div>

    </div>
  );
}
