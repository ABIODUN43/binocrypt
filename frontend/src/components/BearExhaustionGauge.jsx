import React from 'react';
import { Compass, TrendingUp, AlertTriangle, CheckCircle2, ShieldAlert, Zap } from 'lucide-react';

export default function BearExhaustionGauge({ exhaustion }) {
  if (!exhaustion) return null;

  const score = exhaustion.score;

  // Color & Badge based on specifications
  let statusColor = 'text-rose-400 border-rose-800 bg-rose-950/40';
  let barColor = 'bg-rose-500';
  if (score >= 75) {
    statusColor = 'text-emerald-300 border-emerald-700 bg-emerald-950/60';
    barColor = 'bg-emerald-400';
  } else if (score >= 60) {
    statusColor = 'text-teal-300 border-teal-700 bg-teal-950/60';
    barColor = 'bg-teal-400';
  } else if (score >= 40) {
    statusColor = 'text-amber-300 border-amber-800 bg-amber-950/40';
    barColor = 'bg-amber-400';
  }

  const subFactors = [
    { label: 'Trend Deterioration', score: exhaustion.trend_deterioration, desc: 'Diminishing lower-low magnitude & higher-low structure' },
    { label: 'Momentum Recovery', score: exhaustion.momentum_recovery, desc: 'Positive RSI rebound & MACD histogram expansion' },
    { label: 'Volume Accumulation', score: exhaustion.volume_accumulation, desc: 'Selling volume exhaustion & green accumulation surges' },
    { label: 'Market Context', score: exhaustion.market_context, desc: 'Universe breadth above 50 EMA & altcoin health' },
    { label: 'Volatility Transition', score: exhaustion.volatility_transition, desc: 'ATR compression & Bollinger Band pinch stabilization' },
    { label: 'BTC Confirmation', score: exhaustion.btc_confirmation, desc: 'Macro benchmark structural trend alignment' },
  ];

  return (
    <div className="terminal-card flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4 text-emerald-400" />
            <h3 className="text-xs font-bold text-gray-200 tracking-wider uppercase">
              Bear Exhaustion & Bottoming Engine
            </h3>
          </div>
          <span className="text-[10px] text-gray-400 font-mono">
            0 to 100 Scale
          </span>
        </div>

        {/* Big Score Display */}
        <div className="flex items-center justify-between p-3 rounded-xl bg-gray-950/80 border border-gray-800/80 mb-3">
          <div>
            <div className="text-[10px] text-gray-400 uppercase tracking-wider font-mono">
              Composite Exhaustion Score
            </div>
            <div className="flex items-baseline gap-1.5 mt-0.5">
              <span className={`text-3xl font-extrabold font-mono tracking-tight ${score >= 60 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {score.toFixed(0)}
              </span>
              <span className="text-xs text-gray-400 font-mono">/ 100</span>
            </div>
          </div>

          <div className="text-right">
            <div className="text-[10px] text-gray-400 uppercase tracking-wider font-mono mb-1">
              Exhaustion State
            </div>
            <span className={`inline-block px-2.5 py-1 rounded-md text-[11px] font-mono font-bold tracking-wider border ${statusColor}`}>
              {exhaustion.status}
            </span>
          </div>
        </div>

        {/* Master Progress Bar */}
        <div className="w-full h-2.5 bg-gray-950 rounded-full overflow-hidden p-0.5 border border-gray-800 mb-4">
          <div 
            className={`h-full rounded-full transition-all duration-700 ${barColor}`} 
            style={{ width: `${Math.min(100, Math.max(3, score))}%` }}
          />
        </div>

        {/* 6 Subcomponents Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 mb-3">
          {subFactors.map((f, idx) => (
            <div key={idx} className="p-2 rounded-lg bg-gray-900/50 border border-gray-800/60">
              <div className="flex items-center justify-between text-xs mb-1 font-mono">
                <span className="text-gray-300 font-medium">{f.label}</span>
                <span className={`font-bold ${f.score >= 70 ? 'text-emerald-400' : f.score >= 50 ? 'text-teal-300' : 'text-gray-400'}`}>
                  {f.score.toFixed(0)}
                </span>
              </div>
              <div className="w-full h-1 bg-gray-950 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full ${f.score >= 70 ? 'bg-emerald-400' : f.score >= 50 ? 'bg-teal-500' : 'bg-gray-600'}`}
                  style={{ width: `${f.score}%` }}
                />
              </div>
              <div className="text-[9px] text-gray-400 mt-1 truncate">
                {f.desc}
              </div>
            </div>
          ))}
        </div>

        {/* Model Structural Reasons */}
        {exhaustion.reasons && exhaustion.reasons.length > 0 && (
          <div className="space-y-1.5 pt-2 border-t border-gray-800/80">
            <div className="text-[10px] text-gray-400 uppercase font-mono tracking-wider">
              Observed Transition Signatures
            </div>
            {exhaustion.reasons.map((r, idx) => (
              <div key={idx} className="flex items-start gap-1.5 text-xs text-emerald-300/90 leading-tight">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                <span>{r}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="pt-2.5 mt-2 border-t border-gray-800/80 flex items-center justify-between text-[10px] text-gray-400 font-mono">
        <span>Framework: Cyclical Exhaustion Model</span>
        <span>Anti-Lag Multi-Factor</span>
      </div>
    </div>
  );
}
