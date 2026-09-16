import React from 'react';
import { Layers, ShieldAlert, Sparkles, Activity } from 'lucide-react';

const REGIME_CONFIG = {
  STRONG_BULL: { label: 'Strong Bull', color: 'bg-emerald-500', text: 'text-emerald-400', border: 'border-emerald-500/40' },
  BULL: { label: 'Bull', color: 'bg-green-500', text: 'text-green-400', border: 'border-green-500/40' },
  RECOVERY: { label: 'Recovery', color: 'bg-teal-400', text: 'text-teal-300', border: 'border-teal-400/40' },
  NEUTRAL: { label: 'Neutral', color: 'bg-gray-400', text: 'text-gray-300', border: 'border-gray-500/40' },
  DISTRIBUTION: { label: 'Distribution', color: 'bg-amber-500', text: 'text-amber-400', border: 'border-amber-500/40' },
  BEAR: { label: 'Bear', color: 'bg-rose-500', text: 'text-rose-400', border: 'border-rose-500/40' },
  STRONG_BEAR: { label: 'Strong Bear', color: 'bg-red-700', text: 'text-red-400', border: 'border-red-600/40' }
};

export default function RegimeProbabilityGauge({ distribution, symbol }) {
  if (!distribution) return null;

  const regimes = [
    { key: 'STRONG_BULL', prob: distribution.strong_bull },
    { key: 'BULL', prob: distribution.bull },
    { key: 'RECOVERY', prob: distribution.recovery },
    { key: 'NEUTRAL', prob: distribution.neutral },
    { key: 'DISTRIBUTION', prob: distribution.distribution },
    { key: 'BEAR', prob: distribution.bear },
    { key: 'STRONG_BEAR', prob: distribution.strong_bear }
  ];

  const primaryKey = distribution.primary_regime.toUpperCase();
  const primaryMeta = REGIME_CONFIG[primaryKey] || REGIME_CONFIG.NEUTRAL;

  return (
    <div className="terminal-card flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold text-gray-200 tracking-wider uppercase">
              7-State Regime Probability Distribution
            </h3>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-gray-400 uppercase font-mono">Primary:</span>
            <span className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold tracking-wider border ${primaryMeta.border} ${primaryMeta.text} bg-gray-900/80`}>
              {primaryMeta.label.toUpperCase()}
            </span>
          </div>
        </div>

        {/* Note on Probabilistic Modeling */}
        <div className="text-[11px] text-gray-400 mb-3 leading-relaxed">
          Calibrated ensemble across multi-timeframe trend alignment, momentum vector, and universe breadth.
        </div>

        {/* Probability Bars */}
        <div className="space-y-2">
          {regimes.map(({ key, prob }) => {
            const isPrimary = key === primaryKey;
            const meta = REGIME_CONFIG[key] || REGIME_CONFIG.NEUTRAL;
            return (
              <div 
                key={key} 
                className={`p-1.5 rounded transition ${isPrimary ? 'bg-gray-800/60 border border-gray-700/60' : 'hover:bg-gray-900/40'}`}
              >
                <div className="flex items-center justify-between text-xs mb-1">
                  <div className="flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-full ${meta.color}`} />
                    <span className={`font-mono text-[11px] ${isPrimary ? 'font-bold text-white' : 'text-gray-300'}`}>
                      {meta.label}
                    </span>
                    {isPrimary && (
                      <span className="text-[9px] px-1 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/60 font-mono">
                        DOMINANT
                      </span>
                    )}
                  </div>
                  <span className={`font-mono font-bold text-xs ${isPrimary ? meta.text : 'text-gray-400'}`}>
                    {prob.toFixed(1)}%
                  </span>
                </div>
                
                {/* Progress bar */}
                <div className="w-full h-1.5 bg-gray-950 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-500 ${meta.color}`} 
                    style={{ width: `${Math.min(100, Math.max(2, prob))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="pt-3 mt-3 border-t border-gray-800/80 flex items-center justify-between text-[10px] text-gray-400 font-mono">
        <span>Σ Probability: 100.0%</span>
        <span>Softmax Normalized</span>
      </div>
    </div>
  );
}
