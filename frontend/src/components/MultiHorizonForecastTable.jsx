import React from 'react';
import { Calendar, TrendingUp, TrendingDown, Minus, Info, ShieldCheck } from 'lucide-react';

export default function MultiHorizonForecastTable({ forecasts, currentPrice }) {
  if (!forecasts || forecasts.length === 0) return null;

  return (
    <div className="terminal-card">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 mb-3 border-b border-gray-800 gap-2">
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-blue-400" />
          <h3 className="text-xs font-bold text-gray-200 tracking-wider uppercase">
            Multi-Horizon Probabilistic Forecast Matrix
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-blue-950/60 text-blue-300 border border-blue-800/60">
            Regime Transition Drifts
          </span>
          <span className="text-[10px] text-gray-400 font-mono">
            Uncertainty Bounded
          </span>
        </div>
      </div>

      <div className="text-[11px] text-gray-400 mb-3 leading-relaxed">
        Scenario projections evaluate volatility expansion, regime transition matrices, and multi-timeframe directional bias. 
        <span className="text-gray-400 italic"> Scenarios represent estimated risk bounds, not guaranteed price targets.</span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="border-b border-gray-800 text-gray-400 text-[11px] uppercase tracking-wider bg-gray-950/40">
              <th className="py-2.5 px-3">Horizon</th>
              <th className="py-2.5 px-3">Bias</th>
              <th className="py-2.5 px-3">Bull / Bear Prob</th>
              <th className="py-2.5 px-3">Expected Return Range</th>
              <th className="py-2.5 px-3">Bear Case</th>
              <th className="py-2.5 px-3 text-white">Base Case</th>
              <th className="py-2.5 px-3">Bull Case</th>
              <th className="py-2.5 px-3 text-right">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60">
            {forecasts.map((f) => {
              const isBull = f.direction === 'BULLISH';
              const isBear = f.direction === 'BEARISH';
              const directionMeta = isBull
                ? { text: 'text-emerald-400', bg: 'bg-emerald-950/60 border-emerald-800', icon: TrendingUp }
                : isBear
                ? { text: 'text-rose-400', bg: 'bg-rose-950/60 border-rose-800', icon: TrendingDown }
                : { text: 'text-gray-300', bg: 'bg-gray-900 border-gray-700', icon: Minus };

              const Icon = directionMeta.icon;

              return (
                <tr key={f.horizon} className="hover:bg-gray-800/30 transition">
                  {/* Horizon Badge */}
                  <td className="py-3 px-3 font-bold text-white text-xs">
                    <span className="px-2 py-0.5 rounded bg-gray-800 text-gray-200 border border-gray-700 font-mono">
                      {f.horizon}
                    </span>
                  </td>

                  {/* Direction Bias */}
                  <td className="py-3 px-3">
                    <div className="flex items-center gap-1.5">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold border flex items-center gap-1 ${directionMeta.bg} ${directionMeta.text}`}>
                        <Icon className="w-3 h-3" />
                        {f.direction}
                      </span>
                    </div>
                  </td>

                  {/* Probabilities */}
                  <td className="py-3 px-3">
                    <div className="flex items-center gap-2">
                      <span className="text-emerald-400 font-semibold">{f.bull_prob.toFixed(0)}%</span>
                      <span className="text-gray-400">/</span>
                      <span className="text-rose-400 font-semibold">{f.bear_prob.toFixed(0)}%</span>
                    </div>
                  </td>

                  {/* Return Range */}
                  <td className="py-3 px-3">
                    <span className="font-semibold text-gray-200">
                      {f.expected_return_min >= 0 ? `+${f.expected_return_min}%` : `${f.expected_return_min}%`} to {f.expected_return_max >= 0 ? `+${f.expected_return_max}%` : `${f.expected_return_max}%`}
                    </span>
                  </td>

                  {/* Bear Case Price */}
                  <td className="py-3 px-3 text-rose-400/90 font-mono">
                    ${f.bear_case_price < 1 ? f.bear_case_price.toFixed(4) : f.bear_case_price.toFixed(2)}
                  </td>

                  {/* Base Case Price */}
                  <td className="py-3 px-3 text-cyan-300 font-bold font-mono">
                    ${f.base_case_price < 1 ? f.base_case_price.toFixed(4) : f.base_case_price.toFixed(2)}
                  </td>

                  {/* Bull Case Price */}
                  <td className="py-3 px-3 text-emerald-400/90 font-mono">
                    ${f.bull_case_price < 1 ? f.bull_case_price.toFixed(4) : f.bull_case_price.toFixed(2)}
                  </td>

                  {/* Confidence */}
                  <td className="py-3 px-3 text-right">
                    <div className="flex items-center justify-end gap-1 text-gray-300">
                      <span>{f.confidence_pct.toFixed(0)}%</span>
                      <span className="text-[10px] text-gray-400 font-mono">(±{f.uncertainty_pct.toFixed(0)}%)</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Footer Info */}
      <div className="pt-3 mt-3 border-t border-gray-800/80 flex flex-col sm:flex-row sm:items-center justify-between text-[10px] text-gray-400 font-mono gap-1">
        <div className="flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
          <span>Multi-Horizon Monte Carlo & Volatility Interval Calibration</span>
        </div>
        <span>Model Version: RegimeEnsemble v2.4</span>
      </div>
    </div>
  );
}
