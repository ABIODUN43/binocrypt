import React from 'react';
import { Globe, ArrowRight, ShieldCheck, TrendingUp, AlertTriangle } from 'lucide-react';

export default function MarketHierarchyBar({ hierarchy, symbol }) {
  if (!hierarchy) return null;

  let alignColor = 'text-gray-300 bg-gray-800 border-gray-700';
  if (hierarchy.asset_alignment === 'STRONG_MACRO_TAILWIND') {
    alignColor = 'text-emerald-300 bg-emerald-950/60 border-emerald-800';
  } else if (hierarchy.asset_alignment === 'SYSTEMIC_MACRO_HEADWIND') {
    alignColor = 'text-rose-300 bg-rose-950/60 border-rose-800';
  } else if (hierarchy.asset_alignment === 'DECOUPLED_COUNTER_TREND') {
    alignColor = 'text-amber-300 bg-amber-950/60 border-amber-800';
  }

  return (
    <div className="p-3 rounded-xl bg-gray-900/40 border border-gray-800/80 mb-4 font-mono">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-2.5">
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <Globe className="w-4 h-4 text-cyan-400 shrink-0" />
          <span className="font-bold text-gray-200 uppercase tracking-wider text-[11px]">Macro Hierarchy:</span>
        </div>

        {/* Hierarchy Chain */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          {/* Global Market */}
          <span className="px-2 py-0.5 rounded bg-gray-950 border border-gray-800 text-gray-300 text-[11px]">
            Global: <strong className="text-white">{hierarchy.global_regime}</strong>
          </span>

          <ArrowRight className="w-3 h-3 text-gray-600 shrink-0" />

          {/* BTC */}
          <span className="px-2 py-0.5 rounded bg-gray-950 border border-gray-800 text-gray-300 text-[11px]">
            BTC: <strong className={hierarchy.btc_regime.includes('BULL') ? 'text-emerald-400' : 'text-rose-400'}>{hierarchy.btc_regime}</strong>
          </span>

          <ArrowRight className="w-3 h-3 text-gray-600 shrink-0" />

          {/* ETH */}
          <span className="px-2 py-0.5 rounded bg-gray-950 border border-gray-800 text-gray-300 text-[11px]">
            ETH: <strong className={hierarchy.eth_regime.includes('BULL') ? 'text-emerald-400' : 'text-cyan-400'}>{hierarchy.eth_regime}</strong>
          </span>

          <ArrowRight className="w-3 h-3 text-gray-600 shrink-0" />

          {/* Sector */}
          <span className="px-2 py-0.5 rounded bg-gray-950 border border-gray-800 text-gray-300 text-[11px]">
            Sector: <strong className="text-purple-300">{hierarchy.sector_strength}</strong>
          </span>

          <ArrowRight className="w-3 h-3 text-gray-600 shrink-0" />

          {/* Asset Alignment */}
          <span className={`px-2.5 py-0.5 rounded border text-[11px] font-bold ${alignColor}`}>
            {symbol ? symbol.replace('USDT', '') : 'Asset'}: {hierarchy.asset_alignment.replace(/_/g, ' ')}
          </span>
        </div>
      </div>
    </div>
  );
}
