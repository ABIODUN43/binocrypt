import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, Compass, CheckCircle2, XCircle } from 'lucide-react';

export default function MarketRegimeBanner({ regime }) {
  if (!regime) {
    return (
      <div className="terminal-card p-4 animate-pulse">
        <div className="h-6 bg-gray-800 rounded w-1/4 mb-2"></div>
        <div className="h-4 bg-gray-800 rounded w-3/4"></div>
      </div>
    );
  }

  const getPermissionBadge = () => {
    switch (regime.trade_permission) {
      case 'AGGRESSIVE_LONGS':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-950/80 border border-emerald-500/50 text-emerald-400 text-xs font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5" />
            AGGRESSIVE LONGS PERMITTED
          </span>
        );
      case 'SELECTIVE_LONGS':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-blue-950/80 border border-blue-500/50 text-blue-400 text-xs font-semibold">
            <Compass className="w-3.5 h-3.5" />
            SELECTIVE SUPPORT PLAYS ONLY
          </span>
        );
      case 'DEFENSIVE_CASH':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-950/80 border border-amber-500/50 text-amber-400 text-xs font-semibold">
            <AlertTriangle className="w-3.5 h-3.5" />
            DEFENSIVE CASH / TIGHT STOPS
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-950/80 border border-rose-500/50 text-rose-400 text-xs font-semibold">
            <XCircle className="w-3.5 h-3.5" />
            NO NEW TRADES / CASH DEFENSE
          </span>
        );
    }
  };

  return (
    <div className="terminal-card p-4 sm:p-5 relative overflow-hidden">
      {/* Glow highlight */}
      <div className={`absolute top-0 left-0 right-0 h-1 ${
        regime.regime === 'BULL_TREND' ? 'bg-emerald-500' :
        regime.regime === 'BEAR_TREND' ? 'bg-rose-500' :
        regime.regime === 'PANIC' ? 'bg-rose-600' : 'bg-blue-500'
      }`} />

      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Left: Regime Description */}
        <div className="space-y-1.5 max-w-2xl">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-medium text-gray-400 uppercase tracking-wider">MARKET REGIME:</span>
              <span className="text-base font-bold text-white tracking-wide">
                {regime.regime.replace('_', ' ')}
              </span>
            </div>
            <div className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-[#1e2638] text-gray-300">
              Confidence: {regime.confidence}%
            </div>
            {getPermissionBadge()}
          </div>
          <p className="text-xs text-gray-300 leading-relaxed">{regime.summary}</p>
        </div>

        {/* Right: Quant Trend Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono-nums shrink-0">
          
          <div className="bg-[#0b0e14] border border-[#1e2638] p-2 rounded">
            <div className="text-[10px] text-gray-400 font-sans uppercase">BTC 4H / 1H</div>
            <div className="font-semibold text-gray-200 mt-0.5">
              <span className={regime.btc_trend_4h.includes('BULLISH') ? 'text-emerald-400' : 'text-rose-400'}>
                {regime.btc_trend_4h.replace('STRONG_', '')}
              </span>
              <span className="text-gray-500 mx-1">/</span>
              <span className={regime.btc_trend_1h.includes('BULLISH') ? 'text-emerald-400' : 'text-rose-400'}>
                {regime.btc_trend_1h.replace('STRONG_', '')}
              </span>
            </div>
          </div>

          <div className="bg-[#0b0e14] border border-[#1e2638] p-2 rounded">
            <div className="text-[10px] text-gray-400 font-sans uppercase">ETH 4H / 1H</div>
            <div className="font-semibold text-gray-200 mt-0.5">
              <span className={regime.eth_trend_4h.includes('BULLISH') ? 'text-emerald-400' : 'text-rose-400'}>
                {regime.eth_trend_4h.replace('STRONG_', '')}
              </span>
              <span className="text-gray-500 mx-1">/</span>
              <span className={regime.eth_trend_1h.includes('BULLISH') ? 'text-emerald-400' : 'text-rose-400'}>
                {regime.eth_trend_1h.replace('STRONG_', '')}
              </span>
            </div>
          </div>

          <div className="bg-[#0b0e14] border border-[#1e2638] p-2 rounded">
            <div className="text-[10px] text-gray-400 font-sans uppercase">BREADTH (&gt;50 EMA)</div>
            <div className="flex items-center gap-1.5 mt-0.5 font-semibold text-gray-200">
              <span className={regime.market_breadth_pct >= 55 ? 'text-emerald-400' : 'text-amber-400'}>
                {regime.market_breadth_pct}%
              </span>
              <div className="w-10 bg-gray-800 h-1.5 rounded-full overflow-hidden">
                <div 
                  className={`h-full ${regime.market_breadth_pct >= 50 ? 'bg-emerald-500' : 'bg-amber-500'}`} 
                  style={{ width: `${Math.min(100, regime.market_breadth_pct)}%` }} 
                />
              </div>
            </div>
          </div>

          <div className="bg-[#0b0e14] border border-[#1e2638] p-2 rounded">
            <div className="text-[10px] text-gray-400 font-sans uppercase">VOLATILITY</div>
            <div className={`font-semibold mt-0.5 ${
              regime.volatility_state === 'EXTREME' ? 'text-rose-400' :
              regime.volatility_state === 'HIGH' ? 'text-amber-400' : 'text-emerald-400'
            }`}>
              {regime.volatility_state}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
