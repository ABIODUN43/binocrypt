import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import RegimeProbabilityGauge from './RegimeProbabilityGauge';
import BearExhaustionGauge from './BearExhaustionGauge';
import MultiHorizonForecastTable from './MultiHorizonForecastTable';
import DecisionIntelligenceCard from './DecisionIntelligenceCard';
import MarketHierarchyBar from './MarketHierarchyBar';
import { Sparkles, RefreshCw, Layers, Compass, TrendingUp, AlertCircle, PlayCircle } from 'lucide-react';

export default function AssetIntelligenceView({ 
  selectedSymbol, 
  onSelectCoin, 
  opportunities, 
  onExecuteTrade 
}) {
  const [intelligence, setIntelligence] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchIntelligence = async () => {
    if (!selectedSymbol) return;
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getAssetIntelligence(selectedSymbol);
      setIntelligence(data);
    } catch (err) {
      console.error("Intelligence fetch error:", err);
      setError(err.message || "Failed to load intelligence");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchIntelligence();
  }, [selectedSymbol]);

  const topCoins = (opportunities || []).slice(0, 8);

  return (
    <div className="space-y-4">
      {/* Quick Switcher Bar */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none font-mono text-xs">
        <span className="text-gray-400 text-[11px] uppercase tracking-wider font-semibold mr-1 shrink-0">
          Inspect:
        </span>
        {topCoins.map((opp) => {
          const isActive = opp.symbol === selectedSymbol;
          return (
            <button
              key={opp.symbol}
              onClick={() => onSelectCoin(opp.symbol)}
              className={`px-2.5 py-1 rounded-md transition shrink-0 flex items-center gap-1.5 border text-xs ${
                isActive
                  ? 'bg-cyan-950 text-cyan-300 border-cyan-700 font-bold shadow-sm'
                  : 'bg-gray-900/60 text-gray-400 border-gray-800 hover:text-gray-200 hover:bg-gray-800/80'
              }`}
            >
              <span>{opp.symbol.replace('USDT', '')}</span>
              <span className={`text-[10px] font-normal ${opp.change_24h >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {opp.change_24h >= 0 ? '+' : ''}{opp.change_24h.toFixed(1)}%
              </span>
            </button>
          );
        })}
      </div>

      {/* Asset Title Bar */}
      <div className="terminal-card flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-950/60 border border-cyan-800 text-cyan-400">
            <Sparkles className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-extrabold text-white tracking-wide">
                {selectedSymbol ? selectedSymbol.replace('USDT', '') : 'ARB'}
              </h1>
              <span className="text-xs text-gray-400 font-mono">/ USDT</span>
              {intelligence && (
                <span className="px-2.5 py-0.5 rounded text-[11px] font-mono font-bold uppercase tracking-wider bg-gray-900 border border-gray-700 text-gray-200">
                  {intelligence.regime_distribution.primary_regime}
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 text-xs font-mono mt-0.5">
              <span className="text-gray-200 font-bold">
                ${intelligence?.price < 1 ? intelligence?.price.toFixed(4) : intelligence?.price.toFixed(2)}
              </span>
              <span className={intelligence?.change_24h >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                {intelligence?.change_24h >= 0 ? '+' : ''}{intelligence?.change_24h?.toFixed(2)}% (24h)
              </span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2.5 font-mono text-xs">
          <button
            onClick={fetchIntelligence}
            disabled={isLoading}
            className="btn btn-secondary flex items-center gap-1.5 py-1.5 px-3"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Recalibrate Models</span>
          </button>

          {onExecuteTrade && (
            <button
              onClick={() => onExecuteTrade(selectedSymbol)}
              className="btn btn-primary flex items-center gap-1.5 py-1.5 px-3.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold"
            >
              <PlayCircle className="w-4 h-4" />
              <span>Paper Trade Setup</span>
            </button>
          )}
        </div>
      </div>

      {isLoading && !intelligence ? (
        <div className="terminal-card h-80 flex flex-col items-center justify-center font-mono text-xs text-gray-400 gap-2">
          <RefreshCw className="w-8 h-8 animate-spin text-cyan-400" />
          <span>Evaluating multi-timeframe quantitative features for {selectedSymbol}...</span>
        </div>
      ) : error ? (
        <div className="terminal-card p-6 flex flex-col items-center justify-center text-center font-mono text-xs text-rose-400 gap-2">
          <AlertCircle className="w-8 h-8 text-rose-500" />
          <span>{error}</span>
          <button onClick={fetchIntelligence} className="btn btn-secondary text-xs mt-2">Retry Analysis</button>
        </div>
      ) : intelligence ? (
        <div className="space-y-4">
          {/* Market Hierarchy Context */}
          <MarketHierarchyBar hierarchy={intelligence.hierarchy} symbol={selectedSymbol} />

          {/* Row 1: 7-State Regime Probabilities & Bear Exhaustion Gauge */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <RegimeProbabilityGauge 
              distribution={intelligence.regime_distribution} 
              symbol={selectedSymbol} 
            />
            <BearExhaustionGauge 
              exhaustion={intelligence.bear_exhaustion} 
            />
          </div>

          {/* Row 2: Decision Engine & Confirmation Triggers */}
          <DecisionIntelligenceCard intelligence={intelligence} />

          {/* Row 3: Multi-Horizon Forecast Scenarios */}
          <MultiHorizonForecastTable 
            forecasts={intelligence.multi_horizon_forecasts} 
            currentPrice={intelligence.price} 
          />
        </div>
      ) : null}
    </div>
  );
}
