import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Compass, RefreshCw, ArrowRight, ShieldAlert, Sparkles, TrendingUp, Filter } from 'lucide-react';

export default function RecoveryScannerView({ onSelectCoin, onNavigateToSetup }) {
  const [opportunities, setOpportunities] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastRefreshed, setLastRefreshed] = useState(null);

  const fetchRecovery = async () => {
    setIsLoading(true);
    try {
      const data = await api.getTopRecoveryOpportunities(15);
      setOpportunities(data);
      setLastRefreshed(new Date().toLocaleTimeString());
    } catch (err) {
      console.error("Failed to load recovery opportunities:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchRecovery();
  }, []);

  return (
    <div className="space-y-4">
      {/* Header Banner */}
      <div className="terminal-card flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Compass className="w-5 h-5 text-emerald-400" />
            <h2 className="text-base font-bold text-white tracking-wide">
              Bear-Market Exhaustion & Recovery Scanner
            </h2>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
              Institutional Screener
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Ranks cryptocurrencies approaching bear-trend exhaustion, selling volume dry-up, and high-expectancy accumulation transitions.
          </p>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          {lastRefreshed && (
            <span className="text-[11px] text-gray-400">Scanned: {lastRefreshed}</span>
          )}
          <button
            onClick={fetchRecovery}
            disabled={isLoading}
            className="btn btn-secondary flex items-center gap-1.5 text-xs py-1.5 px-3"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Rescan Universe</span>
          </button>
        </div>
      </div>

      {/* Opportunities Table */}
      <div className="terminal-card overflow-hidden">
        {isLoading && opportunities.length === 0 ? (
          <div className="h-64 flex flex-col items-center justify-center gap-2 text-gray-400 font-mono text-xs">
            <RefreshCw className="w-6 h-6 animate-spin text-cyan-400" />
            <span>Scanning universe across 6 bear-exhaustion factors...</span>
          </div>
        ) : opportunities.length === 0 ? (
          <div className="h-48 flex items-center justify-center text-gray-400 font-mono text-xs">
            No active bear exhaustion setups detected matching threshold criteria.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 text-[11px] uppercase tracking-wider bg-gray-950/60">
                  <th className="py-3 px-3 text-center">Rank</th>
                  <th className="py-3 px-4">Asset</th>
                  <th className="py-3 px-4">Price</th>
                  <th className="py-3 px-4">Exhaustion Score</th>
                  <th className="py-3 px-4">Recovery Prob</th>
                  <th className="py-3 px-4">Decision</th>
                  <th className="py-3 px-4">Est. Transition Window</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60">
                {opportunities.map((opp, idx) => {
                  const exScore = opp.bear_exhaustion.score;
                  const recProb = opp.regime_distribution.recovery;

                  return (
                    <tr 
                      key={opp.symbol}
                      onClick={() => onSelectCoin(opp.symbol)}
                      className="hover:bg-gray-800/40 cursor-pointer transition"
                    >
                      {/* Rank */}
                      <td className="py-3 px-3 text-center font-bold text-gray-400">
                        #{idx + 1}
                      </td>

                      {/* Asset */}
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-1.5 font-bold text-white text-sm">
                          <span>{opp.symbol.replace('USDT', '')}</span>
                          <span className="text-[10px] text-gray-400 font-normal">/USDT</span>
                        </div>
                      </td>

                      {/* Price */}
                      <td className="py-3 px-4 font-semibold text-gray-200">
                        ${opp.price < 1 ? opp.price.toFixed(4) : opp.price.toFixed(2)}
                      </td>

                      {/* Exhaustion Score */}
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <span className={`font-bold text-xs ${exScore >= 70 ? 'text-emerald-400' : 'text-teal-300'}`}>
                            {exScore.toFixed(0)}/100
                          </span>
                          <div className="w-16 h-1.5 bg-gray-950 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${exScore >= 70 ? 'bg-emerald-400' : 'bg-teal-400'}`}
                              style={{ width: `${exScore}%` }}
                            />
                          </div>
                        </div>
                        <div className="text-[9px] text-gray-400 truncate max-w-[130px]">
                          {opp.bear_exhaustion.status}
                        </div>
                      </td>

                      {/* Recovery Probability */}
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-800 font-bold">
                          {recProb.toFixed(1)}%
                        </span>
                      </td>

                      {/* Decision Badge */}
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold border border-gray-700 bg-gray-900 text-gray-200">
                          {opp.decision_badge}
                        </span>
                      </td>

                      {/* Transition Window */}
                      <td className="py-3 px-4 text-cyan-300">
                        {opp.transition_window}
                      </td>

                      {/* Action */}
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectCoin(opp.symbol);
                            if (onNavigateToSetup) onNavigateToSetup();
                          }}
                          className="px-2.5 py-1 rounded bg-cyan-600/80 hover:bg-cyan-500 text-white font-mono text-[11px] font-semibold transition flex items-center gap-1 ml-auto"
                        >
                          <span>Analyze</span>
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
