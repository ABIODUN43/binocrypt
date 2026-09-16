import React, { useState } from 'react';
import { Sparkles, ChevronRight, Eye, Play, Search, Filter } from 'lucide-react';

export default function OpportunityTable({ 
  opportunities, 
  onSelectCoin, 
  onOpenExplain, 
  onQuickPaperTrade,
  selectedSymbol 
}) {
  const [search, setSearch] = useState('');
  const [filterTier, setFilterTier] = useState('ALL');

  const getGradeBadge = (grade) => {
    switch (grade) {
      case 'EXCELLENT':
        return <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-500/50">EXCELLENT</span>;
      case 'STRONG':
        return <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-blue-950 text-blue-400 border border-blue-500/50">STRONG</span>;
      case 'GOOD':
        return <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-cyan-950 text-cyan-400 border border-cyan-500/50">GOOD</span>;
      case 'WATCH':
        return <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-amber-950 text-amber-400 border border-amber-500/50">WATCH</span>;
      default:
        return <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-gray-800 text-gray-400 border border-gray-700">IGNORE</span>;
    }
  };

  const filtered = opportunities.filter(opp => {
    const matchesSearch = opp.symbol.toLowerCase().includes(search.toLowerCase());
    const matchesTier = filterTier === 'ALL' || opp.grade === filterTier;
    return matchesSearch && matchesTier;
  });

  return (
    <div className="terminal-card overflow-hidden">
      {/* Header & Filters */}
      <div className="p-4 border-b border-[#1e2638] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>Market Scanner Opportunities</span>
            <span className="text-xs font-mono font-normal text-gray-400">({filtered.length} Liquid Spot Pairs)</span>
          </h2>
          <p className="text-xs text-gray-400">Ranked by multi-factor quantitative confluence & risk-adjusted score</p>
        </div>

        <div className="flex items-center gap-2">
          {/* Search */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-gray-400" />
            <input
              type="text"
              placeholder="Filter pair..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-[#0b0e14] border border-[#1e2638] rounded-md pl-8 pr-3 py-1.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 w-32 sm:w-40"
            />
          </div>

          {/* Tier Filter */}
          <select
            value={filterTier}
            onChange={(e) => setFilterTier(e.target.value)}
            className="bg-[#0b0e14] border border-[#1e2638] rounded-md px-2.5 py-1.5 text-xs text-gray-300 focus:outline-none focus:border-blue-500"
          >
            <option value="ALL">All Tiers</option>
            <option value="EXCELLENT">Excellent (90+)</option>
            <option value="STRONG">Strong (80+)</option>
            <option value="GOOD">Good (70+)</option>
            <option value="WATCH">Watch (60+)</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[#1e2638] bg-[#0e121a] text-[11px] font-mono text-gray-400 uppercase tracking-wider">
              <th className="py-3 px-3 w-12 text-center">#</th>
              <th className="py-3 px-4">Asset</th>
              <th className="py-3 px-4">Price / 24h</th>
              <th className="py-3 px-4">24h Volume</th>
              <th className="py-3 px-4">Opportunity Score</th>
              <th className="py-3 px-4 hidden md:table-cell">Factor Breakdown</th>
              <th className="py-3 px-4">Win Prob</th>
              <th className="py-3 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1e2638]/60 text-xs">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan="8" className="py-8 text-center text-gray-500">
                  No opportunities match the current criteria.
                </td>
              </tr>
            ) : (
              filtered.map((opp) => {
                const isSelected = selectedSymbol === opp.symbol;
                return (
                  <tr 
                    key={opp.symbol}
                    onClick={() => onSelectCoin(opp.symbol)}
                    className={`cursor-pointer transition duration-150 ${
                      isSelected 
                        ? 'bg-blue-900/30 border-l-4 border-l-blue-400 text-white font-medium' 
                        : 'hover:bg-[#182030]/80'
                    }`}
                  >
                    {/* Rank */}
                    <td className="py-3 px-3 text-center font-mono font-bold text-gray-400">
                      {opp.rank}
                    </td>

                    {/* Asset */}
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white text-sm tracking-wide">{opp.symbol.replace('USDT', '')}</span>
                        <span className="text-[10px] text-gray-400 font-mono">/USDT</span>
                        {isSelected && (
                          <span className="px-1.5 py-0.5 rounded bg-blue-600 text-white font-mono text-[9px] font-bold tracking-wider animate-pulse">
                            ACTIVE
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Price & 24h Change */}
                    <td className="py-3 px-4 font-mono-nums">
                      <div className="font-semibold text-gray-200">
                        ${opp.price < 1 ? opp.price.toFixed(4) : opp.price.toFixed(2)}
                      </div>
                      <div className={`text-[11px] ${opp.change_24h >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {opp.change_24h >= 0 ? '+' : ''}{opp.change_24h.toFixed(2)}%
                      </div>
                    </td>

                    {/* Volume */}
                    <td className="py-3 px-4 font-mono-nums text-gray-300">
                      ${(opp.volume_24h / 1_000_000).toFixed(1)}M
                    </td>

                    {/* Opportunity Score */}
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-12 bg-gray-800 h-2 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${
                              opp.total_score >= 88 ? 'bg-emerald-500' :
                              opp.total_score >= 78 ? 'bg-blue-500' :
                              opp.total_score >= 68 ? 'bg-cyan-500' : 'bg-amber-500'
                            }`}
                            style={{ width: `${opp.total_score}%` }}
                          />
                        </div>
                        <span className="font-mono-nums font-bold text-sm text-white">{opp.total_score}</span>
                        {getGradeBadge(opp.grade)}
                      </div>
                    </td>

                    {/* Factor Breakdown (Desktop) */}
                    <td className="py-3 px-4 hidden md:table-cell text-[11px] font-mono-nums text-gray-400">
                      <div className="flex items-center gap-2">
                        <span title="Trend (max 20)">T:{opp.trend_score}</span>
                        <span>•</span>
                        <span title="Momentum (max 20)">M:{opp.momentum_score}</span>
                        <span>•</span>
                        <span title="Volume (max 20)">V:{opp.volume_score}</span>
                        <span>•</span>
                        <span title="Structure (max 20)">S:{opp.structure_score}</span>
                      </div>
                    </td>

                    {/* Win Probability */}
                    <td className="py-3 px-4 font-mono-nums font-semibold text-emerald-400">
                      {(opp.win_probability * 100).toFixed(0)}%
                    </td>

                    {/* Actions */}
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onOpenExplain(opp.symbol);
                          }}
                          title="AI Quantitative Thesis"
                          className="px-2 py-1 rounded bg-[#1e2638] hover:bg-purple-900/40 text-purple-300 hover:text-purple-200 border border-[#2a364f] hover:border-purple-600/50 text-[11px] font-medium flex items-center gap-1 transition"
                        >
                          <Sparkles className="w-3 h-3 text-purple-400" />
                          <span className="hidden lg:inline">AI Thesis</span>
                        </button>

                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectCoin(opp.symbol);
                          }}
                          className={`px-2 py-1 rounded text-[11px] font-medium flex items-center gap-1 transition ${
                            isSelected
                              ? 'bg-blue-600 text-white font-bold shadow-sm'
                              : 'bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30'
                          }`}
                        >
                          <Eye className="w-3 h-3" />
                          <span>{isSelected ? 'Viewing' : 'View Setup'}</span>
                        </button>

                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onQuickPaperTrade(opp.symbol);
                          }}
                          title="1-Click Risk-Sized Paper Trade"
                          className="px-2 py-1 rounded bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-500/30 text-[11px] font-medium flex items-center gap-1 transition"
                        >
                          <Play className="w-3 h-3" />
                          <span className="hidden sm:inline">Trade</span>
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
