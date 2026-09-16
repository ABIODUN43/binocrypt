import React from 'react';
import { 
  Activity, 
  ShieldAlert, 
  TrendingUp, 
  TrendingDown, 
  RefreshCw, 
  Sliders, 
  PieChart, 
  FlaskConical, 
  Terminal,
  Layers,
  Sparkles,
  Compass,
  Award,
  Skull
} from 'lucide-react';

export default function Navbar({ 
  activeTab, 
  setActiveTab, 
  regime, 
  account, 
  onRefresh, 
  isRefreshing 
}) {
  const getRegimeBadge = () => {
    if (!regime) return <span className="text-gray-500">Loading...</span>;
    
    let colorClass = "bg-blue-900/30 text-blue-400 border-blue-800/50";
    let dotClass = "bg-blue-400";
    
    if (regime.regime === "BULL_TREND") {
      colorClass = "bg-emerald-900/40 text-emerald-400 border-emerald-700/60";
      dotClass = "bg-emerald-400";
    } else if (regime.regime === "BEAR_TREND" || regime.regime === "PANIC") {
      colorClass = "bg-rose-900/40 text-rose-400 border-rose-700/60";
      dotClass = "bg-rose-400";
    } else if (regime.regime === "SIDEWAYS") {
      colorClass = "bg-amber-900/30 text-amber-400 border-amber-700/50";
      dotClass = "bg-amber-400";
    } else if (regime.regime === "BREAKOUT_ENVIRONMENT") {
      colorClass = "bg-cyan-900/40 text-cyan-400 border-cyan-700/60";
      dotClass = "bg-cyan-400";
    }

    return (
      <div className={`flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-semibold ${colorClass}`}>
        <span className={`w-2 h-2 rounded-full ${dotClass} animate-pulse`} />
        <span>{regime.regime.replace('_', ' ')}</span>
        <span className="opacity-70 font-mono-nums font-normal">({regime.confidence}%)</span>
      </div>
    );
  };

  const navItems = [
    { id: 'dashboard', label: 'Terminal', icon: Terminal },
    { id: 'intelligence', label: 'Market Intelligence', icon: Sparkles },
    { id: 'recovery', label: 'Recovery Scanner', icon: Compass },
    { id: 'quantlab', label: 'Quant Lab', icon: Sliders },
    { id: 'graveyard', label: 'Graveyard', icon: Skull },
    { id: 'history', label: 'Track Record', icon: Award },
    { id: 'paper', label: 'Paper Portfolio ($40)', icon: PieChart },
    { id: 'backtest', label: 'Backtest Lab', icon: FlaskConical },
  ];

  return (
    <header className="border-b border-[#1e2638] bg-[#0b0e14]/95 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Subtitle */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-emerald-500 to-blue-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Activity className="w-5 h-5 text-white stroke-[2.5]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-wider text-white">BINOCRYPT</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-950 text-blue-400 font-mono font-medium border border-blue-800/40">
                  SPOT v1.0
                </span>
              </div>
              <p className="text-[11px] text-gray-400 -mt-0.5 hidden sm:block">Quantitative Market Scanner & Decision Support</p>
            </div>
          </div>

          {/* Center: Live Regime & BTC Price */}
          <div className="hidden md:flex items-center gap-4">
            {getRegimeBadge()}
            
            {regime && (
              <div className="flex items-center gap-2 text-xs text-gray-300 font-mono-nums bg-[#121722] px-3 py-1 rounded-md border border-[#1e2638]">
                <span className="text-gray-400 font-sans">BTC:</span>
                <span className="font-medium">${regime.btc_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                <span className={regime.btc_change_24h >= 0 ? "text-emerald-400" : "text-rose-400"}>
                  {regime.btc_change_24h >= 0 ? "+" : ""}{regime.btc_change_24h.toFixed(2)}%
                </span>
              </div>
            )}
          </div>

          {/* Right: $40 Account Growth Tracker & Refresh */}
          <div className="flex items-center gap-3">
            {account && (
              <div className="bg-[#121722] border border-[#1e2638] rounded-md px-3 py-1 flex items-center gap-3">
                <div>
                  <div className="text-[10px] uppercase text-gray-400 font-semibold tracking-wider">$40 EXP</div>
                  <div className="text-sm font-bold font-mono-nums text-white">
                    ${account.current_equity.toFixed(2)}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] uppercase text-gray-400 font-semibold tracking-wider">RETURN</div>
                  <div className={`text-xs font-bold font-mono-nums ${account.total_pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {account.total_pnl_usd >= 0 ? "+" : ""}{account.total_return_pct.toFixed(1)}%
                  </div>
                </div>
              </div>
            )}

            <button
              onClick={onRefresh}
              disabled={isRefreshing}
              title="Refresh Market Data"
              className="p-2 rounded-md bg-[#121722] border border-[#1e2638] hover:border-gray-600 text-gray-300 hover:text-white transition disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isRefreshing ? "animate-spin text-emerald-400" : ""}`} />
            </button>
          </div>

        </div>

        {/* Tab Navigation */}
        <div className="flex items-center space-x-1 sm:space-x-2 py-2 overflow-x-auto border-t border-[#1e2638]/60">
          {navItems.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition whitespace-nowrap ${
                  isActive
                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-[#121722]'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            );
          })}
        </div>

      </div>
    </header>
  );
}
