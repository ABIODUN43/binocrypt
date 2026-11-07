import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { 
  ShieldAlert, 
  RefreshCw, 
  TrendingDown, 
  TrendingUp, 
  Target, 
  Layers, 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  ArrowRight, 
  Sparkles, 
  Compass,
  Zap,
  Info,
  ShieldCheck,
  Lock,
  Radio
} from 'lucide-react';

export default function BottomIntelligenceView({ 
  selectedSymbol, 
  onSelectCoin, 
  onExecuteTrade 
}) {
  const [symbol, setSymbol] = useState(selectedSymbol || 'ARBUSDT');
  const [data, setData] = useState(null);
  const [scannerItems, setScannerItems] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isScannerLoading, setIsScannerLoading] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const [activeTab, setActiveTab] = useState('inspector'); // 'inspector' or 'scanner'
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionMessage, setExecutionMessage] = useState(null);

  // Update symbol when parent changes
  useEffect(() => {
    if (selectedSymbol && selectedSymbol !== symbol) {
      setSymbol(selectedSymbol);
    }
  }, [selectedSymbol]);

  const loadData = async (targetSym) => {
    const sym = targetSym || symbol;
    setIsLoading(true);
    try {
      const res = await api.getReversalIntelligence(sym);
      if (res && res.data) {
        setData(res.data);
        setLastRefreshed(new Date().toLocaleTimeString());
      }
    } catch (err) {
      console.error("Failed to load reversal intelligence:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadScanner = async () => {
    setIsScannerLoading(true);
    try {
      const res = await api.getReversalScanner();
      if (res && res.items) {
        setScannerItems(res.items);
      }
    } catch (err) {
      console.error("Failed to load reversal scanner:", err);
    } finally {
      setIsScannerLoading(false);
    }
  };

  useEffect(() => {
    loadData(symbol);
    loadScanner();
  }, [symbol]);

  const handleSelectCoin = (newSym) => {
    setSymbol(newSym);
    if (onSelectCoin) {
      onSelectCoin(newSym);
    }
  };

  // State color helper
  const getStateBadge = (state, label) => {
    switch (state) {
      case 'REVERSAL_CONFIRMED':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-700/80 font-bold text-xs">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            {label || 'Reversal Confirmed'}
          </span>
        );
      case 'BASE_FORMING':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-700/80 font-bold text-xs">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            {label || 'Base Forming'}
          </span>
        );
      case 'DOWNSIDE_EXHAUSTION_WATCH':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-950/80 text-amber-300 border border-amber-700/80 font-bold text-xs">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            {label || 'Exhaustion Watch'}
          </span>
        );
      case 'FAILED_REVERSAL':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-950/80 text-rose-300 border border-rose-700/80 font-bold text-xs">
            <span className="w-2 h-2 rounded-full bg-rose-400" />
            {label || 'Failed Reversal'}
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-900 text-slate-300 border border-slate-700 font-bold text-xs">
            <span className="w-2 h-2 rounded-full bg-slate-400" />
            {label || 'Declining Trend'}
          </span>
        );
    }
  };

  // BR-008 Lifecycle Guard Badge Helper
  const getLifecycleBadge = (guard) => {
    if (!guard) return null;
    const st = guard.lifecycle_state;
    switch (st) {
      case 'READY_TO_EXECUTE':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-700 font-mono font-bold text-[11px]">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            READY TO EXECUTE
          </span>
        );
      case 'TARGET_REACHED_BEFORE_ENTRY':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-950 text-rose-300 border border-rose-700 font-mono font-bold text-[11px]">
            <AlertTriangle className="w-3 h-3 text-rose-400" />
            TARGET REACHED (CLOSED)
          </span>
        );
      case 'MISSED_ENTRY':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-950 text-amber-300 border border-amber-700 font-mono font-bold text-[11px]">
            <AlertTriangle className="w-3 h-3 text-amber-400" />
            MISSED ENTRY (RUNAWAY)
          </span>
        );
      case 'POSITION_OPEN':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-cyan-950 text-cyan-300 border border-cyan-700 font-mono font-bold text-[11px]">
            <CheckCircle2 className="w-3 h-3 text-cyan-400" />
            POSITION OPEN
          </span>
        );
      case 'INVALIDATED':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-950 text-rose-400 border border-rose-800 font-mono font-bold text-[11px]">
            <ShieldAlert className="w-3 h-3" />
            INVALIDATED
          </span>
        );
      case 'EXPIRED':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900 text-slate-400 border border-slate-700 font-mono font-bold text-[11px]">
            <Clock className="w-3 h-3" />
            EXPIRED (&gt;2H)
          </span>
        );
      case 'STALE_MARKET_DATA':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-950 text-amber-300 border border-amber-800 font-mono font-bold text-[11px]">
            <AlertTriangle className="w-3 h-3 text-amber-400" />
            STALE FEED
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900 text-slate-300 border border-slate-700 font-mono font-bold text-[11px]">
            {guard.lifecycle_label || st || 'MONITORING'}
          </span>
        );
    }
  };

  const handleExecuteReversalTrade = async () => {
    if (!data || !data.execution_setup || !data.execution_setup.can_execute) return;
    setIsExecuting(true);
    setExecutionMessage(null);
    try {
      const res = await api.executeReversalSignal({
        symbol: data.symbol,
        signal_id: data.execution_setup.signal_id
      });
      if (res && res.success) {
        setExecutionMessage({
          type: 'success',
          text: `Atomic Reversal Trade Filled: ${res.symbol} @ $${res.fill_price} ($${res.notional_usd?.toFixed(2)} USDT)`
        });
        await loadData(symbol);
        await loadScanner();
        if (onExecuteTrade) {
          onExecuteTrade(symbol, res);
        }
      } else {
        setExecutionMessage({
          type: 'error',
          text: res?.error || 'Execution rejected by safety guard.'
        });
      }
    } catch (err) {
      setExecutionMessage({
        type: 'error',
        text: err.message || 'Execution error encountered.'
      });
    } finally {
      setIsExecuting(false);
    }
  };

  const quickPicks = ['ARBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'OPUSDT', 'LINKUSDT', 'AVAXUSDT', 'ENAUSDT'];

  return (
    <div className="space-y-5">
      {/* 1. Header Banner & Coin Selector */}
      <div className="terminal-card bg-gradient-to-r from-[#0d131f] via-[#0f172a] to-[#0d131f] border-emerald-900/40 p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center">
                <Target className="w-5 h-5 text-emerald-400" />
              </div>
              <h1 className="text-lg font-bold text-white tracking-wide">
                BR-007: Swing Reversal & Bottom-Formation Intelligence
              </h1>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-950 text-blue-300 border border-blue-800">
                PROSPECTIVE PILOT
              </span>
            </div>
            <p className="text-xs text-gray-400 mt-1.5 max-w-2xl">
              Causal detection of downside exhaustion, stabilization envelopes, and confirmed swing recovery setups. 
              Eliminates the falling-knife fallacy by separating exhaustion monitoring from structural confirmation.
            </p>
          </div>

          {/* Quick Coin Selector & Refresh */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center bg-[#141b2d] rounded-lg p-1 border border-gray-800 text-xs font-mono">
              {quickPicks.slice(0, 5).map((p) => {
                const label = p.replace('USDT', '');
                const isSelected = symbol.toUpperCase().replace('/', '').replace('-', '') === p;
                return (
                  <button
                    key={p}
                    onClick={() => handleSelectCoin(p)}
                    className={`px-2.5 py-1 rounded transition ${
                      isSelected 
                        ? 'bg-emerald-600 text-white font-bold shadow' 
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => { loadData(symbol); loadScanner(); }}
              disabled={isLoading}
              className="btn btn-secondary flex items-center gap-1.5 text-xs py-2 px-3"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-emerald-400' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Sub-navigation tabs: Inspector vs Universe Scanner */}
        <div className="flex items-center gap-3 mt-4 pt-3 border-t border-gray-800/80 text-xs font-mono">
          <button
            onClick={() => setActiveTab('inspector')}
            className={`px-3 py-1.5 rounded-md font-semibold transition flex items-center gap-1.5 ${
              activeTab === 'inspector'
                ? 'bg-emerald-950 text-emerald-300 border border-emerald-700'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            <span>Asset Inspector ({data ? data.display_symbol : symbol})</span>
          </button>

          <button
            onClick={() => setActiveTab('scanner')}
            className={`px-3 py-1.5 rounded-md font-semibold transition flex items-center gap-1.5 ${
              activeTab === 'scanner'
                ? 'bg-emerald-950 text-emerald-300 border border-emerald-700'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Universe Bottom Screener ({scannerItems.length} Assets)</span>
          </button>

          {lastRefreshed && (
            <span className="text-[11px] text-gray-500 ml-auto hidden sm:inline">
              Data Updated: {lastRefreshed}
            </span>
          )}
        </div>
      </div>

      {/* 2. Main Content View */}
      {activeTab === 'inspector' && data && (
        <div className="space-y-5">
          
          {/* Top 4 KPI Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* KPI 1: Current State */}
            <div className="terminal-card bg-[#111726]/80 p-4 border-gray-800 flex flex-col justify-between">
              <div>
                <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider">Causal Market State</span>
                <div className="mt-2.5">
                  {getStateBadge(data.state, data.state_label)}
                </div>
              </div>
              <div className="mt-4 pt-2 border-t border-gray-800/60">
                <div className="flex justify-between text-[11px] font-mono text-gray-400 mb-1">
                  <span>State Progression</span>
                  <span className="font-bold text-white">Stage {data.state_stage}/5</span>
                </div>
                <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ${
                      data.state_stage >= 4 ? 'bg-emerald-400' :
                      data.state_stage === 3 ? 'bg-cyan-400' :
                      data.state_stage === 2 ? 'bg-amber-400' : 'bg-slate-500'
                    }`} 
                    style={{ width: `${(data.state_stage / 5) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* KPI 2: Potential-Bottom Zone */}
            <div className="terminal-card bg-[#111726]/80 p-4 border-gray-800 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider">Potential-Bottom Zone</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-950/60 text-blue-300 font-mono border border-blue-800/40">
                    {data.bottom_zone.zone_status}
                  </span>
                </div>
                <div className="mt-2 text-xl font-bold font-mono text-emerald-400">
                  ${data.bottom_zone.potential_bottom_low.toFixed(4)}
                </div>
                <div className="text-[11px] text-gray-400 font-mono mt-0.5">
                  Envelope: {data.bottom_zone.bottom_zone_summary}
                </div>
              </div>
              <div className="mt-3 text-[11px] font-mono flex items-center justify-between pt-2 border-t border-gray-800/60">
                <span className="text-gray-400">Current Distance:</span>
                <span className={`font-bold ${data.bottom_zone.dist_to_floor_pct <= 2 ? 'text-amber-400' : 'text-gray-300'}`}>
                  {data.bottom_zone.dist_to_floor_pct > 0 ? '+' : ''}{data.bottom_zone.dist_to_floor_pct}% above floor
                </span>
              </div>
            </div>

            {/* KPI 3: Downside Exhaustion Score */}
            <div className="terminal-card bg-[#111726]/80 p-4 border-gray-800 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider">Exhaustion Radar</span>
                  <span className="text-[11px] font-mono font-bold text-amber-400">
                    {data.exhaustion_score}/100
                  </span>
                </div>
                <div className="mt-2 text-xl font-bold font-mono text-white">
                  {data.exhaustion_score >= 70 ? 'Extreme Selling Exhaustion' :
                   data.exhaustion_score >= 45 ? 'Decelerating Momentum' : 'Orderly Trend'}
                </div>
                <div className="text-[11px] text-gray-400 font-mono mt-0.5">
                  RSI: {data.features_summary.rsi_14} | 24h Drop: {data.features_summary.dd_from_24h_high_pct}%
                </div>
              </div>
              <div className="mt-3 w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 via-amber-400 to-rose-400" 
                  style={{ width: `${data.exhaustion_score}%` }}
                />
              </div>
            </div>

            {/* KPI 4: Structural Invalidation Stop */}
            <div className="terminal-card bg-[#111726]/80 p-4 border-gray-800 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider">Structural Invalidation</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-950 text-rose-300 font-mono border border-rose-800">
                    HARD STOP
                  </span>
                </div>
                <div className="mt-2 text-xl font-bold font-mono text-rose-400">
                  ${data.bottom_zone.invalidation_price.toFixed(4)}
                </div>
                <div className="text-[11px] text-gray-400 font-mono mt-0.5">
                  Distance: -{data.bottom_zone.dist_to_invalidation_pct}% from current
                </div>
              </div>
              <div className="mt-3 text-[11px] font-mono text-gray-400 flex items-center justify-between pt-2 border-t border-gray-800/60">
                <span>Rule:</span>
                <span className="text-gray-300">1H close below invalidation</span>
              </div>
            </div>
          </div>

          {/* Middle Row: Potential-Bottom Visualizer & Actionable Execution Card */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            
            {/* Visualizer & Structural Analysis (7 cols) */}
            <div className="lg:col-span-7 terminal-card p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-gray-800 pb-3">
                <div className="flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4 text-emerald-400" />
                  <h3 className="font-bold text-sm text-white font-mono">
                    Potential-Bottom Zone Architecture
                  </h3>
                </div>
                <span className="text-xs font-mono text-gray-400">
                  Spot: <span className="text-white font-bold">${data.current_price.toFixed(4)}</span>
                </span>
              </div>

              {/* Price Ladder Diagram */}
              <div className="bg-[#0b0e14] p-4 rounded-xl border border-gray-800/80 font-mono text-xs space-y-3">
                
                {/* Take Profit 2 */}
                <div className="flex items-center justify-between p-2 rounded bg-emerald-950/20 border border-emerald-900/30">
                  <span className="text-emerald-400 font-semibold flex items-center gap-1.5">
                    <Target className="w-3.5 h-3.5" /> TP2 Target (Structural Swing High)
                  </span>
                  <span className="text-emerald-300 font-bold">${data.execution_setup.tp2_price.toFixed(4)} (+{((data.execution_setup.tp2_price - data.current_price)/data.current_price * 100).toFixed(1)}%)</span>
                </div>

                {/* Take Profit 1 */}
                <div className="flex items-center justify-between p-2 rounded bg-emerald-950/40 border border-emerald-800/40">
                  <span className="text-emerald-300 font-semibold flex items-center gap-1.5">
                    <Target className="w-3.5 h-3.5" /> TP1 Target (1.8x Risk)
                  </span>
                  <span className="text-emerald-300 font-bold">${data.execution_setup.tp1_price.toFixed(4)} (+{((data.execution_setup.tp1_price - data.current_price)/data.current_price * 100).toFixed(1)}%)</span>
                </div>

                {/* 1H EMA20 Resistance */}
                <div className="flex items-center justify-between p-2 rounded bg-blue-950/20 border border-blue-900/30">
                  <span className="text-blue-300 flex items-center gap-1.5">
                    <TrendingUp className="w-3.5 h-3.5" /> 1H EMA20 Trend Barrier
                  </span>
                  <span className="text-blue-300 font-bold">${data.features_summary.ema20_1h.toFixed(4)} ({data.features_summary.dist_to_ema20_pct > 0 ? '+' : ''}{data.features_summary.dist_to_ema20_pct}%)</span>
                </div>

                {/* Current Price Marker */}
                <div className="flex items-center justify-between p-2.5 rounded bg-gray-800/60 border-2 border-white/40 shadow-md">
                  <span className="text-white font-bold flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping" />
                    LIVE SPOT PRICE
                  </span>
                  <span className="text-white font-bold text-sm">${data.current_price.toFixed(4)}</span>
                </div>

                {/* Potential Floor Support */}
                <div className="flex items-center justify-between p-2 rounded bg-emerald-950/30 border border-emerald-600/40">
                  <span className="text-emerald-400 font-semibold flex items-center gap-1.5">
                    <Compass className="w-3.5 h-3.5" /> Estimated Potential Floor (L_pivot)
                  </span>
                  <span className="text-emerald-400 font-bold">${data.bottom_zone.potential_bottom_low.toFixed(4)}</span>
                </div>

                {/* Structural Invalidation Stop */}
                <div className="flex items-center justify-between p-2 rounded bg-rose-950/40 border border-rose-800/50">
                  <span className="text-rose-400 font-semibold flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5" /> Hard Invalidation Stop (Breakdown Level)
                  </span>
                  <span className="text-rose-400 font-bold">${data.bottom_zone.invalidation_price.toFixed(4)} (-{data.bottom_zone.dist_to_invalidation_pct}%)</span>
                </div>
              </div>

              {/* State Transition Justification Bullets */}
              <div className="space-y-1.5 pt-2">
                <span className="text-xs font-mono font-bold text-gray-300 uppercase">Causal Evidence & State Triggers:</span>
                <ul className="space-y-1 text-xs font-mono text-gray-400">
                  {data.state_reasons.map((r, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-0.5">•</span>
                      <span>{r}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Actionable Setup Card & Signal Radar (5 cols) */}
            <div className="lg:col-span-5 space-y-5">
              
              {/* Action Recommendation Card */}
              <div className="terminal-card p-5 border-emerald-800/40 bg-[#0f172a]/90 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-xs font-mono uppercase text-gray-400 font-bold block">Execution Directive & Signal Guard</span>
                    <span className="text-[10px] font-mono text-gray-500">BR-008 Causal Lifecycle Control</span>
                  </div>
                  <div>
                    {getLifecycleBadge(data.lifecycle_guard)}
                  </div>
                </div>

                {/* Feed Provenance Strip */}
                {data.data_provenance && (
                  <div className="flex items-center justify-between px-3 py-1.5 rounded bg-[#0b0e14] border border-gray-800 text-[10px] font-mono text-gray-400">
                    <span className="flex items-center gap-1.5">
                      <Radio className={`w-3 h-3 ${data.data_provenance.feed_status === 'LIVE_HEALTHY' ? 'text-emerald-400 animate-pulse' : 'text-amber-400'}`} />
                      Feed: <strong className="text-gray-200">{data.data_provenance.source} Spot</strong>
                    </span>
                    <span>
                      Latency: <strong className="text-gray-200">{typeof data.data_provenance.age_seconds === 'number' ? `${data.data_provenance.age_seconds.toFixed(1)}s` : '0s'}</strong>
                    </span>
                  </div>
                )}

                {/* Critical Lifecycle Alert Callouts */}
                {data.lifecycle_guard?.lifecycle_state === 'TARGET_REACHED_BEFORE_ENTRY' && (
                  <div className="p-3.5 rounded-lg bg-rose-950/60 border border-rose-600/70 text-xs font-mono space-y-1.5 text-rose-200">
                    <div className="flex items-center gap-2 font-bold text-rose-300">
                      <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                      <span>TARGET REACHED BEFORE ENTRY — ORDER BLOCKED</span>
                    </div>
                    <p className="text-[11px] leading-relaxed text-rose-200/90 font-sans">
                      Live market price (<strong>${data.current_price?.toFixed(4)}</strong>) has reached or exceeded TP1 target (<strong>${data.execution_setup?.tp1_price?.toFixed(4)}</strong>). The original swing opportunity is closed. Under causal risk governance, chasing past target is strictly forbidden.
                    </p>
                  </div>
                )}

                {data.lifecycle_guard?.lifecycle_state === 'MISSED_ENTRY' && (
                  <div className="p-3.5 rounded-lg bg-amber-950/60 border border-amber-600/70 text-xs font-mono space-y-1.5 text-amber-200">
                    <div className="flex items-center gap-2 font-bold text-amber-300">
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                      <span>MISSED ENTRY — RUNAWAY PRICE ABOVE ZONE</span>
                    </div>
                    <p className="text-[11px] leading-relaxed text-amber-200/90 font-sans">
                      Current price (<strong>${data.current_price?.toFixed(4)}</strong>) has moved beyond the safe entry boundary (<strong>${data.execution_setup?.entry_zone_max?.toFixed(4)}</strong>). Entering now degrades the risk-to-reward ratio ({data.execution_setup?.live_rr ? `live 1:${data.execution_setup.live_rr}` : 'impaired'}). Wait for a pullback into the confirmed zone.
                    </p>
                  </div>
                )}

                {data.lifecycle_guard?.lifecycle_state === 'POSITION_OPEN' && (
                  <div className="p-3.5 rounded-lg bg-cyan-950/60 border border-cyan-600/70 text-xs font-mono space-y-1.5 text-cyan-200">
                    <div className="flex items-center gap-2 font-bold text-cyan-300">
                      <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                      <span>POSITION ACTIVE IN PAPER TRADER</span>
                    </div>
                    <p className="text-[11px] leading-relaxed text-cyan-200/90 font-sans">
                      A position for this signal is currently active in your Paper Trading portfolio (Signal: <span className="font-mono">{data.execution_setup?.signal_id}</span>). Duplicate entry is locked by the database.
                    </p>
                  </div>
                )}

                {data.lifecycle_guard?.lifecycle_state === 'STALE_MARKET_DATA' && (
                  <div className="p-3.5 rounded-lg bg-amber-950/60 border border-amber-600/70 text-xs font-mono space-y-1.5 text-amber-200">
                    <div className="flex items-center gap-2 font-bold text-amber-300">
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                      <span>MARKET DATA FEED STALE</span>
                    </div>
                    <p className="text-[11px] leading-relaxed text-amber-200/90 font-sans">
                      Market data feed age exceeds the safety threshold ({data.data_provenance?.age_seconds?.toFixed(0)}s &gt; 180s). Trading is locked to protect against stale execution.
                    </p>
                  </div>
                )}

                <div className="p-3 rounded-lg bg-[#0b0e14] border border-gray-800 text-xs font-mono space-y-2">
                  <div className="text-gray-300 leading-relaxed font-sans">
                    {data.execution_setup.action_recommendation}
                  </div>
                  {data.execution_setup.signal_id && (
                    <div className="text-[10px] text-gray-500 font-mono flex items-center justify-between border-t border-gray-800/60 pt-1.5">
                      <span>Signal ID:</span>
                      <span className="text-gray-400">{data.execution_setup.signal_id}</span>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2 pt-2 border-t border-gray-800/80 text-[11px]">
                    <div>
                      <span className="text-gray-500">Entry Zone:</span>
                      <div className="text-white font-bold">${data.execution_setup.entry_zone_min.toFixed(4)} - ${data.execution_setup.entry_zone_max.toFixed(4)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Live Spot Price:</span>
                      <div className="text-emerald-400 font-bold">${data.current_price.toFixed(4)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Stop Distance:</span>
                      <div className="text-rose-400 font-bold">-{data.execution_setup.stop_distance_pct}%</div>
                    </div>
                    <div>
                      <span className="text-gray-500">TP1 Target:</span>
                      <div className="text-emerald-400 font-bold">${data.execution_setup.tp1_price.toFixed(4)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Original R/R:</span>
                      <div className="text-cyan-400 font-bold">1 : {data.execution_setup.risk_reward_ratio}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Live R/R:</span>
                      <div className={`font-bold ${data.execution_setup.live_rr && data.execution_setup.live_rr < 1.3 ? 'text-rose-400' : 'text-emerald-400'}`}>
                        1 : {data.execution_setup.live_rr || data.execution_setup.risk_reward_ratio}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Execution Feedback Banner */}
                {executionMessage && (
                  <div className={`p-3 rounded-lg text-xs font-mono flex items-center gap-2 ${
                    executionMessage.type === 'success' 
                      ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-700' 
                      : 'bg-rose-950/80 text-rose-300 border border-rose-700'
                  }`}>
                    {executionMessage.type === 'success' ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                    )}
                    <span>{executionMessage.text}</span>
                  </div>
                )}

                <button
                  onClick={handleExecuteReversalTrade}
                  disabled={!data.execution_setup.can_execute || isExecuting}
                  className={`w-full py-2.5 px-4 rounded-lg font-mono font-bold text-xs flex items-center justify-center gap-2 transition ${
                    data.execution_setup.can_execute && !isExecuting
                      ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/30 cursor-pointer'
                      : 'bg-gray-800/60 text-gray-500 cursor-not-allowed border border-gray-700/40'
                  }`}
                >
                  {isExecuting ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
                      <span>Executing Atomic Signal...</span>
                    </>
                  ) : data.execution_setup.can_execute ? (
                    <>
                      <Zap className="w-4 h-4" />
                      <span>Execute Confirmed Swing Trade (Spot Paper)</span>
                    </>
                  ) : data.lifecycle_guard?.lifecycle_state === 'TARGET_REACHED_BEFORE_ENTRY' ? (
                    <>
                      <Lock className="w-4 h-4 text-rose-400" />
                      <span>Target Reached Before Entry (Locked)</span>
                    </>
                  ) : data.lifecycle_guard?.lifecycle_state === 'MISSED_ENTRY' ? (
                    <>
                      <Lock className="w-4 h-4 text-amber-400" />
                      <span>Missed Entry: Runaway Price (Locked)</span>
                    </>
                  ) : data.lifecycle_guard?.lifecycle_state === 'POSITION_OPEN' ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                      <span>Position Already Active (Locked)</span>
                    </>
                  ) : data.lifecycle_guard?.lifecycle_state === 'EXPIRED' ? (
                    <>
                      <Clock className="w-4 h-4 text-gray-400" />
                      <span>Signal Expired (&gt;2 Hours) (Locked)</span>
                    </>
                  ) : data.lifecycle_guard?.lifecycle_state === 'STALE_MARKET_DATA' ? (
                    <>
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      <span>Feed Stale (&gt;180s) (Locked)</span>
                    </>
                  ) : (
                    <>
                      <Lock className="w-4 h-4 text-gray-400" />
                      <span>Entry Locked (Awaiting Confirmation)</span>
                    </>
                  )}
                </button>
              </div>

              {/* Causal Factor Radar Scores */}
              <div className="terminal-card p-4 space-y-3">
                <span className="text-xs font-mono font-bold text-gray-300 uppercase">21-Feature Causal Signal Stack</span>
                <div className="space-y-2 text-xs font-mono">
                  {Object.entries(data.radar_signals).map(([key, score]) => (
                    <div key={key} className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-gray-400 capitalize">{key.replace('_', ' ')}</span>
                        <span className="font-bold text-white">{score}/100</span>
                      </div>
                      <div className="w-full bg-gray-800 h-1 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-emerald-500" 
                          style={{ width: `${score}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* 3. Triple-Barrier Target Scenarios (BR-007 Configs A, B, C) */}
          <div className="terminal-card p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-gray-800 pb-3">
              <div>
                <h3 className="font-bold text-sm text-white font-mono flex items-center gap-2">
                  <Target className="w-4 h-4 text-cyan-400" />
                  Pre-Registered Triple-Barrier Recovery Scenarios
                </h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Evaluates likelihood of hitting upside recovery target (+U) before downside failure stop (-D) within fixed swing horizons.
                </p>
              </div>
              <span className="text-[11px] font-mono text-gray-400 bg-gray-900 px-2.5 py-1 rounded border border-gray-800">
                Friction Modeled: 0.40% round-trip
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(data.triple_barriers).map(([key, barrier]) => (
                <div 
                  key={key} 
                  className={`p-4 rounded-xl border flex flex-col justify-between space-y-3 font-mono text-xs ${
                    barrier.is_primary 
                      ? 'bg-gradient-to-b from-[#101b2b] to-[#0c121d] border-emerald-700/60 shadow-lg' 
                      : 'bg-[#0f1422] border-gray-800'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-xs">{barrier.name}</span>
                    {barrier.is_primary && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 font-bold border border-emerald-800">
                        PRIMARY
                      </span>
                    )}
                  </div>

                  <div className="space-y-1.5 py-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Horizon:</span>
                      <span className="text-white font-bold">{barrier.horizon_label}</span>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Target Price (+{barrier.upside_target_pct}%):</span>
                      <span className="text-emerald-400 font-bold">${barrier.target_price.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Stop Price (-{barrier.downside_stop_pct}%):</span>
                      <span className="text-rose-400 font-bold">${barrier.stop_price.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Risk/Reward:</span>
                      <span className="text-cyan-400 font-bold">1 : {barrier.risk_reward_ratio}</span>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-gray-800/80 flex items-center justify-between">
                    <div>
                      <span className="text-[10px] text-gray-500 uppercase block">Estimated Probability</span>
                      <span className="text-base font-bold text-emerald-400">{barrier.win_probability_pct}%</span>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] text-gray-500 uppercase block">Net Expectancy</span>
                      <span className={`text-sm font-bold ${barrier.net_expectancy_pct > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {barrier.net_expectancy_pct > 0 ? '+' : ''}{barrier.net_expectancy_pct}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 4. Research Governance Disclaimer */}
          <div className="p-4 rounded-xl bg-[#0d121c] border border-gray-800 text-xs font-mono text-gray-400 flex items-start gap-3">
            <Info className="w-5 h-5 text-blue-400 mt-0.5 shrink-0" />
            <div className="space-y-1">
              <span className="font-bold text-gray-200">Scientific Governance & Truth-in-Advertising Notice:</span>
              <p className="leading-relaxed">
                BR-007 calculates statistical exhaustion indicators and confirmation pivots strictly point-in-time. 
                It makes no claim of knowing single-tick exact lows before the market produces confirmation. 
                All target barriers are subject to execution slippage and 0.40% round-trip friction.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 3. Universe Screener View (When activeTab === 'scanner') */}
      {activeTab === 'scanner' && (
        <div className="terminal-card overflow-hidden">
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <div>
              <h3 className="font-bold text-sm text-white font-mono flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-400" />
                Universe Bottom-Formation Screener
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                Ranks all liquid universe assets by downside exhaustion and stabilization state.
              </p>
            </div>
            <button
              onClick={loadScanner}
              disabled={isScannerLoading}
              className="btn btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5 font-mono"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isScannerLoading ? 'animate-spin' : ''}`} />
              <span>Rescan</span>
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 text-[11px] uppercase tracking-wider bg-gray-950/60">
                  <th className="py-3 px-4">Asset</th>
                  <th className="py-3 px-4">Price</th>
                  <th className="py-3 px-4">Causal State</th>
                  <th className="py-3 px-4">Lifecycle Guard</th>
                  <th className="py-3 px-4 text-center">Exhaustion</th>
                  <th className="py-3 px-4 text-center">Stabilization</th>
                  <th className="py-3 px-4">Est. Floor</th>
                  <th className="py-3 px-4">Distance</th>
                  <th className="py-3 px-4">Target Prob</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60">
                {scannerItems.map((item) => (
                  <tr 
                    key={item.symbol}
                    onClick={() => {
                      handleSelectCoin(item.symbol);
                      setActiveTab('inspector');
                    }}
                    className="hover:bg-gray-800/40 cursor-pointer transition"
                  >
                    <td className="py-3 px-4 font-bold text-white flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      {item.display_symbol}
                    </td>
                    <td className="py-3 px-4 font-mono font-medium text-gray-200">
                      ${item.current_price.toFixed(4)}
                    </td>
                    <td className="py-3 px-4">
                      {getStateBadge(item.state, item.state_label)}
                    </td>
                    <td className="py-3 px-4">
                      {item.is_actionable ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-300 bg-emerald-950/80 border border-emerald-700/80 px-2 py-0.5 rounded">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                          READY TO EXECUTE
                        </span>
                      ) : item.lifecycle_state === 'TARGET_REACHED_BEFORE_ENTRY' ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-bold text-rose-300 bg-rose-950/80 border border-rose-800/80 px-2 py-0.5 rounded">
                          <AlertTriangle className="w-3 h-3 text-rose-400" />
                          Target Reached
                        </span>
                      ) : item.lifecycle_state === 'MISSED_ENTRY' ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-bold text-amber-300 bg-amber-950/80 border border-amber-800/80 px-2 py-0.5 rounded">
                          <AlertTriangle className="w-3 h-3 text-amber-400" />
                          Missed Entry
                        </span>
                      ) : item.lifecycle_state === 'POSITION_OPEN' ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-bold text-cyan-300 bg-cyan-950/80 border border-cyan-800/80 px-2 py-0.5 rounded">
                          <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                          In Position
                        </span>
                      ) : (
                        <span className="text-[10px] font-mono text-gray-500">
                          {item.lifecycle_label || item.lifecycle_state || 'Standby'}
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                        item.exhaustion_score >= 70 ? 'bg-rose-950 text-rose-300 border border-rose-800' :
                        item.exhaustion_score >= 45 ? 'bg-amber-950 text-amber-300 border border-amber-800' :
                        'bg-gray-900 text-gray-400'
                      }`}>
                        {item.exhaustion_score}/100
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                        item.stabilization_score >= 60 ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' :
                        item.stabilization_score >= 40 ? 'bg-cyan-950 text-cyan-300 border border-cyan-800' :
                        'bg-gray-900 text-gray-400'
                      }`}>
                        {item.stabilization_score}/100
                      </span>
                    </td>
                    <td className="py-3 px-4 text-emerald-400 font-bold">
                      ${item.potential_floor.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 font-medium text-gray-300">
                      {item.dist_to_floor_pct > 0 ? '+' : ''}{item.dist_to_floor_pct}%
                    </td>
                    <td className="py-3 px-4 text-cyan-400 font-bold">
                      {item.primary_target_prob}%
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button className="text-emerald-400 hover:text-emerald-300 font-bold flex items-center gap-1 ml-auto">
                        <span>Inspect</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
