import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { 
  Target, 
  Layers, 
  RefreshCw, 
  TrendingUp, 
  TrendingDown, 
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
  Radio,
  Sliders,
  Award,
  ChevronRight,
  BarChart2,
  Percent
} from 'lucide-react';

export default function RecoveryIntelligenceView({ 
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
  const [activeTab, setActiveTab] = useState('inspector'); // 'inspector', 'scanner', 'benchmark'
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionMessage, setExecutionMessage] = useState(null);

  useEffect(() => {
    if (selectedSymbol && selectedSymbol !== symbol) {
      setSymbol(selectedSymbol);
    }
  }, [selectedSymbol]);

  const loadData = async (targetSym) => {
    const sym = targetSym || symbol;
    setIsLoading(true);
    setExecutionMessage(null);
    try {
      const res = await api.getRecoveryIntelligence(sym);
      if (res && res.data) {
        setData(res.data);
        setLastRefreshed(new Date().toLocaleTimeString());
      }
    } catch (err) {
      console.error("Failed to load recovery intelligence:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadScanner = async () => {
    setIsScannerLoading(true);
    try {
      const res = await api.getRecoveryScanner();
      if (res && res.items) {
        setScannerItems(res.items);
      }
    } catch (err) {
      console.error("Failed to load recovery scanner:", err);
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

  const handleExecuteTrade = async () => {
    if (!data || data.decision !== 'ENTER') return;
    setIsExecuting(true);
    setExecutionMessage(null);
    try {
      const res = await api.executeRecoveryTrade({
        symbol: data.symbol,
        preferred_entry: data.candidate_entry.preferred,
        stop_loss: data.invalidation.price,
        tp1: data.target_forecasts[0]?.target,
        tp2: data.target_forecasts[1]?.target,
        signal_id: data.execution_guard?.signal_id
      });
      if (res && res.success) {
        setExecutionMessage({
          type: 'success',
          text: `Atomic Recovery Order Filled: ${res.symbol} @ $${res.entry_price} ($${res.notional_usd?.toFixed(2)} USDT)`
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

  const getDecisionBadge = (decision, label) => {
    switch (decision) {
      case 'ENTER':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/90 text-emerald-300 border border-emerald-600 font-mono font-bold text-xs shadow-lg shadow-emerald-900/30">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            {label || 'ENTER (SPOT BUY)'}
          </span>
        );
      case 'WAIT':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-950/90 text-cyan-300 border border-cyan-600 font-mono font-bold text-xs shadow-lg shadow-cyan-900/30">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            {label || 'WAIT (AWAITING OPTIMAL ENTRY)'}
          </span>
        );
      case 'AVOID':
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-950/90 text-rose-300 border border-rose-600 font-mono font-bold text-xs">
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
            {label || 'AVOID (ADVERSE RISK)'}
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-900 text-slate-300 border border-slate-700 font-mono font-bold text-xs">
            <Info className="w-3.5 h-3.5 text-slate-400" />
            {label || 'INSUFFICIENT EVIDENCE'}
          </span>
        );
    }
  };

  const quickPicks = ['ARBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'OPUSDT', 'LINKUSDT', 'AVAXUSDT', 'ENAUSDT'];

  return (
    <div className="space-y-5">
      {/* 1. Header Banner & Coin Selector */}
      <div className="terminal-card bg-gradient-to-r from-[#0d131f] via-[#0f172a] to-[#0d131f] border-cyan-900/40 p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-cyan-400" />
              </div>
              <h1 className="text-lg font-bold text-white tracking-wide">
                BR-009: Probabilistic Recovery & Optimal Entry Intelligence
              </h1>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">
                14-DAY HORIZON
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-950 text-blue-300 border border-blue-800">
                RESEARCH PILOT
              </span>
            </div>
            <p className="text-xs text-gray-400 mt-1.5 max-w-2xl">
              Identifies favorable buying opportunities during depressed market conditions before conventional confirmation. 
              Jointly optimizes candidate entry price and estimates target-specific recovery probabilities over 14 calendar days.
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
                        ? 'bg-cyan-600 text-white font-bold shadow' 
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
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Sub-navigation tabs */}
        <div className="flex items-center gap-3 mt-4 pt-3 border-t border-gray-800/80 text-xs font-mono">
          <button
            onClick={() => setActiveTab('inspector')}
            className={`px-3 py-1.5 rounded-md font-semibold transition flex items-center gap-1.5 ${
              activeTab === 'inspector'
                ? 'bg-cyan-950 text-cyan-300 border border-cyan-700'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            <span>Recovery Inspector ({data ? data.display_symbol : symbol})</span>
          </button>

          <button
            onClick={() => setActiveTab('benchmark')}
            className={`px-3 py-1.5 rounded-md font-semibold transition flex items-center gap-1.5 ${
              activeTab === 'benchmark'
                ? 'bg-cyan-950 text-cyan-300 border border-cyan-700'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>6-Strategy Entry Benchmark</span>
          </button>

          <button
            onClick={() => setActiveTab('scanner')}
            className={`px-3 py-1.5 rounded-md font-semibold transition flex items-center gap-1.5 ${
              activeTab === 'scanner'
                ? 'bg-cyan-950 text-cyan-300 border border-cyan-700'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>14-Day Recovery Screener ({scannerItems.length} Assets)</span>
          </button>

          {lastRefreshed && (
            <span className="text-[11px] text-gray-500 ml-auto hidden sm:inline font-mono">
              Last Evaluated: {lastRefreshed}
            </span>
          )}
        </div>
      </div>

      {/* 2. Main Inspector View */}
      {activeTab === 'inspector' && data && (
        <div className="space-y-5">
          
          {/* Top 4 KPI Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* KPI 1: Decision Directive */}
            <div className="terminal-card bg-[#111726]/80 p-4 border-gray-800 flex flex-col justify-between">
              <div>
                <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider">Model Decision Directive</span>
                <div className="mt-2.5">
                  {getDecisionBadge(data.decision, data.decision_badge)}
                </div>
              </div>
              <div className="mt-4 pt-2 border-t border-gray-800/60 text-[11px] font-mono text-gray-400">
                <span className="text-gray-500 block">Market Regime:</span>
                <span className="text-gray-200 font-semibold">{data.market_state}</span>
              </div>
            </div>

            {/* KPI 2: Preferred Optimal Entry */}
            <div className="terminal-card bg-[#111726]/80 p-4 border-gray-800 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider">Optimal Entry Price</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950/60 text-cyan-300 font-mono border border-cyan-800/40">
                    {data.candidate_entry.optimal_strategy_name?.split('—')[0]?.trim()}
                  </span>
                </div>
                <div className="mt-2 text-xl font-bold font-mono text-cyan-400">
                  ${data.candidate_entry.preferred.toFixed(4)}
                </div>
                <div className="text-[11px] text-gray-400 font-mono mt-0.5">
                  Zone: ${data.candidate_entry.lower.toFixed(4)} - ${data.candidate_entry.upper.toFixed(4)}
                </div>
              </div>
              <div className="mt-3 text-[11px] font-mono flex items-center justify-between pt-2 border-t border-gray-800/60">
                <span className="text-gray-400">Current Spot:</span>
                <span className={`font-bold ${data.candidate_entry.is_current_price_in_zone ? 'text-emerald-400' : 'text-gray-300'}`}>
                  ${data.current_price.toFixed(4)} ({data.candidate_entry.distance_from_current_pct > 0 ? '+' : ''}{data.candidate_entry.distance_from_current_pct}%)
                </span>
              </div>
            </div>

            {/* KPI 3: 14-Day TP1 Probability */}
            <div className="terminal-card bg-[#111726]/80 p-4 border-gray-800 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider">14D Recovery Probability</span>
                  <span className="text-[11px] font-mono font-bold text-emerald-400">
                    TP1: +{data.target_forecasts[0]?.gain_pct}%
                  </span>
                </div>
                <div className="mt-2 text-xl font-bold font-mono text-white flex items-baseline gap-2">
                  <span>{data.target_forecasts[0]?.target_first_probability}%</span>
                  <span className="text-xs text-gray-400 font-normal font-mono">
                    [{data.target_forecasts[0]?.confidence_interval_95[0]}% - {data.target_forecasts[0]?.confidence_interval_95[1]}%]
                  </span>
                </div>
                <div className="text-[11px] text-emerald-400 font-mono mt-0.5 font-bold">
                  Net Expectancy: +{data.expected_net_return}% (Friction 0.40% ded.)
                </div>
              </div>
              <div className="mt-3 w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 via-cyan-400 to-emerald-400" 
                  style={{ width: `${data.target_forecasts[0]?.target_first_probability}%` }}
                />
              </div>
            </div>

            {/* KPI 4: Structural Invalidation Level */}
            <div className="terminal-card bg-[#111726]/80 p-4 border-gray-800 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider">Structural Invalidation</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-300 font-mono border border-rose-800/40">
                    HARD STOP
                  </span>
                </div>
                <div className="mt-2 text-xl font-bold font-mono text-rose-400">
                  ${data.invalidation.price.toFixed(4)}
                </div>
                <div className="text-[11px] text-gray-400 font-mono mt-0.5">
                  Stop Distance: -{data.invalidation.distance_pct}% from spot
                </div>
              </div>
              <div className="mt-3 text-[11px] font-mono flex items-center justify-between pt-2 border-t border-gray-800/60">
                <span className="text-gray-400">Invalidation Prob:</span>
                <span className="font-bold text-rose-300">
                  {data.barrier_probabilities.invalidation_first_prob}% (Timeout: {data.barrier_probabilities.timeout_14d_prob}%)
                </span>
              </div>
            </div>
          </div>

          {/* Main 2-Column Grid: Matrix (7 cols) & Execution Directive (5 cols) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            
            {/* Left: 14-Day Probabilistic Recovery Matrix (7 cols) */}
            <div className="lg:col-span-7 space-y-5">
              <div className="terminal-card p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-gray-800 pb-3">
                  <div>
                    <h3 className="font-bold text-sm text-white font-mono flex items-center gap-2">
                      <Target className="w-4 h-4 text-cyan-400" />
                      14-Day Multi-Barrier Recovery Probability Matrix
                    </h3>
                    <p className="text-xs text-gray-400 mt-0.5">
                      First-passage target-first probabilities conditioned on entry before invalidation ($S = ${data.invalidation.price.toFixed(4)}).
                    </p>
                  </div>
                  <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800">
                    Monotonic Kernel P(TP1) &ge; P(TP2) &ge; P(TP3) &ge; P(TP4)
                  </span>
                </div>

                <div className="space-y-3 font-mono text-xs">
                  {data.target_forecasts.map((tf, idx) => (
                    <div 
                      key={tf.target_name}
                      className="p-3.5 rounded-xl border border-gray-800 bg-[#0c121e]/90 hover:border-cyan-800/60 transition space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            idx === 0 ? 'bg-emerald-950 text-emerald-300 border border-emerald-700' :
                            idx === 1 ? 'bg-cyan-950 text-cyan-300 border border-cyan-700' :
                            'bg-blue-950 text-blue-300 border border-blue-700'
                          }`}>
                            {tf.target_name} (+{tf.gain_pct}%)
                          </span>
                          <span className="font-bold text-white">${tf.target.toFixed(4)}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-gray-400 text-[11px]">R:R 1:{tf.risk_reward_ratio}</span>
                          <span className="font-bold text-sm text-emerald-400">{tf.target_first_probability}%</span>
                        </div>
                      </div>

                      {/* Progress bar */}
                      <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${
                            idx === 0 ? 'bg-emerald-400' :
                            idx === 1 ? 'bg-cyan-400' :
                            idx === 2 ? 'bg-blue-400' : 'bg-purple-400'
                          }`}
                          style={{ width: `${tf.target_first_probability}%` }}
                        />
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-gray-400 pt-1">
                        <span>95% CI: [{tf.confidence_interval_95[0]}% - {tf.confidence_interval_95[1]}%]</span>
                        <span>Est. Days: ~{tf.expected_days_to_target}d</span>
                        <span className={`font-bold ${tf.net_expectancy_pct > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          Net Exp: {tf.net_expectancy_pct > 0 ? '+' : ''}{tf.net_expectancy_pct}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Downside Failure & Timeout Row */}
                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div className="p-3 rounded-lg bg-rose-950/30 border border-rose-900/40 text-xs font-mono">
                    <span className="text-[10px] text-rose-400 uppercase font-bold block">Invalidation First (14D)</span>
                    <span className="text-base font-bold text-rose-300 mt-1 block">
                      {data.barrier_probabilities.invalidation_first_prob}%
                    </span>
                    <span className="text-[10px] text-gray-400 mt-0.5 block">Price hits stop before TP1</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-900/60 border border-gray-800 text-xs font-mono">
                    <span className="text-[10px] text-gray-400 uppercase font-bold block">14-Day Timeout (Unresolved)</span>
                    <span className="text-base font-bold text-gray-300 mt-1 block">
                      {data.barrier_probabilities.timeout_14d_prob}%
                    </span>
                    <span className="text-[10px] text-gray-500 mt-0.5 block">Neither barrier reached in 14d</span>
                  </div>
                </div>
              </div>

              {/* 28-Feature Causal Summary */}
              <div className="terminal-card p-4 space-y-3 font-mono text-xs">
                <span className="text-xs font-bold text-gray-300 uppercase">Point-in-Time Predictive Feature Summary</span>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <div className="p-2.5 rounded bg-[#0b0e14] border border-gray-800">
                    <span className="text-[10px] text-gray-500 block">30D Drawdown</span>
                    <span className="text-white font-bold">-{data.features_summary.drawdown_30d_pct}%</span>
                  </div>
                  <div className="p-2.5 rounded bg-[#0b0e14] border border-gray-800">
                    <span className="text-[10px] text-gray-500 block">RSI 14</span>
                    <span className="text-white font-bold">{data.features_summary.rsi_14}</span>
                  </div>
                  <div className="p-2.5 rounded bg-[#0b0e14] border border-gray-800">
                    <span className="text-[10px] text-gray-500 block">Base Compression</span>
                    <span className="text-white font-bold">{data.features_summary.base_compression_ratio}x ATR</span>
                  </div>
                  <div className="p-2.5 rounded bg-[#0b0e14] border border-gray-800">
                    <span className="text-[10px] text-gray-500 block">Volume Exhaustion</span>
                    <span className="text-white font-bold">{data.features_summary.volume_exhaustion_ratio}x MA20</span>
                  </div>
                  <div className="p-2.5 rounded bg-[#0b0e14] border border-gray-800">
                    <span className="text-[10px] text-gray-500 block">Higher Lows (14d)</span>
                    <span className="text-white font-bold">{data.features_summary.higher_low_count} bars</span>
                  </div>
                  <div className="p-2.5 rounded bg-[#0b0e14] border border-gray-800">
                    <span className="text-[10px] text-gray-500 block">Rel. Strength vs BTC</span>
                    <span className="text-cyan-400 font-bold">{data.features_summary.relative_strength_vs_btc > 0 ? '+' : ''}{data.features_summary.relative_strength_vs_btc}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Action Recommendation Card & Optimal Entry Inspector (5 cols) */}
            <div className="lg:col-span-5 space-y-5">
              
              {/* Action Directive Card */}
              <div className="terminal-card p-5 border-cyan-800/40 bg-[#0f172a]/90 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-xs font-mono uppercase text-gray-400 font-bold block">Model Action Directive</span>
                    <span className="text-[10px] font-mono text-gray-500">BR-009 Joint Optimization</span>
                  </div>
                  <div>
                    {getDecisionBadge(data.decision, data.decision_badge)}
                  </div>
                </div>

                {/* Feed Provenance Strip */}
                {data.data_provenance && (
                  <div className="flex items-center justify-between px-3 py-1.5 rounded bg-[#0b0e14] border border-gray-800 text-[10px] font-mono text-gray-400">
                    <span className="flex items-center gap-1.5">
                      <Radio className="w-3 h-3 text-cyan-400 animate-pulse" />
                      Feed: <strong className="text-gray-200">{data.data_provenance.source} Spot</strong>
                    </span>
                    <span>
                      Latency: <strong className="text-gray-200">{typeof data.data_provenance.age_seconds === 'number' ? `${data.data_provenance.age_seconds.toFixed(1)}s` : '0s'}</strong>
                    </span>
                  </div>
                )}

                {/* Decision Rationale */}
                <div className="p-3.5 rounded-lg bg-[#0b0e14] border border-gray-800 text-xs font-mono space-y-2">
                  <div className="text-gray-300 leading-relaxed font-sans">
                    {data.reason}
                  </div>
                  <div className="grid grid-cols-2 gap-2 pt-2 border-t border-gray-800/80 text-[11px]">
                    <div>
                      <span className="text-gray-500">Selected Strategy:</span>
                      <div className="text-cyan-300 font-bold">{data.candidate_entry.optimal_strategy_name?.split('—')[1]?.trim() || data.candidate_entry.optimal_strategy_name}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Preferred Entry:</span>
                      <div className="text-white font-bold">${data.candidate_entry.preferred.toFixed(4)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Entry Zone:</span>
                      <div className="text-white font-bold">${data.candidate_entry.lower.toFixed(4)} - ${data.candidate_entry.upper.toFixed(4)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Distance to Spot:</span>
                      <div className={`font-bold ${data.candidate_entry.is_current_price_in_zone ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {data.candidate_entry.distance_from_current_pct > 0 ? '+' : ''}{data.candidate_entry.distance_from_current_pct}%
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-500">TP1 / Invalidation:</span>
                      <div className="text-gray-200 font-bold">${data.target_forecasts[0]?.target.toFixed(4)} / ${data.invalidation.price.toFixed(4)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Net Expectancy:</span>
                      <div className="text-emerald-400 font-bold">+{data.expected_net_return}%</div>
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

                {/* Action Execution Button */}
                <button
                  onClick={handleExecuteTrade}
                  disabled={data.decision !== 'ENTER' || isExecuting}
                  className={`w-full py-2.5 px-4 rounded-lg font-mono font-bold text-xs flex items-center justify-center gap-2 transition ${
                    data.decision === 'ENTER' && !isExecuting
                      ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/30 cursor-pointer'
                      : 'bg-gray-800/60 text-gray-500 cursor-not-allowed border border-gray-700/40'
                  }`}
                >
                  {isExecuting ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
                      <span>Executing Optimal Trade...</span>
                    </>
                  ) : data.decision === 'ENTER' ? (
                    <>
                      <Zap className="w-4 h-4" />
                      <span>Execute Optimal Entry (Paper Spot)</span>
                    </>
                  ) : data.decision === 'WAIT' ? (
                    <>
                      <Clock className="w-4 h-4 text-cyan-400" />
                      <span>Wait for Optimal Entry Region (Locked)</span>
                    </>
                  ) : (
                    <>
                      <Lock className="w-4 h-4 text-rose-400" />
                      <span>Avoid: Downside Risk Exceeds Target (Locked)</span>
                    </>
                  )}
                </button>
              </div>

              {/* Calibration & Empirical Reliability Panel */}
              <div className="terminal-card p-4 space-y-3 font-mono text-xs">
                <div className="flex items-center justify-between border-b border-gray-800 pb-2">
                  <span className="font-bold text-gray-300 uppercase flex items-center gap-1.5">
                    <Award className="w-4 h-4 text-amber-400" />
                    Out-of-Sample Calibration
                  </span>
                  <span className="text-[10px] text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800 font-bold">
                    Brier: {data.calibration.brier_score}
                  </span>
                </div>

                <p className="text-[11px] text-gray-400 leading-relaxed font-sans">
                  Target-first probability calibration validated on walk-forward out-of-sample data across {data.calibration.sample_size_evaluated} historical setups.
                </p>

                <div className="space-y-1.5 pt-1">
                  <div className="flex justify-between text-[11px] text-gray-500 font-bold uppercase">
                    <span>Probability Bin</span>
                    <span>Observed Hit Rate</span>
                  </div>
                  {data.calibration.reliability_bins?.map((bin, i) => (
                    <div key={i} className="flex justify-between text-[11px] py-1 border-t border-gray-800/50">
                      <span className="text-gray-400">{bin.bin_range} (pred: {bin.predicted_avg}%)</span>
                      <span className="text-white font-bold">{bin.observed_hit_rate}% ({bin.samples} events)</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Research Governance Notice */}
          <div className="p-4 rounded-xl bg-[#0d121c] border border-gray-800 text-xs font-mono text-gray-400 flex items-start gap-3">
            <Info className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
            <div className="space-y-1">
              <span className="font-bold text-gray-200">BR-009 Scientific Governance Notice:</span>
              <p className="leading-relaxed">
                BR-009 operates on a strict 14-day chronological horizon with point-in-time causal features. 
                All probabilities are conditional first-passage survival estimates incorporating 0.40% round-trip friction. 
                A high probability does not override risk limits or authorize live execution without explicit user authorization.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 3. 6-Strategy Entry Benchmark View (When activeTab === 'benchmark') */}
      {activeTab === 'benchmark' && data && (
        <div className="space-y-5">
          <div className="terminal-card p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-gray-800 pb-3">
              <div>
                <h3 className="font-bold text-sm text-white font-mono flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-cyan-400" />
                  6-Strategy Entry Comparison Benchmark ({data.display_symbol})
                </h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Direct empirical comparison of 6 candidate entry strategies on identical current market structure and historical data.
                </p>
              </div>
              <span className="text-[11px] font-mono text-gray-400 bg-gray-900 px-2.5 py-1 rounded border border-gray-800">
                Current Spot: ${data.current_price.toFixed(4)}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 font-mono text-xs">
              {Object.entries(data.entry_strategy_benchmarks || {}).map(([key, strat]) => {
                const isOptimal = key === data.candidate_entry.optimal_strategy_key;
                return (
                  <div 
                    key={key}
                    className={`p-4 rounded-xl border flex flex-col justify-between space-y-3 ${
                      isOptimal 
                        ? 'bg-gradient-to-b from-[#102236] to-[#0c1422] border-cyan-500/80 shadow-lg shadow-cyan-950/40 ring-1 ring-cyan-500/50' 
                        : 'bg-[#0f1422] border-gray-800'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white text-xs">{strat.name}</span>
                        {isOptimal && (
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 font-bold border border-cyan-800 flex items-center gap-1">
                            <Sparkles className="w-3 h-3 text-cyan-400" />
                            OPTIMAL
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-gray-400 mt-1 leading-relaxed font-sans">
                        {strat.description}
                      </p>
                    </div>

                    <div className="space-y-1.5 py-2 border-t border-b border-gray-800/80 text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Entry Price:</span>
                        <span className="font-bold text-cyan-300">${strat.entry_price.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Distance to Spot:</span>
                        <span className="font-medium text-gray-200">{strat.distance_from_current_pct > 0 ? '+' : ''}{strat.distance_from_current_pct}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Risk Distance:</span>
                        <span className="text-rose-400 font-medium">-{strat.stop_distance_pct}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">TP1 Prob / R:R:</span>
                        <span className="text-white font-bold">{(strat.target_first_probability * 100).toFixed(1)}% (1:{strat.risk_reward_ratio})</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Max Adverse Excursion:</span>
                        <span className="text-amber-400 font-medium">{strat.max_adverse_excursion_pct}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Profit Factor:</span>
                        <span className="text-cyan-400 font-bold">{strat.profit_factor}</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-1">
                      <span className="text-[10px] text-gray-500 uppercase">Net Expectancy</span>
                      <span className={`text-sm font-bold ${strat.net_expectancy_pct > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {strat.net_expectancy_pct > 0 ? '+' : ''}{strat.net_expectancy_pct}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* 4. 14-Day Universe Screener View (When activeTab === 'scanner') */}
      {activeTab === 'scanner' && (
        <div className="terminal-card overflow-hidden">
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <div>
              <h3 className="font-bold text-sm text-white font-mono flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                14-Day Probabilistic Recovery Screener
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                Ranks all liquid universe assets by 14-day recovery net expectancy, depression depth, and optimal entry readiness.
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
                  <th className="py-3 px-4">Decision</th>
                  <th className="py-3 px-4 text-center">Depression</th>
                  <th className="py-3 px-4">Optimal Entry</th>
                  <th className="py-3 px-4">Entry Dist</th>
                  <th className="py-3 px-4">TP1 Prob</th>
                  <th className="py-3 px-4">Net Expectancy</th>
                  <th className="py-3 px-4">Strategy</th>
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
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                      {item.display_symbol}
                    </td>
                    <td className="py-3 px-4 font-mono font-medium text-gray-200">
                      ${item.current_price.toFixed(4)}
                    </td>
                    <td className="py-3 px-4">
                      {getDecisionBadge(item.decision, item.decision_badge)}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                        item.depression_score >= 70 ? 'bg-rose-950 text-rose-300 border border-rose-800' :
                        item.depression_score >= 45 ? 'bg-amber-950 text-amber-300 border border-amber-800' :
                        'bg-gray-900 text-gray-400'
                      }`}>
                        {item.depression_score}/100
                      </span>
                    </td>
                    <td className="py-3 px-4 text-cyan-300 font-bold">
                      ${item.preferred_entry.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 font-medium text-gray-300">
                      {item.entry_distance_pct > 0 ? '+' : ''}{item.entry_distance_pct}%
                    </td>
                    <td className="py-3 px-4 text-emerald-400 font-bold">
                      {item.tp1_prob}%
                    </td>
                    <td className={`py-3 px-4 font-bold ${item.net_expectancy_pct > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {item.net_expectancy_pct > 0 ? '+' : ''}{item.net_expectancy_pct}%
                    </td>
                    <td className="py-3 px-4 text-gray-400 text-[10px]">
                      {item.optimal_strategy?.split('—')[1]?.trim() || item.optimal_strategy}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button className="text-cyan-400 hover:text-cyan-300 font-bold flex items-center gap-1 ml-auto">
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
