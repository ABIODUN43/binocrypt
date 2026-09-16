import React, { useState, useEffect, useCallback, useRef } from 'react';
import Navbar from './components/Navbar';
import MarketRegimeBanner from './components/MarketRegimeBanner';
import OpportunityTable from './components/OpportunityTable';
import SetupSpotlightCard from './components/SetupSpotlightCard';
import CandlestickChart from './components/CandlestickChart';
import MultiTimeframeGrid from './components/MultiTimeframeGrid';
import PaperPortfolioView from './components/PaperPortfolioView';
import BacktestLabView from './components/BacktestLabView';
import ExperimentMilestone from './components/ExperimentMilestone';
import RiskCalculatorModal from './components/RiskCalculatorModal';
import AIExplanationModal from './components/AIExplanationModal';
import AssetIntelligenceView from './components/AssetIntelligenceView';
import RecoveryScannerView from './components/RecoveryScannerView';
import PredictionHistoryView from './components/PredictionHistoryView';
import QuantResearchLabView from './components/QuantResearchLabView';
import ResearchGraveyardView from './components/ResearchGraveyardView';

import { api } from './services/api';
import { RefreshCw, AlertCircle, CheckCircle2 } from 'lucide-react';

// Helper to immediately create trade setup from opportunity (0ms instant response)
function createSetupFromOpportunity(opp) {
  if (!opp) return null;
  const current_price = opp.price || 1.0;
  const stop_distance_pct = 2.5; // calibrated 2.5% structural stop
  const stop_loss = Number((current_price * (1 - stop_distance_pct / 100)).toFixed(current_price < 1 ? 4 : 2));
  const risk_dist = current_price - stop_loss;
  const tp1 = Number((current_price + (risk_dist * 1.8)).toFixed(current_price < 1 ? 4 : 2));
  const tp2 = Number((current_price + (risk_dist * 3.0)).toFixed(current_price < 1 ? 4 : 2));

  return {
    symbol: opp.symbol,
    direction: "LONG",
    signal_state: opp.total_score >= 80 ? "READY" : "SETUP_FORMING",
    score: opp.total_score,
    grade: opp.grade,
    entry_zone_min: Number((current_price * 0.997).toFixed(current_price < 1 ? 4 : 2)),
    entry_zone_max: Number((current_price * 1.002).toFixed(current_price < 1 ? 4 : 2)),
    current_price: current_price,
    stop_loss: stop_loss,
    tp1: tp1,
    tp2: tp2,
    risk_reward: 1.81,
    stop_distance_pct: stop_distance_pct,
    invalidation: `1H candle close below $${stop_loss < 1 ? stop_loss.toFixed(4) : stop_loss.toFixed(2)} (${stop_distance_pct}% from entry)`,
    setup_name: opp.total_score >= 80 ? "High-Confluence Trend Continuation" : "Momentum Breakout & Pullback",
    confidence_pct: Math.round((opp.win_probability || 0.72) * 100),
    rationale_bullets: [
      `Opportunity Score ${opp.total_score}/100 with ${opp.grade} rating`,
      `Trend score ${opp.trend_score}/20, Momentum ${opp.momentum_score}/20`,
      `Volume score ${opp.volume_score}/20 with 24h volume of $${(opp.volume_24h / 1_000_000).toFixed(1)}M`,
      `Structure score ${opp.structure_score}/20 with 1:1.81 R:R target profile`
    ]
  };
}

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  // Persistent selected symbol from localStorage so page reload never wipes user choice!
  const getInitialSymbol = () => {
    try {
      return localStorage.getItem('binocrypt_selected_symbol') || '';
    } catch {
      return '';
    }
  };
  
  // Data States
  const [regime, setRegime] = useState(null);
  const [opportunities, setOpportunities] = useState([]);
  const [activeSetups, setActiveSetups] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState(getInitialSymbol);
  const [currentSetup, setCurrentSetup] = useState(null);
  const [account, setAccount] = useState(null);
  const [positions, setPositions] = useState([]);
  const [history, setHistory] = useState([]);

  // Ref to track the current selectedSymbol without triggering useCallback recreation
  const selectedSymbolRef = useRef(selectedSymbol);
  useEffect(() => {
    selectedSymbolRef.current = selectedSymbol;
  }, [selectedSymbol]);

  // UI States
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);
  
  // Modals
  const [isRiskModalOpen, setIsRiskModalOpen] = useState(false);
  const [explainSymbol, setExplainSymbol] = useState(null);

  const showToast = (text, type = 'success') => {
    setToastMessage({ text, type });
    setTimeout(() => setToastMessage(null), 4000);
  };

  // Master Data Refresh - STABLE: empty dependency array, never recreated on setup or selection change!
  const loadAllData = useCallback(async (force = false) => {
    setIsRefreshing(true);
    try {
      const [regimeRes, oppsRes, accountRes, posRes, histRes] = await Promise.all([
        api.getRegime().catch(() => null),
        api.getOpportunities(30, 0, force).catch(() => []),
        api.getPaperAccount().catch(() => null),
        api.getPaperPositions().catch(() => []),
        api.getPaperHistory().catch(() => [])
      ]);

      if (regimeRes) setRegime(regimeRes);
      if (oppsRes && oppsRes.length > 0) {
        setOpportunities(oppsRes);
        
        // Only initialize selectedSymbol if nothing is currently selected (first load)
        if (!selectedSymbolRef.current) {
          const saved = localStorage.getItem('binocrypt_selected_symbol');
          const defaultSym = (saved && oppsRes.some(o => o.symbol === saved)) ? saved : oppsRes[0].symbol;
          setSelectedSymbol(defaultSym);
          try { localStorage.setItem('binocrypt_selected_symbol', defaultSym); } catch (e) {}
        }
      }
      if (accountRes) setAccount(accountRes);
      if (posRes) setPositions(posRes);
      if (histRes) setHistory(histRes);

    } catch (err) {
      console.error("Data refresh error:", err);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  // Initial load & 30s polling
  useEffect(() => {
    loadAllData(false);
    const timer = setInterval(() => {
      loadAllData(false);
    }, 30000);
    return () => clearInterval(timer);
  }, [loadAllData]);

  // SINGLE SOURCE OF TRUTH for currentSetup: responds ONLY when selectedSymbol changes
  useEffect(() => {
    if (!selectedSymbol) return;
    let isCurrent = true;

    // 1. Instantly set setup from opportunities in memory (0ms, no network delay)
    const matched = opportunities.find(o => o.symbol === selectedSymbol);
    if (matched) {
      setCurrentSetup(createSetupFromOpportunity(matched));
    }

    // 2. Fetch enriched deep calculation with exact pivots from backend
    api.getSymbolSetup(selectedSymbol).then(s => {
      // ONLY apply if user is STILL on this exact symbol! (prevents stale responses from previous coin)
      if (isCurrent && s && s.symbol === selectedSymbol) {
        setCurrentSetup(s);
      }
    }).catch(err => {
      console.warn("Using instant opportunity setup for", selectedSymbol);
    });

    return () => {
      isCurrent = false; // Discard any pending response if user clicks another coin
    };
  }, [selectedSymbol]);

  // Fallback: if opportunities arrived on first boot when currentSetup was null
  useEffect(() => {
    if (!currentSetup && selectedSymbol && opportunities.length > 0) {
      const matched = opportunities.find(o => o.symbol === selectedSymbol) || opportunities[0];
      if (matched) {
        setCurrentSetup(createSetupFromOpportunity(matched));
      }
    }
  }, [opportunities, selectedSymbol, currentSetup]);

  // Handle Quick Paper Trade
  const handleQuickPaperTrade = async (symbol) => {
    try {
      const res = await api.placePaperOrder({ symbol: symbol });
      if (res.success) {
        showToast(`Paper order filled: ${res.symbol} ($${res.notional_usd.toFixed(2)} USDT)`);
        loadAllData(false);
      }
    } catch (err) {
      showToast(err.message || 'Order failed to execute', 'error');
    }
  };

  // Handle Setup Card Paper Trade
  const handleExecuteSetupTrade = async (setup) => {
    try {
      const res = await api.placePaperOrder({
        symbol: setup.symbol,
        entry_price: setup.current_price,
        stop_loss: setup.stop_loss,
        tp1: setup.tp1,
        tp2: setup.tp2
      });
      if (res.success) {
        showToast(`Paper order filled: ${res.symbol} @ $${res.entry_price} ($${res.notional_usd.toFixed(2)} USDT)`);
        loadAllData(false);
      }
    } catch (err) {
      showToast(err.message || 'Order failed to execute', 'error');
    }
  };

  // Handle Manual Close
  const handleClosePosition = async (posId) => {
    try {
      const res = await api.closePaperPosition(posId);
      if (res.success) {
        showToast(`Position closed: Net P/L $${res.net_pnl_usd.toFixed(2)} (${res.pnl_pct.toFixed(2)}%)`);
        loadAllData(false);
      }
    } catch (err) {
      showToast(err.message || 'Failed to close position', 'error');
    }
  };

  // Handle Reset Account
  const handleResetAccount = async (capital = 40.0) => {
    try {
      await api.resetPaperAccount(capital);
      showToast(`Account reset to $${capital.toFixed(2)}`);
      loadAllData(false);
    } catch (err) {
      showToast(err.message || 'Failed to reset account', 'error');
    }
  };

  // Handle Coin Selection (switch spotlight card & chart immediately, scroll to view)
  const handleSelectCoin = (symbol) => {
    if (!symbol || symbol === selectedSymbol) return;
    setSelectedSymbol(symbol);
    try {
      localStorage.setItem('binocrypt_selected_symbol', symbol);
    } catch (e) {}

    // Instant 0ms optimistic UI update from existing opportunity data
    const matchedOpp = opportunities.find(o => o.symbol === symbol);
    if (matchedOpp) {
      setCurrentSetup(createSetupFromOpportunity(matchedOpp));
    }

    // Smooth scroll up to the spotlight card & chart
    window.scrollTo({ top: 120, behavior: 'smooth' });
  };

  const selectedOpportunity = opportunities.find(o => o.symbol === selectedSymbol);

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0e14] text-[#f8fafc]">
      
      {/* Toast Notification */}
      {toastMessage && (
        <div className={`fixed bottom-5 right-5 z-50 px-4 py-2.5 rounded-lg shadow-xl text-xs font-semibold flex items-center gap-2 border animate-bounce ${
          toastMessage.type === 'error'
            ? 'bg-rose-950 text-rose-300 border-rose-800'
            : 'bg-emerald-950 text-emerald-300 border-emerald-800'
        }`}>
          {toastMessage.type === 'error' ? <AlertCircle className="w-4 h-4 text-rose-400" /> : <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
          <span>{toastMessage.text}</span>
        </div>
      )}

      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        regime={regime}
        account={account}
        onRefresh={() => loadAllData(true)}
        isRefreshing={isRefreshing}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-5 space-y-5">
        
        {/* Market Regime Banner (Persistent Across Terminal and Scanner) */}
        {(activeTab === 'dashboard' || activeTab === 'scanner') && (
          <MarketRegimeBanner regime={regime} />
        )}

        {/* 1. Dashboard / Terminal View */}
        {activeTab === 'dashboard' && (
          <div className="space-y-5">
            {/* Top Row: Spotlight Setup Card & Candlestick Chart */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-5">
                <SetupSpotlightCard
                  setup={currentSetup}
                  onExecutePaperTrade={handleExecuteSetupTrade}
                  onOpenExplain={(sym) => setExplainSymbol(sym)}
                  account={account}
                  opportunities={opportunities}
                  onSelectSymbol={handleSelectCoin}
                />
              </div>

              <div className="lg:col-span-7">
                <CandlestickChart
                  symbol={selectedSymbol}
                  setup={currentSetup}
                />
              </div>
            </div>

            {/* $40 Experiment Milestone Widget */}
            <ExperimentMilestone account={account} />

            {/* Top Opportunities Scanner Grid */}
            <OpportunityTable
              opportunities={opportunities}
              onSelectCoin={handleSelectCoin}
              onOpenExplain={(sym) => setExplainSymbol(sym)}
              onQuickPaperTrade={handleQuickPaperTrade}
              selectedSymbol={selectedSymbol}
            />
          </div>
        )}

        {/* 2. Market Intelligence & Decision Engine Tab */}
        {activeTab === 'intelligence' && (
          <AssetIntelligenceView
            selectedSymbol={selectedSymbol}
            onSelectCoin={handleSelectCoin}
            opportunities={opportunities}
            onExecuteTrade={handleQuickPaperTrade}
          />
        )}

        {/* 3. Bear Exhaustion & Recovery Scanner Tab */}
        {activeTab === 'recovery' && (
          <RecoveryScannerView
            onSelectCoin={handleSelectCoin}
            onNavigateToSetup={() => {
              setActiveTab('intelligence');
              window.scrollTo({ top: 120, behavior: 'smooth' });
            }}
          />
        )}

        {/* 4. Auditable Prediction Track Record Tab */}
        {activeTab === 'history' && (
          <PredictionHistoryView
            symbol={selectedSymbol}
            onSelectCoin={(sym) => {
              handleSelectCoin(sym);
              setActiveTab('intelligence');
              window.scrollTo({ top: 120, behavior: 'smooth' });
            }}
          />
        )}

        {/* 5. Quant Research Lab Tab (BR-001) */}
        {activeTab === 'quantlab' && (
          <QuantResearchLabView selectedSymbol={selectedSymbol || 'ARB/USDT'} />
        )}

        {/* 6. Research Graveyard Tab */}
        {activeTab === 'graveyard' && (
          <ResearchGraveyardView />
        )}

        {/* 7. Paper Trading Portfolio Tab */}
        {activeTab === 'paper' && (
          <div className="space-y-5">
            <ExperimentMilestone account={account} />
            <PaperPortfolioView
              account={account}
              positions={positions}
              history={history}
              onClosePosition={handleClosePosition}
              onResetAccount={handleResetAccount}
            />
          </div>
        )}

        {/* 5. Backtesting Lab Tab */}
        {activeTab === 'backtest' && (
          <div className="space-y-5">
            <BacktestLabView />
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-[#1e2638] py-4 bg-[#0b0e14] text-center text-xs text-gray-500">
        <p>Binocrypt Spot Decision Support System • Real-Money Execution Disabled by Default</p>
      </footer>

      {/* Risk Engine Modal */}
      {isRiskModalOpen && (
        <RiskCalculatorModal
          isOpen={isRiskModalOpen}
          onClose={() => setIsRiskModalOpen(false)}
          initialEquity={account ? account.current_equity : 40.0}
          initialPrice={currentSetup ? currentSetup.current_price : 100.0}
          initialStop={currentSetup ? currentSetup.stop_loss : 96.0}
          symbol={selectedSymbol}
          onConfirmOrder={(order) => {
            api.placePaperOrder(order).then((res) => {
              if (res.success) {
                showToast(`Paper order filled: ${res.symbol} ($${res.notional_usd.toFixed(2)} USDT)`);
                loadAllData(false);
              }
            }).catch((err) => {
              showToast(err.message || 'Order failed', 'error');
            });
          }}
        />
      )}

      {/* AI Explanation Modal */}
      {explainSymbol && (
        <AIExplanationModal
          isOpen={!!explainSymbol}
          onClose={() => setExplainSymbol(null)}
          symbol={explainSymbol}
        />
      )}

    </div>
  );
}
