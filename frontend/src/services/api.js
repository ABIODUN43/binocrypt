const RAW_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '');
const BASE_URL = RAW_BASE ? (RAW_BASE.endsWith('/api') ? RAW_BASE : `${RAW_BASE}/api`) : '/api';

export async function apiRequest(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {})
  };
  
  try {
    const response = await fetch(url, { ...options, headers });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Request failed with status ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.error(`API Error on ${endpoint}:`, err);
    throw err;
  }
}

export const api = {
  // Market Data & Regime
  getRegime: () => apiRequest('/market/regime'),
  getTicker: (symbol) => apiRequest(`/market/ticker/${symbol}`),
  getKlines: (symbol, interval = '1h', limit = 100) => 
    apiRequest(`/market/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`),
  getDepth: (symbol) => apiRequest(`/market/depth/${symbol}`),

  // Scanner
  getUniverse: () => apiRequest('/scanner/universe'),
  getOpportunities: (limit = 25, minScore = 0, forceRefresh = false) => 
    apiRequest(`/scanner/opportunities?limit=${limit}&min_score=${minScore}&force_refresh=${forceRefresh}`),

  // Setups
  getActiveSetups: (limit = 5) => apiRequest(`/setups/active?limit=${limit}`),
  getSymbolSetup: (symbol) => apiRequest(`/setups/${symbol}`),

  // Risk Engine
  calculateRisk: (accountEquity, riskPct, entryPrice, stopLoss) => 
    apiRequest('/risk/calculate', {
      method: 'POST',
      body: JSON.stringify({
        account_equity: accountEquity,
        risk_pct: riskPct,
        entry_price: entryPrice,
        stop_loss: stopLoss
      })
    }),
  getRiskRules: () => apiRequest('/risk/rules'),

  // Paper Trading
  getPaperAccount: () => apiRequest('/paper/account'),
  getPaperPositions: () => apiRequest('/paper/positions'),
  getPaperHistory: () => apiRequest('/paper/history'),
  placePaperOrder: (orderData) => apiRequest('/paper/order', {
    method: 'POST',
    body: JSON.stringify(orderData)
  }),
  closePaperPosition: (positionId) => apiRequest(`/paper/close/${positionId}`, {
    method: 'POST'
  }),
  resetPaperAccount: (capital = 40.0) => apiRequest(`/paper/reset?capital=${capital}`, {
    method: 'POST'
  }),

  // Backtester
  runBacktest: (params) => apiRequest('/backtest/run', {
    method: 'POST',
    body: JSON.stringify(params)
  }),

  // AI Quantitative Explainer
  getExplanation: (symbol) => apiRequest(`/explain/${symbol}`),

  // Market Intelligence & Regimes
  getAssetIntelligence: (symbol) => apiRequest(`/intelligence/${symbol}`),
  getTopRecoveryOpportunities: (limit = 10) => apiRequest(`/intelligence/scanner/recovery?limit=${limit}`),
  getPredictionHistory: (symbol) => apiRequest(`/intelligence/history/${symbol}`),

  // BR-001 Quantitative Research Blueprint
  getResearchExperiments: () => apiRequest('/research/experiments'),
  getExperimentDetail: (id) => apiRequest(`/research/experiments/${id}`),
  runAblationLadder: (symbol = 'ARB/USDT', horizon = 14, targetLabel = 'R-C') => 
    apiRequest(`/research/run-ladder?symbol=${encodeURIComponent(symbol)}&target_horizon=${horizon}&target_label=${targetLabel}`, { method: 'POST' }),
  getResearchGraveyard: () => apiRequest('/research/graveyard'),
  getHMMState: (symbol) => apiRequest(`/research/hmm/${encodeURIComponent(symbol)}`),
  getLargeMoveSurface: (symbol) => apiRequest(`/research/large-move/${encodeURIComponent(symbol)}`),
  getEntryStrategies: (symbol) => apiRequest(`/research/entry-strategies/${encodeURIComponent(symbol)}`),
  getFalseRecoveryRisk: (symbol) => apiRequest(`/research/false-recovery/${encodeURIComponent(symbol)}`),
  getUniverseRanking: (limit = 10) => apiRequest(`/research/ranking?limit=${limit}`)
};

