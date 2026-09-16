import React, { useState, useEffect } from 'react';
import { X, ShieldAlert, DollarSign, Calculator, CheckCircle2 } from 'lucide-react';
import { api } from '../services/api';

export default function RiskCalculatorModal({ 
  isOpen, 
  onClose, 
  initialEquity = 40.0, 
  initialPrice = 100.0, 
  initialStop = 96.0,
  symbol = 'SOLUSDT',
  onConfirmOrder 
}) {
  const [equity, setEquity] = useState(initialEquity);
  const [riskPct, setRiskPct] = useState(2.0);
  const [entryPrice, setEntryPrice] = useState(initialPrice);
  const [stopLoss, setStopLoss] = useState(initialStop);
  const [sizingResult, setSizingResult] = useState(null);

  useEffect(() => {
    setEquity(initialEquity);
    setEntryPrice(initialPrice);
    setStopLoss(initialStop);
  }, [initialEquity, initialPrice, initialStop]);

  useEffect(() => {
    async function updateSizing() {
      if (entryPrice <= 0 || stopLoss <= 0 || stopLoss >= entryPrice) {
        setSizingResult(null);
        return;
      }
      try {
        const res = await api.calculateRisk(equity, riskPct, entryPrice, stopLoss);
        setSizingResult(res);
      } catch (e) {
        console.error("Risk calc error:", e);
      }
    }
    updateSizing();
  }, [equity, riskPct, entryPrice, stopLoss]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fadeIn">
      <div className="terminal-card w-full max-w-md p-5 bg-[#121722] border-[#2a364f] shadow-2xl relative">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1e2638] pb-3 mb-4">
          <div className="flex items-center gap-2">
            <Calculator className="w-5 h-5 text-blue-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Risk Engine Position Sizer
            </h3>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Inputs */}
        <div className="space-y-3.5 text-xs">
          
          {/* Account Equity & Risk % */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-gray-400 font-mono mb-1">ACCOUNT EQUITY ($)</label>
              <input
                type="number"
                step="1"
                value={equity}
                onChange={(e) => setEquity(parseFloat(e.target.value) || 40)}
                className="w-full bg-[#0b0e14] border border-[#1e2638] rounded p-2 text-white font-mono font-semibold focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-gray-400 font-mono mb-1">RISK / TRADE (%)</label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  step="0.1"
                  min="0.5"
                  max="5.0"
                  value={riskPct}
                  onChange={(e) => setRiskPct(parseFloat(e.target.value) || 2)}
                  className="w-full bg-[#0b0e14] border border-[#1e2638] rounded p-2 text-white font-mono font-semibold focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          </div>

          {/* Entry & Stop Loss */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-gray-400 font-mono mb-1">ENTRY PRICE ($)</label>
              <input
                type="number"
                step="any"
                value={entryPrice}
                onChange={(e) => setEntryPrice(parseFloat(e.target.value) || 0)}
                className="w-full bg-[#0b0e14] border border-[#1e2638] rounded p-2 text-white font-mono font-semibold focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-gray-400 font-mono mb-1">STOP LOSS ($)</label>
              <input
                type="number"
                step="any"
                value={stopLoss}
                onChange={(e) => setStopLoss(parseFloat(e.target.value) || 0)}
                className="w-full bg-[#0b0e14] border border-[#1e2638] rounded p-2 text-rose-400 font-mono font-semibold focus:outline-none focus:border-rose-500"
              />
            </div>
          </div>

          {/* Sizing Output Box */}
          {sizingResult && (
            <div className="bg-[#0b0e14] border border-[#1e2638] p-3 rounded space-y-2 font-mono-nums">
              <div className="flex justify-between text-gray-400">
                <span>Stop Distance:</span>
                <span className="text-rose-400 font-semibold">-{sizingResult.stop_distance_pct}%</span>
              </div>
              <div className="flex justify-between text-gray-400">
                <span>Total Dollar Risk:</span>
                <span className="text-white font-semibold">${sizingResult.risk_amount_usd.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-gray-300 pt-1.5 border-t border-gray-800">
                <span className="font-bold text-white">Recommended Position:</span>
                <span className="font-bold text-emerald-400 text-sm">
                  ${sizingResult.recommended_position_usd.toFixed(2)} USDT
                </span>
              </div>
              <div className="flex justify-between text-[11px] text-gray-500">
                <span>Calculated Quantity:</span>
                <span>{sizingResult.recommended_quantity} tokens</span>
              </div>
              {sizingResult.warning_or_notes && (
                <div className="text-[11px] text-amber-400/90 pt-1 font-sans leading-tight">
                  {sizingResult.warning_or_notes}
                </div>
              )}
            </div>
          )}

          {/* Formula Callout */}
          <div className="p-2 bg-blue-950/20 border border-blue-900/30 rounded text-[11px] text-blue-300 font-mono">
            Position Size = Risk Amount ($) / ((Entry - Stop) / Entry)
          </div>

          {/* Action Button */}
          <button
            onClick={() => {
              if (sizingResult && onConfirmOrder) {
                onConfirmOrder({
                  symbol: symbol,
                  entry_price: entryPrice,
                  stop_loss: stopLoss,
                  custom_notional_usd: sizingResult.recommended_position_usd
                });
                onClose();
              }
            }}
            disabled={!sizingResult || !sizingResult.is_valid}
            className="w-full py-2.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs shadow-lg shadow-emerald-600/30 flex items-center justify-center gap-2 transition disabled:opacity-50"
          >
            <CheckCircle2 className="w-4 h-4" />
            <span>Place Calculated Paper Order</span>
          </button>

        </div>
      </div>
    </div>
  );
}
