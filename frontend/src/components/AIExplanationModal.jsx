import React, { useState, useEffect } from 'react';
import { X, Sparkles, AlertTriangle, ShieldCheck, CheckCircle2, RefreshCw } from 'lucide-react';
import { api } from '../services/api';

export default function AIExplanationModal({ isOpen, onClose, symbol }) {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    async function loadExplanation() {
      if (!isOpen || !symbol) return;
      setIsLoading(true);
      setError(null);
      try {
        const res = await api.getExplanation(symbol);
        if (isMounted) setData(res);
      } catch (err) {
        if (isMounted) setError(err.message || 'Failed to synthesize explanation');
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }
    loadExplanation();
    return () => { isMounted = false; };
  }, [isOpen, symbol]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fadeIn">
      <div className="terminal-card w-full max-w-2xl max-h-[90vh] overflow-y-auto p-5 sm:p-6 bg-[#121722] border-purple-900/40 shadow-2xl relative">
        
        {/* Header */}
        <div className="flex items-start justify-between border-b border-[#1e2638] pb-4 mb-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-purple-900/50 border border-purple-700/50 flex items-center justify-center text-purple-400">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Quantitative Thesis &amp; AI Synthesis
              </h3>
              <p className="text-xs text-gray-400">
                Ground-truth mathematical synthesis • Zero hallucinations
              </p>
            </div>
          </div>

          <button onClick={onClose} className="text-gray-400 hover:text-white transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        {isLoading ? (
          <div className="py-16 flex flex-col items-center justify-center text-center">
            <RefreshCw className="w-8 h-8 text-purple-400 animate-spin mb-3" />
            <p className="text-xs text-gray-400">Analyzing indicator matrix and market structure for {symbol}...</p>
          </div>
        ) : error ? (
          <div className="py-8 text-center text-rose-400 text-xs">
            <AlertTriangle className="w-6 h-6 mx-auto mb-2" />
            {error}
          </div>
        ) : data ? (
          <div className="space-y-4 text-xs">
            
            {/* Title & Regime Alignment */}
            <div className="bg-[#0b0e14] border border-[#1e2638] p-3 rounded">
              <div className="font-bold text-white text-sm">{data.summary_title}</div>
              <div className="text-gray-400 mt-1 flex items-center gap-2">
                <span className="font-mono text-emerald-400 font-semibold">{data.regime_alignment}</span>
              </div>
            </div>

            {/* Why it ranked high */}
            <div className="space-y-2">
              <h4 className="font-mono font-bold text-purple-300 uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-purple-400" />
                <span>Why This Asset Ranked in Top Tier</span>
              </h4>
              <div className="bg-[#0e121a] border border-[#1e2638] p-3 rounded space-y-2">
                {data.why_ranked_high.map((item, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-gray-300">
                    <span className="text-emerald-400 font-bold">•</span>
                    <span className="leading-relaxed">{item}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Technical Triggers */}
            <div className="space-y-2">
              <h4 className="font-mono font-bold text-blue-300 uppercase tracking-wider text-[11px]">
                Exact Technical Triggers &amp; Execution Zone
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {data.technical_triggers.map((item, idx) => (
                  <div key={idx} className="bg-[#0b0e14] border border-[#1e2638] p-2 rounded text-gray-300 font-mono text-[11px]">
                    {item}
                  </div>
                ))}
              </div>
            </div>

            {/* Invalidation & Risk Factors */}
            <div className="space-y-2">
              <h4 className="font-mono font-bold text-rose-300 uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                <span>Invalidation &amp; Risk Guardrails</span>
              </h4>
              <div className="bg-rose-950/20 border border-rose-900/40 p-3 rounded space-y-1.5">
                {data.invalidation_and_risks.map((item, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-rose-200">
                    <span className="text-rose-400 font-bold">•</span>
                    <span className="leading-relaxed">{item}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Small Account Verdict */}
            <div className="bg-blue-950/30 border border-blue-800/50 p-3.5 rounded">
              <div className="text-[11px] uppercase font-mono font-bold text-blue-300 flex items-center gap-1.5 mb-1">
                <ShieldCheck className="w-4 h-4 text-blue-400" />
                <span>Small Account Execution Verdict ($40 Target)</span>
              </div>
              <p className="text-gray-200 leading-relaxed">
                {data.small_account_verdict}
              </p>
            </div>

          </div>
        ) : null}

        <div className="mt-5 pt-3 border-t border-[#1e2638] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded bg-gray-800 hover:bg-gray-700 text-white text-xs font-semibold"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
}
