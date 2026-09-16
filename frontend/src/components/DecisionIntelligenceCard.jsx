import React from 'react';
import { 
  CheckCircle2, Clock, AlertTriangle, ShieldCheck, 
  HelpCircle, ArrowRight, Activity, Zap, Info 
} from 'lucide-react';

const DECISION_STYLES = {
  ENTER: { badge: 'bg-emerald-500 text-black border-emerald-400', banner: 'bg-emerald-950/40 border-emerald-800' },
  ACCUMULATE: { badge: 'bg-teal-500 text-black border-teal-400', banner: 'bg-teal-950/40 border-teal-800' },
  WATCH: { badge: 'bg-amber-500 text-black border-amber-400', banner: 'bg-amber-950/40 border-amber-800' },
  WAIT: { badge: 'bg-gray-700 text-gray-200 border-gray-600', banner: 'bg-gray-900/40 border-gray-800' },
  HOLD: { badge: 'bg-blue-500 text-black border-blue-400', banner: 'bg-blue-950/40 border-blue-800' },
  REDUCE: { badge: 'bg-orange-500 text-black border-orange-400', banner: 'bg-orange-950/40 border-orange-800' },
  TAKE_PROFIT: { badge: 'bg-rose-500 text-black border-rose-400', banner: 'bg-rose-950/40 border-rose-800' },
  EXIT: { badge: 'bg-red-700 text-white border-red-600', banner: 'bg-red-950/40 border-red-800' },
  AVOID: { badge: 'bg-red-900 text-red-200 border-red-800', banner: 'bg-red-950/50 border-red-900' }
};

export default function DecisionIntelligenceCard({ intelligence }) {
  if (!intelligence) return null;

  const decisionKey = intelligence.decision.toUpperCase();
  const style = DECISION_STYLES[decisionKey] || DECISION_STYLES.WAIT;

  return (
    <div className="terminal-card flex flex-col justify-between">
      <div>
        {/* Top Header */}
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold text-gray-200 tracking-wider uppercase">
              Decision Support Engine
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-gray-400 font-mono">Ensemble Agreement:</span>
            <span className="px-1.5 py-0.5 rounded text-[11px] font-mono font-bold bg-gray-900 text-cyan-300 border border-gray-700">
              {intelligence.model_agreement_pct.toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Big Recommendation Banner */}
        <div className={`p-3.5 rounded-xl border mb-4 ${style.banner}`}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
            <div>
              <span className="text-[10px] text-gray-400 uppercase font-mono tracking-wider">
                Action Recommendation
              </span>
              <div className="text-sm font-bold text-white mt-0.5">
                {intelligence.decision_summary}
              </div>
            </div>

            <span className={`self-start sm:self-center px-3 py-1 rounded-md text-xs font-black tracking-wider uppercase font-mono border shadow-md ${style.badge}`}>
              {intelligence.decision_badge}
            </span>
          </div>

          {/* Model Forecast Windows */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-2.5 mt-2 border-t border-gray-800/80">
            <div className="p-2 rounded bg-gray-950/60 border border-gray-800/60 font-mono">
              <div className="text-[10px] text-gray-400 uppercase tracking-wider">
                Estimated Transition Window
              </div>
              <div className="text-xs font-bold text-cyan-300 mt-0.5">
                {intelligence.transition_window}
              </div>
            </div>

            <div className="p-2 rounded bg-gray-950/60 border border-gray-800/60 font-mono">
              <div className="text-[10px] text-gray-400 uppercase tracking-wider">
                Potential Accumulation Window
              </div>
              <div className="text-xs font-bold text-emerald-300 mt-0.5">
                {intelligence.accumulation_window}
              </div>
            </div>
          </div>
        </div>

        {/* 5-Point Confirmation Checklist */}
        <div className="mb-4">
          <div className="text-[11px] font-bold text-gray-300 uppercase tracking-wider font-mono mb-2 flex items-center justify-between">
            <span>Entry & Confirmation Criteria</span>
            <span className="text-[10px] text-gray-400 lowercase font-normal">
              ({intelligence.triggers_checklist.filter(t => t.satisfied).length} of {intelligence.triggers_checklist.length} satisfied)
            </span>
          </div>

          <div className="space-y-1.5">
            {intelligence.triggers_checklist.map((t, idx) => (
              <div 
                key={idx} 
                className={`p-2 rounded-lg border text-xs flex items-center justify-between gap-2 ${
                  t.satisfied 
                    ? 'bg-emerald-950/20 border-emerald-900/50 text-emerald-200' 
                    : 'bg-gray-950/40 border-gray-800/60 text-gray-400'
                }`}
              >
                <div className="flex items-center gap-2">
                  {t.satisfied ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  ) : (
                    <Clock className="w-4 h-4 text-amber-500/80 shrink-0" />
                  )}
                  <span className="font-mono">{t.label}</span>
                </div>
                <span className="text-[10px] font-mono text-gray-400 truncate max-w-[180px]">
                  {t.detail}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Structured Evidence & Risk Breakdown (Anti-Hallucination) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3 border-t border-gray-800/80 font-mono">
          {/* Evidence */}
          <div className="space-y-2">
            <div className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Quantitative Supporting Evidence</span>
            </div>
            <ul className="space-y-1.5 text-xs text-gray-300">
              {intelligence.evidence_bullets.map((e, idx) => (
                <li key={idx} className="flex items-start gap-1.5 leading-tight">
                  <span className="text-emerald-400 shrink-0 mt-0.5">✓</span>
                  <span>{e}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Risks */}
          <div className="space-y-2">
            <div className="text-[10px] font-bold text-rose-400 uppercase tracking-wider flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Model Invalidation & Risk Factors</span>
            </div>
            <ul className="space-y-1.5 text-xs text-gray-300">
              {intelligence.risk_warnings.map((r, idx) => (
                <li key={idx} className="flex items-start gap-1.5 leading-tight">
                  <span className="text-rose-400 shrink-0 mt-0.5">⚠</span>
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Footer Model Accuracy Stats */}
      <div className="pt-3 mt-4 border-t border-gray-800/80 flex flex-wrap items-center justify-between text-[10px] text-gray-400 font-mono gap-2">
        <div className="flex items-center gap-3">
          <span>Historical Directional Accuracy: <strong className="text-white">{intelligence.historical_directional_accuracy_pct}%</strong></span>
          <span>Recovery Accuracy: <strong className="text-white">{intelligence.historical_recovery_accuracy_pct}%</strong></span>
        </div>
        <span>Confidence: <strong className="text-cyan-300">{intelligence.prediction_confidence_pct}%</strong></span>
      </div>
    </div>
  );
}
