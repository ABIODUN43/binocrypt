import React from 'react';
import { Target, Shield, AlertTriangle, TrendingUp, Compass, Award } from 'lucide-react';

export default function ExperimentMilestone({ account }) {
  if (!account) return null;

  const current = account.current_equity;
  const start = account.starting_equity; // $40.00
  const targetMin = 150.0;
  const targetMax = 200.0;
  const targetMid = 175.0;

  // Percentage towards midpoint ($175)
  const progressPct = Math.max(0, Math.min(100, ((current - start) / (targetMid - start)) * 100));

  // Survival score: starts at 100, penalized by high drawdown
  const survivalScore = Math.max(0, 100 - (account.max_drawdown_pct * 4));

  return (
    <div className="terminal-card p-5 bg-gradient-to-br from-[#121722] to-[#0e131d] border-blue-900/30">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#1e2638] pb-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-amber-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              $40 &rarr; $150–$200 Research Experiment
            </h3>
          </div>
          <p className="text-xs text-gray-400 mt-0.5">
            Objective: Validate positive mathematical expectancy &amp; capital preservation without leverage
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="text-right">
            <div className="text-[10px] uppercase font-mono text-gray-400">SURVIVAL SCORE</div>
            <div className="text-sm font-bold font-mono-nums text-emerald-400">
              {survivalScore.toFixed(0)}/100
            </div>
          </div>
        </div>
      </div>

      {/* Progress Bar with Milestones */}
      <div className="space-y-2 mb-4">
        <div className="flex justify-between text-xs font-mono-nums">
          <span className="text-gray-400">Base: <strong>${start.toFixed(2)}</strong></span>
          <span className="text-blue-400 font-bold">Current: ${current.toFixed(2)} ({account.total_return_pct >= 0 ? '+' : ''}{account.total_return_pct.toFixed(1)}%)</span>
          <span className="text-amber-400 font-bold">Target: $150 – $200</span>
        </div>

        {/* Bar */}
        <div className="w-full bg-[#0b0e14] h-3.5 rounded-full p-0.5 border border-[#1e2638] overflow-hidden relative">
          <div 
            className="h-full bg-gradient-to-r from-blue-600 via-emerald-500 to-amber-500 rounded-full transition-all duration-500"
            style={{ width: `${Math.max(3, progressPct)}%` }}
          />
          {/* Milestone tick marks */}
          <div className="absolute top-0 bottom-0 left-[26%] w-0.5 bg-gray-600/60" title="$75 Milestone" />
          <div className="absolute top-0 bottom-0 left-[52%] w-0.5 bg-gray-600/60" title="$110 Milestone" />
          <div className="absolute top-0 bottom-0 left-[81%] w-0.5 bg-amber-500/80" title="$150 Target" />
        </div>

        <div className="flex justify-between text-[10px] font-mono text-gray-500">
          <span>$40.00</span>
          <span>$75 (Phase 1)</span>
          <span>$110 (Phase 2)</span>
          <span className="text-amber-400/80">$150 - $200</span>
        </div>
      </div>

      {/* Realistic Quant Reality Guardrails */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
        
        <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
          <div className="text-[10px] uppercase font-mono text-gray-400 flex items-center gap-1">
            <Compass className="w-3 h-3 text-blue-400" />
            <span>Compounding Reality</span>
          </div>
          <p className="text-gray-300 mt-1 leading-snug">
            3.75x–5.0x over 3 months requires <strong className="text-white font-mono">+55% to +71% monthly</strong>. The engine prioritizes survival over reckless chasing.
          </p>
        </div>

        <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
          <div className="text-[10px] uppercase font-mono text-gray-400 flex items-center gap-1">
            <Shield className="w-3 h-3 text-emerald-400" />
            <span>Small Account Armor</span>
          </div>
          <p className="text-gray-300 mt-1 leading-snug">
            Max risk strictly capped at <strong className="text-emerald-400 font-mono">1.5% - 2.0% ($0.80)</strong> per trade. Positions sized from stop distance.
          </p>
        </div>

        <div className="bg-[#0b0e14] border border-[#1e2638] p-2.5 rounded">
          <div className="text-[10px] uppercase font-mono text-gray-400 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3 text-amber-400" />
            <span>Circuit Breakers</span>
          </div>
          <p className="text-gray-300 mt-1 leading-snug">
            Trading automatically locks if daily drawdown exceeds <strong className="text-amber-400 font-mono">5% ($2.00)</strong> to prevent revenge trading.
          </p>
        </div>

      </div>

    </div>
  );
}
