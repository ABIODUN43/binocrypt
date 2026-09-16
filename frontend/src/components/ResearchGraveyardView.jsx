import React, { useState, useEffect } from 'react';
import { 
  Skull, 
  AlertTriangle, 
  ShieldAlert, 
  BookOpen, 
  Filter, 
  RefreshCw, 
  CheckCircle2, 
  XCircle, 
  Search, 
  FileText,
  Sliders,
  History
} from 'lucide-react';
import { api } from '../services/api';

export default function ResearchGraveyardView() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterFlaw, setFilterFlaw] = useState('ALL');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadGraveyard();
  }, []);

  const loadGraveyard = async () => {
    setLoading(true);
    try {
      const data = await api.getResearchGraveyard();
      if (data) {
        setEntries(data);
      }
    } catch (err) {
      console.error('Failed to load research graveyard:', err);
    } finally {
      setLoading(false);
    }
  };

  const FLAW_TYPES = [
    'ALL',
    'Zero-Cost Illusion & Overtrading',
    'Deep Overfitting & Probability Distortion',
    'Look-Ahead Bias & Bottom-Fishing Fallacy',
    'Timeframe Incompatibility & Spoofing Noise',
    'Stationary Indicator in Non-Stationary Regime'
  ];

  const filteredEntries = entries.filter((e) => {
    const matchesFilter = filterFlaw === 'ALL' || e.flaw_type === filterFlaw;
    const matchesSearch = 
      e.experiment_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.hypothesis.toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.flaw_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.lessons_learned.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 shrink-0 mt-0.5">
            <Skull className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-white">BR-001 Research Graveyard &amp; Rejection Catalog</h2>
              <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                ARCHIVED NEGATIVE FINDINGS
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl">
              A scientific institutional trading engine must catalog what <em>does not</em> work. This graveyard records rejected hypotheses, 
              fatal look-ahead leaks, and probability distortions to prevent repeat mistakes and survive genuine out-of-sample markets.
            </p>
          </div>
        </div>

        <button
          onClick={loadGraveyard}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors border border-slate-700 self-start md:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Catalog
        </button>
      </div>

      {/* Core Scientific Integrity Audit Bar */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
        <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-emerald-400" />
          Anti-Fraud &amp; Zero-Leakage Scientific Principles
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold mb-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Zero Look-Ahead Bias
            </div>
            <p className="text-[11px] text-slate-400">
              Features at time t strictly use observations at or before t. Target returns are completely quarantined.
            </p>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold mb-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Strict Net Cost Gating
            </div>
            <p className="text-[11px] text-slate-400">
              Signals require EV_net &gt; 0 after 0.40% roundtrip taker fee, spread, and slippage deductions.
            </p>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold mb-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Calibrated Probabilities
            </div>
            <p className="text-[11px] text-slate-400">
              Raw scores are scaled via Platt regression; Brier Score &lt; 0.20 and ECE &lt; 0.08 are required.
            </p>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold mb-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Probabilistic Framing
            </div>
            <p className="text-[11px] text-slate-400">
              Never claims guaranteed profits, certainty, or exact future dates. Always expresses model confidence ranges.
            </p>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-2 bg-slate-950/60 border border-slate-800 rounded-lg px-3 py-1.5">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={filterFlaw}
              onChange={(e) => setFilterFlaw(e.target.value)}
              className="bg-transparent text-xs text-slate-200 focus:outline-none cursor-pointer"
            >
              {FLAW_TYPES.map((f) => (
                <option key={f} value={f} className="bg-slate-900 text-white">
                  {f === 'ALL' ? 'All Flaw Types' : f}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="relative w-full md:w-64">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search graveyard entries..."
            className="w-full bg-slate-950/60 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-slate-700"
          />
        </div>
      </div>

      {/* Graveyard Cards */}
      {loading ? (
        <div className="p-12 text-center text-slate-500 bg-slate-900/40 rounded-xl border border-slate-800 flex items-center justify-center gap-3">
          <RefreshCw className="w-5 h-5 animate-spin text-rose-400" />
          <span>Loading archived failure post-mortems...</span>
        </div>
      ) : filteredEntries.length === 0 ? (
        <div className="p-8 text-center text-slate-500 bg-slate-900/40 rounded-xl border border-slate-800 text-xs">
          No graveyard records match the selected filter.
        </div>
      ) : (
        <div className="space-y-4">
          {filteredEntries.map((entry) => (
            <div 
              key={entry.id || entry.experiment_id}
              className="bg-slate-900/60 border border-slate-800 hover:border-slate-700 rounded-xl p-5 transition-colors space-y-3"
            >
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
                <div className="flex items-center gap-3">
                  <div className="px-2.5 py-1 rounded bg-rose-500/20 text-rose-400 font-mono font-bold text-xs border border-rose-500/30">
                    {entry.experiment_id}
                  </div>
                  <span className="px-2.5 py-0.5 rounded bg-amber-500/10 text-amber-400 text-xs font-medium border border-amber-500/20">
                    {entry.flaw_type}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-slate-500">
                  Archived: {entry.archived_at}
                </div>
              </div>

              {/* Hypothesis */}
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Original Hypothesis:</div>
                <p className="text-xs text-slate-200 mt-0.5 font-medium leading-relaxed">
                  "{entry.hypothesis}"
                </p>
              </div>

              {/* Rejection Reason & Post-Mortem */}
              <div className="bg-rose-950/20 border border-rose-500/20 rounded-lg p-3">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-rose-400 mb-1">
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                  Post-Mortem &amp; Empirical Rejection Rationale:
                </div>
                <p className="text-xs text-rose-300 leading-relaxed">
                  {entry.rejection_reason}
                </p>
              </div>

              {/* Lessons Learned */}
              <div className="bg-emerald-950/20 border border-emerald-500/20 rounded-lg p-3">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400 mb-1">
                  <BookOpen className="w-3.5 h-3.5 shrink-0" />
                  Permanent System Rule &amp; Safeguard Implemented:
                </div>
                <p className="text-xs text-emerald-300 leading-relaxed">
                  {entry.lessons_learned}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
