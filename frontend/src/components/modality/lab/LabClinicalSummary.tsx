import React, { useState } from "react";
import { ClinicalSummary, ClinicalPanelSection, ClinicalFindingItem } from "@/lib/types";

interface LabClinicalSummaryProps {
  clinicalSummary?: ClinicalSummary | null;
}

export const LabClinicalSummary: React.FC<LabClinicalSummaryProps> = ({ clinicalSummary }) => {
  const [activePanelId, setActivePanelId] = useState<string | null>(null);
  const [showNormalTests, setShowNormalTests] = useState<boolean>(false);

  if (!clinicalSummary) {
    return null;
  }

  const {
    overview,
    panels,
    abnormal_findings,
    correlated_patterns,
    in_range_findings,
    clinical_guidance,
  } = clinicalSummary;

  const highFindings = abnormal_findings?.high || [];
  const lowFindings = abnormal_findings?.low || [];
  const totalAbnormal = highFindings.length + lowFindings.length;
  const hasPatterns = correlated_patterns && correlated_patterns.length > 0;

  return (
    <div className="space-y-6">
      {/* 1. Overview Metric Banner */}
      <div className="rounded-2xl border border-sky-100 bg-white p-5 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-sky-50 text-sky-600 rounded-xl">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
              </svg>
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Clinical Laboratory Summary & Interpretation
              </h3>
              <p className="text-xs text-slate-500">
                Automated multi-analyte physiological evaluation across detected metabolic and organ panels
              </p>
            </div>
          </div>
          <span className="self-start sm:self-auto text-[11px] font-mono font-medium px-2.5 py-1 bg-slate-50 text-slate-600 border border-slate-200 rounded-full">
            Evidence-Based Analysis
          </span>
        </div>

        {/* Metric Counter Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="rounded-xl border border-slate-100 bg-slate-50/60 p-3 text-center">
            <span className="text-[11px] font-medium text-slate-500 block uppercase tracking-wider">
              Total Analytes
            </span>
            <span className="text-2xl font-bold text-slate-800">
              {overview.total_parameters}
            </span>
          </div>

          <div className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-3 text-center">
            <span className="text-[11px] font-medium text-emerald-700 block uppercase tracking-wider">
              Within Reference Range
            </span>
            <span className="text-2xl font-bold text-emerald-700">
              {overview.in_range_count}
            </span>
          </div>

          <div className={`rounded-xl border p-3 text-center ${
            totalAbnormal > 0
              ? "border-rose-100 bg-rose-50/50"
              : "border-slate-100 bg-slate-50/40"
          }`}>
            <span className={`text-[11px] font-medium block uppercase tracking-wider ${
              totalAbnormal > 0 ? "text-rose-700" : "text-slate-500"
            }`}>
              Abnormal Values
            </span>
            <span className={`text-2xl font-bold ${
              totalAbnormal > 0 ? "text-rose-700" : "text-slate-700"
            }`}>
              {totalAbnormal}
            </span>
          </div>

          <div className="rounded-xl border border-amber-100 bg-amber-50/40 p-3 text-center">
            <span className="text-[11px] font-medium text-amber-700 block uppercase tracking-wider">
              Uncertain / Review
            </span>
            <span className="text-2xl font-bold text-amber-800">
              {overview.uncertain_count}
            </span>
          </div>
        </div>

        {/* Narrative Executive Summary */}
        <div className="rounded-xl border border-slate-200/80 bg-slate-50/70 p-4">
          <div className="flex items-start gap-3">
            <span className="text-sky-600 mt-0.5 shrink-0">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </span>
            <div className="space-y-1">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700 block">
                Executive Finding Overview
              </span>
              <p className="text-xs text-slate-700 leading-relaxed font-normal">
                {overview.narrative_summary}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Correlated Clinical Patterns (Multi-analyte relationships) */}
      {hasPatterns && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-violet-600" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Correlated Clinical Patterns & Multi-Analyte Relationships
            </h4>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {correlated_patterns.map((pat, idx) => (
              <div
                key={idx}
                className={`rounded-xl border p-4 shadow-xs space-y-3 ${
                  pat.severity === "ATTENTION"
                    ? "border-rose-200 bg-linear-to-br from-rose-50/60 via-white to-white"
                    : "border-sky-200 bg-linear-to-br from-sky-50/50 via-white to-white"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full uppercase ${
                      pat.severity === "ATTENTION"
                        ? "bg-rose-100 text-rose-800"
                        : "bg-sky-100 text-sky-800"
                    }`}>
                      {pat.severity}
                    </span>
                    <h5 className="text-xs font-bold text-slate-900">
                      {pat.title}
                    </h5>
                  </div>
                </div>

                <div className="space-y-2 text-xs">
                  <div>
                    <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide block">
                      Contributing Analytes
                    </span>
                    <div className="flex flex-wrap gap-1.5 mt-1">
                      {pat.analytes.map((a) => (
                        <span key={a} className="px-2 py-0.5 bg-white border border-slate-200 text-slate-700 rounded text-[11px] font-mono font-medium">
                          {a.toUpperCase()}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div>
                    <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide block">
                      Physiological Correlation
                    </span>
                    <p className="text-slate-700 text-xs leading-relaxed mt-0.5">
                      {pat.summary}
                    </p>
                  </div>

                  <div>
                    <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide block">
                      Clinical Considerations
                    </span>
                    <p className="text-slate-600 text-[11px] leading-relaxed mt-0.5 italic">
                      {pat.clinical_considerations}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 3. Abnormal Findings Detail */}
      {totalAbnormal > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Abnormal Parameters & Evidence-Based Considerations ({totalAbnormal})
            </h4>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[...highFindings, ...lowFindings].map((finding: ClinicalFindingItem) => (
              <div
                key={finding.normalized_name}
                className="rounded-xl border border-rose-100 bg-white p-4 shadow-xs space-y-3"
              >
                <div className="flex items-start justify-between gap-2 border-b border-slate-100 pb-2.5">
                  <div>
                    <h5 className="text-xs font-bold text-slate-900">
                      {finding.test_name}
                    </h5>
                    <span className="text-[10px] text-slate-500 uppercase font-mono">
                      Category: {finding.category}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full uppercase ${
                      finding.status === "HIGH"
                        ? "bg-rose-100 text-rose-700 border border-rose-200"
                        : "bg-blue-100 text-blue-700 border border-blue-200"
                    }`}>
                      {finding.status}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                  <div>
                    <span className="text-[10px] uppercase text-slate-500 font-bold block">Measured Result</span>
                    <span className="text-sm font-bold text-slate-900">
                      {finding.value} {finding.unit || ""}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase text-slate-500 font-bold block">Reference Range</span>
                    <span className="text-xs font-semibold text-slate-700">
                      {finding.reference_range || "N/A"} {finding.unit || ""}
                    </span>
                    <span className="text-[10px] text-slate-400 block">
                      ({finding.reference_range_source === "REPORT" ? "Printed in report" : "Standard interval"})
                    </span>
                  </div>
                </div>

                {finding.possible_causes && finding.possible_causes.length > 0 && (
                  <div className="space-y-1">
                    <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wide block">
                      Medically Supported Possible Causes
                    </span>
                    <ul className="list-disc list-inside text-xs text-slate-600 space-y-0.5 pl-1">
                      {finding.possible_causes.map((cause, idx) => (
                        <li key={idx} className="leading-relaxed">
                          {cause}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 4. Diagnostic Clinical Panels */}
      {panels && panels.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-sky-500" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Diagnostic Clinical Panels ({panels.length})
            </h4>
          </div>

          <div className="space-y-3">
            {panels.map((panel: ClinicalPanelSection) => {
              const isExpanded = activePanelId === panel.panel_id || activePanelId === null;

              return (
                <div key={panel.panel_id} className="rounded-xl border border-slate-200 bg-white overflow-hidden shadow-xs">
                  <button
                    type="button"
                    onClick={() => setActivePanelId(activePanelId === panel.panel_id ? null : panel.panel_id)}
                    className="w-full flex items-center justify-between p-3.5 bg-slate-50/70 hover:bg-slate-100/70 transition-colors text-left border-b border-slate-200 cursor-pointer"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-bold text-xs text-slate-800">
                        {panel.title}
                      </span>
                      <span className="text-[11px] text-slate-500">
                        {panel.items.length} tests
                      </span>
                      {panel.abnormal_count > 0 ? (
                        <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-rose-100 text-rose-700">
                          {panel.abnormal_count} abnormal
                        </span>
                      ) : (
                        <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700">
                          all normal
                        </span>
                      )}
                    </div>
                    <svg
                      className={`w-4 h-4 text-slate-400 transition-transform ${isExpanded ? "rotate-180" : ""}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {isExpanded && (
                    <div className="p-3 overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 text-slate-500 font-bold text-[10px] uppercase">
                            <th className="py-2 px-3">Test Name</th>
                            <th className="py-2 px-3">Measured Result</th>
                            <th className="py-2 px-3">Reference Range</th>
                            <th className="py-2 px-3">Evaluation Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {panel.items.map((item) => {
                            const isHigh = item.status === "HIGH";
                            const isLow = item.status === "LOW";
                            const isNormal = item.status === "WITHIN_RANGE";

                            return (
                              <tr key={item.normalized_name} className="hover:bg-slate-50/50">
                                <td className="py-2 px-3 font-medium text-slate-800">
                                  {item.test_name}
                                </td>
                                <td className="py-2 px-3 font-mono font-bold text-slate-900">
                                  {item.value} <span className="text-slate-500 font-normal">{item.unit || ""}</span>
                                </td>
                                <td className="py-2 px-3 font-mono text-slate-600">
                                  {item.reference_range ? `${item.reference_range} ${item.unit || ""}` : "N/A"}
                                </td>
                                <td className="py-2 px-3">
                                  {isNormal && (
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                      WITHIN RANGE
                                    </span>
                                  )}
                                  {isHigh && (
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-50 text-rose-700 border border-rose-200">
                                      <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
                                      HIGH
                                    </span>
                                  )}
                                  {isLow && (
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-200">
                                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                                      LOW
                                    </span>
                                  )}
                                  {!isNormal && !isHigh && !isLow && (
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-50 text-amber-700 border border-amber-200">
                                      REVIEW NEEDED
                                    </span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 5. In-Range / Normal Baseline Findings (Collapsible) */}
      {in_range_findings && in_range_findings.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-xs">
          <button
            type="button"
            onClick={() => setShowNormalTests(!showNormalTests)}
            className="w-full flex items-center justify-between text-left cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Within Reference Range Baseline ({in_range_findings.length} parameters)
              </span>
            </div>
            <span className="text-xs text-sky-600 font-semibold hover:underline">
              {showNormalTests ? "Hide Normal Tests" : "View All Normal Tests"}
            </span>
          </button>

          {showNormalTests && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 pt-3 mt-3 border-t border-slate-100">
              {in_range_findings.map((item) => (
                <div key={item.normalized_name} className="p-2 rounded-lg bg-slate-50 border border-slate-100 text-xs">
                  <span className="font-semibold text-slate-800 block truncate" title={item.test_name}>
                    {item.test_name}
                  </span>
                  <div className="text-slate-600 font-mono text-[11px] mt-0.5">
                    {item.value} {item.unit || ""}
                  </div>
                  <div className="text-[10px] text-slate-400 font-mono">
                    ref: {item.reference_range}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 6. Clinical Uncertainty & Physician Next Steps */}
      <div className="rounded-2xl border border-teal-200 bg-linear-to-br from-teal-50/60 via-white to-sky-50/40 p-5 shadow-xs space-y-4">
        <div className="flex items-center gap-2 text-teal-900 font-bold text-xs uppercase tracking-wider">
          <svg className="w-4 h-4 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <span>Recommended Next Steps & Clinical Guidance</span>
        </div>

        {clinical_guidance?.actionable_recommendations && clinical_guidance.actionable_recommendations.length > 0 && (
          <ul className="space-y-1.5 text-xs text-slate-700 pl-1">
            {clinical_guidance.actionable_recommendations.map((step, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-500 mt-1.5 shrink-0" />
                <span className="leading-relaxed">{step}</span>
              </li>
            ))}
          </ul>
        )}

        <div className="pt-2 border-t border-teal-100 text-[11px] text-slate-500 italic leading-relaxed">
          {clinical_guidance?.disclaimer}
        </div>
      </div>
    </div>
  );
};
