import React from "react";
import { ClinicalSummary, ClinicalFindingItem, CorrelatedClinicalPattern } from "@/lib/types";

interface LabKeyFindingsProps {
  clinicalSummary?: ClinicalSummary | null;
}

export function formatReferenceInterval(refRange?: string | null, unit?: string | null): string {
  if (!refRange) return "Not provided";
  const trimmed = refRange.trim();
  if (!unit) return trimmed;
  const cleanUnit = unit.trim();
  if (!cleanUnit) return trimmed;
  if (trimmed.toLowerCase().includes(cleanUnit.toLowerCase())) {
    return trimmed;
  }
  return `${trimmed} ${cleanUnit}`;
}

export function formatValueWithUnit(value: string | number | null | undefined, unit?: string | null): string {
  if (value === null || value === undefined) return "—";
  const valStr = String(value).trim();
  if (!unit) return valStr;
  const cleanUnit = unit.trim();
  if (!cleanUnit) return valStr;
  if (valStr.toLowerCase().includes(cleanUnit.toLowerCase())) {
    return valStr;
  }
  return `${valStr} ${cleanUnit}`;
}

export const LabKeyFindings: React.FC<LabKeyFindingsProps> = ({
  clinicalSummary,
}) => {
  if (!clinicalSummary) {
    return null;
  }

  const { overview, abnormal_findings, uncertain_findings = [], correlated_patterns = [], clinical_guidance } = clinicalSummary;

  const highItems = abnormal_findings?.high || [];
  const lowItems = abnormal_findings?.low || [];
  const allAbnormal = [...highItems, ...lowItems];
  const hasPatterns = correlated_patterns && correlated_patterns.length > 0;

  const highCount = overview?.high_count ?? highItems.length;
  const lowCount = overview?.low_count ?? lowItems.length;
  const inRangeCount = overview?.in_range_count ?? 0;
  const uncertainCount = overview?.uncertain_count ?? uncertain_findings.length;
  const totalCount = overview?.total_parameters ?? (highCount + lowCount + inRangeCount + uncertainCount);

  return (
    <div className="space-y-6">
      {/* =========================================================================
          A. KEY FINDINGS — Immediately display abnormal findings extracted
         ========================================================================= */}
      <section className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs space-y-4">
        {/* Section Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3.5">
          <div>
            <h3 className="text-lg font-extrabold text-slate-900 tracking-tight">
              Key Findings
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Laboratory values identified outside standard reference intervals.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto">
            <span className="text-xs font-mono font-medium px-2.5 py-1 bg-slate-100 text-slate-700 border border-slate-200 rounded-full">
              {totalCount} Tests Evaluated
            </span>
            {allAbnormal.length > 0 ? (
              <span className="text-xs font-semibold px-2.5 py-1 bg-rose-50 text-rose-700 border border-rose-200 rounded-full">
                {allAbnormal.length} Out of Range ({highCount} High, {lowCount} Low)
              </span>
            ) : (
              <span className="text-xs font-semibold px-2.5 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full">
                All Within Range
              </span>
            )}
          </div>
        </div>

        {/* Immediate Abnormal Findings Display */}
        {allAbnormal.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {allAbnormal.map((item: ClinicalFindingItem, idx: number) => {
              const isHigh = item.status === "HIGH";
              return (
                <div
                  key={idx}
                  className={`rounded-xl border p-4 shadow-2xs transition-all ${
                    isHigh
                      ? "bg-rose-50/30 border-rose-200/90 hover:border-rose-300"
                      : "bg-blue-50/30 border-blue-200/90 hover:border-blue-300"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <span className="text-sm font-bold text-slate-900 block tracking-tight">
                        {item.test_name}
                      </span>
                      {item.category && (
                        <span className="text-[10px] text-slate-500 font-medium block mt-0.5">
                          {item.category}
                        </span>
                      )}
                    </div>
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide border shrink-0 ${
                        isHigh
                          ? "bg-rose-100 text-rose-800 border-rose-200"
                          : "bg-blue-100 text-blue-800 border-blue-200"
                      }`}
                    >
                      {isHigh ? "High" : "Low"}
                    </span>
                  </div>

                  <div className="mt-3 flex items-baseline justify-between pt-2.5 border-t border-slate-100">
                    <div>
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block">
                        Reported Value
                      </span>
                      <span className="font-mono font-black text-base text-slate-900">
                        {formatValueWithUnit(item.value, item.unit)}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block">
                        Reference Range
                      </span>
                      <span className="font-mono text-xs text-slate-600 font-medium">
                        {formatReferenceInterval(item.reference_range, item.unit)}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-6 text-center space-y-1.5">
            <div className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-emerald-100 text-emerald-700 mx-auto">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h4 className="text-sm font-bold text-emerald-950">
              No Out-of-Range Values Identified
            </h4>
            <p className="text-xs text-emerald-800 max-w-md mx-auto">
              No abnormal findings were detected. All {totalCount} evaluated laboratory tests fall within standard reference intervals.
            </p>
          </div>
        )}
      </section>

      {/* =========================================================================
          B. WHAT YOUR RESULTS MAY MEAN — Plain-Language Interpretations
         ========================================================================= */}
      <section className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs space-y-4">
        {/* Section Header */}
        <div className="border-b border-slate-100 pb-3">
          <h3 className="text-base font-extrabold text-slate-900 tracking-tight">
            What Your Results May Mean
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Plain-language context for each out-of-range laboratory result.
          </p>
        </div>

        {allAbnormal.length > 0 ? (
          <div className="space-y-4">
            {allAbnormal.map((item: ClinicalFindingItem, idx: number) => {
              const isHigh = item.status === "HIGH";
              const causesList =
                item.possible_causes && item.possible_causes.length > 0
                  ? item.possible_causes
                  : item.possible_causes_text
                  ? [item.possible_causes_text]
                  : [];

              const meaning =
                item.what_it_means ||
                "This parameter is measured outside the typical reference interval established for healthy individuals.";

              const healthImpact =
                item.next_steps ||
                item.guidance ||
                "A single abnormal value does not establish a diagnosis. It should be evaluated alongside symptoms, medical history, and clinical follow-up.";

              return (
                <div
                  key={idx}
                  className="rounded-xl border border-slate-200/90 bg-white p-4 sm:p-5 shadow-xs space-y-3.5 transition-all hover:border-slate-300"
                >
                  {/* Finding: Test name and reported value */}
                  <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-2 h-2 rounded-full shrink-0 ${
                          isHigh ? "bg-rose-500" : "bg-blue-500"
                        }`}
                      />
                      <span className="font-bold text-sm text-slate-900">
                        Finding: <strong className="font-extrabold text-slate-950">{item.test_name}</strong>
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide border ${
                          isHigh
                            ? "bg-rose-50 text-rose-700 border-rose-200"
                            : "bg-blue-50 text-blue-700 border-blue-200"
                        }`}
                      >
                        {isHigh ? "High" : "Low"}
                      </span>
                    </div>

                    <div className="text-xs text-slate-600 font-medium">
                      Reported: <strong className="font-mono font-bold text-slate-900">{formatValueWithUnit(item.value, item.unit)}</strong>{" "}
                      <span className="text-slate-400 font-mono">(Ref: {formatReferenceInterval(item.reference_range, item.unit)})</span>
                    </div>
                  </div>

                  {/* 3-Part Clinical Breakdown: Why it matters, Possible causes, What it may mean for health */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                    {/* Why it matters */}
                    <div className="rounded-lg bg-sky-50/50 border border-sky-100 p-3 space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-sky-900 block">
                        Why It Matters
                      </span>
                      <p className="text-slate-700 leading-relaxed font-normal text-[11px]">
                        {meaning}
                      </p>
                    </div>

                    {/* Possible causes */}
                    <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-slate-700 block">
                        Possible Causes
                      </span>
                      {causesList.length > 0 ? (
                        <ul className="list-disc list-inside space-y-1 text-slate-600 pl-0.5 leading-relaxed text-[11px]">
                          {causesList.slice(0, 3).map((cause, cIdx) => (
                            <li key={cIdx} className="font-normal">{cause}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-slate-500 text-[11px] italic">
                          Non-specific variation; requires clinical correlation with a healthcare provider.
                        </p>
                      )}
                    </div>

                    {/* What it may mean for health */}
                    <div className="rounded-lg bg-emerald-50/50 border border-emerald-100 p-3 space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-900 block">
                        What It May Mean for Health
                      </span>
                      <p className="text-slate-700 leading-relaxed font-normal text-[11px]">
                        {healthImpact}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 text-xs text-slate-600">
            All tested parameters fall within expected physiological intervals. No specific abnormalities require clinical explanation.
          </div>
        )}

        {/* Correlated Physiological Patterns (If backend detected multi-analyte correlations) */}
        {hasPatterns && (
          <div className="pt-3 border-t border-slate-100 space-y-2.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-700 block">
              Correlated Physiological Patterns
            </span>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {correlated_patterns.map((pat: CorrelatedClinicalPattern, idx: number) => (
                <div
                  key={idx}
                  className="rounded-xl border border-slate-200 bg-slate-50/60 p-3.5 space-y-1.5 text-xs"
                >
                  <div className="flex items-center justify-between gap-2 border-b border-slate-200/60 pb-1.5">
                    <span className="font-bold text-slate-900 text-xs">{pat.title}</span>
                    <span className="text-[10px] font-mono text-slate-500 uppercase">{pat.category}</span>
                  </div>
                  <p className="text-slate-700 leading-relaxed text-[11px]">{pat.summary}</p>
                  {pat.clinical_considerations && (
                    <p className="text-slate-500 italic text-[10px] pt-1 border-t border-slate-200/40">
                      {pat.clinical_considerations}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Parameters with Interpretation Limitations */}
        {uncertain_findings.length > 0 && (
          <div className="rounded-xl border border-amber-200/80 bg-amber-50/40 p-3.5 text-xs text-amber-900 space-y-1.5">
            <div className="flex items-center gap-2 font-bold text-amber-950 text-xs">
              <svg className="w-4 h-4 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Parameters with Interpretation Limitations ({uncertain_findings.length})</span>
            </div>
            <p className="text-[11px] text-amber-800 leading-relaxed">
              {uncertain_findings.map((u) => u.test_name).join(", ")}: Mathematical indices or measurements without independent standalone reference intervals. Consult your physician for clinical interpretation.
            </p>
          </div>
        )}

        {/* Actionable Recommendations */}
        {clinical_guidance?.actionable_recommendations && clinical_guidance.actionable_recommendations.length > 0 && (
          <div className="rounded-xl border border-teal-200 bg-teal-50/40 p-4 space-y-2">
            <span className="text-xs font-bold uppercase tracking-wider text-teal-900 block">
              Suggested Next Steps for Discussion with Your Doctor
            </span>
            <ul className="space-y-1 text-xs text-slate-700 pl-1">
              {clinical_guidance.actionable_recommendations.map((step: string, idx: number) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-teal-500 mt-1.5 shrink-0" />
                  <span className="leading-relaxed font-normal">{step}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </div>
  );
};
