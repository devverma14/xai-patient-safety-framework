import React from "react";
import { ClinicalSummary, ClinicalFindingItem, CorrelatedClinicalPattern } from "@/lib/types";

interface LabClinicalInterpretationProps {
  clinicalSummary?: ClinicalSummary | null;
}

export const LabClinicalInterpretation: React.FC<LabClinicalInterpretationProps> = ({
  clinicalSummary,
}) => {
  if (!clinicalSummary) {
    return null;
  }

  const { abnormal_findings, correlated_patterns, clinical_guidance } = clinicalSummary;
  const highFindings = abnormal_findings?.high || [];
  const lowFindings = abnormal_findings?.low || [];
  const abnormalItems = [...highFindings, ...lowFindings];
  const hasPatterns = correlated_patterns && correlated_patterns.length > 0;

  return (
    <div className="rounded-2xl border border-teal-100 bg-white p-5 sm:p-6 shadow-xs space-y-6">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-teal-600" />
            <span className="text-[11px] font-bold tracking-wider uppercase text-teal-800">
              Section B — Clinical Interpretation
            </span>
          </div>
          <h3 className="text-base font-extrabold text-slate-900 mt-0.5">
            Important Findings & Evidence-Based Context
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Clinical context and physiological relationships for out-of-range results. Non-diagnostic analysis.
          </p>
        </div>

        <span className="self-start sm:self-auto text-[11px] font-mono font-medium px-2.5 py-1 bg-teal-50 text-teal-800 border border-teal-200 rounded-full">
          Evidence-Based Guidance
        </span>
      </div>

      {/* Multi-Analyte Correlated Patterns (Generated ONLY when contributing sources are verified) */}
      {hasPatterns && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-indigo-500" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800">
              Multi-Analyte Correlated Patterns ({correlated_patterns.length})
            </h4>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {correlated_patterns.map((pat: CorrelatedClinicalPattern, idx: number) => (
              <div
                key={idx}
                className={`rounded-xl border p-4 shadow-2xs space-y-3 ${
                  pat.severity === "ATTENTION"
                    ? "border-rose-200 bg-linear-to-br from-rose-50/40 via-white to-white"
                    : "border-sky-200 bg-linear-to-br from-sky-50/30 via-white to-white"
                }`}
              >
                <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-2">
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded-full uppercase ${
                        pat.severity === "ATTENTION"
                          ? "bg-rose-100 text-rose-800"
                          : "bg-sky-100 text-sky-800"
                      }`}
                    >
                      {pat.severity}
                    </span>
                    <h5 className="text-xs font-bold text-slate-900">{pat.title}</h5>
                  </div>
                  <span className="text-[10px] font-mono text-slate-400 uppercase">
                    {pat.category}
                  </span>
                </div>

                <div className="space-y-2 text-xs">
                  <div>
                    <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide block">
                      Contributing Analytes
                    </span>
                    <div className="flex flex-wrap gap-1.5 mt-1">
                      {pat.analytes.map((a: string) => (
                        <span
                          key={a}
                          className="px-2 py-0.5 bg-white border border-slate-200 text-slate-800 rounded text-[10px] font-mono font-semibold"
                        >
                          {a.toUpperCase()}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div>
                    <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide block">
                      Physiological Relationship
                    </span>
                    <p className="text-slate-700 text-xs leading-relaxed mt-0.5">
                      {pat.summary}
                    </p>
                  </div>

                  <div>
                    <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide block">
                      Clinical Context for Discussion
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

      {/* Useful Cards for Abnormal Findings */}
      {abnormalItems.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800">
              Clinical Interpretation Cards ({abnormalItems.length})
            </h4>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {abnormalItems.map((finding: ClinicalFindingItem, idx: number) => {
              const isHigh = finding.status === "HIGH";

              return (
                <div
                  key={idx}
                  className="rounded-xl border border-slate-200 bg-white p-4 shadow-2xs space-y-3"
                >
                  <div className="flex items-start justify-between gap-2 border-b border-slate-100 pb-2.5">
                    <div>
                      <h5 className="text-xs font-bold text-slate-900">{finding.test_name}</h5>
                      <span className="text-[10px] text-slate-500 uppercase font-mono">
                        {finding.category || "Laboratory Parameter"}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wide ${
                          isHigh
                            ? "bg-rose-100 text-rose-800 border border-rose-200"
                            : "bg-blue-100 text-blue-800 border border-blue-200"
                        }`}
                      >
                        {isHigh ? "High" : "Low"}
                      </span>
                      <span className="font-mono font-bold text-xs text-slate-900">
                        {finding.value} {finding.unit || ""}
                      </span>
                    </div>
                  </div>

                  {/* Context for clinician */}
                  <div className="space-y-2 text-xs">
                    <div>
                      <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wide block">
                        What This Finding May Indicate
                      </span>
                      <p className="text-xs text-slate-700 leading-relaxed mt-0.5">
                        {finding.guidance ||
                          `Measured ${finding.test_name} deviates from standard population limits (${finding.reference_range || "N/A"} ${finding.unit || ""}).`}
                      </p>
                    </div>

                    <div>
                      <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wide block">
                        Physiological Context & Clinician Considerations
                      </span>
                      <p className="text-xs text-slate-600 leading-relaxed mt-0.5">
                        {finding.possible_causes && finding.possible_causes.length > 0
                          ? `Clinicians typically correlate with hydration, medication effects, dietary timing, and follow-up confirmatory assays: ${finding.possible_causes.slice(0, 3).join(", ")}.`
                          : "Discuss clinical correlation and follow-up monitoring with your physician."}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Recommended Next Steps & Medical Disclaimer */}
      <div className="rounded-xl border border-teal-200 bg-linear-to-br from-teal-50/50 via-white to-sky-50/30 p-4 space-y-3">
        <div className="flex items-center gap-2 text-teal-900 font-bold text-xs uppercase tracking-wider">
          <svg className="w-4 h-4 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <span>Recommended Next Steps & Clinical Consultation</span>
        </div>

        {clinical_guidance?.actionable_recommendations && clinical_guidance.actionable_recommendations.length > 0 && (
          <ul className="space-y-1.5 text-xs text-slate-700 pl-1">
            {clinical_guidance.actionable_recommendations.map((step: string, idx: number) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-500 mt-1.5 shrink-0" />
                <span className="leading-relaxed">{step}</span>
              </li>
            ))}
          </ul>
        )}

        <div className="pt-2 border-t border-teal-100 text-[11px] text-slate-500 italic leading-relaxed">
          {clinical_guidance?.disclaimer ||
            "Notice: Automated laboratory analysis is provided for educational and clinical decision-support purposes only. It does not replace independent clinical judgment or physical examination by a licensed medical practitioner."}
        </div>
      </div>
    </div>
  );
};
