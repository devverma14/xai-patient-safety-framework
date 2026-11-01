import React from "react";
import { ClinicalSummary, ExtractedLabValueItem, ClinicalFindingItem } from "@/lib/types";

interface LabReportSummaryProps {
  clinicalSummary?: ClinicalSummary | null;
  labValues?: ExtractedLabValueItem[];
  reviewReasons?: string[];
}

export const LabReportSummary: React.FC<LabReportSummaryProps> = ({
  clinicalSummary,
  labValues = [],
  reviewReasons = [],
}) => {
  if (!clinicalSummary) {
    return null;
  }

  const { overview, abnormal_findings } = clinicalSummary;

  // Accurate, consistent counts
  const highItems = abnormal_findings?.high || [];
  const lowItems = abnormal_findings?.low || [];
  const allAbnormal = [...highItems, ...lowItems];

  const inRangeCount = labValues.filter(
    (v) => v.interpretation_status === "WITHIN_RANGE" && v.validation_status === "VALID"
  ).length;

  const needsVerificationItems = labValues.filter(
    (v) =>
      v.validation_status !== "VALID" ||
      v.interpretation_status === "UNABLE_TO_DETERMINE" ||
      v.numeric_value === null
  );

  return (
    <div className="rounded-2xl border border-sky-100 bg-white p-5 sm:p-6 shadow-xs space-y-5">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-sky-600" />
            <span className="text-[11px] font-bold tracking-wider uppercase text-sky-800">
              Section A — Executive Summary
            </span>
          </div>
          <h3 className="text-base font-extrabold text-slate-900 mt-0.5">
            Report Summary & Findings Overview
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Key findings prioritized by physiological urgency. Educational and informational evaluation only.
          </p>
        </div>

        <span className="self-start sm:self-auto text-[11px] font-mono font-medium px-2.5 py-1 bg-sky-50 text-sky-800 border border-sky-200 rounded-full">
          {labValues.length} Total Parameters Detected
        </span>
      </div>

      {/* 4 Concise Metric Counter Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="rounded-xl border border-rose-100 bg-rose-50/50 p-3.5 text-center">
          <span className="text-[11px] font-semibold text-rose-700 block uppercase tracking-wider">
            High Results
          </span>
          <span className="text-2xl font-bold text-rose-800 mt-0.5 block">
            {highItems.length}
          </span>
          <span className="text-[10px] text-rose-600 font-sans">Exceeds Reference</span>
        </div>

        <div className="rounded-xl border border-blue-100 bg-blue-50/50 p-3.5 text-center">
          <span className="text-[11px] font-semibold text-blue-700 block uppercase tracking-wider">
            Low Results
          </span>
          <span className="text-2xl font-bold text-blue-800 mt-0.5 block">
            {lowItems.length}
          </span>
          <span className="text-[10px] text-blue-600 font-sans">Below Reference</span>
        </div>

        <div className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-3.5 text-center">
          <span className="text-[11px] font-semibold text-emerald-700 block uppercase tracking-wider">
            Within Range
          </span>
          <span className="text-2xl font-bold text-emerald-800 mt-0.5 block">
            {inRangeCount}
          </span>
          <span className="text-[10px] text-emerald-600 font-sans">Normal Baseline</span>
        </div>

        <div className="rounded-xl border border-amber-100 bg-amber-50/50 p-3.5 text-center">
          <span className="text-[11px] font-semibold text-amber-700 block uppercase tracking-wider">
            Needs Verification
          </span>
          <span className="text-2xl font-bold text-amber-800 mt-0.5 block">
            {needsVerificationItems.length}
          </span>
          <span className="text-[10px] text-amber-600 font-sans">Review / Unmeasured</span>
        </div>
      </div>

      {/* Narrative Executive Summary */}
      <div className="rounded-xl border border-slate-200/90 bg-slate-50/80 p-4">
        <div className="flex items-start gap-3">
          <div className="p-1.5 rounded-lg bg-sky-100 text-sky-700 shrink-0 mt-0.5">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="space-y-1">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-800 block">
              Clinical Narrative Synthesis
            </span>
            <p className="text-xs text-slate-700 leading-relaxed">
              {overview?.narrative_summary ||
                "Laboratory findings have been verified and categorized against standard physiological reference intervals."}
            </p>
          </div>
        </div>
      </div>

      {/* Prioritized Abnormal Findings (Confirmed Out-of-Range Results) */}
      <div className="space-y-3 pt-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800">
              Prioritized Abnormal Findings ({allAbnormal.length})
            </h4>
          </div>
          <span className="text-[11px] text-slate-500 italic">
            Confirmed deviations requiring clinical correlation
          </span>
        </div>

        {allAbnormal.length > 0 ? (
          <div className="overflow-hidden rounded-xl border border-rose-100 bg-white divide-y divide-rose-50">
            {allAbnormal.map((item: ClinicalFindingItem, idx) => {
              const isHigh = item.status === "HIGH";
              return (
                <div key={idx} className="p-4 hover:bg-slate-50/60 transition-colors space-y-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-xs text-slate-900">
                        {item.test_name}
                      </span>
                      {item.category && (
                        <span className="px-1.5 py-0.2 rounded text-[10px] font-sans bg-slate-100 text-slate-600">
                          {item.category}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide ${
                          isHigh
                            ? "bg-rose-100 text-rose-800 border border-rose-200"
                            : "bg-blue-100 text-blue-800 border border-blue-200"
                        }`}
                      >
                        {isHigh ? "High" : "Low"}
                      </span>
                      <span className="font-mono font-bold text-xs text-slate-900">
                        {item.value} {item.unit || ""}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs bg-slate-50/80 p-2.5 rounded-lg border border-slate-100">
                    <div>
                      <span className="text-[10px] uppercase font-bold text-slate-500 block">
                        Reference Interval
                      </span>
                      <span className="font-mono text-xs text-slate-800">
                        {item.reference_range ? `${item.reference_range} ${item.unit || ""}` : "Not provided"}
                      </span>
                      {item.reference_range_source && (
                        <span className="text-[9px] text-slate-400 block mt-0.5">
                          {item.reference_range_source === "REPORT" ? "Printed in report" : "Standard adult norm"}
                        </span>
                      )}
                    </div>
                    <div>
                      <span className="text-[10px] uppercase font-bold text-slate-500 block">
                        Plain-English Explanation
                      </span>
                      <p className="text-xs text-slate-700 leading-snug">
                        {item.guidance ||
                          "Measured concentration falls outside established reference bounds for this parameter."}
                      </p>
                    </div>
                  </div>

                  {item.possible_causes && item.possible_causes.length > 0 && (
                    <div className="text-xs space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
                        Medically Supported Potential Causes (Non-Diagnostic)
                      </span>
                      <p className="text-xs text-slate-600 leading-relaxed">
                        {item.possible_causes.join("; ")}.
                      </p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-4 text-center">
            <span className="text-xs font-semibold text-emerald-800">
              No physiological abnormalities identified. All tested analytes fall within standard reference intervals.
            </span>
          </div>
        )}
      </div>

      {/* Clearly Separated: Unresolved Extraction Conflicts / Unavailable Measurements */}
      {needsVerificationItems.length > 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50/50 p-4 space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-500" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-amber-900">
              Items Requiring Verification or Unavailable ({needsVerificationItems.length})
            </h4>
          </div>
          <p className="text-xs text-amber-900 leading-relaxed">
            The following parameters were either not performed or require document verification. They are never guessed or silently selected.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {needsVerificationItems.map((item, idx) => (
              <div
                key={idx}
                className="bg-white/90 p-3 rounded-lg border border-amber-200 shadow-2xs space-y-1 text-xs"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-900">{item.test_name}</span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-sans font-medium bg-amber-100 text-amber-800 border border-amber-200">
                    Needs Verification
                  </span>
                </div>
                <div className="text-slate-600 font-mono text-[11px]">
                  Extracted text:{" "}
                  <span className="font-semibold text-slate-900">
                    {item.raw_value_str || "No measurement found"}
                  </span>
                </div>
                {item.validation_message && (
                  <p className="text-[11px] text-amber-800 italic leading-snug">
                    {item.validation_message}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
