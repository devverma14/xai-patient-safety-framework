import React, { useState } from "react";
import { ExtractedLabValueItem } from "@/lib/types";
import { formatReferenceInterval, formatValueWithUnit } from "./LabKeyFindings";

interface LabExtractionResultsProps {
  labValues?: ExtractedLabValueItem[];
  extractionMethod?: string;
}

export const LabExtractionResults: React.FC<LabExtractionResultsProps> = ({
  labValues = [],
}) => {
  const [filterTab, setFilterTab] = useState<"ALL" | "ABNORMAL" | "NORMAL" | "VERIFICATION">("ALL");

  const isDemographic = (item: ExtractedLabValueItem) => {
    const norm = (item.normalized_name || "").toLowerCase().trim();
    const name = (item.test_name || "").toLowerCase().trim();
    return (
      ["age", "gender", "sex", "race", "patient_age", "patient_gender"].includes(norm) ||
      ["age", "gender", "sex", "race"].includes(name) ||
      name.startsWith("age ") ||
      name.startsWith("gender ")
    );
  };

  // Pure laboratory analytes (demographics excluded from lab table)
  const laboratoryValues = labValues.filter((v) => !isDemographic(v));

  const isItemNeedsVerification = (item: ExtractedLabValueItem) => {
    return (
      item.validation_status !== "VALID" ||
      item.interpretation_status === "UNABLE_TO_DETERMINE" ||
      item.numeric_value === null
    );
  };

  const highCount = laboratoryValues.filter((v) => v.interpretation_status === "HIGH").length;
  const lowCount = laboratoryValues.filter((v) => v.interpretation_status === "LOW").length;
  const abnormalTotal = highCount + lowCount;
  const normalCount = laboratoryValues.filter(
    (v) => v.interpretation_status === "WITHIN_RANGE" && v.validation_status === "VALID"
  ).length;
  const verificationCount = laboratoryValues.filter((v) => isItemNeedsVerification(v)).length;
  const totalCount = laboratoryValues.length;

  const filteredValues = laboratoryValues.filter((v) => {
    if (filterTab === "ABNORMAL") {
      return v.interpretation_status === "HIGH" || v.interpretation_status === "LOW";
    }
    if (filterTab === "NORMAL") {
      return v.interpretation_status === "WITHIN_RANGE" && v.validation_status === "VALID";
    }
    if (filterTab === "VERIFICATION") {
      return isItemNeedsVerification(v);
    }
    return true;
  });

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs space-y-4">
      {/* Section Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-base font-extrabold text-slate-900 tracking-tight">
            Your Report Data
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Complete list of all {totalCount} tested parameters and standard reference ranges.
          </p>
        </div>

        {/* Filter Tabs */}
        <div className="flex items-center flex-nowrap gap-1.5 overflow-x-auto no-scrollbar bg-slate-100/90 p-1 rounded-xl border border-slate-200/80 shrink-0 whitespace-nowrap">
          <button
            type="button"
            onClick={() => setFilterTab("ALL")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer shrink-0 ${
              filterTab === "ALL"
                ? "bg-white text-slate-900 shadow-xs border border-slate-200/60 font-bold"
                : "text-slate-600 hover:text-slate-900 hover:bg-white/50"
            }`}
          >
            All ({totalCount})
          </button>
          <button
            type="button"
            onClick={() => setFilterTab("ABNORMAL")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer shrink-0 ${
              filterTab === "ABNORMAL"
                ? "bg-rose-600 text-white shadow-xs font-bold"
                : "text-rose-700 hover:bg-rose-50 hover:text-rose-800"
            }`}
          >
            Abnormal ({abnormalTotal})
          </button>
          <button
            type="button"
            onClick={() => setFilterTab("NORMAL")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer shrink-0 ${
              filterTab === "NORMAL"
                ? "bg-emerald-600 text-white shadow-xs font-bold"
                : "text-emerald-700 hover:bg-emerald-50 hover:text-emerald-800"
            }`}
          >
            Within Range ({normalCount})
          </button>
          {verificationCount > 0 && (
            <button
              type="button"
              onClick={() => setFilterTab("VERIFICATION")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer shrink-0 ${
                filterTab === "VERIFICATION"
                  ? "bg-amber-600 text-white shadow-xs font-bold"
                  : "text-amber-700 hover:bg-amber-50 hover:text-amber-800"
              }`}
            >
              Needs Review ({verificationCount})
            </button>
          )}
        </div>
      </div>

      {/* Report Data Table */}
      {filteredValues.length > 0 ? (
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-50 uppercase text-[10px] tracking-wider text-slate-500 font-bold border-b border-slate-200">
              <tr>
                <th className="px-4 py-3">Test Name</th>
                <th className="px-4 py-3">Reported Value</th>
                <th className="px-4 py-3">Reference Range</th>
                <th className="px-4 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredValues.map((v, idx) => {
                const interpStatus = v.interpretation_status || "UNABLE_TO_DETERMINE";
                const isUnperformed =
                  v.raw_value_str &&
                  ["test not taken", "not performed", "not tested", "cancelled"].some((u) =>
                    v.raw_value_str!.toLowerCase().includes(u)
                  );
                const isUnreferenced = interpStatus === "UNABLE_TO_DETERMINE" && v.numeric_value !== null;

                return (
                  <tr
                    key={idx}
                    className="hover:bg-slate-50/80 transition-colors"
                  >
                    {/* Test Name */}
                    <td className="px-4 py-3">
                      <div className="font-semibold text-slate-900 text-xs">
                        {v.test_name}
                      </div>
                      {v.explanation?.category && (
                        <div className="mt-0.5">
                          <span className="px-1.5 py-0.5 rounded text-[9.5px] font-sans bg-slate-100 text-slate-600">
                            {v.explanation.category}
                          </span>
                        </div>
                      )}
                    </td>

                    {/* Reported Value */}
                    <td className="px-4 py-3 font-mono font-bold text-slate-900">
                      {v.numeric_value !== null && v.numeric_value !== undefined ? (
                        <span>
                          {formatValueWithUnit(v.raw_value_str, v.unit)}
                        </span>
                      ) : v.raw_value_str ? (
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-sans font-medium ${
                            isUnperformed
                              ? "bg-slate-100 text-slate-700 border border-slate-300"
                              : "bg-amber-50 text-amber-900 border border-amber-200"
                          }`}
                        >
                          {v.raw_value_str}
                        </span>
                      ) : (
                        <span className="text-slate-400 font-sans italic font-normal">
                          Value unreadable / not available
                        </span>
                      )}
                    </td>

                    {/* Reference Range */}
                    <td className="px-4 py-3">
                      <div className="font-mono text-slate-700 text-xs">
                        {formatReferenceInterval(v.reference_range, v.unit)}
                      </div>
                    </td>

                    {/* Status */}
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide ${
                          isUnperformed
                            ? "bg-slate-100 text-slate-700 border border-slate-300"
                            : isUnreferenced
                            ? "bg-slate-100 text-slate-800 border border-slate-300"
                            : interpStatus === "WITHIN_RANGE"
                            ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                            : interpStatus === "HIGH"
                            ? "bg-rose-50 text-rose-800 border border-rose-200"
                            : interpStatus === "LOW"
                            ? "bg-blue-50 text-blue-800 border border-blue-200"
                            : "bg-amber-50 text-amber-800 border border-amber-200"
                        }`}
                      >
                        {isUnperformed
                          ? "Test not performed"
                          : isUnreferenced
                          ? "Reference range unavailable"
                          : interpStatus === "WITHIN_RANGE"
                          ? "Within Range"
                          : interpStatus === "HIGH"
                          ? "High"
                          : interpStatus === "LOW"
                          ? "Low"
                          : "Needs Review"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-6 text-center text-xs text-slate-500">
          No extracted parameters match the selected &quot;{filterTab.toLowerCase()}&quot; filter criteria.
        </div>
      )}
    </div>
  );
};
