import React, { useState } from "react";
import { ClinicalPanelSection, ClinicalPanelItem } from "@/lib/types";

interface LabClinicalPanelsProps {
  panels?: ClinicalPanelSection[];
}

export const LabClinicalPanels: React.FC<LabClinicalPanelsProps> = ({ panels = [] }) => {
  // Collapsed by default as requested
  const [expandedPanels, setExpandedPanels] = useState<Record<string, boolean>>({});

  const togglePanel = (panelId: string) => {
    setExpandedPanels((prev) => ({
      ...prev,
      [panelId]: !prev[panelId],
    }));
  };

  const expandAll = () => {
    const allExpanded: Record<string, boolean> = {};
    panels.forEach((p) => {
      allExpanded[p.panel_id] = true;
    });
    setExpandedPanels(allExpanded);
  };

  const collapseAll = () => {
    setExpandedPanels({});
  };

  if (!panels || panels.length === 0) {
    return null;
  }

  return (
    <div className="rounded-2xl border border-sky-100 bg-white p-5 sm:p-6 shadow-xs space-y-5">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-sky-500" />
            <span className="text-[11px] font-bold tracking-wider uppercase text-sky-800">
              Section D — Diagnostic Clinical Panels
            </span>
          </div>
          <h3 className="text-base font-extrabold text-slate-900 mt-0.5">
            Organ & Physiological Diagnostic Panels
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Grouped panels for CBC, glycemic, lipid, renal, hepatic, electrolytes, thyroid, and inflammatory markers. Collapsed by default.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            type="button"
            onClick={expandAll}
            className="px-2.5 py-1 text-xs font-semibold text-sky-700 bg-sky-50 hover:bg-sky-100 rounded-lg border border-sky-200 cursor-pointer"
          >
            Expand All
          </button>
          <button
            type="button"
            onClick={collapseAll}
            className="px-2.5 py-1 text-xs font-semibold text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-lg border border-slate-200 cursor-pointer"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Accordion Panels List */}
      <div className="space-y-3">
        {panels.map((panel) => {
          const isExpanded = !!expandedPanels[panel.panel_id];
          const hasAbnormal = panel.abnormal_count > 0;

          return (
            <div
              key={panel.panel_id}
              className="rounded-xl border border-slate-200 bg-white overflow-hidden shadow-2xs transition-all"
            >
              {/* Accordion Header */}
              <button
                type="button"
                onClick={() => togglePanel(panel.panel_id)}
                className={`w-full flex items-center justify-between p-3.5 transition-colors text-left cursor-pointer ${
                  isExpanded ? "bg-slate-100/80 border-b border-slate-200" : "bg-slate-50/70 hover:bg-slate-100/70"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="font-bold text-xs text-slate-800">{panel.title}</span>
                  <span className="text-[11px] text-slate-500 font-mono">
                    ({panel.items.length} {panel.items.length === 1 ? "test" : "tests"})
                  </span>
                  {hasAbnormal ? (
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-rose-100 text-rose-700 border border-rose-200">
                      {panel.abnormal_count} abnormal
                    </span>
                  ) : (
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 border border-emerald-200">
                      all within range
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-slate-400 font-medium">
                    {isExpanded ? "Hide" : "Inspect"}
                  </span>
                  <svg
                    className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                      isExpanded ? "rotate-180" : ""
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {/* Accordion Expanded Table */}
              {isExpanded && (
                <div className="p-3.5 space-y-2">
                  {panel.description && (
                    <p className="text-[11px] text-slate-500 italic px-1">
                      {panel.description}
                    </p>
                  )}
                  <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
                    <table className="w-full text-left text-xs text-slate-700">
                      <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold text-[10px] uppercase">
                        <tr>
                          <th className="py-2.5 px-3">Parameter</th>
                          <th className="py-2.5 px-3">Measured Result</th>
                          <th className="py-2.5 px-3">Reference Interval</th>
                          <th className="py-2.5 px-3">Evaluation Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {panel.items.map((item: ClinicalPanelItem, itemIdx: number) => {
                          const isHigh = item.status === "HIGH";
                          const isLow = item.status === "LOW";
                          const isNormal = item.status === "WITHIN_RANGE";
                          const isNeedsVerification = !isHigh && !isLow && !isNormal;

                          return (
                            <tr key={itemIdx} className="hover:bg-slate-50/50">
                              <td className="py-2.5 px-3 font-semibold text-slate-900">
                                {item.test_name}
                              </td>
                              <td className="py-2.5 px-3 font-mono font-bold text-slate-900">
                                {item.value}{" "}
                                <span className="text-slate-500 font-sans font-normal">
                                  {item.unit || ""}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 font-mono text-slate-600">
                                {item.reference_range ? `${item.reference_range} ${item.unit || ""}` : "—"}
                              </td>
                              <td className="py-2.5 px-3">
                                {isNormal && (
                                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-50 text-emerald-800 border border-emerald-200">
                                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                    Within Range
                                  </span>
                                )}
                                {isHigh && (
                                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-rose-50 text-rose-800 border border-rose-200">
                                    <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
                                    High
                                  </span>
                                )}
                                {isLow && (
                                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-blue-50 text-blue-800 border border-blue-200">
                                    <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                                    Low
                                  </span>
                                )}
                                {isNeedsVerification && (
                                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-amber-50 text-amber-800 border border-amber-200">
                                    Needs Verification
                                  </span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
