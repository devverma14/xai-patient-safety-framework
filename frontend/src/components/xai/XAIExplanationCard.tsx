import React from "react";
import { ExplanationDetails } from "@/lib/types";

interface XAIExplanationCardProps {
  explanation: ExplanationDetails;
}

export const XAIExplanationCard: React.FC<XAIExplanationCardProps> = ({
  explanation,
}) => {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
        <div>
          <span className="text-[10px] font-semibold tracking-wider uppercase text-slate-500">
            Explainable AI (XAI) Attribution
          </span>
          <h3 className="text-sm font-semibold text-slate-800 mt-0.5">
            {explanation.method} Method Details
          </h3>
        </div>
        <span
          className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${
            explanation.available
              ? "bg-teal-50 text-teal-700 border-teal-200"
              : "bg-slate-100 text-slate-500 border-slate-200"
          }`}
        >
          {explanation.available ? "Computed" : "Pending / Unavailable"}
        </span>
      </div>

      {explanation.summary && (
        <p className="text-xs text-slate-700 leading-relaxed mb-4 bg-slate-50 p-3 rounded-lg border border-slate-200">
          {explanation.summary}
        </p>
      )}

      {explanation.details && Object.keys(explanation.details).length > 0 && (
        <div className="space-y-2">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block">
            Explanation Layer & Activation Metrics
          </span>
          <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-[11px] font-mono text-slate-800 border border-slate-200">
            {JSON.stringify(explanation.details, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};
