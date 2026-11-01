import React, { useState } from "react";
import { SHAPDetails, SHAPFactorItem } from "@/lib/types";
import { formatDecimal } from "@/lib/utils";

interface LabShapVisualizationProps {
  shapDetails: SHAPDetails;
}

export const LabShapVisualization: React.FC<LabShapVisualizationProps> = ({
  shapDetails,
}) => {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const increasingFactors = shapDetails.top_risk_increasing_factors || [];
  const decreasingFactors = shapDetails.top_risk_decreasing_factors || [];

  const maxMag = Math.max(
    ...increasingFactors.map((f) => f.magnitude || Math.abs(f.shap_value)),
    ...decreasingFactors.map((f) => f.magnitude || Math.abs(f.shap_value)),
    0.01
  );

  const renderFactorRow = (factor: SHAPFactorItem, isPositive: boolean) => {
    const shapVal = factor.shap_value;
    const mag = factor.magnitude || Math.abs(shapVal);
    const widthPct = Math.min(100, Math.max(12, (mag / maxMag) * 100));

    return (
      <div key={factor.feature_name} className="space-y-1.5">
        <div className="flex justify-between items-center text-xs">
          <div className="font-semibold text-slate-800 truncate pr-2">
            {factor.human_readable_name || factor.feature_name}
            <span className="ml-1.5 text-[11px] font-mono text-slate-500 font-normal">
              (Value: {String(factor.feature_value_scaled)})
            </span>
          </div>
          <span
            className={`font-mono text-xs font-bold shrink-0 ${
              isPositive ? "text-rose-600" : "text-sky-600"
            }`}
          >
            {shapVal > 0 ? `+${formatDecimal(shapVal, 4)}` : formatDecimal(shapVal, 4)}
          </span>
        </div>

        <div className="h-2.5 w-full rounded-full bg-slate-100 overflow-hidden flex items-center border border-slate-200">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isPositive
                ? "bg-rose-500 border-r border-rose-600"
                : "bg-sky-500 border-r border-sky-600"
            }`}
            style={{ width: `${widthPct}%` }}
          />
        </div>
      </div>
    );
  };

  return (
    <div className="rounded-2xl border border-sky-100 bg-white/95 backdrop-blur-md p-6 sm:p-7 shadow-xs space-y-6">
      {/* Primary Section Header — DIRECTLY VISIBLE CORE XAI */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 rounded-full bg-sky-500 animate-pulse" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-sky-700 font-mono">
              FEATURE ATTRIBUTION • TREE SHAP
            </span>
          </div>
          <h2 className="text-xl sm:text-2xl font-black text-slate-900 mt-1 tracking-tight">
            Why did the model make this prediction?
          </h2>
        </div>
        <div className="flex items-center gap-3 text-xs font-semibold">
          <span className="flex items-center gap-1 text-rose-600">
            <span className="text-sm">↑</span> Increases risk
          </span>
          <span className="flex items-center gap-1 text-sky-600">
            <span className="text-sm">↓</span> Decreases risk
          </span>
        </div>
      </div>

      {/* Factors Associated with Higher Model Estimate */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-rose-600" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-rose-900">
            Factors associated with a higher model estimate
          </h3>
        </div>

        {increasingFactors.length > 0 ? (
          <div className="space-y-3.5 rounded-xl bg-slate-50/80 p-4 border border-slate-200/80">
            {increasingFactors.map((f) => renderFactorRow(f, true))}
          </div>
        ) : (
          <p className="text-xs text-slate-500 italic bg-slate-50 p-3 rounded-lg border border-slate-200">
            No significant factors associated with a higher model estimate identified for this report.
          </p>
        )}
      </div>

      {/* Factors Associated with Lower Model Estimate */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-sky-600" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-sky-900">
            Factors associated with a lower model estimate
          </h3>
        </div>

        {decreasingFactors.length > 0 ? (
          <div className="space-y-3.5 rounded-xl bg-slate-50/80 p-4 border border-slate-200/80">
            {decreasingFactors.map((f) => renderFactorRow(f, false))}
          </div>
        ) : (
          <p className="text-xs text-slate-500 italic bg-slate-50 p-3 rounded-lg border border-slate-200">
            No significant factors associated with a lower model estimate identified for this report.
          </p>
        )}
      </div>

      {/* Non-Causal Explanation Note */}
      <div className="rounded-xl border border-sky-100 bg-sky-50/70 p-4 text-xs text-slate-700 leading-relaxed space-y-1">
        <p className="font-semibold text-slate-900">
          How to read feature contributions
        </p>
        <p className="text-slate-600">
          These factors show how the model&apos;s inputs contributed to its prediction. They describe model associations and do not prove that any factor caused the outcome.
        </p>
      </div>

      {/* SECONDARY TECHNICAL DETAILS */}
      <div className="border-t border-slate-100 pt-3">
        <button
          type="button"
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="text-xs font-medium text-slate-500 hover:text-slate-800 flex items-center gap-1 cursor-pointer transition-colors"
        >
          <span>{showTechnicalDetails ? "Hide Technical Details" : "Show Technical Details"}</span>
          <svg
            className={`w-3.5 h-3.5 transition-transform ${showTechnicalDetails ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {showTechnicalDetails && (
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono bg-slate-50 p-3.5 rounded-xl border border-slate-200 text-slate-700">
            <div>
              <span className="text-[10px] text-slate-500 uppercase block font-sans font-bold">XAI Method</span>
              <span className="font-bold text-slate-900">SHAP</span>
            </div>
            <div>
              <span className="text-[10px] text-slate-500 uppercase block font-sans font-bold">Base Model Value</span>
              <span className="font-bold text-slate-900">
                {shapDetails.base_value !== undefined ? formatDecimal(shapDetails.base_value, 4) : "N/A"}
              </span>
            </div>
            <div>
              <span className="text-[10px] text-slate-500 uppercase block font-sans font-bold">Model</span>
              <span className="font-bold text-slate-900">Regularized XGBoost</span>
            </div>
            <div>
              <span className="text-[10px] text-slate-500 uppercase block font-sans font-bold">Features Evaluated</span>
              <span className="font-bold text-slate-900">49 Parameters</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
