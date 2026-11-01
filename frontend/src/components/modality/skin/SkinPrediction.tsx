import React, { useState } from "react";
import { PredictionDetails } from "@/lib/types";
import { PatientFriendlyExplanation } from "@/components/prediction/PatientFriendlyExplanation";

interface SkinPredictionProps {
  prediction: PredictionDetails;
  reviewRequired?: boolean;
}

export const SkinPrediction: React.FC<SkinPredictionProps> = ({
  prediction,
}) => {
  const [viewMode, setViewMode] = useState<"patient" | "clinical">("patient");

  const isHigherRisk = prediction.label?.toUpperCase() === "HIGHER-RISK";

  const prob = prediction.probability ?? 0;
  const probPct = (prob * 100).toFixed(1);
  const threshold = prediction.threshold ?? 0.40;
  const thresholdPct = (threshold * 100).toFixed(0);

  const displayLabel = isHigherRisk
    ? "Higher-Risk Pattern Predicted"
    : "Lower-Risk Pattern Predicted";

  const modelName =
    prediction.model_name ||
    prediction.model_architecture ||
    (prediction.raw_output?.model_architecture as string | undefined) ||
    "DenseNet121";

  const targetLayer =
    (prediction.raw_output?.target_layer as string | undefined) ||
    "features.denseblock4.denselayer16.conv2";

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs space-y-5">
      {/* Header with Dual-View Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3.5">
        <div>
          <span className="text-[11px] font-bold uppercase tracking-wider text-rose-700 block font-mono">
            Benign vs. Malignant Classification • {modelName}
          </span>
          <h2
            className={`text-xl sm:text-2xl font-extrabold tracking-tight mt-1 ${
              isHigherRisk ? "text-amber-800" : "text-emerald-800"
            }`}
          >
            {displayLabel}
          </h2>
        </div>

        {/* Dual-View Mode Switcher */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-100 border border-slate-200/80" role="tablist" aria-label="Explanation View Selection">
          <button
            type="button"
            role="tab"
            aria-selected={viewMode === "patient"}
            onClick={() => setViewMode("patient")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              viewMode === "patient"
                ? "bg-white text-slate-900 shadow-2xs font-bold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Patient View
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={viewMode === "clinical"}
            onClick={() => setViewMode("clinical")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              viewMode === "clinical"
                ? "bg-white text-slate-900 shadow-2xs font-bold"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Clinical / Technical View
          </button>
        </div>
      </div>

      {/* Prominent Probability & Threshold */}
      <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-4 space-y-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs font-semibold text-slate-700">
            Model-estimated probability:{" "}
            <strong className="font-mono text-slate-950 font-bold text-sm">
              {probPct}%
            </strong>
          </span>
          <span className="text-[11px] text-slate-500 font-medium">
            Decision threshold:{" "}
            <span className="font-mono font-semibold text-slate-700">
              {thresholdPct}%
            </span>
          </span>
        </div>

        {/* Probability Bar */}
        <div className="w-full h-2.5 bg-slate-200 rounded-full overflow-hidden relative">
          <div
            className={`h-full transition-all duration-500 rounded-full ${
              isHigherRisk ? "bg-amber-500" : "bg-emerald-500"
            }`}
            style={{ width: `${Math.min(Math.max(prob * 100, 2), 100)}%` }}
          />
        </div>

        <p className="text-[11px] text-slate-500 leading-snug">
          A model estimate reflects pattern likelihood in the dermoscopic image; it is not a confirmed medical diagnosis.
        </p>
      </div>

      {/* Patient View Branch */}
      {viewMode === "patient" && (
        <div className="space-y-4">
          <PatientFriendlyExplanation
            modality="skin"
            isPositiveOrHighRisk={isHigherRisk}
          />

          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-2">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
              <svg className="w-4 h-4 text-rose-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              Understanding the Visual
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed font-normal">
              The heatmaps shown below display where the AI model focused its attention when analyzing dermoscopic patterns. Highlighted regions correspond to pigment networks, asymmetry, and localized lesion borders that influenced the risk triage grouping.
            </p>
          </div>
        </div>
      )}

      {/* Clinical / Technical View Branch */}
      {viewMode === "clinical" && (
        <div className="space-y-4">
          {/* Healthcare Review Guidance */}
          <div className="rounded-xl bg-amber-50/70 p-3.5 border border-amber-200 text-xs text-amber-900 leading-relaxed flex items-start gap-2.5">
            <svg
              className="w-4 h-4 text-amber-600 shrink-0 mt-0.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <div>
              <span className="font-bold text-amber-900 block mb-0.5">
                Recommended Dermatological Review
              </span>
              This screening estimate does not substitute for an in-person skin evaluation. Any new, evolving, bleeding, or irregular lesion should be examined directly by a qualified dermatologist or medical practitioner.
            </div>
          </div>

          {/* Technical Specifications */}
          <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-4 space-y-3 text-xs text-slate-600">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800 uppercase tracking-wider text-[11px]">
                Technical Specifications
              </span>
              <span className="text-[10px] text-slate-400 font-mono">
                {modelName} • Grad-CAM
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono">
              <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans">
                  Architecture
                </span>
                <span className="font-bold text-slate-800">{modelName}</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans">
                  Decision Threshold
                </span>
                <span className="font-bold text-slate-800">{threshold.toFixed(2)}</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans">
                  Raw Probability
                </span>
                <span className="font-bold text-slate-800">{prob.toFixed(4)}</span>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans">
                  Target Layer
                </span>
                <span className="font-bold text-slate-800 truncate block" title={targetLayer}>
                  {targetLayer}
                </span>
              </div>
            </div>
            <p className="text-[10.5px] text-slate-500 leading-normal">
              Binary lesion classification grouping (melanoma/carcinoma vs. benign nevi/keratoses) evaluated using DenseNet121 and Grad-CAM target layer {targetLayer} against a validated decision threshold of {threshold.toFixed(2)}.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
