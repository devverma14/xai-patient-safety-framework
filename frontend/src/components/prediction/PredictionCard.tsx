import React, { useState } from "react";
import { PredictionDetails } from "@/lib/types";
import { PatientFriendlyExplanation } from "@/components/prediction/PatientFriendlyExplanation";

interface PredictionCardProps {
  prediction: PredictionDetails;
  modality?: string;
  task?: string;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  prediction,
  modality = "xray",
  task,
}) => {
  const [viewMode, setViewMode] = useState<"patient" | "clinical">("patient");

  const prob = prediction.probability ?? 0;
  const probPct = (prob * 100).toFixed(1);
  const threshold = prediction.threshold ?? 0.50;
  const thresholdPct = (threshold * 100).toFixed(0);
  const isHighRisk = prob >= threshold;

  const displayLabel =
    prediction.label || (isHighRisk ? "Positive Pattern Predicted" : "Negative Pattern Predicted");

  const modelName =
    prediction.model_name ||
    prediction.model_architecture ||
    "Deep Learning Classifier";

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs space-y-5">
      {/* Header with Dual-View Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3.5">
        <div>
          <span className="text-[11px] font-bold tracking-wider uppercase text-sky-700 block font-mono">
            Model Prediction Result • {modelName}
          </span>
          <h2
            className={`text-xl sm:text-2xl font-extrabold tracking-tight mt-1 ${
              isHighRisk ? "text-amber-800" : "text-emerald-800"
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
              isHighRisk ? "bg-amber-500" : "bg-emerald-500"
            }`}
            style={{ width: `${Math.min(Math.max(prob * 100, 2), 100)}%` }}
          />
        </div>

        <p className="text-[11px] text-slate-500 leading-snug">
          A model estimate reflects pattern likelihood in the input data; it is not a confirmed medical diagnosis.
        </p>
      </div>

      {/* Patient View Branch */}
      {viewMode === "patient" && (
        <div className="space-y-4">
          <PatientFriendlyExplanation
            modality={modality as "lab" | "xray" | "skin" | "retina"}
            task={task}
            isPositiveOrHighRisk={isHighRisk}
          />

          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-2">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
              <svg className="w-4 h-4 text-sky-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              Understanding the Visual
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed font-normal">
              Visual feature attributions highlight specific patterns in the input that the AI model considered most relevant. Warm color regions represent stronger influences on the prediction.
            </p>
          </div>
        </div>
      )}

      {/* Clinical / Technical View Branch */}
      {viewMode === "clinical" && (
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-4 space-y-3 text-xs text-slate-600">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800 uppercase tracking-wider text-[11px]">
                Technical Specifications
              </span>
              <span className="text-[10px] text-slate-400 font-mono">
                {modelName}
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono">
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
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
