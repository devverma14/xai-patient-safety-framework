import React, { useState } from "react";

interface TechnicalDetailsProps {
  modelName: string;
  probability: number;
  threshold: number;
  xaiMethod: "Grad-CAM" | "SHAP";
  targetLayer?: string;
  gridResolution?: string;
  baseValue?: number;
  rawOutput?: Record<string, unknown>;
  defaultOpen?: boolean;
  className?: string;
}

export const TechnicalDetails: React.FC<TechnicalDetailsProps> = ({
  modelName,
  probability,
  threshold,
  xaiMethod,
  targetLayer,
  gridResolution = "7 × 7",
  baseValue,
  rawOutput,
  defaultOpen = false,
  className = "",
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  React.useEffect(() => {
    setIsOpen(defaultOpen);
  }, [defaultOpen]);

  return (
    <div className={`rounded-xl border border-slate-200 bg-white p-4 shadow-xs ${className}`}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between text-xs font-semibold text-slate-700 hover:text-slate-900 transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-sky-500/30 rounded-lg p-1"
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
          <span className="uppercase tracking-wider font-bold">Technical Details</span>
          <span className="text-[10px] font-mono text-slate-400 font-normal">
            ({modelName} • {xaiMethod})
          </span>
        </div>
        <div className="flex items-center gap-1 text-[11px] text-sky-700 font-medium">
          <span>{isOpen ? "Hide Details" : "Show Details"}</span>
          <svg
            className={`w-3.5 h-3.5 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {isOpen && (
        <div className="mt-3 pt-3 border-t border-slate-100 space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">
                Inference Model
              </span>
              <span className="font-bold text-slate-900 text-[11px] truncate block">
                {modelName}
              </span>
            </div>

            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">
                Raw Probability
              </span>
              <span className="font-bold text-slate-900 text-[11px]">
                {probability.toFixed(4)}
              </span>
            </div>

            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">
                Decision Threshold
              </span>
              <span className="font-bold text-slate-900 text-[11px]">
                {threshold.toFixed(2)}
              </span>
            </div>

            <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">
                XAI Method
              </span>
              <span className="font-bold text-slate-900 text-[11px]">
                {xaiMethod}
              </span>
            </div>

            {targetLayer && (
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">
                  Grad-CAM Target Layer
                </span>
                <span className="font-bold text-slate-900 text-[11px] truncate block" title={targetLayer}>
                  {targetLayer}
                </span>
              </div>
            )}

            {xaiMethod === "Grad-CAM" && (
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">
                  Activation Resolution
                </span>
                <span className="font-bold text-slate-900 text-[11px]">
                  {gridResolution}
                </span>
              </div>
            )}

            {baseValue !== undefined && (
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">
                  SHAP Expected Value
                </span>
                <span className="font-bold text-slate-900 text-[11px]">
                  {baseValue.toFixed(4)}
                </span>
              </div>
            )}
          </div>

          {rawOutput && Object.keys(rawOutput).length > 0 && (
            <div className="space-y-1">
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block font-sans">
                Raw Output Class Probabilities
              </span>
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-2.5 text-[11px] font-mono text-slate-800 border border-slate-200">
                {JSON.stringify(rawOutput, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
