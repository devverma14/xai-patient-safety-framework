import React, { useEffect } from "react";
import { ExplanationDetails, GradCAMDetails } from "@/lib/types";

interface XRayGradCAMProps {
  explanation: ExplanationDetails;
  selectedFile: File | null;
}

export const XRayGradCAM: React.FC<XRayGradCAMProps> = ({
  explanation,
  selectedFile,
}) => {
  const imagePreviewUrl = React.useMemo(() => {
    if (!selectedFile) return null;
    return URL.createObjectURL(selectedFile);
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
    };
  }, [imagePreviewUrl]);

  const details = (explanation.details || {}) as GradCAMDetails;
  const grid7x7 = details.spatial_summary_grid_7x7 || [];
  const hasGrid = grid7x7.length === 49;

  // Helper to map 0.0-1.0 activation to heatmap background color
  const getActivationColor = (val: number) => {
    // Normalizing between min/max if present, or absolute 0-1
    const minVal = details.heatmap_min ?? 0;
    const maxVal = details.heatmap_max ?? 1;
    const norm = maxVal > minVal ? (val - minVal) / (maxVal - minVal) : val;

    if (norm < 0.2) return "bg-blue-950/80 text-blue-300 border-blue-900/40";
    if (norm < 0.4) return "bg-cyan-900/80 text-cyan-200 border-cyan-700/50";
    if (norm < 0.6) return "bg-emerald-700/80 text-emerald-100 border-emerald-600/60";
    if (norm < 0.8) return "bg-amber-600/90 text-amber-100 border-amber-500/80 font-bold";
    return "bg-rose-600 text-rose-100 border-rose-400 font-extrabold shadow-sm shadow-rose-500/50 animate-pulse";
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 space-y-4 shadow-xs">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-sky-700 font-mono">
            Explainable AI (XAI) Attribution
          </span>
          <h2 className="text-sm font-bold text-slate-900 mt-0.5">
            Grad-CAM Visual Feature Attribution
          </h2>
        </div>
        <span
          className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${
            explanation.available
              ? "bg-emerald-50 text-emerald-700 border-emerald-200"
              : "bg-amber-50 text-amber-700 border-amber-200"
          }`}
        >
          {explanation.available ? "Grad-CAM Heatmap Computed" : "Unavailable"}
        </span>
      </div>

      {/* Textual Summary */}
      {explanation.summary && (
        <p className="text-xs text-slate-700 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-200">
          {explanation.summary}
        </p>
      )}

      {/* Target Layer & Activation Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">Target Layer</span>
          <span className="font-bold text-sky-800 text-[11px] truncate block" title={details.target_layer || "N/A"}>
            {details.target_layer || "N/A"}
          </span>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">Heatmap Min</span>
          <span className="font-bold text-slate-800">
            {details.heatmap_min !== undefined ? details.heatmap_min.toFixed(4) : "N/A"}
          </span>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">Heatmap Mean</span>
          <span className="font-bold text-slate-800">
            {details.heatmap_mean !== undefined ? details.heatmap_mean.toFixed(4) : "N/A"}
          </span>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-sans font-bold">Heatmap Max</span>
          <span className="font-bold text-amber-700">
            {details.heatmap_max !== undefined ? details.heatmap_max.toFixed(4) : "N/A"}
          </span>
        </div>
      </div>

      {/* Visual Activation Display Grid & Radiograph Overlay */}
      {explanation.available && (
        <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-800">
              Grad-CAM 7x7 Spatial Activation Summary Grid
            </span>
            <span className="text-[11px] font-mono text-slate-500">
              Low (Blue) → High Activation (Red/Yellow)
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            {/* Left: Original Image Preview */}
            {imagePreviewUrl && (
              <div className="space-y-2">
                <span className="text-[10px] font-mono text-slate-500 uppercase block text-center font-bold">
                  Original Radiograph
                </span>
                <div className="relative aspect-square max-h-64 mx-auto rounded-lg overflow-hidden border border-slate-200 bg-slate-100 flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imagePreviewUrl}
                    alt="Original Radiograph"
                    className="max-h-full max-w-full object-contain"
                  />
                </div>
              </div>
            )}

            {/* Right: 7x7 Heatmap Activation Grid */}
            <div className="space-y-2">
              <span className="text-[10px] font-mono text-slate-500 uppercase block text-center font-bold">
                7x7 Spatial Feature Attribution Grid
              </span>

              {hasGrid ? (
                <div className="relative aspect-square max-h-64 mx-auto grid grid-cols-7 grid-rows-7 gap-1 p-2 rounded-lg border border-slate-200 bg-white">
                  {grid7x7.map((val, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center justify-center rounded text-[9px] font-mono border transition-all ${getActivationColor(
                        val
                      )}`}
                      title={`Cell ${idx + 1} (Row ${Math.floor(idx / 7) + 1}, Col ${(idx % 7) + 1}): Activation ${val}`}
                    >
                      {val.toFixed(2)}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-6 text-center text-xs text-slate-500 bg-white rounded-lg border border-slate-200">
                  Compact 7x7 grid visualization unavailable for this response.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Concise Limitations Notice */}
      <div className="rounded-xl border border-sky-100 bg-sky-50/70 p-4 space-y-1.5 text-xs text-slate-700 leading-relaxed">
        <div className="flex items-center gap-1.5 text-sky-900 font-bold text-xs uppercase tracking-wider">
          <span>How to read this visualization</span>
        </div>
        <p className="text-slate-600 text-[11.5px]">
          Highlighted regions indicate stronger model activation during analysis. This visualization illustrates the areas the model focused on; it does not prove disease presence, identify the exact anatomical cause, or serve as a standalone diagnosis.
        </p>
      </div>
    </div>
  );
};
