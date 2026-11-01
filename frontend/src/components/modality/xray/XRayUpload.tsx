import React, { useEffect } from "react";
import { UploadDropzone } from "@/components/upload/UploadDropzone";
import { XRayTask } from "@/lib/types";

interface XRayUploadProps {
  selectedTask: XRayTask;
  selectedFile: File | null;
  onFileSelect: (file: File | null) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  disabled?: boolean;
}

export const XRayUpload: React.FC<XRayUploadProps> = ({
  selectedTask,
  selectedFile,
  onFileSelect,
  onAnalyze,
  isAnalyzing,
  disabled = false,
}) => {
  const previewUrl = React.useMemo(() => {
    if (!selectedFile) return null;
    return URL.createObjectURL(selectedFile);
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const isPneumonia = selectedTask === "pneumonia";
  const taskTitle = isPneumonia
    ? "Pneumonia Detection (DenseNet121)"
    : "Tuberculosis Detection (DenseNet121)";
  const taskHeading = isPneumonia
    ? "Chest X-Ray — Pneumonia Screening"
    : "Chest X-Ray — Tuberculosis Screening";
  const taskInstructions = isPneumonia
    ? "Upload a chest X-ray image to evaluate pulmonary consolidation and infiltrates using DenseNet121 deep visual feature extraction."
    : "Upload a chest X-ray image to evaluate apical cavities and tuberculous opacities using DenseNet121 deep visual feature extraction.";

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3.5">
        <div>
          <span className={`text-[11px] font-bold uppercase tracking-wider block ${isPneumonia ? "text-teal-700" : "text-violet-700"}`}>
            Thoracic Imaging • DenseNet121 • Grad-CAM
          </span>
          <h2 className="text-lg font-bold text-slate-900 mt-0.5">
            {taskHeading}
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl leading-relaxed">
            {taskInstructions}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-slate-700 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200 font-medium">
            DenseNet121
          </span>
          <span className={`text-xs font-mono px-2.5 py-1 rounded-md border font-semibold ${
            isPneumonia ? "bg-teal-50 text-teal-800 border-teal-200" : "bg-violet-50 text-violet-800 border-violet-200"
          }`}>
            Task: {selectedTask.toUpperCase()}
          </span>
        </div>
      </div>

      <UploadDropzone
        acceptTypes={[".jpg", ".jpeg", ".png", ".bmp", ".tiff"]}
        maxSizeBytes={15 * 1024 * 1024}
        selectedFile={selectedFile}
        onFileSelect={onFileSelect}
        onClearFile={() => onFileSelect(null)}
        disabled={isAnalyzing || disabled}
        accent={isPneumonia ? "teal" : "violet"}
        title="Upload a chest X-ray image"
        subtitle={
          isPneumonia
            ? "Chest radiograph for pneumonia screening. In-memory DenseNet121 inference with Grad-CAM feature attribution."
            : "Chest radiograph for tuberculosis screening. In-memory DenseNet121 inference with Grad-CAM feature attribution."
        }
      />

      {/* Selected Image Preview & Metadata */}
      {selectedFile && previewUrl && (
        <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-4 space-y-3 shadow-2xs">
          <div className="flex items-center justify-between text-xs border-b border-slate-100 pb-2">
            <span className="font-bold text-slate-800">Chest X-Ray Preview</span>
            <span className="text-slate-500 font-mono text-[11px]">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || "image/x-ray"}
            </span>
          </div>

          <div className="relative aspect-square max-h-52 mx-auto rounded-lg overflow-hidden border border-slate-300 bg-black flex items-center justify-center shadow-xs">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Uploaded Chest Radiograph Preview"
              className="max-h-full max-w-full object-contain"
            />
          </div>
        </div>
      )}

      {/* Action Button */}
      <button
        type="button"
        onClick={onAnalyze}
        disabled={!selectedFile || isAnalyzing || disabled}
        className={`w-full py-2.5 px-4 rounded-lg font-semibold text-xs transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-2xs ${
          !selectedFile || isAnalyzing || disabled
            ? "bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200"
            : isPneumonia
            ? "bg-teal-700 hover:bg-teal-800 text-white"
            : "bg-indigo-700 hover:bg-indigo-800 text-white"
        }`}
      >
        {isAnalyzing ? (
          <>
            <span className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
            Analyzing {taskTitle} & Computing Grad-CAM...
          </>
        ) : (
          <>
            Analyze Chest X-Ray & Compute Grad-CAM
          </>
        )}
      </button>

      <p className="text-[11px] text-slate-500 text-center font-medium">
        Supported Formats: PNG, JPG, JPEG, BMP, TIFF (Up to 15MB). Image artifacts are processed in-memory for inference and are not saved to persistent storage.
      </p>
    </div>
  );
};
