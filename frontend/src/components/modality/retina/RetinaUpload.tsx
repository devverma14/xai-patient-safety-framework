import React, { useEffect } from "react";
import { UploadDropzone } from "@/components/upload/UploadDropzone";

interface RetinaUploadProps {
  selectedFile: File | null;
  onFileSelect: (file: File | null) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  disabled?: boolean;
}

export const RetinaUpload: React.FC<RetinaUploadProps> = ({
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

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3.5">
        <div>
          <span className="text-[11px] font-bold uppercase tracking-wider text-amber-700 block">
            Ophthalmic Imaging • EfficientNet-B0 • Grad-CAM
          </span>
          <h2 className="text-lg font-bold text-slate-900 mt-0.5">
            Retinal Fundus — Diabetic Retinopathy
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl leading-relaxed font-medium">
            Upload a digital retinal fundus photograph to evaluate diabetic retinopathy patterns using EfficientNet-B0 deep visual feature extraction.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-slate-700 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200 font-medium">
            EfficientNet-B0
          </span>
          <span className="text-xs font-mono bg-amber-50 text-amber-800 border border-amber-200 px-2.5 py-1 rounded-md font-semibold">
            Task: DIABETIC RETINOPATHY
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
        accent="amber"
        title="Upload retinal fundus photograph"
        subtitle="Standardized digital fundus view. In-memory EfficientNet-B0 inference with Grad-CAM feature attribution."
      />

      {/* Selected Image Preview & Metadata */}
      {selectedFile && previewUrl && (
        <div className="rounded-2xl border border-slate-200/80 bg-white/90 p-4 space-y-3 shadow-xs">
          <div className="flex items-center justify-between text-xs border-b border-slate-100 pb-2">
            <span className="font-semibold text-slate-800">Retinal Fundus Preview</span>
            <span className="text-slate-500 font-mono text-[11px]">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || "image/retina"}
            </span>
          </div>

          <div className="relative aspect-square max-h-56 mx-auto rounded-xl overflow-hidden border border-slate-700 bg-black flex items-center justify-center shadow-md">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Uploaded Retinal Fundus Photograph Preview"
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
            : "bg-amber-700 hover:bg-amber-800 text-white"
        }`}
      >
        {isAnalyzing ? (
          <>
            <span className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
            Analyzing Retinal Image...
          </>
        ) : (
          <>
            Analyze Retinal Image
          </>
        )}
      </button>

      <p className="text-[11px] text-slate-500 text-center">
        Supported formats: PNG, JPG, JPEG, BMP, TIFF (Up to 15MB). Image artifacts are processed in-memory for model inference.
      </p>
    </div>
  );
};

