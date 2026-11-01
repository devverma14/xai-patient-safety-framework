import React, { useState } from "react";
import { SkinUpload } from "./SkinUpload";
import { SkinPrediction } from "./SkinPrediction";
import { ThreeViewGradCAM } from "@/components/xai/ThreeViewGradCAM";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { ErrorAlert } from "@/components/common/ErrorAlert";
import { SafetyDisclaimer } from "@/components/common/SafetyDisclaimer";
import { CommonResponsePayload } from "@/lib/types";
import { apiClient } from "@/lib/api/client";

type AnalysisState = "IDLE" | "FILE_SELECTED" | "ANALYZING" | "SUCCESS" | "ERROR";

export const SkinAnalysis: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>("IDLE");
  const [analysisResult, setAnalysisResult] = useState<CommonResponsePayload | null>(null);
  const [errorState, setErrorState] = useState<{
    message: string;
    details?: Record<string, unknown>;
  } | null>(null);

  // Handle file selection
  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    setAnalysisResult(null);
    setErrorState(null);
    setAnalysisState(file ? "FILE_SELECTED" : "IDLE");
  };

  // Execute Skin Analysis via API Client (Task: ham10000)
  const handleRunAnalysis = async () => {
    if (!selectedFile) return;

    setAnalysisState("ANALYZING");
    setErrorState(null);
    setAnalysisResult(null);

    try {
      const result = await apiClient.analyzeSkin(selectedFile, "ham10000");
      setAnalysisResult(result);
      setAnalysisState("SUCCESS");
    } catch (err: unknown) {
      const errObj = err as { message?: string; details?: Record<string, unknown> };
      setErrorState({
        message: errObj.message || "An unexpected error occurred during Skin Lesion analysis.",
        details: errObj.details,
      });
      setAnalysisState("ERROR");
    }
  };

  // Full reset handler
  const handleReset = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setErrorState(null);
    setAnalysisState("IDLE");
  };

  return (
    <div className="space-y-6">
      {/* Upload & Analysis Layout */}
      <div className="space-y-6">
        {!analysisResult && (
          <SkinUpload
            selectedFile={selectedFile}
            onFileSelect={handleFileSelect}
            onAnalyze={handleRunAnalysis}
            isAnalyzing={analysisState === "ANALYZING"}
            disabled={analysisState === "ANALYZING"}
          />
        )}

        {/* ANALYZING State */}
        {analysisState === "ANALYZING" && (
          <LoadingSpinner
            label="Analyzing Skin Lesion Artifact..."
            sublabel="Executing DenseNet121 model inference (Benign vs. Malignant Classification, decision threshold: 0.40) and computing Grad-CAM feature activation maps."
          />
        )}

        {/* ERROR State */}
        {analysisState === "ERROR" && errorState && (
          <ErrorAlert
            title="Skin Analysis Failed"
            message={errorState.message}
            details={errorState.details}
            onRetry={handleRunAnalysis}
          />
        )}

        {/* SUCCESS State */}
        {analysisState === "SUCCESS" && analysisResult && (
          <div className="space-y-6">
            {/* Header Action Bar */}
            <div className="flex items-center justify-between bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
              <div>
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block">
                  Analysis Artifact Output (Benign vs. Malignant Classification)
                </span>
                <span className="text-xs font-bold text-slate-900">
                  File: {selectedFile?.name || "Uploaded Dermoscopic Image"}
                </span>
              </div>
              <button
                type="button"
                onClick={handleReset}
                className="px-3.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-xs font-semibold text-slate-700 transition-colors border border-slate-200 cursor-pointer"
              >
                Analyze Another Image
              </button>
            </div>

            <SafetyDisclaimer compact />

            {/* Prediction Display */}
            {analysisResult.prediction && (
              <SkinPrediction
                prediction={analysisResult.prediction}
                reviewRequired={analysisResult.review_required}
              />
            )}

            {/* THREE-VIEW GRAD-CAM DISPLAY — IMMEDIATE VISIBILITY */}
            {analysisResult.explanation && (
              <ThreeViewGradCAM
                explanation={analysisResult.explanation}
                selectedFile={selectedFile}
                modalityTitle="Benign vs. Malignant Classification"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};
