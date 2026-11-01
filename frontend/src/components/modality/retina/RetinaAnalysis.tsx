import React, { useState } from "react";
import { RetinaUpload } from "./RetinaUpload";
import { RetinaPrediction } from "./RetinaPrediction";
import { ThreeViewGradCAM } from "@/components/xai/ThreeViewGradCAM";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { ErrorAlert } from "@/components/common/ErrorAlert";
import { SafetyDisclaimer } from "@/components/common/SafetyDisclaimer";
import { CommonResponsePayload } from "@/lib/types";
import { apiClient } from "@/lib/api/client";

type AnalysisState = "IDLE" | "FILE_SELECTED" | "ANALYZING" | "SUCCESS" | "ERROR";

export const RetinaAnalysis: React.FC = () => {
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

  // Execute Retina Analysis via API Client (Task: diabetic_retinopathy)
  const handleRunAnalysis = async () => {
    if (!selectedFile) return;

    setAnalysisState("ANALYZING");
    setErrorState(null);
    setAnalysisResult(null);

    try {
      const result = await apiClient.analyzeRetina(selectedFile, "diabetic_retinopathy");
      setAnalysisResult(result);
      setAnalysisState("SUCCESS");
    } catch (err: unknown) {
      const errObj = err as { message?: string; details?: Record<string, unknown> };
      setErrorState({
        message: errObj.message || "An unexpected error occurred during Retinal analysis.",
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
          <RetinaUpload
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
            label="Analyzing Retinal Image..."
            sublabel="Executing EfficientNet-B0 inference (diabetic retinopathy) and computing Grad-CAM visualization."
          />
        )}

        {/* ERROR State */}
        {analysisState === "ERROR" && errorState && (
          <ErrorAlert
            title="Retinal Analysis Failed"
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
                  Analysis Artifact Output (Diabetic Retinopathy)
                </span>
                <span className="text-xs font-bold text-slate-900">
                  File: {selectedFile?.name || "Uploaded Retinal Fundus Photograph"}
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

            {/* Warnings / Safety Disclaimer Banner */}
            {analysisResult.warnings.length > 0 && (
              <SafetyDisclaimer compact />
            )}

            {/* Prediction Display */}
            {analysisResult.prediction ? (
              <RetinaPrediction
                prediction={analysisResult.prediction}
                reviewRequired={analysisResult.review_required}
              />
            ) : (
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-800">
                <span className="font-semibold block mb-1">
                  Status: {analysisResult.status}
                </span>
                {analysisResult.review_reasons.map((r, i) => (
                  <p key={i}>• {r}</p>
                ))}
              </div>
            )}

            {/* Immediate XAI Visualization section for Imaging (Three-View Grad-CAM) */}
            {analysisResult.explanation && (
              <ThreeViewGradCAM
                explanation={analysisResult.explanation}
                selectedFile={selectedFile}
                modalityTitle="Retinal Fundus Screening"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

