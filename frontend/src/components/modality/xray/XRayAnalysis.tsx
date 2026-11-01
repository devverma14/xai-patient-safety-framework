import React, { useState, useEffect } from "react";
import { XRayUpload } from "./XRayUpload";
import { XRayPrediction } from "./XRayPrediction";
import { ThreeViewGradCAM } from "@/components/xai/ThreeViewGradCAM";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { ErrorAlert } from "@/components/common/ErrorAlert";
import { SafetyDisclaimer } from "@/components/common/SafetyDisclaimer";
import { XRayTask, CommonResponsePayload } from "@/lib/types";
import { apiClient } from "@/lib/api/client";

type AnalysisState = "IDLE" | "FILE_SELECTED" | "ANALYZING" | "SUCCESS" | "ERROR";

interface XRayAnalysisProps {
  initialTask?: XRayTask;
  task?: XRayTask;
}

export const XRayAnalysis: React.FC<XRayAnalysisProps> = ({
  initialTask = "pneumonia",
  task,
}) => {
  const [currentTask, setCurrentTask] = useState<XRayTask>(task || initialTask || "pneumonia");
  const activeTask: XRayTask = currentTask;
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>("IDLE");
  const [analysisResult, setAnalysisResult] = useState<CommonResponsePayload | null>(null);
  const [errorState, setErrorState] = useState<{
    message: string;
    details?: Record<string, unknown>;
  } | null>(null);

  useEffect(() => {
    if (task) setCurrentTask(task);
    else if (initialTask) setCurrentTask(initialTask);
  }, [task, initialTask]);

  // Reset state when task changes to prevent stale results
  useEffect(() => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setErrorState(null);
    setAnalysisState("IDLE");
  }, [activeTask]);

  // Handle file selection
  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    setAnalysisResult(null);
    setErrorState(null);
    setAnalysisState(file ? "FILE_SELECTED" : "IDLE");
  };

  // Execute Analysis via API Client
  const handleRunAnalysis = async () => {
    if (!selectedFile) return;

    setAnalysisState("ANALYZING");
    setErrorState(null);
    setAnalysisResult(null);

    try {
      const result = await apiClient.analyzeXRay(selectedFile, activeTask);
      setAnalysisResult(result);
      setAnalysisState("SUCCESS");
    } catch (err: unknown) {
      const errObj = err as { message?: string; details?: Record<string, unknown> };
      setErrorState({
        message: errObj.message || "An unexpected error occurred during X-Ray analysis.",
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

  const isPneumonia = activeTask === "pneumonia";
  const taskLabel = isPneumonia ? "Pneumonia Detection" : "Tuberculosis Detection";

  return (
    <div className="space-y-6">
      {/* Upload & Controls Section */}
      <div className="space-y-6">
        {!analysisResult && (
          <XRayUpload
            selectedTask={activeTask}
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
            label={`Analyzing Chest Radiograph (${taskLabel})...`}
            sublabel="Executing DenseNet121 model inference and computing Grad-CAM visual feature activation maps."
          />
        )}

        {/* ERROR State */}
        {analysisState === "ERROR" && errorState && (
          <ErrorAlert
            title="X-Ray Analysis Failed"
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
                  Analysis Artifact Output (Chest Radiograph / {taskLabel})
                </span>
                <span className="text-xs font-bold text-slate-900">
                  File: {selectedFile?.name || "Uploaded Image"}
                </span>
              </div>
              <button
                type="button"
                onClick={handleReset}
                className="px-3.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-xs font-semibold text-slate-700 transition-colors border border-slate-200 cursor-pointer"
              >
                Analyze Another Radiograph
              </button>
            </div>

            <SafetyDisclaimer compact />

            {/* Prediction Display */}
            {analysisResult.prediction && (
              <XRayPrediction
                prediction={analysisResult.prediction}
                task={activeTask}
                reviewRequired={analysisResult.review_required}
                targetLayer={analysisResult.explanation?.details?.target_layer as string | undefined}
              />
            )}

            {/* THREE-VIEW GRAD-CAM DISPLAY — IMMEDIATE VISIBILITY */}
            {analysisResult.explanation && (
              <ThreeViewGradCAM
                explanation={analysisResult.explanation}
                selectedFile={selectedFile}
                modalityTitle={`Chest Radiograph (${taskLabel})`}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};
