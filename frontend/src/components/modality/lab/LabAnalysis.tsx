import React, { useState } from "react";
import { UploadDropzone } from "@/components/upload/UploadDropzone";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { ErrorAlert } from "@/components/common/ErrorAlert";
import { SafetyDisclaimer } from "@/components/common/SafetyDisclaimer";
import { LabKeyFindings } from "./LabKeyFindings";
import { LabExtractionResults } from "./LabExtractionResults";
import { CommonResponsePayload, LabPayloadDetails } from "@/lib/types";
import { apiClient } from "@/lib/api/client";

export const LabAnalysis: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [result, setResult] = useState<CommonResponsePayload | null>(null);
  const [errorState, setErrorState] = useState<{ message: string; details?: Record<string, unknown> } | null>(null);
  const [showConflictsExpanded, setShowConflictsExpanded] = useState<boolean>(false);

  const handleRunAnalysis = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setResult(null);
    setErrorState(null);

    try {
      const res = await apiClient.analyzeLab(selectedFile);
      setResult(res);
    } catch (err: unknown) {
      const errObj = err as { message?: string; details?: Record<string, unknown> };
      setErrorState({
        message: errObj.message || "An error occurred during Lab Report analysis.",
        details: errObj.details,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setErrorState(null);
    setIsAnalyzing(false);
    setShowConflictsExpanded(false);
  };

  const payloadDetails = result?.payload_details as (LabPayloadDetails & { patient_demographics?: Record<string, any> }) | undefined;
  const clinicalSummary = payloadDetails?.clinical_summary;
  const rawLabValues = payloadDetails?.laboratory_values || [];

  // Filter actual unresolved conflicts from review reasons
  const unresolvedReasons = (result?.review_reasons || []).filter(
    (r) =>
      !r.toLowerCase().includes("unextracted/missing clinical features") &&
      !r.toLowerCase().includes("population mode/median")
  );

  // Extract patient demographic metadata for display
  const patientDemographics = (clinicalSummary?.patient_demographics || payloadDetails?.patient_demographics || {}) as Record<string, any>;
  const ageItem =
    patientDemographics["age"] ||
    rawLabValues.find((v) => ["age", "patient_age"].includes((v.normalized_name || "").toLowerCase()));
  const genderItem =
    patientDemographics["gender"] ||
    rawLabValues.find((v) => ["gender", "sex"].includes((v.normalized_name || "").toLowerCase()));

  // Pure laboratory values (strictly excluding demographic records from all lab lists and counts)
  const isDemographic = (item: { normalized_name?: string; test_name?: string }) => {
    const norm = (item.normalized_name || "").toLowerCase().trim();
    const name = (item.test_name || "").toLowerCase().trim();
    return (
      ["age", "gender", "sex", "race", "patient_age", "patient_gender"].includes(norm) ||
      ["age", "gender", "sex", "race"].includes(name) ||
      name.startsWith("age ") ||
      name.startsWith("gender ")
    );
  };
  const labValues = rawLabValues.filter((v) => !isDemographic(v));

  const hasDemographics = Boolean(ageItem || genderItem);
  const demographicsVerified =
    (!ageItem || ageItem.validation_status === "VALID") &&
    (!genderItem || genderItem.validation_status === "VALID");

  return (
    <div className="space-y-6">
      {/* Unified Lab Report Upload Card */}
      {!result && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
          <div className="border-b border-slate-100 pb-3.5">
            <h2 className="text-lg font-bold text-slate-900">
              Understand Your Lab Report
            </h2>
            <p className="text-xs text-slate-500 mt-1 max-w-2xl leading-relaxed font-medium">
              Upload a laboratory report to review abnormal results and understand what they may mean.
            </p>
          </div>

          <UploadDropzone
            acceptTypes={[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"]}
            onFileSelect={setSelectedFile}
            selectedFile={selectedFile}
            onClearFile={() => setSelectedFile(null)}
            disabled={isAnalyzing}
            accent="sky"
            title="Upload your laboratory report (PDF or image)"
            subtitle="Supports blood, metabolic, and routine lab panels in PDF, PNG, JPEG, or TIFF formats."
          />

          <button
            type="button"
            onClick={handleRunAnalysis}
            disabled={!selectedFile || isAnalyzing}
            className={`w-full py-2.5 px-4 rounded-lg font-semibold text-xs transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-2xs ${
              !selectedFile || isAnalyzing
                ? "bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200"
                : "bg-sky-700 hover:bg-sky-800 text-white"
            }`}
          >
            {isAnalyzing ? (
              <>
                <span className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
                Reviewing Laboratory Report...
              </>
            ) : (
              <>
                Review Lab Report
              </>
            )}
          </button>

          <p className="text-[11px] text-slate-500 text-center font-medium">
            Supported Formats: PDF, PNG, JPG, JPEG, BMP, TIFF (Up to 15MB). Reports are processed securely in memory and are not permanently stored.
          </p>
        </div>
      )}

      {/* Loading Spinner */}
      {isAnalyzing && (
        <LoadingSpinner
          label="Reviewing Laboratory Report..."
          sublabel="Reading test values and comparing against standard reference ranges."
        />
      )}

      {/* Error Alert */}
      {errorState && (
        <ErrorAlert
          title="Lab Report Processing Failed"
          message={errorState.message}
          details={errorState.details}
          onRetry={handleRunAnalysis}
        />
      )}

      {/* Analysis Results Display */}
      {result && (
        <div className="space-y-6">
          {/* Header Action Bar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white p-4 rounded-2xl border border-slate-200 shadow-xs">
            <div>
              <span className="text-[10px] font-bold text-sky-700 uppercase tracking-wider block font-semibold">
                Laboratory Report Review
              </span>
              <span className="text-sm font-extrabold text-slate-900 mt-0.5 block">
                File: {selectedFile?.name || "Uploaded Report"}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleReset}
                className="px-3.5 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-700 text-xs font-semibold text-white transition-colors cursor-pointer shadow-xs"
              >
                Analyze Another Report
              </button>
            </div>
          </div>

          {/* Verified Patient Demographics Banner */}
          {hasDemographics && (
            <div
              className={`flex flex-wrap items-center justify-between gap-3 p-3.5 px-4 rounded-xl shadow-2xs border ${
                demographicsVerified
                  ? "bg-linear-to-r from-sky-50/80 via-white to-slate-50 border-sky-200/80"
                  : "bg-amber-50 border-amber-200"
              }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`p-1.5 rounded-lg shrink-0 ${
                    demographicsVerified
                      ? "bg-sky-100 text-sky-700"
                      : "bg-amber-100 text-amber-700"
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                    />
                  </svg>
                </div>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
                  <span className="font-bold text-slate-700 uppercase tracking-wide text-[10px]">
                    Patient Demographics
                  </span>
                  {ageItem && (
                    <span className="text-slate-900 font-medium">
                      Age: <strong className="font-mono text-slate-950 font-bold">{ageItem.raw_value_str || String(ageItem.value || "—")} yrs</strong>
                    </span>
                  )}
                  {genderItem && (
                    <span className="text-slate-900 font-medium">
                      Gender: <strong className="font-mono text-slate-950 font-bold">{genderItem.raw_value_str || String(genderItem.value || "—")}</strong>
                    </span>
                  )}
                </div>
              </div>

              <div>
                {demographicsVerified ? (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide bg-emerald-50 text-emerald-800 border border-emerald-200">
                    <svg className="w-3 h-3 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                    </svg>
                    Verified Patient Metadata
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide bg-amber-100 text-amber-800 border border-amber-200">
                    Demographic Verification Needed
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Medical Safety Disclaimer */}
          <SafetyDisclaimer compact />

          {/* Compact Warning Banner for Unresolved Conflicts */}
          {unresolvedReasons.length > 0 && (
            <div className="rounded-xl border border-amber-300 bg-amber-50/80 p-3.5 shadow-2xs space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-amber-900 font-bold text-xs">
                  <svg className="w-4 h-4 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <span>
                    Some results need verification — {unresolvedReasons.length} potential extraction conflict{unresolvedReasons.length > 1 ? "s" : ""}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setShowConflictsExpanded(!showConflictsExpanded)}
                  className="text-xs font-semibold text-amber-800 hover:text-amber-950 underline cursor-pointer"
                >
                  {showConflictsExpanded ? "Hide Details" : "View Details"}
                </button>
              </div>

              {showConflictsExpanded && (
                <div className="pt-2 border-t border-amber-200/80 text-xs text-amber-900 space-y-1.5">
                  <p className="text-[11px] text-amber-800">
                    The following items have competing extraction candidates or unperformed statuses. Verify marked values against your physical report:
                  </p>
                  <ul className="list-disc list-inside space-y-1 pl-1">
                    {unresolvedReasons.map((reason, idx) => (
                      <li key={idx} className="font-mono text-[11px]">{reason}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* A. Key Findings & B. What Each Finding May Mean */}
          {clinicalSummary && (
            <LabKeyFindings clinicalSummary={clinicalSummary} />
          )}

          {/* C. Your Report Data */}
          {payloadDetails && (
            <LabExtractionResults
              extractionMethod={payloadDetails.extraction_method}
              labValues={labValues}
            />
          )}
        </div>
      )}
    </div>
  );
};
