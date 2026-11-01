import React from "react";

interface LabValidationStatusProps {
  predictionStatus?: string;
  humanReviewRequired: boolean;
  reviewReasons: string[];
  missingFeatures: string[];
  readinessSummary?: Record<string, string>;
  disclaimer?: string;
  onCorrectInput?: () => void;
}

export const LabValidationStatus: React.FC<LabValidationStatusProps> = ({
  humanReviewRequired,
  reviewReasons,
  missingFeatures,
  disclaimer,
  onCorrectInput,
}) => {
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50/90 p-5 sm:p-6 space-y-4 text-left shadow-xs">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 border-b border-amber-200/80 pb-3">
        <div className="flex items-center gap-2.5 text-amber-950 font-bold text-sm">
          <div className="w-8 h-8 rounded-lg bg-amber-100 flex items-center justify-center shrink-0 border border-amber-300">
            <svg className="w-4 h-4 text-amber-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div>
            <h3 className="font-extrabold text-amber-950 text-sm">Laboratory Parameter Review Required</h3>
            <span className="text-[11px] font-normal text-amber-800">
              Input validation safety gate: extracted parameters or physiological findings require clinician confirmation.
            </span>
          </div>
        </div>
        {humanReviewRequired && (
          <span className="shrink-0 px-2.5 py-1 rounded-md text-[10px] font-extrabold bg-amber-200 text-amber-900 border border-amber-300 uppercase tracking-wider">
            Review Required
          </span>
        )}
      </div>

      <div className="rounded-lg bg-white/80 p-3.5 border border-amber-200 text-xs text-amber-950 leading-relaxed space-y-2">
        <p className="font-medium">
          The laboratory analysis module executes document OCR, regex parameter extraction, and deterministic clinical reference-range comparison across established physiological intervals (e.g., fasting glucose, HbA1c, serum creatinine, and hematology panels).
        </p>
        <p className="text-[11px] text-amber-900">
          Extracted laboratory parameters are compared deterministically against biological reference intervals. To prevent unverified optical extractions from impacting clinical decision workflows, ambiguous or flagged findings pause for clinician confirmation.
        </p>
        <div className="flex items-start gap-2 pt-1 border-t border-amber-100 text-amber-900 text-[11px]">
          <span className="font-bold text-amber-700 uppercase tracking-wider">Zero-Default Guarantee:</span>
          <span>Missing parameters are never silently fabricated, assumed, or statistically imputed.</span>
        </div>
      </div>

      {/* Review Reasons */}
      {reviewReasons.length > 0 && (
        <div className="rounded-lg bg-white p-3.5 border border-amber-200 text-xs space-y-1.5">
          <span className="font-bold text-amber-900 block text-[11px] uppercase tracking-wider">Clinical Safety Reasons:</span>
          {reviewReasons.map((reason, idx) => (
            <div key={idx} className="flex items-start gap-2 text-amber-900/90 text-xs">
              <span className="text-amber-600 font-bold mt-0.5">•</span>
              <span>{reason}</span>
            </div>
          ))}
        </div>
      )}

      {/* Missing Required Features */}
      {missingFeatures.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-bold text-amber-900">
              Missing Required Clinical Variables for Readmission Model ({missingFeatures.length}):
            </span>
            <span className="text-[11px] text-amber-700 font-mono">
              Explicit clinical documentation required
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto p-2 rounded-lg bg-white/70 border border-amber-200">
            {missingFeatures.map((feat, idx) => (
              <span
                key={idx}
                className="px-2 py-0.5 rounded text-[11px] font-mono bg-rose-50 border border-rose-200 text-rose-800 font-medium"
              >
                {feat}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Actionable Next Steps & Correction */}
      <div className="rounded-lg bg-amber-100/60 border border-amber-300 p-3.5 text-xs text-amber-950 space-y-2">
        <span className="font-bold text-amber-950 block text-[11px] uppercase tracking-wider">
          Next Steps for Complete Encounter Assessment:
        </span>
        <ul className="list-disc list-inside space-y-1 text-amber-900 text-xs">
          <li>The laboratory report findings above remain fully available for clinical review and educational reference.</li>
          <li>To run the 30-day readmission prediction model, upload documentation containing the patient&apos;s complete hospital discharge summary or supply explicit clinical encounter variables.</li>
        </ul>
        {onCorrectInput && (
          <div className="pt-2">
            <button
              type="button"
              onClick={onCorrectInput}
              className="px-4 py-2 rounded-lg bg-amber-800 hover:bg-amber-900 text-white text-xs font-bold transition-colors shadow-xs cursor-pointer inline-flex items-center gap-2"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              Upload Additional Documentation or Analyze Another File
            </button>
          </div>
        )}
      </div>

      {disclaimer && (
        <p className="text-[11px] text-amber-800 italic border-t border-amber-200 pt-2 font-medium">
          {disclaimer}
        </p>
      )}
    </div>
  );
};
