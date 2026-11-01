import React from "react";

interface PatientFriendlyExplanationProps {
  modality: "lab" | "xray" | "skin" | "retina";
  task?: string;
  isPositiveOrHighRisk: boolean;
  className?: string;
}

export const PatientFriendlyExplanation: React.FC<PatientFriendlyExplanationProps> = ({
  modality,
  task,
  isPositiveOrHighRisk,
  className = "",
}) => {
  let explanationText = "";

  if (modality === "xray") {
    if (task === "tb") {
      explanationText =
        "The model found patterns in this chest X-ray that were more similar to tuberculosis examples in the data used to train the model.";
    } else {
      // pneumonia (default for xray)
      explanationText =
        "The model found patterns in this chest X-ray that were more similar to pneumonia examples in the data used to train the model.";
    }
  } else if (modality === "skin") {
    if (isPositiveOrHighRisk) {
      explanationText =
        "The image contains visual patterns that the model associates with the project's higher-risk category.";
    } else {
      explanationText =
        "The model found a pattern that is more similar to the project's lower-risk category.";
    }
  } else if (modality === "retina") {
    explanationText =
      "The model found patterns in the retinal image that are associated with diabetic retinopathy in the data used to train the model.";
  } else if (modality === "lab") {
    explanationText =
      "Laboratory test results were reviewed against standard physiological reference ranges to identify any values outside normal limits.";
  }

  return (
    <div className={`rounded-xl border border-sky-100 bg-sky-50/70 p-4 space-y-1.5 ${className}`}>
      <div className="flex items-center gap-2">
        <svg className="w-4 h-4 text-sky-700 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 className="text-xs font-bold uppercase tracking-wider text-sky-900">
          What does this mean?
        </h3>
      </div>
      <p className="text-xs text-slate-700 leading-relaxed font-normal">
        {explanationText}
      </p>
    </div>
  );
};
