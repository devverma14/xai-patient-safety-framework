import React from "react";

interface NextStepGuidanceProps {
  modality: "lab" | "xray" | "skin" | "retina";
  className?: string;
}

export const NextStepGuidance: React.FC<NextStepGuidanceProps> = ({
  modality,
  className = "",
}) => {
  let guidanceText = "";

  if (modality === "xray") {
    guidanceText =
      "Discuss this result with a qualified healthcare professional. They can interpret the image together with your symptoms, medical history, examination, and any additional tests.";
  } else if (modality === "skin") {
    guidanceText =
      "Consider review by a dermatologist or qualified healthcare professional, especially if the lesion is changing, persistent, bleeding, painful, or otherwise concerning.";
  } else if (modality === "retina") {
    guidanceText =
      "Discuss the result with an eye-care professional or qualified clinician.";
  } else if (modality === "lab") {
    guidanceText =
      "Discuss the result with your healthcare professional, who can interpret it alongside your complete medical record.";
  }

  return (
    <div className={`rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-1.5 ${className}`}>
      <div className="flex items-center gap-2">
        <svg className="w-4 h-4 text-slate-700 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900">
          What should you do next?
        </h3>
      </div>
      <p className="text-xs text-slate-600 leading-relaxed font-normal">
        {guidanceText}
      </p>
    </div>
  );
};
