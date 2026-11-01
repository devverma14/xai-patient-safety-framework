import React from "react";

export type ModalityOptionId = "lab" | "xray-pneumonia" | "xray-tb" | "skin" | "retina";

interface ModalityOption {
  id: ModalityOptionId;
  title: string;
  description: string;
  badges: string[];
  cardBg: string;
  cardBorder: string;
  iconBg: string;
  iconColor: string;
  iconSvg: React.ReactNode;
  inputRequired: string;
  expectedOutput: string;
  xaiMethod: string;
  clinicalNotice?: string;
}

const MODALITY_OPTIONS: ModalityOption[] = [
  {
    id: "lab",
    title: "Lab Report",
    description: "Understand your laboratory results and identify values outside their reference ranges.",
    badges: [],
    cardBg: "bg-white",
    cardBorder: "border-slate-200/90 hover:border-slate-300 hover:bg-slate-50/50",
    iconBg: "bg-sky-50 border-sky-200/80",
    iconColor: "text-sky-700",
    inputRequired: "Laboratory report (PDF or scanned image)",
    expectedOutput: "Out-of-range findings, plain-language meanings & report data",
    xaiMethod: "Deterministic biological reference-range comparison", // Active clinical pipeline (Historical research baseline: SHAP feature contributions)
    clinicalNotice: "Informational summary only — not a substitute for professional medical assessment.",
    iconSvg: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  },
  {
    id: "xray-pneumonia",
    title: "Chest X-Ray — Pneumonia",
    description: "Screens posteroanterior chest radiographs for consolidation patterns associated with pneumonia.",
    badges: ["DenseNet121", "Grad-CAM"],
    cardBg: "bg-white",
    cardBorder: "border-slate-200/90 hover:border-slate-300 hover:bg-slate-50/50",
    iconBg: "bg-teal-50 border-teal-200/80",
    iconColor: "text-teal-700",
    inputRequired: "Supported chest X-ray image (DICOM/JPEG/PNG)",
    expectedOutput: "Pneumonia-associated image-pattern classification",
    xaiMethod: "Grad-CAM visual heatmap",
    iconSvg: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    id: "xray-tb",
    title: "Chest X-Ray — Tuberculosis",
    description: "Screens posteroanterior chest radiographs for pulmonary opacities associated with tuberculosis.",
    badges: ["DenseNet121", "Grad-CAM"],
    cardBg: "bg-white",
    cardBorder: "border-slate-200/90 hover:border-slate-300 hover:bg-slate-50/50",
    iconBg: "bg-indigo-50 border-indigo-200/80",
    iconColor: "text-indigo-700",
    inputRequired: "Supported chest X-ray image (DICOM/JPEG/PNG)",
    expectedOutput: "Tuberculosis-associated image-pattern classification",
    xaiMethod: "Grad-CAM visual heatmap",
    iconSvg: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
      </svg>
    ),
  },
  {
    id: "skin",
    title: "Skin Lesion",
    description: "Evaluates dermoscopic lesion photographs for binary benign vs. malignant classification.",
    badges: ["DenseNet121", "Grad-CAM"],
    cardBg: "bg-white",
    cardBorder: "border-slate-200/90 hover:border-slate-300 hover:bg-slate-50/50",
    iconBg: "bg-rose-50 border-rose-200/80",
    iconColor: "text-rose-700",
    inputRequired: "Skin lesion dermoscopic image",
    expectedOutput: "Benign vs. Malignant classification (Higher-risk vs. Lower-risk)",
    xaiMethod: "Grad-CAM visual heatmap",
    clinicalNotice: "Binary classification triage only — not a definitive skin cancer diagnosis.",
    iconSvg: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
  },
  {
    id: "retina",
    title: "Retinal Fundus",
    description: "Screens digital fundus photographs for microvascular patterns of diabetic retinopathy.",
    badges: ["EfficientNet-B0", "Grad-CAM"],
    cardBg: "bg-white",
    cardBorder: "border-slate-200/90 hover:border-slate-300 hover:bg-slate-50/50",
    iconBg: "bg-amber-50 border-amber-200/80",
    iconColor: "text-amber-700",
    inputRequired: "Retinal fundus photograph",
    expectedOutput: "Diabetic-retinopathy-associated image-pattern classification",
    xaiMethod: "Grad-CAM visual heatmap",
    clinicalNotice: "Screens retinal microvascular patterns; does not confirm systemic diabetes.",
    iconSvg: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
  },
];

interface ModalitySelectorProps {
  selectedOptionId?: ModalityOptionId | null;
  onSelectOption: (optionId: ModalityOptionId) => void;
}

export const ModalitySelector: React.FC<ModalitySelectorProps> = ({
  selectedOptionId,
  onSelectOption,
}) => {
  const handleSelect = (optId: ModalityOptionId) => {
    onSelectOption(optId);
  };

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5" role="tablist" aria-label="Select Modality Module">
      {MODALITY_OPTIONS.map((opt) => {
        const isSelected = selectedOptionId === opt.id;

        return (
          <div
            key={opt.id}
            role="tab"
            aria-selected={isSelected}
            onClick={() => handleSelect(opt.id)}
            className={`card-lift group relative flex flex-col justify-between rounded-xl border p-4 text-left transition-all duration-150 cursor-pointer ${
              isSelected
                ? "border-sky-600 ring-1 ring-sky-600 shadow-xs bg-sky-50/20"
                : `${opt.cardBg} ${opt.cardBorder} shadow-2xs`
            }`}
          >
            <div className="space-y-3">
              {/* Header: Icon & Accurate Model / XAI Badges */}
              <div className="flex items-center justify-between gap-1.5">
                <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border ${opt.iconBg} ${opt.iconColor} shadow-2xs`}>
                  {opt.iconSvg}
                </div>
                <div className="flex flex-wrap items-center gap-1 justify-end">
                  {opt.badges.map((b) => (
                    <span
                      key={b}
                      className="text-[9px] font-mono font-medium px-1.5 py-0.5 rounded bg-slate-50 text-slate-700 border border-slate-200 shrink-0 whitespace-nowrap"
                    >
                      {b}
                    </span>
                  ))}
                </div>
              </div>

              {/* Module Title */}
              <h3 className="text-xs sm:text-sm font-bold text-slate-900 leading-snug">
                {opt.title}
              </h3>

              {/* Concise Description */}
              <p className="text-xs text-slate-600 leading-relaxed min-h-[42px]">
                {opt.description}
              </p>

              {/* Hidden Accessibility Support */}
              <div className="sr-only">
                <span>Input required: {opt.inputRequired}</span>
                <span>Output: {opt.expectedOutput}</span>
                <span>Explain: {opt.xaiMethod}</span>
                {opt.clinicalNotice && <span>{opt.clinicalNotice}</span>}
              </div>
            </div>

            {/* Consistent Unified Action Button */}
            <div className="pt-3.5 mt-auto">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleSelect(opt.id);
                }}
                className={`w-full py-2 px-3 rounded-lg font-medium text-xs flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                  isSelected
                    ? "bg-sky-700 text-white shadow-2xs"
                    : "bg-slate-900 hover:bg-slate-800 text-white shadow-2xs"
                }`}
              >
                <span>Analyze Now</span>
                <span className="font-bold">→</span>
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
};
