import React, { useState } from "react";

interface ExplanationDisclosureProps {
  modality: "lab" | "xray" | "skin" | "retina";
  task?: string;
  hasShap?: boolean;
  modelName?: string;
  className?: string;
}

export const ExplanationDisclosure: React.FC<ExplanationDisclosureProps> = ({
  modality,
  task,
  hasShap = false,
  modelName,
  className = "",
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  const getDetails = () => {
    switch (modality) {
      case "xray": {
        const isTb = task === "tb";
        const taskName = isTb ? "Tuberculosis" : "Pneumonia";
        return {
          title: `How does the Chest X-Ray (${taskName}) prediction work?`,
          model: modelName || "DenseNet121",
          method: "Grad-CAM (Gradient-Weighted Class Activation Mapping)",
          inferencePoints: [
            `DenseNet121 evaluates the standardized posteroanterior (PA) radiograph after normalization and resizing.`,
            `The network extracts deep visual features across 121 layers to estimate pattern probability for ${taskName} against a validated decision threshold.`,
          ],
          xaiPoints: [
            `Grad-CAM tracks the gradient flow from the predicted class score back to the final convolutional layer (features.denseblock4.denselayer16.conv2).`,
            `Warmer colors (red/orange) highlight thoracic regions that most strongly increased the model's score; cooler colors (blue) had minimal impact.`,
            `The visualization reflects algorithmic focus—it does not define anatomical boundaries or pathological margins.`,
          ],
        };
      }
      case "skin": {
        return {
          title: "How does the Skin Lesion Benign vs. Malignant classification work?",
          model: modelName || "DenseNet121",
          method: "Grad-CAM (Gradient-Weighted Class Activation Mapping)",
          inferencePoints: [
            `DenseNet121 binary classifier evaluates dermoscopic patterns for benign vs. malignant classification (melanoma, basal cell carcinoma, actinic keratosis vs. benign nevi and keratoses).`,
            `Applies a frozen decision threshold (t = 0.40) to assign lesions into research-defined higher-risk versus lower-risk triage groupings.`,
          ],
          xaiPoints: [
            `Grad-CAM maps gradients from the final convolutional layer into an attribution grid upsampled to the original image dimensions.`,
            `Highlights localized dermatological structures, pigment networks, and borders prioritized by the network.`,
            `Visual heatmaps show model weight attribution and cannot replace a biopsy or in-person clinical exam.`,
          ],
        };
      }
      case "retina": {
        return {
          title: "How does the Retinal Fundus prediction work?",
          model: modelName || "EfficientNet-B0",
          method: "Grad-CAM (Gradient-Weighted Class Activation Mapping)",
          inferencePoints: [
            `EfficientNet-B0 convolutional network trained on APTOS-2019 fundus photographs evaluates vascular and retinal features.`,
            `Uses a calibrated decision threshold (t = 0.50) to classify visual patterns associated with Diabetic Retinopathy.`,
          ],
          xaiPoints: [
            `Grad-CAM visualizes spatial feature activations from the final stage feature extractor of EfficientNet-B0.`,
            `Warmer regions pinpoint retinal quadrants, microvascular patterns, or lesions that contributed most to the model estimate.`,
            `The heatmap illustrates machine focus and does not represent confirmed clinical grading or systemic diabetes diagnosis.`,
          ],
        };
      }
      case "lab": {
        return {
          title: "How does the Laboratory Report Review work?",
          model: "Standard Laboratory Reference Ranges",
          method: "Biological Reference Interval Comparison",
          inferencePoints: [
            `Laboratory test values, measurement units, and reference ranges are read directly from your uploaded document.`,
            `Reported values are organized into standard clinical panels for easy viewing.`,
            `Results are evaluated directly against clinical reference intervals without speculative prediction models.`,
          ],
          xaiPoints: [
            `Values outside standard physiological ranges are clearly flagged as High or Low.`,
            `Plain-language summaries provide context on what each out-of-range finding typically means.`,
            `All findings are informational and meant to guide your conversation with a healthcare professional.`,
          ],
        };
      }
    }
  };

  const details = getDetails();

  return (
    <div className={`rounded-xl border border-slate-200 bg-white overflow-hidden shadow-2xs transition-all ${className}`}>
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3.5 sm:p-4 text-left hover:bg-slate-50/80 transition-colors cursor-pointer group"
        aria-expanded={isExpanded}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-sky-50 text-sky-600 border border-sky-200">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="min-w-0">
            <span className="text-xs font-bold text-slate-800 group-hover:text-sky-700 transition-colors block truncate">
              {details.title}
            </span>
            <span className="text-[10px] text-slate-500 font-mono block">
              {details.model} · {details.method}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0 pl-2">
          <span className="text-[11px] font-semibold text-sky-600 group-hover:text-sky-700 hidden sm:inline">
            {isExpanded ? "Hide Explanation" : "Understand this result"}
          </span>
          <svg
            className={`w-4 h-4 text-slate-400 group-hover:text-sky-600 transition-transform duration-200 ${
              isExpanded ? "rotate-180 text-sky-600" : ""
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {isExpanded && (
        <div className="border-t border-slate-100 bg-slate-50/60 p-4 sm:p-5 space-y-4 text-xs leading-relaxed">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Column 1: Model Prediction Process */}
            <div className="bg-white p-3.5 rounded-xl border border-slate-200/80 space-y-2">
              <div className="flex items-center gap-1.5 text-slate-800 font-bold text-[11px] uppercase tracking-wider">
                <span className="h-1.5 w-1.5 rounded-full bg-sky-500" />
                <span>Prediction Process ({details.model})</span>
              </div>
              <ul className="space-y-1.5 text-slate-600 text-[11px]">
                {details.inferencePoints.map((pt, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-sky-500 font-bold">•</span>
                    <span>{pt}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Column 2: Explainability Method */}
            <div className="bg-white p-3.5 rounded-xl border border-slate-200/80 space-y-2">
              <div className="flex items-center gap-1.5 text-slate-800 font-bold text-[11px] uppercase tracking-wider">
                <span className="h-1.5 w-1.5 rounded-full bg-teal-500" />
                <span>Explainability Method ({details.method})</span>
              </div>
              <ul className="space-y-1.5 text-slate-600 text-[11px]">
                {details.xaiPoints.map((pt, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-teal-500 font-bold">•</span>
                    <span>{pt}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="flex items-center gap-2 text-[10.5px] text-slate-500 pt-1 border-t border-slate-200/60">
            <svg className="w-3.5 h-3.5 text-slate-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span>
              Explanations reflect internal mathematical attribution maps and do not constitute independent clinical diagnosis.
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
