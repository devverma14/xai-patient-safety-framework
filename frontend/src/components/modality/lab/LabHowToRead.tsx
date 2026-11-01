import React, { useState } from "react";

export const LabHowToRead: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);

  return (
    <div className="rounded-xl border border-sky-200/80 bg-linear-to-r from-sky-50/70 via-white to-teal-50/50 shadow-2xs overflow-hidden">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-3.5 sm:px-4 text-left hover:bg-sky-50/50 transition-colors cursor-pointer"
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-sky-100 text-sky-700 shrink-0">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </span>
          <div>
            <span className="text-xs font-bold text-slate-900 block">
              How to Read This Report
            </span>
            <span className="text-[11px] text-slate-500 block">
              Reference intervals, status definitions, and clinical interpretation context
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold text-sky-700 hidden sm:inline">
            {isOpen ? "Hide Guide" : "Read Guide"}
          </span>
          <svg
            className={`w-4 h-4 text-slate-500 transition-transform duration-200 ${
              isOpen ? "rotate-180" : ""
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {isOpen && (
        <div className="p-4 pt-1 sm:px-5 pb-4 border-t border-sky-100/80 space-y-3.5 text-xs">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 pt-2">
            {/* High */}
            <div className="rounded-lg bg-white p-3 border border-rose-100 space-y-1 shadow-2xs">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-rose-500" />
                <span className="font-bold text-rose-800 text-[11px] uppercase tracking-wider">
                  High
                </span>
              </div>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Measured analyte concentration is <strong className="text-rose-900">above</strong> the applicable physiological reference interval reported by the testing laboratory or standard clinical reference.
              </p>
            </div>

            {/* Low */}
            <div className="rounded-lg bg-white p-3 border border-blue-100 space-y-1 shadow-2xs">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-blue-500" />
                <span className="font-bold text-blue-800 text-[11px] uppercase tracking-wider">
                  Low
                </span>
              </div>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Measured analyte concentration is <strong className="text-blue-900">below</strong> the applicable physiological reference interval reported by the testing laboratory or standard clinical reference.
              </p>
            </div>

            {/* Within Range */}
            <div className="rounded-lg bg-white p-3 border border-emerald-100 space-y-1 shadow-2xs">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span className="font-bold text-emerald-800 text-[11px] uppercase tracking-wider">
                  Within Range
                </span>
              </div>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Measured analyte concentration falls <strong className="text-emerald-900">inside</strong> the expected physiological reference interval for healthy individuals.
              </p>
            </div>

            {/* Needs Verification */}
            <div className="rounded-lg bg-white p-3 border border-amber-200 space-y-1 shadow-2xs">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-amber-500" />
                <span className="font-bold text-amber-800 text-[11px] uppercase tracking-wider">
                  Needs Verification
                </span>
              </div>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Extraction, source measurement, reference interval, or evaluation remains <strong className="text-amber-900">unresolved</strong> (e.g. unperformed tests, OCR candidate conflict, or missing units). Never auto-guessed.
              </p>
            </div>
          </div>

          {/* Contextual & Diagnostic Limitation Note */}
          <div className="rounded-lg bg-sky-50/80 p-3 border border-sky-100 text-[11px] text-slate-700 space-y-1 leading-relaxed">
            <span className="font-bold text-sky-950 block text-[11px] uppercase tracking-wide">
              Clinical Context & Reference Variation
            </span>
            <p>
              Reference intervals represent statistical percentiles derived from healthy reference populations and naturally vary depending on testing methodology, laboratory instrument calibration, patient age, biological sex, fasting status, and pregnancy.
            </p>
            <p className="pt-0.5 text-slate-600 italic">
              <strong>Important:</strong> An isolated out-of-range finding does not independently establish a medical diagnosis. Transient variations often occur due to hydration, diet, exercise, or medications. All abnormal results must be interpreted by a qualified healthcare professional in the context of your complete clinical history.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
