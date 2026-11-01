import React from "react";

export const Footer: React.FC = () => {
  return (
    <footer className="mt-12 border-t border-slate-200 bg-white py-6 text-xs text-slate-500">
      <div className="mx-auto max-w-[1380px] px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Left: Copyright & Descriptor */}
        <div className="flex items-center gap-2 text-slate-500 font-normal text-[11px] sm:text-xs">
          <span>© 2026 Patient Safety XAI. Explainable AI for Patient Safety.</span>
        </div>

        {/* Right: Technical Stack Pill Badges */}
        <div className="flex flex-wrap items-center gap-2.5 text-[11px] font-medium text-slate-500">
          <span className="hover:text-slate-800 transition-colors">Lab Report Review</span>
          <span className="text-slate-300">•</span>
          <span className="hover:text-slate-800 transition-colors">Imaging Grad-CAM</span>
          <span className="text-slate-300">•</span>
          <span className="hover:text-slate-800 transition-colors">DenseNet121</span>
          <span className="text-slate-300">•</span>
          <span className="hover:text-slate-800 transition-colors">EfficientNet-B0</span>
        </div>
      </div>
    </footer>
  );
};
