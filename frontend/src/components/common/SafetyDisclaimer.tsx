import React from "react";

interface SafetyDisclaimerProps {
  compact?: boolean;
  className?: string;
}

export const SafetyDisclaimer: React.FC<SafetyDisclaimerProps> = ({
  compact = false,
  className = "",
}) => {
  if (compact) {
    return (
      <div className={`flex items-start gap-2.5 text-xs text-amber-900 bg-amber-50/90 border border-amber-200/80 rounded-xl p-3 shadow-2xs ${className}`}>
        <svg className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <div>
          <span className="font-bold text-amber-950">Notice: </span>
          AI-generated results are informational and do not replace professional medical assessment.
        </div>
      </div>
    );
  }

  return (
    <div className={`rounded-xl border border-amber-200/80 bg-amber-50/80 p-4 text-xs text-amber-900 shadow-2xs ${className}`}>
      <div className="flex items-center gap-2 mb-1.5 font-bold text-amber-950 text-xs uppercase tracking-wider">
        <svg className="w-4 h-4 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        Clinical Information Notice
      </div>
      <p className="leading-relaxed text-amber-900/90 font-medium">
        AI-generated results are informational and do not replace professional medical assessment.
      </p>
    </div>
  );
};
