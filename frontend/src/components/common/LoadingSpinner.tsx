import React from "react";

interface LoadingSpinnerProps {
  label?: string;
  sublabel?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  label = "Analyzing Input...",
  sublabel = "Evaluating model inference and computing visual/feature XAI attributions.",
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center rounded-xl border border-slate-200 bg-white shadow-xs">
      <div className="relative flex h-12 w-12 items-center justify-center">
        <div className="absolute h-12 w-12 rounded-full border-2 border-sky-100" />
        <div className="h-12 w-12 rounded-full border-2 border-sky-600 border-t-transparent animate-spin" />
      </div>
      <p className="mt-4 text-sm font-bold text-slate-900">{label}</p>
      {sublabel && <p className="mt-1 text-xs text-slate-500 max-w-md leading-relaxed">{sublabel}</p>}
    </div>
  );
};
