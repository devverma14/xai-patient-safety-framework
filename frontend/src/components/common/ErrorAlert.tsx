import React from "react";

interface ErrorAlertProps {
  title?: string;
  message: string;
  details?: Record<string, unknown>;
  onRetry?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  title = "Unable to analyze this file",
  message,
  details,
  onRetry,
}) => {
  // Replace technical internal messages with patient-friendly wording if needed
  const displayTitle =
    message.toLowerCase().includes("image") || message.toLowerCase().includes("file")
      ? "Unable to analyze this image"
      : title;

  const displayMessage =
    message.includes("INVALID_INPUT") || message.includes("corrupt") || message.includes("unsupported")
      ? "Please upload a supported image file and try again."
      : message;

  return (
    <div className="rounded-xl border border-rose-200 bg-rose-50 p-5 text-left text-rose-900 shadow-xs">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2 font-bold text-rose-900 text-sm">
          <svg className="w-5 h-5 text-rose-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {displayTitle}
        </div>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="text-xs font-semibold text-rose-700 hover:text-rose-900 underline underline-offset-2 cursor-pointer"
          >
            Try Again
          </button>
        )}
      </div>
      <p className="mt-2 text-xs text-rose-800 leading-relaxed font-medium">{displayMessage}</p>
      {details && Object.keys(details).length > 0 && (
        <div className="mt-3 rounded-lg bg-white p-3 border border-rose-200 text-xs font-mono text-slate-700">
          <span className="font-bold block text-[10px] text-slate-400 uppercase mb-1">Details</span>
          {Object.entries(details)
            .filter(([k]) => !["traceback", "stack", "file_path", "python_exception", "internal_error"].includes(k.toLowerCase()))
            .map(([k, v]) => (
              <div key={k} className="flex justify-between py-0.5">
                <span className="text-slate-500 font-sans">{k}:</span>
                <span className="font-semibold">{String(v)}</span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
};
