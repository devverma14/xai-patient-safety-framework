import React, { useRef, useState } from "react";

export type UploadAccent = "sky" | "teal" | "violet" | "rose" | "amber";

interface UploadDropzoneProps {
  acceptTypes?: string[];
  maxSizeBytes?: number;
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClearFile: () => void;
  disabled?: boolean;
  accent?: UploadAccent;
  title?: string;
  subtitle?: string;
}

const ACCENT_STYLES: Record<
  UploadAccent,
  {
    borderIdle: string;
    borderActive: string;
    bgIdle: string;
    bgActive: string;
    iconBg: string;
    iconText: string;
    fileCardBg: string;
    fileCardBorder: string;
    fileIconBg: string;
    fileIconText: string;
  }
> = {
  sky: {
    borderIdle: "border-slate-300 hover:border-slate-400",
    borderActive: "border-sky-600 ring-2 ring-sky-600/10",
    bgIdle: "bg-slate-50/40 hover:bg-slate-50/70",
    bgActive: "bg-sky-50/60",
    iconBg: "bg-slate-100 text-sky-700 border border-slate-200/80 shadow-2xs",
    iconText: "text-sky-700",
    fileCardBg: "bg-white",
    fileCardBorder: "border-slate-200",
    fileIconBg: "bg-sky-50 text-sky-700 border-sky-200/80",
    fileIconText: "text-sky-700",
  },
  teal: {
    borderIdle: "border-slate-300 hover:border-slate-400",
    borderActive: "border-teal-600 ring-2 ring-teal-600/10",
    bgIdle: "bg-slate-50/40 hover:bg-slate-50/70",
    bgActive: "bg-teal-50/60",
    iconBg: "bg-slate-100 text-teal-700 border border-slate-200/80 shadow-2xs",
    iconText: "text-teal-700",
    fileCardBg: "bg-white",
    fileCardBorder: "border-slate-200",
    fileIconBg: "bg-teal-50 text-teal-700 border-teal-200/80",
    fileIconText: "text-teal-700",
  },
  violet: {
    borderIdle: "border-slate-300 hover:border-slate-400",
    borderActive: "border-indigo-600 ring-2 ring-indigo-600/10",
    bgIdle: "bg-slate-50/40 hover:bg-slate-50/70",
    bgActive: "bg-indigo-50/60",
    iconBg: "bg-slate-100 text-indigo-700 border border-slate-200/80 shadow-2xs",
    iconText: "text-indigo-700",
    fileCardBg: "bg-white",
    fileCardBorder: "border-slate-200",
    fileIconBg: "bg-indigo-50 text-indigo-700 border-indigo-200/80",
    fileIconText: "text-indigo-700",
  },
  rose: {
    borderIdle: "border-slate-300 hover:border-slate-400",
    borderActive: "border-rose-600 ring-2 ring-rose-600/10",
    bgIdle: "bg-slate-50/40 hover:bg-slate-50/70",
    bgActive: "bg-rose-50/60",
    iconBg: "bg-slate-100 text-rose-700 border border-slate-200/80 shadow-2xs",
    iconText: "text-rose-700",
    fileCardBg: "bg-white",
    fileCardBorder: "border-slate-200",
    fileIconBg: "bg-rose-50 text-rose-700 border-rose-200/80",
    fileIconText: "text-rose-700",
  },
  amber: {
    borderIdle: "border-slate-300 hover:border-slate-400",
    borderActive: "border-amber-600 ring-2 ring-amber-600/10",
    bgIdle: "bg-slate-50/40 hover:bg-slate-50/70",
    bgActive: "bg-amber-50/60",
    iconBg: "bg-slate-100 text-amber-700 border border-slate-200/80 shadow-2xs",
    iconText: "text-amber-700",
    fileCardBg: "bg-white",
    fileCardBorder: "border-slate-200",
    fileIconBg: "bg-amber-50 text-amber-700 border-amber-200/80",
    fileIconText: "text-amber-700",
  },
};

export const UploadDropzone: React.FC<UploadDropzoneProps> = ({
  acceptTypes = [".jpg", ".jpeg", ".png", ".pdf", ".bmp", ".tiff"],
  maxSizeBytes = 20 * 1024 * 1024,
  onFileSelect,
  selectedFile,
  onClearFile,
  disabled = false,
  accent = "sky",
  title = "Drop your medical artifact here or click to browse",
  subtitle,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const style = ACCENT_STYLES[accent] || ACCENT_STYLES.sky;

  const handleFileChange = (file: File) => {
    setErrorMsg(null);
    if (file.size > maxSizeBytes) {
      setErrorMsg(`File size exceeds limit (${(maxSizeBytes / (1024 * 1024)).toFixed(0)} MB max).`);
      return;
    }
    onFileSelect(file);
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="w-full">
      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            handleFileChange(e.target.files[0]);
          }
        }}
        accept={acceptTypes.join(",")}
        className="hidden"
        disabled={disabled}
      />

      {selectedFile ? (
        <div className={`flex items-center justify-between rounded-xl border p-3.5 sm:p-4 shadow-2xs transition-all ${style.fileCardBg} ${style.fileCardBorder}`}>
          <div className="flex items-center space-x-3 overflow-hidden">
            <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg font-bold text-xs border shadow-2xs ${style.fileIconBg}`}>
              {selectedFile.name.split(".").pop()?.toUpperCase()}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs sm:text-sm font-bold text-slate-900">{selectedFile.name}</p>
              <div className="flex items-center gap-2 pt-0.5">
                <span className="text-[11px] font-mono font-medium text-slate-500">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </span>
                <span className="text-slate-300">•</span>
                <span className="text-[11px] font-semibold text-emerald-700 flex items-center gap-1">
                  <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Ready for analysis
                </span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
              className="px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 transition-colors cursor-pointer"
            >
              Replace
            </button>
            <button
              type="button"
              onClick={onClearFile}
              disabled={disabled}
              className="p-1.5 rounded-lg text-slate-400 hover:bg-rose-50 hover:text-rose-600 transition-colors cursor-pointer"
              title="Remove file"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      ) : (
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
          className={`group flex flex-col items-center justify-center rounded-xl border border-dashed p-6 sm:p-7 text-center cursor-pointer transition-all duration-150 shadow-2xs ${
            isDragging ? `${style.borderActive} ${style.bgActive}` : `${style.borderIdle} ${style.bgIdle} hover:border-slate-400`
          } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          <div className={`flex h-11 w-11 items-center justify-center rounded-xl mb-2.5 transition-transform group-hover:scale-102 shadow-2xs ${style.iconBg}`}>
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </div>

          <p className="text-xs sm:text-sm font-semibold text-slate-900 group-hover:text-sky-700 transition-colors">
            {title}
          </p>
          <p className="mt-0.5 text-xs text-slate-500 max-w-sm">
            {subtitle || "Files are securely processed in-memory for deep feature extraction and explainability."}
          </p>

          <div className="mt-3 flex flex-wrap items-center justify-center gap-1.5">
            {acceptTypes.map((t) => (
              <span key={t} className="px-2 py-0.5 rounded bg-white border border-slate-200 text-[10px] font-mono font-medium text-slate-600 shadow-2xs">
                {t.replace(".", "").toUpperCase()}
              </span>
            ))}
            <span className="px-2 py-0.5 rounded bg-slate-100 text-[10px] font-medium text-slate-500">
              Max {(maxSizeBytes / (1024 * 1024)).toFixed(0)} MB
            </span>
          </div>
        </div>
      )}

      {errorMsg && (
        <p className="mt-2 text-xs font-semibold text-rose-600 flex items-center gap-1">
          <svg className="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <span>{errorMsg}</span>
        </p>
      )}
    </div>
  );
};
