import React from "react";
import { XRayTask } from "@/lib/types";

interface XRayTaskSelectorProps {
  selectedTask: XRayTask;
  onSelectTask: (task: XRayTask) => void;
  disabled?: boolean;
}

export const XRayTaskSelector: React.FC<XRayTaskSelectorProps> = ({
  selectedTask,
  onSelectTask,
  disabled = false,
}) => {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <span className="text-[11px] font-bold uppercase tracking-wider text-sky-700">
            Task Configuration
          </span>
          <h2 className="text-base font-bold text-slate-900 mt-0.5">
            Select Chest X-Ray Analysis Task
          </h2>
        </div>
        <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 border border-slate-200 font-medium">
          Modality: Chest Radiograph
        </span>
      </div>

      <p className="text-xs text-slate-600 leading-relaxed font-medium">
        Select the target diagnostic task. The selected option configures the corresponding deep learning vision architecture and Grad-CAM visual attribution engine.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Pneumonia Task Option */}
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSelectTask("pneumonia")}
          className={`relative p-4 rounded-xl border text-left transition-all flex flex-col justify-between space-y-3 ${
            selectedTask === "pneumonia"
              ? "bg-white border-sky-600 ring-2 ring-sky-600/20 shadow-xs"
              : "bg-slate-50/50 border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-white"
          } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  selectedTask === "pneumonia" ? "bg-sky-600" : "bg-slate-300"
                }`}
              />
              <span className="text-xs font-bold text-slate-900">
                Pneumonia Detection
              </span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-sky-800 border border-slate-200 font-semibold">
              DenseNet121
            </span>
          </div>

          <p className="text-[11px] text-slate-600 leading-normal">
            Evaluates pulmonary consolidation and infiltrates from chest radiographs using DenseNet121.
          </p>

          <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-500 font-medium">
            <span>Target: Pneumonia vs Normal</span>
            <span className="font-bold text-sky-700">Grad-CAM XAI</span>
          </div>
        </button>

        {/* Tuberculosis Task Option */}
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSelectTask("tb")}
          className={`relative p-4 rounded-xl border text-left transition-all flex flex-col justify-between space-y-3 ${
            selectedTask === "tb"
              ? "bg-white border-sky-600 ring-2 ring-sky-600/20 shadow-xs"
              : "bg-slate-50/50 border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-white"
          } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  selectedTask === "tb" ? "bg-sky-600" : "bg-slate-300"
                }`}
              />
              <span className="text-xs font-bold text-slate-900">
                Tuberculosis (TB) Detection
              </span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-teal-800 border border-slate-200 font-semibold">
              DenseNet121
            </span>
          </div>

          <p className="text-[11px] text-slate-600 leading-normal">
            Evaluates apical cavities and tuberculous opacities from chest radiographs using DenseNet121.
          </p>

          <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-500 font-medium">
            <span>Target: Tuberculosis vs Normal</span>
            <span className="font-bold text-teal-700">Grad-CAM XAI</span>
          </div>
        </button>
      </div>
    </div>
  );
};
