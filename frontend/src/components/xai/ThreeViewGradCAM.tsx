import React, { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { ExplanationDetails, GradCAMDetails } from "@/lib/types";

interface ThreeViewGradCAMProps {
  explanation: ExplanationDetails;
  selectedFile: File | null;
  modalityTitle?: string;
  modality?: "xray" | "skin" | "retina";
}

// Jet colormap mapping from 0.0-1.0 normalized activation
function getJetColor(val: number): [number, number, number] {
  const v = Math.min(1.0, Math.max(0.0, val));
  let r = 0, g = 0, b = 0;

  if (v < 0.25) {
    b = Math.round(128 + v * 4 * 127);
  } else if (v < 0.5) {
    g = Math.round((v - 0.25) * 4 * 255);
    b = Math.round(255 - (v - 0.25) * 4 * 255);
  } else if (v < 0.75) {
    r = Math.round((v - 0.5) * 4 * 255);
    g = 255;
  } else {
    r = 255;
    g = Math.round(255 - (v - 0.75) * 4 * 255);
  }

  return [r, g, b];
}

export const ThreeViewGradCAM: React.FC<ThreeViewGradCAMProps> = ({
  explanation,
  selectedFile,
}) => {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);

  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Cached decoded in-memory assets for instantaneous, flicker-free slider blending
  const cachedImageRef = useRef<HTMLImageElement | null>(null);
  const cachedHeatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const currentOpacityRef = useRef<number>(overlayOpacity);
  currentOpacityRef.current = overlayOpacity;

  const imagePreviewUrl = useMemo(() => {
    if (!selectedFile) return null;
    return URL.createObjectURL(selectedFile);
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
    };
  }, [imagePreviewUrl]);

  const details = (explanation.details || {}) as GradCAMDetails;
  const grid7x7Json = JSON.stringify(details.spatial_summary_grid_7x7 || []);

  // Synchronous draw function for the blended overlay (Panel 3)
  const drawOverlay = useCallback((opacity: number) => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    const width = 224;
    const height = 224;

    if (overlayCanvas.width !== width) overlayCanvas.width = width;
    if (overlayCanvas.height !== height) overlayCanvas.height = height;

    const overCtx = overlayCanvas.getContext("2d");
    if (!overCtx) return;

    overCtx.clearRect(0, 0, width, height);

    const img = cachedImageRef.current;
    const heatCanvas = cachedHeatmapCanvasRef.current;

    // 1. Draw solid base image
    if (img && img.naturalWidth > 0) {
      overCtx.globalAlpha = 1.0;
      overCtx.drawImage(img, 0, 0, width, height);
    }

    // 2. Blend pre-rendered heatmap with requested opacity
    if (heatCanvas) {
      const clampedAlpha = Math.min(1.0, Math.max(0.0, opacity));
      overCtx.globalAlpha = clampedAlpha;
      overCtx.drawImage(heatCanvas, 0, 0, width, height);
      overCtx.globalAlpha = 1.0;
    }
  }, []);

  // Pipeline 1: Preload & cache original uploaded image into memory
  useEffect(() => {
    if (!imagePreviewUrl) {
      cachedImageRef.current = null;
      drawOverlay(currentOpacityRef.current);
      return;
    }

    let isCancelled = false;
    const img = new Image();
    // Note: Do NOT set crossOrigin on local blob URLs to prevent browser CORS decode rejection
    img.onload = () => {
      if (isCancelled) return;
      cachedImageRef.current = img;
      drawOverlay(currentOpacityRef.current);
    };
    img.onerror = () => {
      if (isCancelled) return;
      drawOverlay(currentOpacityRef.current);
    };
    img.src = imagePreviewUrl;

    if (img.complete && img.naturalWidth > 0) {
      cachedImageRef.current = img;
      drawOverlay(currentOpacityRef.current);
    }

    return () => {
      isCancelled = true;
    };
  }, [imagePreviewUrl, drawOverlay]);

  // Pipeline 2: Compute & cache 224x224 Grad-CAM heatmap from 7x7 spatial summary
  useEffect(() => {
    let grid7x7: number[] = [];
    try {
      grid7x7 = JSON.parse(grid7x7Json);
    } catch {
      return;
    }
    if (grid7x7.length !== 49) return;

    const width = 224;
    const height = 224;

    if (!cachedHeatmapCanvasRef.current) {
      cachedHeatmapCanvasRef.current = document.createElement("canvas");
      cachedHeatmapCanvasRef.current.width = width;
      cachedHeatmapCanvasRef.current.height = height;
    }
    const offCanvas = cachedHeatmapCanvasRef.current;
    const offCtx = offCanvas.getContext("2d");
    if (!offCtx) return;

    // A. Bilinear upsample from 7x7 to 224x224
    const tinyCanvas = document.createElement("canvas");
    tinyCanvas.width = 7;
    tinyCanvas.height = 7;
    const tinyCtx = tinyCanvas.getContext("2d");
    if (!tinyCtx) return;

    const tinyImgData = tinyCtx.createImageData(7, 7);
    for (let i = 0; i < 49; i++) {
      const val = grid7x7[i];
      const gray = Math.round(val * 255);
      const idx = i * 4;
      tinyImgData.data[idx] = gray;
      tinyImgData.data[idx + 1] = gray;
      tinyImgData.data[idx + 2] = gray;
      tinyImgData.data[idx + 3] = 255;
    }
    tinyCtx.putImageData(tinyImgData, 0, 0);

    const upCanvas = document.createElement("canvas");
    upCanvas.width = width;
    upCanvas.height = height;
    const upCtx = upCanvas.getContext("2d");
    if (!upCtx) return;

    upCtx.imageSmoothingEnabled = true;
    upCtx.imageSmoothingQuality = "high";
    upCtx.drawImage(tinyCanvas, 0, 0, width, height);

    const upImgData = upCtx.getImageData(0, 0, width, height);
    const colorImgData = offCtx.createImageData(width, height);

    // B. Colorize with Jet colormap
    for (let i = 0; i < upImgData.data.length; i += 4) {
      const normVal = upImgData.data[i] / 255.0;
      const [r, g, b] = getJetColor(normVal);
      colorImgData.data[i] = r;
      colorImgData.data[i + 1] = g;
      colorImgData.data[i + 2] = b;
      colorImgData.data[i + 3] = 255;
    }
    offCtx.putImageData(colorImgData, 0, 0);

    // C. Render to visible heatmap canvas (Panel 2)
    const heatCanvas = heatmapCanvasRef.current;
    if (heatCanvas) {
      heatCanvas.width = width;
      heatCanvas.height = height;
      const heatCtx = heatCanvas.getContext("2d");
      if (heatCtx) {
        heatCtx.drawImage(offCanvas, 0, 0, width, height);
      }
    }

    // D. Update blended overlay (Panel 3)
    drawOverlay(currentOpacityRef.current);
  }, [grid7x7Json, drawOverlay]);

  // Pipeline 3: Synchronize overlay when opacity state changes
  useEffect(() => {
    drawOverlay(overlayOpacity);
  }, [overlayOpacity, drawOverlay]);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-7 shadow-xs space-y-5 text-slate-900">
      {/* Primary Section Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-sky-50 text-sky-700 border border-sky-200 font-mono">
            VISUAL EXPLANATION • GRAD-CAM
          </span>
          <h2 className="text-lg sm:text-xl font-extrabold text-slate-900 mt-1 tracking-tight">
            Visual Feature Attribution Map
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Regions where the model showed higher visual activation during prediction
          </p>
        </div>

        {/* Blend Opacity Slider Control */}
        <div className="flex items-center gap-2.5 text-xs bg-slate-50 px-3.5 py-2 rounded-xl border border-slate-200 shadow-2xs">
          <label
            htmlFor="overlay-opacity-slider"
            className="font-semibold text-slate-700 text-[11px] whitespace-nowrap"
          >
            Heatmap Blend: {Math.round(overlayOpacity * 100)}%
          </label>
          <input
            id="overlay-opacity-slider"
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={overlayOpacity}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              setOverlayOpacity(val);
              currentOpacityRef.current = val;
              drawOverlay(val);
            }}
            className="h-1.5 w-24 accent-sky-600 cursor-pointer"
            aria-label="Heatmap Blend Opacity"
          />
        </div>
      </div>

      {/* CORE THREE-VIEW GRAD-CAM PANELS */}
      {explanation.available ? (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-5">
            {/* Panel 1: Original Image */}
            <div className="flex flex-col items-center rounded-xl border border-slate-200 bg-slate-50/70 p-4 space-y-2.5 shadow-2xs">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Original Image
              </span>
              <div className="relative aspect-square w-full max-w-[224px] h-[224px] rounded-lg overflow-hidden border border-slate-200 bg-slate-100 flex items-center justify-center shadow-2xs">
                {imagePreviewUrl ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img
                    src={imagePreviewUrl}
                    alt="Original Uploaded Image"
                    className="max-h-full max-w-full object-contain"
                  />
                ) : (
                  <span className="text-xs text-slate-400">No Image Uploaded</span>
                )}
              </div>
              <p className="text-[11px] text-slate-500 text-center font-medium">
                Uploaded image file
              </p>
            </div>

            {/* Panel 2: Grad-CAM Heatmap */}
            <div className="flex flex-col items-center rounded-xl border border-slate-200 bg-slate-50/70 p-4 space-y-2.5 shadow-2xs">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Grad-CAM Heatmap
              </span>
              <div className="relative aspect-square w-full max-w-[224px] h-[224px] rounded-lg overflow-hidden border border-slate-200 bg-slate-100 flex items-center justify-center shadow-2xs">
                <canvas
                  ref={heatmapCanvasRef}
                  className="w-full h-full object-contain"
                  title="Grad-CAM Visual Activation Heatmap"
                />
              </div>
              <p className="text-[11px] text-slate-500 text-center font-medium">
                Visual feature activation map
              </p>
            </div>

            {/* Panel 3: Blended Overlay */}
            <div className="flex flex-col items-center rounded-xl border border-slate-200 bg-slate-50/70 p-4 space-y-2.5 shadow-2xs">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Blended Overlay
              </span>
              <div className="relative aspect-square w-full max-w-[224px] h-[224px] rounded-lg overflow-hidden border border-slate-200 bg-slate-100 flex items-center justify-center shadow-2xs">
                <canvas
                  ref={overlayCanvasRef}
                  className="w-full h-full object-contain"
                  title="Grad-CAM Transparent Overlay over Original Image"
                />
              </div>
              <p className="text-[11px] text-slate-500 text-center font-medium">
                Image with heatmap overlay
              </p>
            </div>
          </div>

          {/* Color Intensity Scale Legend */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl border border-slate-200 bg-slate-50 text-xs">
            <div className="flex items-center gap-2 text-slate-700 font-semibold">
              <svg className="w-4 h-4 text-sky-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
              </svg>
              <span>Model Activation Scale</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-[11px] font-mono font-medium text-slate-500">0.0 (Min Focus)</span>
              <div className="h-2.5 w-36 rounded-full bg-gradient-to-r from-blue-700 via-cyan-400 via-emerald-400 via-amber-400 to-rose-600 border border-slate-300 shadow-inner" />
              <span className="text-[11px] font-mono font-medium text-slate-500">1.0 (Peak Focus)</span>
            </div>
          </div>

          {/* Patient-Friendly Explanation */}
          <div className="rounded-xl border border-sky-100 bg-sky-50/70 p-4 space-y-1.5 text-xs text-slate-700 leading-relaxed">
            <h4 className="font-bold text-sky-900 text-xs uppercase tracking-wider flex items-center gap-1.5">
              <svg className="w-4 h-4 text-sky-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              How to read this visualization
            </h4>
            <p className="font-semibold text-slate-800">
              The highlighted areas show where the model had stronger visual activation while producing its prediction.
            </p>
            <p className="text-slate-600 text-[11.5px]">
              This visualization illustrates which image regions drew the model&apos;s computational focus. It does not prove disease presence, identify the exact anatomical cause, or serve as a clinical diagnosis.
            </p>
          </div>
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-xs text-slate-500 font-medium">
          Grad-CAM visual feature attribution computation is pending or unavailable for this artifact.
        </div>
      )}

      {/* SECONDARY TECHNICAL DETAILS */}
      <div className="border-t border-slate-100 pt-3">
        <button
          type="button"
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="text-xs font-semibold text-slate-500 hover:text-sky-700 flex items-center gap-1 cursor-pointer transition-colors"
        >
          <span>{showTechnicalDetails ? "Hide Technical Specifications" : "Show Technical Specifications"}</span>
          <svg
            className={`w-3.5 h-3.5 transition-transform ${showTechnicalDetails ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {showTechnicalDetails && (
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs font-mono bg-slate-50 p-3.5 rounded-xl border border-slate-200 text-slate-700">
            <div>
              <span className="text-[10px] text-slate-500 uppercase block font-sans font-bold">XAI Method</span>
              <span className="font-bold text-slate-900">Grad-CAM</span>
            </div>
            <div>
              <span className="text-[10px] text-slate-500 uppercase block font-sans font-bold">Target Layer</span>
              <span className="font-bold text-sky-800 truncate block">{details.target_layer || "N/A"}</span>
            </div>
            <div>
              <span className="text-[10px] text-slate-500 uppercase block font-sans font-bold">Feature Map</span>
              <span className="font-bold text-slate-900">7 × 7 grid</span>
            </div>
            <div>
              <span className="text-[10px] text-slate-500 uppercase block font-sans font-bold">Display Resolution</span>
              <span className="font-bold text-slate-900">224 × 224 px</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
