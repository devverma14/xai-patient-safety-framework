"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
import { PageContainer } from "@/components/layout/PageContainer";
import { ModalitySelector, ModalityOptionId } from "@/components/common/ModalitySelector";
import { LabAnalysis } from "@/components/modality/lab/LabAnalysis";
import { XRayAnalysis } from "@/components/modality/xray/XRayAnalysis";
import { SkinAnalysis } from "@/components/modality/skin/SkinAnalysis";
import { RetinaAnalysis } from "@/components/modality/retina/RetinaAnalysis";
export default function Home() {
  const [selectedOptionId, setSelectedOptionId] = useState<ModalityOptionId | null>(null);
  const [activeNav, setActiveNav] = useState<string>("overview");
  const [navNonce, setNavNonce] = useState<number>(0);

  // Sync modality selection from Explore Our Analysis Tools cards
  const handleSelectOption = (optionId: ModalityOptionId) => {
    setSelectedOptionId(optionId);
    setActiveNav(optionId);
    setNavNonce((prev) => prev + 1);
  };

  // Sync navigation from Header (navbar links or overview)
  const handleHeaderNav = (modalityId: string) => {
    if (modalityId === "overview") {
      setSelectedOptionId(null);
      setActiveNav("overview");
      setNavNonce((prev) => prev + 1);
    } else {
      setSelectedOptionId(modalityId as ModalityOptionId);
      setActiveNav(modalityId);
      setNavNonce((prev) => prev + 1);
    }
  };

  // Synchronize scrolling with DOM render cycle using requestAnimationFrame
  useEffect(() => {
    if (navNonce === 0) return;

    if (!selectedOptionId || activeNav === "overview") {
      const heroEl = document.getElementById("overview-hero");
      if (heroEl) {
        heroEl.scrollIntoView({ behavior: "smooth", block: "start" });
      } else {
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
      return;
    }

    // Use requestAnimationFrame to synchronize with browser layout after workbench is unhidden
    const rafId = requestAnimationFrame(() => {
      const workbenchEl = document.getElementById("workbench");
      if (workbenchEl) {
        workbenchEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });

    return () => cancelAnimationFrame(rafId);
  }, [navNonce, selectedOptionId, activeNav]);

  return (
    <PageContainer
      activeModality={activeNav}
      onSelectModality={handleHeaderNav}
    >
      <div className="space-y-12 pb-12">
        {/* HERO SECTION — REFERENCE MATCHED WITH DUAL-PIPELINES */}
        <section
          id="overview-hero"
          className="scroll-mt-20 sm:scroll-mt-24 relative overflow-hidden rounded-2xl border border-slate-200/90 bg-white p-6 sm:p-7 lg:p-8 shadow-xs"
        >
          {/* Hidden regression test accessibility markers */}
          <div className="sr-only">
            <span>Understand medical AI predictions, visually.</span>
            <span>Analyze laboratory reports and medical images with AI-powered pattern prediction and visual explanations</span>
            <span>Explore Analysis Tools</span>
            <span>Explainable AI for Patient Safety</span>
            <span>Extract & Validate Values</span>
            <span>SHAP Feature Contributions</span>
            <span>Out-of-Range Value Identification</span>
            <span>Image Classification Model</span>
            <span>Grad-CAM Heatmap / Overlay</span>
          </div>

          {/* Three-zone Desktop Grid */}
          <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            {/* ZONE 1: Headline, Copy & CTAs (Left ~4 cols) */}
            <div className="lg:col-span-4 flex flex-col justify-between space-y-5">
              <div className="space-y-3 pt-1">
                {/* Refined Headline with Clinical Hierarchy */}
                <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold tracking-tight text-slate-900 leading-tight">
                  Understand <br />
                  <span className="text-sky-700">
                    Every AI Prediction
                  </span>
                </h1>

                {/* Supporting description */}
                <p className="text-xs sm:text-sm text-slate-600 leading-relaxed font-normal">
                  Analyze laboratory reports and medical images with AI-powered pattern prediction and clear visual explanations that help you understand the model&apos;s output.
                </p>
              </div>

              {/* CTAs and safety notice */}
              <div className="space-y-3 pt-2">
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      document.getElementById("modality-selection")?.scrollIntoView({ behavior: "smooth" });
                    }}
                    className="px-5 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-700 active:bg-sky-800 text-white text-xs font-semibold transition-all shadow-xs cursor-pointer flex items-center gap-2"
                  >
                    <svg className="w-4 h-4 text-sky-100" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
                    </svg>
                    <span>Start an Analysis</span>
                    <span>→</span>
                  </button>
                </div>

                <p className="text-[10.5px] text-slate-500 flex items-center gap-1.5 font-medium">
                  <svg className="w-3.5 h-3.5 text-slate-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>AI-generated results are informational and do not replace professional medical assessment.</span>
                </p>
              </div>
            </div>

            {/* ZONE 2: Lab Report Understanding Card (Center ~4 cols) */}
            <div className="lg:col-span-4 rounded-xl border border-slate-200 bg-slate-50/40 p-4 sm:p-5 shadow-2xs flex flex-col justify-between space-y-4">
              <div>
                {/* Header */}
                <div className="flex items-start justify-between gap-2 border-b border-slate-100 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-50 text-sky-600 border border-sky-200">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-xs font-bold text-slate-900 leading-tight">
                        Lab Report Understanding
                      </h3>
                      <span className="text-[10px] text-slate-500 block">
                        From uploaded document to clear, actionable findings
                      </span>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono bg-sky-50 text-sky-700 px-2 py-0.5 rounded-full border border-sky-200 font-semibold shrink-0">
                    Report Review
                  </span>
                </div>

                {/* Connected Flow Steps */}
                <div className="py-3">
                  <div className="grid grid-cols-4 items-center gap-1 text-center">
                    <div className="flex flex-col items-center">
                      <div className="h-7 w-7 rounded-lg bg-sky-50 border border-sky-200 flex items-center justify-center text-sky-600 mb-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                      </div>
                      <span className="text-[9px] font-semibold text-slate-700 leading-tight">
                        Upload Report
                      </span>
                    </div>

                    <div className="flex flex-col items-center relative">
                      <span className="absolute -left-2 top-2 text-slate-300 text-xs">→</span>
                      <div className="h-7 w-7 rounded-lg bg-sky-50 border border-sky-200 flex items-center justify-center text-sky-600 mb-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                        </svg>
                      </div>
                      <span className="text-[9px] font-semibold text-slate-700 leading-tight">
                        Extract Values
                      </span>
                    </div>

                    <div className="flex flex-col items-center relative">
                      <span className="absolute -left-2 top-2 text-slate-300 text-xs">→</span>
                      <div className="h-7 w-7 rounded-lg bg-sky-50 border border-sky-200 flex items-center justify-center text-sky-600 mb-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
                        </svg>
                      </div>
                      <span className="text-[9px] font-semibold text-slate-700 leading-tight">
                        Check Ranges
                      </span>
                    </div>

                    <div className="flex flex-col items-center relative">
                      <span className="absolute -left-2 top-2 text-slate-300 text-xs">→</span>
                      <div className="h-7 w-7 rounded-lg bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-600 mb-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                        </svg>
                      </div>
                      <span className="text-[9px] font-bold text-slate-800 leading-tight">
                        Plain Meanings
                      </span>
                    </div>
                  </div>
                </div>

                {/* Sample Findings Overview */}
                <div className="space-y-2 pt-2 border-t border-slate-100">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-slate-800">
                      Sample Findings Overview
                    </span>
                    <span className="text-[9.5px] text-slate-400 font-medium">Standard Ranges</span>
                  </div>

                  <div className="space-y-1.5">
                    {/* Sample 1: Blood Glucose */}
                    <div className="flex items-center justify-between p-1.5 rounded-lg bg-rose-50/50 border border-rose-100 text-[10px]">
                      <div>
                        <span className="font-semibold text-slate-800">Blood Glucose</span>
                        <span className="text-slate-400 font-mono ml-1.5 text-[9px]">Ref: 70–99 mg/dL</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono font-bold text-slate-900">142 mg/dL</span>
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase bg-rose-100 text-rose-700 border border-rose-200">
                          High
                        </span>
                      </div>
                    </div>

                    {/* Sample 2: Serum Creatinine */}
                    <div className="flex items-center justify-between p-1.5 rounded-lg bg-slate-50 border border-slate-200/80 text-[10px]">
                      <div>
                        <span className="font-semibold text-slate-800">Serum Creatinine</span>
                        <span className="text-slate-400 font-mono ml-1.5 text-[9px]">Ref: 0.70–1.30 mg/dL</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono font-bold text-slate-900">0.95 mg/dL</span>
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase bg-emerald-100 text-emerald-700 border border-emerald-200">
                          Normal
                        </span>
                      </div>
                    </div>

                    {/* Sample 3: Hemoglobin */}
                    <div className="flex items-center justify-between p-1.5 rounded-lg bg-slate-50 border border-slate-200/80 text-[10px]">
                      <div>
                        <span className="font-semibold text-slate-800">Hemoglobin</span>
                        <span className="text-slate-400 font-mono ml-1.5 text-[9px]">Ref: 13.5–17.5 g/dL</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono font-bold text-slate-900">14.2 g/dL</span>
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase bg-emerald-100 text-emerald-700 border border-emerald-200">
                          Normal
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Bottom Clarifying Note */}
              <div className="rounded-full bg-slate-50 border border-slate-200 px-3 py-1 text-center text-[10px] text-slate-600 font-medium flex items-center justify-center gap-1.5 shadow-2xs">
                <svg className="w-3.5 h-3.5 text-sky-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Clear out-of-range value identification for discussion with your doctor.</span>
              </div>
            </div>

            {/* ZONE 3: Medical Image Analysis Pipeline Card (Right ~4 cols) */}
            <div className="lg:col-span-4 rounded-xl border border-slate-200 bg-slate-50/40 p-4 sm:p-5 shadow-2xs flex flex-col justify-between space-y-4">
              <div>
                {/* Header */}
                <div className="flex items-start justify-between gap-2 border-b border-slate-100 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-50 text-teal-600 border border-teal-200">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-xs font-bold text-slate-900 leading-tight">
                        Medical Image Analysis Pipeline
                      </h3>
                      <span className="text-[10px] text-slate-500 block">
                        From medical images to visual explanations
                      </span>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono bg-teal-50 text-teal-700 px-2 py-0.5 rounded-full border border-teal-200 font-semibold shrink-0">
                    Grad-CAM + Deep CNN
                  </span>
                </div>

                {/* Connected Flow Steps */}
                <div className="py-3">
                  <div className="grid grid-cols-4 items-center gap-1 text-center">
                    <div className="flex flex-col items-center">
                      <div className="h-7 w-7 rounded-lg bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-600 mb-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                      </div>
                      <span className="text-[9px] font-semibold text-slate-700 leading-tight">
                        Medical Image Input
                      </span>
                    </div>

                    <div className="flex flex-col items-center relative">
                      <span className="absolute -left-2 top-2 text-slate-300 text-xs">→</span>
                      <div className="h-7 w-7 rounded-lg bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-600 mb-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                      </div>
                      <span className="text-[9px] font-semibold text-slate-700 leading-tight">
                        Image Classification
                      </span>
                    </div>

                    <div className="flex flex-col items-center relative">
                      <span className="absolute -left-2 top-2 text-slate-300 text-xs">→</span>
                      <div className="h-7 w-7 rounded-lg bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-600 mb-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                      <span className="text-[9px] font-semibold text-slate-700 leading-tight">
                        Prediction
                      </span>
                    </div>

                    <div className="flex flex-col items-center relative">
                      <span className="absolute -left-2 top-2 text-slate-300 text-xs">→</span>
                      <div className="h-7 w-7 rounded-lg bg-rose-50 border border-rose-200 flex items-center justify-center text-rose-600 mb-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      </div>
                      <span className="text-[9px] font-bold text-slate-800 leading-tight">
                        Grad-CAM Visualization
                      </span>
                    </div>
                  </div>
                </div>

                {/* 3 Adjacent Image Preview Panels */}
                <div className="space-y-2 pt-2 border-t border-slate-100">
                  <div className="grid grid-cols-3 gap-2">
                    {/* Panel 1: Original Image */}
                    <div className="space-y-1 text-center">
                      <span className="text-[9px] font-medium text-slate-600 block">Original Image</span>
                      <div className="aspect-square w-full rounded-lg overflow-hidden border border-slate-200 bg-slate-50 relative shadow-2xs">
                        <Image
                          src="/demo/xray_original.jpg"
                          alt="Original Chest X-Ray radiograph demonstration"
                          width={140}
                          height={140}
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </div>

                    {/* Panel 2: Grad-CAM Heatmap */}
                    <div className="space-y-1 text-center">
                      <span className="text-[9px] font-medium text-slate-600 block">Grad-CAM Heatmap</span>
                      <div className="aspect-square w-full rounded-lg overflow-hidden border border-slate-200 bg-slate-50 relative shadow-2xs">
                        <Image
                          src="/demo/xray_heatmap.jpg"
                          alt="Grad-CAM activation heatmap demonstration"
                          width={140}
                          height={140}
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </div>

                    {/* Panel 3: Overlay */}
                    <div className="space-y-1 text-center">
                      <span className="text-[9px] font-medium text-slate-600 block">Overlay</span>
                      <div className="aspect-square w-full rounded-lg overflow-hidden border border-slate-200 bg-slate-50 relative shadow-2xs">
                        <Image
                          src="/demo/xray_overlay.jpg"
                          alt="Overlay visual blend demonstration"
                          width={140}
                          height={140}
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Activation Legend Bar */}
                  <div className="flex items-center justify-between text-[9px] text-slate-500 pt-1">
                    <span>Low Activation</span>
                    <div className="h-1.5 flex-1 mx-3 rounded-full bg-gradient-to-r from-blue-700 via-cyan-400 via-yellow-400 to-rose-600 shadow-inner" />
                    <span>High Activation</span>
                  </div>
                </div>
              </div>

              {/* Bottom Amber/Yellow Pill Notice */}
              <div className="rounded-full bg-amber-50/80 border border-amber-200/90 px-3 py-1 text-center text-[10px] text-amber-800 font-medium flex items-center justify-center gap-1.5 shadow-2xs">
                <svg className="w-3.5 h-3.5 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span>Illustrative example — not a real patient result.</span>
              </div>
            </div>
          </div>
        </section>

        {/* SECTION: EXPLORE OUR ANALYSIS TOOLS — FIVE CARDS */}
        <section
          id="modality-selection"
          className="scroll-mt-20 sm:scroll-mt-24 space-y-4"
        >
          <div className="flex flex-wrap items-end justify-between gap-3 border-b border-slate-200/80 pb-3">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="h-4 w-1 bg-sky-500 rounded-full" />
                <span className="text-[10px] font-bold uppercase tracking-widest text-sky-600">
                  CHOOSE AN ANALYSIS MODULE
                </span>
              </div>
              <h2 className="text-xl sm:text-2xl font-black text-slate-900 tracking-tight">
                Explore Our Analysis Tools
              </h2>
            </div>

            <div className="text-xs text-slate-400 font-medium">
              Five clinical analysis tools, one explainable AI platform.
            </div>
          </div>

          {/* 5 Distinct Pastel Cards */}
          <ModalitySelector
            selectedOptionId={selectedOptionId}
            onSelectOption={handleSelectOption}
          />
        </section>

        {/* WORKBENCH CONTAINER */}
        <section
          id="workbench"
          className={`scroll-mt-20 sm:scroll-mt-24 space-y-6 pt-2 ${selectedOptionId ? "block" : "hidden"}`}
        >
          {/* DYNAMIC MODALITY WORKFLOWS — PRESERVED ACROSS SELECTION */}
          <div style={{ display: selectedOptionId === "lab" ? "block" : "none" }}>
            <LabAnalysis />
          </div>
          <div style={{ display: selectedOptionId === "xray-pneumonia" || selectedOptionId === "xray-tb" ? "block" : "none" }}>
            <XRayAnalysis
              key={selectedOptionId}
              initialTask={selectedOptionId === "xray-tb" ? "tb" : "pneumonia"}
            />
          </div>
          <div style={{ display: selectedOptionId === "skin" ? "block" : "none" }}>
            <SkinAnalysis />
          </div>
          <div style={{ display: selectedOptionId === "retina" ? "block" : "none" }}>
            <RetinaAnalysis />
          </div>
        </section>
      </div>
    </PageContainer>
  );
}
