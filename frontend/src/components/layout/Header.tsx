import React, { useState } from "react";
import Link from "next/link";
import { Logo } from "@/components/common/Logo";

interface HeaderProps {
  activeModality?: string;
  onSelectModality?: (modalityId: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeModality,
  onSelectModality,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    {
      id: "lab",
      label: "Lab Reports",
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      ),
    },
    {
      id: "xray-pneumonia",
      label: "Pneumonia",
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      ),
    },
    {
      id: "xray-tb",
      label: "TB",
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
        </svg>
      ),
    },
    {
      id: "skin",
      label: "Skin Lesion",
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
      ),
    },
    {
      id: "retina",
      label: "Retina",
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        </svg>
      ),
    },
  ];

  const handleNavClick = (id: string) => {
    if (onSelectModality) {
      onSelectModality(id);
    }
    setMobileMenuOpen(false);
  };

  const isOverviewActive = activeModality === "overview" || !activeModality;

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/95 backdrop-blur-sm shadow-[0_1px_2px_0_rgba(15,23,42,0.03)]">
      <div className="mx-auto flex max-w-[1380px] items-center justify-between px-4 py-2.5 sm:px-6 lg:px-8">
        {/* Brand & Logo */}
        <Link
          href="/"
          onClick={(e) => {
            e.preventDefault();
            if (onSelectModality) onSelectModality("overview");
          }}
          className="flex items-center gap-3 text-slate-900 transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-sky-600 rounded-lg p-1"
          aria-label="Patient Safety XAI Home"
        >
          <Logo size={36} showText={true} textClassName="text-base sm:text-lg" />
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-1 sm:gap-1.5 text-xs" aria-label="Main Navigation">
          <button
            type="button"
            onClick={() => {
              if (onSelectModality) onSelectModality("overview");
            }}
            className={`px-2.5 py-1.5 rounded-lg transition-all duration-150 cursor-pointer flex items-center gap-1.5 ${
              isOverviewActive
                ? "bg-sky-50 text-sky-900 border border-sky-300 font-bold shadow-2xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/50 font-medium"
            }`}
          >
            <svg className="w-3.5 h-3.5 text-sky-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
            </svg>
            <span>Overview</span>
          </button>

          {navLinks.map((item) => {
            const isActive = activeModality === item.id;

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleNavClick(item.id)}
                className={`px-2.5 py-1.5 rounded-lg transition-all duration-150 cursor-pointer flex items-center gap-1.5 ${
                  isActive
                    ? "bg-sky-50 text-sky-900 border border-sky-300 font-bold shadow-2xs"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/50 font-medium"
                }`}
              >
                <span className={isActive ? "text-sky-600" : "text-slate-400"}>
                  {item.icon}
                </span>
                <span>{item.label}</span>
              </button>
            );
          })}

          {/* User Profile Badge */}
          <div className="flex items-center pl-2 ml-1 border-l border-slate-200/80">
            <div
              className="h-7 w-7 rounded-full bg-linear-to-tr from-[#0c2a5e] to-sky-600 flex items-center justify-center text-white text-[11px] font-bold shadow-2xs cursor-default"
              title="Clinical AI Workspace"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
          </div>
        </nav>

        {/* Mobile Hamburger Button */}
        <div className="flex md:hidden items-center gap-2">
          <button
            type="button"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-lg text-slate-600 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-sky-600 cursor-pointer"
            aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-slate-200/80 bg-slate-50/95 backdrop-blur-md px-4 py-3 space-y-1 shadow-sm">
          <button
            type="button"
            onClick={() => {
              if (onSelectModality) onSelectModality("overview");
              setMobileMenuOpen(false);
            }}
            className="w-full text-left px-3 py-2 rounded-lg text-xs font-semibold text-slate-700 hover:bg-slate-100 flex items-center gap-2"
          >
            <svg className="w-4 h-4 text-sky-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
            </svg>
            <span>Overview</span>
          </button>
          {navLinks.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => handleNavClick(item.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-xs font-semibold transition-colors flex items-center gap-2 ${
                activeModality === item.id
                  ? "bg-sky-50 text-sky-800 font-bold"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              <span className="text-slate-400">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      )}
    </header>
  );
};
