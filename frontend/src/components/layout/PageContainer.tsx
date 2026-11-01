import React from "react";
import { Header } from "./Header";
import { Footer } from "./Footer";

interface PageContainerProps {
  children: React.ReactNode;
  activeModality?: string;
  onSelectModality?: (modalityId: string) => void;
}

export const PageContainer: React.FC<PageContainerProps> = ({
  children,
  activeModality,
  onSelectModality,
}) => {
  return (
    <div className="min-h-screen flex flex-col text-slate-900 font-sans antialiased relative selection:bg-sky-500 selection:text-white">
      {/* Ambient background glow accents */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden -z-10">
        <div className="absolute -top-32 -left-32 w-96 h-96 rounded-full bg-cyan-200/40 blur-3xl" />
        <div className="absolute top-1/4 -right-32 w-96 h-96 rounded-full bg-teal-200/30 blur-3xl" />
        <div className="absolute top-2/3 -left-32 w-96 h-96 rounded-full bg-purple-200/35 blur-3xl" />
        <div className="absolute -bottom-32 right-1/4 w-96 h-96 rounded-full bg-amber-100/40 blur-3xl" />
      </div>

      <Header
        activeModality={activeModality}
        onSelectModality={onSelectModality}
      />
      
      <main className="flex-1 mx-auto w-full max-w-[1380px] px-4 py-8 sm:px-6 lg:px-8 relative z-10">
        {children}
      </main>
      
      <Footer />
    </div>
  );
};
