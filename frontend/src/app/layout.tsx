import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Patient Safety XAI — Explainable AI for Patient Safety",
  description:
    "Explainable AI for Patient Safety. Analyze laboratory reports and medical images with AI-powered pattern prediction and visual explanations that help users understand the model's output.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="bg-slate-50/70 text-slate-900 font-sans antialiased min-h-screen selection:bg-sky-100 selection:text-sky-900">
        {children}
      </body>
    </html>
  );
}
