import React from "react";

interface LogoProps {
  size?: number;
  className?: string;
  showText?: boolean;
  textClassName?: string;
}

export const Logo: React.FC<LogoProps> = ({
  size = 34,
  className = "",
  showText = false,
  textClassName = "",
}) => {
  return (
    <div className={`inline-flex items-center gap-2.5 ${className}`}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="shrink-0"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="shieldNavyGrad" x1="12" y1="5" x2="88" y2="95" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#0c2a5e" />
            <stop offset="100%" stopColor="#081c3c" />
          </linearGradient>
          <linearGradient id="patientBlueGrad" x1="24" y1="26" x2="48" y2="54" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#0284c7" />
            <stop offset="100%" stopColor="#0369a1" />
          </linearGradient>
          <linearGradient id="patientTealGrad" x1="26" y1="36" x2="68" y2="88" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#00bfa5" />
            <stop offset="100%" stopColor="#0d9488" />
          </linearGradient>
          <linearGradient id="patientHeadGrad" x1="40" y1="23" x2="54" y2="37" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#0099e6" />
            <stop offset="100%" stopColor="#0077c8" />
          </linearGradient>
        </defs>

        {/* Outer Protective Shield */}
        <path
          d="M50 5C67 8 83 11 88 20C90 42 85 68 50 95C15 68 10 42 12 20C17 11 33 8 50 5Z"
          fill="url(#shieldNavyGrad)"
        />

        {/* Inner White Field */}
        <path
          d="M50 11C64 14 77 16 81 24C83 40 80 62 50 88C20 62 17 40 19 24C23 16 36 14 50 11Z"
          fill="#ffffff"
        />

        {/* Right Neural Network Shield Wing Accent */}
        <path
          d="M56 12C67 15 78 18 81 24C83 40 80 62 50 88C55 83 62 74 67 64C74 50 75 36 73 24C70 18 64 14 56 12Z"
          fill="#0c2a5e"
        />

        {/* Patient Figure — Head */}
        <circle cx="46" cy="30" r="7.5" fill="url(#patientHeadGrad)" />

        {/* Patient Figure — Left Upper Arm (Blue Swoosh) */}
        <path
          d="M24 26C22 34 26 43 32 47C37 51 43 51 45 54C44 48 40 40 35 34C30 29 26 27 24 26Z"
          fill="url(#patientBlueGrad)"
        />

        {/* Patient Figure — Right Upper Arm (Teal/Cyan Swoosh) */}
        <path
          d="M67 27C65 35 59 42 53 46C49 49 46 51 45 54C47 48 52 42 57 36C62 30 65 28 67 27Z"
          fill="url(#patientTealGrad)"
        />

        {/* Patient Figure — Swirling Lower Body (Teal to Shield Tip) */}
        <path
          d="M26 42C24 50 27 60 34 68C40 76 46 84 48 88C49 84 48 72 43 62C37 52 30 45 26 42Z"
          fill="url(#patientTealGrad)"
        />

        {/* Patient Figure — Center Body Spine Curve */}
        <path
          d="M46 54C46 64 47 76 48 88C49 84 51 74 51 66C52 59 49 53 46 54Z"
          fill="url(#patientBlueGrad)"
        />

        {/* Neural Network Connector Lines */}
        <g stroke="#ffffff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="75" y1="32" x2="68" y2="48" />
          <line x1="75" y1="32" x2="80" y2="45" />
          <line x1="68" y1="48" x2="80" y2="45" />
          <line x1="68" y1="48" x2="62" y2="63" />
          <line x1="62" y1="63" x2="72" y2="69" />
          <path d="M80 45C81 55 77 64 72 69" fill="none" strokeWidth="1.8" />
        </g>

        {/* Neural Network Nodes */}
        <circle cx="75" cy="32" r="4.5" fill="#00bfa5" stroke="#ffffff" strokeWidth="1.6" />
        <circle cx="68" cy="48" r="4.5" fill="#0080d7" stroke="#ffffff" strokeWidth="1.6" />
        <circle cx="80" cy="45" r="3.2" fill="#ffffff" stroke="#0c2a5e" strokeWidth="1.2" />
        <circle cx="62" cy="63" r="4.5" fill="#00bfa5" stroke="#ffffff" strokeWidth="1.6" />
      </svg>

      {showText && (
        <div className="flex flex-col text-left">
          <div className="flex items-center gap-1.5 leading-none">
            <span className={`text-base font-black tracking-tight text-[#0a254a] ${textClassName}`}>
              Patient Safety
            </span>
            <span className={`text-base font-black tracking-tight text-[#0080d7] ${textClassName}`}>
              XAI
            </span>
          </div>
          <span className="text-[10.5px] font-semibold text-slate-500 leading-tight mt-0.5">
            Explainable AI for Patient Safety
          </span>
        </div>
      )}
    </div>
  );
};
