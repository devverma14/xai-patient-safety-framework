/**
 * Utility helper functions for formatting probabilities, clinical labels, and styling.
 */

export function formatProbability(value?: number): string {
  if (value === undefined || value === null) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatDecimal(value?: number, decimals: number = 4): string {
  if (value === undefined || value === null) return "N/A";
  return value.toFixed(decimals);
}

export function getStatusBadgeColor(status: string): string {
  switch (status) {
    case "success":
      return "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
    case "review_required":
    case "insufficient_data":
      return "bg-amber-500/10 text-amber-400 border-amber-500/20";
    case "model_not_available":
      return "bg-blue-500/10 text-blue-400 border-blue-500/20";
    case "error":
      return "bg-rose-500/10 text-rose-400 border-rose-500/20";
    default:
      return "bg-slate-500/10 text-slate-400 border-slate-500/20";
  }
}
