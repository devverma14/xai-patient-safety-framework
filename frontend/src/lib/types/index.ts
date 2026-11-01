/**
 * TypeScript Type Definitions for XAI Framework Patient Safety Frontend.
 * Matches the exact backend FastAPI Pydantic schema contracts in backend/app/api/schemas/.
 */

export type ModalityType = "lab" | "xray" | "skin" | "retina";

export type LabTask = "report_analysis" | "30_day_readmission";
export type XRayTask = "pneumonia" | "tb";
export type SkinTask = "ham10000";
export type RetinaTask = "diabetic_retinopathy";

export type TaskType = LabTask | XRayTask | SkinTask | RetinaTask;

export type ResponseStatus =
  | "success"
  | "review_required"
  | "insufficient_data"
  | "model_not_available"
  | "error";

export interface PredictionDetails {
  label?: string;
  probability?: number;
  threshold?: number;
  model_name?: string;
  model_architecture?: string;
  raw_output?: Record<string, unknown>;
}

export interface ExplanationDetails {
  method: "Clinical Reference Intervals" | "Grad-CAM" | "SHAP";
  available: boolean;
  summary?: string;
  details: Record<string, unknown>;
}

export interface QualityDetails {
  input_quality: "passed" | "failed" | "warning" | string;
  ocr_confidence?: number;
  extraction_confidence?: number;
  explanation_quality: Record<string, unknown>;
}

export interface ParameterExplanation {
  display_name: string;
  category: string;
  description: string;
  status_summary?: string;
  specific_status?: string;
  what_it_means?: string;
  possible_causes_text?: string;
  possible_causes?: string[];
  next_steps?: string;
  interpretation_guidance?: string;
  general_guidance?: string;
  clinical_disclaimer?: string;
  is_supported?: boolean;
}

export interface ExtractedLabValueItem {
  test_name: string;
  normalized_name: string;
  raw_value_str: string;
  numeric_value?: number | null;
  unit?: string | null;
  reference_range?: string | null;
  status_flag?: "NORMAL" | "HIGH" | "LOW" | "CRITICAL" | string | null;
  extraction_confidence: number;
  validation_status: "VALID" | "REVIEW_REQUIRED" | "OUT_OF_BOUNDS" | string;
  validation_message?: string | null;
  source_context?: string | null;
  source_line?: string | null;
  reference_range_source?: "REPORT" | "STANDARD_REFERENCE" | string | null;
  interpretation_status?: "WITHIN_RANGE" | "HIGH" | "LOW" | "UNABLE_TO_DETERMINE" | string;
  specific_status?: string | null;
  explanation?: ParameterExplanation | null;
  is_supported?: boolean;
}

export interface ClinicalSummaryOverview {
  total_parameters: number;
  total_detected_count?: number;
  in_range_count: number;
  abnormal_count: number;
  high_count: number;
  low_count: number;
  uncertain_count: number;
  narrative_summary: string;
}

export interface ClinicalPanelItem {
  test_name: string;
  normalized_name: string;
  value: string | number;
  unit?: string | null;
  reference_range?: string | null;
  status: string;
}

export interface ClinicalPanelSection {
  panel_id: string;
  title: string;
  description: string;
  total_tests: number;
  abnormal_count: number;
  status: string;
  items: ClinicalPanelItem[];
}

export interface ClinicalFindingItem {
  test_name: string;
  normalized_name: string;
  value: string | number;
  raw_value_str?: string | null;
  numeric_value?: number | null;
  unit?: string | null;
  reference_range?: string | null;
  reference_range_source?: string | null;
  status: string;
  specific_status?: string | null;
  validation_status: string;
  validation_message?: string | null;
  possible_causes: string[];
  possible_causes_text?: string | null;
  what_it_means?: string | null;
  next_steps?: string | null;
  guidance?: string;
  category: string;
}

export interface CorrelatedClinicalPattern {
  category: string;
  title: string;
  severity: "ATTENTION" | "INFORMATIONAL" | "CRITICAL" | string;
  analytes: string[];
  summary: string;
  clinical_considerations: string;
}

export interface ClinicalGuidance {
  uncertainty_statement: string;
  actionable_recommendations: string[];
  disclaimer: string;
}

export interface ClinicalSummary {
  overview: ClinicalSummaryOverview;
  patient_demographics?: Record<string, ClinicalFindingItem>;
  panels: ClinicalPanelSection[];
  abnormal_findings: {
    high: ClinicalFindingItem[];
    low: ClinicalFindingItem[];
  };
  in_range_findings: ClinicalFindingItem[];
  uncertain_findings: ClinicalFindingItem[];
  correlated_patterns: CorrelatedClinicalPattern[];
  clinical_guidance: ClinicalGuidance;
}

export interface LabPayloadDetails {
  document_info?: Record<string, unknown>;
  extraction_method?: string;
  confidence_metrics?: {
    ocr_confidence: number;
    extraction_confidence: number;
  };
  prediction_status?: "READY" | "INSUFFICIENT_INPUT_DATA" | "REVIEW_REQUIRED" | string;
  human_review_required?: boolean;
  review_reasons?: string[];
  feature_availability?: {
    missing_required_features: string[];
    readiness_summary?: Record<string, string>;
  };
  laboratory_values?: ExtractedLabValueItem[];
  clinical_summary?: ClinicalSummary | null;
  disclaimer?: string;
}

export interface SHAPFactorItem {
  feature_index?: number;
  feature_name: string;
  human_readable_name: string;
  feature_value_scaled: number | string;
  shap_value: number;
  contribution_direction: "positive_risk_increase" | "negative_risk_decrease" | "neutral" | string;
  magnitude: number;
  description?: string;
}

export interface SHAPDetails {
  base_value?: number;
  prediction_summary?: Record<string, unknown>;
  top_risk_increasing_factors?: SHAPFactorItem[];
  top_risk_decreasing_factors?: SHAPFactorItem[];
  all_ranked_contributions?: SHAPFactorItem[];
  disclaimer?: string;
}

export interface GradCAMDetails {
  target_layer?: string;
  target_class_index?: number;
  predicted_label?: string;
  evaluated_probability?: number;
  heatmap_shape?: number[];
  heatmap_min?: number;
  heatmap_max?: number;
  heatmap_mean?: number;
  spatial_summary_grid_7x7?: number[];
  error?: string;
  note?: string;
}

export interface CommonResponsePayload {
  status: ResponseStatus;
  modality: ModalityType;
  task: string;
  prediction?: PredictionDetails | null;
  explanation?: ExplanationDetails | null;
  quality: QualityDetails;
  review_required: boolean;
  review_reasons: string[];
  warnings: string[];
  payload_details?: LabPayloadDetails | Record<string, unknown> | null;
}

export interface APIErrorDetail {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface APIErrorResponse {
  status: "error";
  error: APIErrorDetail;
}

export interface ModalityStatusResponse {
  modality: ModalityType;
  supported_tasks: string[];
  model_status: "AVAILABLE" | "MODEL_NOT_AVAILABLE";
  xai_method: "SHAP" | "Grad-CAM";
  note: string;
  decision_threshold?: number;
  model_features_count?: number;
  model_architecture?: string;
  task_architectures?: Record<string, string>;
}
