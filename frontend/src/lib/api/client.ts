import {
  CommonResponsePayload,
  ModalityStatusResponse,
  ModalityType,
  APIErrorResponse,
} from "../types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

class APIError extends Error {
  code: string;
  details?: Record<string, unknown>;

  constructor(message: string, code: string = "API_ERROR", details?: Record<string, unknown>) {
    super(message);
    this.name = "APIError";
    this.code = code;
    this.details = details;
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        ...options?.headers,
      },
    });

    const data = await res.json();

    if (!res.ok || data.status === "error") {
      const errData = data as APIErrorResponse;
      throw new APIError(
        errData.error?.message || `HTTP ${res.status}: ${res.statusText}`,
        errData.error?.code || "UNKNOWN_ERROR",
        errData.error?.details
      );
    }

    return data as T;
  } catch (err: unknown) {
    if (err instanceof APIError) {
      throw err;
    }
    const message = err instanceof Error ? err.message : "Failed to communicate with backend service.";
    throw new APIError(message, "NETWORK_ERROR");
  }
}

export const apiClient = {
  // Status Queries
  async getModalityStatus(modality: ModalityType): Promise<ModalityStatusResponse> {
    return request<ModalityStatusResponse>(`/api/v1/${modality}/status`);
  },

  // Modality Analyses
  async analyzeLab(
    file: File,
    explicitStructuredInput?: Record<string, unknown>
  ): Promise<CommonResponsePayload> {
    const formData = new FormData();
    formData.append("file", file);
    if (explicitStructuredInput) {
      formData.append("structured_input", JSON.stringify(explicitStructuredInput));
    }
    return request<CommonResponsePayload>("/api/v1/lab/analyze", {
      method: "POST",
      body: formData,
    });
  },

  async analyzeXRay(file: File, task: "pneumonia" | "tb"): Promise<CommonResponsePayload> {
    const formData = new FormData();
    formData.append("file", file);
    return request<CommonResponsePayload>(`/api/v1/xray/analyze?task=${task}`, {
      method: "POST",
      body: formData,
    });
  },

  async analyzeSkin(file: File, task: string = "ham10000"): Promise<CommonResponsePayload> {
    const formData = new FormData();
    formData.append("file", file);
    return request<CommonResponsePayload>(`/api/v1/skin/analyze?task=${task}`, {
      method: "POST",
      body: formData,
    });
  },

  async analyzeRetina(
    file: File,
    task: string = "diabetic_retinopathy"
  ): Promise<CommonResponsePayload> {
    const formData = new FormData();
    formData.append("file", file);
    return request<CommonResponsePayload>(`/api/v1/retina/analyze?task=${task}`, {
      method: "POST",
      body: formData,
    });
  },
};
