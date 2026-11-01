"""
Prediction Service orchestrating modality routing, input validation, OCR processing,
safety feature-readiness checks, model inference, and explanation generation.
"""

import os
import json
import tempfile
import pandas as pd
from typing import Dict, Any, Optional, List
import logging

from backend.app.core.config import settings
from backend.app.core.exceptions import (
    InvalidInputError,
    InsufficientInputDataError,
    ModelNotAvailableError,
    UnsupportedModalityError,
    UnsupportedTaskError,
    ProcessingError,
)
from backend.app.services.model_registry import model_registry
from backend.app.services.explanation_service import explanation_service
from backend.app.services.validation_service import validation_service
from backend.app.api.schemas.common import (
    CommonResponsePayload,
    PredictionDetails,
    ExplanationDetails,
    QualityDetails,
)

from ocr.pipeline import process_lab_report_file, convert_payload_to_ml_dataframe
from ocr.schemas import StructuredLabReportPayload

import torch
from PIL import Image
from preprocessing.images import get_image_transforms

logger = logging.getLogger("backend.services.prediction_service")


class PredictionService:
    """Master Prediction Orchestrator."""

    @staticmethod
    def process_lab_analysis(
        file_bytes: bytes,
        filename: str,
        explicit_structured_input: Optional[Dict[str, Any]] = None,
    ) -> CommonResponsePayload:
        """
        Process uploaded Lab report file through OCR, quality check, safety gate,
        Phase 5 XGBoost, and Phase 6 SHAP.
        """
        ext = validation_service.validate_file_safety(filename, len(file_bytes))

        # Save to temporary file safely for processing
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # 1. Execute OCR & Feature Extraction Pipeline
            payload: StructuredLabReportPayload = process_lab_report_file(tmp_path)

            # 2. General Laboratory Report Analysis Flow (Deterministic Reference Intervals)
            payload_dict = payload.to_dict()
            payload_dict["feature_availability"] = {
                "missing_required_features": [],
                "readiness_summary": {},
            }
            payload_dict["prediction_status"] = "ANALYZED"

            clean_review_reasons = [
                r for r in payload.review_reasons
                if not ("unextracted/missing clinical features" in r.lower() or "population mode/median" in r.lower())
            ]

            # In general lab report workflow, review is required if quality check failed, zero values,
            # or genuine conflicting extractions / unvalidated values are present
            has_conflicts_or_unvalidated = (
                len(payload.laboratory_values) == 0
                or any(v.validation_status != "VALID" for v in payload.laboratory_values)
                or len(clean_review_reasons) > 0
            )

            is_insufficient = len(payload.laboratory_values) == 0

            return CommonResponsePayload(
                status="insufficient_data" if is_insufficient else ("review_required" if has_conflicts_or_unvalidated else "success"),
                modality="lab",
                task="report_analysis",
                prediction=None,
                explanation=ExplanationDetails(
                    method="Clinical Reference Intervals",
                    available=False,
                    summary="No ML disease risk prediction generated for general lab report. Parameter findings are evaluated against physiological reference intervals.",
                    details={},
                ),
                quality=QualityDetails(
                    input_quality="passed" if payload.ocr_confidence > 0 else "failed",
                    ocr_confidence=round(payload.ocr_confidence, 4),
                    extraction_confidence=round(payload.overall_extraction_confidence, 4),
                ),
                review_required=has_conflicts_or_unvalidated,
                review_reasons=clean_review_reasons,
                warnings=[],
                payload_details=payload_dict,
            )

        except Exception as e:
            logger.error(f"Error processing lab analysis: {str(e)}", exc_info=True)
            if isinstance(e, (InvalidInputError, InsufficientInputDataError)):
                raise e
            raise ProcessingError(f"Failed to process lab report: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @staticmethod
    def process_image_analysis(
        modality: str,
        task: str,
        file_bytes: bytes,
        filename: str,
    ) -> CommonResponsePayload:
        """
        Process image modality requests (X-Ray, Skin, Retina).
        Validates input quality and checks Model Registry.
        Executes model inference and Grad-CAM explanation when available.
        """
        ext = validation_service.validate_file_safety(filename, len(file_bytes))
        meta = model_registry.get_metadata(modality, task)

        # Save to temporary file safely for processing
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Check model availability first (e.g. Retina is not available)
            if not model_registry.is_model_available(modality, task):
                logger.info(f"Image analysis request for ({modality}, {task}): Model not yet available.")
                return CommonResponsePayload(
                    status="model_not_available",
                    modality=modality,
                    task=task,
                    prediction=None,
                    explanation=ExplanationDetails(
                        method=meta.xai_method,
                        available=False,
                        summary=f"Visual {meta.xai_method} explanation engine will activate upon completion of model training.",
                        details={"note": "Model not available in current development phase."},
                    ),
                    quality=QualityDetails(
                        input_quality="passed",
                        explanation_quality={},
                    ),
                    review_required=True,
                    review_reasons=[
                        f"Model for modality '{modality}' and task '{task}' is not yet available."
                    ],
                    warnings=[
                        "Patient Safety Notice: Retina model requires dataset acquisition and training."
                    ],
                )

            # 1. Input image quality validation
            quality_res = validation_service.validate_image_input(tmp_path)
            if not quality_res["passed"]:
                raise InvalidInputError(
                    quality_res.get("error_message", "Invalid or corrupted image file."),
                    details={"dimensions": quality_res.get("dimensions")}
                )

            # 2. Image Preprocessing
            with Image.open(tmp_path) as img:
                rgb_img = img.convert("RGB")

            eval_transform = get_image_transforms(target_size=(224, 224), is_training=False)
            input_tensor = eval_transform(rgb_img)  # Shape: [3, 224, 224]
            input_batch = input_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]

            # 3. Model Loading & Inference
            model = model_registry.load_model(modality, task)
            first_param = next(model.parameters(), None)
            device = first_param.device if first_param is not None else torch.device("cpu")
            input_batch = input_batch.to(device)
            
            with torch.no_grad():
                outputs = model(input_batch)
                probs = torch.softmax(outputs, dim=1)[0]

            prob_pos = float(probs[1].item())

            # 4. Modality/Task specific prediction label and threshold mapping
            if modality == "xray" and task == "pneumonia":
                threshold = 0.50
                predicted_label = "PNEUMONIA" if prob_pos >= threshold else "NORMAL"
                target_class_idx = 1 if prob_pos >= threshold else 0
                raw_output = {
                    "pneumonia_probability": round(prob_pos, 4),
                    "normal_probability": round(float(probs[0].item()), 4),
                }

            elif modality == "xray" and task == "tb":
                threshold = 0.50
                predicted_label = "Tuberculosis" if prob_pos >= threshold else "Normal"
                target_class_idx = 1 if prob_pos >= threshold else 0
                raw_output = {
                    "tb_probability": round(prob_pos, 4),
                    "normal_probability": round(float(probs[0].item()), 4),
                    "model_architecture": meta.architecture or "DenseNet121",
                }

            elif modality == "skin" and task == "ham10000":
                # CRITICAL: FROZEN THRESHOLD = 0.40 (Phase 16.2 threshold-calibrated)
                threshold = 0.40
                predicted_label = "HIGHER-RISK" if prob_pos >= threshold else "LOWER-RISK"
                target_class_idx = 1 if prob_pos >= threshold else 0
                raw_output = {
                    "higher_risk_probability": round(prob_pos, 4),
                    "lower_risk_probability": round(float(probs[0].item()), 4),
                    "frozen_threshold": threshold,
                    "risk_group": predicted_label,
                }
            elif modality == "retina" and task == "diabetic_retinopathy":
                threshold = 0.50
                predicted_label = "Diabetic Retinopathy Present" if prob_pos >= threshold else "No Diabetic Retinopathy"
                target_class_idx = 1 if prob_pos >= threshold else 0
                raw_output = {
                    "dr_present_probability": round(prob_pos, 4),
                    "no_dr_probability": round(float(probs[0].item()), 4),
                    "threshold": threshold,
                    "task": "diabetic_retinopathy",
                }
            else:
                raise UnsupportedTaskError(modality, task, model_registry.get_supported_tasks(modality))

            # 5. Grad-CAM Visual Explanation Generation
            exp_engine = explanation_service.get_engine(modality)
            explanation_dict = exp_engine.explain(
                modality=modality,
                task=task,
                input_data={
                    "model": model,
                    "input_tensor": input_batch,
                    "target_class": target_class_idx,
                    "predicted_label": predicted_label,
                    "probability": prob_pos,
                },
            )

            # 6. Build CommonResponsePayload with Patient-Safety Notices
            return CommonResponsePayload(
                status="success",
                modality=modality,
                task=task,
                prediction=PredictionDetails(
                    label=predicted_label,
                    probability=round(prob_pos, 4),
                    threshold=threshold,
                    model_name=meta.model_name or meta.architecture,
                    model_architecture=meta.architecture,
                    raw_output=raw_output,
                ),
                explanation=ExplanationDetails(**explanation_dict),
                quality=QualityDetails(
                    input_quality="passed",
                    explanation_quality={"gradcam_status": "computed" if explanation_dict.get("available") else "failed"},
                ),
                review_required=True,
                review_reasons=[
                    "AI-assisted analysis — For research/decision-support purposes; clinical review required."
                ],
                warnings=[
                    "Patient Safety Notice: AI-assisted analysis — For research/decision-support purposes; clinical review required."
                ],
            )

        except Exception as e:
            logger.error(f"Error processing image analysis ({modality}/{task}): {str(e)}", exc_info=True)
            if isinstance(e, (InvalidInputError, ModelNotAvailableError, UnsupportedModalityError, UnsupportedTaskError)):
                raise e
            raise ProcessingError(f"Failed to process image analysis: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


prediction_service = PredictionService()
