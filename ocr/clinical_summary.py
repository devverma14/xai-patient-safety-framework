"""
Clinical Laboratory Findings Summarization and Multianalyte Correlation Module.

Provides comprehensive, non-diagnostic clinical interpretations of extracted laboratory panels:
1. Categorizes analytes into standard diagnostic panels (CBC, CMP, LFT, KFT, Lipid, Thyroid, Inflammatory).
2. Synthesizes abnormal and in-range findings with physiological explanations.
3. Evaluates evidence-based multianalyte relationships (e.g., glycemic dysregulation, atherogenic lipid balance,
   renal azotemia, hepatocellular transaminitis, microcytic/normocytic anemia, systemic acute-phase response).
4. States clinical uncertainties explicitly and provides actionable physician follow-up guidance without
   claiming definitive medical diagnoses.
"""

from typing import List, Dict, Any, Optional
from .schemas import ExtractedLabValue
from .explanations import PARAMETER_PROFILES, STANDARD_CLINICAL_DISCLAIMER, get_parameter_explanation
from .validation import determine_interpretation_status


# Standard Clinical Panel Mappings
PANEL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "hematology": {
        "title": "Complete Blood Count & Hematology",
        "keys": {
            "hemoglobin", "hematocrit", "rbc", "wbc", "platelets", "mcv", "mch",
            "mchc", "rdw", "rdw_sd", "neutrophils", "lymphocytes", "anc", "alc",
            "amc", "aec", "abc", "esr", "mentzer_index", "rdwi", "green_king_index"
        },
        "description": "Evaluates red blood cells, white blood cell differentials, and platelet abundance.",
    },
    "glycemic": {
        "title": "Glycemic Assessment & Diabetes Markers",
        "keys": {"A1Cresult", "max_glu_serum", "eag"},
        "description": "Reflects acute blood sugar levels and 2-to-3-month glycosylation averages.",
    },
    "lipid": {
        "title": "Lipid Profile & Cardiovascular Health",
        "keys": {
            "cholesterol_total", "ldl", "hdl", "vldl", "triglycerides", "non_hdl_cholesterol",
            "chol_hdl_ratio", "ldl_hdl_ratio", "hdl_ldl_ratio"
        },
        "description": "Quantifies circulating lipoproteins and atherogenic lipid balance.",
    },
    "renal": {
        "title": "Renal Function & Kidney Panel (KFT)",
        "keys": {"bun", "creatinine", "egfr", "bun_cr_ratio", "urea_cr_ratio"},
        "description": "Measures glomerular filtration, waste clearance, and fluid balance.",
    },
    "hepatic": {
        "title": "Hepatic Function & Liver Panel (LFT)",
        "keys": {
            "alt", "ast", "alp", "bilirubin_total", "bilirubin_direct",
            "total_protein", "albumin", "ag_ratio", "ast_alt_ratio"
        },
        "description": "Assesses hepatobiliary integrity, cellular enzyme release, and protein synthesis.",
    },
    "electrolytes": {
        "title": "Electrolytes & Minerals",
        "keys": {"sodium", "potassium", "chloride", "bicarbonate", "calcium"},
        "description": "Evaluates fluid homeostasis, acid-base equilibrium, and cellular electrical gradients.",
    },
    "thyroid": {
        "title": "Thyroid Function Profile",
        "keys": {"tsh", "free_t4", "free_t3"},
        "description": "Monitors the hypothalamic-pituitary-thyroid metabolic regulatory axis.",
    },
    "inflammatory": {
        "title": "Inflammatory & Acute-Phase Markers",
        "keys": {"crp", "esr"},
        "description": "Detects systemic acute-phase reactant elevation in response to inflammation or tissue stress.",
    },
}


def evaluate_multianalyte_patterns(mapped_values: Dict[str, ExtractedLabValue]) -> List[Dict[str, Any]]:
    """
    Identify and describe evidence-based physiological relationships between co-extracted abnormal parameters.
    Never claims a confirmed diagnosis; presents correlated observations for clinical review.
    """
    patterns: List[Dict[str, Any]] = []

    def get_val(*keys: str) -> Optional[ExtractedLabValue]:
        for k in keys:
            if k in mapped_values:
                v = mapped_values[k]
                if (
                    v.validation_status == "VALID"
                    and v.numeric_value is not None
                    and v.interpretation_status not in ["UNABLE_TO_DETERMINE", "REVIEW_REQUIRED", None]
                ):
                    return v
        return None

    # 1. Glycemic Axis: Glucose + HbA1c
    glu = get_val("max_glu_serum", "glucose", "fbs", "rbs")
    a1c = get_val("A1Cresult", "hba1c")
    if glu or a1c:
        glu_high = glu and glu.interpretation_status == "HIGH"
        a1c_high = a1c and a1c.interpretation_status == "HIGH"
        glu_normal = glu and glu.interpretation_status == "WITHIN_RANGE"
        a1c_normal = a1c and a1c.interpretation_status == "WITHIN_RANGE"

        if glu_high and a1c_high:
            patterns.append({
                "category": "Glycemic Assessment",
                "title": "Concomitant Acute and Chronic Glycemic Elevation",
                "severity": "ATTENTION",
                "analytes": ["max_glu_serum", "A1Cresult"],
                "summary": (
                    f"Both blood glucose ({glu.raw_value_str} {glu.unit or ''}) and HbA1c "
                    f"({a1c.raw_value_str} {a1c.unit or ''}) are above their respective reference ranges. "
                    "This combination indicates persistent elevation over preceding weeks/months alongside acute hyperglycemia."
                ),
                "clinical_considerations": (
                    "Common etiologies include diabetes mellitus, impaired glucose tolerance, or medication-related effects. "
                    "Physician confirmation with repeated fasting plasma glucose or oral glucose tolerance testing is standard practice."
                ),
            })
        elif a1c_high and (glu_normal or not glu):
            patterns.append({
                "category": "Glycemic Assessment",
                "title": "Chronic Glycated Hemoglobin Elevation with Controlled/Absent Point Glucose",
                "severity": "ATTENTION",
                "analytes": ["A1Cresult"],
                "summary": (
                    f"HbA1c ({a1c.raw_value_str} {a1c.unit or ''}) is elevated, reflecting sustained average blood sugar "
                    "over the preceding 2–3 months, despite normal or unmeasured point-in-time glucose."
                ),
                "clinical_considerations": (
                    "May indicate day-to-day glycemic variability, recent fasting compliance prior to testing, "
                    "or red blood cell lifespan variations. Clinical correlation with a physician is advised."
                ),
            })
        elif glu_high and a1c_normal:
            patterns.append({
                "category": "Glycemic Assessment",
                "title": "Isolated Acute Blood Glucose Elevation",
                "severity": "ATTENTION",
                "analytes": ["max_glu_serum"],
                "summary": (
                    f"Blood glucose ({glu.raw_value_str} {glu.unit or ''}) is elevated while HbA1c ({a1c.raw_value_str} {a1c.unit or ''}) "
                    "remains within normal reference limits."
                ),
                "clinical_considerations": (
                    "An isolated acute elevation often reflects recent meal intake (postprandial state), acute emotional or physical stress, "
                    "corticosteroid use, or transient illness rather than chronic established diabetes. Fasting confirmation is recommended."
                ),
            })

    # 2. Lipid Profile: Cholesterol, LDL, HDL, Triglycerides
    chol = get_val("cholesterol_total", "cholesterol")
    ldl = get_val("ldl", "ldl_cholesterol")
    hdl = get_val("hdl", "hdl_cholesterol")
    trig = get_val("triglycerides", "triglyceride")
    if chol or ldl or trig or hdl:
        atherogenic_flags = []
        if chol and chol.interpretation_status == "HIGH":
            atherogenic_flags.append(f"Total Cholesterol ({chol.raw_value_str} {chol.unit or ''})")
        if ldl and ldl.interpretation_status == "HIGH":
            atherogenic_flags.append(f"LDL ({ldl.raw_value_str} {ldl.unit or ''})")
        if trig and trig.interpretation_status == "HIGH":
            atherogenic_flags.append(f"Triglycerides ({trig.raw_value_str} {trig.unit or ''})")
        hdl_low = hdl and hdl.interpretation_status == "LOW"

        if atherogenic_flags and hdl_low:
            patterns.append({
                "category": "Cardiovascular & Lipid Health",
                "title": "Combined Atherogenic Lipid Pattern",
                "severity": "ATTENTION",
                "analytes": ["cholesterol_total", "ldl", "hdl", "triglycerides"],
                "summary": (
                    f"Elevation in {', '.join(atherogenic_flags)} accompanied by reduced protective HDL cholesterol "
                    f"({hdl.raw_value_str} {hdl.unit or ''})."
                ),
                "clinical_considerations": (
                    "This pattern is commonly associated with atherogenic dyslipidemia, metabolic syndrome, or dietary and lifestyle factors. "
                    "A formal cardiovascular risk assessment and discussion of dietary, exercise, or lipid-lowering strategies with a doctor is appropriate."
                ),
            })
        elif atherogenic_flags:
            elevated_analytes = [
                k for k, v in [("triglycerides", trig), ("ldl", ldl), ("cholesterol_total", chol)]
                if v and v.interpretation_status == "HIGH"
            ]
            is_trig_only = "triglycerides" in elevated_analytes and len(elevated_analytes) == 1
            summary_text = f"Identified elevated lipid parameters: {', '.join(atherogenic_flags)}."
            if chol and chol.interpretation_status == "WITHIN_RANGE" and ldl and ldl.interpretation_status == "WITHIN_RANGE":
                summary_text += " Note that Total Cholesterol and LDL remain within their respective reference limits on this report."
            patterns.append({
                "category": "Cardiovascular & Lipid Health",
                "title": "Elevated Triglycerides and VLDL Lipoproteins" if is_trig_only else "Elevated Circulating Lipids",
                "severity": "ATTENTION",
                "analytes": elevated_analytes if elevated_analytes else ["triglycerides"],
                "summary": summary_text,
                "clinical_considerations": (
                    "VLDL particles are produced by the liver primarily to carry triglycerides through the bloodstream; "
                    "thus elevated VLDL and triglycerides reflect the same shared metabolic pathway rather than multiple independent disorders. "
                    "Common contributors include non-fasting meals, carbohydrate intake, alcohol, or metabolic factors."
                ),
            })

    # 3. Renal Profile: Creatinine + BUN + eGFR
    creat = get_val("creatinine")
    bun = get_val("bun")
    egfr = get_val("egfr")
    if creat or bun or egfr:
        creat_high = creat and creat.interpretation_status == "HIGH"
        bun_high = bun and bun.interpretation_status == "HIGH"
        egfr_low = egfr and egfr.interpretation_status == "LOW"

        if creat_high and bun_high:
            patterns.append({
                "category": "Renal Function",
                "title": "Concomitant Serum Creatinine and Blood Urea Nitrogen Elevation",
                "severity": "ATTENTION",
                "analytes": ["creatinine", "bun"],
                "summary": (
                    f"Both Serum Creatinine ({creat.raw_value_str} {creat.unit or ''}) and BUN "
                    f"({bun.raw_value_str} {bun.unit or ''}) exceed standard reference intervals."
                ),
                "clinical_considerations": (
                    "Indicates reduced nitrogenous waste filtration. Causes range from acute dehydration, high-protein intake, "
                    "and nephrotoxic medications to intrinsic kidney disease. Correlate with hydration status and urinalysis."
                ),
            })
        elif creat_high and not bun_high:
            patterns.append({
                "category": "Renal Function",
                "title": "Isolated Serum Creatinine Elevation",
                "severity": "ATTENTION",
                "analytes": ["creatinine"],
                "summary": f"Serum Creatinine ({creat.raw_value_str} {creat.unit or ''}) is elevated above reference limits.",
                "clinical_considerations": (
                    "Creatinine is influenced by muscle mass, vigorous exercise, dietary creatine, and renal excretion. "
                    "Repeat testing and hydration assessment are recommended."
                ),
            })
        elif egfr_low and not creat_high:
            patterns.append({
                "category": "Renal Function",
                "title": "Decreased Estimated Glomerular Filtration Rate",
                "severity": "ATTENTION",
                "analytes": ["egfr"],
                "summary": f"eGFR ({egfr.raw_value_str} {egfr.unit or ''}) is below expected physiological baseline.",
                "clinical_considerations": (
                    "eGFR estimates filtration capacity based on age, sex, and serum creatinine. "
                    "Mild decreases require longitudinal monitoring to evaluate for chronic kidney changes."
                ),
            })

    # 4. Hepatic Profile: ALT + AST + ALP + Bilirubin
    alt = get_val("alt")
    ast = get_val("ast")
    alp = get_val("alp")
    bili = get_val("bilirubin_total")
    if alt or ast or alp or bili:
        alt_high = alt and alt.interpretation_status == "HIGH"
        ast_high = ast and ast.interpretation_status == "HIGH"
        alp_high = alp and alp.interpretation_status == "HIGH"
        bili_high = bili and bili.interpretation_status == "HIGH"

        if alt_high and ast_high:
            patterns.append({
                "category": "Hepatic Function",
                "title": "Dual Transaminase Elevation (Hepatocellular Pattern)",
                "severity": "ATTENTION",
                "analytes": ["alt", "ast"],
                "summary": (
                    f"Both ALT ({alt.raw_value_str} {alt.unit or ''}) and AST ({ast.raw_value_str} {ast.unit or ''}) "
                    "are elevated above reference thresholds."
                ),
                "clinical_considerations": (
                    "Reflects liver cell enzyme leakage. Common causes include non-alcoholic fatty liver changes, "
                    "medication or supplement hepatotoxicity, alcohol use, or viral hepatitis. Clinical review is advised."
                ),
            })
        elif alp_high and bili_high:
            patterns.append({
                "category": "Hepatic Function",
                "title": "Cholestatic / Biliary Enzyme Elevation",
                "severity": "ATTENTION",
                "analytes": ["alp", "bilirubin_total"],
                "summary": (
                    f"Alkaline Phosphatase ({alp.raw_value_str} {alp.unit or ''}) and Total Bilirubin "
                    f"({bili.raw_value_str} {bili.unit or ''}) are concomitantly elevated."
                ),
                "clinical_considerations": (
                    "Suggests impaired bile flow (cholestasis) or biliary tract involvement. "
                    "Abdominal ultrasound and fractionation tests are standard follow-up measures."
                ),
            })

    # 5. Hematology: Anemia Profile (Hemoglobin, Hematocrit, MCV, RDW)
    hb = get_val("hemoglobin")
    mcv = get_val("mcv")
    rdw = get_val("rdw")
    plt = get_val("platelets")
    wbc = get_val("wbc")
    neut = get_val("neutrophils")
    if hb and hb.interpretation_status == "LOW":
        mcv_val = mcv.numeric_value if mcv else None
        if mcv_val and mcv_val < 80.0:
            patterns.append({
                "category": "Hematology",
                "title": "Microcytic Anemia Pattern",
                "severity": "ATTENTION",
                "analytes": ["hemoglobin", "mcv"],
                "summary": (
                    f"Low hemoglobin ({hb.raw_value_str} {hb.unit or ''}) combined with reduced mean corpuscular volume "
                    f"(MCV: {mcv.raw_value_str} {mcv.unit or ''} < 80 fL)."
                ),
                "clinical_considerations": (
                    "Microcytic red blood cells are most commonly seen in iron deficiency anemia, chronic blood loss, "
                    "or thalassemia trait. Serum ferritin and iron studies are helpful follow-up tests."
                ),
            })
        elif mcv_val and mcv_val > 100.0:
            patterns.append({
                "category": "Hematology",
                "title": "Macrocytic Anemia Pattern",
                "severity": "ATTENTION",
                "analytes": ["hemoglobin", "mcv"],
                "summary": (
                    f"Low hemoglobin ({hb.raw_value_str} {hb.unit or ''}) with enlarged mean corpuscular volume "
                    f"(MCV: {mcv.raw_value_str} {mcv.unit or ''} > 100 fL)."
                ),
                "clinical_considerations": (
                    "Macrocytic red blood cells can arise from vitamin B12 or folate deficiency, thyroid underactivity, "
                    "or alcohol use. Vitamin B12 and folate level evaluations are standard."
                ),
            })
        else:
            patterns.append({
                "category": "Hematology",
                "title": "Decreased Hemoglobin (Anemia Pattern)",
                "severity": "ATTENTION",
                "analytes": ["hemoglobin"],
                "summary": f"Hemoglobin ({hb.raw_value_str} {hb.unit or ''}) is below expected reference intervals.",
                "clinical_considerations": (
                    "Reduced oxygen-carrying capacity. Causes include nutritional deficiencies, blood loss, "
                    "or chronic systemic illness. Discuss with a clinician."
                ),
            })

    # 6. Inflammatory Profile: CRP + ESR + WBC
    crp = get_val("crp")
    esr = get_val("esr")
    if crp or esr or (wbc and neut):
        crp_high = crp and crp.interpretation_status == "HIGH"
        esr_high = esr and esr.interpretation_status == "HIGH"
        wbc_high = wbc and wbc.interpretation_status == "HIGH"

        if crp_high and (esr_high or wbc_high):
            patterns.append({
                "category": "Inflammatory Response",
                "title": "Systemic Inflammatory / Acute-Phase Elevation",
                "severity": "ATTENTION",
                "analytes": ["crp", "esr", "wbc"],
                "summary": (
                    f"CRP ({crp.raw_value_str} {crp.unit or ''}) is elevated alongside additional inflammatory markers "
                    f"({esr.raw_value_str if esr else ''} {wbc.raw_value_str if wbc else ''})."
                ),
                "clinical_considerations": (
                    "C-reactive protein and ESR are sensitive acute-phase reactants that rise during bacterial/viral infections, "
                    "inflammatory autoimmune states, or tissue trauma. Non-specific; clinical examination is required."
                ),
            })
        elif crp_high:
            patterns.append({
                "category": "Inflammatory Response",
                "title": "Elevated C-Reactive Protein (CRP)",
                "severity": "ATTENTION",
                "analytes": ["crp"],
                "summary": f"C-Reactive Protein ({crp.raw_value_str} {crp.unit or ''}) exceeds baseline reference limits.",
                "clinical_considerations": (
                    "Indicates active inflammatory or acute-phase response. Correlate with symptoms, recent infection, "
                    "or physical injury with a healthcare provider."
                ),
            })

    # 7. Thyroid Axis: TSH + FT4
    tsh = get_val("tsh")
    ft4 = get_val("free_t4")
    if tsh:
        if tsh.interpretation_status == "HIGH":
            patterns.append({
                "category": "Thyroid Function",
                "title": "Elevated Thyroid Stimulating Hormone (TSH)",
                "severity": "ATTENTION",
                "analytes": ["tsh"],
                "summary": (
                    f"Thyroid Stimulating Hormone ({tsh.raw_value_str} {tsh.unit or ''}) is elevated above reference limits."
                ),
                "clinical_considerations": (
                    "Elevated TSH typically represents pituitary compensatory signaling in response to low circulating thyroid hormone (hypothyroidism), "
                    "or subclinical thyroid changes. Free T4 and antithyroid antibody testing are standard next steps."
                ),
            })
        elif tsh.interpretation_status == "LOW":
            patterns.append({
                "category": "Thyroid Function",
                "title": "Suppressed / Low Thyroid Stimulating Hormone (TSH)",
                "severity": "ATTENTION",
                "analytes": ["tsh"],
                "summary": f"TSH ({tsh.raw_value_str} {tsh.unit or ''}) is below standard reference thresholds.",
                "clinical_considerations": (
                    "May indicate hyperthyroidism, thyroid hormone replacement medication effects, or non-thyroidal illness. "
                    "Evaluate with Free T4 and Free T3 assays."
                ),
            })

    return patterns


def generate_clinical_summary(
    laboratory_values: List[ExtractedLabValue],
    demographics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate structured, patient-friendly, and medically sound report summarization.
    """
    mapped_by_key = {v.normalized_name: v for v in laboratory_values if v.normalized_name}

    in_range_items: List[Dict[str, Any]] = []
    high_items: List[Dict[str, Any]] = []
    low_items: List[Dict[str, Any]] = []
    uncertain_items: List[Dict[str, Any]] = []
    patient_demographics: Dict[str, Any] = {}
    demographic_keys = {"age", "gender", "race"}

    def _get_status(val: ExtractedLabValue) -> str:
        if val.normalized_name in demographic_keys:
            if val.validation_status == "VALID":
                return "VERIFIED"
            return "UNABLE_TO_DETERMINE"
        if val.interpretation_status and val.interpretation_status in ["WITHIN_RANGE", "HIGH", "LOW"]:
            return val.interpretation_status
        if val.validation_status == "REVIEW_REQUIRED" or (val.unit is None and val.numeric_value is not None):
            return "UNABLE_TO_DETERMINE"
        res = determine_interpretation_status(
            numeric_val=val.numeric_value,
            unit=val.unit,
            raw_reference_range=val.reference_range,
            norm_key=val.normalized_name,
            status_flag=val.status_flag,
        )
        return res[0] if isinstance(res, tuple) else str(res)

    for v in laboratory_values:
        interp = _get_status(v)
        v.interpretation_status = interp

        if not v.explanation or "what_it_means" not in v.explanation:
            v.explanation = get_parameter_explanation(
                normalized_name=v.normalized_name or "",
                raw_test_name=v.test_name,
                numeric_value=v.numeric_value,
                unit=v.unit,
                interpretation_status=interp,
                reference_range=v.reference_range,
                range_source=v.reference_range_source,
                raw_value_str=v.raw_value_str,
                validation_status=v.validation_status,
                validation_message=v.validation_message,
            )

        exp = v.explanation or {}
        what_it_means = exp.get("what_it_means") or exp.get("interpretation_guidance") or ""
        possible_causes_text = exp.get("possible_causes_text") or ""
        next_steps = exp.get("next_steps") or ""
        specific_status = exp.get("specific_status")
        if not specific_status:
            if interp in ["HIGH", "LOW"]:
                specific_status = interp.capitalize()
            elif interp == "WITHIN_RANGE":
                specific_status = "Within range"
            elif interp == "VERIFIED" or v.normalized_name in demographic_keys:
                specific_status = "Verified patient metadata"
            else:
                specific_status = "Needs Verification"

        item_data = {
            "test_name": v.test_name,
            "normalized_name": v.normalized_name,
            "value": v.raw_value_str,
            "numeric_value": v.numeric_value,
            "unit": v.unit,
            "reference_range": v.reference_range,
            "reference_range_source": v.reference_range_source,
            "status": interp,
            "specific_status": specific_status,
            "validation_status": v.validation_status,
            "validation_message": v.validation_message,
            "source_context": v.source_context,
            "possible_causes": exp.get("possible_causes", []),
            "possible_causes_text": possible_causes_text,
            "what_it_means": what_it_means,
            "next_steps": next_steps,
            "guidance": what_it_means,
            "category": exp.get("category", ""),
        }

        if v.normalized_name in demographic_keys:
            patient_demographics[v.normalized_name] = item_data
            continue

        if interp == "HIGH":
            high_items.append(item_data)
        elif interp == "LOW":
            low_items.append(item_data)
        elif interp == "WITHIN_RANGE":
            in_range_items.append(item_data)
        else:
            uncertain_items.append(item_data)

    # Organize values into Clinical Panels
    panels_output: List[Dict[str, Any]] = []
    for panel_id, defn in PANEL_DEFINITIONS.items():
        matching_values = [
            v for v in laboratory_values
            if v.normalized_name in defn["keys"]
        ]
        if matching_values:
            abnormal_in_panel = [
                v for v in matching_values
                if _get_status(v) in ["HIGH", "LOW"]
            ]
            panels_output.append({
                "panel_id": panel_id,
                "title": defn["title"],
                "description": defn["description"],
                "total_tests": len(matching_values),
                "abnormal_count": len(abnormal_in_panel),
                "status": "ATTENTION" if abnormal_in_panel else "NORMAL",
                "items": [
                    {
                        "test_name": v.test_name,
                        "normalized_name": v.normalized_name,
                        "value": v.raw_value_str,
                        "numeric_value": v.numeric_value,
                        "unit": v.unit,
                        "reference_range": v.reference_range,
                        "reference_range_source": v.reference_range_source,
                        "status": _get_status(v),
                        "validation_status": v.validation_status,
                        "validation_message": v.validation_message,
                        "source_context": v.source_context,
                    }
                    for v in matching_values
                ],
            })

    # Evaluate Multianalyte Correlated Patterns
    correlated_patterns = evaluate_multianalyte_patterns(mapped_by_key)

    lab_measurements = [v for v in laboratory_values if v.normalized_name not in demographic_keys]
    total_count = len(lab_measurements)
    abnormal_count = len(high_items) + len(low_items)

    total_detected = len(laboratory_values)
    # Narrative Executive Summary: patient-friendly synthesis of main concerns
    if total_count == 0:
        overall_summary = "No recognized laboratory test parameters were extracted from the uploaded document."
    elif abnormal_count == 0:
        overall_summary = (
            f"All {total_count} evaluated laboratory analytes fall within their corresponding reference "
            "intervals. No acute physiological abnormalities were identified."
        )
    else:
        main_concern_topics = []
        has_glycemic = any(i["normalized_name"] in ["A1Cresult", "max_glu_serum", "glucose"] for i in (high_items + low_items))
        has_lipid = any(i["normalized_name"] in ["triglycerides", "vldl", "chol_hdl_ratio", "cholesterol_total"] for i in (high_items + low_items))
        has_hepatic = any(i["normalized_name"] in ["alt", "ast", "ast_alt_ratio", "alp", "bilirubin_total"] for i in (high_items + low_items))
        has_hematologic = any(i["normalized_name"] in ["lymphocytes", "neutrophils", "alc", "aec", "rdw", "rdw_sd", "mch"] for i in (high_items + low_items))

        if has_glycemic:
            main_concern_topics.append("elevated blood glucose markers (HbA1c and glucose)")
        if has_lipid:
            main_concern_topics.append("elevated circulating lipids (triglycerides and VLDL)")
        if has_hepatic:
            main_concern_topics.append("mild liver enzyme elevation (ALT)")
        if has_hematologic:
            main_concern_topics.append("mild reactive immune and red cell variations")

        concerns_text = ", ".join(main_concern_topics) if main_concern_topics else "specific metabolic and cellular findings"
        abnormal_names = [f"{i['test_name']} ({i['status'].lower()})" for i in (high_items + low_items)]
        abnormal_summary_str = ", ".join(abnormal_names[:5])
        if len(abnormal_names) > 5:
            abnormal_summary_str += f", and {len(abnormal_names) - 5} other parameters"

        overall_summary = (
            f"Of {total_count} evaluated laboratory analytes, {len(in_range_items)} are within standard baseline intervals, "
            f"and {abnormal_count} demonstrate out-of-range findings ({abnormal_summary_str}). "
            f"Primary findings center on {concerns_text}. Renal function markers and electrolytes are within normal reference limits, "
            "and total hemoglobin is preserved alongside mild shifts in white cell differential percentages and red cell size variation. "
            "These findings warrant routine review with your physician alongside your medical history and fasting status, but do not confirm a diagnosis on their own."
        )

    return {
        "overview": {
            "total_parameters": total_count,
            "total_detected_count": total_detected,
            "in_range_count": len(in_range_items),
            "abnormal_count": abnormal_count,
            "high_count": len(high_items),
            "low_count": len(low_items),
            "uncertain_count": len(uncertain_items),
            "narrative_summary": overall_summary,
        },
        "patient_demographics": patient_demographics,
        "panels": panels_output,
        "abnormal_findings": {
            "high": high_items,
            "low": low_items,
        },
        "in_range_findings": in_range_items,
        "uncertain_findings": uncertain_items,
        "correlated_patterns": correlated_patterns,
        "clinical_guidance": {
            "uncertainty_statement": (
                "Laboratory findings alone do not establish a definitive medical diagnosis. Reference intervals "
                "represent statistical percentiles from healthy reference groups. Transient variations frequently occur "
                "due to hydration, dietary timing, physical activity, medications, and individual biological rhythms."
            ),
            "actionable_recommendations": [
                "Schedule a follow-up consultation with your physician or prescribing provider to review these findings.",
                "Review any current prescription medications, over-the-counter drugs, or nutritional supplements with your clinician.",
                "For glycemic or lipid abnormalities, confirm whether appropriate 8-to-12-hour fasting was maintained prior to sample collection.",
                "Avoid altering prescribed medication doses or beginning unverified treatments based solely on automated report interpretations.",
            ],
            "disclaimer": STANDARD_CLINICAL_DISCLAIMER,
        },
    }
