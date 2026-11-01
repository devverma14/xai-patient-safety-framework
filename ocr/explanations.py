"""
Patient-Friendly Laboratory Parameter Explanation Module.
Provides accessible, medically sound, non-diagnostic descriptions for laboratory parameters,
possible non-diagnostic explanations for abnormal results, and reminders to consult licensed clinicians.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


STANDARD_CLINICAL_DISCLAIMER = (
    "Educational Notice: This explanation is for patient understanding only and does not constitute medical "
    "advice or a diagnosis. Reference ranges represent statistical percentiles of healthy reference populations. "
    "A single abnormal value does not confirm disease. Always consult your physician or healthcare team for "
    "clinical interpretation in the context of your overall health and history."
)


@dataclass
class ParameterMedicalProfile:
    """Standardized medical profile describing a laboratory test parameter."""

    display_name: str
    category: str
    description: str
    possible_causes_high: List[str]
    possible_causes_low: List[str]
    general_guidance: str
    high_meaning: Optional[str] = None
    low_meaning: Optional[str] = None
    high_next_steps: Optional[str] = None
    low_next_steps: Optional[str] = None
    is_percentage: bool = False
    is_derived: bool = False


PARAMETER_PROFILES: Dict[str, ParameterMedicalProfile] = {
    "A1Cresult": ParameterMedicalProfile(
        display_name="Hemoglobin A1c (HbA1c)",
        category="Glycemic Assessment",
        description=(
            "Measures the percentage of hemoglobin in red blood cells that is coated with sugar (glycated). "
            "It reflects average blood glucose levels over the preceding 2 to 3 months (the typical red blood cell lifespan)."
        ),
        possible_causes_high=[
            "Persistent or chronic blood glucose elevation",
            "Impaired insulin secretion or cellular insulin resistance",
            "Conditions that prolong red blood cell lifespan (e.g., iron deficiency anemia, splenectomy)",
            "Certain hemoglobin variants or laboratory assay interferences",
        ],
        possible_causes_low=[
            "Shortened red blood cell survival (e.g., hemolytic anemia, recent blood loss)",
            "Recent blood transfusion",
            "Chronic kidney disease or erythropoietin therapy",
            "Frequent low blood sugar episodes (hypoglycemia)",
        ],
        general_guidance=(
            "Reference ranges (typically 4.0% - 5.6% for non-diabetic individuals) differ from clinical decision "
            "thresholds (5.7%-6.4% prediabetes; 6.5%+ diabetes). Diagnostic confirmation requires repeated testing."
        ),
    ),
    "max_glu_serum": ParameterMedicalProfile(
        display_name="Blood Glucose (Serum / Plasma)",
        category="Metabolic Panel",
        description=(
            "Measures the concentration of glucose (sugar) currently circulating in your bloodstream, "
            "providing the primary source of energy for your cells and tissues."
        ),
        possible_causes_high=[
            "Recent food or carbohydrate intake (non-fasting state)",
            "Acute physical, surgical, or emotional stress (stress hyperglycemia)",
            "Corticosteroid or other medication effects",
            "Infection, acute illness, or dehydration",
            "Impaired glucose tolerance or diabetes mellitus",
        ],
        possible_causes_low=[
            "Prolonged fasting, delayed meals, or skipped carbohydrate intake",
            "Strenuous or prolonged physical exertion",
            "Insulin or diabetes medication dose timing",
            "Severe liver illness or adrenal insufficiency",
        ],
        general_guidance=(
            "Fasting glucose reference intervals (typically 70-99 mg/dL) differ from postprandial (after meal) ranges. "
            "A single elevated glucose value does not constitute a diabetes diagnosis."
        ),
    ),
    "hemoglobin": ParameterMedicalProfile(
        display_name="Hemoglobin (Hgb)",
        category="Complete Blood Count",
        description=(
            "An iron-rich protein inside red blood cells responsible for transporting oxygen from the lungs "
            "to tissues throughout the body and returning carbon dioxide to the lungs."
        ),
        possible_causes_high=[
            "Dehydration or reduced plasma fluid volume (hemoconcentration)",
            "Chronic low oxygen levels (e.g., cigarette smoking, chronic lung disease, high altitude residence)",
            "Bone marrow overproduction (polycythemia)",
        ],
        possible_causes_low=[
            "Iron, vitamin B12, or folate deficiency",
            "Acute or chronic blood loss (e.g., gastrointestinal or menstrual)",
            "Decreased bone marrow production or chronic kidney disease",
            "Increased red blood cell destruction (hemolysis)",
        ],
        general_guidance="Standard adult reference ranges typically span 12.0 - 17.5 g/dL with normal physiological variations by sex and altitude.",
    ),
    "hematocrit": ParameterMedicalProfile(
        display_name="Hematocrit (Hct)",
        category="Complete Blood Count",
        description=(
            "The proportion (percentage) of total blood volume that consists of red blood cells. "
            "It closely parallels hemoglobin levels."
        ),
        possible_causes_high=[
            "Dehydration or fluid restriction",
            "Chronic hypoxia (sleep apnea, smoking, high altitude)",
            "Erythrocytosis or polycythemia",
        ],
        possible_causes_low=[
            "Anemia of various causes",
            "Fluid overload or excessive IV hydration (hemodilution)",
            "Blood loss or nutritional deficiencies",
        ],
        general_guidance="Typically evaluated alongside hemoglobin and red blood cell indices to assess oxygen-carrying capacity.",
    ),
    "rbc": ParameterMedicalProfile(
        display_name="Red Blood Cell Count (RBC)",
        category="Complete Blood Count",
        description="The total number of circulating red blood cells per volume of blood.",
        possible_causes_high=["Dehydration", "Chronic low oxygen conditions", "Polycythemia vera"],
        possible_causes_low=["Anemia", "Blood loss", "Bone marrow suppression", "Hemolysis"],
        general_guidance="Interpreted in combination with hemoglobin, hematocrit, and RBC size indices (MCV, MCH).",
    ),
    "wbc": ParameterMedicalProfile(
        display_name="White Blood Cell Count (WBC)",
        category="Complete Blood Count",
        description=(
            "The total number of immune system cells circulating in the blood, essential for defending "
            "against bacterial, viral, and fungal pathogens, and responding to tissue injury."
        ),
        possible_causes_high=[
            "Infection (bacterial, viral, or fungal)",
            "Systemic inflammation or physical trauma/injury",
            "Physiological or emotional stress",
            "Medications (particularly corticosteroids)",
            "Bone marrow or hematologic conditions",
        ],
        possible_causes_low=[
            "Viral infections suppressing marrow activity",
            "Autoimmune conditions",
            "Medication side effects (e.g., chemotherapy, immunosuppressants)",
            "Severe overwhelming infection (sepsis)",
            "Bone marrow disorders",
        ],
        general_guidance="WBC differential sub-types (neutrophils, lymphocytes, monocytes) provide further context regarding elevated or depressed counts.",
    ),
    "platelets": ParameterMedicalProfile(
        display_name="Platelet Count (PLT)",
        category="Complete Blood Count",
        description="Small cell fragments produced in bone marrow that play a central role in blood clotting and wound repair.",
        possible_causes_high=[
            "Reactive thrombocytosis (response to inflammation, infection, iron deficiency, or surgery)",
            "Splenectomy (absence of spleen filtering)",
            "Primary bone marrow myeloproliferative disorders",
        ],
        possible_causes_low=[
            "Immune-mediated platelet destruction (ITP)",
            "Splenic sequestration (enlarged spleen)",
            "Viral infections or medication side effects",
            "Impaired bone marrow production",
        ],
        general_guidance="Standard reference ranges typically span 150,000 to 450,000 per microliter (150 - 450 x10^3/uL).",
    ),
    "mcv": ParameterMedicalProfile(
        display_name="Mean Corpuscular Volume (MCV)",
        category="Complete Blood Count",
        description="Measures the average physical size or volume of individual red blood cells, helping categorize types of anemia.",
        possible_causes_high=["Vitamin B12 or folate deficiency", "Alcohol consumption", "Liver disease", "Hypothyroidism"],
        possible_causes_low=["Iron deficiency anemia", "Thalassemia trait", "Anemia of chronic disease"],
        general_guidance="Normal range is typically 80 - 100 fL. Values <80 indicate microcytic cells; >100 indicate macrocytic cells.",
    ),
    "mch": ParameterMedicalProfile(
        display_name="Mean Corpuscular Hemoglobin (MCH)",
        category="Complete Blood Count",
        description="The average amount of hemoglobin contained inside a single red blood cell.",
        possible_causes_high=["Macrocytic anemias", "Vitamin B12 / folate deficiency"],
        possible_causes_low=["Iron deficiency anemia", "Thalassemia"],
        general_guidance="Closely correlates with MCV and helps characterize anemia morphology.",
    ),
    "mchc": ParameterMedicalProfile(
        display_name="Mean Corpuscular Hemoglobin Concentration (MCHC)",
        category="Complete Blood Count",
        description="The average concentration of hemoglobin within a given volume of packed red blood cells.",
        possible_causes_high=["Hereditary spherocytosis", "Severe dehydration", "Cold agglutinin disease"],
        possible_causes_low=["Severe iron deficiency", "Thalassemia"],
        general_guidance="Standard range is typically 32 - 36 g/dL.",
    ),
    "rdw": ParameterMedicalProfile(
        display_name="Red Cell Distribution Width (RDW)",
        category="Complete Blood Count",
        description="Measures the variation in size and volume among circulating red blood cells (anisocytosis).",
        possible_causes_high=["Early iron, B12, or folate deficiency", "Mixed nutritional anemias", "Recent blood transfusion"],
        possible_causes_low=["Uniform red blood cell size (rarely of clinical significance)"],
        general_guidance="Helps distinguish iron deficiency (elevated RDW) from uncomplicated thalassemia trait (often normal RDW).",
    ),
    "neutrophils": ParameterMedicalProfile(
        display_name="Neutrophil Count / Percentage",
        category="Complete Blood Count Differential",
        description="The most abundant white blood cell type, acting as first responders against acute bacterial infections and tissue damage.",
        possible_causes_high=["Acute bacterial infection", "Inflammatory conditions", "Physical stress or tissue injury", "Corticosteroids"],
        possible_causes_low=["Viral infections", "Autoimmune neutropenia", "Drug-induced marrow suppression", "Severe nutritional deficit"],
        general_guidance="Interpreted alongside total WBC count and clinical symptoms.",
        is_percentage=True,
    ),
    "lymphocytes": ParameterMedicalProfile(
        display_name="Lymphocyte Count / Percentage",
        category="Complete Blood Count Differential",
        description="White blood cells (including T cells, B cells, and NK cells) central to adaptive immunity, viral defense, and antibody production.",
        possible_causes_high=["Acute or chronic viral infections", "Bacterial infections like pertussis", "Lymphoproliferative disorders"],
        possible_causes_low=["Acute stress response", "Corticosteroid therapy", "Immunodeficiency conditions", "Systemic illness"],
        general_guidance="Proportions naturally shift in response to common respiratory or viral illnesses.",
        is_percentage=True,
    ),
    "sodium": ParameterMedicalProfile(
        display_name="Sodium (Na)",
        category="Electrolytes & Fluid Balance",
        description=(
            "The primary extracellular electrolyte regulating water balance, blood volume, blood pressure, "
            "and electrical signaling in nerve and muscle cells."
        ),
        possible_causes_high=[
            "Dehydration or inadequate water intake",
            "Excessive fluid loss from fever, burns, or sweating",
            "Diabetes insipidus",
            "Excessive intravenous saline administration",
        ],
        possible_causes_low=[
            "Diuretic therapy",
            "Excessive water intake or fluid retention (heart failure, liver cirrhosis, SIADH)",
            "Vomiting, diarrhea, or excessive sweating",
            "Adrenal insufficiency",
        ],
        general_guidance="Normal range is typically 135 - 145 mEq/L. Both rapid elevations and drops require prompt clinical evaluation.",
    ),
    "potassium": ParameterMedicalProfile(
        display_name="Potassium (K)",
        category="Electrolytes & Cardiac Rhythm",
        description="A vital intracellular mineral critical for cardiac muscle rhythm, nerve signal transmission, and skeletal muscle contraction.",
        possible_causes_high=[
            "Decreased kidney excretion (kidney injury or failure)",
            "Potassium-sparing medications or ACE inhibitors/ARBs",
            "Cellular breakdown (hemolysis, tissue trauma)",
            "Acidosis",
        ],
        possible_causes_low=[
            "Diuretic medications (loop or thiazide diuretics)",
            "Vomiting, diarrhea, or gastrointestinal loss",
            "Inadequate dietary potassium intake",
            "Alkalosis or hyperaldosteronism",
        ],
        general_guidance=(
            "Normal range is typically 3.5 - 5.0 mEq/L. Significant variations can impact cardiac electrical stability "
            "and warrant timely clinician review."
        ),
    ),
    "chloride": ParameterMedicalProfile(
        display_name="Chloride (Cl)",
        category="Electrolytes & Acid-Base Balance",
        description="An essential electrolyte that works with sodium and potassium to maintain osmotic pressure and acid-base equilibrium.",
        possible_causes_high=["Dehydration", "Kidney dysfunction", "Metabolic acidosis", "Excessive saline infusion"],
        possible_causes_low=["Prolonged vomiting or nasogastric suction", "Diuretic therapy", "Congestive heart failure", "Metabolic alkalosis"],
        general_guidance="Typically mirrors changes in sodium levels and aids in calculating the anion gap.",
    ),
    "bicarbonate": ParameterMedicalProfile(
        display_name="Bicarbonate (CO2 / HCO3)",
        category="Acid-Base Balance",
        description="A physiological chemical buffer that prevents blood from becoming too acidic or too alkaline.",
        possible_causes_high=["Metabolic alkalosis", "Severe vomiting", "Chronic lung conditions with CO2 retention (e.g., COPD)", "Diuretic use"],
        possible_causes_low=["Metabolic acidosis (e.g., diabetic ketoacidosis, lactic acidosis)", "Severe kidney disease", "Severe diarrhea"],
        general_guidance="Normal adult range is typically 22 - 29 mEq/L. Evaluated in the context of respiration and kidney function.",
    ),
    "bun": ParameterMedicalProfile(
        display_name="Blood Urea Nitrogen (BUN)",
        category="Renal & Metabolic Panel",
        description="Measures the amount of urea nitrogen in the blood, a natural byproduct of dietary protein digestion cleared by the kidneys.",
        possible_causes_high=[
            "Dehydration or hypovolemia",
            "Reduced kidney filtration (acute or chronic kidney disease)",
            "High dietary protein intake or catabolic state",
            "Gastrointestinal bleeding",
            "Congestive heart failure",
        ],
        possible_causes_low=["Very low protein diet or malnutrition", "Severe liver disease (impaired urea synthesis)", "Overhydration"],
        general_guidance="Standard range is typically 7 - 20 mg/dL. The BUN/Creatinine ratio provides additional insight into hydration status.",
    ),
    "creatinine": ParameterMedicalProfile(
        display_name="Serum Creatinine",
        category="Renal Function",
        description="A waste molecule generated by normal muscle breakdown that is filtered almost entirely by healthy kidneys.",
        possible_causes_high=[
            "Reduced kidney filtration capacity (acute kidney injury or chronic kidney disease)",
            "Severe dehydration",
            "Heavy physical exertion or large dietary cooked meat intake",
            "Medications altering tubular secretion (e.g., trimethoprim, cimetidine)",
        ],
        possible_causes_low=["Low muscle mass, frailty, or muscle-wasting conditions", "Severe liver disease", "Pregnancy"],
        general_guidance="Standard adult range is typically 0.6 - 1.3 mg/dL, varying with baseline muscle mass. Used to calculate eGFR.",
    ),
    "egfr": ParameterMedicalProfile(
        display_name="Estimated Glomerular Filtration Rate (eGFR)",
        category="Renal Function",
        description="An estimate of how many milliliters of blood the kidneys filter per minute, calculated from serum creatinine, age, and sex.",
        possible_causes_high=["Normal or hyperfiltration (pregnancy, early diabetes)"],
        possible_causes_low=[
            "Chronic kidney disease (CKD)",
            "Acute kidney injury from medications, contrast, or infection",
            "Severe dehydration or reduced blood flow to kidneys",
        ],
        general_guidance="Values >=60 mL/min/1.73m2 are generally considered adequate in the absence of other kidney damage markers (e.g., proteinuria).",
    ),
    "calcium": ParameterMedicalProfile(
        display_name="Serum Calcium (Total)",
        category="Minerals & Bone Metabolism",
        description="A vital mineral essential for skeletal integrity, neuromuscular transmission, cellular signaling, and blood clotting.",
        possible_causes_high=["Hyperparathyroidism", "Certain malignancies", "Excessive vitamin D intake", "Prolonged immobilization"],
        possible_causes_low=["Hypoparathyroidism", "Vitamin D deficiency", "Low blood albumin level (pseudohypocalcemia)", "Chronic kidney disease"],
        general_guidance="Standard range is typically 8.5 - 10.2 mg/dL. Approximately 40-50% is bound to albumin; ionized calcium may clarify borderline values.",
    ),
    "total_protein": ParameterMedicalProfile(
        display_name="Total Protein",
        category="Hepatic & Nutritional Panel",
        description="Measures the combined amount of the two major classes of blood proteins: albumin and globulins.",
        possible_causes_high=["Dehydration", "Chronic inflammation or infection", "Bone marrow plasma cell disorders (multiple myeloma)"],
        possible_causes_low=["Malnutrition or malabsorption", "Advanced liver disease", "Kidney protein loss (nephrotic syndrome)", "Severe burns"],
        general_guidance="Normal range is typically 6.0 - 8.3 g/dL.",
    ),
    "albumin": ParameterMedicalProfile(
        display_name="Serum Albumin",
        category="Hepatic & Nutritional Panel",
        description="The primary protein synthesized by the liver, essential for maintaining blood oncotic pressure and transporting hormones and drugs.",
        possible_causes_high=["Dehydration (relative concentration)"],
        possible_causes_low=["Chronic liver disease", "Systemic inflammation", "Kidney protein loss", "Malnutrition or protein-losing enteropathy"],
        general_guidance="Normal range is typically 3.5 - 5.0 g/dL. Albumin serves as an indicator of nutritional and hepatic synthetic state.",
    ),
    "bilirubin_total": ParameterMedicalProfile(
        display_name="Total Bilirubin",
        category="Hepatic & Biliary Panel",
        description="An orange-yellow pigment formed during normal breakdown of red blood cells, processed by the liver and excreted in bile.",
        possible_causes_high=[
            "Increased red blood cell breakdown (hemolysis)",
            "Liver cellular injury (hepatitis, toxic injury, cirrhosis)",
            "Bile duct obstruction (gallstones, biliary stricture)",
            "Benign genetic variations in clearance (e.g., Gilbert syndrome)",
        ],
        possible_causes_low=["Generally not clinically significant"],
        general_guidance="Normal range is typically 0.1 - 1.2 mg/dL. Clinicians differentiate between direct (conjugated) and indirect bilirubin.",
    ),
    "alp": ParameterMedicalProfile(
        display_name="Alkaline Phosphatase (ALP)",
        category="Hepatic & Bone Enzymes",
        description="An enzyme found primarily in the cells lining bile ducts within the liver, as well as in active bone-forming cells.",
        possible_causes_high=["Bile duct obstruction or cholestasis", "Liver disease", "Rapid bone turnover or healing fractures", "Pregnancy (placental ALP)"],
        possible_causes_low=["Severe malnutrition, magnesium or zinc deficiency", "Hypophosphatasia (rare)"],
        general_guidance="Standard range is typically 44 - 147 U/L. GGT or bone-specific testing helps determine tissue origin.",
    ),
    "alt": ParameterMedicalProfile(
        display_name="Alanine Aminotransferase (ALT / SGPT)",
        category="Hepatic Enzymes",
        description="An enzyme located predominantly inside liver cells; release into blood indicates hepatocellular stress or irritation.",
        possible_causes_high=[
            "Viral hepatitis (A, B, C)",
            "Metabolic-associated fatty liver disease (MASLD/NAFLD)",
            "Medication or herbal supplement toxicity (e.g., acetaminophen, statins)",
            "Alcohol consumption",
            "Liver ischemia or reduced blood flow",
        ],
        possible_causes_low=["Normal physiological state; occasionally vitamin B6 deficiency"],
        general_guidance="Standard range is typically 7 - 56 U/L. ALT is relatively specific to liver tissue compared to AST.",
    ),
    "ast": ParameterMedicalProfile(
        display_name="Aspartate Aminotransferase (AST / SGOT)",
        category="Hepatic & Muscular Enzymes",
        description="An enzyme present in liver cells, cardiac muscle, skeletal muscle, kidneys, and brain.",
        possible_causes_high=["Liver inflammation or damage", "Intense skeletal muscle exercise or muscle injury (rhabdomyolysis)", "Acute cardiac injury"],
        possible_causes_low=["Normal physiological state"],
        general_guidance="Standard range is typically 10 - 40 U/L. The AST/ALT ratio is useful in identifying alcoholic vs non-alcoholic liver patterns.",
    ),
    "cholesterol_total": ParameterMedicalProfile(
        display_name="Total Cholesterol",
        category="Lipid Panel",
        description="The total amount of cholesterol carried in your blood, including HDL, LDL, and VLDL fractions.",
        possible_causes_high=["Diet high in saturated fats", "Familial hypercholesterolemia", "Hypothyroidism", "Sedentary lifestyle", "Metabolic syndrome"],
        possible_causes_low=["Severe malnutrition", "Hyperthyroidism", "Advanced liver disease", "Severe chronic illness"],
        general_guidance="Desirable range is typically <200 mg/dL. Best evaluated alongside individual lipid fractions (HDL, LDL, triglycerides).",
    ),
    "triglycerides": ParameterMedicalProfile(
        display_name="Triglycerides",
        category="Lipid Panel",
        description="The most common type of fat in the body, derived from dietary fats and extra calories stored for future energy.",
        possible_causes_high=["Recent meal (non-fasting state)", "Excess alcohol intake", "Metabolic syndrome, obesity, or diabetes", "Hypothyroidism", "Genetic hypertriglyceridemia"],
        possible_causes_low=["Very low fat diet", "Malnutrition or malabsorption", "Hyperthyroidism"],
        general_guidance="Normal fasting level is typically <150 mg/dL. Extremely high levels (>500 mg/dL) require evaluation for pancreatitis risk.",
    ),
    "hdl": ParameterMedicalProfile(
        display_name="HDL Cholesterol (High-Density)",
        category="Lipid Panel",
        description="Often termed 'good' cholesterol because it absorbs circulating cholesterol and transports it back to the liver for excretion.",
        possible_causes_high=["Regular aerobic exercise", "Moderate healthy fat intake", "Genetic factors (generally protective)"],
        possible_causes_low=["Sedentary lifestyle", "Cigarette smoking", "Obesity, metabolic syndrome, or uncontrolled diabetes"],
        general_guidance="Optimal target is >=40 mg/dL for men and >=50 mg/dL for women.",
    ),
    "ldl": ParameterMedicalProfile(
        display_name="LDL Cholesterol (Low-Density)",
        category="Lipid Panel",
        description="Often termed 'bad' cholesterol because elevated levels can contribute to atherosclerotic plaque buildup in arterial walls.",
        possible_causes_high=["Dietary saturated/trans fats", "Genetic predisposition (familial hypercholesterolemia)", "Hypothyroidism", "Metabolic syndrome"],
        possible_causes_low=["Statin or lipid-lowering therapy", "Malnutrition or hyperthyroidism"],
        general_guidance="Optimal level is typically <100 mg/dL, with stricter targets (<70 or <55 mg/dL) for individuals with existing cardiovascular risk.",
    ),
    "crp": ParameterMedicalProfile(
        display_name="C-Reactive Protein (CRP)",
        category="Inflammatory Markers",
        description="An acute-phase reactant protein synthesized rapidly by the liver in response to systemic inflammation or infection.",
        possible_causes_high=["Acute bacterial or viral infection", "Autoimmune conditions (e.g., rheumatoid arthritis, lupus)", "Tissue injury or post-surgical state", "Cardiovascular vascular inflammation"],
        possible_causes_low=["Normal baseline physiological state"],
        general_guidance="Standard CRP is typically <3.0 mg/L. It is a non-specific marker of inflammation and does not pinpoint an anatomical location.",
    ),
    "esr": ParameterMedicalProfile(
        display_name="Erythrocyte Sedimentation Rate (ESR)",
        category="Inflammatory Markers",
        description="Measures how quickly red blood cells settle at the bottom of a test tube in one hour, reflecting plasma protein concentration.",
        possible_causes_high=["Systemic infection", "Chronic autoimmune inflammatory disorders", "Malignancy", "Anemia or renal disease"],
        possible_causes_low=["Polycythemia", "Sickle cell disease", "Extreme leukocytosis"],
        general_guidance="Normal range is typically 0 - 20 mm/hr, with higher physiological norms with advancing age and in females.",
    ),
    "tsh": ParameterMedicalProfile(
        display_name="Thyroid Stimulating Hormone (TSH)",
        category="Endocrine / Thyroid Panel",
        description="Produced by the pituitary gland to regulate hormone production by the thyroid gland via negative feedback.",
        possible_causes_high=["Underactive thyroid (primary hypothyroidism)", "Subclinical hypothyroidism", "Inadequate thyroid hormone replacement"],
        possible_causes_low=["Overactive thyroid (hyperthyroidism)", "Subclinical hyperthyroidism", "Excessive thyroid replacement medication", "Acute severe non-thyroidal illness"],
        general_guidance="Standard reference interval is typically 0.4 - 4.0 uIU/mL. Follow-up Free T4 testing is standard when TSH is abnormal.",
    ),
    "age": ParameterMedicalProfile(
        display_name="Patient Age",
        category="Demographics",
        description="Chronological age of the patient recorded on the laboratory requisition header.",
        possible_causes_high=[],
        possible_causes_low=[],
        general_guidance="Patient demographic identifier used for physiological context and reference calibration.",
    ),
    "gender": ParameterMedicalProfile(
        display_name="Patient Gender",
        category="Demographics",
        description="Biological sex or legal gender recorded on the laboratory requisition header.",
        possible_causes_high=[],
        possible_causes_low=[],
        general_guidance="Patient demographic identifier used for sex-stratified reference intervals.",
    ),
    "race": ParameterMedicalProfile(
        display_name="Patient Race / Ethnicity",
        category="Demographics",
        description="Self-reported race or ethnic background recorded in patient registration metadata.",
        possible_causes_high=[],
        possible_causes_low=[],
        general_guidance="Patient demographic metadata.",
    ),
    "vldl": ParameterMedicalProfile(
        display_name="VLDL Cholesterol",
        category="Lipid Profile",
        description="Very Low-Density Lipoprotein particles produced by the liver that primarily transport triglycerides.",
        possible_causes_high=[
            "Elevated circulating triglycerides or recent dietary fat intake",
            "Metabolic syndrome, insulin resistance, or diabetes",
            "Excess alcohol intake",
            "Sedentary lifestyle or weight gain",
        ],
        possible_causes_low=["Low fat diet", "Malnutrition or severe liver disease"],
        general_guidance="Desirable levels are typically <30 mg/dL. Evaluated alongside triglycerides and total lipid profile.",
        high_meaning="VLDL cholesterol travels with triglycerides and rises in tandem when circulating blood fats are elevated.",
        high_next_steps="Discuss with your physician in conjunction with your triglycerides and fasting duration.",
    ),
    "chol_hdl_ratio": ParameterMedicalProfile(
        display_name="Total Cholesterol / HDL Ratio",
        category="Lipid Profile",
        description="A calculated ratio comparing total circulating cholesterol to protective HDL cholesterol.",
        possible_causes_high=["Elevated total cholesterol or triglycerides relative to protective HDL", "Metabolic syndrome", "Sedentary lifestyle"],
        possible_causes_low=["High HDL levels relative to total cholesterol (generally favorable)"],
        general_guidance="Desirable adult ratio is typically below 4.4. Evaluated as a composite cardiovascular risk indicator.",
        high_meaning="A slightly higher ratio reflects a proportionally lower level of protective HDL relative to total circulating lipids.",
        high_next_steps="Review this ratio with your clinician as part of your overall cardiovascular risk assessment.",
        is_derived=True,
    ),
    "ldl_hdl_ratio": ParameterMedicalProfile(
        display_name="LDL / HDL Ratio",
        category="Lipid Profile",
        description="A calculated ratio comparing atherogenic LDL cholesterol to protective HDL cholesterol.",
        possible_causes_high=["Elevated LDL or reduced HDL cholesterol"],
        possible_causes_low=["Favorable lipid balance"],
        general_guidance="Desirable ratio is typically below 3.0. Used in cardiovascular risk estimation.",
        is_derived=True,
    ),
    "ast_alt_ratio": ParameterMedicalProfile(
        display_name="AST / ALT Ratio (SGOT/SGPT Ratio)",
        category="Hepatic Function",
        description="A calculated ratio comparing aspartate aminotransferase to alanine aminotransferase in blood.",
        possible_causes_high=["Alcohol-related liver injury (typically >2.0)", "Advanced fibrosis or cirrhosis", "Strenuous muscle exercise"],
        possible_causes_low=["Metabolic-associated fatty liver changes (MASLD/NAFLD)", "Medication or supplement-induced liver stress", "Characteristic non-alcoholic enzyme pattern"],
        general_guidance="A ratio below 1.0 commonly occurs when ALT rises more than AST, typical of metabolic liver changes.",
        low_meaning="A ratio below 1.0 reflects a pattern where ALT elevation exceeds AST, characteristic of metabolic or medication-related liver stress.",
        low_next_steps="Review alongside your mild ALT elevation with your physician to evaluate potential lifestyle or medication factors.",
        is_derived=True,
    ),
    "rdw_sd": ParameterMedicalProfile(
        display_name="Red Cell Distribution Width - SD (RDW-SD)",
        category="Complete Blood Count",
        description="Measures the actual width of the red blood cell volume distribution curve at the 20% height level in femtoliters.",
        possible_causes_high=["Early nutritional changes in iron, vitamin B12, or folate", "Reticulocyte release during blood recovery", "Mixed cell populations"],
        possible_causes_low=["Uniform red blood cell size"],
        general_guidance="Evaluated alongside RDW-CV (percentage) and hemoglobin to assess red cell size uniformity.",
        high_meaning="A slightly wider red cell size distribution curve indicates mild variation in individual red blood cell volumes.",
        high_next_steps="Your doctor will review this alongside your RDW-CV and normal hemoglobin on routine follow-up.",
    ),
    "alc": ParameterMedicalProfile(
        display_name="Absolute Lymphocyte Count (ALC)",
        category="Complete Blood Count Differential",
        description="The actual absolute concentration of lymphocytes per volume of blood.",
        possible_causes_high=["Recent viral infection (common cold, respiratory illness)", "Recent vaccination", "Immune system activation", "Temporary physical stress"],
        possible_causes_low=["Acute stress", "Corticosteroid therapy", "Immunosuppression"],
        general_guidance="Standard adult range is typically 1.0 - 3.0 x10^3/uL. Proliferates normally during immune responses.",
        high_meaning="An elevated absolute lymphocyte count indicates active proliferation of immune defense cells, frequently following common viral exposure.",
        high_next_steps="Discuss any recent viral illness, cold, or vaccination with your physician; transient elevations typically normalize over time.",
    ),
    "anc": ParameterMedicalProfile(
        display_name="Absolute Neutrophil Count (ANC)",
        category="Complete Blood Count Differential",
        description="The actual absolute number of circulating neutrophils available to fight bacterial infections.",
        possible_causes_high=["Bacterial infection", "Acute inflammation", "Physical stress", "Corticosteroids"],
        possible_causes_low=["Viral illness", "Drug side effects", "Autoimmune conditions"],
        general_guidance="Standard adult range is typically 1.5 - 7.5 x10^3/uL. The primary indicator of infection-fighting capacity.",
    ),
    "aec": ParameterMedicalProfile(
        display_name="Absolute Eosinophil Count (AEC)",
        category="Complete Blood Count Differential",
        description="The absolute concentration of eosinophils, white blood cells involved in allergic and parasitic responses.",
        possible_causes_high=["Allergic rhinitis, seasonal allergies, or asthma", "Eczema or skin sensitivities", "Adverse drug reactions", "Environmental allergens"],
        possible_causes_low=["Normal physiological variation", "Acute stress or corticosteroid use"],
        general_guidance="Standard adult range is typically 0.02 - 0.50 x10^3/uL.",
        high_meaning="A mild elevation in eosinophils is most commonly linked to seasonal allergies, allergic sensitivities, or skin reactions.",
        high_next_steps="Discuss with your doctor in the context of any known allergies, seasonal symptoms, or skin rashes.",
    ),
    "amc": ParameterMedicalProfile(
        display_name="Absolute Monocyte Count (AMC)",
        category="Complete Blood Count Differential",
        description="The absolute number of circulating monocytes, which differentiate into macrophages in tissues.",
        possible_causes_high=["Recovery from acute infection", "Chronic inflammatory states"],
        possible_causes_low=["Acute infection stress response"],
        general_guidance="Standard adult range is typically 0.2 - 0.8 x10^3/uL.",
    ),
    "abc": ParameterMedicalProfile(
        display_name="Absolute Basophil Count (ABC)",
        category="Complete Blood Count Differential",
        description="The absolute number of circulating basophils, which release histamine during allergic responses.",
        possible_causes_high=["Allergic reactions", "Chronic inflammation"],
        possible_causes_low=["Normal physiological baseline"],
        general_guidance="Standard adult range is typically 0.0 - 0.1 x10^3/uL.",
    ),
    "monocytes": ParameterMedicalProfile(
        display_name="Monocyte Percentage",
        category="Complete Blood Count Differential",
        description="The relative percentage of white blood cells that are monocytes.",
        possible_causes_high=["Infection recovery", "Chronic inflammation"],
        possible_causes_low=["Acute stress"],
        general_guidance="Normal range is typically 2 - 10%. Interpreted alongside absolute monocyte count.",
        is_percentage=True,
    ),
    "eosinophils": ParameterMedicalProfile(
        display_name="Eosinophil Percentage",
        category="Complete Blood Count Differential",
        description="The relative percentage of white blood cells that are eosinophils.",
        possible_causes_high=["Allergies, asthma, eczema", "Drug reactions"],
        possible_causes_low=["Normal baseline"],
        general_guidance="Normal range is typically 1 - 6%. Interpreted alongside absolute eosinophil count.",
        is_percentage=True,
    ),
    "basophils": ParameterMedicalProfile(
        display_name="Basophil Percentage",
        category="Complete Blood Count Differential",
        description="The relative percentage of white blood cells that are basophils.",
        possible_causes_high=["Allergic phenomena"],
        possible_causes_low=["Normal baseline"],
        general_guidance="Normal range is typically 0 - 2%.",
        is_percentage=True,
    ),
    "eag": ParameterMedicalProfile(
        display_name="Estimated Average Glucose (eAG)",
        category="Glycemic Assessment",
        description="Calculates estimated average daily blood sugar level over the prior 2 to 3 months directly from HbA1c.",
        possible_causes_high=["Direct mathematical reflection of elevated HbA1c"],
        possible_causes_low=["Direct reflection of lower HbA1c"],
        general_guidance="Derived from the formula: eAG (mg/dL) = 28.7 x HbA1c - 46.7. Independent reference range is not reported.",
        is_derived=True,
    ),
    "bun_cr_ratio": ParameterMedicalProfile(
        display_name="BUN / Creatinine Ratio",
        category="Renal Panel",
        description="Calculates the ratio of blood urea nitrogen to creatinine, helping assess fluid balance and kidney function.",
        possible_causes_high=["Dehydration, reduced renal perfusion, high protein diet, gastrointestinal bleeding"],
        possible_causes_low=["Low protein diet, advanced liver disease, malnutrition"],
        general_guidance="Values between 10:1 and 20:1 are typical in adults, but vary with hydration and muscle mass.",
        is_derived=True,
    ),
    "urea_cr_ratio": ParameterMedicalProfile(
        display_name="Urea / Creatinine Ratio",
        category="Renal Panel",
        description="Calculates the ratio of circulating urea to creatinine to evaluate protein turnover and fluid status.",
        possible_causes_high=["Dehydration, catabolic states, high dietary protein"],
        possible_causes_low=["Low dietary protein, liver disease"],
        general_guidance="Interpreted by clinicians in the context of primary kidney function and hydration.",
        is_derived=True,
    ),
    "mentzer_index": ParameterMedicalProfile(
        display_name="Mentzer Index",
        category="Hematology Indices",
        description="Calculated from MCV divided by RBC count (MCV / RBC). Used in microcytic anemia workups.",
        possible_causes_high=["Values > 13 suggest iron deficiency when microcytic anemia is present"],
        possible_causes_low=["Values < 13 suggest beta-thalassemia trait when microcytic anemia is present"],
        general_guidance="Standalone diagnostic utility requires the presence of microcytic anemia (MCV < 80).",
        is_derived=True,
    ),
    "rdwi": ParameterMedicalProfile(
        display_name="Red Cell Distribution Width Index (RDWI)",
        category="Hematology Indices",
        description="Calculated from (MCV x RDW) / RBC. Evaluates red blood cell distribution in specialized hematology.",
        possible_causes_high=["Iron deficiency patterns in microcytic anemia"],
        possible_causes_low=["Thalassemia trait patterns in microcytic anemia"],
        general_guidance="Interpreted alongside complete blood count morphology by hematology specialists.",
        is_derived=True,
    ),
    "green_king_index": ParameterMedicalProfile(
        display_name="Green and King Index",
        category="Hematology Indices",
        description="Calculated from (MCV^2 x RDW) / (Hemoglobin x 100). Evaluates microcytic red blood cell conditions.",
        possible_causes_high=["Iron deficiency patterns in microcytic anemia"],
        possible_causes_low=["Thalassemia trait patterns in microcytic anemia"],
        general_guidance="A specialized mathematical index for distinguishing microcytosis causes.",
        is_derived=True,
    ),
}

UNPERFORMED_STATUS_LABELS = {
    "Test not taken", "Not Tested", "Not Done", "Not Performed",
    "Pending", "Cancelled", "Sample Clotted", "Sample Hemolyzed",
    "Quantity Not Sufficient", "QNS"
}


def extract_range_bounds(range_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Extract numeric lower and upper bounds from a reference range string."""
    if not range_str:
        return None, None
    clean = re.sub(r"[^\d\.\-<>]", " ", range_str).strip()
    m_range = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", clean)
    if m_range:
        try:
            return float(m_range.group(1)), float(m_range.group(2))
        except ValueError:
            pass
    m_lt = re.search(r"<[=\s]*(\d+(?:\.\d+)?)", range_str)
    if m_lt:
        try:
            return None, float(m_lt.group(1))
        except ValueError:
            pass
    m_gt = re.search(r">[=\s]*(\d+(?:\.\d+)?)", range_str)
    if m_gt:
        try:
            return float(m_gt.group(1)), None
        except ValueError:
            pass
    return None, None


def format_clean_reference_range(range_str: Optional[str], unit: Optional[str]) -> str:
    """Format reference range string ensuring unit is not duplicated."""
    if not range_str or not range_str.strip():
        return "Not provided"
    trimmed = range_str.strip()
    if not unit or not unit.strip():
        return trimmed
    u = unit.strip()
    if u.lower() in trimmed.lower():
        return trimmed
    return f"{trimmed} {u}"


def get_parameter_explanation(
    normalized_name: str,
    raw_test_name: str,
    numeric_value: Optional[float],
    unit: Optional[str],
    interpretation_status: str,
    reference_range: Optional[str] = None,
    range_source: Optional[str] = None,
    raw_value_str: Optional[str] = None,
    validation_status: Optional[str] = None,
    validation_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate structured, patient-friendly, non-diagnostic explanation for an extracted lab parameter.
    Produces analyte-specific 'what_it_means', natural 'possible_causes_text', actionable 'next_steps',
    and specific verification statuses without generic repetitive sentences.
    """
    raw_val_clean = (raw_value_str or (str(numeric_value) if numeric_value is not None else "")).strip()

    # 1. Unperformed tests
    if raw_val_clean in UNPERFORMED_STATUS_LABELS or (
        numeric_value is None and any(w in raw_val_clean.lower() for w in ["not taken", "not tested", "not done", "not performed", "cancelled", "qns"])
    ):
        status_label = "Test not performed"
        what_it_means = f"This test ({raw_test_name}) was listed on the laboratory report panel, but was recorded as '{raw_val_clean or 'Test not taken'}' by the testing laboratory; no measurement was reported."
        causes_text = "Tests may remain unperformed due to laboratory cancellation, sample volume limitations, specimen clotting, or patient preference."
        next_steps = "Discuss with your healthcare provider whether this test was intended and should be rescheduled if clinically relevant."
        return {
            "display_name": raw_test_name or normalized_name,
            "category": "Unperformed Investigations",
            "description": f"Laboratory test '{raw_test_name}' with no measured result reported.",
            "status_summary": status_label,
            "specific_status": status_label,
            "what_it_means": what_it_means,
            "possible_causes_text": causes_text,
            "possible_causes": ["Test not performed by laboratory", "Specimen or collection limitation"],
            "next_steps": next_steps,
            "interpretation_guidance": what_it_means,
            "general_guidance": what_it_means,
            "clinical_disclaimer": STANDARD_CLINICAL_DISCLAIMER,
            "is_supported": True,
        }

    # 2. Patient Demographics
    if interpretation_status in ["VERIFIED", "RECORDED"] or normalized_name in ["age", "gender", "race"]:
        status_label = "Verified patient metadata"
        what_it_means = f"Verified patient demographic characteristic ({raw_test_name}: {raw_val_clean}) extracted from the laboratory report header and confirmed valid."
        causes_text = "Recorded patient demographic metadata used for clinical context and reference interval stratification."
        next_steps = "No action required; recorded as verified patient metadata."
        return {
            "display_name": raw_test_name or normalized_name,
            "category": "Patient Demographics",
            "description": f"Patient demographic information: {raw_test_name}.",
            "status_summary": status_label,
            "specific_status": status_label,
            "what_it_means": what_it_means,
            "possible_causes_text": causes_text,
            "possible_causes": [],
            "next_steps": next_steps,
            "interpretation_guidance": what_it_means,
            "general_guidance": what_it_means,
            "clinical_disclaimer": STANDARD_CLINICAL_DISCLAIMER,
            "is_supported": True,
        }

    # 3. Conflicting Extraction
    if validation_status == "REVIEW_REQUIRED" and validation_message and any(
        w in validation_message.lower() for w in ["conflicting", "duplicate"]
    ):
        status_label = "Conflicting extraction; inspect source"
        what_it_means = f"Multiple competing numeric readings were detected in the source document for {raw_test_name}. Visual audit of the original report is required to verify the printed result."
        causes_text = "Conflicting numbers can occur when historical readings, reference intervals, or adjacent table columns are positioned near the test label in the PDF."
        next_steps = "Use the Inspect Source tool in Section 2 to examine the original extracted lines and verify the printed measurement."
        return {
            "display_name": raw_test_name or normalized_name,
            "category": "Unresolved Extractions",
            "description": f"Laboratory test '{raw_test_name}' with competing candidate readings.",
            "status_summary": status_label,
            "specific_status": status_label,
            "what_it_means": what_it_means,
            "possible_causes_text": causes_text,
            "possible_causes": ["Competing numbers in document layout", "Adjacent table or historical data column"],
            "next_steps": next_steps,
            "interpretation_guidance": what_it_means,
            "general_guidance": what_it_means,
            "clinical_disclaimer": STANDARD_CLINICAL_DISCLAIMER,
            "is_supported": True,
        }

    # Profile lookup
    profile = PARAMETER_PROFILES.get(normalized_name)
    display_name = profile.display_name if profile else raw_test_name
    category = profile.category if profile else "Laboratory Investigation"
    description = profile.description if profile else f"Laboratory parameter '{raw_test_name}' extracted from the clinical report."

    # Bounds & formatted strings
    low_bound, high_bound = extract_range_bounds(reference_range)
    val_disp = f"{numeric_value} {unit}".strip() if (numeric_value is not None and unit) else (raw_val_clean or str(numeric_value))
    ref_disp = format_clean_reference_range(reference_range, unit)

    # 4. Abnormal Findings (HIGH)
    if interpretation_status == "HIGH":
        specific_status = "High"
        degree = "above"
        if high_bound is not None and numeric_value is not None and high_bound > 0:
            ratio = numeric_value / high_bound
            if ratio <= 1.35 or (normalized_name in ["alt", "ast", "alp"] and ratio <= 2.0):
                degree = "mildly above"
            elif ratio <= 1.8:
                degree = "above"
            else:
                degree = "elevated above"

        # Analyte-specific tailoring
        if normalized_name == "A1Cresult":
            what_it_means = f"Hemoglobin A1c (HbA1c) reflects average blood glucose levels over approximately 2 to 3 months based on red blood cell turnover. Your value of {val_disp} is {degree} the report's reference interval ({ref_disp})."
            causes_text = "Elevations are commonly associated with impaired insulin action or secretion, prediabetes, or diabetes. Conditions that extend red blood cell survival, such as iron deficiency, can also alter HbA1c levels."
            next_steps = "Review this result with your clinician for appropriate context, lifestyle discussion, and confirmatory evaluation (such as fasting blood glucose or repeat testing)."
            causes_list = ["Impaired insulin secretion or cellular insulin resistance", "Prediabetes or diabetes mellitus", "Conditions prolonging red blood cell survival (e.g. iron deficiency)", "Recent dietary and lifestyle patterns"]
        elif normalized_name == "triglycerides":
            what_it_means = f"Triglycerides represent circulating and stored dietary fats in the bloodstream. Your level of {val_disp} is {degree} the report limit ({ref_disp})."
            causes_text = "Common contributors include recent non-fasting meals, higher carbohydrate or refined sugar intake, alcohol consumption, metabolic factors, or familial predisposition. Other primary lipids, including Total Cholesterol and LDL, remain within normal reference limits on this report."
            next_steps = "Verify that the blood draw followed an appropriate 9 to 12 hour fast and discuss overall cardiovascular health and lifestyle factors with your clinician."
            causes_list = ["Recent non-fasting meal or carbohydrate intake", "Metabolic syndrome or insulin resistance", "Alcohol intake or dietary fats", "Familial or genetic predisposition"]
        elif normalized_name == "alt":
            what_it_means = f"Alanine Aminotransferase (ALT) is an enzyme found primarily inside liver cells that enters the bloodstream when liver cells experience stress or turnover. Your level of {val_disp} is {degree} the laboratory's reference interval ({ref_disp})."
            causes_text = "Mild elevations frequently stem from non-specific causes, including recent viral illness, strenuous exercise, nutritional or metabolic changes (such as hepatic fat accumulation), alcohol intake, or certain prescription medications and over-the-counter supplements. This result alone does not indicate a specific disease."
            next_steps = "Follow up with your healthcare provider to review potential contributors such as medications, supplements, or recent illness. A repeat test in a few weeks is commonly advised to evaluate whether the elevation is transient."
            causes_list = ["Metabolic or fatty liver changes", "Medication or supplement effects", "Alcohol consumption", "Recent viral illness or strenuous physical exertion"]
        elif normalized_name in ["max_glu_serum", "glucose"]:
            what_it_means = f"Blood glucose measures the concentration of sugar currently circulating in the bloodstream. Your level of {val_disp} is {degree} the reference interval ({ref_disp}). Clinical interpretation depends critically on whether you were fasting for 8 to 12 hours prior to the draw, as blood sugar naturally rises after food or beverage intake."
            causes_text = "If non-fasting, recent dietary intake is the primary explanation. In fasting conditions, elevations may stem from acute physiological stress, illness, medications (such as corticosteroids), or impaired glucose regulation."
            next_steps = "Confirm with your clinician whether this sample was obtained under true fasting conditions, and evaluate the result alongside your HbA1c."
            causes_list = ["Recent meal or carbohydrate consumption (non-fasting state)", "Acute illness, infection, or physiological stress", "Corticosteroid or other medication effects", "Impaired glucose regulation"]
        elif normalized_name == "vldl":
            what_it_means = f"Very Low-Density Lipoprotein (VLDL) is produced by the liver primarily to carry triglycerides through the bloodstream. Your VLDL level of {val_disp} is {degree} the reference limit ({ref_disp})."
            causes_text = "Because VLDL particles carry triglycerides, elevated VLDL directly mirrors the elevated triglycerides seen on this panel. These findings represent a single shared lipid particle pathway rather than multiple independent disorders."
            next_steps = "Review VLDL together with your triglyceride result and fasting state with your healthcare provider."
            causes_list = ["Elevated circulating triglycerides", "Recent dietary fat or carbohydrate intake", "Metabolic syndrome or insulin resistance"]
        elif normalized_name == "chol_hdl_ratio":
            what_it_means = f"Your Total Cholesterol / HDL ratio of {val_disp} is {degree} the desirable limit ({ref_disp}). This mathematical ratio compares total circulating cholesterol against protective HDL ('good') cholesterol."
            causes_text = "A slightly higher ratio occurs when total cholesterol or triglycerides are elevated relative to protective HDL. It provides an atherogenic risk indicator rather than a standalone diagnosis."
            next_steps = "Review this ratio with your clinician as part of your overall cardiovascular risk assessment."
            causes_list = ["Proportionally lower HDL relative to total lipids", "Elevated triglycerides or total cholesterol", "Sedentary lifestyle or dietary factors"]
        elif normalized_name == "mch":
            what_it_means = f"Your MCH of {val_disp} is {degree} the reference interval ({ref_disp}). MCH measures the average amount of hemoglobin inside each red blood cell."
            causes_text = "A borderline reading is typically an incidental physiological variation when total hemoglobin and red blood cell count are normal, or may accompany slight red blood cell enlargement."
            next_steps = "No urgent follow-up is needed for a borderline MCH in the setting of normal hemoglobin; your doctor can review it during routine checkups."
            causes_list = ["Incidental normal physiological variation", "Slight red cell size enlargement", "Adequate iron stores"]
        elif normalized_name in ["rdw", "rdw_sd"]:
            what_it_means = f"Your Red Cell Distribution Width ({raw_test_name}) of {val_disp} is {degree} the reference range ({ref_disp}). RDW measures the variation in red blood cell volume (anisocytosis)."
            causes_text = "Mild elevation indicates a wider mix of cell sizes than usual. This can occur during early nutritional shifts in iron, vitamin B12, or folate stores, after recent blood recovery, or in minor chronic inflammation. Because your total hemoglobin is normal, this is not an active anemia."
            next_steps = "Your physician may check iron stores (serum ferritin) if clinically indicated or monitor on routine checkups."
            causes_list = ["Early or changing iron, B12, or folate stores", "Reticulocyte release during blood cell renewal", "Mild chronic inflammation or recent physical recovery"]
        elif normalized_name == "lymphocytes":
            what_it_means = f"Lymphocytes are white blood cells responsible for viral immunity and antibody production. Your lymphocyte percentage of {val_disp} is {degree} the reference interval ({ref_disp}). As a relative percentage, this indicates a higher proportion of lymphocytes among circulating white cells."
            causes_text = "Relative percentage increases commonly occur during recovery from recent viral infections (such as colds or upper respiratory illness), recent vaccinations, or as a reciprocal balance to lower relative neutrophil percentages."
            next_steps = "Mention any recent cold or viral symptoms or immunizations to your doctor, who can review the relative differential in overall clinical context."
            causes_list = ["Recent common cold, respiratory infection, or viral illness", "Recent vaccination or immune stimulation", "Reciprocal shift from lower neutrophil percentage"]
        elif normalized_name == "alc":
            what_it_means = f"Absolute Lymphocyte Count (ALC) measures the actual number of lymphocytes circulating per volume of blood. Your ALC of {val_disp} is {degree} the reference range ({ref_disp})."
            causes_text = "The total number of circulating lymphocytes often increases temporarily as the immune system actively responds to common viral infections, recent vaccinations, or minor inflammatory triggers."
            next_steps = "Review this mild elevation with your doctor alongside your absolute neutrophil count and general symptoms; transient elevations are commonly monitored with a routine follow-up."
            causes_list = ["Active immune response to viral illness", "Recent vaccination or antigenic stimulus", "Temporary physiological or inflammatory response"]
        elif normalized_name == "aec":
            what_it_means = f"Absolute Eosinophil Count (AEC) measures the circulating number of eosinophils, a specialized white blood cell involved in immune defenses. Your AEC of {val_disp} is {degree} the reference interval ({ref_disp})."
            causes_text = "Elevations can be associated with allergic conditions (such as allergic rhinitis, eczema, or asthma), drug or medication sensitivities, or less commonly, parasitic infections. These associations depend on your personal clinical history, symptoms, and exposures."
            next_steps = "Discuss this finding with your clinician, noting any personal or seasonal allergy symptoms, respiratory concerns, skin reactions, or newly started medications."
            causes_list = ["Seasonal allergies, asthma, or eczema", "Medication or environmental sensitivity", "Less commonly, parasitic or mucosal response"]
        else:
            what_it_means = f"Your {display_name} of {val_disp} is {degree} the reference interval ({ref_disp}). {profile.description if profile else ''}"
            causes_text = f"Possible factors associated with higher readings include {', '.join(profile.possible_causes_high[:3]) if (profile and profile.possible_causes_high) else 'physiological variations, medication effects, or metabolic factors'}. This result alone cannot identify the cause."
            next_steps = f"Discuss this result with your healthcare provider for clinical evaluation in the context of your overall health."
            causes_list = profile.possible_causes_high[:4] if (profile and profile.possible_causes_high) else ["Physiological variation", "Medication or dietary influence"]

        return {
            "display_name": display_name,
            "category": category,
            "description": description,
            "status_summary": "Higher than reference range",
            "specific_status": specific_status,
            "what_it_means": what_it_means,
            "possible_causes_text": causes_text,
            "possible_causes": causes_list,
            "next_steps": next_steps,
            "interpretation_guidance": what_it_means,
            "general_guidance": what_it_means,
            "clinical_disclaimer": STANDARD_CLINICAL_DISCLAIMER,
            "is_supported": True,
        }

    # 5. Abnormal Findings (LOW)
    if interpretation_status == "LOW":
        specific_status = "Low"
        degree = "below"
        if low_bound is not None and numeric_value is not None and low_bound > 0:
            ratio = numeric_value / low_bound
            if ratio >= 0.85:
                degree = "mildly below"
            elif ratio >= 0.6:
                degree = "below"
            else:
                degree = "significantly below"

        if normalized_name == "neutrophils":
            what_it_means = f"Neutrophils are white blood cells that defend against bacterial infections. Your neutrophil percentage of {val_disp} is {degree} the reference interval ({ref_disp}). This reflects a relative differential percentage rather than a deficient total number of cells. Notably, your Absolute Neutrophil Count (ANC) falls within the report's reference interval (2.0 - 7.0 10^3/µL)."
            causes_text = "Differential percentages represent portions of 100%. When another white cell percentage (such as lymphocytes at 47.2%) shifts upward during immune activity, the relative neutrophil percentage decreases proportionally without an actual drop in total neutrophil numbers."
            next_steps = "Your physician will interpret this relative differential in conjunction with your total white blood cell count and clinical symptoms; isolated relative shifts with normal absolute counts rarely require specific intervention."
            causes_list = ["Reciprocal shift due to elevated lymphocyte percentage", "Recent viral illness recovery", "Normal individual differential distribution"]
        elif normalized_name == "ast_alt_ratio":
            what_it_means = f"Your AST/ALT ratio of {val_disp} is {degree} the reference interval ({ref_disp}). This ratio compares AST to ALT in circulation."
            causes_text = "A ratio below 1.0 is characteristic when ALT increases more than AST, typically observed in metabolic liver changes, non-alcoholic hepatic stress, or medication effects (in contrast to alcoholic liver disease where AST typically exceeds ALT)."
            next_steps = "Discuss with your doctor in conjunction with your mild ALT elevation."
            causes_list = ["Metabolic or fatty liver changes", "Medication or supplement-related liver stress", "Characteristic non-alcoholic enzyme pattern"]
        else:
            what_it_means = f"Your {display_name} of {val_disp} is {degree} the reference interval ({ref_disp}). {profile.description if profile else ''}"
            causes_text = f"Possible factors associated with lower readings include {', '.join(profile.possible_causes_low[:3]) if (profile and profile.possible_causes_low) else 'nutritional factors, fluid shifts, or physiological variation'}. This result alone cannot confirm a specific diagnosis."
            next_steps = f"Review this result with your doctor for clinical correlation."
            causes_list = profile.possible_causes_low[:4] if (profile and profile.possible_causes_low) else ["Nutritional factors", "Physiological variation"]

        return {
            "display_name": display_name,
            "category": category,
            "description": description,
            "status_summary": "Lower than reference range",
            "specific_status": specific_status,
            "what_it_means": what_it_means,
            "possible_causes_text": causes_text,
            "possible_causes": causes_list,
            "next_steps": next_steps,
            "interpretation_guidance": what_it_means,
            "general_guidance": what_it_means,
            "clinical_disclaimer": STANDARD_CLINICAL_DISCLAIMER,
            "is_supported": True,
        }

    # 6. Within Reference Range
    if interpretation_status == "WITHIN_RANGE":
        specific_status = "Within range"
        what_it_means = f"Your {display_name} of {val_disp} falls within the standard reference interval ({ref_disp}). This indicates normal expected physiological balance."
        causes_text = "Values within reference limits reflect healthy baseline physiological function for this parameter."
        next_steps = "Continue routine preventive care and healthy lifestyle habits as advised by your healthcare provider."
        return {
            "display_name": display_name,
            "category": category,
            "description": description,
            "status_summary": "Within reference range",
            "specific_status": specific_status,
            "what_it_means": what_it_means,
            "possible_causes_text": causes_text,
            "possible_causes": [],
            "next_steps": next_steps,
            "interpretation_guidance": what_it_means,
            "general_guidance": what_it_means,
            "clinical_disclaimer": STANDARD_CLINICAL_DISCLAIMER,
            "is_supported": True,
        }

    # 7. Unresolved Findings (UNABLE_TO_DETERMINE)
    if numeric_value is not None and (reference_range is None or (profile and profile.is_derived)):
        specific_status = "Reference interval unavailable"
        if normalized_name == "eag":
            what_it_means = f"Your Average Estimated Glucose (eAG) was extracted as {val_disp}. This is an estimated average blood sugar value calculated directly from your HbA1c result using the American Diabetes Association formula. The laboratory report did not publish an independent reference interval for this calculated number."
            causes_text = "Because eAG is derived from HbA1c, its value directly mirrors your HbA1c level. It estimates your daily average glucose over recent months."
            next_steps = "Clinical evaluation should focus on the primary HbA1c (7.80%) and fasting glucose results with your healthcare provider."
            causes_list = ["Mathematical derivation from HbA1c", "Reflection of estimated 2-to-3-month glucose average"]
        elif normalized_name in ["bun_cr_ratio", "urea_cr_ratio"]:
            what_it_means = f"Your {display_name} was extracted as {val_disp}. This is a mathematically calculated ratio from valid source measurements. The testing laboratory did not report a standalone reference interval on this report."
            causes_text = "Ratios reflect the relationship between protein turnover and kidney filtration, and vary with hydration state and dietary protein intake. Both source measurements are within normal limits on this report."
            next_steps = "Your physician will interpret this calculated ratio in the context of your fluid status and primary kidney function tests."
            causes_list = ["Calculated ratio from valid primary analytes", "Hydration and dietary protein factors"]
        elif normalized_name in ["mentzer_index", "rdwi", "green_king_index"]:
            what_it_means = f"Your {display_name} was extracted as {val_disp}. This is a specialized hematologic index calculated from red blood cell indices. The testing laboratory did not report an independent reference interval."
            causes_text = "These indices are used by hematologists specifically when microcytic anemia is present to help differentiate iron deficiency from thalassemia trait. In the presence of normal hemoglobin, standalone diagnostic utility is limited."
            next_steps = "No specific standalone action is required in the setting of normal hemoglobin; discuss with your doctor if comprehensive hematologic workup is indicated."
            causes_list = ["Calculated hematologic discriminator index", "Evaluated primarily in microcytic anemia workups"]
        else:
            what_it_means = f"Your {display_name} was extracted as {val_disp}. The laboratory report did not provide a standalone reference interval for this parameter."
            causes_text = "Calculated parameters or specialized tests often lack fixed reference intervals and require clinical interpretation alongside primary test panels."
            next_steps = "Review this test value with your healthcare provider."
            causes_list = ["No standalone reference interval reported by laboratory", "Requires individualized clinical interpretation"]

        return {
            "display_name": display_name,
            "category": category,
            "description": description,
            "status_summary": specific_status,
            "specific_status": specific_status,
            "what_it_means": what_it_means,
            "possible_causes_text": causes_text,
            "possible_causes": causes_list,
            "next_steps": next_steps,
            "interpretation_guidance": what_it_means,
            "general_guidance": what_it_means,
            "clinical_disclaimer": STANDARD_CLINICAL_DISCLAIMER,
            "is_supported": True,
        }

    # Default fallback
    specific_status = "Interpretation rule unavailable"
    what_it_means = f"The test value '{raw_val_clean}' was extracted from the report, but an automated interpretation rule or verified reference range is unavailable for this specialized parameter."
    causes_text = "Specialized clinical tests require individualized interpretation by a licensed healthcare professional."
    next_steps = "Consult your physician to review this result in the context of your complete medical history."
    return {
        "display_name": display_name,
        "category": category,
        "description": description,
        "status_summary": specific_status,
        "specific_status": specific_status,
        "what_it_means": what_it_means,
        "possible_causes_text": causes_text,
        "possible_causes": ["Specialized or qualitative laboratory parameter", "Reference interval or clinical rule unavailable"],
        "next_steps": next_steps,
        "interpretation_guidance": what_it_means,
        "general_guidance": what_it_means,
        "clinical_disclaimer": STANDARD_CLINICAL_DISCLAIMER,
        "is_supported": False,
    }
