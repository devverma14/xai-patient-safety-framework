"""
Laboratory value extraction and PII redaction engine.
Extracts structured lab parameters from raw OCR text lines using pattern matching and robust parsing,
preserves original extracted text and source context for traceability, never silently guesses missing units,
and strictly redacts PII before returning structured output.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from .schemas import ExtractedLabValue
from .normalization import normalize_test_name, normalize_unit, clean_test_string, TEST_NAME_SYNONYM_MAP
from .validation import validate_extracted_value


# PII Detection & Redaction Regular Expressions
PII_PATTERNS = [
    (r"(?i)\b(patient|pt)\s*name\s*:\s*([A-Za-z\s,\.]+?)(?=\s*(DOB|Date of birth|MRN|SSN|Patient ID|ID|Phone|Tel|HbA1c|Glucose|Test|\n|$))", r"Patient Name: [REDACTED_PII] "),
    (r"(?i)\b(dob|date of birth)\s*:\s*([\d\/\-]+)", r"DOB: [REDACTED_PII] "),
    (r"(?i)\b(mrn|ssn|patient id|id)\s*:\s*([\w\-]+)", r"Patient ID: [REDACTED_PII] "),
    (r"(?i)\b(phone|tel)\s*:\s*([\d\-\(\)\s]+?)(?=\s*(HbA1c|Glucose|Patient|DOB|Test|\n|$))", r"Phone: [REDACTED_PII] "),
]

# Common unit patterns used in laboratory reports
UNIT_PATTERN = r"(?:%|percent|mg/dl|mg/100ml|mmol/l|g/dl|gm/dl|g/l|10\^3/ul|10\^3/µl|k/ul|k/ull|thous/ul|x10\^3/ul|10\*3/ul|/ul|/ull|cells/ul|10\^6/ul|10\^6/µl|m/ul|mil/ul|x10\^6/ul|10\*6/ul|fl|pg|meq/l|mmol/mol|u/l|iu/l|ml/min/1\.73m2|ml/min/1\.73m²|ml/min|mg/l|mm/hr|uiu/ml|µiu/ml|miu/l|ng/ml|pg/ml|ug/dl|µg/dl|ratio|days|day|d|visits|wisits|count)"

# Common status flags on reports
STATUS_FLAG_PATTERN = r"(?i)\b(CRITICAL|PANIC|OUT_OF_RANGE|ABNORMAL|ABN|HIGH|HI|LOW|NORMAL|NORM|WITHIN_RANGE)\b"


def redact_pii_from_text(raw_text: str) -> str:
    """Redact Patient Identifiable Information (PII) from text string."""
    if not raw_text:
        return ""
    redacted = raw_text
    for pattern, replacement in PII_PATTERNS:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


def clean_ocr_line(line: str) -> str:
    """Clean common OCR noise, unprintable characters, smart quotes, and replacement chars."""
    if not line:
        return ""
    # Strip smart quotes, apostrophes, and replacement characters inside word boundaries (e.g. HbA‘1c -> HbA1c)
    cleaned = re.sub(r"([A-Za-z0-9])[\ufffd\?\*\u2018\u2019\u201c\u201d\u00b4\'\`\"]([A-Za-z0-9])", r"\1\2", line)
    return cleaned.strip()


def extract_reference_range_from_line(line: str, after_pos: int = 0) -> Tuple[Optional[str], int, int]:
    """
    Search for a reference range in the line after a given character position.
    Supports '(4.0-5.6)', '[135-145]', 'Ref: 70-100', '< 200', '> 60'.

    Returns:
        Tuple of (reference_range_string, start_pos, end_pos) or (None, -1, -1).
    """
    search_sub = line[after_pos:]

    # 1. Parenthesized or bracketed range: '(4.0-5.6)', '[135 - 145]', '(0 - 30)'
    paren_match = re.search(r"[\(\[]\s*([0-9\.]+\s*(?:-|–|—|to)\s*[0-9\.]+|[<>]=?\s*[0-9\.]+)\s*[\)\]]", search_sub)
    if paren_match:
        ref_str = paren_match.group(1).strip()
        start = after_pos + paren_match.start()
        end = after_pos + paren_match.end()
        return ref_str, start, end

    # 2. Labeled range: 'Ref: 70 - 100', 'Reference: < 200', 'Range: 3.5 - 5.0'
    labeled_match = re.search(
        r"(?i)\b(?:ref|reference|range|normal|interval)[\s:]*([0-9\.]+\s*(?:-|–|—|to)\s*[0-9\.]+|[<>]=?\s*[0-9\.]+)",
        search_sub,
    )
    if labeled_match:
        ref_str = labeled_match.group(1).strip()
        start = after_pos + labeled_match.start()
        end = after_pos + labeled_match.end()
        return ref_str, start, end

    # 3. Explicit comparison range: '< 200', '>= 60', '<5'
    comp_match = re.search(r"([<>]=?\s*[0-9]+(?:\.[0-9]+)?)", search_sub)
    if comp_match:
        ref_str = comp_match.group(1).strip()
        start = after_pos + comp_match.start()
        end = after_pos + comp_match.end()
        return ref_str, start, end

    # 4. Plain numeric range: '4.2 - 5.7', '70 - 140', '13.0-17.0', '40 - 80'
    range_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?\s*(?:-|–|—|to)\s*[0-9]+(?:\.[0-9]+)?)\b", search_sub)
    if range_match:
        ref_str = range_match.group(1).strip()
        start = after_pos + range_match.start()
        end = after_pos + range_match.end()
        return ref_str, start, end

    return None, -1, -1


def extract_status_flag_from_line(line: str, after_pos: int = 0) -> Optional[str]:
    """Extract clinical status flag from the line (e.g. HIGH, LOW, NORMAL, CRITICAL, H, L)."""
    search_sub = line[after_pos:]

    if re.search(r"(?i)\b(CRITICAL|PANIC)\b", search_sub):
        return "CRITICAL"
    if re.search(r"(?i)\b(HIGH|HI|OUT_OF_RANGE|ABNORMAL|ABN)\b", search_sub):
        return "HIGH"
    if re.search(r"(?i)\b(LOW)\b", search_sub):
        return "LOW"
    if re.search(r"(?i)\b(NORMAL|NORM|WITHIN_RANGE)\b", search_sub):
        return "NORMAL"

    # Standalone single-letter flags (surrounded by word boundaries, space, or end of line)
    # Check for standalone 'H' or 'L' at the end of the line or before/after parentheses
    if re.search(r"(?:^|\s)(?:H|\*H)(?:\s|$)", search_sub):
        return "HIGH"
    if re.search(r"(?:^|\s)(?:L|\*L)(?:\s|$)", search_sub):
        return "LOW"

    return None


# Section headers that indicate narrative / non-measurement sections
NARRATIVE_SECTION_PATTERNS = [
    r"(?i)^(?:interpretation|test\s+interpretation)\b",
    r"(?i)^as\s+per\s+american\s+diabetes\s+association",
    r"(?i)^reference\s+group\b",
    r"(?i)^therapeutic\s+goals\b",
    r"(?i)^remarks?\b",
    r"(?i)^treatment\s+targets\b",
    r"(?i)^updated\s+risk\s+stratification\b",
    r"(?i)^major\s+risk\s+ascvd\b",
    r"(?i)^(?:smart\s+report\s+3\.0\s+)?laboratory\s+equipment\s+dashboard\b",
    r"(?i)^(?:smart\s+report\s+3\.0\s+)?machine\s+&\s+qc\s+details\b",
    r"(?i)^terms\s+&\s+conditions\b",
    r"(?i)^(?:smart\s+report\s+3\.0\s+)?advisory\b",
    r"(?i)^(?:smart\s+report\s+3\.0\s+)?health\s+advisory\b",
    r"(?i)^suggested\s+(?:nutrition|lifestyle|future\s+tests)\b",
    r"(?i)^vitamin\s+d\s+status\b",
    r"(?i)^pregnancy\s+interval\b",
    r"(?i)^egfr-national\s+kidney\s+disease\b",
    r"(?i)^risk\s+category\b",
    r"(?i)^(?:table\s+of\s+contents|index\s+of\s+tests|test\s+directory|tests?\s+index)\b",
]

# Section headers that reset or indicate test results sections
TEST_SECTION_PATTERNS = [
    r"(?i)^department\s+of\b",
    r"(?i)^test\s+name\s+value\s+unit\b",
    r"(?i)^--- page\b",
    r"(?i)^\*\*\*\s*end\s+of\s+report\b",
    r"(?i)^personalized\s+summary\b",
    r"(?i)^10\s+vital\s+health\s+parameters\b",
    r"(?i)^(?:lipid\s+profile|complete\s+blood\s+count|liver\s+function|kidney\s+function|thyroid\s+profile|iron\s+study|glucose-random|c-reactive\s+protein|vitamin\s+b12|vitamin\s+d)\b",
]

# Narrative prose patterns (clinical commentary, guideline sentences, disease descriptions, advisory notices)
NARRATIVE_PROSE_PATTERNS = [
    r"(?i)\b(?:is|are|was|were|may\s+be|can\s+be|has\s+been|have\s+been|should\s+be)\s+(?:used\s+to|suggestive\s+of|more\s+likely|seen\s+in|elevated|decreased|an\s+important|associated\s+with|consistent\s+with|influenced\s+by|caused\s+by|found\s+in|increased\s+in|produced\s+in|largely\s+comprised|the\s+primary|commonly\s+measured|one\s+of\s+the|a\s+steroid\s+hormone|a\s+constituent|termed\s+the|reported\s+due\s+to)\b",
    r"(?i)\b(?:if\s+the\s+value|a\s+value\s+greater\s+than|a\s+value\s+less\s+than|increased\s+levels\s+are|decreased\s+levels\s+are|elevated\s+levels\s+are|levels\s+must\s+be\s+made|levels\s+may\s+also\s+increase)\b",
    r"(?i)\b(?:please\s+note\s+that|note\s*:|notice\s*:|change\s+in\s+the\s+biological\s+reference|with\s+effect\s+from\s+\d)\b",
    r"(?i)\bunlike\s+[A-Za-z\s\(\)\-]+,",
    r"(?i)\b(?:are\s+not\s+influenced\s+by|is\s+not\s+influenced\s+by)\b",
    r"(?i)\b(?:volume\s*\(mcv\)\s+by\s+rbc\s+count)\b",
    r"(?i)\b(?:a\s+significant\s+increase\s+in)\b",
    r"(?i)\b(?:after\s+onset\s+of|measuring\s+changes\s+in|rises\s+rapidly)\b",
    r"(?i)\b(?:participates\s+in\s+a\s+variety|vital\s+role\s+in\s+the\s+excretion|play\s+a\s+vital\s+role)\b",
    r"(?i)\b(?:along\s+with\s+\d+[\s°\ufffd]*homogeneous\s+sample\s+mixing|homogeneous\s+sample\s+mixing)\b",
    r"(?i)\b(?:dated\s*:\s*\d{1,2}[\.\/]\d{1,2}[\.\/]\d{2,4})\b",
]

# Patterns for explicit unperformed/unmeasured test statuses
UNPERFORMED_STATUS_PATTERNS = [
    (r"(?i)\b(?:test\s+not\s+taken|not\s+taken)\b", "Test not taken"),
    (r"(?i)\b(?:test\s+not\s+done|not\s+done)\b", "Not Done"),
    (r"(?i)\b(?:test\s+not\s+performed|not\s+performed)\b", "Not Performed"),
    (r"(?i)\b(?:not\s+tested)\b", "Not Tested"),
    (r"(?i)\b(?:cancelled|canceled)\b", "Cancelled"),
    (r"(?i)\b(?:pending(?:\s+laboratory\s+recalibration)?)\b", "Pending"),
    (r"(?i)\b(?:sample\s+clotted|clotted)\b", "Sample Clotted"),
    (r"(?i)\b(?:sample\s+hemoly[sz]ed|hemoly[sz]ed)\b", "Sample Hemolyzed"),
    (r"(?i)\b(?:quantity\s+not\s+sufficient|qns)\b", "Quantity Not Sufficient"),
]


def is_narrative_start(line: str) -> bool:
    """Check if line initiates an educational, interpretation, advisory, or remarks block."""
    for pat in NARRATIVE_SECTION_PATTERNS:
        if re.search(pat, line):
            return True
    return False


def is_narrative_prose_line(line: str) -> bool:
    """Check if an individual line contains narrative clinical prose, guideline threshold notes, or disclaimer text."""
    for pat in NARRATIVE_PROSE_PATTERNS:
        if re.search(pat, line):
            return True
    return False


def is_test_section_start(line: str) -> bool:
    """Check if line initiates a department or test results table block."""
    if re.search(r"(?i)^(?:performed\s+test\(s\)|frequency\s+of\s+quality|description\s*:)", line):
        return False
    for pat in TEST_SECTION_PATTERNS:
        if re.search(pat, line):
            return True
    return False


def extract_patient_demographics(raw_text: str) -> List[ExtractedLabValue]:
    """
    Extract patient demographic characteristics (Age, Gender) strictly from demographic headers,
    preventing reference range tables or guideline thresholds from corrupting patient demographics.
    """
    results: List[ExtractedLabValue] = []

    # 1. Patient Age
    age_match = re.search(
        r"(?i)\b(?:age(?:/gender|\s*/\s*sex)?\s*[:=]\s*(\d+)|(?:male|female)\s*,\s*(\d+)\s*(?:yrs|y|years)?|patient\s*age\s*[:=]\s*(\d+))",
        raw_text,
    )
    if age_match:
        age_str = age_match.group(1) or age_match.group(2) or age_match.group(3)
        if age_str:
            try:
                val = float(age_str)
                item = ExtractedLabValue(
                    test_name="Age",
                    normalized_name="age",
                    raw_value_str=age_str,
                    numeric_value=val,
                    unit=None,
                    reference_range=None,
                    status_flag=None,
                    extraction_confidence=0.95,
                    validation_status="VALID",
                    source_line=age_match.group(0),
                    source_context="Patient demographic header",
                    is_supported=True,
                )
                results.append(validate_extracted_value(item))
            except ValueError:
                pass

    # 2. Patient Gender
    gender_match = re.search(
        r"(?i)\b(?:age/gender\s*:\s*\d+[^/\n]*?/(male|female)|gender\s*[:=]\s*(male|female)|sex\s*[:=]\s*(male|female)|^(male|female)\s*,\s*\d+)\b",
        raw_text,
        re.MULTILINE,
    )
    if gender_match:
        gen = gender_match.group(1) or gender_match.group(2) or gender_match.group(3) or gender_match.group(4)
        if gen:
            item = ExtractedLabValue(
                test_name="Gender",
                normalized_name="gender",
                raw_value_str=gen.capitalize(),
                numeric_value=None,
                unit=None,
                reference_range=None,
                status_flag=None,
                extraction_confidence=0.95,
                validation_status="VALID",
                source_line=gender_match.group(0),
                source_context="Patient demographic header",
                is_supported=True,
            )
            results.append(validate_extracted_value(item))

    return results


def extract_lab_values_from_text(raw_text: str) -> Tuple[List[ExtractedLabValue], str]:
    """
    Parse raw OCR text, redact PII, and reliably extract laboratory test values.
    Preserves original source lines and context, handles aliases and OCR errors,
    supports multi-line table row lookahead, filters narrative guidelines and thresholds,
    and never silently guesses missing units or values.

    Returns:
        Tuple of (list_of_ExtractedLabValue, pii_redacted_text).
    """
    redacted_text = redact_pii_from_text(raw_text)
    extracted_items: List[ExtractedLabValue] = []

    # 1. Extract patient demographic header fields (Age, Gender) safely
    extracted_items.extend(extract_patient_demographics(raw_text))

    lines = redacted_text.split("\n")

    # Sort synonyms by length so longer specific names match before sub-parts
    sorted_synonyms = sorted(TEST_NAME_SYNONYM_MAP.keys(), key=len, reverse=True)

    in_narrative_block = False
    in_toc_block = False
    line_idx = 0

    while line_idx < len(lines):
        line = lines[line_idx]
        clean_line = clean_ocr_line(line)
        line_idx += 1

        if not clean_line or len(clean_line) < 3:
            continue

        # Check section boundaries
        if is_test_section_start(clean_line) and not re.search(r"(?i)^(?:performed\s+test\(s\)|frequency\s+of\s+quality)", clean_line):
            in_narrative_block = False
            in_toc_block = False

        if is_narrative_start(clean_line):
            in_narrative_block = True
            continue

        if re.search(r"(?i)^(?:table\s+of\s+contents|index\s+of\s+tests|test\s+directory)\b", clean_line):
            in_toc_block = True
            continue

        if in_narrative_block or in_toc_block:
            # Skip narrative / educational / advisory / guideline blocks and TOC
            continue

        if is_narrative_prose_line(clean_line):
            # Skip explanatory narrative sentences and commentary
            continue

        # Skip document headers / summary titles that do not contain values
        if re.match(r"(?i)^[A-Z\s\-_]+(REPORT|PANEL|SUMMARY|DIAGNOSTIC|LABORATORY|RESULTS|PAGE\s*\d+)", clean_line):
            if not re.search(r"\d", clean_line):
                continue

        # Skip header column names line (e.g. "TEST PARAMETER RESULT UNIT REF RANGE")
        if re.search(r"(?i)\b(test\s*parameter|parameter|test\s*name)\b.*\b(result|value)\b", clean_line):
            continue

        # Skip Table of Contents / Index column header lines (e.g. "TEST NAME ... PAGE NO")
        if re.search(r"(?i)\b(test\s*name|parameter|test)\b.*\b(page\s*no|page\s*#|page)\b", clean_line):
            continue

        # Skip metadata lines like "Method: ...", "Machine: ...", "Frequency of Quality Control...", "Description:..."
        if re.match(r"(?i)^(?:method|machine|sample\s*type|frequency\s*of|performed\s*test\(s\)|no\.\s*of\s*test|description)\s*:", clean_line):
            continue

        # Strip section numbering like "1.1 Complete Blood Count" or "2. Lipid Profile"
        clean_line_stripped = re.sub(r"^\s*(?:\d+(?:\.\d+)+|\d+\.)\s+", "", clean_line)

        # Identify all distinct, non-overlapping test name matches on this line
        matched_spans: List[Tuple[int, int, str, str]] = []
        for syn in sorted_synonyms:
            # Demographics (age, gender, race) are extracted exclusively from demographic headers
            if TEST_NAME_SYNONYM_MAP[syn] in ["age", "gender", "race"]:
                continue

            pattern = rf"(?i)\b{re.escape(syn)}\b"
            for m in re.finditer(pattern, clean_line_stripped):
                m_start, m_end = m.start(), m.end()
                # Guard against tokens matched inside units, dates, or ratios (e.g. 'K' in 'K/uL')
                if (m_end < len(clean_line_stripped) and clean_line_stripped[m_end] == "/") or (m_start > 0 and clean_line_stripped[m_start - 1] == "/"):
                    continue
                # Check for overlap with already chosen longer spans
                if any(not (m_end <= s[0] or m_start >= s[1]) for s in matched_spans):
                    continue
                norm_target = TEST_NAME_SYNONYM_MAP[syn]
                # Guard against regular hemoglobin matching on HbA1c lines
                if norm_target == "hemoglobin" and re.search(r"(?i)\b(?:glycated|glycosylated)\s+hemoglobin\b|hba1c|hb\s*a1c", clean_line_stripped):
                    continue
                # Avoid duplicate extraction of the same test parameter from alias synonyms on the same line
                if any(norm_target == s[3] for s in matched_spans):
                    continue
                matched_spans.append((m_start, m_end, syn, norm_target))

        if not matched_spans:
            continue

        # Sort matches by appearance order on the line
        matched_spans.sort(key=lambda s: s[0])

        for span_idx, (t_start, t_end, matched_synonym, matched_norm_key) in enumerate(matched_spans):
            next_start = matched_spans[span_idx + 1][0] if span_idx + 1 < len(matched_spans) else len(clean_line_stripped)
            segment = clean_line_stripped[t_start:next_start]
            after_test_text = clean_line_stripped[t_end:next_start].strip()
            test_display_name = clean_line_stripped[t_start:t_end].strip()

            if after_test_text.startswith(":"):
                after_test_text = after_test_text[1:].strip()
            # Strip trailing parameter descriptors like "Count", "Level", "Concentration", "Quantitative" if not part of synonym
            after_test_text = re.sub(r"(?i)^(?:count|level|levels|concentration|quantitative|total)\b", "", after_test_text).strip()
            if after_test_text.startswith(":"):
                after_test_text = after_test_text[1:].strip()

            # Handle Categorical Features (Diabetes Meds: Insulin, Metformin, etc.)
            if matched_norm_key in ["insulin", "metformin", "glipizide", "glyburide", "diabetesMed", "change"]:
                cat_match = re.search(
                    r"(?i)\b(no|steady|up|down|yes|ch)\b",
                    after_test_text,
                )
                if cat_match:
                    raw_val = cat_match.group(1).capitalize()
                    lab_item = ExtractedLabValue(
                        test_name=test_display_name,
                        normalized_name=matched_norm_key,
                        raw_value_str=raw_val,
                        numeric_value=None,
                        unit=None,
                        reference_range=None,
                        status_flag=None,
                        extraction_confidence=0.95,
                        validation_status="VALID",
                        source_line=clean_line,
                        source_context=f"Line {line_idx}: {clean_line}",
                        is_supported=True,
                    )
                    extracted_items.append(validate_extracted_value(lab_item))
                continue

            # Check if numeric value exists inline on the same line
            # Guard against table of contents / index page numbers (e.g. "C-Reactive Protein (CRP) ..... 6" or "HbA1c ... 4")
            is_index_page_ref = bool(
                re.search(r"(?:\.{2,}|…|\-{2,}|—{2,}|\s{3,})(?:page\s*)?(\d{1,3})\s*$", clean_line_stripped, re.IGNORECASE)
                or re.match(r"^[\s:=.\-–—\(\)]*(?:page\s*|p\.\s*|pg\.\s*)?\d{1,3}\s*$", after_test_text, re.IGNORECASE)
            ) and not re.search(rf"\b({UNIT_PATTERN})\b", after_test_text, re.IGNORECASE) and not re.search(r"[<>]", after_test_text)

            # Match inline numeric value starting directly after test name
            inline_match = None
            if not is_index_page_ref:
                inline_match = re.match(r"^[\s:=.\-–—]*(?:([><]=?)\s*)?([0-9]+(?:\.[0-9]+)?)(.*)$", after_test_text)

            value_line = clean_line
            value_seg = segment
            val_found = False
            unperformed_status = None

            if inline_match:
                # Value found inline directly on the same line
                val_found = True
                val_str = (inline_match.group(1) or "") + inline_match.group(2)
                val_str = val_str.replace(" ", "")
                after_val_text = inline_match.group(3)
                val_end_in_seg = len(val_str)
            else:
                # Value not on same line: inspect subsequent lines for multi-line table format
                lookahead_idx = line_idx
                while lookahead_idx < len(lines) and lookahead_idx < line_idx + 4:
                    cand_line = clean_ocr_line(lines[lookahead_idx])
                    lookahead_idx += 1

                    if not cand_line:
                        continue
                    if is_narrative_start(cand_line) or is_test_section_start(cand_line) or is_narrative_prose_line(cand_line):
                        break
                    # Skip metadata lines like Method: or Machine:
                    if re.match(r"(?i)^(?:method|machine|sample|sample\s*type|technique)\s*:", cand_line):
                        continue
                    if re.match(r"^\s*\([A-Za-z\s]+\)\s*$", cand_line):
                        continue

                    # Check for explicit unperformed status (e.g. "Test not taken", "Pending", "Not Tested")
                    for pat, status_label in UNPERFORMED_STATUS_PATTERNS:
                        if re.search(pat, cand_line):
                            unperformed_status = status_label
                            value_line = f"{clean_line} \n {cand_line}"
                            break
                    if unperformed_status:
                        break

                    # Does this line start with a numeric value or result?
                    # e.g. "7.80 % 4.2 - 5.7" or "169 mg/dl 70 - 140"
                    cand_val_match = re.match(r"^\s*(?:([><]=?)\s*)?([0-9]+(?:\.[0-9]+)?)\s*(.*)$", cand_line)
                    if cand_val_match:
                        val_str = (cand_val_match.group(1) or "") + cand_val_match.group(2)
                        after_val_text = cand_val_match.group(3)
                        value_line = cand_line
                        value_seg = cand_line
                        val_found = True
                        val_end_in_seg = len(val_str)
                        break
                    else:
                        break

            if unperformed_status:
                lab_item = ExtractedLabValue(
                    test_name=test_display_name,
                    normalized_name=matched_norm_key,
                    raw_value_str=unperformed_status,
                    numeric_value=None,
                    unit=None,
                    reference_range=None,
                    status_flag=None,
                    extraction_confidence=0.90,
                    validation_status="REVIEW_REQUIRED",
                    validation_message=f"Measurement unavailable: {unperformed_status}.",
                    source_line=value_line,
                    source_context=f"Line {line_idx}: {value_line}",
                    is_supported=True,
                )
                lab_item = validate_extracted_value(lab_item)
                extracted_items.append(lab_item)
                continue

            if not val_found:
                # Check inline unperformed status
                for pat, status_label in UNPERFORMED_STATUS_PATTERNS:
                    if re.search(pat, after_test_text):
                        unperformed_status = status_label
                        break

                if unperformed_status:
                    lab_item = ExtractedLabValue(
                        test_name=test_display_name,
                        normalized_name=matched_norm_key,
                        raw_value_str=unperformed_status,
                        numeric_value=None,
                        unit=None,
                        reference_range=None,
                        status_flag=None,
                        extraction_confidence=0.90,
                        validation_status="REVIEW_REQUIRED",
                        validation_message=f"Measurement unavailable: {unperformed_status}.",
                        source_line=clean_line,
                        source_context=f"Line {line_idx}: {clean_line}",
                        is_supported=True,
                    )
                    lab_item = validate_extracted_value(lab_item)
                    extracted_items.append(lab_item)
                    continue

                # Only emit unmeasured item if line was a real test row header, not narrative text
                if len(clean_line) < 60 or ":" in clean_line:
                    lab_item = ExtractedLabValue(
                        test_name=test_display_name,
                        normalized_name=matched_norm_key,
                        raw_value_str="",
                        numeric_value=None,
                        unit=None,
                        reference_range=None,
                        status_flag=None,
                        extraction_confidence=0.50,
                        validation_status="REVIEW_REQUIRED",
                        validation_message=f"No valid numeric result found on line for '{test_display_name}'.",
                        source_line=clean_line,
                        source_context=f"Line {line_idx}: {clean_line}",
                        is_supported=True,
                    )
                    lab_item = validate_extracted_value(lab_item)
                    extracted_items.append(lab_item)
                continue

            # Extract Unit from after_val_text
            unit_match = re.search(rf"(?:^|\s|\b)({UNIT_PATTERN})(?=$|\s|[\(\[\],])", after_val_text, re.IGNORECASE)
            raw_unit_str: Optional[str] = None
            after_unit_pos = val_end_in_seg
            is_ambiguous = False

            if unit_match:
                raw_unit_str = unit_match.group(1).strip()
                unit_start = after_val_text.find(unit_match.group(0))
                pre_unit = after_val_text[:unit_start]
                if len(re.findall(r"\b\d+(?:\.\d+)?\b", pre_unit)) > 0:
                    is_ambiguous = True
                after_unit_pos = val_end_in_seg + unit_start + len(unit_match.group(0))
            else:
                no_paren = re.sub(r"[\(\[].*?[\)\]]", "", after_val_text)
                if len(re.findall(r"\b\d+(?:\.\d+)?\b", no_paren)) > 0:
                    is_ambiguous = True

            clean_num_str = re.sub(r"[><=]", "", val_str)
            try:
                num_val = float(clean_num_str)
            except ValueError:
                num_val = None

            standardized_unit: Optional[str] = None
            if raw_unit_str:
                norm_u, is_u_valid = normalize_unit(raw_unit_str, matched_norm_key)
                standardized_unit = norm_u if is_u_valid else raw_unit_str

            ref_range_str, _, _ = extract_reference_range_from_line(value_seg, after_pos=after_unit_pos)
            status_flag = extract_status_flag_from_line(value_seg, after_pos=after_unit_pos)

            conf = 0.95
            if not standardized_unit:
                conf -= 0.15
            if "\ufffd" in clean_line or "?" in clean_line:
                conf -= 0.10
            if num_val is None:
                conf -= 0.20
            conf = max(0.50, min(1.0, conf))

            lab_item = ExtractedLabValue(
                test_name=test_display_name,
                normalized_name=matched_norm_key,
                raw_value_str=val_str,
                numeric_value=num_val,
                unit=standardized_unit,
                reference_range=ref_range_str,
                status_flag=status_flag,
                extraction_confidence=conf,
                validation_status="REVIEW_REQUIRED" if is_ambiguous else "PENDING",
                validation_message=f"Ambiguous multiple numeric values found for '{test_display_name}'." if is_ambiguous else None,
                source_line=value_line,
                source_context=f"Line {line_idx}: {value_line}",
                is_supported=True,
            )

            lab_item = validate_extracted_value(lab_item)
            extracted_items.append(lab_item)

    return extracted_items, redacted_text
