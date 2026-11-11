"""Fluoroscopy / angiography organ-dose coefficients (placeholder).

No coefficients are provided for fluoroscopy or angiography (XA / RF).
Every value in the tables below is 0.0, so these exams yield no organ dose
and no effective dose.

Units
-----
* ``coefficients_fluoro`` — organ doses in mGy per Gy·cm²
* ``eff_coefficients_fluoro`` — effective dose in mSv per Gy·cm²

Lookup hierarchy
----------------
  exam name → sex (F / M / U) → "default" → organ dict / float

This module is consumed by ``exam_doses.compute_exam_organ_doses``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from ._parse_utils import to_float as _xa_to_float


_FLUORO_EXAMS = (
    "Coronary Angiography",
    "Coronary Angioplasty",
    "Carotid Angiography",
    "ERCP",
    "Peripheral Angiography",
    "Abdominal Angiography",
    "Cystography",
    "Unknown",
)

_PLACEHOLDER_ORGAN_COEFFS: Dict[str, float] = {
    "Active_Bone_Marrow": 0.0,
    "Adrenals": 0.0,
    "Brain": 0.0,
    "Breast": 0.0,
    "Colon": 0.0,
    "Endosteum": 0.0,
    "Extrathoracic_Airways": 0.0,
    "Gallbladder": 0.0,
    "Heart": 0.0,
    "Kidneys": 0.0,
    "Liver": 0.0,
    "Lungs": 0.0,
    "Lymph_Nodes": 0.0,
    "Muscle": 0.0,
    "Oesophagus": 0.0,
    "Oral_Mucosa": 0.0,
    "Ovaries": 0.0,
    "Pancreas": 0.0,
    "Prostate": 0.0,
    "Salivary_Glands": 0.0,
    "Skin": 0.0,
    "Small_Intestine": 0.0,
    "Spleen": 0.0,
    "Stomach": 0.0,
    "Testicles": 0.0,
    "Thymus": 0.0,
    "Thyroid": 0.0,
    "Urinary_Bladder": 0.0,
    "Uterus": 0.0,
}

coefficients_fluoro: Dict = {
    _exam: {_s: {"default": dict(_PLACEHOLDER_ORGAN_COEFFS)} for _s in ("F", "M", "U")}
    for _exam in _FLUORO_EXAMS
}

eff_coefficients_fluoro: Dict = {
    _exam: {_s: {"default": 0.0} for _s in ("F", "M", "U")}
    for _exam in _FLUORO_EXAMS
}


_REGION_TO_EXAM = {
    "Cystography": "Cystography",
    "CarotidAngiography": "Carotid Angiography",
    "CoronaryAngiography": "Coronary Angiography",
    "CoronaryAngioplasty": "Coronary Angioplasty",
    "PeripheralAngiography": "Peripheral Angiography",
    "AbdominalAngiography": "Abdominal Angiography",
    "ERCP": "ERCP",
    # Generic region labels
    "Chest": "Coronary Angiography",
    "Neck": "Carotid Angiography",
    "Abdomen": "Abdominal Angiography",
    "Pelvis": "Cystography",
    "Extremities": "Peripheral Angiography",
    "Unknown": "Unknown",
    "Default": "Unknown",
}

_EXAM_KEYWORDS = [
    ("ptca",          "Coronary Angioplasty"),
    ("angioplast",    "Coronary Angioplasty"),
    ("pci",           "Coronary Angioplasty"),
    ("coron",         "Coronary Angiography"),
    ("cardiac",       "Coronary Angiography"),
    ("cardio",        "Coronary Angiography"),
    ("ercp",          "ERCP"),
    ("erc ",          "ERCP"),
    ("endoscop",      "ERCP"),
    ("cysto",         "Cystography"),
    ("voiding",       "Cystography"),
    ("carotid",       "Carotid Angiography"),
    ("aorto",         "Abdominal Angiography"),
    ("aorta",         "Abdominal Angiography"),
    ("peripher",      "Peripheral Angiography"),
    ("femoral",       "Peripheral Angiography"),
    ("lower limb",    "Peripheral Angiography"),
    ("iliac",         "Cystography"),
    ("pelvic",        "Cystography"),
    ("pelvis",        "Cystography"),
    ("thorax",        "Coronary Angiography"),
    ("heart",         "Coronary Angiography"),
]


@lru_cache(maxsize=8192)
def _resolve_fluoro_exam_cached(text: Optional[str], region: str) -> str:
    """Pure, memoised core of :func:`_resolve_fluoro_exam`.

    ``text`` is the already-lowercased ``"{Protocol} {Description}"`` string
    when a row was supplied, else ``None`` (reproducing the row-absent path).
    The fluoroscopy coefficient tables and ``_EXAM_KEYWORDS`` are static after
    import, so memoising on ``(text, region)`` is identical.
    """
    if text is not None:
        for keyword, exam in _EXAM_KEYWORDS:
            if keyword in text:
                return exam
    rl = str(region or "").lower()
    for keyword, exam in _EXAM_KEYWORDS:
        if keyword in rl:
            return exam
    mapped = _REGION_TO_EXAM.get(region)
    if mapped:
        return mapped
    return "Unknown"


def _resolve_fluoro_exam(region: str, row: dict = None) -> str:
    """Map Protocol/Description text or region string to a fluoroscopy exam name."""
    # Build the lookup text as an f-string of the raw row values (so a
    # present-but-None field still stringifies to "None"), then
    # delegate to the memoised core.
    text = (f"{row.get('Protocol', '')} {row.get('Description', '')}".lower()
            if row else None)
    return _resolve_fluoro_exam_cached(text, region)


@lru_cache(maxsize=2048)
def _xa_organ_unit_coeffs(exam: str, sex: str):
    """Per-(exam, sex) tuple of ``(organ, coeff_mGy_per_Gy_cm2)`` with the
    positive-coefficient filtering done once (coefficients are static).

    Pre-filters positive coefficients once so the caller only applies
    ``dap * c / 1000.0`` per organ without re-checking each entry.
    """
    exam_table = coefficients_fluoro.get(exam, {})
    sex_table = exam_table.get(sex) or exam_table.get("U") or {}
    coeffs = sex_table.get("default")
    if not isinstance(coeffs, dict) or not coeffs:
        return None
    out = []
    for organ, coeff in coeffs.items():
        c = _xa_to_float(coeff)
        if c and c > 0:
            out.append((organ, c))
    return tuple(out) if out else None


@lru_cache(maxsize=2048)
def _xa_eff_unit_coeff(exam: str, sex: str) -> Optional[float]:
    """Per-(exam, sex) effective-dose coefficient (mSv per Gy*cm2), memoised."""
    exam_eff = eff_coefficients_fluoro.get(exam, {})
    sex_eff = exam_eff.get(sex) or exam_eff.get("U") or {}
    coeff = sex_eff.get("default")
    c = _xa_to_float(coeff)
    return c if (c is not None and c > 0) else None


def _sex_key(sex_code: str) -> str:
    s = str(sex_code or "U").strip().upper()
    return s if s in ("F", "M") else "U"


def xa_organ_doses_gy_from_dap(
    dap_gy_cm2: float,
    region: str,
    sex_code: str = "U",
    row: dict = None,
) -> Optional[Dict[str, float]]:
    """Return organ absorbed doses in Gy from DAP (Gy*cm2).

    Uses the exam-based fluoroscopy conversion coefficients.
    dose_Gy = DAP_Gy_cm2 * coeff_mGy_per_Gy_cm2 / 1000
    """
    dap = _xa_to_float(dap_gy_cm2)
    if dap is None or dap <= 0:
        return None
    exam = _resolve_fluoro_exam(region, row)
    sex = _sex_key(sex_code)
    unit = _xa_organ_unit_coeffs(exam, sex)
    if not unit:
        return None
    out: Dict[str, float] = {}
    for organ, c in unit:
        dose_gy = dap * c / 1000.0
        if dose_gy > 0:
            out[organ] = dose_gy
    return out if out else None


def xa_effective_dose_msv_from_dap(
    dap_gy_cm2: float,
    region: str,
    sex_code: str = "U",
    row: dict = None,
) -> Optional[float]:
    """Return effective dose in mSv from DAP (Gy*cm2).

    ed_mSv = DAP_Gy_cm2 * coeff_mSv_per_Gy_cm2
    """
    dap = _xa_to_float(dap_gy_cm2)
    if dap is None or dap <= 0:
        return None
    exam = _resolve_fluoro_exam(region, row)
    sex = _sex_key(sex_code)
    c = _xa_eff_unit_coeff(exam, sex)
    if c is None:
        return None
    return dap * c
