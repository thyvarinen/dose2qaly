"""CT organ-dose extraction from Excel input columns.

CT organ doses come exclusively from per-organ equivalent-dose columns
in the Excel input (e.g. "Brain (CT) mSv", "Liver (CT) mSv").  No
DLP-based coefficient fallback exists.

For CT (photons, wR = 1) the equivalent dose in mSv equals absorbed
organ dose in mGy, so the conversion to Gy is simply mSv / 1000.

This module is consumed by ``exam_doses.compute_exam_organ_doses``.
"""
from __future__ import annotations

from typing import Dict, Optional

from ._parse_utils import to_float as _to_float_local

# Column mapping: Excel column name -> canonical organ name for the risk model.
# These columns are loaded via excel_io.DEFAULT_NEEDED_COLS.

_EXCEL_CT_COLUMN_TO_ORGAN: Dict[str, str] = {
    "Adrenals (CT) mSv": "Adrenals",
    "Brain (CT) mSv": "Brain",
    "Breasts (CT) mSv": "Breast",
    "Colon (CT) mSv": "Colon",
    "Esophagus (CT) mSv": "Oesophagus",
    "Eye Lenses (CT) mSv": "Lens",
    "Gall Bladder (CT) mSv": "Gallbladder",
    "Heart (CT) mSv": "Heart",
    "Kidneys (CT) mSv": "Kidney",
    "Liver (CT) mSv": "Liver",
    "Lungs (CT) mSv": "Lung",
    "Muscle (CT) mSv": "Muscle",
    "Ovaries (CT) mSv": "Ovary",
    "Pancreas (CT) mSv": "Pancreas",
    "Red Marrow (CT) mSv": "Active_Bone_Marrow",
    "Salivary Glands (CT) mSv": "Salivary_Glands",
    "Skeleton (CT) mSv": "Skeleton",
    "Skin (CT) mSv": "Skin",
    "Small Intestine (CT) mSv": "Small_Intestine",
    "Spleen (CT) mSv": "Spleen",
    "Stomach (CT) mSv": "Stomach",
    "Testicles (CT) mSv": "Testicles",
    "Thymus (CT) mSv": "Thymus",
    "Thyroid (CT) mSv": "Thyroid",
    "Upper Large Intestine (CT) mSv": "Upper_Large_Intestine",
    "Urinary Bladder (CT) mSv": "Bladder",
    "Uterus (CT) mSv": "Uterus",
    "Fetus (CT) mSv": "Fetus",
    "Remainder ICRP103 mSv": "Remainder",  # IMPORTANT: Aggregate dose to remaining tissues
}


def ct_organ_doses_from_excel_row(row: dict) -> Optional[Dict[str, float]]:
    """Extract CT organ doses directly from Excel input columns.

    Returns dict of organ_name -> dose in Gy, or None if no valid CT organ
    dose columns are present.  For CT (photons, wR=1) mSv == mGy, so we
    convert mSv -> Gy by dividing by 1000.
    """
    out: Dict[str, float] = {}
    for col, organ in _EXCEL_CT_COLUMN_TO_ORGAN.items():
        val = _to_float_local(row.get(col))
        if val is not None and val > 0:
            dose_gy = val / 1000.0  # mSv -> Gy
            out[organ] = max(out.get(organ, 0.0), dose_gy)
    return out if out else None


def ct_effective_dose_from_excel_row(row: dict) -> Optional[float]:
    """Extract CT effective dose from Excel input columns.

    Returns effective dose in mSv, or None if not available.
    Prefers 'ICRP 103 mSv', falls back to 'Total Effective Dose mSv'.
    """
    ed = _to_float_local(row.get("ICRP 103 mSv"))
    if ed is not None and ed > 0:
        return ed
    ed = _to_float_local(row.get("Total Effective Dose mSv"))
    if ed is not None and ed > 0:
        return ed
    return None
