"""Mammography dose conversions.

This module turns a mammography **mean glandular dose (MGD)**, reported
by the imaging device in milligray (mGy), into the two quantities the rest of
the pipeline needs:

* the absorbed **breast** organ dose in gray (Gy), used by the risk engine, and
* the whole-body **effective dose** in millisievert (mSv), used for reporting.

It is called from :mod:`exam_doses` when an examination is a mammography study.
Both helpers accept messy spreadsheet input (strings, comma decimal separators)
and return ``None`` for anything that is missing, non-numeric, zero, or
negative, so callers can simply skip invalid rows.
"""
from __future__ import annotations

from typing import Dict, Optional


def mg_organ_doses_gy_from_mgd(mgd_mgy: float) -> Optional[Dict[str, float]]:
    """Convert mammography mean glandular dose (mGy) to absorbed breast dose (Gy).

    Returns ``{"Breast": dose_Gy}`` or ``None`` when the input is invalid.
    """
    try:
        v = float(str(mgd_mgy).replace(",", "."))
    except Exception:
        return None

    # ``v != v`` is the standard NaN test (NaN is the only value not equal to
    # itself); reject NaN and non-positive doses as invalid.
    if not (v == v) or v <= 0.0:
        return None

    return {"Breast": v / 1000.0}  # mGy -> Gy


def mg_effective_dose_msv_from_mgd(mgd_mgy: float) -> Optional[float]:
    """Convert mammography mean glandular dose (mGy) to effective dose (mSv).

    Uses the conversion factor 0.12 mSv / mGy: the MGD is taken as the
    breast absorbed dose and multiplied by the ICRP Publication 103 breast
    tissue-weighting factor w_T = 0.12 (Table 3), with all other organ
    doses taken as zero.
    Returns ``None`` when the input is invalid.
    """
    try:
        v = float(str(mgd_mgy).replace(",", "."))
    except Exception:
        return None

    # Reject NaN (``v != v``) and non-positive doses as invalid.
    if not (v == v) or v <= 0.0:
        return None

    return v * 0.12
