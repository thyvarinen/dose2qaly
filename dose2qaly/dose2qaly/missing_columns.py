"""Categorise and format the user-facing "missing columns" warning.

The simulation tolerates absent Excel columns -- ``row.get(...)`` simply
returns ``None`` and the matching dose calculation is skipped.  When that
happens we want to tell the user *which* simulation features each absent
column would have driven, instead of dumping a flat alphabetical list.

This module is the single source of truth for that grouping.  It is used by:

* ``dose2qaly.excel_io.report_missing_headers``  (runtime stderr +
  ``on_progress("missing_columns", ...)`` event; called from
  ``preprocess.load_and_preprocess`` on every run).
* ``dose2qaly.config``                  (manual "Check
  Against Selected Files" button and the pre-flight ``Continue?`` dialog).

The category order matches the order in which a typical user thinks about
the input data:  patient/exam metadata first, then per-modality dose inputs
in the modality dispatch order used by ``compute_exam_organ_doses``.

Public API
----------
* ``categorize(columns)``         -> ``OrderedDict[category, list[col]]``
* ``format_per_file_summary(...)``  -> multi-line string for stderr / dialog
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Tuple
import os


# Category definitions (display order)
# Each entry is (category_label, set_of_exact_column_names).  The "(CT) mSv",
# "(NM) mSv" and "(Total) mSv" per-organ columns are matched by suffix
# inside ``categorize`` rather than enumerated here.

_CATEGORY_COMMON = "Common (patient, demographics & exam metadata)"
_CATEGORY_CT = "CT (per-organ doses from Excel)"
_CATEGORY_RADXA = "Radiography / Fluoroscopy / Angiography (DAP & beam quality)"
_CATEGORY_MG = "Mammography"
_CATEGORY_ED = "Pre-calculated effective dose"
_CATEGORY_OTHER = "Other (unrecognised)"

_EXACT_COLUMN_TO_CATEGORY: Dict[str, str] = {
    # Common
    "A#": _CATEGORY_COMMON,
    "Patient MRN": _CATEGORY_COMMON,
    "Gender": _CATEGORY_COMMON,
    "Sex": _CATEGORY_COMMON,
    "Patient Weight kg": _CATEGORY_COMMON,
    "Population": _CATEGORY_COMMON,
    "Age at Exam yrs": _CATEGORY_COMMON,
    "Date Performed": _CATEGORY_COMMON,
    "DatePerformed": _CATEGORY_COMMON,
    "Description": _CATEGORY_COMMON,
    "Protocol": _CATEGORY_COMMON,
    "ExamType": _CATEGORY_COMMON,
    "Modality": _CATEGORY_COMMON,
    "Examination Modalities": _CATEGORY_COMMON,
    "Acquisition Modality": _CATEGORY_COMMON,
    # CT
    "Remainder ICRP103 mSv": _CATEGORY_CT,
    # (per-organ "(CT) mSv" columns are matched by suffix below)
    # Radiography / Fluoroscopy / Angiography
    "DAP (Total) Gy-cm2": _CATEGORY_RADXA,
    "Fluoro DAP (Total) Gy-cm2": _CATEGORY_RADXA,
    "Fluoro DAP (A/B) Gy-cm2": _CATEGORY_RADXA,
    "kVp kV": _CATEGORY_RADXA,
    "Xray Filter Max Thickness (DX)": _CATEGORY_RADXA,
    "kVp": _CATEGORY_RADXA,
    "kV": _CATEGORY_RADXA,
    # Mammography
    "Mean Glandular Dose mGy": _CATEGORY_MG,
    # Pre-calculated effective dose
    "Total Effective Dose mSv": _CATEGORY_ED,
    "ICRP 103 mSv": _CATEGORY_ED,
}

# Order in which categories are emitted by ``categorize`` (categories absent
# from the input simply do not appear in the result).
_CATEGORY_ORDER: Tuple[str, ...] = (
    _CATEGORY_COMMON,
    _CATEGORY_CT,
    _CATEGORY_RADXA,
    _CATEGORY_MG,
    _CATEGORY_ED,
    _CATEGORY_OTHER,
)


def _classify_one(col: str) -> str:
    """Return the category label for a single column name."""
    cat = _EXACT_COLUMN_TO_CATEGORY.get(col)
    if cat is not None:
        return cat
    # Per-organ CT equivalent doses -- e.g. "Liver (CT) mSv".
    if col.endswith(" (CT) mSv"):
        return _CATEGORY_CT
    # NM / Total per-organ columns aren't loaded by default but a user
    # might still list them; keep them under CT for visual grouping.
    if col.endswith(" (NM) mSv") or col.endswith(" (Total) mSv"):
        return _CATEGORY_CT
    return _CATEGORY_OTHER


def categorize(columns: Iterable[str]) -> "OrderedDict[str, List[str]]":
    """Group ``columns`` by simulation purpose.

    Parameters
    ----------
    columns : iterable of str
        Column names to classify (typically the missing-column list).

    Returns
    -------
    OrderedDict[str, list[str]]
        Maps each non-empty category label to the (alphabetically sorted)
        list of input column names that fall in it.  Categories with zero
        members are omitted; the surviving categories are emitted in the
        canonical display order.
    """
    buckets: Dict[str, List[str]] = {}
    for col in columns:
        cat = _classify_one(col)
        buckets.setdefault(cat, []).append(col)
    out: "OrderedDict[str, List[str]]" = OrderedDict()
    for cat in _CATEGORY_ORDER:
        cols = buckets.get(cat)
        if cols:
            out[cat] = sorted(cols)
    return out


# Explicit consequences for columns whose absence is a model choice rather
# than a calculated estimate.  Rendered by ``format_consequences`` in both
# the stdout note and the GUI window.
COLUMN_CONSEQUENCES: Dict[str, str] = {
    "Population": "no per-row population: every row uses the population chosen at start-up",
    "Gender": "sex unknown for every row: the mean of the male and female risk models is used",
    "Sex": "sex unknown for every row: the mean of the male and female risk models is used",
}


def format_consequences(
    missing_per_file: Mapping[str, Iterable[str]],
    startup_population: str | None = None,
) -> str:
    """One line per distinct missing column listed in ``COLUMN_CONSEQUENCES``.

    ``Population`` renders as "every row uses the start-up population
    (<name>)" when ``startup_population`` is given, else the generic
    sentence.  Columns without an entry (e.g. CT organ doses) are skipped,
    so a file missing only those returns an empty string.
    """
    missing_cols = {c for cols in missing_per_file.values() for c in cols}
    lines: List[str] = []
    for col in sorted(missing_cols):
        text = COLUMN_CONSEQUENCES.get(col)
        if text is None:
            continue
        if col == "Population" and startup_population:
            text = f"no per-row population: every row uses the start-up population ({startup_population})"
        lines.append(f"{col}: {text}")
    return "\n".join(lines)


_CATEGORY_SHORT: Dict[str, str] = {
    _CATEGORY_COMMON: "Common",
    _CATEGORY_CT:     "CT",
    _CATEGORY_RADXA:  "Rad/Fluoro",
    _CATEGORY_MG:     "Mammography",
    _CATEGORY_ED:     "Eff.Dose",
    _CATEGORY_OTHER:  "Other",
}

# Maximum number of column names printed inline before switching to a count.
_MAX_INLINE_COLS = 4


def format_per_file_summary(
    missing_per_file: Mapping[str, Iterable[str]],
    *,
    indent: str = "  ",
    bullet: str = "- ",
) -> str:
    """Render a compact per-file missing-columns summary as plain text.

    One line per file; columns are grouped by category and truncated when
    there are many, so the output stays readable regardless of file count.

    Layout (single file or all files identical)
    -------------------------------------------
    ::

        Missing columns (1 file):
          data.xlsx  →  Common: Gender, Population  |  CT: Brain (CT) mSv (+23 more)

    Layout (multiple files with different missing sets)
    ---------------------------------------------------
    ::

        Missing columns (2 files):
          file_a.xlsx  →  Common: Gender  |  CT: Brain (CT) mSv, Liver (CT) mSv
          file_b.xlsx  →  CT: Brain (CT) mSv
    """
    if not missing_per_file:
        return ""

    def _file_line(fp: str, cols: Iterable[str]) -> str:
        col_list = sorted(cols)
        grouped = categorize(col_list)
        parts: List[str] = []
        for cat, ccols in grouped.items():
            short = _CATEGORY_SHORT.get(cat, cat)
            if len(ccols) <= _MAX_INLINE_COLS:
                parts.append(f"{short}: {', '.join(ccols)}")
            else:
                shown = ", ".join(ccols[:_MAX_INLINE_COLS])
                parts.append(f"{short}: {shown} (+{len(ccols) - _MAX_INLINE_COLS} more)")
        detail = "  |  ".join(parts) if parts else "(none)"
        return f"{indent}{bullet}{os.path.basename(fp)}  →  {detail}"

    n = len(missing_per_file)
    lines: List[str] = [f"Missing columns ({n} file{'s' if n != 1 else ''}):"]
    for fp, cols in missing_per_file.items():
        lines.append(_file_line(fp, cols))
    return "\n".join(lines)
