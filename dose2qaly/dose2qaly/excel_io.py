"""Excel-loading helpers for the Monte Carlo simulation.

``preprocess.load_and_preprocess`` reads input files through this
module.

The loader uses a column whitelist (``usecols`` push-down) so the
Calamine parser only materialises the columns the engine actually
reads -- typically a 2-4× speed-up on the first/uncached read of
large XLSX files.  Missing columns are reported per-file through the
``on_progress("missing_columns", {file: [cols], ...})`` channel.

Public API
----------
* ``DEFAULT_NEEDED_COLS``    Column whitelist used by the engine.
* ``check_excel_headers``    Fast header-only column pre-check (streaming row 0).
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional

import pandas as pd


# Excel column push-down
# Restricting the Excel parser to these columns speeds it up.
# Missing columns are tolerated (row.get returns None) and reported via on_progress.
# Sources: exam_doses.compute_exam_organ_doses, ct_doses._EXCEL_CT_COLUMN_TO_ORGAN, radiography_doses.
# Column equivalence groups
# Columns where any one member being present satisfies the requirement.
# Used by check_excel_headers and config._scan_files_for_missing_cols to
# suppress spurious "missing column" warnings when an equivalent alternative
# is present in the file.
#
# "A#" (examination) and "Patient MRN" (person) are distinct keys and are
# each reported on their own when absent.
#
# "Sex" is an accepted alternative header for the "Gender" demographic
# column (some exports label it that way); either satisfies the requirement.
_SEX_COLS = frozenset({"Gender", "Sex"})
# Modality can arrive under any of these three spellings; the engine prefers
# Acquisition Modality, then Examination Modalities, then Modality.
_MODALITY_COL_CANDIDATES = ("Acquisition Modality", "Examination Modalities", "Modality")
_MODALITY_COLS = frozenset(_MODALITY_COL_CANDIDATES)

# Every member of an equivalence group maps to the whole group, so a file
# that has any one member is not reported as missing the others.
COLUMN_ALTERNATIVES: dict[str, frozenset[str]] = {
    **{c: _SEX_COLS for c in _SEX_COLS},
    **{c: _MODALITY_COLS for c in _MODALITY_COLS},
}

# Columns that drive optional sub-study features; loaded if present but
# never reported as missing because the simulation does not depend on them.
# "Population" is optional too: rows without it take the population chosen
# at start-up (main._prepare_cohort), so its absence is not worth a note.
_OPTIONAL_NEVER_REPORT: frozenset[str] = frozenset({
    "Device",
    "Equipment",
    "Population",
})

# Mapping of a column to the set of canonical modality strings that must
# be present in the file for the column to be reported as missing.
# Common columns (patient ID, demographics, age, date, protocol) have no
# entry and are always reported if absent.
_COLUMN_MODALITY_REQUIREMENTS: Dict[str, frozenset[str]] = {
    # CT organ-dose columns
    "Scan Regions": frozenset({"ct"}),
    "Remainder ICRP103 mSv": frozenset({"ct"}),
    # (per-organ "(CT) mSv" / "(NM) mSv" / "(Total) mSv" columns are matched
    # by suffix in _filter_missing_by_modality.)
    # Radiography / Fluoroscopy / Angiography
    "DAP (Total) Gy-cm2": frozenset({"radiography", "angiography", "cbct"}),
    "Fluoro DAP (Total) Gy-cm2": frozenset({"angiography"}),
    "Fluoro DAP (A/B) Gy-cm2": frozenset({"angiography"}),
    "kVp kV": frozenset({"radiography", "angiography", "cbct"}),
    "kVp": frozenset({"radiography", "angiography", "cbct"}),
    "kV": frozenset({"radiography", "angiography", "cbct"}),
    # Beam filtration: reported only if rad/fluoro/angio is present
    "Xray Filter Max Thickness (DX)": frozenset({"radiography", "angiography"}),
    # CBCT / radiography body-part tags
    "Body Part Examined (CBCT)": frozenset({"cbct"}),
    "Body Part Examined (DX)": frozenset({"radiography"}),
    # Mammography
    "Mean Glandular Dose mGy": frozenset({"mg"}),
    # Pre-calculated effective dose (optional override, but report if any
    # imaging modality is present).
    "Total Effective Dose mSv": frozenset({"ct", "radiography", "angiography", "cbct", "mg"}),
    "ICRP 103 mSv": frozenset({"ct", "radiography", "angiography", "cbct", "mg"}),
}


def _has_alternative(col: str, present_cols: set[str]) -> bool:
    """Return True if any equivalent alternative for *col* is in *present_cols*."""
    alts = COLUMN_ALTERNATIVES.get(col)
    if alts is None:
        return False
    return bool(alts & present_cols)


def _normalize_modality_for_check(raw: object) -> frozenset[str]:
    """Return canonical modality tokens for a raw Excel modality cell.

    Mirrors the normalization in ``exam_doses.compute_exam_organ_doses``.
    A single raw value can yield multiple tokens (e.g. "CT/PT" -> {"ct"}).
    """
    s = str(raw).strip().upper()
    if not s:
        return frozenset()
    out: set[str] = set()
    # Multi-value strings such as "CT/PT" are split and each part tokenised.
    for part in s.replace("/", " ").replace("\\", " ").split():
        p = part.strip()
        if p in ("CR", "DX", "DR"):
            out.add("radiography")
        elif p in ("XA", "RF") or "ANGIO" in p:
            out.add("angiography")
        elif "CBCT" in p:
            out.add("cbct")
        elif "CT" in p:
            out.add("ct")
        elif p == "MG" or "MAMMO" in p:
            out.add("mg")
    return frozenset(out)


def _sample_file_modalities(
    fp: str, header_cols: list[str], max_rows: int = 500
) -> frozenset[str]:
    """Collect canonical modality tokens from the first ``max_rows`` data rows.

    Only the modality column is read, so the cost is small even for
    multi-million-row workbooks.  Returns an empty frozenset when the file has
    no recognisable modality column or cannot be sampled.
    """
    header_set = set(str(c) for c in header_cols)
    mod_col = next((c for c in _MODALITY_COL_CANDIDATES if c in header_set), None)
    if mod_col is None:
        return frozenset()
    try:
        if fp.lower().endswith((".xlsx", ".xlsm")):
            # Streaming zip/XML read of one column over the first rows only;
            # calamine/openpyxl would parse the whole sheet (minutes on a
            # 300k-row export) just to yield these few hundred cells.
            try:
                src_idx = [str(c) for c in header_cols].index(mod_col)
            except ValueError:
                return frozenset()
            rows = _read_xlsx_rows_fast(fp, max_rows=max_rows + 1, only_cols={src_idx})
            if not rows:
                return frozenset()
            vals = [r[src_idx] for r in rows[1:] if src_idx in r]
        else:
            df = pd.read_excel(fp, usecols=[mod_col], nrows=max_rows)
            if df.empty:
                return frozenset()
            vals = df[mod_col].dropna()
        out: set[str] = set()
        for v in vals:
            out.update(_normalize_modality_for_check(v))
        return frozenset(out)
    except Exception:
        return frozenset()


def _filter_missing_by_modality(
    miss: list[str], modalities: frozenset[str]
) -> list[str]:
    """Drop modality-specific columns that are not needed for a file's content.

    Also drops optional sub-study columns.
    """
    out: list[str] = []
    for c in miss:
        if c in _OPTIONAL_NEVER_REPORT:
            continue
        req = _COLUMN_MODALITY_REQUIREMENTS.get(c)
        if req is None and (
            c.endswith(" (CT) mSv")
            or c.endswith(" (NM) mSv")
            or c.endswith(" (Total) mSv")
        ):
            req = frozenset({"ct"})
        if req is None:
            out.append(c)
            continue
        if req & modalities:
            out.append(c)
    return out


# Single source of truth for the column whitelist.  NEEDED_COLS_ORDERED
# keeps the fixed display order config.py's "Input File Columns" tab needs;
# DEFAULT_NEEDED_COLS derives from it
# for the engine's set-membership checks.  config.py imports the ordered
# tuple instead of re-declaring its own copy.
NEEDED_COLS_ORDERED: tuple = (
    # Patient identifiers / demographics
    "A#", "Patient MRN", "Gender", "Sex", "Patient Weight kg", "Population",
    # Exam metadata
    # "Date Performed" / "Device" / "Equipment" are for the device-renewal sub-study;
    # optional — files lacking them still run the MC engine unchanged.
    "Age at Exam yrs",
    "Date Performed", "Device", "Equipment",
    # Exam descriptions
    # ("ExamType" is only used as a fallback for "Protocol" -- not loaded
    # by default since the input always populates "Protocol".)
    "Description",
    "Protocol",
    # CT region classification is driven entirely by this field
    # (region_resolver.resolve_ct_region_from_scan_regions).
    "Scan Regions",
    # Radiography (CR/DX) and CBCT region classification is driven
    # primarily by these DICOM tags (region_resolver.resolve_radiography_region /
    # resolve_cbct_region).
    "Body Part Examined (DX)", "Body Part Examined (CBCT)",
    # Modality
    # Engine chain: Examination Modalities > Modality > Acquisition Modality.
    # Only "Acquisition Modality" is loaded; the others are pure fallbacks.
    "Acquisition Modality",
    # Radiography / XA / Fluoroscopy
    "DAP (Total) Gy-cm2",
    # ("Fluoro DAP (Total) Gy-cm2" is a fallback for missing DAP; omitted by default.)
    # Radiography beam quality: the Radimetrics export carries kVp as
    # "kVp kV" and the added filter plates as the DICOM multi-value set
    # "Xray Filter Max Thickness (DX)" (engine: radiography_doses._beam_quality_from_row).
    "kVp kV",
    "Xray Filter Max Thickness (DX)",
    # Mammography
    "Mean Glandular Dose mGy",
    # Pre-calculated effective doses
    # Engine prefers "ICRP 103 mSv".
    "ICRP 103 mSv",
    # Per-organ CT equivalent doses (mSv)
    # Source: _EXCEL_CT_COLUMN_TO_ORGAN in ct_doses.py.  Only "(CT) mSv" columns are used.
    "Adrenals (CT) mSv", "Brain (CT) mSv", "Breasts (CT) mSv",
    "Colon (CT) mSv", "Esophagus (CT) mSv", "Eye Lenses (CT) mSv",
    "Gall Bladder (CT) mSv", "Heart (CT) mSv", "Kidneys (CT) mSv",
    "Liver (CT) mSv", "Lungs (CT) mSv", "Muscle (CT) mSv",
    "Ovaries (CT) mSv", "Pancreas (CT) mSv", "Red Marrow (CT) mSv",
    "Salivary Glands (CT) mSv", "Skeleton (CT) mSv", "Skin (CT) mSv",
    "Small Intestine (CT) mSv", "Spleen (CT) mSv", "Stomach (CT) mSv",
    "Testicles (CT) mSv", "Thymus (CT) mSv", "Thyroid (CT) mSv",
    "Upper Large Intestine (CT) mSv", "Urinary Bladder (CT) mSv",
    "Uterus (CT) mSv", "Fetus (CT) mSv",
    # Aggregate "remainder" tissues (ICRP 103) -- used as the CT
    # remainder organ dose when present.
    "Remainder ICRP103 mSv",
)
DEFAULT_NEEDED_COLS = frozenset(NEEDED_COLS_ORDERED)

_CT_ORGAN_COLS: tuple = tuple(c for c in NEEDED_COLS_ORDERED if c.endswith("(CT) mSv"))

# Reader-facing guide to the whitelist, shown in the "Input File Columns" tab of
# the simulation setup.  Each entry is (group, columns, what the engine does
# with them); the groups follow the modality routing in
# exam_doses.compute_exam_organ_doses.  Every whitelist column appears in
# exactly one entry (checked below), so the guide cannot drift from the list.
COLUMN_GUIDE: tuple = (
    ("Every modality", ("Acquisition Modality",),
     "CT / CR / DX / MG / XA / RF / CBCT - decides which dose model is used."),
    ("Every modality", ("Age at Exam yrs",),
     "Age at exposure: risk model, latency and remaining lifetime."),
    ("Every modality", ("Sex", "Gender"),
     "Patient sex: sex-specific risk model and organs. Either header name."),
    ("Every modality", ("A#",),
     "Examination identifier (accession number)."),
    ("Every modality", ("Patient MRN",),
     "Per-person key: repeat examinations and competing-mortality strata."),
    ("Every modality", ("Date Performed",),
     "Calendar year of the examination: per-year runs and annual results."),
    ("Every modality", ("Description", "Protocol"),
     "Free text; imaging region when the DICOM region tag is blank, and "
     "identifies dental and scoliosis examinations."),
    ("Every modality", ("Patient Weight kg",),
     "Weight adjustment (Golikov and Druzhinina) for adult chest and "
     "abdomen CT. Optional."),
    ("Every modality", ("Population",),
     "Per-row reference population (Euro-American / Asian). Optional; "
     "otherwise the start-up choice."),
    ("Every modality", ("Device", "Equipment"),
     "Device-level breakdown only. Optional."),
    ("CT", ("Scan Regions",),
     "Imaging region for CT (sole source)."),
    ("CT", _CT_ORGAN_COLS,
     "Organ equivalent doses in mSv from the dose-tracking system - the "
     "input to the cancer risk model (%d columns)." % len(_CT_ORGAN_COLS)),
    ("CT", ("Remainder ICRP103 mSv",),
     "Dose to the ICRP 103 remainder tissues."),
    ("CT", ("ICRP 103 mSv",),
     "Effective dose, for reporting and collective dose; not used for "
     "risk. Other modalities derive effective dose from DAP or MGD."),
    ("Radiography (CR / DX)", ("DAP (Total) Gy-cm2",),
     "Dose-area product; converted to organ doses with projection-specific "
     "coefficients."),
    ("Radiography (CR / DX)", ("kVp kV", "Xray Filter Max Thickness (DX)"),
     "Beam quality for the conversion coefficients. An exam without kVp "
     "has no dose data; missing filtration takes the kVp default."),
    ("Radiography (CR / DX)", ("Body Part Examined (DX)",),
     "Imaging region for radiography."),
    ("Mammography (MG)", ("Mean Glandular Dose mGy",),
     "The only dose quantity: breast dose and effective dose are derived "
     "from it."),
    ("Dental CBCT", ("DAP (Total) Gy-cm2", "kVp kV"),
     "DAP converted with kVp-specific dental coefficients. Only dental "
     "CBCT has a dose model."),
    ("Dental CBCT", ("Body Part Examined (CBCT)",),
     "Imaging region for CBCT."),
    ("Angiography / fluoroscopy (XA / RF) - placeholder",
     ("DAP (Total) Gy-cm2", "kVp kV", "Xray Filter Max Thickness (DX)"),
     "Placeholder only: a DAP-based model exists in the code but was not "
     "used in the study."),
)

_guide_cols = {c for _, cols, _ in COLUMN_GUIDE for c in cols}
assert _guide_cols == DEFAULT_NEEDED_COLS, (
    "COLUMN_GUIDE out of sync with NEEDED_COLS_ORDERED: "
    f"missing {DEFAULT_NEEDED_COLS - _guide_cols}, extra {_guide_cols - DEFAULT_NEEDED_COLS}"
)
del _guide_cols


def _xlsx_col_to_idx(ref: str) -> int:
    """Map an A1-style cell reference to a 0-based column index ('C7' -> 2)."""
    idx = 0
    for ch in ref:
        if ch.isalpha():
            idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
        else:
            break
    return idx - 1 if idx > 0 else 0


def _xlsx_first_sheet_path(zf) -> str:
    """Resolve the zip path of the workbook's first worksheet.

    Walks ``xl/workbook.xml`` (first ``<sheet>`` element) and the workbook
    relationships to find the real target, falling back to ``sheet1.xml`` or
    the first ``xl/worksheets/*.xml`` member when the rels lookup fails.
    """
    from xml.etree.ElementTree import fromstring
    try:
        wb = fromstring(zf.read("xl/workbook.xml"))
        rid = None
        for el in wb.iter():
            if el.tag.split("}")[-1] == "sheet":
                for k, v in el.attrib.items():
                    if k.split("}")[-1] == "id":
                        rid = v
                        break
                break
        if rid:
            rels = fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            for rel in rels:
                if rel.get("Id") == rid:
                    target = rel.get("Target") or ""
                    if target:
                        if target.startswith("/"):
                            return target.lstrip("/")
                        if not target.startswith("xl/"):
                            target = "xl/" + target
                        return target
    except Exception:
        pass
    names = set(zf.namelist())
    if "xl/worksheets/sheet1.xml" in names:
        return "xl/worksheets/sheet1.xml"
    for n in zf.namelist():
        if n.startswith("xl/worksheets/") and n.endswith(".xml"):
            return n
    return "xl/worksheets/sheet1.xml"


def _read_xlsx_rows_fast(
    fp: str, max_rows: int = 1, only_cols: Optional[set] = None
) -> Optional[List[Dict[int, str]]]:
    """Stream the first ``max_rows`` rows of an .xlsx/.xlsm sheet, cheaply.

    ``openpyxl.load_workbook`` -- even with ``read_only=True`` -- eagerly
    parses the **entire** shared-strings table on open (``reader.read_strings``),
    and ``python-calamine`` parses the entire sheet before yielding the first
    row; on a multi-million-row workbook either is hundreds of MB of XML and
    takes minutes (what looks like a "freeze" after pressing Run).  This
    reader instead opens the .xlsx as a zip, stream-parses only the first
    ``max_rows`` ``<row>`` elements of the first worksheet, and resolves only
    the shared-string indices those cells actually reference -- stopping as
    soon as the largest needed index is seen.

    ``only_cols`` restricts the *data* rows (row 2 onwards) to the given
    0-based column indices; the header row is always read in full.  Returns
    one ``{col_idx: text}`` dict per row (row 0 = header), or ``None`` on
    any failure so the caller can fall back to the slower readers.
    """
    import zipfile
    from xml.etree.ElementTree import iterparse
    try:
        with zipfile.ZipFile(fp) as zf:
            sheet_path = _xlsx_first_sheet_path(zf)

            # ── Stream the rows' cells: [row] -> {col_idx: (type, value, inline)}
            rows: List[Dict[int, tuple]] = []
            cells: Dict[int, tuple] = {}
            with zf.open(sheet_path) as fh:
                in_row = False
                row_seen = 0
                for event, elem in iterparse(fh, events=("start", "end")):
                    tag = elem.tag.split("}")[-1]
                    if event == "start" and tag == "row":
                        row_seen += 1
                        in_row = True
                        cells = {}
                    elif event == "end" and tag == "c" and in_row:
                        ref = elem.get("r") or ""
                        col = _xlsx_col_to_idx(ref) if ref else len(cells)
                        if row_seen > 1 and only_cols is not None and col not in only_cols:
                            elem.clear()
                            continue
                        ctype = elem.get("t")
                        v_text = None
                        inline = None
                        for child in elem:
                            ctag = child.tag.split("}")[-1]
                            if ctag == "v":
                                v_text = child.text
                            elif ctag == "is":
                                inline = "".join(
                                    t.text or "" for t in child.iter()
                                    if t.tag.split("}")[-1] == "t"
                                )
                        cells[col] = (ctype, v_text, inline)
                        elem.clear()
                    elif event == "end" and tag == "row" and in_row:
                        elem.clear()
                        rows.append(cells)
                        in_row = False
                        if row_seen >= max_rows:
                            break
            if not rows or not rows[0]:
                return []

            # ── Resolve only the shared-string indices the rows use.
            needed = {
                int(v) for row in rows for (t, v, _i) in row.values()
                if t == "s" and v is not None and v.lstrip("-").isdigit()
            }
            shared: Dict[int, str] = {}
            if needed:
                max_idx = max(needed)
                try:
                    with zf.open("xl/sharedStrings.xml") as sh:
                        si = -1
                        for _ev, el in iterparse(sh, events=("end",)):
                            if el.tag.split("}")[-1] == "si":
                                si += 1
                                if si in needed:
                                    shared[si] = "".join(
                                        t.text or "" for t in el.iter()
                                        if t.tag.split("}")[-1] == "t"
                                    )
                                el.clear()
                                if si >= max_idx:
                                    break
                except KeyError:
                    pass  # no shared strings part (all inline / numeric)

            def _text(cell) -> str:
                t, v, inline = cell
                if t == "s" and v is not None:
                    return shared.get(int(v), "") if v.lstrip("-").isdigit() else ""
                if t == "inlineStr":
                    return inline or ""
                return "" if v is None else str(v)

            return [{col: _text(c) for col, c in row.items()} for row in rows]
    except Exception:
        return None


def _read_xlsx_header_fast(fp: str) -> Optional[List[str]]:
    """Stream just the first row of an .xlsx/.xlsm header, in O(header width).

    Thin wrapper over :func:`_read_xlsx_rows_fast`; returns the header as an
    ordered list (gaps filled with ``""``), or ``None`` on failure.
    """
    rows = _read_xlsx_rows_fast(fp, max_rows=1)
    if rows is None:
        return None
    if not rows:
        return []
    header = rows[0]
    return [header.get(col, "") for col in range(max(header) + 1)]


def _read_header_columns(fp: str) -> Optional[List[str]]:
    """Return the column names from row 0 only, as cheaply as possible.

    IMPORTANT: ``pd.read_excel(fp, engine='calamine', nrows=0)`` is *not*
    a header-only read.  The calamine engine parses the **entire sheet**
    into memory and pandas only slices to ``nrows`` afterwards, so on a
    multi-million-row workbook the "just check the header" step ends up
    reading the whole file (tens of seconds to minutes).

    This helper instead streams only the first row:

    1. ``_read_xlsx_header_fast`` — zip + streaming XML parse of the first
       row only, for .xlsx/.xlsm.  Unlike ``openpyxl.load_workbook`` (which
       eagerly reads the *entire* shared-strings table on open), this is
       O(header width) and completes in milliseconds on multi-million-row
       workbooks.
    2. ``python-calamine`` streaming ``iter_rows`` — take the first row
       and stop (used for .xls and as a fallback).
    3. ``pd.read_excel(nrows=0)`` — correct but full-parse last resort.

    Returns the list of header names, or ``None`` if every reader failed.
    """
    # 1. Streaming zip+XML header read (the cheap path for modern .xlsx).
    if fp.lower().endswith((".xlsx", ".xlsm")):
        cols = _read_xlsx_header_fast(fp)
        if cols is not None:
            return cols

    # 2. python-calamine streaming: first row only.
    try:
        from python_calamine import CalamineWorkbook  # lazy import
        wb = CalamineWorkbook.from_path(fp)
        try:
            sheet = wb.get_sheet_by_index(0)
            for row in sheet.iter_rows():
                return [("" if c is None else str(c)) for c in row]
            return []
        finally:
            try:
                wb.close()
            except Exception:
                pass
    except Exception:
        pass

    # 3. Last resort: full parse (still correct, just not cheap).
    try:
        return list(pd.read_excel(fp, nrows=0).columns)
    except Exception:
        return None


def check_excel_headers(
    file_list,
    needed_cols=None,
    on_progress=None,
    context_aware: bool = True,
) -> dict[str, list[str]]:
    """Read the header row of each file and return missing required columns.

    Streams just the first row of each workbook via
    :func:`_read_header_columns` (openpyxl read-only / calamine streaming),
    so the header check is O(1) in file size.  When ``context_aware=True``,
    the first 500 data rows of the modality column are also sampled to
    suppress modality-specific columns that are irrelevant to the file's
    actual content (e.g. CT organ-dose columns for a radiography-only
    export).  This keeps the pre-flight warning accurate without loading
    the whole workbook.

    Parameters
    ----------
    file_list : list of str
        Paths to Excel files to inspect.
    needed_cols : iterable of str, optional
        Required column names.  Defaults to :data:`DEFAULT_NEEDED_COLS`.
    on_progress : callable, optional
        If provided, called as ``on_progress("header_check", missing_per_file)``
        after the check finishes.
    context_aware : bool, default True
        If True, filter the missing list per file by the modalities present
        in the first 500 rows.  Set to False to get a flat list.

    Returns
    -------
    dict[str, list[str]]
        Mapping of file path → sorted list of missing column names.  Empty
        dict means all required columns are present in every file.
    """
    if needed_cols is None:
        needed_cols = DEFAULT_NEEDED_COLS
    needed_set = frozenset(needed_cols)
    missing_per_file: dict[str, list[str]] = {}

    for fp in file_list:
        cols = _read_header_columns(fp)
        if cols is None:
            missing_per_file[fp] = ["<could not read headers>"]
            continue
        present = set(cols)
        miss = sorted(c for c in (needed_set - present) if not _has_alternative(c, present))
        if context_aware:
            modalities = _sample_file_modalities(fp, cols)
            miss = _filter_missing_by_modality(miss, modalities)
        if miss:
            missing_per_file[fp] = miss

    if missing_per_file and on_progress:
        try:
            on_progress("header_check", missing_per_file)
        except Exception:
            pass

    return missing_per_file


def _estimate_excel_rows(fp: str) -> int:
    """Fast data-row estimate for .xlsx/.xlsm (excludes the header row).

    Avoids ``openpyxl.load_workbook`` entirely: that call eagerly parses the
    full shared-strings table on open (minutes on multi-million-row files).
    Instead this streams the first worksheet's XML and (a) returns the row
    span from the ``<dimension>`` element the moment it appears -- present in
    virtually every writer's output, so this is the O(1) common case -- or
    (b) falls back to counting ``<row>`` start tags, which still reads only
    the sheet XML and never touches the shared-strings table.  Returns 0 on
    any failure (the value is only used for progress percentages).
    """
    if not fp.lower().endswith((".xlsx", ".xlsm")):
        return 0
    import zipfile
    from xml.etree.ElementTree import iterparse
    try:
        with zipfile.ZipFile(fp) as zf:
            sheet_path = _xlsx_first_sheet_path(zf)
            with zf.open(sheet_path) as fh:
                in_sheetdata = False
                row_count = 0
                for ev, elem in iterparse(fh, events=("start", "end")):
                    tag = elem.tag.split("}")[-1]
                    if ev == "start" and tag == "dimension":
                        ref = elem.get("ref") or ""
                        end = ref.split(":")[-1] if ref else ""
                        digits = "".join(ch for ch in end if ch.isdigit())
                        if digits:
                            return max(0, int(digits) - 1)
                    elif ev == "start" and tag == "sheetData":
                        in_sheetdata = True
                    elif ev == "start" and tag == "row" and in_sheetdata:
                        row_count += 1
                    elif ev == "end" and tag in ("c", "row"):
                        elem.clear()
                return max(0, row_count - 1)
    except Exception:
        return 0


def _estimate_file_rows(fp: str) -> int:
    return _estimate_excel_rows(fp)



def _read_excel_streaming(
    fp: str,
    needed_cols=None,
    on_progress=None,
    total_rows: int = 0,
    rows_processed_ref=None,
    last_pct_ref=None,
    progress_lock=None,
):
    """Stream-read an .xlsx/.xlsm file with python-calamine, emitting 1% progress.

    Returns a pandas DataFrame.  Falls back to ``pd.read_excel`` if the
    streaming reader fails.  Progress is reported only when ``total_rows > 0``
    and the provided thread-safe references are supplied.
    """
    import pandas as pd
    # Tolerant column push-down: a name list
    # makes pandas raise when the sheet is missing any of the 52 default
    # columns, while the predicate simply keeps whatever is present.  Files
    # legitimately omit columns (a radiography-only export has no "(CT) mSv"
    # columns at all), so the strict form turned a fallback into a hard error.
    _needed_set = frozenset(needed_cols) if needed_cols else None
    _usecols = (lambda c: c in _needed_set) if _needed_set else None
    try:
        from python_calamine import CalamineWorkbook
    except Exception:
        return pd.read_excel(fp, engine="calamine", usecols=_usecols)

    try:
        wb = CalamineWorkbook.from_path(fp)
        sheet = wb.get_sheet_by_index(0)
        rows_iter = sheet.iter_rows()
        header = next(rows_iter, None)
        if header is None:
            wb.close()
            return pd.DataFrame()
        header_names = [str(c) if c is not None else "" for c in header]
        needed_set = frozenset(needed_cols) if needed_cols else None

        # Determine which columns to keep
        col_indices = []
        col_names = []
        for idx, name in enumerate(header_names):
            if needed_set is None or name in needed_set:
                col_indices.append(idx)
                col_names.append(name)

        columns = {name: [] for name in col_names}
        for row in rows_iter:
            for out_idx, src_idx in enumerate(col_indices):
                val = row[src_idx] if src_idx < len(row) else None
                columns[col_names[out_idx]].append(val)
            # ``on_progress`` is optional: headless callers (scripts, tests,
            # year_batch's internal runs) pass None.  Emitting unconditionally
            # raised TypeError on the first row, and because the whole loop is
            # wrapped in a bare ``except`` the failure was invisible -- every
            # such run silently fell back to the slower, less tolerant
            # ``pd.read_excel`` below instead of streaming.
            _report = on_progress is not None and total_rows > 0
            if progress_lock is not None:
                with progress_lock:
                    rows_processed_ref[0] += 1
                    if _report:
                        pct = min(100, int(rows_processed_ref[0] * 100 / total_rows))
                        if pct != last_pct_ref[0]:
                            last_pct_ref[0] = pct
                            on_progress("init", rows_processed_ref[0], total_rows,
                                        f"Reading Excel: {rows_processed_ref[0]:,} / {total_rows:,}")
            else:
                rows_processed_ref[0] += 1
                if _report:
                    pct = min(100, int(rows_processed_ref[0] * 100 / total_rows))
                    if pct != last_pct_ref[0]:
                        last_pct_ref[0] = pct
                        on_progress("init", rows_processed_ref[0], total_rows,
                                    f"Reading Excel: {rows_processed_ref[0]:,} / {total_rows:,}")
        wb.close()
        return pd.DataFrame(columns)
    except Exception:
        try:
            wb.close()
        except Exception:
            pass
        return pd.read_excel(fp, engine="calamine", usecols=_usecols)


def parse_exam_dates(values) -> "pd.Series":
    """Parse a "Date Performed" column into datetimes, whatever it holds.

    ``Date Performed`` reaches the engine in whatever form the export used,
    and a multi-file run routinely mixes several encodings in one column:
    datetime/date objects, ISO strings (``"2017-10-05"``), locale strings
    (``"5.10.2017"``) and **bare Excel serial day numbers** (``43013``).

    The serial case has to be split out *before* ``to_datetime``, not after.
    ``pd.to_datetime(43013.0)`` does not fail — it reads the number as
    nanoseconds since the epoch and returns ``1970-01-01``.  A post-hoc
    ``isna()`` guard therefore never fires, and every serial-dated row is
    silently filed under **1970**: one calendar-year bucket swallowing a whole
    file, which inflates each patient's Mataac CED (all their years pooled
    into one) and mislabels ``year_stats`` and "Auto-Sort by Year" to match.

    The ``20000 < n < 80000`` window (~1954-2119) keeps a year-only string
    such as ``"2017"`` out of the serial branch.

    The second trap is ``errors="coerce"`` on a heterogeneous string column:
    pandas infers ONE format from the first non-null element and quietly
    coerces every value that does not match it to ``NaT``.  A column whose
    first row reads ``"2017"`` turns every real ``"2021-04-04"`` below it into
    ``NaT`` — so in a multi-file run one file's encoding can wipe out the
    dates of all the others.  Rows that fail the fast vectorised pass are
    therefore re-parsed element-wise (``format="mixed"``), which costs
    dateutil only on the rows that actually need it.

    Day/month order for ambiguous strings ("5.10.2017" vs "10/05/2017") is
    left at pandas' default (month first).  Only the calendar year is used by
    the Mataac stratum and by ``year_stats``, and both readings agree on it.

    Returns a datetime64 Series, ``NaT`` where nothing could be parsed.
    """
    s = pd.Series(list(values), dtype=object)
    num = pd.to_numeric(s, errors="coerce")
    is_serial = num.notna() & (num > 20000) & (num < 80000)
    out = pd.to_datetime(s.where(~is_serial), errors="coerce")
    retry = out.isna() & s.notna() & ~is_serial
    if bool(retry.any()):
        out = out.copy()
        out[retry] = pd.to_datetime(s[retry], errors="coerce", format="mixed")
    if bool(is_serial.any()):
        out = out.copy()
        out[is_serial] = pd.to_datetime(
            num[is_serial], unit="D", origin="1899-12-30", errors="coerce")
    return out


def get_file_calendar_years(fp: str, date_col: str = "Date Performed") -> Optional[List[int]]:
    """Return the sorted, distinct calendar years found in *fp*'s date column.

    Used by the setup dialog's "Auto-Sort by Year" button to decide which
    single-year bucket a whole file belongs to.  Reads only ``date_col`` --
    a calamine streaming column push-down (see
    :func:`_read_excel_streaming`) for .xlsx/.xlsm -- so no other column is
    materialised.  Note this still has to visit every row (row-major
    storage gives no cheaper way to collect *every* distinct year in the
    column), unlike the O(1) header-only readers above.

    Returns ``None`` if the column is absent from the file, the file cannot
    be read, or the column contains no parseable dates.
    """
    cols = _read_header_columns(fp)
    if cols is not None and date_col not in cols:
        return None

    try:
        if fp.lower().endswith((".xlsx", ".xlsm")):
            df = _read_excel_streaming(
                fp, needed_cols=[date_col],
                rows_processed_ref=[0], last_pct_ref=[-1],
            )
        else:
            df = pd.read_excel(fp, usecols=[date_col])
    except Exception:
        return None

    if date_col not in df.columns:
        return None

    try:
        parsed = parse_exam_dates(df[date_col]).dropna()
        if parsed.empty:
            return None
        return sorted(int(y) for y in parsed.dt.year.unique())
    except Exception:
        return None


def report_missing_headers(file_list, needed_set, on_progress=None) -> dict:
    """Run the header-only pre-check on ``file_list`` and report the result.

    Reads only row 0 of each file (fast).  When expected columns are
    absent, prints a per-file summary to stderr and reports it through
    ``on_progress("missing_columns", {file: [cols], ...})``.  Returns the
    ``{file: [missing cols]}`` mapping (empty when nothing is missing).
    Informational only: never raises.
    """
    try:
        _header_missing = check_excel_headers(file_list, needed_set)
        if _header_missing:
            from .missing_columns import (
                format_consequences as _fmt_conseq,
                format_per_file_summary as _fmt_missing,
            )
            _summary = _fmt_missing(_header_missing)
            print(
                "\n[NOTE] Some expected columns are absent from the input "
                "file(s); rows that depend on them use calculated "
                "estimates (see below for exceptions).\n",
                file=sys.stderr,
            )
            print(_summary, file=sys.stderr)
            _consequences = _fmt_conseq(_header_missing)
            if _consequences:
                print(_consequences, file=sys.stderr)
            if on_progress:
                try:
                    on_progress("missing_columns", _header_missing)
                except Exception:
                    pass
        return _header_missing
    except Exception:
        return {}

