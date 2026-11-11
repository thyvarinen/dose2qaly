"""Participant-flow ledger (STROBE/RECORD "numbers at each stage").

Tracks how many examinations and distinct patients enter the cohort at
each exclusion stage, so the Summary tab and the manuscript can report a
transparent flow diagram:

    dataset (merged exams)
      - Non-Dose regions (QA / phantom / post-mortem: no living patient in
        the beam; dropped by :func:`partition_non_dose`)
      - CT / Dental-CBCT rows with no organ dose (dropped upstream in
        ``main.run_simulation`` when CT dose data are incomplete)
      - non-CBCT dental radiography (the ``DENTAL_IMAGING_SUBTYPE_REGIONS``
        rows, i.e. the "Dental imaging" pseudo-modality; dropped
        unconditionally)
      - exams with no organ-dose model (empty ``_cancer_doses``: non-dental
        CBCT regions, missing KAP / MGD, male mammography, unlocalised
        radiography, fluoroscopy / angiography, NM/PT, ...; dropped by
        :func:`partition_no_dose_model`)
    = analysed cohort == risk-modelled exams

Every examination that reaches ``exam_data`` after these partitions is
risk-modelled (non-empty ``_cancer_doses``, i.e.
``vectorized.build_exam_arrays``'s ``is_modelled`` rule holds for all of
them), so every downstream count -- Table 1 exams / patients / sex / age,
the age pyramid, the per-100 000 denominators -- is the analysed cohort.
The dropped exams are kept as a ledger (``stats["_excluded_cohorts"]``,
count tables per exclusion reason) so the By Body Region tab can still
list them as excluded examinations.  Each dropped exam is tagged with a
human-readable ``_exclusion_reason`` by :func:`exclusion_reason`.

Patient identity uses the same pseudonymised-id representation and
None/NaN handling as ``analysis.build_patients_by_modality``
(``_person_id`` falling back to ``_patient_id``; None or NaN ids each
count as one patient in ``patients_missing`` rather than joining the
``patients`` set).  Within one per-year batch the pseudonymisation salt
is pinned (``exam_doses.pin_patient_id_salt``), so cross-year unions in
:func:`union_participant_flows` are valid -- modelled on
``year_batch._cross_year_patients_by_modality``.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from .columnar_exams import ColumnarExamsList
from .style import fmt_thousands
from .region_resolver import (
    CT_UNLOCALIZED,
    DENTAL_IMAGING_SUBTYPE_REGIONS,
    RADIOGRAPHY_OTHER,
)


# Exclusion-ledger keys (``stats["_excluded_cohorts"]`` and the
# ``_exclusion_reason`` tag group under these) and their display labels.
EXCLUSION_NON_DOSE = "non_dose"
EXCLUSION_NO_ORGAN_DOSE = "no_organ_dose"
EXCLUSION_DENTAL_RADIOGRAPHY = "dental_radiography"
EXCLUSION_NO_DOSE_MODEL = "no_dose_model"
# Flow order; also the order of the By Body Region "Excluded" section.
EXCLUSION_ORDER: Tuple[str, ...] = (
    EXCLUSION_NON_DOSE, EXCLUSION_NO_ORGAN_DOSE,
    EXCLUSION_DENTAL_RADIOGRAPHY, EXCLUSION_NO_DOSE_MODEL)
# The two stages whose exams carry no risk model although they are real
# patient exposures the app could otherwise count; the "excluded dose"
# bookkeeping (male mammography MGD etc.) covers only these.
RISK_MODEL_EXCLUSIONS: Tuple[str, ...] = (EXCLUSION_NON_DOSE, EXCLUSION_NO_DOSE_MODEL)
EXCLUSION_LABELS: Dict[str, str] = {
    EXCLUSION_NON_DOSE: "Non-Dose regions: no living patient in the beam "
                        "(QA / phantom / post-mortem)",
    EXCLUSION_NO_ORGAN_DOSE: "CT without organ-dose data (incomplete dose export); "
                             "radiography without a kerma-area product or tube "
                             "voltage; dental CBCT without a kerma-area product",
    EXCLUSION_DENTAL_RADIOGRAPHY: "Dental panoramic / intraoral / cephalometric "
                                  "radiography (no organ-dose conversion available)",
    EXCLUSION_NO_DOSE_MODEL: "No organ-dose model (not risk-modelled)",
}

_UNLOCALISED_REGIONS = {"", "unknown", CT_UNLOCALIZED.lower(), RADIOGRAPHY_OTHER.lower()}


def exclusion_reason(ex: dict) -> str:
    """Human-readable reason why *ex* carries no risk model.

    Derived from the exam's modality, region, sex and dose fields -- the
    same facts ``exam_doses.compute_exam_organ_doses`` used when it
    declined to produce organ doses -- so the By Body Region tab can show
    *why* an examination was excluded, not only that it was.
    """
    region = str(ex.get("region") or "")
    if region.startswith("Non-Dose"):
        return "no living patient in the beam (QA / phantom / post-mortem)"
    if ex.get("reporting_region", region) in DENTAL_IMAGING_SUBTYPE_REGIONS:
        return ("dental panoramic / intraoral / cephalometric radiography: "
                "no organ-dose conversion available")
    mod = str(ex.get("modality") or "").strip()
    mod_u = mod.upper()
    region_l = region.strip().lower()
    has_organ_dose = bool(ex.get("organ_doses_gy"))
    is_mg = mod_u in ("MAMMOGRAPHY", "MG") or "MAMMO" in mod_u or region_l == "breast"
    if mod_u == "CBCT":
        if region_l != "dental cbct":
            return "non-dental CBCT: no organ-dose conversion available"
        # exam_doses._dental_doses returns no organ doses only when the
        # DAP column is missing or <= 0 (the kVp lookup always resolves).
        return "dental CBCT: kerma-area product missing or zero"
    if is_mg:
        if has_organ_dose and str(ex.get("sex_code") or "U").upper()[:1] == "M":
            return "male patient: breast cancer risk not modelled"
        return "mean glandular dose missing or zero"
    if mod_u == "CT":
        return "no organ-dose data (incomplete dose export)"
    if mod_u in ("RADIOGRAPHY", "DX", "CR", "DR", "RG"):
        if ex.get("_dose_quantity_missing"):
            return "kerma-area product or tube voltage missing"
        if region_l in _UNLOCALISED_REGIONS:
            return "unknown region: no organ-dose conversion available"
        return "no organ-dose conversion available for this region"
    if has_organ_dose:
        return "organ doses map to no modelled cancer site"
    return "no organ-dose model for this modality"


# Field access (plain list-of-dicts and columnar _cols path)

def _iter_exam_fields(exam_data):
    """Yield (modality, region_key, region_raw, pid, cancer_doses) per exam.

    ``region_key`` is the sweep/analysis keying: ``reporting_region``
    falling back to ``region``.  ``region_raw`` is the raw ``region``
    value ``vectorized.build_exam_arrays`` uses for its Non-Dose mask.
    """
    exam_data = ColumnarExamsList.from_dicts(exam_data or [])
    cols = exam_data._cols
    n = len(exam_data)
    mod_col = cols.get("modality", [None] * n)
    rep_col = cols.get("reporting_region")
    reg_col = cols.get("region", [None] * n)
    if rep_col is None:
        rep_col = reg_col
    pid_col = cols.get("_person_id")
    if pid_col is None:
        pid_col = cols.get("_patient_id", [None] * n)
    cd_col = cols.get("_cancer_doses", [None] * n)
    for i in range(n):
        yield (mod_col[i], rep_col[i], reg_col[i], pid_col[i], cd_col[i])


def _pid_is_missing(pid) -> bool:
    return pid is None or (isinstance(pid, float) and pid != pid)


def _is_non_dose_region(region_raw) -> bool:
    return str(region_raw or "").startswith("Non-Dose")


def _is_missing_region(region_key) -> bool:
    r = str(region_key or "").strip()
    return (not r) or r.lower() == "unknown"


def partition_dental_radiography(exam_data) -> Tuple[list, list]:
    """Split ``exam_data`` into (kept, dropped).

    Dropped = exams whose region key (``reporting_region`` falling back
    to ``region``, the same keying as
    ``analysis.build_patients_by_modality``) is in
    ``DENTAL_IMAGING_SUBTYPE_REGIONS`` -- the rows the app regroups into
    the "Dental imaging" pseudo-modality.  "Dental CBCT" is NOT dropped.
    """
    kept: list = []
    dropped: list = []
    cols = getattr(exam_data, "_cols", None)
    if cols is not None:
        n = len(exam_data)
        rep_col = cols.get("reporting_region")
        if rep_col is None:
            rep_col = cols.get("region", [None] * n)
        for i, ex in enumerate(exam_data):
            if rep_col[i] in DENTAL_IMAGING_SUBTYPE_REGIONS:
                dropped.append(ex)
            else:
                kept.append(ex)
    else:
        for ex in (exam_data or []):
            if ex.get("reporting_region", ex.get("region")) in DENTAL_IMAGING_SUBTYPE_REGIONS:
                dropped.append(ex)
            else:
                kept.append(ex)
    return kept, dropped


def _partition_by(exam_data, pred, reason_key: str) -> Tuple[list, list]:
    """Split ``exam_data`` into (kept, dropped) on ``pred``; tag the dropped
    exams with ``_exclusion_key`` / ``_exclusion_reason``.  Input order is
    preserved inside each partition."""
    kept: list = []
    dropped: list = []
    for ex in (exam_data or []):
        if pred(ex):
            try:
                ex["_exclusion_key"] = reason_key
                ex["_exclusion_reason"] = exclusion_reason(ex)
            except TypeError:
                # Read-only mapping (e.g. a columnar row view): the ledger
                # still gets the exam; only the tag is skipped.
                pass
            dropped.append(ex)
        else:
            kept.append(ex)
    return kept, dropped


def partition_non_dose(exam_data) -> Tuple[list, list]:
    """Split ``exam_data`` into (kept, dropped) where dropped = exams whose
    raw ``region`` is a ``Non-Dose`` placeholder (QA / phantom / post-mortem
    records that document no X-ray exposure of a living patient)."""
    return _partition_by(
        exam_data, lambda ex: _is_non_dose_region(ex.get("region")),
        EXCLUSION_NON_DOSE)


def partition_no_dose_model(exam_data) -> Tuple[list, list]:
    """Split ``exam_data`` into (kept, dropped) where dropped = exams with
    dose data but an empty ``_cancer_doses`` map, i.e. examinations the
    organ-dose models cannot describe (non-dental CBCT, male mammography,
    unlocalised radiography, ...).  Every exam in ``kept`` is risk-modelled
    in the sense of ``vectorized.build_exam_arrays``'s ``is_modelled``.
    Exams whose primary dose quantity is missing are filed by the caller
    (``main._prepare_cohort``) under the no-dose-data stage instead."""
    return _partition_by(
        exam_data, lambda ex: not ex.get("_cancer_doses"),
        EXCLUSION_NO_DOSE_MODEL)


def excluded_by_modality(exams) -> Dict[str, int]:
    """``{modality: n}`` over an excluded-exam list (raw modality strings)."""
    out: Dict[str, int] = {}
    for mod, _rk, _rr, _pid, _cd in _iter_exam_fields(exams):
        m = str(mod or "Unknown")
        out[m] = out.get(m, 0) + 1
    return out


def excluded_dose_sv_by_modality(exams) -> Dict[str, float]:
    """``{modality: Sv}`` -- effective dose carried by excluded exams (e.g.
    male mammography keeps an MGD-derived effective dose although breast
    cancer is not modelled).  Zero for most exclusion reasons."""
    out: Dict[str, float] = {}
    for ex in (exams or []):
        try:
            ed = float(ex.get("ed_msv", 0.0) or 0.0)
        except (TypeError, ValueError):
            ed = 0.0
        if ed <= 0.0:
            continue
        m = str(ex.get("modality") or "Unknown")
        out[m] = out.get(m, 0.0) + ed / 1000.0
    return out


def build_excluded_cohorts(excluded: Optional[Dict[str, list]], file_list,
                           group_label: Optional[str] = None) -> Dict[str, Any]:
    """``{exclusion key: CohortByFile}`` count tables for the excluded exams.

    The tables are built by :func:`file_cohort.build_file_cohort` from
    shallow copies whose ``debug_description`` is the exam's
    ``_exclusion_reason`` (orientation debug fields blanked), so the
    ``region_debug`` table reads ``(modality, region, reason) -> n`` and the
    By Body Region tab can list excluded examinations per region with the
    reason they were excluded.  Mergeable across years like
    ``stats["_file_cohort"]``.  Keys with no exams are omitted.
    """
    from .file_cohort import build_file_cohort
    out: Dict[str, Any] = {}
    for key, exams in (excluded or {}).items():
        if not exams:
            continue
        rows = []
        for ex in exams:
            row = dict(ex)
            row["debug_description"] = str(
                ex.get("_exclusion_reason") or exclusion_reason(ex))
            row["orientation_inferred"] = ""
            row["orientation_detected"] = ""
            row["orientation_resolved"] = ""
            rows.append(row)
        out[str(key)] = build_file_cohort(rows, file_list, group_label=group_label)
    return out


def merge_excluded_cohorts(per_run) -> Optional[Dict[str, Any]]:
    """Merge several runs' ``{key: CohortByFile}`` ledgers (sum of counts)."""
    from .file_cohort import merge_file_cohorts
    keys: list = []
    ledgers = [d for d in (per_run or ()) if isinstance(d, dict)]
    if not ledgers:
        return None
    for d in ledgers:
        for k in d:
            if k not in keys:
                keys.append(k)
    out: Dict[str, Any] = {}
    for k in keys:
        merged = merge_file_cohorts(d.get(k) for d in ledgers)
        if merged is not None:
            out[k] = merged
    return out


# Ledger construction

def _is_mg_modality(mod: str) -> bool:
    """Same mammography test as ``exclusion_reason``: modality code upper
    in ("MAMMOGRAPHY", "MG") or contains "MAMMO"."""
    mu = str(mod or "").strip().upper()
    return mu in ("MAMMOGRAPHY", "MG") or "MAMMO" in mu


def _stage(key: str, label: str, exams_list) -> Dict[str, Any]:
    patients: set = set()
    missing = 0
    by_modality: Dict[str, int] = {}
    n = 0
    mg_male = 0
    for mod, _rk, _rr, pid, _cd in _iter_exam_fields(exams_list):
        n += 1
        m = str(mod or "Unknown")
        by_modality[m] = by_modality.get(m, 0) + 1
        if _pid_is_missing(pid):
            missing += 1
        else:
            patients.add(pid)
    # Male-mammography count: iterate the raw exam dicts (sex_code is not
    # yielded by _iter_exam_fields).  Region "breast" also counts as MG
    # (same test as exclusion_reason).
    for ex in (exams_list or []):
        mod = str(ex.get("modality") or "")
        region_l = str(ex.get("region") or "").strip().lower()
        if _is_mg_modality(mod) or region_l == "breast":
            if str(ex.get("sex_code") or "U").upper()[:1] == "M":
                mg_male += 1
    return {"key": key, "label": label, "exams": n,
            "patients": patients, "patients_missing": missing,
            "by_modality": by_modality, "mg_male": mg_male}


def _filter_stage(key: str, label: str, exams_list, pred) -> Dict[str, Any]:
    return _stage(key, label, [ex for ex in exams_list if pred(ex)])


def build_participant_flow(
    *,
    rows_loaded: int,
    rows_after_preprocess: int,
    exams_merged: int,
    dataset_exams,
    non_dose_exams,
    no_organ_dose_exams,
    dental_exams,
    analysed_exams,
    no_dose_model_exams=None,
) -> Dict[str, Any]:
    """Assemble the per-year flow ledger.

    The exam lists are expected to be disjoint partitions of
    ``dataset_exams`` in precedence order Non-Dose -> no-organ-dose ->
    dental -> no-dose-model -> analysed (``main._prepare_cohort`` builds
    them that way so the invariant ``dataset = non_dose + no_organ_dose +
    dental + no_dose_model + analysed`` holds by construction).  The
    analysed cohort is, by construction, the risk-modelled cohort.
    """
    stages = [
        _stage("dataset", "Dataset (merged examinations)", dataset_exams),
        _stage("excl_non_dose",
               "Excluded: Non-Dose regions (QA/phantom/post-mortem)",
               non_dose_exams),
        _stage("excl_no_organ_dose",
               "Excluded: no dose data",
               no_organ_dose_exams),
        _stage("excl_dental_radiography",
               "Excluded: non-CBCT dental radiography",
               dental_exams),
        _stage("excl_no_dose_model",
               "Excluded: dose data but no organ-dose model",
               no_dose_model_exams or []),
        _stage("analysed", "Analysed cohort (risk-modelled)", analysed_exams),
    ]
    # Sub-partition of the analysed cohort.
    def _missing_reg(ex):
        return _is_missing_region(ex.get("reporting_region", ex.get("region")))
    stages.append(_filter_stage(
        "analysed_missing_region", "  of which missing region",
        analysed_exams, _missing_reg))
    stages.append(_filter_stage(
        "analysed_ct_adult_trunk",
        "  of which adult chest/abdominal CT (weight-correction eligible)",
        analysed_exams, lambda ex: bool(ex.get("_golikov_eligible"))))
    stages.append(_filter_stage(
        "analysed_ct_no_weight",
        "  of which adult chest/abdominal CT without recorded weight",
        analysed_exams, lambda ex: bool(ex.get("_golikov_no_weight"))))
    # Patients lost entirely by the exclusions (dental-only, Non-Dose-only,
    # no-organ-dose-only, no-dose-model-only): in the dataset set but in
    # no analysed exam.
    _ds = stages[0]
    _an = next(s for s in stages if s["key"] == "analysed")
    patients_dataset = len(_ds["patients"]) + _ds["patients_missing"]
    patients_analysed = len(_an["patients"]) + _an["patients_missing"]
    patients_excluded_only = (
        len(_ds["patients"] - _an["patients"])
        + (_ds["patients_missing"] - _an["patients_missing"])
    )
    return {
        "rows_loaded": int(rows_loaded or 0),
        "rows_after_preprocess": int(rows_after_preprocess or 0),
        "exams_merged": int(exams_merged or 0),
        "patients_dataset": patients_dataset,
        "patients_analysed": patients_analysed,
        "patients_excluded_only": patients_excluded_only,
        "stages": stages,
    }


# Cross-year union

def union_participant_flows(per_year: Dict[Any, dict]) -> Optional[Dict[str, Any]]:
    """Union per-year flow dicts into one cross-year ledger.

    Exams and ``patients_missing`` are summed; ``patients`` becomes the
    size of the union of all years' sets plus the summed missing count
    (valid because the batch pins one pseudonymisation salt).  Returns
    ``None`` if any year's flow is missing or malformed.
    """
    flows = [f for f in (per_year or {}).values() if isinstance(f, dict)]
    if not flows or len(flows) != len(per_year or {}):
        return None
    keys = [s["key"] for s in flows[0].get("stages", [])]
    out_stages: List[Dict[str, Any]] = []
    _set_by_key: Dict[str, set] = {}
    _missing_by_key: Dict[str, int] = {}
    for key in keys:
        exams = 0
        union: set = set()
        missing = 0
        by_mod: Dict[str, int] = {}
        mg_male = 0
        label = None
        for f in flows:
            st = next((s for s in f.get("stages", []) if s.get("key") == key), None)
            if st is None:
                return None
            label = st["label"]
            exams += int(st.get("exams", 0) or 0)
            union |= set(st.get("patients") or set())
            missing += int(st.get("patients_missing", 0) or 0)
            mg_male += int(st.get("mg_male", 0) or 0)
            for m, c in (st.get("by_modality") or {}).items():
                by_mod[m] = by_mod.get(m, 0) + int(c or 0)
        _set_by_key[key] = union
        _missing_by_key[key] = missing
        out_stages.append({"key": key, "label": label, "exams": exams,
                           "patients": len(union) + missing,
                           "patients_missing": missing,
                           "by_modality": by_mod, "mg_male": mg_male})
    # Patients lost entirely by the exclusions, computed from the UNIONED
    # sets (not by summing per-year values): a patient dental-only in one
    # year but CT-imaged in another is still in the analysed cohort.
    _ds_set = _set_by_key.get("dataset", set())
    _an_set = _set_by_key.get("analysed", set())
    _ds_miss = _missing_by_key.get("dataset", 0)
    _an_miss = _missing_by_key.get("analysed", 0)
    return {
        "rows_loaded": sum(int(f.get("rows_loaded", 0) or 0) for f in flows),
        "rows_after_preprocess": sum(int(f.get("rows_after_preprocess", 0) or 0) for f in flows),
        "exams_merged": sum(int(f.get("exams_merged", 0) or 0) for f in flows),
        "patients_dataset": len(_ds_set) + _ds_miss,
        "patients_analysed": len(_an_set) + _an_miss,
        "patients_excluded_only": len(_ds_set - _an_set)
                                  + (_ds_miss - _an_miss),
        "n_years": len(flows),
        "stages": out_stages,
    }


# Rendering

def _stage_patients(st: dict) -> int:
    pat = st["patients"]
    if isinstance(pat, (set, frozenset)):
        return len(pat) + int(st.get("patients_missing", 0) or 0)
    return int(pat or 0)


def _fmt_int_comma(n) -> str:
    return f"{int(n):,}"


def _fmt_int_fig(n) -> str:
    """Thousands separated with a thin space (U+2009), the manuscript's
    number style, for text drawn into the flow-diagram figure."""
    return fmt_thousands(int(n))


def _mg_male_modality_parts(st: dict, fmt: Callable[[int], str] = _fmt_int_comma
                            ) -> Tuple[str, Optional[str]]:
    """Build the by-modality parenthesis for an ``excl_*`` stage, marking
    mammography exams that are all male (asterisk + footnote) or partially
    male (``(x male)``).  Only the no-dose-model stage is marked: there
    sex is the reason for exclusion; elsewhere (no living patient, no dose
    data) it is irrelevant.  ``fmt`` renders the integers.  Returns
    ``(parts_text, footnote_or_None)``."""
    bm = st.get("by_modality") or {}
    mg_male = (int(st.get("mg_male", 0) or 0)
               if st.get("key") == "excl_no_dose_model" else 0)
    parts = []
    footnote = None
    for m, c in sorted(bm.items()):
        if _is_mg_modality(m) and mg_male > 0 and c > 0:
            if mg_male == c:
                parts.append(f"{m} {fmt(c)}*")
                footnote = "*male patients (breast cancer risk not modelled)"
            else:
                parts.append(f"{m} {fmt(c)} ({fmt(mg_male)} male)")
        else:
            parts.append(f"{m} {fmt(c)}")
    return ", ".join(parts), footnote



# Selection-of-examinations flow diagram (the manuscript's STROBE-style
# figure: a vertical chain acquisitions -> examinations -> analysed
# cohort, with one side box per exclusion stage branching off the arrow
# between the examinations and analysed boxes).

_FLOW_EXCLUSION_TEXT: Dict[str, str] = {
    "excl_non_dose": "examinations with no living patient in the beam "
                     "(quality control, phantom, post-mortem)",
    "excl_no_organ_dose": "examinations without dose data",
    "excl_dental_radiography": "dental panoramic, intraoral and "
                               "cephalometric radiographs (no organ-dose "
                               "conversion available)",
    "excl_no_dose_model": "examinations with dose data but no organ-dose "
                          "model",
}

# Analysed-cohort sub-stages that the flow-diagram figure hides (the text
# and LaTeX ledgers still list them).  The two weight-correction buckets
# are manuscript footnotes, not diagram rows.
_FLOW_BOX_HIDDEN_SUBSTAGES = frozenset(
    {"analysed_ct_adult_trunk", "analysed_ct_no_weight"})


def participant_flow_boxes(flow: Optional[dict], *, n_years: Optional[int] = None,
                           period: Optional[str] = None
                           ) -> Optional[Tuple[List[str], List[str]]]:
    """Text of the flow diagram as ``(main_boxes, side_boxes)``.

    ``main_boxes`` is the vertical chain (acquisitions, examinations after
    merge, analysed cohort); ``side_boxes`` has one entry per non-empty
    ``excl_*`` stage in ledger order.  Pure function (no matplotlib) so the
    wording can be tested; :func:`plot_participant_flow` draws it.
    Returns ``None`` when ``flow`` carries no stages.
    """
    if not flow or not flow.get("stages"):
        return None
    stages = {s["key"]: s for s in flow["stages"]}
    ny = int(n_years or flow.get("n_years") or 1)
    ds = stages.get("dataset")
    an = stages.get("analysed")
    n_merged = int((ds or {}).get("exams") or flow.get("exams_merged") or 0)
    n_rows = int(flow.get("rows_loaded") or flow.get("rows_after_preprocess") or 0)
    fi = _fmt_int_fig
    main: List[str] = []
    period_s = f", {period}" if period else ""
    if n_rows:
        main.append("Acquisitions exported from the dose-management "
                    f"platform{period_s}\nn = {fi(n_rows)}")
    pd_ = flow.get("patients_dataset")
    main.append("Examinations\n"
                f"n = {fi(n_merged)}"
                + (f" in {fi(pd_)} patients" if pd_ is not None else ""))
    side: List[str] = []
    for st in flow["stages"]:
        if not st["key"].startswith("excl_"):
            continue
        n = int(st.get("exams") or 0)
        if n <= 0:
            continue
        desc = _FLOW_EXCLUSION_TEXT.get(
            st["key"], st["label"].split(":", 1)[-1].strip().lower())
        txt = f"Excluded: {fi(n)} {desc}"
        bm = st.get("by_modality") or {}
        if bm:
            parts, footnote = _mg_male_modality_parts(st, fi)
            txt += "\n(" + parts + ")"
            if footnote:
                txt += "\n" + footnote
        side.append(txt)
    if an is not None:
        n_an = int(an.get("exams") or 0)
        pa = flow.get("patients_analysed")
        if pa is None:
            pa = _stage_patients(an)
        lines = ["Analysed cohort"]
        line2 = f"n = {fi(n_an)} examinations"
        if n_merged:
            line2 += f" ({fi(n_merged - n_an)} excluded)"
        if ny > 1:
            line2 += f"; {fi(round(n_an / ny))} per year"
        lines.append(line2)
        line3 = f"in {fi(pa)} patients"
        pe = flow.get("patients_excluded_only")
        if pe:
            line3 += f" ({fi(pe)} had only excluded examinations)"
        lines.append(line3)
        for st in flow["stages"]:
            if (st["key"].startswith("analysed_")
                    and st["key"] not in _FLOW_BOX_HIDDEN_SUBSTAGES
                    and int(st.get("exams") or 0) > 0):
                lines.append(f"of which {fi(st['exams'])} "
                             f"{st['label'].strip().removeprefix('of which').strip()}")
        main.append("\n".join(lines))
    return main, side


def plot_participant_flow(ax, flow: Optional[dict], *, n_years: Optional[int] = None,
                          period: Optional[str] = None, fontsize: float = 8.5,
                          fit: str = "shrink") -> None:
    """Draw the selection-of-examinations diagram on ``ax`` (axes-fraction
    coordinates, axis hidden).  Box heights are estimated from the line
    count at the current figure size.  ``fit="shrink"`` steps the font
    down until the whole diagram fits the figure as it is; ``fit="grow"``
    keeps the nominal font and instead enlarges the figure HEIGHT
    (``figure.set_size_inches(..., forward=False)``) until it fits -- the
    GUI uses this inside a scrollable canvas and resizes the Tk widget to
    the new figure size afterwards."""
    import textwrap
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = participant_flow_boxes(flow, n_years=n_years, period=period)
    if boxes is None:
        ax.text(0.5, 0.5, "Participant flow not available for this result",
                ha="center", va="center", transform=ax.transAxes, fontsize=fontsize)
        return
    main, side = boxes
    fig = ax.figure
    bb = ax.get_position()
    xm, xs, right = 0.27, 0.53, 0.99          # main centre / side left edge
    main_frac, side_frac = 0.46, right - xs   # box widths incl. padding
    gap0, gap_s0 = 0.055, 0.03                # gaps at the nominal font size

    def _wrap(txt: str, width: int) -> str:
        return "\n".join(textwrap.fill(p, width) for p in txt.split("\n"))

    def _layout(fs):
        fw, fh = fig.get_size_inches()
        aw_in, ah_in = max(fw * bb.width, 0.5), max(fh * bb.height, 0.5)
        char_w = 0.58 * fs / 72.0          # inches per character (DejaVu Sans)
        pad_in = 0.5 * fs / 72.0           # bbox pad=0.5 (font-size units)
        mw = max(14, int((main_frac * aw_in - 2 * pad_in) / char_w))
        sw = max(14, int((side_frac * aw_in - 2 * pad_in) / char_w))
        m_txt = [_wrap(t, mw) for t in main]
        s_txt = [_wrap(t, sw) for t in side]
        lh = 1.4 * fs / 72.0 / ah_in       # line height, axes units
        pad = pad_in / ah_in
        g, gs = gap0 * fs / fontsize, gap_s0 * fs / fontsize
        h = lambda t: (t.count("\n") + 1) * lh + 2 * pad
        m_h = [h(t) for t in m_txt]
        s_h = [h(t) for t in s_txt]
        span = (g + sum(s_h) + gs * (len(s_h) - 1) + g) if s_h else g
        total = sum(m_h) + g * max(len(m_h) - 2, 0) + span
        return m_txt, s_txt, m_h, s_h, span, total, pad_in / aw_in, g, gs

    fs = fontsize
    if fit == "grow":
        # Resize the figure height so the content fills 90-98 % of it
        # (grows for long ledgers, tightens when the figure is too tall).
        for _ in range(5):
            total = _layout(fs)[5]
            if 0.90 <= total <= 0.98:
                break
            fw, fh = fig.get_size_inches()
            fig.set_size_inches(fw, max(2.0, fh * total / 0.95), forward=False)
        m_txt, s_txt, m_h, s_h, span, total, pad_x, gap, gap_s = _layout(fs)
    else:
        for step in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            fs = fontsize * step
            m_txt, s_txt, m_h, s_h, span, total, pad_x, gap, gap_s = _layout(fs)
            if total <= 0.98:
                break
    top = 1.0 - max(0.0, (1.0 - total)) / 2.0
    box_kw = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=0.9)
    arrow_kw = dict(arrowstyle="-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0,
                    mutation_scale=9)

    def _arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                    textcoords="axes fraction", arrowprops=arrow_kw, zorder=2)

    # Main chain: every box but the last sits above the side boxes.
    y = top
    centres: List[float] = []
    n_main = len(m_txt)
    for i, (txt, hh) in enumerate(zip(m_txt, m_h)):
        if i == n_main - 1 and n_main > 1 and s_h:
            y -= span - gap          # leave room for the side boxes
        yc = y - hh / 2.0
        ax.text(xm, yc, txt, ha="center", va="center", fontsize=fs,
                transform=ax.transAxes, bbox=box_kw, zorder=3)
        centres.append(yc)
        y = yc - hh / 2.0 - gap
    for i in range(1, n_main):
        _arrow(xm, centres[i - 1] - m_h[i - 1] / 2.0, xm, centres[i] + m_h[i] / 2.0)
    # Side boxes branch off the arrow between the last two main boxes.
    if s_txt and n_main >= 2:
        ys = centres[-2] - m_h[-2] / 2.0 - gap
        for txt, hh in zip(s_txt, s_h):
            yc = ys - hh / 2.0
            ax.text(xs + pad_x, yc, txt, ha="left", va="center", fontsize=fs,
                    transform=ax.transAxes, bbox=box_kw, zorder=3)
            _arrow(xm, yc, xs, yc)
            ys = yc - hh / 2.0 - gap_s
