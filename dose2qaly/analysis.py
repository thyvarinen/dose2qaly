"""
Dose2QALY Analysis Module
=========================================
Provides analysis and visualization functions imported by Monte Carlo.
All plotting functions accept matplotlib axes so the GUI can embed them.
"""
from __future__ import annotations

import random as _rng
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import Z_95, BACKGROUND_MSV_PER_YEAR
from .risk_models import (
    MAX_ATTAINED_AGE_ICRP152,
    OTHER_SOLID_REMAINDER_SITES,
    REMAINDER_ORGAN_MULTIPLICITY,
    _delta_lambda_vectorized,
    _get_cancer_incidence_rate,
    cancer_risk_params,
    compute_lifetime_excess_cancer_risk_REIC,
    get_expected_remaining_years,
    get_organ_dose_for_cancer,
    lifetime_baseline_cancer_incidence_ICRP152,
    lookup_cancer_params,
    DDREF_SOLID_CANCERS,
    DDREF_LEUKAEMIA,
)
# Shared sex-exclusion table and modeled-cancer iterator — canonical versions
# live in exam_doses so the main engine and this analysis module
# cannot drift out of sync.
from .exam_doses import (
    _SEX_EXCLUDED_CANCER_TYPES,
    _iter_modeled_cancer_types,
)
from .columns import ExamColumns, EventColumns, is_event_columns
from .columnar_exams import ColumnarExamsList
from .region_resolver import fold_rare_region_combos as _fold_rare_region_combos
# Centralised colour-blind-safe palette + semantic tokens.  Prefer these
# over ad-hoc hex strings so every chart in the GUI stays consistent and
# deuteranope/protanope-friendly.
from .style import (
    PALETTE_LIST as _PALETTE_LIST,
    SEMANTIC as _SEMANTIC,
    OKABE_ITO as _OKABE_ITO,
    stable_color as _stable_color,
    shade_color as _shade_color,
    fmt_thousands as _fmt_thousands,
)


# Modality display names
MODALITY_DISPLAY = {
    "CT": "CT (CT+CT/PT)",
    "MG": "MG (Mammography)",
    "RADIOGRAPHY": "Radiography (CR+DX+DR)",
    "ANGIOGRAPHY": "Angiography (XA+RF)",
    "CBCT": "CBCT",
    "CR": "CR (Computed Radiography)",
    "DX": "DX (Digital Radiography)",
    "XA": "XA (Angiography)",
    "RF": "RF (Fluoroscopy)",
    "DR": "DR (Digital Radiography)",
    "CT/PT": "CT/PT (PET-CT)",
    "PT": "PT (PET)",
    "NM": "NM (Nuclear Medicine)",
    "US": "US (Ultrasound)",
    "MR": "MR (MRI)",
    "IO": "IO (Intraoral)",
    "PX": "PX (Panoramic)",
    "DENTAL IMAGING": "Dental imaging (Panoramic+Intraoral+Kefalometry)",
}


def modality_name(code: str) -> str:
    """Return a human-readable modality name from a raw code.

    Examples: ``"CT"`` → ``"CT scan"``, ``"CR"`` → ``"X-ray"``.
    Unknown codes are returned unchanged.
    """
    key = str(code).strip().upper()
    return MODALITY_DISPLAY.get(key, str(code))


# Plain-language dose reference values
# Used in format_summary to translate abstract numbers into everyday terms.
CHEST_XRAY_MSV = 0.02                    # typical chest PA projection; Mettler et al, Radiology 2008;248:254-63

# Shared DDREF / leukaemia helpers

# ICRP 152 uses different DDREF values for solid cancers and leukaemia.
# The exact values are selected by the caller (e.g. 2.0 / 1.0 for default
# ICRP 152, but sensitivity analyses may override them).  Capping at 0.01
# prevents division by zero or absurdly large weights if a misconfiguration
# passes a tiny DDREF.
_LEUK_TYPES = ("leukaemia", "leukemia")

# ICRP 152 p.26: average years of life lost per radiogenic cancer (the 15 y
# the l_rel factors are normalised to).  Defined here rather than imported
# from vectorized/research_questions to avoid a circular import.
_ICRP_AVG_LIFE_LOST_YEARS: float = 15.0


def _ddref_scales(ddref_solid: float, ddref_leuk: float) -> Tuple[float, float]:
    """Return (scale_solid, scale_leuk) for ICRP 152 DDREF weights."""
    return 2.0 / max(ddref_solid, 0.01), 1.0 / max(ddref_leuk, 0.01)


def _is_leukaemia_type(cancer_type) -> bool:
    """Return True if ``cancer_type`` is a leukaemia/non-CLL bone-marrow site."""
    return str(cancer_type or "").lower() in _LEUK_TYPES


class _FilteredEvents:
    """Uniform columnar view over filtered events."""

    def __init__(self, event_columns, event_mask):
        self._event_columns = event_columns
        self._event_mask = event_mask

    def __getitem__(self, field: str):
        """Return a columnar slice for ``field``."""
        return getattr(self._event_columns, field)[self._event_mask]

    def __len__(self):
        return int(self._event_mask.sum())


def _filter_events(event_columns, filters: dict) -> Optional[_FilteredEvents]:
    """Return a filtered columnar event view, or None if no events match."""
    if event_columns is None or len(event_columns) == 0:
        return None
    event_mask = _filter_columns(event_columns, filters, age_attr="age_at_exposure", sex_attr="sex")
    if not event_mask.any():
        return None
    return _FilteredEvents(event_columns, event_mask)


def _ensure_exam_columns(exam_data) -> Optional[ExamColumns]:
    """Normalize a list of dicts or an ExamColumns to ExamColumns."""
    if exam_data is None or len(exam_data) == 0:
        return None
    if hasattr(exam_data, "age"):
        return exam_data
    return ExamColumns.from_exam_data(exam_data)


def _columnar_dict_to_event_columns(d: Optional[dict]) -> EventColumns:
    """Build an EventColumns from the columnar dict-of-arrays returned by the engine."""
    if not d:
        return EventColumns.empty()
    if is_event_columns(d):
        return d
    def _get_arr(key, dtype):
        val = d.get(key)
        if val is None:
            return np.array([], dtype=dtype)
        return np.asarray(val, dtype=dtype)
    return EventColumns(
        cancer_type=_get_arr("cancer_type", object),
        modality=_get_arr("modality", object),
        region=_get_arr("region", object),
        protocol=_get_arr("protocol", object),
        age_at_exposure=_get_arr("age_at_exposure", np.float64),
        sex=_get_arr("sex", "U1"),
        died=_get_arr("died", bool),
        exam_index=_get_arr("exam_index", np.int64),
        iteration=_get_arr("iteration", np.int32),
        ed_msv=_get_arr("ed_msv", np.float64),
        QALY_lost=_get_arr("QALY_lost", np.float64),
        life_lost_years=_get_arr("life_lost_years", np.float64),
        age_at_death=_get_arr("age_at_death", np.float64),
        projected_lifetime=_get_arr("projected_lifetime", np.float64),
    )


def _ensure_event_columns(cancer_events) -> Optional[EventColumns]:
    """Normalise a list of dicts, a columnar dict, or an EventColumns to EventColumns.

    Handles three input shapes:
    * ``EventColumns`` — returned as-is (checked via :func:`is_event_columns`)
    * ``dict`` of arrays (engine output) — converted via
      :func:`_columnar_dict_to_event_columns`
    * ``list`` of event dicts — converted via :meth:`EventColumns.from_events`

    Hosting the dict-of-arrays path here lets every call site
    (``analysis``, ``adv_ui``, ``year_batch``, ``research_questions``)
    share the same normalisation.
    """
    if cancer_events is None or len(cancer_events) == 0:
        return None
    if is_event_columns(cancer_events):
        return cancer_events
    if isinstance(cancer_events, dict):
        return _columnar_dict_to_event_columns(cancer_events)
    return EventColumns.from_events(cancer_events)




# Formatting helpers

def _format_one_in(rate_per_person: float) -> str:
    """Express a small probability as '1 in N' (or similar) for lay reading."""
    if rate_per_person is None or rate_per_person <= 0:
        return "effectively 0 (none observed)"
    if rate_per_person >= 1.0:
        return f"~{rate_per_person:.2f} events per person"
    n = 1.0 / rate_per_person
    if n >= 1e9:
        return "1 in >1 billion"
    if n >= 1e6:
        return f"1 in {n / 1e6:.1f} million"
    if n >= 10_000:
        return f"1 in {n:,.0f}"
    if n >= 100:
        return f"1 in {n:.0f}"
    if n >= 10:
        return f"1 in {n:.1f}"
    return f"1 in {n:.2f}"


def _format_chest_xrays(msv: float) -> str:
    if msv <= 0:
        return "0"
    n = msv / CHEST_XRAY_MSV
    if n < 1:
        return f"{n:.2f}"
    if n < 10:
        return f"{n:.1f}"
    return f"{n:,.0f}"


def _format_background_equiv(msv: float, background_msv_per_year: float = BACKGROUND_MSV_PER_YEAR) -> str:
    """Return human-friendly duration of natural-background dose equivalent."""
    if msv <= 0:
        return "0"
    years = msv / background_msv_per_year
    if years < 1.0 / 365.25:
        hours = years * 365.25 * 24
        return f"{hours:.1f} hours"
    if years < 1.0 / 12:
        days = years * 365.25
        return f"{days:.1f} days"
    if years < 1.0:
        months = years * 12
        return f"{months:.1f} months"
    if years < 100:
        return f"{years:.1f} years"
    return f"{years:,.0f} years"


def _build_patient_aggregates(exam_data: list) -> Tuple[dict, dict, dict, int, int]:
    """Compute patient aggregates in a single pass over exam_data.

    Returns
    -------
    tuple
        (pid_counts, pid_to_dose, pid_to_exams, pid_none_n, pid_nan_n) where:
        - pid_counts: dict mapping patient_id -> exam count (for diagnostics)
        - pid_to_dose: empty dict (unpopulated placeholder slot)
        - pid_to_exams: empty dict (see pid_to_dose)
        - pid_none_n: number of exams with missing patient id
        - pid_nan_n: number of exams with NaN patient id

    Only ``pid_counts`` and the none/nan tallies are populated; the dose and
    exam-index dicts stay empty placeholders so the 5-tuple shape stays
    compatible with cache readers (``format_summary`` / ``main.run_simulation``).
    """
    pid_counts: dict = {}
    pid_to_dose: dict = {}
    pid_to_exams: dict = {}
    pid_none_n = 0
    pid_nan_n = 0

    exam_columns = _ensure_exam_columns(exam_data)
    if exam_columns is None:
        return pid_counts, pid_to_dose, pid_to_exams, pid_none_n, pid_nan_n

    patient_id_arr = exam_columns.person_id
    for i in range(len(exam_columns)):
        pid = patient_id_arr[i]
        if pid is None:
            pid_none_n += 1
        elif isinstance(pid, float) and pid != pid:
            pid_nan_n += 1
        else:
            pid_counts[pid] = pid_counts.get(pid, 0) + 1

    return pid_counts, pid_to_dose, pid_to_exams, pid_none_n, pid_nan_n


def build_patients_by_modality(exam_data) -> dict:
    """Distinct patient ids and unidentifiable-exam counts per modality.

    Returns ``{"sets": {modality: set_of_person_ids},
    "missing": {modality: int}}``.  Modality keying mirrors the
    Region x Age sweep's cell keying in
    ``research_questions._region_age_baseline``: exams whose
    ``reporting_region`` (falling back to ``region``) is a dental-imaging
    subtype region are keyed under the ``"Dental imaging"``
    pseudo-modality.  Person id is ``_person_id`` falling back to
    ``_patient_id`` (columns.py convention); a missing or NaN id counts
    towards ``missing`` instead of the set.  ``exam_data`` is normalised
    to the columnar ``ColumnarExamsList`` on entry.
    """
    from .region_resolver import DENTAL_IMAGING_SUBTYPE_REGIONS

    sets: Dict[str, set] = {}
    missing: Dict[str, int] = {}

    def _mod_key(mod, reg) -> str:
        m = str(mod or "")
        if reg in DENTAL_IMAGING_SUBTYPE_REGIONS:
            m = "Dental imaging"
        return m

    def _add(mod, pid) -> None:
        if pid is None or (isinstance(pid, float) and pid != pid):
            missing[mod] = missing.get(mod, 0) + 1
        else:
            sets.setdefault(mod, set()).add(pid)
            missing.setdefault(mod, missing.get(mod, 0))

    exam_data = ColumnarExamsList.from_dicts(exam_data or [])
    cols = exam_data._cols
    n = len(exam_data)
    mod_col = cols.get("modality", [None] * n)
    reg_col = cols.get("reporting_region")
    if reg_col is None:
        reg_col = cols.get("region", [None] * n)
    pid_col = cols.get("_person_id")
    if pid_col is None:
        pid_col = cols.get("_patient_id", [None] * n)
    for i in range(n):
        _add(_mod_key(mod_col[i], reg_col[i]), pid_col[i])
    return {"sets": sets, "missing": missing}


def _count_unique_examinations(exam_data) -> Optional[int]:
    """Count distinct examinations by ``_patient_id`` (the ``A#`` accession
    number), the same key ``merge_acquisition_series`` groups on to collapse
    multi-acquisition rows into one exam.

    A row with no A# never merges with any other row (see
    ``merge_acquisition_series``), so each such row is its own examination
    here too -- mirrors how :func:`_build_patient_aggregates` treats a
    missing/NaN person id as its own patient.

    Returns ``None`` when ``exam_data`` cannot be read as columns, so the
    caller can fall back to the acquisitions/iterations estimate.
    """
    exam_columns = _ensure_exam_columns(exam_data)
    if exam_columns is None:
        return None
    aid_arr = exam_columns.patient_id
    seen: set = set()
    n_missing = 0
    for i in range(len(exam_columns)):
        aid = aid_arr[i]
        if aid is None or (isinstance(aid, float) and aid != aid):
            n_missing += 1
        else:
            seen.add(aid)
    return len(seen) + n_missing


def resolve_unique_examinations(stats: dict, exam_data: Optional[list] = None) -> int:
    """Resolve the unique-examination count for a ``(stats, exam_data)`` pair.

    Prefers a direct count of distinct ``A#`` accession numbers in
    ``exam_data`` (:func:`_count_unique_examinations`) and falls back to the
    ``acquisitions / iterations`` quotient when ``exam_data`` is unavailable
    -- e.g. the live Summary tab, which omits ``exam_data`` from
    :func:`format_summary` calls to avoid blocking the worker thread.

    Shared by :func:`format_summary` (headline "per iteration" -> "per
    100,000 examinations" rescale) and ``gui._render_summary`` (the matching
    rescale of the per-iteration distribution charts), so both always agree
    on the same cohort size for a given modality-filtered view.
    """
    stats = stats or {}
    iterations = int(stats.get("iterations", 0) or 0)
    acquisitions = int(stats.get("acquisitions", 0) or 0)
    _unique_exams_by_aid = _count_unique_examinations(exam_data) if exam_data else None
    return (
        _unique_exams_by_aid if _unique_exams_by_aid is not None
        else (int(acquisitions / iterations) if iterations > 0 else 0)
    )


def _per_year_total_examinations(stats: dict) -> Optional[int]:
    """Total unique examinations summed across every year in a per-year batch.

    Uses exactly the per-year ``_unique_examinations`` logic (with the
    acquisitions/iterations fallback for batches that predate that
    precomputation) that the dataset-size lines display, so the two never
    disagree.  Returns ``None`` when ``stats`` is not a per-year-average run
    or has no ``_year_batches``.
    """
    stats = stats or {}
    if stats.get("_run_mode") != "per_year_average":
        return None
    batches = stats.get("_year_batches")
    if not batches:
        return None
    tot = 0
    for _y in batches.values():
        if "_unique_examinations" in _y:
            tot += int(_y.get("_unique_examinations", 0) or 0)
            continue
        _yit = float(_y.get("iterations", 0) or 0)
        if _yit > 0:
            tot += int(float(_y.get("acquisitions", 0) or 0) / _yit)
    return tot


def _per_year_summed_patients(stats: dict) -> Optional[int]:
    """Sum of per-year unique-patient counts across a per-year batch (patient-years).

    The basis for the per-year mean: a patient imaged in several years
    counts once per year, so this is a patient-year total, not a distinct
    head-count.  Returns ``None`` when ``stats`` is not a per-year-average
    run or has no ``_year_batches``.
    """
    stats = stats or {}
    if stats.get("_run_mode") != "per_year_average":
        return None
    batches = stats.get("_year_batches")
    if not batches:
        return None
    return sum(int(y.get("_unique_patients", 0) or 0) for y in batches.values())


def _per_year_total_patients(stats: dict) -> Optional[int]:
    """Distinct patients across the whole per-year batch.

    A patient imaged in several years counts once.  Reads the batch-wide
    cross-year union stored on ``stats["_unique_patients_total"]`` (built
    by ``year_batch._cross_year_patient_totals``, valid only because the
    batch pins one pseudonymisation salt).  Falls back to
    :func:`_per_year_summed_patients` (the per-year sum) for batches
    recorded before the cross-year union existed, so the headline still
    shows a number instead of disappearing.  Returns ``None`` when
    ``stats`` is not a per-year-average run or has no ``_year_batches``.
    """
    stats = stats or {}
    if stats.get("_run_mode") != "per_year_average":
        return None
    batches = stats.get("_year_batches")
    if not batches:
        return None
    tot = stats.get("_unique_patients_total")
    if tot is not None:
        return int(tot)
    return _per_year_summed_patients(stats)


def _per_year_mean(stats: dict, total: Optional[int]) -> Optional[int]:
    """Per-year mean of ``total`` over the year-batch count, or ``None``."""
    if total is None:
        return None
    batches = (stats or {}).get("_year_batches") or {}
    n_yrs = len(batches) or 1
    return int(round(total / n_yrs))


def _per_year_mean_examinations(stats: dict) -> Optional[int]:
    """Per-year-average mean examination count, or ``None`` when not applicable.

    In ``per_year_average`` mode the consistent denominator for every
    per-exam figure (plain-language interpretation, the per-100k
    projection) is the *mean* annual examination count, not the last
    processed year's ``exam_data`` (which is what
    :func:`resolve_unique_examinations` would count).  Because the
    per-iteration case figures are themselves equal-weighted means across
    years (``mean(cases_y) / mean(exams_y) == sum(cases_y) / sum(exams_y)``),
    dividing by this mean recovers the true per-exam rate.  Returns ``None``
    when ``stats`` is not a per-year-average run or has no ``_year_batches``.
    """
    return _per_year_mean(stats, _per_year_total_examinations(stats))


def _per_year_mean_patients(stats: dict) -> Optional[int]:
    """Per-year-average mean unique-patient count, or ``None`` when not applicable.

    Mirrors :func:`_per_year_mean_examinations` for the unique-patient
    count, so the unique-patient count reported for the cohort
    (which is per-person, not per-exam) uses the cross-year average rather than the last processed
    year's patient table.  The mean is patient-years / n_years (a patient
    imaged in several years counts once per year), so it divides
    :func:`_per_year_summed_patients` -- NOT the distinct cross-year total
    -- by the year count.  Returns ``None`` when ``stats`` is not a
    per-year-average run or has no ``_year_batches``.
    """
    return _per_year_mean(stats, _per_year_summed_patients(stats))


def resolve_cohort_examinations(stats: dict, exam_data: Optional[list] = None) -> int:
    """Resolve the cohort examination count used as the per-exam denominator.

    In ``per_year_average`` mode this is the per-year mean examination count
    (see :func:`_per_year_mean_examinations`), so per-exam figures derived
    from equal-weighted per-year means divide by the matching mean
    denominator instead of the last processed year's ``exam_data``.  In any
    other mode it falls back to :func:`resolve_unique_examinations`.
    """
    mean_exams = _per_year_mean_examinations(stats)
    if mean_exams is not None:
        return mean_exams
    return resolve_unique_examinations(stats, exam_data)


def resolve_cohort_patients(stats: dict, fallback: int) -> int:
    """Resolve the cohort unique-patient count.

    In ``per_year_average`` mode this is the per-year mean unique-patient
    count (see :func:`_per_year_mean_patients`); otherwise ``fallback``
    (the single-run unique-patient count computed by the caller).
    """
    mean_patients = _per_year_mean_patients(stats)
    if mean_patients is not None:
        return mean_patients
    return fallback


def _exams_by_modality_for_summary(stats: dict, key: str = "examinations_by_modality",
                                   as_float: bool = False) -> dict:
    """Per-modality count matching the format_summary denominator.

    By default this aggregates ``examinations_by_modality``; pass ``key`` to
    aggregate a sibling count table (e.g. ``excluded_no_organ_dose_by_modality``)
    with identical per-year / per-run-year-count logic.  ``as_float`` keeps
    the per-year means as floats (for Sv tables such as
    ``excluded_dose_sv_by_modality``) instead of rounding to int.

    In ``per_year_average`` mode this sums each year's ``key`` across
    ``_year_batches`` and divides by the number of years (the per-year mean),
    mirroring :func:`resolve_cohort_examinations`.  In a ``combined`` batch
    (one pooled run over multiple year groups) it divides the top-level
    ``key`` by ``_run_year_count`` when that is > 1.  Otherwise it returns
    ``stats[key]`` as-is.
    """
    _cast = (lambda v: float(v)) if as_float else (lambda v: int(round(v)))
    batches = _year_batches_of(stats)
    if batches:
        n_yrs = len(batches) or 1
        tot: Counter = Counter()
        for ys in batches.values():
            for m, v in (ys.get(key) or {}).items():
                tot[m] += float(v or 0) if as_float else int(v or 0)
        return {m: _cast(v / n_yrs) for m, v in tot.items()}
    ebm = dict(stats.get(key) or {})
    n_yrs = _run_year_count(stats)
    if n_yrs > 1:
        ebm = {m: _cast(v / n_yrs) for m, v in ebm.items()}
    elif as_float:
        ebm = {m: float(v or 0) for m, v in ebm.items()}
    return ebm


# Flat ``{modality: n}`` stats keys of the source-drop exclusion ledger
# (main.run_simulation; see participant_flow.py).  Absent when the run has no ledger.
_EXCLUDED_LEDGER_KEYS = ("excluded_non_dose_by_modality",
                         "excluded_no_dose_model_by_modality")



def _modelled_by_modality(stats: dict, iterations: int) -> Optional[dict]:
    """Per-modality risk-modelled exam counts and dose (per iteration).

    Returns ``{mod: {"exams", "dose_sv", "excl_nondose", "excl_no_model"}}``
    for every modality in ``modality_stats`` that carries the ``scans_model``
    key (i.e. results from a vectorized run that built the modelled mask).
    Returns ``None`` when no modality has the key, so callers fall back to
    the all-exams / dose-masked behaviour (old results, synthetic test dicts).

    * ``exams``      = ``scans_model / iterations`` (modelled exam count)
    * ``dose_sv``    = ``dose_sv_model / iterations`` (modelled collective dose)
    * ``excl_nondose`` = ``(scans - scans_dose) / iterations`` (Non-Dose rows)
    * ``excl_no_model`` = ``(scans_dose - scans_model) / iterations`` (dose
      present but no cancer-dose model -- e.g. non-dental CBCT, NM/PT, male MG)

    Every exam in the simulation is
    risk-modelled (participant_flow.py), so the two mask-derived exclusion
    counts are 0; :func:`_modality_dose_table` then takes the exclusion
    counts from the ledger keys instead.
    """
    ms = stats.get("modality_stats") or {}
    iter_f = float(iterations) if iterations > 0 else 1.0
    out: dict = {}
    for mod, mv in ms.items():
        if "scans_model" not in mv:
            continue
        scans = float(mv.get("scans", 0) or 0) / iter_f
        scans_dose = float(mv.get("scans_dose", 0) or 0) / iter_f
        scans_model = float(mv.get("scans_model", 0) or 0) / iter_f
        dose_sv_model = float(mv.get("dose_sv_model", 0.0) or 0.0) / iter_f
        out[mod] = {
            "exams": int(round(scans_model)),
            "dose_sv": dose_sv_model,
            "excl_nondose": int(round(scans - scans_dose)),
            "excl_no_model": int(round(scans_dose - scans_model)),
        }
    return out if out else None


def resolve_modelled_examinations(stats: dict, iterations: int, fallback: int) -> int:
    """Total risk-modelled examination count (per iteration), or ``fallback``.

    Sums the per-modality modelled counts from :func:`_modelled_by_modality`;
    returns ``fallback`` (the all-exams denominator) when the helper is None
    (old results / synthetic dicts without ``scans_model``).
    """
    mm = _modelled_by_modality(stats, iterations)
    if mm is None:
        return fallback
    return int(sum(v["exams"] for v in mm.values()))


def _modelled_dose_sv(stats: dict, iterations: int, fallback: float) -> float:
    """Total risk-modelled collective dose per iteration (Sv), or ``fallback``."""
    mm = _modelled_by_modality(stats, iterations)
    if mm is None:
        return fallback
    return float(sum(v["dose_sv"] for v in mm.values()))


def _modality_dose_table(stats: dict, iterations: int) -> List[dict]:
    """One row per modality for the collective-dose table (section 2).

    Each row: ``{"modality", "exams", "dose_sv", "share_pct", "msv_per_exam",
    "excl_dropped", "excl_nondose", "excl_no_model", "excluded",
    "dose_excl_sv"}``.

    When ``_modelled_by_modality`` returns a table (vectorized runs with the
    modelled mask), ``exams`` / ``dose_sv`` are the *risk-modelled* exam count
    and collective dose (per iteration); ``excluded`` breaks that down into
    exams dropped upstream (no organ dose, from
    ``excluded_no_organ_dose_by_modality``), Non-Dose rows, and dose-bearing
    exams with no cancer-dose model.  The latter two come from the
    source-drop ledger (``excluded_non_dose_by_modality`` /
    ``excluded_no_dose_model_by_modality``, per-year means) when the run
    carries it -- those exams are absent from the simulation -- and
    from the ``scans - scans_model`` mask arithmetic otherwise.
    ``dose_excl_sv`` is the dose carried by the excluded exams (ledger
    ``excluded_dose_sv_by_modality`` plus ``dose_sv/it - dose_sv_model/it``).
    Mean dose/exam is ``n/a`` (rendered as 0.0 by the caller's choice) when
    ``exams == 0`` but the row is kept when ``excluded > 0`` (e.g. NM/PT
    fully excluded).

    When the helper is None (results without the ledger / synthetic test
    dicts), rows fall back to the all-exams count and dose-masked dose,
    with the ledger exclusions only (zero without a ledger).
    Sorted by collective dose desc.
    """
    ms = stats.get("modality_stats") or {}
    exams_by_mod = _exams_by_modality_for_summary(stats)
    excluded_dropped_by_mod = _exams_by_modality_for_summary(
        stats, key="excluded_no_organ_dose_by_modality"
    )
    ledger_nondose_by_mod = _exams_by_modality_for_summary(
        stats, key="excluded_non_dose_by_modality")
    ledger_no_model_by_mod = _exams_by_modality_for_summary(
        stats, key="excluded_no_dose_model_by_modality")
    ledger_dose_by_mod = _exams_by_modality_for_summary(
        stats, key="excluded_dose_sv_by_modality", as_float=True)
    iter_f = float(iterations) if iterations > 0 else 1.0
    modelled = _modelled_by_modality(stats, iterations)
    rows = []
    total_dose = 0.0
    # Union of modalities from modality_stats and the exams/excluded tables.
    _all_mods = list(ms.keys())
    for _tbl in (exams_by_mod, excluded_dropped_by_mod,
                 ledger_nondose_by_mod, ledger_no_model_by_mod):
        for _m in _tbl:
            if _m not in _all_mods:
                _all_mods.append(_m)
    for mod in _all_mods:
        mv = ms.get(mod, {})
        if modelled is not None and mod in modelled:
            mm = modelled[mod]
            exams = int(mm["exams"])
            dose_sv_iter = float(mm["dose_sv"])
            excl_nondose = int(mm["excl_nondose"])
            excl_no_model = int(mm["excl_no_model"])
            dose_sv_all_iter = float(mv.get("dose_sv", 0.0) or 0.0) / iter_f
            dose_excl_sv = max(0.0, dose_sv_all_iter - dose_sv_iter)
        else:
            dose_sv_iter = float(mv.get("dose_sv", 0.0) or 0.0) / iter_f
            exams = int(exams_by_mod.get(mod, 0) or 0)
            excl_nondose = 0
            excl_no_model = 0
            dose_excl_sv = 0.0
        # Source-drop ledger: exams removed before the simulation, so they
        # are invisible to the mask arithmetic above and must be added.
        excl_nondose += int(ledger_nondose_by_mod.get(mod, 0) or 0)
        excl_no_model += int(ledger_no_model_by_mod.get(mod, 0) or 0)
        dose_excl_sv += float(ledger_dose_by_mod.get(mod, 0.0) or 0.0)
        excl_dropped = int(excluded_dropped_by_mod.get(mod, 0) or 0)
        excluded = excl_dropped + excl_nondose + excl_no_model
        if exams == 0 and excluded == 0 and dose_sv_iter == 0.0:
            continue
        rows.append({
            "modality": mod, "exams": exams, "dose_sv": dose_sv_iter,
            "excl_dropped": excl_dropped, "excl_nondose": excl_nondose,
            "excl_no_model": excl_no_model, "excluded": excluded,
            "dose_excl_sv": dose_excl_sv,
        })
        total_dose += dose_sv_iter
    rows.sort(key=lambda r: -r["dose_sv"])
    for r in rows:
        r["share_pct"] = (r["dose_sv"] / total_dose * 100.0) if total_dose > 0 else 0.0
        r["msv_per_exam"] = (r["dose_sv"] * 1000.0 / r["exams"]) if r["exams"] > 0 else 0.0
    return rows



def _render_population_burden_table(pop: dict, filter_note: str,
                                    per_exam_notes: Optional[List[str]] = None,
                                    background_msv_per_year: float = BACKGROUND_MSV_PER_YEAR) -> List[str]:
    """Render section 1 from :func:`_population_burden_by_modality`'s output
    as fixed-width lines (3-space indent): population, annual burden and
    per-exam lifetime risk sub-tables, Total row last in each."""
    rows = pop["rows"]
    total = pop["total"]
    flags = pop["note_flags"]

    def _sex_cells(sex) -> Tuple[str, str, str]:
        if sex is None:
            return ("n/a",) * 3
        f, m, u = (float(v) for v in sex)
        denom = f + m + u
        if denom <= 0:
            return ("n/a",) * 3
        return tuple(f"{v:,.0f} ({100 * v / denom:.1f})" for v in (f, m, u))

    def _age_cell(age):
        if age is None:
            return "n/a"
        q1, med, q3 = age
        return f"{med:.0f} ({q1:.0f}-{q3:.0f})"

    def _val_cell(v, rng, count=True):
        """Value with its optional range: counts (cancers / fatal) keep
        integer range endpoints once the upper bound reaches 10, QALYs keep
        one decimal below 100."""
        if v is None:
            return "n/a"
        s = f"{v:.1f}" if v >= 10 else f"{v:.2f}"
        if rng is not None:
            lo, hi = max(0.0, rng[0]), rng[1]
            rf = (".0f" if hi >= 10 else ".1f") if count else (".1f" if hi < 100 else ".0f")
            s += f" ({lo:{rf}}-{hi:{rf}})"
        return s

    def _qps_cell(qps):
        return f"{qps:.2f}" if qps is not None else "n/a"

    def _mod_label(mod, width):
        s = str(mod)
        return s if len(s) <= width else s[:width - 1] + "\u2026"

    def _n_pct(v, tot, fmt=",.0f"):
        """``value (percent of tot)``; ``n/a`` for a missing value."""
        if v is None:
            return "n/a"
        pct = f" ({100.0 * v / tot:.1f})" if tot else ""
        return f"{v:{fmt}}{pct}"

    # Sub-table 1: population / sex / age / collective dose.  Widths:
    # Modality 14 / Exams 16 / Patients 16 / Female 16 / Male 16 /
    # Unknown 14 / Age 15 / Collective dose 18 = 125 (+3 indent).
    def _row1(label, r):
        f_s, m_s, u_s = _sex_cells(r["sex"])
        return (f"   {label:<14}{_n_pct(r['exams'], total['exams']):>16}"
                f"{_n_pct(r['patients'], total['patients']):>16}"
                f"{f_s:>16}{m_s:>16}{u_s:>14}{_age_cell(r['age']):>15}"
                f"{_n_pct(r['dose_sv'], total['dose_sv'], ',.1f'):>18}")

    r1 = "   " + "-" * 125
    t1_lines = [
        f"   {'':<14}{'':>16}{'':>16}{'':>16}{'':>16}{'':>14}{'':>15}"
        f"{'Collective dose':>18}",
        f"   {'Modality':<14}{'Exams/yr (%)':>16}{'Patients (%)':>16}"
        f"{'Female (%)':>16}{'Male (%)':>16}{'Unknown (%)':>14}"
        f"{'Age med (IQR)':>15}{'person Sv/yr (%)':>18}",
        r1,
    ]
    t1_lines.extend(_row1(_mod_label(r["modality"], 14), r) for r in rows)
    t1_lines.append(r1)
    t1_lines.append(_row1("Total", total))
    t1_lines.append("")

    # Sub-table 2: burden.  Widths: Modality 12 / Cancers 20 / Fatal 18 /
    # QALYs 22 / QALYs per Sv 16 = 88 (+3 indent).
    def _row2(label, r):
        return (f"   {label:<12}"
                f"{_val_cell(r['cases'], r['cases_rng']):>20}"
                f"{_val_cell(r['deaths'], r['deaths_rng']):>18}"
                f"{_val_cell(r['qaly'], r['qaly_rng'], count=False):>22}"
                f"{_qps_cell(r['qaly_per_sv']):>16}")

    r2 = "   " + "-" * 88
    t2_lines = [
        f"   {'Modality':<12}{'Cancers/yr':>20}"
        f"{'Fatal/yr':>18}{'QALYs lost/yr':>22}{'QALYs/person Sv':>16}",
        r2,
    ]
    t2_lines.extend(_row2(_mod_label(r["modality"], 12), r) for r in rows)
    t2_lines.append(r2)
    t2_lines.append(_row2("Total", total))
    t2_lines.append("")

    # Sub-table 3: per-exam lifetime risk and dose equivalents.  Widths:
    # Modality 14 / mSv 10 / Cancer 18 / Fatal 18 / Cases per 10k 11 /
    # Chest X-rays 14 / Background 14 = 99 (+3 indent).
    def _one_in(n, exams):
        if n is None or not exams:
            return "n/a"
        rate = n / exams
        return _format_one_in(rate) if rate > 0 else "none observed"

    def _row3(label, r):
        ex = r["exams"]
        msv_v = r["dose_sv"] * 1000.0 / ex if ex else None
        msv = f"{msv_v:.2f}" if msv_v is not None else "n/a"
        c10k = (f"{r['cases'] / ex * 10_000.0:.2f}"
                if ex and r["cases"] is not None else "n/a")
        cxr = _format_chest_xrays(msv_v) if msv_v is not None else "n/a"
        bg = (_format_background_equiv(msv_v, background_msv_per_year)
              if msv_v is not None else "n/a")
        return (f"   {label:<14}{msv:>10}{_one_in(r['cases'], ex):>18}"
                f"{_one_in(r['deaths'], ex):>18}{c10k:>11}{cxr:>14}{bg:>14}")

    r3 = "   " + "-" * 99
    t3_lines = [
        f"   {'Modality':<14}{'mSv/exam':>10}{'Cancer (1 in N)':>18}"
        f"{'Fatal (1 in N)':>18}{'Cases/10k':>11}{'Chest X-rays':>14}"
        f"{'Background':>14}",
        r3,
    ]
    t3_lines.extend(_row3(_mod_label(r["modality"], 14), r) for r in rows)
    t3_lines.append(r3)
    t3_lines.append(_row3("Average", total))
    t3_lines.extend(per_exam_notes or [])

    if flags.get("any_na"):
        t3_lines.append(
            "   n/a: not available until the run completes (sex/age, QALYs by "
            "modality and ranges need the final event data).")
    if filter_note:
        t3_lines.append(
            "   All cancer types are counted here; the cancer-type filter "
            "applies to the later sections.")
    return t1_lines + t2_lines + t3_lines


def _population_burden_by_modality(stats, iterations, exam_columns=None,
                                   event_columns=None, iter_cases_list=None,
                                   iter_deaths_list=None) -> Optional[dict]:
    """Section-1 population + annual burden table, one row per modality.

    Returns ``None`` when no modality has risk-modelled exams (the table is
    not emitted at all); otherwise ``{"rows": [...], "total": row,
    "note_flags": {...}}``.  See the section-1 rendering in
    :func:`format_summary` for the row schema and the per-cell formatting.

    Sources are read defensively: every key may be missing (synthetic test
    dicts carry almost nothing).  Sex/age come from the per-year-averaged
    Region x Age sweep in batch mode (``_rq_sweep_det_avg``) or from
    ``exam_columns`` for a single/combined run; the cancer-type visibility
    filter is NOT applied here.
    """
    rows = [r for r in _modality_dose_table(stats, iterations) if r["exams"] > 0]
    if not rows:
        return None
    iter_f = float(iterations) if iterations > 0 else 1.0
    ms = stats.get("modality_stats") or {}

    # Patients per modality (raw modality keying, same as
    # build_patients_by_modality).
    pbm = stats.get("_patients_by_modality") or {}
    sets = pbm.get("sets") or {}
    missing = pbm.get("missing") or {}
    row_mods = [r["modality"] for r in rows]

    def _patients_for(mod):
        if mod in sets or mod in missing:
            return len(sets.get(mod, ())) + int(missing.get(mod, 0))
        return None

    # Sex / age per modality.
    per_year = stats.get("_run_mode") == "per_year_average"
    sex_by_mod: Dict[str, Optional[np.ndarray]] = {}
    age_hist_by_mod: Dict[str, Optional[np.ndarray]] = {}
    age_arrs_by_mod: Dict[str, Optional[np.ndarray]] = {}
    if per_year:
        # Lazy import: research_questions imports this module at top level.
        from .research_questions import SEX_COUNT_INDEX as _sex_idx
        _fmu = [_sex_idx["F"], _sex_idx["M"], _sex_idx["U"]]
        sweep = stats.get("_rq_sweep_det_avg") or {}
        for cell in sweep.get("per_region_age") or []:
            mod = cell.get("modality")
            if mod is None:
                continue
            sc = cell.get("sex_counts_per_year")
            if sc is not None:
                s = np.asarray(sc, dtype=float)[_fmu]
                sex_by_mod[mod] = (sex_by_mod[mod] + s
                                   if mod in sex_by_mod else s.copy())
            h = cell.get("age_hist_per_year")
            if h is not None:
                a = np.asarray(h, dtype=float)
                age_hist_by_mod[mod] = (age_hist_by_mod[mod] + a
                                        if mod in age_hist_by_mod else a.copy())
    elif exam_columns is not None and len(exam_columns):
        em = np.asarray(exam_columns.modality, dtype=object)
        sx = np.asarray(exam_columns.sex_code, dtype=object)
        ag = np.asarray(exam_columns.age, dtype=np.float64)
        for mod in row_mods:
            mask = (em == mod)
            if not mask.any():
                continue
            sexs = sx[mask]
            f = int(np.sum(sexs == "F"))
            m = int(np.sum(sexs == "M"))
            u = int(mask.sum()) - f - m
            sex_by_mod[mod] = np.array([f, m, u], dtype=float)
            ages = ag[mask]
            ages = ages[~np.isnan(ages)]
            if ages.size:
                age_arrs_by_mod[mod] = np.floor(ages)

    def _age_quantiles_from_hist(h):
        if h is None:
            return None
        total = float(h.sum())
        if total <= 0:
            return None
        cum = np.cumsum(h)
        return tuple(
            float(np.searchsorted(cum, p * cum[-1], side="left"))
            for p in (0.25, 0.5, 0.75))

    def _age_quantiles_from_arr(a):
        if a is None or a.size == 0:
            return None
        return tuple(float(v) for v in np.percentile(
            a, [25, 50, 75], method="inverted_cdf"))

    # Per-modality series for ranges + QALY mean.
    any_range = False
    any_na = False
    out_rows = []
    for r in rows:
        mod = r["modality"]
        mod_stat = ms.get(mod, {}) or {}
        cases = (float(mod_stat.get("cases", 0) or 0) / iter_f) if iter_f else None
        deaths = (float(mod_stat.get("deaths", 0) or 0) / iter_f) if iter_f else None
        ser = _simulated_annual_series(stats, [mod], events=event_columns)
        cases_rng = None
        deaths_rng = None
        qaly = None
        qaly_rng = None
        if ser["cases"] is not None and len(ser["cases"]):
            cases_rng = (float(np.percentile(ser["cases"], 2.5)),
                         float(np.percentile(ser["cases"], 97.5)))
            any_range = True
        if ser["deaths"] is not None and len(ser["deaths"]):
            deaths_rng = (float(np.percentile(ser["deaths"], 2.5)),
                          float(np.percentile(ser["deaths"], 97.5)))
        if ser["qaly"] is not None and len(ser["qaly"]):
            qaly = float(np.mean(ser["qaly"]))
            qaly_rng = (float(np.percentile(ser["qaly"], 2.5)),
                        float(np.percentile(ser["qaly"], 97.5)))
            any_range = True
        if cases is None or qaly is None:
            any_na = True
        sex = sex_by_mod.get(mod)
        if per_year:
            age = _age_quantiles_from_hist(age_hist_by_mod.get(mod))
        else:
            age = _age_quantiles_from_arr(age_arrs_by_mod.get(mod))
        if sex is None or age is None:
            any_na = True
        dose_sv = r["dose_sv"]
        qaly_per_sv = (qaly / dose_sv) if (qaly is not None and dose_sv > 0) else None
        out_rows.append({
            "modality": mod, "exams": r["exams"],
            "patients": _patients_for(mod),
            "sex": sex, "age": age,
            "dose_sv": dose_sv,
            "cases": cases, "cases_rng": cases_rng,
            "deaths": deaths, "deaths_rng": deaths_rng,
            "qaly": qaly, "qaly_rng": qaly_rng,
            "qaly_per_sv": qaly_per_sv,
        })

    # Total row.
    tot_exams = sum(r["exams"] for r in out_rows)
    tot_dose = sum(r["dose_sv"] for r in out_rows)
    tot_cases = (float(stats.get("cancer_cases", 0) or 0) / iter_f) if iter_f else None
    tot_deaths = (float(stats.get("cancer_deaths", 0) or 0) / iter_f) if iter_f else None
    tot_qaly = (float(stats.get("QALY_lost", 0) or 0) / iter_f) if iter_f else None
    tot_cases_rng = tot_deaths_rng = tot_qaly_rng = None
    if (iter_cases_list and iter_deaths_list
            and len(iter_cases_list) >= 2 and len(iter_deaths_list) >= 2):
        rng = simulated_annual_range(
            stats, iter_cases=iter_cases_list, iter_deaths=iter_deaths_list,
            iter_qaly=stats.get("iter_QALY_list"))
        tot_cases_rng = rng.get("cases")
        tot_deaths_rng = rng.get("deaths")
        tot_qaly_rng = rng.get("qaly")
        if any(v is not None for v in rng.values()):
            any_range = True
    if tot_cases is None or tot_qaly is None:
        any_na = True
    # Total sex: elementwise sum over rows that have it.
    tot_sex = None
    for r in out_rows:
        if r["sex"] is None:
            continue
        tot_sex = r["sex"].copy() if tot_sex is None else tot_sex + r["sex"]
    # Total age: from the summed histogram (batch) or concatenated ages.
    tot_age = None
    if per_year:
        tot_hist = None
        for mod in row_mods:
            h = age_hist_by_mod.get(mod)
            if h is None:
                continue
            tot_hist = h.copy() if tot_hist is None else tot_hist + h
        if tot_hist is not None:
            tot_age = _age_quantiles_from_hist(tot_hist)
    else:
        cat = [age_arrs_by_mod[m] for m in row_mods
               if age_arrs_by_mod.get(m) is not None]
        if cat:
            tot_age = _age_quantiles_from_arr(np.concatenate(cat))
    if tot_sex is None or tot_age is None:
        any_na = True
    # Total patients: union of sets over row modalities + sum of missing.
    tot_patients = None
    if any((m in sets or m in missing) for m in row_mods):
        union = set()
        miss_sum = 0
        for m in row_mods:
            union |= set(sets.get(m, ()))
            miss_sum += int(missing.get(m, 0))
        tot_patients = len(union) + miss_sum
    tot_qaly_per_sv = (tot_qaly / tot_dose) if (tot_qaly is not None and tot_dose > 0) else None
    total = {
        "modality": "Total", "exams": tot_exams, "patients": tot_patients,
        "sex": tot_sex, "age": tot_age, "dose_sv": tot_dose,
        "cases": tot_cases, "cases_rng": tot_cases_rng,
        "deaths": tot_deaths, "deaths_rng": tot_deaths_rng,
        "qaly": tot_qaly, "qaly_rng": tot_qaly_rng,
        "qaly_per_sv": tot_qaly_per_sv,
    }
    return {
        "rows": out_rows, "total": total,
        "note_flags": {"any_range": any_range, "any_na": any_na},
    }



def _cohort_weighted_baseline_incidence_with_n(exam_data, exam_columns=None):
    """Cohort-weighted lifetime baseline cancer incidence from ICRP 152
    Annex A (see :func:`_cohort_baseline_patient_values`), also returning
    the number of distinct patients the value was averaged over.

    Returns ``(probability, source_key, n_patients)`` or ``None``.
    """
    values = _cohort_baseline_patient_values(exam_data, exam_columns=exam_columns)
    if not values:
        return None
    return sum(values) / len(values), "icrp152", len(values)


def _cohort_baseline_patient_values(exam_data, exam_columns=None):
    """Per-patient lifetime baseline incidence list for the distinct
    patients in ``exam_data`` in ``exam_data``.
    """
    exam_columns = exam_columns or _ensure_exam_columns(exam_data)
    if exam_columns is None or len(exam_columns) == 0:
        return None
    seen: dict = {}

    patient_id = exam_columns.person_id
    sex_code = exam_columns.sex_code
    population = exam_columns.population
    for i in range(len(exam_columns)):
        pid = patient_id[i]
        if pid is None or (isinstance(pid, float) and pid != pid):  # NaN check
            # Fall back to object identity for rows that lack a stable id
            pid = i
        if pid in seen:
            continue
        sx = str(sex_code[i] or "U")
        sex = sx if sx in ("M", "F") else "U"
        pop = str(population[i])
        seen[pid] = (sex, pop)

    if not seen:
        return None
    cache: dict = {}
    values = []
    for key in seen.values():
        val = cache.get(key)
        if val is None:
            val = float(lifetime_baseline_cancer_incidence_ICRP152(key[0], key[1]))
            cache[key] = val
        values.append(val)
    return values



def _build_all_site_excess_cumsum(
    organ_doses_gy: Dict[str, float],
    age_at_exposure: float,
    sex_code: str,
    population: str,
    max_attained_age: int,
    ddref_solid: float,
    ddref_leuk: float,
) -> np.ndarray:
    """Σ_a Σ_c Δλ_c(a) cumulative excess incidence — ICRP 152 para 47.

    Sums the annual radiogenic excess incidence over every modelled
    cancer site at each site's own absorbed dose, returning an array
    indexed by attained age (length ``max_attained_age + 1``).  Used as
    the all-site dose-corrected survival decrement so every site in a
    given exam shares the same Δλ_total — the strict para 47 form
    rather than the per-site fallback inside
    ``compute_lifetime_excess_cancer_risk_REIC``.

    Bone and Skin are excluded (lifetime-not-annual risks per Table 3.1);
    Other-solid sub-sites are folded once via the multiplicity-weighted
    mean dose × Remainder kernel, matching the MC pipeline.
    """
    end_age = int(min(max_attained_age, MAX_ATTAINED_AGE_ICRP152))
    out = np.zeros(end_age + 1, dtype=float)
    if not organ_doses_gy or sex_code not in ("M", "F", "U"):
        return out
    e_int = int(max(0, min(int(age_at_exposure), end_age - 1)))
    ages_arr = np.arange(0, end_age + 1, dtype=float)
    annual = np.zeros(end_age + 1, dtype=float)

    # Multiplicity-weighted mean Other-solid (Remainder) dose.
    rem_weighted = 0.0
    rem_total_mult = float(sum(REMAINDER_ORGAN_MULTIPLICITY.values()))
    for ct in sorted(OTHER_SOLID_REMAINDER_SITES):
        if ct in _SEX_EXCLUDED_CANCER_TYPES.get(sex_code, _SEX_EXCLUDED_CANCER_TYPES["U"]):
            continue
        d = float(organ_doses_gy.get(ct, 0.0))
        if d <= 0:
            d = float(get_organ_dose_for_cancer(ct, organ_doses_gy))
        if d > 0:
            rem_weighted += d * REMAINDER_ORGAN_MULTIPLICITY.get(ct, 1)
    mean_rem_dose = rem_weighted / rem_total_mult if rem_total_mult > 0 else 0.0

    if mean_rem_dose > 0 and "Remainder" in cancer_risk_params:
        try:
            params = lookup_cancer_params("Remainder")
            baseline_rates = np.array(
                [_get_cancer_incidence_rate(int(a), sex_code, "Remainder", population) for a in ages_arr]
            )
            d_lam = _delta_lambda_vectorized(
                "Remainder", mean_rem_dose, e_int, ages_arr,
                sex_code, ddref_solid, baseline_rates, params,
            )
            annual += np.maximum(0.0, d_lam)
        except Exception:
            pass

    for ct in _iter_modeled_cancer_types(organ_doses_gy, sex_code):
        if ct in OTHER_SOLID_REMAINDER_SITES:
            continue
        if ct in ("Bone", "Skin"):
            continue  # lifetime-not-annual; not part of S(a,e|d) integral
        d = float(organ_doses_gy.get(ct, 0.0))
        if d <= 0:
            d = float(get_organ_dose_for_cancer(ct, organ_doses_gy))
        if d <= 0:
            continue
        try:
            params = lookup_cancer_params(ct)
            ddref_ct = ddref_leuk if ct == "Leukaemia" else ddref_solid
            baseline_rates = np.array(
                [_get_cancer_incidence_rate(int(a), sex_code, ct, population) for a in ages_arr]
            )
            d_lam = _delta_lambda_vectorized(
                ct, d, e_int, ages_arr, sex_code, ddref_ct, baseline_rates, params,
            )
            annual += np.maximum(0.0, d_lam)
        except Exception:
            continue

    return np.cumsum(annual)


def _deterministic_lifetime_risk_by_site(
    organ_doses_gy: Dict[str, float],
    age_at_exposure: float,
    sex_code: str,
    expected_lifespan: float,
    population: str,
    risk_model: str,
    ddref_solid: float,
    ddref_leuk: float,
) -> Dict[str, float]:
    """Compute each cancer site's expected lifetime risk for one exam
    analytically (no Monte Carlo draw).  ``organ_doses_gy`` maps organ
    name to absorbed dose in Gy; ``age_at_exposure`` and
    ``expected_lifespan`` are in years.  Returns the per-site
    expected-incidence / detriment contributions used by the
    deterministic-baseline path."""
    lifetime_risk: Dict[str, float] = {}
    if not organ_doses_gy:
        return lifetime_risk

    max_a = int(min(120, expected_lifespan))

    # ICRP 152 para 47-strict all-site cumulative excess incidence —
    # built once per exam so every per-site call shares the same
    # Δλ_total dose-corrected survival decrement.
    excess_cumsum_all = _build_all_site_excess_cumsum(
        organ_doses_gy, age_at_exposure, sex_code,
        population, max_a, ddref_solid, ddref_leuk,
    )

    rem_dose_map: Dict[str, float] = {}
    for ct in sorted(OTHER_SOLID_REMAINDER_SITES):
        if ct not in cancer_risk_params:
            continue
        if ct in _SEX_EXCLUDED_CANCER_TYPES.get(sex_code, _SEX_EXCLUDED_CANCER_TYPES["U"]):
            continue
        dose = get_organ_dose_for_cancer(ct, organ_doses_gy)
        if dose <= 0:
            dose = organ_doses_gy.get(ct, 0.0)
        if dose > 0:
            rem_dose_map[ct] = float(dose)

    if rem_dose_map:
        weighted_sum = sum(d * REMAINDER_ORGAN_MULTIPLICITY.get(ct, 1) for ct, d in rem_dose_map.items())
        total_mult_icrp = sum(REMAINDER_ORGAN_MULTIPLICITY.values())
        mean_rem_dose = weighted_sum / max(total_mult_icrp, 1)
        lifetime_risk["Remainder"] = compute_lifetime_excess_cancer_risk_REIC(
            "Remainder",
            mean_rem_dose,
            age_at_exposure,
            sex_code,
            population=population,
            max_attained_age=max_a,
            ddref=ddref_solid,
            method=risk_model,
            survival_excess_cumsum=excess_cumsum_all,
        )

    for ct in _iter_modeled_cancer_types(organ_doses_gy, sex_code):
        if ct in OTHER_SOLID_REMAINDER_SITES:
            continue
        dose = get_organ_dose_for_cancer(ct, organ_doses_gy)
        if dose <= 0:
            dose = organ_doses_gy.get(ct, 0.0)
        if dose <= 0:
            continue
        if ct in ("Bone", "Skin"):
            params = lookup_cancer_params(ct)
            w_err = float(params.get("ERR_weight", 0.0))
            w_ear = float(params.get("EAR_weight", 0.0))
            err_b = params.get("ERR_beta_base", {})
            ear_b = params.get("EAR_beta_base", {})
            if sex_code in ("M", "F"):
                beta = w_err * float(err_b.get(sex_code, 0.0)) + w_ear * float(ear_b.get(sex_code, 0.0))
            else:
                beta = (
                    w_err * 0.5 * (float(err_b.get("M", 0.0)) + float(err_b.get("F", 0.0)))
                    + w_ear * 0.5 * (float(ear_b.get("M", 0.0)) + float(ear_b.get("F", 0.0)))
                )
            lifetime_risk[ct] = max(0.0, float(dose * beta))
            continue
        ddref = ddref_leuk if ct == "Leukaemia" else ddref_solid
        lifetime_risk[ct] = compute_lifetime_excess_cancer_risk_REIC(
            ct,
            dose,
            age_at_exposure,
            sex_code,
            population=population,
            max_attained_age=max_a,
            ddref=ddref,
            method=risk_model,
            survival_excess_cumsum=excess_cumsum_all,
        )
    return lifetime_risk


def _empty_curve(n_bins: int) -> dict:
    """Return the per-bin zero arrays for an empty risk curve.

    Includes ``detriment*`` keys so every caller — including the
    early-return paths in :func:`compute_filtered_risk_curve` and
    :func:`compute_combined_bayesian_curve` — returns a dict with the
    same key set as the non-empty path, avoiding latent KeyErrors.
    """
    return {
        "risk_per_sv": np.zeros(n_bins), "risk_low": np.zeros(n_bins),
        "risk_high": np.zeros(n_bins), "death_per_sv": np.zeros(n_bins),
        "death_low": np.zeros(n_bins), "death_high": np.zeros(n_bins),
        "dose_sv": np.zeros(n_bins), "cases": np.zeros(n_bins),
        "deaths": np.zeros(n_bins),
        "detriment": np.zeros(n_bins), "detriment_per_sv": np.zeros(n_bins),
        "detriment_low": np.zeros(n_bins), "detriment_high": np.zeros(n_bins),
    }


def _empty_curve_result(bin_edges: np.ndarray, n_bins: int,
                        dose_field: str = "ed_msv") -> dict:
    """Full empty-curve dict including ``bin_edges``, ``centers``, ``n_exams``.

    Single source of truth for the early-return paths of
    :func:`compute_filtered_risk_curve` and
    :func:`compute_combined_bayesian_curve` — giving every early-return path the
    same key set.
    """
    return {
        "bin_edges": bin_edges,
        "centers": (bin_edges[:-1] + bin_edges[1:]) / 2,
        **_empty_curve(n_bins),
        "n_exams": 0,
        "dose_field": dose_field,
    }


def _apply_ci_bands(
    risk_low: np.ndarray, risk_high: np.ndarray,
    death_low: np.ndarray, death_high: np.ndarray,
    detriment_low: np.ndarray, detriment_high: np.ndarray,
    risk_per_sv: np.ndarray, death_per_sv: np.ndarray, detriment_per_sv: np.ndarray,
    c_se: np.ndarray, d_se: np.ndarray, det_se: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Fill ``*_low``/``*_high`` CI bands in place from per-bin SEs.

    Shared between :func:`compute_combined_bayesian_curve` and
    :func:`compute_filtered_risk_curve` — the 6-line ``Z_95 * se`` block
    was duplicated verbatim (with different SE array names) between them.
    The low band is clamped at 0 (rates cannot be negative).
    """
    risk_low[mask] = np.maximum(0.0, risk_per_sv[mask] - Z_95 * c_se[mask])
    risk_high[mask] = risk_per_sv[mask] + Z_95 * c_se[mask]
    death_low[mask] = np.maximum(0.0, death_per_sv[mask] - Z_95 * d_se[mask])
    death_high[mask] = death_per_sv[mask] + Z_95 * d_se[mask]
    detriment_low[mask] = np.maximum(0.0, detriment_per_sv[mask] - Z_95 * det_se[mask])
    detriment_high[mask] = detriment_per_sv[mask] + Z_95 * det_se[mask]


# Combined Bayesian Curve (single curve merging all adjustments)

# Mataac dose strata, binned on the engine's stratification variable: the
# patient's CT-only effective dose within the calendar year of the exam
# (``_mataac_index_ced_sv``).  The displayed factor is the MEAN of the
# correction factors the engine actually applied to the exams in the group,
# not a ratio of the published 10-year survivals.  Those published ratios
# would divide each group's survival by Group A's own survival (0.846),
# which silently makes Group A the reference and therefore reports numbers
# that do not match the age- and sex-resolved factors used in the run.
_MATAAC_CED_STRATA = [
    ("A: <10 mSv",    0.010),
    ("B: 10-50 mSv",  0.050),
    ("C: 50-100 mSv", 0.100),
    ("D: \u2265100 mSv",   float("inf")),
]


def _accumulate_bayesian_exam_bins(exam_data, filters, bin_width, n_iterations,
                                   dose_field="ed_msv", compare_model=False,
                                   compare_mataac=False, exam_columns=None):
    """Exam-side accumulation of :func:`compute_combined_bayesian_curve`.

    Returns ``None`` when no exams match the filters (same early-return
    condition as the original fused function).  Otherwise a dict of per-bin
    raw sums -- ``dose_by_bin`` (Sv, iteration-scaled), the dose-weighted
    frailty accumulators, per-bin exam lists for the model comparison, the
    Mataac strata ``group_stats`` -- so the batch combiner can pool them
    across years before any ratio or factor is formed.
    """
    exam_columns = exam_columns or _ensure_exam_columns(exam_data)

    exam_mask = None
    filtered_ages = filtered_dose = filtered_frailty = filtered_ced = None
    filtered_exams = None
    # Use columnar filter if available
    if exam_columns is not None:
        exam_mask = _filter_columns(exam_columns, filters)
        if not exam_mask.any():
            return None
        # Extract filtered data from columns
        filtered_ages = exam_columns.age[exam_mask]
        filtered_dose = exam_columns.ed_msv[exam_mask]
        # For model comparison or non-columnar dose fields (e.g. ed_msv_no_golikov),
        # need full exam objects (fall back to dict)
        if compare_model or dose_field != "ed_msv":
            filtered_exams = [ex for ex in exam_data if _match_record(ex, filters)]
    else:
        # Dict-based filtering path for non-columnar exam_data
        filtered_exams = [ex for ex in exam_data if _match_record(ex, filters)]

    if not filtered_exams and (exam_columns is None or not exam_mask.any()):
        return None

    bin_edges = np.arange(0, 101, bin_width)
    n_bins = len(bin_edges) - 1
    n_it = max(1, n_iterations)

    # Step 1: Dose per bin + collect per-bin metadata
    dose_by_bin = np.zeros(n_bins)
    # DOSE-weighted, not exam-count-weighted -- see _bayesian_bin_weights.
    frailty_dose_sum = np.zeros(n_bins)
    frailty_dose_weight = np.zeros(n_bins)
    bin_exam_lists = [[] for _ in range(n_bins)] if compare_model else None

    group_stats = ({g[0]: {"n": 0, "dose_sv": 0.0, "frailty_sum": 0.0}
                    for g in _MATAAC_CED_STRATA} if compare_mataac else None)

    if exam_columns is not None and dose_field == "ed_msv":
        if compare_mataac:
            _fr = getattr(exam_columns, "mataac_frailty_factor", None)
            _ced_col = getattr(exam_columns, "mataac_index_ced_sv", None)
            if _fr is None or _ced_col is None:
                raise ValueError(
                    "compare_mataac requires mataac_frailty_factor and "
                    "mataac_index_ced_sv columns on exam_columns")
            filtered_frailty = _fr[exam_mask]
            filtered_ced = _ced_col[exam_mask]
        # Columnar path — vectorised (np.add.at keeps the same deterministic
        # summation order as the per-exam Python loop it replaces).
        _ages_ex = np.asarray(filtered_ages, dtype=np.float64)
        _dose_ex = np.asarray(filtered_dose, dtype=np.float64)
        if _ages_ex.size:
            _idx_ex = np.clip((_ages_ex // bin_width).astype(np.int64), 0, n_bins - 1)
            _ed_sv = _dose_ex / 1000.0
            np.add.at(dose_by_bin, _idx_ex, _ed_sv * n_it)

            if compare_mataac:
                # Replicate the ``float(x or 1.0)`` guard: a 0.0 frailty
                # factor is treated as 1.0 (no correction).  The columnar
                # array defaults to 1.0 already, but be explicit.
                _frailty = np.asarray(filtered_frailty, dtype=np.float64)
                _frailty = np.where(_frailty == 0.0, 1.0, _frailty)
                _ced = np.asarray(filtered_ced, dtype=np.float64)
                np.add.at(frailty_dose_sum, _idx_ex, _ed_sv * _frailty)
                np.add.at(frailty_dose_weight, _idx_ex, _ed_sv)
                # Mataac group assignment via searchsorted on the thresholds.
                # _MATAAC_CED_STRATA = [(A, 0.010), (B, 0.050), (C, 0.100),
                #                       (D, inf)] → group index is the
                # number of thresholds strictly below ced.
                _thresh = np.array([t for _, t in _MATAAC_CED_STRATA[:-1]],
                                   dtype=np.float64)
                _grp = np.searchsorted(_thresh, _ced, side="right")
                for gi, (gname, _) in enumerate(_MATAAC_CED_STRATA):
                    _gmask = _grp == gi
                    if _gmask.any():
                        group_stats[gname]["n"] += int(_gmask.sum())
                        group_stats[gname]["dose_sv"] += float(_ed_sv[_gmask].sum())
                        group_stats[gname]["frailty_sum"] += float(_frailty[_gmask].sum())
    else:
        # Dict-based path (non-columnar exam_data or dose_field != "ed_msv")
        for ex in (filtered_exams or []):
            age = float(ex.get("age", 0) or 0)
            idx = min(max(int(age // bin_width), 0), n_bins - 1)
            ed_sv = float(ex.get(dose_field, ex.get("ed_msv", 0)) or 0) / 1000.0
            dose_by_bin[idx] += ed_sv * n_it

            if compare_mataac:
                frailty = float(ex.get("_mataac_frailty_factor", 1.0) or 1.0)
                frailty_dose_sum[idx] += ed_sv * frailty
                frailty_dose_weight[idx] += ed_sv
                ced = float(ex.get("_mataac_index_ced_sv", 0.0) or 0.0)
                for gname, thresh in _MATAAC_CED_STRATA:
                    if ced < thresh:
                        group_stats[gname]["n"] += 1
                        group_stats[gname]["dose_sv"] += ed_sv
                        group_stats[gname]["frailty_sum"] += frailty
                        break

            if compare_model:
                bin_exam_lists[idx].append(ex)

    n_exams = len(filtered_exams) if filtered_exams is not None else int(exam_mask.sum())
    return {
        "bin_edges": bin_edges,
        "dose_by_bin": dose_by_bin,
        "frailty_dose_sum": frailty_dose_sum,
        "frailty_dose_weight": frailty_dose_weight,
        "bin_exam_lists": bin_exam_lists,
        "group_stats": group_stats,
        "n_exams": n_exams,
    }


def _bayesian_bin_weights(exam_acc, n_bins, ddref_solid, ddref_leuk,
                          compare_model=False, compare_mataac=False,
                          selected_model="REIC", comparison_model="LAR",
                          population="Euro-American"):
    """Per-bin event weights of :func:`compute_combined_bayesian_curve`.

    Returns ``(bin_weight_solid, bin_weight_leuk, model_ratio, avg_frailty)``.
    ``model_ratio`` is all-ones unless ``compare_model`` and the accumulator
    carried per-bin exam lists (single-run only -- the per-year digest keeps
    no ``organ_doses_gy``, so the batch path never has them; the dose-weighted
    frailty un-do factor pools fine as combined numerator/denominator).
    """
    scale_solid, scale_leuk = _ddref_scales(ddref_solid, ddref_leuk)

    # Model ratio (sampled — max 50 exams/bin for speed)
    bin_exam_lists = exam_acc.get("bin_exam_lists") if exam_acc else None
    model_ratio = np.ones(n_bins)
    if compare_model and bin_exam_lists is not None:
        _MAX_SAMPLE = 50
        for i in range(n_bins):
            if not bin_exam_lists[i]:
                continue
            sample = bin_exam_lists[i]
            if len(sample) > _MAX_SAMPLE:
                sample = _rng.sample(sample, _MAX_SAMPLE)
            sel_s, cmp_s = 0.0, 0.0
            for ex in sample:
                age = float(ex.get("age", 0) or 0)
                sex_code = str(ex.get("sex_code", "U") or "U")
                remaining = float(get_expected_remaining_years(age, sex_code))
                lifespan = min(120.0, age + max(0.0, remaining))
                organ_doses = ex.get("organ_doses_gy", {}) or {}
                _ex_pop = str(ex.get("_population") or population)
                r_sel = _deterministic_lifetime_risk_by_site(
                    organ_doses, age, sex_code, lifespan, _ex_pop,
                    selected_model, ddref_solid, ddref_leuk)
                r_cmp = _deterministic_lifetime_risk_by_site(
                    organ_doses, age, sex_code, lifespan, _ex_pop,
                    comparison_model, ddref_solid, ddref_leuk)
                sel_s += sum(max(0.0, float(r)) for r in r_sel.values())
                cmp_s += sum(max(0.0, float(r)) for r in r_cmp.values())
            if sel_s > 0:
                model_ratio[i] = cmp_s / sel_s

    # Inverse frailty: DOSE-weighted, not per-exam
    # The curve is cases/dose, and the expected cases an exam contributes go
    # like dose_i x frailty_i, so undoing the correction in a bin needs
    #     sum(dose_i) / sum(dose_i x frailty_i)  =  1 / dose-weighted mean.
    # Averaging the factors over exam COUNTS instead answers a question nobody
    # asked ("what if the average exam were unadjusted") and is badly biased
    # low here, because frailty and dose are the same variable twice: only
    # heavily imaged patients are adjusted at all, so a bin's many 1.0-factor
    # low-dose exams drown out the few 0.48-factor exams that produced most of
    # its dose and most of its events.  On a bin of 200 x 1 mSv unadjusted
    # exams plus 20 x 30 mSv group-D exams, the count mean is 0.953 (a 5%
    # correction) where the dose-weighted mean is 0.610 (a 64% correction) --
    # the overlay recovered a thirteenth of the effect it claims to show.
    inv_frailty = np.ones(n_bins)
    avg_frailty = np.ones(n_bins)
    if compare_mataac and exam_acc is not None:
        fd_sum = exam_acc["frailty_dose_sum"]
        fd_weight = exam_acc["frailty_dose_weight"]
        mask_f = fd_weight > 0
        avg_frailty[mask_f] = fd_sum[mask_f] / fd_weight[mask_f]
        inv_frailty[mask_f] = 1.0 / np.maximum(avg_frailty[mask_f], 0.01)

    # Per-event DDREF (solid vs leuk), pre-compute other factors
    other_factors = model_ratio * inv_frailty
    return scale_solid * other_factors, scale_leuk * other_factors, model_ratio, avg_frailty


def _accumulate_bayesian_event_bins(event_columns, filters, bin_width,
                                    bin_weight_solid, bin_weight_leuk,
                                    sampled_yll: bool = False):
    """Event-side accumulation of :func:`compute_combined_bayesian_curve`.

    Each event contributes its per-bin weight (DDREF scale x model ratio x
    inverse frailty).  Returns the four raw per-bin sums so the batch
    combiner can pool them across years before forming ratios.
    """
    bin_edges = np.arange(0, 101, bin_width)
    n_bins = len(bin_edges) - 1
    cases_by_bin = np.zeros(n_bins)
    deaths_by_bin = np.zeros(n_bins)
    detriment_by_bin = np.zeros(n_bins)
    detriment_sq_by_bin = np.zeros(n_bins)

    # Vectorised: ``np.add.at`` applies duplicate-index additions in array
    # order, so per-bin float sums have a deterministic summation order.
    _ev_ages = None
    _ev_types = None
    _ev_died = None
    filtered = _filter_events(event_columns, filters)
    if filtered is not None:
        _ev_ages = np.asarray(filtered["age_at_exposure"], dtype=np.float64)
        _ev_types = [str(t) if t else "" for t in filtered["cancer_type"]]
        _ev_died = np.asarray(filtered["died"]).astype(bool)

    if _ev_ages is not None and _ev_ages.size:
        _idx_ev = np.clip((_ev_ages // bin_width).astype(np.int64), 0, n_bins - 1)
        _types_arr = np.asarray(_ev_types, dtype=object)
        _uniq, _inv = np.unique(_types_arr, return_inverse=True)
        _leuk_uniq = np.array([_is_leukaemia_type(u) for u in _uniq], dtype=bool)
        _leuk_arr = _leuk_uniq[_inv]
        _w = np.where(_leuk_arr, bin_weight_leuk[_idx_ev], bin_weight_solid[_idx_ev])
        _detw = _event_detriment_weights(filtered, _uniq, _inv, sampled_yll) * _w
        np.add.at(cases_by_bin, _idx_ev, _w)
        np.add.at(deaths_by_bin, _idx_ev, np.where(_ev_died, _w, 0.0))
        np.add.at(detriment_by_bin, _idx_ev, _detw)
        np.add.at(detriment_sq_by_bin, _idx_ev, _detw * _detw)

    return {
        "cases_by_bin": cases_by_bin,
        "deaths_by_bin": deaths_by_bin,
        "detriment_by_bin": detriment_by_bin,
        "detriment_sq_by_bin": detriment_sq_by_bin,
    }


def _mataac_group_rows(group_stats) -> list:
    """Render the per-stratum ``group_stats`` dict as dose-group rows."""
    dg = []
    for gname, _thresh in _MATAAC_CED_STRATA:
        gs = (group_stats or {}).get(gname)
        if gs and gs["n"] > 0:
            dg.append({"group": gname, "n_exams": gs["n"],
                       "dose_sv": gs["dose_sv"],
                       "frailty_factor": min(1.0, gs["frailty_sum"] / gs["n"])})
    return dg


def compute_combined_bayesian_curve(
    exam_data: list,
    events: list,
    n_iterations: int,
    filters: dict,
    bin_width: int = 5,
    ddref_solid: float = 2.0,
    ddref_leuk: float = 1.0,
    use_no_golikov_dose: bool = False,
    compare_model: bool = False,
    selected_model: str = "REIC",
    comparison_model: str = "LAR",
    population: str = "Euro-American",
    compare_mataac: bool = False,
    exam_columns=None,
    event_columns=None,
    sampled_yll: bool = False,
) -> Dict:
    """Single combined Bayesian curve incorporating all active adjustments.

    Starts from actual simulation events and applies:
    - DDREF scaling (always; equals 1.0 at defaults solid=2.0, leuk=1.0)
    - Model ratio per age bin (if compare_model; sampled for speed)
    - Inverse frailty per age bin (if compare_mataac)
    - Different dose field (if use_no_golikov_dose)

    Parameters
    ----------
    exam_data : list
        List of exam dicts (the non-columnar input form)
    events : list
        List of cancer event dicts (the non-columnar input form)
    n_iterations : int
        Number of Monte Carlo iterations
    filters : dict
        Filter criteria
    bin_width : int
        Age bin width
    ddref_solid : float
        DDREF for solid cancers
    ddref_leuk : float
        DDREF for leukemia
    use_no_golikov_dose : bool
        Use ed_msv_no_golikov instead of ed_msv
    compare_model : bool
        Compare risk models
    selected_model : str
        Selected risk model
    comparison_model : str
        Comparison risk model
    population : str
        Population for risk calculations
    compare_mataac : bool
        Apply MATAAC frailty correction
    exam_columns : ExamColumns, optional
        Columnar exam data for efficient processing
    event_columns : EventColumns, optional
        Columnar event data for efficient processing

    Returns
    -------
    dict
        Combined Bayesian curve data
    """
    dose_field = "ed_msv_no_golikov" if use_no_golikov_dose else "ed_msv"
    event_columns = event_columns or _ensure_event_columns(events)
    # Sampled-YLL detriment needs the event QALY_lost column (present in
    # QALY_mode="full"); digests without it fall back to the l_rel
    # factor and report ``detriment_yll_fallback``.
    _yll_ok = getattr(event_columns, "QALY_lost", None) is not None
    _eff_sampled = bool(sampled_yll and _yll_ok)
    exam_acc = _accumulate_bayesian_exam_bins(
        exam_data, filters, bin_width, n_iterations,
        dose_field=dose_field, compare_model=compare_model,
        compare_mataac=compare_mataac, exam_columns=exam_columns)
    bin_edges = np.arange(0, 101, bin_width)
    n_bins = len(bin_edges) - 1
    if exam_acc is None:
        out = _empty_curve_result(bin_edges, n_bins)
        out["detriment_yll_basis"] = "sampled" if _eff_sampled else "icrp_average"
        if sampled_yll and not _eff_sampled:
            out["detriment_yll_fallback"] = True
        return out

    w_solid, w_leuk, _model_ratio, avg_frailty = _bayesian_bin_weights(
        exam_acc, n_bins, ddref_solid, ddref_leuk,
        compare_model=compare_model, compare_mataac=compare_mataac,
        selected_model=selected_model, comparison_model=comparison_model,
        population=population)

    ev = _accumulate_bayesian_event_bins(
        event_columns, filters, bin_width, w_solid, w_leuk,
        sampled_yll=_eff_sampled)

    result = _finalise_curve(
        bin_edges, exam_acc["dose_by_bin"], ev["cases_by_bin"],
        ev["deaths_by_bin"], ev["detriment_by_bin"], ev["detriment_sq_by_bin"],
        exam_acc["n_exams"], dose_field=dose_field)
    result["detriment_yll_basis"] = "sampled" if _eff_sampled else "icrp_average"
    if sampled_yll and not _eff_sampled:
        result["detriment_yll_fallback"] = True

    if compare_mataac and exam_acc["group_stats"]:
        result["mataac_dose_groups"] = _mataac_group_rows(exam_acc["group_stats"])
        result["avg_frailty_by_bin"] = avg_frailty

    return result


# Region-by-Modality Analysis

def region_by_modality(exam_data: list, exam_columns=None) -> Dict[str, Dict[str, int]]:
    """Return {modality: {region: count}} from precomputed exam data.

    Parameters
    ----------
    exam_data : list
        List of exam dicts (converted to ExamColumns internally), **or** a
        :class:`file_cohort.FileCohort` whose ``mod_region`` count tables
        are returned directly without per-exam expansion.
    exam_columns : ExamColumns, optional
        Columnar exam data for efficient processing

    Returns
    -------
    dict
        {modality: {region: count}}
    """
    # FileCohort shortcut: the count tables already hold modality x region.
    if hasattr(exam_data, "mod_region_dict") and callable(getattr(exam_data, "mod_region_dict")):
        return exam_data.mod_region_dict()

    out: Dict[str, Dict[str, int]] = {}

    exam_columns = exam_columns or _ensure_exam_columns(exam_data)
    if exam_columns is None:
        return out

    from .region_resolver import radiography_reporting_region as _rpt_reg
    modality = exam_columns.modality
    region = exam_columns.region
    for i in range(len(exam_columns)):
        mod = str(modality[i] or "Unknown")
        raw_reg = str(region[i] or "Unknown")
        reg = _rpt_reg(raw_reg)
        out.setdefault(mod, {})
        out[mod][reg] = out[mod].get(reg, 0) + 1
    return out


def _is_dental_region(region) -> bool:
    r = str(region or "").strip().lower()
    return r.startswith("dental") or r in ("kefalometry", "intraoral")


def _is_dental_imaging_subtype(region) -> bool:
    """True for the cross-modality "Dental imaging" subtypes shown in the
    shared dental panel (Panoramic, Kefalometry, Intraoral, Dental CBCT).

    Unlike :func:`_is_dental_region`, callers use this only to decide what
    also gets *counted* in the shared dental panel -- "Dental CBCT" stays
    True here, but is additionally kept in CBCT's own by-body-region panel
    (see :func:`plot_all_regions_by_modality`), since it is real CBCT
    modality data with its own dose model there.
    """
    r = str(region or "").strip().lower()
    return r.startswith("dental") or r in ("kefalometry", "intraoral")


# Display labels for the dental subtypes shown in the dental panel.
_DENTAL_SUBTYPE_DISPLAY = {
    "Dental Panoramic": "Panoramic",
    "Dental CBCT": "CBCT",
    "Kefalometry": "Kefalometry",
    "Dental_intraoral": "Intraoral",
    "Dental": "Dental (other)",
}


def plot_region_by_modality(ax, rbm: Dict[str, Dict[str, int]], modality: str,
                             no_risk_regions: Optional[set] = None):
    """Bar chart of region counts for a single modality.

    ``no_risk_regions``, when given, marks the listed region labels with a
    trailing "*" and adds a footnote -- used for CBCT, where only "Dental
    CBCT" (HEAD/SKULL) has a dose model; every other body part reached by
    the same dental cone-beam units (Ankle, Wrist, ...) is real anatomy with
    no risk calculated.
    """
    data = rbm.get(modality, {})
    if not data:
        ax.set_title(f"{modality_name(modality)}: No data")
        return
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:15]
    regions = [r for r, _ in sorted_items]
    counts = [c for _, c in sorted_items]
    y_pos = range(len(regions))
    ax.barh(y_pos, counts, color=_PALETTE_LIST[0], alpha=0.85)
    ax.set_yticks(y_pos)
    if no_risk_regions:
        labels = [f"{r} *" if r in no_risk_regions else r for r in regions]
    else:
        labels = regions
    ax.set_yticklabels(labels, fontsize=8)
    xlabel = "Count (1 iteration)"
    if no_risk_regions:
        xlabel += "  (* = no dose model, risk not calculated)"
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(f"{modality_name(modality)}: Regions", fontsize=10)
    ax.invert_yaxis()


def plot_dental_by_subtype(ax, dental_counts: Dict[str, int]):
    """Bar chart of dental-imaging exam counts per subtype.

    Dental imaging is a single cross-modality category, so its panel breaks
    the dental total down by subtype (Dental Panoramic, Dental CBCT,
    Kefalometry, Dental_intraoral, …) on the y-axis rather than by modality
    or anatomical region.
    """
    if not dental_counts:
        ax.set_title("Dental imaging: No data")
        return
    sorted_items = sorted(dental_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [_DENTAL_SUBTYPE_DISPLAY.get(k, k) for k, _ in sorted_items]
    counts = [c for _, c in sorted_items]
    total = sum(dental_counts.values())
    y_pos = range(len(labels))
    ax.barh(y_pos, counts, color=_PALETTE_LIST[1 % len(_PALETTE_LIST)], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Count (1 iteration)")
    ax.set_title(f"Dental imaging: by subtype ({total:,} total)", fontsize=10)
    ax.invert_yaxis()


def plot_all_regions_by_modality(fig, rbm: Dict[str, Dict[str, int]]):
    """Create subplots for each modality with region distribution.

    The cross-modality dental-imaging subtypes (Panoramic, Kefalometry,
    Intraoral, Dental CBCT) are pulled out of their raw modality panel and
    shown together in a single dedicated panel broken down by subtype.
    "Dental CBCT" is additionally kept in CBCT's own by-body-region panel
    -- it already has a real dose model there -- alongside the other,
    unmodelled CBCT body parts (Ankle, Wrist, ...), which are marked with
    "*" in that panel. So "Dental CBCT" appears in both panels; every other
    dental subtype appears only in the shared dental panel.
    """
    fig.clear()
    # Split cross-modality dental-imaging subtypes out of the per-modality
    # region dicts into the shared dental panel. "Dental CBCT" is *also*
    # kept in its raw modality's (CBCT's) own dict, so it shows in both.
    non_dental: Dict[str, Dict[str, int]] = {}
    dental_by_subtype: Dict[str, int] = {}
    for mod, regions in rbm.items():
        kept = {}
        for reg, cnt in (regions or {}).items():
            is_dental = _is_dental_imaging_subtype(reg)
            if is_dental:
                dental_by_subtype[reg] = dental_by_subtype.get(reg, 0) + cnt
            if not is_dental or str(reg).strip().lower() == "dental cbct":
                kept[reg] = cnt
        if kept:
            # Fold rare multi-region combos (e.g. "Neck, Pelvis" with <30
            # scans) into "Other combinations" so the bar chart isn't
            # cluttered with n=1-13 combo labels.
            non_dental[mod] = _fold_rare_region_combos(kept)

    panels = [("region", mod) for mod in sorted(non_dental.keys())]
    if dental_by_subtype:
        panels.append(("dental", None))

    n = len(panels)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    for i, (kind, mod) in enumerate(panels):
        ax = fig.add_subplot(rows, cols, i + 1)
        if kind == "dental":
            plot_dental_by_subtype(ax, dental_by_subtype)
        elif str(mod).strip().upper() == "CBCT":
            # Only "Dental CBCT" (HEAD/SKULL) has a dose model; every other
            # CBCT body part is real anatomy with no risk calculated.
            no_risk = {r for r in non_dental[mod] if r != "Dental CBCT"}
            plot_region_by_modality(ax, non_dental, mod, no_risk_regions=no_risk)
        else:
            plot_region_by_modality(ax, non_dental, mod)
    fig.tight_layout(pad=1.5)


# Age Histogram

def _nice_tick_step(max_val: float) -> float:
    """Compute a clean rounded tick step for the x-axis.

    Picks from the set {1, 2, 5} × 10^n so that ~5–7 ticks fit on each side.
    """
    if max_val <= 0:
        return 1.0
    raw_step = max_val / 5.0
    magnitude = 10 ** np.floor(np.log10(raw_step))
    normalized = raw_step / magnitude
    # The 1k→2k boundary is set 30% lower than the "nice" 1.5 point so a
    # tight x-axis switches to 2k ticks earlier: on the Patient Cohort
    # pyramid a dense 1k grid crowds the value labels at the bar ends, and
    # 2k reads cleaner as soon as the bars are even moderately tall.
    if normalized <= 1.05:
        step = 1.0 * magnitude
    elif normalized <= 3.5:
        step = 2.0 * magnitude
    elif normalized <= 7.5:
        step = 5.0 * magnitude
    else:
        step = 10.0 * magnitude
    return step


def _nice_axis_limit(max_val: float, step: float) -> float:
    """Round the axis limit up to the next multiple of *step*.

    Adds one extra step of headroom so labels at the edge are not clipped.
    """
    if step <= 0:
        return max_val * 1.1
    return np.ceil(max_val / step + 0.5) * step


def _format_count(val: float) -> str:
    """Format a count for bar labels: integer if whole, one decimal if .5.
    Thin-space thousands (the manuscript's figure number style)."""
    val = abs(val)
    if abs(val - round(val)) < 1e-9:
        return _fmt_thousands(int(round(val)))
    return _fmt_thousands(val, 1)


def _label_tick_step(step: float, axis_limit: float, max_labels: int = 4) -> float:
    """Tick-label step for one side of the pyramid: the grid step
    ``step`` doubled until at most ``max_labels`` labels fit per side.
    A {1, 2, 5} grid step doubles to {2, 4, 10}, all readable; the
    bold tick labels of a 120 000-wide axis otherwise run into each
    other at 20 000 spacing (six labels a side)."""
    tick = step
    while tick > 0 and axis_limit / tick > max_labels:
        tick *= 2
    return tick


def _label_fits_inside(count_val: float, label: str,
                        threshold_int: float = 1800, threshold_half: float = 2500) -> bool:
    """Decide whether a label fits inside a bar.

    Default thresholds (integer labels need count ≥ 1800, .5 labels need
    count ≥ 2500) suit the plain, whole-cohort pyramid where bars run into
    the thousands.  Callers plotting smaller totals (e.g. one modality's
    contribution) should pass thresholds scaled to their own axis range so
    labels don't just always fall outside.
    """
    is_half = "." in label
    threshold = threshold_half if is_half else threshold_int
    return count_val >= threshold


def _add_bar_labels(ax, male_counts, female_counts,
                     centers, bin_width, axis_limit,
                     fontsize_inside=11, fontsize_outside=10,
                     min_label_distance=700,
                     inside_threshold_int=1800, inside_threshold_half=2500):
    """Place count labels inside or outside bars depending on bar width.

    Label placement logic:
    - If the bar is long enough (≥``inside_threshold_int`` for integers,
      ≥``inside_threshold_half`` for .5 labels), place the label inside in
      white bold text, centred in the bar.
    - Otherwise, place the label just outside the bar end in dark grey
      semibold text, but enforce a minimum distance from the zero line
      (``min_label_distance``) so labels never sit on top of the central
      axis.  The defaults suit the plain, whole-cohort pyramid (bars in
      the thousands); callers with smaller totals should scale all three
      threshold arguments to their own ``axis_limit``.
    - Male outside: x = min(-count - offset, -min_label_distance), ha = "right".
    - Female outside: x = max(count + offset, min_label_distance), ha = "left".
    - Slight vertical staggering (0.04 * bin_width) on alternating rows
      to reduce overlap between adjacent small-bar labels.
    - All non-zero bars get a label, including very small ones like 95+.
    - Labels show the plotted count (including half-unknown contributions).
    """
    dark = "#333333"
    outside_offset = axis_limit * 0.008
    stagger = bin_width * 0.04

    for i in range(len(centers)):
        m_val = male_counts[i]
        f_val = female_counts[i]
        y = centers[i]

        # Male (left side, negative x)
        if m_val > 0:
            lbl = _format_count(m_val)
            if _label_fits_inside(m_val, lbl, inside_threshold_int, inside_threshold_half):
                ax.text(-m_val / 2.0, y, lbl,
                        ha="center", va="center", fontsize=fontsize_inside,
                        color="white", fontweight="bold", clip_on=True,
                        zorder=6)
            else:
                x_pos = min(-m_val - outside_offset, -min_label_distance)
                y_off = stagger if (i % 2 == 0) else -stagger
                ax.text(x_pos, y + y_off, lbl,
                        ha="right", va="center", fontsize=fontsize_outside,
                        color=dark, fontweight="semibold",
                        clip_on=False, zorder=7)

        # Female (right side, positive x)
        if f_val > 0:
            lbl = _format_count(f_val)
            if _label_fits_inside(f_val, lbl, inside_threshold_int, inside_threshold_half):
                ax.text(f_val / 2.0, y, lbl,
                        ha="center", va="center", fontsize=fontsize_inside,
                        color="white", fontweight="bold", clip_on=True,
                        zorder=6)
            else:
                x_pos = max(f_val + outside_offset, min_label_distance)
                y_off = -stagger if (i % 2 == 0) else stagger
                ax.text(x_pos, y + y_off, lbl,
                        ha="left", va="center", fontsize=fontsize_outside,
                        color=dark, fontweight="semibold",
                        clip_on=False, zorder=7)


def _draw_age_pyramid_by_modality(ax, exam_columns, bins, n_bins, centers, age_labels,
                                   title="Patient age distribution by sex",
                                   subtitle=None, fontsize_title=16, fontsize_axis=13,
                                   fontsize_tick=11, fontsize_bar=10, fontsize_legend=12,
                                   top_n=8):
    """Draw a single age pyramid where each male/female bar is itself a
    stacked bar broken down by modality (e.g. a female bar of length 6234
    at 60-64 might be OOOMMB: mostly radiography, some mammography, a
    little CT).

    Colour coding keeps the original sex hues intact — blue for male,
    pink for female (``SEMANTIC["male"]``/``["female"]``) — with each
    modality getting its own dark→light shade of that hue (via
    ``shade_color``), so the chart still reads as "the usual pyramid"
    at a glance while remaining sex/modality-legible.  Each bar also
    keeps the same total-count label as the plain pyramid.  Modalities
    beyond ``top_n`` (by total exam count) are folded into 'Other'.
    """
    bin_width = int(bins[1] - bins[0])
    ages = np.asarray(exam_columns.age, dtype=np.float64)
    sexes = np.asarray(exam_columns.sex_code, dtype="U1")
    mods = np.asarray([m or "Unknown" for m in exam_columns.modality], dtype=object)

    valid = ages >= 0
    ages_v, sexes_v, mods_v = ages[valid], sexes[valid], mods[valid]
    bin_idx = np.clip((ages_v // bin_width).astype(np.int64), 0, n_bins - 1)

    ordered = Counter(mods_v.tolist()).most_common()
    has_other = len(ordered) > top_n
    top_names = [m for m, _ in (ordered[:top_n] if has_other else ordered)]
    stack_names = top_names + (["Other"] if has_other else [])

    male_mask = sexes_v == "M"
    female_mask = sexes_v == "F"
    unknown_mask = ~(male_mask | female_mask)

    n_stack = len(stack_names)
    male_stack = np.zeros((n_stack, n_bins))
    female_stack = np.zeros((n_stack, n_bins))
    for i, name in enumerate(stack_names):
        mod_mask = ~np.isin(mods_v, top_names) if name == "Other" else (mods_v == name)
        half_unknown = np.bincount(bin_idx[mod_mask & unknown_mask], minlength=n_bins).astype(np.float64) / 2.0
        male_stack[i] = np.bincount(bin_idx[mod_mask & male_mask], minlength=n_bins).astype(np.float64) + half_unknown
        female_stack[i] = np.bincount(bin_idx[mod_mask & female_mask], minlength=n_bins).astype(np.float64) + half_unknown

    # Dark → light shade per modality, applied on top of the base male
    # (blue) / female (pink) hues.  The ramp deliberately spans *both*
    # directions from the base hue (darker than base → lighter than base)
    # so adjacent modality segments stay easy to tell apart even with
    # 8-9 of them stacked; the biggest modality is always the darkest,
    # which keeps the ordering intuitive.
    #
    # The light end is capped well short of white on purpose.  Sex is the
    # chart's *primary* encoding and modality only its secondary one, so
    # every shade has to stay unmistakably blue or pink.  Lightening a hue
    # desaturates it, and past roughly +0.45 both ramps converge on the
    # same washed-out pastel: a very pale blue and a very pale pink are
    # near-indistinguishable, which is exactly the cue that must not be
    # lost.  Segment boundaries do not depend on the shade gap alone --
    # each bar is drawn with a white edge -- so trading some shade
    # contrast for hue fidelity costs nothing in legibility.
    #
    # The dark end differs per hue on purpose: a heavily darkened blue still
    # reads as blue, whereas a heavily darkened pink drifts toward maroon and
    # would weaken the "pink = female" cue.
    if n_stack > 1:
        male_steps = np.linspace(-0.50, 0.38, n_stack)
        female_steps = np.linspace(-0.30, 0.42, n_stack)
    else:
        male_steps = female_steps = np.zeros(1)
    male_colors = [_shade_color(_SEMANTIC["male"], s) for s in male_steps]
    female_colors = [_shade_color(_SEMANTIC["female"], s) for s in female_steps]

    male_left = np.zeros(n_bins)
    female_left = np.zeros(n_bins)
    for i in range(n_stack):
        ax.barh(centers, -male_stack[i], left=-male_left, height=bin_width * 0.85,
                color=male_colors[i], edgecolor="white", linewidth=0.7, alpha=1.0, zorder=4)
        ax.barh(centers, female_stack[i], left=female_left, height=bin_width * 0.85,
                color=female_colors[i], edgecolor="white", linewidth=0.7, alpha=1.0, zorder=4)
        male_left += male_stack[i]
        female_left += female_stack[i]

    max_val = max(male_left.max() if n_bins else 0.0, female_left.max() if n_bins else 0.0, 1.0)
    step = _nice_tick_step(max_val)
    axis_limit = _nice_axis_limit(max_val, step)
    axis_limit += step * 0.5
    # Extra room on both sides: every total label sits outside its bar here.
    axis_limit *= 1.15
    ax.set_xlim(-axis_limit, axis_limit)

    # Total-count labels per bar.  Unlike the plain pyramid these are
    # *always* placed outside the bar end (infinite "fits inside"
    # thresholds): inside labels would land on top of the white lines
    # separating the modality segments and become hard to read.
    _add_bar_labels(ax, male_left, female_left,
                    centers, bin_width, axis_limit,
                    fontsize_inside=fontsize_bar + 1,
                    fontsize_outside=fontsize_bar,
                    min_label_distance=axis_limit * 0.04,
                    inside_threshold_int=float("inf"),
                    inside_threshold_half=float("inf"))

    tick_step = _label_tick_step(step, axis_limit)
    tick_vals = np.arange(0, axis_limit + tick_step / 2, tick_step)
    ticks = np.concatenate([-tick_vals[1:][::-1], tick_vals[1:]])
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_thousands(abs(v)) for v in ticks], fontsize=fontsize_tick,
                       fontweight="bold")
    ax.set_yticks(centers)
    ax.set_yticklabels(age_labels, fontsize=fontsize_tick, fontweight="bold")
    ax.set_xlabel("Number of examinations", fontsize=fontsize_axis, fontweight="bold")
    ax.set_ylabel("Age group (years)", fontsize=fontsize_axis, fontweight="bold")
    ax.axvline(0, color=_SEMANTIC["neutral"], linewidth=0.8, zorder=5)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.text(0.02, 0.98, "MALE", transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize_axis, fontweight="bold", color=_SEMANTIC["male"])
    ax.text(0.98, 0.98, "FEMALE", transform=ax.transAxes, ha="right", va="top",
            fontsize=fontsize_axis, fontweight="bold", color=_SEMANTIC["female"])

    # The subtitle is anchored to the top of the axes (va="bottom" at
    # y=1.0), so the title has to be padded clear of a whole line of
    # subtitle text.  The pad is derived from the subtitle's own font size
    # because a hardcoded one only ever suits the single font size it is
    # tuned against, and silently overlaps the moment the sizes are scaled.
    _fs_subtitle = fontsize_title - 1
    ax.set_title(title, fontsize=fontsize_title, fontweight="bold",
                 pad=(_fs_subtitle * 1.4 + 6) if subtitle else 14)
    if subtitle:
        ax.text(0.5, 1.0, subtitle,
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=_fs_subtitle, fontweight="normal",
                color="#555555", style="italic")

    # Legend: one row per modality, shown as a paired male-shade/female-shade
    # swatch so the same row communicates "this shade of blue and this shade
    # of pink both mean <modality>".
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    legend_pairs = [
        (Patch(facecolor=male_colors[i], edgecolor="white"),
         Patch(facecolor=female_colors[i], edgecolor="white"))
        for i in range(n_stack)
    ]
    ax.legend(legend_pairs, stack_names,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
              fontsize=max(fontsize_legend - 4, 8),
              loc="lower right", ncol=2, frameon=True, framealpha=0.9,
              title="Modality",
              title_fontsize=max(fontsize_legend - 3, 9),
              handlelength=2.8, handleheight=1.2,
              labelspacing=0.5, borderpad=0.6, columnspacing=1.4,
              prop={"weight": "bold", "size": max(fontsize_legend - 4, 8)})


def plot_age_histogram(ax, exam_data: list, font_sizes: dict | None = None,
                        subtitle: str | None = None,
                        modality_filter: str | None = None,
                        modality_top_n: int = 8):
    """Draw a population-age pyramid on the given matplotlib Axes.

    Every male/female bar is a stacked bar broken down by modality (hue =
    sex, shade = modality), which is a strict superset of the plain
    one-colour-per-sex pyramid: the totals and their labels are identical,
    with the modality mix added inside each bar.  There is therefore no
    plain variant and no toggle -- a single modality can still be isolated
    via ``modality_filter``, which renders as a one-segment stack.

    Parameters
    ----------
    font_sizes : dict, optional
        Override default font sizes. Keys: title, subtitle, axis, tick,
        bar, legend. If None, uses publication-quality defaults.
    subtitle : str, optional
        Override the auto-generated subtitle. Pass "" to suppress it.
        If None, auto-generates from unrecognised-sex count.
    modality_filter : str, optional
        If given (and not "All modalities"), restrict the pyramid to
        exams of that single modality before plotting.
    modality_top_n : int
        Number of top modalities to show individually (remainder folded
        into "Other").

    Returns
    -------
    int
        Number of pyramid panels drawn (one per distinct population tag;
        1 for the single-population and empty-data paths).
    """
    # Defaults are sized for the 14x8 in figure this chart is drawn at, so
    # the value labels and legend stay readable on screen and in print.
    # Font size is a *point* size, so it is independent of the responsive
    # dpi scaling the GUI applies to the same figure -- the two compose.
    fs = font_sizes or {}
    _fs_title = fs.get("title", 21)
    _fs_axis = fs.get("axis", 17)
    _fs_tick = fs.get("tick", 16)
    _fs_bar = fs.get("bar", 15)
    _fs_legend = fs.get("legend", 20)

    exam_columns = _ensure_exam_columns(exam_data)
    if exam_columns is None or len(exam_columns) == 0:
        ax.set_title("No age data")
        return 1

    if modality_filter and modality_filter != "All modalities":
        mod_arr = np.asarray([m or "Unknown" for m in exam_columns.modality], dtype=object)
        exam_columns = exam_columns.subset(mod_arr == modality_filter)
        if len(exam_columns) == 0:
            fig = ax.figure
            fig.clear()
            no_data_ax = fig.add_subplot(111)
            no_data_ax.set_title(f"No exams for modality '{modality_filter}'")
            no_data_ax.axis("off")
            return 1

    bin_width = 5
    bins = np.arange(0, 101, bin_width)
    n_bins = len(bins) - 1
    centers = (bins[:-1] + bins[1:]) / 2

    age_labels = []
    for b in bins[:-1]:
        if b == 95:
            age_labels.append("95+")
        else:
            age_labels.append(f"{int(b)}-{int(b + bin_width - 1)}")

    fig = ax.figure
    fig.clear()

    title_suffix = ""
    if modality_filter and modality_filter != "All modalities":
        title_suffix = f" — {modality_filter}"

    # Distinct population tags in the selection, ordered Euro-American →
    # Asian → anything else sorted.  More than one means one pyramid panel
    # per population; a single (or absent) tag keeps the original layout.
    pops = np.asarray(
        [str(p) if p else "Euro-American" for p in exam_columns.population],
        dtype=object)
    distinct = set(pops.tolist())
    pop_order = [p for p in ("Euro-American", "Asian") if p in distinct]
    pop_order += sorted(distinct - set(pop_order))

    if len(pop_order) <= 1:
        single_ax = fig.add_subplot(111)
        _draw_age_pyramid_by_modality(single_ax, exam_columns, bins, n_bins,
                                      centers, age_labels,
                                      title=f"Patient age distribution{title_suffix}",
                                      subtitle=subtitle,
                                      fontsize_title=_fs_title,
                                      fontsize_axis=_fs_axis,
                                      fontsize_tick=_fs_tick,
                                      fontsize_bar=_fs_bar,
                                      fontsize_legend=_fs_legend,
                                      top_n=modality_top_n)
        return 1

    # sharey=False: each pyramid keeps its own age labels on the y axis;
    # the shared bins still line the rows up across panels.  Every panel
    # draws at the *same* font sizes and 14:8 proportions as the single
    # pyramid -- the caller widens the figure to 14 in per panel instead
    # of squeezing N panels into one 14x8 in canvas (the GUI's dpi scaling
    # then shrinks the whole thing uniformly, as it does for one panel).
    n_pop = len(pop_order)
    axes = fig.subplots(1, n_pop, sharey=False)
    for i, name in enumerate(pop_order):
        sub = exam_columns.subset(pops == name)
        _draw_age_pyramid_by_modality(
            axes[i], sub, bins, n_bins, centers, age_labels,
            title=f"Patient age distribution — {name}{title_suffix}",
            subtitle=subtitle if subtitle is not None
                     else f"{len(sub):,} exams",
            fontsize_title=_fs_title, fontsize_axis=_fs_axis,
            fontsize_tick=_fs_tick, fontsize_bar=_fs_bar,
            fontsize_legend=_fs_legend, top_n=modality_top_n)
    return n_pop


# Patient Cohort — sex + modality mix

def plot_sex_breakdown(ax, exam_data: list):
    """Pie chart of the exposed cohort's sex split.

    Complements the age pyramid: reviewers routinely ask "is the dose /
    risk imbalance driven by a sex imbalance in who gets imaged?".  Uses
    the CB-safe sex palette (``male``/``female``/``unknown``) so it lines
    up visually with every other sex-coded chart in the GUI.
    """
    ax.clear()
    exam_columns = _ensure_exam_columns(exam_data)
    if exam_columns is None or len(exam_columns) == 0:
        ax.set_title("Sex distribution: No data")
        ax.axis("off")
        return
    # Vectorised sex counts: one array pass + C-level boolean masks.
    sex_chars = np.asarray(exam_columns.sex_code, dtype="U1")
    n_male    = int(np.count_nonzero(sex_chars == "M"))
    n_female  = int(np.count_nonzero(sex_chars == "F"))
    n_unknown = len(sex_chars) - n_male - n_female
    counts = {"Male": n_male, "Female": n_female, "Unknown": n_unknown}
    palette = {
        "Male":    _SEMANTIC["male"],
        "Female":  _SEMANTIC["female"],
        "Unknown": _SEMANTIC["unknown"],
    }
    total = n_male + n_female + n_unknown
    if total <= 0:
        ax.set_title("Sex distribution: No data")
        ax.axis("off")
        return

    # Slices below this share cannot carry a readable on-pie label: their
    # wedge is a hairline and the label's leader angle is indistinguishable
    # from its neighbour's, so it lands on top of it (the "Unknown (n=91)"
    # = 0.07% case).  Those go into the caption line instead, where the
    # number is still reported but stops fighting for space on the pie.
    _MIN_LABELLED_SHARE = 2.0

    labels, values, colors, footnotes = [], [], [], []
    for name in ("Male", "Female", "Unknown"):
        n = int(counts.get(name, 0))
        if n <= 0:
            continue
        share = n / total * 100.0
        values.append(n)
        colors.append(palette[name])
        # One label per slice carrying both the count and the share.
        # Keeping them together inside the wedge makes each slice legible
        # in one glance -- separate inside/outside texts would force the
        # reader to pair two pieces of text per slice.
        if share >= _MIN_LABELLED_SHARE:
            labels.append(f"{name}\n{n:,}  ({share:.1f}%)")
        else:
            labels.append("")
            # "0.0%" would read as "none", so anything that rounds to zero
            # at one decimal is reported as an upper bound instead.
            share_txt = f"{share:.1f}%" if share >= 0.1 else "<0.1%"
            footnotes.append(f"{name.lower()} sex: {n:,} ({share_txt})")

    ax.pie(
        values, colors=colors, labels=labels,
        startangle=90, counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 10},
    )

    # Which reference population the cohort was risk-modelled against
    # (Euro-American vs Asian baselines give materially different lifetime
    # risks), so the reader can attribute the numbers without cross-checking
    # the run configuration.  A mixed cohort reports its split rather than
    # silently picking one name.
    pops = [str(p) for p in exam_columns.population if p]
    named = sorted(set(pops))
    pop_mix = ""
    if len(named) == 1:
        pop_line = f"{named[0]} cohort"
    elif len(named) > 1:
        pop_line = "mixed cohort"
        # The per-population split goes on its own caption line rather than
        # into the title.  This axes is only half the figure wide, so a title
        # that grows with the number of populations runs off the edge.
        pop_mix = ", ".join(
            f"{name} {n / len(pops) * 100:.0f}%"
            for name, n in Counter(pops).most_common()
        )
    else:
        pop_line = "population not specified"

    # Caption: one line for the volume (plus any slice too small to label),
    # a second only when there is a population split to report.
    cap_lines = [f"{total:,} exams"]
    if footnotes:
        cap_lines[0] += " · " + "; ".join(footnotes)
    if pop_mix:
        cap_lines.append(pop_mix)

    # Two-tier heading: the bold line says *what cohort*, the grey line(s)
    # below it carry the volume and the caveats.  The title pad is derived
    # from the caption's font size and line count so the two never collide.
    _fs_caption = 9
    ax.set_title(f"Sex distribution — {pop_line}",
                 fontsize=11, fontweight="bold",
                 pad=_fs_caption * 1.5 * len(cap_lines) + 4)
    ax.text(0.5, 1.0, "\n".join(cap_lines), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=_fs_caption,
            color="#555555", style="italic", linespacing=1.4)


def plot_modality_mix(ax, exam_data: list, top_n: int = 10, exam_columns=None,
                       split_by_sex: bool = False):
    """Horizontal bar chart of exam counts per modality (top-N + 'Other').

    Horizontal layout so long modality names remain readable even with
    20+ categories.  Everything below rank ``top_n`` is folded into a
    single 'Other' bar so the chart stays uncluttered on large
    multi-modality datasets.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to plot on
    exam_data : list
        List of exam dicts (converted to ExamColumns internally)
    top_n : int
        Number of top modalities to show individually
    exam_columns : ExamColumns, optional
        Columnar exam data for efficient processing
    split_by_sex : bool
        If True, each bar is stacked into male/female/unknown segments
        (see ``_plot_modality_mix_by_sex``) instead of one solid colour
        per modality.
    """
    ax.clear()
    exam_columns = exam_columns or _ensure_exam_columns(exam_data)
    if exam_columns is None or len(exam_columns) == 0:
        ax.set_title("Modality Mix: No data")
        ax.axis("off")
        return

    if split_by_sex:
        _plot_modality_mix_by_sex(ax, exam_columns, top_n)
        return

    # Build a plain list once; Counter's C fast-path is faster on lists than generators.
    modality_list = [m or "Unknown" for m in exam_columns.modality]

    counts = Counter(modality_list)
    if not counts:
        ax.set_title("Modality Mix: No data")
        ax.axis("off")
        return
    ordered = counts.most_common()
    if len(ordered) > top_n:
        head = ordered[:top_n]
        tail_total = sum(v for _, v in ordered[top_n:])
        ordered = head + [("Other", tail_total)]
    modalities = [m for m, _ in ordered]
    values = [int(v) for _, v in ordered]
    total = sum(values)
    ys = np.arange(len(modalities))
    colors = [
        _SEMANTIC["reference"] if m == "Other" else _stable_color(m)
        for m in modalities
    ]
    bars = ax.barh(ys, values, color=colors, alpha=0.9, edgecolor="white")
    ax.set_yticks(ys)
    ax.set_yticklabels(modalities, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Examinations")
    ax.set_title(f"Modality Mix — {total:,} exams")
    max_v = max(values) if values else 1
    for bar, v in zip(bars, values):
        pct = (v / total * 100) if total else 0.0
        ax.text(
            bar.get_width() + max_v * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{v:,}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=8,
        )
    # Give the annotations room on the right.
    ax.set_xlim(0, max_v * 1.22)


def _plot_modality_mix_by_sex(ax, exam_columns, top_n: int = 10):
    """Horizontal stacked bar chart of exam counts per modality, split by sex.

    Colour coding is deliberately two-layered so the sex split reads at a
    glance while the modality breakdown stays available:

    * **Hue** encodes sex — blue for male, pink (Okabe-Ito reddish-purple)
      for female, grey for unknown — matching every other sex-coded chart
      in the GUI (``SEMANTIC["male"]``/``["female"]``/``["unknown"]``).
    * **Shade** (dark → light) encodes the modality, so each bar segment
      within a hue is still visually distinguishable per modality.
    """
    modality_arr = np.asarray([m or "Unknown" for m in exam_columns.modality], dtype=object)
    sex_arr = np.asarray(exam_columns.sex_code, dtype="U1")

    counts = Counter(modality_arr.tolist())
    if not counts:
        ax.set_title("Modality Mix: No data")
        ax.axis("off")
        return
    ordered = counts.most_common()
    if len(ordered) > top_n:
        top_names = [m for m, _ in ordered[:top_n]]
        modalities = top_names + ["Other"]
        is_other = ~np.isin(modality_arr, top_names)
    else:
        modalities = [m for m, _ in ordered]
        is_other = np.zeros(len(modality_arr), dtype=bool)

    n_mod = len(modalities)
    male_counts = np.zeros(n_mod, dtype=int)
    female_counts = np.zeros(n_mod, dtype=int)
    unknown_counts = np.zeros(n_mod, dtype=int)
    for i, m in enumerate(modalities):
        mask = is_other if m == "Other" else (modality_arr == m)
        sex_here = sex_arr[mask]
        male_counts[i] = int(np.count_nonzero(sex_here == "M"))
        female_counts[i] = int(np.count_nonzero(sex_here == "F"))
        unknown_counts[i] = mask.sum() - male_counts[i] - female_counts[i]

    values = male_counts + female_counts + unknown_counts
    total = int(values.sum())
    ys = np.arange(n_mod)

    # Dark → light shade per modality, applied on top of the male/female/
    # unknown base hues so sex stays the dominant, instantly-readable signal.
    shade_steps = np.linspace(0.0, 0.55, n_mod) if n_mod > 1 else np.zeros(1)
    male_colors = [_shade_color(_SEMANTIC["male"], s) for s in shade_steps]
    female_colors = [_shade_color(_SEMANTIC["female"], s) for s in shade_steps]
    unknown_colors = [_shade_color(_SEMANTIC["unknown"], s) for s in shade_steps]

    ax.barh(ys, male_counts, color=male_colors, alpha=0.95, edgecolor="white",
            label="Male")
    ax.barh(ys, female_counts, left=male_counts, color=female_colors,
            alpha=0.95, edgecolor="white", label="Female")
    ax.barh(ys, unknown_counts, left=male_counts + female_counts,
            color=unknown_colors, alpha=0.95, edgecolor="white", label="Unknown")

    ax.set_yticks(ys)
    ax.set_yticklabels(modalities, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Examinations")
    ax.set_title(f"Modality Mix by Sex — {total:,} exams")
    max_v = int(values.max()) if len(values) else 1
    for i, v in enumerate(values):
        pct = (v / total * 100) if total else 0.0
        ax.text(
            v + max_v * 0.01, ys[i],
            f"{v:,}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=8,
        )
    # Give the annotations room on the right.
    ax.set_xlim(0, max_v * 1.22)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_SEMANTIC["male"], edgecolor="white", label="Male"),
        Patch(facecolor=_SEMANTIC["female"], edgecolor="white", label="Female"),
    ]
    if unknown_counts.sum() > 0:
        legend_handles.append(
            Patch(facecolor=_SEMANTIC["unknown"], edgecolor="white", label="Unknown")
        )
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.9)


# Cancer Distribution Bar Chart

def filter_cancer_breakdown(cancer_breakdown: Dict[str, Dict], visible_cancers: Optional[set] = None):
    """Filter the cancer-breakdown dict to only ``visible_cancers``.

    Returns the full dict when ``visible_cancers`` is ``None``.
    """
    if visible_cancers is None:
        return dict(cancer_breakdown or {})
    return {
        ct: cv for ct, cv in (cancer_breakdown or {}).items()
        if ct in visible_cancers
    }


def plot_cancer_pie(ax, cancer_breakdown: Dict[str, Dict], visible_cancers: Optional[set] = None):
    """Draw a pie chart of cancer-type distribution on the given Axes."""
    cancer_breakdown = filter_cancer_breakdown(cancer_breakdown, visible_cancers)
    ax.clear()
    if not cancer_breakdown:
        ax.set_title("No cancer data yet")
        return
    items = []
    for ct, cv in sorted(cancer_breakdown.items(), key=lambda x: x[1].get("developed", 0), reverse=True):
        n = float(cv.get("developed", 0) or 0)
        if n > 0:
            items.append((ct, n))
    if not items:
        ax.set_title("No cancers developed")
        return

    labels = [ct for ct, _ in items]
    counts = np.array([n for _, n in items], dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        ax.set_title("No cancers developed")
        return

    # Percentages are easier to compare across very uneven cancer frequencies
    # than a pie chart, especially when Skin dominates the distribution.
    percentages = counts / total * 100.0

    # Color-blind-friendly palette. The chart does not rely on color for
    # interpretation, but the palette helps distinguish categories visually.
    colors = [
        "#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7",
        "#56B4E9", "#F0E442", "#999999", "#000000",
    ]
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

    y = np.arange(len(labels))
    bars = ax.barh(y, percentages, color=bar_colors, edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Share of cancers (%)", fontsize=14, fontweight="bold")
    ax.set_title("Cancer Type Distribution", fontsize=17.5, fontweight="bold",
                 loc="center", pad=8)
    ax.set_xlim(0, 110)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.tick_params(axis="both", labelsize=12.5)
    for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
        _lbl.set_fontweight("bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for sp_name in ("top", "right", "bottom", "left"):
        ax.spines[sp_name].set_visible(True)
        ax.spines[sp_name].set_color("#333333")
        ax.spines[sp_name].set_linewidth(0.8)

    x_offset = 1.0
    for bar, pct in zip(bars, percentages):
        ax.text(
            min(pct + x_offset, 109.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=14,
            fontweight="bold",
        )


# Cancer Frequency vs QALYL Burden — two-panel aligned horizontal bar chart

def _compute_qalyl_by_cancer_type(event_columns, visible_cancers=None):
    """Sum QALY_lost per cancer type from EventColumns.

    Returns a ``dict[str, float]`` mapping cancer-type name to total QALYL.
    """
    if event_columns is None or len(event_columns) == 0:
        return {}
    ct_arr = event_columns.cancer_type
    qaly_arr = np.asarray(event_columns.QALY_lost, dtype=np.float64)
    if visible_cancers is not None:
        mask = np.array([str(t) in visible_cancers for t in ct_arr], dtype=bool)
        ct_arr = ct_arr[mask]
        qaly_arr = qaly_arr[mask]
    if len(ct_arr) == 0:
        return {}
    result: Dict[str, float] = {}
    for i in range(len(ct_arr)):
        ct = str(ct_arr[i]) if ct_arr[i] else "Unknown"
        result[ct] = result.get(ct, 0.0) + float(qaly_arr[i])
    return result


def plot_cancer_frequency_vs_burden(
    ax_freq,
    ax_burden,
    cancer_breakdown: Dict[str, Dict],
    event_columns=None,
    visible_cancers: Optional[set] = None,
):
    """Draw two aligned horizontal bar charts comparing cancer frequency
    with QALYs lost (QALYL) contribution.

    Panel A — Share of simulated cancers (%)
    Panel B — Share of total QALYL (%)

    Both panels share an identical cancer-type ordering (sorted by
    frequency, descending) and **identical per-cancer-type colours** so
    readers can instantly compare a given cancer between frequency and
    burden and see rank shifts.

    The colour assignment, font sizes, bar height, and grid style are
    consistent with the single-panel cancer-type chart convention.

    Parameters
    ----------
    ax_freq : matplotlib Axes
        Left panel — cancer frequency.
    ax_burden : matplotlib Axes
        Right panel — QALYL contribution.
    cancer_breakdown : dict
        ``{cancer_type: {"developed": int, "deaths": int}}`` from stats.
    event_columns : EventColumns, optional
        Columnar event data with ``cancer_type`` and ``QALY_lost`` fields.
        When ``None`` or empty, Panel B shows "No QALYL data".
    visible_cancers : set, optional
        Restrict to this subset of cancer types.
    """
    ax_freq.clear()
    ax_burden.clear()

    cancer_breakdown = filter_cancer_breakdown(cancer_breakdown, visible_cancers)

    # Gather frequency data
    freq_items = []
    for ct, cv in cancer_breakdown.items():
        n = float(cv.get("developed", 0) or 0)
        if n > 0:
            freq_items.append((ct, n))

    if not freq_items:
        for ax in (ax_freq, ax_burden):
            ax.set_title("No cancer data yet", fontsize=13.65, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        return

    # Sort by frequency descending — determines order shared by both panels
    freq_items.sort(key=lambda x: x[1], reverse=True)
    labels = [ct for ct, _ in freq_items]
    freq_counts = np.array([n for _, n in freq_items], dtype=float)
    freq_total = float(np.sum(freq_counts))
    freq_pct = freq_counts / freq_total * 100.0 if freq_total > 0 else np.zeros_like(freq_counts)

    # Gather QALYL data
    qalyl_by_type = _compute_qalyl_by_cancer_type(event_columns, visible_cancers)
    qalyl_vals = np.array([qalyl_by_type.get(ct, 0.0) for ct in labels], dtype=float)
    qalyl_total = float(np.sum(qalyl_vals))
    qalyl_pct = qalyl_vals / qalyl_total * 100.0 if qalyl_total > 0 else np.zeros_like(qalyl_vals)

    has_qalyl = qalyl_total > 0

    # Per-cancer-type colours (same as original plot_cancer_pie)
    # Okabe-Ito colour-blind-safe palette, assigned by position.
    colors = [
        "#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7",
        "#56B4E9", "#F0E442", "#999999", "#000000",
    ]
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

    # Shared layout parameters
    y = np.arange(len(labels))
    bar_height = 0.72  # match original
    x_offset = 1.0     # match original

    # Panel A: Cancer frequency
    bars_a = ax_freq.barh(y, freq_pct, color=bar_colors, edgecolor="white",
                          height=bar_height)
    ax_freq.set_yticks(y)
    ax_freq.set_yticklabels(labels, fontsize=18.85, fontweight="bold")
    ax_freq.invert_yaxis()
    ax_freq.set_xlabel("Share of simulated cancers (%)", fontsize=22.75, fontweight="bold")
    ax_freq.set_title("Cancer Type Distribution", fontsize=22.75, fontweight="bold",
                      loc="center", pad=8)
    ax_freq.set_xlim(0, 100)
    ax_freq.set_xticks(np.arange(0, 101, 10))
    ax_freq.tick_params(axis="both", labelsize=16.25)
    for _lbl in ax_freq.get_xticklabels() + ax_freq.get_yticklabels():
        _lbl.set_fontweight("bold")
    ax_freq.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax_freq.set_axisbelow(True)
    for sp_name in ("top", "right", "bottom", "left"):
        ax_freq.spines[sp_name].set_visible(True)
        ax_freq.spines[sp_name].set_color("#333333")
        ax_freq.spines[sp_name].set_linewidth(0.8)
    for bar, pct in zip(bars_a, freq_pct):
        ax_freq.text(
            min(pct + x_offset, 99.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=18.2, fontweight="bold",
        )

    # Panel B: QALYL contribution
    if has_qalyl:
        bars_b = ax_burden.barh(y, qalyl_pct, color=bar_colors, edgecolor="white",
                                height=bar_height)
        _burden_max = max(qalyl_pct.max() * 1.15, 10)
        _burden_xlim = np.ceil(_burden_max / 5) * 5  # round up to nearest 5
        for bar, pct in zip(bars_b, qalyl_pct):
            ax_burden.text(
                min(pct + x_offset, _burden_xlim - 1.0),
                bar.get_y() + bar.get_height() / 2.0,
                f"{pct:.1f}%",
                va="center", ha="left", fontsize=18.2, fontweight="bold",
            )
    else:
        ax_burden.text(0.5, 0.5, "No QALY burden data yet",
                       transform=ax_burden.transAxes, ha="center", va="center",
                       fontsize=13.65, color="#999999")

    ax_burden.set_yticks(y)
    ax_burden.set_yticklabels(labels, fontsize=18.85, fontweight="bold")
    ax_burden.invert_yaxis()
    ax_burden.set_xlabel("Share of total QALY burden (%)", fontsize=22.75, fontweight="bold")
    ax_burden.set_title("Contribution to Total QALY Burden by Cancer Type",
                        fontsize=22.75, fontweight="bold", loc="center", pad=8)
    if has_qalyl:
        ax_burden.set_xlim(0, _burden_xlim)
        _burden_tick = 10 if _burden_xlim <= 40 else 10
        ax_burden.set_xticks(np.arange(0, _burden_xlim + 1, _burden_tick))
    else:
        ax_burden.set_xlim(0, 50)
        ax_burden.set_xticks(np.arange(0, 51, 10))
    ax_burden.tick_params(axis="both", labelsize=16.25)
    for _lbl in ax_burden.get_xticklabels() + ax_burden.get_yticklabels():
        _lbl.set_fontweight("bold")
    ax_burden.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax_burden.set_axisbelow(True)
    for sp_name in ("top", "right", "bottom", "left"):
        ax_burden.spines[sp_name].set_visible(True)
        ax_burden.spines[sp_name].set_color("#333333")
        ax_burden.spines[sp_name].set_linewidth(0.8)


# Iteration Distribution Histogram

# One legend size for every Summary-tab distribution panel (cancer / death
# counts, QALYs, life-years) so the four charts read identically.
_SUMMARY_LEGEND_FS = 13
_SUMMARY_LEGEND_HEADROOM = 1.4

def plot_iteration_histogram(ax_cases, ax_deaths, iter_cases: list, iter_deaths: list,
                              *, scale: float = 1.0,
                              per_label: str = "per simulated iteration"):
    """Histogram where x = number of cancers (or deaths) in one simulated
    year and y = number of years (iterations) with that count.

    Adaptive binning.  The plotted values are counts, so they are integers,
    and every bar must therefore span the same whole number of integers.
    A fractional bin width (e.g. span/40 = 1.075) puts two integers into
    the occasional bin and one into the rest, which renders as isolated
    bars about twice the height of their neighbours — an aliasing artefact
    that looks like a real mode.  Both branches below keep the bin edges on
    half-integers with an integer width so this cannot happen:
      * Ranges up to ``_MAX_BARS`` counts get one bar per integer, which is
        what the user expects for cohorts where every count matters.
      * Wider ranges (e.g. 1,200 … 15,000 cancers per year on large
        datasets) group a whole number of counts per bar, sized by
        Freedman–Diaconis and capped at ``_MAX_BARS`` bars.  This keeps bar
        widths visible (instead of 0.8-wide hairlines over a 14,000-unit
        span) and lets matplotlib place a handful of readable x-ticks with
        thousands separators instead of ~14,000 stacked integer labels.

    ``scale`` re-expresses each iteration's count at a different, fixed
    examination count (e.g. ``100_000 / unique_examinations`` to project
    "per 100,000 examinations" -- the same linear-rescale convention as
    ``research_questions.compute_region_age_dose_sweep``'s ``per_n_scans``).
    Once scaled the values are not integer counts, so this switches
    to the same continuous Freedman-Diaconis binning as
    :func:`plot_QALY_histogram` instead of the integer-count branches above.
    """
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    _fmt_int = FuncFormatter(lambda v, _p: f"{int(round(v)):,}")
    _scaled = (scale != 1.0)

    def _draw(ax, values, noun, bar_color):
        """Draw the per-iteration ``values`` count histogram on ``ax``
        (integer bins; ``scale`` applied first when set).  Shows a
        "No data" title when ``values`` is empty."""
        if not values:
            ax.set_title(f"{noun.capitalize()} Count Distribution: No data",
                         fontsize=17.5, fontweight="bold", loc="center", pad=8)
            return
        if _scaled:
            vals = np.asarray([float(v) * scale for v in values], dtype=np.float64)
        else:
            vals = np.fromiter((int(v) for v in values), dtype=np.int64,
                               count=len(values))
        vmin, vmax = float(vals.min()), float(vals.max())
        span = vmax - vmin
        n = vals.size
        mean_v = float(vals.mean())
        median_v = float(np.median(vals))
        p05, p95 = np.percentile(vals, [2.5, 97.5])

        _MAX_BARS = 60
        if not _scaled and span < _MAX_BARS:
            # Small dynamic range → one bar per integer.
            vmin_i, vmax_i = int(vmin), int(vmax)
            counts = Counter(int(v) for v in vals.tolist())
            xs = list(range(vmin_i, vmax_i + 1))
            ys = [counts.get(x, 0) for x in xs]
            ax.bar(xs, ys, width=0.9, color=bar_color, alpha=0.85,
                   edgecolor="white", align="center")
            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=min(len(xs), 12)))
            ax.set_xlim(vmin_i - 0.5, vmax_i + 0.5)
        elif not _scaled:
            # Wide dynamic range → group a whole number of counts per bar.
            # Freedman–Diaconis sets the target width; it is rounded up to
            # an integer and then to whatever is needed to stay within
            # _MAX_BARS bars.  Edges sit on half-integers so each bar
            # covers exactly ``step`` integer values.
            q75, q25 = np.percentile(vals, [75, 25])
            iqr = float(q75 - q25)
            if iqr > 0:
                fd_width = 2.0 * iqr / (n ** (1.0 / 3.0))
            else:
                fd_width = span / max(float(np.sqrt(n)), 1.0)
            step = max(1, int(np.ceil(max(fd_width, 1.0))))
            step = max(step, int(np.ceil((span + 1) / float(_MAX_BARS))))
            edges = np.arange(vmin - 0.5, vmax + 0.5 + step, step)
            ax.hist(vals, bins=edges, color=bar_color, alpha=0.85,
                    edgecolor="white", linewidth=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            ax.xaxis.set_major_formatter(_fmt_int)
            ax.set_xlim(float(edges[0]) - span * 0.02,
                        float(edges[-1]) + span * 0.02)
        else:
            # Scaled (continuous) domain: same Freedman-Diaconis binning as
            # plot_QALY_histogram, since a projected count is not an
            # integer.
            if span <= 0:
                ax.bar([vmin], [n], width=max(abs(vmin) * 0.02, 1.0),
                       color=bar_color, alpha=0.85, edgecolor="white",
                       align="center")
            else:
                q75, q25 = np.percentile(vals, [75, 25])
                iqr = float(q75 - q25)
                if iqr > 0:
                    fd_width = 2.0 * iqr / (n ** (1.0 / 3.0))
                    nbins = int(np.clip(np.ceil(span / max(fd_width, 1e-9)), 15, 40))
                else:
                    nbins = int(np.clip(np.sqrt(n), 15, 40))
                ax.hist(vals, bins=nbins, color=bar_color, alpha=0.85,
                        edgecolor="white", linewidth=0.6)
                ax.set_xlim(vmin - span * 0.02, vmax + span * 0.02)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.xaxis.set_major_formatter(_fmt_int)

        ax.yaxis.set_major_formatter(_fmt_int)
        ax.set_xlabel(f"Number of {noun} {per_label}",
                      fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of iterations",
                      fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11.5)
        for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
            _lbl.set_fontweight("bold")
        # Title describes only the plotted quantity (no numerical results).
        if noun == "cancers":
            _title = "Cancer Count Distribution"
        else:
            _title = "Fatal Cancer Count Distribution"
        if _scaled:
            _title += f" ({per_label})"
        ax.set_title(_title, fontsize=17.5, fontweight="bold",
                     loc="center", pad=8)
        # Numerical summaries in legend, not in title.
        ax.axvline(mean_v, color=_SEMANTIC["highlight"], linestyle="--",
                   linewidth=1.8, label=f"Mean: {mean_v:,.1f}")
        ax.axvline(median_v, color=_SEMANTIC["neutral"], linestyle=":",
                   linewidth=1.8, label=f"Median: {median_v:,.1f}")
        # Central 95 % inter-percentile band — slightly darker for visibility.
        _ci_label = (f"95% range of simulated years: {p05:,.1f}\u2013{p95:,.1f}" if _scaled
                     else f"95% range of simulated years: {int(round(p05)):,}\u2013{int(round(p95)):,}")
        ax.axvspan(float(p05), float(p95),
                   color=_SEMANTIC["neutral"], alpha=0.11,
                   label=_ci_label)
        _leg = ax.legend(loc="upper right", framealpha=0.9,
                  prop={"weight": "bold", "size": _SUMMARY_LEGEND_FS},
                  handlelength=1.56, handleheight=0.78)
        # Headroom so the three-line legend sits above the bars instead of
        # covering the peak of the distribution.
        ax.set_ylim(top=ax.get_ylim()[1] * _SUMMARY_LEGEND_HEADROOM)
        # Grid and borders matching Figure 8 style.
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        for sp_name in ("top", "right", "bottom", "left"):
            ax.spines[sp_name].set_visible(True)
            ax.spines[sp_name].set_color("#333333")
            ax.spines[sp_name].set_linewidth(0.8)

    _draw(ax_cases,  iter_cases,  "cancers", _SEMANTIC["induced"])
    _draw(ax_deaths, iter_deaths, "deaths",  _SEMANTIC["died"])


def plot_QALY_histogram(ax_QALY, ax_life, iter_QALY: list, iter_life: list,
                         *, scale: float = 1.0,
                         per_label: str = "per simulated iteration"):
    """Per-iteration distribution of total QALYs lost and life-years lost.

    The continuous-valued counterpart of ``plot_iteration_histogram``:
    x = total QALYs lost (overall, all cancers) -- or life-years lost from
    fatal cancers only -- in one simulated year, and y = number of years
    (iterations) falling in that bin.  Both panels always use
    Freedman-Diaconis binning (capped at 40 bins) because the per-iteration
    totals are continuous floats, not the small integer counts the cancer /
    death distribution deals with.

    ``scale`` re-expresses each iteration's total at a different, fixed
    examination count (e.g. ``100_000 / unique_examinations``), matching
    :func:`plot_iteration_histogram`'s ``scale``/``per_label`` kwargs.
    """
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    _fmt_int = FuncFormatter(lambda v, _p: f"{int(round(v)):,}")

    def _draw(ax, values, title, xlabel, bar_color):
        """Draw a histogram of ``values`` on ``ax`` after dropping
        non-finite entries and applying ``scale``.  Shows a "No data"
        title when nothing positive remains."""
        vals = np.asarray([float(v) for v in (values or [])], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or not np.any(vals > 0.0):
            ax.set_title(f"{title}: No data",
                         fontsize=17.5, fontweight="bold", loc="center", pad=8)
            return
        if scale != 1.0:
            vals = vals * scale
        vmin, vmax = float(vals.min()), float(vals.max())
        span = vmax - vmin
        n = vals.size
        mean_v = float(vals.mean())
        median_v = float(np.median(vals))
        p05, p95 = (float(x) for x in np.percentile(vals, [2.5, 97.5]))

        if span <= 0:
            # Every iteration identical → a single representative bar.
            _bw = max(abs(vmin) * 0.02, 1.0)
            ax.bar([vmin], [n], width=_bw,
                   color=bar_color, alpha=0.85, edgecolor="white",
                   align="center")
            ax.set_xlim(max(0.0, vmin - _bw) if vmin >= 0.0 else vmin - _bw,
                        vmin + _bw)
        else:
            q75, q25 = np.percentile(vals, [75, 25])
            iqr = float(q75 - q25)
            if iqr > 0:
                fd_width = 2.0 * iqr / (n ** (1.0 / 3.0))
                nbins = int(np.clip(np.ceil(span / max(fd_width, 1e-9)), 15, 40))
            else:
                nbins = int(np.clip(np.sqrt(n), 15, 40))
            ax.hist(vals, bins=nbins, color=bar_color, alpha=0.85,
                    edgecolor="white", linewidth=0.6)
            _lo = vmin - span * 0.02
            ax.set_xlim(max(0.0, _lo) if vmin >= 0.0 else _lo,
                        vmax + span * 0.02)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_major_formatter(_fmt_int)
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of iterations", fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11.5)
        for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
            _lbl.set_fontweight("bold")
        # Title describes only the plotted quantity (no numerical results).
        _title = f"{title} ({per_label})" if scale != 1.0 else title
        ax.set_title(_title, fontsize=17.5, fontweight="bold",
                     loc="center", pad=8)
        # Numerical summaries in legend, not in title.
        ax.axvline(mean_v, color=_SEMANTIC["highlight"], linestyle="--",
                   linewidth=1.8, label=f"Mean: {mean_v:,.1f}")
        ax.axvline(median_v, color=_SEMANTIC["neutral"], linestyle=":",
                   linewidth=1.8, label=f"Median: {median_v:,.1f}")
        # Central 95 % inter-percentile band — slightly darker for visibility.
        ax.axvspan(p05, p95, color=_SEMANTIC["neutral"], alpha=0.11,
                   label=f"95% range of simulated years: {p05:,.1f}–{p95:,.1f}")
        _leg = ax.legend(loc="upper right", framealpha=0.9,
                  prop={"weight": "bold", "size": _SUMMARY_LEGEND_FS},
                  handlelength=1.56, handleheight=0.78)
        # Headroom so the three-line legend sits above the bars instead of
        # covering the peak of the distribution.
        ax.set_ylim(top=ax.get_ylim()[1] * _SUMMARY_LEGEND_HEADROOM)
        # Grid and borders matching Figure 8 style.
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        for sp_name in ("top", "right", "bottom", "left"):
            ax.spines[sp_name].set_visible(True)
            ax.spines[sp_name].set_color("#333333")
            ax.spines[sp_name].set_linewidth(0.8)

    _draw(ax_QALY, iter_QALY, "QALYs Lost Distribution",
          f"QALYs lost {per_label}", "#8C564B")
    _draw(ax_life, iter_life, "Life Years Lost (fatal only) Distribution",
          f"Life-years lost {per_label}", _OKABE_ITO["black"])


def plot_cancer_age_histogram(ax, cancer_events: list, visible_cancers: Optional[set] = None):
    """Age pyramid of cancer events (age at exposure) split by sex.

    Mirrors the patient-cohort age pyramid: horizontal bars with male on
    the left, female on the right, 5-year bins from 0 to 95+.
    """
    ax.clear()
    event_columns = _ensure_event_columns(cancer_events)
    if event_columns is None or len(event_columns) == 0:
        ax.set_title("Cancer Age Distribution: No data yet")
        ax.axis("off")
        return

    # Filter by visible cancer types if specified
    if visible_cancers is not None:
        mask = np.array([ct in visible_cancers for ct in event_columns.cancer_type], dtype=bool)
        ages = event_columns.age_at_exposure[mask]
        sexes = np.asarray(event_columns.sex, dtype="U1")[mask]
    else:
        ages = event_columns.age_at_exposure
        sexes = np.asarray(event_columns.sex, dtype="U1")

    if ages.size == 0:
        ax.set_title("Cancer Age Distribution: No visible cancers")
        ax.axis("off")
        return

    ages = np.asarray(ages, dtype=np.float64)
    valid = ages >= 0
    ages_v = ages[valid]
    sexes_v = sexes[valid]

    if len(ages_v) == 0:
        ax.set_title("Cancer Age Distribution: No valid ages")
        ax.axis("off")
        return

    bin_width = 5
    bins = np.arange(0, 101, bin_width)
    n_bins = len(bins) - 1
    centers = (bins[:-1] + bins[1:]) / 2

    age_labels = []
    for b in bins[:-1]:
        if b == 95:
            age_labels.append("95+")
        else:
            age_labels.append(f"{int(b)}-{int(b + bin_width - 1)}")

    bin_idx = np.clip((ages_v // bin_width).astype(np.int64), 0, n_bins - 1)
    male_mask    = sexes_v == "M"
    female_mask  = sexes_v == "F"
    unknown_mask = ~(male_mask | female_mask)

    male_counts    = np.bincount(bin_idx[male_mask], minlength=n_bins).astype(np.float64)
    female_counts  = np.bincount(bin_idx[female_mask], minlength=n_bins).astype(np.float64)
    unknown_counts = np.bincount(bin_idx[unknown_mask], minlength=n_bins).astype(np.float64)

    raw_male   = male_counts.copy()
    raw_female = female_counts.copy()

    # Distribute unrecognised sex evenly across both sexes
    half_unknown  = unknown_counts / 2.0
    male_counts   += half_unknown
    female_counts += half_unknown

    ax.barh(centers, -male_counts,   height=bin_width * 0.85,
            color=_SEMANTIC["male"], alpha=0.85, edgecolor="white", label="Male")
    ax.barh(centers,  female_counts, height=bin_width * 0.85,
            color=_SEMANTIC["female"], alpha=0.85, edgecolor="white", label="Female")

    max_val = max(male_counts.max(), female_counts.max(), 1)
    ax.set_xlim(-max_val * 1.3, max_val * 1.3)

    # Absolute count annotations inside each bar
    for i in range(n_bins):
        unk_add  = unknown_counts[i] / 2.0
        if unk_add > 0:
            unk_sfx = f"(+{int(unk_add)})" if unk_add == int(unk_add) else f"(+{unk_add:.1f})"
        else:
            unk_sfx = ""
        m_total  = male_counts[i]
        f_total  = female_counts[i]
        if m_total > 0:
            lbl = f"{int(raw_male[i]):,}{unk_sfx}"
            ax.text(-m_total / 2.0, centers[i], lbl,
                    ha="center", va="center", fontsize=7.5, color="white",
                    fontweight="bold", clip_on=True)
        if f_total > 0:
            lbl = f"{int(raw_female[i]):,}{unk_sfx}"
            ax.text(f_total / 2.0, centers[i], lbl,
                    ha="center", va="center", fontsize=7.5, color="white",
                    fontweight="bold", clip_on=True)

    tick_vals = np.linspace(0, max_val * 1.1, 5)
    ax.set_xticks(np.concatenate([-tick_vals[1:][::-1], tick_vals[1:]]))
    ax.set_xticklabels(
        [f"{abs(v):,.0f}" for v in np.concatenate([-tick_vals[1:][::-1], tick_vals[1:]])],
        fontsize=9)

    ax.set_yticks(centers)
    ax.set_yticklabels(age_labels, fontsize=7)
    ax.set_xlabel("Number of Cancer Cases", fontsize=8)
    ax.set_ylabel("Age at Exposure (years)", fontsize=8)
    ax.axvline(0, color=_SEMANTIC["neutral"], linewidth=0.8)
    ax.text(-max_val * 0.06, centers[-1] + bin_width * 0.9, "Male",
                ha="center", va="bottom", fontsize=8, color=_SEMANTIC["male"], fontweight="bold")
    ax.text( max_val * 0.06, centers[-1] + bin_width * 0.9, "Female",
                ha="center", va="bottom", fontsize=8, color=_SEMANTIC["female"], fontweight="bold")

    n_unk = int(unknown_counts.sum())
    title_str = f"Cancer Age Pyramid (n={len(ages_v):,} cases)"
    if n_unk > 0:
        title_str += f"\n(unrecognised sex: {n_unk:,} → split equally)"
    ax.set_title(title_str, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")


# Advanced Risk Analysis

def _normalize_sex(value) -> str:
    s = str(value).strip().upper()
    if s in ("M", "MALE", "1"):
        return "M"
    if s in ("F", "FEMALE", "2"):
        return "F"
    return str(value).strip()


def _match_record(record: dict, filters: dict,
                  age_key: str = "age", sex_key: str = "sex_code") -> bool:
    """Return True if a dict record matches the filter criteria."""
    sex_filter = filters.get("sex", "All")
    modality_filter = filters.get("modality", "All")
    region_filter = filters.get("region", "All")
    min_age = filters.get("min_age")
    max_age = filters.get("max_age")

    age = float(record.get(age_key, record.get("age", 0)) or 0)
    sex = _normalize_sex(record.get(sex_key, record.get("sex", "")))
    modality = str(record.get("modality", ""))
    region = str(record.get("region", ""))

    if sex_filter != "All" and sex != sex_filter:
        return False
    if modality_filter != "All" and modality != modality_filter:
        return False
    if region_filter != "All" and region != region_filter:
        return False
    if min_age is not None and age < float(min_age):
        return False
    if max_age is not None and age > float(max_age):
        return False
    return True


def _filter_columns(columns, filters: dict,
                    age_attr: str = "age", sex_attr: str = "sex_code") -> np.ndarray:
    """Return boolean mask of columnar records matching filters.

    Parameters
    ----------
    columns : ExamColumns or EventColumns
        Columnar data object.
    filters : dict
        Filter criteria (sex, modality, region, min_age, max_age).
    age_attr, sex_attr : str
        Attribute names to read on ``columns``.

    Returns
    -------
    np.ndarray[bool_]
        Boolean mask where True means the record passes all filters.
    """
    sex_filter = filters.get("sex", "All")
    modality_filter = filters.get("modality", "All")
    region_filter = filters.get("region", "All")
    min_age = filters.get("min_age")
    max_age = filters.get("max_age")

    mask = np.ones(len(columns), dtype=bool)

    if min_age is not None:
        mask &= getattr(columns, age_attr) >= float(min_age)
    if max_age is not None:
        mask &= getattr(columns, age_attr) <= float(max_age)

    if sex_filter != "All":
        sex_mask = np.array([_normalize_sex(s) == sex_filter for s in getattr(columns, sex_attr)])
        mask &= sex_mask

    if modality_filter != "All":
        mask &= columns.modality == modality_filter

    if region_filter != "All":
        mask &= columns.region == region_filter

    return mask


def union_risk_analysis_filter_values(exam_sources) -> Dict[str, List[str]]:
    """Unique Sex/Modality/Region filter values across multiple sources.

    Each source may be an exam-data list or an already-columnar object
    (``ExamColumns`` or the risk-curve digest's compact columns).  In a
    ``per_year_average`` batch the GUI feeds one digest per year group, so
    a modality that exists only in an early year still appears in the
    dropdowns.
    """
    sexes: set = set()
    modalities: set = set()
    regions: set = set()
    for src in exam_sources or []:
        cols = src if hasattr(src, "age") else _ensure_exam_columns(src)
        if cols is None:
            continue
        for sex_raw in cols.sex_code:
            sex_raw = str(sex_raw).strip()
            if sex_raw:
                sexes.add(_normalize_sex(sex_raw))
        for mod_raw in cols.modality:
            mod_raw = str(mod_raw).strip()
            if mod_raw:
                modalities.add(mod_raw)
        for reg_raw in cols.region:
            reg_raw = str(reg_raw).strip()
            if reg_raw:
                regions.add(reg_raw)
    return {
        "sexes": ["All"] + sorted(sexes),
        "modalities": ["All"] + sorted(modalities),
        "regions": ["All"] + sorted(regions),
    }


_EVENT_DET_FACTOR_CACHE: Dict[str, float] = {}


def _event_detriment_factor(cancer_type) -> float:
    """ICRP 152 detriment factor for one realised incidence event.

    Mirrors ``risk_models.get_cancer_params_cached``'s detriment factor:
    ``det = (k + q_nf*(1-k)) * l`` with ``q_nf = k + q_min*(1-k)``.  Summing
    this over realised cancer events and dividing by dose is an unbiased
    estimator of the simulation's expected detriment per Sv (the quantity
    the ICRP 152 Table 3.9 reference of 5.74 %/Sv is compared against),
    exactly as realised deaths/dose estimate the mortality reference.
    """
    ct = str(cancer_type or "").strip()
    key = ct.lower()
    cached = _EVENT_DET_FACTOR_CACHE.get(key)
    if cached is not None:
        return cached
    if _is_leukaemia_type(ct):
        site = "Leukaemia"
    else:
        site = ct
    p = lookup_cancer_params(site) or {}
    k = float(p.get("k_lethality", 0.0))
    q = float(p.get("q_min", 0.1))
    l_rel = float(p.get("l_rel", 1.0))
    q_nonfatal = k + q * (1.0 - k)
    det_factor = (k + q_nonfatal * (1.0 - k)) * l_rel
    _EVENT_DET_FACTOR_CACHE[key] = det_factor
    return det_factor


def _event_detriment_weights(filtered, types_uniq, inv, sampled_yll: bool) -> np.ndarray:
    """Per-event detriment weights for one filtered event set.

    With ``sampled_yll`` and an event-column set that carries ``QALY_lost``
    (``q_individual x YLL`` in the default ``QALY_mode="full"``), each event's
    weight is its own realised YLL basis ``QALY_lost / 15``; otherwise the
    ICRP 152 population-average ``_event_detriment_factor`` per cancer type.
    """
    if sampled_yll and getattr(filtered._event_columns, "QALY_lost", None) is not None:
        return np.asarray(filtered["QALY_lost"], dtype=np.float64) / _ICRP_AVG_LIFE_LOST_YEARS
    return np.array([_event_detriment_factor(u) for u in types_uniq],
                    dtype=np.float64)[inv]


# Cross-year batch average for the Risk Analysis tab
#
# A "per_year_average" batch discards each year's exam_data/cancer_events
# before loading the next (see year_batch.py's memory note), so
# AdvancedRiskAnalysisPanel cannot recompute the "Simulation" curve from
# whatever survives in memory without silently mislabelling one year's
# numbers as the batch average -- the same problem research_questions.py
# solves for the Region x Age sweep.  The fix follows the same two-part
# pattern: keep a compact per-year DIGEST (just the columns
# compute_filtered_risk_curve's default-dose-field path reads) alive for
# every year of the batch, and recompute + average the curve from every
# digest whenever the user changes filters.
#
# The digest also carries the two Mataac columns, so the Bayesian overlay can
# be averaged across the batch on the same footing as the Simulation curve
# (:func:`compute_batch_averaged_bayesian_curve`).  Two float32 columns per
# exam is ~8 bytes/exam/year -- around 20 MB on a nine-year, 300k-exam-a-year
# archive -- and it buys the only comparison that is otherwise structurally
# impossible to average: without it the two plotted curves describe
# different cohorts and the overlay can appear BELOW a curve it can only
# ever raise.
#
# Still deliberately absent, and still forcing those two comparisons back onto
# the single detail year:
#   * ``ed_msv_no_golikov`` -- not carried by ``ExamColumns`` either, so the
#     "no Golikov" comparison needs full exam dicts whatever we store here;
#   * ``organ_doses_gy`` -- the model-comparison recomputation needs per-organ
#     doses for every exam, which is exactly the payload this digest exists to
#     avoid retaining for the whole batch.

class _RiskDigestExamColumns:
    """Duck-types ``ExamColumns`` for the fields the risk curves read on their
    default-dose-field (``"ed_msv"``) columnar path: ``age``, ``sex_code``,
    ``modality``, ``region``, ``ed_msv``, plus ``mataac_frailty_factor`` and
    ``mataac_index_ced_sv`` for the Bayesian overlay.  Everything else
    ``ExamColumns`` carries (protocol, patient_id, device, equipment,
    date_performed, ...) is left out on purpose -- see the note above
    :func:`build_risk_curve_digest`.

    The two Mataac columns default to ``None`` so a digest without them
    still loads; consumers must probe with
    ``getattr(cols, "mataac_frailty_factor", None) is not None`` and fall back
    to the single-year overlay rather than assume they are present.
    """
    __slots__ = ("age", "sex_code", "modality", "region", "ed_msv",
                 "mataac_frailty_factor", "mataac_index_ced_sv")

    def __init__(self, age, sex_code, modality, region, ed_msv,
                 mataac_frailty_factor=None, mataac_index_ced_sv=None):
        self.age = age
        self.sex_code = sex_code
        self.modality = modality
        self.region = region
        self.ed_msv = ed_msv
        self.mataac_frailty_factor = mataac_frailty_factor
        self.mataac_index_ced_sv = mataac_index_ced_sv

    def __len__(self) -> int:
        return len(self.age)


class _RiskDigestEventColumns:
    """Duck-types ``EventColumns`` for the fields ``compute_filtered_risk_curve``
    reads: ``age_at_exposure``, ``sex``, ``modality``, ``region``,
    ``cancer_type``, ``died``, plus ``QALY_lost`` for the sampled-YLL
    detriment basis.  ``QALY_lost`` defaults to ``None`` so a digest
    without it still loads; consumers
    must probe with ``getattr(cols, "QALY_lost", None) is not None``.
    """
    __slots__ = ("age_at_exposure", "sex", "modality", "region", "cancer_type",
                 "died", "QALY_lost")

    def __init__(self, age_at_exposure, sex, modality, region, cancer_type,
                 died, QALY_lost=None):
        self.age_at_exposure = age_at_exposure
        self.sex = sex
        self.modality = modality
        self.region = region
        self.cancer_type = cancer_type
        self.died = died
        self.QALY_lost = QALY_lost

    def __len__(self) -> int:
        return len(self.age_at_exposure)

    def subset(self, mask):
        """Same shape as ``EventColumns.subset`` -- used by adv_ui's
        ``_filter_events_by_visible_cancers`` on a whole digest."""
        return _RiskDigestEventColumns(
            age_at_exposure=self.age_at_exposure[mask],
            sex=self.sex[mask],
            modality=self.modality[mask],
            region=self.region[mask],
            cancer_type=self.cancer_type[mask],
            died=self.died[mask],
            QALY_lost=(None if self.QALY_lost is None
                       else self.QALY_lost[mask]),
        )


def build_risk_curve_digest(exam_columns, event_columns) -> Dict[str, Any]:
    """Compact per-year snapshot for the Risk Analysis "Simulation" curve.

    Called once per year of a ``per_year_average`` batch while that year's
    ``exam_data``/``cancer_events`` are still alive (see
    ``year_batch.run_yearly_simulation``), so ``{"exam_columns":
    _RiskDigestExamColumns(...), "event_columns": _RiskDigestEventColumns(...)}``
    can be retained for the WHOLE batch and fed straight back into
    :func:`compute_filtered_risk_curve` for any filter combination the user
    picks later, at a fraction of the memory of the full columns.
    """
    # A year with no exams contributes nothing to any curve; None (not an
    # empty digest) keeps the batch helpers' "is there a digest" check honest.
    if exam_columns is None or len(exam_columns) == 0:
        return None
    _frailty = getattr(exam_columns, "mataac_frailty_factor", None)
    _ced = getattr(exam_columns, "mataac_index_ced_sv", None)
    ex = _RiskDigestExamColumns(
        age=np.asarray(exam_columns.age, dtype=np.float32),
        sex_code=np.asarray(exam_columns.sex_code),
        modality=np.asarray(exam_columns.modality),
        region=np.asarray(exam_columns.region),
        ed_msv=np.asarray(exam_columns.ed_msv, dtype=np.float32),
        mataac_frailty_factor=(None if _frailty is None
                               else np.asarray(_frailty, dtype=np.float32)),
        mataac_index_ced_sv=(None if _ced is None
                             else np.asarray(_ced, dtype=np.float32)),
    )
    if event_columns is not None and len(event_columns) > 0:
        _qaly = getattr(event_columns, "QALY_lost", None)
        ev = _RiskDigestEventColumns(
            age_at_exposure=np.asarray(event_columns.age_at_exposure, dtype=np.float32),
            sex=np.asarray(event_columns.sex),
            modality=np.asarray(event_columns.modality),
            region=np.asarray(event_columns.region),
            cancer_type=np.asarray(event_columns.cancer_type),
            died=np.asarray(event_columns.died, dtype=bool),
            QALY_lost=(None if _qaly is None
                       else np.asarray(_qaly, dtype=np.float32)),
        )
    else:
        ev = _RiskDigestEventColumns(
            age_at_exposure=np.zeros(0, dtype=np.float32),
            sex=np.zeros(0, dtype="<U1"),
            modality=np.zeros(0, dtype=object),
            region=np.zeros(0, dtype=object),
            cancer_type=np.zeros(0, dtype=object),
            died=np.zeros(0, dtype=bool),
            QALY_lost=np.zeros(0, dtype=np.float32),
        )
    return {"exam_columns": ex, "event_columns": ev}


def _accumulate_filtered_bins(exam_data: list, cancer_events: list, n_iterations: int,
                              filters: dict, bin_width: int = 5,
                              ddref_solid: float = 2.0, ddref_leuk: float = 1.0,
                              dose_field: str = "ed_msv",
                              exam_columns=None, event_columns=None,
                              sampled_yll: bool = False) -> Dict:
    """Accumulation stage of :func:`compute_filtered_risk_curve`.

    Returns the raw per-bin sums -- ``dose_by_bin`` (Sv, already multiplied by
    ``n_iterations``), ``cases_by_bin``/``deaths_by_bin``/``detriment_by_bin``
    (DDREF-weighted event counts), ``detriment_sq_by_bin`` (sum of squared
    event weights, for the weighted-Poisson SE) and ``n_exams`` -- so the batch
    combiner can pool them across years before any ratio is formed.  Keeping
    accumulation and ratio/CI finalisation apart is what lets the cross-year
    curve be ``Σ w_y·cases / Σ w_y·dose`` rather than an average of per-year
    ratios (which bins with little dose make noisy).
    """
    exam_columns = exam_columns or _ensure_exam_columns(exam_data)
    event_columns = event_columns or _ensure_event_columns(cancer_events)
    # Sampled-YLL detriment is only possible when the event columns carry
    # QALY_lost; a digest recorded before that column was added reports the
    # l_rel-factor basis via this flag instead.
    _yll_used = bool(sampled_yll
                    and getattr(event_columns, "QALY_lost", None) is not None)

    bin_edges = np.arange(0, 101, bin_width)
    n_bins = len(bin_edges) - 1
    dose_by_bin = np.zeros(n_bins)
    cases_by_bin = np.zeros(n_bins)
    deaths_by_bin = np.zeros(n_bins)
    detriment_by_bin = np.zeros(n_bins)      # Σ (det_factor · scale) per bin
    detriment_sq_by_bin = np.zeros(n_bins)   # Σ (det_factor · scale)² for CI
    n_exams = 0
    exam_mask = None
    filtered_exams = None

    # Use columnar filter if available
    if exam_columns is not None:
        exam_mask = _filter_columns(exam_columns, filters)
        n_exams = int(exam_mask.sum())
        # Extract filtered data from columns
        filtered_ages = exam_columns.age[exam_mask]
        filtered_dose = exam_columns.ed_msv[exam_mask] if dose_field == "ed_msv" else np.zeros(exam_mask.sum())
        # Note: ed_msv_no_golikov not in ExamColumns yet, fall back to dict for that field
        if dose_field != "ed_msv":
            filtered_exams = [ex for ex in exam_data if _match_record(ex, filters)]
    else:
        # Dict-based filtering path for non-columnar exam_data
        filtered_exams = [ex for ex in exam_data if _match_record(ex, filters)]
        n_exams = len(filtered_exams)

    if n_exams:
        # Dose accumulation: np.add.at applies duplicate-index additions in array
        # order, so float summation order — and therefore the exact bit pattern of
        # the totals — is deterministic and reproducible across runs.
        _iter_scale = float(max(1, n_iterations))
        if exam_columns is not None and dose_field == "ed_msv":
            _ages_ex = np.asarray(filtered_ages, dtype=np.float64)
            _dose_ex = np.asarray(filtered_dose, dtype=np.float64)
        else:
            _flist = filtered_exams or []
            _ages_ex = np.array([float(ex.get("age", 0) or 0) for ex in _flist], dtype=np.float64)
            _dose_ex = np.array(
                [float(ex.get(dose_field, ex.get("ed_msv", 0)) or 0) for ex in _flist],
                dtype=np.float64,
            )
        if _ages_ex.size:
            _idx_ex = np.clip((_ages_ex // bin_width).astype(np.int64), 0, n_bins - 1)
            np.add.at(dose_by_bin, _idx_ex, _dose_ex / 1000.0 * _iter_scale)

        # Process events (case/death/detriment accumulation) — vectorised
        scale_solid, scale_leuk = _ddref_scales(ddref_solid, ddref_leuk)

        _ev_ages = None
        _ev_types = None
        _ev_died = None
        filtered = _filter_events(event_columns, filters)
        if filtered is not None:
            _ev_ages = np.asarray(filtered["age_at_exposure"], dtype=np.float64)
            _ev_types = [str(t) if t else "" for t in filtered["cancer_type"]]
            _ev_died = np.asarray(filtered["died"]).astype(bool)

        if _ev_ages is not None and _ev_ages.size:
            _idx_ev = np.clip((_ev_ages // bin_width).astype(np.int64), 0, n_bins - 1)
            # Map each event's cancer type to its DDREF scale + detriment factor
            # via the unique-type table (cheap; avoids a per-event Python call).
            _types_arr = np.asarray(_ev_types, dtype=object)
            _uniq, _inv = np.unique(_types_arr, return_inverse=True)
            _s_uniq = np.array(
                [scale_leuk if _is_leukaemia_type(u) else scale_solid for u in _uniq],
                dtype=np.float64,
            )
            _s_arr = _s_uniq[_inv]
            _detw_arr = _event_detriment_weights(
                filtered, _uniq, _inv, sampled_yll) * _s_arr
            np.add.at(cases_by_bin, _idx_ev, _s_arr)
            np.add.at(deaths_by_bin, _idx_ev, np.where(_ev_died, _s_arr, 0.0))
            np.add.at(detriment_by_bin, _idx_ev, _detw_arr)
            np.add.at(detriment_sq_by_bin, _idx_ev, _detw_arr * _detw_arr)

    return {
        "bin_edges": bin_edges,
        "dose_by_bin": dose_by_bin,
        "cases_by_bin": cases_by_bin,
        "deaths_by_bin": deaths_by_bin,
        "detriment_by_bin": detriment_by_bin,
        "detriment_sq_by_bin": detriment_sq_by_bin,
        "detriment_yll_sampled": _yll_used,
        "n_exams": n_exams,
    }


def _finalise_curve(bin_edges: np.ndarray, dose_by_bin: np.ndarray,
                    cases_by_bin: np.ndarray, deaths_by_bin: np.ndarray,
                    detriment_by_bin: np.ndarray, detriment_sq_by_bin: np.ndarray,
                    n_exams: int, dose_field: str = "ed_msv",
                    se_cases_num=None, se_deaths_num=None, se_detriment_num=None) -> Dict:
    """Finalisation stage of :func:`compute_filtered_risk_curve`.

    Turns raw per-bin sums into per-Sv rates plus their 95 % bands.
    ``se_*_num`` are the "effective counts" whose square root forms the SE
    numerator -- they default to the plain sums (single cohort), while the
    batch combiner passes ``Σ w_y²·sum_y`` so the interval narrows with the
    number of pooled years instead of being read off one year's counts.
    """
    n_bins = len(bin_edges) - 1
    risk_per_sv = np.zeros(n_bins)
    risk_low = np.zeros(n_bins)
    risk_high = np.zeros(n_bins)
    death_per_sv = np.zeros(n_bins)
    death_low = np.zeros(n_bins)
    death_high = np.zeros(n_bins)
    detriment_per_sv = np.zeros(n_bins)
    detriment_low = np.zeros(n_bins)
    detriment_high = np.zeros(n_bins)

    if se_cases_num is None:
        se_cases_num = cases_by_bin
    if se_deaths_num is None:
        se_deaths_num = deaths_by_bin
    if se_detriment_num is None:
        se_detriment_num = detriment_sq_by_bin

    mask = dose_by_bin > 0
    risk_per_sv[mask] = cases_by_bin[mask] / dose_by_bin[mask]
    death_per_sv[mask] = deaths_by_bin[mask] / dose_by_bin[mask]
    detriment_per_sv[mask] = detriment_by_bin[mask] / dose_by_bin[mask]

    case_se = np.zeros(n_bins)
    death_se = np.zeros(n_bins)
    detriment_se = np.zeros(n_bins)
    case_se[mask] = np.sqrt(np.maximum(se_cases_num[mask], 0.0)) / dose_by_bin[mask]
    death_se[mask] = np.sqrt(np.maximum(se_deaths_num[mask], 0.0)) / dose_by_bin[mask]
    # Weighted-Poisson SE: sqrt(Σ w²) / dose (reduces to sqrt(N)/dose when w=1).
    detriment_se[mask] = np.sqrt(np.maximum(se_detriment_num[mask], 0.0)) / dose_by_bin[mask]
    _apply_ci_bands(risk_low, risk_high, death_low, death_high,
                    detriment_low, detriment_high,
                    risk_per_sv, death_per_sv, detriment_per_sv,
                    case_se, death_se, detriment_se, mask)

    return {
        "bin_edges": bin_edges,
        "centers": (bin_edges[:-1] + bin_edges[1:]) / 2,
        "dose_sv": dose_by_bin,
        "cases": cases_by_bin,
        "deaths": deaths_by_bin,
        "risk_per_sv": risk_per_sv,
        "risk_low": risk_low,
        "risk_high": risk_high,
        "death_per_sv": death_per_sv,
        "death_low": death_low,
        "death_high": death_high,
        "detriment": detriment_by_bin,
        "detriment_per_sv": detriment_per_sv,
        "detriment_low": detriment_low,
        "detriment_high": detriment_high,
        "detriment_sq": detriment_sq_by_bin,
        "n_exams": n_exams,
        "dose_field": dose_field,
    }


def compute_filtered_risk_curve(exam_data: list, cancer_events: list, n_iterations: int,
                                filters: dict, bin_width: int = 5,
                                ddref_solid: float = 2.0, ddref_leuk: float = 1.0,
                                dose_field: str = "ed_msv", exam_columns=None, event_columns=None,
                                sampled_yll: bool = False) -> Dict:
    """Compute age-stratified risk curve with optional columnar data.
    ...
    """
    acc = _accumulate_filtered_bins(
        exam_data, cancer_events, n_iterations, filters, bin_width=bin_width,
        ddref_solid=ddref_solid, ddref_leuk=ddref_leuk, dose_field=dose_field,
        exam_columns=exam_columns, event_columns=event_columns,
        sampled_yll=sampled_yll)
    basis = "sampled" if acc["detriment_yll_sampled"] else "icrp_average"
    if acc["n_exams"] == 0:
        n_bins = len(acc["bin_edges"]) - 1
        out = _empty_curve_result(acc["bin_edges"], n_bins, dose_field)
    else:
        out = _finalise_curve(
            acc["bin_edges"], acc["dose_by_bin"], acc["cases_by_bin"],
            acc["deaths_by_bin"], acc["detriment_by_bin"],
            acc["detriment_sq_by_bin"], acc["n_exams"], dose_field=dose_field)
    out["detriment_yll_basis"] = basis
    if sampled_yll and basis != "sampled":
        out["detriment_yll_fallback"] = True
    return out


# Cross-year curves are combined as a RATIO OF EQUAL-WEIGHTED SUMS, never as
# an average of per-year ratios: with ``w_y = 1/(I_y · n_years)``, each year's
# raw per-bin sums are divided by its own iteration count (giving that year's
# per-one-simulated-year expectation) and by the year count, then added --
# so ``risk_per_sv = (Σ_y w_y·cases_y) / (Σ_y w_y·dose_y)``.  Per-iteration
# rates are averaged across years; a year never gets extra weight for having
# needed more MC iterations to converge or more exams.  This is the same
# normalisation ``year_batch._to_per_iteration_rates`` applies to every
# other cumulative total the platform averages across years.
#
# The 95 % bands stay valid Poisson intervals of the weighted sum:
# ``SE = sqrt(Σ_y w_y²·count_y) / Dose`` (and likewise for deaths and the
# detriment sum-of-squares).  With one year this reduces exactly to the
# single-cohort formula; with k identical years it narrows by ~1/√k.


def _iter_year_digests(year_digests):
    """Yield ``(year_label, iterations, digest)`` for each usable digest.

    ``year_digests`` maps a year-group label to its
    :func:`build_risk_curve_digest` dict; the digest's own ``"iterations"``
    (set by ``year_batch`` while that year's stats were alive) is that year's
    OWN MC iteration count -- required both to build the dose sums and to
    weight the year equally in the pool.  Digests that are missing, have no
    exams, or lack a positive iteration count are skipped (not counted as
    zero years -- a missing digest means the payload could not be built,
    not that the year contributed no dose); the
    fraction of the batch actually contributing is reported via ``n_years``
    and ``years`` on the result.
    """
    for year, digest in (year_digests or {}).items():
        if not digest:
            continue
        exam_columns = digest.get("exam_columns")
        if exam_columns is None or len(exam_columns) == 0:
            continue
        year_iters = int(digest.get("iterations", 0) or 0)
        if year_iters <= 0:
            continue
        yield year, year_iters, digest


def _pool_year_bins(per_year_accs, n_bins):
    """Pool ``(iterations, acc)`` pairs into equal-weighted batch sums.

    ``acc`` holds one year's raw per-bin sums (``dose_by_bin`` already
    iteration-scaled).  Returns ``(dose, cases, deaths, detriment,
    detriment_sq, se_cases, se_deaths, se_detriment)`` where the three
    ``se_*`` arrays are the effective counts whose square root forms the SE
    numerator, i.e. ``Σ_y w_y²·count_y`` (see the note above).
    """
    n_years = len(per_year_accs)
    dose = np.zeros(n_bins)
    cases = np.zeros(n_bins)
    deaths = np.zeros(n_bins)
    detriment = np.zeros(n_bins)
    detriment_sq = np.zeros(n_bins)
    se_c = np.zeros(n_bins)
    se_d = np.zeros(n_bins)
    se_t = np.zeros(n_bins)
    for year_iters, acc in per_year_accs:
        w = 1.0 / (float(year_iters) * n_years)
        dose += w * acc["dose_by_bin"]
        cases += w * acc["cases_by_bin"]
        deaths += w * acc["deaths_by_bin"]
        detriment += w * acc["detriment_by_bin"]
        detriment_sq += w * acc["detriment_sq_by_bin"]
        w2 = w * w
        se_c += w2 * acc["cases_by_bin"]
        se_d += w2 * acc["deaths_by_bin"]
        se_t += w2 * acc["detriment_sq_by_bin"]
    return dose, cases, deaths, detriment, detriment_sq, se_c, se_d, se_t


def _empty_batch_curve(bin_edges, dose_field="ed_msv"):
    out = _empty_curve_result(bin_edges, len(bin_edges) - 1, dose_field)
    out["n_years"] = 0
    out["years"] = []
    out["_averaged_across_years"] = 0
    return out


def compute_batch_averaged_risk_curve(
    year_digests: Dict[str, Any],
    filters: dict,
    bin_width: int = 5,
    ddref_solid: float = 2.0,
    ddref_leuk: float = 1.0,
    sampled_yll: bool = False,
) -> Dict[str, Any]:
    """Equal-weighted cross-year "Simulation" risk curve for a
    ``per_year_average`` batch, for any filter combination.

    ``year_digests`` maps a year-group label to that year's
    :func:`build_risk_curve_digest` dict (with ``"iterations"`` = that
    year's own iteration count).  Each year is accumulated by
    :func:`_accumulate_filtered_bins` at its own iteration count and the
    raw sums are pooled with ``w_y = 1/(I_y·n_years)`` -- combined
    numerator over combined denominator, never an average of per-year
    ratios -- then finalised by :func:`_finalise_curve` with the
    weighted-Poisson SEs.

    ``dose_sv``/``cases``/``deaths``/``detriment`` come out on a
    per-one-simulated-year basis (equal-weighted mean across the
    contributing years); ``n_exams`` is the SUM of matched examinations
    over all contributing years.  ``n_years``/``years`` report which years
    contributed; ``_averaged_across_years`` mirrors ``n_years`` for the
    plot/report labels.
    """
    bin_edges = np.arange(0, 101, bin_width)
    n_bins = len(bin_edges) - 1
    # Sampled-YLL detriment is applied for ALL pooled years or none: if any
    # contributing digest predates the QALY_lost column, every year uses the
    # l_rel factor so the pool stays on one basis.
    _eff_sampled = sampled_yll
    _yll_fallback = False
    if sampled_yll:
        for _y, _i, _d in _iter_year_digests(year_digests):
            if getattr(_d.get("event_columns"), "QALY_lost", None) is None:
                _eff_sampled = False
                _yll_fallback = True
                break
    per_year = []
    n_exams = 0
    for _year, year_iters, digest in _iter_year_digests(year_digests):
        acc = _accumulate_filtered_bins(
            [], [], year_iters, filters, bin_width=bin_width,
            ddref_solid=ddref_solid, ddref_leuk=ddref_leuk,
            exam_columns=digest.get("exam_columns"),
            event_columns=digest.get("event_columns"),
            sampled_yll=_eff_sampled)
        per_year.append((_year, year_iters, acc))
        n_exams += acc["n_exams"]
    if not per_year:
        out = _empty_batch_curve(bin_edges)
        out["detriment_yll_basis"] = "sampled" if _eff_sampled else "icrp_average"
        return out

    dose, cases, deaths, detriment, detriment_sq, se_c, se_d, se_t = \
        _pool_year_bins([(iters, acc) for _, iters, acc in per_year], n_bins)
    out = _finalise_curve(
        bin_edges, dose, cases, deaths, detriment, detriment_sq, n_exams,
        dose_field="ed_msv",
        se_cases_num=se_c, se_deaths_num=se_d, se_detriment_num=se_t)
    out["n_years"] = len(per_year)
    out["years"] = [y for y, _, _ in per_year]
    out["_averaged_across_years"] = len(per_year)
    out["detriment_yll_basis"] = "sampled" if _eff_sampled else "icrp_average"
    if _yll_fallback:
        out["detriment_yll_fallback"] = True
    return out


def batch_bayesian_curve_supported(
    year_digests: Dict[str, Any],
    compare_model: bool = False,
    use_no_golikov_dose: bool = False,
) -> bool:
    """True if the Bayesian overlay can be averaged across the whole batch.

    Requires that no active comparison needs data the per-year digest does not
    keep -- the model comparison needs per-organ doses, the no-Golikov
    comparison needs ``ed_msv_no_golikov`` (which ``ExamColumns`` does not
    carry either) -- and that every contributing digest actually has the two
    Mataac columns, which digests recorded before they were added do not.
    """
    if compare_model or use_no_golikov_dose:
        return False
    seen = False
    for _year, _iters, digest in _iter_year_digests(year_digests):
        if getattr(digest.get("exam_columns"), "mataac_frailty_factor", None) is None:
            return False
        seen = True
    return seen


def _merge_mataac_dose_groups(per_year: List[list], n_years: int) -> list:
    """Combine each year's Mataac dose-group table into per-year averages.

    ``n_exams``/``dose_sv`` are extensive and are divided by the batch size so
    they read as "per simulated year", matching the averaged curve beside
    them.  ``frailty_factor`` is an intensity and is combined as an
    exam-weighted mean over the years that actually had exams in the group,
    which reproduces exactly what a single year reports when only one year
    contributes.
    """
    if n_years <= 0:
        return []
    order: List[str] = []
    acc: Dict[str, Dict[str, float]] = {}
    for groups in per_year:
        for g in (groups or []):
            name = g.get("group")
            if name not in acc:
                acc[name] = {"n": 0.0, "dose_sv": 0.0, "frailty_weighted": 0.0}
                order.append(name)
            n = float(g.get("n_exams", 0) or 0)
            acc[name]["n"] += n
            acc[name]["dose_sv"] += float(g.get("dose_sv", 0.0) or 0.0)
            acc[name]["frailty_weighted"] += n * float(g.get("frailty_factor", 1.0) or 1.0)
    out = []
    for name in order:
        a = acc[name]
        if a["n"] <= 0:
            continue
        out.append({
            "group": name,
            "n_exams": int(round(a["n"] / n_years)),
            "dose_sv": a["dose_sv"] / n_years,
            "frailty_factor": min(1.0, a["frailty_weighted"] / a["n"]),
        })
    return out


def compute_batch_averaged_bayesian_curve(
    year_digests: Dict[str, Any],
    filters: dict,
    bin_width: int = 5,
    ddref_solid: float = 2.0,
    ddref_leuk: float = 1.0,
    use_no_golikov_dose: bool = False,
    selected_model: str = "REIC",
    comparison_model: str = "LAR",
    population: str = "Euro-American",
    compare_mataac: bool = False,
    sampled_yll: bool = False,
) -> Dict[str, Any]:
    """Equal-weighted cross-year Bayesian comparison curve for a batch.

    The batch twin of :func:`compute_batch_averaged_risk_curve`, combined the
    same way so the two plotted curves describe the SAME cohort.  Without it
    the overlay is the single detail year while the Simulation curve is the
    batch mean, and the pair can be ordered either way round -- including
    the overlay sitting below a curve that removing frailty can only raise.

    The per-bin adjustment factors are formed from POOLED exam-side sums
    (the dose-weighted inverse-frailty term is itself a combined
    numerator/denominator ratio, so it pools exactly); per-year events are
    then accumulated with those pooled bin weights and combined with the
    ``w_y`` weights like the Simulation curve.

    Call :func:`batch_bayesian_curve_supported` first: comparisons that need
    per-organ doses or the no-Golikov dose field cannot be rebuilt from the
    digest and must stay on the detail year.
    """
    dose_field = "ed_msv_no_golikov" if use_no_golikov_dose else "ed_msv"
    bin_edges = np.arange(0, 101, bin_width)
    n_bins = len(bin_edges) - 1

    # Same all-or-none sampled-YLL rule as compute_batch_averaged_risk_curve.
    _eff_sampled = sampled_yll
    _yll_fallback = False
    if sampled_yll:
        for _y, _i, _d in _iter_year_digests(year_digests):
            if getattr(_d.get("event_columns"), "QALY_lost", None) is None:
                _eff_sampled = False
                _yll_fallback = True
                break

    # Exam-side accumulation per year (dose iteration-scaled; frailty and
    # group stats cohort-scaled, pooled as plain sums below).
    per_year = []
    pooled_frailty_sum = np.zeros(n_bins)
    pooled_frailty_weight = np.zeros(n_bins)
    n_exams = 0
    for year, year_iters, digest in _iter_year_digests(year_digests):
        acc = _accumulate_bayesian_exam_bins(
            [], filters, bin_width, year_iters,
            dose_field=dose_field,
            compare_model=False,  # per-organ doses are not in the digest
            compare_mataac=compare_mataac,
            exam_columns=digest.get("exam_columns"))
        if acc is None:
            # The year's digest exists but no exam matches the filters:
            # still a contributing year (zero dose), keep it in n_years.
            acc = {"bin_edges": bin_edges, "dose_by_bin": np.zeros(n_bins),
                   "frailty_dose_sum": np.zeros(n_bins),
                   "frailty_dose_weight": np.zeros(n_bins),
                   "bin_exam_lists": None, "group_stats": None, "n_exams": 0}
        per_year.append((year, year_iters, acc, digest.get("event_columns")))
        pooled_frailty_sum += acc["frailty_dose_sum"]
        pooled_frailty_weight += acc["frailty_dose_weight"]
        n_exams += acc["n_exams"]
    if not per_year:
        out = _empty_batch_curve(bin_edges, dose_field)
        out["detriment_yll_basis"] = "sampled" if _eff_sampled else "icrp_average"
        return out

    n_years = len(per_year)
    # Pooled per-bin weights: the dose-weighted frailty term is itself a
    # combined numerator/denominator ratio, so pooling the sums gives the
    # correct all-years factor (identical to a single run at n_years = 1).
    pooled_exam_acc = {
        "bin_exam_lists": None,
        "frailty_dose_sum": pooled_frailty_sum,
        "frailty_dose_weight": pooled_frailty_weight,
    }
    w_solid, w_leuk, _model_ratio, avg_frailty = _bayesian_bin_weights(
        pooled_exam_acc, n_bins, ddref_solid, ddref_leuk,
        compare_model=False, compare_mataac=compare_mataac,
        selected_model=selected_model, comparison_model=comparison_model,
        population=population)

    # Event-side accumulation per year with the pooled bin weights, then
    # pooled with the same w_y = 1/(I_y·n_years) as the Simulation curve.
    dose = np.zeros(n_bins)
    cases = np.zeros(n_bins)
    deaths = np.zeros(n_bins)
    detriment = np.zeros(n_bins)
    detriment_sq = np.zeros(n_bins)
    se_c = np.zeros(n_bins)
    se_d = np.zeros(n_bins)
    se_t = np.zeros(n_bins)
    dose_groups = []
    for _year, year_iters, acc, event_columns in per_year:
        w = 1.0 / (float(year_iters) * n_years)
        dose += w * acc["dose_by_bin"]
        ev = _accumulate_bayesian_event_bins(
            event_columns, filters, bin_width, w_solid, w_leuk,
            sampled_yll=_eff_sampled)
        cases += w * ev["cases_by_bin"]
        deaths += w * ev["deaths_by_bin"]
        detriment += w * ev["detriment_by_bin"]
        detriment_sq += w * ev["detriment_sq_by_bin"]
        w2 = w * w
        se_c += w2 * ev["cases_by_bin"]
        se_d += w2 * ev["deaths_by_bin"]
        se_t += w2 * ev["detriment_sq_by_bin"]
        if acc["group_stats"]:
            dose_groups.append(_mataac_group_rows(acc["group_stats"]))

    out = _finalise_curve(
        bin_edges, dose, cases, deaths, detriment, detriment_sq, n_exams,
        dose_field=dose_field,
        se_cases_num=se_c, se_deaths_num=se_d, se_detriment_num=se_t)
    out["n_years"] = n_years
    out["years"] = [y for y, _, _, _ in per_year]
    out["_averaged_across_years"] = n_years
    out["detriment_yll_basis"] = "sampled" if _eff_sampled else "icrp_average"
    if _yll_fallback:
        out["detriment_yll_fallback"] = True
    if dose_groups:
        out["mataac_dose_groups"] = _merge_mataac_dose_groups(dose_groups, n_years)
    if compare_mataac:
        out["avg_frailty_by_bin"] = avg_frailty
    return out


def summarize_mataac_strata(exam_columns_sources) -> Optional[Dict[str, Any]]:
    """All-years-combined Mataac competing-mortality strata summary.

    Every source's exams carry ``mataac_index_ced_sv`` -- the patient's
    cumulative CT effective dose within THAT exam's own calendar year, as
    assigned by the engine -- so summing per-source counts and doses pools
    per-calendar-year stratum membership without recomputing cumulative
    dose over the pooled period (which would promote patients into strata
    they never belonged to).  All modalities are counted, matching how the
    engine applies the correction to every exam of a stratum patient.

    Counts and collective dose are SUMMED over sources (not averaged);
    the mean adjustment factor per stratum is the per-exam mean -- the
    same weighting the single-year dose-group table uses
    (``frailty_sum / n``).

    Returns ``None`` when no source carries the Mataac columns (e.g.
    digests recorded before they were added).
    """
    totals = {name: {"n": 0, "dose_sv": 0.0, "frailty_sum": 0.0}
              for name, _ in _MATAAC_CED_STRATA}
    n_all = 0
    seen = False
    _thresh = np.array([t for _, t in _MATAAC_CED_STRATA[:-1]], dtype=np.float64)
    for cols in exam_columns_sources or []:
        if cols is None or len(cols) == 0:
            continue
        ced = getattr(cols, "mataac_index_ced_sv", None)
        if ced is None:
            continue
        seen = True
        ced = np.asarray(ced, dtype=np.float64)
        frailty = getattr(cols, "mataac_frailty_factor", None)
        frailty = (np.ones(len(cols), dtype=np.float64) if frailty is None
                   else np.asarray(frailty, dtype=np.float64))
        ed_sv = np.asarray(cols.ed_msv, dtype=np.float64) / 1000.0
        grp = np.searchsorted(_thresh, ced, side="right")
        n_all += len(cols)
        for gi, (gname, _t) in enumerate(_MATAAC_CED_STRATA):
            m = grp == gi
            if m.any():
                t = totals[gname]
                t["n"] += int(m.sum())
                t["dose_sv"] += float(ed_sv[m].sum())
                t["frailty_sum"] += float(frailty[m].sum())
    if not seen:
        return None
    groups = [{
        "group": name,
        "n_exams": t["n"],
        "dose_sv": t["dose_sv"],
        "frailty_factor": (min(1.0, t["frailty_sum"] / t["n"]) if t["n"] else 1.0),
    } for name, t in ((gname, totals[gname]) for gname, _ in _MATAAC_CED_STRATA)]
    n_total = sum(t["n"] for t in totals.values())
    f_total = sum(t["frailty_sum"] for t in totals.values())
    return {
        "groups": groups,
        "n_exams": n_all,
        "n_stratified": n_total,
        "dose_sv": sum(t["dose_sv"] for t in totals.values()),
        "frailty_factor": (min(1.0, f_total / n_total) if n_total else 1.0),
    }


def format_mataac_strata_summary(strata: dict, scope: str,
                                 heading: bool = True) -> str:
    """Render :func:`summarize_mataac_strata` in manuscript-sentence form:
    examinations (n and %), share of collective dose (%), mean adjustment
    factor per stratum, and a total row.  ``scope`` names which years and
    which examinations were counted; ``heading=False`` drops the title
    lines for callers that print their own.
    """
    n_all = max(1, int(strata.get("n_exams", 0)))
    d_all = float(strata.get("dose_sv", 0.0) or 0.0)
    lines = ([
        "-" * 60,
        "MATAAC COMPETING-MORTALITY STRATA -- ALL YEARS COMBINED",
    ] if heading else []) + [
        "Stratum = patient's cumulative CT effective dose in the exam's own",
        "calendar year (as the engine assigns it); the correction then applies",
        "to all of that patient's exams in that year regardless of modality.",
        f"Counted: {scope}.",
        "",
        f"{'Stratum':<18} {'Exams':>10} {'Exams %':>8} {'Dose Sv':>10} "
        f"{'Dose %':>8} {'Competing-mortality adjustment factor':>38}",
        "-" * 100,
    ]
    for g in strata.get("groups", []):
        lines.append(
            f"{g['group']:<18} {g['n_exams']:>10,} "
            f"{100.0 * g['n_exams'] / n_all:>7.1f}% "
            f"{g['dose_sv']:>10.4f} "
            f"{100.0 * g['dose_sv'] / d_all if d_all > 0 else 0.0:>7.1f}% "
            f"{g['frailty_factor']:>38.3f}"
        )
    lines.append("-" * 100)
    lines.append(
        f"{'All strata':<18} {strata.get('n_stratified', 0):>10,} "
        f"{100.0 * strata.get('n_stratified', 0) / n_all:>7.1f}% "
        f"{d_all:>10.4f} {100.0:>7.1f}% {strata.get('frailty_factor', 1.0):>38.3f}"
    )
    return "\n".join(lines)



_WEIGHT_ADJ_REGIONS = ("Chest", "Abdomen")
_WEIGHT_ADJ_FIELDS = ("n_eligible", "n_no_weight", "n_scaled",
                      "factor_sum", "dose_ref_sv", "dose_adj_sv")


def _empty_weight_adj_row() -> Dict[str, float]:
    row = {k: 0 for k in _WEIGHT_ADJ_FIELDS}
    row["factor_max"] = 1.0
    return row


def summarize_weight_adjustment(exam_data) -> Optional[Dict[str, Any]]:
    """Per-region summary of the Golikov and Druzhinina weight adjustment
    over adult chest and abdominal CT (``_golikov_eligible`` exams).

    Per region: eligible exams, exams without a recorded weight, exams
    whose dose was scaled (factor > 1), the sum and maximum of the scale
    factor over scaled exams, and the collective effective dose before
    (``ed_msv_no_golikov``) and after (``ed_msv``) the adjustment, in Sv.
    The factor of an exam is ``ed_msv / ed_msv_no_golikov``, which also
    holds for exams merged from several acquisition series.  Returns
    ``None`` when the exams carry no eligibility flag.
    """
    if exam_data is None or len(exam_data) == 0:
        return None
    cols = getattr(exam_data, "_cols", None)

    def _col(key):
        if cols is not None:
            return cols.get(key)
        vals = [ex.get(key) for ex in exam_data]
        return vals if any(v is not None for v in vals) else None

    elig = _col("_golikov_eligible")
    if elig is None:
        return None
    n = len(elig)
    region = _col("region") or [None] * n
    ed = _col("ed_msv") or [0.0] * n
    ed_ref = _col("ed_msv_no_golikov") or ed
    no_wt = _col("_golikov_no_weight") or [False] * n
    out = {r: _empty_weight_adj_row() for r in _WEIGHT_ADJ_REGIONS}
    for i in range(n):
        t = out.get(region[i]) if elig[i] else None
        if t is None:
            continue
        adj = float(ed[i] or 0.0)
        ref = float(ed_ref[i] or 0.0) or adj
        t["n_eligible"] += 1
        t["dose_adj_sv"] += adj / 1000.0
        t["dose_ref_sv"] += ref / 1000.0
        if no_wt[i]:
            t["n_no_weight"] += 1
        f = adj / ref if ref > 0 else 1.0
        if f > 1.0 + 1e-9:
            t["n_scaled"] += 1
            t["factor_sum"] += f
            t["factor_max"] = max(t["factor_max"], f)
    return {"regions": out}


def merge_weight_adjustments(summaries) -> Optional[Dict[str, Any]]:
    """Sum :func:`summarize_weight_adjustment` results over year groups
    (counts and doses added, maximum factor kept)."""
    summaries = [x for x in summaries or [] if x]
    if not summaries:
        return None
    out = {r: _empty_weight_adj_row() for r in _WEIGHT_ADJ_REGIONS}
    for x in summaries:
        for r, t in (x.get("regions") or {}).items():
            o = out.setdefault(r, _empty_weight_adj_row())
            for k in _WEIGHT_ADJ_FIELDS:
                o[k] += t.get(k, 0)
            o["factor_max"] = max(o["factor_max"], t.get("factor_max", 1.0))
    return {"regions": out}


def format_weight_adjustment(summary: dict, scope: str) -> str:
    """Render :func:`summarize_weight_adjustment` as a region table with a
    total row."""
    regions = summary.get("regions") or {}
    tot = _empty_weight_adj_row()
    for t in regions.values():
        for k in _WEIGHT_ADJ_FIELDS:
            tot[k] += t[k]
        tot["factor_max"] = max(tot["factor_max"], t["factor_max"])

    def _row(label, t):
        mean_f = (f"{t['factor_sum'] / t['n_scaled']:.3f}"
                  if t["n_scaled"] else "---")
        max_f = f"{t['factor_max']:.3f}" if t["n_scaled"] else "---"
        chg = (f"{100.0 * (t['dose_adj_sv'] / t['dose_ref_sv'] - 1.0):+.1f}%"
               if t["dose_ref_sv"] > 0 else "---")
        return (f"{label:<10}{t['n_eligible']:>10,}{t['n_no_weight']:>11,}"
                f"{t['n_scaled']:>9,}{mean_f:>13}{max_f:>12}"
                f"{t['dose_ref_sv']:>16.4f}{t['dose_adj_sv']:>16.4f}{chg:>9}")

    rule = "-" * 106
    lines = [
        "Adult chest and abdominal CT. Doses are scaled up for patients",
        "lighter than 70 kg; heavier patients and exams without a recorded",
        "weight keep the reference-phantom dose. Factor = adjusted / reference dose.",
        f"Counted: {scope}.",
        "",
        f"{'Region':<10}{'Eligible':>10}{'No weight':>11}{'Scaled':>9}"
        f"{'Mean factor':>13}{'Max factor':>12}{'Dose ref. Sv':>16}"
        f"{'Dose adj. Sv':>16}{'Change':>9}",
        rule,
    ]
    lines.extend(_row(r, regions[r]) for r in regions)
    lines.append(rule)
    lines.append(_row("Total", tot))
    lines.append("Mean factor: mean over scaled exams.")
    return "\n".join(lines)

def plot_filtered_risk_analysis(ax_cases, ax_deaths, sim_curve: dict,
                                bayes_curve: Optional[dict] = None):
    """Plot at most two curves: Simulation (always) + Bayesian (if provided)."""
    ax_cases.clear()
    ax_deaths.clear()

    centers = sim_curve.get("centers", np.array([]))
    if len(centers) == 0 or not np.any(sim_curve.get("dose_sv", np.array([])) > 0):
        ax_cases.set_title("Cancer risk / Sv: No filtered data")
        ax_deaths.set_title("Death risk / Sv: No filtered data")
        return

    # A cross-year batch average keeps a true 95 % CI: the pooled
    # weighted-Poisson SE narrows with the number of contributing years
    # (see compute_batch_averaged_risk_curve / _pool_year_bins).
    _n_yrs = sim_curve.get("_averaged_across_years")
    _band_label = "95% CI"
    _title_suffix = f" (avg across {_n_yrs} yrs)" if _n_yrs else ""

    ax_cases.plot(centers, sim_curve["risk_per_sv"] * 100.0, color=_SEMANTIC["induced"],
                  linewidth=2, marker="o", markersize=4, label="Simulation")
    ax_cases.fill_between(centers, sim_curve["risk_low"] * 100.0,
                          sim_curve["risk_high"] * 100.0,
                          color=_SEMANTIC["induced"], alpha=0.18,
                          label=f"Simulation {_band_label}")
    ax_cases.set_xlabel("Age at Exposure (years)")
    ax_cases.set_ylabel("Cancer risk / Sv (%)")
    ax_cases.set_title(f"Cancer risk per same-age dose bin{_title_suffix}")
    # ICRP 152 nominal cancer incidence: Table 3.7 sex/population-balanced
    # total = 1715 / 10,000 / Sv = 17.15 %/Sv (solid + leukaemia + heritable).
    ax_cases.axhline(17.150, color=_SEMANTIC["reference"], linestyle="--", linewidth=2,
                    alpha=0.7, label="ICRP 152 incidence (17.15 %/Sv)")

    ax_deaths.plot(centers, sim_curve["death_per_sv"] * 100.0, color=_OKABE_ITO["blue"],
                   linewidth=2, marker="s", markersize=4, label="Death risk (Simulation)")
    ax_deaths.fill_between(centers, sim_curve["death_low"] * 100.0,
                           sim_curve["death_high"] * 100.0,
                           color=_OKABE_ITO["blue"], alpha=0.18,
                           label=f"Death risk {_band_label}")
    # Detriment per Sv per age bin (Simulation): realised incidence events
    # weighted by each site's ICRP 152 detriment factor / dose — the direct
    # counterpart to the ICRP 152 detriment reference line (5.74 %/Sv).
    if "detriment_per_sv" in sim_curve:
        ax_deaths.plot(centers, sim_curve["detriment_per_sv"] * 100.0, color=_OKABE_ITO["reddish_purple"],
                       linewidth=2, marker="D", markersize=4, label="Detriment (Simulation)")
        ax_deaths.fill_between(centers, sim_curve.get("detriment_low", sim_curve["detriment_per_sv"]) * 100.0,
                               sim_curve.get("detriment_high", sim_curve["detriment_per_sv"]) * 100.0,
                               color=_OKABE_ITO["reddish_purple"], alpha=0.15,
                               label=f"Detriment {_band_label}")
    ax_deaths.set_xlabel("Age at Exposure (years)")
    ax_deaths.set_ylabel("Death risk / Detriment per Sv (%)")
    ax_deaths.set_title(f"Cancer death risk & detriment per same-age dose bin{_title_suffix}")
    # Two ICRP 152 reference lines on the death plot:
    #   * Detriment (Table 3.9 total = 574 / 10,000 / Sv = 5.74 %/Sv) –
    #     the dose-limit quantity that weights mortality, morbidity,
    #     lethality and life-years lost; this is what the simulation's
    #     "detriment" output is compared against.
    #   * Nominal mortality (≈ Σ_sites R_site × k_lethality_site over
    #     Table 3.9 R column = 398 / 10,000 / Sv ≈ 3.98 %/Sv) – the older
    #     "fatal cancer risk coefficient" (~4 %/Sv) often quoted from
    #     ICRP 103 / BEIR VII; this is the direct counterpart to the
    #     simulated death-per-Sv curve since each developed cancer in
    #     `risk.py` becomes a death with probability k_lethality.
    ax_deaths.axhline(5.740, color=_SEMANTIC["reference"], linestyle="--", linewidth=2,
                     alpha=0.7, label="ICRP 152 detriment (5.74 %/Sv)")
    ax_deaths.axhline(3.980, color=_SEMANTIC["reference"], linestyle=":", linewidth=2,
                     alpha=0.7, label="ICRP 152 mortality (3.98 %/Sv)")

    if bayes_curve is not None and np.any(bayes_curve.get("dose_sv", np.array([])) > 0):
        ax_cases.plot(centers, bayes_curve["risk_per_sv"] * 100.0, color=_SEMANTIC["highlight"],
                      linewidth=2, marker="^", markersize=4, label="Bayesian")
        ax_cases.fill_between(centers, bayes_curve["risk_low"] * 100.0,
                              bayes_curve["risk_high"] * 100.0,
                              color=_SEMANTIC["highlight"], alpha=0.15, label="Bayesian 95% CI")
        ax_deaths.plot(centers, bayes_curve["death_per_sv"] * 100.0, color=_OKABE_ITO["sky_blue"],
                       linewidth=2, marker="^", markersize=4, label="Death risk (Bayesian)")
        ax_deaths.fill_between(centers, bayes_curve["death_low"] * 100.0,
                               bayes_curve["death_high"] * 100.0,
                               color=_OKABE_ITO["sky_blue"], alpha=0.15, label="Death risk 95% CI (Bayesian)")
        if "detriment_per_sv" in bayes_curve:
            ax_deaths.plot(centers, bayes_curve["detriment_per_sv"] * 100.0, color=_OKABE_ITO["bluish_green"],
                           linewidth=2, marker="v", markersize=4, label="Detriment (Bayesian)")

    ax_cases.legend(fontsize=7, loc="upper right")
    ax_deaths.legend(fontsize=7, loc="upper right")


def format_filtered_risk_summary(sim_curve: dict, bayes_curve: Optional[dict],
                                 filters: dict, adjustments: dict,
                                 heritable_risk: float = 0.0,
                                 heritable_detriment: float = 0.0,
                                 total_dose_sv: Optional[float] = None,
                                 mataac_strata: Optional[dict] = None,
                                 strata_scope: str = "") -> str:
    """Format text summary. sim_curve = simulation baseline. bayes_curve = combined
    Bayesian curve (None if no adjustments active).

    adjustments keys: ddref_solid, ddref_leuk, ddref_changed, golikov,
    model_compare, selected_model, comparison_model, mataac, mataac_dose_groups.

    When ``total_dose_sv`` is supplied it is used as the headline simulation
    dose denominator instead of the recomputed curve dose. This lets the
    Risk Analysis tab match the Summary tab exactly when no filters are
    applied (the simulation accumulates dose in float32 while the curve is
    recomputed in float64, which can give a ~0.1% difference).
    """
    sel_model = adjustments.get("selected_model", "REIC")
    # Actual feature state of this run, independent of any comparison checkboxes.
    sim_use_mataac = bool(adjustments.get("sim_use_mataac", True))
    sim_use_golikov = bool(adjustments.get("sim_use_golikov", True))

    _curve_dose = float(np.sum(sim_curve.get("dose_sv", np.array([]))) or 0.0)
    sim_dose = total_dose_sv if (total_dose_sv is not None and total_dose_sv > 0) else _curve_dose
    sim_cases = float(np.sum(sim_curve.get("cases", np.array([]))) or 0.0)
    sim_deaths = float(np.sum(sim_curve.get("deaths", np.array([]))) or 0.0)
    sim_detriment = float(np.sum(sim_curve.get("detriment", np.array([]))) or 0.0)
    sim_risk = ((sim_cases + heritable_risk) / sim_dose * 100.0) if sim_dose > 0 else 0.0
    sim_death = (sim_deaths / sim_dose * 100.0) if sim_dose > 0 else 0.0
    sim_det = ((sim_detriment + heritable_detriment) / sim_dose * 100.0) if sim_dose > 0 else 0.0

    # A cross-year batch average (see compute_batch_averaged_risk_curve):
    # every "Simulation" quantity above is a per-simulated-year figure
    # (equal-weighted mean across year groups), not the archive total.
    _n_yrs = sim_curve.get("_averaged_across_years")
    _exams_label = "Filtered examinations"
    if _n_yrs:
        _exams_label += f" (summed over {_n_yrs} year group(s))"

    _fb_note = ("(Per-patient YLL requested but the per-year digest "
                "lacks it -- detriment uses the ICRP 152 site-average YLL "
                "for all years.)")
    _sim_fb = bool(sim_curve.get("detriment_yll_fallback"))

    lines = [
        "=" * 60,
        "FILTERED RISK ANALYSIS",
        "=" * 60,
        f"Sex: {filters.get('sex', 'All')} | Modality: {filters.get('modality', 'All')} | Region: {filters.get('region', 'All')}",
        f"Age range: {filters.get('min_age', 0)} - {filters.get('max_age', 100)} years",
        f"Selected model: {sel_model}",
        f"{_exams_label}: {int(sim_curve.get('n_exams', 0)):,}",
        f"Simulation settings: weight adjustment {'ACTIVE' if sim_use_golikov else 'INACTIVE'} | "
        f"competing-mortality adjustment {'ACTIVE' if sim_use_mataac else 'INACTIVE'}",
        "",
        f"Simulation cancer risk / Sv: {sim_risk:.4f}% (incl. heritable)",
        f"Simulation death risk / Sv:  {sim_death:.4f}%",
        f"Simulation detriment / Sv:   {sim_det:.4f}%",
    ]
    if _sim_fb:
        lines.append(_fb_note)

    has_any = (adjustments.get("ddref_changed") or adjustments.get("golikov")
               or adjustments.get("model_compare") or adjustments.get("mataac"))

    if has_any and bayes_curve is not None:
        b_dose = float(np.sum(bayes_curve.get("dose_sv", np.array([]))) or 0.0)
        b_cases = float(np.sum(bayes_curve.get("cases", np.array([]))) or 0.0)
        b_deaths = float(np.sum(bayes_curve.get("deaths", np.array([]))) or 0.0)
        b_detriment = float(np.sum(bayes_curve.get("detriment", np.array([]))) or 0.0)
        # Add heritable risk/detriment to the Bayesian headline too, using the
        # same whole-cohort heritable totals.  Gonadal dose does not depend on
        # DDREF or cancer model, so the heritable total is the same as in the
        # simulation; only the dose-per-Sv denominator differs when the Bayesian
        # comparison uses a different effective dose (e.g., no-Golikov).
        b_risk = ((b_cases + heritable_risk) / b_dose * 100.0) if b_dose > 0 else 0.0
        b_death = (b_deaths / b_dose * 100.0) if b_dose > 0 else 0.0
        b_det = ((b_detriment + heritable_detriment) / b_dose * 100.0) if b_dose > 0 else 0.0
        dr = b_risk - sim_risk
        dd = b_death - sim_death
        ddet = b_det - sim_det
        dr_pct = (dr / sim_risk * 100.0) if sim_risk > 0 else 0.0
        dd_pct = (dd / sim_death * 100.0) if sim_death > 0 else 0.0
        ddet_pct = (ddet / sim_det * 100.0) if sim_det > 0 else 0.0

        lines.extend([
            "",
            f"Bayesian cancer risk / Sv: {b_risk:.4f}% (incl. heritable)",
            f"Bayesian death risk / Sv:  {b_death:.4f}%",
            f"Bayesian detriment / Sv:   {b_det:.4f}%",
        ])
        if not _sim_fb and bayes_curve.get("detriment_yll_fallback"):
            lines.append(_fb_note)
        lines.extend([
            f"Difference vs simulation: cancer {dr:+.4f} pp ({dr_pct:+.1f}%), "
            f"death {dd:+.4f} pp ({dd_pct:+.1f}%), "
            f"detriment {ddet:+.4f} pp ({ddet_pct:+.1f}%)",
        ])

        lines.extend(["", "-" * 60, "ACTIVE ADJUSTMENTS", "-" * 60])

        if adjustments.get("ddref_changed"):
            lines.extend([
                "",
                f"DDREF Solid: {adjustments['ddref_solid']:.2f} (default: 2.00)",
                f"DDREF Leukaemia: {adjustments['ddref_leuk']:.2f} (default: 1.00)",
            ])

        if adjustments.get("golikov"):
            lines.extend([
                "",
                "WEIGHT ADJUSTMENT REMOVED" if sim_use_golikov else "WEIGHT ADJUSTMENT: was already INACTIVE in simulation",
                "Golikov and Druzhinina: CT organ doses come from a single 70 kg",
                "adult reference phantom. The weight adjustment scales them up for",
                "adult chest and abdominal CT in patients lighter than 70 kg;",
                "heavier patients keep the reference-phantom doses.",
                f"The simulation ran {'WITH' if sim_use_golikov else 'WITHOUT'} weight adjustment.",
                ("Bayesian curve uses reference-phantom doses (no weight adjustment)."
                 if sim_use_golikov else
                 "Bayesian curve matches the simulation here (adjustment already off) — no change."),
            ])

        if adjustments.get("model_compare"):
            cmp_model = adjustments.get("comparison_model", "LAR")
            lines.extend([
                "",
                f"RISK MODEL SWITCHED TO {cmp_model}",
                "ICRP Publication 152 defines two risk projection models:",
                "",
                "REIC (Risk of Exposure-Induced Cancer):",
                "  Probability that an observed cancer was caused by the radiation",
                "  exposure. Accounts for baseline cancer incidence — a cancer is",
                "  only 'attributed' to radiation proportionally to the excess risk",
                "  relative to the total (baseline + excess) risk.",
                "",
                "LAR (Lifetime Attributable Risk):",
                "  Excess probability of developing (or dying from) cancer over a",
                "  lifetime due to radiation. Integrates excess absolute risk over",
                "  remaining life years, adjusted for competing mortality.",
                "",
                "Key difference: REIC is generally lower than LAR because it",
                "attributes only the fraction of cancers above baseline, while LAR",
                "counts all excess cancers regardless of baseline incidence.",
            ])

        if adjustments.get("mataac"):
            lines.extend([
                "",
                ("COMPETING-MORTALITY ADJUSTMENT REMOVED" if sim_use_mataac
                 else "COMPETING-MORTALITY ADJUSTMENT: was already INACTIVE in simulation"),
                "Mataac et al. 2024: CT patients in higher cumulative effective",
                "dose (CED) groups show lower 10-year survival, likely due to",
                "confounding by indication — sicker patients undergo more scans.",
                f"The simulation ran {'WITH' if sim_use_mataac else 'WITHOUT'} the competing-mortality adjustment (reduced lifespan).",
                ("Bayesian curve removes the adjustment (standard lifespan → higher risk)."
                 if sim_use_mataac else
                 "Bayesian curve matches the simulation here (adjustment already off) — no change."),
            ])
            # The strata table below shows the same groups; this one is
            # only the fallback when it is unavailable.
            mataac_dg = (adjustments.get("mataac_dose_groups") or []
                         if mataac_strata is None else [])
            if mataac_dg:
                lines.extend([
                    "",
                    f"{'Dose group':<18} {'Exams':>7} {'Dose Sv':>10} "
                    f"{'Competing-mortality adjustment factor':>38}",
                    "-" * 76,
                ])
                for dg in mataac_dg:
                    lines.append(
                        f"{dg['group']:<18} {dg['n_exams']:>7,} "
                        f"{dg['dose_sv']:>10.4f} {dg['frailty_factor']:>38.3f}"
                    )

            # Removing frailty can only RAISE a curve, so an overlay that lands
            # below the simulation is never the frailty term -- but on screen it
            # reads as a flat contradiction of the sentence above.  Name the
            # actual cause instead of leaving the reader to reconcile the two.
            _sim_r = np.asarray(sim_curve.get("risk_per_sv", []), dtype=float)
            _bay_r = np.asarray(bayes_curve.get("risk_per_sv", []), dtype=float)
            _n_below = (int(np.count_nonzero(_bay_r < _sim_r - 1e-12))
                        if _sim_r.size and _sim_r.size == _bay_r.size else 0)
            if _n_below:
                _others = [name for flag, name in (
                    (adjustments.get("model_compare"),
                     f"the risk model was switched to "
                     f"{adjustments.get('comparison_model', 'LAR')}, which can be lower than "
                     f"{adjustments.get('selected_model', 'REIC')}"),
                    (adjustments.get("ddref_changed"),
                     "DDREF was changed from its defaults"),
                    (adjustments.get("golikov"),
                     "the Golikov body-size correction was removed, changing the dose denominator"),
                ) if flag]
                lines.extend([
                    "",
                    f"NOTE: the Bayesian curve sits BELOW the simulation in {_n_below} age bin(s).",
                    "Removing frailty on its own can only raise the curve, so that is not",
                    "what produced it. This single curve combines EVERY active adjustment:",
                ])
                if _others:
                    lines.extend(f"  - {o}." for o in _others)
                elif not adjustments.get("bayes_scope_matches_sim", True):
                    lines.append("  - no other adjustment is active, so the two curves are")
                    lines.append("    describing different cohorts: the simulation curve is averaged")
                    lines.append("    over all year groups while this overlay could not be and is")
                    lines.append("    the detail year alone (see the note above).")
                else:
                    lines.append("  - no other adjustment is active and both curves cover the same")
                    lines.append("    cohort, so this should not be possible. Please report it.")

    if mataac_strata is not None:
        lines.append("")
        lines.append(format_mataac_strata_summary(mataac_strata, strata_scope))

    lines.append("=" * 60)
    return "\n".join(lines)

def _year_batches_of(stats: dict) -> "dict | None":
    """Return ``stats['_year_batches']`` iff this is a per_year_average batch.

    Lets callers use each year's own counts instead of the pre-averaged
    top-level ones.
    """
    if stats.get("_run_mode") != "per_year_average":
        return None
    return stats.get("_year_batches") or None


def batch_completion_totals(stats: dict, exam_data) -> tuple:
    """``(total_examinations, total_iterations)`` for the GUI's "Processed
    ... examinations across ... years" completion message.

    For an "average across years" batch, ``exam_data`` passed in here is
    only the *last processed year* (see year_batch.py's memory note) and
    ``stats['iterations']`` is the per-year MEAN MC-iteration count (see
    ``_scale_rates_to_mean_iterations``) -- neither is the batch's real
    total, so quoting them directly silently understates both numbers.
    Sums each year's own count from ``stats['_year_batches']`` instead,
    the same convention :func:`format_summary` uses for the Summary tab's
    "Dataset size, TOTAL across all N year group(s)" line.

    Any other run ("combined" -- which pools every year group's files into
    one simulation, so ``exam_data``/``stats['iterations']`` already hold
    the true combined totals -- or no batching at all) returns
    the values unchanged.
    """
    year_batches = (
        stats.get("_year_batches")
        if stats.get("_run_mode") == "per_year_average" else None
    )
    if not year_batches:
        return len(exam_data or []), int(stats.get("iterations", 0) or 0)
    total_examinations = 0
    total_iterations = 0
    for y in year_batches.values():
        if "_unique_examinations" in y:
            total_examinations += int(y.get("_unique_examinations", 0) or 0)
        else:
            _yit = float(y.get("iterations", 0) or 0)
            if _yit > 0:
                total_examinations += int(float(y.get("acquisitions", 0) or 0) / _yit)
        total_iterations += int(y.get("iterations", 0) or 0)
    return total_examinations, total_iterations


def _run_year_count(stats: dict) -> int:
    """Number of year groups this run spans, for a Total/Average display.

    ``per_year_average`` genuinely ran one simulation per year group, so
    the denominator is how many of those actually produced results (see
    ``_year_batches``).  ``combined`` instead pools every year group's
    files into ONE simulation -- there is no per-year run to look at, but
    it still spans ``stats['_n_years']`` year groups (set by
    year_batch.run_yearly_simulation), and "per year" is still a
    meaningful average to show alongside the already-correct combined
    total.  Any other run (a single year group, or no batching at all)
    returns 1, meaning "no Total/Average distinction to draw".
    """
    mode = stats.get("_run_mode")
    if mode == "per_year_average":
        return len(stats.get("_year_batches") or {}) or 1
    if mode == "combined":
        return int(stats.get("_n_years", 1) or 1)
    return 1


# Sankey Diagram

# MC Diagnostics (convergence trajectory, per-patient dose CDF, dose-vs-age)




def plot_sankey_diagram(fig, stats: dict, visible_cancers: set = None):
    """Draw a multi-level Sankey diagram on the given matplotlib Figure.

    Levels (left to right):
      1. Total Examinations
      2. Examinations per modality
      3. No Cancer / Cancer (per modality, aggregated)
      4. Cancer types
      5. Died / Survived per cancer type
      6. Total Died / Total Survived

    Draw is short-circuited when the caller hasn't changed any input since
    the previous invocation: the last-used figure remains displayed and we
    skip the ~40-60-patch rebuild.  The cache key is stored as a private
    attribute on ``fig`` so independent figures (e.g. one per tab) do not
    interfere with each other.
    """
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MplPath

    # Immutable cache key: skip re-rendering when the inputs haven't changed.
    if isinstance(stats, dict):
        _ms = stats.get("modality_stats") or {}
        _bd = stats.get("cancer_breakdown") or {}
        _vc = frozenset(visible_cancers) if visible_cancers else None
        _cache_key = (
            int(stats.get("iterations", 0) or 0),
            int(stats.get("cancer_cases", 0) or 0),
            int(stats.get("cancer_deaths", 0) or 0),
            tuple(sorted(
                (str(m),
                 float(v.get("scans", 0) or 0),
                 float(v.get("cases", 0) or 0),
                 float(v.get("deaths", 0) or 0))
                for m, v in _ms.items()
            )),
            tuple(sorted(
                (str(c),
                 float(v.get("developed", 0) or 0),
                 float(v.get("deaths", 0) or 0))
                for c, v in _bd.items()
            )),
            _vc,
        )
        if getattr(fig, "_sankey_cache_key", None) == _cache_key:
            return  # identical to last draw — keep the cached pixels
        fig._sankey_cache_key = _cache_key
    else:
        # ``stats`` is not a dict (e.g. early during GUI init) — force a
        # redraw and drop any stale cache so the next valid call re-renders.
        fig._sankey_cache_key = None

    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Per-box hit regions for the GUI hover tooltip; cleared on every redraw.
    box_hits: List[Tuple[float, float, float, float, str]] = []
    fig._sankey_box_hits = box_hits
    fig._sankey_ax = ax
    fig._sankey_tooltip = None  # GUI re-creates on first hover post-redraw

    if not isinstance(stats, dict):
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", fontsize=14)
        return

    iterations = max(1, int(stats.get("iterations", 1) or 1))
    mod_stats = stats.get("modality_stats", {})
    cancer_bd = stats.get("cancer_breakdown", {})

    if not mod_stats:
        ax.text(0.5, 0.5, "No modality data yet", ha="center", va="center", fontsize=14)
        return

    # Filter cancer types by visibility
    if visible_cancers is not None:
        cancer_bd = {ct: cv for ct, cv in cancer_bd.items() if ct in visible_cancers}

    # Compute per-iteration values
    total_exams = sum(ms.get("scans", 0) for ms in mod_stats.values()) / iterations
    total_cases = sum(cv.get("developed", 0) for cv in cancer_bd.values()) / iterations
    total_deaths = sum(cv.get("deaths", 0) for cv in cancer_bd.values()) / iterations
    total_survived_cancer = total_cases - total_deaths
    total_no_cancer = total_exams - total_cases

    # Per-modality data
    mod_data = {}
    for mod, ms in sorted(mod_stats.items(), key=lambda x: x[1].get("scans", 0), reverse=True):
        scans = ms.get("scans", 0) / iterations
        cases = ms.get("cases", 0) / iterations
        deaths_mod = ms.get("deaths", 0) / iterations
        mod_data[mod] = {"scans": scans, "cases": cases, "deaths": deaths_mod,
                         "no_cancer": scans - cases}

    # Per cancer-type data
    ct_data = {}
    for ct, cv in sorted(cancer_bd.items(), key=lambda x: x[1].get("developed", 0), reverse=True):
        dev = cv.get("developed", 0) / iterations
        dth = cv.get("deaths", 0) / iterations
        if dev > 0:
            ct_data[ct] = {"developed": dev, "deaths": dth, "survived": dev - dth}

    if total_exams <= 0:
        ax.text(0.5, 0.5, "No examinations", ha="center", va="center", fontsize=14)
        return

    # Layout constants
    # X positions for each column (6 columns)
    x_cols = [0.02, 0.18, 0.38, 0.56, 0.74, 0.92]
    box_w = 0.08
    margin_top = 0.94
    margin_bot = 0.06
    usable_h = margin_top - margin_bot
    min_box_h = 0.012
    gap = 0.008

    # Modality and cancer colour pools use the Okabe-Ito cycle so every
    # category in the Sankey diagram stays colour-blind-safe.  Modality
    # and cancer palettes start at different offsets so a modality box
    # directly above a cancer box is never the same colour.
    colors_mod = list(_PALETTE_LIST)
    colors_cancer = list(_PALETTE_LIST[1:]) + list(_PALETTE_LIST[:1])
    color_no_cancer = _SEMANTIC["no_cancer"]
    color_survived  = _SEMANTIC["survived"]
    color_died      = _SEMANTIC["died"]

    def _draw_box(x, y, w, h, color, label, fontsize=7, alpha=0.85):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, max(h, 0.001), boxstyle="round,pad=0.003",
            facecolor=color, edgecolor="white", linewidth=0.5, alpha=alpha,
        )
        ax.add_patch(rect)
        # Record the box region for hover-tooltip lookup (full label even when text is shrunk).
        box_hits.append((float(x), float(y), float(w), float(max(h, 0.001)), str(label)))
        if h >= min_box_h * 0.8:
            # Collapse newline to space when box is too short for two lines
            two_line_threshold = min_box_h * 3.5
            if h < two_line_threshold and "\n" in label:
                display_label = label.replace("\n", " ")
            else:
                display_label = label
            # Scale font down for small boxes
            if h < min_box_h * 2.0:
                effective_fs = max(4.0, fontsize - 1.5)
            elif h < min_box_h * 3.5:
                effective_fs = max(4.5, fontsize - 0.5)
            else:
                effective_fs = fontsize
            ax.text(x + w / 2, y + h / 2, display_label, ha="center", va="center",
                    fontsize=effective_fs, color="white", fontweight="bold",
                    clip_on=True)

    def _draw_flow(x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.18):
        """Draw a curved flow band between two vertical segments."""
        verts = [
            (x0, y0_bot),
            (x0 + (x1 - x0) * 0.4, y0_bot),
            (x1 - (x1 - x0) * 0.4, y1_bot),
            (x1, y1_bot),
            (x1, y1_top),
            (x1 - (x1 - x0) * 0.4, y1_top),
            (x0 + (x1 - x0) * 0.4, y0_top),
            (x0, y0_top),
            (x0, y0_bot),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        path = MplPath(verts, codes)
        # Rasterize dense flow bands in exported figures to shrink file size.
        patch = mpatches.PathPatch(path, facecolor=color, edgecolor="none",
                                    alpha=alpha, rasterized=True)
        ax.add_patch(patch)

    def _allocate_boxes(items_values, total_value):
        """Compute (y_bottom, height) for each item proportional to value."""
        if total_value <= 0 or not items_values:
            return []
        n = len(items_values)
        total_gap = gap * max(0, n - 1)
        available = usable_h - total_gap
        boxes = []
        y = margin_top
        for val in items_values:
            frac = val / total_value if total_value > 0 else 0
            h = max(min_box_h, frac * available)
            y -= h
            boxes.append((y, h))
            y -= gap
        # Rescale if overflow
        total_used = sum(h for _, h in boxes) + total_gap
        if total_used > usable_h:
            scale = (usable_h - total_gap) / sum(h for _, h in boxes)
            y = margin_top
            new_boxes = []
            for _, h in boxes:
                nh = max(min_box_h * 0.5, h * scale)
                y -= nh
                new_boxes.append((y, nh))
                y -= gap
            boxes = new_boxes
        return boxes

    # Column 1: Total Examinations
    col1_h = usable_h
    col1_y = margin_bot
    _draw_box(x_cols[0], col1_y, box_w, col1_h, "#34495E",
              f"All Exams\n{total_exams:,.0f}", fontsize=8)

    # Column 2: Per Modality
    mod_names = list(mod_data.keys())
    mod_values = [mod_data[m]["scans"] for m in mod_names]
    col2_boxes = _allocate_boxes(mod_values, total_exams)

    # Flow from Total → Modalities (top-down so flows match col2 top-down order)
    src_top = col1_y + col1_h
    for i, (mname, (by, bh)) in enumerate(zip(mod_names, col2_boxes)):
        color = colors_mod[i % len(colors_mod)]
        frac = mod_values[i] / total_exams if total_exams > 0 else 0
        src_h = frac * col1_h
        src_bot = src_top - src_h
        _draw_flow(x_cols[0] + box_w, src_bot, src_top,
                   x_cols[1], by, by + bh, color, alpha=0.15)
        disp_name = modality_name(mname) if len(mname) <= 6 else mname
        _draw_box(x_cols[1], by, box_w, bh, color,
                  f"{disp_name}\n{mod_values[i]:,.0f}", fontsize=6)
        src_top = src_bot

    # Column 3: Cancer / No Cancer
    # Two boxes: No Cancer (top) and Cancer (bottom)
    col3_items = [total_no_cancer, total_cases]
    col3_boxes = _allocate_boxes(col3_items, total_exams)

    if len(col3_boxes) >= 2:
        nc_y, nc_h = col3_boxes[0]
        ca_y, ca_h = col3_boxes[1]

        _draw_box(x_cols[2], nc_y, box_w, nc_h, color_no_cancer,
                  f"No Cancer\n{total_no_cancer:,.1f}", fontsize=7)
        _draw_box(x_cols[2], ca_y, box_w, ca_h, _SEMANTIC["cancer"],
                  f"Cancer\n{total_cases:,.4f}", fontsize=7)

        # Flows from each modality to No Cancer / Cancer, top-down to avoid crossings.
        nc_dst_top = nc_y + nc_h  # fill No Cancer box top-down
        ca_dst_top = ca_y + ca_h  # fill Cancer box top-down
        for i, (mname, (my, mh)) in enumerate(zip(mod_names, col2_boxes)):
            color = colors_mod[i % len(colors_mod)]
            scans = mod_data[mname]["scans"]
            cases = mod_data[mname]["cases"]
            no_ca = mod_data[mname]["no_cancer"]

            if scans <= 0:
                continue
            # Source: top portion of modality box → No Cancer (top of col3)
            no_ca_ratio = no_ca / scans if scans > 0 else 0
            src_nc_h = mh * no_ca_ratio
            src_nc_top = my + mh          # top of modality box
            src_nc_bot = src_nc_top - src_nc_h
            nc_frac = no_ca / total_no_cancer if total_no_cancer > 0 else 0
            nc_dst_bot = nc_dst_top - nc_h * nc_frac
            _draw_flow(x_cols[1] + box_w, src_nc_bot, src_nc_top,
                       x_cols[2], nc_dst_bot, nc_dst_top, color, alpha=0.10)
            nc_dst_top = nc_dst_bot
            # Source: bottom portion of modality box → Cancer (bottom of col3)
            if cases > 0:
                ca_frac = cases / total_cases if total_cases > 0 else 0
                src_ca_top = src_nc_bot    # just below the no-cancer portion
                src_ca_bot = my            # bottom of modality box
                ca_dst_bot = ca_dst_top - ca_h * ca_frac
                _draw_flow(x_cols[1] + box_w, src_ca_bot, src_ca_top,
                           x_cols[2], ca_dst_bot, ca_dst_top, color, alpha=0.20)
                ca_dst_top = ca_dst_bot

    # Column 4: Cancer Types
    ct_names = list(ct_data.keys())
    ct_values = [ct_data[ct]["developed"] for ct in ct_names]
    col4_boxes = _allocate_boxes(ct_values, total_cases)

    if len(col3_boxes) >= 2:
        ca_y, ca_h = col3_boxes[1]
        # Source goes top-down through the Cancer box to match col4 top-down order
        src_ct_top = ca_y + ca_h
        for j, (ctname, (cty, cth)) in enumerate(zip(ct_names, col4_boxes)):
            color = colors_cancer[j % len(colors_cancer)]
            frac = ct_values[j] / total_cases if total_cases > 0 else 0
            src_h = frac * ca_h
            src_ct_bot = src_ct_top - src_h
            _draw_flow(x_cols[2] + box_w, src_ct_bot, src_ct_top,
                       x_cols[3], cty, cty + cth, color, alpha=0.20)
            pct_ct = ct_values[j] / total_cases * 100 if total_cases > 0 else 0
            short = ctname[:14] if len(ctname) > 14 else ctname
            _draw_box(x_cols[3], cty, box_w, cth, color,
                      f"{short}\n{ct_values[j]:.3f}  {pct_ct:.1f}%", fontsize=6.5)
            src_ct_top = src_ct_bot

    # Column 5: Died / Survived per cancer type
    # Build list: for each cancer type, two sub-boxes (survived, died)
    col5_items = []
    col5_labels = []
    col5_colors = []
    col5_ct_source = []  # index into ct_names
    for j, ctname in enumerate(ct_names):
        surv = ct_data[ctname]["survived"]
        died = ct_data[ctname]["deaths"]
        # Emit Died first so it lands at top of each cancer-type group in col5,
        # matching the top-down source flow from col4 (avoids crossing within pair)
        if died > 0:
            col5_items.append(died)
            col5_labels.append("Died")
            col5_colors.append(color_died)
            col5_ct_source.append(j)
        if surv > 0:
            col5_items.append(surv)
            col5_labels.append("Surv")
            col5_colors.append(color_survived)
            col5_ct_source.append(j)

    col5_boxes = _allocate_boxes(col5_items, total_cases)

    # Flows from each cancer type to Survived / Died, top-down to avoid crossings.
    ct_src_top = {}  # track top-down position in each col4 cancer box
    for k, (by5, bh5) in enumerate(col5_boxes):
        j = col5_ct_source[k]
        if j < len(col4_boxes):
            cty, cth = col4_boxes[j]
            ctname = ct_names[j]
            dev = ct_data[ctname]["developed"]
            val = col5_items[k]
            if j not in ct_src_top:
                ct_src_top[j] = cty + cth  # start from top of box
            src_top = ct_src_top[j]
            frac_h = (val / dev * cth) if dev > 0 else 0
            src_bot = src_top - frac_h
            _draw_flow(x_cols[3] + box_w, src_bot, src_top,
                       x_cols[4], by5, by5 + bh5,
                       col5_colors[k], alpha=0.18)
            ct_src_top[j] = src_bot
        _draw_box(x_cols[4], by5, box_w * 0.85, bh5, col5_colors[k],
                  f"{col5_labels[k]}\n{col5_items[k]:.4f}", fontsize=5)

    # Column 6: Total Died / Survived / No Cancer
    # Top-to-bottom order matches the source flows so bands don't cross.
    col6_items = []
    col6_colors_list = []
    col6_labels_list = []
    if total_deaths > 0:
        col6_items.append(total_deaths)
        col6_colors_list.append(color_died)
        col6_labels_list.append(f"Died\n{total_deaths:.4f}")
    if total_survived_cancer > 0:
        col6_items.append(total_survived_cancer)
        col6_colors_list.append(color_survived)
        col6_labels_list.append(f"Survived\n{total_survived_cancer:.4f}")
    if total_no_cancer > 0:
        col6_items.append(total_no_cancer)
        col6_colors_list.append(color_no_cancer)
        col6_labels_list.append(f"No Cancer\n{total_no_cancer:,.1f}")

    # Scale col6 proportional to total_exams so No Cancer box has correct size
    col6_boxes = _allocate_boxes(col6_items, total_exams)

    # Track accumulated positions in each col6 box (top-down fill).
    surv_top = None
    died_top = None
    surv_box = None
    died_box = None
    nc6_box = None
    for idx6, (by6, bh6) in enumerate(col6_boxes):
        _draw_box(x_cols[5], by6, box_w, bh6, col6_colors_list[idx6],
                  col6_labels_list[idx6], fontsize=7)
        if col6_colors_list[idx6] == color_survived:
            surv_box = (by6, bh6)
            surv_top = by6 + bh6
        elif col6_colors_list[idx6] == color_died:
            died_box = (by6, bh6)
            died_top = by6 + bh6
        else:
            nc6_box = (by6, bh6)

    # Flows from col5 → col6 (top-down, no crossings)
    for k, (by5, bh5) in enumerate(col5_boxes):
        val = col5_items[k]
        if col5_colors[k] == color_survived and surv_box is not None:
            _, sh = surv_box
            frac_h = (val / total_survived_cancer * sh) if total_survived_cancer > 0 else 0
            dst_top = surv_top
            dst_bot = surv_top - frac_h
            _draw_flow(x_cols[4] + box_w * 0.85, by5, by5 + bh5,
                       x_cols[5], dst_bot, dst_top,
                       color_survived, alpha=0.15)
            surv_top = dst_bot
        elif col5_colors[k] == color_died and died_box is not None:
            _, dh = died_box
            frac_h = (val / total_deaths * dh) if total_deaths > 0 else 0
            dst_top = died_top
            dst_bot = died_top - frac_h
            _draw_flow(x_cols[4] + box_w * 0.85, by5, by5 + bh5,
                       x_cols[5], dst_bot, dst_top,
                       color_died, alpha=0.15)
            died_top = dst_bot

    # Flow from col3 No Cancer box → col6 No Cancer box
    if nc6_box is not None and len(col3_boxes) >= 1:
        nc3_y, nc3_h = col3_boxes[0]
        nc6_y, nc6_h = nc6_box
        _draw_flow(x_cols[2] + box_w, nc3_y, nc3_y + nc3_h,
                   x_cols[5], nc6_y, nc6_y + nc6_h,
                   color_no_cancer, alpha=0.12)

    # Column headers
    headers = ["Total\nExaminations", "Per\nModality", "Cancer\nOutcome",
               "Cancer\nType", "Per Type\nOutcome", "Overall\nOutcome"]
    for i, hdr in enumerate(headers):
        ax.text(x_cols[i] + box_w / 2, 0.98, hdr, ha="center", va="top",
                fontsize=7, fontweight="bold", color=_SEMANTIC["neutral"])

    ax.set_title("Radiation-Induced Cancer Risk Flow (per year, Sankey Diagram)",
                 fontsize=10, fontweight="bold", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


_STRAT_SEX_LABELS = {"M": "Male", "F": "Female", "U": "Unspecified"}
_STRAT_BANDS = [(0, 15, "0-14 (paed)"), (15, 40, "15-39"),
                (40, 65, "40-64"), (65, 150, "65+")]

# Sentinel default for ``format_stratified_breakdown(header=...)``: means
# "render the current box header (using ``detail_label``)".  Distinct from
# ``None`` which means "no box header at all" (embedded-section mode).
_BOX_HEADER_DEFAULT = object()


def _stratified_accumulate(
    exam_columns,
    event_columns,
    cancer_events,
    visible_cancers,
) -> dict:
    """Single fused pass over exam_data + events building every accumulator
    the stratified-breakdown sections need.

    Returns a dict with: ``n_exams``, ``sex_scan_counts``, ``sex_dose_msv``,
    ``sex_cases``, ``sex_deaths``, ``age_scan_counts``, ``age_dose_msv``,
    ``age_cases``, ``age_deaths``, ``proc_dose_msv``, ``proc_exam_counts``,
    ``proc_cases``, ``proc_deaths``.
    """
    bands = _STRAT_BANDS
    n_bands = len(bands)
    n_exams = len(exam_columns) if exam_columns is not None else 0

    sex_scan_counts: Counter = Counter()
    sex_dose_msv: Counter = Counter()
    age_scan_counts = [0] * n_bands
    age_dose_msv = [0.0] * n_bands
    proc_dose_msv: Counter = Counter()
    proc_exam_counts: Counter = Counter()

    if exam_columns is not None:
        ed_msv_arr = exam_columns.ed_msv
        sex_arr = exam_columns.sex_code
        age_arr = exam_columns.age
        protocol_arr = exam_columns.protocol
        modality_arr = exam_columns.modality
        ed_msv_np = np.asarray(ed_msv_arr, dtype=np.float64)

        sex_chars = np.asarray(sex_arr, dtype="U1")
        sex_code_idx = np.where(sex_chars == "M", 0, np.where(sex_chars == "F", 1, 2))
        sex_counts_arr = np.bincount(sex_code_idx, minlength=3)
        sex_dose_arr = np.bincount(sex_code_idx, weights=ed_msv_np, minlength=3)
        for _code, _ci in (("M", 0), ("F", 1), ("U", 2)):
            sex_scan_counts[_code] += int(sex_counts_arr[_ci])
            sex_dose_msv[_code] += float(sex_dose_arr[_ci])

        age_np = np.asarray(age_arr, dtype=np.float64)
        band_edges = np.array([b[0] for b in bands] + [bands[-1][1]], dtype=np.float64)
        band_idx = np.digitize(age_np, band_edges[1:-1], right=False)
        in_range = (age_np >= band_edges[0]) & (age_np < band_edges[-1])
        band_idx = np.where(in_range, band_idx, n_bands)
        age_counts_arr = np.bincount(band_idx, minlength=n_bands + 1)[:n_bands]
        age_dose_arr = np.bincount(band_idx, weights=ed_msv_np, minlength=n_bands + 1)[:n_bands]
        for bi in range(n_bands):
            age_scan_counts[bi] += int(age_counts_arr[bi])
            age_dose_msv[bi] += float(age_dose_arr[bi])

        for i in range(len(exam_columns)):
            ed_msv = float(ed_msv_arr[i] or 0.0)
            proto = str(protocol_arr[i] or "").strip() or str(modality_arr[i] or "Unknown")
            proc_dose_msv[proto] += ed_msv
            proc_exam_counts[proto] += 1

    sex_cases: Counter = Counter()
    sex_deaths: Counter = Counter()
    age_cases = [0] * n_bands
    age_deaths = [0] * n_bands
    proc_cases: Counter = Counter()
    proc_deaths: Counter = Counter()

    _ev_cols = event_columns if event_columns is not None else cancer_events
    if _ev_cols is not None and hasattr(_ev_cols, "cancer_type"):
        if visible_cancers is not None:
            mask = np.array([ct in visible_cancers for ct in _ev_cols.cancer_type], dtype=bool)
        else:
            mask = np.ones(len(_ev_cols), dtype=bool)
        sex_arr = _ev_cols.sex[mask]
        died_arr = _ev_cols.died[mask]
        age_arr = _ev_cols.age_at_exposure[mask]
        proto_arr = _ev_cols.protocol[mask]
        for sx in ("M", "F", "U"):
            s_mask = sex_arr == sx
            sex_cases[sx] = int(s_mask.sum())
            sex_deaths[sx] = int(died_arr[s_mask].sum())
        for bi, (lo, hi, _lbl) in enumerate(bands):
            a_mask = (age_arr >= lo) & (age_arr < hi)
            age_cases[bi] = int(a_mask.sum())
            age_deaths[bi] = int(died_arr[a_mask].sum())
        proto_list = proto_arr.tolist()
        died_list = died_arr.tolist()
        for p, d in zip(proto_list, died_list):
            proc_cases[p] += 1
            if d:
                proc_deaths[p] += 1
    else:
        if visible_cancers is None:
            for ev in cancer_events:
                sx = str(ev.get("sex") or "U").upper()[:1]
                if sx not in _STRAT_SEX_LABELS:
                    sx = "U"
                died = bool(ev.get("died"))
                sex_cases[sx] += 1
                if died:
                    sex_deaths[sx] += 1
                a = float(ev.get("age_at_exposure", ev.get("age", 0)) or 0)
                for bi, (lo, hi, _lbl) in enumerate(bands):
                    if lo <= a < hi:
                        age_cases[bi] += 1
                        if died:
                            age_deaths[bi] += 1
                        break
                proto = str(ev.get("protocol") or "").strip() or str(ev.get("modality") or "Unknown")
                proc_cases[proto] += 1
                if died:
                    proc_deaths[proto] += 1
        else:
            for ev in cancer_events:
                if ev.get("cancer_type") not in visible_cancers:
                    continue
                sx = str(ev.get("sex") or "U").upper()[:1]
                if sx not in _STRAT_SEX_LABELS:
                    sx = "U"
                died = bool(ev.get("died"))
                sex_cases[sx] += 1
                if died:
                    sex_deaths[sx] += 1
                a = float(ev.get("age_at_exposure", ev.get("age", 0)) or 0)
                for bi, (lo, hi, _lbl) in enumerate(bands):
                    if lo <= a < hi:
                        age_cases[bi] += 1
                        if died:
                            age_deaths[bi] += 1
                        break
                proto = str(ev.get("protocol") or "").strip() or str(ev.get("modality") or "Unknown")
                proc_cases[proto] += 1
                if died:
                    proc_deaths[proto] += 1

    return {
        "n_exams": n_exams,
        "sex_scan_counts": sex_scan_counts,
        "sex_dose_msv": sex_dose_msv,
        "sex_cases": sex_cases,
        "sex_deaths": sex_deaths,
        "age_scan_counts": age_scan_counts,
        "age_dose_msv": age_dose_msv,
        "age_cases": age_cases,
        "age_deaths": age_deaths,
        "proc_dose_msv": proc_dose_msv,
        "proc_exam_counts": proc_exam_counts,
        "proc_cases": proc_cases,
        "proc_deaths": proc_deaths,
    }


def _render_strat_sex(acc, iter_f, indent="  "):
    lines = []
    if acc["sex_scan_counts"]:
        lines.append(f"{indent}-- By sex --")
        lines.append(
            f"{indent}{'Sex':<12}{'Exams':>10}{'Dose (Sv)':>12}"
            f"{'Cases/yr':>12}{'Deaths/yr':>12}{'Cases/10k':>12}"
        )
        lines.append(indent + "-" * 70)
        for sx in ("M", "F", "U"):
            nsc = int(round(acc["sex_scan_counts"].get(sx, 0)))
            if nsc == 0:
                continue
            dsv = acc["sex_dose_msv"].get(sx, 0.0) / 1000.0
            c = acc["sex_cases"].get(sx, 0) / iter_f
            d = acc["sex_deaths"].get(sx, 0) / iter_f
            rate_10k = (c / nsc * 10_000.0) if nsc > 0 else 0.0
            lines.append(
                f"{indent}{_STRAT_SEX_LABELS[sx]:<12}{nsc:>10,d}{dsv:>12.2f}"
                f"{c:>12.1f}{d:>12.1f}{rate_10k:>12.2f}"
            )
    return lines


def _render_strat_age(acc, iter_f, indent="  "):
    """Render the "By age at exposure" summary lines from accumulator
    ``acc``.  ``iter_f`` divides cumulative counts back to per-iteration
    means (counts in ``acc`` are summed over iterations)."""
    lines = []
    total_scans_age = sum(acc["age_scan_counts"])
    total_dose_msv = sum(acc["age_dose_msv"])
    total_cases_age = sum(acc["age_cases"]) / iter_f if iter_f > 0 else 0.0
    if total_scans_age > 0:
        lines.append(f"{indent}-- By age at exposure --")
        lines.append(
            f"{indent}{'Age band':<14}{'Exams':>9}{'Exam %':>9}"
            f"{'Dose %':>9}{'mSv/exam':>10}{'Cases/yr':>11}{'Case %':>9}{'Cases/10k':>12}"
        )
        lines.append(indent + "-" * 82)
        for i, (_lo, _hi, lbl) in enumerate(_STRAT_BANDS):
            nsc = int(round(acc["age_scan_counts"][i]))
            if nsc == 0:
                continue
            dsv = acc["age_dose_msv"][i]
            c = acc["age_cases"][i] / iter_f
            ex_pct = nsc / total_scans_age * 100.0
            dose_pct = (dsv / total_dose_msv * 100.0) if total_dose_msv > 0 else 0.0
            case_pct = (c / total_cases_age * 100.0) if total_cases_age > 0 else 0.0
            rate_10k = (c / nsc * 10_000.0) if nsc > 0 else 0.0
            msv_per = (dsv / nsc) if nsc > 0 else 0.0
            lines.append(
                f"{indent}{lbl:<14}{nsc:>9,d}{ex_pct:>8.1f}%"
                f"{dose_pct:>8.1f}%{msv_per:>10.2f}{c:>11.1f}{case_pct:>8.1f}%"
                f"{rate_10k:>12.2f}"
            )
    return lines


def _render_strat_procedures(acc, iter_f, top_procedures, embedded, indent="  "):
    lines = []
    if not acc["proc_cases"]:
        return lines
    if not embedded:
        lines.append(f"{indent}-- Top {top_procedures} procedures by annual cases --")
    lines.append(
        f"{indent}{'Procedure':<38}{'Dose (Sv/yr)':>12}{'mSv/exam':>10}"
        f"{'Cases/yr':>10}{'Fatal/yr':>10}"
    )
    lines.append(indent + "-" * 80)
    for proto, nc in acc["proc_cases"].most_common(top_procedures):
        dsv = acc["proc_dose_msv"].get(proto, 0.0) / 1000.0
        n_ex = acc["proc_exam_counts"].get(proto, 0)
        msv_per = (acc["proc_dose_msv"].get(proto, 0.0) / n_ex) if n_ex > 0 else 0.0
        c = nc / iter_f
        d = acc["proc_deaths"].get(proto, 0) / iter_f
        short = proto if len(proto) <= 36 else proto[:35] + "\u2026"
        lines.append(
            f"{indent}{short:<38}{dsv:>12.2f}{msv_per:>10.2f}{c:>10.1f}{d:>10.1f}"
        )
    return lines


def stratified_digest(acc: dict, iterations: int) -> dict:
    """A JSON-ish, pid-free snapshot of one year's stratified accumulators.

    Returns plain dicts / lists (no Counters) so the digest can be averaged
    across years without dragging per-patient data along.
    """
    return {
        "iterations": int(iterations or 0),
        "sex_scan_counts": dict(acc["sex_scan_counts"]),
        "sex_dose_msv": dict(acc["sex_dose_msv"]),
        "sex_cases": dict(acc["sex_cases"]),
        "sex_deaths": dict(acc["sex_deaths"]),
        "age_scan_counts": list(acc["age_scan_counts"]),
        "age_dose_msv": list(acc["age_dose_msv"]),
        "age_cases": list(acc["age_cases"]),
        "age_deaths": list(acc["age_deaths"]),
        "proc_dose_msv": dict(acc["proc_dose_msv"]),
        "proc_exam_counts": dict(acc["proc_exam_counts"]),
        "proc_cases": dict(acc["proc_cases"]),
        "proc_deaths": dict(acc["proc_deaths"]),
        "n_exams": int(acc["n_exams"]),
    }


def merge_stratified_digests(digests: list) -> Optional[dict]:
    """Equal-weight average of per-year :func:`stratified_digest` outputs.

    Returns an acc-shaped dict renderable with ``iter_f = 1.0`` (i.e. all
    case/death fields are already per-year means), or ``None`` when the list
    is empty.  A key missing in a year contributes 0 (like
    :func:`_average_numeric_tree``).  Dict fields are converted back to
    :class:`Counter` so ``.most_common`` works in the renderers.
    """
    if not digests:
        return None
    n = float(len(digests))

    def _avg_sum(key):
        # Sum across years, divide by n. Works for dict or list values.
        if not digests:
            return 0.0
        first = digests[0].get(key)
        if isinstance(first, dict):
            # Preserve insertion order across years (Counter.most_common
            # uses it for ties) by collecting keys in first-seen order.
            keys: dict = {}
            for d in digests:
                for k in (d.get(key) or {}):
                    keys.setdefault(k, None)
            return Counter({k: sum(float((d.get(key) or {}).get(k, 0.0)) for d in digests) / n
                            for k in keys})
        # list
        length = max((len(d.get(key) or []) for d in digests), default=0)
        out = [0.0] * length
        for d in digests:
            vals = d.get(key) or []
            for i, v in enumerate(vals):
                out[i] += float(v or 0.0)
        return [v / n for v in out]

    def _avg_cases(key):
        """Equal-weight mean across the per-year digests for one counter
        key: sums ``v_y / iterations_y`` over years with iterations > 0,
        then divides by the number of digests."""
        # sum_y(v_y / iterations_y) / n, skipping years with iterations 0.
        if not digests:
            return Counter() if isinstance(digests[0].get(key), Counter) else []
        first = digests[0].get(key)
        if isinstance(first, dict) or first is None:
            # Preserve insertion order across years (see _avg_sum).
            keys: dict = {}
            for d in digests:
                for k in (d.get(key) or {}):
                    keys.setdefault(k, None)
            out = Counter()
            for k in keys:
                total = 0.0
                for d in digests:
                    it = int(d.get("iterations", 0) or 0)
                    if it <= 0:
                        continue
                    total += float((d.get(key) or {}).get(k, 0.0)) / it
                out[k] = total / n
            return out
        # list
        length = max((len(d.get(key) or []) for d in digests), default=0)
        out = [0.0] * length
        for d in digests:
            it = int(d.get("iterations", 0) or 0)
            if it <= 0:
                continue
            vals = d.get(key) or []
            for i, v in enumerate(vals):
                out[i] += float(v or 0.0) / it
        return [v / n for v in out]

    return {
        "n_exams": sum(int(d.get("n_exams", 0) or 0) for d in digests) / n,
        "sex_scan_counts": _avg_sum("sex_scan_counts"),
        "sex_dose_msv": _avg_sum("sex_dose_msv"),
        "sex_cases": _avg_cases("sex_cases"),
        "sex_deaths": _avg_cases("sex_deaths"),
        "age_scan_counts": _avg_sum("age_scan_counts"),
        "age_dose_msv": _avg_sum("age_dose_msv"),
        "age_cases": _avg_cases("age_cases"),
        "age_deaths": _avg_cases("age_deaths"),
        "proc_dose_msv": _avg_sum("proc_dose_msv"),
        "proc_exam_counts": _avg_sum("proc_exam_counts"),
        "proc_cases": _avg_cases("proc_cases"),
        "proc_deaths": _avg_cases("proc_deaths"),
    }


def format_stratified_breakdown(
    exam_data: Optional[list],
    cancer_events: Optional[list],
    iterations: int,
    visible_cancers: Optional[set] = None,
    top_procedures: int = 10,
    exam_columns: Optional["ExamColumns"] = None,
    event_columns: Optional["EventColumns"] = None,
    detail_label: str = "",
    sections=("procedures", "sex", "age"),
    header=_BOX_HEADER_DEFAULT,
    _acc: Optional[dict] = None,
    indent: str = "  ",
    iter_f_override: Optional[float] = None,
) -> str:
    """Return a multi-line stratified breakdown of the cohort results.

    Sections (each omitted when the underlying data is empty):

    * **By sex**   — cases / deaths per M / F / U, normalised per cohort-year.
    * **By age group at exposure** — 0-14 (paed), 15-39, 40-64, 65+.
    * **Top-N procedures** by annual cases (``protocol`` field, modality fallback).

    ``iterations`` normalises counts so the numbers match the header of
    ``format_summary``.  When ``visible_cancers`` is non-None, events whose
    ``cancer_type`` is not in the set are dropped (matches the existing
    GUI filter).  Returns an empty string when both inputs are empty so
    callers can safely ``if strat: ...`` before appending.

    ``detail_label`` replaces the default "per cohort-year" qualifier in
    the box header -- used by :func:`format_summary` in ``per_year_average``
    mode to flag that the breakdown is computed from the detail year's
    ``exam_data``/``cancer_events`` only (not the cross-year average).

    ``sections`` selects which sub-sections to render and in what order
    (default all three, in the order listed above).  ``header`` controls the box:
    the default sentinel renders the current box header (using
    ``detail_label``); ``None`` omits the box entirely (embedded mode, used
    when :func:`format_summary` writes its own numbered headings); a string
    renders a box with that text.  ``_acc`` is a precomputed
    :func:`_stratified_accumulate` result so a caller rendering sections in
    two calls need not scan the data twice; ``None`` (the default) computes
    it here.
    """
    if iterations <= 0:
        return ""
    exam_data = exam_data or []
    cancer_events = cancer_events or []
    exam_columns = exam_columns or _ensure_exam_columns(exam_data)
    event_columns = event_columns or _ensure_event_columns(cancer_events)
    if _acc is None and (
        (exam_columns is None or len(exam_columns) == 0)
        and (event_columns is None or len(event_columns) == 0)
    ):
        return ""

    if _acc is None:
        _acc = _stratified_accumulate(
            exam_columns, event_columns, cancer_events, visible_cancers)

    iter_f = float(iterations) if iter_f_override is None else float(iter_f_override)
    n_exams = _acc["n_exams"]
    embedded = header is None

    lines: List[str] = []
    if header is not None:
        if header is _BOX_HEADER_DEFAULT:
            _qual = detail_label if detail_label else "per cohort-year"
            _header_text = f"Stratified breakdown ({_qual}; {n_exams:,} exams)"
            if len(_header_text) > 72:
                _header_text = _header_text[:72]
        else:
            _header_text = str(header)
            if len(_header_text) > 72:
                _header_text = _header_text[:72]
        lines = [
            "+" + "-" * 73 + "+",
            f"| {_header_text}".ljust(74) + "|",
            "+" + "-" * 73 + "+",
        ]

    _renderers = {
        "procedures": lambda: _render_strat_procedures(_acc, iter_f, top_procedures, embedded, indent),
        "sex": lambda: _render_strat_sex(_acc, iter_f, indent),
        "age": lambda: _render_strat_age(_acc, iter_f, indent),
    }
    for sec in sections:
        _sub = _renderers.get(sec)
        if _sub is None:
            continue
        _sub_lines = _sub()
        if _sub_lines:
            if lines and lines[-1] != "" and not embedded:
                lines.append("")
            lines.extend(_sub_lines)

    return "\n".join(lines)


def _simulated_annual_series(stats=None, modality_keys=None, *,
                             iter_cases=None, iter_deaths=None,
                             iter_qaly=None, events=None) -> Dict[str, Optional[np.ndarray]]:
    """Per-iteration (one simulated year) annual outcome series, or None.

    Shared body of :func:`simulated_annual_range`: resolves the
    ``cases`` / ``deaths`` / ``qaly`` per-iteration arrays for the given
    ``modality_keys`` (or the pooled lists when ``None``) and returns them
    as ``{"cases": arr|None, "deaths": arr|None, "qaly": arr|None}``.

    In ``per_year_average`` mode the pooled lists (one row per iteration
    of every processed year) describe the pooled distribution, so the
    same call covers all years.  With several modality keys the
    per-modality series are summed **elementwise** — iteration by
    iteration — never concatenated (concatenating would describe one
    year of a single modality, not the combined annual outcome).  A
    ``ValueError`` is raised if the series lengths differ.

    A modality absent from every pooled year contributes all zeros, so
    its series is a zero array (not ``None``).
    """
    n_iters = 0
    for _s in (iter_cases, iter_deaths, iter_qaly):
        if _s is not None and len(_s):
            n_iters = len(_s)
            break
    if n_iters <= 0:
        n_iters = int((stats or {}).get("iterations", 0) or 0)

    def _elem_add(acc, vals, what):
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return acc
        if acc is None:
            acc = np.zeros(arr.shape, dtype=np.float64)
        if acc.shape != arr.shape:
            raise ValueError(
                f"{what}: per-iteration series length {arr.size} != "
                f"{acc.size}; modality series must be combined elementwise "
                "(per simulated year), never concatenated"
            )
        return acc + arr

    keys: Optional[List[str]] = None
    if modality_keys:
        keys = [modality_keys] if isinstance(modality_keys, str) else [
            str(k) for k in modality_keys
        ]

    if keys:
        series: Dict[str, Optional[np.ndarray]] = {
            "cases": None, "deaths": None, "qaly": None
        }
        by_mod = (stats or {}).get("_iter_lists_by_modality")
        if by_mod is not None:
            for m in keys:
                e = by_mod.get(m) or {}
                for name, field in (
                    ("cases", "iter_cases"),
                    ("deaths", "iter_deaths"),
                    ("qaly", "iter_qaly"),
                ):
                    v = e.get(field) or []
                    if len(v) == 0 and n_iters > 0:
                        # Modality absent from every pooled year: genuine
                        # all-zero contribution, same convention as the
                        # Summary tab's modality filter.
                        v = [0.0] * n_iters
                    series[name] = _elem_add(series[name], v, f"{name} of {m!r}")
        elif events is not None and len(events):
            ev = _ensure_event_columns(events)
            if ev is not None and len(ev):
                it = np.asarray(ev.iteration)
                mod = np.asarray(ev.modality)
                died = np.asarray(ev.died)
                qaly_ev = np.asarray(ev.QALY_lost, dtype=np.float64)
                n_bins = n_iters if n_iters > 0 else (
                    int(it.max()) + 1 if it.size else 0)
                if n_bins > 0:
                    valid = (it >= 0) & (it < n_bins)
                    for m in keys:
                        sel = valid & (mod == m)
                        idx = it[sel].astype(np.int64)
                        series["cases"] = _elem_add(
                            series["cases"],
                            np.bincount(idx, minlength=n_bins), f"{m!r}")
                        series["deaths"] = _elem_add(
                            series["deaths"],
                            np.bincount(idx, weights=died[sel].astype(np.float64),
                                        minlength=n_bins), f"{m!r}")
                        series["qaly"] = _elem_add(
                            series["qaly"],
                            np.bincount(idx, weights=qaly_ev[sel],
                                        minlength=n_bins), f"{m!r}")
    else:
        series = {"cases": iter_cases, "deaths": iter_deaths,
                  "qaly": iter_qaly}

    out: Dict[str, Optional[np.ndarray]] = {}
    for name, s in series.items():
        if s is None or (hasattr(s, "__len__") and len(s) == 0):
            out[name] = None
        else:
            out[name] = np.asarray(s, dtype=np.float64)
    return out


def simulated_annual_range(stats=None, modality_keys=None, *,
                           iter_cases=None, iter_deaths=None,
                           iter_qaly=None, events=None):
    """Central 95% range of simulated annual outcomes.

    The 2.5th/97.5th percentiles of the per-iteration (one simulated year)
    annual outcome series — the quantity the Summary tab displays as the
    "95% range of simulated years".  In ``per_year_average`` mode the pooled
    lists (one row per iteration of every processed year) describe exactly
    this pooled distribution, so the same call covers all years.

    Parameters
    ----------
    stats : dict, optional
        Run/batch stats; ``iterations`` and ``_iter_lists_by_modality`` are
        read when resolving a ``modality_keys`` selection.
    modality_keys : str or iterable of str, optional
        Raw modality code(s) — the keys of ``stats['_iter_lists_by_modality']``
        / ``events.modality``.  ``None`` or empty means "all modalities": the
        pooled ``iter_cases``/``iter_deaths``/``iter_qaly`` lists are used.
        With several keys the per-modality series are summed **elementwise**
        — iteration by iteration — never concatenated (concatenating would
        describe one year of a single modality, not the combined annual
        outcome).  A ``ValueError`` is raised if the series lengths differ.
    iter_cases, iter_deaths, iter_qaly : sequence, optional
        The pooled per-iteration series (``results['cases_l']`` /
        ``results['deaths_l']`` / ``stats['iter_QALY_list']``).
    events : EventColumns | list | dict, optional
        Cancer events used to bin a per-modality series when
        ``stats['_iter_lists_by_modality']`` is absent (``single`` /
        ``combined`` runs).  Ignored in batch mode, where the surviving
        event payload is only the last processed year's.

    Returns
    -------
    dict
        ``{"cases": (lo, hi)|None, "deaths": (lo, hi)|None,
        "qaly": (lo, hi)|None}`` — ``None`` when no series is available.
        A modality absent from every pooled year contributes all zeros, so
        its range is ``(0.0, 0.0)``, not ``None``.
    """
    series = _simulated_annual_series(
        stats, modality_keys, iter_cases=iter_cases, iter_deaths=iter_deaths,
        iter_qaly=iter_qaly, events=events)
    out: Dict[str, Optional[Tuple[float, float]]] = {}
    for name, s in series.items():
        if s is None or len(s) == 0:
            out[name] = None
        else:
            out[name] = (float(np.percentile(s, 2.5)),
                         float(np.percentile(s, 97.5)))
    return out


def _mataac_strata_section(stats: dict, exam_columns,
                           cohort_examinations=None) -> List[str]:
    """Indented lines of the all-years-combined Mataac competing-mortality
    strata table for :func:`format_adjustments`.

    For a ``per_year_average`` batch the sources are each year's
    ``_risk_curve_digest`` exam columns; for a single/combined run the
    run's own ``exam_columns``.  When ``cohort_examinations`` (the analysed
    examination count) disagrees with the strata count, the digests
    predate the dental-radiography exclusion and a NOTE line says so.

    Returns ``[]`` when the Mataac correction is off or no source carries
    the Mataac columns.
    """
    if not stats.get("sim_config", {}).get("use_mataac", True):
        return []
    if stats.get("_run_mode") == "per_year_average" and stats.get("_year_batches"):
        year_batches = stats["_year_batches"]
        sources = [
            (y.get("_risk_curve_digest") or {}).get("exam_columns")
            for y in year_batches.values()
        ]
        _ys = sorted(str(y) for y in year_batches)
        _span = f"{_ys[0]}\u2013{_ys[-1]}" if len(_ys) > 1 else _ys[0]
        scope = f"all modalities, {len(year_batches)} year group(s) ({_span})"
    elif exam_columns is not None and len(exam_columns) > 0:
        sources = [exam_columns]
        scope = "all modalities, this run"
    else:
        return []
    strata = summarize_mataac_strata(sources)
    if strata is None:
        return []
    out = ["   " + ln for ln in
           format_mataac_strata_summary(strata, scope, heading=False).splitlines()]
    if (cohort_examinations is not None
            and int(cohort_examinations) != int(strata["n_exams"])):
        diff = abs(int(cohort_examinations) - int(strata["n_exams"]))
        out.append(
            f"   NOTE: the strata count differs from the analysed examinations "
            f"by {diff:,}; digests may predate the dental-radiography "
            f"exclusion (re-run the batch).")
    return out


def format_adjustments(stats: dict, exam_columns=None, exam_data=None) -> str:
    """Text for the Summary tab's Adjustments view: the competing-mortality
    strata (Mataac et al.) and the weight adjustment (Golikov and
    Druzhinina), each marked off when the run did not apply it."""
    stats = stats or {}
    sc = stats.get("sim_config") or {}
    if exam_columns is None and exam_data:
        exam_columns = ExamColumns.from_exam_data(exam_data)
    year_batches = _year_batches_of(stats)
    lines = ["COMPETING-MORTALITY ADJUSTMENT (Mataac et al.)"]
    if not sc.get("use_mataac", True):
        lines.append("   Off in this run.")
    else:
        lines.extend(_mataac_strata_section(
            stats, exam_columns,
            cohort_examinations=(_per_year_total_examinations(stats)
                                 if year_batches else None))
            or ["   No stratum data in this run."])
    lines += ["", "WEIGHT ADJUSTMENT (Golikov and Druzhinina)"]
    if not sc.get("use_golikov", True):
        lines.append("   Off in this run.")
    else:
        wa = stats.get("_weight_adjustment") or summarize_weight_adjustment(exam_data)
        if wa is None:
            lines.append("   No weight-adjustment data in this run.")
        else:
            if year_batches:
                _ys = sorted(str(y) for y in year_batches)
                _span = f"{_ys[0]}\u2013{_ys[-1]}" if len(_ys) > 1 else _ys[0]
                scope = f"{len(year_batches)} year group(s) ({_span}), all years combined"
            else:
                scope = "this run"
            lines.extend("   " + ln for ln in
                         format_weight_adjustment(wa, scope).splitlines())
    return "\n".join(lines)

def _year_label(label) -> str:
    """``"Year 2019"`` for a bare year label; labels that already start with
    "Year" are returned unchanged."""
    s = str(label)
    return s if s.lower().startswith("year") else f"Year {s}"


def format_summary(
    stats: dict,
    visible_cancers: set = None,
    background_msv_per_year: float = BACKGROUND_MSV_PER_YEAR,
    exam_data: Optional[list] = None,
    cancer_events: Optional[list] = None,
    iter_cases_list: Optional[list] = None,
    iter_deaths_list: Optional[list] = None,
    exam_columns: Optional["ExamColumns"] = None,
    event_columns: Optional["EventColumns"] = None,
    per_100k_examinations: bool = False,
) -> str:
    """Format live summary text.

    If *visible_cancers* is given, cases/deaths are recomputed from only those
    cancer types and detriment/QALYs/life-lost are scaled proportionally.

    If *per_100k_examinations* is set, the "Averages per iteration" headline
    block (cancers, deaths, heritable effects, detriment, QALYs lost,
    life-years lost -- including their 95% sampled range) is linearly
    rescaled from "per iteration" to "per 100,000 examinations", using
    :func:`resolve_unique_examinations` for the cohort size. This is the
    same ``per_n_scans`` projection convention as
    ``research_questions.compute_region_age_dose_sweep``: since the
    per-iteration expectation is a random quantity for a cohort of a
    particular (fixed) size, multiplying every term -- point estimate AND
    CI -- by ``100_000 / unique_examinations`` re-expresses the same
    distribution at a different, comparable examination count. Every other
    section (dataset-size lines and the stratified breakdown) is left
    in its original units.

    When ``exam_data`` and/or ``cancer_events`` are provided, a stratified
    breakdown section is appended (sex / age-group / top-N procedures; see
    ``format_stratified_breakdown``).
    """
    if not isinstance(stats, dict):
        return ""
    exam_columns = exam_columns or _ensure_exam_columns(exam_data)
    event_columns = event_columns or _ensure_event_columns(cancer_events)
    iterations = int(stats.get("iterations", 0) or 0)
    acquisition_rows = int(stats.get("acquisition_rows", 0) or 0)
    # Prefer a direct count of distinct A# accession numbers (the exact key
    # merge_acquisition_series groups on) over the acquisitions/iterations
    # quotient -- exam_data is the ground truth and this ties the reported
    # number unambiguously to "#A", independent of any future change to how
    # ``acquisitions`` is accumulated per iteration.  Falls back to the
    # quotient when exam_data isn't available (e.g. a cached/aggregated
    # stats dict with no per-exam detail).  In per_year_average mode the
    # per-exam denominator must be the per-year MEAN examination count (not
    # the last processed year's exam_data), so the per-exam figures derived
    # from equal-weighted per-year means divide by the matching mean -- see
    # resolve_cohort_examinations.
    unique_examinations = resolve_cohort_examinations(stats, exam_data)
    total_sv = float(stats.get("total_sv", 0.0) or 0.0)
    cases = float(stats.get("cancer_cases", 0.0) or 0.0)
    deaths = float(stats.get("cancer_deaths", 0.0) or 0.0)
    detriment = float(stats.get("total_detriment", 0.0) or 0.0)
    QALY_lost = float(stats.get("QALY_lost", 0.0) or 0.0)
    life_lost = float(stats.get("life_lost", 0.0) or 0.0)

    # Filter by visible cancer types if a subset is selected
    filter_note = ""
    if visible_cancers is not None:
        cbd = stats.get("cancer_breakdown", {})
        all_types = set(cbd.keys())
        hidden = all_types - visible_cancers
        if hidden:
            filtered_cases = sum(
                cv.get("developed", 0) for ct, cv in cbd.items() if ct in visible_cancers
            )
            filtered_deaths = sum(
                cv.get("deaths", 0) for ct, cv in cbd.items() if ct in visible_cancers
            )
            # Scale detriment/QALYs/life proportionally to the case ratio
            ratio = (filtered_cases / cases) if cases > 0 else 1.0
            cases = float(filtered_cases)
            deaths = float(filtered_deaths)
            detriment *= ratio
            QALY_lost *= ratio
            life_lost *= ratio
            filter_note = f"  [Filtered: {len(hidden)} cancer type(s) hidden]"

    dose_per_iteration_sv = (total_sv / iterations) if iterations > 0 else 0.0
    dose_per_iteration_msv = dose_per_iteration_sv * 1000.0
    # Risk-modelled per-exam denominator / dose. Falls back to the all-exams
    # count and total dose when the run carries no modelled mask
    # (results without ``scans_model`` / synthetic test dicts).
    _n_model_exams = resolve_modelled_examinations(stats, iterations, unique_examinations)
    _model_dose_sv_iter = _modelled_dose_sv(stats, iterations, dose_per_iteration_sv)
    _model_dose_msv = _model_dose_sv_iter * 1000.0
    _has_modelled = _modelled_by_modality(stats, iterations) is not None
    # Exams dropped upstream by the CT/Dental-CBCT no-organ-dose filter (not
    # in the simulation at all). Per-year mean in per_year_average mode.
    _dropped_upstream_by_mod = _exams_by_modality_for_summary(
        stats, key="excluded_no_organ_dose_by_modality"
    ) if _has_modelled else {}
    _total_dropped_upstream = int(sum(_dropped_upstream_by_mod.values()) or 0)
    # Exams dropped at source because they carry no risk model (Non-Dose
    # rows, no organ-dose model; participant_flow.py).  Per-year mean in
    # per_year_average mode.  0 for runs without
    # the ledger, where the same exams are still in exam_data and counted
    # via
    # ``unique_examinations - _n_model_exams`` instead.
    _total_dropped_upstream += int(sum(
        sum(_exams_by_modality_for_summary(stats, key=_k).values())
        for _k in _EXCLUDED_LEDGER_KEYS) or 0) if _has_modelled else 0
    avg_cases_sampled = (cases / iterations) if iterations > 0 else 0.0
    avg_deaths_sampled = (deaths / iterations) if iterations > 0 else 0.0

    # Calculate 95% sampled range (2.5th to 97.5th percentiles from iteration lists)
    if iter_cases_list and iter_deaths_list and len(iter_cases_list) >= 2:
        _sim_ranges = simulated_annual_range(
            stats, iter_cases=iter_cases_list, iter_deaths=iter_deaths_list)
        cases_ci_low, cases_ci_high = _sim_ranges["cases"]
        deaths_ci_low, deaths_ci_high = _sim_ranges["deaths"]
        _interval_label = "95% range of simulated years"
    else:
        cases_ci_low = cases_ci_high = avg_cases_sampled
        deaths_ci_low = deaths_ci_high = avg_deaths_sampled
        _interval_label = "Sampled"
    sc = stats.get("sim_config") or {}
    use_mataac = bool(sc.get("use_mataac", True))
    use_golikov = bool(sc.get("use_golikov", True))
    icrp_strict = bool(sc.get("icrp152_strict", False))
    # Sampled-YLL detriment basis: same rule as the Risk Analysis tab
    # (_event_detriment_weights) -- active unless the ICRP 152 reference
    # assumptions are on or the run used the single-tier "icrp152" QALY mode.
    _sampled_yll = (not icrp_strict
                    and str(sc.get("QALY_mode", "full")).lower() == "full")

    # Recompute detriment using ICRP Table 3.9 method (sampled counts × severity
    # factors): the l_rel path matches the By Cancer Type per-Sv table, while the
    # sampled path (non-strict, QALY_mode "full") uses per-patient YLL =
    # QALY_lost/15 and therefore sits below it for adult cohorts.
    from .risk_models import lookup_cancer_params as _lcp
    _t39_det_per_iter = 0.0
    _t39_nom_total_10k = 0.0
    _cbd_t39 = stats.get("cancer_breakdown", {})
    if iterations > 0:
        for _ct, _cv in _cbd_t39.items():
            if visible_cancers is not None and _ct not in visible_cancers:
                continue
            _dev = float(_cv.get("developed", 0))
            _param_key = {"Other solid": "Remainder"}.get(_ct, _ct)
            _p = _lcp(_param_key)
            _k = _p.get("k_lethality", 0.5)
            _qm = _p.get("q_min", 0.1)
            _lr = _p.get("l_rel", 1.0)
            _qa = _k + _qm * (1 - _k)
            _df = _k + _qa * (1 - _k)
            _t39_det_per_iter += (_dev / iterations) * _df * _lr
            if total_sv > 0:
                _t39_nom_total_10k += _dev / total_sv * 10000
    avg_QALY = (QALY_lost / iterations) if iterations > 0 else 0.0
    # In QALY_mode "full", QALY_lost = Σ q_individual × sampled YLL over cancer
    # events, whose expectation is the nested ICRP severity factor × E[YLL];
    # dividing by the 15 y ICRP average YLL gives the per-patient analogue of
    # the Table 3.9 detriment (heritable detriment is added as-is below).
    avg_detriment = (avg_QALY / _ICRP_AVG_LIFE_LOST_YEARS
                     if _sampled_yll else _t39_det_per_iter)
    avg_life_lost = (life_lost / iterations) if iterations > 0 else 0.0
    heritable_risk_acc = float(stats.get("heritable_risk", 0.0) or 0.0)
    heritable_det_acc = float(stats.get("heritable_detriment", 0.0) or 0.0)
    avg_heritable = (heritable_risk_acc / iterations) if iterations > 0 else 0.0

    # ICRP Table 3.9 total nominal risk (per Gy) and radiation detriment (per Sv)
    _heritable_nom_10k = (heritable_risk_acc / total_sv * 10000) if total_sv > 0 else 0.0
    _heritable_det_avg = (heritable_det_acc / iterations) if iterations > 0 else 0.0
    _t39_nom_per_Gy = (_t39_nom_total_10k + _heritable_nom_10k) / 10000
    _t39_det_per_Sv = (avg_detriment + _heritable_det_avg) / dose_per_iteration_sv if dose_per_iteration_sv > 0 else 0.0

    # Compute unique patient count for the summary header.
    # Derived from the pseudonymised per-person key (_person_id), which
    # exam_doses.py resolves as "Patient MRN" → A# accession.
    # Exams with a missing ID are each treated as a separate patient.
    _pid_counts: dict = {}
    _pid_none_n = 0
    _pid_nan_n = 0
    # Prefer cached aggregates (precomputed in main.run_simulation while
    # exam_data was alive) so the live Summary tab -- which calls
    # format_summary with exam_data=None to avoid blocking the worker
    # thread -- still shows the true unique-patient count instead of
    # falling back to unique_examinations.
    if "_pid_aggregates" in stats:
        _pid_agg = stats["_pid_aggregates"]
        if len(_pid_agg) == 5:
            _pid_counts, _, _, _pid_none_n, _pid_nan_n = _pid_agg
        else:
            _pid_counts, _, _ = _pid_agg
        unique_patients = len(_pid_counts) + _pid_none_n + _pid_nan_n
    elif exam_data:
        _pid_counts, _, _, _pid_none_n, _pid_nan_n = _build_patient_aggregates(exam_data)
        stats["_pid_aggregates"] = (_pid_counts, {}, {}, _pid_none_n, _pid_nan_n)
        unique_patients = len(_pid_counts) + _pid_none_n + _pid_nan_n
    else:
        unique_patients = unique_examinations

    # ``unique_patients`` so far is the DETAIL year's count (the retained
    # exam_data, i.e. the last processed year in per_year_average mode, or
    # the whole cohort otherwise).
    # The header / baseline arm need the cross-year AVERAGE patient
    # count in per_year_average mode (the baseline is per-person-per-lifetime,
    # and the per-iteration case figures are equal-weighted means), not the
    # single detail year.  Falls back to the detail-year count otherwise.
    unique_patients = resolve_cohort_patients(stats, unique_patients)

    # Diagnostic: how many rows actually had a "Patient MRN" value vs.
    # falling back to the A# accession number (which is ~unique per exam).
    # A near-1:1 patient:exam ratio with a high fallback share means the
    # input file has no usable "Patient MRN" column, not a code bug.
    _pid_src_note = ""
    if "_pid_source_counts" in stats:
        _src_counts = stats["_pid_source_counts"]
    elif exam_data:
        _src_counts = Counter(
            ex.get("_person_id_source", "unknown") for ex in exam_data
        )
        stats["_pid_source_counts"] = _src_counts
    else:
        _src_counts = None
    if _src_counts:
        _n_total_src = int(sum(_src_counts.values()) or 0) or 1
        _n_mrn = int(_src_counts.get("patient_mrn", 0) or 0)
        _n_fallback = int(_src_counts.get("accession_fallback", 0) or 0)
        _pid_src_note = (
            f"  (from 'Patient MRN' column: {_n_mrn:,} rows [{_n_mrn / _n_total_src * 100:.1f}%]"
            f"; fell back to A# accession number: {_n_fallback:,} rows [{_n_fallback / _n_total_src * 100:.1f}%])"
        )

    # For an "average across years" batch (run_mode == "per_year_average"),
    # ``exam_data`` above is only the *last* processed year (see
    # year_batch.py's memory note), so unique_examinations/unique_patients
    # as computed above would silently understate the real dataset.  Every
    # year's own raw (pre-average) stats are kept in ``_year_batches``
    # though, so the true dataset size is summed straight from there; the
    # per-year mean is shown alongside it for comparison with the
    # per-iteration figures below, which remain equal-weighted averages
    # only -- dose/cancers/detriment/QALY are not dataset-size counts, so
    # totalling them across years would not be a meaningful number.
    _year_batches = (
        stats.get("_year_batches")
        if stats.get("_run_mode") == "per_year_average" else None
    )
    _per_year = bool(_year_batches)
    _detail_year = stats.get("_detail_year")
    _dataset_lines: list = []
    if _year_batches:
        _n_yrs = len(_year_batches) or 1
        _tot_acq_rows = sum(int(y.get("acquisition_rows", 0) or 0) for y in _year_batches.values())
        # "_unique_examinations" / "_unique_patients" are direct per-year
        # counts precomputed in year_batch.py while each year's exam_data
        # was still alive (with an acquisitions/iterations fallback for
        # batches that predate that precomputation).  The per-year SUM of
        # examinations lives in _per_year_total_examinations so the
        # dataset-size totals and the per-exam denominator
        # (resolve_cohort_examinations, which divides this total by _n_yrs)
        # share one source of truth.  Unique PATIENTS, by contrast, are
        # UNIONED across years (a patient imaged in several years counts
        # once) via stats["_unique_patients_total"] -- built by
        # year_batch._cross_year_patient_totals, valid only because the
        # batch pins one pseudonymisation salt -- so the headline is a
        # distinct head-count, while the "/year" average still divides the
        # per-year SUM (_per_year_summed_patients, i.e. patient-years) by
        # _n_yrs.
        _tot_unique_exams = _per_year_total_examinations(stats) or 0
        _tot_unique_patients = _per_year_total_patients(stats) or 0
        _summed_patients = _per_year_summed_patients(stats) or 0
        _multi = stats.get("_patients_multi_year")
        _multi_note = f"; {_multi:,} imaged in more than one year" if _multi is not None else ""
        _dataset_lines.append(
            f"Dataset size, TOTAL across all {_n_yrs} year group(s) "
            "(per-year average in parentheses):"
        )
        if _tot_acq_rows > 0:
            _dataset_lines.append(
                f"  Acquisition series (input): {_tot_acq_rows:,}"
                f"  (avg {_tot_acq_rows / _n_yrs:,.0f}/year)"
            )
        _dataset_lines.append(
            f"  Unique examinations: {_tot_unique_exams:,}"
            f"  (avg {_tot_unique_exams / _n_yrs:,.0f}/year)"
        )
        _dataset_lines.append(
            f"  Unique patients: {_tot_unique_patients:,}"
            f"  (avg {_summed_patients / _n_yrs:,.0f}/year{_multi_note}){_pid_src_note}"
        )
    elif _run_year_count(stats) > 1:
        # "combined" batch: every year group's files were pooled into ONE
        # simulation, so acquisition_rows/unique_examinations/unique_patients
        # above already ARE the true combined totals (exam_data at this
        # point is the whole pooled cohort, not just one year -- unlike
        # per_year_average, nothing here was discarded).  Only the
        # per-year average needs adding, dividing by the number of year
        # groups that contributed files.
        _n_yrs = _run_year_count(stats)
        _dataset_lines.append(
            f"Dataset size, TOTAL across all {_n_yrs} year group(s) "
            "(per-year average in parentheses):"
        )
        if acquisition_rows > 0:
            _dataset_lines.append(
                f"  Acquisition series (input): {acquisition_rows:,}"
                f"  (avg {acquisition_rows / _n_yrs:,.0f}/year)"
            )
        _dataset_lines.append(
            f"  Unique examinations: {unique_examinations:,}"
            f"  (avg {unique_examinations / _n_yrs:,.0f}/year)"
        )
        _dataset_lines.append(
            f"  Unique patients: {unique_patients:,}"
            f"  (avg {unique_patients / _n_yrs:,.0f}/year){_pid_src_note}"
        )
    else:
        if acquisition_rows > 0:
            _dataset_lines.append(f"Acquisition series (input): {acquisition_rows:,}")
        _dataset_lines.append(f"Unique examinations: {unique_examinations:,}")
        _dataset_lines.append(f"Unique patients: {unique_patients:,}{_pid_src_note}")

    # Headline "per iteration" -> "per 100,000 examinations" rescale (see
    # the *per_100k_examinations* docstring paragraph above).  Only the
    # headline block below is affected: everything computed from
    # avg_cases_sampled / cases_ci_* / avg_deaths_sampled / etc. further
    # down (plain-language interpretation, decision-framing/NNH, the
    # stratified breakdown) intentionally keeps reading the un-rescaled
    # variables, so a display-only copy is used here instead of overwriting
    # them in place.
    _headline_label = "Averages per iteration:"
    (_cases_disp, _cases_lo_disp, _cases_hi_disp,
     _deaths_disp, _deaths_lo_disp, _deaths_hi_disp,
     _heritable_disp, _detriment_disp, _QALY_disp, _life_lost_disp) = (
        avg_cases_sampled, cases_ci_low, cases_ci_high,
        avg_deaths_sampled, deaths_ci_low, deaths_ci_high,
        avg_heritable, avg_detriment, avg_QALY, avg_life_lost,
    )
    if per_100k_examinations and unique_examinations > 0:
        _scale_100k = 100_000.0 / (_n_model_exams if _has_modelled else unique_examinations)
        if _has_modelled:
            _headline_label = (
                f"Averages per 100,000 examinations "
                f"(projected from {_n_model_exams:,} risk-modelled exams):"
            )
        else:
            _headline_label = (
                f"Averages per 100,000 examinations "
                f"(projected from {unique_examinations:,} actual):"
            )
        (_cases_disp, _cases_lo_disp, _cases_hi_disp,
         _deaths_disp, _deaths_lo_disp, _deaths_hi_disp,
         _heritable_disp, _detriment_disp, _QALY_disp, _life_lost_disp) = (
            v * _scale_100k for v in
            (avg_cases_sampled, cases_ci_low, cases_ci_high,
             avg_deaths_sampled, deaths_ci_low, deaths_ci_high,
             avg_heritable, avg_detriment, avg_QALY, avg_life_lost)
        )

    # Pre-compute section data
    _run_mode = stats.get("_run_mode")
    _year = stats.get("_year")
    if _per_year:
        _yr_keys = sorted(_year_batches.keys())
        _mode_line = (f"per-year average over {len(_yr_keys)} years "
                      f"({_yr_keys[0]}\u2013{_yr_keys[-1]}), equal-weighted"
                      if len(_yr_keys) > 1 else
                      f"one year group ({_yr_keys[0]})")
    elif _run_year_count(stats) > 1:
        _mode_line = f"{_run_year_count(stats)} year groups pooled into one run"
    else:
        _mode_line = "single run" + (f" [year {_year}]" if _year else "")
    if filter_note:
        _mode_line += filter_note

    _per_yr_qualifier = "mean per year" if _per_year else "per year"
    _detail_qualifier = (f"detail year {_detail_year} only, not averaged"
                         if _per_year else "per cohort-year")
    if per_100k_examinations and unique_examinations > 0:
        if _has_modelled:
            _headline_qualifier = (f"per 100,000 examinations "
                                  f"(projected from {_n_model_exams:,} risk-modelled exams)")
        else:
            _headline_qualifier = (f"per 100,000 examinations "
                                   f"(projected from {unique_examinations:,} actual)")
    elif _per_year:
        _headline_qualifier = "mean per year"
    else:
        _headline_qualifier = "per year"

    _risk_model = sc.get("risk_model", "ICRP 152")
    _population = sc.get("population", "EA")

    # Per-exam derived figures (used by sections 2 and 3). Denominator and
    # dose numerator are the risk-modelled exams when the modelled mask is
    # available, so per-exam figures describe only exams with a risk model.
    _denom_exams = _n_model_exams if _has_modelled else unique_examinations
    _denom_dose_msv = _model_dose_msv if _has_modelled else dose_per_iteration_msv
    heritable_rate_per_exam = (avg_heritable / _denom_exams) if _denom_exams > 0 else 0.0

    lines = [
        "=" * 75,
        f"SUMMARY  {_mode_line}",
        "=" * 75,
    ]

    # Partial calendar years
    # One line per partial year: annual totals scaled (mode "scale") or the
    # year excluded from the annual average (mode "exclude").  Only years
    # whose coverage fell below ``partial_year_min_coverage`` are reported;
    # full years are silent.
    _partial_scaling = stats.get("_partial_year_scaling") or {}
    _partial_excluded = stats.get("_partial_years_excluded") or []
    _year_coverage = stats.get("_year_coverage") or {}
    _partial_mode = (stats.get("sim_config") or {}).get("partial_year_mode", "scale")
    for _py in sorted(_partial_scaling.keys(), key=str):
        _cov = float(_year_coverage.get(_py, 0.0) or 0.0)
        _factor = float(_partial_scaling.get(_py, 1.0) or 1.0)
        lines.append(
            f"   {_year_label(_py)} covers {_cov * 100:.0f}% of the calendar year; "
            f"annual totals scaled by {_factor:.2f} "
            f"(partial_year_mode={_partial_mode})."
        )
    for _py in sorted(_partial_excluded, key=str):
        _cov = float(_year_coverage.get(_py, 0.0) or 0.0)
        lines.append(
            f"   {_year_label(_py)} covers {_cov * 100:.0f}% of the calendar year; "
            f"excluded from annual averages (partial_year_mode={_partial_mode})."
        )
    if _partial_scaling or _partial_excluded:
        lines.append("")

    # Achieved precision per year (per-year-average runs)
    # One line per year: iteration count, stop reason, the relative SE of
    # that year's per-iteration annual cancers and QALYs, the partial-year
    # handling (annualised / excluded).  Mirrors ``_precision_by_year``
    # built by year_batch.
    _precision_by_year = stats.get("_precision_by_year") or {}
    for _py in sorted(_precision_by_year.keys(), key=str):
        _p = _precision_by_year.get(_py)
        if not isinstance(_p, dict):
            continue
        _it = int(_p.get("iterations", 0) or 0)
        _stop = _p.get("stop_reason")
        _stop_note = ""
        if _stop == "stability_and_precision":
            _stop_note = " (stop: stability+precision)"
        elif _stop == "stability_only":
            _stop_note = " (stop: stability only)"
        elif _stop == "max_iterations":
            _stop_note = " (stop: max iterations)"
        elif _stop == "not_converged":
            _stop_note = " (stop: not converged)"
        _rse_cases = _p.get("rel_se_cases")
        _rse_qaly = _p.get("rel_se_qaly")
        _rse_cases_s = f"{_rse_cases * 100:.1f}%" if _rse_cases is not None else "n/a"
        _rse_qaly_s = f"{_rse_qaly * 100:.1f}%" if _rse_qaly is not None else "n/a"
        _cov = float(_p.get("coverage", 1.0) or 1.0)
        if _p.get("excluded"):
            _cov_note = "excluded"
        else:
            _sf = float(_p.get("scale_factor", 1.0) or 1.0)
            if _sf != 1.0:
                _cov_note = f"coverage {_cov * 100:.0f}%, annualised x{_sf:.2f}"
            else:
                _cov_note = f"coverage {_cov * 100:.0f}%"
        lines.append(
            f"   {_year_label(_py)}: {_it:,} iterations{_stop_note}; "
            f"rel. SE cancers {_rse_cases_s}, QALYs {_rse_qaly_s}; "
            f"{_cov_note}."
        )
    if _precision_by_year:
        lines.append("")
    elif not _per_year and isinstance(stats.get("_convergence"), dict):
        # Single-run (non-batch) precision line from the convergence record.
        _c = stats["_convergence"]
        _it = int(_c.get("iterations", 0) or 0)
        _stop = _c.get("stop_reason")
        _stop_note = ""
        if _stop == "stability_and_precision":
            _stop_note = " (stop: stability+precision)"
        elif _stop == "stability_only":
            _stop_note = " (stop: stability only)"
        elif _stop == "max_iterations":
            _stop_note = " (stop: max iterations)"
        elif _stop == "not_converged":
            _stop_note = " (stop: not converged)"
        _rse_cases = _c.get("rel_se_cases")
        _rse_qaly = _c.get("rel_se_qaly")
        _rse_cases_s = f"{_rse_cases * 100:.1f}%" if _rse_cases is not None else "n/a"
        _rse_qaly_s = f"{_rse_qaly * 100:.1f}%" if _rse_qaly is not None else "n/a"
        lines.append(
            f"   Run: {_it:,} iterations{_stop_note}; "
            f"rel. SE cancers {_rse_cases_s}, QALYs {_rse_qaly_s}."
        )
        lines.append("")

    # Section 1: imaged population, annual burden and per-exam risk
    _pop = _population_burden_by_modality(
        stats, iterations, exam_columns=exam_columns,
        event_columns=event_columns, iter_cases_list=iter_cases_list,
        iter_deaths_list=iter_deaths_list)
    if _pop is not None:
        _per_exam_notes = []
        if heritable_rate_per_exam > 0:
            _per_exam_notes.append(
                f"   Heritable effect (gonadal): {_format_one_in(heritable_rate_per_exam)} per exam")
        _per_exam_notes.append(
            f"   Chest X-rays: equivalent number at {CHEST_XRAY_MSV:g} mSv each. Background: time to "
            f"receive the same dose from natural background radiation ({background_msv_per_year:.1f} "
            f"mSv/yr, UNSCEAR worldwide average incl. radon).")
        _per_exam_notes.append("   Cancer risk is lifetime, not per year.")
        lines.append(f"1. Imaged population, annual burden and per-exam risk by modality "
                     f"({_per_yr_qualifier})")
        lines.extend(_render_population_burden_table(
            _pop, filter_note, per_exam_notes=_per_exam_notes,
            background_msv_per_year=background_msv_per_year))
        lines.append("")

    # Section 2: Model settings
    lines.append("2. Model settings")
    lines.append(f"   Risk model / population         : {_risk_model} / {_population}")
    lines.append(f"   DDREF solid / leukaemia         : {sc.get('ddref_solid', DDREF_SOLID_CANCERS):g} / {sc.get('ddref_leuk', DDREF_LEUKAEMIA):g}")
    lines.append(f"   Weight adjustment               : {'ON' if use_golikov else 'OFF'}")
    lines.append(f"   Competing-mortality adjustment  : {'ON' if use_mataac else 'OFF'}")
    lines.append(f"   ICRP 152 reference assumptions  : {'ON' if icrp_strict else 'OFF'}")
    if not _precision_by_year and not isinstance(stats.get("_convergence"), dict):
        lines.append(f"   MC iterations                   : {iterations:,}")
    lines.append("")

    # Sections 3 & 4: stratified breakdown
    # In per_year_average mode, prefer the cross-year averaged digest
    # (computed in year_batch.py while each year's exam_data was alive) when
    # it exists AND the visible_cancers filter hides no cancer type (the
    # digest is built with visible_cancers=None, so it would be inconsistent
    # with a filtered view).  Otherwise fall back to the detail-year path.
    _strat_digest_avg = stats.get("_strat_digest_avg") if isinstance(stats, dict) else None
    _hidden_types = set()
    if visible_cancers is not None:
        _hidden_types = set((stats.get("cancer_breakdown") or {}).keys()) - visible_cancers
    _use_avg_digest = bool(_per_year and _strat_digest_avg and not _hidden_types)
    if _use_avg_digest:
        _strat_acc = _strat_digest_avg
        _strat_iter_f = 1.0
        _strat_qualifier = f"mean per year over {len(_year_batches)} years"
    elif exam_data or cancer_events or exam_columns is not None or event_columns is not None:
        _strat_acc = _stratified_accumulate(
            exam_columns, event_columns, cancer_events or [], visible_cancers)
        _strat_iter_f = None  # use float(iterations)
        _strat_qualifier = _detail_qualifier
    else:
        _strat_acc = None
        _strat_iter_f = None
        _strat_qualifier = _detail_qualifier

    if _strat_acc is not None:
        # Read the top_procedures default from format_stratified_breakdown's
        # signature so the section heading stays in sync with the rendering.
        import inspect as _inspect
        _top_n = _inspect.signature(format_stratified_breakdown).parameters["top_procedures"].default
        _s4 = format_stratified_breakdown(
            exam_data=exam_data or [], cancer_events=cancer_events or [],
            iterations=iterations, visible_cancers=visible_cancers,
            exam_columns=exam_columns, event_columns=event_columns,
            sections=("procedures",), header=None, _acc=_strat_acc, indent="   ",
            iter_f_override=_strat_iter_f)
        if _s4:
            lines.append(f"3. Top {_top_n} procedures by cancer cases/yr ({_strat_qualifier})")
            lines.append(_s4)
            lines.append("")
        _s5 = format_stratified_breakdown(
            exam_data=exam_data or [], cancer_events=cancer_events or [],
            iterations=iterations, visible_cancers=visible_cancers,
            exam_columns=exam_columns, event_columns=event_columns,
            sections=("sex", "age"), header=None, _acc=_strat_acc, indent="   ",
            iter_f_override=_strat_iter_f)
        if _s5:
            lines.append(f"4. Who carries the burden ({_strat_qualifier})")
            lines.append(_s5)
            lines.append("")

    lines.append("=" * 75)
    return "\n".join(lines)

