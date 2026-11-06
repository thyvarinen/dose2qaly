"""Dose2QALY exam-dose computation.

This module is the canonical home for the per-exam dose dispatcher
and its modality-specific helpers.  ``main.py`` imports
``compute_exam_organ_doses`` and ``_compute_exam_batch`` from here;
``analysis.py`` imports ``_SEX_EXCLUDED_CANCER_TYPES`` and
``_iter_modeled_cancer_types`` from here.

Key design choices
------------------

* ``_INDIVIDUAL_REMAINDER_ORGANS`` includes ``"Salivary_Glands"`` so
  that dental/radiography helpers that emit a salivary-glands key
  fold it into the aggregated ICRP-103 Remainder dose alongside the
  other untracked remainder organs.
* Patient-id and exam-date are resolved with an ``or``-chain so a
  legitimately falsy id (integer 0, empty string) is treated as
  missing.
* ``_compute_exam_batch`` uses the zero-allocation
  ``_ColumnRowView`` so a column the active modality branch never
  reads costs nothing during preprocessing.

Purpose
-------
Turn a single Excel exam row into the per-organ absorbed dose dict (Gy)
and effective dose (mSv) that the ICRP 152 Monte Carlo risk engine
consumes.  This module owns all of the modality-specific dose lookups
(CT / mammography / radiography / fluoroscopy / dental).

Pipeline
--------
Excel row -> region (``region_resolver``) -> modality
dispatch -> dose coefficients (CT / XA / radiography /
mammography / dental) -> Golikov patient-size scaling for CT and XA
(see ``_golikov_relative_scale``) -> organ-name normalisation
(``_normalize_organ``) -> ``organ_doses_gy`` dict.

Dose-coefficient sources
------------------------
* CT organ-dose coefficients from ``ct_doses``.
* PCXMC radiography coefficients (mGy / Gy*cm²) from ``radiography_doses``.
* XA / fluoroscopy coefficients from ``fluoro_doses``.
* Mammography mean glandular dose to organ-dose conversion from
  ``mammography_doses`` (based on ICRP Publication 116).
* Dental DAP-to-organ mapping from ``dental_doses``.

Modalities handled
------------------
  CT / CT/PT            Direct per-organ doses from Excel columns.
  CBCT                  Dental CBCT only: DAP * PCXMC dental coefficients.
  Mammography (MG)      MGD taken as breast absorbed dose.
  Radiography (CR/DX/DR) DAP * PCXMC radiography coefficients.
  Angiography (XA/RF)   DAP * PCXMC fluoroscopy coefficients.

The Golikov body-size correction (CT and XA only) scales doses from the
70 kg reference phantom to the actual patient weight using a published
effective-diameter relation.  It is applied to patients BELOW that weight
only, and for CT only to the "Chest" and "Abdomen" region labels.  It is
OUTSIDE the ICRP 152 risk model
(which works directly from absorbed organ doses) and can be disabled
with ``use_golikov=False`` in the simulation config.

Public API
----------
  compute_exam_organ_doses(row)    Main per-row dispatcher used by the engine.
  _compute_exam_batch(args)        ProcessPoolExecutor worker entry point.

Module-private helpers
----------------------
  _to_float, _patient_weight_kg
  _get_golikov_coeffs, _golikov_relative_scale
  _resolved_orientation_from_region
  _ORGAN_NAME_MAP, _INDIVIDUAL_REMAINDER_ORGANS, _normalize_organ
  _resolve_fluoro_dap, _age_key
  _mg_organ_doses, _mg_effective_dose_msv
  _xa_doses, _fluoroscopy_doses, _radiography_doses, _dental_doses
  _SEX_EXCLUDED_CANCER_TYPES
"""
from __future__ import annotations

import hmac
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

# Project modules
from . import region_resolver as region_resolver
from .dental_doses import (
    DENTAL_DOSE_MAPPING,
    DENTAL_CBCT_DOSES_BY_KVP,
    DENTAL_CBCT_REFERENCE_DAP_GY_CM2,
    nearest_dental_cbct_kvp,
)
from .mammography_doses import mg_organ_doses_gy_from_mgd, mg_effective_dose_msv_from_mgd
from .fluoro_doses import (
    xa_organ_doses_gy_from_dap,
    xa_effective_dose_msv_from_dap,
)
from .ct_doses import (
    ct_organ_doses_from_excel_row,
    ct_effective_dose_from_excel_row,
)
from .projection_orientation import (
    get_projection_orientation,
    infer_projection_orientation_from_text,
)
from .risk_models import (
    normalize_sex_code,
    cancer_risk_params,
    CANCER_TO_ORGAN_DOSE_KEY,
    get_organ_dose_for_cancer,
    HERITABLE_GONAD_DOSE_KEY,
    REMAINDER_ORGAN_MULTIPLICITY,
)
from ._parse_utils import to_float as _to_float


# 1. LIGHTWEIGHT COLUMN-MAJOR ROW VIEW
# ``_ColumnRowView`` is a ``dict``-shaped read-only view backed by the
# column-major ``col_data`` dict + a row index.  It supports just enough
# of the dict API (``row.get(key, default)``, ``row[key]``, ``key in row``)
# for the modality dispatcher and its helpers to consume.  Because each
# access is a single ``col_data.get(key)`` plus one list index, columns
# that the active modality branch never reads (e.g. fluoroscopy never
# touches the ~30 ``... (CT) mSv`` organ-dose columns) cost nothing
# during preprocessing.

class _ColumnRowView:
    """Read-only view of one row of column-major Excel data.

    Constructed once per row (two ``__slots__``, no per-column allocation)
    and discarded when the row has been processed.  Honours the subset of
    the ``dict`` API that the dose dispatcher and its downstream helpers
    actually invoke, so modality branches that never mention a column
    (e.g. RF → ``Fetus (CT) mSv``) never incur the cost of reading it.
    """

    __slots__ = ("_data", "_i")

    def __init__(self, col_data: Dict[str, list], i: int):
        self._data = col_data
        self._i = i

    def get(self, key, default=None):
        """Return this row's value for column ``key``, or ``default`` when
        the column is absent or the entry is ``None``."""
        col = self._data.get(key)
        if col is None:
            return default
        return col[self._i]

    def __getitem__(self, key):
        return self._data[key][self._i]

    def __contains__(self, key):
        return key in self._data


# 2. GENERIC PARSING / PATIENT HELPERS

def _clean_id(x) -> Optional[str]:
    """Normalise a patient/accession id to a non-empty string or None.

    Empty Excel cells arrive as float ``NaN`` (truthy in Python) or as
    blank/whitespace strings; both must map to None so that rows lacking an
    accession id are never grouped together as the same examination/patient.
    """
    if x is None:
        return None
    if isinstance(x, float) and not np.isfinite(x):  # NaN / inf
        return None
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "unknown", "n/a", "null"):
        return None
    return s


# Patient-id pseudonymization
# The real patient/accession id (``A#`` / ``Patient MRN``) is only ever used
# downstream as an *opaque equality key* -- grouping multi-acquisition rows
# into one exam, computing per-patient cumulative dose, Mataac frailty
# patient boundaries, and duplicate-imaging diagnostics.  Its literal value
# is never displayed or otherwise needed.  To keep real patient identifiers
# out of every on-disk artifact (the .raw_excel_cache and
# .preprocess_cache pickle files), it is hashed into an opaque per-run
# token the moment it is read
# from Excel, here, before it ever reaches an exam dict.
#
# The salt is generated once per simulation run (see ``set_patient_id_salt``,
# called by ``preprocess.py`` before any row processing starts) and shared
# with every ``ProcessPoolExecutor`` worker via an explicit initializer, so
# the same real patient hashes to the same token everywhere within one run
# (including across multiple input files and across worker processes).  A
# fresh random salt is used on every run -- pseudonyms are NOT stable across
# separate runs, which is intentional (avoids ever needing to persist a
# re-identification key to disk).
#
# Within a ``per_year_average`` batch the salt is additionally PINNED across
# every year's run (still random per batch, still never persisted) by
# ``year_batch.run_yearly_simulation`` via :func:`pin_patient_id_salt`, so a
# patient imaged in several calendar years hashes to the same token in each
# year and the cross-year distinct-patient count can be taken as a set union.
# That mode therefore bypasses the full preprocess cache (see
# ``main.run_simulation``'s ``skip_preprocess_cache`` flag), since a cache hit
# would carry tokens written under a different salt and break the union.
_PATIENT_ID_SALT: Optional[bytes] = None
_PINNED_PATIENT_ID_SALT: Optional[bytes] = None


def set_patient_id_salt(salt: bytes) -> None:
    """Set the process-wide salt used to pseudonymize patient ids.

    Called once per run by ``preprocess.py`` in the main process, and again
    inside each ``ProcessPoolExecutor`` worker via ``initializer=`` so every
    worker shares the exact same salt as the parent process.
    """
    global _PATIENT_ID_SALT
    _PATIENT_ID_SALT = salt


def pin_patient_id_salt(salt: Optional[bytes]) -> None:
    """Pin the salt every run started in this process will use until unpinned (``None``).

    ``year_batch`` pins one salt for a whole per_year_average batch so a
    patient's pseudonym is identical in every year's run and the
    cross-year distinct-patient count can be taken as a set union.
    """
    global _PINNED_PATIENT_ID_SALT
    _PINNED_PATIENT_ID_SALT = salt


def new_run_patient_id_salt() -> bytes:
    """Salt for a new run: the pinned batch salt if set, else fresh ``os.urandom(16)``."""
    return _PINNED_PATIENT_ID_SALT if _PINNED_PATIENT_ID_SALT is not None else os.urandom(16)


def _get_or_init_salt() -> bytes:
    """Return the active salt, lazily generating one if unset.

    The lazy path only matters for callers that invoke
    ``compute_exam_organ_doses`` directly (e.g. tests, ad-hoc scripts)
    without going through ``preprocess.py``'s pipeline.
    """
    global _PATIENT_ID_SALT
    if _PATIENT_ID_SALT is None:
        _PATIENT_ID_SALT = os.urandom(16)
    return _PATIENT_ID_SALT


def _pseudonymize_patient_id(pid: str, salt: bytes) -> str:
    """Hash a raw patient/accession id into an opaque, equality-preserving token."""
    return hmac.new(salt, pid.encode("utf-8", "surrogatepass"), "sha256").hexdigest()[:16]


def _patient_weight_kg(row: dict) -> Optional[float]:
    weight = _to_float(row.get("Patient Weight kg"))
    if weight is None or weight <= 0:
        return None
    return weight


# 3. GOLIKOV PATIENT-SIZE CORRECTION
# CT/XA organ doses are scaled from the single adult reference phantom
# (Radimetrics / Cristy-Eckerman) to the actual patient weight using the
# Golikov/Druzhinina effective-dose conversion coefficient model.
#
# The one adult phantom is described by two masses in the literature: the
# nominal 70 kg ICRP Reference Man it models, and its own ~73 kg geometric
# mass.  We normalise on the nominal 70 kg and use the same value as the
# cut-off above which no correction is applied, so the scale factor reaches
# 1.0 continuously (see _golikov_relative_scale).  Do not set these two to
# different weights -- that reintroduces a step at the boundary.
_GOLIKOV_REF_WEIGHT_KG = 70.0  # ICRP Publication 23 Reference Man

_GOLIKOV_PARAMS = {
    "CT": {
        "Chest": {"a1": 0.11, "a3": 11.4},
        "Abdomen": {"a1": 0.13, "a3": 12.7},
    },
    "XA": {
        "default": {"a1": 0.13, "a3": 13.0},
    },
}

@lru_cache(maxsize=32)
def _get_golikov_coeffs(modality: str, region: str = None) -> Tuple[float, float, float]:
    a1, a3, a2 = 0.13, 13.0, 7.0
    modality_key = str(modality or "").strip().upper()
    region_key = str(region or "").strip()
    modality_params = _GOLIKOV_PARAMS.get(modality_key)
    if not modality_params:
        return a1, a3, a2
    if region_key in modality_params:
        region_params = modality_params[region_key]
        return region_params.get("a1", a1), region_params.get("a3", a3), a2
    default_params = modality_params.get("default", {})
    return default_params.get("a1", a1), default_params.get("a3", a3), a2


def _golikov_relative_scale(weight_kg: Optional[float], modality: str, region: str) -> float:
    """Golikov/Druzhinina body-size dose multiplier for CT and XA.

    Returns the ratio of the patient-size correction evaluated at
    ``weight_kg`` (kg) to the same correction at the reference phantom's
    weight -- a unitless factor >= 1.  Returns 1.0 (no scaling) when
    ``weight_kg`` is missing or non-positive, when the modality is not
    ``"CT"`` or ``"XA"`` (and, for CT, when ``region`` is not "Chest" or
    "Abdomen"), when the patient is at or above the reference weight, and
    on any evaluation error.
    """
    if weight_kg is None or weight_kg <= 0:
        return 1.0
    modality_key = str(modality or "").upper()
    if modality_key == "CT":
        if region not in ("Chest", "Abdomen"):
            return 1.0
    elif modality_key != "XA":
        return 1.0
    # ONE-SIDED BY CHOICE, not because heavy patients are better represented.
    # The Cristy-Eckerman series has a SINGLE adult phantom -- nominally the
    # 70 kg ICRP Reference Man, ~73 kg by the phantom's own geometry -- so a
    # 120 kg patient is matched to exactly the same phantom as a 71 kg one and
    # is, if anything, the case most in need of scaling.  Applying the Golikov
    # model in both directions would therefore be the physically complete
    # thing to do; it is deliberately not done, because that direction only
    # ever *reduces* an estimated dose and we would rather not talk a dose
    # down on the strength of a recorded weight.  Patients at or above the
    # reference keep the phantom dose unchanged.
    #
    # The cut-off and the ratio's normalisation point must be the SAME weight
    # or the function steps at the boundary: normalising at 73.2 while cutting
    # at 70 left a 4.5% jump between a 69.9 kg and a 70.0 kg patient.  Both are
    # the phantom's nominal 70 kg (see REF_WEIGHT_KG).
    if weight_kg >= _GOLIKOV_REF_WEIGHT_KG:
        return 1.0
    try:
        a1, a3, a2 = _get_golikov_coeffs(modality, region)
        ref_weight = _GOLIKOV_REF_WEIGHT_KG
        cce_patient = a3 * (1.0 - np.exp(-a1 * a2 * weight_kg ** (1.0 / 3.0))) / weight_kg
        cce_ref = a3 * (1.0 - np.exp(-a1 * a2 * ref_weight ** (1.0 / 3.0))) / ref_weight
        if not np.isfinite(cce_patient) or not np.isfinite(cce_ref) or cce_ref <= 0:
            return 1.0
        return float(cce_patient) / float(cce_ref)
    except Exception:
        return 1.0


# 4. ORIENTATION HELPER

def _resolved_orientation_from_region(region: Optional[str], fallback: Optional[str] = None) -> Optional[str]:
    region_text = str(region or "").strip()
    if not region_text:
        return fallback
    region_lower = region_text.lower()
    if region_lower.endswith(" pa"):
        return "PA"
    if region_lower.endswith(" ap"):
        return "AP"
    if region_lower.endswith(" lat"):
        return "Lat"
    if region_lower.endswith(" oblique"):
        return "RObl"
    return fallback


# Combined region + projection-orientation resolution (memoised)
# Region and the three orientation values are all pure functions of the
# region_resolver._RESOLVE_REGION_KEY_FIELDS values, so a single key lookup
# replaces ~20 per-row function calls — get_projection_orientation alone
# re-scans 8 fields x (normalize + infer) on every row (~17 calls/row).  The
# cache is cleared in lockstep with region_resolver._RESOLVE_REGION_CACHE (via
# the registered hook below) so the resolver's unknown-description side effect
# re-fires on each run's first encounter of a key.
_REGION_ORI_CACHE: Dict[tuple, tuple] = {}


def _resolve_region_and_orientation(row, protocol_text, description_text):
    """Return (region, orientation_detected, orientation_inferred,
    orientation_resolved) for ``row``, memoised on the resolver key fields.

    Identical to computing these inline: ``region`` still flows through
    ``region_resolver.resolve_region`` (preserving its unknown-description
    registration on first encounter), and the orientation values use the same
    expressions and call order as the non-memoised path (region resolution has
    no dependency on the orientation text, so evaluating it first changes nothing
    observable).
    """
    key = tuple(
        str(row.get(f, "") or "")
        for f in region_resolver._RESOLVE_REGION_KEY_FIELDS
    )
    cached = _REGION_ORI_CACHE.get(key)
    if cached is not None:
        return cached
    region = region_resolver.resolve_region(row)
    orientation_text = " ".join(
        part for part in (protocol_text, description_text)
        if part
    )
    detected = get_projection_orientation(row, orientation_text)
    inferred = infer_projection_orientation_from_text(orientation_text)
    resolved = _resolved_orientation_from_region(region, detected or inferred)
    result = (region, detected, inferred, resolved)
    _REGION_ORI_CACHE[key] = result
    return result


def _clear_region_ori_cache() -> None:
    _REGION_ORI_CACHE.clear()


region_resolver.register_cache_clear_hook(_clear_region_ori_cache)


# 5. ORGAN NAME NORMALIZATION

_ORGAN_NAME_MAP = {
    "Breasts": "Breast", "Breast_tissue": "Breast",
    "Lungs": "Lung",
    "Kidneys": "Kidney",
    "Urinary_Bladder": "Bladder", "Urinary bladder": "Bladder",
    "Stomach_Wall": "Stomach",
    "Rectosigmoid": "Colon",
    "Red_Marrow": "Leukaemia", "Active_Bone_Marrow": "Leukaemia",
    "Red bone marrow": "Leukaemia", "Active bone marrow": "Leukaemia",
    "Bone": "Bone", "Skeleton": "Bone", "Bone_surface": "Bone", "Endosteum": "Bone",
    "Ovaries": "Ovary",
    "Testicles": "Testis",
    "Esophagus": "Oesophagus",
}

# Individual ICRP-103 Remainder organs (after normalization) that should be
# aggregated into a single "Remainder" dose key when a precomputed Remainder
# value isn't supplied.  Per ICRP 103 §A.158 brain and salivary glands are
# NOT in the 13-organ Remainder list — they have their own w_T = 0.01 — so
# they are tracked individually under OTHER_SOLID_REMAINDER_SITES alongside
# Kidney and Pancreas (which ARE in the ICRP-103 Remainder list but get a
# dedicated dose key for higher fidelity).  The 10 organs below correspond
# to REMAINDER_ORGAN_MULTIPLICITY["Remainder"] = 10 untracked ICRP-103
# Remainder organs (the 13-Remainder list minus Kidney, Pancreas, prostate/
# uterus).  Prostate (M)/uterus (F) is excluded because its dose is captured
# elsewhere (gonadal dose for heritable detriment uses Testis/Ovary, not
# prostate/uterus); its REIC contribution to "Other solid" is implicitly 0.
_INDIVIDUAL_REMAINDER_ORGANS = frozenset({
    "Salivary_Glands", "Oral_Mucosa", "Extrathoracic_Airways",
    "Lymph_Nodes", "Muscle", "Thymus", "Adrenals", "Spleen",
    "Heart", "Small_Intestine", "Gallbladder",
})


@lru_cache(maxsize=256)
def _normalize_organ(name: str, sex_code: str = "U") -> Optional[str]:
    """Map raw organ/site name to canonical risk-model key."""
    raw = str(name or "").strip()
    if not raw:
        return None
    mapped = _ORGAN_NAME_MAP.get(raw)
    if mapped is not None:
        return mapped
    low = raw.lower().replace(" ", "_")
    if low == "prostate_uterus":
        return "Prostate" if sex_code == "M" else ("Uterus" if sex_code == "F" else None)
    if low == "gonads":
        return "Testis" if sex_code == "M" else ("Ovary" if sex_code == "F" else None)
    return raw


# 6. MODALITY-SPECIFIC ORGAN DOSE CALCULATION

def _resolve_fluoro_dap(row: dict) -> Optional[float]:
    """Return fluoroscopy DAP (Gy·cm²) from DAP (Total) Gy-cm2."""
    fdap = _to_float(row.get("DAP (Total) Gy-cm2"))
    return fdap if fdap and fdap > 0 else None


def _mg_organ_doses(row: dict) -> Optional[Dict[str, float]]:
    """Mammography organ doses from Mean Glandular Dose. Returns Gy."""
    mgd = _to_float(row.get("Mean Glandular Dose mGy"))
    if mgd is None or mgd <= 0:
        return None
    doses = mg_organ_doses_gy_from_mgd(mgd)
    if doses:
        return doses
    return {"Breast": mgd / 1000.0}


def _mg_effective_dose_msv(row: dict) -> Optional[float]:
    """Mammography effective dose from MGD in mSv."""
    mgd = _to_float(row.get("Mean Glandular Dose mGy"))
    if mgd is None or mgd <= 0:
        return None
    return mg_effective_dose_msv_from_mgd(mgd)


def _xa_doses(row: dict, region: str, age: float, sex_code: str) -> Tuple[
    Optional[Dict[str, float]], Optional[float], float
]:
    """Compute XA (angiography) organ doses, effective dose and Golikov scale.

    Single-pass: parses DAP once and queries the fluoroscopy/XA
    coefficient tables (``xa_organ_doses_gy_from_dap`` and
    ``xa_effective_dose_msv_from_dap``) for both organ doses and ED.
    The Golikov body-size scale is also resolved here so the caller can
    apply it to ED if patient-weight scaling is enabled.

    Returns ``(organ_doses_gy, ed_msv_no_golikov, golikov_scale)``.
    """
    dap = _to_float(row.get("DAP (Total) Gy-cm2"))
    if dap is None or dap <= 0:
        return None, None, 1.0
    doses_raw = xa_organ_doses_gy_from_dap(dap, region, row=row)
    ed_msv_raw = xa_effective_dose_msv_from_dap(dap, region, sex_code=sex_code, row=row)
    if age is not None and age >= 18:
        xa_scale = _golikov_relative_scale(_patient_weight_kg(row), "XA", region)
    else:
        xa_scale = 1.0
    if doses_raw:
        if xa_scale == 1.0:
            organ_doses = dict(doses_raw)
        else:
            organ_doses = {organ: dose * xa_scale for organ, dose in doses_raw.items() if dose > 0}
        organ_doses = organ_doses or None
    else:
        organ_doses = None
    ed_msv_no_golikov = ed_msv_raw if (ed_msv_raw is not None and ed_msv_raw > 0) else None
    return organ_doses, ed_msv_no_golikov, xa_scale


def _fluoroscopy_doses(row: dict, region: str, sex_code: str) -> Tuple[
    Optional[Dict[str, float]], Optional[float]
]:
    """Combined RF (fluoroscopy) organ-dose + effective-dose computation.

    Parses DAP once and calls the XA coefficient lookups; no Golikov scaling
    for RF (reference phantom doses).
    """
    fdap = _resolve_fluoro_dap(row)
    if fdap is None:
        return None, None
    doses = xa_organ_doses_gy_from_dap(fdap, region, row=row)
    ed_msv = xa_effective_dose_msv_from_dap(fdap, region, sex_code=sex_code, row=row)
    return (doses or None), (ed_msv if (ed_msv is not None and ed_msv > 0) else None)


def _radiography_doses(row: dict, region: str, age: float, sex_code: str) -> Tuple[
    Optional[Dict[str, float]], Optional[float]
]:
    """Compute radiography (CR / DX / DR) organ doses and effective dose.

    Single-pass: parses DAP, derives the projection orientation, beam
    quality and radiography exam key once, then passes them as keyword
    arguments to ``icrp_radiography_organ_doses_gy_from_dap`` and
    ``icrp_radiography_effective_dose_msv_from_dap`` so the radiography
    module does not need to re-derive them internally.

    The returned organ doses are passed through ``_normalize_organ`` so
    multiple raw keys (e.g. ``Gonads`` -> ``Testis`` for males,
    ``Ovary`` for females; ``Active_Bone_Marrow`` -> ``Leukaemia``) are
    folded onto the canonical risk-site names expected by the risk
    engine.

    Returns ``(organ_doses_gy, ed_msv)``.
    """
    from .radiography_doses import (
        icrp_radiography_organ_doses_gy_from_dap,
        icrp_radiography_effective_dose_msv_from_dap,
        _beam_quality_from_row as _rad_beam_quality_from_row,
        _normalize_exam_key as _rad_normalize_exam_key,
    )

    dap = _to_float(row.get("DAP (Total) Gy-cm2"))
    if dap is None or dap <= 0:
        return None, None
    # An unlocalised exam (blank / non-anatomical Body Part Examined) or
    # one outside the radiography exam regions has no organ-dose conversion:
    # the resolver's verdict stands, so free text in Protocol/Description
    # must not rescue it via the keyword match in ``_normalize_exam_key``.
    # The exam carries no organ doses / effective dose and is excluded from
    # the analysed cohort (listed in the By Body Region tab).
    if (not region
            or region == region_resolver.CT_UNLOCALIZED
            or region == region_resolver.RADIOGRAPHY_OTHER):
        return None, None
    projection = (
        get_projection_orientation(row, None)
        or infer_projection_orientation_from_text(str(region or ""))
    )
    beam_quality_key = _rad_beam_quality_from_row(row)
    _group = "paediatric" if (age is not None and age < 18) else "adult"
    exam_key = _rad_normalize_exam_key(
        region,
        " ".join([
            str(region or ""),
            str(row.get("Protocol") or ""),
            str(row.get("Description") or ""),
        ]),
    )
    raw = icrp_radiography_organ_doses_gy_from_dap(
        dap, region, age, sex_code, row=row,
        exam=exam_key, projection=projection, beam_quality_key=beam_quality_key,
    )
    ed_msv = icrp_radiography_effective_dose_msv_from_dap(
        dap, region, age, sex_code, row=row,
        exam=exam_key, projection=projection, beam_quality_key=beam_quality_key,
    )
    if raw:
        out: Dict[str, float] = {}
        for organ, dose in raw.items():
            mapped = _normalize_organ(organ, sex_code)
            if mapped:
                out[mapped] = out.get(mapped, 0.0) + dose
        organ_doses = out or None
    else:
        organ_doses = None
    return organ_doses, (ed_msv if (ed_msv is not None and ed_msv > 0) else None)


def _dental_doses(row: dict, region: str, sex_code: str) -> Tuple[
    Optional[Dict[str, float]], Optional[float]
]:
    """Compute dental-region organ doses and effective dose.

    Single-pass: parses DAP and queries ``DENTAL_DOSE_MAPPING`` once.
    Each entry in the mapping carries a per-organ coefficient (mGy /
    Gy*cm^2) plus an ``Average_Total_Body`` value used as the effective
    dose surrogate.  Organ doses are passed through ``_normalize_organ``
    to fold raw names onto canonical risk-site keys.

    Returns ``(organ_doses_gy, ed_msv)`` (ED already in mSv).
    """
    dap = _to_float(row.get("DAP (Total) Gy-cm2"))
    if dap is None or dap <= 0:
        return None, None

    if region == "Dental CBCT":
        kvp = _to_float(row.get("kVp kV", row.get("kVp", row.get("kV"))))
        entry = DENTAL_CBCT_DOSES_BY_KVP[nearest_dental_cbct_kvp(kvp)]
        scale = dap / DENTAL_CBCT_REFERENCE_DAP_GY_CM2
        out: Dict[str, float] = {}
        for organ, coeff_mgy in entry["coeffs"].items():
            if organ == "Average_Total_Body" or coeff_mgy <= 0:
                continue
            dose_gy = coeff_mgy * 1e-3 * scale      # mGy → Gy
            mapped = _normalize_organ(organ, sex_code)
            if mapped:
                out[mapped] = out.get(mapped, 0.0) + dose_gy
        organ_doses = out or None
        ed_msv = entry["effective_dose_msv"] * scale   # already in mSv
        return organ_doses, (ed_msv if ed_msv > 0 else None)

    mapping = DENTAL_DOSE_MAPPING.get(region)
    if not mapping:
        return None, None
    ref_dap = mapping["reference_dap_gy_cm2"]
    if ref_dap <= 0:
        return None, None
    scale = dap / ref_dap
    coeffs = mapping["coeffs"]
    out: Dict[str, float] = {}
    for organ, coeff_mgy in coeffs.items():
        if organ == "Average_Total_Body" or coeff_mgy <= 0:
            continue
        dose_gy = coeff_mgy * 1e-3 * scale          # mGy → Gy
        mapped = _normalize_organ(organ, sex_code)
        if mapped:
            out[mapped] = out.get(mapped, 0.0) + dose_gy
    organ_doses = out or None
    avg_body = coeffs.get("Average_Total_Body", 0.0)
    ed_msv = (avg_body * scale) if avg_body > 0 else None   # already in mSv
    return organ_doses, ed_msv


# 7. EXAM DOSE DISPATCHER

# Cancer types excluded by sex (used here and in risk calculation).
_SEX_EXCLUDED_CANCER_TYPES = {
    "M": {"Ovary", "Uterus"},
    "F": {"Prostate", "Testis"},
    "U": {"Ovary", "Uterus", "Prostate", "Testis"},
}


@lru_cache(maxsize=16)
def _cancer_dose_plan(sex_code: str) -> Tuple[str, ...]:
    """Per-sex tuple of cancer types to evaluate for this exam.

    Filters the static ``CANCER_TO_ORGAN_DOSE_KEY`` list by sex-based
    exclusions and ``cancer_risk_params`` membership.  The iteration order
    is preserved so ``modeled`` list / ``cancer_doses`` insertion order is
    unchanged.  Actual organ-dose lookup is delegated to
    ``get_organ_dose_for_cancer``.
    """
    excluded = _SEX_EXCLUDED_CANCER_TYPES.get(sex_code, _SEX_EXCLUDED_CANCER_TYPES["U"])
    return tuple(
        ct
        for ct in CANCER_TO_ORGAN_DOSE_KEY
        if ct in cancer_risk_params
        and ct not in excluded
        and not (ct == "Breast" and sex_code == "M")
    )


def _build_cancer_dose_map(
    organ_doses_gy: Dict[str, float], sex_code: str,
) -> Tuple[Dict[str, float], List[str]]:
    """Build the per-exam (cancer_dose_dict, modeled_types_list) pair.

    Resolves the absorbed dose for each sex-compatible cancer site via the
    shared ``get_organ_dose_for_cancer`` helper, then falls back to a direct
    cancer-type key in the dose dict.
    """
    cancer_doses: Dict[str, float] = {}
    modeled: List[str] = []
    for ct in _cancer_dose_plan(sex_code):
        dose = get_organ_dose_for_cancer(ct, organ_doses_gy)
        if dose <= 0:
            dose = organ_doses_gy.get(ct, 0.0)
        if dose > 0:
            cancer_doses[ct] = float(dose)
            modeled.append(ct)
    return cancer_doses, modeled


def _iter_modeled_cancer_types(organ_doses_gy: Dict[str, float], sex_code: str) -> List[str]:
    """Return modeled cancer-site names with a positive dose for ``sex_code``.

    Thin wrapper over :func:`_build_cancer_dose_map`; kept as a separate
    public helper so callers that only need the modelled-types list do
    not have to discard the ``cancer_doses`` dict.
    """
    return _build_cancer_dose_map(organ_doses_gy, sex_code)[1]


def _parse_population(value):
    """Resolve an Excel ``Population`` cell to a population label.

    Returns ``"Euro-American"`` or ``"Asian"``; returns ``None`` for an
    empty/missing/NaN cell (the row inherits the start-up population in
    ``main._prepare_cohort``).  Any other non-empty value raises
    ``ValueError`` so a typo can never silently fall back to a default.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    s = str(value).strip()
    sl = s.lower()
    if not s or sl == "nan":
        return None
    if sl.startswith("asian"):
        return "Asian"
    if sl.startswith("euro") or sl.startswith("ea"):
        return "Euro-American"
    raise ValueError(
        f"Unrecognised Population value {value!r}; "
        "expected Euro-American or Asian"
    )


def compute_exam_organ_doses(row: dict) -> Dict:
    """Resolve modality, region, and absorbed organ doses for one exam row.

    Pipeline
    --------
    1. Pull descriptive text fields (Description, Protocol) and resolve
       the imaging region via ``region_resolver.resolve_region``.
    2. Detect the modality from ``Examination Modalities`` / ``Modality``
       / ``Acquisition Modality`` and canonicalise to one of:
       ``CT``, ``Radiography``, ``Angiography``, ``CBCT``, ``MG`` or
       the raw value if unrecognised.
    3. Dispatch to the appropriate dose helper:
       * CT (incl. CT/PT) -- direct Excel organ doses only.
       * Mammography -- ``_mg_organ_doses`` (MGD taken as breast absorbed
         dose; effective dose = MGD * 0.12 mSv/mGy).
       * Radiography -- ``_radiography_doses`` (DAP * PCXMC radiography
         coefficients) with dental fallback for dental regions.
       * Angiography (XA / RF) -- ``_xa_doses`` / ``_fluoroscopy_doses``
         (DAP * fluoroscopy coefficients).
       * CBCT -- dental DAP only (non-dental CBCT yields no organ doses).
    4. Normalise organ keys via ``_normalize_organ`` so the risk engine
       sees canonical names (``Testis`` / ``Ovary`` / ``Leukaemia`` etc.).
    5. Aggregate the ten ``_INDIVIDUAL_REMAINDER_ORGANS`` into a single
       ``"Remainder"`` mean dose (ICRP 103 B.140-141 13-organ group;
       ``REMAINDER_ORGAN_MULTIPLICITY``).
    6. Pre-compute the per-cancer-site dose dict and the heritable gonad
       dose so the MC iteration can avoid repeating the fallback chain.
    7. Record the Golikov scale factor used (or 1.0) so the simulation
       can revert it when ``use_golikov=False``.

    Returns
    -------
    dict with keys
      ``modality``, ``region``, ``age``, ``sex_code``,
      ``organ_doses_gy``         absorbed dose per canonical organ (Gy),
      ``cancer_doses``           pre-resolved per-cancer-site dose (Gy),
      ``modeled_cancer_types``   list of cancer sites with positive dose,
      ``gonad_dose_gy``          gonad dose for heritable detriment,
      ``ed_msv``                 effective dose, mSv (Golikov-weighted),
      ``ed_msv_no_golikov``      effective dose without patient scaling,
      ``dose_source``            tag identifying which helper computed the doses,
      ``_golikov_factor``        scale factor baked into ``organ_doses_gy``,
      various description / patient metadata for downstream reporting.
    """
    description_text = str(row.get("Description", "") or "").strip()
    protocol_text = str(row.get("Protocol", "") or row.get("ExamType", "") or "").strip()
    # When Description is a generic placeholder (e.g. "Unknown Protocol",
    # "Unknown Study Description"), classification actually uses whichever
    # field carries real anatomy text (see resolve_region below) — show
    # that field for debug/audit display too, instead of the uninformative
    # placeholder, so region breakdowns are self-explanatory.
    if description_text and region_resolver._contains_any(
        region_resolver.normalize_text(description_text), region_resolver._GENERIC_UNKNOWN_TOKENS
    ):
        _real_text = next(
            (t for t in (protocol_text,)
             if t and not region_resolver._contains_any(
                 region_resolver.normalize_text(t), region_resolver._GENERIC_UNKNOWN_TOKENS
             )),
            None,
        )
        debug_description = f"{description_text} [Protocol: {_real_text}]" if _real_text else description_text
    else:
        debug_description = description_text or protocol_text

    # Resolve region + projection orientation together.  Memoised on the
    # resolver key fields, so this collapses ~20 per-row function calls
    # (region scan + 8-field x normalize/infer orientation scan) into one
    # dict lookup per distinct protocol/description key.
    region, orientation_detected, orientation_inferred, orientation_resolved = \
        _resolve_region_and_orientation(
            row, protocol_text, description_text,
        )

    # Detect modality — Acquisition Modality is ground truth
    modality_raw = str(
        row.get("Acquisition Modality", None)
        or row.get("Examination Modalities", None)
        or row.get("Modality", None)
        or ""
    ).strip().upper()
    age = _to_float(row.get("Age at Exam yrs"))
    # "Gender" is the primary column; "Sex" is accepted as a fallback for
    # exports that label the same demographic field differently.
    sex_code = normalize_sex_code(row.get("Gender", None) or row.get("Sex", None) or "U")
    region_lower = str(region or "").lower()

    # Force female for mammography if sex unknown
    is_mg = modality_raw == "MG" or "MAMMO" in modality_raw or region_lower == "breast"
    if is_mg and sex_code == "U":
        sex_code = "F"

    # ── Canonical modality normalization (merge related modalities) ──
    if modality_raw in ("CR", "DX", "DR"):
        modality_canon = "Radiography"
    elif modality_raw in ("XA", "RF") or "ANGIO" in modality_raw:
        modality_canon = "Angiography"
    elif "CBCT" in modality_raw:
        modality_canon = "CBCT"
    elif "CT" in modality_raw:  # covers CT, CT/PT, CTPA, etc.
        modality_canon = "CT"
    else:
        modality_canon = modality_raw
    # A "Dental CBCT" examination is a cone-beam CT whatever archive code
    # it carries (combo panoramic/CBCT units export DX/CR), and it is
    # dosed with the dental CBCT model either way, so report it under
    # CBCT everywhere.
    if modality_canon == "Radiography" and str(region or "") == "Dental CBCT":
        modality_canon = "CBCT"

    # CT classification is driven entirely by the DICOM "Scan Regions" field
    # (see region_resolver), not by the Finnish NCSP code/description shown
    # above -- for CT rows, replace the debug description with the raw Scan
    # Regions value so the "By Body Region" tab stays internationally
    # legible (no dependency on Finnish NCSP codes/text ("kuntakoodit") to
    # understand why a row was classified the way it was).
    if modality_canon == "CT":
        _scan_regions_text = str(row.get("Scan Regions", "") or "").strip()
        debug_description = f"Scan Regions: {_scan_regions_text}" if _scan_regions_text else "Scan Regions: (blank)"

    organ_doses = None
    dose_source = "None"
    ed_msv = 0.0
    ed_msv_no_golikov = 0.0
    _xa_golikov_scale = 1.0  # cached for _golikov_factor; avoids third recompute
    _ct_golikov_scale = 1.0  # Golikov for adult CT (Cristy series has ONE adult phantom)
    _golikov_no_weight = False  # True iff adult Chest/Abdomen CT kept unscaled ref-phantom doses because weight was missing
    _golikov_eligible = False  # True iff adult Chest/Abdomen CT (weight-correction eligible, regardless of weight)
    _dose_quantity_missing = False  # True iff the modality's primary dose quantity is absent or <= 0

    # ── CT (including CT/PT) ──
    # Organ doses come exclusively from the direct Excel columns
    # ("Brain (CT) mSv", "Liver (CT) mSv", etc.) produced by the
    # dose-tracking system.  CT rows without direct organ doses will have
    # organ_doses = None and contribute zero modelled cancer risk
    # (they are flagged by the missing-CT-doses prompt in main.py).
    if modality_canon == "CT":
        organ_doses = ct_organ_doses_from_excel_row(row)
        if organ_doses:
            dose_source = "Excel_CT"
            excel_ed = ct_effective_dose_from_excel_row(row)
            if excel_ed is not None and excel_ed > 0:
                ed_msv = excel_ed
                ed_msv_no_golikov = excel_ed
            # Golikov correction for adult CT: the Cristy series contains a single
            # adult phantom, so every adult maps to the same reference phantom
            # regardless of weight (identical to the XA situation).  Applied for the
            # "Chest" and "Abdomen" region labels only -- note that a combined
            # chest-abdomen(-pelvis) scan resolves to "Trunk" and is therefore NOT
            # size-corrected, which the manuscript states explicitly.
            if age is not None and age >= 18:
                _weight_kg = _patient_weight_kg(row)
                _golikov_eligible = (region in ("Chest", "Abdomen"))
                _ct_golikov_scale = _golikov_relative_scale(_weight_kg, "CT", region)
                # Adult Chest/Abdomen CT with no recorded weight keeps the
                # unscaled reference-phantom doses (the Golikov factor is 1.0
                # for missing weight). Flag it so the participant-flow ledger
                # can report how many exams fell into this bucket.
                if _golikov_eligible and _weight_kg is None:
                    _golikov_no_weight = True
                if _ct_golikov_scale != 1.0:
                    organ_doses = {k: v * _ct_golikov_scale for k, v in organ_doses.items()}
                    ed_msv = ed_msv * _ct_golikov_scale
        else:
            # CT row with no organ-dose columns: primary dose quantity missing.
            _dose_quantity_missing = True

    # ── MG ──
    elif is_mg:
        organ_doses = _mg_organ_doses(row)
        mg_ed = _mg_effective_dose_msv(row)
        if mg_ed is not None and mg_ed > 0:
            ed_msv = mg_ed
        else:
            _dose_quantity_missing = True
        dose_source = "MammographyMGD"

    # ── Radiography (CR + DX + DR) ──
    elif modality_canon == "Radiography":
        _rad_dap = _to_float(row.get("DAP (Total) Gy-cm2"))
        if _rad_dap is None or _rad_dap <= 0:
            _dose_quantity_missing = True
        # Check for dental regions first
        if region_lower.startswith("dental") or region_lower in ("kefalometry", "intraoral"):
            organ_doses, dental_ed = _dental_doses(row, region, sex_code)
            if dental_ed is not None and dental_ed > 0:
                ed_msv = dental_ed
            if organ_doses:
                dose_source = "Dental_DAP"
        else:
            from .radiography_doses import _beam_quality_from_row
            if _beam_quality_from_row(row) is None:
                _dose_quantity_missing = True
            organ_doses, cr_ed = _radiography_doses(row, region, age or 0, sex_code)
            if cr_ed is not None and cr_ed > 0:
                ed_msv = cr_ed
            if organ_doses:
                dose_source = "Radiography_DAP"

    # ── Angiography (XA + RF merged, XA dose framework) ──
    elif modality_canon == "Angiography":
        if modality_raw == "RF":
            _rf_dap = _to_float(row.get("DAP (Total) Gy-cm2"))
            if _rf_dap is None or _rf_dap <= 0:
                _dose_quantity_missing = True
            organ_doses, rf_ed = _fluoroscopy_doses(row, region, sex_code)
            if rf_ed is not None and rf_ed > 0:
                ed_msv = rf_ed
            if organ_doses:
                dose_source = "Fluoroscopy"
        else:
            _xa_dap = _to_float(row.get("DAP (Total) Gy-cm2"))
            if _xa_dap is None or _xa_dap <= 0:
                _dose_quantity_missing = True
            organ_doses, xa_ed_no_golikov, _xa_golikov_scale = \
                _xa_doses(row, region, age or 0, sex_code)
            if xa_ed_no_golikov is not None and xa_ed_no_golikov > 0:
                ed_msv_no_golikov = xa_ed_no_golikov
                xa_ed = xa_ed_no_golikov * _xa_golikov_scale
                if xa_ed > 0:
                    ed_msv = xa_ed
            if organ_doses:
                dose_source = "XA"

    # ── CBCT ──
    # Only dental CBCT has a dose model (DAP × dental coefficients).
    # Non-dental CBCT produces organ_doses = None and contributes zero
    # modelled cancer risk (flagged by the missing-CT-doses prompt).
    elif modality_canon == "CBCT":
        if region_lower.startswith("dental cbct"):
            _cbct_dap = _to_float(row.get("DAP (Total) Gy-cm2"))
            if _cbct_dap is None or _cbct_dap <= 0:
                _dose_quantity_missing = True
            organ_doses, dental_ed = _dental_doses(row, region, sex_code)
            if dental_ed is not None and dental_ed > 0:
                ed_msv = dental_ed
            if organ_doses:
                dose_source = "Dental_CBCT_DAP"

    if ed_msv_no_golikov <= 0:
        ed_msv_no_golikov = ed_msv

    # Normalize organ names
    if organ_doses:
        normalized = {}
        for k, v in organ_doses.items():
            mapped = _normalize_organ(k, sex_code)
            if mapped and v > 0:
                normalized[mapped] = max(normalized.get(mapped, 0.0), float(v))
        organ_doses = normalized

    # Aggregate individual remainder organs into a single "Remainder" key
    # while preserving individual doses for dose-proportional sampling.
    # CT already provides "Remainder" directly; dental and other sources
    # may supply individual organs (Salivary_Glands, Oral_Mucosa, etc.) that
    # need aggregation.  Mean over the 10-organ multiplicity so the downstream
    # risk calculation (which weights Remainder ×10 / 13) stays correct.
    remainder_individual_doses = {}
    if organ_doses and "Remainder" not in organ_doses:
        rem_sum = 0.0
        for k, v in organ_doses.items():
            if k in _INDIVIDUAL_REMAINDER_ORGANS:
                rem_sum += v
                remainder_individual_doses[k] = v
        if rem_sum > 0:
            # Add aggregated Remainder key for risk calculation
            organ_doses["Remainder"] = rem_sum / float(REMAINDER_ORGAN_MULTIPLICITY.get("Remainder", 10))

    # Precompute cancer dose map and modeled types once per exam.
    # These are invariant across MC iterations, avoiding redundant
    # dict lookups + fallback chains on every iteration.
    od_final = organ_doses or {}
    cancer_doses, modeled_types = _build_cancer_dose_map(od_final, sex_code)

    # Gonad dose for heritable detriment — extracted from od_final directly because
    # Testis/Ovary are excluded from cancer_risk_params and never enter _cancer_doses.
    _gkey = HERITABLE_GONAD_DOSE_KEY.get(sex_code)
    _gonad_dose_gy = float(get_organ_dose_for_cancer(_gkey, od_final)) if _gkey else 0.0

    # Store the Golikov scale factor baked into organ_doses_gy so it can be
    # undone in run_simulation when use_golikov=False. Adult XA applies a
    # patient-size scale; adult CT does so for the Chest and Abdomen region
    # labels (Cristy series has a single adult phantom).  Paediatric exams
    # (age < 18), patients at or above the reference weight, combined "Trunk"
    # scans and all other sources use reference-phantom doses unchanged
    # (factor = 1.0).
    if dose_source == "XA":
        _golikov_factor = _xa_golikov_scale
    elif dose_source == "Excel_CT" and _ct_golikov_scale != 1.0:
        _golikov_factor = _ct_golikov_scale
    else:
        _golikov_factor = 1.0

    # ``A#`` is the accession number — unique per examination.  It is used
    # as the grouping key in ``merge_acquisition_series`` to combine
    # multi-acquisition rows into one exam.  Empty Excel cells arrive as
    # float ``NaN`` (truthy in Python); ``_clean_id`` maps NaN / blank /
    # whitespace to None so that rows with no accession stay solo.
    patient_id = _clean_id(row.get("A#"))

    # ``_person_id`` is the per-person key: "Patient MRN" (the medical-record
    # number, stable across a patient's repeat visits), falling back to A#
    # when no MRN is present.  The key matters for the Mataac frailty
    # correction, which groups exams by ``_person_id`` to accrue per-patient
    # calendar-year CT dose; without it every exam collapses onto its own
    # accession (A#) and the per-patient stratum is lost.
    _raw_person_id = (
        _clean_id(row.get("Patient MRN"))
        or _clean_id(row.get("PATIENT MRN"))
        or _clean_id(row.get("MRN"))
    )
    if _raw_person_id is not None:
        person_id = _raw_person_id
        person_id_source = "patient_mrn"
    else:
        person_id = _clean_id(row.get("A#"))
        person_id_source = "accession_fallback" if person_id is not None else "none"
    # Pseudonymize immediately: the real id is only needed as an opaque
    # equality key downstream and must never reach exam_data / disk caches.
    # See "Patient-id pseudonymization" above _clean_id.
    if patient_id is not None:
        patient_id = _pseudonymize_patient_id(patient_id, _get_or_init_salt())
    if person_id is not None:
        person_id = _pseudonymize_patient_id(person_id, _get_or_init_salt())
    exam_date = row.get("Date Performed") or row.get("DatePerformed") or None

    # Per-exam population override from Excel (Euro-American or Asian)
    exam_population = _parse_population(row.get("Population"))

    # Reporting region: for radiography, strip the orientation suffix so
    # that "Chest PA", "Chest AP", "Chest Lat" all report as "Chest".
    # The actual risk calculation still uses the full `region` label.
    # Dental-imaging subtypes (Panoramic/Kefalometry/Intraoral/Dental) keep
    # their own subtype name here -- they get grouped into the
    # "Dental imaging" pseudo-modality for display elsewhere (research
    # questions, By Body Region), not by collapsing the region label itself,
    # so per-subtype breakdowns stay available everywhere reporting_region
    # is used. "Dental CBCT" already keeps its own name for the same reason.
    if region in region_resolver.DENTAL_IMAGING_SUBTYPE_REGIONS:
        reporting_region = region
    elif modality_canon == "Radiography":
        reporting_region = region_resolver.radiography_reporting_region(
            region or "Unknown")
    else:
        reporting_region = region or "Unknown"

    return {
        "modality": modality_canon or modality_raw or "Unknown",
        "region": region or "Unknown",
        "reporting_region": reporting_region,
        "age": age or 0.0,
        "sex_code": sex_code,
        "protocol": protocol_text,
        "description": description_text,
        "debug_description": debug_description,
        "orientation_inferred": orientation_inferred or "",
        "orientation_detected": orientation_detected or "",
        "orientation_resolved": orientation_resolved or "",
        "organ_doses_gy": od_final,
        "ed_msv": max(0.0, ed_msv),
        "ed_msv_no_golikov": max(0.0, ed_msv_no_golikov),
        "dose_source": dose_source,
        "_golikov_factor": _golikov_factor,
        "_golikov_no_weight": _golikov_no_weight,
        "_golikov_eligible": _golikov_eligible,
        "_dose_quantity_missing": _dose_quantity_missing,
        "_cancer_doses": cancer_doses,
        "_modeled_types": modeled_types,
        "_gonad_dose_gy": _gonad_dose_gy,
        "_patient_id": patient_id,
        "_person_id": person_id,
        "_person_id_source": person_id_source,
        "_exam_date": exam_date,
        "_remainder_individual_doses": remainder_individual_doses,
        "_population": exam_population,
    }


def _compute_exam_batch(args):
    """Module-level worker: process a batch of exam rows in parallel (ProcessPoolExecutor).

    Receives a column-major data dict already sliced to this batch's rows plus
    a (start, end) range addressed from 0.  Slicing per batch (see
    ``preprocess._batch_gen``) keeps IPC pickle volume and per-worker memory
    O(N) instead of O(batches * N).

    Each row is wrapped in a zero-allocation :class:`_ColumnRowView` instead
    of materialising a per-row dict of all columns — the modality dispatch
    inside ``compute_exam_organ_doses`` then naturally touches only the
    columns relevant to each row's modality (e.g. an RF/fluoroscopy row
    never reads ``Fetus (CT) mSv`` or any other ``(CT) mSv`` field).
    """
    col_data, row_file_index, start, end = args
    results = []
    for i in range(start, end):
        row = _ColumnRowView(col_data, i)
        result = compute_exam_organ_doses(row)
        result["protocol"] = str(row.get("Protocol", "") or "")
        _dev = row.get("Device")
        _eqp = row.get("Equipment")
        _dte = row.get("Date Performed")
        result["device"]         = "" if _dev is None else str(_dev)
        result["equipment"]      = "" if _eqp is None else str(_eqp)
        result["date_performed"] = _dte
        result["_file_index"] = row_file_index[i]
        results.append(result)
    return results, region_resolver.get_unknown_descriptions()
