"""Resolve free-text exam descriptions to canonical body-region labels.

Two-stage classifier:
  1. **Evidence extraction** — :func:`build_evidence` extracts all
     observable facts from the row (description, protocol, NCSP code,
     modality letter, projection, etc.) into an immutable
     :class:`RegionEvidence` object.
  2. **Modality-specific classification** — the dispatcher calls the
     resolver appropriate for the row's modality (CT, XA/RF, radiography,
     nuclear, mammography, CBCT, dental).  Each resolver returns a
     :class:`RegionDecision` carrying the region, confidence, source,
     rule, and evidence string for auditability.

The public API is :func:`resolve_region(row)`, which returns the region
string.

CT region taxonomy (derived from the DICOM "Scan Regions" field only)
-----------------------------------------------------------------------
  Base buckets: Head, Neck, Chest, Abdomen, Pelvis, Spine, Extremities,
  Trunk, Other. "Trunk" covers chest+abdomen(+pelvis) combinations --
  named for the anatomical compartment actually imaged (thorax+abdomen,
  with or without pelvis) rather than "Wholebody", which would overclaim
  anatomical extent for e.g. a plain Chest+Abdomen exam with no head or
  extremities coverage. The vendor's own literal "wholebody" Scan Regions
  atom is folded into this same bucket (see _CT_SCAN_REGION_ATOM_MAP) --
  true full-body scans are rare enough in this dataset that a separate
  bucket is not warranted, and Trunk remains the closer of the two labels
  even for those cases. Upper and lower extremities are merged into a single
  "Extremities" bucket (too few CT exams of either to warrant separate
  buckets). "Unknown/unspecified region" (``CT_UNLOCALIZED``) covers rows
  with blank/unrecognised Scan Regions -- these still carry real per-organ
  doses from the Excel dose-tracking columns (region is a reporting
  dimension only, see below), so they are included in annual
  burden/benefit-by-region estimates, but excluded from the region x
  age-group heatmap (no anatomical identity to plot against). When
  ``Scan Regions`` lists two or more distinct buckets
  (e.g. "Abdomen, Chest"), the resolved region is the comma-joined
  combination of those buckets in canonical order (e.g. "Chest, Abdomen")
  rather than a generic catch-all — this keeps common combo protocols
  (chest/abdomen/pelvis, etc.) distinguishable in reporting.
  Deliberately NOT derived from NCSP procedure codes or free text, so CT
  classification is portable/publishable independent of Finnish
  "kuntakoodit". See :func:`resolve_ct_region_from_scan_regions`.

Radiography (CR/DX/DR) uses the PCXMC radiography region set (Chest,
Skull PA, Abdomen AP, Pelvis AP, Lumbar spine AP, etc.), classified
primarily from the DICOM "Body Part Examined (DX)" field (see
_DX_BODY_PART_BASE_REGION_MAP and resolve_radiography_region) rather than
NCSP procedure codes or free text. An empty or non-anatomical value
resolves to "Unknown/unspecified region", same as CT's convention for
blank Scan Regions -- but unlike CT (whose organ doses come from the Excel
columns) a radiography exam's dose is derived from DAP via region-keyed
coefficients, so an unlocalised radiography row has no organ-dose
conversion and is excluded from the analysed cohort (listed in the By Body
Region tab).

CBCT is classified the same way from "Body Part Examined (CBCT)": HEAD or
SKULL -> "Dental CBCT" (the only CBCT region with a real dose model);
every other value is reported under its own literal body-part label with
no risk attributed to it; empty -> "Unknown/unspecified region". See
resolve_cbct_region.

The resolver is heavily memoised: typical hospital datasets have only a
few hundred distinct (description, protocol, modality) tuples across
millions of rows, so the cache turns an O(N × 100-branch) string scan
into an O(N) dict lookup.
"""
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from .projection_orientation import get_projection_orientation, infer_projection_orientation_from_text, infer_projection_from_code


# CT region taxonomy = derived directly from the DICOM "Scan Regions" field,
# NOT from Finnish NCSP procedure codes or free-text description matching.
# This keeps CT classification code-independent (portable/publishable
# outside Finland). Scan Regions only carries plain anatomy, so clinical
# specializations (Cardiac/Aortic, Vessels, Urinary, Brain vs. Head&neck,
# Nuclear/Hybrid) are folded into plain
# anatomy buckets; see resolve_ct_region_from_scan_regions.
# CT organ doses come per-organ from the Excel dose-tracking columns, so the
# region label is a reporting/grouping dimension only.
#
CT_HEAD = "Head"
CT_NECK = "Neck"
CT_CHEST = "Chest"
CT_DIGESTIVE = "Abdomen"
CT_PELVIS = "Pelvis"
CT_SPINE = "Spine"
CT_EXTREMITIES = "Extremities"
CT_TRUNK = "Trunk"  # chest+abdomen(+pelvis); also covers genuine vendor "wholebody" reports
CT_UNLOCALIZED = "Unknown/unspecified region"  # blank/unrecognised Scan Regions

# Catch-all for rare multi-region combos folded at the reporting layer (see
# ``fold_rare_region_combos``).  Single CT buckets are never folded; only
# combination labels (e.g. "Neck, Pelvis") with fewer than
# ``RARE_COMBO_THRESHOLD`` scans are merged into this bucket so the long tail
# of n=1-13 combos does not clutter reports and heatmaps.
CT_OTHER_COMBINATIONS = "Other combinations"
RARE_COMBO_THRESHOLD = 30  # fold combos with fewer than this many scans; reporting convention only, does not affect dose or risk

# Base (single-atom) CT buckets. Rows whose "Scan Regions" list two or more
# distinct buckets get a dynamically-generated combo label (comma-joined, in
# _CT_BUCKET_ORDER order) instead of one of these — see
# resolve_ct_region_from_scan_regions.  Rare combos (fewer than
# RARE_COMBO_THRESHOLD scans) are folded into CT_OTHER_COMBINATIONS at the
# reporting layer (fold_rare_region_combos), not here — the resolver itself
# always produces the granular label so dose calculation is unaffected.

# Canonical head-to-foot ordering used to name multi-region combo buckets
# deterministically (e.g. always "Chest, Abdomen", never "Abdomen, Chest").
_CT_BUCKET_ORDER = (
    CT_HEAD, CT_NECK, CT_CHEST, CT_DIGESTIVE, CT_PELVIS, CT_SPINE,
    CT_EXTREMITIES, CT_TRUNK,
)

# Atomic "Scan Regions" vendor labels -> canonical CT bucket. Scan Regions
# values are lists of these atoms separated by ",", "/", ";", or "+" (e.g.
# "Abdomen, Chest" or "Chest/Abdomen/Pelvis"). Matching is case-insensitive.
_CT_SCAN_REGION_ATOM_MAP = {
    "head": CT_HEAD,
    "neck": CT_NECK,
    "chest": CT_CHEST,
    "abdomen": CT_DIGESTIVE,
    "pelvis": CT_PELVIS,
    "l spine": CT_SPINE,
    "t spine": CT_SPINE,
    "c spine": CT_SPINE,
    "lower extremities": CT_EXTREMITIES,
    "knee": CT_EXTREMITIES,
    "ankle": CT_EXTREMITIES,
    "foot": CT_EXTREMITIES,
    "runoff": CT_EXTREMITIES,
    "upper extremities": CT_EXTREMITIES,
    "wrist": CT_EXTREMITIES,
    "elbow": CT_EXTREMITIES,
    "hand": CT_EXTREMITIES,
    "wholebody": CT_TRUNK,  # genuine vendor whole-body report, folded into Trunk (see module docstring)
}

_RE_SCAN_REGION_SPLIT = re.compile(r"[,/;+]")

# Used by the (unrelated) Nuclear-modality (NM/PT/PET) resolver below --
# not part of the CT Scan-Regions taxonomy above.
NUCLEAR_REGION = "Nuclear/Hybrid (CT Component)"

ANGIO_FLUORO_UNKNOWN = "UnknownAngiography"

# Official NCSP "Muu"/"Täydentävä" (other/supplementary) procedure codes are
# recognised, real procedures — they are simply not anatomically localised
# by design (e.g. "Muun valtimon stenttaus" = stenting of an unspecified
# artery). These are a genuinely different category from ANGIO_FLUORO_UNKNOWN
# (rows where no procedure could be identified at all, e.g. missing/
# placeholder text) and are tracked separately so the two are not conflated.
ANGIO_FLUORO_OTHER = "Other angiography"

ANGIO_FLUORO_REGIONS = {
    "Cystography",
    "ERCP",
    "CarotidAngiography",
    "Coronary",
    "PeripheralAngiography",
    "AbdominalAngiography",
    "GIFluoroscopy",
    "MSKIntervention",
    ANGIO_FLUORO_OTHER,
    ANGIO_FLUORO_UNKNOWN,
}

DENTAL_ALLOWED_REGIONS = {
    "Dental CBCT",
    "Dental Panoramic",
    "Dental_intraoral",
    "Kefalometry",
    "Dental",
}


RADIOGRAPHY_MODALITIES = {"CR", "DX", "DR"}

# Radiography reporting catch-all for rows whose anatomy cannot be mapped to a
# radiography exam region (keeps foreign-modality labels like "CT urography" / "Cardiac"
# out of the radiography group).
RADIOGRAPHY_OTHER = "Other radiography"

# Mapping from orientation-specific radiography regions to combined
# reporting regions (orientation stripped).  Used only for reporting /
# display — the actual risk calculation still uses the full region label
# so the correct dose coefficients are applied per orientation.
_RADIOGRAPHY_REPORTING_REGION_MAP = {
    "Chest PA": "Chest",
    "Chest AP": "Chest",
    "Chest Lat": "Chest",
    "Pelvis AP": "Pelvis",
    "Abdomen AP": "Abdomen",
    "Lumbar spine AP": "Lumbar spine",
    "Lumbar spine Lat": "Lumbar spine",
    "Lumbar spine oblique": "Lumbar spine",
    "Cervical spine AP": "Cervical spine",
    "Cervical spine Lat": "Cervical spine",
    "Thoracic spine AP": "Thoracic spine",
    "Thoracic spine Lat": "Thoracic spine",
    "Thoraco-lumbar spine Lat": "Thoraco-lumbar spine",
    "Scoliosis AP": "Scoliosis",
    "Scoliosis PA": "Scoliosis",
    "Scoliosis Lat": "Scoliosis",
    "Skull PA": "Skull",
    "Skull Lat": "Skull",
    "Extremities": "Extremities",
}


def radiography_reporting_region(region: str) -> str:
    """Strip the orientation suffix from a radiography region label.

    Returns the combined body-region name (e.g. ``"Chest PA"`` → ``"Chest"``).
    Non-radiography regions and unknown labels are returned unchanged.
    """
    return _RADIOGRAPHY_REPORTING_REGION_MAP.get(region, region)

# Regions a radiography (CR/DX/DR) row is allowed to carry.  This is the
# PCXMC radiography region set (incl. "Extremities") plus the catch-all above.  Anything
# else produced by the CT-oriented code-prefix map is coerced back into this
# set by resolve_radiography_region.
# Loaded lazily from radiography_doses to avoid importing the large coefficient
# tables at startup.
@lru_cache(maxsize=None)
def _radiography_exam_regions() -> Tuple[str, ...]:
    from .radiography_doses import RADIOGRAPHY_EXAM_REGIONS
    return RADIOGRAPHY_EXAM_REGIONS


# CT-style / intermediate base labels -> nearest
# allowed radiography exam region, preserving anatomy.  Urography maps to
# Abdomen AP (IVU is an abdominal radiograph).
_RADIOGRAPHY_BASE_FALLBACK = {
    "head and neck": "Skull PA",
    "head": "Skull PA",
    "chest": "Chest AP",
    "cardiac ct": "Chest AP",
    "ctpa": "Chest AP",
    "abdomen": "Abdomen AP",
    "ct urography": "Abdomen AP",
    "urography": "Abdomen AP",
    "chest-abdomen-pelvis": "Abdomen AP",
    "thoracic / lumbar spine": "Lumbar spine AP",
    "extremities": "Extremities",
    "extremity": "Extremities",
}

_UNKNOWN_DESCRIPTION_CACHE = set()
_UNKNOWN_DESCRIPTIONS = []  # list of (description_text, modality) tuples

# Memoisation cache for resolve_region
# Hospital datasets typically have <1 000 distinct (description, protocol,
# modality, projection-metadata) tuples but millions of exam rows.  Without a
# cache, resolve_region re-runs its ~100-branch string-
# contains chain for every single row; with a cache, every duplicate hits a
# dict lookup instead.  Cleared together with _UNKNOWN_DESCRIPTIONS at the
# start of every simulation so the unknown-descriptions side effect is not
# stale across runs.
_RESOLVE_REGION_CACHE: dict = {}

# Fields that resolve_region and its dependencies actually
# read from ``row``.  The resolver is pure given these fields, so the tuple
# of their string values is a safe cache key.
#
# Top-level reads:  Description, Procedure,
#                   Protocol, Acquisition Protocol, Protocol Name, ExamType,
#                   Scan Regions, Body Part Examined (DX), Body Part Examined (CBCT),
#                   Examination Modalities, Modality, Acquisition Modality
# get_projection_orientation reads _PROJECTION_FIELDS from
# projection_orientation:  Projection,
#                   Position, View, Description, Protocol, Exam Type
# Union (deduped, in stable order):
_RESOLVE_REGION_KEY_FIELDS: tuple = (
    "Description",
    "Procedure",
    "Protocol",
    "Acquisition Protocol",
    "Protocol Name",
    "ExamType",
    "Exam Type",
    "Scan Regions",
    "Body Part Examined (DX)",
    "Body Part Examined (CBCT)",
    "Examination Modalities",
    "Modality",
    "Acquisition Modality",
    "Projection",
    "Position",
    "View",
)


# Callbacks fired whenever the resolver caches are cleared (start of each
# run).  Lets dependent modules (e.g. exam_doses' combined region+orientation
# cache) stay in lockstep with _RESOLVE_REGION_CACHE so their first-encounter
# side effects re-fire correctly.
_CACHE_CLEAR_HOOKS: list = []


def register_cache_clear_hook(fn) -> None:
    """Register a zero-arg callable to run whenever the resolver caches clear."""
    if fn not in _CACHE_CLEAR_HOOKS:
        _CACHE_CLEAR_HOOKS.append(fn)


def get_unknown_descriptions():
    """Return list of (description_text, modality) for unrecognized regions."""
    return list(_UNKNOWN_DESCRIPTIONS)


def clear_unknown_descriptions():
    """Clear the unknown descriptions cache (call before new simulation)."""
    _UNKNOWN_DESCRIPTION_CACHE.clear()
    _UNKNOWN_DESCRIPTIONS.clear()
    # Must also clear the resolve cache — otherwise rows whose first encounter
    # registered an unknown description in a previous run would now be served
    # from cache without re-registering that unknown (because @lru_cache-style
    # memoisation skips the function body entirely on hit).  Clearing both
    # together preserves the exact unknown-description bookkeeping behaviour
    # of the uncached path.
    _RESOLVE_REGION_CACHE.clear()
    # Clear dependent caches (e.g. exam_doses region+orientation memo) so they
    # cannot serve a stale hit that would skip the unknown re-registration.
    for _hook in _CACHE_CLEAR_HOOKS:
        _hook()


# Text normalisation

# Actual mojibake sequences (UTF-8 decoded as Latin-1).  Each key is the
# corrupted byte sequence that appears in real data; the value is the correct
# Finnish string.  Only full-word corruptions are listed — never single-letter
# or short-substring replacements that could corrupt valid text.
_MOJIBAKE_MAP = {
    "rntgen": "röntgen",
    "tydentv": "täydentävä",
    "yksikn": "yksikön",
}

_RE_WS = re.compile(r"\s+")


def _fix_mojibake(value: str) -> str:
    for bad, good in _MOJIBAKE_MAP.items():
        value = value.replace(bad, good)
    return value


def normalize_text(value: str) -> str:
    """Normalise a string for token matching: strip, lower-case, NFKC, fix mojibake, collapse whitespace."""
    value = str(value or "")
    value = value.strip().lower()
    value = unicodedata.normalize("NFKC", value)
    value = _fix_mojibake(value)
    value = _RE_WS.sub(" ", value)
    return value


# Evidence object

@dataclass(frozen=True)
class RegionEvidence:
    """All observable facts extracted from a row, computed once."""
    description_raw: str
    protocol_raw: str
    series_raw: str
    procedure_raw: str
    modality_raw: str

    description: str
    protocol: str
    series: str
    combined: str
    scan_regions: str

    description_lower: str
    protocol_lower: str
    series_lower: str
    combined_lower: str

    code_description: Optional[str]
    code_protocol: Optional[str]
    code_series: Optional[str]
    code: Optional[str]

    code_prefix2: Optional[str]
    code_prefix3: Optional[str]
    code_prefix4: Optional[str]
    code_modality_letter: Optional[str]

    normalized_modality: str
    projection: Optional[str]

    body_part_examined_dx: str
    body_part_examined_cbct: str


# Decision object

@dataclass(frozen=True)
class RegionDecision:
    """Result of a region classification, with auditability metadata."""
    region: str
    confidence: str  # "high", "medium", "low"
    source: str      # "code", "text", "prefix", "fallback", "override"
    rule: str        # human-readable rule name
    evidence: str    # the text/code that triggered the rule


# Evidence builder

# Generic/placeholder NCSP code prefixes that carry no anatomical meaning on
# their own (supplementary fluoro runs, "täydentävä" codes, etc.). When a
# more specific code is available in another field, it should win even if
# the generic code appears in a higher-priority field (e.g. Protocol).
#
# "zxf" is the procedure-interruption family (ZXF05 = "Toimenpiteen
# keskeyttäminen…"). It records *why* a study stopped, never what was imaged,
# so it must not shadow the real code: the archive's ZXF05 rows carry ordinary
# diagnostic protocols ("VATSA_NATIIVI", "Akuuttivatsa yli 70v", "Thoraxin
# röntgen") and real recorded dose. See _is_non_dose for why they are kept.
_GENERIC_PLACEHOLDER_CODE_PREFIXES = ("xx", "zxf")


def _is_generic_placeholder_code(code: Optional[str]) -> bool:
    return bool(code) and code.startswith(_GENERIC_PLACEHOLDER_CODE_PREFIXES)


def _select_best_code(*candidates: Optional[str]) -> Optional[str]:
    """Pick the best code among extraction candidates, in priority order.

    Prefers the first non-generic (non-placeholder) NCSP code (containing
    at least one digit, which all real NCSP codes do, e.g. ``jn4ad``) over
    purely alphabetic protocol shorthands (e.g. ``thx``) that are extracted
    from Protocol fields but are not official NCSP procedure codes.
    Only falls back to a generic placeholder code if no specific code was
    found anywhere.
    """
    # Pass 1: first non-generic code that looks like a real NCSP code (has a digit)
    for candidate in candidates:
        if candidate and not _is_generic_placeholder_code(candidate) and any(c.isdigit() for c in candidate):
            return candidate
    # Pass 2: first non-generic code (may be a protocol shorthand like "thx")
    for candidate in candidates:
        if candidate and not _is_generic_placeholder_code(candidate):
            return candidate
    # Pass 3: generic placeholder as last resort
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def build_evidence(row) -> RegionEvidence:
    """Extract all observable facts from a row into an immutable evidence object."""
    description_raw = str(
        row.get("Description", None)
        or row.get("Procedure", None)
        or ""
    )
    procedure_raw = str(
        row.get("Procedure", None)
        or ""
    )
    protocol_raw = str(
        row.get("Protocol", None)
        or row.get("Acquisition Protocol", None)
        or row.get("Protocol Name", None)
        or row.get("ExamType", None)
        or ""
    )
    series_raw = ""
    acquisition_protocol_raw = str(row.get("Acquisition Protocol", None) or "")
    protocol_name_raw = str(row.get("Protocol Name", None) or "")
    scan_regions_raw = str(row.get("Scan Regions", None) or "")
    modality_raw = str(
        row.get("Acquisition Modality", None)
        or row.get("Examination Modalities", None)
        or row.get("Modality", None)
        or ""
    )
    body_part_examined_dx = str(row.get("Body Part Examined (DX)", None) or "").strip()
    body_part_examined_cbct = str(row.get("Body Part Examined (CBCT)", None) or "").strip()

    description = description_raw.strip()
    protocol = protocol_raw.strip()
    series = series_raw.strip()
    acquisition_protocol = acquisition_protocol_raw.strip()
    protocol_name = protocol_name_raw.strip()
    scan_regions = scan_regions_raw.strip()
    combined = " ".join([
        protocol, acquisition_protocol, protocol_name,
        procedure_raw.strip(), description, scan_regions,
    ]).strip()

    description_lower = normalize_text(description)
    protocol_lower = normalize_text(protocol)
    series_lower = normalize_text(series)
    combined_lower = normalize_text(combined)

    code_description = _extract_code_token(description)
    code_protocol = _extract_code_token(protocol)
    code_acquisition_protocol = _extract_code_token(acquisition_protocol) if acquisition_protocol else None
    code_protocol_name = _extract_code_token(protocol_name) if protocol_name else None
    code_series = None
    # Protocol fields often contain the actual code used, so they are preferred.
    # Description is only the displayed study label in some exports. However,
    # generic placeholder codes (e.g. "XX9LA Täydentävä lpv" = supplementary
    # fluoro run) are sometimes recorded in Protocol/Procedure even when the
    # real procedure code is only present in Description — prefer any
    # non-generic code over a generic placeholder regardless of field order.
    code = _select_best_code(
        code_protocol, code_acquisition_protocol, code_protocol_name,
        code_description,
    )

    code_prefix2 = code[:2] if code and len(code) >= 2 else None
    code_prefix3 = code[:3] if code and len(code) >= 3 else None
    code_prefix4 = code[:4] if code and len(code) >= 4 else None
    code_modality_letter = code[4] if code and len(code) >= 5 else None

    normalized_modality = detect_modality(modality_raw)

    projection = get_projection_orientation(row, combined)

    return RegionEvidence(
        description_raw=description_raw,
        protocol_raw=protocol_raw,
        series_raw=series_raw,
        procedure_raw=procedure_raw,
        modality_raw=modality_raw,
        description=description,
        protocol=protocol,
        series=series,
        combined=combined,
        scan_regions=scan_regions,
        description_lower=description_lower,
        protocol_lower=protocol_lower,
        series_lower=series_lower,
        combined_lower=combined_lower,
        code_description=code_description,
        code_protocol=code_protocol,
        code_series=code_series,
        code=code,
        code_prefix2=code_prefix2,
        code_prefix3=code_prefix3,
        code_prefix4=code_prefix4,
        code_modality_letter=code_modality_letter,
        normalized_modality=normalized_modality,
        projection=projection,
        body_part_examined_dx=body_part_examined_dx,
        body_part_examined_cbct=body_part_examined_cbct,
    )


# Modality detection


def detect_modality(modality_raw: str) -> str:
    """Use Acquisition Modality as ground truth. Do not infer modality if missing."""
    m = str(modality_raw or "").strip().upper()

    if not m:
        return "UNKNOWN"

    if "CBCT" in m:
        return "CBCT"
    if m == "CT":
        return "CT"
    if m in ("CR", "DX", "DR"):
        return m
    if m in ("XA", "RF"):
        return "XA"
    if m == "MG" or "MAMM" in m:
        return "MG"
    if m in ("NM", "PT", "PET"):
        return m
    if m in ("US", "ULTRASOUND", "UÄ"):
        return "US"
    if m in ("MR", "MRI", "MT"):
        return "MR"
    if "ANGIO" in m or "FLUOR" in m:
        return "XA"
    if "CT" in m:
        return "CT"

    return "UNKNOWN"


# Normalised modalities that deliver an X-ray dose, i.e. the ones for which a
# row's own recorded dose is a real patient exposure regardless of what other
# modality the study text happens to mention.  "UNKNOWN" is deliberately NOT
# here: an unpopulated Acquisition Modality is no evidence of an exposure.
_XRAY_MODALITIES = frozenset({"CT", "CR", "DX", "DR", "XA", "MG", "CBCT"})


def _is_xray_modality(normalized_modality: Optional[str]) -> bool:
    """True if the row's ground-truth modality is an X-ray (dose-bearing) one."""
    return str(normalized_modality or "") in _XRAY_MODALITIES


# Shared generic-unknown / non-anatomical token lists
# Used by all modality resolver functions (resolve_ct_region,
# resolve_xa_rf_region, resolve_radiography_region) to short-circuit
# rows that carry no reliable anatomical information into the appropriate
# unknown bucket (CT_UNLOCALIZED, ANGIO_FLUORO_UNKNOWN, RADIOGRAPHY_OTHER)
# rather than forcing them into a specific anatomical region.

_GENERIC_UNKNOWN_TOKENS = [
    "unknown study description",
    "unknown protocol",
    "tieto puuttuu",
]

_GENERIC_NONANATOMICAL_TOKENS = [
    "täydentävä läpivalaisututkimus",
    "täydentävä lpv",
    "täydentävä varjoainetutkimus",
    "täydent. toimenpide",
    "muu varjoainetutkimus",
    "täydentävä tt",
    "täydent. tt",
    "muu tt",
    "angio low",
]

# XA/RF anatomy-evidence token groups
# Used by resolve_xa_rf_region.  The goal is to require positive
# anatomical evidence before assigning a region, rather than defaulting
# everything unresolved to AbdominalAngiography.

_XA_RF_ABDOMINAL_TOKENS = [
    "vatsa", "abdom",
    "aorta", "aortta", "aortic",
    "viskeraali", "visceral", "mesenter", "keliakia", "celiac",
    "maksa", "hepatic",
    "perna", "splenic",
    "sappi", "biliary",
    "portti", "porta",
    "munuaisvaltimo", "renal artery",
    "alacava", "ivc", "onttolask",
    "kives", "spermatic", "munasarja", "ovari",
    "kohtu", "uterine", "gynek",
    "suolisto", "suolen", "peräsuoli", "rektum",
    "defekografia", "anografia",
    "sirt", "radioyttrium",
    "embol",
    "stenttigraft", "stent graft",
    "lantio",
]

_XA_RF_PERIPHERAL_VASCULAR_TOKENS = [
    "raajavaltimo",
    "reisivaltimo", "polvitaive", "säärivaltimo", "nilkkavalt",
    "femoral", "popliteal",
    "lower limb", "upper limb", "peripheral",
    "alaraaja", "yläraaja",
]

_XA_RF_CAROTID_CEREBRAL_TOKENS = [
    "aivojen", "aivov",
    "kaulavaltimo", "karot", "carot",
    "nikamavaltimo", "vertebral",
    "pään ja kaulan",
    "kaulan alue",
]

_XA_RF_CORONARY_TOKENS = [
    "sepelvaltimo", "coronary", "koron",
    "sydämen angio", "sydän", "sydan",
    "tahdistin", "pacemaker", "defibrill", "icd-",
    "lv kine", "ventrikulog", "ventriculog",
    "ptca", "pci", "angioplast", "rotabl", "atherect",
    "stentin asent", "cardiac",
]

_XA_RF_CYSTOGRAPHY_TOKENS = [
    "virtsatie", "virtsarak", "virtsaputk",
    "uretrokyst", "cystograph", "kysto",
    "miktio", "mictur",
    "nefrostom", "pyelografia",
]

_XA_RF_ERCP_TOKENS = [
    "ercp", "erc ", "sappitie",
    "ptc", "kolangiogr", "cholangiogr",
]

_XA_RF_GI_TOKENS = [
    "ruokatorven", "ruokatorvi", "esophag", "nielemis",
    "mahalaukku", "mahalaukun", "duodenum", "gastroduoden",
    "ohutsuol", "paksusuol", "kolongraf", "passaasi", "pasaasi",
    "peräsuol", "rektum", "anografia", "defekografia",
    "ruuansulatuskanava", "ruoansulatuskanava",
    "cystogastro",
]


def _extract_code_token(text: str) -> Optional[str]:
    if not text:
        return None
    raw_upper = str(text).strip().upper()
    match = re.search(r"\b(?:[A-Z]{2,3}\d[A-Z0-9]{1,4}|[A-Z]{3}\d{2})\b", raw_upper)
    if match:
        return match.group(0).lower()
    for token in ("HR_THX", "THX"):
        if re.search(rf"\b{re.escape(token)}\b", raw_upper):
            return token.lower()
    return None


def _contains_any(value: str, tokens: List[str]) -> bool:
    return any(t in value for t in tokens)




def _infer_projection_from_text(text: str) -> Optional[str]:
    return infer_projection_orientation_from_text(text)


_RE_TRAILING_SIDE_SUFFIX = re.compile(r"\((?:oikea|vasen)\)\s*$", re.IGNORECASE)


def _has_trailing_side_suffix(*texts: str) -> bool:
    """True if any raw text field ends with a trailing "(Oikea)"/"(Vasen)".

    Radiography-specific convention: a laterality marker placed in brackets
    at the very *end* of a description (e.g. "Thoraxin röntgen (Oikea)")
    denotes the lateral/side view actually taken, not merely which side of
    the body was imaged. The same word appearing mid-string (not a trailing
    bracket) is body-side-only and must not be treated as a projection.
    """
    return any(text and _RE_TRAILING_SIDE_SUFFIX.search(text.strip()) for text in texts)


# Post-mortem imaging
# Post-mortem studies carry a real recorded dose — an unconstrained one, since
# there is no patient to optimise for — but it is delivered to a deceased
# patient and so produces no stochastic risk.  They must be excluded from the
# cohort rather than modelled: recorded ages are ages at death, so a perinatal
# post-mortem whole-body CT (observed at 65-92 mSv, ~60x a live paediatric CT)
# is otherwise assigned the model's largest risk coefficient and a full
# lifetime to express it, which is enough to dominate the paediatric cells of
# the Region x Age sweep on its own.
#
# The whole WX9* family is post-mortem and is already excluded by code prefix
# below.  The WX1* post-mortem codes predate it (WX1AD ran 2007-2016) and the
# rest of that family is live-patient imaging — WX1AG/BG/CG/DG whole-body MRI,
# WX1MA slot-scan spine radiography, WX1PA/WX1VA DXA body composition — so
# these four are listed explicitly.  A "wx1" prefix rule would wrongly discard
# live scoliosis slot-scans.
_POST_MORTEM_CODES = frozenset({"wx1ad", "wx1dd", "wx1ed", "wx1gd"})

# Text fallback for rows carrying an ordinary anatomical code but a post-mortem
# protocol (e.g. "AA3AD Kasvojen TT" run under protocol "VAINAJA").
# "vainaja" = the deceased; "kuoleman jälkeinen" = post-mortem.  "kuoleman" is
# matched as a whole word so "aivokuoleman" — brain-death confirmation
# angiography, performed on a living patient — is not caught by it.
_RE_POST_MORTEM = re.compile(r"\bvainaja|\bkuoleman\b")


def _is_non_dose(ev: RegionEvidence) -> bool:
    """Detect records that document no X-ray exposure of a living patient.

    The test is *not* "was this a successful examination" but "did ionising
    radiation reach a living patient".  A record is excluded only when there
    was no patient in the beam (QA/phantom images, specimen radiographs, room
    rental) or the patient was already deceased (post-mortem studies, imaging
    of a dead fetus).

    An **interrupted or failed examination is therefore kept**: it is part of
    the hospital's process and the dose it delivered is real, so excluding it
    would under-count both collective dose and risk.  The archive's
    interruption rows (NCSP ZXF05) carry ordinary diagnostic protocols and
    non-zero dose in 34 of 40 cases; ``zxf`` is instead treated as a generic
    placeholder code (see ``_GENERIC_PLACEHOLDER_CODE_PREFIXES``) so the row
    resolves on its real protocol / Scan Regions evidence and falls back to
    the usual unspecified-region bucket only when nothing else is available.

    ``procedure cancellation`` stays excluded: unlike an interruption it does
    not imply that an exposure was ever made.  No row in the reference archive
    matches it, so if one ever appears carrying dose, revisit this.
    """
    cl = ev.combined_lower
    if "procedure cancellation" in cl:
        return True
    if "quality assurance" in cl:
        return True
    if "laatukuva" in cl:
        return True
    # Angio room rental — not a medical procedure, no patient dose
    if "angiohuoneen vuokra" in cl or "angio room" in cl:
        return True
    # Specimen radiographs (removed tissue, not patient dose)
    if "mammopreparaatti" in cl or "rintarauhaspreparaatti" in cl:
        return True
    # Post-mortem imaging (deceased patient — no stochastic risk)
    if _RE_POST_MORTEM.search(cl):
        return True
    # Ultrasound — no ionising radiation.  Gated on the ground-truth modality
    # (``Acquisition Modality``, never inferred) because naming ultrasound is
    # not the same as being ultrasound: "Rintarauhasen yhdistelmätutkimus
    # (mammografia + ultraääni)", the combined breast examination, names
    # ultrasound but its mammography half is a real exposure with a
    # recorded MGD.  Ultrasound-guided procedures are the same
    # case — the guidance is sonographic, the recorded dose is not.
    _xray = _is_xray_modality(ev.normalized_modality)
    if not _xray and ("ultraääni" in cl or "ultraään" in cl
                      or "ultraaan" in cl or "uä" in cl):
        return True
    code = ev.code
    if code:
        # Post-mortem: the whole wx9* family, the older WX1* codes in
        # _POST_MORTEM_CODES, and mb2* ("MB2AA Kuolleen sikiön rtg" —
        # radiograph of a deceased fetus).
        if code.startswith("mb2") or code.startswith("wx9"):
            return True
        if code in _POST_MORTEM_CODES:
            return True
        if code.startswith("tfp") or code.startswith("tun"):
            return True
        # Specimen radiograph code
        if code.startswith("ha6"):
            return True
        # Ultrasound code letter (5th char = e) — same modality gate as above
        if len(code) >= 5 and code[4] == "e" and not _xray:
            return True
    return False


# Dental detection

_DENTAL_CBCT_CODES = {"eb1ci", "eb1bi", "eb1di", "eb1ei", "eb1fi",
                      "dm1ci", "dm1bi", "dm1ai", "eg1ai"}
_DENTAL_PANO_CODES = {"eb1ha", "eb1ka", "eb1ja", "ed1aa", "eg1aa", "ee1ha", "ee1ja", "ee1ka"}
_DENTAL_INTRAORAL_CODES = {"eb1aa", "eb1sa", "eb1ca"}

# Canonical set of the region labels ``resolve_dental_region`` returns for
# the cross-modality "Dental imaging" subtypes -- i.e. every dental region
# EXCEPT "Dental CBCT", which is a cone-beam CT examination rather than a
# plain-film subtype and is reported under the "CBCT" modality whatever
# archive code it carries (see exam_doses.compute_exam_organ_doses).
# Callers that need to
# recognise these rows and fold them into the "Dental imaging"
# pseudo-modality (while keeping the subtype name as the region label,
# instead of collapsing them all into one flat "Dental imaging" bucket)
# should use this constant rather than a private copy of the literal set,
# so exam_doses.py / vectorized.py / research_questions.py cannot drift.
DENTAL_IMAGING_SUBTYPE_REGIONS: frozenset = frozenset({
    "Dental Panoramic", "Dental_intraoral", "Dental", "Kefalometry",
})
# Finnish anatomy terms in source exam descriptions (hammas/hampa = tooth, leuka = jaw, kartiokeila = cone-beam).
_DENTAL_TEXT_TOKENS = ["dental", "intraoral", "kefalometry", "panoramic",
                       "hammas", "hampa", "opg"]
_DENTAL_ANATOMY_TOKENS = ["hammas", "hampa", "leuka", "leuan", "leukojen",
                          "mandible", "maxilla", "tmj"]


def resolve_dental_region(ev: RegionEvidence) -> Optional[RegionDecision]:
    """Classify dental imaging rows. Returns None if not dental."""
    code = ev.code
    if code:
        if code.startswith("eb1ma"):
            return RegionDecision("Kefalometry", "high", "code", "dental_code:eb1ma", ev.combined)
        if code in _DENTAL_CBCT_CODES:
            return RegionDecision("Dental CBCT", "high", "code", f"dental_code:{code}", ev.combined)
        if code in _DENTAL_PANO_CODES:
            return RegionDecision("Dental Panoramic", "high", "code", f"dental_code:{code}", ev.combined)
        if code in _DENTAL_INTRAORAL_CODES:
            return RegionDecision("Dental_intraoral", "high", "code", f"dental_code:{code}", ev.combined)

    # Text-based dental detection
    if _contains_any(ev.combined_lower, _DENTAL_TEXT_TOKENS):
        cl = ev.combined_lower
        if "cbct" in cl:
            return RegionDecision("Dental CBCT", "medium", "text", "dental_text:cbct", ev.combined)
        if "kartiokeila" in cl and _contains_any(cl, _DENTAL_ANATOMY_TOKENS):
            return RegionDecision("Dental CBCT", "medium", "text", "dental_text:cbct_anatomy", ev.combined)
        if "panoramic" in cl or "opg" in cl:
            return RegionDecision("Dental Panoramic", "medium", "text", "dental_text:panoramic", ev.combined)
        if "intraoral" in cl:
            return RegionDecision("Dental_intraoral", "medium", "text", "dental_text:intraoral", ev.combined)
        if "kefalometry" in cl:
            return RegionDecision("Kefalometry", "medium", "text", "dental_text:kefalometry", ev.combined)
        return RegionDecision("Dental", "low", "text", "dental_text:generic", ev.combined)

    # kartiokeila alone (without dental anatomy) is NOT dental — it's extremity CBCT
    return None


# CT resolver
# Driven entirely by the DICOM "Scan Regions" field -- no NCSP-code or
# free-text fallback (deliberately, for code-independent portability).
#
# Combo-reduction rules (applied to the raw bucket set before naming, so rare
# long-tail combos collapse into more clinically meaningful shared labels):
#   0. Explicit overrides (_CT_EXPLICIT_BUCKET_OVERRIDES): a handful of rare
#      leftover combos that don't fit rules 1-3 below are folded into a
#      sensible neighbour bucket by hand (see the table for exact mappings).
#   1. Trunk consolidation: Chest+(Abdomen or Pelvis) -> Trunk. Neck present
#      alongside a trunk combo generalizes to Head (Neck-only-in-this-context).
#   2. Spine-adjacency: {Spine, X} (exactly 2 buckets) -> {X}, except X==Head,
#      which becomes {Head, Neck} (cervical spine is head/neck-adjacent).
#   3. Extremities-adjacency: {Extremities, X} (exactly 2 buckets) -> {Extremities}.
# Rules are applied in a loop until a fixed point, so cascading reductions on
# 3+ bucket sets resolve correctly. See _reduce_ct_bucket_set.

# Rare leftover combos folded into a neighbour bucket by hand (checked first,
# before the general rules, in _reduce_ct_bucket_set). Note: {Chest, Pelvis}
# and {Head, Chest, Pelvis} are NOT listed here -- rule (1) below already
# covers them (Chest + Pelvis triggers trunk consolidation same as Chest +
# Abdomen).
_CT_EXPLICIT_BUCKET_OVERRIDES = {
    # None of the combos below have Chest jointly with Abdomen and/or
    # Pelvis (that combination is genuine trunk coverage, handled by rule
    # (1) below and labelled "Trunk"). Head/Neck combined with only Chest
    # OR only Abdomen/Pelvis (never both) is a different, rarer category --
    # an extended head/neck-to-trunk scan of uncertain exact extent -- so
    # all such combos are pooled into one shared {Head, Abdomen} bucket
    # instead of being split into several anatomically-arbitrary tiny ones.
    frozenset({CT_HEAD, CT_CHEST}): frozenset({CT_HEAD, CT_DIGESTIVE}),
    frozenset({CT_HEAD, CT_NECK, CT_DIGESTIVE}): frozenset({CT_HEAD, CT_DIGESTIVE}),
    frozenset({CT_NECK, CT_DIGESTIVE}): frozenset({CT_HEAD, CT_DIGESTIVE}),
    frozenset({CT_NECK, CT_CHEST}): frozenset({CT_HEAD, CT_DIGESTIVE}),
    frozenset({CT_HEAD, CT_PELVIS}): frozenset({CT_HEAD, CT_DIGESTIVE}),
    frozenset({CT_HEAD, CT_NECK, CT_CHEST}): frozenset({CT_HEAD, CT_DIGESTIVE}),
    frozenset({CT_DIGESTIVE, CT_PELVIS}): frozenset({CT_DIGESTIVE}),
}


def _reduce_ct_bucket_set(buckets: frozenset) -> frozenset:
    """Apply the CT combo-reduction rules (overrides + trunk/spine/extremities)."""
    while True:
        if buckets in _CT_EXPLICIT_BUCKET_OVERRIDES:
            buckets = _CT_EXPLICIT_BUCKET_OVERRIDES[buckets]
            continue

        reduced = set(buckets)

        # (1) Trunk consolidation: Chest+(Abdomen or Pelvis) -> Trunk; Neck -> Head.
        if CT_CHEST in reduced and (CT_DIGESTIVE in reduced or CT_PELVIS in reduced):
            reduced -= {CT_CHEST, CT_DIGESTIVE, CT_PELVIS}
            reduced.add(CT_TRUNK)
            if CT_NECK in reduced:
                reduced.discard(CT_NECK)
                reduced.add(CT_HEAD)

        # (2) Spine-adjacency: exactly {Spine, X} -> {X} (Head -> {Head, Neck}).
        # Deliberately drops Spine rather than the other way around (e.g. NOT
        # {Pelvis, Spine} -> {Spine}): region is a reporting dimension only
        # (organ doses come from the Excel per-organ columns regardless), and
        # a Pelvis+L-spine exam irradiates bladder/rectum/gonads far more than
        # an isolated lumbar-spine survey -- folding it into "Pelvis" reflects
        # its real organ-dose profile, whereas folding into "Spine" would
        # dilute that bucket's true isolated-low-dose character.
        if CT_SPINE in reduced and len(reduced) == 2:
            other = next(iter(reduced - {CT_SPINE}))
            reduced = {CT_HEAD, CT_NECK} if other == CT_HEAD else {other}

        # (3) Extremities-adjacency: exactly {Extremities, X} -> {Extremities}.
        if CT_EXTREMITIES in reduced and len(reduced) == 2:
            reduced = {CT_EXTREMITIES}

        reduced = frozenset(reduced)
        if reduced == buckets:
            return reduced
        buckets = reduced


def resolve_ct_region_from_scan_regions(scan_regions_raw: str) -> RegionDecision:
    """Classify a CT row from the DICOM ``Scan Regions`` field alone.

    Splits the raw value on ``,``/``/``/``;``/``+`` into atomic vendor labels,
    maps each to a canonical bucket via ``_CT_SCAN_REGION_ATOM_MAP``, dedupes,
    and then applies the trunk/spine/extremities adjacency-reduction rules
    (see above). Zero recognised atoms -> Unknown/unspecified region; exactly one distinct
    bucket -> that bucket; two or more distinct buckets -> a combo label
    formed by joining those buckets, in canonical head-to-foot order
    (``_CT_BUCKET_ORDER``), e.g. "Chest, Abdomen". There is intentionally no
    NCSP-code or free-text fallback for CT.

    If some (but not all) segments fail to match ``_CT_SCAN_REGION_ATOM_MAP``
    (e.g. an unrecognised vendor vocabulary word from a different
    institution/scanner), confidence is downgraded to "medium" and the
    unmatched segment(s) are recorded in the rule string instead of being
    silently dropped -- this only matters for future/other datasets; it does
    not change classification for any currently-recognised vocabulary.
    """
    raw = str(scan_regions_raw or "").strip()
    if not raw:
        return RegionDecision(CT_UNLOCALIZED, "low", "fallback", "ct:scanregions:blank", raw)

    segments = [s.strip().lower() for s in _RE_SCAN_REGION_SPLIT.split(raw) if s.strip()]
    unmatched = [s for s in segments if s not in _CT_SCAN_REGION_ATOM_MAP]
    buckets = frozenset(
        _CT_SCAN_REGION_ATOM_MAP[s] for s in segments if s in _CT_SCAN_REGION_ATOM_MAP
    )

    if not buckets:
        return RegionDecision(CT_UNLOCALIZED, "low", "text", "ct:scanregions:unrecognized", raw)

    buckets = _reduce_ct_bucket_set(buckets)

    # Partial match: some segments matched, but at least one didn't. Don't
    # silently drop the unmatched segment(s) -- downgrade confidence and
    # record them, so unrecognised vendor vocabulary from other datasets is
    # auditable instead of invisible.
    confidence = "medium" if unmatched else "high"
    unmatched_note = f":unmatched={'|'.join(unmatched)}" if unmatched else ""

    if len(buckets) == 1:
        (bucket,) = buckets
        return RegionDecision(bucket, confidence, "text", f"ct:scanregions:{bucket}{unmatched_note}", raw)
    ordered = [b for b in _CT_BUCKET_ORDER if b in buckets]
    combo = ", ".join(ordered)
    return RegionDecision(combo, confidence, "text", f"ct:scanregions:multi{unmatched_note}", raw)


def resolve_ct_region(ev: RegionEvidence) -> RegionDecision:
    """Classify a CT row from ``ev.scan_regions`` alone (see module docstring)."""
    return resolve_ct_region_from_scan_regions(ev.scan_regions)


# Rare-combo folding (reporting layer)

def is_combo_region(region: str) -> bool:
    """True if *region* is a multi-region combination label (e.g. ``"Chest, Abdomen"``).

    Combination labels are comma-joined canonical bucket names produced by
    :func:`resolve_ct_region_from_scan_regions`.  Single buckets (``"Head"``,
    ``"Trunk"``, …) and special categories (``"Non-Dose (QA, Cancellation etc.)"``,
    ``CT_UNLOCALIZED``) are NOT combos.
    """
    r = str(region or "")
    if not r:
        return False
    # "Non-Dose (QA, Cancellation etc.)" contains a comma but is not a combo.
    if r.startswith("Non-Dose"):
        return False
    return ", " in r


def fold_rare_region_combos(
    region_counts: dict,
    threshold: int = RARE_COMBO_THRESHOLD,
) -> dict:
    """Fold rare multi-region combos into ``CT_OTHER_COMBINATIONS``.

    Parameters
    ----------
    region_counts
        ``{region_label: count}`` mapping (e.g. from
        :func:`analysis.region_by_modality` or the sweep's per-region totals).
    threshold
        Combos with fewer than this many scans are folded.  Default
        :data:`RARE_COMBO_THRESHOLD` (30).

    Returns
    -------
    dict
        New ``{region_label: count}`` with rare combos merged into
        ``CT_OTHER_COMBINATIONS`` (summed).  Single regions and combos at or
        above the threshold are kept unchanged.  If no combos are folded, the
        input is returned as-is (no new key added).
    """
    if not region_counts:
        return dict(region_counts) if region_counts else {}

    folded = {}
    other_total = 0
    for reg, cnt in region_counts.items():
        if is_combo_region(reg) and cnt < threshold:
            other_total += cnt
        else:
            folded[reg] = cnt
    if other_total > 0:
        folded[CT_OTHER_COMBINATIONS] = folded.get(CT_OTHER_COMBINATIONS, 0) + other_total
    return folded


# XA/RF resolver

_XA_STRONG_ANATOMY_TOKENS = (
    _XA_RF_ERCP_TOKENS + _XA_RF_CYSTOGRAPHY_TOKENS + _XA_RF_CORONARY_TOKENS +
    _XA_RF_CAROTID_CEREBRAL_TOKENS + _XA_RF_PERIPHERAL_VASCULAR_TOKENS +
    _XA_RF_ABDOMINAL_TOKENS + _XA_RF_GI_TOKENS
)


def classify_xa_by_code(code: Optional[str], cl: str = "") -> Optional[str]:
    """Classify an XA/RF row from its NCSP code prefix. Returns a region string or None.

    If cl (combined lower text) is provided, generic vascular PG* codes use
    anatomy text to determine the correct vascular region.
    """
    if not code:
        return None
    code = code.lower()

    # Head/neck/cerebral arteries and veins
    # AA7 = CSF shunt / ventricular hardware contrast studies (sunttiröntgen)
    # — head/neck anatomy, grouped with carotid/cerebral.
    if code.startswith(("pa", "ph4", "aa7")):
        return "CarotidAngiography"

    # Coronary / cardiac catheter procedures (official NCSP)
    if code.startswith(("fa", "fk", "fl", "fm", "fn", "fe1", "fe2")):
        return "Coronary"

    # Local cath-lab / cardiac procedure shorthand codes observed in data.
    # These are consistently used in coronary/cath-lab context.
    if code.startswith(("fme", "ffm", "fpe", "ffc", "fke", "fkc", "tfe")):
        return "Coronary"

    # Pulmonary artery / pulmonary embolism.
    # FB1ST = catheter treatment of pulmonary embolism (great vessel) → AbdominalAngiography
    if code == "fb1st":
        return "AbdominalAngiography"
    # Other FB codes: no dedicated ThoracicAngiography bucket; route to
    # UnknownAngiography to avoid anatomical inaccuracy.
    if code.startswith("fb"):
        return ANGIO_FLUORO_UNKNOWN

    # ERCP / biliary
    if code.startswith(("jk3", "jk4", "jk5", "jl5")):
        return "ERCP"

    # Urinary contrast / cystography / nephrostomy / urinary interventions
    # KA6 = kidney interventions (e.g. renal cyst sclerosis)
    if code.startswith(("ka3", "ka6", "kc1", "kd1", "kh1")):
        return "Cystography"

    # Spinal-canal vascular procedures (spinal angiography / embolization).
    # No dedicated spine-vascular bucket exists; these are consistent with
    # the other spine/nerve-root interventional codes already routed to
    # MSKIntervention, so group them there rather than mislabel them
    # AbdominalAngiography via the broader "pc" prefix below.
    if code.startswith("pc6"):
        return "MSKIntervention"

    # Peripheral upper/lower limb arteries and veins
    if code.startswith(("pb", "pd3", "pd4", "pe", "pf", "ph1", "ph2", "ph9")):
        return "PeripheralAngiography"

    # Abdominal / visceral / aortic / pelvic / caval / gonadal vessels
    if code.startswith(("pc", "pd1", "pd2", "pd5", "pd6", "pd7", "ph3", "ph5", "ph6", "ph7")):
        return "AbdominalAngiography"

    # Penile vasculature (cavernosography) — no dedicated genital bucket;
    # closest anatomical grouping is pelvic/abdominal.
    if code.startswith("ph8"):
        return "AbdominalAngiography"

    # AV fistula / dialysis-access port procedures (PG5*) are almost always
    # in the upper or lower limb → PeripheralAngiography.
    if code.startswith("pg5"):
        return "PeripheralAngiography"

    # Generic vascular codes PG and lymphatic codes PJ are not anatomically
    # specific on their own. Use anatomy text if available; otherwise stay
    # unknown rather than guess.
    if code.startswith(("pg1", "pg2", "pg3", "pg6", "wg6", "pj")):
        if cl and _contains_any(cl, _XA_RF_PERIPHERAL_VASCULAR_TOKENS):
            return "PeripheralAngiography"
        if cl and _contains_any(cl, _XA_RF_CAROTID_CEREBRAL_TOKENS):
            return "CarotidAngiography"
        if cl and _contains_any(cl, _XA_RF_ABDOMINAL_TOKENS):
            return "AbdominalAngiography"
        return ANGIO_FLUORO_OTHER

    # GI contrast fluoroscopy studies (esophagram, swallow study, gastro-
    # duodenal, small bowel/colon passage, defecography) are not angiography
    # but are performed under fluoroscopy — keep them out of
    # UnknownAngiography with a dedicated bucket. QX2CB (fistulography) is
    # GI-adjacent (enterocutaneous/ostomy tract); other QX2 codes are
    # generic soft-tissue procedures with no GI anatomy and are left
    # unmapped here (anatomy text, if any, is checked afterwards).
    if code.startswith(("jc2", "jd1", "jf1", "jf2", "jg1", "jh1", "jx1")) or code in ("qx2cb", "eh1jb"):
        return "GIFluoroscopy"

    # MSK/spine guided procedures (nerve root blocks, bone biopsies, facet
    # injections, vertebroplasty, SI joint injections, hip joint injections)
    # are performed under fluoroscopy but are not angiography — route to a
    # dedicated MSKIntervention bucket.
    if code.startswith(("ab2", "ab4", "ab5", "ab6", "ac1",
                        "na6", "na7", "na8",
                        "nb1", "nb2", "nf1", "nf2",
                        "nk1", "nk2", "nk4", "nk5", "nk6", "nk7")):
        return "MSKIntervention"

    return None


def resolve_xa_rf_region(ev: RegionEvidence) -> RegionDecision:
    """Classify an XA/RF row. Code-first; text fallback only when code is absent or generic.

    Generic-unknown/non-anatomical text checks run against ``combined_lower``,
    which includes benign system placeholders (e.g. "Unknown Protocol Name"
    recorded when a field simply wasn't populated). Those placeholders must
    never override a real NCSP code found elsewhere in the row, so all
    code-based classification happens before any generic-text short-circuit.
    """
    cl = ev.combined_lower
    code = ev.code

    # (1) Generic XX codes → Other angiography unless strong anatomy present
    if code and code.startswith("xx") and not _contains_any(cl, _XA_STRONG_ANATOMY_TOKENS):
        return RegionDecision(ANGIO_FLUORO_OTHER, "low", "code", f"xa:generic_xx:{code}", ev.combined)

    # (2) Code-prefix classification (primary) — takes priority over any
    # generic-unknown placeholder text elsewhere in the row.
    code_region = classify_xa_by_code(code, cl)
    if code_region:
        return RegionDecision(code_region, "high", "code", f"xa:prefix:{code}", ev.combined)

    # (3) Text fallback — anatomy tokens take priority over generic-unknown
    # placeholder text. A row with "Tahdistin" (pacemaker) in Description but
    # "Unknown Protocol Name" in Acquisition Protocol must classify as
    # Coronary, not UnknownAngiography.
    if _contains_any(cl, _XA_RF_ERCP_TOKENS):
        return RegionDecision("ERCP", "medium", "text", "xa:text:ercp", ev.combined)
    if _contains_any(cl, _XA_RF_CYSTOGRAPHY_TOKENS):
        return RegionDecision("Cystography", "medium", "text", "xa:text:cystography", ev.combined)
    if _contains_any(cl, _XA_RF_CORONARY_TOKENS):
        return RegionDecision("Coronary", "medium", "text", "xa:text:coronary", ev.combined)
    if _contains_any(cl, _XA_RF_CAROTID_CEREBRAL_TOKENS):
        return RegionDecision("CarotidAngiography", "medium", "text", "xa:text:carotid", ev.combined)
    if _contains_any(cl, _XA_RF_PERIPHERAL_VASCULAR_TOKENS):
        return RegionDecision("PeripheralAngiography", "medium", "text", "xa:text:peripheral", ev.combined)
    if _contains_any(cl, _XA_RF_ABDOMINAL_TOKENS):
        return RegionDecision("AbdominalAngiography", "medium", "text", "xa:text:abdominal", ev.combined)
    if _contains_any(cl, _XA_RF_GI_TOKENS):
        return RegionDecision("GIFluoroscopy", "medium", "text", "xa:text:gi", ev.combined)

    # (4) Generic-unknown — only after code AND text gave no anatomy
    if _contains_any(cl, _GENERIC_UNKNOWN_TOKENS):
        return RegionDecision(ANGIO_FLUORO_UNKNOWN, "low", "text", "xa:generic_unknown", ev.combined)

    # (5) Generic non-anatomical (täydentävä / muu codes) — recognised but
    # not anatomically localised by design → Other angiography
    if _contains_any(cl, _GENERIC_NONANATOMICAL_TOKENS):
        return RegionDecision(ANGIO_FLUORO_OTHER, "low", "text", "xa:generic_nonanatomical", ev.combined)

    # (6) No anatomical evidence
    return RegionDecision(ANGIO_FLUORO_UNKNOWN, "low", "fallback", "xa:no_evidence", ev.combined)


# Radiography resolver
#
# Primary source: the DICOM "Body Part Examined (DX)" tag, present on ~99%
# of CR/DX rows (measured on the full Radimetrics export). Values are the
# standard DICOM body-part code set (e.g. "CHEST", "LOW_EXM", "TSPINE") --
# a different, coarser vocabulary than NCSP codes and free text (e.g.
# "CHEST" alone covers what NCSP splits across several GA/GD1 codes).
# _DX_BODY_PART_BASE_REGION_MAP maps
# every value observed in the archive (normalised: stripped, upper-cased,
# whitespace-collapsed) onto the same base-region vocabulary the
# base -> Table-4.1-exam block below already understands, so that block is
# unchanged. Spelling/formatting variants of the same anatomy seen in the
# data are folded onto one canonical base (e.g. "LSPINE" / "L SPINE" /
# "LSSPINE" -> "Lumbar spine"). Non-anatomical placeholder values
# ("REPORT", "TEST", "PATTERN", "S") and values with no radiography exam region
# ("WHOLEBODY") are deliberately left out of the map -- see
# resolve_radiography_region for what happens when the map has no entry.
_DX_BODY_PART_BASE_REGION_MAP = {
    "CHEST": "Chest",
    "CLAVICLE": "Chest",
    "SCAPULA": "Chest",
    "STERNUM": "Chest",
    "RIB": "Chest",
    "RIBS": "Chest",
    "SCJOINT": "Chest",
    "ESOPHAGUS": "Chest",

    "SKULL": "Skull",
    "HEAD": "Skull",
    "CRANIUM": "Skull",
    "BRAIN": "Skull",
    "JAW": "Skull",
    "NECK": "Skull",
    "SINUSES": "Skull",
    "PNS": "Skull",
    "FACIAL BONES": "Skull",
    "ORIBIT": "Skull",
    "OPTIC CANAL": "Skull",

    "ABDOMEN": "Abdomen",
    "KIDNEY": "Abdomen",
    "CHESTABDOMEN": "Abdomen",
    "SMALL BOWEL": "Abdomen",
    "SMALL INTESTINE": "Abdomen",

    "PELVIS": "Pelvis",
    "HIP": "Pelvis",
    "FEMUR": "Pelvis",
    "SACROILIACJOINT": "Pelvis",
    "COCCYX": "Pelvis",

    "CSPINE": "Cervical spine",
    "C SPINE": "Cervical spine",
    "ODONTOID": "Cervical spine",

    "TSPINE": "Thoracic spine",
    "T SPINE": "Thoracic spine",

    "LSPINE": "Lumbar spine",
    "L SPINE": "Lumbar spine",
    "LSSPINE": "Lumbar spine",

    "THORACOLUMBAR SP": "Thoraco-lumbar spine",

    "KNEE": "Extremity",
    "HAND": "Extremity",
    "FOOT": "Extremity",
    "LEG": "Extremity",
    "SHOULDER": "Extremity",
    "ANKLE": "Extremity",
    "WRIST": "Extremity",
    "ELBOW": "Extremity",
    "LOW_EXM": "Extremity",
    "UP_EXM": "Extremity",
    "FINGER": "Extremity",
    "ARM": "Extremity",
    "PATELLA": "Extremity",
    "TIBIA FIBULA": "Extremity",
    "TIBIA": "Extremity",
    "EXTREMITY": "Extremity",
    "FOREARM": "Extremity",
    "CALCANEUS": "Extremity",
    "TOES": "Extremity",
    "TOE": "Extremity",
    "HUMERUS": "Extremity",
    "RAAJAT": "Extremity",  # Finnish: "limbs/extremities"
}

# Body-part values that mean "spine" without specifying which level. There is
# no Body Part Examined equivalent of the old NCSP scoliosis code (see the
# "na7" override in resolve_radiography_region below), so a generic spine
# tag is disambiguated from the row's own text where possible and otherwise
# defaults to Lumbar spine (the most commonly ordered plain "spine" film).
_DX_GENERIC_SPINE_BODY_PARTS = frozenset({
    "SPINE", "SSPINE", "CTSPINE",
    "SELKARANKA",  # Finnish: "spine/backbone"
})


def _normalize_body_part_value(raw: str) -> str:
    return _RE_WS.sub(" ", str(raw or "").strip().upper())


def _classify_dx_body_part_base(ev: "RegionEvidence") -> Optional[str]:
    """Map ``ev.body_part_examined_dx`` to a base region, or None if empty/unmapped."""
    bp = _normalize_body_part_value(ev.body_part_examined_dx)
    if not bp:
        return None
    if bp in _DX_GENERIC_SPINE_BODY_PARTS:
        cl = ev.combined_lower
        if _contains_any(cl, ["thoracic", "rintaran"]):
            return "Thoracic spine"
        if _contains_any(cl, ["cervical", "kaularan"]):
            return "Cervical spine"
        return "Lumbar spine"
    return _DX_BODY_PART_BASE_REGION_MAP.get(bp)


def resolve_radiography_region(ev: RegionEvidence) -> RegionDecision:
    """Classify a radiography (CR/DX/DR) row into a radiography exam region.

    Primary source: ``Body Part Examined (DX)`` (see
    ``_DX_BODY_PART_BASE_REGION_MAP``). An empty tag, or one carrying a
    non-anatomical placeholder value (e.g. "REPORT", "TEST") or a value
    with no radiography exam region ("WHOLEBODY"), resolves to ``CT_UNLOCALIZED``
    ("Unknown/unspecified region") -- the same label CT uses for blank
    ``Scan Regions``.  Unlike CT, the row then has no organ-dose
    conversion (``exam_doses._radiography_doses`` returns no organ doses /
    effective dose for it) and is excluded from the analysed cohort
    (listed in the By Body Region tab).

    One NCSP-code override survives the switch to the new primary source:
    the "na7" scoliosis family has no Body Part Examined counterpart
    (scoliosis exams are tagged with a generic spine value like any other
    spine film), but full-spine slot imaging has a genuinely different
    dose profile from a conventional spine radiograph, so it is still
    detected from the code.
    """
    cl = ev.combined_lower

    base_source = "bodypart"
    if ev.code and ev.code.startswith("na7"):
        base = "Scoliosis"
        base_source = "code_na7"
    else:
        base = _classify_dx_body_part_base(ev)

    if not base:
        return RegionDecision(
            CT_UNLOCALIZED, "low", "fallback",
            "rg:bodypart_dx:empty_or_unmapped", ev.body_part_examined_dx or ev.combined,
        )

    # Determine projection — priority: code-based (the NCSP code itself
    # unambiguously encodes projection for GD1E*/GD1Q*/GD1U*, so it must not
    # be silently overridden by a generic/default metadata field), then
    # explicit metadata, then a trailing "(Oikea)"/"(Vasen)" bracket (a
    # lateral-view marker, as opposed to the same word appearing mid-string
    # which only denotes body side), then generic free-text inference.
    proj = (
        infer_projection_from_code(ev.code)
        or ev.projection
        or ("Lat" if _has_trailing_side_suffix(
                ev.description_raw, ev.protocol_raw, ev.series_raw, ev.procedure_raw
            ) else None)
        or _infer_projection_from_text(ev.combined)
    )
    if proj in ("LLat", "RLat"):
        proj = "Lat"

    if base == "Lumbar spine" and ("oblique" in cl or " obl" in cl):
        proj = "RObl"

    # Map base + projection to final region
    if base == "Thoraco-lumbar spine":
        region = "Thoraco-lumbar spine Lat"
    elif base == "Extremity":
        region = "Extremity"
    elif base == "Pelvis":
        region = "Pelvis AP"
    elif base == "Abdomen":
        region = "Abdomen AP"
    elif base == "Chest":
        if proj == "PA":
            region = "Chest PA"
        elif proj == "Lat":
            region = "Chest Lat"
        else:
            region = "Chest AP"
    elif base == "Lumbar spine":
        if proj == "RObl":
            region = "Lumbar spine oblique"
        elif proj == "Lat":
            region = "Lumbar spine Lat"
        else:
            region = "Lumbar spine AP"
    elif base == "Cervical spine":
        region = "Cervical spine Lat" if proj == "Lat" else "Cervical spine AP"
    elif base == "Thoracic spine":
        region = "Thoracic spine Lat" if proj == "Lat" else "Thoracic spine AP"
    elif base == "Scoliosis":
        if proj == "PA":
            region = "Scoliosis PA"
        elif proj == "Lat":
            region = "Scoliosis Lat"
        else:
            region = "Scoliosis AP"
    elif base == "Skull":
        region = "Skull Lat" if proj == "Lat" else "Skull PA"
    else:
        return RegionDecision(RADIOGRAPHY_OTHER, "low", "fallback", "rg:unmapped_base", ev.combined)

    # Enforce allowed region set
    if region in _radiography_exam_regions():
        return RegionDecision(region, "medium", "text", f"rg:{base_source}:{base}", ev.combined)

    # Try fallback mapping
    mapped = _RADIOGRAPHY_BASE_FALLBACK.get(str(region or "").lower())
    if mapped:
        return RegionDecision(mapped, "low", "prefix", f"rg:fallback_map:{region}", ev.combined)

    return RegionDecision(RADIOGRAPHY_OTHER, "low", "fallback", "rg:not_in_exam_regions", ev.combined)


# Nuclear resolver

def resolve_nuclear_region(ev: RegionEvidence) -> RegionDecision:
    """All NM/PT/PET rows resolve to Nuclear/hybrid."""
    return RegionDecision(NUCLEAR_REGION, "high", "modality", "nuclear:always", ev.combined)


# Mammography resolver

def resolve_mammography_region(ev: RegionEvidence) -> RegionDecision:
    """All MG rows resolve to Breast."""
    return RegionDecision("Breast", "high", "modality", "mg:always", ev.combined)


# CBCT resolver
#
# Primary source: the DICOM "Body Part Examined (CBCT)" tag (~99% filled in
# the archive). "HEAD" and "SKULL" both route to Dental CBCT -- the unit is
# a dental cone-beam scanner, so skull-base/sinus/facial-bone CBCT is
# imaged on it too. Dental CBCT is the only CBCT region with a real dose
# model (see
# exam_doses.py), so every other non-empty value is still reported under
# its own literal body-part label (e.g. "Ankle", "Wrist") for anatomical
# bookkeeping, but carries no dose model and therefore no risk; the actual
# body part is preserved as its own label rather than folded into a shared
# catch-all bucket. An empty tag
# resolves to CT_UNLOCALIZED ("Unknown/unspecified region"), the same
# convention used for radiography and CT.
_CBCT_DENTAL_BODY_PARTS = frozenset({"HEAD", "SKULL"})


def resolve_cbct_region(ev: RegionEvidence) -> RegionDecision:
    """Classify a CBCT row from Body Part Examined (CBCT).

    HEAD/SKULL -> "Dental CBCT" (real dose model). Any other non-empty
    value -> that value as a literal region label (no dose model, so no
    risk). Empty -> CT_UNLOCALIZED.
    """
    bp = _normalize_body_part_value(ev.body_part_examined_cbct)
    if bp in _CBCT_DENTAL_BODY_PARTS:
        return RegionDecision("Dental CBCT", "high", "text", f"cbct:bodypart:{bp}", ev.combined)
    if bp:
        return RegionDecision(bp.title(), "medium", "text", f"cbct:bodypart:{bp}", ev.combined)
    return RegionDecision(CT_UNLOCALIZED, "low", "fallback", "cbct:bodypart:empty", ev.combined)


# Main dispatcher

def _resolve_region_internal(row) -> RegionDecision:
    """Build evidence, dispatch to modality-specific resolver, apply overrides."""
    ev = build_evidence(row)

    # (1) Non-dose detection (highest priority, modality-independent)
    if _is_non_dose(ev):
        return RegionDecision(
            "Non-Dose (QA, Cancellation etc.)", "high", "override",
            "non_dose:detected", ev.combined,
        )

    # (2) Early dental detection (modality-independent)
    dental = resolve_dental_region(ev)
    if dental:
        return dental

    # (3) Modality-specific dispatch
    modality = ev.normalized_modality
    if modality == "CT":
        decision = resolve_ct_region(ev)
    elif modality == "XA":
        decision = resolve_xa_rf_region(ev)
    elif modality in RADIOGRAPHY_MODALITIES:
        decision = resolve_radiography_region(ev)
    elif modality in ("NM", "PT", "PET"):
        decision = resolve_nuclear_region(ev)
    elif modality == "MG":
        decision = resolve_mammography_region(ev)
    elif modality == "CBCT":
        decision = resolve_cbct_region(ev)
    else:
        decision = RegionDecision(
            "Unknown Modality", "low", "fallback",
            "modality:missing_or_unrecognized", ev.combined,
        )

    # (4) XA/RF region guard: coerce incompatible labels to UnknownAngiography
    if modality == "XA" and decision.region not in ANGIO_FLUORO_REGIONS and decision.region != "Non-Dose (QA, Cancellation etc.)":
        decision = RegionDecision(
            ANGIO_FLUORO_UNKNOWN, "low", "override",
            "xa:coerce_incompatible_region", ev.combined,
        )

    # (5) Late dental override: if modality-specific logic produced a generic
    # radiography or CT region but the text is dental, override.
    if decision.region not in DENTAL_ALLOWED_REGIONS and decision.region != "Non-Dose (QA, Cancellation etc.)":
        if _contains_any(ev.combined_lower, _DENTAL_TEXT_TOKENS):
            dental = resolve_dental_region(ev)
            if dental:
                decision = dental

    # (6) Unknown-description registration (reported via get_unknown_descriptions for the run's unknown-regions summary)
    if decision.region in ("Unknown Modality", "General"):
        unknown_text = ev.description or ev.protocol
        if unknown_text:
            unknown_key = str(unknown_text).strip().lower()
            if unknown_key and unknown_key not in _UNKNOWN_DESCRIPTION_CACHE:
                _UNKNOWN_DESCRIPTION_CACHE.add(unknown_key)
                _UNKNOWN_DESCRIPTIONS.append((unknown_text, str(modality or "Unknown")))

    return decision


# Public API


def resolve_region(row):
    """Return the resolved region string for ``row``.

    Memoised on the tuple of fields the resolver (and its downstream
    projection-orientation helper) actually read from ``row``.  Typical
    hospital datasets have only hundreds of distinct key tuples across
    millions of rows, so the cache converts an O(N × 100-branch) string
    scan into an O(N) dict lookup.

    The cache is cleared in ``clear_unknown_descriptions`` so the
    unknown-descriptions side effect (registered on cache miss only)
    stays in sync with each simulation run.
    """
    key = tuple(str(row.get(f, "") or "") for f in _RESOLVE_REGION_KEY_FIELDS)
    cached = _RESOLVE_REGION_CACHE.get(key)
    if cached is not None:
        return cached
    decision = _resolve_region_internal(row)
    region = decision.region
    _RESOLVE_REGION_CACHE[key] = region
    return region
