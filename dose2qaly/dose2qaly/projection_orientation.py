"""Projection / view orientation parsing for X-ray imaging.

* ``"AP"``   - anterior-posterior (beam enters the front)
* ``"PA"``   - posterior-anterior (beam enters the back)
* ``"Lat"``  - lateral (side), with ``"LLat"`` / ``"RLat"`` for left / right
* ``"RObl"`` - oblique (angled) view

Two strategies are provided and combined:

* :func:`normalize_projection_orientation` - exact match of a clean code
  (e.g. ``"Lateral"`` -> ``"Lat"``).
* :func:`infer_projection_orientation_from_text` - keyword scan of free text,
  including Finnish synonyms (``"Etukuva"``, ``"Sivu"``, ``"Viistokuva"``...).

:func:`get_projection_orientation` is the main entry point used by
:mod:`exam_doses`; it scans several row fields plus any explicit values and
returns the first recognised projection.  Results are memoised with
``lru_cache`` because the same handful of projection strings recur across
millions of exam rows.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Optional, Tuple
import re

# Row columns scanned (in priority order) for a projection / view label.
# Candidates are tried in order; the first recognised projection wins.
_PROJECTION_FIELDS: Tuple[str, ...] = (
    "Projection",
    "Position",
    "View",
    "Description",
    "Protocol",
    "Exam Type",
)


# Precompile regex patterns once so _normalize_impl / _infer_impl do not
# rebuild them on every call (meaningful saving when these are invoked
# millions of times per simulation from compute_exam_organ_doses).
_RE_ROBL    = re.compile(r"\bROBL\b")
_RE_OBL     = re.compile(r"\bOBL\b")
_RE_LLAT    = re.compile(r"\bLLAT\b")
_RE_RLAT    = re.compile(r"\bRLAT\b")
_RE_LAT     = re.compile(r"\bLAT\b")
_RE_PA      = re.compile(r"\bPA\b")
_RE_AP      = re.compile(r"\bAP\b")
_RE_FRONTAL = re.compile(r"\bFRONTAL\b")
_RE_WS      = re.compile(r"\s+")


@lru_cache(maxsize=4096)
def _normalize_impl(raw: str) -> Optional[str]:
    """String-only core used by normalize_projection_orientation.

    Separated so a module-level lru_cache can memoise repeat inputs
    (typical hospital data: <100 distinct projection strings across
    millions of exam rows).
    """
    p = _RE_WS.sub("", raw.strip().upper())
    p = p.replace("-", "").replace("_", "")
    if p in {"PA", "POSTERIORANTERIOR"}:
        return "PA"
    if p in {"AP", "ANTERIORPOSTERIOR"}:
        return "AP"
    if p in {"LAT", "LATERAL"}:
        return "Lat"
    if p in {"LLAT", "LEFTLATERAL", "LLATERAL"}:
        return "LLat"
    if p in {"RLAT", "RIGHTLATERAL", "RLATERAL"}:
        return "RLat"
    if p in {"ROBL", "RIGHTOBLIQUE", "ROBLQ", "OBL", "OBLIQUE"}:
        return "RObl"
    return None


def normalize_projection_orientation(raw: Optional[str]) -> Optional[str]:
    """Normalise a free-text projection string to a canonical code.

    Examples: ``"AP"`` → ``"AP"``, ``"PA"`` → ``"PA"``,
    ``"Lateral"`` → ``"Lat"``, ``"Oblique"`` → ``"RObl"``.
    Returns ``None`` when no known projection is found.
    """
    if raw is None:
        return None
    return _normalize_impl(str(raw))


@lru_cache(maxsize=4096)
def _infer_impl(t: str) -> Optional[str]:
    """String-only core used by infer_projection_orientation_from_text."""
    if not t:
        return None
    # Oblique
    if _RE_ROBL.search(t) or "RIGHT OBLIQUE" in t or _RE_OBL.search(t) or "OBLIQUE" in t:
        return "RObl"
    if "VIISTOKUVA" in t or "VIISTO" in t or "VINO" in t:
        return "RObl"
    # Lateral — use explicit projection words only, not laterality (Oikea/Vasen)
    if _RE_LLAT.search(t) or "LEFT LATERAL" in t:
        return "LLat"
    if _RE_RLAT.search(t) or "RIGHT LATERAL" in t:
        return "RLat"
    if _RE_LAT.search(t) or "LATERAL" in t or "SIVUKUVA" in t or "KYLKIKUVA" in t:
        return "Lat"
    # AP indicators — bedside, portable, supine
    if "MAKUU" in t or "BEDSIDE" in t or "PORTABLE" in t or "ULKOPUOLELLA" in t:
        return "AP"
    # PA / AP explicit
    if _RE_PA.search(t) or "POSTERIOR-ANTERIOR" in t or "TAKAKUVA" in t or "TAKAPROJEKTIO" in t:
        return "PA"
    if _RE_AP.search(t) or "ANTERIOR-POSTERIOR" in t or "ETUKUVA" in t or "ETUPROJEKTIO" in t or _RE_FRONTAL.search(t):
        return "AP"
    return None


def infer_projection_orientation_from_text(text: str) -> Optional[str]:
    """Infer projection from raw free-text (case-insensitive regex scan).

    Searches for keywords like ``"AP"``, ``"PA"``, ``"Lateral"``,
    ``"Oblique"``, and Finnish variants (``"Etukuva"``, ``"Sivukuva"``, etc.).
    Returns a canonical projection code or ``None``.
    """
    return _infer_impl(str(text or "").upper())


def infer_projection_from_code(code: Optional[str]) -> Optional[str]:
    """Infer projection from NCSP code prefix for radiography.

    Code-based projection defaults take priority over text inference
    because the official code encodes the acquisition type.
    """
    if not code:
        return None
    code = code.lower()
    # Chest radiography codes
    if code.startswith("gd1q"):   # makuuasennossa (bedside/supine)
        return "AP"
    if code.startswith("gd1u"):   # radiologian yksikön ulkopuolella (portable)
        return "AP"
    if code.startswith("gd1e"):   # kylkikuva (lateral)
        return "Lat"
    return None


@lru_cache(maxsize=32768)
def _projection_from_candidates(candidates: Tuple[Any, ...]) -> Optional[str]:
    """Memoised core: first candidate that normalises/infers to a projection.

    ``normalize_projection_orientation`` and ``infer_projection_orientation_from_text``
    are pure, so memoising on the candidate tuple is identical.  Distinct
    candidate tuples across a hospital dataset are bounded by description
    diversity, while ``get_projection_orientation`` is called several times per
    radiography row, so this collapses the repeated full scans.
    """
    for value in candidates:
        proj = normalize_projection_orientation(value)
        if proj:
            return proj
        proj = infer_projection_orientation_from_text(str(value or ""))
        if proj:
            return proj
    return None


def get_projection_orientation(
    row: Optional[Mapping[str, Any]] = None,
    *values: Any,
) -> Optional[str]:
    """Return the best projection from a row and/or explicit values.

    Tries each candidate in order: row fields (``Projection``,
    ``ViewPosition``, etc.), then extra positional arguments.
    The first candidate that yields a valid projection wins.
    Memoised on the candidate tuple for performance.
    """
    candidates = []
    if row is not None:
        candidates.extend(row.get(field) for field in _PROJECTION_FIELDS)
    candidates.extend(values)
    try:
        return _projection_from_candidates(tuple(candidates))
    except TypeError:
        # Unhashable candidate (not expected for Excel scalars) -> uncached path.
        for value in candidates:
            proj = normalize_projection_orientation(value)
            if proj:
                return proj
            proj = infer_projection_orientation_from_text(str(value or ""))
            if proj:
                return proj
        return None


__all__ = [
    "normalize_projection_orientation",
    "infer_projection_orientation_from_text",
    "infer_projection_from_code",
    "get_projection_orientation",
]
