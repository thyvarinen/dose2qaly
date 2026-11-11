"""
Centralised matplotlib styling + palette for the Monte Carlo GUI.

Exposes:

* ``apply_style()``   — idempotently set ``matplotlib.rcParams`` so every
                        figure created after the call shares fonts, grid
                        style, palette, etc.
* ``PALETTE_LIST``    — 9-colour Okabe-Ito colour-blind-safe palette, used
                        both as the rc default ``axes.prop_cycle`` and as
                        the source for ``stable_color``.
* ``SEMANTIC``        — named tokens for recurring plot concepts
                        (male/female, cancer/no-cancer, died/survived,
                        baseline/induced, highlight).  Callers should
                        prefer these over hard-coded hex strings.
* ``stable_color(name)`` — deterministic palette lookup keyed by an
                        arbitrary string, so "Lung" gets the same colour
                        across every chart in the GUI.

Design goals
------------

1. **Colour-blind safety.**  Palette is the Okabe-Ito (2008) set, which is
   the de-facto standard for scientific publication and remains
   distinguishable under deuteranopia, protanopia and tritanopia.
2. **Idempotent.**  Importing this module or calling ``apply_style`` twice
   is a no-op after the first application so unit tests / repeated GUI
   launches don't accumulate state.
3. **No hard dependency on running the GUI.**  The module only imports
   matplotlib, so analysis/export scripts can apply the same style.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import matplotlib as _mpl


# Number style for text drawn into figures
# The manuscript writes thousands with a thin space ("1 257 456"), and the
# journal's typesetters do not redraw figures, so figure text has to carry
# the same separator itself.  DejaVu Sans and Arial both have the glyph.
THIN_SPACE = "\u2009"


def fmt_thousands(v, decimals: int = 0) -> str:
    """``f"{v:,.{decimals}f}"`` with a thin-space thousands separator."""
    return f"{v:,.{decimals}f}".replace(",", THIN_SPACE)


# Okabe-Ito (2008) 9-colour palette
# Primary palette used both as the default matplotlib cycle and as the
# source for ``stable_color``.  The ordering is ICLR-/Nature-convention
# (blue first for foreground, grey last for tertiary categories).
PALETTE_LIST: List[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # grey
    "#000000",  # black
]

OKABE_ITO: Dict[str, str] = {
    "black":            "#000000",
    "orange":           "#E69F00",
    "sky_blue":         "#56B4E9",
    "bluish_green":     "#009E73",
    "yellow":           "#F0E442",
    "blue":             "#0072B2",
    "vermilion":        "#D55E00",
    "reddish_purple":   "#CC79A7",
    "grey":             "#999999",
}

# Semantic tokens for recurring plot concepts
# These map domain concepts to colours so every chart in the GUI uses the
# same colour for "male", "died", "induced" etc.  Preferred over
# ``PALETTE_LIST`` indexing when a meaning is attached to the colour.
SEMANTIC: Dict[str, str] = {
    # Sex
    "male":        OKABE_ITO["blue"],
    "female":      OKABE_ITO["reddish_purple"],
    "unknown":     OKABE_ITO["grey"],
    # Outcome
    "cancer":      OKABE_ITO["vermilion"],
    "no_cancer":   OKABE_ITO["bluish_green"],
    # "died" uses black rather than vermilion (which would clash with the
    # "cancer"/"induced" incidence hue).  Black gives maximal contrast against
    # the vermilion bars, carries conventional "mortality" semantics, and
    # remains distinguishable under every colour-blind deficiency (Okabe-Ito
    # safe).
    "died":        OKABE_ITO["black"],
    "survived":    OKABE_ITO["orange"],
    # Comparative framings
    "baseline":    OKABE_ITO["sky_blue"],
    "induced":     OKABE_ITO["vermilion"],
    "reference":   OKABE_ITO["grey"],
    "highlight":   OKABE_ITO["orange"],
    "neutral":     "#2C3E50",  # dark blue-grey, used for axis/reference lines
    # Convergence band fills tend to need lighter shades; these are the
    # same hue as their line counterparts but rendered with alpha<1.
    "ci_band_cases":  OKABE_ITO["vermilion"],
    "ci_band_deaths": OKABE_ITO["blue"],
    # Ordered status tiers (good → warning → bad).  Uses Okabe-Ito
    # bluish-green and vermilion so the pair stays distinguishable under
    # deuteranopia / protanopia (the classic red-green confusion).
    "status_good": OKABE_ITO["bluish_green"],
    "status_warn": OKABE_ITO["orange"],
    "status_bad":  OKABE_ITO["vermilion"],
}

# Modality colour mapping
# Fixed modality→colour assignment used across every figure in the
# manuscript and GUI, so the same modality always has the same colour and
# readers never need to relearn colours between figures.  Okabe-Ito
# hues, colour-blind-safe.  The mapping is by modality *code* (as used
# in ``analysis.MODALITY_DISPLAY`` and ``_per_modality_burden_points``),
# not by position in a sorted list, so it is stable regardless of which
# modalities are present in a given sweep or what order they sort into.
MODALITY_COLORS: Dict[str, str] = {
    "CT":          OKABE_ITO["blue"],           # #0072B2
    "CBCT":        OKABE_ITO["orange"],         # #E69F00
    "RADIOGRAPHY": OKABE_ITO["bluish_green"],   # #009E73
    "MG":          OKABE_ITO["reddish_purple"], # #CC79A7
    "CR":          OKABE_ITO["bluish_green"],   # radiography family
    "DX":          OKABE_ITO["bluish_green"],   # radiography family
    "DR":          OKABE_ITO["bluish_green"],   # radiography family
    "XA":          OKABE_ITO["vermilion"],      # #D55E00
    "RF":          OKABE_ITO["vermilion"],      # angiography family
    "ANGIOGRAPHY": OKABE_ITO["vermilion"],      # angiography family
    "CT/PT":       OKABE_ITO["sky_blue"],       # #56B4E9
    "PT":          OKABE_ITO["sky_blue"],       # PET family
    "NM":          OKABE_ITO["sky_blue"],       # nuclear medicine family
    "MR":          OKABE_ITO["grey"],           # #999999 (non-ionising)
    "US":          OKABE_ITO["grey"],           # (non-ionising)
    "IO":          OKABE_ITO["orange"],         # dental family
    "PX":          OKABE_ITO["orange"],         # dental family
    "DENTAL IMAGING": OKABE_ITO["orange"],      # dental family
}


def _modality_lookup(table: Dict[str, Any], fallback: list, code: str):
    """Case-insensitive modality lookup with a stable hash-derived fallback.

    Tries ``table[code]`` then ``table[code.strip().upper()]``; on a miss,
    hashes ``code`` with md5 and indexes into ``fallback``.
    """
    v = table.get(code)
    if not v and code:
        v = table.get(str(code).strip().upper())
    if v:
        return v
    h = int(hashlib.md5(str(code).encode("utf-8")).hexdigest(), 16)
    return fallback[h % len(fallback)]


def modality_color(code: str) -> str:
    """Return the fixed colour for a modality code.

    Falls back to a stable hash-derived colour from ``PALETTE_LIST`` for
    any modality not in ``MODALITY_COLORS``, so unknown modalities still
    get a consistent colour rather than reusing the previous modality's.

    The lookup is case-insensitive: ``exam_doses.py`` uses mixed-case
    canonical names (``"Radiography"``, ``"Angiography"``) while
    ``detect_modality`` and ``MODALITY_DISPLAY`` use uppercase codes
    (``"RADIOGRAPHY"``, ``"ANGIOGRAPHY"``).  Both forms must map to the
    same colour.
    """
    return _modality_lookup(MODALITY_COLORS, PALETTE_LIST, code)


# Modality line style / marker mapping
# Redundant encoding so modalities stay distinguishable without colour
# (overlapping lines, greyscale print, colour-vision deficiency).  Same
# family grouping as MODALITY_COLORS.
MODALITY_LINESTYLES: Dict[str, Any] = {
    "CT": "-", "CT/PT": "-", "PT": "-", "NM": "-",
    "RADIOGRAPHY": "--", "CR": "--", "DX": "--", "DR": "--",
    "MG": "-.",
    "CBCT": ":", "IO": ":", "PX": ":", "DENTAL IMAGING": ":",
    "XA": (0, (5, 1, 1, 1, 1, 1)), "RF": (0, (5, 1, 1, 1, 1, 1)), "ANGIOGRAPHY": (0, (5, 1, 1, 1, 1, 1)),
    "MR": "-", "US": "-",
}
MODALITY_MARKERS: Dict[str, str] = {
    "CT": "o", "CT/PT": "o", "PT": "o", "NM": "o",
    "RADIOGRAPHY": "s", "CR": "s", "DX": "s", "DR": "s",
    "MG": "^",
    "CBCT": "D", "IO": "D", "PX": "D", "DENTAL IMAGING": "D",
    "XA": "v", "RF": "v", "ANGIOGRAPHY": "v",
    "MR": "o", "US": "o",
}
_FALLBACK_LINESTYLES = ["-", "--", "-.", ":"]
_FALLBACK_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def modality_linestyle(code: str):
    """Fixed line style for a modality code (case-insensitive; stable hash fallback)."""
    return _modality_lookup(MODALITY_LINESTYLES, _FALLBACK_LINESTYLES, code)


def modality_marker(code: str) -> str:
    """Fixed marker for a modality code (case-insensitive; stable hash fallback)."""
    return _modality_lookup(MODALITY_MARKERS, _FALLBACK_MARKERS, code)

# Ordered ramp for binned "low → high" shading (quartiles, tiers)
# Monotonic warm ramp anchored by the cool bluish-green "good" end, built
# entirely from Okabe-Ito hues so a 4-step low→high encoding stays
# colour-blind-safe while reading green→yellow→orange→red to typical
# viewers.  Use this instead of ad-hoc green/yellow/orange/red gradients.
QUARTILE_RAMP: List[str] = [
    OKABE_ITO["bluish_green"],  # lowest quartile  (best)
    OKABE_ITO["yellow"],        # 2nd quartile
    OKABE_ITO["orange"],        # 3rd quartile
    OKABE_ITO["vermilion"],     # highest quartile (worst)
]

# Pale tints of ``QUARTILE_RAMP`` (each hue mixed ~75% toward white) for use
# as table-row / cell *background* fills where dark text must stay legible.
# The bluish-green retains enough blue to stay separable from the warm
# vermilion tint under red-green deficiency — unlike the pale green/pale red
# pair these replace.
QUARTILE_RAMP_LIGHT: List[str] = [
    "#BFE7DC",  # bluish-green tint  (lowest quartile, best)
    "#FBF8D0",  # yellow tint        (2nd quartile)
    "#F9E7BF",  # orange tint        (3rd quartile)
    "#F5D9BF",  # vermilion tint     (highest quartile, worst)
]

_STYLE_APPLIED: bool = False


def apply_style(force: bool = False) -> None:
    """Apply global matplotlib rcParams once per process.

    Safe to call from any module; the first call wins.  Set ``force=True``
    to re-apply after an external library has monkey-patched rcParams
    (e.g. seaborn).  The palette is registered as the default
    ``axes.prop_cycle`` so any plot that does not specify an explicit
    colour will get the colour-blind-safe palette automatically.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED and not force:
        return
    _mpl.rcParams.update({
        # Typography ---------------------------------------------------
        "font.family":       "DejaVu Sans",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "axes.labelsize":    9,
        "axes.labelweight":  "normal",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.framealpha": 0.9,
        # Grid & spines ------------------------------------------------
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "axes.edgecolor":    "#2C3E50",
        "axes.linewidth":    0.8,
        "grid.linestyle":    "--",
        "grid.alpha":        0.3,
        "grid.color":        "#999999",
        # Palette ------------------------------------------------------
        "axes.prop_cycle":   _mpl.cycler(color=PALETTE_LIST),
        # Figure -------------------------------------------------------
        "figure.facecolor":  "white",
        "figure.edgecolor":  "white",
        "savefig.facecolor": "white",
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
    })
    _STYLE_APPLIED = True


def shade_color(hex_color: str, factor: float) -> str:
    """Shade ``hex_color``: positive ``factor`` lightens toward white,
    negative darkens toward black (0 = unchanged, +1 = white, -1 = black).

    Used to derive a family of same-hue tints/shades from a single semantic
    base colour — e.g. giving each modality its own shade of "male blue" /
    "female pink" so a chart can encode two dimensions (sex via hue,
    category via shade) while the primary colour-coding (hue) stays
    instantly recognisable.  Spanning both directions from the base colour
    (e.g. ``linspace(-0.4, 0.7, n)``) gives a much wider, easier-to-tell-
    apart ramp than lightening alone.
    """
    factor = max(-1.0, min(1.0, factor))
    r, g, b = _mpl.colors.to_rgb(hex_color)
    if factor >= 0.0:
        r += (1.0 - r) * factor
        g += (1.0 - g) * factor
        b += (1.0 - b) * factor
    else:
        keep = 1.0 + factor  # factor is negative → scale toward black
        r *= keep
        g *= keep
        b *= keep
    return _mpl.colors.to_hex((r, g, b))


def stable_color(name: object, palette: Optional[Sequence[str]] = None) -> str:
    """Deterministic colour assignment keyed by ``name``.

    Uses an MD5 hash mod palette size so e.g. ``stable_color("Lung")``
    returns the same colour across every chart in the GUI, without needing
    to maintain a global assignment table.  Falls back to ``PALETTE_LIST``
    when no custom palette is supplied.
    """
    pal = list(palette) if palette is not None else PALETTE_LIST
    if not pal:
        return "#000000"
    h = int(hashlib.md5(str(name).encode("utf-8")).hexdigest(), 16)
    return pal[h % len(pal)]


__all__ = [
    "PALETTE_LIST",
    "OKABE_ITO",
    "SEMANTIC",
    "QUARTILE_RAMP",
    "QUARTILE_RAMP_LIGHT",
    "apply_style",
    "stable_color",
    "shade_color",
]
