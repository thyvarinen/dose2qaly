"""ICRP 152 (2022) cancer-risk model -- parameters, kernels, and look-up tables.

This module implements the risk-of-exposure-induced cancer (REIC) and lifetime
attributable risk (LAR) calculations defined in ICRP Publication 152 (2022),
with the severity-adjusted detriment formulation of ICRP Publication 103 (2007)
Annex A.

Key ICRP citations
------------------
ICRP 152 (2022) -- Radiation detriment calculation methodology:
  - p.31  Table 3.1   ERR/EAR weights, dose-response form, source per cancer site.
  - p.32  para 41     Leukaemia EAR linear-quadratic dose response, time-since-
                      exposure factor (t/25)^alpha.
  - p.32  Table 3.4   Leukaemia EAR coefficients beta and alpha by sex and age-group.
  - p.32  Table 3.5   Non-CLL leukaemia baseline incidence (gamma1..gamma5).
  - p.37  para 46     Cancer-free survival mu_bar(x) = mu_m(x) - mu_mac(x) + mu_iac(x).
  - p.38  para 47     REIC dose-adjusted survival S(a,e|d) = S0 * exp(-Sum delta_lambda).
  - p.41  para 51-52  REIC integral with amax = 94 y.
  - p.45  para 61     Female breast EAR-only model (Preston et al. 2002 pooled analysis).
  - p.47  para 63-64  DDREF = 2 for solid cancers, NOT applied to leukaemia, bone, skin.
  - p.48  para 73     Heritable nominal risk = 20 cases / 10,000 / Gy (both sexes).
  - p.49  Table 3.7   Sex-averaged nominal risk per Gy (Total = 1715 / 10,000 / Gy).
  - p.51  Table 3.9   Detriment-adjusted nominal risk per Sv (Total = 574 / 10,000 / Sv).
  - p.51  paras 78,83 Severity formula: q = k + q_min*(1-k); D = R*(k + q*(1-k))*l.
  - p.55  Table 4.1   Reference detriment parameters (default vs alternative).
  - p.60-61 para 97   Latency: reference 5 y for ALL cancer sites; alternative ranges
                      5-10 y solid, 2-5 y leukaemia.
  - p.79  para 138    REIC at 0.1 Gy * 10 convention with nonlinear survival
                      correction S(a,0.1|e) = S0 * exp(-0.1 * C_all(a,e)).
  - Annex A Tables A.1-A.4   Baseline cancer incidence (cases/100,000/yr) by sex,
                             population (Euro-American / Asian), 5-year age groups.
  - Annex A Table A.5        All-cause mortality rates.
  - Annex A Table A.6        All-cancer mortality rates.
  - Annex A Table A.7        Reference population age distribution (used by
                             modify_icrp_age_distribution.py to weight age averaging).

ICRP 103 (2007) -- Severity-adjusted detriment formulation:
  - p.183 Table A.4.1   Site-specific lethality fractions k, q_min, l_rel.
  - p.193 (A.142-A.144) Detriment formula derivation: D = R*(k + q*(1-k))*l.
  - p.197 (A.158)       Sex-specific risk averaging.
  - p.265 Table B.1     Tissue weighting factors w_T (effective-dose weighting).
  - p.294 (B.140-141)   Remainder tissue group: 13 organs (12 unisex + sex-specific).

ICRP 60 (1991), ICRP 59 (1991) -- Bone and skin nominal risks (ICRP 152 retains
these as published lifetime nominals, not LSS-derived ERR/EAR).

Pipeline (top-down)
-------------------
1. ``configure_simulation`` / ``configure_latency`` set DDREF, population, latency.
2. ``solid_kernel_perGy`` / ``leukaemia_kernels`` build per-(cancer, sex, age-at-
   exposure) ndarrays of annual excess incidence per Gy.
3. ``_baseline_survival_curve`` builds cancer-free survival S(a|e) per ICRP 152
   para 46 from the prefix-summed log-survival in ``_CANCER_FREE_SURV_PREFIX``.
4. ``compute_lifetime_excess_cancer_risk_REIC`` integrates ``Sum kernel*S`` over
   ages from e+L to amax = 94 (ICRP 152 paras 51-52).
5. ``precompute_kernels`` builds in-memory and on-disk caches for the MC engine;
   ``_build_rpg_tables`` collapses each kernel*survival product into cumulative
   tables for batch GPU/CPU lookup in ``vectorized``.
6. ``sample_cancer_latency_bulk`` draws cancer onset ages from the kernel*survival
   probability distribution (inverse-CDF sampling).
7. ``sample_lifespans_vectorized`` samples natural lifespans from ICRP 152
   Table A.5 all-cause mortality rates for life-lost / age-at-death accounting.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import threading
from typing import Dict, Optional, Tuple

import numpy as np


MAX_ATTAINED_AGE_ICRP152 = 94
# Solid-cancer integration cap: ICRP 152 paras 51–52 and 138 specify amax = 94 y
# for REIC ("REIC was calculated ... by cumulating the risk up to the attained
# age of 94 years").  The Annex-A baseline incidence tables (A.1–A.4) ARE
# published with a "≥90" age bin (covering 90–94 for REIC; _get_cancer_incidence_rate
# caps age_group at 90), and the ERR/EAR functional form
# β·d·exp(α₁(e−30)/10 + α₂·ln(a/70)) is specified without an upper age bound
# (ICRP 152 para 37).  Therefore the solid-cancer cap is set equal to the
# leukaemia cap (= MAX_ATTAINED_AGE_ICRP152 = 94) to remain ICRP-compliant.
#
# Known residual: reproducing ICRP 152 Table 3.6 EAR columns with this cap
# shows +5–12 % overshoots for several cancers (Lung F, Oesophagus F, Stomach F,
# Colon F).  This appears to be an ICRP-internal inconsistency: aggregating
# their own Table 3.6 per-sex/pop values through the Table 3.9 weighting
# (50/50 sex + 50/50 pop + DDREF=2) does not exactly reproduce the Table 3.9 R
# values either (3–9 % gap for the same cancers).  The simulation follows ICRP
# methodology to the letter and accepts this residual rather than adjusting the
# integration bound to mask it.  See solid_kernel_perGy, _delta_lambda_vectorized,
# and get_female_breast_EAR_kernel_perGy for the enforcement points.
SOLID_MAX_ATTAINED_AGE_ICRP152 = 94

DDREF_SOLID_CANCERS = 2.0
DDREF_LEUKAEMIA = 1.0


def _max_attained_age_for_risk_model() -> int:
    """Active integration upper bound (ICRP 152 Table 4.1).

    Returns ``_SIM_CONFIG["max_attained_age"]`` when set (sensitivity
    override), otherwise the ICRP-152 reference value of 94 years.  This
    is the SAME for REIC and LAR — ICRP 152 lists 99 y as a separate
    sensitivity row in Table 4.1, not as part of "LAR mode".
    """
    return int(_SIM_CONFIG.get("max_attained_age", SOLID_MAX_ATTAINED_AGE_ICRP152))


def _solid_latency_for_risk_model() -> float:
    """Active minimum latency for solid cancers (years).

    Returns the user-configured value from _LATENCY_PARAMS["Solid"] (set via
    configure_latency / configure_simulation).  ICRP 152 Table 4.1 lists
    10 y as the sensitivity alternative; users opt in via the latency
    spinbox / config dict — selecting LAR does NOT auto-bump it.
    """
    return float(_LATENCY_PARAMS.get("Solid", 5.0))


def _leuk_latency_for_risk_model() -> float:
    """Active minimum latency for leukaemia (years), see above."""
    return float(_LATENCY_PARAMS.get("Leukaemia", 5.0))


# Minimum latency periods (years).
# Defaults match the ICRP 152 reference detriment: 5 y for ALL cancer sites
# (ICRP 152 p.61 in §4.2: "Reference detriment assumes minimum latency of
# 5 years for all cancer sites").  The recommended ranges in ICRP 152 §4.2
# para 97 / Fig. 4.7 are 5-10 y for solid and 2-5 y for leukaemia; those
# are exposed via the startup dialog (config.show_config_dialog
# spinboxes) so the user can override them at initialisation for sensitivity
# analysis.  configure_latency() / configure_simulation() write into this dict.
_LATENCY_PARAMS: dict = {"Leukaemia": 5.0, "Solid": 5.0}

# Fallback latency for sites without a time-resolved REIC kernel
# (Bone via ICRP 60 nominal risk, Skin via ICRP 59 nominal risk, plus any
# unrecognised site that lands in the fallback for defensive reasons).
# ICRP 152 and BEIR VII do not publish a site-specific latency distribution
# for these sites, so we sample the onset age uniformly on the same support
# the REIC integration uses for every other site: [e + min_lat, a_max + 1),
# with a_max = MAX_ATTAINED_AGE_ICRP152 (= 94).  This is the maximum-entropy
# choice given only the published lower bound (the configured minimum
# latency, default 5 y) and the integration upper bound a_max.

# Active simulation parameters — set once at initialisation via configure_simulation().
# ICRP 152 Table 4.1 defaults.
_SIM_CONFIG: dict = {
    "ddref_solid":  2.0,
    "ddref_leuk":   1.0,
    "population":   "Euro-American",
    "risk_model":   "REIC",
}


def configure_latency(params: dict) -> None:
    """Set the active minimum latency periods (years) used by all risk kernels.

    Must be called before ``precompute_kernels()`` so the cached kernels
    reflect the configured latency.

    Parameters
    ----------
    params : dict
        Required keys ``'Leukaemia'`` and ``'Solid'`` (years).

    ICRP 152 references
    -------------------
    - p.61 (in §4.2): the reference detriment uses **5 years for ALL
      cancer sites**.
    - §4.2 para 97: the recommended ranges are **5-10 y for solid** and
      **2-5 y for leukaemia** (used in sensitivity analysis only; the
      lower bounds 5 y / 2 y are NOT defaults).
    - Fig. 4.7: leukaemia detriment increases ~+48 % when the latency
      drops from 5 y to 2 y; solid-cancer detriment changes by only a
      few percent across 5-10 y.

    Defaults supplied by ``DEFAULTS`` in ``config.py``
    therefore set both sites to 5 y to reproduce the ICRP 152 reference
    detriment.  The startup dialog allows the user to override either
    value for sensitivity analysis.
    """
    # IMPORTANT: update the existing dict in place rather than rebinding the
    # name.  Other modules import this dict via ``from risk_models
    # import _LATENCY_PARAMS`` which captures a reference at import time;
    # rebinding here would leave those modules pointing at a stale dict.
    _LATENCY_PARAMS.clear()
    _LATENCY_PARAMS["Leukaemia"] = float(params["Leukaemia"])
    _LATENCY_PARAMS["Solid"]     = float(params["Solid"])


def configure_simulation(cfg: dict) -> None:
    """Set all active simulation parameters from a merged config dict.

    Also calls configure_latency() so only one call is needed before
    precompute_kernels().  Expected keys (all optional, ICRP 152 defaults shown):
      'ddref_solid'        – DDREF for solid cancers           (default 2.0)
      'ddref_leuk'         – DDREF for leukaemia               (default 1.0)
      'population'         – baseline population model         (default 'Euro-American')
      'risk_model'         – 'REIC' or 'LAR'                   (default 'REIC')
      'max_attained_age'   – integration cap, y (ICRP 152 ref. 94, alt. 99)
      'Leukaemia'          – minimum latency for leukaemia, y  (default 5.0)
      'Solid'              – minimum latency for solid, y      (default 5.0)

    ICRP 152 §4.1 para 90 explicitly states sensitivity-analysis alternatives
    are applied "one at a time".  Setting ``max_attained_age = 99`` runs the
    Table 4.1 amax sensitivity row independently of the chosen risk_model;
    similarly, the latency and DDREF spinboxes drive their respective
    sensitivity rows without coupling to risk_model.
    """
    # IMPORTANT: update the existing dict in place rather than rebinding it.
    # Other modules (risk.py, vectorized.py, etc.) import
    # _SIM_CONFIG via ``from risk_models import
    # _SIM_CONFIG`` and capture a reference to the original dict object at
    # import time; rebinding here would silently leave those modules looking
    # at a stale config (so e.g. risk_model='LAR' would not propagate to
    # compute_lifetime_risk's _is_lar branch).
    _SIM_CONFIG.clear()
    _SIM_CONFIG["ddref_solid"]      = float(cfg.get("ddref_solid", 2.0))
    _SIM_CONFIG["ddref_leuk"]       = float(cfg.get("ddref_leuk", 1.0))
    _SIM_CONFIG["population"]       = str(cfg.get("population", "Euro-American"))
    _SIM_CONFIG["risk_model"]       = str(cfg.get("risk_model", "REIC"))
    _SIM_CONFIG["max_attained_age"] = int(cfg.get(
        "max_attained_age", SOLID_MAX_ATTAINED_AGE_ICRP152))
    # --- ICRP severity-weighted QALYs accounting (see vectorized.py _QALY_event_bulk) ---------------
    # "icrp152" (default) → q_severity * l_rel * 15 yr per cancer (the ICRP 152
    #                       site-average construction).
    # "full"              → q_severity * per-patient YLL (cancer onset age to the
    #                       patient's own competing-mortality-adjusted lifespan),
    #                       the per-age generalisation of the ICRP site average,
    #                       with optional exponential discounting (WHO-CHOICE /
    #                       GBD 2019 convention).
    _SIM_CONFIG["QALY_mode"] = str(cfg.get("QALY_mode", "full")).lower()
    _SIM_CONFIG["QALY_discount_rate"] = float(cfg.get("QALY_discount_rate", 0.0))
    configure_latency({
        "Leukaemia": float(cfg["Leukaemia"] if "Leukaemia" in cfg else cfg["latency_leukaemia"]),
        "Solid":     float(cfg["Solid"]     if "Solid"     in cfg else cfg["latency_solid"]),
    })


CANCER_TO_ORGAN_DOSE_KEY = {
    "Breast": "Breast",
    "Lung": "Lung",
    "Skin": "Skin",
    "Stomach": "Stomach",
    "Liver": "Liver",
    "Colon": "Colon",
    "Bladder": "Bladder",
    "Brain": "Brain",
    "Thyroid": "Thyroid",
    "Leukaemia": "Red_Marrow",
    "Bone": "Bone",
    "Ovary": "Ovary",
    "Testis": "Testis",
    "Prostate": "Prostate",
    "Uterus": "Uterus",
    "Kidney": "Kidney",
    "Pancreas": "Pancreas",
    "Salivary_Glands": "Salivary_Glands",
    "Oesophagus": "Oesophagus",
    "Remainder": "Remainder",
}

CANCER_DOSE_FALLBACK_KEYS = {
    "Stomach": ["Stomach_Wall"],
    "Colon": ["Rectosigmoid"],
    "Oesophagus": ["Esophagus"],
    "Esophagus": ["Oesophagus"],
    # Radiography organ table uses different naming conventions:
    "Red_Marrow": ["Active_Bone_Marrow"],   # leukaemia
    "Bladder": ["Urinary_Bladder"],
    "Bone": ["Endosteum"],
    "Ovary": ["Gonads", "Ovaries"],         # F Gonads = ovarian dose
    "Testis": ["Gonads", "Testicles"],      # M Gonads = testicular dose
    "Lung": ["Lungs"],
    "Kidney": ["Kidneys"],
}


# ICRP 152 "Other solid" REIC-relevant sub-sites.
#
# Per ICRP 103 §B.140 the Remainder tissues are 13 per sex:
#   adrenals, ET, gallbladder, heart, kidneys, lymph nodes, muscle, oral mucosa,
#   pancreas, prostate (M)/uterus (F), small intestine, spleen, thymus.
# Per ICRP 103 §A.158 brain and salivary glands are NOT in this Remainder list;
# each carries an explicit w_T = 0.01 because their cancer risk is judged greater
# than other Remainder tissues.  In ICRP 152 there is no separate brain or
# salivary-gland risk model — both are absorbed into the LSS-derived "Other
# solid" cancer category (Table 3.1, 50%ERR:50%EAR).  For REIC of "Other solid"
# we therefore average the absorbed dose over the union of:
#   - the 13 ICRP-103 Remainder organs (per sex),
#   - brain (w_T = 0.01),
#   - salivary glands (w_T = 0.01).
# Within that union, four sub-sites have their own dose key tracked separately
# (Brain, Salivary_Glands, Kidney, Pancreas, each with multiplicity 1) and the
# rest are lumped under the "Remainder" dose key.  Prostate (M)/uterus (F) is
# part of the ICRP-103 Remainder but is not separately tracked here for REIC
# (its REIC contribution is implicitly zero in the dose average; the gonadal
# dose used for heritable detriment is Testis or Ovary, NOT prostate/uterus).
OTHER_SOLID_REMAINDER_SITES = {
    "Remainder", "Brain", "Salivary_Glands", "Kidney", "Pancreas",
}

# Canonical ICRP-103 Remainder count (13 per sex); the ICRP-103
# effective-dose w_T denominator.
N_REMAINDER_ORGANS_ICRP = 13

REMAINDER_ORGAN_MULTIPLICITY: dict = {
    "Brain":           1,   # ICRP 103 §A.158, w_T = 0.01 (NOT in 13 Remainder)
    "Salivary_Glands": 1,   # ICRP 103 §A.158, w_T = 0.01 (NOT in 13 Remainder)
    "Kidney":          1,   # in ICRP-103 13-Remainder list
    "Pancreas":        1,   # in ICRP-103 13-Remainder list
    "Remainder":       10,  # 10 untracked ICRP-103 Remainder organs (the 13 minus Kidney, Pancreas, prostate/uterus)
}


# ICRP 152 Table 3.7 / ICRP 103 Table A.4.4: Heritable (gonad) detriment.
# Nominal risk coefficient for heritable effects: 20 per 10,000 per Gy
# Heritable detriment per Sv from ICRP 152 Table 3.9 (detriment-adjusted, no DDREF).
# Nominal risk = 20/10000/Gy (Table 3.7). With two-level det_factor (k=0.80, q_min=0.1, l=1.32):
#   q = 0.80 + 0.1*(1-0.80) = 0.82; det_factor = (0.80 + 0.82*0.20)*1.32 = 1.272
#   D = 20 * 1.272 = 25.4 per 10,000 per Sv (ICRP 152 Table 3.9). DDREF=1 for heritable.
# Units: per Sv (absorbed gonad dose). For diagnostic photons wR=1 so Gy=Sv.
HERITABLE_DETRIMENT_PER_GY = {
    "M": 25.4 / 10000,   # 25.4 per 10,000 per Sv (ICRP 152 Table 3.9, no DDREF)
    "F": 25.4 / 10000,   # 25.4 per 10,000 per Sv (ICRP 152 Table 3.9, no DDREF)
}
HERITABLE_DETRIMENT_PER_GY["U"] = 0.5 * (
    HERITABLE_DETRIMENT_PER_GY["M"] + HERITABLE_DETRIMENT_PER_GY["F"]
)
# Organ dose key used for heritable detriment calculation
HERITABLE_GONAD_DOSE_KEY = {"M": "Testis", "F": "Ovary"}



def get_organ_dose_for_cancer(cancer_type: str, organ_doses: dict) -> float:
    """Return the absorbed organ dose (Gy) for a given cancer type.

    Uses the primary organ key from ``CANCER_TO_ORGAN_DOSE_KEY`` and falls
    back through ``CANCER_DOSE_FALLBACK_KEYS`` if the primary organ is not
    present in the dose dict.  Returns ``0.0`` when no match is found.
    """
    organ_key = CANCER_TO_ORGAN_DOSE_KEY.get(cancer_type)
    if organ_key and organ_key in organ_doses:
        return float(organ_doses[organ_key])
    if organ_key:
        for fallback_key in CANCER_DOSE_FALLBACK_KEYS.get(organ_key, []):
            if fallback_key in organ_doses:
                return float(organ_doses[fallback_key])
    return 0.0



def normalize_sex_code(sex: str) -> str:
    """Normalise any sex representation to ``"M"``, ``"F"``, or ``"U"``.

    Handles numeric codes (1 → M, 2 → F) and common string variants
    (including Finnish ``"Mies"`` / ``"Nainen"`` and the DICOM PatientSex
    marker ``"O"`` / ``"Other"``); unknown markers map to ``"U"``.  Any
    other value raises ``ValueError`` — no silent guesses.
    """
    try:
        v = float(sex)
        if not np.isfinite(v):
            return "U"
        if v == 1:
            return "M"
        if v == 2:
            return "F"
    except (TypeError, ValueError):
        pass
    s = str(sex).strip().upper()
    if not s or s in {"U", "O", "OTHER", "UNKNOWN", "UNSPECIFIED", "NONE", "NA", "N/A"}:
        return "U"
    if s in {"F", "FEMALE", "FEM", "W", "WOMAN", "WOMEN", "N", "NAINEN"}:
        return "F"
    if s in {"M", "MALE", "MAN", "MIES"}:
        return "M"
    raise ValueError(
        f"Unrecognised sex value {sex!r}; expected M/F, Male/Female, "
        "Mies/Nainen, 1/2, or an unknown marker (empty, U, O/Other, Unknown, NA)"
    )


def _sex_weighted_value(params: dict, key: str, sex_code: str, default: float = 0.0) -> float:
    block = params.get(key, {}) if isinstance(params, dict) else {}
    if sex_code in ("M", "F"):
        return float(block.get(sex_code, default))
    m = block.get("M", None)
    f = block.get("F", None)
    if m is None and f is None:
        return float(default)
    if m is None:
        return float(f)
    if f is None:
        return float(m)
    return float(0.5 * (float(m) + float(f)))



#All-cause mortality rates by age group, sex, and population
# ALL_CAUSE_MORTALITY_RATES: all-cause mortality (deaths per 100,000/year).
# Source: ICRP 152 Table A.5 (modified from ICRP 103 Tables A.4.12/13/16/17).
# Module-level so the single-year interpolation cache (_LIFE_INTERP_CACHE,
# keyed on the persistent sub-dict identity) stays valid across calls.
ALL_CAUSE_MORTALITY_RATES = {
    "M": {
        "Euro-American": {
            0: 143.02, 5: 15.39, 10: 19.43, 15: 66.78, 20: 94.71, 25: 99.79, 30: 124.33,
            35: 160.80, 40: 224.83, 45: 321.50, 50: 505.70, 55: 821.44, 60: 1378.11,
            65: 2241.12, 70: 3590.14, 75: 5634.15, 80: 9122.79, 85: 13879.10, 90: 24029.19,
        },
        "Asian": {
            0: 149.24, 5: 24.88, 10: 23.65, 15: 35.16, 20: 50.43, 25: 59.21, 30: 80.39,
            35: 114.64, 40: 188.22, 45: 276.69, 50: 399.85, 55: 646.43, 60: 1257.04,
            65: 2107.53, 70: 3550.26, 75: 5749.87, 80: 9661.98, 85: 12799.94, 90: 22367.18,
        },
    },
    "F": {
        "Euro-American": {
            0: 114.61, 5: 11.35, 10: 13.28, 15: 28.51, 20: 33.03, 25: 40.17, 30: 55.43,
            35: 81.36, 40: 122.96, 45: 193.21, 50: 309.20, 55: 489.59, 60: 801.25,
            65: 1283.49, 70: 2098.33, 75: 3406.46, 80: 5934.90, 85: 9876.82, 90: 19441.90,
        },
        "Asian": {
            0: 127.18, 5: 16.67, 10: 15.15, 15: 18.31, 20: 27.75, 25: 33.29, 30: 44.91,
            35: 62.83, 40: 107.45, 45: 162.17, 50: 237.87, 55: 399.63, 60: 740.16,
            65: 1239.84, 70: 2184.11, 75: 3682.84, 80: 6509.31, 85: 8923.98, 90: 17750.63,
        },
    },
}


_LIFE_INTERP_CACHE: dict = {}


def _interp_life_rate_per_person(rate_dict: dict, age: int) -> float:
    """Single-year log-linear interpolation of a 5-year abridged ICRP Annex A
    rate table ({age_anchor: rate_per_100k}) -> per-person/year fraction.

    ICRP 152's cancer-free survival / REIC is a SINGLE-YEAR discrete sum
    (paras 46-51), but the Annex A life-table inputs (Tables A.5/A.6 mortality,
    A.1-A.4 incidence) are tabulated in 5-year groups.  Sampling them with a
    lower-edge step (``age//5*5``) gives every age in a band its YOUNGEST rate,
    under-counting the steeply rising mortality hazard within each band.  That
    inflates cancer-free survival and produced a near-uniform ~+7% over-
    prediction versus the published Table 3.6.  Expanding the abridged table to
    single years by log-linear interpolation between the tabulated anchors is
    the standard demographic treatment and reproduces Table 3.6 with ~0
    systematic deviation.

    NOTE: this is applied to the SURVIVAL life table only (all-cause mortality,
    all-cancer mortality, all-cancer incidence).  The age-specific cancer
    INCIDENCE used as the ERR baseline λ_c(a) (``_get_cancer_incidence_rate``)
    is deliberately left as the 5-year step, matching how ICRP applies
    age-group-specific incidence rates (interpolating it re-introduces the
    bias; see the Table 3.6 root-cause investigation).

    Values below the first / above the last anchor use flat extrapolation
    (identical to the capped-step behaviour at age >= 90).
    """
    arr = _LIFE_INTERP_CACHE.get(id(rate_dict))
    if arr is None:
        xs = sorted(rate_dict)
        ys = np.array([rate_dict[x] for x in xs], dtype=float)
        ages = np.arange(121, dtype=float)
        if np.all(ys > 0.0):
            arr = np.exp(np.interp(ages, xs, np.log(ys)))
        else:
            arr = np.interp(ages, xs, ys)        # tables containing zeros
        arr = (arr / 100000.0).astype(float)
        _LIFE_INTERP_CACHE[id(rate_dict)] = arr
    a = 0 if age < 0 else (120 if age > 120 else int(age))
    return float(arr[a])


def _lookup_rate_by_sex_pop(age: int, sex_code: str, population: str,
                            rates_by_sex: dict) -> float:
    """Return interpolated rate per person for a sex and population.

    ``rates_by_sex`` maps ``"M"`` / ``"F"`` to population-specific rate tables.
    Unknown sex is averaged from the male and female rates.
    """
    if sex_code in ("M", "F"):
        pop_rates = rates_by_sex[sex_code].get(population,
                                               rates_by_sex[sex_code]["Euro-American"])
        return _interp_life_rate_per_person(pop_rates, age)
    m = _lookup_rate_by_sex_pop(age, "M", population, rates_by_sex)
    f = _lookup_rate_by_sex_pop(age, "F", population, rates_by_sex)
    return 0.5 * (m + f)


def get_mortality_rate(age: int, sex: str = "M", population: str = "Euro-American") -> float:
    """All-cause mortality rate (deaths per person per year) at ``age``.

    Sources: ICRP 152 Annex A Table A.5 (all-cause mortality).
    For unknown sex, returns the arithmetic mean of male and female rates.
    """
    return _lookup_rate_by_sex_pop(age, sex.upper(), population, ALL_CAUSE_MORTALITY_RATES)



# ALL_CANCER_MORTALITY_RATES: All-cancer mortality rates (deaths per 100,000 persons per year)
# Source: ICRP 152 (2022) Table A.6 (modified from ICRP 103 Tables A.4.12/13/16/17)
ALL_CANCER_MORTALITY_RATES_M = {
    "Euro-American": {
        0: 2.75, 5: 2.74, 10: 2.52, 15: 3.50, 20: 4.50, 25: 5.87, 30: 9.09, 35: 16.28,
        40: 34.98, 45: 69.83, 50: 143.81, 55: 262.09, 60: 457.53, 65: 734.15,
        70: 1065.72, 75: 1427.76, 80: 1880.96, 85: 2208.86, 90: 2677.26,
    },
    "Asian": {
        0: 3.79, 5: 3.96, 10: 4.78, 15: 4.81, 20: 5.06, 25: 7.79, 30: 14.60, 35: 29.41,
        40: 58.32, 45: 95.90, 50: 149.26, 55: 252.16, 60: 482.58, 65: 755.18,
        70: 1065.73, 75: 1365.66, 80: 1661.07, 85: 1586.63, 90: 1838.67,
    },
}
ALL_CANCER_MORTALITY_RATES_F = {
    "Euro-American": {
        0: 2.22, 5: 2.01, 10: 2.05, 15: 2.76, 20: 3.40, 25: 5.97, 30: 12.77, 35: 26.07,
        40: 48.98, 45: 88.79, 50: 150.52, 55: 232.48, 60: 343.06, 65: 487.75,
        70: 654.11, 75: 801.53, 80: 988.90, 85: 1178.13, 90: 1220.69,
    },
    "Asian": {
        0: 3.38, 5: 3.08, 10: 3.52, 15: 3.39, 20: 3.97, 25: 6.37, 30: 13.20, 35: 23.88,
        40: 45.04, 45: 66.72, 50: 94.83, 55: 151.41, 60: 245.00, 65: 357.21,
        70: 508.02, 75: 653.04, 80: 780.83, 85: 712.91, 90: 840.17,
    },
}


def _get_all_cancer_mortality_rate(age: int, sex_code: str, population: str = "Euro-American") -> float:
    """All-cancer mortality rate (per person per year) from ICRP 152 Table A.6.

    Single-year log-linear interpolation of the 5-year table (life-table input
    to cancer-free survival); see :func:`_interp_life_rate_per_person`.
    """
    rates_by_sex = {
        "M": ALL_CANCER_MORTALITY_RATES_M,
        "F": ALL_CANCER_MORTALITY_RATES_F,
    }
    return _lookup_rate_by_sex_pop(age, sex_code, population, rates_by_sex)


# Cancer-free survival prefix per ICRP 152 para 46: μ̄(x) = μ_m(x) − μ_mac(x) + μ_iac(x).
# Built lazily after BASELINE_RATES_ICRP is defined.  Key: (sex, population) → ndarray[121].
_CANCER_FREE_SURV_PREFIX: dict = {}


def _build_allcause_pop() -> dict:
    """Build {(sex, population): ndarray[121]} of annual all-cause death probabilities."""
    result = {}
    for sex in ("M", "F"):
        for pop in ("Euro-American", "Asian"):
            arr = np.zeros(121, dtype=float)
            for age in range(121):
                arr[age] = get_mortality_rate(age, sex, pop)
            result[(sex, pop)] = arr
    return result


_ALLCAUSE_PER_AGE_POP = _build_allcause_pop()


def _get_cancer_incidence_rate(age: int, sex_code: str, cancer_type: str,
                               population: str = "Euro-American") -> float:
    """Baseline cancer incidence rate (per person per year) from ICRP 152 Annex A tables."""
    sex = sex_code if sex_code in ("M", "F") else "M"
    if (cancer_type in ("Breast", "Ovary") and sex == "M") or (
        cancer_type == "Prostate" and sex == "F"
    ):
        return 0.0
    age_group = (age // 5) * 5
    if age_group > 90:
        age_group = 90
    mapped = cancer_type
    if mapped == "Leukaemia":
        mapped = "Non-CLL Leukaemia"
    try:
        pop_rates = BASELINE_RATES_ICRP[sex].get(population,
                    BASELINE_RATES_ICRP[sex].get("Euro-American", {}))
        site_rates = pop_rates.get(mapped, pop_rates.get("Remainder", {}))
        return float(site_rates.get(age_group, 0.0)) / 100000.0
    except (KeyError, AttributeError):
        return 0.0


_BASELINE_INCIDENCE_CACHE: dict = {}

def _baseline_incidence_array_for(cancer_type: str, sex_code: str,
                                  population: str = "Euro-American") -> np.ndarray:
    """Return baseline cancer incidence rate array for ages 0-120 (per person/year).
    Used by solid_kernel_perGy for the ERR term: Δλ_ERR = β × ERR(e,a) × λ_0(a).
    Source: ICRP 152 Annex A Tables A.1-A.4.
    """
    key = (cancer_type, sex_code, population)
    cached = _BASELINE_INCIDENCE_CACHE.get(key)
    if cached is not None:
        return cached
    out = np.zeros(121, dtype=float)
    for age in range(121):
        out[age] = _get_cancer_incidence_rate(age, sex_code, cancer_type, population)
    out.flags.writeable = False
    _BASELINE_INCIDENCE_CACHE[key] = out
    return out


def _get_all_cancer_incidence_rate(age: int, sex_code: str,
                                   population: str = "Euro-American") -> float:
    """Sum of baseline incidence rates over all cancer sites (per person per year).
    Used in cancer-free survival: μ_iac(x) = Σ_c λ_c(x) per ICRP 152 para 46.
    Sources: ICRP 152 Tables A.1 (EA-M), A.2 (EA-F), A.3 (AS-M), A.4 (AS-F).

    As a life-table input to cancer-free survival, the 5-year table is expanded
    to single years by log-linear interpolation per site, then summed; see
    :func:`_interp_life_rate_per_person`.
    """
    sex = sex_code if sex_code in ("M", "F") else "M"
    a = 0 if age < 0 else (120 if age > 120 else int(age))
    total = 0.0
    try:
        pop_rates = BASELINE_RATES_ICRP[sex].get(population,
                    BASELINE_RATES_ICRP[sex].get("Euro-American", {}))
        for site_rates in pop_rates.values():
            if site_rates:
                total += _interp_life_rate_per_person(site_rates, a)
    except (KeyError, AttributeError):
        pass
    return total


def _ensure_cancer_free_surv_prefix() -> None:
    """Lazily build _CANCER_FREE_SURV_PREFIX per ICRP 152 para 46.
    Called at runtime so BASELINE_RATES_ICRP is guaranteed to be fully defined.
    Prefix[(sex, pop)][age] = Σ_{t=0}^{age-1} log(1 − μ̄(t))
    where μ̄(t) = μ_m(t) − μ_mac(t) + μ_iac(t).
    """
    global _CANCER_FREE_SURV_PREFIX
    if _CANCER_FREE_SURV_PREFIX:
        return
    for sex in ("M", "F"):
        for pop in ("Euro-American", "Asian"):
            log_surv = np.zeros(121, dtype=float)
            cumsum = 0.0
            for age in range(121):
                log_surv[age] = cumsum
                mu_m = get_mortality_rate(age, sex, pop)
                mu_mac = _get_all_cancer_mortality_rate(age, sex, pop)
                mu_iac = _get_all_cancer_incidence_rate(age, sex, pop)
                mu_bar = max(0.0, mu_m - mu_mac + mu_iac)
                cumsum += np.log(max(1e-10, 1.0 - mu_bar))
            _CANCER_FREE_SURV_PREFIX[(sex, pop)] = log_surv


# Mataac (2024) clinical competing-mortality adjustment
# ----------------------------------------------------------------------------
# Heavily-imaged patients die sooner than the general population (comorbidity /
# disease severity, not radiation).  We capture this with an EMPIRICAL ADDITIVE
# EXCESS HAZARD derived from the Mataac year-to-death table (Eur J Radiol
# 2024;178:111629), applied INSIDE the ICRP 152 cancer-free survival product.
#
# For dose stratum k, sex s, population pop and observation year t (0..10):
#   h_coh_k[t]        = conditional yearly all-cause hazard from the cohort
#                       counts (deaths[t] / patients still alive at start of t)
#   h_pop[s,pop,k,t]  = INDIRECTLY STANDARDISED ICRP all-cause mortality, i.e.
#                       the expectation E[mu_m(a + t)] taken over the dose
#                       group's own age distribution, NOT mu_m evaluated at the
#                       group mean age.  mu_m is strongly convex in age and the
#                       Mataac groups are wide (SD 14.2-17.6 y, Table 1), so
#                       mu_m(mean age) understates the expected hazard by a
#                       factor of 2.5-3.6 and would inflate the excess by
#                       15% (group D) to 46% (group B).  The age distribution is
#                       reconstructed as a Gaussian on the reported mean +/- SD,
#                       which reproduces the paper's reported inter-quartile
#                       ranges to within ~1 year for groups B, C and D.
#   excess[s,pop,k,t] = max(0, h_coh_k[t] - h_pop[s,pop,k,t])
# The per-patient clinical cancer-free hazard at attained age n (exposure age e)
# is the standard ICRP hazard plus the windowed excess:
#   mu_clinical(n) = mu_bar(n) + excess[s,pop,k, n-e]   for 0 <= n-e <= 10
#   mu_clinical(n) = mu_bar(n)                          for n-e > 10  (revert)
# with mu_bar = mu_m - mu_mac + mu_iac (the existing ICRP definition).  Because
# the excess is the ONLY difference from ICRP survival, there is no mu_mac
# double-count, and nothing is extrapolated beyond the observed 10-year window.
# The excess baseline is population-specific (Asian patients use the Asian
# baseline; Euro-American and unspecified use the Euro-American baseline, the
# Mataac cohort being Western).  The strongly front-loaded mortality (e.g.
# Group D loses ~41% in 2 years) is reproduced directly from the data.
#
# Cumulative-effective-dose strata (frail strata 0..2 = groups B, C, D):
#   <10 mSv  -> reference: standard ICRP survival, no adjustment (group A)
#   10-50    -> group B
#   50-100   -> group C
#   >=100    -> group D
#
# Group A is deliberately NOT a frail stratum.  Indirectly standardising its
# observed deaths (2154/14031 over 10 years) against the ICRP 152 reference
# rates for its own age distribution (61.6 +/- 21.7 y) gives an expected death
# fraction of 0.330 versus 0.154 observed, i.e. a standardised mortality ratio
# of 0.46.  Group A therefore carries no net excess hazard over the ICRP
# reference at all; the only positive residual anywhere in its 11-year window
# is a single 1.2 percentage-point bump in year 0, worth a correction factor of
# 0.988 that is flat across exposure ages.  Holding group A at 1.0 is thus
# conservative rather than approximate: the data would support a factor > 1.
# (For comparison the standardised mortality ratios of groups B, C and D are
# 1.31, 3.35 and 3.59.)
_MATAAC_STRATA_SV = (0.010, 0.050, 0.100)       # Sv: lower edge + upper edges
# Group mean age (years) for frail strata B, C, D (Mataac Table 1 demographics).
_MATAAC_GROUP_MEAN_AGE = (52.5, 50.1, 52.9)
# Group age standard deviation (years) for the same strata (Mataac Table 1).
# Used to indirectly standardise the subtracted ICRP baseline; see above.
_MATAAC_GROUP_AGE_SD = (17.6, 15.4, 14.2)
# Cohort size per frail stratum (Mataac Table 1).
_MATAAC_GROUP_N = (18989, 2756, 769)
# Deaths by years-to-death (t = 0..10), per frail stratum B, C, D
# (Mataac "Years to Death by CED Dose Group").
_MATAAC_DEATH_COUNTS = (
    (1537, 1001, 554, 304, 328, 306, 260, 253, 176, 141, 93),   # B (10-50)
    (412, 420, 184, 110, 74, 58, 50, 55, 26, 20, 23),           # C (50-100)
    (136, 181, 66, 24, 24, 8, 16, 10, 9, 9, 4),                 # D (>=100)
)
_MATAAC_WINDOW_YEARS = 11                        # observed years t = 0..10
_MATAAC_EXCESS: dict = {}                        # (sex, pop, k) -> excess[0..10]
_MU_BAR_CACHE: dict = {}                         # (sex, pop) -> per-age mu_bar array
_MATAAC_CORRECTION_CACHE: dict = {}             # (e, sex, pop, k) -> correction factor


def _mataac_stratum_index(cum_dose_sv: float) -> int:
    """Map cumulative effective dose (Sv) to a Mataac frail-stratum index.

    Returns -1 for the reference stratum (<10 mSv, no adjustment) and 0, 1, 2
    for the 10-50, 50-100 and >=100 mSv strata respectively.
    """
    if cum_dose_sv < _MATAAC_STRATA_SV[0]:
        return -1
    for k, edge in enumerate(_MATAAC_STRATA_SV[1:]):
        if cum_dose_sv < edge:
            return k
    return len(_MATAAC_DEATH_COUNTS) - 1


def _ensure_mu_bar(sex: str, pop: str) -> np.ndarray:
    """Per-age ICRP 152 cancer-free hazard mu_bar = mu_m - mu_mac + mu_iac.

    Cached per (sex, pop); reused to build the clinical survival product.
    """
    cached = _MU_BAR_CACHE.get((sex, pop))
    if cached is not None:
        return cached
    mu_bar = np.empty(121, dtype=float)
    for age in range(121):
        mu_m = get_mortality_rate(age, sex, pop)
        mu_mac = _get_all_cancer_mortality_rate(age, sex, pop)
        mu_iac = _get_all_cancer_incidence_rate(age, sex, pop)
        mu_bar[age] = min(1.0 - 1e-10, max(0.0, mu_m - mu_mac + mu_iac))
    _MU_BAR_CACHE[(sex, pop)] = mu_bar
    return mu_bar


def _ensure_mataac_excess_hazard() -> None:
    """Build the empirical additive excess hazard for every frail stratum.

    ``_MATAAC_EXCESS[(sex, pop, k)]`` is the per-year excess all-cause hazard
    (length ``_MATAAC_WINDOW_YEARS``) of dose group ``k`` over the INDIRECTLY
    STANDARDISED ICRP 152 all-cause mortality of that group, for the given
    sex/population::

        h_coh_k[t]        = deaths_k[t] / (alive at start of year t)
        h_pop[s,pop,k,t]  = Sum_a w_k(a) mu_m(a + t, s, pop)
        excess[s,pop,k,t] = max(0, h_coh_k[t] - h_pop[s,pop,k,t])

    ``w_k`` is the group's age distribution, reconstructed as a Gaussian on the
    mean and standard deviation reported in Mataac Table 1 and renormalised over
    ages 0-100.  Standardising over the distribution rather than evaluating
    ``mu_m`` at the mean age matters because ``mu_m`` is convex in age: at these
    group widths ``mu_m(mean age)`` is 2.5-3.6x smaller than ``E[mu_m(age)]``,
    which would inflate the 10-year summed excess by 15% (group D) to 46%
    (group B) and correspondingly over-discount the risk.

    The ``max(0, .)`` floor is required because vital status in the source
    cohort came from a single institution's registry, so deaths occurring
    outside that institution are not ascertained.  The resulting under-count is
    negligible in the first years, when patients are still in active care, but
    drives the observed hazard implausibly far below the reference in the later
    years of the window (group A reaches 0.08x the reference by year 10).  The
    floor keeps the genuine early excess while refusing to credit an apparent
    late-window survival advantage that is a data-capture artefact.

    The cohort death counts are the same (Western) observations for both
    populations; only the subtracted baseline is population-specific.  Built
    once from the static ICRP 152 Annex A tables.
    """
    if _MATAAC_EXCESS:
        return
    # Conditional yearly cohort hazard h_coh_k[t] (independent of sex/pop).
    h_coh = []
    for k, deaths in enumerate(_MATAAC_DEATH_COUNTS):
        n0 = float(_MATAAC_GROUP_N[k])
        alive = n0
        hz = np.zeros(_MATAAC_WINDOW_YEARS, dtype=float)
        for t in range(_MATAAC_WINDOW_YEARS):
            d = float(deaths[t]) if t < len(deaths) else 0.0
            hz[t] = (d / alive) if alive > 0 else 0.0
            alive -= d
        h_coh.append(hz)
    # Per-group age distribution w_k(a) over ages 0..100 (Mataac Table 1).
    _age_grid = np.arange(0, 101, dtype=int)
    _weights = []
    for k in range(len(_MATAAC_DEATH_COUNTS)):
        w = np.exp(-0.5 * ((_age_grid - _MATAAC_GROUP_MEAN_AGE[k])
                           / _MATAAC_GROUP_AGE_SD[k]) ** 2)
        _weights.append(w / w.sum())
    for sex in ("M", "F"):
        for pop in ("Euro-American", "Asian"):
            # mu_m(a) for every age reachable within the observation window.
            mu_age = np.array([
                get_mortality_rate(int(a), sex, pop)
                for a in range(0, MAX_ATTAINED_AGE_ICRP152 + 1)
            ], dtype=float)
            for k in range(len(_MATAAC_DEATH_COUNTS)):
                w = _weights[k]
                base = np.array([
                    float((w * mu_age[np.minimum(_age_grid + t,
                                                 MAX_ATTAINED_AGE_ICRP152)]).sum())
                    for t in range(_MATAAC_WINDOW_YEARS)
                ], dtype=float)
                _MATAAC_EXCESS[(sex, pop, k)] = np.maximum(0.0, h_coh[k] - base)


def _require_population(population) -> str:
    """Return a concrete population label or raise ``ValueError``.

    Population-dependent tables (baseline incidence, cancer-free survival,
    kernels, onset CDFs) have no valid answer for the mixed label
    ``"Euro-American/Asian"`` or for missing/unknown values: callers must
    resolve the per-exam population upstream.
    """
    if population in ("Euro-American", "Asian"):
        return population
    raise ValueError(
        f"Unknown population {population!r}; expected Euro-American or Asian"
    )


def mataac_survival_correction(
    exposure_age: int, sex_code: str, cum_dose_sv: float,
    population: str = "Euro-American",
) -> float:
    """Age-resolved REIC survival-correction factor for a frail patient.

    Returns the ratio of the lifetime all-cancer incidence probability under
    the clinical-cohort cancer-free survival to that under the standard ICRP
    152 survival, summed from the exposure age ``e`` to the maximum attained
    age and weighted by all-cancer incidence ``mu_iac(a)``::

        c = Sum_a S_clin(a|e) mu_iac(a) / Sum_a S_icrp(a|e) mu_iac(a).

    The clinical hazard is the ICRP hazard plus the empirical dose-group excess
    over the observed window, ``mu_clinical(n) = mu_bar(n) + excess[n-e]`` for
    ``0 <= n-e < window`` and ``mu_bar(n)`` afterwards (Mataac et al. 2024).
    Delivered as the per-exam ``surv_correction`` multiplier
    (vectorized.build_exam_arrays).  Returns 1.0 for the reference stratum
    (<10 mSv).  For unknown sex the M/F factors are averaged equally.
    """
    k = _mataac_stratum_index(cum_dose_sv)
    if k < 0:
        return 1.0
    pop = _require_population(population)
    e = int(exposure_age)
    cached = _MATAAC_CORRECTION_CACHE.get((e, sex_code, pop, k))
    if cached is not None:
        return cached
    if sex_code not in ("M", "F"):
        val = 0.5 * (
            mataac_survival_correction(e, "M", cum_dose_sv, pop)
            + mataac_survival_correction(e, "F", cum_dose_sv, pop)
        )
        _MATAAC_CORRECTION_CACHE[(e, sex_code, pop, k)] = val
        return val
    _ensure_cancer_free_surv_prefix()
    _ensure_mataac_excess_hazard()
    amax = MAX_ATTAINED_AGE_ICRP152
    e = min(max(0, e), amax)
    a = np.arange(e, amax + 1, dtype=int)
    # Standard ICRP cancer-free survival from e (via the precomputed prefix).
    pref_icrp = _CANCER_FREE_SURV_PREFIX[(sex_code, pop)]
    s_icrp = np.exp(pref_icrp[a] - pref_icrp[e])
    # Clinical survival: ICRP hazard plus the windowed empirical excess.
    mu_bar = _ensure_mu_bar(sex_code, pop)
    h_clin = mu_bar[a].copy()
    excess = _MATAAC_EXCESS[(sex_code, pop, k)]
    w = min(_MATAAC_WINDOW_YEARS, h_clin.shape[0])
    h_clin[:w] = np.clip(h_clin[:w] + excess[:w], 0.0, 1.0 - 1e-10)
    log_s_clin = np.zeros(h_clin.shape[0], dtype=float)
    np.cumsum(np.log(1.0 - h_clin)[:-1], out=log_s_clin[1:])
    s_clin = np.exp(log_s_clin)
    mu_iac = np.array([_get_all_cancer_incidence_rate(int(x), sex_code, pop)
                       for x in a], dtype=float)
    den = float((s_icrp * mu_iac).sum())
    num = float((s_clin * mu_iac).sum())
    val = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 1.0
    _MATAAC_CORRECTION_CACHE[(e, sex_code, pop, k)] = val
    return val


_LIFETIME_BASELINE_CACHE: dict = {}


def lifetime_baseline_cancer_incidence_ICRP152(
    sex_code: str = "U",
    population: str = "Euro-American",
) -> float:
    """Cumulative lifetime probability of developing *any* cancer under the
    ICRP 152 Annex A reference rates.

    Used by ``analysis.format_summary`` to report the
    population-baseline cancer probability that a member of the simulated
    cohort would face independently of any radiation exposure.

    Discrete cumulative-incidence formula with competing-risk correction::

        P(cancer in lifetime) = Sum_{a = 0..120} S_cf(a | 0) * mu_iac(a)

    where:

    * ``mu_iac(a)`` = sum of site-specific cancer incidence rates at
      attained age ``a`` (ICRP 152 Annex A Tables A.1-A.4, via
      :func:`_get_all_cancer_incidence_rate`).
    * ``S_cf(a | 0)`` = probability of surviving to age ``a`` without
      cancer and without death, using the ICRP 152 cancer-free survival
      model ``mu_bar(x) = mu_m(x) - mu_mac(x) + mu_iac(x)`` (paras 46-47),
      drawn from the cached log-survival prefix in
      :data:`_CANCER_FREE_SURV_PREFIX`.

    For unknown sex (``sex_code`` not ``"M"`` / ``"F"``) the returned value
    is the arithmetic mean of the M and F probabilities for the given
    population.  ``population`` must be a concrete label — the mixed
    ``"Euro-American/Asian"`` run-level label raises ``ValueError``.

    Typical values (Tables A.1/A.2 + A.5/A.6):

    * ``M, Euro-American`` ~ 0.41-0.44
    * ``F, Euro-American`` ~ 0.37-0.40
    * ``M, Asian``         ~ 0.26-0.29
    * ``F, Asian``         ~ 0.22-0.25

    Results are cached in :data:`_LIFETIME_BASELINE_CACHE` so repeated
    calls are free.
    """
    pop = _require_population(population)
    sex = sex_code if sex_code in ("M", "F") else "U"
    cached = _LIFETIME_BASELINE_CACHE.get((sex, pop))
    if cached is not None:
        return cached
    _ensure_cancer_free_surv_prefix()
    if sex == "U":
        val = 0.5 * (
            lifetime_baseline_cancer_incidence_ICRP152("M", pop)
            + lifetime_baseline_cancer_incidence_ICRP152("F", pop)
        )
        _LIFETIME_BASELINE_CACHE[(sex, pop)] = val
        return val
    pref = _CANCER_FREE_SURV_PREFIX[(sex, pop)]
    # pref[a] = Σ_{t=0}^{a-1} log(1 − μ̄(t)); S_cf(a|0) = exp(pref[a]),
    # noting pref[0] == 0 so S_cf(0|0) == 1.
    s_cf = np.exp(pref)
    mu_iac = np.array(
        [_get_all_cancer_incidence_rate(a, sex, pop) for a in range(121)],
        dtype=float,
    )
    val = float(np.clip((s_cf * mu_iac).sum(), 0.0, 1.0))
    _LIFETIME_BASELINE_CACHE[(sex, pop)] = val
    return val


# cancer_risk_params: ERR/EAR model coefficients, lethality, QOL, and life-lost factors
# Sources:
#   ERR model parameters (beta, alpha1, alpha2): ICRP 152 Table 3.2
#   EAR model parameters (beta, alpha1, alpha2): ICRP 152 Table 3.3
#   Breast EAR model: ICRP 152 para. 38 (Preston et al. 2002 pooled analysis)
#   ERR:EAR weights: ICRP 152 Table 3.1
#   Lethality (k), q_min, relative life-lost (l_rel): ICRP 152 Table 3.9
#   Bone nominal risk: ICRP 60 via BEIR IV (7 cases/10,000/Gy)
#   Skin nominal risk: ICRP 59 (1000 cases/10,000/Gy, no DDREF per ICRP 152 para. 64)
cancer_risk_params = {
    "All solid": {
        # Reference aggregate for ALL solid cancers combined (ICRP 103 Table A.4.1).
        # NOT a modeled risk site — not in CANCER_TO_ORGAN_DOSE_KEY.
        # Individual organs use their own site-specific entries below.
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.35, "F": 0.58},
        "ERR_alpha1": {"M": -0.19, "F": -0.19},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 43.20 / 10000, "F": 59.83 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.5,
        "l_rel": 1.0,
        "q_min": 0.1,
    },
    "Oesophagus": {
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.40, "F": 0.65},
        "ERR_alpha1": {"M": -0.19, "F": -0.19},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 0.48 / 10000, "F": 0.66 / 10000},
        "EAR_alpha1": {"M": 0.49, "F": 0.49},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.93,
        "l_rel": 0.87,
        "q_min": 0.1,
    },
    "Stomach": {
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.23, "F": 0.38},
        "ERR_alpha1": {"M": -0.19, "F": -0.19},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 6.63 / 10000, "F": 9.18 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.83,
        "l_rel": 0.88,
        "q_min": 0.1,
    },
    "Colon": {
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.68, "F": 0.33},
        "ERR_alpha1": {"M": -0.19, "F": -0.19},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 5.76 / 10000, "F": 2.40 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.48,
        "l_rel": 0.97,
        "q_min": 0.1,
    },
    "Liver": {
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.25, "F": 0.40},
        "ERR_alpha1": {"M": -0.19, "F": -0.19},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 4.18 / 10000, "F": 1.30 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.95,
        "l_rel": 0.88,
        "q_min": 0.1,
    },
    "Lung": {
        "ERR_weight": 0.3,
        "EAR_weight": 0.7,
        "ERR_beta_base": {"M": 0.29, "F": 1.36},
        "ERR_alpha1": {"M": 0.16, "F": 0.16},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 6.47 / 10000, "F": 8.97 / 10000},
        "EAR_alpha1": {"M": 0.010, "F": 0.010},
        "EAR_alpha2": {"M": 4.25, "F": 4.25},
        "k_lethality": 0.89,
        "l_rel": 0.80,
        "q_min": 0.1,
    },
    "Breast": {
        "ERR_weight": 0.0,
        "EAR_weight": 1.0,
        "ERR_beta_base": {"M": 0.0, "F": 0.0},
        "ERR_alpha1": {"M": 0.0, "F": 0.0},
        "ERR_alpha2": {"M": 0.0, "F": 0.0},
        "EAR_beta_base": {"M": 0.0, "F": 25.3 / 10000.0},
        "EAR_alpha1": {"M": 0.0, "F": -0.5},
        "EAR_alpha2": {"M": 0.0, "F": 3.5},
        "k_lethality": 0.29,
        "l_rel": 1.29,
        "q_min": 0.1,
        "use_ear_model": True,
    },
    "Ovary": {
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.0, "F": 0.32},
        "ERR_alpha1": {"M": 0.0, "F": -0.19},
        "ERR_alpha2": {"M": 0.0, "F": -1.65},
        "EAR_beta_base": {"M": 0, "F": 1.47 / 10000},
        "EAR_alpha1": {"M": 0.0, "F": -0.27},
        "EAR_alpha2": {"M": 0.0, "F": 2.38},
        "k_lethality": 0.57,
        "l_rel": 1.12,
        "q_min": 0.1,
    },
    "Bladder": {
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.67, "F": 1.10},
        "ERR_alpha1": {"M": -0.19, "F": -0.19},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 2.00 / 10000, "F": 2.77 / 10000},
        "EAR_alpha1": {"M": -0.12, "F": -0.12},
        "EAR_alpha2": {"M": 6.39, "F": 6.39},
        "k_lethality": 0.29,
        "l_rel": 0.71,
        "q_min": 0.1,
    },
    "Thyroid": {
        "ERR_weight": 1.0,
        "EAR_weight": 0.0,
        "ERR_beta_base": {"M": 0.53, "F": 1.05},
        "ERR_alpha1": {"M": -0.82, "F": -0.82},
        "ERR_alpha2": {"M": 0.0, "F": 0.0},
        "EAR_beta_base": {"M": 0.0, "F": 0.0},
        "EAR_alpha1": {"M": 0.0, "F": 0.0},
        "EAR_alpha2": {"M": 0.0, "F": 0.0},
        "k_lethality": 0.07,
        "l_rel": 1.29,
        "q_min": 0.2,
    },
    "Remainder": {
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.22, "F": 0.17},
        "ERR_alpha1": {"M": -0.42, "F": -0.42},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 7.55 / 10000, "F": 10.45 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.49,
        "l_rel": 1.03,
        "q_min": 0.1,
    },
    "Brain": {
        # ICRP 103/152 Table 3.1 "Other solid" (wT=0.01 per ICRP 103 Table B.2).
        # Uses same LSS-derived coefficients as Remainder; Remainder baseline for ERR.
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.22, "F": 0.17},
        "ERR_alpha1": {"M": -0.42, "F": -0.42},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 7.55 / 10000, "F": 10.45 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.49,
        "l_rel": 1.03,
        "q_min": 0.1,
    },
    "Kidney": {
        # ICRP 103/152 Table 3.1 "Other solid" (part of Remainder tissue group).
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.22, "F": 0.17},
        "ERR_alpha1": {"M": -0.42, "F": -0.42},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 7.55 / 10000, "F": 10.45 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.49,
        "l_rel": 1.03,
        "q_min": 0.1,
    },
    "Pancreas": {
        # ICRP 103/152 Table 3.1 "Other solid" (part of Remainder tissue group).
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.22, "F": 0.17},
        "ERR_alpha1": {"M": -0.42, "F": -0.42},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 7.55 / 10000, "F": 10.45 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.49,
        "l_rel": 1.03,
        "q_min": 0.1,
    },
    "Salivary_Glands": {
        # ICRP 103 §A.158: salivary glands carry a separate w_T = 0.01 because
        # their cancer risk is judged greater than other Remainder tissues, but
        # ICRP 152 has no salivary-gland-specific risk model — the cancer is
        # absorbed into "Other solid" (Table 3.1, 50%ERR:50%EAR).  Coefficients
        # therefore mirror Brain / Kidney / Pancreas / Remainder so that the
        # site contributes to the "Other solid" REIC integral via the shared
        # mean-dose pathway.
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.22, "F": 0.17},
        "ERR_alpha1": {"M": -0.42, "F": -0.42},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 7.55 / 10000, "F": 10.45 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.49,
        "l_rel": 1.03,
        "q_min": 0.1,
    },
    "Leukaemia": {
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.0, "F": 0.0},
        "ERR_alpha1": {"M": 0.0, "F": 0.0},
        "ERR_alpha2": {"M": 0.0, "F": 0.0},
        "EAR_beta_base": {"M": 0.0, "F": 0.0},
        "EAR_alpha1": {"M": 0.0, "F": 0.0},
        "EAR_alpha2": {"M": 0.0, "F": 0.0},
        "k_lethality": 0.67,
        "l_rel": 1.63,
        "q_min": 0.1,
    },
    "Bone": {
        "ERR_weight": 0.0,
        "EAR_weight": 1.0,
        "ERR_beta_base": {"M": 0.0, "F": 0.0},
        "ERR_alpha1": {"M": 0.0, "F": 0.0},
        "ERR_alpha2": {"M": 0.0, "F": 0.0},
        # ICRP 103 Table A.4.4 nominal bone cancer coefficient: 7 cases / 10,000
        # per Gy (BEIR IV, average bone dose, ICRP 60 derivation).  This is also
        # the value retained in ICRP 152 Table 3.7 for reference detriment.
        # NOTE: ICRP 152 §3.1.4 reports a recalculated bone-surface nominal of
        # 5/10,000/Gy from updated dosimetry; the reference-detriment column of
        # Table 3.7 keeps 7/10,000/Gy for continuity with ICRP 103.  The 7/10,000
        # value is used here for that consistency.  Switch to 5.0 / 10000 to
        # adopt the §3.1.4 recalculation as a sensitivity test.
        "EAR_beta_base": {"M": 7.0 / 10000, "F": 7.0 / 10000},
        "EAR_alpha1": {"M": 0.0, "F": 0.0},
        "EAR_alpha2": {"M": 0.0, "F": 0.0},
        "k_lethality": 0.45,
        "l_rel": 1.0,
        "q_min": 0.1,
    },
    "Skin": {
        # ICRP Table 3.1: 100% ERR (constant relative risk model, ICRP Pub. 59).
        # Nominal lifetime risk = 1000 cases / 10,000 / Gy stored as ERR_beta_base.
        # No DDREF applied (ICRP 152 §3.1.5 para 64). No year-by-year integration
        # (bypassed in compute_lifetime_risk — total lifetime nominal, not annual rate).
        "ERR_weight": 1.0,
        "EAR_weight": 0.0,
        # 1000 / 10,000 per Gy = 0.1 per Gy (constant, no age/sex modulation)
        "ERR_beta_base": {"M": 1000.0 / 10000, "F": 1000.0 / 10000},
        "ERR_alpha1": {"M": 0.0, "F": 0.0},
        "ERR_alpha2": {"M": 0.0, "F": 0.0},
        "EAR_beta_base": {"M": 0.0, "F": 0.0},
        "EAR_alpha1": {"M": 0.0, "F": 0.0},
        "EAR_alpha2": {"M": 0.0, "F": 0.0},
        "k_lethality": 0.002,
        "l_rel": 1.0,
        "q_min": 0.0,
    },
    "General": {
        # Fallback for any unrecognised cancer type — uses the Remainder (Other
        # solid) parameters per ICRP 103/152 Table 3.1.
        "ERR_weight": 0.5,
        "EAR_weight": 0.5,
        "ERR_beta_base": {"M": 0.22, "F": 0.17},
        "ERR_alpha1": {"M": -0.42, "F": -0.42},
        "ERR_alpha2": {"M": -1.65, "F": -1.65},
        "EAR_beta_base": {"M": 7.55 / 10000, "F": 10.45 / 10000},
        "EAR_alpha1": {"M": -0.27, "F": -0.27},
        "EAR_alpha2": {"M": 2.38, "F": 2.38},
        "k_lethality": 0.49,
        "l_rel": 1.03,
        "q_min": 0.1,
    },
}


def lookup_cancer_params(cancer_type: str) -> dict:
    """Return cancer_risk_params entry for the given cancer type.
    Falls back to 'Remainder' for unrecognised sites.
    """
    if cancer_type in cancer_risk_params:
        return cancer_risk_params[cancer_type]
    return cancer_risk_params["Remainder"]


_CANCER_PARAMS_CACHE: Dict[str, Tuple[float, float, float, float]] = {}

def get_cancer_params_cached(site: str) -> Tuple[float, float, float, float]:
    """Return (k_lethality, q_min, l_rel, detriment_factor) for a cancer site.
    Caches results to avoid repeated dict lookups in the MC hot loop.
    """
    cached = _CANCER_PARAMS_CACHE.get(site)
    if cached is not None:
        return cached
    p = lookup_cancer_params(site) or {}
    k = float(p.get("k_lethality", 0.0))
    q = float(p.get("q_min", 0.1))
    l_rel = float(p.get("l_rel", 1.0))
    q_nonfatal = k + q * (1.0 - k)
    det_factor = (k + q_nonfatal * (1.0 - k)) * l_rel
    result = (k, q, l_rel, det_factor)
    _CANCER_PARAMS_CACHE[site] = result
    return result


def get_leukaemia_EAR_params(age_at_exposure: float, sex: str):
    """Return leukaemia EAR coefficients (beta, alpha) for (age, sex).

    Source: ICRP 152 Table 3.4.
    ``beta`` is the linear-term coefficient (per 10,000 / yr / Gy).
    ``alpha`` is the time-since-exposure power (negative = decreasing
    risk with time).
    """
    # Returns (beta, alpha) for leukaemia EAR model
    # Source: ICRP 152 Table 3.4 (EAR = beta*(d+1.53*d^2)*(t/25)^alpha)
    # beta: linear-term coefficient (per 10,000/year/Gy)
    # alpha: power of time-since-exposure (negative = decreasing risk with time)
    beta = 0.5
    alpha = 0.5
    leukaemia_params = [
        (0, 19, 0.58, -1.54, 0.41, -1.06),
        (20, 39, 0.96, -0.69, 0.69, -0.21),
        (40, 200, 2.03, 0.17, 1.45, 0.66),
    ]
    for age_min, age_max, b_m, a_m, b_f, a_f in leukaemia_params:
        if age_min <= age_at_exposure <= age_max:
            if sex == "M":
                beta, alpha = b_m, a_m
            elif sex == "F":
                beta, alpha = b_f, a_f
            else:
                beta, alpha = 0.5 * (b_m + b_f), 0.5 * (a_m + a_f)
            break
    return beta, alpha


LEUKAEMIA_QUAD_LINEAR_RATIO: float = 1.53  # ICRP 152 para 41 quadratic-to-linear ratio


def leukaemia_quadratic_share(
    age_at_exposure: np.ndarray,
    sex_code: np.ndarray,
    dose_gy: np.ndarray,
) -> np.ndarray:
    """Fraction of leukaemia risk carried by the quadratic dose term.

    ICRP 152 para 41 models leukaemia EAR as ``beta*d + 1.53*d**2`` times a
    time-since-exposure factor.  That time factor, the baseline rates and the
    survival weighting all multiply the linear and quadratic terms identically,
    so the quadratic share of the integrated REIC collapses to a closed form in
    dose alone::

        s = 1.53*d / (beta + 1.53*d)

    No risk integration is required.  ``beta`` is the ICRP 152 Table 3.4
    linear coefficient looked up per (age at exposure, sex).

    The share matters because a dose reduction to a fraction ``f`` of baseline
    removes ``(1-f)`` of the linear term but ``(1-f**2)`` of the quadratic term.

    Parameters
    ----------
    age_at_exposure, sex_code, dose_gy
        Equal-length arrays.  ``sex_code`` holds "M" / "F" / anything else
        (treated as sex-averaged, matching get_leukaemia_EAR_params).
        ``dose_gy`` is the red-marrow absorbed dose in Gy.

    Returns
    -------
    np.ndarray
        Share in [0, 1); zero wherever dose is zero.
    """
    age = np.asarray(age_at_exposure, dtype=float)
    dose = np.maximum(np.asarray(dose_gy, dtype=float), 0.0)
    sex = np.asarray(sex_code, dtype=object)

    # ICRP 152 Table 3.4 linear coefficients by age-at-exposure band.
    beta_m = np.select([age < 20.0, age < 40.0], [0.58, 0.96], default=2.03)
    beta_f = np.select([age < 20.0, age < 40.0], [0.41, 0.69], default=1.45)
    is_m = sex == "M"
    is_f = sex == "F"
    beta = np.where(is_m, beta_m, np.where(is_f, beta_f, 0.5 * (beta_m + beta_f)))

    quad = LEUKAEMIA_QUAD_LINEAR_RATIO * dose
    return np.where(dose > 0.0, quad / (beta + quad), 0.0)


def get_female_breast_EAR_kernel_perGy(exposure_age: float) -> np.ndarray:
    """Female-breast excess-absolute-rate kernel per gray of exposure.

    ``exposure_age`` is the age at exposure in years.  Returns an array of
    attained ages 0..120 holding the EAR coefficient at each age, as excess
    cases per person-year per Gy (the table values are stored already
    divided by 10 000): zero before the solid-cancer minimum
    latency, the EAR model value afterwards, capped at the risk model's
    maximum attained age.  Coefficients default to the female values in
    ``cancer_risk_params["Breast"]`` (ICRP 152 Table 3.4).
    """
    ages_local = np.arange(121, dtype=float)
    a = ages_local.astype(float)
    e = float(exposure_age)
    breast_params = cancer_risk_params.get("Breast", {})
    beta_breast = breast_params.get("EAR_beta_base", {}).get("F", 25.3 / 10000.0)
    alpha1_breast = breast_params.get("EAR_alpha1", {}).get("F", -0.5)
    kernel = np.zeros_like(ages_local, dtype=float)
    t = a - e
    mask = t >= _LATENCY_PARAMS["Solid"]  # minimum latency from configure_latency() / ICRP 152 §4.2 para 97
    if not np.any(mask):
        return kernel.astype(float)
    a_eff = a[mask]
    a_eff_safe = np.where(a_eff == 0, 1, a_eff)
    ear_per_gy = beta_breast * np.exp(
        alpha1_breast * (e - 30.0) / 10.0 + 3.5 * np.log(a_eff_safe / 70.0)
    )
    post_mask = a_eff > 50.0
    if np.any(post_mask):
        ear_per_gy[post_mask] = beta_breast * np.exp(
            alpha1_breast * (e - 30.0) / 10.0
            + 3.5 * np.log(a_eff_safe[post_mask] / 70.0)
            - 2.5 * np.log(a_eff_safe[post_mask] / 50.0)
        )
    kernel[mask] = ear_per_gy
    # Apply the ICRP 152 Table 4.1 integration cap (94 y for REIC, 99 y for
    # LAR) — same dynamic cap as solid_kernel_perGy.  Breast uses an EAR-only
    # model in ICRP 152 Table 3.4 / para 38 (Preston et al. 2002 pooled
    # analysis).
    _amax = _max_attained_age_for_risk_model()
    if _amax < len(kernel) - 1:
        kernel[_amax + 1:] = 0.0
    return kernel.astype(float)


def _get_ageonly_arrays(cancer_type: str, sex_code: str):
    global _AGEONLY_BUILT
    if not _AGEONLY_BUILT:
        _build_ageonly_factors()
        _AGEONLY_BUILT = True
    if sex_code in ("M", "F"):
        err_base = _ERR_AGEONLY[sex_code].get(cancer_type, _ERR_AGEONLY[sex_code].get("Remainder"))
        ear_base = _EAR_AGEONLY[sex_code].get(cancer_type, _EAR_AGEONLY[sex_code].get("Remainder"))
        return err_base.astype(float), ear_base.astype(float)
    err_m = _ERR_AGEONLY["M"].get(cancer_type, _ERR_AGEONLY["M"].get("Remainder"))
    err_f = _ERR_AGEONLY["F"].get(cancer_type, _ERR_AGEONLY["F"].get("Remainder"))
    ear_m = _EAR_AGEONLY["M"].get(cancer_type, _EAR_AGEONLY["M"].get("Remainder"))
    ear_f = _EAR_AGEONLY["F"].get(cancer_type, _EAR_AGEONLY["F"].get("Remainder"))
    return (0.5 * (err_m + err_f)).astype(float), (0.5 * (ear_m + ear_f)).astype(float)


# BASELINE_RATES_ICRP: Baseline cancer incidence rates (cases per 100,000 persons per year)
# Source: ICRP 152 (2022) Annex A
#   Table A.1 – Euro-American male (modified from ICRP 103 Table A.4.11)
#   Table A.2 – Euro-American female (modified from ICRP 103 Table A.4.10)
#   Table A.3 – Asian male (modified from ICRP 103 Table A.4.15)
#   Table A.4 – Asian female (modified from ICRP 103 Table A.4.14)
# Population data from Cancer Incidence in Five Continents Vol. VIII (Parkin et al., 2002)
# 5-year age groups: 0-4, 5-9, ..., 85-89, >=90 (keyed by lower bound)
BASELINE_RATES_ICRP = {
    "M": {
        # ICRP 152 Table A.1: Euro-American male cancer incidence rates (cases per 100,000/year)
        "Euro-American": {
            "Oesophagus": {
                0: 0.00, 5: 0.00, 10: 0.00, 15: 0.00, 20: 0.02, 25: 0.09, 30: 0.21, 35: 0.64, 40: 1.94, 45: 4.26,
                50: 9.47, 55: 15.68, 60: 24.79, 65: 33.72, 70: 46.59, 75: 49.57, 80: 55.88, 85: 59.36, 90: 49.36,
            },
            "Stomach": {
                0: 0.01, 5: 0.00, 10: 0.00, 15: 0.06, 20: 0.10, 25: 0.27, 30: 0.82, 35: 1.45, 40: 3.27, 45: 6.02,
                50: 11.72, 55: 21.64, 60: 36.02, 65: 58.28, 70: 87.72, 75: 114.49, 80: 145.00, 85: 165.76, 90: 137.84,
            },
            "Colon": {
                0: 0.00, 5: 0.00, 10: 0.06, 15: 0.13, 20: 0.33, 25: 0.92, 30: 1.75, 35: 3.15, 40: 6.71, 45: 12.42,
                50: 25.26, 55: 47.90, 60: 84.67, 65: 129.65, 70: 185.35, 75: 248.89, 80: 310.36, 85: 316.71, 90: 335.18,
            },
            "Liver": {
                0: 0.62, 5: 0.10, 10: 0.05, 15: 0.10, 20: 0.15, 25: 0.22, 30: 0.32, 35: 0.72, 40: 2.06, 45: 3.12,
                50: 5.53, 55: 9.60, 60: 15.00, 65: 22.80, 70: 30.88, 75: 36.70, 80: 36.96, 85: 37.73, 90: 39.21,
            },
            "Lung": {
                0: 0.01, 5: 0.00, 10: 0.03, 15: 0.11, 20: 0.19, 25: 0.36, 30: 0.99, 35: 3.19, 40: 9.41, 45: 23.28,
                50: 56.22, 55: 108.53, 60: 189.00, 65: 304.06, 70: 400.78, 75: 456.24, 80: 459.96, 85: 404.07, 90: 337.79,
            },
            "Bladder": {
                0: 0.12, 5: 0.01, 10: 0.02, 15: 0.10, 20: 0.39, 25: 0.60, 30: 1.27, 35: 2.52, 40: 5.70, 45: 12.63,
                50: 25.29, 55: 46.07, 60: 79.67, 65: 132.28, 70: 184.53, 75: 229.94, 80: 275.56, 85: 266.44, 90: 376.32,
            },
            "Thyroid": {
                0: 0.00, 5: 0.05, 10: 0.13, 15: 0.43, 20: 0.77, 25: 1.54, 30: 1.47, 35: 1.78, 40: 2.15, 45: 2.83,
                50: 3.34, 55: 3.81, 60: 4.16, 65: 5.24, 70: 5.69, 75: 5.98, 80: 6.26, 85: 6.26, 90: 6.26,
            },
            "Non-CLL Leukaemia": {
                0: 7.77, 5: 3.80, 10: 3.07, 15: 2.73, 20: 1.98, 25: 2.33, 30: 2.80, 35: 3.20, 40: 3.81, 45: 4.85,
                50: 7.20, 55: 9.56, 60: 14.06, 65: 20.92, 70: 30.97, 75: 39.48, 80: 50.15, 85: 38.53, 90: 43.13,
            },
            "Remainder": {
                0: 11.94, 5: 6.02, 10: 5.89, 15: 10.17, 20: 18.86, 25: 28.54, 30: 38.54, 35: 48.20, 40: 64.71, 45: 105.91,
                50: 200.75, 55: 364.73, 60: 620.00, 65: 965.84, 70: 1313.52, 75: 1539.02, 80: 1693.11, 85: 1909.67, 90: 2009.03,
            },
            "Breast": {
                0: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0, 25: 0.0, 30: 0.0, 35: 0.0, 40: 0.0, 45: 0.0,
                50: 0.0, 55: 0.0, 60: 0.0, 65: 0.0, 70: 0.0, 75: 0.0, 80: 0.0, 85: 0.0, 90: 0.0,
            },
            "Ovary": {
                0: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0, 25: 0.0, 30: 0.0, 35: 0.0, 40: 0.0, 45: 0.0,
                50: 0.0, 55: 0.0, 60: 0.0, 65: 0.0, 70: 0.0, 75: 0.0, 80: 0.0, 85: 0.0, 90: 0.0,
            },
        },
        # ICRP 152 Table A.3: Asian male cancer incidence rates (cases per 100,000/year)
        "Asian": {
            "Oesophagus": {
                0: 0.00, 5: 0.00, 10: 0.00, 15: 0.00, 20: 0.00, 25: 0.10, 30: 0.13, 35: 0.80, 40: 2.94, 45: 7.05,
                50: 14.35, 55: 25.49, 60: 44.55, 65: 58.10, 70: 82.63, 75: 92.66, 80: 94.17, 85: 69.75, 90: 68.97,
            },
            "Stomach": {
                0: 0.08, 5: 0.05, 10: 0.06, 15: 0.33, 20: 0.81, 25: 2.29, 30: 5.05, 35: 14.96, 40: 29.51, 45: 47.43,
                50: 76.73, 55: 127.25, 60: 217.15, 65: 316.67, 70: 412.58, 75: 488.08, 80: 520.98, 85: 716.89, 90: 863.48,
            },
            "Colon": {
                0: 0.00, 5: 0.00, 10: 0.06, 15: 0.10, 20: 0.50, 25: 0.91, 30: 3.54, 35: 5.45, 40: 12.43, 45: 24.55,
                50: 39.96, 55: 72.34, 60: 119.83, 65: 162.08, 70: 186.30, 75: 214.56, 80: 222.27, 85: 326.54, 90: 422.02,
            },
            "Liver": {
                0: 0.74, 5: 0.24, 10: 0.33, 15: 0.13, 20: 0.70, 25: 1.67, 30: 3.60, 35: 11.41, 40: 21.68, 45: 36.58,
                50: 54.82, 55: 95.29, 60: 170.87, 65: 195.63, 70: 192.09, 75: 183.31, 80: 187.30, 85: 232.57, 90: 215.09,
            },
            "Lung": {
                0: 0.03, 5: 0.05, 10: 0.07, 15: 0.14, 20: 0.41, 25: 0.51, 30: 2.30, 35: 5.09, 40: 14.83, 45: 23.27,
                50: 44.64, 55: 80.55, 60: 176.67, 65: 317.21, 70: 439.32, 75: 509.83, 80: 540.57, 85: 682.18, 90: 608.83,
            },
            "Bladder": {
                0: 0.03, 5: 0.00, 10: 0.00, 15: 0.06, 20: 0.31, 25: 0.59, 30: 0.81, 35: 2.20, 40: 3.59, 45: 5.14,
                50: 10.69, 55: 17.08, 60: 33.03, 65: 55.42, 70: 73.66, 75: 108.13, 80: 120.05, 85: 158.97, 90: 264.33,
            },
            "Thyroid": {
                0: 0.00, 5: 0.02, 10: 0.23, 15: 0.59, 20: 0.74, 25: 0.99, 30: 1.16, 35: 1.67, 40: 2.15, 45: 3.17,
                50: 2.82, 55: 2.86, 60: 3.84, 65: 5.13, 70: 5.16, 75: 4.68, 80: 4.35, 85: 4.35, 90: 4.35,
            },
            "Non-CLL Leukaemia": {
                0: 5.09, 5: 4.73, 10: 3.31, 15: 3.51, 20: 2.30, 25: 2.89, 30: 3.49, 35: 2.93, 40: 3.71, 45: 5.30,
                50: 6.67, 55: 9.07, 60: 12.55, 65: 18.61, 70: 25.46, 75: 28.83, 80: 28.87, 85: 44.17, 90: 44.84,
            },
            "Remainder": {
                0: 9.42, 5: 4.18, 10: 4.73, 15: 5.85, 20: 6.21, 25: 9.82, 30: 14.58, 35: 24.00, 40: 44.42, 45: 66.56,
                50: 95.22, 55: 143.58, 60: 253.77, 65: 358.35, 70: 486.41, 75: 579.55, 80: 686.22, 85: 1032.39, 90: 1295.00,
            },
            "Breast": {
                0: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0, 25: 0.0, 30: 0.0, 35: 0.0, 40: 0.0, 45: 0.0,
                50: 0.0, 55: 0.0, 60: 0.0, 65: 0.0, 70: 0.0, 75: 0.0, 80: 0.0, 85: 0.0, 90: 0.0,
            },
            "Ovary": {
                0: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0, 25: 0.0, 30: 0.0, 35: 0.0, 40: 0.0, 45: 0.0,
                50: 0.0, 55: 0.0, 60: 0.0, 65: 0.0, 70: 0.0, 75: 0.0, 80: 0.0, 85: 0.0, 90: 0.0,
            },
        },
    },
    "F": {
        # ICRP 152 Table A.2: Euro-American female cancer incidence rates (cases per 100,000/year)
        "Euro-American": {
            "Oesophagus": {
                0: 0.00, 5: 0.00, 10: 0.00, 15: 0.01, 20: 0.02, 25: 0.04, 30: 0.10, 35: 0.13, 40: 0.50, 45: 1.07,
                50: 2.42, 55: 5.27, 60: 7.92, 65: 11.24, 70: 16.96, 75: 21.52, 80: 26.77, 85: 34.82, 90: 23.34,
            },
            "Stomach": {
                0: 0.01, 5: 0.01, 10: 0.04, 15: 0.08, 20: 0.09, 25: 0.27, 30: 0.75, 35: 1.17, 40: 2.28, 45: 3.31,
                50: 5.02, 55: 8.76, 60: 14.26, 65: 21.99, 70: 33.48, 75: 47.53, 80: 65.22, 85: 76.14, 90: 73.73,
            },
            "Colon": {
                0: 0.01, 5: 0.03, 10: 0.11, 15: 0.25, 20: 0.36, 25: 0.83, 30: 1.27, 35: 3.27, 40: 6.00, 45: 11.90,
                50: 21.92, 55: 41.98, 60: 63.80, 65: 94.46, 70: 138.10, 75: 177.76, 80: 234.14, 85: 241.25, 90: 266.50,
            },
            "Liver": {
                0: 0.32, 5: 0.03, 10: 0.04, 15: 0.07, 20: 0.09, 25: 0.17, 30: 0.24, 35: 0.39, 40: 0.64, 45: 1.42,
                50: 2.43, 55: 4.07, 60: 6.73, 65: 9.82, 70: 14.11, 75: 17.32, 80: 22.02, 85: 21.66, 90: 16.94,
            },
            "Lung": {
                0: 0.01, 5: 0.04, 10: 0.02, 15: 0.04, 20: 0.19, 25: 0.39, 30: 1.04, 35: 3.20, 40: 8.29, 45: 20.20,
                50: 40.44, 55: 67.32, 60: 106.00, 65: 154.72, 70: 190.74, 75: 191.05, 80: 166.82, 85: 127.96, 90: 76.51,
            },
            "Breast": {
                0: 0.02, 5: 0.00, 10: 0.01, 15: 0.12, 20: 1.19, 25: 7.17, 30: 23.53, 35: 54.12, 40: 107.57, 45: 183.33,
                50: 243.57, 55: 263.17, 60: 298.07, 65: 305.57, 70: 328.61, 75: 339.09, 80: 365.99, 85: 335.97, 90: 382.23,
            },
            "Ovary": {
                0: 0.05, 5: 0.23, 10: 0.69, 15: 1.77, 20: 2.89, 25: 4.03, 30: 5.82, 35: 9.00, 40: 13.73, 45: 24.54,
                50: 34.33, 55: 41.39, 60: 49.35, 65: 55.60, 70: 62.04, 75: 61.42, 80: 56.31, 85: 49.39, 90: 38.63,
            },
            "Bladder": {
                0: 0.06, 5: 0.00, 10: 0.00, 15: 0.07, 20: 0.19, 25: 0.31, 30: 0.50, 35: 0.98, 40: 1.85, 45: 4.05,
                50: 7.90, 55: 13.25, 60: 22.38, 65: 33.45, 70: 47.83, 75: 56.59, 80: 68.67, 85: 83.68, 90: 54.69,
            },
            "Thyroid": {
                0: 0.01, 5: 0.08, 10: 0.54, 15: 1.80, 20: 3.87, 25: 5.60, 30: 6.38, 35: 7.00, 40: 7.20, 45: 8.48,
                50: 8.07, 55: 7.97, 60: 7.16, 65: 7.79, 70: 8.53, 75: 8.13, 80: 8.73, 85: 8.73, 90: 8.73,
            },
            "Remainder": {
                0: 10.46, 5: 4.86, 10: 5.12, 15: 6.82, 20: 13.07, 25: 24.77, 30: 36.43, 35: 50.07, 40: 67.41, 45: 96.90,
                50: 146.31, 55: 210.13, 60: 276.08, 65: 353.94, 70: 439.19, 75: 507.31, 80: 550.65, 85: 688.28, 90: 765.31,
            },
            "Non-CLL Leukaemia": {
                0: 6.92, 5: 3.05, 10: 2.15, 15: 2.19, 20: 1.59, 25: 1.61, 30: 1.86, 35: 2.27, 40: 3.41, 45: 3.72,
                50: 5.28, 55: 6.59, 60: 9.82, 65: 12.96, 70: 17.72, 75: 21.96, 80: 26.88, 85: 26.91, 90: 44.90,
            },
        },
        # ICRP 152 Table A.4: Asian female cancer incidence rates (cases per 100,000/year)
        "Asian": {
            "Oesophagus": {
                0: 0.00, 5: 0.00, 10: 0.00, 15: 0.00, 20: 0.00, 25: 0.11, 30: 0.05, 35: 0.20, 40: 0.65, 45: 1.15,
                50: 2.17, 55: 6.38, 60: 12.35, 65: 17.66, 70: 28.42, 75: 34.69, 80: 38.69, 85: 28.65, 90: 27.96,
            },
            "Stomach": {
                0: 0.00, 5: 0.00, 10: 0.00, 15: 0.20, 20: 0.95, 25: 2.41, 30: 8.54, 35: 15.25, 40: 24.58, 45: 27.18,
                50: 34.98, 55: 52.62, 60: 75.78, 65: 113.21, 70: 159.53, 75: 195.44, 80: 260.54, 85: 284.69, 90: 354.64,
            },
            "Colon": {
                0: 0.00, 5: 0.00, 10: 0.00, 15: 0.30, 20: 0.26, 25: 1.52, 30: 2.40, 35: 5.53, 40: 9.34, 45: 16.76,
                50: 28.27, 55: 44.43, 60: 71.50, 65: 89.08, 70: 126.39, 75: 138.59, 80: 152.09, 85: 174.60, 90: 244.83,
            },
            "Liver": {
                0: 0.41, 5: 0.15, 10: 0.15, 15: 0.11, 20: 0.22, 25: 0.32, 30: 0.92, 35: 2.25, 40: 3.69, 45: 5.89,
                50: 11.12, 55: 21.21, 60: 46.70, 65: 75.39, 70: 84.23, 75: 96.89, 80: 111.69, 85: 114.47, 90: 113.01,
            },
            "Lung": {
                0: 0.00, 5: 0.00, 10: 0.05, 15: 0.12, 20: 0.14, 25: 0.86, 30: 1.26, 35: 2.97, 40: 7.70, 45: 12.55,
                50: 19.96, 55: 34.36, 60: 63.49, 65: 89.27, 70: 145.22, 75: 171.64, 80: 176.17, 85: 184.59, 90: 193.15,
            },
            "Breast": {
                0: 0.00, 5: 0.00, 10: 0.00, 15: 0.00, 20: 0.51, 25: 3.62, 30: 14.77, 35: 38.85, 40: 67.94, 45: 86.55,
                50: 81.36, 55: 76.81, 60: 88.33, 65: 86.57, 70: 84.42, 75: 82.73, 80: 82.34, 85: 52.17, 90: 65.36,
            },
            "Ovary": {
                0: 0.02, 5: 0.25, 10: 1.17, 15: 1.49, 20: 2.08, 25: 2.49, 30: 3.45, 35: 5.85, 40: 9.59, 45: 13.05,
                50: 15.14, 55: 16.12, 60: 19.62, 65: 19.89, 70: 20.51, 75: 20.27, 80: 15.48, 85: 21.20, 90: 23.17,
            },
            "Bladder": {
                0: 0.23, 5: 0.00, 10: 0.00, 15: 0.00, 20: 0.06, 25: 0.15, 30: 0.13, 35: 0.43, 40: 0.75, 45: 0.94,
                50: 2.80, 55: 4.62, 60: 7.49, 65: 10.82, 70: 18.15, 75: 25.43, 80: 35.23, 85: 50.41, 90: 34.96,
            },
            "Thyroid": {
                0: 0.00, 5: 0.18, 10: 0.55, 15: 1.54, 20: 3.26, 25: 3.84, 30: 5.74, 35: 6.78, 40: 10.45, 45: 13.31,
                50: 12.54, 55: 11.59, 60: 12.86, 65: 12.59, 70: 13.96, 75: 13.00, 80: 11.16, 85: 11.16, 90: 11.16,
            },
            "Non-CLL Leukaemia": {
                0: 4.63, 5: 2.44, 10: 3.25, 15: 1.62, 20: 1.58, 25: 1.76, 30: 2.02, 35: 3.27, 40: 3.92, 45: 4.18,
                50: 5.89, 55: 5.60, 60: 9.19, 65: 10.75, 70: 14.84, 75: 13.56, 80: 18.58, 85: 19.69, 90: 20.69,
            },
            "Remainder": {
                0: 9.50, 5: 3.46, 10: 4.21, 15: 3.51, 20: 6.29, 25: 11.41, 30: 19.68, 35: 29.60, 40: 42.92, 45: 55.63,
                50: 82.15, 55: 118.03, 60: 167.56, 65: 230.12, 70: 294.06, 75: 348.37, 80: 396.37, 85: 478.79, 90: 585.14,
            },
        },
    },
}


def _apply_latency_and_amax_mask(kernel: np.ndarray, exposure_age: float,
                                 latency: float) -> np.ndarray:
    """Zero a kernel for ages below latency and above the active risk-model cap.

    Parameters
    ----------
    kernel : np.ndarray
        Per-attained-age values (length 121, indices 0..120).
    exposure_age : float
        Age at exposure in years; ``t = age - exposure_age`` is used for the
        latency cut.
    latency : float
        Minimum latency in years (e.g. 5 for solid REIC, 10 for solid LAR).
    """
    ages = np.arange(len(kernel), dtype=float)
    t = ages - exposure_age
    kernel = np.where(t >= latency, kernel, 0.0).astype(float)
    _amax = _max_attained_age_for_risk_model()
    if _amax < len(kernel) - 1:
        kernel[_amax + 1:] = 0.0
    return kernel


def solid_kernel_perGy(cancer_type: str, sex_code: str, exposure_age_int: int,
                       population: str = "Euro-American",
                       force_model: str = None) -> np.ndarray:
    """Annual excess incidence per Gy as a function of attained age (length 121).

    Implements the ICRP 152 risk model (paras 35-38, p.31-32) combining ERR
    and EAR contributions per cancer site with the ICRP 152 Table 3.1
    weights ``ERR_weight`` and ``EAR_weight`` from ``cancer_risk_params``.

    Functional form (ICRP 152 para 37 / Tables 3.2-3.3)::

        ERR(e, a) = beta_ERR * exp(alpha1_ERR * (e - 30) / 10) * (a/70)^alpha2_ERR
        EAR(e, a) = beta_EAR * exp(alpha1_EAR * (e - 30) / 10) * (a/70)^alpha2_EAR

        annual ERR excess = ERR(e,a) * baseline_incidence(a)   (ICRP 152 Eq. 1)
        annual EAR excess = EAR(e,a)
        kernel(a)  = w_ERR * annual_ERR + w_EAR * annual_EAR   (linear, per Gy)

    The age-only factor ``(a/70)^alpha2`` is precomputed in
    ``_build_ageonly_factors``.  Baseline incidence per ICRP 152 Tables
    A.1-A.4 comes from ``_baseline_incidence_array_for``.

    Special cases
    -------------
    * Female breast -- EAR-only model (Preston et al. 2002, ICRP 152
      para 61); delegated to ``get_female_breast_EAR_kernel_perGy``.
    * Leukaemia -- NOT handled here; see ``leukaemia_kernels`` (ICRP 152
      para 41 LQ form).
    * Bone, Skin -- NOT integrated by this function.  ICRP 152 sec
      3.1.4 / 5.1.2 retains the ICRP 60 / ICRP 59 nominal lifetime risks
      unchanged; ``compute_lifetime_risk`` short-circuits Bone and Skin.

    Output is then masked
    ---------------------
    * To zero for ages where ``t = a - e < min_latency`` (ICRP 152 para 97).
    * To zero for attained ages > 94 y (ICRP 152 paras 51-52, 138 amax).

    Parameters
    ----------
    cancer_type : str -- key into ``cancer_risk_params``.
    sex_code : ``'M'``, ``'F'``, or otherwise treated as sex-averaged.
    exposure_age_int : age at exposure in years (capped to MAX_ATTAINED_AGE_ICRP152).
    population : ``'Euro-American'`` or ``'Asian'`` (selects baseline rates).
    force_model : optional ``'ERR'`` / ``'EAR'`` for sensitivity testing
                  (overrides the ICRP 152 Table 3.1 weights).

    Returns ndarray of length 121 (annual excess incidence per Gy at each
    attained age 0..120).
    """
    if cancer_type == "Breast" and sex_code == "F":
        if force_model == "ERR":
            return np.zeros(121, dtype=float)
        return get_female_breast_EAR_kernel_perGy(float(exposure_age_int))
    params = lookup_cancer_params(cancer_type)
    beta_base_err = _sex_weighted_value(params, "ERR_beta_base", sex_code, 0.0)
    beta_base_ear = _sex_weighted_value(params, "EAR_beta_base", sex_code, 0.0)
    Alpha1_ERR = _sex_weighted_value(params, "ERR_alpha1", sex_code, 0.0)
    Alpha1_EAR = _sex_weighted_value(params, "EAR_alpha1", sex_code, 0.0)
    w_err = params.get("ERR_weight")
    w_ear = params.get("EAR_weight")
    err_base, ear_base = _get_ageonly_arrays(cancer_type, sex_code)
    exp_scalar_err = np.exp(Alpha1_ERR * (float(exposure_age_int) - 30.0) / 10.0)
    exp_scalar_ear = np.exp(Alpha1_EAR * (float(exposure_age_int) - 30.0) / 10.0)
    err_age = exp_scalar_err * err_base
    ear_age = exp_scalar_ear * ear_base
    base_arr = _baseline_incidence_array_for(cancer_type, sex_code, population)
    annual_err = beta_base_err * err_age * base_arr
    annual_ear = beta_base_ear * ear_age
    if force_model == "ERR":
        kernel = annual_err
    elif force_model == "EAR":
        kernel = annual_ear
    else:
        kernel = (w_err * annual_err + w_ear * annual_ear)
    # ICRP 152 §4.2 para 97 / Table 4.1: minimum latency is 5 y for REIC and
    # 10 y for LAR (default).  _solid_latency_for_risk_model honours user
    # overrides via configure_latency / configure_simulation.
    _solid_lat = _solid_latency_for_risk_model()
    return _apply_latency_and_amax_mask(kernel, float(exposure_age_int), _solid_lat)


def leukaemia_kernels(sex_code: str, exposure_age_int: int) -> tuple:
    """Build the leukaemia EAR kernels for the linear-quadratic dose model.

    Implements ICRP 152 para 41 / Table 3.4 (p.32)::

        EAR(t; d) = (beta * d + 1.53 * d^2) * (t / 25)^alpha     per 10,000 / yr

    where ``t = a - e`` is time since exposure, ``alpha`` is the time-since-
    exposure exponent and ``beta`` the linear coefficient (sex- and age-at-
    exposure-dependent, looked up via ``get_leukaemia_EAR_params`` against
    ICRP 152 Table 3.4).  The leading factor 1.53 is the ICRP 152 para 41
    quadratic-to-linear ratio.

    Returns the dose-decoupled basis pair ``(L1, L2)`` such that the
    kernel for an arbitrary dose ``d`` is ``L1 * d + L2 * d**2``::

        L1[a] = beta * (t/25)^alpha / 10000
        L2[a] = 1.53 * (t/25)^alpha / 10000

    Both arrays are zero for ages where ``t < min_latency`` (ICRP 152
    para 97).  ICRP 152 leukaemia means non-CLL leukaemia (CLL excluded);
    the baseline incidence used elsewhere lives under the
    ``"Non-CLL Leukaemia"`` key in ``BASELINE_RATES_ICRP``.

    Returns ``(L1, L2)`` as length-121 float arrays.
    """
    beta_leuk, alpha_leuk = get_leukaemia_EAR_params(float(exposure_age_int), sex_code)
    ages_local = np.arange(121, dtype=float)
    t = ages_local - float(exposure_age_int)
    f = np.zeros_like(ages_local, dtype=float)
    # ICRP 152 §4.2 para 97 / Table 4.1: minimum latency is 5 y for REIC and
    # 2 y for LAR (default).  _leuk_latency_for_risk_model honours user
    # overrides via configure_latency / configure_simulation.
    _leuk_lat = _leuk_latency_for_risk_model()
    mask = t >= _leuk_lat
    f[mask] = (t[mask] / 25.0) ** float(alpha_leuk) / 10000.0
    L1 = _apply_latency_and_amax_mask(float(beta_leuk) * f, float(exposure_age_int), _leuk_lat)
    L2 = _apply_latency_and_amax_mask(1.53 * f, float(exposure_age_int), _leuk_lat)
    return L1.astype(float), L2.astype(float)


# ICRP 152 leukaemia reference dose: the EAR coefficients (Table 3.4) are
# anchored at 0.1 Gy (para 41 / para 138).  The per-Gy slope of the LQ model at
# that reference dose is  L1 + d_ref·L2  (d/dGy of L1·d + L2·d² evaluated as the
# secant L1 + d_ref·L2).  This is the value the Table 3.6 reproduction integrates.
_LEUK_REFERENCE_DOSE_GY = 0.1  # ICRP 152 Table 3.4, paras 41 and 138

# ICRP 152 Table 3.5 LSS leukaemia baseline-incidence model parameters
# (γ1..γ5), cases per 10 000 persons / yr.  Used to convert the EAR model to an
# ERR for the ERR-transfer projection (para 42).
_LEUK_LSS_BASELINE_PARAMS = {
    "M": (-0.28, -0.02, 2.99, 1.11, 9.40),
    "F": (-0.82, 0.00, 1.40, 0.73, 7.87),
}


def _leuk_lss_baseline_perGy_person(exposure_age: float, sex_code: str,
                                    ages: np.ndarray) -> np.ndarray:
    """ICRP 152 Table 3.5 LSS leukaemia baseline incidence (per person / yr).

    Implements the para-42 two-branch model EXACTLY as published -- crucially
    INCLUDING the γ5·ln(a/70)² term for a >= 70.  This is the LSS-cohort
    baseline ICRP divides the EAR model by to obtain the leukaemia ERR
    (``ERR = EAR / baseline_LSS``), so it must NOT be clamped to a <= 70 (which
    would zero the γ5 term and under-estimate the baseline at older attained
    ages).  A small floor a >= 1 only guards log(0); the leukaemia kernel is
    already zero before ``e + latency`` so young ages never contribute.
    """
    g1, g2, g3, g4, g5 = _LEUK_LSS_BASELINE_PARAMS.get(
        sex_code if sex_code in ("M", "F") else "M",
        # sex-averaged fallback for "U"/unknown
        tuple(0.5 * (m + f) for m, f in zip(_LEUK_LSS_BASELINE_PARAMS["M"],
                                            _LEUK_LSS_BASELINE_PARAMS["F"])),
    )
    a = np.maximum(np.asarray(ages, dtype=float), 1.0)
    e = float(exposure_age)
    ln_a50 = np.log(a / 50.0)
    expo = g1 + g2 * (e - 30.0) + g3 * ln_a50 + g4 * ln_a50 ** 2
    ge70 = a >= 70.0
    expo[ge70] += g5 * np.log(a[ge70] / 70.0) ** 2
    return np.exp(expo) / 10000.0          # per 10 000 PY -> per person / yr


def leukaemia_err_ear_kernels_perGy(
    sex_code: str, exposure_age_int: int, population: str = "Euro-American",
) -> tuple:
    """Per-Gy leukaemia excess-incidence kernels for the ERR- and EAR-transfer
    projections (ICRP 152 Table 3.6 convention).

    ICRP 152's leukaemia risk is an **EAR (absolute-rate) model** fitted to the
    LSS (Table 3.4).  ICRP reports its lifetime excess risk under two transport
    models:

    * **EAR transfer** -- the LSS excess rate is applied directly and is
      population-independent::

          Δλ_EAR(e, a) = EAR_LSS(e, a)              [per person / yr / Gy]

    * **ERR transfer** -- the LSS excess is first converted to a *relative* risk
      against the LSS leukaemia baseline, then re-expressed on the **target
      population** baseline::

          ERR_LSS(e, a)  = EAR_LSS(e, a) / λ_LSS(e, a)
          Δλ_ERR(e, a)   = ERR_LSS(e, a) × λ_target(a)
                         = EAR_LSS(e, a) × λ_target(a) / λ_LSS(e, a)

    where

    * ``EAR_LSS`` is the per-Gy slope ``L1 + d_ref·L2`` from
      :func:`leukaemia_kernels` at the ICRP reference dose ``d_ref = 0.1 Gy``;
    * ``λ_LSS(e, a)`` is the ICRP 152 Table 3.5 LSS non-CLL leukaemia baseline
      incidence (per 10 000 PY → per person/yr);
    * ``λ_target(a)`` is the **population** non-CLL leukaemia baseline from
      Annex A (:func:`_baseline_incidence_array_for`, per person/yr).

    The Table 3.6 ERR column needs an ERR-transfer kernel (LSS risk scaled by the
    population/LSS baseline-rate ratio), which is distinct from the EAR-transfer
    kernel.  Using the EAR-transfer value for *both* columns leaves the leukaemia
    ERR coefficient ~15 % low against published ICRP 152 Table 3.6 (the
    bone-marrow deviation); the ERR-transfer kernel here removes that bias.

    Returns ``(K_err, K_ear)`` as length-121 per-Gy arrays
    (cases / person / yr / Gy), already latency- and amax-masked (inherited from
    :func:`leukaemia_kernels`).
    """
    L1, L2 = leukaemia_kernels(sex_code, exposure_age_int)
    # Per-Gy EAR slope at the ICRP reference dose (para 41 / 138).
    ear_perGy = L1 + _LEUK_REFERENCE_DOSE_GY * L2          # EAR-transfer kernel

    ages = np.arange(L1.size, dtype=float)
    # LSS leukaemia baseline (Table 3.5), exact two-branch model incl. the
    # γ5 a>=70 term (NOT the [30,70]-clamped helper used elsewhere).
    lam_lss = _leuk_lss_baseline_perGy_person(float(exposure_age_int), sex_code, ages)
    # Target-population non-CLL leukaemia baseline (Annex A), already per person/yr.
    lam_tgt = _baseline_incidence_array_for("Leukaemia", sex_code, population)

    with np.errstate(divide="ignore", invalid="ignore"):
        transfer = np.where(lam_lss > 0.0, lam_tgt[:L1.size] / lam_lss, 0.0)
    err_perGy = ear_perGy * transfer                        # ERR-transfer kernel
    err_perGy[~np.isfinite(err_perGy)] = 0.0
    # ERR transfer is undefined wherever the EAR kernel is zero (pre-latency /
    # post-amax); the multiplicative form already enforces that, but guard the
    # baseline-zero ages explicitly so the masks stay identical to the EAR side.
    err_perGy[ear_perGy == 0.0] = 0.0
    return err_perGy.astype(float), ear_perGy.astype(float)


_ERR_AGEONLY = {"M": {}, "F": {}}
_EAR_AGEONLY = {"M": {}, "F": {}}


def _build_ageonly_factors():
    """Precompute the attained-age factor (a/70)^alpha2 for every cancer site.

    ICRP 152 para 37 expresses ERR / EAR risk as a product of an
    age-at-exposure factor and an attained-age factor.  This helper
    builds the attained-age component ``exp(alpha2 * ln(a/70))`` for
    each cancer site / sex pair from the ICRP 152 Table 3.2 / 3.3
    alpha2 values stored in ``cancer_risk_params``.

    A floor of a = 30 is applied (``ages_safe = max(a, 30)``) because
    the Preston et al. 2007 LSS model that ICRP 152 adopts is fitted
    to attained ages a >= 30.  In practice the latency mask in
    ``solid_kernel_perGy`` zeros out almost all ages below 30 anyway,
    so this floor only affects exposures at very young ages whose
    attained ages remain below 30 after the latency interval.
    """
    ages_local = np.arange(121, dtype=float)
    ages_safe = np.maximum(ages_local, 30)   # ICRP 152 LSS model fit applies for a >= 30
    for sex in ("M", "F"):
        for cancer, params in cancer_risk_params.items():
            if not isinstance(params, dict):
                continue
            a2_err = params.get("ERR_alpha2", {}).get(sex, 0.0)
            a2_ear = params.get("EAR_alpha2", {}).get(sex, 0.0)
            _ERR_AGEONLY[sex][cancer] = np.exp(a2_err * np.log(ages_safe / 70.0)).astype(np.float32)
            _EAR_AGEONLY[sex][cancer] = np.exp(a2_ear * np.log(ages_safe / 70.0)).astype(np.float32)


_AGEONLY_BUILT = False




def compute_lifetime_excess_cancer_risk_REIC(
    cancer_type: str,
    dose_Gy: float,
    age_at_exposure: float,
    sex_code: str,
    population: str = "Euro-American",
    max_attained_age: int = 94,
    ddref: float = None,
    force_model: str = None,
    method: str = "REIC",
    survival_excess_cumsum: Optional[np.ndarray] = None,
) -> float:
    """Compute lifetime excess cancer risk per ICRP 152 (2022).

    method='REIC'  (default): Risk of Exposure-Induced Cancer, ICRP 152 para 51.
        Integral = Σ Δλ(n) × S(n, e|d), where S is dose-adjusted cancer-free survival.
        S(a, e|d) ≈ S(a, e|0) × exp(−Σ_{n'<n} Δλ_total(n')), with Δλ_total summed
        over ALL sites (ICRP 152 para 47).
    method='LAR': Lifetime Attributable Risk, ICRP 152 para 52 footnote.
        Integral = Σ Δλ(n) × S(n, e|0), background cancer-free survival only (no dose correction).

    ``survival_excess_cumsum``: optional ICRP 152 para 47-strict all-site
    cumulative excess incidence array, length ``max_attained_age + 1``,
    indexed by attained age.  When provided (and method='REIC'), the
    survival exponent uses this all-site sum rather than the
    single-site fallback.  Build it once per exam with
    ``compute_all_site_excess_cumsum_REIC`` and pass to every call so
    every site sees the same Δλ_total decrement (matches the MC pipeline).

    In both cases the survival function S(a, e) uses ICRP 152 para 46:
        μ̄(x) = μ_m(x) − μ_mac(x) + μ_iac(x), conditional from exposure age e.
    For diagnostic imaging doses REIC ≈ LAR (ICRP 152 para 52).
    """
    age_at_exposure = max(0, min(age_at_exposure, max_attained_age - 1))
    if ddref is None:
        ddref = determine_ddref(cancer_type)
    min_latency = _LATENCY_PARAMS["Leukaemia"] if cancer_type == "Leukaemia" else _LATENCY_PARAMS["Solid"]  # from configure_latency() / ICRP 152 §4.2 para 97
    start_age = int(age_at_exposure + min_latency)
    end_age = min(max_attained_age, MAX_ATTAINED_AGE_ICRP152)
    if start_age >= end_age:
        return 0.0
    ages_arr = np.arange(start_age, end_age + 1, dtype=float)
    params = lookup_cancer_params(cancer_type)
    # force_model overrides ERR/EAR weights in params for sensitivity testing
    if force_model == "ERR":
        params = dict(params, ERR_weight=1.0, EAR_weight=0.0)
    elif force_model == "EAR":
        params = dict(params, ERR_weight=0.0, EAR_weight=1.0)
    baseline_rates = np.array(
        [_get_cancer_incidence_rate(int(a), sex_code, cancer_type, population) for a in ages_arr]
    )
    delta_lambda = _delta_lambda_vectorized(
        cancer_type, dose_Gy, age_at_exposure,
        ages_arr, sex_code, ddref, baseline_rates, params,
    )
    # Cancer-free survival conditional from exposure age e (ICRP 152 para 47)
    _ensure_cancer_free_surv_prefix()
    e_int = int(age_at_exposure)
    a_int = np.clip(ages_arr.astype(int), 0, 120)
    pop = _require_population(population)
    if sex_code in ("M", "F"):
        pref = _CANCER_FREE_SURV_PREFIX[(sex_code, pop)]
        surv = np.exp(pref[a_int] - pref[e_int]).astype(float)
    else:
        pref_m = _CANCER_FREE_SURV_PREFIX[("M", pop)]
        pref_f = _CANCER_FREE_SURV_PREFIX[("F", pop)]
        surv = 0.5 * (np.exp(pref_m[a_int] - pref_m[e_int]) +
                      np.exp(pref_f[a_int] - pref_f[e_int]))
    surv = np.clip(surv.astype(float), 0.0, 1.0)
    if method == "REIC":
        # Dose-adjusted survival per ICRP 152 para 47:
        #   S(a,e|d) ≈ S(a,e|0) × exp(−Σ_{n<a} Δλ_total(n))
        # where Δλ_total sums excess incidence over ALL sites.  The strict
        # form uses a precomputed all-site cumulative array passed in by the
        # caller; without it we fall back to a single-site decrement
        # (numerically negligible at diagnostic doses but not strictly
        # para 47-compliant).
        if survival_excess_cumsum is not None:
            cumsum_all = np.asarray(survival_excess_cumsum, dtype=float)
            # Index by attained age, shifted by one so each element is the
            # cumulative all-site Δλ BEFORE the integrand's age bin.
            a_idx = np.clip(a_int - 1, 0, cumsum_all.size - 1)
            excess_before = cumsum_all[a_idx]
            # Zero out the first bin (no excess accumulated before start_age).
            excess_before = np.where(ages_arr.astype(int) <= start_age, 0.0, excess_before)
        else:
            # Single-site fallback: per-site cumulative excess (single cancer only).
            excess_before = np.concatenate([[0.0], np.cumsum(np.maximum(0.0, delta_lambda[:-1]))])
        surv = surv * np.exp(-np.maximum(0.0, excess_before))
    return float(np.clip(np.sum(delta_lambda * surv), 0.0, 1.0))


def _delta_lambda_vectorized(
    cancer_type: str,
    dose_Gy: float,
    age_at_exposure: float,
    attained_ages: np.ndarray,
    sex_code: str,
    ddref: float,
    baseline_rates: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Annual excess incidence rate Delta_lambda(a) for a fixed dose.

    Used by ``compute_lifetime_excess_cancer_risk_REIC`` to build the
    integrand of the REIC integral (ICRP 152 para 51, p.41)::

        REIC_c(e, d) = Sum_{n = e+L .. amax} Delta_lambda_c(n, e, d) * S(n, e | d)

    For a given dose, this routine returns the vector of
    ``Delta_lambda_c(n, e, d)`` evaluated at every attained age in
    ``attained_ages``.  It dispatches by cancer type:

    * Female breast -- delegated to ``get_female_breast_EAR_kernel_perGy``
      and scaled by ``dose_Gy / ddref`` (Preston et al. 2002 EAR-only
      model; ICRP 152 para 61 / Table 3.1).
    * Leukaemia -- linear-quadratic dose response per ICRP 152 para 41 /
      Table 3.4: ``EAR = (beta * d + 1.53 * d^2) * (t/25)^alpha / 10000``,
      then divided by ``ddref`` (which is 1.0 for leukaemia per ICRP 152
      para 64).
    * All other solid sites -- weighted ERR + EAR per ICRP 152 paras 35-38
      using the (a/70)^alpha2 attained-age component cached by
      ``_build_ageonly_factors``.

    The returned array is masked to zero for ``a > 94`` (ICRP 152
    paras 51-52, 138 amax).  The minimum-latency mask (``t >= L``) is
    applied at the integral-bounds level by the caller (it sets
    ``start_age = e + L``).

    Returns ``ndarray`` of the same length as ``attained_ages``.
    """
    a = np.asarray(attained_ages, dtype=float)
    t = a - float(age_at_exposure)
    out = np.zeros_like(a, dtype=float)
    if cancer_type == "Breast" and sex_code == "F":
        # Preston 2002 EAR-only model; the piecewise (a <= 50 / a > 50)
        # behaviour lives in get_female_breast_EAR_kernel_perGy
        # (ICRP 152 para 61).
        mask = t >= 0.0
        if not np.any(mask):
            return out
        kernel_perGy = get_female_breast_EAR_kernel_perGy(float(age_at_exposure))
        age_idx = np.clip(attained_ages.astype(int), 0, 120)
        perGy = kernel_perGy[age_idx]
        ear_excess = (dose_Gy / ddref) * perGy
        ear_excess = np.where(mask, ear_excess, 0.0)
        out[:] = ear_excess
        return out
    if cancer_type == "Leukaemia":
        # ICRP 152 para 41 / Table 3.4: linear-quadratic EAR.  DDREF is
        # not applied to leukaemia (ICRP 152 para 64) so callers pass
        # ddref = 1.0.
        beta_leuk, alpha_leuk = get_leukaemia_EAR_params(float(age_at_exposure), sex_code)
        mask = t >= 0.0
        if np.any(mask):
            linear_term    = beta_leuk * dose_Gy
            quadratic_term = 1.53 * (dose_Gy ** 2)
            time_factor    = (t[mask] / 25.0) ** alpha_leuk
            out[mask] = ((linear_term + quadratic_term) * time_factor) / (10000.0 * ddref)
        return out

    # Solid cancers -- weighted ERR + EAR per ICRP 152 paras 35-38, Table 3.1.
    beta_base_err = _sex_weighted_value(params, "ERR_beta_base", sex_code, 0.0)
    beta_base_ear = _sex_weighted_value(params, "EAR_beta_base", sex_code, 0.0)
    Alpha1_ERR    = _sex_weighted_value(params, "ERR_alpha1",    sex_code, 0.0)
    Alpha1_EAR    = _sex_weighted_value(params, "EAR_alpha1",    sex_code, 0.0)
    w_err = params.get("ERR_weight")
    w_ear = params.get("EAR_weight")
    mask = t >= 0.0
    if np.any(mask):
        global _AGEONLY_BUILT
        if not _AGEONLY_BUILT:
            _build_ageonly_factors()
            _AGEONLY_BUILT = True
        sex = "M" if sex_code == "M" else ("F" if sex_code == "F" else "U")
        exp_scalar_err = np.exp(Alpha1_ERR * (float(age_at_exposure) - 30.0) / 10.0)
        exp_scalar_ear = np.exp(Alpha1_EAR * (float(age_at_exposure) - 30.0) / 10.0)
        age_idx = np.clip(attained_ages.astype(int), 0, 120)
        if sex in ("M", "F"):
            err_base = _ERR_AGEONLY[sex].get(cancer_type, _ERR_AGEONLY[sex].get("Remainder"))
            ear_base = _EAR_AGEONLY[sex].get(cancer_type, _EAR_AGEONLY[sex].get("Remainder"))
        else:
            err_m = _ERR_AGEONLY["M"].get(cancer_type, _ERR_AGEONLY["M"].get("Remainder"))
            err_f = _ERR_AGEONLY["F"].get(cancer_type, _ERR_AGEONLY["F"].get("Remainder"))
            ear_m = _EAR_AGEONLY["M"].get(cancer_type, _EAR_AGEONLY["M"].get("Remainder"))
            ear_f = _EAR_AGEONLY["F"].get(cancer_type, _EAR_AGEONLY["F"].get("Remainder"))
            err_base = 0.5 * (err_m + err_f)
            ear_base = 0.5 * (ear_m + ear_f)
        err_age = exp_scalar_err * err_base[age_idx][mask]
        ear_age = exp_scalar_ear * ear_base[age_idx][mask]
        excess_err = beta_base_err * (dose_Gy / ddref) * err_age * baseline_rates[mask]
        excess_ear = beta_base_ear * (dose_Gy / ddref) * ear_age
        out[mask] = (w_err * excess_err + w_ear * excess_ear)
    # ICRP 152 Table 4.1 integration cap: 94 y for REIC, 99 y for LAR.
    out[a > _max_attained_age_for_risk_model()] = 0.0
    return out


ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
life_table_male = [79, 70, 61, 52, 43, 34, 25, 17, 10, 6]
life_table_female = [84, 75, 66, 57, 48, 39, 30, 21, 13, 7]
average_values = [(m + f) / 2 for m, f in zip(life_table_male, life_table_female)]


def get_expected_remaining_years(age, sex="Unknown"):
    """Expected remaining years of life at ``age`` (linear interpolation).

    Uses WHO life-table data embedded as ``life_table_male``,
    ``life_table_female``, and ``average_values``.  Sex-averaged when
    sex is unknown.
    """
    if str(sex).startswith("F"):
        values = life_table_female
    elif str(sex).startswith("M"):
        values = life_table_male
    else:
        values = average_values
    if age <= ages[0]:
        return values[0]
    for i in range(1, len(ages)):
        if age < ages[i]:
            x0, y0 = ages[i - 1], values[i - 1]
            x1, y1 = ages[i], values[i]
            return y0 + (y1 - y0) * (age - x0) / (x1 - x0)
    return values[-1]




# Window (years post-exposure) over which the Mataac competing-mortality
# excess hazard is applied to the sampled lifespan.  Matches the 10-year
# observation horizon of Mataac et al. (2024); after the window the hazard
# reverts to the ICRP 152 baseline.
_MATAAC_EXCESS_WINDOW_YEARS: int = 10


def sample_lifespans_vectorized(
    ages: np.ndarray,
    sexes: np.ndarray,
    population: str = "Euro-American",
    max_attained_age: int = 120,
    rng: np.random.Generator = None,
    excess_hazard: np.ndarray = None,
) -> np.ndarray:
    """Vectorized lifespan sampling using ICRP 152 Table A.5 all-cause mortality rates.

    Note: ``max_attained_age=120`` is the length of the all-cause mortality rate
    tables (Table A.5, ages 0–120), NOT the ICRP 152 risk integration cap.
    ICRP 152 para 51 specifies amax = 94 y only for REIC/LAR cumulation
    (excess risk × cancer-free survival).  Lifespan sampling for ICRP severity-weighted QALYs/YLL
    is a competing-mortality model and is not capped by ICRP 152.

    For each person (age, sex), samples age of death from annual all-cause death
    probabilities in _ALLCAUSE_PER_AGE_POP.  Fully NumPy-vectorised: generates the
    entire (n_group × max_attained_age) random matrix at once, compares against the
    rate vector, then finds the first death age with np.argmax – no Python age loop.
    Returns ndarray of shape (n,) with sampled lifespans (years).

    ``excess_hazard`` (optional, shape (n,)) adds a constant competing-mortality
    hazard ``delta`` over the first ``_MATAAC_EXCESS_WINDOW_YEARS`` years from
    each person's start age, so heavily-imaged (Mataac-frail) patients die
    sooner: ``q'(a) = 1 - (1 - q(a)) * exp(-delta)`` for a in the window.
    ``delta == 0`` (or ``None``) leaves the sampled lifespan unchanged.
    """
    rng = rng or np.random.default_rng()
    ages_arr = np.asarray(ages, dtype=float)
    # Fast path: if caller already handed us a U1 numpy array of sex codes
    # (run_simulation now produces "M"/"F"/"U" U1 strings once per run, and
    # normalize_sex_code() uppercases them up front), skip the 2 M-element
    # Python list-comprehension that would otherwise fire on every Monte Carlo
    # iteration.  Longer U-dtype inputs are truncated to U1; non-U inputs
    # (lists, object arrays, etc.) keep the safe .upper() fallback so mixed-case
    # data from external callers is still handled correctly.
    if isinstance(sexes, np.ndarray) and sexes.dtype.kind == "U":
        sexes_upper = sexes if sexes.dtype.itemsize == 4 else sexes.astype("U1")
    else:
        sexes_upper = np.array([str(s).upper() for s in sexes], dtype="U1")
    n = len(ages_arr)
    result = np.full(n, float(max_attained_age))
    pop = _require_population(population)

    def _sim_group(idx_arr: np.ndarray, rates: np.ndarray) -> None:
        """Draw a natural lifespan for every person in ``idx_arr`` under the
        per-age mortality ``rates`` and write the results into ``result``.
        Each draw is the first integer age whose uniform random number falls
        below that year's death probability (all draws taken up-front as one
        (m x max_attained_age) matrix), plus a uniform fractional part within
        the death year.  A person with no simulated death keeps the
        ``max_attained_age`` default.  When ``excess_hazard`` is supplied it
        raises the annual death probability over the first
        ``_MATAAC_EXCESS_WINDOW_YEARS`` post-start years for that person."""
        m = len(idx_arr)
        if m == 0:
            return
        start_ages_f = ages_arr[idx_arr]                       # continuous start age per person
        start_ages   = np.floor(start_ages_f).astype(int)      # integer start age per person
        start_frac   = start_ages_f - start_ages                # fractional offset within start year
        # Build mortality rate vector for ages 0..max_attained_age-1
        rate_vec = np.minimum(rates[:max_attained_age].astype(float), 1.0)
        # Generate all random draws at once: shape (m, max_attained_age)
        rand_mat = rng.random((m, max_attained_age))
        age_grid = np.arange(max_attained_age, dtype=int)[np.newaxis, :]
        # Death at each (person, age) where random draw < mortality rate.
        # When a competing-mortality excess hazard is supplied, raise the
        # annual death probability over the first _MATAAC_EXCESS_WINDOW_YEARS
        # years post-start so frail patients die sooner.  delta == 0 rows fall
        # through the cheap broadcast path and remain unchanged.
        if excess_hazard is not None:
            delta = np.asarray(excess_hazard, dtype=float)[idx_arr]
            if np.any(delta > 0.0):
                _sa = start_ages[:, np.newaxis]
                in_window = (age_grid >= _sa) & (age_grid < _sa + _MATAAC_EXCESS_WINDOW_YEARS)
                rate_mat = np.broadcast_to(rate_vec[np.newaxis, :],
                                           (m, max_attained_age)).copy()
                _adj = 1.0 - (1.0 - rate_mat) * np.exp(-delta[:, np.newaxis])
                rate_mat = np.where(in_window & (delta[:, np.newaxis] > 0.0),
                                    np.minimum(_adj, 1.0), rate_mat)
                death_mat = rand_mat < rate_mat
            else:
                death_mat = rand_mat < rate_vec[np.newaxis, :]
        else:
            death_mat = rand_mat < rate_vec[np.newaxis, :]
        # Mask out ages before the person's current age
        death_mat &= age_grid >= start_ages[:, np.newaxis]
        # First death age per person (argmax returns 0 when no death found)
        first_idx = np.argmax(death_mat, axis=1)   # shape (m,)
        has_death = death_mat[np.arange(m), first_idx]
        dead_j = np.where(has_death)[0]
        if dead_j.size:
            frac = rng.random(dead_j.size)
            # If death falls in the SAME calendar year as a fractional
            # start age, the within-year offset must be >= the start's own
            # fractional offset, otherwise the sampled lifespan could
            # precede the (continuous) start instant despite matching its
            # integer year. Rescale into [start_frac, 1) only in that case;
            # later years are unaffected (frac stays uniform on [0,1)).
            _same_year = first_idx[dead_j] == start_ages[dead_j]
            _sf = start_frac[dead_j]
            frac = np.where(_same_year, _sf + frac * (1.0 - _sf), frac)
            result[idx_arr[dead_j]] = first_idx[dead_j] + frac

    for sex in ("M", "F"):
        idx = np.where(sexes_upper == sex)[0]
        rates = _ALLCAUSE_PER_AGE_POP.get((sex, pop),
                _ALLCAUSE_PER_AGE_POP.get((sex, "Euro-American")))
        _sim_group(idx, rates)

    unk_idx = np.where((sexes_upper != "M") & (sexes_upper != "F"))[0]
    if unk_idx.size:
        rm = _ALLCAUSE_PER_AGE_POP.get(("M", pop), _ALLCAUSE_PER_AGE_POP.get(("M", "Euro-American")))
        rf = _ALLCAUSE_PER_AGE_POP.get(("F", pop), _ALLCAUSE_PER_AGE_POP.get(("F", "Euro-American")))
        _sim_group(unk_idx, 0.5 * (rm + rf))

    return result


_LIFESPAN_PMF_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}


def expected_lifespan_pmf(
    sex_code: str,
    exposure_age: float,
    population: str = "Euro-American",
    max_attained_age: int = MAX_ATTAINED_AGE_ICRP152,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the discrete lifespan probability mass function.

    The lifespan is the first-passage age of death from the ICRP 152 Table A.5
    all-cause mortality rates, starting at ``exposure_age``.  Ages are integer
    years; the returned ``ages`` array spans ``[e_int, max_attained_age]`` and
    the corresponding ``probs`` sum to 1.

    The final bin (``max_attained_age``) collects the probability of surviving
    to the integration cap.  For deterministic expectation calculations the
    continuous lifespan is approximated by the mid-point of each bin.

    The result depends only on ``(sex bucket, e_int, population,
    max_attained_age)`` — the mortality rates are integer-age indexed and
    ``exposure_age`` is int-cast to ``e_int`` — so it is memoised in
    ``_LIFESPAN_PMF_CACHE``.  The cached arrays are marked read-only so any
    caller that mutated them would raise instead of corrupting the cache.
    """
    pop = _require_population(population)
    e_int = int(max(0, min(float(exposure_age), max_attained_age)))
    sex_key = sex_code if sex_code in ("M", "F") else "U"
    cache_key = (sex_key, e_int, pop, int(max_attained_age))
    cached = _LIFESPAN_PMF_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if sex_code in ("M", "F"):
        rates = _ALLCAUSE_PER_AGE_POP.get((sex_code, pop),
                _ALLCAUSE_PER_AGE_POP.get((sex_code, "Euro-American")))
    else:
        rm = _ALLCAUSE_PER_AGE_POP.get(("M", pop), _ALLCAUSE_PER_AGE_POP.get(("M", "Euro-American")))
        rf = _ALLCAUSE_PER_AGE_POP.get(("F", pop), _ALLCAUSE_PER_AGE_POP.get(("F", "Euro-American")))
        rates = 0.5 * (rm + rf)
    rates = np.asarray(rates, dtype=float)
    rates = np.clip(rates[:max_attained_age], 0.0, 1.0)

    ages = np.arange(e_int, max_attained_age + 1, dtype=int)
    if ages.size == 0:
        probs = np.ones_like(ages, dtype=float)
        ages.flags.writeable = False
        probs.flags.writeable = False
        _LIFESPAN_PMF_CACHE[cache_key] = (ages, probs)
        return ages, probs

    # log-survival prefix from exposure age
    log_surv = np.concatenate(([0.0], np.cumsum(np.log(np.maximum(1.0 - rates[e_int:], 1e-15)))))
    # P(death at age a) = S(a-1) * q(a)
    probs = np.empty(ages.size, dtype=float)
    n = min(ages.size - 1, log_surv.size - 1)
    if n > 0:
        probs[:n] = np.exp(log_surv[:n]) * rates[e_int:e_int + n]
    # Final bin: probability of surviving to max_attained_age
    probs[-1] = max(0.0, 1.0 - float(probs[:n].sum()) if n > 0 else 1.0)
    total = probs.sum()
    if total > 0.0:
        probs /= total
    ages.flags.writeable = False
    probs.flags.writeable = False
    _LIFESPAN_PMF_CACHE[cache_key] = (ages, probs)
    return ages, probs


def sample_cancer_latency_bulk(sites, ages, min_latency_params, u_inv, u_jit,
                              sexes, populations):
    """Bulk / vectorized cancer-onset sampling.

    Takes arrays of ``sites`` (cancer type names, length ``n_hits``),
    exposure ``ages`` (float, length ``n_hits``), and two pre-drawn
    per-hit random variate arrays:

      * ``u_inv`` — Uniform(0, 1) for inverse-CDF sampling against the
        cached REIC onset kernel (and for the uniform fallback used when
        no precomputed kernel is available, i.e. Bone / Skin),
      * ``u_jit`` — Uniform(0, 1) for sub-year jitter on the integer
        onset age.
      * ``sexes`` — array of sex codes ("M"/"F"/"U"), length ``n_hits``.
        Sex-specific kernels are used for the onset CDF lookup.
      * ``populations`` — per-hit population labels ("Euro-American"/"Asian"),
        length ``n_hits``.  The onset kernel and survival are population-
        specific, so the CDF is keyed by the hit's own population.

    Returns a float64 ``(n_hits,)`` array of latencies (years).

    Equivalent to one inverse-CDF onset draw per hit but
    amortises: (a) the Python function-call overhead, (b) the per-call
    RNG dispatch — uniforms are drawn in bulk by the caller — and (c)
    the ``_LATENCY_PDF_CACHE`` lookup, which happens once per unique
    ``(cancer_type, exposure_age_int, sex, population)`` tuple rather than
    once per hit.
    The ``searchsorted`` + arithmetic is vectorized across every hit
    sharing a cached CDF.

    Statistically identical to the scalar version: the same
    inverse-CDF transforms are applied to the same uniform variates.
    Differences vs. the scalar path are
    at most on the order of IEEE-754 re-association of a handful of
    floating-point operations.
    """
    if min_latency_params is None:
        min_latency_params = dict(_LATENCY_PARAMS)
    n = len(sites)
    latencies = np.empty(n, dtype=np.float64)
    ages_f = np.asarray(ages, dtype=np.float64)

    # Group hits by (cancer_type, exposure_age_int, sex, population) so the
    # CDF cache is consulted at most once per distinct key.  The per-iteration
    # count of distinct keys is typically ~n_sites * n_age_bins * 2 sexes *
    # 2 populations (a few hundred), tiny compared with n_hits (~10^3–10^5).
    sexes_arr = np.asarray([str(s).upper()[:1] for s in sexes], dtype='U1')
    pops_arr = np.asarray([_require_population(p) for p in populations],
                          dtype=object)

    groups: dict = {}
    for i in range(n):
        ct = sites[i]
        age_e = ages_f[i]
        e_int = int(max(0, min(age_e, MAX_ATTAINED_AGE_ICRP152)))
        sex_i = str(sexes_arr[i])
        pop_i = pops_arr[i]
        key = (ct, e_int, sex_i, pop_i)
        g = groups.get(key)
        if g is None:
            groups[key] = [i]
        else:
            g.append(i)

    _kernels_ready = kernels_precomputed()

    for (ct, e_int, sex_i, pop_i), idx_list in groups.items():
        idxs = np.asarray(idx_list, dtype=np.int64)
        ct_lower = str(ct).lower()
        if ct_lower == "leukaemia":
            min_lat = float(min_latency_params.get("Leukaemia", 5.0))
        else:
            min_lat = float(min_latency_params.get("Solid", 5.0))

        use_cached = _kernels_ready and ct not in ("Bone", "Skin")
        cdf = None
        start_a = 0
        if use_cached:
            try:
                cache_key = (ct, e_int, min_lat, sex_i, pop_i)
                cached = _LATENCY_PDF_CACHE.get(cache_key)
                if cached is None:
                    if ct_lower == "leukaemia":
                        L1, _L2 = get_cached_leukaemia(sex_i, e_int, pop_i)
                        kernel = L1
                    else:
                        kernel = get_cached_kernel(ct, sex_i, e_int, pop_i)
                    start_a = int(e_int + min_lat)
                    end_a = MAX_ATTAINED_AGE_ICRP152
                    if start_a < end_a:
                        weights = np.maximum(kernel[start_a:end_a + 1], 0.0)
                        total = float(weights.sum())
                        if total > 0.0:
                            cached = (start_a, np.cumsum(weights))
                        else:
                            cached = (start_a, None)
                    else:
                        cached = (start_a, None)
                    _LATENCY_PDF_CACHE[cache_key] = cached
                start_a, cdf = cached
            except Exception:
                cdf = None  # fall through to uniform fallback

        if cdf is not None:
            # Vectorized inverse-CDF for this group: matches the scalar
            # path's ``rng.random() * cdf[-1] → searchsorted(cdf, u)``
            # one-to-one, only with the uniforms already supplied by the
            # caller so no per-hit ``rng.random()`` call is needed.
            u = u_inv[idxs] * float(cdf[-1])
            k = np.searchsorted(cdf, u, side="right")
            # Clamp the right-edge case u == cdf[-1] (searchsorted returns
            # cdf.size in that situation) — same clamp as the scalar path.
            np.minimum(k, cdf.size - 1, out=k)
            onset_age = start_a + k  # (group_size,) int
            latencies[idxs] = (onset_age.astype(np.float64)
                               - ages_f[idxs]
                               + u_jit[idxs])
        else:
            # Uniform fallback for Bone (ICRP 60), Skin (ICRP 59), or any
            # site without a time-resolved REIC kernel.  Re-uses the
            # pre-drawn ``u_inv`` Uniform(0, 1) variates so no extra RNG
            # stream is needed: latency = min_lat + u * span where span is
            # the remaining post-latency window before a_max.  When the
            # window is non-positive (exposure age too late) span is clipped
            # to zero and latency collapses to min_lat.
            span = np.maximum(
                MAX_ATTAINED_AGE_ICRP152 + 1.0 - ages_f[idxs] - min_lat,
                0.0,
            )
            latencies[idxs] = min_lat + u_inv[idxs] * span

    return latencies


def expected_onset_pmf(
    cancer_type: str,
    exposure_age: float,
    sex_code: str = "U",
    population: str = "Euro-American",
    min_latency_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the discrete cancer-onset probability mass function.

    The onset distribution is the normalised REIC kernel (annual excess rate
    times cancer-free survival) used for onset sampling.  For Bone
    and Skin, which have no time-resolved REIC kernel, a discrete uniform over
    ``[e + min_lat, MAX_ATTAINED_AGE_ICRP152]`` is returned.

    Returns ``(onset_ages, probs)`` where onset ages are integer years and
    ``probs`` sums to 1.  Continuous expectations should use the bin mid-points.
    """
    if min_latency_params is None:
        min_latency_params = dict(_LATENCY_PARAMS)
    ct = str(cancer_type).strip()
    ct_lower = ct.lower()
    e_int = int(max(0, min(float(exposure_age), MAX_ATTAINED_AGE_ICRP152)))
    if ct_lower == "leukaemia":
        min_lat = float(min_latency_params.get("Leukaemia", 5.0))
    else:
        min_lat = float(min_latency_params.get("Solid", 5.0))
    start_a = int(e_int + min_lat)
    end_a = MAX_ATTAINED_AGE_ICRP152
    ages = np.arange(start_a, end_a + 1, dtype=int)
    if ages.size == 0:
        # Latency window extends beyond amax (e.g. e=90, lat=5 → start=95 > 94).
        # MC path samples latency = min_lat (constant) in this case, so onset
        # is deterministic at start_a.  Return a degenerate PMF there.
        return np.array([start_a], dtype=int), np.array([1.0], dtype=float)

    if ct in ("Bone", "Skin"):
        probs = np.full(ages.size, 1.0 / ages.size, dtype=float)
        return ages, probs

    sex = sex_code if sex_code in ("M", "F") else "U"
    pop = _require_population(population)
    try:
        if ct_lower == "leukaemia":
            L1, _L2 = get_cached_leukaemia(sex, e_int, pop)
            kernel = L1
        else:
            kernel = get_cached_kernel(ct, sex, e_int, pop)
        weights = np.maximum(kernel[start_a:end_a + 1], 0.0)
        total = float(weights.sum())
        if total > 0.0:
            probs = weights / total
        else:
            probs = np.full(ages.size, 1.0 / ages.size, dtype=float)
    except Exception:
        probs = np.full(ages.size, 1.0 / ages.size, dtype=float)
    return ages, probs


def _discount_interval(t1: np.ndarray, t2: np.ndarray, r: float) -> np.ndarray:
    """Present value of 1 unit/year of utility spanning [t1, t2], years.

    Uses a continuous-time exponential discount factor, equivalent to the
    GBD / WHO-CHOICE convention when expressed on an annual basis:
    ``PV = ∫_{t1}^{t2} e^{-r·t} dt = (e^{-r t1} - e^{-r t2}) / r``.
    Returns ``max(0, t2 - t1)`` when ``r == 0`` so the undiscounted case
    is bit-exact (no ``expm1`` round-off).  Shared by ``vectorized``,
    which imports it from this module.
    """
    dt = np.maximum(0.0, t2 - t1)
    if r <= 0.0:
        return dt
    # Numerically stable form: factor out e^{-r t1} to avoid catastrophic
    # cancellation when r·dt is tiny.
    return np.exp(-r * t1) * (-np.expm1(-r * dt)) / r


def _lifespan_survival_at_onset(
    onset_ages: np.ndarray, lifespan_ages: np.ndarray, lifespan_probs: np.ndarray,
) -> np.ndarray:
    """P(lifespan >= a) for each onset age ``a``, from the unconditional
    (from-exposure) lifespan PMF returned by :func:`expected_lifespan_pmf`.

    The MC path draws a cancer's post-onset lifespan via
    ``sample_lifespans_vectorized(onset_age, ...)`` (see ``vectorized.py``
    ``_QALY_event_bulk``), which only searches for a death age >= onset --
    i.e. it draws from the lifespan distribution CONDITIONAL on having
    survived (all-cause) to onset.  ``expected_lifespan_pmf`` instead
    returns the UNCONDITIONAL distribution from exposure age ``e``, which
    includes probability mass for dying before onset.  Naively pairing that
    unconditional PMF with the onset PMF double-counts competing mortality:
    the REIC probability already discounts for cancer-free survival to
    onset, and the raw lifespan PMF then discounts a second time (dying-
    before-onset pairs contribute zero YLL without their probability mass
    being redistributed among the surviving-to-onset outcomes), biasing
    E[YLL] low relative to the MC estimate.

    Dividing each onset row's lifespan PMF by this survival probability
    (see callers below) restores ``P(lifespan=d | lifespan >= a)`` and
    makes the analytical YLL match the MC conditional draw.
    """
    suffix = np.cumsum(lifespan_probs[::-1])[::-1]
    idx = np.searchsorted(lifespan_ages, onset_ages, side="left")
    out = np.zeros(idx.shape, dtype=float)
    valid = idx < suffix.size
    out[valid] = suffix[idx[valid]]
    return out


def _onset_lifespan_joint_conditional(
    onset_ages: np.ndarray, onset_probs: np.ndarray,
    lifespan_ages: np.ndarray, lifespan_probs: np.ndarray,
) -> np.ndarray:
    """Joint PMF ``P(onset=a) * P(lifespan=d | lifespan >= a)`` (shape N x L).

    Used by :func:`expected_yll_moments_for_site` so the survival-to-onset
    conditioning (see :func:`_lifespan_survival_at_onset`) is implemented
    once.  Columns with ``d < a`` are left with an arbitrary (unnormalised)
    value since callers always weight this joint by a YLL matrix that is
    zero there (``_discount_interval`` clips negative intervals to 0).
    """
    surv_to_onset = _lifespan_survival_at_onset(onset_ages, lifespan_ages, lifespan_probs)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_lifespan = lifespan_probs[np.newaxis, :] / surv_to_onset[:, np.newaxis]
    cond_lifespan = np.where(surv_to_onset[:, np.newaxis] > 0.0, cond_lifespan, 0.0)
    return onset_probs[:, np.newaxis] * cond_lifespan


_YLL_MOMENTS_CACHE: Dict[Tuple, Tuple[float, float]] = {}


def expected_yll_moments_for_site(
    cancer_type: str,
    exposure_age: float,
    sex_code: str,
    population: str = "Euro-American",
    min_latency_params: Optional[dict] = None,
    QALY_discount_rate: float = 0.0,
    lifespan_ages: Optional[np.ndarray] = None,
    lifespan_probs: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Return (E[YLL], E[YLL²]) for one cancer site.

    Uses the ICRP 152 REIC onset distribution and the all-cause mortality
    lifespan distribution.  The second moment is needed for the
    analytical variance of the per-iteration ICRP severity-weighted QALYs and life-lost totals.
    """
    e = float(exposure_age)
    _ml_key = tuple(sorted((min_latency_params or {}).items()))
    # The onset/lifespan PMFs depend only on int(e), and
    # the undiscounted YLL matrix max(0, lifespan_mid - onset_mid) cancels the
    # fractional part of e. The discounted path keeps the exact float key.
    _e_key = int(max(0.0, e)) if float(QALY_discount_rate) <= 0.0 else round(e, 4)
    cache_key = (str(cancer_type), _e_key, str(sex_code), str(population),
                 float(QALY_discount_rate), _ml_key)
    cached = _YLL_MOMENTS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if lifespan_ages is None or lifespan_probs is None:
        lifespan_ages, lifespan_probs = expected_lifespan_pmf(
            sex_code, e, population=population,
        )
    if lifespan_probs.size == 0 or lifespan_probs.sum() <= 0.0:
        return 0.0, 0.0
    lifespan_mid = lifespan_ages.astype(float) + 0.5
    onset_ages, onset_probs = expected_onset_pmf(
        cancer_type, e, sex_code=sex_code, population=population,
        min_latency_params=min_latency_params,
    )
    if onset_probs.size == 0 or onset_probs.sum() <= 0.0:
        return 0.0, 0.0
    onset_mid = onset_ages.astype(float) + 0.5

    t1 = onset_mid - e
    t2 = lifespan_mid - e
    t1_col = t1[:, np.newaxis]
    t2_row = t2[np.newaxis, :]
    yll_matrix = _discount_interval(t1_col, t2_row, float(QALY_discount_rate))
    # Survival-to-onset-conditioned joint PMF -- see
    # _onset_lifespan_joint_conditional.
    joint = _onset_lifespan_joint_conditional(
        onset_ages.astype(float), onset_probs, lifespan_ages.astype(float), lifespan_probs)
    e_yll = float((yll_matrix * joint).sum())
    e_yll2 = float((yll_matrix * yll_matrix * joint).sum())
    _YLL_MOMENTS_CACHE[cache_key] = (e_yll, e_yll2)
    return e_yll, e_yll2


def _baseline_survival_curve(
    ages_int, exposure_age_int: int, sex_code: str,
    population: str = "Euro-American",
) -> np.ndarray:
    """Cancer-free survival S(a, e|0) per ICRP 152 paras 46-47 (p.37-38).

    Implements the conditional cancer-free survival used inside the REIC
    integral (ICRP 152 Eq. on p.41).  The hazard is

        mu_bar(x) = mu_m(x) - mu_mac(x) + mu_iac(x)

    where (all from ICRP 152 Annex A):

    * ``mu_m(x)``  = all-cause mortality rate         (Table A.5).
    * ``mu_mac(x)`` = all-cancer mortality rate       (Table A.6).
    * ``mu_iac(x)`` = sum of all-site cancer incidence rates  (Tables A.1-A.4).

    The survival is conditional on having survived (and being cancer-free)
    at exposure age ``e``::

        S(a, e | 0) = exp( prefix[a] - prefix[e] )

    where ``prefix[a] = Sum_{t = 0..a-1} log(1 - mu_bar(t))`` is the cached
    log-survival prefix in :data:`_CANCER_FREE_SURV_PREFIX`.  For unknown
    sex (``sex_code`` not ``'M'`` / ``'F'``) the M and F survival curves
    are averaged equally.

    Returns the survival values for every age in ``ages_int``, clipped to
    [0, 1].  Used by ``compute_lifetime_risk`` (per-exam REIC integration)
    and by ``_build_rpg_tables`` (pre-baked cumulative tables).
    """
    _ensure_cancer_free_surv_prefix()
    a_arr = np.asarray(ages_int, dtype=int)
    if a_arr.size == 0:
        return np.zeros(0, dtype=float)
    e_int = int(exposure_age_int)
    pop = _require_population(population)
    if sex_code in ("M", "F"):
        pref = _CANCER_FREE_SURV_PREFIX[(sex_code, pop)]
        surv = np.exp(pref[a_arr] - pref[e_int]).astype(float)
    else:
        prefM = _CANCER_FREE_SURV_PREFIX[("M", pop)]
        prefF = _CANCER_FREE_SURV_PREFIX[("F", pop)]
        vM = np.exp(prefM[a_arr] - prefM[e_int]).astype(float)
        vF = np.exp(prefF[a_arr] - prefF[e_int]).astype(float)
        surv = 0.5 * (vM + vF)
    return np.clip(surv, 0.0, 1.0)



def determine_ddref(cancer_type: str) -> float:
    """Return the dose-rate effectiveness factor for a given cancer site.

    DDREF policy per ICRP 152 §3.1.5 paras 63-64 and Table 4.1 (p.55):

    * Solid cancers -- DDREF = ``_SIM_CONFIG['ddref_solid']`` (default 2.0;
      ICRP 152 para 100 / p.62).  Applied to all LSS-derived solid-cancer
      kernels.
    * Leukaemia -- DDREF = ``_SIM_CONFIG['ddref_leuk']`` (default 1.0).
      ICRP 152 para 64 explicitly does NOT apply DDREF to leukaemia
      because the linear-quadratic dose response (ICRP 152 para 41 /
      Table 3.4) already accounts for low-dose risk modification.
    * Bone surface and Skin -- DDREF = 1.0.  ICRP 152 para 64 retains
      the ICRP Publication 60 (1991) bone surface and ICRP Publication
      59 (1991) skin nominal risks unchanged; these are total lifetime
      nominal risks, not LSS-derived annual rates, so DDREF is not
      applicable.

    DDREF does not depend on the dose or dose rate of the individual exam:
    ICRP 152 applies one factor per cancer group across the whole
    diagnostic dose range.
    """
    ct = str(cancer_type).strip().lower()
    if ct == "leukaemia":
        return _SIM_CONFIG["ddref_leuk"]
    if ct in ("bone", "skin"):
        # ICRP 152 para 64: DDREF not applied to bone (ICRP 60 nominal) or
        # skin (ICRP 59 nominal); these are total lifetime risks, not annual rates.
        return 1.0
    return _SIM_CONFIG["ddref_solid"]


# Precomputed kernel cache

_KERNEL_CACHE = {}
_SURV_CACHE = {}
# Cache for onset-distribution CDFs (sample_cancer_latency_bulk).
# Key:   (cancer_type, exposure_age_int, min_latency_years)
# Value: (start_age_int, cumulative_weights_ndarray) or (start_age_int, None)
#        when the distribution has no positive mass.
# The tuple's cumsum array is reusable across all hits with the same
# (cancer_type, age, min_latency), eliminating per-call np.arange / fancy
# index / np.maximum / normalize work that dominated the per-hit cost.
# Cleared by precompute_kernels() whenever the underlying risk kernels are
# rebuilt (population change, re-run, etc.) so stale CDFs are never returned.
_LATENCY_PDF_CACHE: dict = {}

# Serialises kernel-cache reload against any background reader (notably the
# per-year deterministic-baseline sweep run on a background thread by
# year_batch.run_yearly_simulation).  ``precompute_kernels`` takes it for the
# whole reload, and the sweep takes it for the duration of its
# ``_compute_deterministic_baselines`` call so it cannot observe the caches
# mid-clear.  An RLock lets a re-entrant precompute_kernels (e.g. the mixed-
# population double call in main.run_simulation) take it twice safely.
KERNEL_STATE_LOCK = threading.RLock()
# Path of the on-disk cache that is currently loaded into ``_KERNEL_CACHE`` /
# ``_SURV_CACHE`` / the ``_RPG_*`` globals.  The fingerprint embedded in the
# path encodes (max_age, population, DDREF, source hash), so equality of the
# path means the in-memory state is already exactly what a reload would
# produce -- letting a per_year_average batch skip the redundant per-year
# reload (which would also churn the caches while a background sweep reads
# them).
_KERNEL_CACHE_LOADED_PATHS: dict = {}  # population -> on-disk path currently loaded

PRECOMPUTE_CANCER_TYPES = sorted(
    cancer_type
    for cancer_type in CANCER_TO_ORGAN_DOSE_KEY
    if cancer_type in cancer_risk_params
)


def _precompute_kernels_vectorized(max_age: int, population: str) -> None:
    """Bulk vectorised builder for the kernel/survival caches.

    Builds the same _KERNEL_CACHE / _SURV_CACHE entries as the per-(sex, e)
    reference path (``_precompute_one_sex_age`` calling ``solid_kernel_perGy``
    / ``leukaemia_kernels`` / ``_baseline_survival_curve``), but uses one
    (max_age+1, 121) tensor op per (cancer_type, sex_code) pair instead of
    363 separate Python-level calls.

    The result is deterministic: every intermediate
    float-product order matches per element.  On a typical laptop this drops
    precompute from ~125 ms / population to ~15 ms.

    Caveats
    -------
    * `(cancer_type, sex_code) == ("Breast", "F")` uses the dedicated
      Preston et al. 2002 EAR-only model (ICRP 152 para 61) — the same special
      case the per-(sex, e) path delegates to
      ``get_female_breast_EAR_kernel_perGy``.
    * Leukaemia kernels are vectorised per-bucket (3 age buckets × sex)
      because ICRP 152 Table 3.4 supplies (beta, alpha) as a piecewise-
      constant function of exposure age; using a scalar (alpha) per bucket
      means ``np.power`` gets the same C-level ``pow`` dispatch as the
      per-element path, so the result is deterministic.
    """
    # Make sure the lazy ICRP 152 lookup tables exist before we read them.
    if not _AGEONLY_BUILT:
        _build_ageonly_factors()
        globals()["_AGEONLY_BUILT"] = True
    _ensure_cancer_free_surv_prefix()

    # Snapshot config-dependent constants once.
    _amax       = _max_attained_age_for_risk_model()
    _solid_lat  = _solid_latency_for_risk_model()
    _leuk_lat   = _leuk_latency_for_risk_model()

    pop = _require_population(population)

    # Common axes
    ages_local = np.arange(121, dtype=float)               # (121,)
    e_axis     = np.arange(max_age + 1, dtype=float)       # (max_age+1,)
    t          = ages_local[None, :] - e_axis[:, None]      # (max_age+1, 121)
    valid_solid = t >= _solid_lat                           # (max_age+1, 121)
    valid_leuk  = t >= _leuk_lat                            # (max_age+1, 121)

    # 1. Survival tensor S(e, a) per sex
    # Match _baseline_survival_curve():
    #   M / F:  surv = clip(exp(prefix[a] - prefix[e]), 0, 1)
    #   U:      surv = clip(0.5 * (exp(prefM[a]-prefM[e]) + exp(prefF[a]-prefF[e])), 0, 1)
    # _SURV_CACHE[(sex, e)] stores the slice [max(1,e) : _amax+1].
    prefM = _CANCER_FREE_SURV_PREFIX[("M", pop)]   # (121,)
    prefF = _CANCER_FREE_SURV_PREFIX[("F", pop)]
    surv_M_full = np.exp(prefM[None, :] - prefM[:, None])   # (121, 121)
    surv_F_full = np.exp(prefF[None, :] - prefF[:, None])
    surv_U_full = 0.5 * (surv_M_full + surv_F_full)
    np.clip(surv_M_full, 0.0, 1.0, out=surv_M_full)
    np.clip(surv_F_full, 0.0, 1.0, out=surv_F_full)
    np.clip(surv_U_full, 0.0, 1.0, out=surv_U_full)

    end_a = _amax + 1
    _empty = np.array([], dtype=float)
    for sex_code, surv_full in (("M", surv_M_full), ("F", surv_F_full), ("U", surv_U_full)):
        for e in range(max_age + 1):
            start_a = 1 if e < 1 else e
            if start_a >= end_a:
                _SURV_CACHE[(sex_code, e, pop)] = _empty
            else:
                # .copy() so each cache entry is an independent buffer
                # (matches the per-e _baseline_survival_curve allocation
                # and avoids pickling 121 references to one parent buffer).
                _SURV_CACHE[(sex_code, e, pop)] = surv_full[e, start_a:end_a].copy()

    # 2. Solid kernels per (cancer, sex) — vectorised over e
    # Mirrors solid_kernel_perGy() exactly:
    #   exp_e_err   = exp(alpha1_err * (e - 30) / 10)            # (max_age+1,)
    #   err_age     = exp_e_err[:, None] * err_base[None, :]      # (max_age+1, 121)
    #   annual_err  = beta_err * err_age * baseline[None, :]      # left-to-right
    #   kernel      = w_err*annual_err + w_ear*annual_ear
    #   mask_lat    = t >= solid_lat                              # (max_age+1, 121)
    #   kernel[~mask_lat] = 0
    #   kernel[:, _amax+1:] = 0
    _kernel_cache = _KERNEL_CACHE   # local alias for tight loop
    for sex_code in ("M", "F", "U"):
        for ct in PRECOMPUTE_CANCER_TYPES:
            if ct == "Leukaemia":
                continue
            try:
                if ct == "Breast" and sex_code == "F":
                    # ICRP 152 para 61 Preston et al. 2002 pooled-analysis
                    # EAR-only model — same code path as
                    # get_female_breast_EAR_kernel_perGy but vectorised over e.
                    breast_params = cancer_risk_params.get("Breast", {})
                    beta_breast = breast_params.get(
                        "EAR_beta_base", {}).get("F", 25.3 / 10000.0)
                    alpha1_breast = breast_params.get(
                        "EAR_alpha1", {}).get("F", -0.5)
                    a = ages_local
                    a_safe   = np.where(a == 0, 1.0, a)              # (121,)
                    log_a70  = np.log(a_safe / 70.0)                 # (121,)
                    log_a50  = np.log(a_safe / 50.0)                 # (121,)
                    exp_e    = alpha1_breast * (e_axis - 30.0) / 10.0
                    pre_term = exp_e[:, None] + 3.5 * log_a70[None, :]
                    ear_pre  = beta_breast * np.exp(pre_term)
                    a_gt50   = a > 50.0                              # (121,)
                    if a_gt50.any():
                        post_term = pre_term - 2.5 * log_a50[None, :]
                        ear_post  = beta_breast * np.exp(post_term)
                        ear = np.where(a_gt50[None, :], ear_post, ear_pre)
                    else:
                        ear = ear_pre
                    kernel = np.where(valid_solid, ear, 0.0)
                else:
                    params        = lookup_cancer_params(ct)
                    beta_base_err = _sex_weighted_value(params, "ERR_beta_base", sex_code, 0.0)
                    beta_base_ear = _sex_weighted_value(params, "EAR_beta_base", sex_code, 0.0)
                    Alpha1_ERR    = _sex_weighted_value(params, "ERR_alpha1", sex_code, 0.0)
                    Alpha1_EAR    = _sex_weighted_value(params, "EAR_alpha1", sex_code, 0.0)
                    w_err         = params.get("ERR_weight")
                    w_ear         = params.get("EAR_weight")
                    err_base, ear_base = _get_ageonly_arrays(ct, sex_code)        # (121,)
                    base_arr      = _baseline_incidence_array_for(ct, sex_code, pop)  # (121,)
                    exp_e_err     = np.exp(Alpha1_ERR * (e_axis - 30.0) / 10.0)   # (max_age+1,)
                    exp_e_ear     = np.exp(Alpha1_EAR * (e_axis - 30.0) / 10.0)
                    # MUST preserve solid_kernel_perGy's left-to-right multiplication
                    # order so float results are deterministic:
                    #   err_age    = exp_scalar * err_base
                    #   annual_err = beta * err_age * base_arr
                    err_age       = exp_e_err[:, None] * err_base[None, :]        # (max_age+1, 121)
                    ear_age       = exp_e_ear[:, None] * ear_base[None, :]
                    annual_err    = beta_base_err * err_age * base_arr[None, :]
                    annual_ear    = beta_base_ear * ear_age
                    kernel        = w_err * annual_err + w_ear * annual_ear       # (max_age+1, 121)
                    kernel        = np.where(valid_solid, kernel, 0.0)
                if _amax < 120:
                    kernel[:, _amax + 1:] = 0.0
                # Materialise per-e slices so the on-disk pickle stores
                # 95 independent length-121 buffers (matching solid_kernel_perGy's
                # per-e return values).
                for e in range(max_age + 1):
                    _kernel_cache[("solid", ct, sex_code, e, pop)] = kernel[e].copy()
            except Exception:
                # Bit-for-bit fallback: solid_kernel_perGy writes zeros on any
                # exception and downstream consumers tolerate that, so mirror the
                # same recovery here.
                zeros_row = np.zeros(121, dtype=float)
                for e in range(max_age + 1):
                    _kernel_cache[("solid", ct, sex_code, e, pop)] = zeros_row.copy()

    # 3. Leukaemia kernels per (sex, e) — vectorised over e
    # ICRP 152 Table 3.4 supplies (beta, alpha) as a piecewise-constant
    # step function of exposure age (3 buckets).  Within each bucket the
    # power expression ``np.power(t/25, alpha) / 10000`` is a SCALAR-
    # exponent call, matching the per-element path bit-for-bit (numpy dispatches
    # to the same C-level ``pow`` regardless of how many e values share the
    # exponent).  We loop over the 3 buckets × 2 sexes = 6 buckets.
    _leuk_buckets = (
        ( 0, 19),
        (20, 39),
        (40, max_age),
    )
    # 1.53 is the ICRP 152 para 41 quadratic-to-linear ratio.
    _leuk_q_factor = 1.53
    f_M = np.zeros((max_age + 1, 121), dtype=float)
    f_F = np.zeros((max_age + 1, 121), dtype=float)
    L1_M = np.zeros((max_age + 1, 121), dtype=float)
    L1_F = np.zeros((max_age + 1, 121), dtype=float)
    L2_M = np.zeros((max_age + 1, 121), dtype=float)
    L2_F = np.zeros((max_age + 1, 121), dtype=float)
    # Bucket-wise compute keeps alpha as a scalar so np.power matches the
    # per-element ``(t[mask]/25) ** float(alpha)`` exactly.
    for (e_lo, e_hi) in _leuk_buckets:
        if e_lo > max_age:
            continue
        e_hi_eff = min(e_hi, max_age)
        e_slice = slice(e_lo, e_hi_eff + 1)
        # M / F separately so each np.power gets a scalar alpha.
        for sex_code, f_buf, L1_buf, L2_buf in (
            ("M", f_M, L1_M, L2_M),
            ("F", f_F, L1_F, L2_F),
        ):
            beta, alpha = get_leukaemia_EAR_params(float(e_lo), sex_code)
            t_slice = t[e_slice]                                         # (n_e_in_bucket, 121)
            mask_slice = valid_leuk[e_slice]
            if not mask_slice.any():
                continue
            # Avoid (0 ** negative) at masked-out positions by substituting
            # a safe placeholder that the mask zeros out anyway.
            t25_safe = np.where(mask_slice, t_slice / 25.0, 1.0)
            f_slice  = np.power(t25_safe, float(alpha)) / 10000.0
            f_slice  = np.where(mask_slice, f_slice, 0.0)
            f_buf[e_slice]  = f_slice
            L1_buf[e_slice] = float(beta) * f_slice
            L2_buf[e_slice] = _leuk_q_factor * f_slice
    if _amax < 120:
        for buf in (L1_M, L1_F, L2_M, L2_F):
            buf[:, _amax + 1:] = 0.0
    # ICRP 152 averages the M and F sex-specific kernels for sex_code "U"
    # by averaging the (beta, alpha) parameters first (see
    # get_leukaemia_EAR_params), NOT by averaging the M and F kernels.
    # Replicate that here by recomputing per-bucket with averaged params.
    L1_U = np.zeros((max_age + 1, 121), dtype=float)
    L2_U = np.zeros((max_age + 1, 121), dtype=float)
    for (e_lo, e_hi) in _leuk_buckets:
        if e_lo > max_age:
            continue
        e_hi_eff = min(e_hi, max_age)
        e_slice = slice(e_lo, e_hi_eff + 1)
        beta_U, alpha_U = get_leukaemia_EAR_params(float(e_lo), "U")
        t_slice    = t[e_slice]
        mask_slice = valid_leuk[e_slice]
        if not mask_slice.any():
            continue
        t25_safe = np.where(mask_slice, t_slice / 25.0, 1.0)
        f_slice  = np.power(t25_safe, float(alpha_U)) / 10000.0
        f_slice  = np.where(mask_slice, f_slice, 0.0)
        L1_U[e_slice] = float(beta_U) * f_slice
        L2_U[e_slice] = _leuk_q_factor * f_slice
    if _amax < 120:
        L1_U[:, _amax + 1:] = 0.0
        L2_U[:, _amax + 1:] = 0.0

    for sex_code, L1_full, L2_full in (
        ("M", L1_M, L2_M), ("F", L1_F, L2_F), ("U", L1_U, L2_U),
    ):
        for e in range(max_age + 1):
            _kernel_cache[("leuk_L1", sex_code, e, pop)] = L1_full[e].copy()
            _kernel_cache[("leuk_L2", sex_code, e, pop)] = L2_full[e].copy()


# PERSISTENT ON-DISK KERNEL CACHE
# ----------------------------------------------------------------------------
# precompute_kernels() is deterministic; we persist results to a pickle file
# keyed by a cryptographic fingerprint of (max_age, population, DDREF, source hash).
#
# Cache invariants:
# * Fingerprint includes a hash of this source file — any edit invalidates all caches.
# * DDREF changes produce a new fingerprint → new cache file.
# * Corrupt cache files are silently deleted; the slow path runs instead.
# * Writes are atomic (tmp + os.replace).

_KERNEL_CACHE_FORMAT_VERSION = 2  # bump if the persisted blob layout changes


def _kernel_cache_dir() -> str:
    """Return (creating if needed) the on-disk kernel-cache directory."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kernel_cache")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def _kernel_cache_fingerprint(max_age: int, population: str) -> str:
    """Return a hex fingerprint of all inputs that affect kernel values.

    Mixes in (a) call parameters, (b) DDREF values from _SIM_CONFIG,
    (c) risk_model and latency from _SIM_CONFIG / _LATENCY_PARAMS so REIC
    and LAR runs never share a cached blob, and (d) the full source of this
    module so any code change auto-invalidates.
    """
    h = hashlib.blake2b(digest_size=12)
    h.update(f"v{_KERNEL_CACHE_FORMAT_VERSION}|ma{max_age}|pop{population}".encode())
    h.update(f"|dds{_SIM_CONFIG.get('ddref_solid', 1.0)}".encode())
    h.update(f"|ddl{_SIM_CONFIG.get('ddref_leuk', 1.0)}".encode())
    # Risk-model-specific cap & latency feed straight into kernel construction.
    # Including them here ensures a REIC cache file is never reused for a
    # subsequent LAR run (or vice versa).
    h.update(f"|rm{_SIM_CONFIG.get('risk_model', 'REIC')}".encode())
    h.update(f"|amax{_max_attained_age_for_risk_model()}".encode())
    h.update(f"|lats{_solid_latency_for_risk_model()}".encode())
    h.update(f"|latl{_leuk_latency_for_risk_model()}".encode())
    # Fingerprint this module's source so any kernel-code change triggers a
    # recompute.  File read is ~1 ms for ~90 KB — negligible vs save/load.
    try:
        with open(__file__, "rb") as f:
            h.update(f.read())
    except OSError:
        pass
    return h.hexdigest()


def _kernel_cache_path(max_age: int, population: str) -> str:
    fp = _kernel_cache_fingerprint(max_age, population)
    # Include human-readable population tag in the filename for easier
    # debugging — the fingerprint is what guarantees correctness.
    safe_pop = "".join(c if c.isalnum() else "_" for c in population)[:24]
    return os.path.join(_kernel_cache_dir(),
                        f"kernels_v{_KERNEL_CACHE_FORMAT_VERSION}_{safe_pop}_{fp}.pkl")


def _load_kernel_cache_from_disk(max_age: int, population: str) -> bool:
    """Populate in-memory caches from a matching on-disk pickle, if any.

    Returns True on hit; False on miss or corruption.
    """
    path = _kernel_cache_path(max_age, population)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("format_version") != _KERNEL_CACHE_FORMAT_VERSION:
            return False
        # Drop only THIS population's entries: a disk load for one
        # population must not evict kernels another population's run built.
        for _d in (_KERNEL_CACHE, _SURV_CACHE, _LATENCY_PDF_CACHE):
            for _k in [_k for _k in _d if _k[-1] == population]:
                del _d[_k]
        _KERNEL_CACHE.update(blob["kernel_cache"])
        _SURV_CACHE.update(blob["surv_cache"])
        # Restore the RPG module-level arrays via in-place copy (assignment
        # would break any code holding the module attribute).
        global _RPG_SOLID, _RPG_LEUK_L1, _RPG_LEUK_L2
        global _RPG_SOLID_ICRP152, _RPG_LEUK_L1_ICRP152, _RPG_LEUK_L2_ICRP152
        global _RPG_BONE_BETA, _RPG_SKIN_BETA
        global _RPG_CANCER_LIST, _RPG_CT_TO_IDX, _RPG_REM_IDX
        _RPG_SOLID           = blob["rpg_solid"]
        _RPG_LEUK_L1         = blob["rpg_l1"]
        _RPG_LEUK_L2         = blob["rpg_l2"]
        _RPG_SOLID_ICRP152   = blob["rpg_solid_icrp"]
        _RPG_LEUK_L1_ICRP152 = blob["rpg_l1_icrp"]
        _RPG_LEUK_L2_ICRP152 = blob["rpg_l2_icrp"]
        _RPG_BONE_BETA       = blob["rpg_bone"]
        _RPG_SKIN_BETA       = blob["rpg_skin"]
        _RPG_CANCER_LIST     = list(blob["rpg_cancer_list"])
        _RPG_CT_TO_IDX       = dict(blob["rpg_ct_to_idx"])
        _RPG_REM_IDX         = int(blob["rpg_rem_idx"])
        _RPG_BY_POP[population] = blob["rpg_by_pop_entry"]
        return True
    except (OSError, pickle.UnpicklingError, KeyError, EOFError, ValueError):
        # Corrupt cache — delete and fall through to recompute.
        try:
            os.remove(path)
        except OSError:
            pass
        return False


def _save_kernel_cache_to_disk(max_age: int, population: str) -> None:
    """Atomically persist the current in-memory caches to disk.

    Never raises — a save failure must not break the simulation.
    """
    path = _kernel_cache_path(max_age, population)
    tmp = path + ".tmp"
    try:
        blob = {
            "format_version":   _KERNEL_CACHE_FORMAT_VERSION,
            "max_age":          max_age,
            "population":       population,
            # Persist only this population's entries: the in-memory caches
            # are population-keyed and may hold both populations at once.
            "kernel_cache":     {k: v for k, v in _KERNEL_CACHE.items()
                                 if k[-1] == population},
            "surv_cache":       {k: v for k, v in _SURV_CACHE.items()
                                 if k[-1] == population},
            "rpg_solid":        _RPG_SOLID.copy(),
            "rpg_l1":           _RPG_LEUK_L1.copy(),
            "rpg_l2":           _RPG_LEUK_L2.copy(),
            "rpg_solid_icrp":   _RPG_SOLID_ICRP152.copy(),
            "rpg_l1_icrp":      _RPG_LEUK_L1_ICRP152.copy(),
            "rpg_l2_icrp":      _RPG_LEUK_L2_ICRP152.copy(),
            "rpg_bone":         _RPG_BONE_BETA.copy(),
            "rpg_skin":         _RPG_SKIN_BETA.copy(),
            "rpg_cancer_list":  list(_RPG_CANCER_LIST),
            "rpg_ct_to_idx":    dict(_RPG_CT_TO_IDX),
            "rpg_rem_idx":      int(_RPG_REM_IDX),
            "rpg_by_pop_entry": _RPG_BY_POP.get(population, {}),
        }
        with open(tmp, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        # Clean up stray tmp; swallow the error so a read-only disk etc.
        # doesn't abort the run.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def precompute_kernels(max_age: int = 120, population: str = "Euro-American"):
    """Precompute all risk kernels and survival curves. Call once before iterations.
    Uses ICRP 152 para 46 cancer-free survival (population-specific).

    Fast path: if a matching on-disk cache exists, load it (<1 s).  Cache
    key encodes (max_age, population, DDREF, full source hash) so any
    config or code change automatically triggers a recompute.

    Idempotent: a per_year_average batch calls this once per year with an
    identical (max_age, population) config, so the on-disk path is the same
    every call.  When that path already matches the in-memory state
    (``_KERNEL_CACHE_LOADED_PATH``) the reload is skipped entirely -- the
    fingerprint in the path already encodes DDREF/config/source hash, so
    equality means the loaded state is exactly what reloading would
    produce.  Skipping also avoids clearing/repopulating ``_KERNEL_CACHE`` /
    ``_SURV_CACHE`` / the ``_RPG_*`` globals while a background digest
    thread (year_batch.run_yearly_simulation) may be reading them.
    """
    global _KERNEL_CACHE, _SURV_CACHE
    global _RPG_SOLID, _RPG_LEUK_L1, _RPG_LEUK_L2
    global _RPG_SOLID_ICRP152, _RPG_LEUK_L1_ICRP152, _RPG_LEUK_L2_ICRP152
    global _RPG_BONE_BETA, _RPG_SKIN_BETA

    path = _kernel_cache_path(max_age, population)
    # Idempotent fast path: same config -> same on-disk path -> this
    # population's entries are already loaded.  Tracked per population so a
    # second population's precompute cannot masquerade as a hit -- or evict
    # the first population's kernels.
    if (_KERNEL_CACHE_LOADED_PATHS.get(population) == path
            and ("leuk_L1", "M", 0, population) in _KERNEL_CACHE):
        # The kernel/survival caches are unchanged, but a caller may have
        # mutated the _RPG_* module globals in place (e.g. test helpers that
        # overwrite them to collapse every exposure age).  Restore them from
        # the per-population snapshot taken when the state was first loaded --
        # that snapshot is a copy, so it is unaffected by such mutation.  No
        # lock is needed here: the background digest thread reads
        # _KERNEL_CACHE/_SURV_CACHE (held under KERNEL_STATE_LOCK), not the
        # _RPG_* tables, and the rebinds are atomic.
        _pop = _RPG_BY_POP.get(population)
        if _pop and all(k in _pop for k in ("solid", "l1", "l2", "solid_icrp",
                                             "l1_icrp", "l2_icrp", "bone", "skin")):
            _RPG_SOLID = _pop["solid"].copy()
            _RPG_LEUK_L1 = _pop["l1"].copy()
            _RPG_LEUK_L2 = _pop["l2"].copy()
            _RPG_SOLID_ICRP152 = _pop["solid_icrp"].copy()
            _RPG_LEUK_L1_ICRP152 = _pop["l1_icrp"].copy()
            _RPG_LEUK_L2_ICRP152 = _pop["l2_icrp"].copy()
            _RPG_BONE_BETA = _pop["bone"].copy()
            _RPG_SKIN_BETA = _pop["skin"].copy()
        return

    with KERNEL_STATE_LOCK:
        # Re-check inside the lock: another thread may have just loaded it.
        if (_KERNEL_CACHE_LOADED_PATHS.get(population) == path
                and ("leuk_L1", "M", 0, population) in _KERNEL_CACHE):
            return

        # Fast path — try disk cache first.
        if _load_kernel_cache_from_disk(max_age, population):
            _KERNEL_CACHE_LOADED_PATHS[population] = path
            return

        # Slow path — full recompute.  Only this population's entries are
        # dropped: another population's kernels stay valid (mixed-population
        # runs need both).  Latency-onset CDFs derive from _KERNEL_CACHE, so
        # this population's CDFs are invalidated together.
        for _d in (_KERNEL_CACHE, _SURV_CACHE, _LATENCY_PDF_CACHE):
            for _k in [_k for _k in _d if _k[-1] == population]:
                del _d[_k]
        # Vectorised builder: a single in-process pass that broadcasts each kernel
        # formula across all 95 exposure ages at once, instead of a per-(sex, e) loop
        # over _precompute_one_sex_age.  Deterministic output; ~6-8x faster on cold runs.
        _precompute_kernels_vectorized(max_age, population)
        # Build vectorized risk-per-Gy lookup tables for batch simulation
        _build_rpg_tables(max_age, population)
        # Snapshot tables per population for mixed-population per-patient lookup.
        # Both linear (simulation) and ICRP 152 para 138 (Table 3.9 comparison)
        # versions are stored so the GUI can pick the appropriate one.
        _RPG_BY_POP[population] = {
            "solid":        _RPG_SOLID.copy(),
            "l1":           _RPG_LEUK_L1.copy(),
            "l2":           _RPG_LEUK_L2.copy(),
            "bone":         _RPG_BONE_BETA.copy(),
            "skin":         _RPG_SKIN_BETA.copy(),
            "solid_icrp":   _RPG_SOLID_ICRP152.copy(),
            "l1_icrp":      _RPG_LEUK_L1_ICRP152.copy(),
            "l2_icrp":      _RPG_LEUK_L2_ICRP152.copy(),
        }

        # Persist for next startup.  Never fails the run on I/O error.
        _save_kernel_cache_to_disk(max_age, population)
        _KERNEL_CACHE_LOADED_PATHS[population] = path


def get_cached_kernel(cancer_type: str, sex_code: str, exposure_age: int,
                      population: str) -> np.ndarray:
    """Get precomputed kernel. Falls back to live computation if not cached."""
    if cancer_type == "Leukaemia":
        return _KERNEL_CACHE.get(("leuk_L1", sex_code, exposure_age, population),
                                  leukaemia_kernels(sex_code, exposure_age)[0])
    k = _KERNEL_CACHE.get(("solid", cancer_type, sex_code, exposure_age, population))
    if k is not None:
        return k
    return solid_kernel_perGy(cancer_type, sex_code, exposure_age, population)


def get_cached_leukaemia(sex_code: str, exposure_age: int, population: str):
    """Get precomputed leukaemia kernels (L1, L2)."""
    L1 = _KERNEL_CACHE.get(("leuk_L1", sex_code, exposure_age, population))
    L2 = _KERNEL_CACHE.get(("leuk_L2", sex_code, exposure_age, population))
    if L1 is not None and L2 is not None:
        return L1, L2
    return leukaemia_kernels(sex_code, exposure_age)


def get_cached_survival(
    sex_code: str, exposure_age: int,
    population: str,
) -> np.ndarray:
    """Get precomputed ICRP 152 cancer-free survival curve (Table A.5/A.6/A.1-A.4).

    Length matches the active risk-model cap: 94 - max(1, e) + 1 entries for
    REIC, 99 - max(1, e) + 1 entries for LAR (ICRP 152 Table 4.1).
    """
    s = _SURV_CACHE.get((sex_code, exposure_age, population))
    if s is not None:
        return s
    _amax = _max_attained_age_for_risk_model()
    ages_arr = np.arange(max(1, exposure_age), _amax + 1, dtype=int)
    if ages_arr.size > 0:
        return _baseline_survival_curve(ages_arr, exposure_age, sex_code, population)
    return np.array([], dtype=float)


def kernels_precomputed(population=None) -> bool:
    """Check if kernels have been precomputed.

    ``population=None`` → True when any population's kernels exist; a concrete
    population checks only that population's entries (mixed-population runs
    precompute both, so an early ``("leuk_L1", "M", 0, pop)`` sentinel key is
    used rather than scanning the whole cache).
    """
    if population is None:
        return len(_KERNEL_CACHE) > 0
    return ("leuk_L1", "M", 0, population) in _KERNEL_CACHE


# VECTORIZED RISK-PER-GY LOOKUP TABLES (for batch MC iteration)
# _build_rpg_tables() collapses each kernel × survival product into a cumulative
# sum at max_attained_age, enabling a single advanced-index lookup for all exams.

_RPG_SOLID: np.ndarray = np.empty(0)       # (n_types, 3, max_e+1)
_RPG_LEUK_L1: np.ndarray = np.empty(0)     # (3, max_e+1)
_RPG_LEUK_L2: np.ndarray = np.empty(0)     # (3, max_e+1)
# ICRP 152 para 138 convention: REIC at 0.1 Gy × 10 (nonlinear survival
# correction S × exp(-0.1 × C_all)).  Used for Table 3.9 nominal-risk
# comparison; simulation path continues to use the linear per-Gy tables
# above with per-patient dose-dependent correction.
_RPG_SOLID_ICRP152: np.ndarray = np.empty(0)     # (n_types, 3, max_e+1)
_RPG_LEUK_L1_ICRP152: np.ndarray = np.empty(0)   # (3, max_e+1)
_RPG_LEUK_L2_ICRP152: np.ndarray = np.empty(0)   # (3, max_e+1)
_RPG_CANCER_LIST: list = []                 # ordered solid type names
_RPG_CT_TO_IDX: dict = {}                   # name -> axis-0 index
_RPG_REM_IDX: int = -1                      # Remainder index
_RPG_BONE_BETA: np.ndarray = np.empty(0)    # (3,) M/F/U
_RPG_SKIN_BETA: np.ndarray = np.empty(0)    # (3,) M/F/U
_RPG_BY_POP: dict = {}                      # population -> per-population RPG snapshot


def _expand_sex_average(mf: np.ndarray, axis: int = 1) -> np.ndarray:
    """Expand a sex axis from 2 entries (M, F) to 3 (M, F, U=mean).

    The unknown-sex slice is filled with the arithmetic mean of the M/F
    slices, matching the sex-averaged convention used throughout the
    vectorized engine.
    """
    shape = list(mf.shape)
    shape[axis] = 3
    out = np.zeros(shape, dtype=mf.dtype)
    indexer = [slice(None)] * mf.ndim
    indexer[axis] = slice(0, 2)
    out[tuple(indexer)] = mf
    indexer[axis] = 2
    out[tuple(indexer)] = 0.5 * (
        np.take(mf, 0, axis=axis) + np.take(mf, 1, axis=axis)
    )
    return out


def _bone_skin_beta(params: dict, sex_code: str) -> float:
    """Nominal lifetime risk coefficient per Gy for Bone/Skin (ICRP 59/60).

    Bone uses the EAR beta base, Skin uses the ERR beta base (ICRP 152
    Table 3.1).  For unknown sex the M/F coefficients are averaged.
    """
    w_err = float(params.get("ERR_weight", 0.0))
    w_ear = float(params.get("EAR_weight", 0.0))
    err_b = params.get("ERR_beta_base", {})
    ear_b = params.get("EAR_beta_base", {})
    if sex_code in ("M", "F"):
        return w_err * float(err_b.get(sex_code, 0.0)) + w_ear * float(ear_b.get(sex_code, 0.0))
    return (
        w_err * 0.5 * (float(err_b.get("M", 0.0)) + float(err_b.get("F", 0.0)))
        + w_ear * 0.5 * (float(ear_b.get("M", 0.0)) + float(ear_b.get("F", 0.0)))
    )


def _build_rpg_tables(max_age: int = 120, population: str = "Euro-American"):
    """Build cumulative risk-per-Gy (RPG) lookup tables for batch MC iteration.

    The Monte Carlo engine in ``vectorized`` looks up
    each exam's lifetime risk via a single advanced-index operation on
    these tables, replacing the per-exam REIC integral (ICRP 152
    paras 51-52) with an O(1) array lookup.  Two parallel sets of tables
    are built:

    * ``_RPG_SOLID`` / ``_RPG_LEUK_L1`` / ``_RPG_LEUK_L2`` -- linear
      per-Gy tables for the simulation path.  The dose-dependent
      survival correction (ICRP 152 para 47) is applied per-exam at
      look-up time so the simulation handles arbitrary exam doses.

    * ``_RPG_SOLID_ICRP152`` / ``_RPG_LEUK_*_ICRP152`` -- ICRP 152
      para 138 convention (Table 3.9 reference detriment): REIC is
      computed at 0.1 Gy with the nonlinear cancer-free-survival
      correction ``S(a, 0.1 | e) = S0(a|e) * exp(-0.1 * C_all(a, e))``
      baked in, then divided by 0.1 to give a per-Gy table.  Used by
      the analysis layer for nominal-risk reporting; never by the
      simulation hot loop.

    Tensor shapes (after the build)
    -------------------------------
    * ``rpg_solid[ct_idx, sex_idx, e]``
      = Sum_{t = max(1, e) .. max_a} kernel_ct(t, e) * S(t | e)
      = lifetime risk per Gy for ``cancer_type`` index ``ct_idx`` if
        exposed at age ``e``, cumulated to max_attained_age.
    * ``rpg_leuk_l1[sex_idx, e]`` and ``rpg_leuk_l2[sex_idx, e]``
      provide the L1 (linear-in-dose) and L2 (quadratic-in-dose) basis
      arrays for the ICRP 152 leukaemia LQ model: lifetime leukaemia
      risk = ``L1 * d + L2 * d^2``.
    * ``sex_idx``: 0 = M, 1 = F, 2 = U (sex-averaged).

    Excluded from the solid tables
    ------------------------------
    * Leukaemia -- handled separately via the L1 / L2 tables.
    * Bone, Skin -- ICRP 152 para 64 retains the ICRP 60 (bone) /
      ICRP 59 (skin) total lifetime nominals unchanged; no
      kernel*survival integration is needed.  Their per-Gy beta values
      are stored in ``_RPG_BONE_BETA`` and ``_RPG_SKIN_BETA``.
    * Brain, Kidney, Pancreas -- members of the "Other solid /
      Remainder" group (ICRP 103 B.140-141, 13 organs total).  Their
      doses are folded into the Remainder mean-dose calculation
      upstream and their risk is captured by the single ``"Remainder"``
      column.  Listing them as independent Bernoulli columns would
      double-count both probability and reported events.

    Called automatically at the end of ``precompute_kernels``.
    """
    global _RPG_SOLID, _RPG_LEUK_L1, _RPG_LEUK_L2
    global _RPG_SOLID_ICRP152, _RPG_LEUK_L1_ICRP152, _RPG_LEUK_L2_ICRP152
    global _RPG_CANCER_LIST, _RPG_CT_TO_IDX, _RPG_REM_IDX
    global _RPG_BONE_BETA, _RPG_SKIN_BETA

    # ICRP 152 Table 4.1: 94 y for REIC, 99 y for LAR.
    max_a = _max_attained_age_for_risk_model()

    # Build ordered list of solid cancer types for the table.
    # Exclude leukaemia (handled separately), Bone/Skin (nominal lifetime, no survival integration),
    # and OTHER_SOLID_REMAINDER_SITES sub-sites other than "Remainder" (Brain, Salivary_Glands,
    # Kidney, Pancreas) — their doses are folded into rem_mean_dose and their risk is captured by
    # the single "Remainder" column (ICRP 103 §A.158 / §B.140 + ICRP 152 Table 3.1).
    # Listing them separately would double-count probability and cancer events.
    _rem_subsites = OTHER_SOLID_REMAINDER_SITES - {"Remainder"}
    solid_types = [ct for ct in PRECOMPUTE_CANCER_TYPES
                   if ct not in ("Leukaemia", "Bone", "Skin")
                   and ct not in _rem_subsites]
    if "Remainder" not in solid_types:
        solid_types.append("Remainder")
    _RPG_CANCER_LIST = solid_types
    _RPG_CT_TO_IDX = {ct: i for i, ct in enumerate(solid_types)}
    _RPG_REM_IDX = _RPG_CT_TO_IDX.get("Remainder", -1)

    n_types = len(solid_types)
    # ICRP 152 para 46-47 / para 138 dose for the nonlinear survival correction.
    _ICRP_REFERENCE_DOSE = 0.1  # Gy

    # ---- 1. Stack every kernel and survival curve into dense tensors -----
    # Shapes:
    #   K (n_types, 2, max_age+1, max_a+1)  -- per-cancer per-Gy annual kernels
    #   L1/L2 (2, max_age+1, max_a+1)       -- leukaemia LQ basis arrays
    #   surv  (2, max_age+1, max_a+1)       -- cancer-free survival S(a|e)
    # Cells with no precomputed kernel / survival value stay at zero; the
    # downstream cumsum simply treats zero entries as no-op additions, so
    # missing (sex, e) entries do not contribute to the cumulative sums.
    K_tensor    = np.zeros((n_types, 2, max_age + 1, max_a + 1), dtype=np.float64)
    L1_tensor   = np.zeros((2, max_age + 1, max_a + 1), dtype=np.float64)
    L2_tensor   = np.zeros((2, max_age + 1, max_a + 1), dtype=np.float64)
    surv_tensor = np.zeros((2, max_age + 1, max_a + 1), dtype=np.float64)

    for s_idx, sex in enumerate(("M", "F")):
        for e in range(max_age + 1):
            surv = _SURV_CACHE.get((sex, e, population))
            if surv is None or surv.size == 0:
                continue
            # The REIC integral starts at attained age max(1, e) per ICRP 152
            # para 46 (cancer-free survival is conditional on having survived
            # to age e).  Place the survival values into the tensor so
            # surv_tensor[s, e, max(1, e) : max(1, e) + n] carries S(a|e) and
            # earlier ages stay at zero.
            start_a = max(1, e)
            n = min(int(surv.size), max_a + 1 - start_a)
            if n <= 0:
                continue
            surv_tensor[s_idx, e, start_a:start_a + n] = surv[:n]

            for ct_idx, ct in enumerate(solid_types):
                K = _KERNEL_CACHE.get(("solid", ct, sex, e, population))
                if K is not None:
                    # Cached solid kernels are length max_age+1.
                    K_tensor[ct_idx, s_idx, e, :] = K[:max_a + 1]
            L1 = _KERNEL_CACHE.get(("leuk_L1", sex, e, population))
            if L1 is not None:
                L1_tensor[s_idx, e, :] = L1[:max_a + 1]
            L2 = _KERNEL_CACHE.get(("leuk_L2", sex, e, population))
            if L2 is not None:
                L2_tensor[s_idx, e, :] = L2[:max_a + 1]

    # ── 2. Zero-mask ages a < max(1, e) so the cumulative all-cancer sum
    #       starts from the correct age bin (matches ``total += K[ages_arr[:n]]``
    #       which only adds kernels at ages in [max(1, e), max_a]).
    age_grid     = np.arange(max_a + 1, dtype=np.int64)
    start_a_grid = np.maximum(1, np.arange(max_age + 1, dtype=np.int64))
    valid_age_mask = (age_grid[None, :] >= start_a_grid[:, None])   # (max_age+1, max_a+1)
    # In-place multiply by the bool mask (True→1.0, False→0.0).
    K_tensor  *= valid_age_mask[None, None, :, :]
    L1_tensor *= valid_age_mask[None, :, :]
    L2_tensor *= valid_age_mask[None, :, :]

    # ---- 3. All-site cumulative excess C_all(a, e) -- ICRP 152 para 47 -----
    # Per para 47 the dose-corrected survival uses the FULL excess incidence
    # at dose d:  μ_ic(x,e,d) − μ_ic(x).  For solids that is linear in d
    # (kernel_c · d).  For leukaemia the LQ dose-response gives L1·d + L2·d²,
    # so both terms must enter the survival exponent (the quadratic term is
    # carried by ``c_quad``).  Summation order is fixed (cancers first, then
    # leukaemia L1) so the float-add order is deterministic across runs.
    total_ak = np.zeros((2, max_age + 1, max_a + 1), dtype=np.float64)
    for ct_idx in range(n_types):
        total_ak += K_tensor[ct_idx]
    total_ak += L1_tensor
    c_lin_cumul  = np.cumsum(total_ak,  axis=-1)
    c_quad_cumul = np.cumsum(L2_tensor, axis=-1)

    # 4. Modified survival at 0.1 Gy for ICRP 152 §4.2 / Table 3.6
    #     S(a,e | 0.1) ≈ S0 × exp(−0.1·C_lin − 0.01·C_quad)
    surv_icrp = surv_tensor * np.exp(
        -_ICRP_REFERENCE_DOSE       * c_lin_cumul
        - (_ICRP_REFERENCE_DOSE**2) * c_quad_cumul
    )

    # ---- 5. Per-cancer cumulative risk-per-Gy tables -----------------------
    # rpg(ct, sex, e, a) = Sum_{t = max(1, e) .. a} kernel_ct(t, e) * S(t | e).
    # Products vanish at ages where surv_tensor is zero (before max(1, e)),
    # so the cumsum along the attained-age axis is equivalent to a per-
    # (ct, s, e) loop with ages_arr = [max(1, e), ..., max_a].
    rpg_mf          = np.cumsum(K_tensor * surv_tensor[None, :, :, :], axis=-1)
    rpg_mf_icrp     = np.cumsum(K_tensor * surv_icrp[None, :, :, :], axis=-1)
    rpg_l1_mf       = np.cumsum(L1_tensor * surv_tensor, axis=-1)
    rpg_l2_mf       = np.cumsum(L2_tensor * surv_tensor, axis=-1)
    rpg_l1_mf_icrp  = np.cumsum(L1_tensor * surv_icrp,  axis=-1)
    rpg_l2_mf_icrp  = np.cumsum(L2_tensor * surv_icrp,  axis=-1)

    # 6. Expand sex axis to 3 entries (M, F, U=mean)
    rpg         = _expand_sex_average(rpg_mf, axis=1)
    rpg_icrp    = _expand_sex_average(rpg_mf_icrp, axis=1)
    rpg_l1      = _expand_sex_average(rpg_l1_mf, axis=0)
    rpg_l2      = _expand_sex_average(rpg_l2_mf, axis=0)
    rpg_l1_icrp = _expand_sex_average(rpg_l1_mf_icrp, axis=0)
    rpg_l2_icrp = _expand_sex_average(rpg_l2_mf_icrp, axis=0)

    # Collapse attained-age axis: the simulation only ever uses the value at
    # max_attained_age (94), because REIC already bakes in cancer-free survival
    # S(a|e) and the per-exam surv_correction handles prior-dose / frailty.
    # Keeping the full (max_a+1)-wide axis was ~95× wasted memory.
    _RPG_SOLID = rpg[..., max_a]
    _RPG_LEUK_L1 = rpg_l1[..., max_a]
    _RPG_LEUK_L2 = rpg_l2[..., max_a]
    _RPG_SOLID_ICRP152 = rpg_icrp[..., max_a]
    _RPG_LEUK_L1_ICRP152 = rpg_l1_icrp[..., max_a]
    _RPG_LEUK_L2_ICRP152 = rpg_l2_icrp[..., max_a]

    # Bone and Skin: nominal lifetime betas (no survival integration)
    bone_beta = np.zeros(3, dtype=np.float64)
    skin_beta = np.zeros(3, dtype=np.float64)
    for s_idx, sex in enumerate(("M", "F", "U")):
        bone_beta[s_idx] = _bone_skin_beta(cancer_risk_params.get("Bone", {}), sex)
        skin_beta[s_idx] = _bone_skin_beta(cancer_risk_params.get("Skin", {}), sex)
    _RPG_BONE_BETA = bone_beta
    _RPG_SKIN_BETA = skin_beta


def get_vectorized_rpg():
    """Return precomputed tables for vectorized simulation.

    Returns
    -------
    tuple : (rpg_solid, rpg_leuk_l1, rpg_leuk_l2,
             cancer_type_list, ct_to_idx, rem_idx,
             bone_beta, skin_beta)
    """
    return (_RPG_SOLID, _RPG_LEUK_L1, _RPG_LEUK_L2,
            _RPG_CANCER_LIST, _RPG_CT_TO_IDX, _RPG_REM_IDX,
            _RPG_BONE_BETA, _RPG_SKIN_BETA)


def get_vectorized_rpg_2pop():
    """Return RPG tables stacked over both populations for per-patient lookup.

    Index 0 = Euro-American, Index 1 = Asian.

    Returns the same 8-element tuple as ``get_vectorized_rpg`` but with the
    first two axes carrying the population dimension:

    * ``rpg_solid``  shape ``(2, n_types, 3, max_e+1)``
    * ``rpg_l1/l2``  shape ``(2, 3, max_e+1)``
    * ``bone_beta / skin_beta``  shape ``(2, 3)``

    The shared metadata (cancer_type_list, ct_to_idx, rem_idx) is taken from
    whichever population was last precomputed (both have the same structure).

    Returns ``None`` if both populations have not yet been precomputed.
    """
    ea   = _RPG_BY_POP.get("Euro-American")
    asian = _RPG_BY_POP.get("Asian")
    if ea is None or asian is None:
        return None
    return (
        np.stack([ea["solid"], asian["solid"]]),  # (2, n_types, 3, max_e+1)
        np.stack([ea["l1"],    asian["l1"]]),      # (2, 3, max_e+1)
        np.stack([ea["l2"],    asian["l2"]]),
        _RPG_CANCER_LIST, _RPG_CT_TO_IDX, _RPG_REM_IDX,
        np.stack([ea["bone"],  asian["bone"]]),    # (2, 3)
        np.stack([ea["skin"],  asian["skin"]]),
    )


