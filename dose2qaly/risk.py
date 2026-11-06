"""ICRP 152 lifetime cancer-risk calculation and Monte Carlo outcome sampling.

This module is the canonical home for the per-exam ICRP 152 lifetime risk
integral and the Monte Carlo Bernoulli outcome draw.  ``main.py`` imports
``compute_lifetime_risk`` and ``_get_cancer_params_cached`` from here and
re-exports ``compute_lifetime_risk`` through ``dose2qaly.__init__``.

It implements the REIC (default) and LAR (opt-in) integrals, the kernel-cache
fast path, DDREF sensitivity hooks, the ICRP 103 Bone/Skin total-lifetime
treatment, ICRP 152 Remainder dose-proportional sampling, and heritable
detriment via the gonad-dose channel.

Public API
----------
* ``compute_lifetime_risk(...)``   ICRP 152 LAR / REIC integral per cancer site.

Module-private helpers
----------------------
* ``_iter_modeled_cancer_types``   Re-exported convenience over exam_doses.
* ``_is_excluded_cancer_type``     Sex-based cancer-type filter.
* ``_resolve_cancer_dose``         Cancer-site dose lookup with fallback.
* ``_get_cancer_params_cached``    Lethality / detriment-factor cache.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

from .exam_doses import (
    _SEX_EXCLUDED_CANCER_TYPES,
    _iter_modeled_cancer_types,
)
from .risk_models import (
    cancer_risk_params,
    lookup_cancer_params,
    _bone_skin_beta,
    get_organ_dose_for_cancer,
    MAX_ATTAINED_AGE_ICRP152,
    get_cached_kernel,
    get_cached_leukaemia,
    get_cached_survival,
    OTHER_SOLID_REMAINDER_SITES,
    REMAINDER_ORGAN_MULTIPLICITY,
    N_REMAINDER_ORGANS_ICRP,
    _SIM_CONFIG,
)


# 1. CANCER-SITE FILTERS / DOSE LOOKUP

def _is_excluded_cancer_type(ct: str, sex_code: str) -> bool:
    """Return True if cancer type is excluded for the given sex."""
    excluded = _SEX_EXCLUDED_CANCER_TYPES.get(sex_code, _SEX_EXCLUDED_CANCER_TYPES["U"])
    return ct in excluded or (ct == "Breast" and sex_code == "M")


def _resolve_cancer_dose(ct: str, cdoses: Dict[str, float], od: Dict[str, float]) -> float:
    """Resolve organ dose for a cancer type, preferring precomputed doses."""
    if cdoses:
        return cdoses.get(ct, 0.0)
    dose = get_organ_dose_for_cancer(ct, od)
    if dose <= 0:
        dose = od.get(ct, 0.0)
    return dose


# 2. LIFETIME RISK CALCULATION (ICRP 152 LAR / REIC)

def compute_lifetime_risk(
    organ_doses_gy: Dict[str, float],
    age_at_exposure: float,
    sex_code: str,
    max_attained_age: int = MAX_ATTAINED_AGE_ICRP152,
    ddref_solid: float = None,
    ddref_leuk: float = None,
    population: str = None,
    risk_model: str = None,
    precomputed_cancer_doses: Dict[str, float] = None,
    precomputed_modeled_types: List[str] = None,
    remainder_individual_doses: Dict[str, float] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute lifetime excess cancer risk per site (REIC or LAR) per ICRP 152.

    risk_model='REIC' (default) or 'LAR':
      Both use cancer-free survival S(a,e|0) per ICRP 152 para 46:
        μ̄(x) = μ_m(x) − μ_mac(x) + μ_iac(x), conditional from exposure age e.
      For REIC: dose-adjusted survival correction is applied inside
        compute_lifetime_excess_cancer_risk_REIC (negligible at diagnostic doses).
      For LAR: background survival only (ICRP 152 para 52 footnote).
    Uses precomputed kernels when available for speed.
    ddref_solid/ddref_leuk allow Bayesian sensitivity testing.
    """
    if ddref_solid is None:
        ddref_solid = _SIM_CONFIG["ddref_solid"]
    if ddref_leuk is None:
        ddref_leuk = _SIM_CONFIG["ddref_leuk"]
    if population is None:
        population = _SIM_CONFIG["population"]
    if risk_model is None:
        risk_model = _SIM_CONFIG["risk_model"]
    lifetime_risk = {}
    if not organ_doses_gy and not precomputed_cancer_doses:
        return lifetime_risk, {}

    e_int = int(max(0, min(age_at_exposure, MAX_ATTAINED_AGE_ICRP152)))
    end_age = int(max_attained_age)
    ages_arr = np.arange(max(1, e_int), end_age + 1, dtype=int)
    if ages_arr.size == 0:
        return lifetime_risk, {}

    # Use cached survival when available; get_cached_survival falls back to live
    # computation. Trim to the caller's requested age range.
    surv = get_cached_survival(sex_code, e_int, population)
    n = min(surv.size, ages_arr.size)
    surv = surv[:n]
    ages_arr = ages_arr[:n]

    # "Other solid" / Remainder group (ICRP 152 §3.1)
    # ICRP 152 requires the mean absorbed dose across all Remainder organs.
    # Apply the Remainder kernel once to the weighted mean dose, then distribute
    # proportionally so the sum of individual risks equals the ICRP total.
    _cdoses = precomputed_cancer_doses or {}
    rem_dose_map: Dict[str, float] = {}
    for ct in sorted(OTHER_SOLID_REMAINDER_SITES):
        if ct not in cancer_risk_params:
            continue
        if _cdoses:
            dose = _cdoses.get(ct, 0.0)
        else:
            if _is_excluded_cancer_type(ct, sex_code):
                continue
            dose = _resolve_cancer_dose(ct, {}, organ_doses_gy)
        if dose > 0:
            rem_dose_map[ct] = float(dose)

    if rem_dose_map:
        weighted_sum = sum(
            d * REMAINDER_ORGAN_MULTIPLICITY.get(ct, 1) for ct, d in rem_dose_map.items()
        )
        # ICRP 152 §3.1: mean over ALL 13 Remainder organs, including those with zero
        # dose (e.g., abdominal organs in a head CT). Always divide by full group size.
        total_mult_icrp = N_REMAINDER_ORGANS_ICRP
        mean_rem_dose = weighted_sum / total_mult_icrp
        try:
            K_rem = get_cached_kernel("Remainder", sex_code, e_int, population)
            total_rem_risk = float(np.sum((mean_rem_dose / ddref_solid) * K_rem[ages_arr] * surv))
        except Exception:
            _logger.warning(
                f"Remainder kernel lookup failed for sex={sex_code}, age={e_int}, "
                f"population={population}. Setting remainder risk to 0.0."
            )
            total_rem_risk = 0.0
        # Dose-proportional weights for subtype sampling at draw time
        rem_weights = {
            ct: (dose * REMAINDER_ORGAN_MULTIPLICITY.get(ct, 1)) / max(weighted_sum, 1e-15)
            for ct, dose in rem_dose_map.items()
        }
        # Add individual remainder organ doses if available
        individual_doses = remainder_individual_doses or {}
        if individual_doses:
            rem_weights.pop("Remainder", None)
            for organ, dose in individual_doses.items():
                if dose > 0:
                    # Individual organs have multiplicity 1
                    rem_weights[organ] = dose / max(weighted_sum, 1e-15)
        lifetime_risk["Remainder"] = max(0.0, total_rem_risk)
    else:
        rem_weights = {}

    # All other cancer sites
    modeled = precomputed_modeled_types if precomputed_modeled_types is not None else _iter_modeled_cancer_types(organ_doses_gy, sex_code)
    for ct in modeled:
        if ct in OTHER_SOLID_REMAINDER_SITES:
            continue  # already handled above

        dose = _resolve_cancer_dose(ct, _cdoses, organ_doses_gy)
        if dose <= 0:
            continue

        if ct in ("Bone", "Skin"):
            # ICRP 152 §3.1.4/§5.1.2: bone and skin nominal risks (ICRP 59/60) are
            # total LIFETIME risk per Gy, NOT annual rates. Must NOT integrate over years.
            # Bone: 100%EAR (Table 3.1) → coefficient in EAR_beta_base.
            # Skin: 100%ERR (Table 3.1, CRR model) → coefficient in ERR_beta_base.
            params = lookup_cancer_params(ct)
            beta = _bone_skin_beta(params, sex_code)
            lifetime_risk[ct] = max(0.0, float(dose * beta))
            continue

        try:
            if ct == "Leukaemia":
                L1, L2 = get_cached_leukaemia(sex_code, e_int, population)
                k1 = L1[ages_arr]
                k2 = L2[ages_arr]
                yearly = (dose * k1 + (dose ** 2) * k2) / ddref_leuk
                risk = float(np.sum(yearly * surv))
            else:
                K = get_cached_kernel(ct, sex_code, e_int, population)
                kslice = K[ages_arr]
                risk = float(np.sum((dose / ddref_solid) * kslice * surv))

            lifetime_risk[ct] = max(0.0, risk)
        except Exception:
            _logger.warning(
                f"Kernel lookup failed for cancer type={ct}, sex={sex_code}, age={e_int}, "
                f"population={population}. Setting risk to 0.0."
            )
            lifetime_risk[ct] = 0.0

    return lifetime_risk, rem_weights
