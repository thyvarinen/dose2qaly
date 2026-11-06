"""
Research Question analysis module
=================================

Dose-reduction benefit per imaging region × 10-year age group
(per 100 000 imaging procedures).

Thesis question (Finnish original, paraphrased):
    "Study the effect of dose reductions per imaging region in 10-year
    age groups -> what benefits (cancers / deaths / QALYs avoided) are
    achieved per 100 000 examinations in each region?
    Results in e.g. 10% dose-reduction increments (plus baselines).
    Regions: heart, lungs, head, extremities, ..."

Scientific principles
----------------------
1. **QALY is the headline metric** (years of healthy life lost) so the
   user can compare regions in the same unit clinicians use to weigh
   benefit vs. harm.
2. **All projections use linear-no-threshold (LNT) scaling** of the
   per-Sv risk coefficient observed in the simulation, within the small
   dose-change window (-50%..+50%).  This is the ICRP / NCRP convention
   for low-dose extrapolation and matches the risk model used elsewhere
   in this codebase.

Pipeline
--------
  Monte-Carlo output (stats, exam_data, cancer_events)
    → ``compute_region_age_dose_sweep``   (per-(region, age) baseline + LNT sweep)
    → ``plot_region_age_sweep`` / ``format_region_age_sweep_summary``
      (GUI figure / text)
"""
from __future__ import annotations

import bisect
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as _mpl_Figure
import numpy as np
import textwrap

from .columnar_exams import ColumnarExamsList
from .columns import EventColumns, is_event_columns as _is_event_columns
from .analysis import modality_name as _modality_name
from .analysis import (
    _ensure_event_columns as _ensure_event_columns,
    _columnar_dict_to_event_columns as _columnar_dict_to_event_columns,
    _DENTAL_SUBTYPE_DISPLAY as _DENTAL_SUBTYPE_DISPLAY,
)
from .region_resolver import radiography_reporting_region as _rpt_reg
from .region_resolver import (
    CT_UNLOCALIZED as _CT_UNLOCALIZED,
    CT_OTHER_COMBINATIONS as _CT_OTHER_COMBINATIONS,
    RARE_COMBO_THRESHOLD as _RARE_COMBO_THRESHOLD,
    is_combo_region as _is_combo_region,
    DENTAL_IMAGING_SUBTYPE_REGIONS as _DENTAL_IMAGING_SUBTYPE_REGIONS,
)

# Lower-cased mirror of ``_DENTAL_IMAGING_SUBTYPE_REGIONS``, for the few
# call sites that compare against a lower-cased region string.
_DENTAL_IMAGING_SUBTYPE_REGIONS_LOWER = frozenset(
    r.lower() for r in _DENTAL_IMAGING_SUBTYPE_REGIONS)
# "Dental CBCT" exams are frequently coded under another archive
# modality (DX/CR, PX — combo panoramic/CBCT units export plain-film
# codes); exam_doses reports every one of them under the CBCT modality
# whatever the archive code, so every tab sees them as CBCT.
# ``_grouping_modality_array`` applies the same rewrite (and the
# "Dental imaging" pseudo-modality) wherever cells are keyed.
_DENTAL_CBCT_REGION = "Dental CBCT"
_DENTAL_CBCT_MODALITY = "CBCT"


def _grouping_modality_array(mod_arr: np.ndarray, reg_arr: np.ndarray,
                             dental_cbct_raw: bool = False) -> np.ndarray:
    """Sweep-cell modality for each exam/event: the raw modality, except
    dental-imaging subtypes -> "Dental imaging" and "Dental CBCT" ->
    "CBCT".  ``dental_cbct_raw=True`` leaves "Dental CBCT" on its raw
    modality -- used by the simulated per-iteration cell store, which
    keeps the raw split so a modality selection can be corrected exactly
    (see _sim_cohort_series).  Returns a copy only when something
    changes."""
    _dental = np.isin(reg_arr, list(_DENTAL_IMAGING_SUBTYPE_REGIONS))
    _cbct = (reg_arr == _DENTAL_CBCT_REGION) if not dental_cbct_raw else None
    if not (_dental.any() or (_cbct is not None and _cbct.any())):
        return mod_arr
    out = mod_arr.copy()
    out[_dental] = "Dental imaging"
    if _cbct is not None:
        out[_cbct] = _DENTAL_CBCT_MODALITY
    return out


def _modality_matches(mod_val: Any, reg_val: Any, mod_keys) -> bool:
    """Whether an exam/event belongs to the selected modality key set,
    using the same grouping as the sweep cells (dental-imaging subtypes
    only match "DENTAL IMAGING", "Dental CBCT" only matches "CBCT")."""
    reg = str(reg_val or "").strip()
    if reg.lower() in _DENTAL_IMAGING_SUBTYPE_REGIONS_LOWER:
        return "DENTAL IMAGING" in mod_keys
    if reg == _DENTAL_CBCT_REGION:
        return _DENTAL_CBCT_MODALITY in mod_keys
    return _norm_modality(mod_val) in mod_keys


# Display-only short names for the research-question figures' y-axis.
# The region *key* stays as-is (it is shared with the resolver, caches,
# CSV exports and the GUI's unmapped-description list).
_REGION_DISPLAY_SHORT = {_CT_UNLOCALIZED: "Unknown"}
from .risk import compute_lifetime_risk as _compute_lifetime_risk
from .constants import Z_95
from .style import modality_color as _modality_color
from .style import modality_linestyle as _modality_linestyle
from .style import modality_marker as _modality_marker
from .risk_models import (
    get_cancer_params_cached as _get_cancer_params_cached,
    expected_lifespan_pmf as _expected_lifespan_pmf,
    expected_yll_moments_for_site as _expected_yll_moments_for_site,
    get_organ_dose_for_cancer as _get_organ_dose_for_cancer,
    get_leukaemia_EAR_params as _get_leukaemia_EAR_params,
    leukaemia_quadratic_share as _leukaemia_quadratic_share,
    LEUKAEMIA_QUAD_LINEAR_RATIO as _LEUK_QL_RATIO,
    _require_population,
)


# Manuscript-figure font stack
#
# The 11 research-question plotting functions below draw the figures that go
# into the manuscript (Lancet-style sans-serif: Helvetica/Arial).  Setting
# ``mpl.rcParams["font.family"]`` at function entry leaked into the
# process-wide default: once any research figure rendered (the Research
# Questions tab is deferred until the simulation finishes), every later
# Summary-tab redraw created its Text artists in Arial, which at the same
# point size looks visibly smaller than the DejaVu Sans ``style.apply_style``
# pins at GUI start.  The decorator below scopes the font stack to the
# function body via ``mpl.rc_context`` so the global default is restored on
# exit, and pre-creates tick 0 on each axis while the context is still
# active (ticks are created lazily at draw time and new ticks copy their
# font from ``majorTicks[0]``).
_RQ_FONT_RC = {"font.family": "sans-serif"}
# A Text artist stores only the generic name "sans-serif"; the font manager
# resolves it through ``font.sans-serif`` at draw time, i.e. after the
# rc_context has exited, so the fallback list itself must be global.  This
# is safe for the rest of the GUI, whose default family is the concrete
# "DejaVu Sans" and never consults the generic list.
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"]
# Vector-export settings are harmless to the GUI look and only matter at
# savefig time, so they stay global.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["ps.useafm"] = True


def _rq_fonts(func):
    """Draw a manuscript figure with the research-question font stack
    without leaking it into the process-wide rcParams (the GUI's other
    charts are DejaVu Sans via style.apply_style)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(_RQ_FONT_RC):
            result = func(*args, **kwargs)
            for a in (*args, *kwargs.values()):
                if isinstance(a, _mpl_Figure):
                    for ax in a.axes:
                        for axis in (ax.xaxis, ax.yaxis):
                            axis.majorTicks[0]; axis.minorTicks[0]
        return result
    return wrapper


# ICRP 152 reference (nominal) detriment coefficients
#
# Effective dose is NOT a purely physical quantity.  The tissue weighting
# factors w_T that define it are derived from the ICRP reference-population
# radiation detriment, which already embeds lethality (k), quality of life
# for non-fatal disease (q_min) and relative years of life lost (l_rel),
# sex-averaged and age-at-exposure-averaged over the ICRP reference
# population (ICRP 152 Annex A Table A.7).  A dose-only prediction of health
# benefit therefore does not assume "no age" -- it assumes the ICRP
# REFERENCE-POPULATION AVERAGE patient.
#
# ICRP 152 Table 3.9 (p.51), as reproduced by ICRP Task Group 102:
#     Total detriment              = 574.0  per 10,000 per Sv
#     of which Gonads (heritable)  =  25.4  per 10,000 per Sv  (0.044 of total)
#     => cancer-only detriment     = 548.6  per 10,000 per Sv
#
# The heritable term is EXCLUDED from the reference used here because the
# QALYs (QALY) this module compares against are
# accumulated from CANCER events only: ``vectorized.py`` adds heritable


# Shared helpers

def _safe_iters(stats: dict) -> int:
    """Iterations count, never zero (so divisions are safe)."""
    try:
        return max(1, int((stats or {}).get("iterations", 0) or 0))
    except Exception:
        return 1


def _event_arrays(
    events: Any,
    _decade_vec,
    dental_cbct_raw: bool = False,
):
    """Extract modality/region/age/decade/iteration/died/QALY/life arrays from events.

    Also returns ``ctype_ev`` / ``sex_ev`` / ``exidx_ev``, which the caller needs
    to resolve the per-event leukaemia quadratic dose-response share.

    Accepts either a list of event dicts or an ``EventColumns`` instance.
    Iteration indices are returned unchanged; callers clip main events to
    ``[0, iters_main-1]`` if desired.  ``dental_cbct_raw`` is passed to
    ``_grouping_modality_array``.
    """
    if _is_event_columns(events):
        mod_ev = np.asarray(events.modality, dtype=object)
        reg_ev = np.asarray(events.region, dtype=object)
        age_ev = np.asarray(events.age_at_exposure, dtype=float)
        di_ev = _decade_vec(age_ev)
        it_ev = np.asarray(events.iteration, dtype=np.int64)
        died_ev = np.asarray(events.died, dtype=float)
        QALY_ev = np.asarray(events.QALY_lost, dtype=float)
        life_ev = np.asarray(events.life_lost_years, dtype=float)
        ctype_ev = np.asarray(events.cancer_type, dtype=object)
        sex_ev = np.asarray(events.sex, dtype=object)
        exidx_ev = np.asarray(events.exam_index, dtype=np.int64)
    else:
        ev_list = events or []
        mod_ev = np.array([str(ev.get("modality", "") or "") for ev in ev_list], dtype=object)
        reg_ev = np.array([str(ev.get("region") or "Unknown") for ev in ev_list], dtype=object)
        age_ev = np.array([float(ev.get("age", 0) or 0.0) for ev in ev_list], dtype=float)
        di_ev = _decade_vec(age_ev)

        def _it_of(ev) -> int:
            try:
                return int(ev.get("iteration", 0) or 0)
            except (TypeError, ValueError):
                return 0

        it_ev = np.array([_it_of(ev) for ev in ev_list], dtype=np.int64)
        died_ev = np.array([1.0 if ev.get("died") else 0.0 for ev in ev_list], dtype=float)
        QALY_ev = np.array([float(ev.get("QALY_lost", 0.0) or 0.0) for ev in ev_list], dtype=float)
        life_ev = np.array([float(ev.get("life_lost_years", 0.0) or 0.0) for ev in ev_list], dtype=float)
        ctype_ev = np.array([str(ev.get("cancer_type", "") or "") for ev in ev_list], dtype=object)
        sex_ev = np.array([str(ev.get("sex", "U") or "U") for ev in ev_list], dtype=object)
        # NB: plain ``or -1`` would map a valid exam_index of 0 to -1.
        exidx_ev = np.array(
            [-1 if ev.get("exam_index") is None else int(ev.get("exam_index"))
             for ev in ev_list],
            dtype=np.int64,
        )
    # Map orientation-specific radiography regions to combined reporting
    # regions so the sweep groups Chest PA/AP/Lat under a single "Chest".
    # Dental-imaging subtypes (Panoramic/Kefalometry/Intraoral/Dental) keep
    # their own name -- they are NOT collapsed into one flat "Dental
    # imaging" region label, so per-subtype cells stay distinguishable in
    # the sweep table / heatmap / burden charts. They are still grouped
    # into the "Dental imaging" pseudo-modality below (mod_ev override) for
    # display/filtering purposes.
    reg_ev = np.array([_rpt_reg(r) for r in reg_ev], dtype=object)
    # Assign the grouping modality for dental imaging events: exams with a
    # dental-imaging subtype region have actual modality "Radiography" but
    # are keyed under the "Dental imaging" pseudo-modality; "Dental
    # CBCT" events are keyed under "CBCT" whatever modality they carry
    # (see _grouping_modality_array).
    mod_ev = _grouping_modality_array(mod_ev, reg_ev, dental_cbct_raw)
    return (mod_ev, reg_ev, age_ev, di_ev, it_ev, died_ev, QALY_ev, life_ev,
            ctype_ev, sex_ev, exidx_ev)


def _leuk_event_quad_share(
    exam_data: Any,
    ctype_ev: np.ndarray,
    sex_ev: np.ndarray,
    age_ev: np.ndarray,
    exidx_ev: np.ndarray,
) -> np.ndarray:
    """Per-event leukaemia quadratic dose-response share (0 for other sites).

    Only leukaemia events need a red-marrow dose lookup, and those are rare,
    so the per-exam lookup loop runs over leukaemia events only rather than
    over the whole cohort.
    """
    n = int(ctype_ev.size)
    shares = np.zeros(n, dtype=float)
    if n == 0:
        return shares
    leuk_idx = np.flatnonzero(ctype_ev == "Leukaemia")
    if leuk_idx.size == 0:
        return shares
    try:
        n_ex = len(exam_data)
    except Exception:
        n_ex = 0
    doses = np.zeros(leuk_idx.size, dtype=float)
    for j, ev_i in enumerate(leuk_idx):
        i = int(exidx_ev[ev_i])
        if i < 0 or i >= n_ex:
            continue
        try:
            ex = exam_data[i]
            cd = ex.get("_cancer_doses") or None
            d = cd.get("Leukaemia") if cd else None
            if d is None:
                d = _get_organ_dose_for_cancer(
                    "Leukaemia", ex.get("organ_doses_gy") or {})
        except Exception:
            d = 0.0
        doses[j] = float(d or 0.0)
    shares[leuk_idx] = _leukaemia_quadratic_share(
        age_ev[leuk_idx], sex_ev[leuk_idx], doses)
    return shares


def _compute_deterministic_baselines(
    exam_data: list,
    sim_config: dict,
    min_latency_params: Optional[dict] = None,
) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """Compute analytical expected cases/deaths/QALY per (region, age_decade).
    """
    ddref_solid = float(sim_config.get("ddref_solid", 2.0))
    ddref_leuk = float(sim_config.get("ddref_leuk", 1.0))
    risk_model = str(sim_config.get("risk_model", "REIC")).upper()
    max_age = int(sim_config.get("max_attained_age", 94) or 94)
    QALY_discount = float(sim_config.get("QALY_discount_rate", 0.0))
    QALY_mode = str(sim_config.get("QALY_mode", "full")).lower()

    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    _edges_list = [float(x) for x in AGE_DECADE_EDGES]
    _n_dec = len(AGE_DECADE_LABELS)

    def _decade_of(age: float) -> int:
        # bisect_right(edges, age) - 1 matches searchsorted(side="right") - 1;
        # plain ints avoid the per-exam numpy overhead (called ~20k times here).
        idx = bisect.bisect_right(_edges_list, float(age)) - 1
        return min(max(idx, 0), _n_dec - 1)

    _ex_cols = getattr(exam_data, "_cols", None)
    if _ex_cols is not None:
        _n_ex = len(exam_data)
        _mod_col = _ex_cols.get("modality", [None] * _n_ex)
        _reg_col = _ex_cols.get("reporting_region", _ex_cols.get("region", [None] * _n_ex))
        _age_col = _ex_cols.get("age", [None] * _n_ex)
        _sex_col = _ex_cols.get("sex_code", [None] * _n_ex)
        _organ_col = _ex_cols.get("organ_doses_gy", [None] * _n_ex)
        _cdose_col = _ex_cols.get("_cancer_doses", [None] * _n_ex)
        _ctypes_col = _ex_cols.get("_modeled_types", [None] * _n_ex)
        _ed_col = _ex_cols.get("ed_msv", [None] * _n_ex)
        _prior_col = _ex_cols.get("_prior_cumulative_dose_sv", [0.0] * _n_ex)
        _frail_col = _ex_cols.get("_mataac_frailty_factor", [1.0] * _n_ex)
        _pop_col = _ex_cols["_population"]
        exams = zip(_mod_col, _reg_col, _age_col, _sex_col, _organ_col, _cdose_col, _ctypes_col, _ed_col,
                    _prior_col, _frail_col, _pop_col)
    else:
        exams = (
            (
                ex.get("modality", ""),
                ex.get("reporting_region", ex.get("region")),
                ex.get("age", ex.get("age_at_exposure", 0.0)),
                ex.get("sex_code", "U"),
                ex.get("organ_doses_gy", {}),
                ex.get("_cancer_doses"),
                ex.get("_modeled_types"),
                ex.get("ed_msv", 0.0),
                float(ex.get("_prior_cumulative_dose_sv", 0.0) or 0.0),
                float(ex.get("_mataac_frailty_factor", 1.0) or 1.0),
                ex["_population"],
            )
            for ex in (exam_data or [])
        )

    _MAX_AGE = 94  # MAX_ATTAINED_AGE_ICRP152 — matches MC vectorized path's over_age_limit

    for _mod, _reg, _age, _sex, _organ, _cdose, _ctypes, _ed, _prior, _frail, _pop in exams:
        modality = str(_mod or "")
        region = str(_reg or "Unknown")
        # Same modality grouping as ``_region_age_baseline``: dental-
        # imaging subtype exams (Panoramic/Kefalometry/Intraoral/Dental)
        # have actual modality "Radiography" but must be keyed under the
        # "Dental imaging" pseudo-modality, and "Dental CBCT" under
        # "CBCT", so these cells match up with the MC-path baseline cells
        # for the deterministic-vs-MC comparison.
        if region in _DENTAL_IMAGING_SUBTYPE_REGIONS:
            modality = "Dental imaging"
        elif region == _DENTAL_CBCT_REGION:
            modality = _DENTAL_CBCT_MODALITY
        age = float(_age or 0.0)
        sex = str(_sex or "U").strip().upper()[:1]
        if sex not in ("M", "F"):
            sex = "U"
        organ_doses = _organ or {}
        if not organ_doses and not _cdose:
            continue
        decade = _decade_of(age)
        cell = (modality, region, decade)
        if cell not in out:
            out[cell] = {
                "scans": 0.0, "dose_sv": 0.0,
                "cases": 0.0, "deaths": 0.0, "non_mortal": 0.0,
                "QALY_lost": 0.0, "life_lost": 0.0,
                "var_cases": 0.0, "var_deaths": 0.0, "var_non_mortal": 0.0,
                "var_QALY": 0.0, "var_life": 0.0,
                # Portion of the leukaemia contribution carried by the ICRP 152
                # para 41 quadratic dose term.  Needed because a dose reduction
                # removes (1-f**2) of it but only (1-f) of everything else.
                "leukq_cases": 0.0, "leukq_deaths": 0.0, "leukq_QALY": 0.0,
            }

        # Count scans/dose for all exams (matches MC _region_age_baseline).
        ed_msv = float(_ed or 0.0)
        out[cell]["scans"] += 1.0
        out[cell]["dose_sv"] += max(0.0, ed_msv) / 1000.0

        # surv_correction: match MC path exactly.
        # frailty × (age <= 94); MC path zeroes risk for Non-Dose exams and ages > 94.
        if region.startswith("Non-Dose") or age > _MAX_AGE:
            continue
        _surv_corr = float(_frail or 1.0)

        # The exam's own Population row value (resolved at load time), not
        # the run-level label — in a mixed cohort the latter would apply the
        # wrong population's incidence, survival and onset kernels.
        exam_population = _require_population(_pop)

        try:
            lifetime_risk, _rem_weights = _compute_lifetime_risk(
                organ_doses_gy=organ_doses,
                age_at_exposure=age,
                sex_code=sex,
                max_attained_age=max_age,
                ddref_solid=ddref_solid,
                ddref_leuk=ddref_leuk,
                population=exam_population,
                risk_model=risk_model,
                precomputed_cancer_doses=_cdose,
                precomputed_modeled_types=_ctypes,
            )
        except Exception:
            continue

        if not lifetime_risk:
            continue

        # Precompute lifespan PMF once per exam and reuse across sites.
        #
        # ICRP 152 para 51 specifies amax = 94 y ONLY for the REIC/LAR risk
        # integration (cumulating excess cancer incidence × cancer-free survival).
        # It does NOT specify a cap for lifespan / YLL computation.  The 120
        # used here is the length of the all-cause mortality rate tables
        # (ICRP 152 Table A.5, ages 0–120), not an ICRP integration parameter.
        # Using 120 for the lifespan PMF allows YLL to be non-zero when cancer
        # onset > 94 (e.g. Bone/Skin at exposure age 90: onset = 95, patient
        # may survive to 96+).  This matches the MC path's
        # sample_lifespans_vectorized(max_attained_age=120) default.
        lifespan_ages, lifespan_probs = _expected_lifespan_pmf(
            sex, age, population=exam_population, max_attained_age=120)

        for site, risk in lifetime_risk.items():
            if risk <= 0.0:
                continue
            k, q_min, l_rel, _det = _get_cancer_params_cached(site)
            q_severity = k + q_min * (1.0 - k)
            p = float(risk) * _surv_corr
            out[cell]["cases"] += p
            out[cell]["deaths"] += p * k
            out[cell]["non_mortal"] += p * (1.0 - k)

            # Analytical variance (Poisson-binomial)
            out[cell]["var_cases"]      += p * (1.0 - p)
            out[cell]["var_deaths"]     += p * k * (1.0 - p * k)
            out[cell]["var_non_mortal"] += p * (1.0 - k) * (1.0 - p * (1.0 - k))

            if QALY_mode == "icrp152":
                # ICRP 152 §3.2.3 site-average YLL — matches MC path's
                # QALY_event_bulk = q_severity * l_rel * 15.0. This mode
                # intentionally reproduces ICRP's flat aggregate formula, so
                # the single-tier q_severity is used as-is (not nested).
                e_yll = l_rel * 15.0
                e_yll2 = e_yll * e_yll
                q_eff = q_severity
                q_eff2 = q_severity * q_severity
            else:
                try:
                    e_yll, e_yll2 = _expected_yll_moments_for_site(
                        site, age, sex, population=exam_population,
                        min_latency_params=min_latency_params,
                        QALY_discount_rate=QALY_discount,
                        lifespan_ages=lifespan_ages, lifespan_probs=lifespan_probs,
                    )
                except Exception:
                    yll = l_rel * 15.0
                    e_yll = yll
                    e_yll2 = yll * yll
                # Individualized weight: 1 if the cancer is sampled as fatal,
                # q_severity if non-fatal (see vectorized.py). Its expectation
                # over the Bernoulli(k) fatal/non-fatal outcome is the NESTED
                # ICRP 152 detriment severity factor k + q_severity*(1-k),
                # not the single-tier q_severity — confirmed empirically
                # against ICRP 152 Table 4.4 sensitivity ratios.
                q_eff = k + q_severity * (1.0 - k)
                q_eff2 = k + (1.0 - k) * q_severity * q_severity

            out[cell]["QALY_lost"] += p * q_eff * e_yll
            out[cell]["life_lost"] += p * k * e_yll

            if site == "Leukaemia":
                # Scalar form of risk_models.leukaemia_quadratic_share (same
                # constant, same beta lookup) to avoid per-exam array allocation.
                _leuk_dose = None
                if _cdose:
                    _leuk_dose = _cdose.get("Leukaemia")
                if _leuk_dose is None:
                    _leuk_dose = _get_organ_dose_for_cancer("Leukaemia", organ_doses)
                _qd = _LEUK_QL_RATIO * max(0.0, float(_leuk_dose or 0.0))
                _qs = _qd / (_get_leukaemia_EAR_params(age, sex)[0] + _qd) if _qd > 0.0 else 0.0
                if _qs > 0.0:
                    out[cell]["leukq_cases"] += p * _qs
                    out[cell]["leukq_deaths"] += p * k * _qs
                    out[cell]["leukq_QALY"] += p * q_eff * e_yll * _qs

            # Var(QALY) = Σ p·E[Q²]·E[YLL²] - p²·E[Q]²·E[YLL]²
            out[cell]["var_QALY"] += (
                p * q_eff2 * e_yll2 - p * p * q_eff * q_eff * e_yll * e_yll
            )
            # Var(life) = Σ p·k·(E[YLL²] - p·k·E[YLL]²)
            out[cell]["var_life"] += p * k * (e_yll2 - p * k * e_yll * e_yll)

    return out


def _concat_event_columns(ecs: List[Any]) -> EventColumns:
    """Concatenate multiple EventColumns / columnar dicts into one."""
    ecs = [_ensure_event_columns(ec) for ec in ecs]
    ecs = [ec for ec in ecs if ec is not None and len(ec) > 0]
    if not ecs:
        return EventColumns.empty()
    if len(ecs) == 1:
        return ecs[0]
    def _cat(arrs):
        return np.concatenate(arrs) if arrs else np.array([], dtype=object)
    return EventColumns(
        cancer_type=_cat([ec.cancer_type for ec in ecs]),
        modality=_cat([ec.modality for ec in ecs]),
        region=_cat([ec.region for ec in ecs]),
        protocol=_cat([ec.protocol for ec in ecs]),
        age_at_exposure=_cat([ec.age_at_exposure for ec in ecs]),
        sex=_cat([ec.sex for ec in ecs]),
        died=_cat([ec.died for ec in ecs]),
        exam_index=_cat([ec.exam_index for ec in ecs]),
        iteration=_cat([ec.iteration for ec in ecs]),
        ed_msv=_cat([ec.ed_msv for ec in ecs]),
        QALY_lost=_cat([ec.QALY_lost for ec in ecs]),
        life_lost_years=_cat([ec.life_lost_years for ec in ecs]),
        age_at_death=_cat([ec.age_at_death for ec in ecs]),
        projected_lifetime=_cat([ec.projected_lifetime for ec in ecs]),
    )


def _filter_events_by_modality(events: Any, mod_key: Any) -> Any:
    """Filter events by canonical modality key(s).

    ``mod_key`` may be a single canonical string, a set/list of canonical
    strings, or None/empty (no filtering).

    Returns a new ``EventColumns`` when the input is columnar, otherwise a
    list of dicts.  Returns an empty list if the filter matches nothing.
    """
    if not mod_key or not events:
        return events
    # Normalise to a set of canonical keys.
    if isinstance(mod_key, str):
        mod_keys = {mod_key}
    else:
        mod_keys = set(mod_key)
    if not mod_keys:
        return events

    def _event_matches(mod_val: str, reg_val: str) -> bool:
        # Dental imaging events match only "DENTAL IMAGING" and "Dental
        # CBCT" events only "CBCT", not "RADIOGRAPHY" (even though their
        # actual modality is Radiography) -- the sweep-cell grouping.
        return _modality_matches(mod_val, reg_val, mod_keys)

    if _is_event_columns(events):
        mod_arr = np.asarray(events.modality, dtype=object)
        reg_arr = np.asarray(events.region, dtype=object)
        mask = np.array([
            _event_matches(m, r) for m, r in zip(mod_arr, reg_arr)
        ], dtype=bool)
        if not mask.any():
            return []
        return EventColumns(
            cancer_type=events.cancer_type[mask],
            modality=events.modality[mask],
            region=events.region[mask],
            protocol=events.protocol[mask],
            age_at_exposure=events.age_at_exposure[mask],
            sex=events.sex[mask],
            died=events.died[mask],
            exam_index=events.exam_index[mask],
            iteration=events.iteration[mask],
            ed_msv=events.ed_msv[mask],
            QALY_lost=events.QALY_lost[mask],
            life_lost_years=events.life_lost_years[mask],
            age_at_death=events.age_at_death[mask],
            projected_lifetime=events.projected_lifetime[mask],
        )
    return [ev for ev in events
            if _event_matches(ev.get("modality"), ev.get("region"))]


# Dose-reduction benefit per imaging region × 10-year age group
#       (per 100 000 imaging procedures)
#
# Implementation principles:
#   * Linear-no-threshold (LNT) scaling of the simulation's per-imaging
#     risk in the small dose-change window (-50%..+50%).
#   * Baselines come from a single walk over ``exam_data`` (deterministic
#     scan/dose counts) and ``cancer_events`` (stochastic outcome counts
#     already aggregated over all simulated MC years).
#   * Outcome counts are annualised by dividing by ``stats['iterations']``
#     so the table reports "cases per 100 000 examinations per simulated
#     year" — directly comparable across regions of different volume.

AGE_DECADE_EDGES: Tuple[float, ...] = (
    0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 200.0,
)
AGE_DECADE_LABELS: Tuple[str, ...] = (
    "0-9", "10-19", "20-29", "30-39", "40-49",
    "50-59", "60-69", "70-79", "80-89", "90+",
)

# Upper bound (inclusive) of the per-cell 1-year exam-age histogram bins.
# Ages are clipped into [0, AGE_HIST_MAX_AGE], so there are
# AGE_HIST_MAX_AGE + 1 bins (0..120 -> 121 bins).
AGE_HIST_MAX_AGE: int = 120
# Per-cell exam counts by sex (parallel to age_hist): index into the
# last axis of the (n_cells, 3) ``sex_counts`` array.
SEX_COUNT_INDEX = {"F": 0, "M": 1, "U": 2}


def _sex_norm(v: Any) -> str:
    """Normalise an exam sex value to 'F'/'M'/'U' (columns.py convention)."""
    s = str(v or "U").strip().upper()[:1]
    return s if s in ("F", "M") else "U"

DEFAULT_DOSE_REDUCTIONS_PCT: Tuple[float, ...] = (
    0.0, -10.0, -20.0, -30.0, -40.0, -50.0,
)

# ---------------------------------------------------------------------------
# Small-cell reliability thresholds (NCHS-style "unreliable rate" defaults).
#
# A per-(region, age) cell is flagged unreliable when EITHER the TOTAL
# exam count of the cell across all pooled year groups is below
# ``DEFAULT_MIN_SCANS`` OR its reported QALY
# mean fails the precision criterion (zero-event cells are always treated
# as reliable because their estimate is a stable 0).  A cell with at least
# one simulated cancer is *precise* iff BOTH:
#
#   * ``event_iters >= DEFAULT_MIN_EVENT_ITERS`` -- it saw at least one
#     cancer in at least this many distinct Monte-Carlo iterations (an
#     NCHS-style minimum-count bar, the conventional cut-off being ~20
#     events / relative SE <~22 %); and
#   * ``rel_se_QALY <= DEFAULT_MAX_REL_SE`` -- the relative standard error
#     of the per-iteration QALY mean, ``(std / sqrt(n)) / mean`` computed
#     over all ``n`` iterations of the cell (Welford, ddof=1), is at most
#     this fraction.
#
# ``DEFAULT_MIN_EVENTS`` is an absolute-count floor inside the
# ``events_ok`` check (also ~20, matching the NCHS bar); the event-
# iteration and relative-SE terms above are the operative precision test.
# All thresholds can be overridden by callers of
# ``compute_region_age_dose_sweep``.
# ---------------------------------------------------------------------------
DEFAULT_MIN_SCANS: float = 30.0
DEFAULT_MIN_EVENTS: float = 20.0
DEFAULT_MIN_EVENT_ITERS: int = 20
DEFAULT_MAX_REL_SE: float = 0.10
# Canonical value lives in constants.Z_95; aliased to CI_Z here since that
# name is used throughout this module's stopping-rule / CI code below.
CI_Z: float = Z_95


def _project_ci(
    lo_per_year: float,
    hi_per_year: float,
    scale: float,
    mult: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a per-cohort [lo, hi] interval to per-N_exams across LNT levels.

    Expressing the same random quantity per ``per_n_scans`` examinations is a
    deterministic linear rescale by ``scale * mult``, so ``Var(cX) = c**2
    Var(X)`` and the standard error — hence the whole interval — scales by
    exactly the same factor as the point estimate.  Both endpoints are
    therefore transformed directly.

    Scaling the endpoints (rather than decomposing into midpoint and
    half-width) also makes this transform centre-preserving by
    construction: an input interval that is not symmetric about its mean
    cannot displace the output's centre.

    ``mult`` is an array of LNT multipliers (positive for dose reductions,
    so the result is "avoided"; negative for dose increases, so the result
    is "added harm").  When ``mult`` is negative the interval flips, so we
    return the elementwise min / max — the interval is always reported
    lo <= hi regardless of sign.
    """
    factor = scale * mult
    a = lo_per_year * factor
    b = hi_per_year * factor
    return np.minimum(a, b), np.maximum(a, b)


def _lq_share(baseline: float, leuk_quad: float) -> float:
    """Leukaemia-quadratic share ``w`` of a metric, clamped to [0, 1].

    Factored out so the sweep's ``_mult_eff`` and any consumer rebuilding a
    multiplier for an off-grid dose change (``compute_annual_benefit``)
    derive ``w`` from exactly the same expression.
    """
    if baseline <= 0.0 or leuk_quad <= 0.0:
        return 0.0
    return min(max(leuk_quad / baseline, 0.0), 1.0)


def _lnt_mult(pct: float, w: float) -> float:
    """LNT multiplier for a dose change of ``pct`` % with quadratic share ``w``.

    The scalar counterpart of the sweep's ``_mult_eff``, kept beside it so
    the two cannot drift.  ``m = -pct/100`` is positive for a reduction (the
    result is "avoided") and negative for an increase ("added harm"), and
    the ICRP 152 para 41 leukaemia quadratic term contributes
    ``m*(2-m)`` where the linear sites contribute ``m``::

        m_eff = m*(1-w) + m*(2-m)*w = m*(1 + w*(1-m))
    """
    m = -float(pct) / 100.0
    return m * (1.0 + float(w) * (1.0 - m))


def _region_age_baseline(
    exam_data: list,
    cancer_events: Optional[Any],
    iterations: int,
    *,
    extra_cancer_events: Optional[Any] = None,
    extra_iters_by_cell: Optional[Dict[Tuple[str, str, int], int]] = None,
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    """Per-(modality, region, decade) baseline counts, dose AND per-iteration arrays.

    Each cell maintains a per-iteration array of size
    ``iters_main + iters_extra`` where ``iters_extra`` may differ between
    cells (zero for cells that received no extra MC iterations).
    The mean / SE / CI computed downstream are therefore correct for
    each cell individually — cells with extra iterations use a larger
    denominator proportional to the extra MC effort spent on them, while
    other cells continue to use ``iterations``.

    Returns
    -------
    dict keyed by ``(modality, region, decade_idx)`` with annualised values:
        * ``scans``       -- average exams per simulated year
        * ``dose_sv``     -- collective effective dose per simulated year [Sv]
        * ``modelled_scans`` -- count of exams with a risk model
          (_cancer_doses truthy) per simulated year
        * ``cases``       -- average induced cancers per simulated year
        * ``deaths``      -- average induced cancer deaths per simulated year
        * ``QALY_lost``   -- average QALYs lost per simulated year
        * ``life_lost``   -- average life-years lost per simulated year
        * ``non_mortal``  -- average non-fatal induced cancers (cases - deaths)
        * ``iters_main``  -- main MC iterations applicable to this cell
        * ``iters_extra`` -- extra MC iterations beyond the main run (>=0)
        * ``iters_total`` -- iters_main + iters_extra

    plus, for each stochastic metric, the totals and a precomputed Wald
    95% confidence interval (derived from per-iteration counts via
    Welford's online algorithm for numerically-stable variance) so
    downstream callers do not need to keep per-iteration arrays alive:

        * ``total_events`` -- total induced cancers summed over all iters
        * ``total_deaths`` -- total induced deaths summed over all iters
        * ``cases_ci`` / ``deaths_ci`` / ``non_mortal_ci`` / ``QALY_ci``
          -- (lo, hi) tuples of the per-year mean's 95% CI
        * ``age_hist`` -- exams per 1-year age bin (0..AGE_HIST_MAX_AGE),
          same per-simulated-year basis as ``scans``

    ``scans`` and ``dose_sv`` come from ``exam_data`` (one deterministic
    pass, the cohort is identical in every MC iteration) so they are
    already "per simulated year".  Cancer outcomes accumulate over all
    ``iterations`` MC years inside ``cancer_events``; each event's
    ``"iteration"`` field (set by ``vectorized_iteration``) attributes it
    to the correct per-iteration slot for Welford online accumulation.
    Out-of-range or missing iteration indices are clamped to
    ``[0, iters-1]`` defensively (e.g. cached / replayed runs).

    Extra events from targeted MC iterations carry their own
    iteration indices in ``[0, iters_extra_for_cell)`` and are written
    to slots ``[iters_main, iters_main + iters_extra)``.  Cells with no
    extra iterations keep ``iters_extra = 0`` and behave identically
    to the base output.
    """
    iters_main = max(1, int(iterations or 0))
    extra_by_cell = {
        (str(mod), str(reg), int(di)): max(0, int(n))
        for (mod, reg, di), n in (extra_iters_by_cell or {}).items()
    }
    out: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    # Vectorised binning
    # Built with ``np.bincount`` / ``np.add.at`` over a single flat per-cell
    # buffer.  Both ops apply their additions in input (row) order, so the
    # per-cell per-iteration float sums have a deterministic summation order
    # — important for reproducibility with saved caches/snapshots.
    _edges = np.asarray(AGE_DECADE_EDGES, dtype=float)
    _n_dec = len(AGE_DECADE_LABELS)

    def _decade_vec(ages: np.ndarray) -> np.ndarray:
        if ages.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.clip(
            np.searchsorted(_edges, ages, side="right") - 1, 0, _n_dec - 1
        ).astype(np.int64)

    # Per-source column extraction (one pass each; avoids per-exam
    # bucket-lookup-and-scalar-add overhead).
    #
    # ``exam_data`` is normalised to the columnar store first, then the
    # underlying column lists are read directly instead of iterating it —
    # each ``for ex in ex_list`` pass would otherwise materialise N
    # ``ExamProxy`` objects (3N allocations) and route every field access
    # through ``ExamProxy.get``.  Iterating the raw lists is deterministic
    # (same values, same order, same None/falsy handling) and removes that
    # per-exam object churn for million-row cohorts.
    exam_data = ColumnarExamsList.from_dicts(exam_data or [])
    _ex_cols = exam_data._cols
    _n_ex = len(exam_data)
    _mod_col = _ex_cols.get("modality", [None] * _n_ex)
    _reg_col = _ex_cols.get("reporting_region", _ex_cols.get("region"))
    _age_col = _ex_cols.get("age")
    _dose_col = _ex_cols.get("ed_msv")
    if _reg_col is None:
        _reg_col = [None] * _n_ex
    if _age_col is None:
        _age_col = [None] * _n_ex
    if _dose_col is None:
        _dose_col = [None] * _n_ex
    mod_ex = np.array([str(m or "") for m in _mod_col], dtype=object)
    reg_ex = np.array([str(r or "Unknown") for r in _reg_col], dtype=object)
    age_ex = np.array([float(a or 0.0) for a in _age_col], dtype=float)
    dose_ex = np.array(
        [max(0.0, float(d or 0.0)) / 1000.0 for d in _dose_col],
        dtype=float,
    )
    # Per-exam "has a risk model" flag (1.0 if _cancer_doses is truthy,
    # else 0.0) -- matches the vectorized is_modelled definition (a non-
    # empty _cancer_doses dict AND region not Non-Dose; the Non-Dose part
    # is handled separately downstream). organ_doses_gy is deliberately
    # NOT used: an exam can carry organ doses but an empty _cancer_doses
    # (e.g. male mammography, Breast excluded), which is NOT modelled.
    _cd_col = _ex_cols.get("_cancer_doses")
    if _cd_col is None:
        modelled_ex = np.zeros(_n_ex, dtype=float)
    else:
        modelled_ex = np.array(
            [1.0 if bool(cd) else 0.0 for cd in _cd_col],
            dtype=float,
        )
    _sex_ex_col = _ex_cols.get("sex_code")
    if _sex_ex_col is None:
        _sex_ex_col = _ex_cols.get("sex")
    if _sex_ex_col is None:
        _sex_ex_col = ["U"] * _n_ex
    sex_ex = [_sex_norm(v) for v in _sex_ex_col]
    sex_idx_ex = np.array(
        [SEX_COUNT_INDEX.get(s, SEX_COUNT_INDEX["U"]) for s in sex_ex],
        dtype=np.int64,
    )
    # Assign the grouping modality for dental imaging exams: exams with a
    # subtype region (Panoramic/Kefalometry/Intraoral/Dental -- each kept
    # as its own region label, see exam_doses.py) have actual modality
    # "Radiography" but should be treated as "Dental imaging" pseudo-
    # modality for cell keying; "Dental CBCT" exams are keyed under "CBCT"
    # (same mapping as _event_arrays).
    mod_ex = _grouping_modality_array(mod_ex, reg_ex)
    di_ex = _decade_vec(age_ex)


    (mod_ev, reg_ev, age_ev, di_ev, it_ev, died_ev, QALY_ev, life_ev,
     ctype_ev, sex_ev, exidx_ev) = _event_arrays(
        cancer_events, _decade_vec,
    )
    if it_ev.size:
        it_ev = np.clip(it_ev, 0, iters_main - 1)

    (mod_xt, reg_xt, age_xt, di_xt, it_xt, died_xt, QALY_xt, life_xt,
     ctype_xt, sex_xt, exidx_xt) = _event_arrays(
        extra_cancer_events, _decade_vec,
    )

    # Leukaemia quadratic dose-response share per event (ICRP 152 para 41).
    qs_ev = _leuk_event_quad_share(exam_data, ctype_ev, sex_ev, age_ev, exidx_ev)
    qs_xt = _leuk_event_quad_share(exam_data, ctype_xt, sex_xt, age_xt, exidx_xt)

    n_ex, n_ev, n_xt = reg_ex.size, reg_ev.size, reg_xt.size
    if (n_ex + n_ev + n_xt) == 0:
        return out

    # Union of all (modality, region, decade) cells, encoded as a single integer code.
    all_mod = np.concatenate([mod_ex, mod_ev, mod_xt]) if (n_ex or n_ev or n_xt) else np.array([], dtype=object)
    all_reg = np.concatenate([reg_ex, reg_ev, reg_xt]) if (n_ex or n_ev or n_xt) else np.array([], dtype=object)
    all_di = np.concatenate([di_ex, di_ev, di_xt]).astype(np.int64)
    umod, mod_codes = np.unique(all_mod, return_inverse=True)
    ureg, reg_codes = np.unique(all_reg, return_inverse=True)
    _n_ureg = len(ureg)
    combined_codes = mod_codes.astype(np.int64) * _n_ureg + reg_codes.astype(np.int64)
    cell_code_all = combined_codes * _n_dec + all_di
    ucells, cell_ids_all = np.unique(cell_code_all, return_inverse=True)
    n_cells = int(ucells.size)

    cid_ex = cell_ids_all[:n_ex]
    cid_ev = cell_ids_all[n_ex:n_ex + n_ev]
    cid_xt = cell_ids_all[n_ex + n_ev:]

    # Decode each unique cell back to (modality_str, region_str, decade_idx) + its extra-iter
    # budget.  iters_total_arr drives the Welford count per cell.
    cell_combined = (ucells // _n_dec).astype(np.int64)
    cell_mod_code = (cell_combined // _n_ureg).astype(np.int64)
    cell_reg_code = (cell_combined % _n_ureg).astype(np.int64)
    cell_di = (ucells % _n_dec).astype(np.int64)
    cell_mod = [str(umod[c]) for c in cell_mod_code]
    cell_reg = [str(ureg[c]) for c in cell_reg_code]
    iters_extra_arr = np.array(
        [int(extra_by_cell.get((cell_mod[c], cell_reg[c], int(cell_di[c])), 0)) for c in range(n_cells)],
        dtype=np.int64,
    )
    iters_total_arr = iters_main + iters_extra_arr

    # Scans + collective dose per cell (np.bincount sums in row order).
    scans_arr = (np.bincount(cid_ex, minlength=n_cells).astype(float)
                 if n_ex else np.zeros(n_cells, dtype=float))
    dose_arr = (np.bincount(cid_ex, weights=dose_ex, minlength=n_cells)
                if n_ex else np.zeros(n_cells, dtype=float))
    # Count of modelled exams (_cancer_doses truthy) per cell -- a
    # (modality, region) with zero modelled exams across all age decades has
    # no risk model and is shown as n/a in the heatmap.
    modelled_arr = (np.bincount(cid_ex, weights=modelled_ex, minlength=n_cells)
                    if n_ex else np.zeros(n_cells, dtype=float))
    # Collective dose of the modelled exams only -- matches the Summary
    # tab's ``dose_sv_model`` (dose_sv restricted to is_modelled exams).
    modelled_dose_arr = (
        np.bincount(cid_ex, weights=dose_ex * modelled_ex, minlength=n_cells)
        if n_ex else np.zeros(n_cells, dtype=float))

    # Per-cell 1-year exam-age histogram (bins 0..AGE_HIST_MAX_AGE), on the
    # same per-simulated-year basis as ``scans``.
    _n_ab = AGE_HIST_MAX_AGE + 1
    if n_ex:
        _age_bin_ex = np.clip(np.floor(age_ex), 0, AGE_HIST_MAX_AGE).astype(np.int64)
        age_hist_arr = np.bincount(
            cid_ex * _n_ab + _age_bin_ex, minlength=n_cells * _n_ab
        ).astype(float).reshape(n_cells, _n_ab)
        # Per-cell exam counts by sex (F/M/U, same per-year basis as scans).
        sex_counts_arr = np.bincount(
            cid_ex * 3 + sex_idx_ex, minlength=n_cells * 3
        ).astype(float).reshape(n_cells, 3)
    else:
        age_hist_arr = np.zeros((n_cells, _n_ab), dtype=float)
        sex_counts_arr = np.zeros((n_cells, 3), dtype=float)

    # Welford online accumulation
    # Instead of materialising flat per-iteration arrays of length
    # Σ(iters_total) — O(n_cells × iterations) memory — we use Welford's
    # online algorithm to accumulate per-cell running mean and M2 (sum of
    # squared deviations) in O(n_cells) memory.
    #
    # Events are grouped by iteration.  For each iteration that has events,
    # per-cell sums are computed via np.bincount and the Welford
    # accumulators are updated vectorised across all cells.  Iterations
    # with no events for a cell contribute x=0; gaps between
    # event-iterations are handled with a closed-form batch zero-update
    # so we never iterate over individual zero-iterations.
    #
    # The resulting mean and sample variance (ddof=1) are numerically
    # identical to a two-pass centred approach.

    n_arr = iters_total_arr.astype(float)

    # Welford accumulators: (mean, M2) per metric, plus shared count.
    wf_n      = np.zeros(n_cells, dtype=float)
    # Count of iterations in which the cell produced >=1 cancer (the
    # "event_iters" of the precision criterion).  Updated only inside the
    # per-iteration event block, never by the gap / trailing zero-updates.
    wf_evit   = np.zeros(n_cells, dtype=float)
    wf_mean_c = np.zeros(n_cells, dtype=float)
    wf_m2_c   = np.zeros(n_cells, dtype=float)
    wf_mean_d = np.zeros(n_cells, dtype=float)
    wf_m2_d   = np.zeros(n_cells, dtype=float)
    wf_mean_q = np.zeros(n_cells, dtype=float)
    wf_m2_q   = np.zeros(n_cells, dtype=float)
    wf_mean_l = np.zeros(n_cells, dtype=float)
    wf_m2_l   = np.zeros(n_cells, dtype=float)
    wf_mean_nm = np.zeros(n_cells, dtype=float)
    wf_m2_nm   = np.zeros(n_cells, dtype=float)

    # Total counts (for total_events / total_deaths reporting).
    total_cases  = np.zeros(n_cells, dtype=float)
    total_deaths = np.zeros(n_cells, dtype=float)

    # Combine main + extra events with global iteration indices
    ev_cid = np.empty(n_ev + n_xt, dtype=np.int64)
    ev_it  = np.empty(n_ev + n_xt, dtype=np.int64)
    ev_d   = np.empty(n_ev + n_xt, dtype=float)
    ev_q   = np.empty(n_ev + n_xt, dtype=float)
    ev_l   = np.empty(n_ev + n_xt, dtype=float)
    ev_qs  = np.empty(n_ev + n_xt, dtype=float)

    if n_ev:
        ev_cid[:n_ev] = cid_ev
        ev_it[:n_ev]  = it_ev
        ev_d[:n_ev]   = died_ev
        ev_q[:n_ev]   = QALY_ev
        ev_l[:n_ev]   = life_ev
        ev_qs[:n_ev]  = qs_ev

    if n_xt:
        cell_extra_xt = iters_extra_arr[cid_xt]
        keep = cell_extra_xt > 0
        n_keep = int(keep.sum())
        ev_cid[n_ev:n_ev + n_keep] = cid_xt[keep]
        ev_it[n_ev:n_ev + n_keep]  = iters_main + np.clip(it_xt[keep], 0, cell_extra_xt[keep] - 1)
        ev_d[n_ev:n_ev + n_keep]   = died_xt[keep]
        ev_q[n_ev:n_ev + n_keep]   = QALY_xt[keep]
        ev_l[n_ev:n_ev + n_keep]   = life_xt[keep]
        ev_qs[n_ev:n_ev + n_keep]  = qs_xt[keep]
        n_ev_xt = n_keep
    else:
        n_ev_xt = 0

    ev_cid = ev_cid[:n_ev + n_ev_xt]
    ev_it  = ev_it[:n_ev + n_ev_xt]
    ev_d   = ev_d[:n_ev + n_ev_xt]
    ev_q   = ev_q[:n_ev + n_ev_xt]
    ev_l   = ev_l[:n_ev + n_ev_xt]
    ev_qs  = ev_qs[:n_ev + n_ev_xt]

    # Leukaemia quadratic-attributable portion of each metric, per cell.
    # Point estimates only (the share is a deterministic per-exam property),
    # so these bypass the Welford variance machinery: summing over all
    # iterations and dividing by the per-cell iteration count gives the mean.
    if ev_cid.size:
        _leukq_cases = np.bincount(ev_cid, weights=ev_qs, minlength=n_cells)
        _leukq_deaths = np.bincount(ev_cid, weights=ev_qs * ev_d, minlength=n_cells)
        _leukq_QALY = np.bincount(ev_cid, weights=ev_qs * ev_q, minlength=n_cells)
    else:
        _leukq_cases = np.zeros(n_cells, dtype=float)
        _leukq_deaths = np.zeros(n_cells, dtype=float)
        _leukq_QALY = np.zeros(n_cells, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        _inv_iters = np.where(n_arr > 0.0, 1.0 / n_arr, 0.0)
    _leukq_cases = _leukq_cases * _inv_iters
    _leukq_deaths = _leukq_deaths * _inv_iters
    _leukq_QALY = _leukq_QALY * _inv_iters

    # Track totals
    if ev_cid.size:
        np.add.at(total_cases,  ev_cid, 1.0)
        np.add.at(total_deaths, ev_cid, ev_d)

    # Group events by iteration and process via Welford
    # Metrics list: (value_array, mean_array, m2_array) for the update loop.
    wf_metrics = [
        (wf_mean_c,  wf_m2_c),    # cases
        (wf_mean_d,  wf_m2_d),    # deaths
        (wf_mean_q,  wf_m2_q),    # QALY
        (wf_mean_l,  wf_m2_l),    # life
        (wf_mean_nm, wf_m2_nm),   # non_mortal
    ]

    prev_iter = 0

    if ev_cid.size:
        # Sort by iteration to process in order
        sort_idx = np.argsort(ev_it, kind="stable")
        ev_it_s  = ev_it[sort_idx]
        ev_cid_s = ev_cid[sort_idx]
        ev_d_s   = ev_d[sort_idx]
        ev_q_s   = ev_q[sort_idx]
        ev_l_s   = ev_l[sort_idx]

        # Unique iterations that have events, in ascending order
        uniq_its, it_starts = np.unique(ev_it_s, return_index=True)
        it_ends = np.append(it_starts[1:], ev_cid_s.size)

        for ui in range(uniq_its.size):
            cur_it = int(uniq_its[ui])

            # Batch zero-update for gap [prev_iter, cur_it)
            k_gap = cur_it - prev_iter
            if k_gap > 0:
                n_old = wf_n.copy()
                n_new = n_old + k_gap
                ratio = np.where(n_new > 0, n_old / n_new, 0.0)
                inv_n_new = np.where(n_new > 0, 1.0 / n_new, 0.0)
                for mean_arr, m2_arr in wf_metrics:
                    m2_arr += mean_arr ** 2 * n_old * k_gap * inv_n_new
                    mean_arr *= ratio
                wf_n = n_new

            # Per-cell sums for this iteration
            sl = slice(int(it_starts[ui]), int(it_ends[ui]))
            c_ids = ev_cid_s[sl]
            c_sum = np.bincount(c_ids, minlength=n_cells).astype(float)
            d_sum = np.bincount(c_ids, weights=ev_d_s[sl], minlength=n_cells)
            q_sum = np.bincount(c_ids, weights=ev_q_s[sl], minlength=n_cells)
            l_sum = np.bincount(c_ids, weights=ev_l_s[sl], minlength=n_cells)
            nm_sum = c_sum - d_sum

            # This iteration is "event-bearing" for every cell whose
            # cancer count is non-zero here -- accumulate the precision
            # criterion's event_iters.
            wf_evit += (c_sum > 0)

            # Welford update (vectorised across all cells)
            wf_n += 1.0
            for val, (mean_arr, m2_arr) in zip(
                [c_sum, d_sum, q_sum, l_sum, nm_sum], wf_metrics
            ):
                delta = val - mean_arr
                mean_arr += delta / wf_n
                delta2 = val - mean_arr
                m2_arr += delta * delta2

            prev_iter = cur_it + 1

    # Final batch zero-update for remaining iterations
    max_processed = prev_iter if ev_cid.size else 0
    remaining = iters_total_arr.astype(float) - float(max_processed)
    remaining = np.maximum(remaining, 0.0)
    if remaining.any():
        n_old = wf_n
        k = remaining
        n_new = n_old + k
        ratio = np.where(n_new > 0, n_old / n_new, 0.0)
        inv_n_new = np.where(n_new > 0, 1.0 / n_new, 0.0)
        for mean_arr, m2_arr in wf_metrics:
            m2_arr += mean_arr ** 2 * n_old * k * inv_n_new
            mean_arr *= ratio
        wf_n = n_new

    # Derive final statistics from Welford state
    mean_cases  = wf_mean_c.copy()
    mean_deaths = wf_mean_d.copy()
    mean_QALY   = wf_mean_q.copy()
    mean_life   = wf_mean_l.copy()
    mean_nm     = wf_mean_nm.copy()

    sum_cases  = total_cases
    sum_deaths = total_deaths

    def _wald_ci(mean: np.ndarray, m2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wald 95% CI from Welford M2, plus a degenerate-cell mask.

        The interval is left strictly symmetric about ``mean`` — it is NOT
        clamped at zero here.  Rare cells can have a standard error larger
        than their mean, and clamping the lower bound at this stage would
        make ``(lo + hi) / 2`` differ from ``mean``, silently displacing the
        interval for every downstream consumer that rescales it.  Clamping
        to non-negative counts belongs at the reporting boundary instead.

        Returns ``(lo, hi, degenerate)`` where ``degenerate`` flags cells
        whose Wald interval has collapsed to a point: ``n <= 1`` (no
        replicate variance) or ``std == 0`` with a non-zero mean (every
        iteration produced the identical non-zero value).  A genuine
        zero cell (``mean == 0`` and ``std == 0``) is NOT degenerate — its
        ``[0, 0]`` interval is a stable, honest estimate of zero.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(n_arr > 1.0, m2 / np.maximum(n_arr - 1.0, 1.0), 0.0)
            std = np.sqrt(np.maximum(var, 0.0))
            half = CI_Z * std / np.sqrt(n_arr)
        degenerate = (n_arr <= 1.0) | ((std == 0.0) & (mean > 0.0))
        lo = np.where(degenerate, mean, mean - half)
        hi = np.where(degenerate, mean, mean + half)
        return lo, hi, degenerate

    cases_lo, cases_hi, _cases_deg   = _wald_ci(mean_cases, wf_m2_c)
    deaths_lo, deaths_hi, _deaths_deg = _wald_ci(mean_deaths, wf_m2_d)
    nm_lo, nm_hi, _nm_deg           = _wald_ci(mean_nm, wf_m2_nm)
    QALY_lo, QALY_hi, QALY_deg     = _wald_ci(mean_QALY, wf_m2_q)

    # Relative standard error of the per-iteration QALY mean over all n
    # iterations: ``rel_se = (std / sqrt(n)) / mean``.  Uses the same
    # sample std (ddof=1) that ``_wald_ci`` derives from the Welford M2 --
    # ``std = sqrt(m2 / (n-1))`` for n>1, else 0.  This drives the rare-cell
    # precision criterion, so it is computed over every iteration of the
    # cell (event-bearing and zero alike); a zero-mean cell is given 0 (a
    # stable estimate of zero, not an imprecise one).
    with np.errstate(divide="ignore", invalid="ignore"):
        _q_var = np.where(n_arr > 1.0, wf_m2_q / np.maximum(n_arr - 1.0, 1.0), 0.0)
        _q_std = np.sqrt(np.maximum(_q_var, 0.0))
        rel_se_q = np.where(
            mean_QALY > 0.0,
            (_q_std / np.sqrt(n_arr)) / mean_QALY,
            0.0,
        )

    for c in range(n_cells):
        out[(cell_mod[c], cell_reg[c], int(cell_di[c]))] = {
            "scans": float(scans_arr[c]),
            "dose_sv": float(dose_arr[c]),
            "modelled_scans": float(modelled_arr[c]),
            "modelled_dose_sv": float(modelled_dose_arr[c]),
            "cases": float(mean_cases[c]),
            "deaths": float(mean_deaths[c]),
            "QALY_lost": float(mean_QALY[c]),
            "life_lost": float(mean_life[c]),
            "non_mortal": float(mean_nm[c]),
            "iters_main": iters_main,
            "iters_extra": int(iters_extra_arr[c]),
            "iters_total": int(iters_total_arr[c]),
            "total_events": float(sum_cases[c]),
            "total_deaths": float(sum_deaths[c]),
            "cases_ci": (float(cases_lo[c]), float(cases_hi[c])),
            "deaths_ci": (float(deaths_lo[c]), float(deaths_hi[c])),
            "non_mortal_ci": (float(nm_lo[c]), float(nm_hi[c])),
            "QALY_ci": (float(QALY_lo[c]), float(QALY_hi[c])),
            "ci_degenerate": bool(QALY_deg[c]),
            # Precision-criterion inputs: distinct event-bearing iteration
            # count, relative SE of the QALY mean, and the Welford M2 (sum
            # of squared deviations) needed to re-derive the variance when
            # extra iterations are folded in incrementally.
            "event_iters": int(wf_evit[c]),
            "rel_se_QALY": float(rel_se_q[c]),
            "QALY_m2": float(wf_m2_q[c]),
            "leukq_cases": float(_leukq_cases[c]),
            "leukq_deaths": float(_leukq_deaths[c]),
            "leukq_QALY": float(_leukq_QALY[c]),
            "age_hist": age_hist_arr[c].copy(),
            "sex_counts": sex_counts_arr[c].copy(),
        }

    return out


def _norm_modality(s: Any) -> str:
    """Canonical-key form for modality strings (matches ``exam_doses``)."""
    return str(s or "").strip().upper()


# Cache for modality-filtered exam lists.  The key is (id(exam_data), mod_key).
_modality_filter_cache: Dict[Tuple[int, Any], list] = {}
# Cache of the full (all-modality) deterministic-baseline computation,
# keyed by (id(unfiltered exam_data), id(sim_config)).  This is the
# expensive analytical ICRP-152 per-exam integral, so it must NOT be
# recomputed every time the user only changes the modality filter —
# it is computed once over the whole cohort and then filtered per cell.
_deterministic_baseline_cache: Dict[Tuple[int, int], Dict[Tuple[str, str, int], Dict[str, float]]] = {}


def discard_deterministic_baseline_cache(exam_data, stats) -> None:
    """Drop the cached full-cohort deterministic baseline for ``exam_data``.

    The cache is keyed by ``id()``; once a year's exam_data is freed the next
    year's object can reuse the address and hit a stale entry, so batch
    callers discard the entry while the objects are still alive.
    """
    sim_cfg = (stats or {}).get("sim_config") or {}
    _deterministic_baseline_cache.pop((id(exam_data), id(sim_cfg)), None)


def compute_region_age_dose_sweep(
    stats: dict,
    exam_data: list,
    cancer_events: Optional[Any] = None,
    dose_reductions_pct: Tuple[float, ...] = DEFAULT_DOSE_REDUCTIONS_PCT,
    per_n_scans: int = 100_000,
    min_scans: float = DEFAULT_MIN_SCANS,
    min_expected_events: float = DEFAULT_MIN_EVENTS,
    *,
    min_event_iters: int = DEFAULT_MIN_EVENT_ITERS,
    max_rel_se: float = DEFAULT_MAX_REL_SE,
    modality: Union[str, Iterable[str], None] = None,
    extra_cancer_events: Optional[Any] = None,
    extra_iters_by_cell: Optional[Dict[Tuple[str, int], int]] = None,
    deterministic_baseline: bool = False,
    min_latency_params: Optional[dict] = None,
) -> Dict[str, Any]:
    """Avoided cancers / deaths / QALYs per ``per_n_scans`` imaging
    procedures, broken down by (region x 10-year age group x dose-reduction %).

    Dose-response-aware scaling.  Writing the reduced dose as a fraction
    ``f`` of baseline and ``m = 1 - f = -reduction_pct/100``, solid cancers
    are linear in dose so they lose ``m`` of their risk.  Leukaemia carries
    an ICRP 152 para 41 quadratic term that instead loses ``1 - f**2``.
    With ``w`` the leukaemia-quadratic share of a given metric::

        avoided = baseline * m * (1 + w*(1 - m))

    which collapses to the plain linear ``baseline * m`` when ``w == 0``.

    A dose reduction of 0% reproduces the baseline and is kept in the
    output so plots and tables show the reference line next to the
    reductions.

    Per-(region, age) cells with too few scans or an imprecise reported
    QALY mean are still computed (so users can see them) but flagged
    ``is_reliable = False``.  The precision bar follows the U.S. NCHS
    convention plus a relative-SE term: a cell with at least one simulated
    cancer is precise iff it produced a cancer in at least
    ``min_event_iters`` distinct iterations AND the relative standard error
    of its per-iteration QALY mean is at most ``max_rel_se``.  Cells with
    literally zero events are treated as reliable since their point
    estimate is a stable 0.

    Returns
    -------
    dict
        ``levels``          : tuple of dose-change percentages used
                              (negative = reduction).
        ``age_groups``      : tuple of age-decade labels in order.
        ``regions``         : sorted list of regions present in data.
        ``per_n_scans``     : normalisation constant (typically 100 000).
        ``iterations``      : number of MC iterations the rates were
                              averaged over (used for relative-SE notes).
        ``min_scans`` / ``min_expected_events`` : the reliability
                              thresholds in force.
        ``per_region_age``  : list of dicts, one per (region, age_decade)
                              pair with a non-empty scan count.  Each
                              dict carries baseline, per-level arrays
                              AND per-level Wald 95 % confidence-interval
                              arrays plus a boolean ``is_reliable`` flag.
        ``per_region``      : list of dicts, one per region, with arrays
                              summed over all age groups.  Convenient
                              for the "which region pays off most" view.
        ``cohort_totals``   : per-level cohort sums (cases_avoided,
                              deaths_avoided, QALY_saved, total_scans).
    """
    iters = _safe_iters(stats)
    # Optional modality filter — restrict exam_data and event lists to
    # exams / events whose modality matches the caller-provided code(s)
    # (case-insensitive).  ``modality`` may be a single string or an
    # iterable of strings.  When None/empty the full cohort is used.
    # ``modality_filter`` is echoed on the output so downstream UI /
    # formatter can title the view accordingly.
    if modality:
        if isinstance(modality, str):
            mod_keys = {_norm_modality(modality)}
        else:
            mod_keys = {_norm_modality(m) for m in modality if m}
        mod_keys.discard("")
    else:
        mod_keys = set()
    mod_key: Optional[Union[str, frozenset]] = (
        frozenset(mod_keys) if mod_keys else None
    )
    # Keep a reference to the FULL (unfiltered) exam list so the expensive
    # deterministic-baseline integral can be computed once over the whole
    # cohort and cached independently of the modality filter (see below).
    _full_exam_data = exam_data
    if mod_keys:
        _filter_key = (id(exam_data), frozenset(mod_keys))
        if _filter_key in _modality_filter_cache:
            exam_data = _modality_filter_cache[_filter_key]
        else:
            # Dental imaging exams match only the "Dental imaging"
            # pseudo-modality and "Dental CBCT" exams only "CBCT", not
            # "Radiography" (even though their actual modality is
            # Radiography) -- the same grouping the sweep cells use.
            exam_data = [
                ex for ex in (exam_data or [])
                if _modality_matches(
                    ex.get("modality"),
                    ex.get("reporting_region", ex.get("region", "")),
                    mod_keys)
            ]
            _modality_filter_cache[_filter_key] = exam_data
        cancer_events = _filter_events_by_modality(cancer_events, mod_keys)
        extra_cancer_events = _filter_events_by_modality(extra_cancer_events, mod_keys)
    base = _region_age_baseline(
        exam_data or [], cancer_events or [], iters,
        extra_cancer_events=extra_cancer_events,
        extra_iters_by_cell=extra_iters_by_cell,
    )

    # Optional deterministic baseline: replace the Monte Carlo point estimate
    # (mean) with the analytical ICRP 152 expectation, and replace the MC
    # CI with the analytical 95 % range of a SINGLE year's realised
    # outcome for that cell (normal approximation to the
    # Poisson-binomial outcome distribution), computed from per-exam
    # per-site variance formulas as ``mean +/- CI_Z*sqrt(var)``.  This
    # eliminates MC noise entirely -- the interval reflects only the
    # irreducible stochastic uncertainty of the Bernoulli cancer-induction
    # process (plus, after cross-year averaging, the year-to-year
    # variation of the expectation; see ``_average_cell_group``).  It does
    # NOT shrink with the MC iteration count: the analytical expectation
    # has no sampling error, so dividing by ``sqrt(n_it)`` is
    # meaningless and would collapse the interval as
    # compute time grew.
    #
    # This integral is expensive (per-exam ICRP-152 lifetime-risk
    # integration), so it is computed ONCE over the full, unfiltered
    # cohort and cached by exam-data identity + sim-config identity.
    # Changing only the modality selection then just filters the cached
    # per-cell dict (cell key already includes modality) instead of
    # re-running the analytical integral from scratch.
    det_base: Optional[Dict[Tuple[str, str, int], Dict[str, float]]] = None
    # Resolved unconditionally (not just under deterministic_baseline) so the
    # QALY accounting mode can be reported in the returned sweep -- the
    # annual-benefit plot needs it to state whether its ICRP nominal
    # reference is directly comparable to the simulated QALY.
    sim_cfg = (stats or {}).get("sim_config") or {}
    if deterministic_baseline:
        _det_key = (id(_full_exam_data), id(sim_cfg))
        _det_full = _deterministic_baseline_cache.get(_det_key)
        if _det_full is None:
            # Hold the kernel-state lock for the duration of the per-exam
            # analytical integral: it reads _KERNEL_CACHE / _SURV_CACHE / the
            # _RPG_* globals via risk.compute_lifetime_risk, and the next
            # year's run_simulation may call precompute_kernels (which clears
            # and repopulates those) on the main thread while this runs in
            # the background.  The lock is an RLock so the main thread's own
            # precompute_kernels can still re-enter it.
            from .risk_models import KERNEL_STATE_LOCK
            with KERNEL_STATE_LOCK:
                _det_full = _compute_deterministic_baselines(
                    _full_exam_data, sim_cfg,
                    min_latency_params=min_latency_params,
                )
            _deterministic_baseline_cache[_det_key] = _det_full
        if mod_keys:
            det_base = {}
            for cell, d in _det_full.items():
                d_mod, _d_reg, _d_dec = cell
                # ``_compute_deterministic_baselines`` already keys dental-
                # imaging subtype cells under the "Dental imaging"
                # pseudo-modality and Dental CBCT under "CBCT", matching
                # ``base``'s cells, so a plain modality match is
                # sufficient here.
                if _norm_modality(d_mod) in mod_keys:
                    det_base[cell] = d
        else:
            det_base = _det_full
        for cell, b in base.items():
            d = det_base.get(cell)
            if d is None:
                continue

            def _det_ci(mean_val: float, var_val: float) -> Tuple[float, float]:
                # Analytical 95 % range of a single year's realised outcome
                # for this cell (normal approximation to the
                # Poisson-binomial).  Kept symmetric about mean_val (no
                # clamp at zero) so downstream rescaling cannot displace
                # the interval's centre -- see _wald_ci.  Degenerate only
                # when std <= 0 (a genuine zero cell); the analytical
                # expectation has no sampling error, so there is no
                # n_it <= 1 condition.
                var_val = max(float(var_val), 0.0)
                std = float(np.sqrt(var_val))
                if std <= 0.0:
                    return mean_val, mean_val
                half = CI_Z * std
                return mean_val - half, mean_val + half

            b["cases_ci"] = _det_ci(d["cases"], d["var_cases"])
            b["deaths_ci"] = _det_ci(d["deaths"], d["var_deaths"])
            b["non_mortal_ci"] = _det_ci(d["non_mortal"], d["var_non_mortal"])
            b["QALY_ci"] = _det_ci(d["QALY_lost"], d["var_QALY"])
            # Deterministic cells have no MC precision criterion: the
            # analytical expectation has no sampling error, so rel_se_QALY
            # is 0, the interval is never "degenerate" in the MC sense
            # (a zero-variance cell is a genuine zero), and the cell is
            # never flagged for refinement.
            b["ci_degenerate"] = False
            b["rel_se_QALY"] = 0.0
            b["analytical"] = True
            # Replace point estimates with deterministic values
            b["cases"] = d["cases"]
            b["deaths"] = d["deaths"]
            b["non_mortal"] = d["non_mortal"]
            b["QALY_lost"] = d["QALY_lost"]
            b["life_lost"] = d["life_lost"]
            b["leukq_cases"] = d.get("leukq_cases", 0.0)
            b["leukq_deaths"] = d.get("leukq_deaths", 0.0)
            b["leukq_QALY"] = d.get("leukq_QALY", 0.0)
            # Keep scans/dose deterministic from the exam data.  Preserve
            # the MC-path all-exam count first: ``d["scans"]`` counts only
            # exams that have organ or cancer doses (dose-less exams such
            # as dental-imaging subtypes are skipped in
            # ``_compute_deterministic_baselines``), while ``scans_per_year``
            # stays the per-100k denominator.
            b["scans_all"] = b["scans"]
            b["scans"] = d["scans"]
            b["dose_sv"] = d["dose_sv"]

    levels = np.array([float(p) for p in dose_reductions_pct], dtype=float)
    mult = -levels / 100.0  # -20% reduction -> +0.20 fractional benefit
    min_scans = float(min_scans or 0.0)
    min_events = float(min_expected_events or 0.0)

    def _mult_eff(baseline: float, leuk_quad: float) -> np.ndarray:
        """Per-metric multiplier honouring the leukaemia quadratic dose response.

        Solid cancers are linear in dose, so a reduction to a fraction ``f``
        removes ``(1-f)`` of their risk.  The ICRP 152 para 41 leukaemia
        quadratic term instead loses ``(1-f**2) = (1-f)(1+f)``.  Writing
        ``m = 1-f`` and letting ``w`` be the quadratic share of this metric::

            m_eff = m*(1-w) + m*(2-m)*w = m*(1 + w*(1-m))

        Reduces to ``mult`` when ``w == 0``, and holds for dose increases too.
        """
        return mult * (1.0 + _lq_share(baseline, leuk_quad) * (1.0 - mult))

    # Exclude "Other radiography" — a catch-all bucket with no meaningful
    # dose coefficients.  Dental regions are now handled via the
    # "Dental imaging" pseudo-modality (their reporting_region is one of
    # ``_DENTAL_IMAGING_SUBTYPE_REGIONS``, mapped to modality "Dental
    # imaging" by ``_region_age_baseline`` above) and are filtered by the
    # modality selector, not here.
    _EXCLUDE_REGIONS = {"Other radiography"}
    base = {k: v for k, v in base.items() if k[1] not in _EXCLUDE_REGIONS}

    # Fold rare multi-region combos (e.g. "Neck, Pelvis" with <30 scans) into
    # "Other combinations" so the long tail of n=1-13 combos does not clutter
    # the heatmap or per-region table.  Single regions are never folded; only
    # combo labels (comma-joined bucket names) below RARE_COMBO_THRESHOLD are
    # merged.  The merge sums all per-cell numeric fields (scans, dose, cases,
    # deaths, QALY, …) across the folded combos' age groups.
    region_scan_totals: Dict[str, float] = {}
    for (_mod, _reg, _dec) in base:
        region_scan_totals[_reg] = region_scan_totals.get(_reg, 0.0) + base[(_mod, _reg, _dec)]["scans"]

    _folded_base: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for key, cell in base.items():
        _reg = str(key[1])
        if _is_combo_region(_reg) and region_scan_totals.get(_reg, 0.0) < _RARE_COMBO_THRESHOLD:
            folded_key = (key[0], _CT_OTHER_COMBINATIONS, key[2])
        else:
            folded_key = key
        if folded_key not in _folded_base:
            # Shallow-copy the first cell's dict; numeric fields will be
            # summed into it by the accumulation below.  Per-iteration arrays
            # (if present) are not merged — they are only used for CI
            # computation, which is approximate for folded cells anyway.
            _folded_base[folded_key] = dict(cell)
        else:
            existing = _folded_base[folded_key]
            for _k, _v in cell.items():
                if isinstance(_v, (int, float, np.floating, np.integer)):
                    existing[_k] = existing.get(_k, 0) + _v
                elif isinstance(_v, np.ndarray):
                    # Per-iteration arrays: sum element-wise (CI will be
                    # approximate for folded cells, which is acceptable
                    # given they represent <30 scans total).
                    if _k in existing and isinstance(existing[_k], np.ndarray):
                        existing[_k] = existing[_k] + _v
                    else:
                        existing[_k] = _v.copy() if hasattr(_v, "copy") else _v
    base = _folded_base

    # Build (modality, region) pairs from base keys.
    mod_regions = sorted({(mod, reg) for (mod, reg, _) in base.keys()})
    # Per-(modality, region) "has a risk model" flag: True iff the sum of
    # modelled exams (_cancer_doses truthy) over all age decades is > 0.
    # Regions with no model (e.g. non-dental CBCT body parts such as
    # Elbow/Ankle/Wrist, which get _cancer_doses=None) are shown as n/a in
    # the heatmap and excluded from the annual benefit/burden totals.
    region_has_model: Dict[Tuple[str, str], bool] = {
        (mod, reg): sum(
            float(base.get((mod, reg, di), {}).get("modelled_scans", 0.0))
            for di in range(len(AGE_DECADE_LABELS))
        ) > 0.0
        for (mod, reg) in mod_regions
    }
    if not mod_regions or levels.size == 0:
        return {
            "levels": tuple(levels.tolist()),
            "age_groups": AGE_DECADE_LABELS,
            "regions": mod_regions,
            "per_n_scans": int(per_n_scans),
            "iterations": int(iters),
            "min_scans": min_scans,
            "min_expected_events": min_events,
            "min_event_iters": int(min_event_iters),
            "max_rel_se": float(max_rel_se),
            "modality_filter": mod_key,
            "modality_region_map": {},
            "least_scans_cell": None,
            "QALY_mode": str(sim_cfg.get("QALY_mode", "full")).lower(),
            "QALY_discount_rate": float(sim_cfg.get("QALY_discount_rate", 0.0) or 0.0),
            "per_region_age": [],
            "per_region": [],
            "cohort_totals": {
                "cases_avoided_per_100k":      np.zeros_like(levels),
                "deaths_avoided_per_100k":     np.zeros_like(levels),
                "non_mortal_avoided_per_100k": np.zeros_like(levels),
                "QALY_saved_per_100k":         np.zeros_like(levels),
                "total_scans_per_year":        0.0,
            },
        }

    per_region_age: List[Dict[str, Any]] = []
    # Per-(modality, region) accumulators (annualised, before per-100k normalisation
    # — we re-normalise at the end against each (modality, region)'s own total
    # scans so "Chest per 100 k" is comparable to "Skull per 100 k").
    region_totals: Dict[Tuple[str, str], Dict[str, float]] = {
        (mod, reg): {
            "scans": 0.0, "dose_sv": 0.0,
            "cases": 0.0, "deaths": 0.0, "non_mortal": 0.0,
            "QALY_lost": 0.0,
            "leukq_cases": 0.0, "leukq_deaths": 0.0,
            "leukq_non_mortal": 0.0, "leukq_QALY": 0.0,
            "modelled_scans": 0.0, "modelled_dose_sv": 0.0,
        }
        for (mod, reg) in mod_regions
    }
    cohort_scans = 0.0
    cohort_cases = np.zeros_like(levels)
    cohort_deaths = np.zeros_like(levels)
    cohort_non_mortal = np.zeros_like(levels)
    cohort_QALY = np.zeros_like(levels)

    for (mod, reg) in mod_regions:
        for di, lbl in enumerate(AGE_DECADE_LABELS):
            b = base.get((mod, reg, di))
            if not b or b["scans"] <= 0.0:
                continue
            scans = float(b["scans"])
            scale = float(per_n_scans) / scans
            # Leukaemia's quadratic dose-response term responds to a dose
            # change more strongly than the linear sites, so each metric gets
            # its own multiplier weighted by that metric's quadratic share.
            _lq_cases = float(b.get("leukq_cases", 0.0) or 0.0)
            _lq_deaths = float(b.get("leukq_deaths", 0.0) or 0.0)
            _lq_QALY = float(b.get("leukq_QALY", 0.0) or 0.0)
            _lq_nm = max(0.0, _lq_cases - _lq_deaths)
            mult_cases = _mult_eff(float(b["cases"]), _lq_cases)
            mult_deaths = _mult_eff(float(b["deaths"]), _lq_deaths)
            mult_nm = _mult_eff(float(b["non_mortal"]), _lq_nm)
            mult_QALY = _mult_eff(float(b["QALY_lost"]), _lq_QALY)
            cases_arr      = b["cases"]      * scale * mult_cases
            deaths_arr     = b["deaths"]     * scale * mult_deaths
            non_mortal_arr = b["non_mortal"] * scale * mult_nm
            QALY_arr       = b["QALY_lost"]  * scale * mult_QALY

            # Wald 95 % CIs precomputed by _region_age_baseline from the
            # per-iteration counts (numerically-stable centred variance).
            # The point estimate (b["cases"] / b["deaths"] / etc.) is the
            # corresponding per-iteration mean, so it is unchanged.  Non-
            # mortal uses its own per-iteration array (cases - deaths) so
            # the cases/deaths covariance is taken into account.
            (c_lo, c_hi) = b["cases_ci"]
            (d_lo, d_hi) = b["deaths_ci"]
            (n_lo, n_hi) = b["non_mortal_ci"]
            (q_lo, q_hi) = b["QALY_ci"]
            cases_lo_arr,      cases_hi_arr      = _project_ci(c_lo, c_hi, scale, mult_cases)
            deaths_lo_arr,     deaths_hi_arr     = _project_ci(d_lo, d_hi, scale, mult_deaths)
            non_mortal_lo_arr, non_mortal_hi_arr = _project_ci(n_lo, n_hi, scale, mult_nm)
            QALY_lo_arr,       QALY_hi_arr       = _project_ci(q_lo, q_hi, scale, mult_QALY)

            # Reliability: total accumulated events across all MC
            # iterations.  Zero-event cells stay reliable because their
            # estimate is a stable 0 and they cannot pollute the ranking.
            total_events = float(b["total_events"])
            total_deaths = float(b["total_deaths"])
            event_iters = int(b.get("event_iters", 0) or 0)
            rel_se = float(b.get("rel_se_QALY", 0.0) or 0.0)
            ci_degenerate = bool(b.get("ci_degenerate", False))
            analytical = bool(b.get("analytical", False))
            scans_ok  = scans >= min_scans
            # Deterministic-baseline cells have no MC precision criterion:
            # the analytical expectation has no sampling error, so they
            # are reliable iff their scan count clears the floor, never
            # flagged for refinement, never degenerate.
            if analytical:
                events_ok = True
                needs_refinement = False
                ci_degenerate = False
            else:
                # Precision criterion on the reported QALY mean.  A cell with
                # at least one simulated cancer is precise only when it saw a
                # cancer in at least ``min_event_iters`` distinct iterations AND
                # the relative SE of its per-iteration QALY mean is at most
                # ``max_rel_se`` (and it clears the absolute-count floor
                # ``min_events``).  Zero-event cells are reliable because
                # their estimate is a stable 0 and cannot pollute the ranking.
                events_ok = (total_events == 0.0) or (
                    total_events >= min_events
                    and event_iters >= int(min_event_iters)
                    and rel_se <= float(max_rel_se)
                )
                # ``needs_refinement`` flags cells whose CI is wide ONLY
                # because of MC noise (insufficient precision).  These are
                # candidates for extra MC iterations — more iterations on
                # the small underlying cohort will tighten ``events_ok``
                # without changing the cohort itself.  Cells that fail only
                # ``scans_ok`` cannot be fixed by more MC (their denominator
                # is real).  A degenerate CI (n<=1 or std==0 with non-zero
                # mean) is also MC-noise-driven and refinable when there is a
                # real cohort.
                needs_refinement = bool((not events_ok or ci_degenerate) and scans > 0)
            is_reliable = bool(scans_ok and events_ok and not ci_degenerate)
            unreliable_reasons = []
            if not scans_ok:
                unreliable_reasons.append("exams<%g" % min_scans)
            if not analytical:
                if not events_ok and total_events > 0.0:
                    # Report each precision sub-criterion the cell failed.
                    if total_events < min_events:
                        unreliable_reasons.append("events<%g" % min_events)
                    if event_iters < int(min_event_iters):
                        unreliable_reasons.append("event_iters<%d" % int(min_event_iters))
                    if rel_se > float(max_rel_se):
                        unreliable_reasons.append("rel_se>%.0f%%" % (100.0 * float(max_rel_se)))
                if ci_degenerate:
                    unreliable_reasons.append("ci_degenerate (<2 event-bearing iterations)")

            # Baseline CI: reporting the same random quantity per
            # ``per_n_scans`` examinations is a deterministic linear rescale,
            # so Var(cX) = c**2 Var(X) and the interval scales by exactly the
            # same ``scale`` factor as the point estimate.  Transforming the
            # endpoints directly keeps the result centred on
            # ``mean * scale`` == the reported point estimate.
            def _scale_ci(lo: float, hi: float) -> Tuple[float, float]:
                return lo * scale, hi * scale

            per_region_age.append({
                "modality": mod,
                "region": reg,
                "age_group": lbl,
                "age_decade_idx": di,
                "scans_per_year": scans,
                "scans_all_per_year": float(b.get("scans_all", b["scans"])),
                "dose_sv_per_year": float(b["dose_sv"]),
                "mean_dose_msv": (b["dose_sv"] * 1000.0 / scans) if scans > 0 else 0.0,
                "modelled_scans_per_year": float(b.get("modelled_scans", 0.0)),
                "modelled_dose_sv_per_year": float(b.get("modelled_dose_sv", 0.0)),
                "age_hist_per_year": b.get("age_hist"),
                "sex_counts_per_year": b.get("sex_counts"),
                "has_risk_model": region_has_model[(mod, reg)],
                "baseline_cases_per_100k":      float(b["cases"]      * scale),
                "baseline_deaths_per_100k":     float(b["deaths"]     * scale),
                "baseline_non_mortal_per_100k": float(b["non_mortal"] * scale),
                "baseline_QALY_per_100k":       float(b["QALY_lost"]  * scale),
                "baseline_cases_ci_lo_per_100k":  float(_scale_ci(c_lo, c_hi)[0]),
                "baseline_cases_ci_hi_per_100k":  float(_scale_ci(c_lo, c_hi)[1]),
                "baseline_deaths_ci_lo_per_100k": float(_scale_ci(d_lo, d_hi)[0]),
                "baseline_deaths_ci_hi_per_100k": float(_scale_ci(d_lo, d_hi)[1]),
                "baseline_QALY_ci_lo_per_100k":   float(_scale_ci(q_lo, q_hi)[0]),
                "baseline_QALY_ci_hi_per_100k":   float(_scale_ci(q_lo, q_hi)[1]),
                # Leukaemia-quadratic share ``w`` per metric -- the only
                # quantity beyond the baseline needed to rebuild that
                # metric's multiplier at ANY dose change, via _mult_eff's
                # m_eff = m*(1 + w*(1-m)).  Stored so consumers can price a
                # level absent from ``dose_reductions_pct`` without
                # re-running the Monte Carlo (see compute_annual_benefit).
                "leukq_share_cases":      _lq_share(float(b["cases"]), _lq_cases),
                "leukq_share_deaths":     _lq_share(float(b["deaths"]), _lq_deaths),
                "leukq_share_non_mortal": _lq_share(float(b["non_mortal"]), _lq_nm),
                "leukq_share_QALY":       _lq_share(float(b["QALY_lost"]), _lq_QALY),
                "cases_avoided_per_100k":          cases_arr,
                "deaths_avoided_per_100k":         deaths_arr,
                "non_mortal_avoided_per_100k":     non_mortal_arr,
                "QALY_saved_per_100k":             QALY_arr,
                "cases_avoided_ci_lo_per_100k":      cases_lo_arr,
                "cases_avoided_ci_hi_per_100k":      cases_hi_arr,
                "deaths_avoided_ci_lo_per_100k":     deaths_lo_arr,
                "deaths_avoided_ci_hi_per_100k":     deaths_hi_arr,
                "non_mortal_avoided_ci_lo_per_100k": non_mortal_lo_arr,
                "non_mortal_avoided_ci_hi_per_100k": non_mortal_hi_arr,
                "QALY_saved_ci_lo_per_100k":         QALY_lo_arr,
                "QALY_saved_ci_hi_per_100k":         QALY_hi_arr,
                "total_observed_events": total_events,
                "total_observed_deaths": total_deaths,
                # Precision-criterion fields carried on the reported row so
                # the renderer and any precision check share the same
                # numbers.  ``QALY_mean_iter`` is the raw per-iteration QALY
                # mean (NOT scaled per-100k); ``QALY_m2`` is its Welford M2.
                "event_iters": event_iters,
                "rel_se_QALY": rel_se,
                "QALY_m2": float(b.get("QALY_m2", 0.0) or 0.0),
                "QALY_mean_iter": float(b.get("QALY_lost", 0.0) or 0.0),
                "iters_main":  int(b.get("iters_main", iters)),
                "iters_extra": int(b.get("iters_extra", 0)),
                "iters_total": int(b.get("iters_total", iters)),
                "scans_ok":  bool(scans_ok),
                "events_ok": bool(events_ok),
                "is_reliable": is_reliable,
                "needs_refinement": needs_refinement,
                "ci_degenerate": ci_degenerate,
                "unreliable_reason": "; ".join(unreliable_reasons),
                "analytical": analytical,
            })
            rt = region_totals[(mod, reg)]
            rt["scans"] += scans
            rt["dose_sv"] += float(b["dose_sv"])
            rt["cases"] += float(b["cases"])
            rt["deaths"] += float(b["deaths"])
            rt["non_mortal"] += float(b["non_mortal"])
            rt["QALY_lost"] += float(b["QALY_lost"])
            rt["leukq_cases"] += _lq_cases
            rt["leukq_deaths"] += _lq_deaths
            rt["leukq_non_mortal"] += _lq_nm
            rt["leukq_QALY"] += _lq_QALY
            rt["modelled_scans"] += float(b.get("modelled_scans", 0.0))
            rt["modelled_dose_sv"] += float(b.get("modelled_dose_sv", 0.0))
            # Cohort totals are summed in absolute (unnormalised) units
            # so we can re-normalise to per-100k of total cohort scans
            # at the very end.  Each metric keeps its own leukaemia-quadratic
            # weighting rather than the plain linear multiplier.
            cohort_scans += scans
            cohort_cases += b["cases"] * mult_cases
            cohort_deaths += b["deaths"] * mult_deaths
            cohort_non_mortal += b["non_mortal"] * mult_nm
            cohort_QALY += b["QALY_lost"] * mult_QALY

    per_region: List[Dict[str, Any]] = []
    for (mod, reg) in mod_regions:
        rt = region_totals[(mod, reg)]
        if rt["scans"] <= 0.0:
            continue
        scale = float(per_n_scans) / rt["scans"]
        per_region.append({
            "modality": mod,
            "region": reg,
            "scans_per_year": rt["scans"],
            "dose_sv_per_year": rt["dose_sv"],
            "mean_dose_msv": (rt["dose_sv"] * 1000.0 / rt["scans"]) if rt["scans"] > 0 else 0.0,
            "modelled_scans_per_year": float(rt.get("modelled_scans", 0.0)),
            "modelled_dose_sv_per_year": float(rt.get("modelled_dose_sv", 0.0)),
            "has_risk_model": region_has_model[(mod, reg)],
            "baseline_cases_per_100k":      float(rt["cases"]      * scale),
            "baseline_deaths_per_100k":     float(rt["deaths"]     * scale),
            "baseline_non_mortal_per_100k": float(rt["non_mortal"] * scale),
            "baseline_QALY_per_100k":       float(rt["QALY_lost"]  * scale),
            "cases_avoided_per_100k":
                rt["cases"] * scale * _mult_eff(rt["cases"], rt["leukq_cases"]),
            "deaths_avoided_per_100k":
                rt["deaths"] * scale * _mult_eff(rt["deaths"], rt["leukq_deaths"]),
            "non_mortal_avoided_per_100k":
                rt["non_mortal"] * scale * _mult_eff(rt["non_mortal"], rt["leukq_non_mortal"]),
            "QALY_saved_per_100k":
                rt["QALY_lost"] * scale * _mult_eff(rt["QALY_lost"], rt["leukq_QALY"]),
        })
    per_region.sort(key=lambda r: r["baseline_QALY_per_100k"], reverse=True)

    cohort_scale = float(per_n_scans) / cohort_scans if cohort_scans > 0 else 0.0
    # Least-scans (modality, region, age) cell, useful when filtering by modality to spot the smallest-throughput subgroup.
    # Ties are broken deterministically by (modality, region, age_decade_idx); None if no cell has scans.
    least_scans_cell: Optional[Dict[str, Any]] = None
    if per_region_age:
        _least = min(
            per_region_age,
            key=lambda r: (r["scans_per_year"], r["modality"], r["region"], r["age_decade_idx"]),
        )
        least_scans_cell = {
            "modality":               _least["modality"],
            "region":                 _least["region"],
            "age_group":              _least["age_group"],
            "age_decade_idx":         int(_least["age_decade_idx"]),
            "scans_per_year":         float(_least["scans_per_year"]),
            "mean_dose_msv":          float(_least["mean_dose_msv"]),
            "baseline_cases_per_100k":  float(_least["baseline_cases_per_100k"]),
            "baseline_deaths_per_100k": float(_least["baseline_deaths_per_100k"]),
            "baseline_QALY_per_100k":   float(_least["baseline_QALY_per_100k"]),
            "is_reliable":            bool(_least.get("is_reliable", True)),
        }
    # Build modality → regions map so the text summary can group regions
    # by modality.  When no modality filter is active, use the full
    # stats modality_stats.  When a filter is active, build from the
    # filtered exam data so each selected modality maps to its regions.
    modality_region_map: Dict[str, List[str]] = {}
    _DENTAL_IMAGING = "Dental imaging"
    if not mod_key:
        mod_stats = (stats or {}).get("modality_stats") or {}
        for mod_code, ms in mod_stats.items():
            sorted_mod_regions = sorted((ms.get("regions") or {}).keys())
            if sorted_mod_regions:
                modality_region_map[mod_code] = sorted_mod_regions
        # Move the "Dental imaging" subtypes (Panoramic/Kefalometry/
        # Intraoral/Dental) from whichever modality they ended up under
        # (typically "Radiography") into their own "Dental imaging"
        # pseudo-modality group, keeping each subtype's own name rather
        # than collapsing them into one flat region label. "Dental CBCT"
        # is NOT moved: it stays under its raw modality like every other
        # region.
        dental_regs = set()
        for mod_code in list(modality_region_map.keys()):
            _subtypes_here = [
                r for r in modality_region_map[mod_code]
                if r in _DENTAL_IMAGING_SUBTYPE_REGIONS
            ]
            if _subtypes_here:
                dental_regs.update(_subtypes_here)
                modality_region_map[mod_code] = [
                    r for r in modality_region_map[mod_code]
                    if r not in _DENTAL_IMAGING_SUBTYPE_REGIONS
                ]
        if dental_regs:
            modality_region_map[_DENTAL_IMAGING] = sorted(dental_regs)
        # Remove empty modality entries.
        modality_region_map = {
            k: v for k, v in modality_region_map.items() if v
        }
    else:
        # Build from filtered exam data: map each modality to the set
        # of reporting regions seen in the filtered cohort.
        _seen: Dict[str, set] = {}
        for ex in (exam_data or []):
            mod = str(ex.get("modality", "Unknown") or "Unknown")
            rpt = str(ex.get("reporting_region", ex.get("region", "Unknown")) or "Unknown")
            if rpt in _EXCLUDE_REGIONS:
                continue
            _seen.setdefault(mod, set()).add(rpt)
        # "Dental imaging" pseudo-modality: dental-imaging subtype exams
        # (Panoramic/Kefalometry/Intraoral/Dental) have actual modality
        # "Radiography", so they won't appear under their own modality
        # key.  Add them manually, keeping each subtype's own name.
        # "Dental CBCT" is not special-cased: it stays under the raw
        # modality its exams carry.
        if "DENTAL IMAGING" in (mod_keys or set()):
            dental_regs = _seen.get("Radiography", set()) & _DENTAL_IMAGING_SUBTYPE_REGIONS
            if dental_regs:
                modality_region_map["Dental imaging"] = sorted(dental_regs)
        for mod_code, regs in _seen.items():
            mod_k = _norm_modality(mod_code)
            if mod_k == "DENTAL IMAGING":
                continue
            # Skip dental-imaging subtype regions under Radiography when
            # the Dental imaging pseudo-modality is selected separately.
            if "DENTAL IMAGING" in (mod_keys or set()):
                regs = {r for r in regs if r not in _DENTAL_IMAGING_SUBTYPE_REGIONS}
            if regs:
                modality_region_map[mod_code] = sorted(regs)
    # Both branches above list "Dental CBCT" under the RAW modality its
    # exams carry (modality_stats / exam_data); the cells are keyed under
    # "CBCT", so move it there (sorted, like every other entry).
    modality_region_map = _rehome_dental_cbct_map(
        modality_region_map, _DENTAL_CBCT_MODALITY,
        any(r["region"] == _DENTAL_CBCT_REGION for r in per_region_age))
    if _DENTAL_CBCT_MODALITY in modality_region_map:
        modality_region_map[_DENTAL_CBCT_MODALITY].sort()

    return {
        "levels": tuple(levels.tolist()),
        "age_groups": AGE_DECADE_LABELS,
        "regions": mod_regions,
        "per_n_scans": int(per_n_scans),
        "iterations": int(iters),
        "min_scans": min_scans,
        "min_expected_events": min_events,
        "min_event_iters": int(min_event_iters),
        "max_rel_se": float(max_rel_se),
        "modality_filter": mod_key,
        "modality_region_map": modality_region_map,
        "least_scans_cell": least_scans_cell,
        "deterministic_baseline": bool(deterministic_baseline),
        "interval_kind": ("analytical_single_year_range"
                          if deterministic_baseline else "mc_ci_of_mean"),
        "QALY_mode": str(sim_cfg.get("QALY_mode", "full")).lower(),
        "QALY_discount_rate": float(sim_cfg.get("QALY_discount_rate", 0.0) or 0.0),
        "per_region_age": per_region_age,
        "per_region": per_region,
        "cohort_totals": {
            "cases_avoided_per_100k":      cohort_cases * cohort_scale,
            "deaths_avoided_per_100k":     cohort_deaths * cohort_scale,
            "non_mortal_avoided_per_100k": cohort_non_mortal * cohort_scale,
            "QALY_saved_per_100k":         cohort_QALY * cohort_scale,
            "total_scans_per_year":        float(cohort_scans),
        },
    }


# "Average across years" batch support
#
# A "per_year_average" batch (year_batch.run_yearly_simulation) discards
# each year's heavy exam_data/cancer_events before loading the next year,
# so by the time the GUI renders the Research Questions tab only the last
# processed year's detail is still in memory -- recomputing the sweep from
# it would silently mislabel a single year's numbers as "average across
# years".  Instead, year_batch.py calls compute_region_age_dose_sweep()
# once per year (while that year's data is still alive, with the same
# fixed default arguments every time) and average_region_age_sweeps() below
# combines the per-year results with the same equal-weighted, "absent =
# genuine 0" convention used for every other average-mode statistic in
# this codebase (see year_batch._average_numeric_tree).
#
# The Research Questions tab itself (always the analytical ICRP-152
# baseline) uses the POOLED mode, pool_region_age_sweeps(): all year groups
# are treated as one exam-weighted cohort and annualised by dividing by the
# number of year groups, so a cell's point estimate is the pooled count /
# pooled exams and its interval is the analytical 95 % range of the mean
# annual realised outcome (sqrt(Σ var_y)/k).  The between-year Student-t CI
# columns are identical in both modes.  Tagged interval_kind =
# "analytical_pooled_years_range" and combine = "pooled".

_CI_LO_SUFFIX = "_ci_lo_per_100k"
_CI_HI_SUFFIX = "_ci_hi_per_100k"
_YR_MIN_SUFFIX = "_yr_min_per_100k"
_YR_MAX_SUFFIX = "_yr_max_per_100k"
# Between-year statistics of the k annual point estimates of a cell (the
# calendar years treated as replicate observations of the hospital's imaged
# population): sample SD (ddof=1, k >= 2), number of contributing years and
# a Student-t 95 % CI of the k-year mean, ``mean ± t(0.975, k-1) * sd/sqrt(k)``,
# reported only for k >= _YR_CI_MIN_YEARS.  This is cohort-composition
# uncertainty -- it says nothing about risk-model or dosimetric uncertainty.
_YR_SD_SUFFIX = "_yr_sd_per_100k"
_YR_N_SUFFIX = "_yr_n"
_YR_CI_LO_SUFFIX = "_yr_ci_lo_per_100k"
_YR_CI_HI_SUFFIX = "_yr_ci_hi_per_100k"
_YR_CI_MIN_YEARS = 3
_YR_SUFFIXES = (_YR_MIN_SUFFIX, _YR_MAX_SUFFIX, _YR_SD_SUFFIX, _YR_N_SUFFIX,
                _YR_CI_LO_SUFFIX, _YR_CI_HI_SUFFIX)
# Two-sided 97.5 % Student-t quantiles by degrees of freedom (no scipy in
# this project).  Beyond df = 30 the normal limit 1.96 is used.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
         19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
         25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t975(df: int) -> float:
    """Two-sided 97.5 % Student-t quantile for ``df`` degrees of freedom."""
    if df < 1:
        return float("nan")
    return _T975.get(int(df), 1.96)


def between_year_stats(values: "np.ndarray") -> Tuple[int, "np.ndarray", "np.ndarray", "np.ndarray"]:
    """``(k, sd, ci_lo, ci_hi)`` of the annual estimates stacked along axis 0.

    ``sd`` is the sample SD (ddof=1; NaN when k < 2).  The CI is the
    Student-t 95 % interval of the k-year mean, ``mean ± t(0.975, k-1) *
    sd/sqrt(k)``, NaN when ``k < _YR_CI_MIN_YEARS`` (two years give a
    formally computable but meaningless interval, t = 12.7).  Not clamped at
    zero (clamp at the display boundary, like every other interval here).
    """
    stacked = np.asarray(values, dtype=float)
    k = int(stacked.shape[0])
    shape = stacked.shape[1:]
    nan = np.full(shape, np.nan)
    if k < 2:
        return k, nan, nan.copy(), nan.copy()
    sd = stacked.std(axis=0, ddof=1)
    if k < _YR_CI_MIN_YEARS:
        return k, sd, nan, nan.copy()
    half = _t975(k - 1) * sd / np.sqrt(k)
    mean = stacked.mean(axis=0)
    return k, sd, mean - half, mean + half


# Analytical (deterministic-baseline) interval kinds.  A single year's
# sweep is tagged ``"analytical_single_year_range"``; a cross-year POOLED
# average of analytical sweeps is tagged ``"analytical_pooled_years_range"``
# (see :func:`average_region_age_sweeps`).  Both are treated identically
# by the cross-year combiner's analytical branches, so re-averaging an
# already-pooled sweep still works.
ANALYTICAL_INTERVAL_KINDS = frozenset(
    {"analytical_single_year_range", "analytical_pooled_years_range"})


def sweep_is_analytical(sweep) -> bool:
    """True when *sweep* carries an analytical (deterministic-baseline) kind."""
    return str((sweep or {}).get("interval_kind", "")) in ANALYTICAL_INTERVAL_KINDS


# Genuine cumulative counts (not per-iteration rates) -- summed across years
# rather than averaged, since they represent the total evidence feeding the
# average estimate.  ``event_iters`` (distinct event-bearing iterations)
# belongs here too: it is a count of evidence, and the cross-year total is
# the number of event-bearing iterations the pooled estimate rests on.
_SUM_CELL_KEYS = {"total_observed_events", "total_observed_deaths",
                  "iters_main", "iters_extra", "iters_total",
                  "event_iters"}
_AND_CELL_KEYS = {"is_reliable", "scans_ok", "events_ok"}
_OR_CELL_KEYS = {"needs_refinement", "has_risk_model", "ci_degenerate"}
_REASON_CELL_KEY = "unreliable_reason"
# EXTENSIVE per-year quantities: a year in which this cell saw no exams
# genuinely contributed 0 to it, so every year of the batch belongs in the
# denominator ("absent = genuine 0", as year_batch._average_numeric_tree
# does).  These are the ONLY such fields in a sweep cell.
#
# Everything else numeric in a cell is an INTENSITY, already normalised by
# that cell's own scan count -- ``*_per_100k`` (scale = per_n_scans/scans,
# see compute_region_age_dose_sweep), ``mean_dose_msv`` (dose_sv*1000/scans)
# and ``leukq_share_*`` (a fraction in [0, 1]).  A year with no exams of
# this cell leaves those UNDEFINED (0/0), not zero, so it must be left out
# of the denominator entirely -- dividing them by the full ``n_years``
# shrinks a cell present in k of n years to k/n of its true value.  It
# would also put the point estimate BELOW the min-max spread computed just
# after (which is taken over the present years only), an interval that
# cannot contain its own centre.
_PER_YEAR_CELL_KEYS = {"scans_per_year", "scans_all_per_year", "dose_sv_per_year",
                       "modelled_scans_per_year",
                       "modelled_dose_sv_per_year", "age_hist_per_year", "sex_counts_per_year"}


def _average_cell_group(cells: List[Dict[str, Any]], identity_fields: Tuple[str, ...],
                        n_years: int, interval_kind: str = "mc_ci_of_mean",
                        combine: str = "equal_weight",
                        min_scans: float = 0.0) -> Dict[str, Any]:
    """Equal-weighted average of one (modality, region[, age_group]) cell
    across the year groups that produced it.

    ``combine="pooled"`` instead treats every year's exams as ONE cohort:
    each intensity (``*_per_100k`` point estimates and the other
    per-exam-normalised fields) is the exam-weighted mean over the present
    years, ``Σ v_y·s_y / Σ s_y`` with ``s_y = scans_per_year`` of that
    year's cell (equal-weight fallback when ``Σ s_y == 0``), which equals
    the absolute pooled count divided by the pooled exam count.  The
    ``leukq_share_*`` fractions are weighted by the absolute baseline count
    they are a share of (``baseline_<m>_per_100k · s_y``).  The extensive
    per-year fields, the summed / AND / OR flags and the between-year
    statistics are computed exactly as in equal-weight mode.  The interval
    is, for both kinds, that of the k-year MEAN annual realised outcome
    (independent years): ``half = sqrt(Σ (half_y·s_y)²) / Σ s_y`` per
    ``per_n`` exams, i.e. ``CI_Z·sqrt(Σ var_y)/k`` in absolute annual
    counts.

    A cell missing from some year (e.g. a region with zero scans that
    year) contributes 0 for that year to the EXTENSIVE per-year fields
    (``scans_per_year`` / ``dose_sv_per_year``, see
    ``_PER_YEAR_CELL_KEYS``) -- diluting their average, not excluding it
    from the denominator, matching :func:`year_batch._average_numeric_tree`.
    Every other numeric field is an intensity normalised by that cell's own
    scan count, which a year with no such scans leaves undefined rather
    than zero, so those are averaged over the years the cell is actually
    present in.  Per-cell confidence intervals are pooled, not averaged.
    The pooling rule depends on ``interval_kind``:

    * ``"mc_ci_of_mean"`` (Monte Carlo runs): each year's
      ``*_ci_lo/hi_per_100k`` is a Wald interval from an independent MC
      run, so the cross-year point estimate (their equal-weight mean) has
      variance ``Σ Var_y / k²`` and a half-width
      ``sqrt(Σ half_y²) / k`` (``half_y = (hi_y - lo_y) / 2``).
      ``rel_se_QALY`` is then re-derived from the pooled interval.

    * ``"analytical_single_year_range"`` (deterministic baseline): each
      year's interval is the analytical 95 % range of a single year's
      realised outcome (``half_y = CI_Z * sigma_y``).  A randomly chosen
      year's outcome is a k-component mixture, so per element
      ``var_mix = mean(sigma_y²) + mean((mu_y - mu_bar)²)`` and
      ``half = CI_Z * sqrt(var_mix)``.  ``rel_se_QALY`` is left at 0
      (the analytical expectation has no sampling error).

    The year-to-year spread of the point estimates is kept under the
    separate ``*_yr_min/_yr_max_per_100k`` keys as a descriptive "how much does this
    vary year to year" signal, for both kinds.  The inferential version of
    the same information is the BETWEEN-YEAR CI: with the k present years
    as replicate observations, ``*_yr_n`` = k, ``*_yr_sd_per_100k`` = their
    sample SD (ddof=1) and ``*_yr_ci_lo/hi_per_100k`` = ``mean ± t(0.975,
    k-1) * sd/sqrt(k)`` (see :func:`between_year_stats`; NaN when k <
    ``_YR_CI_MIN_YEARS``).  It quantifies how well the k-year mean pins
    down the hospital's typical annual value given the variation in who
    was imaged; it contains no risk-model or dosimetric uncertainty and is
    the interval the manuscript's supplementary tables report.  None of
    these is clamped at zero here; clamping belongs at the reporting
    boundary (same reasoning as :func:`_wald_ci`).

    ``mean_dose_msv`` is an exception: like the ``*_per_100k`` fields it is
    normalised by the cell's scan count, but unlike them it is a
    DETERMINISTIC per-exam cohort property (sum of exam doses / count of
    exams), not a Monte-Carlo rate estimate.  Equal-weighting it across
    years gives a year with 1 high-dose paediatric exam the same weight as
    a year with 100 low-dose exams, inflating low-volume cells (notably
    0-9 in combo regions like "Head, Trunk" / "Head, Abdomen").  It is
    therefore skipped in the generic loop below and RECONSTRUCTED from the
    correctly-pooled extensive fields after the loop:

        mean_dose_msv = (Σ dose_sv_per_year / n_years) * 1000
                        / (Σ scans_per_year / n_years)

    The ``n_years`` denominators cancel, so this is exactly the per-exam
    average across the whole pooled cohort -- the same answer you would
    get by pooling every year's exams into one list and taking the mean
    dose.  O(1) per cell (one division), no extra iteration.

    The small-cell exam floor is applied to the pooled TOTAL: ``scans_ok``
    compares ``min_scans`` against the sum of the cell's exams over all
    year groups (an absent year contributes a genuine 0), not to each
    year's count.

    ``rel_se_QALY`` / ``QALY_m2`` / ``QALY_mean_iter`` are statistical
    intensity fields of the precision criterion; the generic loop averages
    them over present years.  For ``"mc_ci_of_mean"`` sweeps
    ``rel_se_QALY`` is then RE-DERIVED from the pooled ``baseline_QALY``
    interval -- ``(pooled_half / CI_Z) / mean`` -- so it reflects the
    cross-year pooled uncertainty rather than the mean of the per-year
    precisions.  For ``"analytical_single_year_range"`` sweeps it is set
    to 0 (the analytical expectation has no sampling error); it is shown
    only as a year-to-year diagnostic, never as a formal interval.
    """
    present_n = len(cells)
    pooled = combine == "pooled"
    all_keys: set = set()
    for c in cells:
        all_keys.update(c.keys())
    out: Dict[str, Any] = {}
    # Exam weights of the present years (pooled mode only).
    s_y = [float(c.get("scans_per_year") or 0.0) for c in cells]

    def _weighted(vals, weights):
        """Σ v·w / Σ w over (value, weight) pairs; equal-weight mean when
        the weights sum to 0; None when there is nothing to combine."""
        pairs = [(np.asarray(v, dtype=float), float(w)) for v, w in zip(vals, weights)
                 if v is not None]
        if not pairs:
            return None
        w_sum = sum(w for _v, w in pairs)
        if w_sum > 0.0:
            acc = sum(v * w for v, w in pairs) / w_sum
        else:
            acc = sum(v for v, _w in pairs) / len(pairs)
        return float(acc) if np.ndim(acc) == 0 else acc

    for k in all_keys:
        if k in identity_fields:
            out[k] = cells[0][k]
        elif k.endswith(_CI_LO_SUFFIX) or k.endswith(_CI_HI_SUFFIX):
            continue  # filled in the second pass below
        elif k.endswith(_YR_SUFFIXES):
            continue  # filled in the second pass below
        elif k == "mean_dose_msv":
            continue  # reconstructed from pooled dose_sv/scans after the loop
        elif k == _REASON_CELL_KEY:
            # Per-year scan-floor reasons ("exams<30") are dropped here:
            # the floor is on the
            # POOLED exam count over all year groups, recomputed after the
            # loop.  The "present in only k/n" note is also deferred -- it
            # is appended only when the pooled cell ends up unreliable.
            reasons = sorted({
                r for c in cells for r in str(c.get(k) or "").split("; ")
                if r and not r.startswith(("exams<", "scans_per_year<"))
            })
            out[k] = "; ".join(reasons)
        elif k in _SUM_CELL_KEYS:
            out[k] = sum(float(c.get(k, 0) or 0) for c in cells)
        elif k in _AND_CELL_KEYS:
            # The pooled exam-count floor is recomputed after the loop
            # ("scans_ok"); the remaining AND flags combine plainly -- a
            # year with no exams of this cell is a genuine 0 in the pooled
            # count, not a reliability failure by itself.
            out[k] = all(bool(c.get(k)) for c in cells)
        elif k in _OR_CELL_KEYS:
            out[k] = any(bool(c.get(k)) for c in cells)
        else:
            # Generic numeric (scalar or array) mean.  Non-numeric leaves
            # (e.g. a string field this schema-agnostic function was not
            # told about) fall back to the first year's value, same as
            # year_batch._average_numeric_tree does for its own
            # non-numeric leaves, instead of crashing on ``str / int``.
            total = None
            n_present = 0
            numeric = True
            for c in cells:
                v = c.get(k)
                if v is None:
                    continue
                if not isinstance(v, (int, float, np.ndarray)) or isinstance(v, bool):
                    numeric = False
                    break
                total = v if total is None else total + v
                n_present += 1
            if not numeric:
                out[k] = next((c[k] for c in cells if k in c), None)
            elif total is None:
                pass
            elif pooled and k not in _PER_YEAR_CELL_KEYS:
                # Pooled cohort: exam-weighted mean over the present years.
                # leukq_share_<m> is a fraction of the baseline <m> count,
                # so its weight is that absolute count (baseline·scans);
                # falls back to exam weights when those are all zero.
                weights = s_y
                if k.startswith("leukq_share_"):
                    _bk = "baseline_" + k[len("leukq_share_"):] + "_per_100k"
                    _bw = [float(np.sum(np.asarray(c.get(_bk, 0.0) or 0.0, dtype=float))) * w
                           for c, w in zip(cells, s_y)]
                    if sum(_bw) > 0.0:
                        weights = _bw
                out[k] = _weighted([c.get(k) for c in cells], weights)
            else:
                # Extensive per-year fields divide by the whole batch (an
                # absent year really did contribute 0); intensities divide
                # by the years that actually have the field, since an absent
                # year leaves them undefined, not zero.  See the note on
                # _PER_YEAR_CELL_KEYS.
                denom = n_years if k in _PER_YEAR_CELL_KEYS else n_present
                out[k] = total / denom if denom else total
    for lo_key in [k for k in all_keys if k.endswith(_CI_LO_SUFFIX)]:
        prefix = lo_key[: -len(_CI_LO_SUFFIX)]
        base_key = prefix + "_per_100k"
        hi_key = prefix + _CI_HI_SUFFIX
        yr_min_key = prefix + _YR_MIN_SUFFIX
        yr_max_key = prefix + _YR_MAX_SUFFIX
        # Years that carry all three of base/lo/hi for this prefix.  The
        # cross-year point estimate (out[base_key]) was already computed by
        # the generic loop over the present years; reuse it instead of
        # recomputing, so a cell present in k of n years pools only those k.
        years = [c for c in cells
                 if c.get(base_key) is not None
                 and c.get(lo_key) is not None
                 and c.get(hi_key) is not None]
        if not years:
            continue
        mean = out.get(base_key)
        if mean is None:
            continue
        mean = np.broadcast_to(np.asarray(mean, dtype=float), np.shape(years[0][base_key]))
        half_y = np.stack([
            np.broadcast_to(
                (np.asarray(c[hi_key], dtype=float)
                 - np.asarray(c[lo_key], dtype=float)) / 2.0,
                np.shape(years[0][base_key]))
            for c in years
        ])
        k = len(years)
        w_y = np.asarray([float(c.get("scans_per_year") or 0.0) for c in years])
        if pooled and w_y.sum() > 0.0:
            # Pooled cohort, both kinds: the k years are independent, so
            # the mean annual outcome per per_n exams has
            # half = sqrt(Σ (half_y·s_y)²) / Σ s_y  (= CI_Z·sqrt(Σ var_y)/k
            # in absolute annual counts).  Falls through to the
            # equal-weight rules when no present year has exams.
            w_b = w_y.reshape((k,) + (1,) * (half_y.ndim - 1))
            pooled_half = np.sqrt(np.sum((half_y * w_b) ** 2, axis=0)) / w_y.sum()
        elif interval_kind in ANALYTICAL_INTERVAL_KINDS:
            # A randomly chosen year's outcome is a k-component mixture:
            # sigma_y = half_y/CI_Z, mu_y = per-year point estimate,
            # var_mix = mean(sigma_y²) + mean((mu_y - mu_bar)²),
            # half = CI_Z*sqrt(var_mix).  mu_bar equals out[base_key]
            # (already computed by the generic loop).
            sigma_y = half_y / CI_Z
            mu_y = np.stack([
                np.broadcast_to(np.asarray(c[base_key], dtype=float),
                                np.shape(years[0][base_key]))
                for c in years
            ])
            mu_bar = mean
            var_mix = np.mean(sigma_y ** 2, axis=0) + np.mean(
                (mu_y - mu_bar) ** 2, axis=0)
            pooled_half = CI_Z * np.sqrt(np.maximum(var_mix, 0.0))
        else:
            # Var(mean) = Σ Var_y / k²  =>  half = sqrt(Σ half_y²) / k.
            # Not clamped at zero here (see docstring / _wald_ci).
            pooled_half = np.sqrt(np.sum(half_y ** 2, axis=0)) / k
        out[lo_key] = mean - pooled_half
        out[hi_key] = mean + pooled_half
        # Year-to-year spread of the point estimates: the min/max over the
        # present years.
        pts = [c[base_key] for c in years]
        stacked = np.stack([
            np.broadcast_to(np.asarray(p, dtype=float), np.shape(pts[0])) for p in pts
        ])
        out[yr_min_key] = stacked.min(axis=0)
        out[yr_max_key] = stacked.max(axis=0)
        # Between-year SD / n / Student-t 95 % CI of the k-year mean (the
        # calendar years as replicate observations of the imaged cohort).
        # Same statistic for both interval kinds -- it describes how much
        # the annual expectation moved with who was imaged, not MC noise
        # (for the MC kind it also absorbs the small per-year MC error).
        k_yr, yr_sd, yr_lo, yr_hi = between_year_stats(stacked)
        out[prefix + _YR_N_SUFFIX] = k_yr
        out[prefix + _YR_SD_SUFFIX] = yr_sd
        out[prefix + _YR_CI_LO_SUFFIX] = yr_lo
        out[prefix + _YR_CI_HI_SUFFIX] = yr_hi
    # Re-derive rel_se_QALY from the pooled baseline_QALY interval so it
    # reflects the cross-year pooled uncertainty, not the mean of the
    # per-year precisions.  rel_se = (pooled_half / CI_Z) / mean; 0 when the
    # mean is non-positive (a stable estimate of zero, not an imprecise one).
    # Only for the MC kind -- the analytical kind has no sampling error, so
    # rel_se_QALY stays at 0 (the AND/OR-combined flags already reflect it).
    if interval_kind not in ANALYTICAL_INTERVAL_KINDS:
        _qaly_mean = out.get("baseline_QALY_per_100k")
        _qaly_lo = out.get("baseline_QALY_ci_lo_per_100k")
        _qaly_hi = out.get("baseline_QALY_ci_hi_per_100k")
        if (_qaly_mean is not None and _qaly_lo is not None and _qaly_hi is not None
                and "rel_se_QALY" in all_keys):
            _m = np.asarray(_qaly_mean, dtype=float)
            _half = (np.asarray(_qaly_hi, dtype=float) - np.asarray(_qaly_lo, dtype=float)) / 2.0
            with np.errstate(divide="ignore", invalid="ignore"):
                _rel = np.where(_m > 0.0, (_half / CI_Z) / _m, 0.0)
            out["rel_se_QALY"] = float(np.asarray(_rel).flat[0]) if _rel.ndim == 0 else _rel
    else:
        if "rel_se_QALY" in all_keys:
            out["rel_se_QALY"] = 0.0
    # Small-cell floor: ``min_scans`` applies to the TOTAL exam count of
    # the cell pooled over all year groups (a single-year sweep: that
    # sweep's exams), not to each year's count.  ``s_y`` holds each
    # present year's exams; absent years genuinely contribute 0, so the
    # pooled total is the plain sum.
    if "scans_ok" in all_keys:
        total_scans = float(sum(s_y))
        out["scans_ok"] = total_scans >= min_scans
        out["is_reliable"] = bool(
            out["scans_ok"] and out.get("events_ok", True)
            and not out.get("ci_degenerate", False))
        if _REASON_CELL_KEY in all_keys:
            reasons = [r for r in out.get(_REASON_CELL_KEY, "").split("; ") if r]
            if not out["scans_ok"]:
                reasons.append(
                    "exams<%g (%g in %d pooled year group(s))"
                    % (min_scans, total_scans, present_n))
            if present_n < n_years and not out["is_reliable"]:
                reasons.append(
                    f"present in only {present_n}/{n_years} year group(s)")
            out[_REASON_CELL_KEY] = "; ".join(reasons)
    # Reconstruct mean_dose_msv from the correctly-pooled extensive fields
    # (scans_per_year / dose_sv_per_year are already divided by n_years in
    # the generic loop above, so their ratio is the scan-weighted per-exam
    # mean across the whole batch -- NOT the equal-weighted mean of the
    # per-year means, which would inflate low-volume cells).  See the
    # docstring note above.
    if "mean_dose_msv" in all_keys:
        _scans = float(out.get("scans_per_year", 0.0) or 0.0)
        _dose_sv = float(out.get("dose_sv_per_year", 0.0) or 0.0)
        out["mean_dose_msv"] = (_dose_sv * 1000.0 / _scans) if _scans > 0.0 else 0.0
    return out


def _average_cell_list(per_year_lists: List[List[Dict[str, Any]]],
                       key_fields: Tuple[str, ...], n_years: int,
                       interval_kind: str = "mc_ci_of_mean",
                       combine: str = "equal_weight",
                       min_scans: float = 0.0) -> List[Dict[str, Any]]:
    """Group per-year cell lists (``per_region_age`` or ``per_region``) by
    ``key_fields`` and average each group with :func:`_average_cell_group`."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for cells in per_year_lists:
        for c in cells:
            key = tuple(c.get(f) for f in key_fields)
            groups.setdefault(key, []).append(c)
    return [_average_cell_group(cells, key_fields, n_years, interval_kind,
                                combine, min_scans)
            for cells in groups.values()]


def apply_min_scans(sweep: Dict[str, Any], min_scans: float) -> Dict[str, Any]:
    """Return a copy of *sweep* re-flagged for a different small-cell exam floor.

    Only ``scans_ok`` / ``is_reliable`` / ``unreliable_reason`` and the
    sweep-level ``min_scans`` change; every estimate and interval is untouched.
    The floor is tested against the cell's TOTAL exams over the pooled year
    groups: ``scans_per_year * sweep["_averaged_across_years"]`` (1 for a
    single sweep).  ``is_reliable = scans_ok and events_ok and not ci_degenerate``
    (``events_ok`` defaults True, ``ci_degenerate`` False — analytical cells).
    """
    if not sweep:
        return sweep
    min_scans = float(min_scans)
    n_years = int(sweep.get("_averaged_across_years") or 1)
    out = dict(sweep)
    for list_key in ("per_region_age", "per_region"):
        cells = sweep.get(list_key)
        if not cells:
            continue
        out[list_key] = [dict(c) for c in cells]
        for c in out[list_key]:
            if "scans_ok" not in c:
                continue
            total = float(c.get("scans_per_year") or 0.0) * n_years
            c["scans_ok"] = total >= min_scans
            c["is_reliable"] = bool(
                c["scans_ok"] and c.get("events_ok", True)
                and not c.get("ci_degenerate", False))
            old = [r for r in str(c.get("unreliable_reason") or "").split("; ") if r]
            reasons = [
                r for r in old
                if not r.startswith(("exams<", "scans_per_year<", "present in only"))
            ]
            if not c["scans_ok"]:
                reasons.append(
                    "exams<%g (%g in %d pooled year group(s))"
                    % (min_scans, total, n_years))
            # The presence note is informational and, as in
            # _average_cell_group, kept only on cells that end up unreliable.
            if not c["is_reliable"]:
                reasons.extend(r for r in old if r.startswith("present in only"))
            c["unreliable_reason"] = "; ".join(reasons)
    out["min_scans"] = min_scans
    least = out.get("least_scans_cell")
    if isinstance(least, dict):
        key = (least.get("modality"), least.get("region"),
               least.get("age_group"))
        match = next(
            (c for c in out.get("per_region_age") or []
             if (c.get("modality"), c.get("region"), c.get("age_group")) == key),
            None)
        if match is not None:
            least = dict(least)
            least["is_reliable"] = bool(match.get("is_reliable", True))
            out["least_scans_cell"] = least
    return out


def _average_dict_of_arrays(dicts: List[Dict[str, Any]], n_years: int,
                            weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """Elementwise mean of a list of same-shaped numeric dicts (``cohort_totals``).

    With *weights* (pooled mode; one per dict, the year's total exams)
    every key containing ``_per_100k`` becomes the weighted mean
    ``Σ v_y·w_y / Σ w_y`` (plain mean when the weights sum to 0); the
    other keys stay ``Σ / n_years``.
    """
    all_keys: set = set()
    for d in dicts:
        all_keys.update(d.keys())
    w_sum = float(sum(weights)) if weights else 0.0
    out: Dict[str, Any] = {}
    for k in all_keys:
        total = None
        w_tot = None
        n_present = 0
        for i, d in enumerate(dicts):
            v = d.get(k)
            if v is None:
                continue
            total = v if total is None else total + v
            n_present += 1
            if weights is not None:
                wv = np.asarray(v, dtype=float) * float(weights[i])
                w_tot = wv if w_tot is None else w_tot + wv
        if total is None:
            continue
        if weights is not None and "_per_100k" in k:
            out[k] = (w_tot / w_sum) if w_sum > 0.0 else total / n_present
        else:
            out[k] = total / n_years
    return out


def average_region_age_sweeps(sweeps: List[Dict[str, Any]],
                              combine: str = "equal_weight") -> Dict[str, Any]:
    """Cross-year combination of per-year :func:`compute_region_age_dose_sweep`
    results from a "per_year_average" batch.

    ``combine="equal_weight"`` (default) is the equal-weighted mean of the
    per-year sweeps.  ``combine="pooled"`` (see :func:`pool_region_age_sweeps`)
    treats all years as ONE exam-weighted cohort, annualised by dividing
    by the number of year groups -- see :func:`_average_cell_group` for
    the per-field rules.  An analytical input then gets
    ``interval_kind="analytical_pooled_years_range"``; the output carries
    ``combine`` either way.

    Every sweep in *sweeps* must have been computed with the SAME fixed
    arguments (only the underlying exam_data/cancer_events differs per
    year) -- so config-only fields (``levels``, ``per_n_scans``,
    ``modality_filter``, ``deterministic_baseline``, ``QALY_mode``, ...)
    are copied from the first sweep unchanged; only the per-cell numeric
    content (``per_region_age``, ``per_region``, ``cohort_totals``) is
    combined.  Returns ``{}`` for an empty input, or the sole sweep
    unchanged for a single-year batch.
    """
    sweeps = [s for s in (sweeps or []) if s]
    if not sweeps:
        return {}
    if len(sweeps) == 1:
        return sweeps[0]
    n_years = len(sweeps)
    first = sweeps[0]
    interval_kind = first.get("interval_kind", "mc_ci_of_mean")
    pooled = combine == "pooled"
    out_kind = interval_kind
    if pooled and interval_kind in ANALYTICAL_INTERVAL_KINDS:
        out_kind = "analytical_pooled_years_range"

    regions: List[Any] = []
    _seen_regions = set()
    for s in sweeps:
        for r in (s.get("regions") or []):
            key = tuple(r) if isinstance(r, (tuple, list)) else r
            if key not in _seen_regions:
                _seen_regions.add(key)
                regions.append(r)
    regions.sort(key=lambda r: (r[0], r[1]) if isinstance(r, (tuple, list)) else (str(r),))

    modality_region_map: Dict[str, set] = {}
    for s in sweeps:
        for mod, regs in (s.get("modality_region_map") or {}).items():
            modality_region_map.setdefault(mod, set()).update(regs)
    modality_region_map_out = {m: sorted(rs) for m, rs in modality_region_map.items()}

    min_scans = float(first.get("min_scans") or 0.0)
    per_region_age = _average_cell_list(
        [s.get("per_region_age") or [] for s in sweeps],
        ("modality", "region", "age_group"), n_years, interval_kind, combine,
        min_scans,
    )
    per_region = _average_cell_list(
        [s.get("per_region") or [] for s in sweeps],
        ("modality", "region"), n_years, interval_kind, combine,
        min_scans,
    )
    per_region.sort(key=lambda r: r.get("baseline_QALY_per_100k", 0.0), reverse=True)

    cohort_totals = _average_dict_of_arrays(
        [s.get("cohort_totals") or {} for s in sweeps], n_years,
        weights=[float((s.get("cohort_totals") or {}).get("total_scans_per_year") or 0.0)
                 for s in sweeps] if pooled else None,
    )

    least_scans_cell = None
    if per_region_age:
        _least = min(
            per_region_age,
            key=lambda r: (r.get("scans_per_year", 0.0), r.get("modality", ""),
                          r.get("region", ""), r.get("age_decade_idx", 0)),
        )
        least_scans_cell = {
            "modality": _least["modality"], "region": _least["region"],
            "age_group": _least["age_group"],
            "age_decade_idx": int(_least["age_decade_idx"]),
            "scans_per_year": float(_least["scans_per_year"]),
            "mean_dose_msv": float(_least.get("mean_dose_msv", 0.0)),
            "baseline_cases_per_100k": float(_least.get("baseline_cases_per_100k", 0.0)),
            "baseline_deaths_per_100k": float(_least.get("baseline_deaths_per_100k", 0.0)),
            "baseline_QALY_per_100k": float(_least.get("baseline_QALY_per_100k", 0.0)),
            "is_reliable": bool(_least.get("is_reliable", True)),
        }

    mean_iterations = sum(int(s.get("iterations", 0) or 0) for s in sweeps) / n_years

    return {
        "levels": first.get("levels"),
        "age_groups": first.get("age_groups"),
        "regions": regions,
        "per_n_scans": first.get("per_n_scans"),
        "iterations": int(round(mean_iterations)),
        "min_scans": first.get("min_scans"),
        "min_expected_events": first.get("min_expected_events"),
        "modality_filter": first.get("modality_filter"),
        "modality_region_map": modality_region_map_out,
        "least_scans_cell": least_scans_cell,
        "deterministic_baseline": first.get("deterministic_baseline", False),
        "interval_kind": out_kind,
        "combine": combine,
        "QALY_mode": first.get("QALY_mode"),
        "QALY_discount_rate": first.get("QALY_discount_rate"),
        "per_region_age": per_region_age,
        "per_region": per_region,
        "cohort_totals": cohort_totals,
        # Marks this as a cross-year average rather than a single sweep, and
        # records the batch size -- consumers (gui.py) use this to label the
        # tab and to gate features (modality filter, deterministic baseline)
        # that would need per-year exam_data to honour correctly.
        "_averaged_across_years": n_years,
    }


def pool_region_age_sweeps(sweeps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """All year groups pooled as ONE exam-weighted cohort, annualised by
    dividing by the number of year groups -- the Research Questions tab's
    cross-year combination.  See :func:`average_region_age_sweeps`."""
    return average_region_age_sweeps(sweeps, combine="pooled")


def filter_region_age_sweep(
    sweep: Dict[str, Any],
    modality: Union[str, Iterable[str], None],
) -> Dict[str, Any]:
    """Restrict an already-computed sweep to one or more modalities.

    :func:`compute_region_age_dose_sweep` normally applies a modality
    filter to ``exam_data`` *before* it computes anything.  That is not an
    option for an "average across years" batch, whose per-year exam
    payloads are long gone by the time the user touches the modality
    picker -- but it is also not necessary: every ``per_region_age`` /
    ``per_region`` entry is already keyed by modality and was computed
    from that cell's exams alone, so selecting a subset of cells gives
    bit-for-bit the values a pre-filtered run would have produced.  Only
    the four cohort-level fields that aggregate ACROSS cells --
    ``cohort_totals``, ``least_scans_cell``, ``regions`` and
    ``modality_region_map`` -- have to be rebuilt, which this does with
    the same formulas the full computation uses.

    Cells carry their *grouping* modality (``_region_age_baseline``
    rewrites dental imaging out of Radiography into the "Dental imaging"
    pseudo-modality; "Dental CBCT" keeps its raw modality like every other
    region), so matching against it agrees with the pre-filter path.

    ``modality`` may be a single code, an iterable of codes, or None/empty
    for "all modalities" (in which case *sweep* is returned unchanged).
    """
    if not modality:
        return sweep
    if isinstance(modality, str):
        mod_keys = {_norm_modality(modality)}
    else:
        mod_keys = {_norm_modality(m) for m in modality if m}
    mod_keys.discard("")
    if not mod_keys or not sweep:
        return sweep

    def _keep(cell: Dict[str, Any]) -> bool:
        return _norm_modality(cell.get("modality")) in mod_keys

    per_region_age = [c for c in (sweep.get("per_region_age") or []) if _keep(c)]
    per_region = [c for c in (sweep.get("per_region") or []) if _keep(c)]

    out = dict(sweep)
    out["per_region_age"] = per_region_age
    out["per_region"] = per_region
    out["modality_filter"] = frozenset(mod_keys)
    out["regions"] = sorted({(c["modality"], c["region"]) for c in per_region_age})
    _map: Dict[str, set] = {}
    for c in per_region_age:
        _map.setdefault(str(c["modality"]), set()).add(str(c["region"]))
    out["modality_region_map"] = {m: sorted(rs) for m, rs in _map.items()}

    # Cohort totals: re-derive each cell's absolute (un-normalised) annual
    # contribution -- ``value_per_100k * scans / per_n_scans`` inverts the
    # ``scale = per_n_scans / scans`` applied per cell -- sum those, then
    # re-normalise against the SELECTED cells' own total scans, exactly as
    # the full computation does against the whole cohort's.
    per_n = float(sweep.get("per_n_scans") or 0.0)
    levels = np.asarray(sweep.get("levels") or (), dtype=float)
    cohort_scans = float(sum(float(c.get("scans_per_year", 0.0) or 0.0)
                             for c in per_region_age))
    _metrics = ("cases_avoided_per_100k", "deaths_avoided_per_100k",
                "non_mortal_avoided_per_100k", "QALY_saved_per_100k")
    totals: Dict[str, Any] = {"total_scans_per_year": cohort_scans}
    cohort_scale = (per_n / cohort_scans) if cohort_scans > 0 else 0.0
    for key in _metrics:
        acc = np.zeros_like(levels)
        for c in per_region_age:
            v = c.get(key)
            if v is None:
                continue
            acc = acc + np.asarray(v, dtype=float) * (
                float(c.get("scans_per_year", 0.0) or 0.0) / per_n
                if per_n > 0 else 0.0)
        totals[key] = acc * cohort_scale
    out["cohort_totals"] = totals

    out["least_scans_cell"] = None
    if per_region_age:
        _least = min(
            per_region_age,
            key=lambda r: (r.get("scans_per_year", 0.0), r.get("modality", ""),
                           r.get("region", ""), r.get("age_decade_idx", 0)),
        )
        out["least_scans_cell"] = {
            "modality": _least["modality"], "region": _least["region"],
            "age_group": _least["age_group"],
            "age_decade_idx": int(_least["age_decade_idx"]),
            "scans_per_year": float(_least["scans_per_year"]),
            "mean_dose_msv": float(_least.get("mean_dose_msv", 0.0)),
            "baseline_cases_per_100k": float(_least.get("baseline_cases_per_100k", 0.0)),
            "baseline_deaths_per_100k": float(_least.get("baseline_deaths_per_100k", 0.0)),
            "baseline_QALY_per_100k": float(_least.get("baseline_QALY_per_100k", 0.0)),
            "is_reliable": bool(_least.get("is_reliable", True)),
        }
    return out


def rescale_region_age_sweep(sweep: Dict[str, Any], new_per_n_scans: int) -> Dict[str, Any]:
    """Return a copy of *sweep* re-normalised to a different ``per_n_scans``.

    Every ``..._per_100k``-suffixed field is actually "per
    ``sweep['per_n_scans']`` examinations" (the literal "100k" in the name
    is just the conventional default) -- a pure linear rescale, since
    ``scale = per_n_scans / scans`` in :func:`compute_region_age_dose_sweep`
    and Var(cX) = c**2 Var(X) scales the same way.  Used so the "Per: N
    examinations" dropdown keeps working on a precomputed
    :func:`average_region_age_sweeps` result, whose per-year exam_data is
    not available to recompute a fresh sweep at a different
    normaliser.
    """
    old_n = float(sweep.get("per_n_scans") or 0)
    if old_n <= 0 or int(new_per_n_scans) == int(old_n):
        return sweep
    ratio = float(new_per_n_scans) / old_n

    def _scale_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v * ratio if "_per_100k" in k else v) for k, v in d.items()}

    out = dict(sweep)
    out["per_n_scans"] = int(new_per_n_scans)
    out["cohort_totals"] = _scale_dict(sweep.get("cohort_totals") or {})
    out["per_region_age"] = [_scale_dict(c) for c in (sweep.get("per_region_age") or [])]
    out["per_region"] = [_scale_dict(c) for c in (sweep.get("per_region") or [])]
    if sweep.get("least_scans_cell"):
        out["least_scans_cell"] = _scale_dict(sweep["least_scans_cell"])
    return out



def _modality_phrase(mod_filt: Any) -> str:
    """Build a human-readable modality phrase from a ``modality_filter``
    value that may be ``None``, a single canonical key string, or a
    ``frozenset`` of canonical keys.

    Returns a string like ``" CT"`` or ``" CT + Mammography"`` (with a
    leading space), or ``""`` when no filter is set.
    """
    if not mod_filt:
        return ""
    from .analysis import modality_name as _mod_name
    if isinstance(mod_filt, frozenset):
        keys = sorted(mod_filt)
    elif isinstance(mod_filt, (set, list, tuple)):
        keys = sorted(mod_filt)
    else:
        keys = [mod_filt]
    labels = []
    for k in keys:
        _mn = _mod_name(k)
        label = _mn.split(" (")[0] if " (" in _mn else _mn
        labels.append(label)
    if not labels:
        return ""
    return " " + " + ".join(labels)


_HOURS_PER_YEAR = 365.25 * 24.0

# Cell-annotation text style for plot_region_age_sweep:
#   "adaptive"      -- per-cell black/white text by background intensity
#   "dark_halo"     -- uniform near-black text with a white stroke on every cell
#   "light_outline" -- uniform white text with a dark stroke on every cell
# Default is "light_outline": one text style for every cell, so readers never
# take the text colour for a legend entry (the adaptive black/white switch
# was read as carrying meaning).
HEATMAP_TEXT_STYLE: str = "light_outline"


def _fmt_heatmap_value(v: float) -> str:
    """Format a heatmap dose cell value.

    The number of decimals shown depends on the integer part -- two
    decimals below one, one below ten, none at ten or more:

        \\|v\\| < 0.1         -> "<0.1"
        0.1 <= \\|v\\| < 1    -> two decimals ("0.34", "0.20")
        1 <= \\|v\\| < 10     -> one decimal ("2.0", "5.3")
        \\|v\\| >= 10         -> whole number ("709", "1204")
    """
    av = abs(v)
    if av < 0.1:
        return "<0.1"
    if av >= 10.0:
        return f"{v:.0f}"
    if av >= 1.0:
        r = float(f"{v:.1f}")
        return f"{r:.0f}" if abs(r) >= 10.0 else f"{r:.1f}"
    return f"{v:.2f}"


def _fmt_heatmap_hours(v: float) -> str:
    """Format a heatmap QALY cell value (hours per exam or QALYs per N exams).

    The number of decimals shown depends on the integer part -- two
    decimals below one ("0.50", "0.03"), one decimal below ten ("1.6"),
    whole numbers at ten and above ("47", "347").  Values below 0.01
    are shown as ``"<0.01"``.
    """
    av = abs(v)
    if av < 0.01:
        return "<0.01"
    if av >= 10.0:
        return f"{v:.0f}"
    if av >= 1.0:
        r = float(f"{v:.1f}")
        return f"{r:.0f}" if abs(r) >= 10.0 else f"{r:.1f}"
    return f"{v:.2f}"


_MODALITY_ORDER = [
    "CT", "Radiography", "MG", "CBCT", "Dental imaging",
    "Angiography", "NM", "MR", "US",
]


def _mod_sort_key(mod_code: str) -> tuple:
    mod_k = _norm_modality(mod_code)
    for i, prefix in enumerate(_MODALITY_ORDER):
        if mod_k == _norm_modality(prefix) or mod_code == prefix:
            return (i, mod_code)
    return (len(_MODALITY_ORDER), mod_code)


@_rq_fonts
def plot_region_age_sweep(
    fig,
    sweep: Dict[str, Any],
    region_filter: Optional[Union[str, List[str]]] = None,
    value_mode: str = "per_n",
    hide_unmodelled: bool = True,
    dental_cbct_in_cbct: bool = True,
    hide_small_cells: bool = True,
) -> None:
    """Single grouped heatmap with modality labels in the left margin.

    Three default display transforms apply to the heatmap only (exports,
    summaries and the sweep data itself are unchanged):

    * ``hide_unmodelled=True`` — in the default (unfiltered) view, rows
      whose regions have no risk/dose model (``has_risk_model=False``,
      e.g. non-dental CBCT body parts that would render as all-"n/a") are
      dropped entirely, along with regions that have no per-age cell.
      An explicit ``region_filter`` always shows what was selected.
    * ``dental_cbct_in_cbct=True`` — "Dental CBCT" exams coded under other
      raw modalities (Radiography, PX, ...) are pooled into the CBCT group
      row (exam-weighted merge per age cell) instead of appearing as a
      second "Dental CBCT" row inside another modality's group.
    * ``hide_small_cells=True`` — cells whose row fails the exam-count
      floor (``scans_ok=False``, i.e. fewer than
      ``sweep["min_scans"]`` exams in total across the pooled year
      groups) are rendered exactly like
      no-data cells (grey, italic "n/a") and excluded from the shared
      colour scale, since their per-exam estimates rest on too few exams
      to be reliable.  ``False`` draws them with values.


    All regions appear in one shared heatmap with one shared colour
    scale.  ``value_mode`` selects the cell unit:

    * ``"per_n"`` (default) — expected QALYs lost per ``per_n``
      examinations (the ``baseline_QALY_per_100k`` field is already "per
      ``per_n_scans`` exams").
    * ``"hours_per_exam"`` — hours of QALY lost per single examination
      (the per-``per_n``-exams field re-expressed as hours per exam).

    Any other ``value_mode`` is treated as ``"per_n"``.  Modality names
    are placed as bold text in a separate left-side column, vertically
    centred on each group.  Horizontal white separator lines are drawn
    between modality groups.  No header rows are inserted -- the y-axis
    contains only region labels.

    When no modality information is available, renders as a simple
    heatmap without group labels.
    """
    import matplotlib.transforms as mtransforms
    import matplotlib.patheffects as mpatheffects
    from matplotlib.colors import PowerNorm

    fig.clear()
    fig.set_layout_engine(None)
    ages = list((sweep or {}).get("age_groups") or ())
    rows = (sweep or {}).get("per_region_age") or []
    per_region = (sweep or {}).get("per_region") or []
    per_n = int((sweep or {}).get("per_n_scans") or 100_000)
    if value_mode != "hours_per_exam":
        value_mode = "per_n"

    if not rows or not ages:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No simulation data yet", transform=ax.transAxes,
                ha="center", va="center", color="#888")
        ax.set_axis_off()
        return

    # Filter regions. "Unknown/unspecified region" (CT rows with blank/
    # unrecognised Scan Regions) is excluded here -- it has no anatomical
    # identity to plot against age groups -- but is intentionally still
    # included in the annual burden/benefit-by-region estimates
    # (compute_annual_burden / compute_annual_benefit), which sum over age
    # instead of plotting it.  That only applies to CT, whose organ doses
    # come from the Excel columns; an unlocalised radiography/CBCT row has
    # no dose model and drops out everywhere via has_risk_model=False.  "Other combinations" (rare multi-region
    # combos folded at the reporting layer) is also excluded from the
    # heatmap -- it is not a single anatomical region and clutters the
    # grid -- but is kept everywhere else (annual burden, benefit-by-
    # region, CSV export, text summaries).
    mod_regions_all = [(r.get("modality", ""), r["region"]) for r in per_region
                       if r["region"] != _CT_UNLOCALIZED
                       and r["region"] != _CT_OTHER_COMBINATIONS
                       and not r["region"].startswith("Non-Dose")]
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = [(("", region_filter))]
        elif isinstance(region_filter, list) and region_filter:
            region_filter = [
                (("", s)) if isinstance(s, str) else (str(s[0]), str(s[1]))
                for s in region_filter
            ]
        filter_set = set(region_filter)
        mod_regions = [(mod, reg) for (mod, reg) in mod_regions_all
                       if (mod, reg) in filter_set]
    else:
        mod_regions = mod_regions_all
    if not mod_regions:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No matching region", transform=ax.transAxes,
                ha="center", va="center", color="#888")
        ax.set_axis_off()
        return

    # Build modality groups
    modality_region_map = dict(
        (k, list(v)) for k, v in ((sweep or {}).get("modality_region_map") or {}).items())
    # (2) Re-home every "Dental CBCT" cell to the CBCT modality key and
    # pool duplicates per age cell (exam-weighted on scans_per_year), so a
    # region coded under several raw modalities shows one merged row in
    # the CBCT group instead of a row per raw modality.
    if dental_cbct_in_cbct:
        _cbct_key = next(
            (k for k in (list(modality_region_map)
                         + [str(r.get("modality", "")) for r in rows]
                         + [m for m, _ in mod_regions])
             if _norm_modality(k) == "CBCT"), "CBCT")

        def _pool_dental_cells(cells):
            s = [float(c.get("scans_per_year", 0.0) or 0.0) for c in cells]
            tot = sum(s)
            m = dict(cells[0])
            for key in ("baseline_QALY_per_100k", "baseline_cases_per_100k",
                        "baseline_deaths_per_100k", "mean_dose_msv"):
                vs = [float(c.get(key, 0.0) or 0.0) for c in cells]
                m[key] = (sum(v * w for v, w in zip(vs, s)) / tot
                          if tot > 0 else sum(vs) / len(vs))
            m["scans_per_year"] = tot
            m["modelled_scans_per_year"] = sum(
                float(c.get("modelled_scans_per_year", 0.0) or 0.0) for c in cells)
            m["ci_degenerate"] = any(c.get("ci_degenerate") for c in cells)
            m["has_risk_model"] = True
            # The exam-count floor applies to the merged cell's TOTAL
            # exams over the pooled year groups: ``tot`` is annualised
            # (Σ s_y / n_years) when the sweep is a cross-year pool, so
            # scale it back by the number of pooled years.
            n_yr = int((sweep or {}).get("_averaged_across_years") or 1)
            m["scans_ok"] = tot * n_yr >= float(
                (sweep or {}).get("min_scans", DEFAULT_MIN_SCANS) or 0.0)
            m["is_reliable"] = bool(
                m["scans_ok"]
                and all(c.get("events_ok", True) for c in cells)
                and not m["ci_degenerate"])
            m["modality"] = _cbct_key
            return m

        _d_rows = [dict(r, modality=_cbct_key)
                   for r in rows if r["region"] == "Dental CBCT"]
        if _d_rows:
            _by_age: Dict[int, list] = {}
            for r in _d_rows:
                _by_age.setdefault(int(r.get("age_decade_idx", 0)), []).append(r)
            rows = ([r for r in rows if r["region"] != "Dental CBCT"]
                    + [_pool_dental_cells(g) for g in _by_age.values()])
        _d_pr = [dict(r, modality=_cbct_key)
                 for r in per_region if r["region"] == "Dental CBCT"]
        if _d_pr:
            per_region = ([r for r in per_region if r["region"] != "Dental CBCT"]
                          + [_pool_dental_cells(_d_pr)])
        mod_regions = list(dict.fromkeys(
            (_cbct_key if reg == "Dental CBCT" else mod, reg)
            for mod, reg in mod_regions))
        if modality_region_map:
            modality_region_map = {
                k: [r for r in v if r != "Dental CBCT"]
                for k, v in modality_region_map.items()}
            modality_region_map = {
                k: v for k, v in modality_region_map.items() if v}
            modality_region_map.setdefault(_cbct_key, [])
            if "Dental CBCT" not in modality_region_map[_cbct_key]:
                modality_region_map[_cbct_key].append("Dental CBCT")
    # (1) Drop unmodelled/no-cell regions from the default view.
    if hide_unmodelled and not region_filter:
        _modelled = {(str(r.get("modality", "")), r["region"]) for r in rows
                     if r.get("has_risk_model", True) is not False}
        _keep = {(m, rg) for (m, rg) in mod_regions if (m, rg) in _modelled}
        mod_regions = [p for p in mod_regions if p in _keep]
        if not mod_regions:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No matching region", transform=ax.transAxes,
                    ha="center", va="center", color="#888")
            ax.set_axis_off()
            return
        rows = [r for r in rows
                if (str(r.get("modality", "")), r["region"]) in _keep]
        per_region = [r for r in per_region
                      if (str(r.get("modality", "")), r["region"]) in _keep]

    # Build groups from per_region's modality field (handles duplicate
    # regions across modalities correctly).
    modality_groups: List[Tuple[str, List[str]]] = []
    if modality_region_map:
        region_set = {reg for (_, reg) in mod_regions}
        for mod_code in sorted(modality_region_map.keys(), key=_mod_sort_key):
            # Match on region name only (already filtered via region_set).
            # modality_region_map is the authoritative logical grouping and
            # may reassign a region to a different modality key than its
            # per-row modality field (e.g. dental-imaging subtypes are moved
            # from "Radiography" into the "Dental imaging" group), so
            # requiring an exact (mod_code, reg) match against the raw field
            # would drop such regions into a spurious "All modalities"
            # leftover group.
            mod_regs = [reg for reg in region_set
                        if reg in modality_region_map.get(mod_code, [])]
            if mod_regs:
                modality_groups.append((mod_code, mod_regs))
        covered = {r for _, rl in modality_groups for r in rl}
        leftover = [reg for (_, reg) in mod_regions if reg not in covered]
        if leftover:
            modality_groups.append(("", leftover))
    else:
        # No modality map: group by the modality field in per_region.
        _by_mod: Dict[str, List[str]] = {}
        for mod, reg in mod_regions:
            _by_mod.setdefault(mod, []).append(reg)
        for mod_code in sorted(_by_mod.keys(), key=_mod_sort_key):
            modality_groups.append((mod_code, _by_mod[mod_code]))

    _QALY_KEY = "baseline_QALY_per_100k"

    # Sort regions within each modality group by descending total QALY
    # lost, computed as the literal sum across all age-group cells
    # shown in the heatmap for that region (i.e. what a reader would
    # get by adding the per-exam-hours values across the row) -- summed
    # from ``rows`` (per_region_age), the exact same source used to
    # populate M_q below, so the ordering matches the printed table.
    # (per_region's own baseline_QALY_per_100k is a separate,
    # scan-weighted whole-population statistic and does not equal this
    # row sum, so it must not be used for this ordering.)  The per-100k
    # field is used here as the sort key only because the per-exam-hours
    # transform is a positive monotonic scaling of it, so the ordering
    # is unchanged.
    _region_total: Dict[str, float] = {}
    for r in rows:
        reg = r["region"]
        val = float(r.get(_QALY_KEY, 0.0) or 0.0)
        _region_total[reg] = _region_total.get(reg, 0.0) + val
    for gi, (mod_code, mod_regs) in enumerate(modality_groups):
        modality_groups[gi] = (mod_code, sorted(
            mod_regs, key=lambda reg: -_region_total.get(reg, 0.0)))

    # Build flat region list with group bounds
    regions_ordered: List[str] = []
    regions_mod_ordered: List[str] = []  # parallel mod_code per region
    group_bounds: List[Tuple[int, int, str, str]] = []  # (start_idx, end_idx, mod_label, mod_code)
    di = 0
    for gi, (mod_code, mod_regs) in enumerate(modality_groups):
        mod_label = _modality_name(mod_code) if mod_code else "All modalities"
        mod_label = mod_label.split(" (")[0] if " (" in mod_label else mod_label
        if mod_label == "MG":
            mod_label = "Mammography"
        # NB: "CBCT" is deliberately left as-is (not renamed to "Dental
        # CBCT") -- the group can now contain literal non-dental CBCT
        # regions (Ankle, Wrist, ...) alongside "Dental CBCT", so a
        # dental-only header would be misleading.
        start_di = di
        for reg in mod_regs:
            regions_ordered.append(reg)
            regions_mod_ordered.append(mod_code)
            di += 1
        group_bounds.append((start_di, di - 1, mod_label, mod_code))

    n_reg_data = len(regions_ordered)

    # Build per-row data arrays (no gap rows).
    M_q = np.full((n_reg_data, len(ages)), np.nan, dtype=float)
    M_d = np.full((n_reg_data, len(ages)), np.nan, dtype=float)
    has_data = np.zeros((n_reg_data, len(ages)), dtype=bool)
    # Cells whose Wald CI collapsed to a point (``ci_degenerate``): the point
    # estimate is still drawn, but marked with a dagger so the figure agrees
    # with the text/CSV exports, which print "n/a" for the interval.
    is_degen = np.zeros((n_reg_data, len(ages)), dtype=bool)

    # Map (modality, region) -> data row index.
    mod_reg_to_di = {}
    di = 0
    for gi, (mod_code, mod_regs) in enumerate(modality_groups):
        for reg in mod_regs:
            mod_reg_to_di[(mod_code, reg)] = di
            di += 1

    for r in rows:
        mi = mod_reg_to_di.get((r.get("modality", ""), r["region"]))
        if mi is None:
            continue
        j = int(r.get("age_decade_idx", 0))
        if j < 0 or j >= len(ages):
            continue
        # Cells below the exam-count floor render as no-data (grey "n/a")
        # and stay out of the shared colour scale -- has_data stays False
        # and M_q/M_d stay NaN, the same state the no-model clearing below
        # produces.  Done before global_vmax so blanked cells do not
        # drive the colour scale.
        if hide_small_cells and not r.get("scans_ok", True):
            continue
        # baseline_QALY_per_100k is actually "per per_n_scans exams"
        # (see ``scale = per_n_scans / scans`` in
        # compute_region_age_dose_sweep).  ``per_n`` mode keeps that
        # field as-is (QALYs lost per ``per_n`` examinations);
        # ``hours_per_exam`` mode re-expresses it as hours of QALY lost
        # per single examination.
        _q = float(r.get(_QALY_KEY, 0.0) or 0.0)
        if value_mode == "hours_per_exam":
            M_q[mi, j] = _q / per_n * _HOURS_PER_YEAR
        else:
            M_q[mi, j] = _q
        M_d[mi, j] = float(r.get("mean_dose_msv", 0.0) or 0.0)
        has_data[mi, j] = True
        is_degen[mi, j] = bool(r.get("ci_degenerate", False))

    # Regions with no risk/dose model (e.g. non-dental CBCT body parts such
    # as Elbow/Ankle/Wrist, which get organ_doses_gy=None) are shown as n/a
    # across every age column: clear their data so the existing has_data=False
    # path renders a grey cell with italic "n/a" text.  Match on the
    # (mod_code, region) tuple from mod_reg_to_di; the leftover ("", reg)
    # group matches on region name alone since rows carry the real modality.
    no_model = {(str(r.get("modality", "")), r["region"])
                for r in rows if r.get("has_risk_model", True) is False}
    _no_model_regions = {reg for (_, reg) in no_model}
    for (mod_code, reg), di in mod_reg_to_di.items():
        if (mod_code, reg) in no_model or (mod_code == "" and reg in _no_model_regions):
            M_q[di, :] = np.nan
            M_d[di, :] = np.nan
            has_data[di, :] = False

    # Single shared vmax.
    global_vmax = float(np.nanmax(M_q)) if M_q.size > 0 else 1e-9
    if global_vmax < 1e-9:
        global_vmax = 1e-9

    def _display_region(reg: str, mod_code: str = "") -> str:
        """Shorten verbose region names for the y-axis (display only), and
        wrap long ones onto a second line rather than letting them run into
        the left margin or get squeezed against the plot. Dental-imaging
        subtypes get their own short label (Panoramic, CBCT, Kefalometry,
        Intraoral) instead of all collapsing to one generic "Dental" --
        they are separate rows within the "Dental imaging" group now, so
        collapsing them to the same label would make the rows indistinguishable.
        "Dental CBCT" is shown as "Dental" inside the CBCT group (raw-CBCT
        dental exams, beside non-dental CBCT regions); under any other
        modality it keeps "Dental CBCT"."""
        if _norm_modality(mod_code) == "CBCT" and reg == "Dental CBCT":
            reg = "Dental"
        elif reg in _DENTAL_IMAGING_SUBTYPE_REGIONS:
            reg = _DENTAL_SUBTYPE_DISPLAY.get(reg, reg)
        reg = _REGION_DISPLAY_SHORT.get(reg, reg)
        if len(reg) > 16:
            reg = "\n".join(textwrap.wrap(reg, width=16, break_long_words=False))
        return reg

    # Compute custom y-coordinates with real gaps
    # Each data row is 1.0 unit tall.  Between modality groups, a
    # fractional gap (_GAP_FRAC * 1.0) is added as pure axes background.
    _GAP_FRAC = 0.7  # gap between modality groups, in row-height units
    y_edges = []      # bottom edge of each row
    y_centers = []    # center of each row
    group_y_bounds = []  # (y_top, y_bottom, mod_label, mod_code) for labels
    cy = 0.0
    for gi, (start_di, end_di, mod_label, mod_code) in enumerate(group_bounds):
        group_start_y = cy
        for ri in range(start_di, end_di + 1):
            y_edges.append(cy)
            y_centers.append(cy + 0.5)
            cy += 1.0
        group_y_bounds.append((group_start_y, cy, mod_label, mod_code))
        if gi < len(group_bounds) - 1:
            cy += _GAP_FRAC  # real gap between groups

    total_y = cy
    n_ages = len(ages)

    # Plot: single unified heatmap via per-group pcolormesh calls
    # A single combined y-edge array would stretch the last row of each
    # group to absorb the gap (since pcolormesh shares edges between
    # adjacent cells).  To get a genuinely blank gap with every row at
    # exactly the same height, each modality group is drawn with its own
    # pcolormesh call (same axes, same cmap/norm) positioned at its true
    # y-offset; nothing is drawn in the gap itself, leaving it blank.
    ax = fig.add_subplot(111)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#d9d9d9")  # true no-data cells get grey
    _GAMMA = 0.5
    norm = PowerNorm(gamma=_GAMMA, vmin=0.0, vmax=global_vmax)
    x_edge_arr = np.arange(n_ages + 1) - 0.5

    im = None  # reference for colour bar
    for start_di, end_di, mod_label, mod_code in group_bounds:
        group_y_top = y_edges[start_di]
        n_rows_g = end_di - start_di + 1
        group_y_edge_arr = group_y_top + np.arange(n_rows_g + 1, dtype=float)
        M_q_group = M_q[start_di:end_di + 1, :]
        M_q_group_masked = np.ma.array(M_q_group, mask=np.isnan(M_q_group))
        im_g = ax.pcolormesh(x_edge_arr, group_y_edge_arr, M_q_group_masked,
                             cmap=cmap, norm=norm, edgecolors="#404040", linewidth=0.5)
        if im is None:
            im = im_g

    # Thin black separator lines in the centre of each white gap.  Kept
    # visually distinct (pure black, slightly thicker) from the softer
    # dark-grey interior cell gridlines so the modality-group boundary
    # reads as a stronger structural cue than ordinary cell divisions.
    # Span only the heatmap columns (x from -0.5 to n_ages-0.5).  Each
    # line's y-position is kept so the modality header for the group
    # below can be aligned exactly to it (see below).
    sep_ys: List[float] = []
    for i in range(1, len(group_y_bounds)):
        prev_bottom = group_y_bounds[i - 1][1]
        curr_top = group_y_bounds[i][0]
        sep_y = (prev_bottom + curr_top) / 2.0
        sep_ys.append(sep_y)
        ax.plot([-0.5, n_ages - 0.5], [sep_y, sep_y],
                color="black", linewidth=1.0, zorder=3, clip_on=True)

    # A small pad above row 0 gives the *first* group's modality header
    # somewhere to sit -- every later group already has its own gap for
    # this, but row 0 starts right at the top edge with no blank band.
    _TOP_PAD = 0.55 if len(group_y_bounds) > 1 else 0.0
    ax.set_xlim(-0.5, n_ages - 0.5)
    ax.set_ylim(total_y, -_TOP_PAD)  # inverted: row 0 at top

    # Give the very first group a top separator line too, matching the
    # black lines that already sit above every later group -- otherwise
    # the first modality header looks visually inconsistent (floating,
    # with no line) compared to the rest.  Every later separator sits
    # in the *middle* of its blank gap (equal space above and below),
    # so this one is placed at the midpoint of the top pad too, rather
    # than flush against row 0 -- flush placement left no breathing
    # room below the line while leaving all the slack above it.
    if len(group_y_bounds) > 1:
        top_sep_y = -_TOP_PAD / 2.0
        ax.plot([-0.5, n_ages - 0.5], [top_sep_y, top_sep_y],
                color="black", linewidth=1.0, zorder=3, clip_on=False)

    # X-ticks.
    ax.set_xticks(np.arange(n_ages))
    ax.set_xticklabels(ages, rotation=0, fontsize=12, ha="center")
    ax.set_xlabel("Age group, years", fontsize=13, fontweight="bold")

    # Y-ticks at row centers.
    ax.set_yticks(y_centers)
    ax.set_yticklabels([_display_region(r, m)
                       for r, m in zip(regions_ordered, regions_mod_ordered)],
                      fontsize=12)

    # One overall light frame.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
        spine.set_linewidth(0.8)

    # Modality names are shown as a compact header sitting in the blank
    # gap directly above each group, right-aligned with the region tick
    # labels -- not as a dedicated extra column.  A reader scanning a
    # generic region name like "Breast" or "Chest" needs an explicit,
    # spatially-tied cue for which modality it belongs to; the title
    # alone lists every modality but isn't tied to specific rows.  Using
    # the gap's existing blank space (rather than reserving new width)
    # keeps cell width -- and the auto font-size tier below -- unaffected
    # by how many modalities happen to be selected.  Skipped only when
    # there's a single group, since the title qualifier already makes
    # the (sole) modality unambiguous for every row.
    mod_label_texts: List[Any] = []
    if len(group_y_bounds) > 1:
        for gi, (_, _, mod_label, _) in enumerate(group_y_bounds):
            # Every group's header sits vertically centred on its own
            # black separator line (the first group's line is
            # top_sep_y, drawn above) -- since the text lives outside
            # the plot area (x<0) while the line spans the data
            # columns (x>=-0.5), they occupy different x ranges and
            # never visually overlap.
            y_header = top_sep_y if gi == 0 else sep_ys[gi - 1]
            t = ax.text(-0.01, y_header, mod_label,
                        transform=mtransforms.blended_transform_factory(
                            ax.transAxes, ax.transData),
                        ha="right", va="center", fontsize=12.2,
                        fontweight="bold", color="#1a1a1a", clip_on=False)
            mod_label_texts.append(t)

    # Cell annotations: dose (grey, top) + QALY value (bold, bottom).  The
    # QALY value is in ``value_mode`` units -- QALYs lost per ``per_n``
    # examinations (``per_n``) or hours of QALY lost per exam
    # (``hours_per_exam``); ``_fmt_heatmap_hours`` buckets suit both.
    # Cells are sized to a fixed physical constant (_CELL_IN, used again
    # further below for the actual box sizing) -- every render has the
    # *same* cell dimensions regardless of how many modalities/regions
    # are selected, chosen to comfortably fit this annotation text; the
    # figure grows taller instead of squeezing cells for large
    # selections (see the fixed-size box section below), so the font
    # size here is simply a fixed constant too, sized against the
    # 0.8in cell: the widest routine dose string (e.g. "<0.1 mSv", ~8
    # chars) at 10.5pt is ~0.6in wide, and the bold QALY value line at
    # 13pt is well clear of both the dose line above it and the cell's
    # edges -- comfortably larger than before without crowding.
    _CELL_IN = 0.8  # fixed square cell size (inches) -- see box-sizing below
    _val_fs, _dose_fs, _na_fs = 15.6, 11.55, 10.5

    # A thin contrasting outline (halo) is added to every annotation so
    # text stays legible on any cell colour, not just the light/dark
    # extremes of the colormap -- mid-intensity orange cells otherwise
    # give grey dose text too little contrast to read.  Kept thin so it
    # reads as a subtle legibility aid rather than a heavy drop-shadow.
    _halo_light = [mpatheffects.withStroke(linewidth=1.4, foreground="#ffffff")]
    _halo_dark = [mpatheffects.withStroke(linewidth=1.0, foreground="#111111")]
    # Uniform-style variants selected by HEATMAP_TEXT_STYLE (see the module
    # constant near _HOURS_PER_YEAR): heavier strokes than the adaptive halo
    # because the same text colour must survive every background.
    _halo_dark_heavy = [mpatheffects.withStroke(linewidth=2.6, foreground="#ffffff")]
    _halo_dark_heavy_dose = [mpatheffects.withStroke(linewidth=1.9, foreground="#ffffff")]
    _halo_light_heavy = [mpatheffects.withStroke(linewidth=2.0, foreground="#222222")]
    _halo_light_heavy_dose = [mpatheffects.withStroke(linewidth=1.4, foreground="#222222")]

    # Adaptive text colour based on cell background brightness (three
    # tiers, thresholded on the same normalised intensity the colormap
    # uses): dark cells get plain white text (already high-contrast, no
    # outline needed); medium cells get white text with a subtle dark
    # outline (white alone starts losing contrast against mid-tone
    # orange); light cells get plain black text.
    _DARK_THRESH = 0.6
    _MED_THRESH = 0.3

    for ri in range(n_reg_data):
        yc = y_centers[ri]
        for c_i in range(n_ages):
            if not has_data[ri, c_i]:
                ax.text(c_i, yc, "n/a", ha="center", va="center",
                        fontsize=_na_fs, color="#999999", style="italic",
                        zorder=4)
                continue
            v = float(M_q[ri, c_i])
            dose_v = float(M_d[ri, c_i])
            qfmt = _fmt_heatmap_hours(v)
            if is_degen[ri, c_i]:
                qfmt += "\u2020"
            dfmt = f"{_fmt_heatmap_value(dose_v)} mSv"
            cell_intensity = (v / global_vmax) ** _GAMMA if global_vmax > 0 else 0.0
            if HEATMAP_TEXT_STYLE == "dark_halo":
                color = "#111111"
                dose_color = "#333333"
                val_halo = _halo_dark_heavy
                dose_halo = _halo_dark_heavy_dose
            elif HEATMAP_TEXT_STYLE == "light_outline":
                color = "#ffffff"
                dose_color = "#ffffff"
                val_halo = _halo_light_heavy
                dose_halo = _halo_light_heavy_dose
            elif cell_intensity > _DARK_THRESH:
                color = "#ffffff"
                dose_color = "#f0f0f0"
                dose_halo = None
                val_halo = None
            elif cell_intensity > _MED_THRESH:
                color = "#ffffff"
                dose_color = "#f5f5f5"
                dose_halo = _halo_dark
                val_halo = _halo_dark
            else:
                color = "#111111"
                dose_color = "#444444"
                dose_halo = None
                val_halo = None
            ax.text(c_i, yc + 0.28, dfmt,
                    ha="center", va="center",
                    fontsize=_dose_fs, color=dose_color, zorder=4,
                    path_effects=dose_halo)
            ax.text(c_i, yc - 0.22, qfmt,
                    ha="center", va="center", path_effects=val_halo,
                    fontsize=_val_fs, color=color, fontweight="bold", zorder=4)

    # Single shared colour bar spanning full heatmap height.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    if value_mode == "hours_per_exam":
        _cb_label = "Expected QALYs lost per examination (hours)"
    else:
        _cb_label = f"Expected QALYs lost per {per_n:,} examinations"
    cb.set_label(_cb_label,
                 fontsize=12.1, fontweight="bold")
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_linewidth(0.8)
    cb.outline.set_edgecolor("black")

    # Title text
    # A short modality qualifier keeps the figure self-explanatory even
    # if it's ever viewed without its caption (e.g. extracted into a
    # slide deck or repository preview) -- otherwise every modality
    # selection renders an identical title.  Placed on its own line
    # below the main title (rather than appended in parentheses) so it
    # reads as a clear subtitle instead of crowding the title line.
    _mod_names = []
    for _, _, _mlabel, _ in group_bounds:
        if _mlabel and _mlabel not in _mod_names:
            _mod_names.append(_mlabel)
    if len(_mod_names) == 1:
        _mod_line = _mod_names[0]
    elif 1 < len(_mod_names) <= 3:
        _mod_line = f"{', '.join(_mod_names[:-1])}, and {_mod_names[-1]}"
    else:
        _mod_line = ""
    if value_mode == "hours_per_exam":
        title = "Expected QALYs Lost per Examination by Imaging Region and Age Group"
    else:
        title = f"Expected QALYs Lost per {per_n:,} Examinations by Imaging Region and Age Group"

    # Fixed-size cell box
    # Box dimensions come from the fixed physical _CELL_IN constant
    # (set up above, near the font-tiering logic), so every render has
    # genuinely constant cell dimensions -- independent of how many
    # rows/columns are selected or how wide the region/modality labels
    # happen to be.  Rather than squeezing cells to fit a fixed-height
    # canvas (which was distorting/shrinking dimensions for larger
    # modality selections), the *figure itself* grows taller to fit;
    # the GUI wraps this figure in a scrollable container so the full
    # height can still be viewed.  All margins below are fixed absolute
    # inches, not fractions -- so they stay correct regardless of the
    # figure's final height.
    _TITLE_FS = 15.525           # shared font size/colour/weight for both
    _TITLE_TOP_GAP_IN = 0.12    # gap from figure top edge to title text
    _TITLE_LINE_IN = 0.299      # rendered height of the 15.525pt bold title
    _SUBTITLE_GAP_IN = 0.06     # gap between title and modality subtitle
    _SUBTITLE_LINE_IN = 0.299   # rendered height of the subtitle (same size as title)
    # Extra clearance so the first group's modality header/separator
    # line (drawn just above row 0, see _TOP_PAD) doesn't crowd the
    # title block above it.
    _BOX_TOP_GAP_IN = 0.16 if len(group_y_bounds) > 1 else 0.08
    _BOTTOM_MARGIN_IN = 0.9     # x-tick labels + xlabel + padding

    top_margin_in = _TITLE_TOP_GAP_IN + _TITLE_LINE_IN + _BOX_TOP_GAP_IN
    if _mod_line:
        top_margin_in += _SUBTITLE_GAP_IN + _SUBTITLE_LINE_IN

    axes_width_in = _CELL_IN * n_ages
    axes_height_in = _CELL_IN * total_y
    fig_w_in = fig.get_figwidth()  # width is never changed, only height
    new_fig_h_in = top_margin_in + axes_height_in + _BOTTOM_MARGIN_IN
    fig.set_figheight(new_fig_h_in)

    # Left margin needs the labels' actual rendered width (which varies
    # with the region/modality names in this selection), measured via a
    # throwaway draw before the title is placed -- the title is centred
    # on the resulting box (below), so its horizontal position depends
    # on this margin too.
    fig.subplots_adjust(left=0.20, right=0.98, top=0.9, bottom=0.1)
    fig.canvas.draw()
    for _lbl in cb.ax.get_yticklabels():
        _lbl.set_fontweight("bold")
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig_w_in * fig.dpi

    tick_labels = ax.get_yticklabels()
    tick_w_px = max((t.get_window_extent(renderer=renderer).width
                      for t in tick_labels), default=0.0)
    mod_w_px = max((t.get_window_extent(renderer=renderer).width
                     for t in mod_label_texts), default=0.0)
    label_w_px = max(tick_w_px, mod_w_px)
    _GAP_PX = 10.0  # small fixed gap between figure edge and region labels
    required_left = (_GAP_PX + label_w_px + _GAP_PX) / fig_w_px
    required_left = min(max(required_left, 0.10), 0.40)

    box_width_frac = axes_width_in / fig_w_in
    box_height_frac = axes_height_in / new_fig_h_in
    box_bottom = _BOTTOM_MARGIN_IN / new_fig_h_in
    ax.set_position([required_left, box_bottom, box_width_frac, box_height_frac])

    # The colour-bar label runs along the bar's height, which is set by the
    # number of rows: a full modality selection leaves room for the label,
    # but a two-region filter does not, and a rotated label that overruns is
    # clipped at BOTH ends -- which reads as a deliberately short label
    # rather than as an error.  Shrink to fit against the rendered extents
    # (a fixed font size tuned on one selection fails on another), then wrap
    # onto two lines if even the floor size overruns.
    fig.canvas.draw()
    _cb_label = cb.ax.yaxis.label

    def _cb_label_overruns() -> bool:
        avail = cax.get_window_extent(renderer=renderer).height
        return _cb_label.get_window_extent(renderer=renderer).height > avail

    if _cb_label_overruns():
        for _fs in (11.0, 10.0, 9.0):
            _cb_label.set_fontsize(_fs)
            fig.canvas.draw()
            if not _cb_label_overruns():
                break
    if _cb_label_overruns():
        if value_mode == "hours_per_exam":
            _cb_label.set_text("Expected QALYs lost\nper examination (hours)")
        else:
            _cb_label.set_text(f"Expected QALYs lost\nper {per_n:,} examinations")
        fig.canvas.draw()

    # Title/subtitle are centred on the plotted box itself (region
    # labels + heatmap cells), not the full figure -- the box's left
    # margin (labels) and right margin (colorbar) aren't symmetric, so
    # centring on the whole figure width visibly offsets the title from
    # the content it describes.
    _title_center_x = required_left + box_width_frac / 2.0
    fig.suptitle(title, fontsize=_TITLE_FS, fontweight="bold",
                 x=_title_center_x, va="top",
                 y=1.0 - _TITLE_TOP_GAP_IN / new_fig_h_in)
    if _mod_line:
        # Same size/colour/weight as the main title -- a subtitle in a
        # visibly different style read as inconsistent/unintentional,
        # so it's styled identically, just on its own line below.
        _mod_y = 1.0 - (_TITLE_TOP_GAP_IN + _TITLE_LINE_IN
                        + _SUBTITLE_GAP_IN) / new_fig_h_in
        fig.text(_title_center_x, _mod_y, _mod_line, ha="center", va="top",
                 fontsize=_TITLE_FS, fontweight="bold")


# Annual-benefit analysis: expected QALYs / cancers / deaths saved per year
# from a 10% dose reduction, combining per-examination burden with local
# annual examination volumes.

def _per_100k_ci_to_annual(
    lo_per_100k: float, hi_per_100k: float, annual_factor: float,
) -> Tuple[float, float]:
    """Convert a per-N-exams Wald CI to an annual CI.

    Going from "per ``per_n`` examinations" back to "per year" multiplies the
    same random quantity by ``scans / per_n``, a deterministic constant, so
    the standard error and hence both endpoints scale by that identical
    factor.  Transforming the endpoints directly keeps the annual interval
    centred on the annual point estimate.
    """
    a = lo_per_100k * annual_factor
    b = hi_per_100k * annual_factor
    return min(a, b), max(a, b)


def _ci_var(lo: float, hi: float) -> float:
    """Variance implied by a Wald CI (half-width = CI_Z * SE)."""
    w = (hi - lo) / (2.0 * CI_Z)
    return w * w if w > 0.0 else 0.0


def _ci_from_var(point: float, var: float) -> Tuple[float, float]:
    """Build a Wald CI from accumulated point and variance.

    The interval is clamped at the reporting boundary of zero, because none
    of these quantities (life-years, cancers, deaths avoided) can cross it:
    a dose *reduction* cannot add harm and a dose *increase* cannot save
    anyone.  Which side to clamp therefore follows the sign of the point
    estimate -- clamping the lower bound unconditionally would invert the
    interval for a dose increase, whose values are negative by convention
    (e.g. point -14.2 gave [0, -14.2]).
    """
    std = float(np.sqrt(var)) if var > 0.0 else 0.0
    half = CI_Z * std
    if point < 0.0:
        return point - half, min(0.0, point + half)
    return max(0.0, point - half), point + half


# Central 95% range of simulated annual outcomes
#
# The interval the Research Questions tab reports is the same quantity the
# Summary tab shows as the "95% range of simulated years": the 2.5th/97.5th
# percentiles of the pooled per-iteration annual outcome series (one row =
# one simulated year; in per_year_average mode the pooled list already
# covers every processed year).  It is NOT a confidence interval of a mean
# — the width is the year-to-year spread of the simulated annual outcome
# itself, so it does not narrow with more iterations.
#
# ``attach_simulated_ranges`` stores the series needed to reproduce it on
# ``sweep["_sim"]`` and rewrites every per-cell ``*_ci_*_per_100k`` field to
# the percentile of that cell's own pooled per-iteration series (NaN → the
# renderer shows "n/a" when no series was recorded; there is no fallback to
# the analytical interval).  Per-year cell series are collected in
# ``year_batch`` while that year's events are still alive; in single /
# combined mode they are built here from the run's surviving event payload.

SIM_ANNUAL_RANGE_LABEL = (
    "Central 95% range of simulated annual outcomes "
    "(single year; all years pooled)"
)

# Display-only cell-interval modes for the Research Questions tab.  The tag
# lives on the sweep dict as ``sweep["cell_interval"]``; on ``_sim``-
# attached sweeps the text summary treats an absent tag
# as ``CELL_INTERVAL_COHORT`` and renders the analytical 95% range for a
# cohort of ``per_n_scans`` examinations
# (``baseline_<m>_cohort_lo/hi_per_100k``), while ``CELL_INTERVAL_SIMULATED``
# renders the central 95% range of simulated annual outcomes.  No
# ``*_ci_*_per_100k`` field and no other consumer is affected.
CELL_INTERVAL_SIMULATED = "simulated_annual"
CELL_INTERVAL_COHORT = "cohort_per_n"


def set_cell_interval(sweep: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Return a shallow copy of *sweep* tagged with cell-interval *mode*.

    The tag is display-only: it selects which interval
    :func:`format_region_age_sweep_summary` renders, and changes no data.
    """
    if mode not in (CELL_INTERVAL_SIMULATED, CELL_INTERVAL_COHORT):
        raise ValueError(
            f"unknown cell_interval mode {mode!r}; expected "
            f"{CELL_INTERVAL_SIMULATED!r} or {CELL_INTERVAL_COHORT!r}")
    out = dict(sweep)
    out["cell_interval"] = mode
    return out


# Regions the sweep drops before building cells / region_scan_totals —
# mirrors the ``_EXCLUDE_REGIONS`` local in compute_region_age_dose_sweep.
_SWEEP_EXCLUDE_REGIONS = frozenset({"Other radiography"})


def _sim_decade_index(ages) -> np.ndarray:
    """Age-decade index identical to the sweep's ``_decade_vec``."""
    ages = np.asarray(ages, dtype=float)
    if ages.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.clip(
        np.searchsorted(np.asarray(AGE_DECADE_EDGES, dtype=float), ages,
                        side="right") - 1,
        0, len(AGE_DECADE_LABELS) - 1).astype(np.int64)


def _exam_region_scan_totals(exam_data) -> Dict[str, float]:
    """Region -> scan counts, mirroring the sweep's ``region_scan_totals``
    (built from the same ``base`` bins, i.e. post region-resolution,
    ``_SWEEP_EXCLUDE_REGIONS`` excluded)."""
    totals: Dict[str, float] = {}
    if exam_data is None:
        return totals
    _cols = ColumnarExamsList.from_dicts(exam_data)._cols
    _regs = _cols.get("reporting_region", _cols.get("region")) or []
    _it = (str(r or "Unknown") for r in _regs)
    for r in _it:
        if r in _SWEEP_EXCLUDE_REGIONS:
            continue
        totals[r] = totals.get(r, 0.0) + 1.0
    return totals


def per_iteration_cell_series(events, iterations, exam_data=None):
    """Per-(modality, region, decade) simulated annual outcome series.

    Returns ``{cell_key: {"pos": int32[k], "cases": f32[k],
    "deaths": f32[k], "qaly": f32[k], "raw_mods": tuple}}`` — only
    (cell, iteration) pairs that saw at least one event.  ``pos`` is the
    iteration index inside *this* run (this year in a batch); densify with
    ``a[pos] += v``.  Counts stay float32 so the pooled store stays small;
    cases/deaths are integer-valued.

    Cell keys use the sweep's grouping: ``_event_arrays``'s "Dental imaging"
    pseudo-modality rewrite, reporting-region labels, and — when
    *exam_data* is given — the rare-combo fold into "Other combinations"
    driven by the same exam-count totals as the sweep's own
    ``region_scan_totals``.  Without *exam_data* the fold cannot be
    evaluated and combo cells keep their true region (single-run callers
    should always pass it).  The ONE deliberate departure: "Dental CBCT"
    cells are stored per RAW modality (``("Radiography", "Dental CBCT",
    di)`` and ``("CBCT", "Dental CBCT", di)``) although the sweep keys them
    all under "CBCT" -- ``_sim_cell_entries`` folds the raw variants back
    together for every per-cell lookup, and the raw split lets
    ``_sim_cohort_series`` correct a modality selection exactly.
    ``raw_mods`` records each cell's raw event modalities (identical to the
    key except for the "Dental imaging" pseudo-modality) so a
    pseudo-modality's events can be recognised as already covered by a
    canonical-modality selection.
    """
    n = int(iterations or 0)
    out: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    if n <= 0 or events is None or len(events) == 0:
        return out
    if _is_event_columns(events):
        raw_mod_ev = np.asarray(events.modality, dtype=object)
    else:
        raw_mod_ev = np.array(
            [str(e.get("modality", "") or "") for e in events],
            dtype=object,
        )
    (mod_ev, reg_ev, _age_ev, di_ev, it_ev, died_ev, qaly_ev,
     _life_ev, _ct, _sx, _xi) = _event_arrays(events, _sim_decade_index,
                                             dental_cbct_raw=True)
    valid = (it_ev >= 0) & (it_ev < n)
    if not valid.any():
        return out
    it_v = it_ev[valid].astype(np.int64)
    mod_v = mod_ev[valid]
    reg_v = np.array([str(r) for r in reg_ev[valid]], dtype=object)
    died_v = died_ev[valid]
    qaly_v = qaly_ev[valid]
    di_v = di_ev[valid].astype(np.int64)
    raw_mod_v = raw_mod_ev[valid]

    # Same rare-combo fold as compute_region_age_dose_sweep (per year).
    if exam_data is not None:
        totals = _exam_region_scan_totals(exam_data)
        fold_regs = [
            r for r, c in totals.items()
            if _is_combo_region(r) and c < _RARE_COMBO_THRESHOLD
        ]
        if fold_regs:
            reg_v = np.where(
                np.isin(reg_v, fold_regs), _CT_OTHER_COMBINATIONS, reg_v)

    _n_dec = len(AGE_DECADE_LABELS)
    umod, mc = np.unique(mod_v, return_inverse=True)
    ureg, rc = np.unique(reg_v, return_inverse=True)
    cell_code = ((mc.astype(np.int64) * len(ureg) + rc) * _n_dec
                 + di_v)
    # (cell, iteration) pairs -> summed outcome per slot.
    flat = cell_code * n + it_v
    uflat, invf = np.unique(flat, return_inverse=True)
    c_cases = np.bincount(invf, minlength=uflat.size).astype(np.float32)
    c_deaths = np.bincount(
        invf, weights=died_v, minlength=uflat.size).astype(np.float32)
    c_qaly = np.bincount(
        invf, weights=qaly_v, minlength=uflat.size).astype(np.float32)
    cell_of = uflat // n          # non-decreasing (uflat sorted)
    pos_all = (uflat % n).astype(np.int32)

    # raw modality set per cell (for the pseudo-modality coverage test).
    uraw, rawc = np.unique(raw_mod_v, return_inverse=True)
    pair = np.unique(cell_code * len(uraw) + rawc)
    cell_raws: Dict[int, List[str]] = {}
    for p in pair:
        cell_raws.setdefault(
            int(p // len(uraw)), []).append(str(uraw[int(p % len(uraw))]))

    uniq_cells, starts = np.unique(cell_of, return_index=True)
    ends = np.append(starts[1:], uflat.size)
    for i, cc in enumerate(uniq_cells):
        cc = int(cc)
        di = cc % _n_dec
        comb = cc // _n_dec
        mod_i = comb // len(ureg)
        reg_i = comb % len(ureg)
        s, e = int(starts[i]), int(ends[i])
        out[(str(umod[mod_i]), str(ureg[reg_i]), di)] = {
            "pos": pos_all[s:e],
            "cases": c_cases[s:e],
            "deaths": c_deaths[s:e],
            "qaly": c_qaly[s:e],
            "raw_mods": tuple(cell_raws.get(cc, ())),
        }
    return out


def _iter_series_by_modality(events, iterations):
    """Raw modality -> ``{"iter_cases","iter_deaths","iter_qaly"}`` dense
    per-iteration lists, the same binning as
    ``year_batch._per_modality_breakdown``."""
    n = int(iterations or 0)
    out: Dict[str, Dict[str, list]] = {}
    if n <= 0 or events is None or len(events) == 0:
        return out
    ev = events if _is_event_columns(events) else _ensure_event_columns(events)
    if ev is None or len(ev) == 0:
        return out
    it = np.asarray(ev.iteration)
    mod = np.asarray(ev.modality)
    died = np.asarray(ev.died, dtype=np.float64)
    qaly = np.asarray(ev.QALY_lost, dtype=np.float64)
    valid = (it >= 0) & (it < n)
    for m in np.unique(mod):
        sel = valid & (mod == m)
        idx = it[sel].astype(np.int64)
        out[str(m)] = {
            "iter_cases": np.bincount(idx, minlength=n).tolist(),
            "iter_deaths": np.bincount(
                idx, weights=died[sel], minlength=n).tolist(),
            "iter_qaly": np.bincount(
                idx, weights=qaly[sel], minlength=n).tolist(),
        }
    return out


def _sim_cell_entries(cells: Dict[Any, Any], key: Tuple[str, str, int]) -> list:
    """Cell-store entries for a sweep cell key.  A "Dental CBCT" key (the
    sweep keys these under "CBCT") collects every raw-modality variant
    stored for that (region, decade); any other key is a plain lookup."""
    mod, reg, di = key
    if reg == _DENTAL_CBCT_REGION:
        return [e for (m, r, d), e in cells.items()
                if r == reg and int(d) == int(di)]
    e = cells.get(key)
    return [e] if e is not None else []


def _sim_cell_dense(cell_entries, n: int, field: str) -> np.ndarray:
    """Densify one entry, or a list of entries summed, to length ``n``."""
    a = np.zeros(n, dtype=np.float64)
    if cell_entries is None:
        return a
    if isinstance(cell_entries, dict):
        cell_entries = [cell_entries]
    for e in cell_entries:
        a[np.asarray(e["pos"], dtype=np.int64)] += np.asarray(
            e[field], dtype=np.float64)
    return a


def _sim_cells_series(sim: Dict[str, Any], cells_or_rows) -> Optional[Dict[str, np.ndarray]]:
    """Pooled per-iteration {cases, deaths, qaly, non_mortal} summed over
    the given sweep cells (keys or per_region_age row dicts).

    Returns ``None`` when no cell store was recorded — the caller then
    reports "n/a" rather than falling back to an analytical interval.
    """
    cells = sim.get("cells")
    n = int(sim.get("n") or 0)
    # A missing year digest would silently read as a zero-outcome year.
    if cells is None or n <= 0 or sim.get("missing_years"):
        return None
    c = np.zeros(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    q = np.zeros(n, dtype=np.float64)
    for item in cells_or_rows:
        if isinstance(item, dict):
            key = (str(item.get("modality") or ""),
                   str(item.get("region") or ""),
                   int(item.get("age_decade_idx", 0) or 0))
        else:
            key = item
        for e in _sim_cell_entries(cells, key):
            pos = np.asarray(e["pos"], dtype=np.int64)
            c[pos] += np.asarray(e["cases"], dtype=np.float64)
            d[pos] += np.asarray(e["deaths"], dtype=np.float64)
            q[pos] += np.asarray(e["qaly"], dtype=np.float64)
    return {"cases": c, "deaths": d, "qaly": q, "non_mortal": c - d}


def _sim_saved_series(sim: Dict[str, Any], cell_rows, metric: str,
                      pct: float) -> Optional[np.ndarray]:
    """Per-iteration saved/harm series for a dose change of ``pct`` %.

    Σ over the given sweep cells of ``series_metric(cell) × m_cell`` where
    ``m_cell = _lnt_mult(pct, leukq_share_<metric>)`` — the same per-cell
    multiplier the point estimate uses, so a dose increase (negative m)
    flips the series and its percentile band orders itself accordingly.
    ``None`` when no cell store was recorded.
    """
    cells = sim.get("cells")
    n = int(sim.get("n") or 0)
    if cells is None or n <= 0 or sim.get("missing_years"):
        return None
    share_key = {
        "cases": "leukq_share_cases",
        "deaths": "leukq_share_deaths",
        "non_mortal": "leukq_share_non_mortal",
        "qaly": "leukq_share_QALY",
    }[metric]
    s = np.zeros(n, dtype=np.float64)
    for r in cell_rows:
        key = (str(r.get("modality") or ""), str(r.get("region") or ""),
               int(r.get("age_decade_idx", 0) or 0))
        m = _lnt_mult(pct, float(r.get(share_key, 0.0) or 0.0))
        for e in _sim_cell_entries(cells, key):
            s[np.asarray(e["pos"], dtype=np.int64)] += (
                np.asarray(e[metric], dtype=np.float64) * m)
    return s


def _sim_cohort_series(sim: Dict[str, Any], modality_filter,
                       ) -> Optional[Dict[str, np.ndarray]]:
    """Pooled per-iteration cohort series for the sweep's modality
    selection.

    ``None`` selection = "All modalities" -> the same pooled totals the
    Summary tab uses.  A selection sums the selected canonical modalities
    **elementwise** from ``stats['_iter_lists_by_modality']`` — identical to
    what the Summary tab's modality filter produces (a grouped label like
    Radiography = CR+DX+DR is the sum over the same simulated year, never
    a concatenation of independent years).  Sweep-only pseudo-modalities
    (e.g. "Dental imaging"), which are subsets of a canonical modality's
    events, fall back to summing their own cells — except cells whose raw
    modalities are already covered by a selected canonical modality, which
    would otherwise be counted twice.

    One exact correction on top of that parity: the sweep keys every
    "Dental CBCT" exam under "CBCT" whatever raw modality it carries,
    while the raw-modality lists file a DX-coded dental CBCT event under
    Radiography.  The cell store keeps dental CBCT per raw modality, so
    those events are added when CBCT is selected without their raw
    modality, and removed when their raw modality is selected without
    CBCT -- the cohort series then covers exactly the cells the point
    estimates were summed over.
    """
    n = int(sim.get("n") or 0)
    totals = sim.get("totals") or {}
    if not modality_filter:
        return {
            "cases": np.asarray(totals.get("cases") or np.zeros(n),
                                dtype=np.float64),
            "deaths": np.asarray(totals.get("deaths") or np.zeros(n),
                                 dtype=np.float64),
            "qaly": np.asarray(totals.get("qaly") or np.zeros(n),
                               dtype=np.float64),
        }
    mods = ([modality_filter] if isinstance(modality_filter, str)
            else list(modality_filter))
    sel_norm = {_norm_modality(m) for m in mods}
    by_mod = sim.get("by_modality") or {}
    by_mod_norm = {_norm_modality(k): v for k, v in by_mod.items()}
    cells = sim.get("cells") or {}
    c = np.zeros(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    q = np.zeros(n, dtype=np.float64)
    for m in mods:
        m_norm = _norm_modality(m)
        e = by_mod_norm.get(m_norm) if by_mod_norm else None
        if e is not None:
            for arr, field in (
                (c, "iter_cases"), (d, "iter_deaths"), (q, "iter_qaly"),
            ):
                v = np.asarray(e.get(field) or [], dtype=np.float64)
                if v.size == 0:
                    continue
                if v.size != n:
                    raise ValueError(
                        f"{m!r}: per-iteration series length {v.size} != "
                        f"{n}; modality series must be combined elementwise "
                        "(per simulated year), never concatenated"
                    )
                arr += v
            # NB: the canonical lists are keyed by RAW event modality, so
            # "Radiography" still carries the dental-subtype events the
            # sweep regrouped under "Dental imaging" — exactly what the
            # Summary tab's modality dropdown (``_compute_modality_filtered_
            # results``: raw ``ev.modality == code``) reports for the same
            # selection.  Keep that parity; the pseudo-modality itself is
            # covered by the cell fallback below when selected.
        else:
            # Absent canonical modality (by_mod populated) contributes
            # zeros; a pseudo-modality (or an absent by_mod store) sums its
            # own cells, skipping cells already covered by the selection.
            m_norm = _norm_modality(m)
            for key, ce in cells.items():
                if _norm_modality(key[0]) != m_norm:
                    continue
                raws = {_norm_modality(r) for r in (ce.get("raw_mods") or ())}
                if raws and raws <= sel_norm:
                    continue
                pos = np.asarray(ce["pos"], dtype=np.int64)
                c[pos] += np.asarray(ce["cases"], dtype=np.float64)
                d[pos] += np.asarray(ce["deaths"], dtype=np.float64)
                q[pos] += np.asarray(ce["qaly"], dtype=np.float64)
    # Dental CBCT correction (see docstring): a raw-keyed dental CBCT
    # cell was counted above iff its raw modality is selected AND present
    # in the raw-modality lists; it BELONGS iff CBCT is selected.
    want = _DENTAL_CBCT_MODALITY in sel_norm
    for (raw, reg, _di), ce in cells.items():
        if reg != _DENTAL_CBCT_REGION:
            continue
        raw_norm = _norm_modality(raw)
        counted = raw_norm in sel_norm and by_mod_norm.get(raw_norm) is not None
        if counted == want:
            continue
        sign = 1.0 if want else -1.0
        pos = np.asarray(ce["pos"], dtype=np.int64)
        c[pos] += sign * np.asarray(ce["cases"], dtype=np.float64)
        d[pos] += sign * np.asarray(ce["deaths"], dtype=np.float64)
        q[pos] += sign * np.asarray(ce["qaly"], dtype=np.float64)
    return {"cases": c, "deaths": d, "qaly": q, "non_mortal": c - d}


def _sim_range(series) -> Optional[Tuple[float, float]]:
    """(2.5, 97.5) percentiles of a pooled annual series, or None."""
    if series is None:
        return None
    a = np.asarray(series, dtype=np.float64)
    if a.size == 0:
        return None
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def _apply_sim_cell_ranges(sweep: Dict[str, Any], sim: Dict[str, Any]) -> None:
    """Rewrite every per-cell ``*_ci_*_per_100k`` field on *sweep* in place
    to the central 95% range of that cell's pooled simulated annual series.

    NaN where no series was recorded (no cell store, or missing year
    digests) — the renderers show "n/a"; the analytical interval is never
    re-substituted.  Point estimates are untouched.
    """
    cells = sim.get("cells")
    if sim.get("missing_years"):
        # A year whose digest lacks the series would silently read as a
        # zero-outcome year — under-reporting the spread.  Cannot compute
        # -> n/a for every cell-derived interval.
        cells = None
    n = int(sim.get("n") or 0)
    per_n = float(sweep.get("per_n_scans") or 0.0)
    analytical = sweep_is_analytical(sweep)
    n_years = int(sweep.get("_averaged_across_years") or 0) or 1
    levels = np.asarray(sweep.get("levels") or (), dtype=float)
    mult = -levels / 100.0
    nan = float("nan")
    for r in sweep.get("per_region_age") or []:
        scans = float(r.get("scans_per_year") or 0.0)
        scale = per_n / scans if scans > 0.0 else 0.0
        # Analytical 95% range of a per_n-exam cohort drawn from this
        # cell's mix — stored for every analytical cell with a non-NaN
        # analytical interval, including cells whose simulated series is
        # missing.
        # The pooled analytical half per per_n exams is
        # CI_Z·per_n·sqrt(ΣV_y)/Σs_y with Σs_y = scans_per_year·n_years,
        # while the variance of a per_n-exam cohort from the same mix is
        # per_n·ΣV_y/Σs_y, so its half is h·sqrt(Σs_y/per_n).  Holds for
        # both analytical kinds the tab uses (a single year is n_years=1).
        if analytical:
            exams_total = scans * n_years
            for bk in ("baseline_cases", "baseline_deaths", "baseline_QALY"):
                cohort_lo_key = f"{bk}_cohort_lo_per_100k"
                cohort_hi_key = f"{bk}_cohort_hi_per_100k"
                if cohort_lo_key in r:
                    continue  # keep the stored interval; the ci fields
                    # hold simulated ranges after the first application
                lo_a = r.get(f"{bk}_ci_lo_per_100k")
                hi_a = r.get(f"{bk}_ci_hi_per_100k")
                if lo_a is None or hi_a is None:
                    continue
                lo_a = float(lo_a)
                hi_a = float(hi_a)
                if np.isnan(lo_a) or np.isnan(hi_a):
                    continue
                h_c = 0.0
                if per_n > 0.0 and exams_total > 0.0:
                    h_c = ((hi_a - lo_a) / 2.0
                           * np.sqrt(exams_total / per_n))
                mean_b = float(r.get(f"{bk}_per_100k") or 0.0)
                r[cohort_lo_key] = mean_b - h_c
                r[cohort_hi_key] = mean_b + h_c
        if cells is None or n < 2:
            for bk in ("baseline_cases", "baseline_deaths", "baseline_QALY"):
                r[f"{bk}_ci_lo_per_100k"] = nan
                r[f"{bk}_ci_hi_per_100k"] = nan
                r[f"{bk}_range_source"] = "n/a"
            for fk in ("cases_avoided", "deaths_avoided",
                       "non_mortal_avoided", "QALY_saved"):
                r[f"{fk}_ci_lo_per_100k"] = np.full(levels.size, nan)
                r[f"{fk}_ci_hi_per_100k"] = np.full(levels.size, nan)
            continue
        key = (str(r.get("modality") or ""), str(r.get("region") or ""),
               int(r.get("age_decade_idx", 0) or 0))
        entry = _sim_cell_entries(cells, key)
        ser_c = _sim_cell_dense(entry, n, "cases")
        ser_d = _sim_cell_dense(entry, n, "deaths")
        ser = {
            "cases": ser_c,
            "deaths": ser_d,
            "qaly": _sim_cell_dense(entry, n, "qaly"),
            "non_mortal": ser_c - ser_d,
        }
        ranges = {m: _sim_range(a) for m, a in ser.items()}
        for bk, m in (("baseline_cases", "cases"),
                      ("baseline_deaths", "deaths"),
                      ("baseline_QALY", "qaly")):
            mean_b = float(r.get(f"{bk}_per_100k") or 0.0)
            # Fallback interval for cells whose simulated range is
            # degenerate: the stored analytical cohort interval from
            # above (the ci fields hold simulated ranges, so it is not
            # recomputed from them).
            cohort_lo_key = f"{bk}_cohort_lo_per_100k"
            cohort_hi_key = f"{bk}_cohort_hi_per_100k"
            if cohort_lo_key in r:
                h_cohort = (float(r[cohort_hi_key])
                            - float(r[cohort_lo_key])) / 2.0
                cohort_stored = True
            else:
                cohort_stored = False
                h_cohort = 0.0
            lo, hi = ranges[m] if ranges[m] else (nan, nan)
            lo_s, hi_s = lo * scale, hi * scale
            r[f"{bk}_ci_lo_per_100k"] = lo_s
            r[f"{bk}_ci_hi_per_100k"] = hi_s
            if np.isnan(lo_s) or np.isnan(hi_s):
                r[f"{bk}_range_source"] = "n/a"
            elif (analytical and mean_b > 0.0 and hi_s <= lo_s
                  and cohort_stored and h_cohort > 0.0):
                r[f"{bk}_range_source"] = "cohort_100k"
            else:
                r[f"{bk}_range_source"] = "simulated"
        for fk, m, share_key in (
            ("cases_avoided", "cases", "leukq_share_cases"),
            ("deaths_avoided", "deaths", "leukq_share_deaths"),
            ("non_mortal_avoided", "non_mortal", "leukq_share_non_mortal"),
            ("QALY_saved", "qaly", "leukq_share_QALY"),
        ):
            w = float(r.get(share_key, 0.0) or 0.0)
            m_eff = mult * (1.0 + w * (1.0 - mult))
            lo, hi = ranges[m] if ranges[m] else (nan, nan)
            lo_arr, hi_arr = _project_ci(lo, hi, scale, m_eff)
            r[f"{fk}_ci_lo_per_100k"] = lo_arr
            r[f"{fk}_ci_hi_per_100k"] = hi_arr


def attach_simulated_ranges(sweep, stats=None, *, events=None,
                            exam_data=None, iter_cases=None,
                            iter_deaths=None):
    """Return *sweep* with ``_sim`` simulated-outcome series attached and all
    per-cell intervals rewritten to central-95% simulated annual ranges.

    *stats* supplies the pooled ``per_year_average`` products
    (``_iter_lists_by_modality``, ``iter_QALY_list``, ``_sim_cell_series``,
    ``_sim_cell_series_len`` / ``_sim_cell_series_missing``); *events* and
    *exam_data* are the single/combined-run payload from which the same
    series are built directly.  ``iter_cases`` / ``iter_deaths`` are the
    pooled per-iteration totals (``results['cases_l']`` /
    ``results['deaths_l']``) — the "All modalities" selection.  Point
    estimates, levels, scans and every non-interval field are untouched.
    """
    if not sweep:
        return sweep
    stats = stats or {}
    out = dict(sweep)
    out["per_region_age"] = [
        dict(r) for r in (sweep.get("per_region_age") or [])]

    n_iters = 0
    for s in (iter_cases, iter_deaths):
        if s:
            n_iters = len(s)
            break
    if not n_iters:
        ql = stats.get("iter_QALY_list")
        if ql:
            n_iters = len(ql)
    if not n_iters:
        n_iters = int(stats.get("iterations", 0) or 0)

    is_batch = (stats.get("_run_mode") == "per_year_average"
                or stats.get("_iter_lists_by_modality") is not None)
    if is_batch:
        cells = stats.get("_sim_cell_series")
        by_mod = stats.get("_iter_lists_by_modality") or {}
        missing = list(stats.get("_sim_cell_series_missing") or [])
        n_series = int(stats.get("_sim_cell_series_len") or 0) or n_iters
    else:
        cells = per_iteration_cell_series(events, n_iters, exam_data)
        by_mod = _iter_series_by_modality(events, n_iters)
        missing = []
        n_series = n_iters

    sim = {
        "n": n_series,
        "totals": {"cases": iter_cases, "deaths": iter_deaths,
                   "qaly": stats.get("iter_QALY_list")},
        "by_modality": by_mod,
        "cells": cells,
        "missing_years": missing,
    }
    out["_sim"] = sim
    _apply_sim_cell_ranges(out, sim)
    return out


def _extrapolation_factor(
    sweep: Dict[str, Any],
    extrapolate_to_exams: Optional[float],
) -> Tuple[float, float]:
    """Return (factor, current_total_scans_per_year) for manual extrapolation.

    `current_total_scans_per_year` is the sum of the displayed examination
    count (``modelled_scans_per_year``, falling back to ``scans_per_year``)
    over the same rows that the annual benefit / burden functions use.
    `factor` is `extrapolate_to_exams / current_total`.  If
    `extrapolate_to_exams` is None, not a positive number, or the current
    total is zero, factor is 1.0 (no extrapolation).
    """
    current = 0.0
    for r in list((sweep or {}).get("per_region_age") or []):
        reg_name = str(r.get("region", ""))
        if reg_name.startswith("Non-Dose"):
            continue
        if r.get("has_risk_model", True) is False:
            continue
        s = float(r.get("scans_per_year", 0.0) or 0.0)
        if s <= 0.0:
            continue
        current += (float(r["modelled_scans_per_year"])
                    if "modelled_scans_per_year" in r else s)
    if extrapolate_to_exams is None:
        return 1.0, current
    try:
        target = float(extrapolate_to_exams)
    except (TypeError, ValueError):
        return 1.0, current
    if not (target > 0.0) or current <= 0.0:
        return 1.0, current
    return target / current, current


def _cbct_modality_key(sweep: Dict[str, Any]) -> str:
    """The sweep's CBCT modality key (first key normalising to "CBCT"
    among the modality map and the per-cell rows; ``"CBCT"`` if none).
    Same lookup as the region/age heatmap's ``dental_cbct_in_cbct``."""
    sweep = sweep or {}
    candidates = (list(sweep.get("modality_region_map") or {})
                  + [str(r.get("modality", "") or "")
                     for r in (sweep.get("per_region_age") or [])])
    return next((k for k in candidates if _norm_modality(k) == "CBCT"), "CBCT")


def _rehome_dental_cbct_map(modality_region_map, cbct_key: str,
                            present: bool) -> Dict[str, List[str]]:
    """Copy of ``modality_region_map`` with "Dental CBCT" listed only
    under ``cbct_key`` (added there when ``present``), mirroring the
    heatmap's ``dental_cbct_in_cbct`` regrouping.  An absent map stays
    ``{}``."""
    if not modality_region_map:
        return {}
    out = {k: [r for r in v if r != "Dental CBCT"]
           for k, v in modality_region_map.items()}
    out = {k: v for k, v in out.items() if v}
    if present:
        out.setdefault(cbct_key, [])
        if "Dental CBCT" not in out[cbct_key]:
            out[cbct_key].append("Dental CBCT")
    return out


def compute_annual_benefit(
    sweep: Dict[str, Any],
    target_pct: float = -10.0,
    extrapolate_to_exams: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate annual QALYs / cancers / deaths saved from a dose reduction.

    Combines the per-*per_n_scans* burden (generalisable) with the local
    annual scan count to produce a hospital-level expected benefit.

    Returns a dict with:
    * ``per_region_age``: list of {region, age_group, age_decade_idx,
      scans_per_year, dose_sv_annual, dose_msv_annual,
      QALY_lost_annual, cases_lost_annual, deaths_lost_annual,
      QALY_saved_annual, cases_saved_annual, deaths_saved_annual}
    * ``per_region``: list of {region, QALY_saved_annual,
      cases_saved_annual, deaths_saved_annual, total_scans,
      total_scans_all, dose_sv_annual, dose_msv_annual,
      QALY_lost_annual, cases_lost_annual, deaths_lost_annual} sorted
      by QALY_saved_annual descending.
    * ``cohort_total``: {QALY_saved_annual, cases_saved_annual,
      deaths_saved_annual, total_scans, total_scans_all, dose_sv_annual,
      dose_msv_annual,
      QALY_lost_annual, cases_lost_annual, deaths_lost_annual}
    * ``target_pct``: the dose change used.

    ``total_scans`` / ``dose_sv_annual`` / ``dose_msv_annual`` count the
    risk-MODELLED exams only (``modelled_scans_per_year`` /
    ``modelled_dose_sv_per_year`` -- non-empty ``_cancer_doses``), matching
    the Summary tab's section-2 denominator; ``total_scans_all`` keeps the
    all-exams count.  Sweeps that predate the ``modelled_*`` keys fall back
    to the all-exams values.  The outcome totals (``*_annual``) are
    unchanged by this -- they are scaled by ``scans_per_year``.

    ``target_pct`` is a signed dose *change*: negative is a reduction (the
    ``*_saved`` / ``*_avoided`` fields are then positive benefits) and
    positive is an increase (they turn negative, i.e. added harm).  Any
    percentage is accepted, not only the levels the sweep priced -- an
    off-grid value is computed exactly from each cell's baseline and its
    leukaemia-quadratic share rather than snapped to a nearby level.
    """
    rows = list((sweep or {}).get("per_region_age") or [])
    levels = list((sweep or {}).get("levels") or ())
    per_n = int((sweep or {}).get("per_n_scans") or 100_000)
    ex_factor, ex_current_total = _extrapolation_factor(sweep, extrapolate_to_exams)
    # Pooled simulated annual-outcome series, when attach_simulated_ranges
    # ran on this sweep — intervals below then become the central 95% range
    # of simulated annual outcomes rather than the analytical range of the
    # mean.
    _sim = (sweep or {}).get("_sim")

    # Locate the requested dose change among the levels the sweep priced.
    # A miss is normal: the GUI lets the user type any percentage, while the
    # sweep only evaluates ``DEFAULT_DOSE_REDUCTIONS_PCT``.  Rather than
    # silently substituting a level the user did not ask for, the off-grid case is priced directly
    # from each cell's baseline and its stored leukaemia-quadratic share.
    #
    # That is exact, not interpolated: the sweep's own per-level values are
    # baseline x scale x m_eff(level), so evaluating the same closed-form
    # m_eff at an arbitrary level reproduces what the sweep would have
    # computed had that level been in the list.  ``_verify_offgrid`` in
    # verify_offgrid_levels.py checks this to ~1e-12 against a re-run sweep.
    try:
        level_idx = levels.index(float(target_pct))
    except ValueError:
        level_idx = None

    def _at_level(r: Dict[str, Any], avoided_key: str,
                  baseline_key: str, share_key: str) -> float:
        """This cell's avoided/added amount per ``per_n`` at ``target_pct``."""
        if level_idx is not None:
            return float(np.asarray(r.get(avoided_key))[level_idx])
        return (float(r.get(baseline_key, 0.0) or 0.0)
                * _lnt_mult(target_pct, r.get(share_key, 0.0) or 0.0))

    def _ci_at_level(r: Dict[str, Any], lo_key: str, hi_key: str,
                     base_lo_key: str, base_hi_key: str,
                     share_key: str) -> Tuple[float, float]:
        """The matching 95% interval, per ``per_n`` at ``target_pct``.

        Off-grid this rescales the baseline interval by the same multiplier
        as the point estimate -- a deterministic linear map, so the interval
        scales exactly (see _project_ci) -- and orders the endpoints, since
        a dose increase makes the multiplier negative and flips them.
        """
        if level_idx is not None:
            return (float(np.asarray(r.get(lo_key, [0.0]))[level_idx]),
                    float(np.asarray(r.get(hi_key, [0.0]))[level_idx]))
        _m = _lnt_mult(target_pct, r.get(share_key, 0.0) or 0.0)
        _a = float(r.get(base_lo_key, 0.0) or 0.0) * _m
        _b = float(r.get(base_hi_key, 0.0) or 0.0) * _m
        return min(_a, _b), max(_a, _b)

    per_ra: List[Dict[str, Any]] = []
    region_acc: Dict[Tuple[str, str], Dict[str, float]] = {}
    cohort_rows: List[Dict[str, Any]] = []

    _cbct_key = _cbct_modality_key(sweep)
    _dental_cbct_present = False

    for r in rows:
        reg_name = str(r.get("region", ""))
        # "Unknown/unspecified region" (blank/unrecognised Scan Regions) is
        # intentionally kept here -- for CT it still carries real per-organ
        # doses from the Excel columns, so it contributes a genuine annual
        # burden/benefit total -- only Non-Dose (QA, cancellations,
        # mislabelled non-CT) is excluded.  An unlocalised radiography/CBCT
        # row has no dose model and is dropped by the has_risk_model test.
        if reg_name.startswith("Non-Dose"):
            continue
        # Regions with no dose/risk model (e.g. non-dental CBCT body parts)
        # are shown as n/a in the heatmap and carry no modelled burden, so
        # they are left out of the annual totals rather than counted as zero.
        if r.get("has_risk_model", True) is False:
            continue
        scans = float(r.get("scans_per_year", 0.0))
        if scans <= 0.0:
            continue
        scans = scans * ex_factor
        # QALY_saved_per_100k is an array over the sweep's dose-change
        # levels; off-grid requests are priced from the baseline instead.
        QALY_per_n = _at_level(r, "QALY_saved_per_100k",
                               "baseline_QALY_per_100k", "leukq_share_QALY")
        cases_per_n = _at_level(r, "cases_avoided_per_100k",
                                "baseline_cases_per_100k", "leukq_share_cases")
        deaths_per_n = _at_level(r, "deaths_avoided_per_100k",
                                 "baseline_deaths_per_100k",
                                 "leukq_share_deaths")

        # Scale per-N → annual: multiply by (annual scans / per_n).
        annual_factor = scans / per_n
        QALY_annual = QALY_per_n * annual_factor
        cases_annual = cases_per_n * annual_factor
        deaths_annual = deaths_per_n * annual_factor

        # Baseline burden (no dose reduction) for this cell.
        baseline_QALY_per_n = float(r.get("baseline_QALY_per_100k", 0.0) or 0.0)
        baseline_cases_per_n = float(r.get("baseline_cases_per_100k", 0.0) or 0.0)
        baseline_deaths_per_n = float(r.get("baseline_deaths_per_100k", 0.0) or 0.0)
        baseline_QALY_annual = baseline_QALY_per_n * annual_factor
        baseline_cases_annual = baseline_cases_per_n * annual_factor
        baseline_deaths_annual = baseline_deaths_per_n * annual_factor

        QALY_saved_ci = _per_100k_ci_to_annual(
            *_ci_at_level(r, "QALY_saved_ci_lo_per_100k",
                          "QALY_saved_ci_hi_per_100k",
                          "baseline_QALY_ci_lo_per_100k",
                          "baseline_QALY_ci_hi_per_100k",
                          "leukq_share_QALY"),
            annual_factor)
        cases_saved_ci = _per_100k_ci_to_annual(
            *_ci_at_level(r, "cases_avoided_ci_lo_per_100k",
                          "cases_avoided_ci_hi_per_100k",
                          "baseline_cases_ci_lo_per_100k",
                          "baseline_cases_ci_hi_per_100k",
                          "leukq_share_cases"),
            annual_factor)
        deaths_saved_ci = _per_100k_ci_to_annual(
            *_ci_at_level(r, "deaths_avoided_ci_lo_per_100k",
                          "deaths_avoided_ci_hi_per_100k",
                          "baseline_deaths_ci_lo_per_100k",
                          "baseline_deaths_ci_hi_per_100k",
                          "leukq_share_deaths"),
            annual_factor)
        baseline_QALY_ci = _per_100k_ci_to_annual(
            float(r.get("baseline_QALY_ci_lo_per_100k", 0.0) or 0.0),
            float(r.get("baseline_QALY_ci_hi_per_100k", 0.0) or 0.0),
            annual_factor)
        baseline_cases_ci = _per_100k_ci_to_annual(
            float(r.get("baseline_cases_ci_lo_per_100k", 0.0) or 0.0),
            float(r.get("baseline_cases_ci_hi_per_100k", 0.0) or 0.0),
            annual_factor)
        baseline_deaths_ci = _per_100k_ci_to_annual(
            float(r.get("baseline_deaths_ci_lo_per_100k", 0.0) or 0.0),
            float(r.get("baseline_deaths_ci_hi_per_100k", 0.0) or 0.0),
            annual_factor)

        # Displayed exam count / collective dose count the risk-MODELLED
        # exams only (non-empty ``_cancer_doses``), matching the Summary
        # tab's section-2 denominator.  ``modelled_*`` keys fall back to
        # the all-exams values for sweeps that predate them.
        modelled_scans = (float(r["modelled_scans_per_year"]) * ex_factor
                          if "modelled_scans_per_year" in r else scans)
        dose_sv_per_year = (float(r["modelled_dose_sv_per_year"])
                            if "modelled_dose_sv_per_year" in r
                            else float(r.get("dose_sv_per_year", 0.0) or 0.0)) * ex_factor
        dose_msv_annual = dose_sv_per_year * 1000.0

        mod = str(r.get("modality", "") or "")
        reg = str(r.get("region", "?"))
        age = str(r.get("age_group", "?"))
        di = int(r.get("age_decade_idx", 0))
        # "Dental CBCT" exams coded under other raw modalities are pooled
        # into the CBCT group, as the region/age heatmap does.  Only the
        # DISPLAY key changes: ``acc["_rows"]`` / ``cohort_rows`` keep the
        # original sweep row, whose raw modality the simulated cell store
        # is keyed by.
        if reg == "Dental CBCT":
            mod = _cbct_key
            _dental_cbct_present = True

        per_ra.append({
            "modality": mod,
            "region": reg,
            "age_group": age,
            "age_decade_idx": di,
            "scans_per_year": scans,
            "dose_sv_annual": dose_sv_per_year,
            "dose_msv_annual": dose_msv_annual,
            "QALY_lost_annual": baseline_QALY_annual,
            "cases_lost_annual": baseline_cases_annual,
            "deaths_lost_annual": baseline_deaths_annual,
            "QALY_saved_annual": QALY_annual,
            "cases_saved_annual": cases_annual,
            "deaths_saved_annual": deaths_annual,
            "QALY_saved_annual_ci_lo": QALY_saved_ci[0],
            "QALY_saved_annual_ci_hi": QALY_saved_ci[1],
            "cases_saved_annual_ci_lo": cases_saved_ci[0],
            "cases_saved_annual_ci_hi": cases_saved_ci[1],
            "deaths_saved_annual_ci_lo": deaths_saved_ci[0],
            "deaths_saved_annual_ci_hi": deaths_saved_ci[1],
            "QALY_lost_annual_ci_lo": baseline_QALY_ci[0],
            "QALY_lost_annual_ci_hi": baseline_QALY_ci[1],
            "cases_lost_annual_ci_lo": baseline_cases_ci[0],
            "cases_lost_annual_ci_hi": baseline_cases_ci[1],
            "deaths_lost_annual_ci_lo": baseline_deaths_ci[0],
            "deaths_lost_annual_ci_hi": baseline_deaths_ci[1],
        })

        acc = region_acc.setdefault((mod, reg), {
            "QALY_saved_annual": 0.0,
            "cases_saved_annual": 0.0,
            "deaths_saved_annual": 0.0,
            "total_scans": 0.0,
            "total_scans_all": 0.0,
            "dose_sv_annual": 0.0,
            "dose_msv_annual": 0.0,
            "QALY_lost_annual": 0.0,
            "cases_lost_annual": 0.0,
            "deaths_lost_annual": 0.0,
            "_var_QALY_saved": 0.0,
            "_var_cases_saved": 0.0,
            "_var_deaths_saved": 0.0,
            "_var_QALY_lost": 0.0,
            "_var_cases_lost": 0.0,
            "_var_deaths_lost": 0.0,
            "_rows": [],
        })
        acc["_rows"].append(r)
        cohort_rows.append(r)
        acc["QALY_saved_annual"] += QALY_annual
        acc["cases_saved_annual"] += cases_annual
        acc["deaths_saved_annual"] += deaths_annual
        acc["total_scans"] += modelled_scans
        acc["total_scans_all"] += scans
        acc["dose_sv_annual"] += dose_sv_per_year
        acc["dose_msv_annual"] += dose_msv_annual
        acc["QALY_lost_annual"] += baseline_QALY_annual
        acc["cases_lost_annual"] += baseline_cases_annual
        acc["deaths_lost_annual"] += baseline_deaths_annual
        acc["_var_QALY_saved"] += _ci_var(*QALY_saved_ci)
        acc["_var_cases_saved"] += _ci_var(*cases_saved_ci)
        acc["_var_deaths_saved"] += _ci_var(*deaths_saved_ci)
        acc["_var_QALY_lost"] += _ci_var(*baseline_QALY_ci)
        acc["_var_cases_lost"] += _ci_var(*baseline_cases_ci)
        acc["_var_deaths_lost"] += _ci_var(*baseline_deaths_ci)

    def _metric_range(series):
        rng = _sim_range(series)
        if rng is None:
            return (float("nan"), float("nan"))
        return rng[0] * ex_factor, rng[1] * ex_factor

    per_region = []
    for (mod, reg), vals in region_acc.items():
        v = {k: val for k, val in vals.items() if not k.startswith("_")}
        v["modality"] = mod
        v["region"] = reg
        if _sim is not None:
            # Central 95% range of the pooled simulated annual outcome —
            # elementwise sum of this (modality, region)'s cells, with each
            # cell's saved/harm series at its own leukaemia-quadratic share.
            _rows = vals["_rows"]
            ser = _sim_cells_series(_sim, _rows)
            v["QALY_saved_annual_ci_lo"], v["QALY_saved_annual_ci_hi"] = \
                _metric_range(_sim_saved_series(_sim, _rows, "qaly", target_pct))
            v["cases_saved_annual_ci_lo"], v["cases_saved_annual_ci_hi"] = \
                _metric_range(_sim_saved_series(_sim, _rows, "cases", target_pct))
            v["deaths_saved_annual_ci_lo"], v["deaths_saved_annual_ci_hi"] = \
                _metric_range(_sim_saved_series(_sim, _rows, "deaths", target_pct))
            v["QALY_lost_annual_ci_lo"], v["QALY_lost_annual_ci_hi"] = \
                _metric_range((ser or {}).get("qaly"))
            v["cases_lost_annual_ci_lo"], v["cases_lost_annual_ci_hi"] = \
                _metric_range((ser or {}).get("cases"))
            v["deaths_lost_annual_ci_lo"], v["deaths_lost_annual_ci_hi"] = \
                _metric_range((ser or {}).get("deaths"))
        else:
            v["QALY_saved_annual_ci_lo"], v["QALY_saved_annual_ci_hi"] = _ci_from_var(
                vals["QALY_saved_annual"], vals["_var_QALY_saved"])
            v["cases_saved_annual_ci_lo"], v["cases_saved_annual_ci_hi"] = _ci_from_var(
                vals["cases_saved_annual"], vals["_var_cases_saved"])
            v["deaths_saved_annual_ci_lo"], v["deaths_saved_annual_ci_hi"] = _ci_from_var(
                vals["deaths_saved_annual"], vals["_var_deaths_saved"])
            v["QALY_lost_annual_ci_lo"], v["QALY_lost_annual_ci_hi"] = _ci_from_var(
                vals["QALY_lost_annual"], vals["_var_QALY_lost"])
            v["cases_lost_annual_ci_lo"], v["cases_lost_annual_ci_hi"] = _ci_from_var(
                vals["cases_lost_annual"], vals["_var_cases_lost"])
            v["deaths_lost_annual_ci_lo"], v["deaths_lost_annual_ci_hi"] = _ci_from_var(
                vals["deaths_lost_annual"], vals["_var_deaths_lost"])
        per_region.append(v)
    per_region.sort(key=lambda x: x["QALY_saved_annual"], reverse=True)

    _c_QALY_saved = sum(v["QALY_saved_annual"] for v in region_acc.values())
    _c_cases_saved = sum(v["cases_saved_annual"] for v in region_acc.values())
    _c_deaths_saved = sum(v["deaths_saved_annual"] for v in region_acc.values())
    _c_QALY_lost = sum(v["QALY_lost_annual"] for v in region_acc.values())
    _c_cases_lost = sum(v["cases_lost_annual"] for v in region_acc.values())
    _c_deaths_lost = sum(v["deaths_lost_annual"] for v in region_acc.values())
    _v_QALY_saved = sum(v["_var_QALY_saved"] for v in region_acc.values())
    _v_cases_saved = sum(v["_var_cases_saved"] for v in region_acc.values())
    _v_deaths_saved = sum(v["_var_deaths_saved"] for v in region_acc.values())
    _v_QALY_lost = sum(v["_var_QALY_lost"] for v in region_acc.values())
    _v_cases_lost = sum(v["_var_cases_lost"] for v in region_acc.values())
    _v_deaths_lost = sum(v["_var_deaths_lost"] for v in region_acc.values())
    if _sim is not None:
        # Baseline cohort = the Summary-tab selection (pooled totals or the
        # elementwise sum of the selected modalities' pooled series); the
        # saved/harm band is the elementwise sum of the per-cell saved
        # series at each cell's own leukaemia-quadratic share.
        _cser = _sim_cohort_series(_sim, (sweep or {}).get("modality_filter"))
        _cohort_ci = {
            "QALY_saved": _metric_range(
                _sim_saved_series(_sim, cohort_rows, "qaly", target_pct)),
            "cases_saved": _metric_range(
                _sim_saved_series(_sim, cohort_rows, "cases", target_pct)),
            "deaths_saved": _metric_range(
                _sim_saved_series(_sim, cohort_rows, "deaths", target_pct)),
            "QALY_lost": _metric_range((_cser or {}).get("qaly")),
            "cases_lost": _metric_range((_cser or {}).get("cases")),
            "deaths_lost": _metric_range((_cser or {}).get("deaths")),
        }
    else:
        _cohort_ci = {
            "QALY_saved": _ci_from_var(_c_QALY_saved, _v_QALY_saved),
            "cases_saved": _ci_from_var(_c_cases_saved, _v_cases_saved),
            "deaths_saved": _ci_from_var(_c_deaths_saved, _v_deaths_saved),
            "QALY_lost": _ci_from_var(_c_QALY_lost, _v_QALY_lost),
            "cases_lost": _ci_from_var(_c_cases_lost, _v_cases_lost),
            "deaths_lost": _ci_from_var(_c_deaths_lost, _v_deaths_lost),
        }
    cohort = {
        "QALY_saved_annual": _c_QALY_saved,
        "cases_saved_annual": _c_cases_saved,
        "deaths_saved_annual": _c_deaths_saved,
        "total_scans": sum(v["total_scans"] for v in region_acc.values()),
        "total_scans_all": sum(v["total_scans_all"] for v in region_acc.values()),
        "dose_sv_annual": sum(v["dose_sv_annual"] for v in region_acc.values()),
        "dose_msv_annual": sum(v["dose_msv_annual"] for v in region_acc.values()),
        "QALY_lost_annual": _c_QALY_lost,
        "cases_lost_annual": _c_cases_lost,
        "deaths_lost_annual": _c_deaths_lost,
        "QALY_saved_annual_ci_lo": _cohort_ci["QALY_saved"][0],
        "QALY_saved_annual_ci_hi": _cohort_ci["QALY_saved"][1],
        "cases_saved_annual_ci_lo": _cohort_ci["cases_saved"][0],
        "cases_saved_annual_ci_hi": _cohort_ci["cases_saved"][1],
        "deaths_saved_annual_ci_lo": _cohort_ci["deaths_saved"][0],
        "deaths_saved_annual_ci_hi": _cohort_ci["deaths_saved"][1],
        "QALY_lost_annual_ci_lo": _cohort_ci["QALY_lost"][0],
        "QALY_lost_annual_ci_hi": _cohort_ci["QALY_lost"][1],
        "cases_lost_annual_ci_lo": _cohort_ci["cases_lost"][0],
        "cases_lost_annual_ci_hi": _cohort_ci["cases_lost"][1],
        "deaths_lost_annual_ci_lo": _cohort_ci["deaths_lost"][0],
        "deaths_lost_annual_ci_hi": _cohort_ci["deaths_lost"][1],
    }

    return {
        "extrapolation_factor": ex_factor,
        "current_total_scans": ex_current_total,
        "target_pct": float(target_pct),
        "per_n_scans": per_n,
        "modality_filter": (sweep or {}).get("modality_filter"),
        "modality_region_map": _rehome_dental_cbct_map(
            (sweep or {}).get("modality_region_map"), _cbct_key,
            _dental_cbct_present),
        # Carried through so plots can state whether the ICRP nominal
        # detriment reference is directly comparable to the simulated QALY.
        "QALY_mode": str((sweep or {}).get("QALY_mode", "full")).lower(),
        "QALY_discount_rate": float((sweep or {}).get("QALY_discount_rate", 0.0) or 0.0),
        "per_region_age": per_ra,
        "per_region": per_region,
        "cohort_total": cohort,
    }


def _format_dose_psv(value: float) -> str:
    """Adaptive-precision formatter for an annual collective effective
    dose in person Sv/yr.  A single fixed number of decimals (e.g. one)
    rounds every low-dose modality (radiography, mammography, dental
    CBCT, ...) down to an uninformative "0.0".  Scaling to a different
    unit (mSv) doesn't fix this either -- the real problem is precision,
    not the unit -- so instead the number of decimals grows as the
    value shrinks, always keeping roughly two significant figures,
    while staying in the same unit (person Sv/yr) throughout.  Scientific
    notation is a last resort, used only below 1e-4 (what four decimals
    cannot show).
    """
    v = abs(value)
    if v == 0:
        return "0.0"
    if v >= 1:
        return f"{value:.1f}"
    if v >= 0.1:
        return f"{value:.2f}"
    if v >= 0.01:
        return f"{value:.3f}"
    if v >= 0.0001:
        return f"{value:.4f}"
    return f"{value:.1e}"


def _fmt_annual_qaly(v: float) -> str:
    """Bar-label formatter for an annual QALY value or interval bound.

    Zero is plain ``"0"`` (never ``0.00e+00``); three decimals normally,
    four when the value would otherwise read ``0.000``, and scientific
    notation only for non-zero values below 1e-4.
    """
    v = float(v)
    if v == 0.0:
        return "0"
    a = abs(v)
    if a < 1e-4:
        return f"{v:.2e}"
    if a < 1e-3:
        return f"{v:.4f}"
    return f"{v:.3f}"


def _fit_dose_texts_inside(fig, ax, legend, dose_texts,
                          legend_anchor_y: float = 0.55,
                          pad_px: float = 10.0,
                          max_right_frac: float = 0.995,
                          max_iter: int = 12) -> None:
    """Keep every dose annotation inside the axes frame and clear of the
    legend, on one line.

    Two stages, in order of how little they disturb the chart:

    1. *Grow the right frame.*  The axes is widened to the right and the
       x limit is extended by exactly the matching data range, so the
       pixels-per-data scale is unchanged: bars, error bars and labels
       stay exactly where they were and only the frame (and the
       axes-anchored legend) moves right.  The legend is re-anchored if
       that would push it past the figure edge.
    2. *Widen the data range* (fallback, only if stage 1 hit the figure
       edge).  This does shift the labels leftwards, so it is a last
       resort.  Each annotation sits at a fixed offset in points from a
       data-anchored position, so only the anchor moves with ``xlim``
       while the gap and the text width do not -- a single proportional
       rescale therefore undershoots (badly for a wide label on a short
       bar, e.g. a lone mammography row), hence the exact per-label
       solve plus re-measure loop.
    """
    if not dose_texts:
        return

    # Stage 1: move the right spine out, leaving the content put
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w = float(fig.get_size_inches()[0] * fig.dpi)
    ax_bbox = ax.get_window_extent(renderer=renderer)
    text_x1 = max(t.get_window_extent(renderer=renderer).x1
                  for t, _ in dose_texts)
    overflow = text_x1 - (ax_bbox.x1 - pad_px)
    if overflow > 0.5 and ax_bbox.width > 0:
        grow = min(overflow, fig_w * max_right_frac - ax_bbox.x1)
        if grow > 0.5:
            lo, hi = ax.get_xlim()
            px_per_data = ax_bbox.width / float(hi - lo)
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0,
                             pos.width + grow / fig_w, pos.height])
            ax.set_xlim(lo, lo + (ax_bbox.width + grow) / px_per_data)
            # An axes-anchored legend just moved right with the frame --
            # pull it back if it now overhangs the figure.  A figure
            # legend (axes is None) does not move and must not be
            # re-anchored to the axes.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_bbox = legend.get_window_extent(renderer=renderer)
            if (getattr(legend, "axes", None) is not None
                    and legend_bbox.x1 > fig_w - pad_px):
                new_ax_bbox = ax.get_window_extent(renderer=renderer)
                legend.set_bbox_to_anchor(
                    ((fig_w - pad_px - new_ax_bbox.x0) / new_ax_bbox.width,
                     legend_anchor_y),
                    transform=ax.transAxes)

    # Stage 2: fallback data-range widening
    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        ax_bbox = ax.get_window_extent(renderer=renderer)
        lo, hi = ax.get_xlim()
        span = float(hi - lo)
        if span <= 0 or ax_bbox.width <= 0:
            return
        needed_hi = hi
        for text, anchor_x in dose_texts:
            tbbox = text.get_window_extent(renderer=renderer)
            blocked_by_legend = (tbbox.y1 >= legend_bbox.y0
                                 and tbbox.y0 <= legend_bbox.y1)
            limit_x = ax_bbox.x1
            if blocked_by_legend:
                limit_x = min(limit_x, legend_bbox.x0)
            limit_x -= pad_px
            if tbbox.x1 <= limit_x:
                continue
            # Split the label's right edge into the part that scales with
            # the data range (its anchor) and the part that does not.
            anchor_px = ax.transData.transform((anchor_x, 0.0))[0] - ax_bbox.x0
            fixed_px = (tbbox.x1 - ax_bbox.x0) - anchor_px
            target_anchor_px = (limit_x - ax_bbox.x0) - fixed_px
            if target_anchor_px <= 1.0 or anchor_px <= 0:
                continue
            needed_hi = max(needed_hi, lo + span * (anchor_px / target_anchor_px))
        if needed_hi <= hi + 1e-12:
            return
        ax.set_xlim(lo, needed_hi)


def _display_region_bar(reg: str, mod_code: str = "") -> str:
    # Dental-imaging subtypes get their own short label (Panoramic,
    # CBCT, Kefalometry, Intraoral) instead of collapsing to one
    # generic "Dental" -- they are now separate rows within the
    # "Dental imaging" group, so a shared label would make them
    # indistinguishable.  "Dental CBCT" shows as "Dental" inside the
    # CBCT group (raw-CBCT dental exams); under any other modality it
    # keeps "Dental CBCT".
    if _norm_modality(mod_code) == "CBCT" and reg == "Dental CBCT":
        reg = "Dental"
    elif reg in _DENTAL_IMAGING_SUBTYPE_REGIONS:
        reg = _DENTAL_SUBTYPE_DISPLAY.get(reg, reg)
    reg = _REGION_DISPLAY_SHORT.get(reg, reg)
    if len(reg) > 16:
        reg = "\n".join(textwrap.wrap(reg, width=16, break_long_words=False))
    return reg


@_rq_fonts
def plot_annual_QALY_bar(
    benefit: Dict[str, Any],
    fig: Any,
    *,
    burden_mode: bool = False,
) -> None:
    """Two panels sharing a region axis: benefit gained, and the avoided
    dose that bought it.

    With ``burden_mode`` the same figure shows the current practice
    instead: panel A is the annual QALYs lost and panel B the annual
    collective effective dose (see :func:`plot_annual_burden_bar`).

    **Panel A (left, wide) — how much is gained.**  One bar per region,
    the actual expected benefit of the dose reduction, stacked by age
    decade and computed with age-specific REIC/LAR coefficients, with
    95% CI whiskers.  Each bar is labelled with its QALY and 95% CI.

    **Panel B (right, narrow) — the dose that bought it.**  One bar per
    region giving the avoided collective effective dose (person Sv/yr):
    panel A's benefit divided out, back to the denominator.  The dose is
    deterministic, so the bars carry no whiskers.

    Why the dose panel exists.  Panel A's ordering is dominated by
    workload -- the busiest, highest-dose regions come top -- which tells
    the reader where the benefit is but not whether dose is a good proxy
    for it.  Panel B puts the avoided dose on an axis of its own beside
    the benefit it bought, and the finding is visible side by side:
    regions with similar avoided dose yield very different QALY, so
    benefit per unit avoided dose is not constant and avoided collective
    effective dose is not by itself a sufficient statistic for the
    health benefit it buys.  Panel A's age stacking shows the mechanism
    -- benefit concentrates in younger patients, and the regions differ
    in their age mix.

    No external coefficient appears in either panel; the figure compares
    the simulation against itself.  See the in-body comment at the top of
    the function for why no ICRP 152 nominal reference line is drawn,
    and the module-level ``ICRP152_*`` constants for its derivation.

    The title names the quantity shown and the dose-reduction target
    only.

    A positive ``benefit["target_pct"]`` is a dose *increase*.  Every
    "saved" quantity is then negative -- harm added rather than benefit
    avoided -- so the figure is drawn in magnitudes and the wording flips
    to "lost"/"added" throughout.

    Modality names are shown as compact headers right-aligned with the
    region tick labels, sitting directly on the black separator line
    above each group (the same styling as the region/age heatmap).  No
    header rows are inserted — the y-axis contains only region labels.
    """
    import matplotlib.ticker as mticker
    import matplotlib.transforms as mtransforms
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Single font scale for the whole figure: the figure must read from
    # the back row when pasted into a slide, so every hard-coded font
    # size below is multiplied by this.
    _FS = 1.6

    fig.clear()
    fig.set_layout_engine(None)
    # The GUI creates a bare Figure and calls this function before
    # wrapping it in FigureCanvasTkAgg, so fig.canvas is a generic
    # FigureCanvasBase with no get_renderer() -- needed below for the
    # two-pass margin fitting.  Attaching a throwaway Agg canvas here
    # gives it one; the GUI's later FigureCanvasTkAgg(fig, ...) call
    # replaces fig.canvas again afterwards, so this has no side effect
    # on the final rendered widget.
    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    per_region = list((benefit or {}).get("per_region") or [])
    per_ra = list((benefit or {}).get("per_region_age") or [])
    target_pct = float((benefit or {}).get("target_pct", -10.0))
    modality_region_map = (benefit or {}).get("modality_region_map") or {}

    # A positive ``target_pct`` is a dose INCREASE, for which every
    # "saved" field comes back negative: harm added rather than benefit
    # avoided.  Everything below -- the descending sort, the "> 0" filters
    # that pick the active age decades, the bar geometry and the axis
    # limits -- is written for positive magnitudes, and all values share
    # one sign, so normalise to magnitudes here and carry the direction in
    # ``_is_incr`` for the wording.  Rows are shallow-copied rather than
    # mutated: the caller's ``benefit`` dict is also the source for the
    # window's text summary.
    _is_incr = target_pct > 0.0 and not burden_mode
    if _is_incr:
        _FLIP = ("QALY_saved_annual", "QALY_saved_annual_ci_lo",
                 "QALY_saved_annual_ci_hi")
        def _abs_rows(rows):
            out = []
            for _r in rows:
                _c = dict(_r)
                for _k in _FLIP:
                    if _k in _c and _c[_k] is not None:
                        _c[_k] = abs(float(_c[_k]))
                # abs() alone would leave lo > hi once both are negated.
                _lo = _c.get("QALY_saved_annual_ci_lo")
                _hi = _c.get("QALY_saved_annual_ci_hi")
                if _lo is not None and _hi is not None and _lo > _hi:
                    _c["QALY_saved_annual_ci_lo"] = _hi
                    _c["QALY_saved_annual_ci_hi"] = _lo
                out.append(_c)
            return out
        per_region = _abs_rows(per_region)
        per_ra = _abs_rows(per_ra)

    # Wording that follows the direction of the dose change.  Panel A's
    # quantity is QALYs gained (reduction) or lost
    # (increase); panel B's dose is avoided or added.
    _w_saved = "lost" if (_is_incr or burden_mode) else "gained"
    _w_avoided = "added" if _is_incr else "avoided"

    if not per_region or not per_ra:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#888")
        return

    age_labels = list(AGE_DECADE_LABELS)
    _cb_palette = [
        "#08306B", "#2171B5", "#6BAED6", "#C6DBEF", "#FEE0D2",
        "#FDAE6B", "#FD8D3C", "#E6550D", "#A63603", "#7F2704",
    ]
    age_color = {lbl: _cb_palette[i % len(_cb_palette)]
                 for i, lbl in enumerate(age_labels)}

    # Build modality groups

    all_regions = {r["region"] for r in per_region}
    all_mod_regions = {(r.get("modality", ""), r["region"]) for r in per_region}
    modality_groups: List[Tuple[str, List[str]]] = []
    if modality_region_map:
        for mod_code in sorted(modality_region_map.keys(), key=_mod_sort_key):
            mod_regs = [r for r in all_regions
                        if r in modality_region_map.get(mod_code, [])
                        and (mod_code, r) in all_mod_regions]
            if mod_regs:
                modality_groups.append((mod_code, mod_regs))
        covered = {r for _, rl in modality_groups for r in rl}
        leftover = [reg for (mod, reg) in all_mod_regions if reg not in covered]
        if leftover:
            modality_groups.append(("", leftover))
    else:
        _by_mod: Dict[str, List[str]] = {}
        for mod, reg in all_mod_regions:
            _by_mod.setdefault(mod, []).append(reg)
        for mod_code in sorted(_by_mod.keys(), key=_mod_sort_key):
            modality_groups.append((mod_code, _by_mod[mod_code]))

    _QALY_order = {(r.get("modality", ""), r["region"]): r["QALY_saved_annual"] for r in per_region}
    for gi, (mod_code, mod_regs) in enumerate(modality_groups):
        modality_groups[gi] = (mod_code, sorted(
            mod_regs, key=lambda r: -_QALY_order.get((mod_code, r), 0.0)))

    # Include CI upper bounds in the global max so that whiskers which
    # extend past the bar total (e.g. Chest, where the 95% CI is wide
    # relative to the mean) are not clipped by the x-axis limit below.
    _ci_hi_lookup_early = {(r.get("modality", ""), r["region"]):
                            float(r.get("QALY_saved_annual_ci_hi", 0.0) or 0.0)
                            for r in per_region}
    _global_max = max(
        (max(_QALY_order.get((mod_code, r), 0.0), _ci_hi_lookup_early.get((mod_code, r), 0.0))
         for mod_code, rl in modality_groups for r in rl),
        default=1e-9,
    )

    # Determine active age groups.
    active_ages = []
    for lbl in age_labels:
        has = any(
            float(r.get("QALY_saved_annual", 0.0)) > 0
            for r in per_ra
            if r["region"] in all_regions and r.get("age_group") == lbl
        )
        if has:
            active_ages.append(lbl)

    # Build flat region list with group bounds
    # No header rows — y-axis contains only region labels.
    # Add a small vertical gap between modality groups for visual separation.
    _GROUP_GAP = 0.8  # extra space between modality groups
    # A wrapped (two-line) tick label on the last row of a group hangs
    # about half a row below its row centre, into the gap where the next
    # group's header sits centred on the separator; widen that gap so
    # the label and the header cannot touch.
    _GROUP_GAP_WRAPPED = 1.4


    regions_ordered: List[str] = []
    regions_mod_ordered: List[str] = []  # parallel mod_code per region
    group_bounds: List[Tuple[float, float, str]] = []  # (y_start, y_end, mod_label)
    y_positions_list: List[float] = []
    y_pos = 0.0
    for gi, (mod_code, mod_regs) in enumerate(modality_groups):
        mod_label = _modality_name(mod_code) if mod_code else "All modalities"
        mod_label = mod_label.split(" (")[0] if " (" in mod_label else mod_label
        if mod_label == "MG":
            mod_label = "Mammography"
        # NB: "CBCT" is deliberately left as-is (not renamed to "Dental
        # CBCT") -- the group can now contain literal non-dental CBCT
        # regions (Ankle, Wrist, ...) alongside "Dental CBCT", so a
        # dental-only header would be misleading.
        start_y = y_pos
        for reg in mod_regs:
            regions_ordered.append(reg)
            regions_mod_ordered.append(mod_code)
            y_positions_list.append(y_pos)
            y_pos += 1.0
        group_bounds.append((start_y, y_pos - 1.0, mod_label))
        if gi < len(modality_groups) - 1:
            # gap only between groups, not after last
            _last_wrapped = "\n" in _display_region_bar(mod_regs[-1], mod_code)
            y_pos += _GROUP_GAP_WRAPPED if _last_wrapped else _GROUP_GAP

    n_reg = len(regions_ordered)
    y_positions = np.array(y_positions_list)

    # Build data matrix.
    mod_reg_idx = {}
    di = 0
    for gi, (mod_code, mod_regs) in enumerate(modality_groups):
        for reg in mod_regs:
            mod_reg_idx[(mod_code, reg)] = di
            di += 1
    age_idx = {lbl: j for j, lbl in enumerate(active_ages)}
    M = np.zeros((n_reg, len(active_ages)), dtype=float)
    for r in per_ra:
        ri = mod_reg_idx.get((r.get("modality", ""), r["region"]))
        aj = age_idx.get(r["age_group"])
        if ri is not None and aj is not None:
            # Accumulate: several sweep cells can share a displayed
            # (modality, region, age) once "Dental CBCT" coded under
            # different raw modalities is pooled into the CBCT group.
            M[ri, aj] += float(r["QALY_saved_annual"])

    # No external coefficient is used anywhere in this figure.  Both
    # panels are the simulation compared against itself: panel A is the
    # modelled benefit, panel B is that benefit divided by the avoided
    # collective effective dose that bought it.
    #
    # No ICRP 152 nominal-coefficient reference line is drawn.  The
    # nominal value (5.486 %/Sv cancer-only detriment x 15 yr average
    # YLL = 0.823 QALY/Sv, see the module-level ``ICRP152_*`` constants)
    # sits ~6x above every modelled bar for this cohort, and that gap is
    # dominated by age at exposure -- the dose-weighted mean age is 68 y,
    # while the nominal value is an all-ages population average.  Showing
    # it would invite the reading that effective dose is mis-calibrated
    # by 6x, when the honest reading is that an all-ages coefficient does
    # not apply to an elderly cohort.  The spread between the bars needs
    # no such reference to be meaningful.
    _dose_reduction_frac = 1.0 if burden_mode else abs(target_pct) / 100.0

    # Per-region CI lookup for whiskers.
    _ci_lookup = {(r.get("modality", ""), r["region"]): r for r in per_region}

    # Two panels sharing the region axis.  The benefit panel is scaled
    # by the actual QALY values alone, and the dose panel gets an axis
    # of its own on which the differences between regions are actually
    # visible.
    _gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.07)
    ax = fig.add_subplot(_gs[0, 0])
    ax_dose = fig.add_subplot(_gs[0, 1], sharey=ax)

    # Inverse lookup: y-position index → (modality, region).
    _regions_inv = []
    for _gi, (_mc, _mrs) in enumerate(modality_groups):
        for _reg in _mrs:
            _regions_inv.append((_mc, _reg))

    # Per-region avoided collective effective dose (person Sv/yr): the quantity
    # drawn in panel B.  Taken from
    # per_region rather than by summing a per-(region, age) matrix --
    # active_ages is filtered to decades that produced non-zero QALY, so
    # summing the matrix would silently drop dose delivered in decades
    # that happened to yield no modelled benefit, understating the dose.
    dose_saved = np.zeros(n_reg, dtype=float)
    for _i, (_mc, _reg) in enumerate(_regions_inv):
        _ci_d = _ci_lookup.get((_mc, _reg), {})
        dose_saved[_i] = (float(_ci_d.get("dose_sv_annual", 0.0) or 0.0)
                          * _dose_reduction_frac)

    # Panel A: actual benefit, one bar per region, stacked by age
    # Only the actual benefit is drawn here.  It is the only quantity in
    # the figure that uses age-specific REIC/LAR coefficients evaluated
    # at the real age at exposure, so it is the only one entitled to the
    # age palette; the dose-based prediction carries no per-scan age
    # information (see the reference-line note above) and colouring it by
    # age would falsely imply otherwise.  With the prediction in
    # panel B the bar can take the full row height, and the x-limit is
    # set by the actual values alone.
    _BAR_H = 0.72
    y_act = y_positions

    totals = M.sum(axis=1)

    left_a = np.zeros(n_reg)
    for j, age_lbl in enumerate(active_ages):
        wa = M[:, j]
        ax.barh(y_act, wa, left=left_a, height=_BAR_H,
                color=age_color[age_lbl],
                edgecolor="white", linewidth=0.4, zorder=2)
        left_a += wa

    # Draw 95% CI whiskers on the actual-benefit bar for each region.
    # A thin white underlay is drawn first so the CI line remains
    # visible even when it crosses dark bar segments.  Only the actual
    # bar carries a CI: the predicted bar is a deterministic rescaling
    # of the dose, not an independently sampled estimate.
    for i, (_mc, _reg) in enumerate(_regions_inv):
        ci = _ci_lookup.get((_mc, _reg), {})
        lo = float(ci.get("QALY_saved_annual_ci_lo", 0.0))
        hi = float(ci.get("QALY_saved_annual_ci_hi", 0.0))
        if hi > lo and totals[i] > 0:
            # White underlay for contrast against dark bars.
            ax.plot([lo, hi], [y_act[i], y_act[i]],
                    color="white", linewidth=4.4, zorder=4,
                    solid_capstyle="round")
            ax.plot([lo, hi], [y_act[i], y_act[i]],
                    color="black", linewidth=2.6, zorder=5,
                    solid_capstyle="round")
            ax.plot([lo, lo], [y_act[i] - 0.16, y_act[i] + 0.16],
                    color="black", linewidth=2.6, zorder=5)
            ax.plot([hi, hi], [y_act[i] - 0.16, y_act[i] + 0.16],
                    color="black", linewidth=2.6, zorder=5)

    # Set y-axis ticks and labels (regions only, no headers).  Long
    # region names are wrapped onto a second line -- matches the
    # region/age heatmap's y-axis label wrapping -- instead of running
    # into the left margin (``_display_region_bar`` is defined above the
    # layout loop, which needs it to size the inter-group gaps).
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_display_region_bar(r, m)
                        for r, m in zip(regions_ordered, regions_mod_ordered)],
                       fontsize=11 * _FS)
    ax.invert_yaxis()
    _TOP_PAD = _GROUP_GAP
    # Same wrapped-label allowance at the bottom edge, so a two-line
    # label on the very last row does not run into the x tick labels.
    _BOT_PAD = ((_GROUP_GAP_WRAPPED - _GROUP_GAP) / 2.0
                if "\n" in _display_region_bar(regions_ordered[-1],
                                               regions_mod_ordered[-1])
                else 0.0)
    ax.set_ylim(y_positions[-1] + 0.5 + _BOT_PAD, y_positions[0] - 0.5 - _TOP_PAD)
    # Headroom for the two-line label column past each bar.  Smaller than
    # it needed to be when a dose-predicted bar shared the panel and could
    # run several times the longest actual bar; _fit_dose_texts_inside
    # below still tops it up if a label would overflow.
    ax.set_xlim(0, _global_max * 1.26)
    # Prune the top tick label so it cannot run into panel B's zero
    # across the narrow shared boundary.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="upper"))
    ax.set_xlabel(f"QALYs {_w_saved} annually",
                  fontsize=12 * _FS, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12 * _FS)
    for _lbl in ax.get_xticklabels():
        _lbl.set_fontweight("bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    # One thin, light frame around the panel -- matches the region/age
    # heatmap's box styling instead of matplotlib's default heavier
    # black spines, for visual consistency across the report's figures.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
        spine.set_linewidth(0.8)

    # Horizontal separator lines at every modality boundary, each
    # centred in the middle of its blank gap (above first group,
    # between groups, and below last group) for equal spacing above
    # and below every line.
    _sep_kw = dict(color="black", linewidth=1.6, linestyle="-", zorder=1)
    first_start = group_bounds[0][0] if group_bounds else 0.0
    top_sep_y = first_start - 0.5 - _TOP_PAD / 2.0
    ax.axhline(y=top_sep_y, **_sep_kw)
    sep_ys: List[float] = []
    for i in range(1, len(group_bounds)):
        prev_end_y = group_bounds[i - 1][1]
        curr_start_y = group_bounds[i][0]
        sep_y = (prev_end_y + curr_start_y) / 2.0
        sep_ys.append(sep_y)
        ax.axhline(y=sep_y, **_sep_kw)
    last_end = group_bounds[-1][1] if group_bounds else 0.0
    ax.axhline(y=last_end + 0.5 + _BOT_PAD, **_sep_kw)

    # Modality names as compact headers right-aligned with the region
    # tick labels, sitting directly on each group's separator line --
    # rather than a dedicated left-side column -- freeing horizontal
    # space, matching the region/age heatmap's modality-header styling.
    mod_label_texts: List[Any] = []
    mod_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for gi, (_, _, mod_label) in enumerate(group_bounds):
        y_header = top_sep_y if gi == 0 else sep_ys[gi - 1]
        t = ax.text(-0.01, y_header, mod_label,
                    transform=mod_trans, ha="right", va="center",
                    fontsize=12.2 * _FS, fontweight="bold", color="#1a1a1a",
                    clip_on=False)
        mod_label_texts.append(t)

    # Annotate each region on one shared x anchor, placed past whichever
    # of (bar, CI whisker) reaches furthest right, so no label can overlap
    # a bar or a CI line and the numbers form a tidy column.  One line
    # per region, centred on the bar: the benefit and its 95% CI.  The
    # avoided dose that bought it is shown in panel B instead.
    _offset = _global_max * 0.015
    _dose_texts: List[Tuple[Any, float]] = []
    for i, total in enumerate(totals):
        if total <= 0:
            continue
        _mc, _reg = _regions_inv[i]
        ci = _ci_lookup.get((_mc, _reg), {})
        ci_lo = float(ci.get("QALY_saved_annual_ci_lo", total))
        ci_hi = float(ci.get("QALY_saved_annual_ci_hi", total))
        tfmt = _fmt_annual_qaly(total)
        has_ci = ci_hi > ci_lo and (ci_hi - ci_lo) / max(total, 1e-12) > 1e-6
        if has_ci:
            label = (f"{tfmt} [{_fmt_annual_qaly(ci_lo)}, "
                     f"{_fmt_annual_qaly(ci_hi)}]")
        else:
            label = tfmt
        label_x = max(total, ci_hi if has_ci else 0.0) + _offset

        t_act = ax.text(label_x, y_act[i], label,
                        va="center", ha="left", fontsize=11 * _FS,
                        color="#333", fontweight="bold", zorder=3)
        _dose_texts.append((t_act, label_x))

    # Panel B: avoided collective effective dose
    # The avoided collective effective dose that bought panel A's
    # benefit, one bar per region.  The dose is deterministic, so the
    # bars carry no whiskers.
    _DOSE_COLOR = "#9E9E9E"   # neutral grey: this panel is not age-resolved
    ax_dose.barh(y_act, dose_saved, height=_BAR_H, color=_DOSE_COLOR,
                 edgecolor="white", linewidth=0.4, zorder=2)

    # Value labels sit just past the bar tips in the otherwise dead space
    # to the right of each row.
    _dose_max = float(dose_saved.max() if n_reg else 0.0)
    for i in range(n_reg):
        if dose_saved[i] <= 0:
            continue
        ax_dose.text(dose_saved[i] + _dose_max * 0.02, y_act[i],
                     _format_dose_psv(dose_saved[i]), va="center", ha="left",
                     fontsize=9.5 * _FS, color="#333", fontweight="bold",
                     zorder=3)
    ax_dose.set_xlim(0, _dose_max * 1.22)
    ax_dose.set_xlabel(
        "Collective effective dose\nannually\n(person Sv/yr)" if burden_mode
        else f"Collective effective dose\n{_w_avoided} annually\n(person Sv/yr)",
        fontsize=11 * _FS, fontweight="bold")
    ax_dose.tick_params(axis="x", labelsize=11 * _FS)
    for _lbl in ax_dose.get_xticklabels():
        _lbl.set_fontweight("bold")
    # y ticks are shared with panel A, so only that panel labels them.
    ax_dose.tick_params(axis="y", left=False, labelleft=False)
    ax_dose.grid(True, axis="x", alpha=0.3, linestyle="--", zorder=0)
    ax_dose.set_axisbelow(True)
    for spine in ax_dose.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
        spine.set_linewidth(0.8)
    # Repeat panel A's modality separators so the rows read across.
    ax_dose.axhline(y=top_sep_y, **_sep_kw)
    for _sy in sep_ys:
        ax_dose.axhline(y=_sy, **_sep_kw)
    ax_dose.axhline(y=last_end + 0.5, **_sep_kw)


    # Title text (built here, drawn during layout)
    # The title is descriptive only -- quantity and dose target.  The
    # fold-spread of panel B is left for the reader to see in the panel.
    _title_txt = ("QALYs Lost and Collective Effective Dose Annually"
                  if burden_mode else
                  f"QALYs {'Lost' if _is_incr else 'Gained'} and "
                  f"Collective Effective Dose {'Added' if _is_incr else 'Avoided'} "
                  f"Annually from {abs(float(target_pct)):g}% Dose "
                  f"{'Increase' if _is_incr else 'Reduction'}")

    _sub = ""

    # Margin fitting
    # Every margin is solved from rendered pixel extents rather than from
    # figure fractions.  A fraction that looks right at one figure size
    # silently fails at another -- the same 9.5 pt subtitle occupies a
    # larger fraction of a short figure, so a hard-coded ``top`` runs the
    # text into the panel frame whenever the caller shrinks the figure or
    # the region count drops.  Pixel extents are size-invariant.
    fig.subplots_adjust(left=0.20, right=0.96, top=0.90, bottom=0.08)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_px, _fig_h_px = fig.get_size_inches() * fig.dpi

    # Left: widest region tick label / modality header.
    tick_w_px = max((t.get_window_extent(renderer=renderer).width
                      for t in ax.get_yticklabels()), default=0.0)
    mod_w_px = max((t.get_window_extent(renderer=renderer).width
                     for t in mod_label_texts), default=0.0)
    _GAP_PX = 14.0
    required_left = (_GAP_PX + max(tick_w_px, mod_w_px) + _GAP_PX) / fig_w_px
    required_left = min(max(required_left, 0.10), 0.40)
    _right = 0.96

    # Title/subtitle centred on the plotted panel itself (region labels
    # + bars), not the full figure -- matches the region/age heatmap's
    # title styling and avoids visibly offsetting the title from the
    # content it describes when the label margin is wide.
    _title_center_x = (required_left + _right) / 2.0
    _PAD_PX = 9.0

    # The legend sits below the panels rather than inside panel A:
    # groups are sorted independently, so with several modality groups
    # no corner of panel A is guaranteed free (a large mammography bar
    # at the bottom collided with a lower-right legend).
    from matplotlib.patches import Patch as _Patch
    _legend_handles = [
        _Patch(facecolor=age_color[lbl], edgecolor="none", label=lbl)
        for lbl in age_labels
    ]
    _legend = fig.legend(
        handles=_legend_handles, title="Age group",
        ncol=5, loc="lower center",
        bbox_to_anchor=(_title_center_x, 6.0 / _fig_h_px),
        fontsize=11 * _FS, title_fontsize=12 * _FS,
        frameon=False, handlelength=1.6, handleheight=1.1,
        columnspacing=1.4, handletextpad=0.5, borderaxespad=0.2)

    _sup = fig.suptitle(
        _title_txt, x=_title_center_x, y=1.0 - _PAD_PX / _fig_h_px,
        va="top", fontsize=13.5 * _FS, fontweight="bold")
    fig.canvas.draw()
    # Fit the title to the figure width.  It is a single line whose length
    # depends on the reduction percentage and the font scale, so it can
    # overrun a narrow figure -- and because it is
    # centred it then gets clipped at BOTH ends, which reads as a
    # deliberately short title rather than as an error.
    #
    # Centred text anchored at ``x`` can only use twice its distance to the
    # nearer edge, so the usable width is not the full figure width.  Two
    # remedies, cheapest first: re-anchor to the figure centre (which
    # maximises that distance, at the cost of the panel-centred alignment),
    # then shrink.  Floor at 9 pt; below that a clipped larger title is the
    # lesser evil.
    def _avail_px_for(_cx_frac: float) -> float:
        _cx = _cx_frac * fig_w_px
        return 2.0 * max(0.0, min(_cx - _PAD_PX, fig_w_px - _PAD_PX - _cx))

    def _title_w_px() -> float:
        fig.canvas.draw()
        return _sup.get_window_extent(renderer=renderer).width

    if _title_w_px() > _avail_px_for(_title_center_x):
        if _title_w_px() <= _avail_px_for(0.5):
            _sup.set_x(0.5)
        else:
            _sup.set_x(0.5)
            for _fs in (12.5 * _FS, 11.5 * _FS, 10.5 * _FS, 9.5 * _FS, 9.0 * _FS):
                _sup.set_fontsize(_fs)
                if _title_w_px() <= _avail_px_for(0.5):
                    break
    _block_y0_px = _sup.get_window_extent(renderer=renderer).y0
    if _sub:
        _sub_text = fig.text(
            _title_center_x, (_block_y0_px - 5.0) / _fig_h_px, _sub,
            ha="center", va="top", fontsize=9.5 * _FS, color="#555")
        fig.canvas.draw()
        _block_y0_px = _sub_text.get_window_extent(renderer=renderer).y0

    # Bottom: tick labels plus the x-axis label, taking whichever panel
    # needs more (panel A's label is a point larger than panel B's).
    def _axis_h_px(_a) -> float:
        _t = max((t.get_window_extent(renderer=renderer).height
                  for t in _a.get_xticklabels()), default=0.0)
        _l = _a.xaxis.get_label().get_window_extent(renderer=renderer).height
        return _t + _l

    fig.canvas.draw()
    _legend_h_px = _legend.get_window_extent(renderer=renderer).height
    _bottom_px = (_PAD_PX + max(_axis_h_px(ax), _axis_h_px(ax_dose))
                  + 14.0 + _legend_h_px + 8.0)
    fig.subplots_adjust(
        left=required_left, right=_right,
        top=float(np.clip((_block_y0_px - _PAD_PX) / _fig_h_px, 0.35, 0.95)),
        bottom=float(np.clip(_bottom_px / _fig_h_px, 0.05, 0.45)),
    )
    for _lbl in ax.get_xticklabels():
        _lbl.set_fontweight("bold")

    # ── Safety pass: extend the x-axis if needed so no label overflows
    # the panel frame -------------------------------------------------
    # The label texts' pixel position scales with the data xlim, so
    # widening xlim pulls an overflowing label back inside the frame.
    # Run last, so it sees the final panel geometry.
    #
    # Stage 1 of the helper widens the axes itself, which in a two-panel
    # layout would run panel A straight over panel B.  Cap the growth at
    # panel B's left edge so it can only use slack that is actually free,
    # and fall back to widening the data range beyond that.
    _dose_x0 = float(ax_dose.get_position().x0)
    _fit_dose_texts_inside(fig, ax, _legend, _dose_texts,
                           max_right_frac=max(_dose_x0 - 0.012, 0.30))


def compute_annual_burden(
    sweep: Dict[str, Any],
    extrapolate_to_exams: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate annual QALYs lost / cancers / deaths (baseline, no dose reduction).

    Combines the per-*per_n_scans* baseline burden with the local annual
    scan count to produce a hospital-level annual burden estimate.

    Returns a dict with:
    * ``per_region_age``: list of {region, age_group, age_decade_idx,
      scans_per_year, dose_sv_annual, dose_msv_annual,
      QALY_annual, cases_annual, deaths_annual}
    * ``per_region``: list of {region, QALY_annual, cases_annual,
      deaths_annual, total_scans, total_scans_all, dose_sv_annual,
      dose_msv_annual} sorted by QALY_annual descending.
    * ``cohort_total``: {QALY_annual, cases_annual, deaths_annual,
      total_scans, total_scans_all, dose_sv_annual, dose_msv_annual,
      QALY_annual_sim, cases_annual_sim, deaths_annual_sim} —
      ``*_annual_sim`` is the Monte Carlo mean of the pooled simulated
      annual series when ``attach_simulated_ranges`` ran; NaN otherwise.

    ``total_scans`` / ``dose_sv_annual`` / ``dose_msv_annual`` count the
    risk-MODELLED exams only (``modelled_scans_per_year`` /
    ``modelled_dose_sv_per_year`` -- non-empty ``_cancer_doses``), matching
    the Summary tab's section-2 denominator; ``total_scans_all`` keeps the
    all-exams count.  Sweeps that predate the ``modelled_*`` keys fall back
    to the all-exams values.  The outcome totals (``*_annual``) are
    unchanged by this -- they are scaled by ``scans_per_year``.
    """
    rows = list((sweep or {}).get("per_region_age") or [])
    per_n = int((sweep or {}).get("per_n_scans") or 100_000)
    ex_factor, ex_current_total = _extrapolation_factor(sweep, extrapolate_to_exams)
    # Pooled simulated annual-outcome series, when attach_simulated_ranges
    # ran on this sweep — the intervals below then become the central 95%
    # range of simulated annual outcomes rather than the analytical
    # range of the mean.
    _sim = (sweep or {}).get("_sim")

    per_ra: List[Dict[str, Any]] = []
    region_acc: Dict[Tuple[str, str], Dict[str, float]] = {}

    _cbct_key = _cbct_modality_key(sweep)
    _dental_cbct_present = False

    for r in rows:
        reg_name = str(r.get("region", ""))
        # "Unknown/unspecified region" (blank/unrecognised Scan Regions) is
        # intentionally kept here -- for CT it still carries real per-organ
        # doses from the Excel columns, so it contributes a genuine annual
        # burden total -- only Non-Dose (QA, cancellations, mislabelled
        # non-CT) is excluded.  An unlocalised radiography/CBCT row has no
        # dose model and is dropped by the has_risk_model test.
        if reg_name.startswith("Non-Dose"):
            continue
        # Regions with no dose/risk model (e.g. non-dental CBCT body parts)
        # are shown as n/a in the heatmap and carry no modelled burden, so
        # they are left out of the annual totals rather than counted as zero.
        if r.get("has_risk_model", True) is False:
            continue
        scans = float(r.get("scans_per_year", 0.0))
        if scans <= 0.0:
            continue
        scans = scans * ex_factor
        QALY_per_n = float(r.get("baseline_QALY_per_100k", 0.0))
        cases_per_n = float(r.get("baseline_cases_per_100k", 0.0))
        deaths_per_n = float(r.get("baseline_deaths_per_100k", 0.0))

        annual_factor = scans / per_n
        QALY_annual = QALY_per_n * annual_factor
        cases_annual = cases_per_n * annual_factor
        deaths_annual = deaths_per_n * annual_factor

        QALY_ci = _per_100k_ci_to_annual(
            float(r.get("baseline_QALY_ci_lo_per_100k", 0.0) or 0.0),
            float(r.get("baseline_QALY_ci_hi_per_100k", 0.0) or 0.0),
            annual_factor)
        cases_ci = _per_100k_ci_to_annual(
            float(r.get("baseline_cases_ci_lo_per_100k", 0.0) or 0.0),
            float(r.get("baseline_cases_ci_hi_per_100k", 0.0) or 0.0),
            annual_factor)
        deaths_ci = _per_100k_ci_to_annual(
            float(r.get("baseline_deaths_ci_lo_per_100k", 0.0) or 0.0),
            float(r.get("baseline_deaths_ci_hi_per_100k", 0.0) or 0.0),
            annual_factor)

        mod = str(r.get("modality", "") or "")
        reg = str(r.get("region", "?"))
        age = str(r.get("age_group", "?"))
        di = int(r.get("age_decade_idx", 0))
        # "Dental CBCT" under other raw modalities is pooled into the CBCT
        # group (see compute_annual_benefit); ``acc["_rows"]`` keeps the
        # original row for the raw-modality-keyed simulated cell store.
        if reg == "Dental CBCT":
            mod = _cbct_key
            _dental_cbct_present = True

        # Displayed exam count / collective dose count the risk-MODELLED
        # exams only (non-empty ``_cancer_doses``), matching the Summary
        # tab's section-2 denominator.  ``modelled_*`` keys fall back to
        # the all-exams values for sweeps that predate them.
        modelled_scans = (float(r["modelled_scans_per_year"]) * ex_factor
                          if "modelled_scans_per_year" in r else scans)
        dose_sv_per_year = (float(r["modelled_dose_sv_per_year"])
                            if "modelled_dose_sv_per_year" in r
                            else float(r.get("dose_sv_per_year", 0.0) or 0.0)) * ex_factor
        dose_msv_annual = dose_sv_per_year * 1000.0

        per_ra.append({
            "modality": mod,
            "region": reg,
            "age_group": age,
            "age_decade_idx": di,
            "scans_per_year": scans,
            "dose_sv_annual": dose_sv_per_year,
            "dose_msv_annual": dose_msv_annual,
            "QALY_annual": QALY_annual,
            "cases_annual": cases_annual,
            "deaths_annual": deaths_annual,
            "QALY_annual_ci_lo": QALY_ci[0],
            "QALY_annual_ci_hi": QALY_ci[1],
            "cases_annual_ci_lo": cases_ci[0],
            "cases_annual_ci_hi": cases_ci[1],
            "deaths_annual_ci_lo": deaths_ci[0],
            "deaths_annual_ci_hi": deaths_ci[1],
        })

        acc = region_acc.setdefault((mod, reg), {
            "QALY_annual": 0.0,
            "cases_annual": 0.0,
            "deaths_annual": 0.0,
            "total_scans": 0.0,
            "total_scans_all": 0.0,
            "dose_sv_annual": 0.0,
            "dose_msv_annual": 0.0,
            "_var_QALY": 0.0,
            "_var_cases": 0.0,
            "_var_deaths": 0.0,
            "_rows": [],
        })
        acc["_rows"].append(r)
        acc["QALY_annual"] += QALY_annual
        acc["cases_annual"] += cases_annual
        acc["deaths_annual"] += deaths_annual
        acc["total_scans"] += modelled_scans
        acc["total_scans_all"] += scans
        acc["dose_sv_annual"] += dose_sv_per_year
        acc["dose_msv_annual"] += dose_msv_annual
        acc["_var_QALY"] += _ci_var(*QALY_ci)
        acc["_var_cases"] += _ci_var(*cases_ci)
        acc["_var_deaths"] += _ci_var(*deaths_ci)

    def _metric_range(series_dict, key):
        """(lo, hi) for *key* of a pooled series dict, or a NaN pair."""
        rng = _sim_range((series_dict or {}).get(key))
        if rng is None:
            return (float("nan"), float("nan"))
        return rng[0] * ex_factor, rng[1] * ex_factor

    def _metric_mean(series_dict, key):
        """Mean of *key* of a pooled series dict (x ex_factor), or NaN."""
        a = (series_dict or {}).get(key)
        if a is None or np.asarray(a).size == 0:
            return float("nan")
        return float(np.mean(np.asarray(a, dtype=np.float64))) * ex_factor

    per_region = []
    for (mod, reg), vals in region_acc.items():
        v = {k: val for k, val in vals.items() if not k.startswith("_")}
        v["modality"] = mod
        v["region"] = reg
        if _sim is not None:
            # Central 95% range of the pooled simulated annual outcome of
            # this (modality, region) — elementwise sum of its cells'
            # per-iteration series, percentiles in annual units.
            ser = _sim_cells_series(_sim, vals["_rows"])
            v["QALY_annual_ci_lo"], v["QALY_annual_ci_hi"] = _metric_range(
                ser, "qaly")
            v["cases_annual_ci_lo"], v["cases_annual_ci_hi"] = _metric_range(
                ser, "cases")
            v["deaths_annual_ci_lo"], v["deaths_annual_ci_hi"] = _metric_range(
                ser, "deaths")
        else:
            v["QALY_annual_ci_lo"], v["QALY_annual_ci_hi"] = _ci_from_var(
                vals["QALY_annual"], vals["_var_QALY"])
            v["cases_annual_ci_lo"], v["cases_annual_ci_hi"] = _ci_from_var(
                vals["cases_annual"], vals["_var_cases"])
            v["deaths_annual_ci_lo"], v["deaths_annual_ci_hi"] = _ci_from_var(
                vals["deaths_annual"], vals["_var_deaths"])
        per_region.append(v)
    per_region.sort(key=lambda x: x["QALY_annual"], reverse=True)

    _c_QALY = sum(v["QALY_annual"] for v in region_acc.values())
    _c_cases = sum(v["cases_annual"] for v in region_acc.values())
    _c_deaths = sum(v["deaths_annual"] for v in region_acc.values())
    _v_QALY = sum(v["_var_QALY"] for v in region_acc.values())
    _v_cases = sum(v["_var_cases"] for v in region_acc.values())
    _v_deaths = sum(v["_var_deaths"] for v in region_acc.values())
    if _sim is not None:
        # Cohort = the Summary-tab selection: pooled totals ("All
        # modalities") or the elementwise sum of the selected modalities'
        # pooled per-iteration series.
        _cser = _sim_cohort_series(_sim, (sweep or {}).get("modality_filter"))
        _cohort_ci = {
            "QALY": _metric_range(_cser, "qaly"),
            "cases": _metric_range(_cser, "cases"),
            "deaths": _metric_range(_cser, "deaths"),
        }
        _cohort_sim = {
            "QALY": _metric_mean(_cser, "qaly"),
            "cases": _metric_mean(_cser, "cases"),
            "deaths": _metric_mean(_cser, "deaths"),
        }
    else:
        _cohort_ci = {
            "QALY": _ci_from_var(_c_QALY, _v_QALY),
            "cases": _ci_from_var(_c_cases, _v_cases),
            "deaths": _ci_from_var(_c_deaths, _v_deaths),
        }
        _cohort_sim = {"QALY": float("nan"), "cases": float("nan"),
                       "deaths": float("nan")}
    cohort = {
        "QALY_annual": _c_QALY,
        "cases_annual": _c_cases,
        "deaths_annual": _c_deaths,
        "total_scans": sum(v["total_scans"] for v in region_acc.values()),
        "total_scans_all": sum(v["total_scans_all"] for v in region_acc.values()),
        "dose_sv_annual": sum(v["dose_sv_annual"] for v in region_acc.values()),
        "dose_msv_annual": sum(v["dose_msv_annual"] for v in region_acc.values()),
        "QALY_annual_ci_lo": _cohort_ci["QALY"][0],
        "QALY_annual_ci_hi": _cohort_ci["QALY"][1],
        "cases_annual_ci_lo": _cohort_ci["cases"][0],
        "cases_annual_ci_hi": _cohort_ci["cases"][1],
        "deaths_annual_ci_lo": _cohort_ci["deaths"][0],
        "deaths_annual_ci_hi": _cohort_ci["deaths"][1],
        "QALY_annual_sim": _cohort_sim["QALY"],
        "cases_annual_sim": _cohort_sim["cases"],
        "deaths_annual_sim": _cohort_sim["deaths"],
    }

    return {
        "extrapolation_factor": ex_factor,
        "current_total_scans": ex_current_total,
        "per_n_scans": per_n,
        "modality_filter": (sweep or {}).get("modality_filter"),
        "modality_region_map": _rehome_dental_cbct_map(
            (sweep or {}).get("modality_region_map"), _cbct_key,
            _dental_cbct_present),
        "per_region_age": per_ra,
        "per_region": per_region,
        "cohort_total": cohort,
    }


@_rq_fonts
def plot_annual_burden_bar(
    burden: Dict[str, Any],
    fig: Any,
) -> None:
    """Annual QALYs lost and collective effective dose of the current
    practice, in the same two-panel layout as :func:`plot_annual_QALY_bar`:
    panel A stacked by age decade with 95% whiskers, panel B the annual
    collective effective dose per region."""
    def _as_saved(rows):
        out = []
        for r in rows or []:
            c = dict(r)
            c["QALY_saved_annual"] = float(r.get("QALY_annual", 0.0) or 0.0)
            c["QALY_saved_annual_ci_lo"] = r.get("QALY_annual_ci_lo", 0.0)
            c["QALY_saved_annual_ci_hi"] = r.get("QALY_annual_ci_hi", 0.0)
            out.append(c)
        return out

    b = burden or {}
    plot_annual_QALY_bar.__wrapped__(
        {"per_region": _as_saved(b.get("per_region")),
         "per_region_age": _as_saved(b.get("per_region_age")),
         "modality_region_map": b.get("modality_region_map") or {}},
        fig, burden_mode=True)


def format_region_age_sweep_summary(
    sweep: Dict[str, Any],
    region_filter: Optional[Union[str, List[str]]] = None,
) -> str:
    """Human-readable text summary for the GUI text panel.

    Produces a table of baseline CT burden per ``per_n_scans`` examinations
    by patient age decade and imaging region.  Each region block shows
    four metric rows — QALYs lost, fatal cancers, cancers, and average
    effective dose (mSv) — with 95% confidence intervals on the first
    three.  Regions are ordered by baseline QALYs lost (descending).
    """
    if not sweep:
        return "No simulation data yet."
    per_region = list(sweep.get("per_region") or [])
    rows = list(sweep.get("per_region_age") or [])
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = [(("", region_filter))]
        elif isinstance(region_filter, list) and region_filter:
            # Normalise: list of strings → list of ("", str) tuples
            region_filter = [
                (("", s)) if isinstance(s, str) else (str(s[0]), str(s[1]))
                for s in region_filter
            ]
        filter_set = set(region_filter)
        per_region = [r for r in per_region
                      if (r.get("modality", ""), r["region"]) in filter_set]
        rows = [r for r in rows
                if (r.get("modality", ""), r["region"]) in filter_set]
    per_n = int(sweep.get("per_n_scans") or 100_000)
    age_groups = list(sweep.get("age_groups") or AGE_DECADE_LABELS)
    if not per_region or not rows:
        return "No simulation data yet."

    # Build (modality, region, age_decade_idx) → row lookup
    cell_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for r in rows:
        cell_map[(r.get("modality", ""), r["region"], r["age_decade_idx"])] = r

    # Region order: sorted by baseline QALYs lost descending (per_region is
    # already sorted this way by compute_region_age_dose_sweep).
    # Use (modality, region) pairs from per_region.
    mod_region_order = [(pr.get("modality", ""), pr["region"]) for pr in per_region]

    mod_filt = sweep.get("modality_filter")
    mod_phrase = _modality_phrase(mod_filt)

    n_iters = int(sweep.get("iterations") or 0)
    min_scans = float(sweep.get("min_scans") or 0.0)
    n_unreliable = sum(1 for r in rows if not r.get("is_reliable", True))
    _is_analytical = sweep_is_analytical(sweep)
    _is_pooled = sweep.get("combine") == "pooled"
    _n_avg_years = int(sweep.get("_averaged_across_years", 0) or 0)
    _show_between_yr = _n_avg_years >= 2
    # attach_simulated_ranges rewrote the per-cell intervals to the central
    # 95% range of simulated annual outcomes (same quantity as the Summary
    # tab's "95% range of simulated years").
    _sim_attached = isinstance(sweep.get("_sim"), dict)
    _cell_cohort = (_sim_attached
                    and sweep.get("cell_interval", CELL_INTERVAL_COHORT)
                    != CELL_INTERVAL_SIMULATED)
    _cohort_marked = _sim_attached and not _cell_cohort and any(
        r.get(f"{bk}_range_source") == "cohort_100k"
        for r in cell_map.values()
        for bk in ("baseline_cases", "baseline_deaths", "baseline_QALY"))

    # Column layout — single width for both point-estimate and CI rows
    # so every value aligns under its age-decade header.
    label_w = 22
    metric_w = 20
    col_w = 22        # width per age column (wide enough for [lo, hi] CI strings)

    # Build header
    val_header = f"  {'Region':<{label_w}} {'Metric':<{metric_w}}"
    for ag in age_groups:
        val_header += f" {ag:>{col_w}}"
    sep = "  " + "─" * (len(val_header) - 2)

    lines: List[str] = []
    lines.append("=" * len(val_header))
    lines.append(
        f"Radiation burden per {per_n:,}{mod_phrase} examinations by patient age decade"
        f" and imaging region."
    )
    lines.append("=" * len(val_header))
    if _is_analytical and _is_pooled:
        lines.append(
            "  Point estimates: analytical ICRP 152 expectation for all year "
            "groups pooled as one cohort (exam-weighted), annualised by "
            "dividing by the number of year groups."
        )
        if _sim_attached and _cell_cohort:
            lines.append(
                f"  Interval: 95% range for a cohort of {per_n:,} "
                "examinations with this cell's recorded dose/age/sex mix "
                "(normal approximation)."
            )
        elif _sim_attached:
            lines.append(
                f"  Interval: {SIM_ANNUAL_RANGE_LABEL} \u2014 2.5th\u201397.5th "
                "percentiles of the pooled per-iteration annual totals (the "
                "Summary tab's '95% range of simulated years'); 'n/a' where "
                "no simulated series was recorded.  Year-to-year min\u2013max "
                "of the expectation is in the CSV export."
            )
            if _cohort_marked:
                lines.append(
                    f"  \u2020: the simulated range is degenerate ([0, 0]: too "
                    f"few examinations for a cancer to be realised in a "
                    f"typical simulated year); shown instead is the analytical "
                    f"95% range for a cohort of {per_n:,} examinations drawn "
                    f"from this cell's dose/age mix (normal approximation, "
                    f"mean \u00b1 1.96\u00b7sqrt({per_n:,}\u00b7\u03a3p(1\u2212p)"
                    f"/n_exams) \u2014 the value a Monte Carlo resample of the "
                    f"cell to {per_n:,} examinations converges to). Not "
                    f"comparable with the un-marked ranges, which are the "
                    f"cell's own annual outcome scaled to {per_n:,}."
                )
        else:
            lines.append(
                "  Interval: analytical 95 % range of the mean annual realised "
                "outcome over the pooled years (normal approximation, mean \u00b1 "
                "1.96\u00b7sqrt(\u03a3 var_y)/k). It does not depend on the "
                "iteration count. Year-to-year min\u2013max of the expectation is "
                "in the CSV export."
            )
        lines.append(
            f"  '*' marks cells with fewer than {min_scans:g} examinations in "
            f"total (all year groups pooled).  Unreliable: {n_unreliable}/{len(rows)}."
        )
    elif _is_analytical:
        lines.append(
            "  Point estimates: analytical ICRP 152 expectation (no Monte "
            "Carlo sampling error), equal-weighted mean over the year groups."
        )
        if _sim_attached and _cell_cohort:
            lines.append(
                f"  Interval: 95% range for a cohort of {per_n:,} "
                "examinations with this cell's recorded dose/age/sex mix "
                "(normal approximation)."
            )
        elif _sim_attached:
            lines.append(
                f"  Interval: {SIM_ANNUAL_RANGE_LABEL} \u2014 2.5th\u201397.5th "
                "percentiles of the pooled per-iteration annual totals (the "
                "Summary tab's '95% range of simulated years'); 'n/a' where "
                "no simulated series was recorded."
            )
            if _cohort_marked:
                lines.append(
                    f"  \u2020: the simulated range is degenerate ([0, 0]: too "
                    f"few examinations for a cancer to be realised in a "
                    f"typical simulated year); shown instead is the analytical "
                    f"95% range for a cohort of {per_n:,} examinations drawn "
                    f"from this cell's dose/age mix (normal approximation, "
                    f"mean \u00b1 1.96\u00b7sqrt({per_n:,}\u00b7\u03a3p(1\u2212p)"
                    f"/n_exams) \u2014 the value a Monte Carlo resample of the "
                    f"cell to {per_n:,} examinations converges to). Not "
                    f"comparable with the un-marked ranges, which are the "
                    f"cell's own annual outcome scaled to {per_n:,}."
                )
        else:
            lines.append(
                "  Interval: analytical 95 % range of a single year's realised "
                "outcome (normal approximation; includes year-to-year variation "
                "of the expectation). It does not depend on the iteration count. "
                "Year-to-year min\u2013max of the expectation is in the CSV export."
            )
        lines.append(
            f"  '*' marks cells with fewer than {min_scans:g} examinations in "
            f"total (all year groups pooled).  Unreliable: {n_unreliable}/{len(rows)}."
        )
    else:
        lines.append(
            "  95% CI (Wald: mean \u00b1 1.96\u00b7SE) shown for QALYs lost, "
            "fatal cancers, and cancers."
        )
        lines.append(
            f"  Based on {n_iters:,} MC iterations."
            f"  '*' marks unreliable cells"
            f" (exams < {min_scans:g} OR imprecise QALY mean"
            f" OR degenerate CI)."
            f"  Degenerate CIs (n<=1 or std==0 with non-zero mean) are shown as 'n/a'."
            f"  Unreliable: {n_unreliable}/{len(rows)}."
        )
        _min_ev_iters = int(sweep.get("min_event_iters") or DEFAULT_MIN_EVENT_ITERS)
        _max_rel_se = float(sweep.get("max_rel_se") or DEFAULT_MAX_REL_SE)
        lines.append(
            f"  Precision: >= {_min_ev_iters:d} event-bearing iterations and "
            f"relative SE of the QALY mean <= {_max_rel_se * 100:.0f} %; "
            f"cells below this are marked * and n/a."
        )
    if _show_between_yr:
        lines.append(
            "  Between-yr 95% CI: mean reflects variation in the imaged "
            "population between calendar years, not risk-model or "
            "dosimetric uncertainty."
        )
    lines.append("")

    # Table header
    lines.append(val_header)
    lines.append(sep)

    # (display name, baseline-value key, CI-lo key, CI-hi key, has_ci)
    metric_specs = [
        ("QALYs lost",            "baseline_QALY_per_100k",
         "baseline_QALY_ci_lo_per_100k",   "baseline_QALY_ci_hi_per_100k",   True),
        ("Fatal cancers",    "baseline_deaths_per_100k",
         "baseline_deaths_ci_lo_per_100k", "baseline_deaths_ci_hi_per_100k", True),
        ("Cancers",          "baseline_cases_per_100k",
         "baseline_cases_ci_lo_per_100k",  "baseline_cases_ci_hi_per_100k",  True),
        ("Average E (mSv)",  "mean_dose_msv",
         None, None, False),
    ]

    # Build modality-grouped region iteration order.
    # When modality_region_map is available (no modality filter), group
    # regions by modality with a section header.  Otherwise iterate flat.
    modality_region_map = sweep.get("modality_region_map") or {}

    if modality_region_map:
        # Ordered: modality codes sorted by display order, regions within
        # each modality keep the QALYs descending order from per_region.
        modality_groups: List[Tuple[str, List[str]]] = []
        for mod_code in sorted(modality_region_map.keys(), key=_mod_sort_key):
            mod_regions = [
                reg for (mod, reg) in mod_region_order
                if reg in modality_region_map.get(mod_code, [])
                and mod == mod_code
            ]
            if mod_regions:
                modality_groups.append((mod_code, mod_regions))
        # (modality, region) pairs not found in any modality group go last.
        covered = {(mod_code, r) for mod_code, rl in modality_groups for r in rl}
        leftover = [reg for (mod, reg) in mod_region_order if (mod, reg) not in covered]
        if leftover:
            modality_groups.append(("", leftover))
    else:
        # No modality map: group by the modality field in per_region.
        _by_mod: Dict[str, List[str]] = {}
        for mod, reg in mod_region_order:
            _by_mod.setdefault(mod, []).append(reg)
        modality_groups = list(_by_mod.items())

    for mod_code, mod_regions in modality_groups:
        if mod_code:
            mod_label = _modality_name(mod_code)
            lines.append(f"  {mod_label}")
            lines.append(sep)

        for reg in mod_regions:
            for mi, (metric_name, base_key, ci_lo_key, ci_hi_key, has_ci) in enumerate(metric_specs):
                # Region label only on the first metric row of the block.
                reg_label = reg[:label_w] if mi == 0 else ""

                # Point-estimate row
                line = f"  {reg_label:<{label_w}} {metric_name:<{metric_w}}"
                for di, _ in enumerate(age_groups):
                    r = cell_map.get((mod_code, reg, di))
                    if r is None:
                        line += f" {'—':>{col_w}}"
                    else:
                        val = float(r.get(base_key, 0.0))
                        flag = "" if r.get("is_reliable", True) else "*"
                        line += f" {f'{val:.3f}{flag}':>{col_w}}"
                lines.append(line)

                # 95% CI / 95% range row (for metrics that have one)
                if has_ci:
                    _ci_label = "95% range" if _is_analytical else "95% CI"
                    ci_line = f"  {'':<{label_w}} {_ci_label:<{metric_w}}"
                    for di, _ in enumerate(age_groups):
                        r = cell_map.get((mod_code, reg, di))
                        if r is None:
                            ci_line += f" {'—':>{col_w}}"
                        elif r.get("ci_degenerate", False):
                            # Degenerate Wald CI (n<=1 or std==0 with non-zero
                            # mean): the interval collapsed to [x, x] and
                            # carries no uncertainty information, so it is
                            # reported as n/a rather than a misleading point.
                            ci_line += f" {'n/a':>{col_w}}"
                        elif _cell_cohort:
                            # Cohort display mode: the analytical 95% range
                            # for a cohort of per_n examinations with this
                            # cell's recorded mix; n/a where never stored.
                            _prefix = ci_lo_key[: -len("_ci_lo_per_100k")]
                            lo_c = r.get(_prefix + "_cohort_lo_per_100k")
                            hi_c = r.get(_prefix + "_cohort_hi_per_100k")
                            if (lo_c is None or hi_c is None
                                    or np.isnan(float(lo_c))
                                    or np.isnan(float(hi_c))):
                                ci_line += f" {'n/a':>{col_w}}"
                            else:
                                ci_str = (f"[{max(0.0, float(lo_c)):.3f}, "
                                          f"{float(hi_c):.3f}]")
                                ci_line += f" {ci_str:>{col_w}}"
                        else:
                            # Stored intervals are symmetric about the mean so
                            # that rescaling stays centre-preserving; a count
                            # cannot be negative, so clamp for display only.
                            lo = float(r.get(ci_lo_key, 0.0) or 0.0)
                            hi = float(r.get(ci_hi_key, 0.0) or 0.0)
                            if np.isnan(lo) or np.isnan(hi):
                                # No simulated series recorded for this cell
                                # (missing year digest) -- n/a, no fallback.
                                ci_line += f" {'n/a':>{col_w}}"
                            else:
                                _prefix = ci_lo_key[: -len("_ci_lo_per_100k")]
                                if r.get(_prefix + "_range_source") == "cohort_100k":
                                    # Simulated range collapsed to a point;
                                    # show the per_n-exam cohort range instead.
                                    lo_c = float(r.get(_prefix + "_cohort_lo_per_100k", 0.0) or 0.0)
                                    hi_c = float(r.get(_prefix + "_cohort_hi_per_100k", 0.0) or 0.0)
                                    ci_str = f"[{max(0.0, lo_c):.3f}, {hi_c:.3f}]†"
                                else:
                                    ci_str = f"[{max(0.0, lo):.3f}, {hi:.3f}]"
                                ci_line += f" {ci_str:>{col_w}}"
                    lines.append(ci_line)

                # Between-year 95 % CI of the k-year mean (cross-year
                # averaged sweeps only; see between_year_stats).
                if has_ci and _show_between_yr:
                    prefix = base_key[: -len("_per_100k")]
                    yr_line = f"  {'':<{label_w}} {'Between-yr 95% CI':<{metric_w}}"
                    for di, _ in enumerate(age_groups):
                        r = cell_map.get((mod_code, reg, di))
                        yr_str = "\u2014"
                        if r is not None:
                            lo_v = r.get(prefix + _YR_CI_LO_SUFFIX)
                            hi_v = r.get(prefix + _YR_CI_HI_SUFFIX)
                            if lo_v is not None and hi_v is not None:
                                lo_f, hi_f = float(lo_v), float(hi_v)
                                if not (np.isnan(lo_f) or np.isnan(hi_f)):
                                    yr_str = f"[{max(0.0, lo_f):.3f}, {hi_f:.3f}]"
                        yr_line += f" {yr_str:>{col_w}}"
                    lines.append(yr_line)

            if _show_between_yr:
                # Exams/yr: the epidemiological denominator of every cell,
                # and the pooled total over the k year groups (the count the
                # small-cell floor is tested against; a cell can show
                # "0" exams/yr yet hold a few exams in total).
                n_line = f"  {'':<{label_w}} {'Exams/yr':<{metric_w}}"
                tot_line = f"  {'':<{label_w}} {'Exams, all years':<{metric_w}}"
                for di, _ in enumerate(age_groups):
                    r = cell_map.get((mod_code, reg, di))
                    if r is None:
                        n_str = tot_str = "\u2014"
                    else:
                        spy = float(r.get("scans_per_year", 0.0) or 0.0)
                        n_str = f"{spy:,.0f}"
                        tot_str = f"{round(spy * _n_avg_years):,d}"
                    n_line += f" {n_str:>{col_w}}"
                    tot_line += f" {tot_str:>{col_w}}"
                lines.append(n_line)
                lines.append(tot_line)

            lines.append(sep)

    lines.append("=" * len(val_header))
    return "\n".join(lines)



DOSE_BURDEN_ALL_LABEL: str = "All modalities"


def dose_burden_modality_title(label: str) -> str:
    """Human-readable modality heading, e.g. ``"CT (CT+CT/PT)"``.

    ``analysis.modality_name`` already embeds the code in its display string
    ("CT" -> "CT (CT+CT/PT)"), so it is used unprefixed here — a
    "<code> - <name>" form would repeat the code twice.
    """
    if label == DOSE_BURDEN_ALL_LABEL:
        return "All modalities combined"
    return _modality_name(label) or str(label)


def _fit_fig_text(
    fig: Any,
    text_obj: Any,
    raw: str,
    *,
    max_width_frac: float = 0.96,
    fontsizes: Tuple[float, ...] = (10.0, 9.0, 8.0),
    max_lines: int = 3,
) -> None:
    """Wrap (then, if needed, shrink) a figure-level text so it cannot clip.

    Figure titles and footnotes are sized in *points* while the space they
    have is a *fraction of the figure*, so a string that fits the 10x15 in
    export clips badly on the GUI's smaller canvas — and the same figure is
    rendered at both sizes here.  Wrapping is tried first at the largest
    font (adding a line costs less legibility than losing a point size),
    and the font is only reduced once ``max_lines`` is exhausted.

    Widths are measured from the rendered pixel extent, never estimated
    from character counts, so the result holds for any figure size, dpi or
    installed font.
    """
    renderer = fig.canvas.get_renderer()
    avail_px = fig.get_figwidth() * fig.dpi * max_width_frac
    raw = " ".join(str(raw).split())

    def _too_wide() -> bool:
        return text_obj.get_window_extent(renderer=renderer).width > avail_px

    for fs in fontsizes:
        text_obj.set_fontsize(fs)
        for n_lines in range(1, max_lines + 1):
            if n_lines == 1:
                text_obj.set_text(raw)
            else:
                # Aim for n equal-length lines; textwrap only ever produces
                # <= n when the target width is len/n, so the loop is
                # monotone in the number of lines actually used.
                width = max(10, int(np.ceil(len(raw) / n_lines)) + 2)
                text_obj.set_text("\n".join(
                    textwrap.wrap(raw, width=width) or [raw]))
            if not _too_wide():
                return
    # Exhausted every option: leave the smallest font and widest wrap.


def _outside(renderer, a, side: str) -> float:
    box = a.get_window_extent()
    tight = a.get_tightbbox(renderer)
    if tight is None:
        return 0.0
    return max(0.0, {
        "left": box.x0 - tight.x0, "right": tight.x1 - box.x1,
        "bottom": box.y0 - tight.y0, "top": tight.y1 - box.y1,
    }[side])


# Burden vs collective effective dose (scaled-practice curve)
# The dose-bin curves above answer a per-examination *rate* question
# ("how much riskier is a high-dose exam"), which needs quantile bins for
# statistical power.  This view answers the complementary *population-
# burden* question -- "given this population and practice, if the collective
# effective dose were X person-Sv, how many cancers / fatal cancers / QALYs
# lost would result" -- by scaling the current practice up or down and
# plotting the resulting burden against collective dose.
#
# At diagnostic doses the per-examination risk is fixed (scaling the practice
# means performing more or fewer of the *same* exams, not changing any one
# exam's dose), so the expected burden is exactly linear in the scale factor
# f:  B(f) = f * B0.  The Monte-Carlo variance scales linearly too
# (Poisson-binomial: f independent copies), so the 95% CI half-width grows
# as sqrt(f) while the mean grows as f -- the *relative* CI narrows as the
# practice scales up, which is the statistically correct behaviour.
#
# The current practice point (f = 1) is marked on the curve.  See
# compute_annual_burden for the per-cell annual counts that feed the cohort
# total.

# How far past the current practice to extend the x-axis, as a multiple of
# the current collective dose.  2.0 shows the trend without pushing the
# current-practice marker into a corner.
DEFAULT_BURDEN_VS_DOSE_MAX_SCALE: float = 2.0
DEFAULT_BURDEN_VS_DOSE_N_POINTS: int = 120


def _sweep_age_quantiles(
    sweep: Dict[str, Any],
) -> Optional[Tuple[float, float, float]]:
    """(Q1, median, Q3) of the weighted exam-age distribution of the sweep,
    from the per-cell 1-year age histograms (``age_hist_per_year``): the
    smallest integer age at which the cumulative exam count reaches
    p x the total, for p = 0.25 / 0.5 / 0.75.  None when no cell carries
    a histogram or the total is zero."""
    total = None
    for c in (sweep or {}).get("per_region_age") or []:
        h = c.get("age_hist_per_year")
        if h is None:
            continue
        h = np.asarray(h, dtype=float)
        total = h.copy() if total is None else total + h
    if total is None or total.sum() <= 0.0:
        return None
    cum = np.cumsum(total)
    return tuple(
        float(np.searchsorted(cum, p * cum[-1], side="left"))
        for p in (0.25, 0.5, 0.75)
    )


def _sweep_sex_counts(sweep: Dict[str, Any]) -> Optional[np.ndarray]:
    """Exam counts by sex (F/M/U, per-year basis) summed over the sweep's
    cells (``sex_counts_per_year``).  None when no cell carries it."""
    total = None
    for c in (sweep or {}).get("per_region_age") or []:
        s = c.get("sex_counts_per_year")
        if s is None:
            continue
        s = np.asarray(s, dtype=float)
        total = s.copy() if total is None else total + s
    return total


def _collective_burden_point(
    sweep: Dict[str, Any],
    *,
    allow_zero_dose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract the cohort-total annual burden + collective dose + per-metric
    variance from the Region × Age sweep.

    Returns ``None`` when there are no dosed examinations (no cohort
    total), or when the cohort's collective dose is zero and
    ``allow_zero_dose`` is False (the default keeps the burden-vs-dose
    plots free of zero-dose points).
    The variance is recovered from the Wald CI half-width
    (``half = CI_Z * sqrt(var)`` => ``var = (half / CI_Z)**2``), which is
    exact for the symmetric CIs :func:`compute_annual_burden` produces via
    :func:`_ci_from_var` (the lower-bound clamp at zero does not affect the
    upper bound, so ``half = ci_hi - point`` is always the true half-width).
    ``cases_sim`` / ``deaths_sim`` / ``qaly_sim`` carry the Monte Carlo
    means of the pooled simulated annual series (NaN when no ``_sim`` is
    attached).
    """
    sweep = sweep or {}
    if not sweep.get("per_region"):
        return None
    cohort = (compute_annual_burden(sweep) or {}).get("cohort_total") or {}
    dose_sv = float(cohort.get("dose_sv_annual", 0.0) or 0.0)
    if dose_sv <= 0.0 and not allow_zero_dose:
        return None

    def _metric(key: str) -> Tuple[float, float, float, float]:
        point = float(cohort.get(f"{key}_annual", 0.0) or 0.0)
        ci_lo = float(cohort.get(f"{key}_annual_ci_lo", 0.0) or 0.0)
        ci_hi = float(cohort.get(f"{key}_annual_ci_hi", 0.0) or 0.0)
        half = max(ci_hi - point, 0.0)
        var = (half / CI_Z) ** 2 if half > 0.0 else 0.0
        return point, var, ci_lo, ci_hi

    cases, var_cases, cases_lo, cases_hi = _metric("cases")
    deaths, var_deaths, deaths_lo, deaths_hi = _metric("deaths")
    qaly, var_qaly, qaly_lo, qaly_hi = _metric("QALY")
    age_q = _sweep_age_quantiles(sweep)
    sex_c = _sweep_sex_counts(sweep)
    # All exams in the sweep's cells (the same basis as age_hist / sex
    # counts) -- the cohort total misses cells skipped by
    # compute_annual_burden (has_risk_model=False / Non-Dose), and in an
    # analytical sweep ``scans_per_year`` itself counts only exams with
    # organ/cancer doses, so prefer the preserved all-exam count.
    _cells = sweep.get("per_region_age") or []
    _cell_scans = sum(float(c.get("scans_all_per_year",
                                 c.get("scans_per_year", 0.0)) or 0.0)
                      for c in _cells)
    total_scans_all = (_cell_scans if _cells else
                       float(cohort.get("total_scans_all",
                                        cohort.get("total_scans", 0.0))
                             or 0.0))
    return {
        "dose_sv": dose_sv,
        "total_scans": float(cohort.get("total_scans", 0.0) or 0.0),
        "total_scans_all": total_scans_all,
        "cases": cases, "var_cases": var_cases,
        "deaths": deaths, "var_deaths": var_deaths,
        "qaly": qaly, "var_qaly": var_qaly,
        # Absolute interval endpoints — with a simulated-annual-outcome
        # sweep attached (``_sim``) these are the central-95% percentiles,
        # which are not symmetric about the point estimate.
        "cases_ci_lo": cases_lo, "cases_ci_hi": cases_hi,
        "deaths_ci_lo": deaths_lo, "deaths_ci_hi": deaths_hi,
        "qaly_ci_lo": qaly_lo, "qaly_ci_hi": qaly_hi,
        "cases_sim": float(cohort.get("cases_annual_sim")
                           if cohort.get("cases_annual_sim") is not None
                           else float("nan")),
        "deaths_sim": float(cohort.get("deaths_annual_sim")
                            if cohort.get("deaths_annual_sim") is not None
                            else float("nan")),
        "qaly_sim": float(cohort.get("QALY_annual_sim")
                          if cohort.get("QALY_annual_sim") is not None
                          else float("nan")),
        "median_age": age_q[1] if age_q is not None else None,
        "age_q1": age_q[0] if age_q is not None else None,
        "age_q3": age_q[2] if age_q is not None else None,
        "female_scans": (float(sex_c[SEX_COUNT_INDEX["F"]])
                         if sex_c is not None else None),
        "male_scans": (float(sex_c[SEX_COUNT_INDEX["M"]])
                       if sex_c is not None else None),
        "unknown_sex_scans": (float(sex_c[SEX_COUNT_INDEX["U"]])
                              if sex_c is not None else None),
    }


# Per-modality small multiples of the burden-vs-dose curve
# One auto-scaled panel per modality so the large disparity in collective
# dose (mammography <1 person-Sv vs CT ~50) is shown at each panel's own
# resolution.  The metric plotted is QALYs lost, because the rate -- QALYs
# lost per person-Sv -- is exactly what varies between modalities and is the
# real content of this view.
DEFAULT_BURDEN_OVERLAY_MAX_DOSE_SV: float = 15.0


def _clean_extrapolation_dose(value: Optional[float]) -> Optional[float]:
    """Return a positive, finite float (person-Sv), or None (= no extrapolation)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v <= 0.0:
        return None
    return v


def _per_modality_burden_points(
    sweep: Dict[str, Any],
    *,
    allow_zero_dose: bool = False,
) -> List[Tuple[str, Dict[str, float]]]:
    """Collect ``_collective_burden_point`` for every modality in the sweep.

    Returns ``[(modality_code, point), ...]`` sorted by modality code,
    skipping modalities with no dosed examinations unless
    ``allow_zero_dose`` (``_collective_burden_point`` returns None).
    Uses ``filter_region_age_sweep`` to subset the cached sweep per
    modality -- no recomputation, see its docstring.
    """
    mod_map = (sweep or {}).get("modality_region_map") or {}
    out: List[Tuple[str, Dict[str, float]]] = []
    for mod in sorted(mod_map.keys()):
        sub = filter_region_age_sweep(sweep, mod)
        pt = _collective_burden_point(sub, allow_zero_dose=allow_zero_dose)
        if pt is not None:
            out.append((str(mod), pt))
    return out


def _burden_point_value(pt: Dict[str, Any], key: str, sim_attached: bool) -> float:
    """Current-practice point for *key* ("cases"/"deaths"/"qaly"): the Monte
    Carlo mean (``<key>_sim``) when the simulated series is attached and
    finite, else the analytical expectation ``pt[key]``."""
    if sim_attached:
        v = pt.get(f"{key}_sim")
        if v is not None and np.isfinite(v):
            return float(v)
    return float(pt[key])


def _burden_band(xx: np.ndarray, d0: float, v0: float,
                 interval: Optional[Tuple[float, float]],
                 var: float, sim_attached: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yy, yy_lo, yy_hi) of the scaled-practice line and its band.

    The line is ``xx * v0 / d0``.  The band is anchored to the
    current-practice marker at ``x = d0`` — exactly its interval there —
    and its half-widths grow as ``sqrt(xx / d0)`` (independent-count
    noise: f copies of the same practice).  With the simulated range
    attached the (asymmetric) half-widths are ``v0 - lo`` / ``hi - v0``
    of *interval*; otherwise the analytical Wald half-width from *var*.
    No interval (None) and no variance -> band collapses onto the line.
    """
    xx = np.asarray(xx, dtype=float)
    slope = (v0 / d0) if d0 > 0.0 else 0.0
    yy = xx * slope
    s = np.sqrt(np.maximum(xx / d0, 0.0)) if d0 > 0.0 else np.zeros_like(xx)
    if sim_attached:
        h_lo = max(v0 - interval[0], 0.0) if interval is not None else 0.0
        h_hi = max(interval[1] - v0, 0.0) if interval is not None else 0.0
    else:
        h_lo = h_hi = CI_Z * np.sqrt(var) if var > 0 else 0.0
    return yy, np.maximum(yy - h_lo * s, 0.0), yy + h_hi * s


@_rq_fonts
def plot_burden_vs_dose_small_multiples(
    sweep: Dict[str, Any],
    fig: Any,
    *,
    overlay_max_dose_sv: float = DEFAULT_BURDEN_OVERLAY_MAX_DOSE_SV,
    extrapolate_to_dose_sv: Optional[float] = None,
) -> None:
    """Annual radiation-related burden vs collective dose, by modality.

    Three side-by-side panels sharing one collective-dose x-axis, each
    showing the same per-modality lines for a different outcome measure:

    * **QALY burden** (left) — QALYs lost, the severity-weighted metric.
    * **Cancer burden** (middle) — radiation-induced cancers.
    * **Fatal cancer burden** (right) — radiation-induced cancer deaths.

    Putting the three measures side by side at each panel's own y-scale
    makes the modality *ranking* comparable across outcome measures: a
    modality's slope is its burden per person-Sv for that measure, so a
    modality that ranks differently between panels does so because
    incidence, mortality and severity weighting are not interchangeable.

    All three panels share the capped x-axis (``overlay_max_dose_sv``,
    default 15 person-Sv) so modality slopes are visually comparable.
    Each modality's line is solid (with CI band) up to 2x its current
    workload; the y-range is set by the drawn artists.  Modalities whose
    workload exceeds the cap get an arrow at the cap pointing off-panel
    plus an italic label along the line stating workload and burden;
    headroom is added so the label stays inside the panel.

    Current-practice markers are the Monte Carlo means of the pooled
    per-iteration annual series when the simulated range is attached
    (matching the text companion), else the analytical expectation; the
    band passes through the marker's own 95% range at ``x = d0`` and its
    half-widths grow as ``sqrt(x / d0)`` (:func:`_burden_band`).

    Modality identity is communicated via a single external legend (below
    the panels) rather than direct line labels.  This is a global solution
    to the label-collision problem: any future dataset can produce close
    slopes, overlapping CI bands, or clustered endpoints that defeat
    direct-label placement, but a legend is always clean.  Each modality
    also carries a fixed line style and marker
    (``style.MODALITY_LINESTYLES`` / ``MODALITY_MARKERS``) as a redundant,
    colour-independent encoding, so overlapping lines and greyscale print
    stay distinguishable.

    The actual per-modality workload values (collective dose, cancers,
    fatal cancers, QALY burden, QALY/person-Sv) are provided by the text
    companion :func:`format_burden_vs_dose_by_modality_summary`, which also
    emits a LaTeX table for the manuscript.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    for _stale in list(fig.axes):
        try:
            _stale.set_xscale("linear")
        except Exception:
            pass
    fig.clear()
    fig.set_layout_engine(None)
    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)

    pts = _per_modality_burden_points(sweep)
    n_avg_years = int((sweep or {}).get("_averaged_across_years", 0) or 0)
    _sim_attached = isinstance((sweep or {}).get("_sim"), dict)
    # Number of input files / year groups pooled into this sweep (1 when the
    # sweep is a single year).  Lines are drawn for the POOLED total over
    # these years; the current-practice markers stay annual (per year).
    k_years = max(n_avg_years, 1)

    if not pts:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "No dosed examinations to plot.\n"
                "Open the Region \u00d7 Age sub-tab first so the per-cell "
                "sweep is built,\nthen return here.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#888")
        ax.set_axis_off()
        return

    overlay_max = float(overlay_max_dose_sv)
    # Manual extrapolation target (person-Sv).  None = unchanged behaviour.
    _extrap = _clean_extrapolation_dose(extrapolate_to_dose_sv)
    if _extrap is not None:
        overlay_max = _extrap

    # Three side-by-side comparison panels
    # Same per-modality lines in each panel, one outcome measure each, so
    # the modality ranking can be read across measures.  sharex because
    # all three use the identical capped collective-dose axis; each keeps
    # its own y-scale (the three measures differ by orders of magnitude).
    outer = fig.add_gridspec(1, 3, wspace=0.18)
    ov_axes: List[Any] = []
    for _i in range(3):
        ov_axes.append(fig.add_subplot(
            outer[0, _i], sharex=(ov_axes[0] if ov_axes else None)))

    _BAND_ALPHA = 0.20

    # (point key, variance key, y-axis label, panel title, metric noun)
    # Order follows the logical progression cancer cases -> fatal cases ->
    # overall health burden (QALYs): a reader first sees the tangible
    # outcomes, then the fatal subset, then the broader population-health
    # metric that incorporates age and life expectancy.  "Annual" lives in
    # the figure title, not the panel titles, to keep the panels short.
    _PANEL_METRICS = (
        ("cases", "var_cases", "Cancers", "A. Cancer cases", "cancers"),
        ("deaths", "var_deaths", "Fatal cancers", "B. Fatal cancers",
         "fatal cancers"),
        ("qaly", "var_qaly", "QALYs lost", "C. QALYs lost", "QALYs"),
    )

    def _short_mod_label(mod_code: str) -> str:
        """Compact modality label for panel titles and direct line labels."""
        label = _modality_name(mod_code) if mod_code else "All modalities"
        label = label.split(" (")[0] if " (" in label else label
        if label == "MG":
            label = "Mammography"
        return label

    def _metric_interval(pt, key, var_key):
        """Absolute (lo, hi) annual interval for the current-practice
        marker — the simulated percentile range when ``_sim`` is attached
        (asymmetric endpoints are used as-is), else the analytical Wald
        half-width derived from *var_key*.  ``None`` when no interval can
        be drawn (missing simulated series -> "n/a" in the text)."""
        if _sim_attached:
            lo = pt.get(f"{key}_ci_lo")
            hi = pt.get(f"{key}_ci_hi")
            if lo is None or hi is None or not (
                    np.isfinite(lo) and np.isfinite(hi)):
                return None
            return float(lo), float(hi)
        v0 = _burden_point_value(pt, key, _sim_attached)
        vv = pt[var_key]
        h = CI_Z * np.sqrt(vv) if vv > 0 else 0.0
        return max(v0 - h, 0.0), v0 + h

    def _draw_marker(ax, d0, v0, interval, col, marker):
        # Diamond renders larger than "o" at the same markersize.
        ms = 7.0 if marker == "D" else 8.0
        if interval is None:
            ax.plot(d0, v0, marker, color=col, markersize=ms,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=4)
            return
        lo, hi = interval
        ax.errorbar(d0, v0,
                    yerr=[[max(v0 - lo, 0.0)], [max(hi - v0, 0.0)]],
                    fmt=marker, color=col, ecolor=col, markersize=ms,
                    markeredgecolor="black", markeredgewidth=0.8,
                    elinewidth=1.4, capsize=3.5, zorder=4)

    # Overlay panels: slopes comparable, actual workload in text
    # The x-axis is capped at ``overlay_max_dose_sv`` (default 15 person-Sv)
    # so that modality slopes are visually comparable.  Each modality's line
    # is solid (with CI band) to 2x its current workload; the y-range is
    # set by the drawn artists.  Modalities whose workload exceeds the cap
    # get an arrow at the cap pointing off-panel plus an italic label along
    # the line stating workload and burden (placed in the per-modality
    # loop below; headroom/fontsize adjusted after the first draw).
    panel_off: List[List[Tuple[str, float, float, float, str, float]]] = [
        [] for _ in ov_axes]
    for _pi, (_key, _var_key, _ylab, _panel_title, _noun) in zip(
            range(len(ov_axes)), _PANEL_METRICS):
        main = ov_axes[_pi]
        off_panel: List[Tuple[str, float, float, float, str]] = []
        y_need = 0.0
        _cap_line = False
        for mod, pt in pts:
            # Marker = ANNUAL current practice: the Monte Carlo mean of the
            # pooled per-iteration annual series when _sim is attached
            # (matching the text companion), else the analytical expectation.
            # The line extends to 2x the pooled k-year workload; the band is
            # anchored to the marker's own interval at x = d0 and grows as
            # sqrt(x / d0) (_burden_band).
            d0 = pt["dose_sv"]
            v0 = _burden_point_value(pt, _key, _sim_attached)
            vv = pt[_var_key]
            col = _modality_color(mod)
            ls = _modality_linestyle(mod)
            mk = _modality_marker(mod)
            if d0 <= 0.0:
                continue
            slope = v0 / d0
            iv = _metric_interval(pt, _key, _var_key)
            d_pool = d0 * k_years
            # Solid segment + CI band cover up to 2x the POOLED workload.
            x_draw = (overlay_max if _extrap is not None
                      else min(2.0 * d_pool, overlay_max))
            xx = np.linspace(0.0, x_draw, 60)
            yy, yy_lo, yy_hi = _burden_band(
                xx, d0, v0, iv, vv, _sim_attached)
            main.fill_between(xx, yy_lo, yy_hi,
                              color=col, alpha=_BAND_ALPHA, zorder=1)
            # Thin band edges in the modality's own line style so
            # overlapping bands stay separable without colour.
            main.plot(xx, yy_hi, color=col, linewidth=0.9,
                      linestyle=ls, alpha=0.75, zorder=1.5)
            main.plot(xx, yy_lo, color=col, linewidth=0.9,
                      linestyle=ls, alpha=0.75, zorder=1.5)
            main.plot(xx, yy, color=col, linewidth=3.0,
                      linestyle=ls, zorder=2)
            y_need = max(y_need, float(np.nanmax(yy_hi)))
            # Marker = ANNUAL current practice (sits on the line at x = d0).
            if d0 <= overlay_max:
                _draw_marker(main, d0, v0, iv, col, mk)
                y_need = max(y_need, iv[1] if iv is not None else v0)
            # Arrow only when the ANNUAL current practice itself lies past
            # the capped x-axis (not merely the pooled line).
            if d0 > overlay_max:
                off_panel.append((_short_mod_label(mod), d0, v0, vv, col,
                                  slope))
                # Arrow at the cap pointing off-panel: the line continues
                # beyond the capped axis.  The workload / burden numbers are
                # reported in the text summary, not on the figure.
                # A dashed continuation past a dotted "axis limit" line
                # reads as "runs on beyond the plotted range".
                y_edge = overlay_max * slope
                x_end = overlay_max * 1.085
                main.annotate("", xy=(x_end, x_end * slope),
                              xytext=(overlay_max, y_edge),
                              arrowprops=dict(arrowstyle="-", color=col,
                                              lw=3.0,
                                              linestyle=(0, (2.2, 1.6)),
                                              shrinkA=0, shrinkB=0),
                              annotation_clip=False, zorder=3)
                y_need = max(y_need, x_end * slope)
                _cap_line = True

        if _cap_line:
            main.axvline(overlay_max, color="#555555", linewidth=1.1,
                         linestyle=(0, (1.0, 2.0)), alpha=0.8, zorder=2.5)
        main.set_xlim(0.0, overlay_max * (1.11 if _cap_line else 1.04))
        main.set_ylim(0.0, 1.08 * y_need)

        main.grid(True, axis="y", alpha=0.22, linestyle="-", linewidth=0.5,
                  color="#BBBBBB")
        main.set_axisbelow(True)
        main.spines["top"].set_visible(False)
        main.spines["right"].set_visible(False)
        for sp in ("left", "bottom"):
            main.spines[sp].set_color("#333333")
            main.spines[sp].set_linewidth(0.8)
        main.tick_params(labelsize=19.2, color="#333333")
        for _tl in main.get_xticklabels() + main.get_yticklabels():
            _tl.set_fontweight("bold")
        main.set_xlabel("Collective dose (person Sv)",
                        fontsize=20.2, fontweight="bold")
        main.set_ylabel(_ylab, fontsize=20.2, fontweight="bold")
        main.set_title(_panel_title, fontsize=20.2, fontweight="bold",
                       loc="center", pad=8.0)

        panel_off[_pi] = off_panel

    # Title block, pixel-measured margins
    # The pooled-years / annual-marker distinction is left to the
    # manuscript caption rather than spelled out in the figure title.
    head = ("Projected annual radiation-related health outcomes vs. "
            "collective effective dose")
    if _extrap is not None:
        head += f" (current practice extrapolated to {_extrap:g} person Sv)"

    fig_w_px = fig.get_figwidth() * fig.dpi
    fig_h_px = fig.get_figheight() * fig.dpi
    _PAD_PX = 6.0
    # Extra gap between the suptitle and the panel titles, separate from the
    # edge padding so the other margins stay tight.  A bit more than _PAD_PX
    # to establish a clear figure-title -> panel-title -> plots hierarchy
    # without spending much vertical real estate (~1.8% of figure height).
    _TOP_GAP_PX = 13.2
    renderer = fig.canvas.get_renderer()

    # Legend (below the panels, horizontal)
    # One shared legend for all three panels: the modality colours are
    # identical across panels, so repeating it three times would waste
    # width.  A legend is used instead of direct line labels because it is
    # immune to the collision problems that arise when modality slopes
    # are close, CI bands overlap, or endpoints cluster — problems that
    # any future dataset can produce.  With only a handful of
    # modalities the legend is not burdensome and the plot stays clean.
    from matplotlib.lines import Line2D
    legend_handles = []
    for mod, pt in pts:
        if pt["dose_sv"] <= 0.0:
            continue
        col = _modality_color(mod)
        _mk = _modality_marker(mod)
        legend_handles.append(
            Line2D([], [], color=col, linewidth=3.0,
                   linestyle=_modality_linestyle(mod), marker=_mk,
                   markersize=7.0 if _mk == "D" else 8.0,
                   markeredgecolor="black", markeredgewidth=0.8,
                   label=_short_mod_label(mod)))
    _ov_legend = fig.legend(
        handles=legend_handles, ncol=max(1, len(legend_handles)),
        loc="lower center",
        bbox_to_anchor=(0.5, _PAD_PX / fig_h_px),
        fontsize=21.8, frameon=False, handlelength=3.0,
        columnspacing=1.6, handletextpad=0.6, borderaxespad=0.0)
    for _t in _ov_legend.get_texts():
        _t.set_fontweight("bold")

    t_head = fig.suptitle(head, x=0.5, y=1.0 - _PAD_PX / fig_h_px, va="top",
                          fontsize=23.8, fontweight="bold")
    _fit_fig_text(fig, t_head, head, max_width_frac=0.965,
                  fontsizes=(23.8, 21.9, 20.1, 18.2), max_lines=2)
    block_y0 = t_head.get_window_extent(renderer=renderer).y0

    fig.canvas.draw()

    fig.canvas.draw()


    # Margins: measured across the three panels.  The shared legend is a
    # single horizontal row centred below the panels, so the right margin
    # reserves no legend space; the bottom margin must reserve the legend
    # height so the axes tight-bbox bottom sits above it.  wspace lives
    # on the gridspec, not here.
    left_px = _PAD_PX + max(_outside(renderer, a, "left") for a in ov_axes)
    right_px = _PAD_PX + max(_outside(renderer, a, "right") for a in ov_axes)
    lb = _ov_legend.get_window_extent(renderer=renderer)
    bottom_px = _PAD_PX + lb.height + 4.0 + max(_outside(renderer, a, "bottom")
                                                for a in ov_axes)
    top_px = block_y0 - _TOP_GAP_PX - max(_outside(renderer, a, "top") for a in ov_axes)
    fig.subplots_adjust(
        left=float(np.clip(left_px / fig_w_px, 0.04, 0.30)),
        right=float(np.clip(1.0 - right_px / fig_w_px, 0.50, 0.985)),
        top=float(np.clip(top_px / fig_h_px, 0.55, 0.96)),
        bottom=float(np.clip(bottom_px / fig_h_px, 0.05, 0.45)),
    )


def format_burden_vs_dose_by_modality_summary(
    sweep: Dict[str, Any],
    extrapolate_to_dose_sv: Optional[float] = None,
    *,
    patients_by_modality: Optional[Dict[str, Any]] = None,
) -> str:
    """Fixed-width text companion to :func:`plot_burden_vs_dose_small_multiples`.

    Reports the per-modality current-practice values as plain text, so the
    exact numbers can be copied directly instead of being read off the
    rendered chart.

    When ``attach_simulated_ranges`` has run on the sweep (``_sim``),
    cancers / fatal cancers / QALYs lost are reported as the Monte Carlo
    means of the pooled per-iteration annual series — the same series
    behind the Summary tab's distributions — rather than the analytical
    per-cell expectations; collective dose, exams, sex counts and age
    quantiles stay sweep-derived, and distinct patients come from
    ``patients_by_modality`` (``stats["_patients_by_modality"]``).
    """
    pts = _per_modality_burden_points(sweep, allow_zero_dose=True)
    if not pts:
        return ("No dosed examinations found.\n"
                "Open the Region \u00d7 Age sub-tab first so the per-cell "
                "sweep is built, then return here.")
    n_avg_years = int((sweep or {}).get("_averaged_across_years", 0) or 0)
    per_yr = "/yr" if n_avg_years else ""
    _is_analytical = sweep_is_analytical(sweep)
    _is_pooled = (sweep or {}).get("interval_kind") == "analytical_pooled_years_range"
    _sim_attached = isinstance((sweep or {}).get("_sim"), dict)
    # Since exams without a risk model are dropped at source
    # (participant_flow.py) the all-exams and risk-modelled counts coincide;
    # the "modelled" column / footnote then only restate the Exams column
    # and are replaced by a one-line note.  Sweeps that keep unmodelled
    # exams masked (``total_scans_all`` > ``total_scans``) still show the
    # split.
    _all_modelled = all(
        abs(float(pt.get("total_scans_all", pt.get("total_scans", 0.0)) or 0.0)
            - float(pt.get("total_scans", 0.0) or 0.0)) < 0.5
        for _, pt in pts)

    def _ci(point: float, var: float) -> Tuple[float, float]:
        half = CI_Z * np.sqrt(var) if var > 0 else 0.0
        return max(point - half, 0.0), point + half

    def _rng_txt(lo: float, hi: float) -> str:
        """'[lo, hi]' or 'n/a' for a missing simulated series."""
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return "n/a"
        return f"[{max(lo, 0.0):.2f}, {hi:.2f}]"

    def _vals(pt) -> Tuple[float, float, float]:
        """(cancers, fatal, QALYs): Monte Carlo means when the simulated
        series is attached, else the analytical expectation."""
        if _sim_attached:
            return pt["cases_sim"], pt["deaths_sim"], pt["qaly_sim"]
        return pt["cases"], pt["deaths"], pt["qaly"]

    def _num(v, fmt) -> str:
        """format(v, fmt), or 'n/a' for a non-finite value."""
        if v is None or not np.isfinite(v):
            return "n/a"
        return format(v, fmt)

    L: List[str] = []
    L.append("CURRENT PRACTICE BY MODALITY "
             "(radiation-related QALY burden vs collective dose)")
    if n_avg_years and (sweep or {}).get("combine") == "pooled":
        L.append(f"POOLED ACROSS {n_avg_years} YEAR GROUP(S) \u2014 counts "
                 f"are annual means of the pooled cohort (total / {n_avg_years}).")
    elif n_avg_years:
        L.append(f"AVERAGE ACROSS {n_avg_years} YEAR GROUP(S) \u2014 counts "
                 "are per-year means.")
    if _sim_attached:
        L.append("Cancers, fatal cancers and QALYs lost are Monte Carlo "
                 "means of the pooled per-iteration annual series \u2014 "
                 "the same series behind the Summary tab's cancer-count, "
                 "fatal-cancer-count and QALYs-lost distributions for each "
                 "modality; the ranges are their central 95% percentiles. "
                 "Collective dose, exams, patients, sex and age are "
                 "exam-data totals; patients are distinct persons over "
                 "all years (a patient imaged with several modalities is "
                 "counted in each). "
                 + ("Exams are the analysed cohort: every examination "
                    "carries an organ-dose model. Examinations without one "
                    "(Non-Dose rows, non-dental CBCT, radiographs without "
                    "KAP, mammography without MGD or in men, ...) were "
                    "excluded upstream; see the Summary tab's participant "
                    "flow and the By Body Region tab's 'Excluded "
                    "examinations'."
                    if _all_modelled else
                    "Exams are all examinations in the dataset; the "
                    "'modelled' column gives the risk-modelled subset."))
    elif _is_analytical:
        L.append("Cancers, fatal cancers and QALYs are expected annual "
                 "values (ICRP 152 expectation computed per exam, not "
                 "Monte Carlo means); the ranges below are analytical 95% "
                 "ranges.")
    L.append("")
    # Total row: the whole sweep's cohort point.  Its simulated range is
    # the elementwise sum of the shown modalities' pooled series (the
    # sweep's modality selection), not a sum of the per-modality ranges.
    _total_pt = _collective_burden_point(sweep, allow_zero_dose=True)
    _rows = list(pts) + ([(None, _total_pt)] if _total_pt is not None else [])

    def _row_label(mod) -> str:
        return "Total" if mod is None else dose_burden_modality_title(mod)

    _pbm = patients_by_modality or {}
    _pbm_sets = _pbm.get("sets") or {}
    _pbm_missing = _pbm.get("missing") or {}

    def _patients_n(mod) -> Optional[int]:
        """Distinct patients for one modality, or the union for Total
        (a patient imaged with several modalities is counted once in
        Total but once per modality in the rows)."""
        if mod is None:
            # Dataset-wide distinct count: union over ALL modality sets
            # (not just the shown rows) + every missing-exam count.  Equals
            # year_batch._cross_year_patient_totals(...)[0].
            if not _pbm_sets and not _pbm_missing:
                return None
            u: set = set()
            for ids in _pbm_sets.values():
                u |= set(ids)
            miss = sum(int(v or 0) for v in _pbm_missing.values())
            return len(u) + miss
        if mod not in _pbm_sets:
            return None
        return len(_pbm_sets[mod]) + int(_pbm_missing.get(mod, 0) or 0)

    def _patients_str(mod, na: str) -> str:
        n = _patients_n(mod)
        return f"{n:,}" if n is not None else na

    def _female_str(pt, pct: str, na: str) -> str:
        f = pt.get("female_scans")
        m = pt.get("male_scans")
        u = pt.get("unknown_sex_scans")
        if f is None and m is None and u is None:
            return na
        denom = (f or 0.0) + (m or 0.0) + (u or 0.0)
        if denom <= 0.0:
            return na
        return f"{(f or 0.0):,.0f} ({100.0 * (f or 0.0) / denom:.1f}{pct})"

    def _age_str(pt, dash: str, na: str) -> str:
        med = pt.get("median_age")
        q1 = pt.get("age_q1")
        q3 = pt.get("age_q3")
        if med is None or q1 is None or q3 is None:
            return na
        return f"{med:.0f} ({q1:.0f}{dash}{q3:.0f})"

    L.append(f"{'Modality':<26s} {'exams' + per_yr:>10s} "
             f"{'modelled' + per_yr:>10s} {'patients':>10s} "
             f"{('female' + per_yr + ' (%)'):>16s} {'age med (IQR)':>14s} "
             f"{'person-Sv':>10s} {'cancers':>9s} {'fatal':>8s} "
             f"{'QALYs':>8s} {'QALY/person-Sv':>15s}")
    for mod, pt in _rows:
        d0 = pt["dose_sv"]
        c0, dth, q0 = _vals(pt)
        n_scans = int(round(pt.get("total_scans_all",
                                   pt.get("total_scans", 0.0)) or 0.0))
        n_modelled = int(round(pt.get("total_scans", 0.0) or 0.0))
        slope = (q0 / d0) if d0 > 0.0 else None
        if mod is None:
            L.append("-" * 146)
        L.append(f"{_row_label(mod):<26s} {n_scans:>10,} "
                 f"{n_modelled:>10,} "
                 f"{_patients_str(mod, 'n/a'):>10s} "
                 f"{_female_str(pt, '%', 'n/a'):>16s} "
                 f"{_age_str(pt, '-', 'n/a'):>14s} {d0:>10.2f} "
                 f"{_num(c0, '.2f'):>9s} {_num(dth, '.2f'):>8s} "
                 f"{_num(q0, '.1f'):>8s} {_num(slope, '.2f'):>15s}")
    L.append("")
    if _sim_attached:
        L.append(f"{SIM_ANNUAL_RANGE_LABEL}:")
    else:
        L.append("95% analytical ranges (mean annual realised outcome over the "
                 "pooled years):" if _is_pooled
                 else "95% analytical single-year ranges:" if _is_analytical
                 else "95% Monte Carlo CIs:")
    for mod, pt in _rows:
        label = _row_label(mod)
        if _sim_attached:
            L.append(f"  {label}: cancers "
                     f"{_rng_txt(pt['cases_ci_lo'], pt['cases_ci_hi'])}, "
                     f"fatal {_rng_txt(pt['deaths_ci_lo'], pt['deaths_ci_hi'])}, "
                     f"QALYs {_rng_txt(pt['qaly_ci_lo'], pt['qaly_ci_hi'])}")
        else:
            c_lo, c_hi = _ci(pt["cases"], pt["var_cases"])
            d_lo, d_hi = _ci(pt["deaths"], pt["var_deaths"])
            q_lo, q_hi = _ci(pt["qaly"], pt["var_qaly"])
            L.append(f"  {label}: cancers [{c_lo:.2f}, {c_hi:.2f}], "
                     f"fatal [{d_lo:.2f}, {d_hi:.2f}], "
                     f"QALYs [{q_lo:.2f}, {q_hi:.2f}]")

    doses = [pt["dose_sv"] for _, pt in pts if pt["dose_sv"] > 0.0] or [0.0]
    d_max, d_min = max(doses), min(doses)
    disparity = (d_max / d_min) if d_min > 0 else float("inf")
    L.append("")
    L.append(f"Collective dose range across dosed modalities: {d_min:.2f}"
             f"\u2013{d_max:.2f} person-Sv{per_yr} (~{disparity:.0f}-fold).")

    # Manual extrapolation block (only when a target dose was typed)
    _extrap = _clean_extrapolation_dose(extrapolate_to_dose_sv)
    if _extrap is not None:

        def _scaled_rng(pt: Dict[str, float], key: str, var_key: str,
                        f: float) -> str:
            _vc, _vd, _vq = _vals(pt)
            v0 = {"cases": _vc, "deaths": _vd, "qaly": _vq}[key]
            s_f = float(np.sqrt(f))
            if _sim_attached:
                lo = pt.get(f"{key}_ci_lo")
                hi = pt.get(f"{key}_ci_hi")
                if lo is None or hi is None or not (
                        np.isfinite(lo) and np.isfinite(hi)):
                    return "n/a"
                h_lo = max(v0 - lo, 0.0) * s_f
                h_hi = max(hi - v0, 0.0) * s_f
            else:
                var = pt[var_key]
                h_lo = h_hi = CI_Z * np.sqrt(f * var) if var > 0 else 0.0
            return _rng_txt(v0 * f - h_lo, v0 * f + h_hi)

        L.append("")
        L.append(f"EXTRAPOLATED TO {_extrap:g} person-Sv{per_yr} FOR EACH "
                 "MODALITY")
        L.append("Linear: burden = current burden x target dose / current "
                 "dose. Assumes the same exam, region and age mix as the "
                 "current practice.")
        L.append(f"{'Modality':<26s} {'current Sv':>10s} {'x factor':>9s} "
                 f"{'cancers':>9s} {'fatal':>8s} {'QALYs':>9s}")
        _ex_lines: List[str] = []
        for mod, pt in pts:
            d0 = pt["dose_sv"]
            if d0 <= 0.0:
                continue
            f = _extrap / d0
            label = _row_label(mod)
            _xc, _xd, _xq = _vals(pt)
            L.append(f"{label:<26s} {d0:>10.2f} {f:>9.2f} "
                     f"{_num(_xc * f, '.2f'):>9s} "
                     f"{_num(_xd * f, '.2f'):>8s} "
                     f"{_num(_xq * f, '.1f'):>9s}")
            _ex_lines.append(
                f"  {label}: cancers "
                f"{_scaled_rng(pt, 'cases', 'var_cases', f)}, "
                f"fatal {_scaled_rng(pt, 'deaths', 'var_deaths', f)}, "
                f"QALYs {_scaled_rng(pt, 'qaly', 'var_qaly', f)}")
        L.append("")
        L.append("Ranges at the extrapolated dose (current range scaled by "
                 "sqrt(target dose / current dose)):")
        L.extend(_ex_lines)

    return "\n".join(L)


__all__ = [
    "compute_region_age_dose_sweep",
    "average_region_age_sweeps",
    "filter_region_age_sweep",
    "rescale_region_age_sweep",
    "apply_min_scans",
    "plot_region_age_sweep",
    "compute_annual_benefit",
    "plot_annual_QALY_bar",
    "compute_annual_burden",
    "plot_annual_burden_bar",
    "attach_simulated_ranges",
    "set_cell_interval",
    "CELL_INTERVAL_SIMULATED",
    "CELL_INTERVAL_COHORT",
    "per_iteration_cell_series",
    "SIM_ANNUAL_RANGE_LABEL",
    "plot_burden_vs_dose_small_multiples",
    "format_burden_vs_dose_by_modality_summary",
    "DEFAULT_BURDEN_VS_DOSE_MAX_SCALE",
    "DEFAULT_BURDEN_VS_DOSE_N_POINTS",
    "DEFAULT_BURDEN_OVERLAY_MAX_DOSE_SV",
    "format_region_age_sweep_summary",
    "DOSE_BURDEN_ALL_LABEL",
    "DEFAULT_DOSE_REDUCTIONS_PCT",
    "AGE_DECADE_EDGES",
    "AGE_DECADE_LABELS",
    "DEFAULT_MIN_SCANS",
    "DEFAULT_MIN_EVENTS",
    "DEFAULT_MIN_EVENT_ITERS",
    "DEFAULT_MAX_REL_SE",
]
