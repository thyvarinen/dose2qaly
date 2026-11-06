"""
Dose2QALY — Monte Carlo simulation of imaging radiation risk
================================================================

Purpose
-------
Estimate the population-level lifetime cancer risk and detriment caused by
medical X-ray exams listed in Excel files.
For every exam row we (1) compute absorbed organ doses, (2) convert them
to a per-site lifetime cancer risk via the ICRP 152 risk model, and then
(3) run a Monte Carlo simulation that, year by year, draws one Bernoulli
outcome per cancer site per exam and aggregates cases, deaths, life-years
lost and ICRP detriment until the stopping rule is met (stable per-age
risk estimates and a relative standard error of the annual cancer count
and annual QALYs lost below the precision target).

Pipeline (one exam → one risk vector)
-------------------------------------
  Excel row
    → Region resolver           (region_resolver)
    → Modality dispatch         (exam_doses)
    → Organ doses [Gy]          (Conversion coefficients per modality, exam_doses)
    → ICRP 152 lifetime risk    (risk.compute_lifetime_risk)
    → Bernoulli draw / site     (vectorized engine)
    → Aggregate over MC years   (this file — vectorized engine)
    → Convergence + GUI         (this file)

Module layout
-------------
  exam_doses        modality dispatch + per-modality dose helpers
  risk              ICRP 152 lifetime risk + Bernoulli outcome sampling
  preprocess        per-exam preprocessing (population, Golikov, frailty)
  excel_io          Excel ingestion + column whitelist
  mc_warnings       diagnostic warning ring + CuPy/GPU probes
  risk_models       ICRP 152 risk-model parameters & kernels
  region_resolver   exam description → body region
  vectorized        GPU / CPU vectorised iteration engine
  gui               Tkinter results GUI
  analysis          post-simulation plots & summaries
  config            pre-run configuration dialog
  year_batch        multi-year batch driver (per-year average / combined)
  research_questions  region x age burden sweeps and dose-reduction projections
  participant_flow  examination selection flow counts
  file_cohort       per-file cohort bookkeeping
  preprocess_cache  on-disk cache of preprocessed inputs

This file hosts ``run_simulation`` (the orchestrator), the GUI launcher
(``start_gui``), the ``__main__`` entry point, and the per-iteration
``_merge_partial`` aggregator that is tied to ``run_simulation``'s
convergence / streaming logic.

Public API (re-exported from ``dose2qaly/__init__.py``)
---------------------------------------------------------------------
  compute_exam_organ_doses(row)   See ``exam_doses``.
  compute_lifetime_risk(...)      See ``risk``.
  run_simulation(...)             Full Monte Carlo engine (this file).
  start_gui(...)                  Launch the results GUI (this file).
  get_backend_info()              See ``mc_warnings``.

Domain terms (LAR, REIC, DDREF, QALYs, collective effective dose, ...) are
defined in ``docs/GLOSSARY.md``.
"""
from __future__ import annotations  # forward-reference types in annotations

# Standard library
import os              # paths, environment checks, file existence
import sys             # sys.path bootstrap, thread frame dumps
import time            # wall-clock timers for profiling and watchdog
import shutil          # cleanup of temporary event directories
import tempfile        # per-run event buffer directories
import multiprocessing as mp  # CPU count and worker pool
import threading       # background cache save and stall-dump watchdog
import functools      # partial binding of the per-iteration worker
from collections import deque, Counter  # FIFO for inflight futures and modality counts
from dataclasses import dataclass       # _SimConfig / _CohortState containers
from concurrent.futures import ThreadPoolExecutor  # parallel CPU iteration dispatch

# Third-party
import numpy as np     # vectorised arrays and numerical ops

try:
    # Spawned pool workers re-import this module as __mp_main__; importing
    # CuPy there costs ~400 MB commit + a CUDA context per worker for nothing.
    if mp.parent_process() is not None:
        raise ImportError("GPU is never used in spawned worker processes")
    import cupy as cp   # optional GPU acceleration (None if not installed)
except Exception:
    cp = None

# Direct-execution bootstrap
# main.py can be started in three ways:
#   1. As a script:          python ".../dose2qaly/main.py"
#   2. In a spawned worker:  multiprocessing re-imports it as __mp_main__
#   3. As a package module:  import dose2qaly.main
#
# Cases 1 and 2 leave __package__ as None, so the relative imports below
# (``from .risk_models import ...``) would fail.  This block adds the
# workspace root to sys.path and restores the package context.  Case 3
# already has the right context, so the guard does nothing.
if __name__ in ("__main__", "__mp_main__") and __package__ in (None, ""):
    from pathlib import Path as _Path
    _pkg_root = _Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    __package__ = "dose2qaly"
    del _Path, _pkg_root

# Project modules
from .constants import Z_95, BACKGROUND_MSV_PER_YEAR
from .risk_models import (
    CANCER_TO_ORGAN_DOSE_KEY,
    MAX_ATTAINED_AGE_ICRP152,
    DDREF_SOLID_CANCERS,
    DDREF_LEUKAEMIA,
    precompute_kernels,
    configure_simulation,
    get_vectorized_rpg,
    get_vectorized_rpg_2pop,
    get_cancer_params_cached,
)
from .exam_doses import compute_exam_organ_doses  # noqa: F401  (re-exported)
from .risk import compute_lifetime_risk  # noqa: F401  (re-exported)
# NOTE: excel_io pulls in pandas at import time; it is imported lazily inside
# run_simulation() (its only call site) so that importing this module -- and
# therefore the whole package, since submodule import always runs
# __init__.py -- does not force pandas to load for callers that only need
# the dose/risk computation API.
from .preprocess import (
    load_and_preprocess,
    merge_acquisition_series,
    analyze_population_and_ct_doses,
    apply_golikov_and_cumulative_dose,
)
from .columns import EventColumns
from .mc_warnings import (
    check_optional_packages,
    get_backend_info,
)
# NOTE: gui pulls in tkinter + matplotlib; imported lazily inside start_gui()
# (its only call site), same rationale as excel_io above.
from .vectorized import (
    build_exam_arrays as _vec_build_exam_arrays,
    vectorized_iteration as _vec_iteration,
    invalidate_gpu_cache as _vec_invalidate_gpu_cache,
    plan_iteration_slots as _vec_plan_iteration_slots,
    invalidate_persistent_cp_rng as _vec_invalidate_persistent_cp_rng,
    precompute_risk_invariants as _vec_precompute_risk_invariants,
    gpu_run_section as _vec_gpu_run_section,
    MAIN_NS,
)


# 1. PER-ITERATION STATS HELPERS

def _merge_partial(stats, partial):
    """Accumulate one iteration's results into the running "stats" dict.

    This is called once per completed MC year in the caller's (main)
    thread.  It updates totals, modality breakdowns, cancer-type
    breakdowns, and heritable-risk counters.
    """
    stats["total_sv"] += partial["dose_sv"]
    stats["cancer_cases"] += partial["cases"]
    stats["cancer_deaths"] += partial["deaths"]
    stats["expected_deaths"] += partial.get("expected_deaths", 0.0)
    stats["expected_incidence"] += partial.get("expected_incidence", 0.0)
    stats["total_detriment"] += partial["detriment"]
    stats["QALY_lost"] += partial["QALY"]
    stats["life_lost"] += partial["life_lost"]
    for mod, ms in partial["modality"].items():
        t = stats["modality_stats"].setdefault(mod, {
            "scans": 0, "scans_dose": 0, "scans_model": 0,
            "dose_sv": 0.0, "dose_sv_model": 0.0, "cases": 0, "deaths": 0,
            "expected_deaths": 0.0,
            "detriment": 0.0, "QALY": 0.0, "life_lost": 0.0, "regions": {},
        })
        for k in ("scans", "scans_dose", "scans_model", "cases", "deaths"):
            t[k] += ms.get(k, 0)
        for k in ("dose_sv", "dose_sv_model", "expected_deaths", "detriment", "QALY", "life_lost"):
            t[k] += ms.get(k, 0.0)
        for reg, rs in ms.get("regions", {}).items():
            rt = t["regions"].setdefault(reg, {
                "scans": 0, "scans_dose": 0, "scans_model": 0,
                "dose_sv": 0.0, "dose_sv_model": 0.0, "cases": 0, "deaths": 0,
                "expected_deaths": 0.0,
            })
            for k in ("scans", "scans_dose", "scans_model", "cases", "deaths"):
                rt[k] += rs.get(k, 0)
            for k in ("dose_sv", "dose_sv_model", "expected_deaths"):
                rt[k] += rs.get(k, 0.0)
    for ct, cv in partial["cancer_bd"].items():
        t = stats["cancer_breakdown"].setdefault(ct, {"developed": 0, "deaths": 0})
        t["developed"] += cv["developed"]
        t["deaths"] += cv["deaths"]
    stats["heritable_risk"] = stats.get("heritable_risk", 0.0) + partial.get("heritable_risk", 0.0)
    stats["heritable_detriment"] = stats.get("heritable_detriment", 0.0) + partial.get("heritable_detriment", 0.0)
    for rct, rcv in partial.get("rem_subtype_bd", {}).items():
        t = stats["rem_subtype_breakdown"].setdefault(rct, {"developed": 0, "deaths": 0})
        t["developed"] += rcv["developed"]
        t["deaths"] += rcv["deaths"]

    # Per-modality-per-device accumulation.
    for mod, dev_dict in (partial.get("device_stats") or {}).items():
        md = stats["device_stats"].setdefault(mod, {})
        for dev, ds in dev_dict.items():
            t = md.setdefault(dev, {
                "scans": 0, "scans_dose": 0, "scans_model": 0,
                "dose_sv": 0.0, "dose_sv_model": 0.0, "cases": 0, "deaths": 0,
                "expected_deaths": 0.0, "detriment": 0.0,
            })
            for k in ("scans", "scans_dose", "scans_model", "cases", "deaths"):
                t[k] += ds.get(k, 0)
            for k in ("dose_sv", "dose_sv_model", "expected_deaths", "detriment"):
                t[k] += ds.get(k, 0.0)

    # Per-modality-per-year accumulation.
    for mod, yr_dict in (partial.get("year_stats") or {}).items():
        my = stats["year_stats"].setdefault(mod, {})
        for yr, ys in yr_dict.items():
            t = my.setdefault(yr, {
                "scans": 0, "scans_dose": 0, "scans_model": 0,
                "dose_sv": 0.0, "dose_sv_model": 0.0, "cases": 0, "deaths": 0,
                "expected_deaths": 0.0, "detriment": 0.0,
            })
            for k in ("scans", "scans_dose", "scans_model", "cases", "deaths"):
                t[k] += ys.get(k, 0)
            for k in ("dose_sv", "dose_sv_model", "expected_deaths", "detriment"):
                t[k] += ys.get(k, 0.0)


def _bin_index(arr: np.ndarray, bins: np.ndarray, n_bins: int) -> np.ndarray:
    """np.searchsorted → 0-based bin index, clipped to [0, n_bins-1]."""
    return np.clip(np.searchsorted(bins, arr, side='right') - 1, 0, n_bins - 1)


def _bind_exam_array_view(ea: dict):
    """Return the sorted exam view and common host arrays stored in ``ea``.

    Called after ``build_exam_arrays`` (warm cache, cold cache, or RPG mismatch
    rebuild) to avoid copy-pasting the same rebinding in three places.
    """
    exam_data = ea["exam_data_sorted"]
    ages = ea["age_int_f64"]
    n_cpu = max(1, mp.cpu_count() or 1)
    return exam_data, ages, n_cpu


_EVENT_COLUMN_KEYS = (
    "cancer_type", "modality", "region", "protocol", "age_at_exposure",
    "sex", "died", "exam_index", "iteration", "ed_msv", "QALY_lost",
    "life_lost_years", "age_at_death", "projected_lifetime",
)

# Minimum number of simulated years before the stopping rule may stop a
# run; author decision.
DEFAULT_MIN_ITERATIONS = 1000


def _count_by_modality(exam_data) -> Counter:
    """Count exams by normalised modality string."""
    if hasattr(exam_data, "_cols"):
        return Counter(str(m or "Unknown") for m in exam_data._cols.get("modality", []))
    return Counter(str(ex.get("modality") or "Unknown") for ex in exam_data)


def _sum_count_tables(tables) -> dict:
    """Elementwise sum of several ``{key: number}`` tables (plain dict out)."""
    out: dict = {}
    for t in tables or ():
        for k, v in (t or {}).items():
            out[k] = out.get(k, 0) + v
    return out


def _disambiguated_basenames(file_list: list) -> list:
    """Return one human-readable label per entry in ``file_list``.

    Uses the basename, appending a `" (2)"``, ``" (3)"`` ... suffix when two
    files in different directories share the same basename, so every label
    is unique.  Order matches ``file_list``, which is what lets the
    per-file breakdown line up 1:1 with ``exam_data``'s ``_file_index``.
    """
    seen: dict = {}
    labels = []
    for f in file_list:
        base = os.path.basename(str(f)) or str(f)
        n = seen.get(base, 0) + 1
        seen[base] = n
        labels.append(base if n == 1 else f"{base} ({n})")
    return labels


def _count_by_file(exam_data, file_labels: list) -> dict:
    """Count exams per source input file, keyed by disambiguated basename.

    Every label in ``file_labels`` is included (initialised to 0) so the
    result preserves the input file order and shows files that contributed
    zero exams, rather than only the files a Counter happened to see.
    """
    counts = {label: 0 for label in file_labels}
    n_labels = len(file_labels)

    def _bump(fi):
        fi = int(fi) if fi is not None else 0
        if 0 <= fi < n_labels:
            counts[file_labels[fi]] += 1

    if hasattr(exam_data, "_cols"):
        total = len(exam_data)
        for fi in exam_data._cols.get("_file_index", [0] * total):
            _bump(fi)
    else:
        for ex in exam_data:
            _bump(ex.get("_file_index", 0))
    return counts


def _empty_event_columns() -> EventColumns:
    """Return an empty EventColumns with the correct dtype for each field.

    Delegates to :meth:`EventColumns.empty` (the single source of truth
    in ``columns.py``); a thin wrapper for call sites in this module that
    take no arguments.
    """
    return EventColumns.empty()


def _stopping_decision(n_done: int, min_iterations: int, stable_count: int,
                       stability_window: int, precision_enabled: bool,
                       rel_se_cases, rel_se_qaly, target_rel_se: float):
    """Decide whether the MC iteration loop should stop, and why.

    Pure helper (no globals) so it can be unit-tested directly.  The rule:
    stop once ``n_done >= min_iterations`` AND ``stable_count >=
    stability_window`` AND (precision disabled OR both ``rel_se_cases``
    and ``rel_se_qaly`` are not None and at or below ``target_rel_se``).
    A ``None`` rel SE (n<2 or mean<=0) means the precision criterion is not
    met while enabled.  Returns ``(should_stop, reason)`` with ``reason``
    in ``{"stability_and_precision", "stability_only", "not_converged"}``.
    """
    stable = n_done >= min_iterations and stable_count >= stability_window
    if not stable:
        return (False, "not_converged")
    if not precision_enabled:
        return (True, "stability_only")
    precision_ok = (
        rel_se_cases is not None and rel_se_cases <= target_rel_se
        and rel_se_qaly is not None and rel_se_qaly <= target_rel_se
    )
    if precision_ok:
        return (True, "stability_and_precision")
    return (False, "not_converged")


def _rel_se_of_iter_list(values: list):
    """Relative SE of a per-iteration list: std(ddof=1)/sqrt(n)/mean.

    Returns ``None`` when ``n < 2`` or the mean is non-positive (a stable
    estimate of zero, not an imprecise one) -- matching the convention
    :func:`research_questions._wald_ci` uses for ``rel_se_QALY``.
    """
    n = len(values or [])
    if n < 2:
        return None
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if mean <= 0.0:
        return None
    std = float(arr.std(ddof=1))
    return (std / np.sqrt(n)) / mean


# 2. SIMULATION ENGINE


@dataclass(frozen=True)
class _SimConfig:
    """Parsed ``sim_config`` for one run_simulation call (see its docstring)."""
    ddref_solid: float
    ddref_leuk: float
    use_golikov: bool
    use_mataac: bool
    merge_series: bool
    keep_events: bool
    convergence_pct: float
    precision_stop_enabled: bool
    precision_target_rel_se: float
    min_iterations_enabled: bool
    icrp152_strict: bool
    min_iterations: int
    background_rad: float
    QALY_mode: str
    QALY_discount_rate: float
    profile: bool
    gpu_iters_per_device: int
    needed_cols: object
    skip_preprocess_cache: bool

    @classmethod
    def from_dict(cls, sim_config: dict | None) -> _SimConfig:
        """Unpack simulation config (with safe defaults)."""
        _cfg = sim_config or {}
        ddref_solid = float(_cfg.get("ddref_solid", DDREF_SOLID_CANCERS))
        ddref_leuk  = float(_cfg.get("ddref_leuk", DDREF_LEUKAEMIA))
        use_golikov = bool(_cfg.get("use_golikov", True))
        use_mataac  = bool(_cfg.get("use_mataac", True))
        merge_series = bool(_cfg.get("merge_acquisition_series", True))
        keep_events = bool(_cfg.get("keep_events", True))
        convergence_pct = float(_cfg.get("convergence_pct", 1.0))
        precision_stop_enabled = bool(_cfg.get("precision_stop_enabled", True))
        precision_target_rel_se = float(_cfg.get("precision_target_rel_se", 0.01) or 0.01)
        min_iterations_enabled = bool(_cfg.get("min_iterations_enabled", False))
        icrp152_strict = bool(_cfg.get("icrp152_strict", False))
        if icrp152_strict:
            use_golikov = False
            use_mataac  = False
        _raw_min = _cfg.get("min_iterations", DEFAULT_MIN_ITERATIONS)
        try:
            min_iterations = int(_raw_min)
        except (TypeError, ValueError):
            raise ValueError(
                f"min_iterations must be a positive integer, got {_raw_min!r}"
            )
        if min_iterations < 1:
            raise ValueError(
                f"min_iterations must be a positive integer, got {_raw_min!r}"
            )
        background_rad = float(_cfg.get("background_radiation_msv_per_year", BACKGROUND_MSV_PER_YEAR))
        # QALY mode and discount rate are mirrored in sim_config for downstream reporting.
        QALY_mode          = str(_cfg.get("QALY_mode", "full")).lower()
        QALY_discount_rate = float(_cfg.get("QALY_discount_rate", 0.0))
        # Performance knobs: profile toggles per-phase timing; gpu_iters_per_device is capped by VRAM.
        profile = bool(_cfg.get("profile", False)) or os.environ.get("DOSE2QALY_PROFILE", "") not in ("", "0")
        # Per-device concurrency defaults to 1: profiling showed >1 in-flight iteration on a single GPU adds overhead, not overlap.
        gpu_iters_per_device = max(1, int(_cfg.get("gpu_iters_per_device", 1) or 1))
        needed_cols = _cfg.get("needed_cols")  # None → use DEFAULT_NEEDED_COLS
        # Per-year batch pins one pseudonymisation salt across every year's run
        # (see year_batch.run_yearly_simulation / exam_doses.pin_patient_id_salt)
        # so the cross-year distinct-patient count can be taken as a set union.
        # A cached year would carry tokens written under a different salt and
        # break that union, so the batch sets this flag to bypass the full
        # preprocess cache entirely (load + save).
        skip_preprocess_cache = bool(_cfg.get("skip_preprocess_cache", False))
        return cls(
            ddref_solid=ddref_solid, ddref_leuk=ddref_leuk,
            use_golikov=use_golikov, use_mataac=use_mataac,
            merge_series=merge_series, keep_events=keep_events,
            convergence_pct=convergence_pct,
            precision_stop_enabled=precision_stop_enabled,
            precision_target_rel_se=precision_target_rel_se,
            min_iterations_enabled=min_iterations_enabled,
            icrp152_strict=icrp152_strict, min_iterations=min_iterations,
            background_rad=background_rad,
            QALY_mode=QALY_mode, QALY_discount_rate=QALY_discount_rate,
            profile=profile, gpu_iters_per_device=gpu_iters_per_device,
            needed_cols=needed_cols, skip_preprocess_cache=skip_preprocess_cache,
        )


@dataclass
class _CohortState:
    """Cohort-loading result for :func:`_prepare_cohort`.

    ``error`` short-circuits the run (same strings run_simulation returned
    directly before the split).  ``full_cached`` is the raw full-cache tuple
    (or None); ``ea``/``ages``/``n_cpu`` are only populated on the warm path.
    """
    error: str | None = None
    full_cached: object = None
    ea: object = None
    exam_data: object = None
    total: int = 0
    ages: object = None
    n_cpu: object = None
    population: object = None
    ea_ratio: float = 0.5
    as_ratio: float = 0.5
    startup_population: object = None
    use_mixed_kernels: bool = False
    pop_counts: object = None
    file_pop_counts: object = None
    ct_no_dose: object = None
    n_acquisition_rows: int = 0
    acq_rows_by_modality: object = None
    acq_rows_by_file: object = None
    file_labels: object = None
    excluded_no_organ_dose: object = None
    participant_flow: object = None
    # Exclusion ledger for the exams dropped at source because they carry
    # no risk model (participant_flow.EXCLUSION_* keys).  ``excluded_exams``
    # holds the raw exam dicts (cold path only; None on the warm path);
    # the per-modality counters / dose and the CohortByFile count tables
    # are what gets persisted, reported and rendered.
    excluded_exams: object = None
    excluded_by_modality: object = None
    excluded_dose_sv_by_modality: object = None
    excluded_cohorts: object = None


def _prepare_cohort(file_list, cfg: _SimConfig, population, on_progress) -> _CohortState:
    """Load/preprocess the exam cohort (full-cache probe, cold load, warm restore).

    Covers run_simulation stages 1-2c: the full post-build_exam_arrays cache
    probe; on a miss the cold load + merge + population/CT-dose scan +
    missing-CT prompt + no-organ-dose filter + dental partition +
    participant-flow ledger + Golikov/cumulative dose; on a hit the warm
    restore of every cached counter.  ``population`` may be reassigned on
    either path -- the caller must take ``state.population``.
    """
    _startup_population = population  # start-up config value; ``population``
    # is reassigned to the cohort mix ("Euro-American/Asian") below.
    # 1. Try the full post-build_exam_arrays cache first
    # A hit skips all preprocessing and returns the columnar exam arrays directly.
    _full_cached = None
    if not cfg.skip_preprocess_cache:
        try:
            from .preprocess_cache import load_preprocess_full_cache as _load_full_cache
            _full_cached = _load_full_cache(
                file_list, cfg.use_golikov, cfg.use_mataac,
                needed_cols=cfg.needed_cols, merge_series=cfg.merge_series,
                population=population,
            )
        except Exception:
            pass

    # 1½. Cold path: load input files and preprocess
    if _full_cached is None:
        exam_data, total, _row_file_index = load_and_preprocess(
            file_list, needed_cols=cfg.needed_cols, on_progress=on_progress
        )
        del _row_file_index

        if len(exam_data) == 0:
            return _CohortState(error="No data")

        # Resolve the per-exam population: rows whose Population cell was
        # empty take the start-up population.  A start-up value that is not
        # a concrete label — e.g. the mixed run-level label, which has no
        # meaning for a single row — falls back to Euro-American (the same
        # default run_simulation uses).
        if _startup_population not in ("Euro-American", "Asian"):
            _orig_startup_population = _startup_population
            _startup_population = "Euro-American"
            if on_progress:
                on_progress("init", total, total,
                            f"Start-up population {_orig_startup_population!r} "
                            "is not a concrete label -> using Euro-American "
                            "for rows without a Population value")
        _n_pop_defaulted = 0
        for _ex in exam_data:
            if _ex.get("_population") not in ("Euro-American", "Asian"):
                _ex["_population"] = _startup_population
                _n_pop_defaulted += 1
        if _n_pop_defaulted and on_progress:
            on_progress("init", total, total,
                        f"{_n_pop_defaulted:,} rows without a Population value "
                        f"→ using the start-up population {_startup_population}")

        # Participant-flow ledger inputs: rows loaded vs acquisition rows.
        _rows_loaded = total
        # Capture the raw acquisition row count before merging sub-series.
        _n_acquisition_rows = len(exam_data)
        _acq_rows_by_modality = _count_by_modality(exam_data)
        _file_labels = _disambiguated_basenames(file_list)
        _acq_rows_by_file = _count_by_file(exam_data, _file_labels)

    # Per-modality count of exams dropped by the CT/Dental-CBCT no-organ-dose
    # filter below. Populated on the cold path when _ct_no_dose is non-empty;
    # on the warm-cache path it is restored from the full cache (the filter
    # is not re-run, so the persisted count is the only source).
    _excluded_no_organ_dose = Counter()
    # Exams actually dropped by the CT/Dental-CBCT no-organ-dose filter
    # (cold path only; the participant-flow ledger needs the exam dicts,
    # not just the count).  None on the warm-cache path.
    _dropped_no_organ_dose: list = []
    # STROBE/RECORD participant-flow ledger: built on the cold path,
    # restored from the full cache on the warm path.
    _participant_flow = None
    # Exclusion ledger (Non-Dose / no organ-dose model exams dropped at
    # source, see participant_flow.py): raw exam lists on the cold path,
    # counters + CohortByFile tables restored from the cache on the warm path.
    _excluded_exams = None
    _excluded_by_modality: dict = {}
    _excluded_dose_sv_by_modality: dict = {}
    _excluded_cohorts = None

    if _full_cached is not None:
        # ── Warm path: cache already holds ea and pop counts; only kernels/splits remain.
        (
            ea,
            _pop_counts,
            _file_pop_counts,
            _ct_no_dose,
            _unknown_descs,
            _cached_n_acquisition_rows,
            _cached_acq_rows_by_modality,
            _cached_acq_rows_by_file,
            _cached_excluded_no_organ_dose,
            _cached_participant_flow,
            _cached_excluded_by_modality,
            _cached_excluded_dose_sv_by_modality,
            _cached_excluded_cohorts,
        ) = _full_cached
        _participant_flow = _cached_participant_flow
        if _cached_excluded_no_organ_dose:
            _excluded_no_organ_dose = Counter(_cached_excluded_no_organ_dose)
        _excluded_by_modality = dict(_cached_excluded_by_modality or {})
        _excluded_dose_sv_by_modality = dict(_cached_excluded_dose_sv_by_modality or {})
        _excluded_cohorts = _cached_excluded_cohorts

        exam_data = ea["exam_data_sorted"]
        total = len(exam_data)
        _n_acquisition_rows = _cached_n_acquisition_rows if _cached_n_acquisition_rows is not None else total
        _acq_rows_by_modality = _cached_acq_rows_by_modality if _cached_acq_rows_by_modality is not None else _count_by_modality(exam_data)
        _file_labels = _disambiguated_basenames(file_list)
        _acq_rows_by_file = _cached_acq_rows_by_file if _cached_acq_rows_by_file is not None else _count_by_file(exam_data, _file_labels)
        _, ages, n_cpu = _bind_exam_array_view(ea)

        # Re-derive population mix parameters from pop counts stored in cache
        _n_tagged = _pop_counts.get("Euro-American", 0) + _pop_counts.get("Asian", 0)
        _ea_ratio = 0.5
        _as_ratio = 0.5
        _use_mixed_kernels = False
        if _n_tagged > 0:
            _ea_ratio = _pop_counts["Euro-American"] / _n_tagged
            _as_ratio = _pop_counts["Asian"] / _n_tagged
            if _pop_counts["Euro-American"] > 0 and _pop_counts["Asian"] > 0:
                population = "Euro-American/Asian"
                _use_mixed_kernels = True
            elif _pop_counts["Asian"] > _pop_counts["Euro-American"]:
                population = "Asian"
            else:
                population = "Euro-American"

        # Restore unknown-description cache so the GUI warning reproduces.
        try:
            from . import region_resolver as _rr
            for _d, _m in (_unknown_descs or []):
                _k = str(_d or "").strip().lower()
                if _k and _k not in _rr._UNKNOWN_DESCRIPTION_CACHE:
                    _rr._UNKNOWN_DESCRIPTION_CACHE.add(_k)
                    _rr._UNKNOWN_DESCRIPTIONS.append((_d, _m))
        except Exception:
            pass
        # Push exam_data to the GUI so filter tables populate on cache hit.
        if on_progress:
            on_progress("exam_data", exam_data)
        return _CohortState(
            full_cached=_full_cached, ea=ea, exam_data=exam_data, total=total,
            ages=ages, n_cpu=n_cpu, population=population,
            ea_ratio=_ea_ratio, as_ratio=_as_ratio,
            startup_population=_startup_population,
            use_mixed_kernels=_use_mixed_kernels,
            pop_counts=_pop_counts, file_pop_counts=_file_pop_counts,
            ct_no_dose=_ct_no_dose, n_acquisition_rows=_n_acquisition_rows,
            acq_rows_by_modality=_acq_rows_by_modality,
            acq_rows_by_file=_acq_rows_by_file, file_labels=_file_labels,
            excluded_no_organ_dose=_excluded_no_organ_dose,
            participant_flow=_participant_flow,
            excluded_by_modality=_excluded_by_modality,
            excluded_dose_sv_by_modality=_excluded_dose_sv_by_modality,
            excluded_cohorts=_excluded_cohorts,
        )

    # 2⁺. Merge multi-acquisition sub-series into one exam per examination
    # Merging per-acquisition rows keeps counts and doses meaningful; update ``total``.
    if cfg.merge_series:
        exam_data = merge_acquisition_series(exam_data, on_progress)
        total = len(exam_data)

    # Participant-flow ledger: snapshot of the merged dataset BEFORE any
    # exclusions (the "dataset" stage; counts and patient ids only).
    _flow_dataset_exams = list(exam_data)

    from .participant_flow import (
        partition_dental_radiography as _partition_dental,
        partition_non_dose as _partition_non_dose,
        partition_no_dose_model as _partition_no_dose_model,
        build_participant_flow as _build_participant_flow,
        excluded_by_modality as _excl_by_modality,
        excluded_dose_sv_by_modality as _excl_dose_sv_by_modality,
        EXCLUSION_NON_DOSE as _EXCL_NON_DOSE,
        EXCLUSION_NO_ORGAN_DOSE as _EXCL_NO_ORGAN_DOSE,
        EXCLUSION_DENTAL_RADIOGRAPHY as _EXCL_DENTAL,
        EXCLUSION_NO_DOSE_MODEL as _EXCL_NO_DOSE_MODEL,
        RISK_MODEL_EXCLUSIONS as _RISK_MODEL_EXCLUSIONS,
        exclusion_reason as _exclusion_reason,
    )

    # 2⁻³. Exclude Non-Dose rows (QA / phantom / post-mortem: no living
    # patient in the beam).  First of the exclusion stages, so the ledger
    # partitions below are disjoint by construction and a Non-Dose CT row
    # without organ doses is a Non-Dose exclusion, not a missing-dose gap.
    exam_data, _dropped_non_dose = _partition_non_dose(exam_data)
    total = len(exam_data)
    if _dropped_non_dose:
        _msg = (f"Excluded {len(_dropped_non_dose):,} Non-Dose exam(s) (no living "
                f"patient in the beam) from the analysed cohort ({total:,} exams remain)")
        if on_progress:
            on_progress("init", total, total, _msg)
        else:
            print(f"[INFO] {_msg}", flush=True)
    if total == 0:
        return _CohortState(error="No exams remain after excluding Non-Dose rows")

    # 2⁻¹ / 2⁰ / 2c: single O(N) scan
    population, _ea_ratio, _as_ratio, _use_mixed_kernels, _ct_no_dose, \
        _pop_counts, _file_pop_counts = \
        analyze_population_and_ct_doses(exam_data, file_list, population, total, on_progress)

    # Missing CT dose prompt
    if _ct_no_dose:
        _missing_summary = {
            region: {
                "n_patients": len(d["patient_ids"]) if d["patient_ids"] else d["n_exams"],
                "n_exams": d["n_exams"],
                "expected_organs": d["expected_organs"],
            }
            for region, d in _ct_no_dose.items()
        }
        _will_answer = False
        if on_progress:
            _continue_event = threading.Event()
            _continue_result = [True]
            # Protocol: a handler that will answer the prompt (set the event)
            # returns True; anything else (None/False, e.g. a progress
            # printer in a script) means no one will answer, so the engine
            # continues immediately with the warning printed instead of
            # paying a silent 300 s wait.
            _will_answer = (
                on_progress("missing_ct_doses", _missing_summary,
                            _continue_event, _continue_result) is True
            )
        if _will_answer:
            _continue_event.wait(timeout=300)
            if not _continue_result[0]:
                return _CohortState(error="Cancelled by user: missing CT doses")
        else:
            for region, info in _missing_summary.items():
                print(
                    f"[WARNING] Missing CT organ doses: region='{region}', "
                    f"exams={info['n_exams']}, patients={info['n_patients']}",
                    flush=True,
                )

    # Drop CT exams with no organ dose so they don't dilute risk averages
    # (a genuine data gap -- CT organ doses come straight from Excel
    # columns, so a missing value means the export was incomplete).
    # For CBCT, only "Dental CBCT" rows with no organ dose are a
    # comparable data gap (exam_doses._dental_doses yields none only when
    # the DAP column is missing or <= 0); every other CBCT region (Ankle,
    # Wrist, ...) has no dose model *by design* -- see exam_doses.py -- and
    # is excluded by the no-dose-model stage below with that reason.
    # Runs unconditionally (not only when CT doses are missing) so a
    # dental CBCT without DAP lands in this ledger stage regardless of
    # what the CT rows of the same file look like.
    _before_filter = len(exam_data)
    _kept = []
    for ex in exam_data:
        if not ex.get("organ_doses_gy") and (
            ex.get("modality") == "CT"
            or (ex.get("modality") == "CBCT"
                and str(ex.get("region") or "").lower() == "dental cbct")
        ):
            _excluded_no_organ_dose[str(ex.get("modality") or "Unknown")] += 1
            _dropped_no_organ_dose.append(ex)
        else:
            _kept.append(ex)
    exam_data = _kept
    _n_excluded = _before_filter - len(exam_data)
    total = len(exam_data)
    if _n_excluded:
        _msg = (f"Excluded {_n_excluded:,} CT exam(s) without organ doses / dental CBCT "
                f"exam(s) without KAP from the simulation ({total:,} exams remain)")
        if on_progress:
            on_progress("init", total, total, _msg)
        else:
            print(f"[INFO] {_msg}", flush=True)
    if total == 0:
        return _CohortState(error="No exams with organ dose data remain after filtering")

    # 2⁻². Exclude non-CBCT dental radiography from the cohort
    # Exams whose region is one of the "Dental imaging" subtype regions
    # ("Dental Panoramic", "Dental_intraoral", "Dental", "Kefalometry")
    # are removed unconditionally: they carry no dose model, so they
    # only dilute the cohort.  "Dental CBCT" is NOT affected.  Runs
    # unconditionally (not inside the _ct_no_dose block) so the dropped
    # rows never reach ea, the pyramid, sweeps, or any downstream count.
    exam_data, _dropped_dental = _partition_dental(exam_data)
    total = len(exam_data)
    if _dropped_dental:
        _msg = (f"Excluded {len(_dropped_dental):,} non-CBCT dental radiography "
                f"exam(s) from the simulation ({total:,} exams remain)")
        if on_progress:
            on_progress("init", total, total, _msg)
        else:
            print(f"[INFO] {_msg}", flush=True)
    if total == 0:
        return _CohortState(error="No exams remain after excluding dental radiography")

    # 2⁻⁴. Exclude exams with no organ-dose model (empty ``_cancer_doses``:
    # non-dental CBCT, radiography without KAP or with an unmodelled
    # projection, mammography without MGD or in men, NM/PT, ...).  These
    # are missing-data exclusions, not zero-dose examinations, so they
    # leave the analysed cohort here instead of sitting in every count and
    # per-100 000 denominator with zero burden.  Every exam that remains
    # is risk-modelled.  The dropped exams are kept as a ledger so the By
    # Body Region tab can still list them (with the reason) as excluded.
    exam_data, _dropped_no_model = _partition_no_dose_model(exam_data)
    total = len(exam_data)
    # Split the no-dose-model drops by whether the primary dose quantity was
    # missing (no dose data) vs present but no organ-dose model.  The
    # missing-dose-data exams move to the no-organ-dose stage so the two
    # "dose" exclusion stages match the manuscript wording.
    _dropped_no_model_no_dose = [
        ex for ex in _dropped_no_model if bool(ex.get("_dose_quantity_missing"))]
    _dropped_no_model = [
        ex for ex in _dropped_no_model if not bool(ex.get("_dose_quantity_missing"))]
    if _dropped_no_model_no_dose:
        _dropped_no_organ_dose.extend(_dropped_no_model_no_dose)
        for ex in _dropped_no_model_no_dose:
            _excluded_no_organ_dose[str(ex.get("modality") or "Unknown")] += 1
            # Retag so the By Body Region ledger lists them under the
            # no-dose-data stage with the right reason.
            try:
                ex["_exclusion_key"] = _EXCL_NO_ORGAN_DOSE
                ex["_exclusion_reason"] = _exclusion_reason(ex)
            except TypeError:
                pass
    if _dropped_no_model:
        _msg = (f"Excluded {len(_dropped_no_model):,} exam(s) with no organ-dose "
                f"model from the analysed cohort ({total:,} risk-modelled exams remain)")
        if on_progress:
            on_progress("init", total, total, _msg)
        else:
            print(f"[INFO] {_msg}", flush=True)
    if total == 0:
        return _CohortState(error="No risk-modelled exams remain after excluding exams without an organ-dose model")
    # All four exclusion stages, in flow order, so the By Body Region tab
    # can list every excluded exam with its reason.  The "excluded dose"
    # bookkeeping covers only the two risk-model stages (male mammography
    # MGD etc.); CT without organ doses and dental radiography carry no
    # dose the app would otherwise have counted.
    _excluded_exams = {
        _EXCL_NON_DOSE: _dropped_non_dose,
        _EXCL_NO_ORGAN_DOSE: _dropped_no_organ_dose,
        _EXCL_DENTAL: _dropped_dental,
        _EXCL_NO_DOSE_MODEL: _dropped_no_model,
    }
    _excluded_by_modality = {k: _excl_by_modality(v) for k, v in _excluded_exams.items()}
    _excluded_dose_sv_by_modality = {
        k: _excl_dose_sv_by_modality(v) for k, v in _excluded_exams.items()
        if k in _RISK_MODEL_EXCLUSIONS}

    # Participant-flow ledger (STROBE/RECORD "numbers at each stage")
    # Disjoint partition in precedence order Non-Dose -> no-organ-dose
    # -> dental -> no-dose-model -> analysed, exactly the order the
    # exclusions ran above, so the stage exams sum to the dataset total
    # and the analysed cohort is the risk-modelled cohort.
    try:
        _participant_flow = _build_participant_flow(
            rows_loaded=_rows_loaded,
            rows_after_preprocess=_n_acquisition_rows,
            exams_merged=len(_flow_dataset_exams),
            dataset_exams=_flow_dataset_exams,
            non_dose_exams=_dropped_non_dose,
            no_organ_dose_exams=_dropped_no_organ_dose,
            dental_exams=_dropped_dental,
            no_dose_model_exams=_dropped_no_model,
            analysed_exams=exam_data,
        )
    except Exception:
        _participant_flow = None

    # 2a. Apply Golikov weight adjustment toggle
    # 2b. Compute cumulative prior dose per patient
    apply_golikov_and_cumulative_dose(exam_data, total, cfg.use_golikov, cfg.use_mataac, on_progress)

    # Push exam_data to the GUI once after dose adjustments to avoid redundant rebuilds.
    if on_progress:
        on_progress("exam_data", exam_data)

    return _CohortState(
        full_cached=None, ea=None, exam_data=exam_data, total=total,
        ages=None, n_cpu=None, population=population,
        ea_ratio=_ea_ratio, as_ratio=_as_ratio,
        startup_population=_startup_population,
        use_mixed_kernels=_use_mixed_kernels,
        pop_counts=_pop_counts, file_pop_counts=_file_pop_counts,
        ct_no_dose=_ct_no_dose, n_acquisition_rows=_n_acquisition_rows,
        acq_rows_by_modality=_acq_rows_by_modality,
        acq_rows_by_file=_acq_rows_by_file, file_labels=_file_labels,
        excluded_no_organ_dose=_excluded_no_organ_dose,
        participant_flow=_participant_flow,
        excluded_exams=_excluded_exams,
        excluded_by_modality=_excluded_by_modality,
        excluded_dose_sv_by_modality=_excluded_dose_sv_by_modality,
        excluded_cohorts=None,
    )


def _build_exam_arrays_and_cache(cohort: _CohortState, rpg_ct_to_idx,
                                 cfg: _SimConfig, file_list, on_progress):
    """(Re)build vectorised exam arrays when needed; save the full cache in background.

    Rebuilds when the full cache missed (``cohort.ea is None``) or when the
    cached kernels do not match the current RPG width.  Returns
    ``(ea, exam_data, ages, n_cpu)`` -- the warm values unchanged when no
    rebuild was needed.
    """
    ea = cohort.ea
    exam_data = cohort.exam_data
    total = cohort.total
    ages = cohort.ages
    n_cpu = cohort.n_cpu
    # Rebuild exam arrays if cached kernels don't match the current config or no cache exists.
    _needs_array_rebuild = False
    if cohort.full_cached is None:
        _needs_array_rebuild = True
        _rebuild_msg = f"Building vectorized exam arrays ({total:,} exams)..."
    else:
        _cached_n_solid = ea.get("dose_solid", np.empty((0, 0))).shape[1]
        if _cached_n_solid != len(rpg_ct_to_idx):
            _needs_array_rebuild = True
            _rebuild_msg = f"RPG mismatch — rebuilding exam arrays ({total:,} exams)..."
    if _needs_array_rebuild:
        _vec_invalidate_gpu_cache(ns=MAIN_NS)  # ensure clean GPU state for new simulation
        if on_progress:
            on_progress("init", total, total, _rebuild_msg)
        ea = _vec_build_exam_arrays(exam_data, rpg_ct_to_idx, on_progress=on_progress)
        # Rebind to the sorted array view so downstream lookups match ea's internal order.
        exam_data, ages, n_cpu = _bind_exam_array_view(ea)

        # Save the full cache in a background thread so the next run can skip preprocessing.
        # Cache failures are non-fatal.  Skipped entirely when
        # ``skip_preprocess_cache`` is set (per-year batch pins one salt; a
        # cached year would carry tokens from another salt and break the
        # cross-year patient union -- see ``pin_patient_id_salt``).
        if not cfg.skip_preprocess_cache:
            try:
                from .preprocess_cache import save_preprocess_full_cache as _save_full_cache
                from . import region_resolver as _rr
                # Population counts were already accumulated in the single
                # analyze_population_and_ct_doses scan (pre-CT-no-dose-filter);
                # reuse them here instead of a second O(N) pass.
                _pop_counts_for_save = {
                    "Euro-American": cohort.pop_counts["Euro-American"],
                    "Asian": cohort.pop_counts["Asian"],
                    "unspecified": cohort.pop_counts["unspecified"],
                }
                _file_pop_for_save = [
                    {
                        "Euro-American": _fc["Euro-American"],
                        "Asian": _fc["Asian"],
                        "unspecified": _fc["unspecified"],
                    }
                    for _fc in cohort.file_pop_counts
                ]
                # Shallow snapshot: protects against concurrent updates while arrays remain read-only.
                _ea_cache_snapshot = dict(ea)

                def _save_full_cache_bg():
                    try:
                        _save_full_cache(
                            file_list, _ea_cache_snapshot,
                            _pop_counts_for_save, _file_pop_for_save, cohort.ct_no_dose,
                            cfg.use_golikov, cfg.use_mataac,
                            _rr.get_unknown_descriptions(),
                            cohort.n_acquisition_rows,
                            cohort.acq_rows_by_modality,
                            cohort.acq_rows_by_file,
                            needed_cols=cfg.needed_cols, merge_series=cfg.merge_series,
                            excluded_no_organ_dose_by_modality=dict(cohort.excluded_no_organ_dose),
                            participant_flow=cohort.participant_flow,
                            population=cohort.startup_population,
                            excluded_by_modality=cohort.excluded_by_modality,
                            excluded_dose_sv_by_modality=cohort.excluded_dose_sv_by_modality,
                            excluded_cohorts=cohort.excluded_cohorts,
                        )
                    except Exception:
                        pass
                    finally:
                        # Let the snapshot be garbage-collected once the save finishes.
                        _ea_cache_snapshot.clear()

                threading.Thread(target=_save_full_cache_bg, daemon=True).start()
            except Exception:
                pass
    return ea, exam_data, ages, n_cpu


class _EventSpool:
    """Per-iteration event buffer that spills to temp chunks to bound RAM use.

    Inactive when ``keep_events`` is False (no temp dir, ``collect`` returns
    the empty-keys dict).  Chunks live in ``chunk_{n:06d}`` dirs, one ``.npy``
    per column (avoids np.savez ZIP overhead).
    """

    _MAX_BYTES = 256 * 1024 * 1024

    def __init__(self, keep_events: bool):
        self._keep = keep_events
        self._dir = tempfile.mkdtemp(prefix="pir_mc_events_") if keep_events else None
        self._buffer: list = []
        self._files: list = []
        self._buffer_bytes_total: int = 0

    def flush(self) -> None:
        if not self._buffer or not self._dir:
            return
        concatenated: dict = {}
        keys = list(self._buffer[0].keys())
        for k in keys:
            arrs = [ev[k] for ev in self._buffer if k in ev]
            if arrs:
                concatenated[k] = np.concatenate(arrs)
        if concatenated:
            # One directory per chunk; one .npy file per column avoids np.savez ZIP overhead.
            chunk_dir = os.path.join(self._dir, f"chunk_{len(self._files):06d}")
            os.makedirs(chunk_dir, exist_ok=True)
            for k, arr in concatenated.items():
                np.save(os.path.join(chunk_dir, f"{k}.npy"), arr)
            self._files.append(chunk_dir)
        self._buffer.clear()
        self._buffer_bytes_total = 0

    def add(self, ev_col) -> None:
        if not self._keep or not ev_col:
            return
        if any(isinstance(v, np.ndarray) and v.size > 0 for v in ev_col.values()):
            self._buffer.append(ev_col)
            self._buffer_bytes_total += sum(int(arr.nbytes) for arr in ev_col.values() if isinstance(arr, np.ndarray))
            if self._buffer_bytes_total > self._MAX_BYTES:
                self.flush()

    def collect(self) -> dict[str, list]:
        """Merge flushed chunks or in-memory events into one per-column list."""
        concatenated = {k: [] for k in _EVENT_COLUMN_KEYS}
        if self._keep:
            if self._files:
                self.flush()
                for chunk_dir in self._files:
                    # Object arrays are pickled inside .npy files; data is self-generated and trusted.
                    for key in _EVENT_COLUMN_KEYS:
                        col_path = os.path.join(chunk_dir, f"{key}.npy")
                        if os.path.exists(col_path):
                            concatenated[key].append(np.load(col_path, allow_pickle=True))
            else:
                for ev_col in self._buffer:
                    for key in _EVENT_COLUMN_KEYS:
                        if key in ev_col:
                            concatenated[key].append(ev_col[key])
        return concatenated

    def cleanup(self) -> None:
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)


class _ConvergenceMonitor:
    """Per-bin risk-per-Sv stability tracker + precision stopping rule.

    Welford's online algorithm per age bin (1 y bins 0..105): per-bin 95 %
    CI half-width must be stable (relative change < ``convergence_pct``) for
    20 consecutive iterations after the minimum-iterations floor.  When the
    precision rule is enabled, the rel. SE of the annual cancer count and
    annual QALYs must also be at or below ``precision_target_rel_se``.
    """

    def __init__(self, ages, ed_sv, cfg: _SimConfig):
        # Use the user minimum if enabled, otherwise a default of 30.
        self.min_iterations = cfg.min_iterations if cfg.min_iterations_enabled else DEFAULT_MIN_ITERATIONS
        self.rel_threshold = cfg.convergence_pct / 100.0
        self.window = 20
        self._precision_enabled = cfg.precision_stop_enabled
        self._target_rel_se = cfg.precision_target_rel_se
        self.age_bins = np.arange(0, 106, 1)   # 0,1,2,...,105 → 105 bins
        self.n_bins = len(self.age_bins) - 1
        exam_age_bin_idx = _bin_index(ages, self.age_bins, self.n_bins)
        # dose per bin (constant, in Sv) — reuse ed_sv from ea dict (already computed)
        self.dose_per_bin = np.zeros(self.n_bins, dtype=float)
        np.add.at(self.dose_per_bin, exam_age_bin_idx, ed_sv)
        # mask bins with meaningful dose for convergence (ignore empty / near-zero bins)
        self.meaningful_bin = self.dose_per_bin > 1e-9

        # Welford's online algorithm per bin: track mean & M2 of per-iteration
        # risk_per_sv values so we can compute a running standard deviation.
        self._welford_count = np.zeros(self.n_bins, dtype=float)   # n
        self._welford_mean  = np.zeros(self.n_bins, dtype=float)   # running mean
        self._welford_M2    = np.zeros(self.n_bins, dtype=float)   # sum of squared deviations
        self.prev_ci_hw     = None          # previous 95% CI half-width per bin

        self.converged = False
        self.n_iter_done = 0
        self.stable_count = 0              # consecutive iterations where ALL metrics stable
        self.stop_reason = "not_converged"
        self.rel_se_cases = None           # rel. SE of annual cancer count (std/sqrt(n)/mean)
        self.rel_se_qaly = None            # rel. SE of annual QALYs (std/sqrt(n)/mean)

    def update(self, partial_result, iter_cases_list, iter_QALY_list):
        """Check 95% CI half-width stabilization for per-bin risk-per-Sv.

        Uses Welford's online algorithm. Per-bin metric must be stable
        (relative CI half-width change < threshold) for 20 consecutive
        iterations after minimum iterations.  When the precision-based
        stopping rule is enabled, the stability window alone is not
        enough: the relative SE of the annual cancer count and the annual
        QALYs (std(ddof=1)/sqrt(n)/mean over the per-iteration lists)
        must also fall at or below ``precision_target_rel_se``.
        """
        self.n_iter_done += 1

        # Per-bin case counts from this iteration's events (columnar path).
        iter_case = np.zeros(self.n_bins, dtype=float)
        ev = partial_result.get("events")
        if ev and len(ev["age_at_exposure"]) > 0:
            _ev_bins = _bin_index(ev["age_at_exposure"], self.age_bins, self.n_bins)
            np.add.at(iter_case, _ev_bins, 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            iter_risk = np.where(self.dose_per_bin > 0, iter_case / self.dose_per_bin, 0.0)

        m_bin = self.meaningful_bin
        self._welford_count[m_bin] += 1.0
        delta = iter_risk[m_bin] - self._welford_mean[m_bin]
        self._welford_mean[m_bin] += delta / self._welford_count[m_bin]
        delta2 = iter_risk[m_bin] - self._welford_mean[m_bin]
        self._welford_M2[m_bin] += delta * delta2

        if self.n_iter_done < self.min_iterations:
            return

        # Per-bin 95% CI half-width (only meaningful bins matter)
        n_m = self._welford_count[m_bin]
        m2_m = self._welford_M2[m_bin]
        with np.errstate(divide="ignore", invalid="ignore"):
            std_m = np.where(n_m > 1, np.sqrt(m2_m / np.maximum(n_m - 1, 1)), 0.0)
            ci_hw_m = np.where(n_m > 0, Z_95 * std_m / np.sqrt(np.maximum(n_m, 1)), 0.0)

        # Check per-bin metric for stabilization
        bin_stable = False
        if self.prev_ci_hw is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_change_bins = np.where(
                    self.prev_ci_hw > 1e-15,
                    np.abs(ci_hw_m - self.prev_ci_hw) / self.prev_ci_hw,
                    0.0,
                )
            if rel_change_bins.size > 0 and float(np.max(rel_change_bins)) < self.rel_threshold:
                bin_stable = True

        if bin_stable:
            self.stable_count += 1
        else:
            self.stable_count = 0

        # Relative SE of the annual cancer count and annual QALYs over the
        # per-iteration lists (std(ddof=1)/sqrt(n)/mean; None when n<2 or
        # mean<=0).  Recomputed each iteration -- n is small, so the std
        # over the list is cheap.
        self.rel_se_cases = _rel_se_of_iter_list(iter_cases_list)
        self.rel_se_qaly = _rel_se_of_iter_list(iter_QALY_list)

        # Converge once the stability window is reached AND (when enabled)
        # the precision criterion on cancers & QALYs is met.
        _should, _reason = _stopping_decision(
            self.n_iter_done, self.min_iterations, self.stable_count, self.window,
            self._precision_enabled, self.rel_se_cases, self.rel_se_qaly,
            self._target_rel_se)
        if _should:
            self.converged = True
            self.stop_reason = _reason

        self.prev_ci_hw = ci_hw_m.copy()

    def status_str(self):
        """Short convergence status for GUI detail line."""
        if self.converged:
            return " [CONVERGED]"
        if self.n_iter_done < self.min_iterations:
            return f" [min {self.min_iterations}]"
        if self._precision_enabled and (self.rel_se_cases is not None
                                        or self.rel_se_qaly is not None):
            _cs = f"{self.rel_se_cases * 100:.1f}%" if self.rel_se_cases is not None else "n/a"
            _qs = f"{self.rel_se_qaly * 100:.1f}%" if self.rel_se_qaly is not None else "n/a"
            _tgt = self._target_rel_se * 100
            _ok_c = self.rel_se_cases is not None and self.rel_se_cases <= self._target_rel_se
            _ok_q = self.rel_se_qaly is not None and self.rel_se_qaly <= self._target_rel_se
            _c_mark = "<=" if _ok_c else ">"
            _q_mark = "<=" if _ok_q else ">"
            return (f" [stable {self.stable_count}/{self.window}, "
                    f"relSE c {_cs}{_c_mark}{_tgt:.0f}% q {_qs}{_q_mark}{_tgt:.0f}%]")
        if self.stable_count > 0:
            return f" [stable {self.stable_count}/{self.window}]"
        return " [monitoring]"

    def stop_reason_message(self) -> str:
        """Human-readable reason a converged run stopped (for the progress line)."""
        return {
            "stability_and_precision":
                f"95% CI half-width stable (<{self.rel_threshold*100:.0f}% "
                f"change) for {self.window} consecutive years AND rel. SE "
                f"of cancers/QALYs <= {self._target_rel_se*100:.1f}%",
            "stability_only":
                f"95% CI half-width stable (<{self.rel_threshold*100:.0f}% "
                f"change) for {self.window} consecutive years "
                f"(precision stop disabled)",
        }.get(self.stop_reason, "")

    def metadata(self, stats_iterations, n_iterations) -> dict:
        """Convergence metadata dict stored as ``stats["_convergence"]``."""
        return {
            "iterations": int(stats_iterations or self.n_iter_done),
            "stop_reason": self.stop_reason,
            "min_iterations": self.min_iterations,
            "max_iterations": n_iterations,
            "stability_threshold": self.rel_threshold,
            "stability_window": self.window,
            "precision_stop_enabled": self._precision_enabled,
            "precision_target_rel_se": self._target_rel_se,
            "rel_se_cases": self.rel_se_cases,
            "rel_se_qaly": self.rel_se_qaly,
        }


def _run_one_iteration(args, *, ea, exam_data, rpg, min_latency_params,
                       population, use_gpu, max_parallel, cfg: _SimConfig):
    """Execute a single MC iteration. Thread-safe: only local state.

    ``args`` is ``(it_idx, seed, device_id, slot_idx)``:
    * ``device_id`` — CUDA device to pin this worker to (``None`` on CPU).
    * ``slot_idx``  — index into that device's scratch-buffer ring
      (``_gpu_caches[device_id].slots[slot_idx]`` on GPU,
      ``_cpu_cache.slots[slot_idx]`` on CPU).  Each of the
      ``max_parallel`` in-flight iterations owns a unique
      (device, slot) pair so there is no contention on scratch buffers.
    """
    it_idx, seed, device_id, slot_idx = args
    # Pin this worker to its assigned CUDA device.
    if use_gpu and cp is not None and device_id is not None:
        cp.cuda.Device(int(device_id)).use()
    # Create per-thread CUDA stream for overlapping operations
    if use_gpu and cp is not None:
        stream = cp.cuda.Stream()
    else:
        stream = None
    # Derive independent child seeds: ls_child is kept (unused) to
    # preserve the existing seed-derivation contract; mc_child seeds
    # the MC Bernoulli draws AND the per-hit conditional lifespan draw
    # inside vectorized_iteration (see vectorized.py's Notes for
    # vectorized_iteration -- a whole-cohort, exposure-age-rooted
    # lifespan is not sampled here: it is only needed for the hit
    # subset, and doing so independently of the cancer's onset
    # age could make onset exceed the sampled lifespan even for hits
    # the case-fatality draw marked as fatal, producing a "died" event
    # with 0 life-years-lost).
    iter_ss = np.random.SeedSequence(seed)
    _, mc_child = iter_ss.spawn(2)
    _t_s = time.perf_counter() if cfg.profile else 0.0
    if cfg.profile:
        # Sync GPU kernels so profiling splits are accurate.
        if use_gpu and cp is not None:
            try:
                cp.cuda.runtime.deviceSynchronize()
            except Exception:
                pass
        _t_i = time.perf_counter()
    else:
        _t_i = 0.0
    mc_seed = int(mc_child.generate_state(1, dtype=np.uint64)[0])
    # Run the vectorised iteration on the per-device CUDA stream if available.
    if stream is not None:
        with stream:
            result = _vec_iteration(
                ea, exam_data, rpg, min_latency_params,
                seed=mc_seed, iteration_idx=it_idx,
                get_cancer_params_cached_fn=get_cancer_params_cached,
                ddref_solid=cfg.ddref_solid, ddref_leuk=cfg.ddref_leuk,
                population=population, use_gpu=use_gpu,
                n_slots=max_parallel,
                slot_idx=slot_idx,
            )
        stream.synchronize()
    else:
        result = _vec_iteration(
            ea, exam_data, rpg, min_latency_params,
            seed=mc_seed, iteration_idx=it_idx,
            get_cancer_params_cached_fn=get_cancer_params_cached,
            ddref_solid=cfg.ddref_solid, ddref_leuk=cfg.ddref_leuk,
            population=population, use_gpu=use_gpu,
            n_slots=max_parallel,
            slot_idx=slot_idx,
        )
    if cfg.profile:
        result["_prof_sampling"] = _t_i - _t_s
        result["_prof_iter"] = time.perf_counter() - _t_i
    return result


def run_simulation(file_list, n_iterations, min_latency_params,
                   population="Euro-American", on_progress=None, risk_model="REIC",
                   sim_config: dict = None, seed=None):
    """Run the full Monte Carlo lifetime cancer-risk simulation.

    Pipeline
    --------
    1. Load Excel files (parallel when multiple files).
    2. Pre-process exams: modality dispatch, organ-dose lookup,
       population tagging, Golikov weight correction,
       Mataac frailty multiplier, and cumulative dose tracking.
    3. Merge multi-acquisition sub-series into one exam per
       examination (optional, controlled by ``merge_acquisition_series``).
    4. Pre-compute ICRP 152 risk kernels and survival curves once.
    5. Build vectorised exam arrays (stable-sorted by patient).
    6. Run Monte Carlo iterations until convergence or until
       ``n_iterations`` is reached.
    7. Each iteration: draw Bernoulli outcomes per cancer site per exam
       → sample a natural lifespan per cancer hit, conditioned on that
       hit's onset age → collect cases, deaths, QALYs lost.
    8. Track 95 % CI half-width with Welford's online algorithm;
       stop early when stable for 20 consecutive iterations.

    Parameters
    ----------
    file_list : list of str or Path
        Excel files to load.  May be a single file or multiple files.
    n_iterations : int
        Hard maximum number of MC years to simulate.
    min_latency_params : dict
        Minimum latency years per cancer type (passed to
        ``configure_simulation``).
    population : {"Euro-American", "Asian", "Euro-American/Asian"}
        Demographic baseline for the risk model.  ``"Euro-American/Asian"``
        triggers per-patient mixed kernels when both populations are
        present in the data.
    on_progress : callable, optional
        GUI callback: ``on_progress(stage, *args)``.  Stages include
        ``"init"``, ``"precompute"``, ``"prepare_iterations"``,
        ``"iteration"``, ``"unknown_regions"``, ``"missing_ct_doses"``.
        For ``"missing_ct_doses"`` the handler must return ``True`` if it
        will answer the prompt (set the passed ``threading.Event`` via the
        ``continue_result`` list); any other return value means no one will
        answer and the engine continues immediately.
    risk_model : {"REIC", "LAR"}
        Risk integral type (REIC = default, LAR = opt-in).
    sim_config : dict, optional
        Fine-grained controls:

        * ``ddref_solid`` — DDREF for solid cancers (default 2.0).
        * ``ddref_leuk`` — DDREF for leukaemia (default 1.0).
        * ``use_golikov`` — Enable Golikov body-size correction (default True).
        * ``use_mataac`` — Enable Mataac frailty multiplier (default True).
        * ``merge_acquisition_series`` — Collapse per-acquisition rows
          (default True).
        * ``convergence_pct`` — Relative CI change threshold [%] for
          early stopping (default 1.0).
        * ``min_iterations_enabled`` / ``min_iterations`` — Custom
          minimum iterations before convergence may stop (default
          disabled, built-in minimum 1000).
        * ``icrp152_strict`` — Disable Golikov and Mataac (default False).
        * ``QALY_mode`` / ``QALY_discount_rate`` — QALYs accounting mode.
        * ``gpu_iters_per_device`` — Concurrent iterations per GPU
          (default 1, VRAM-clamped downstream).
        * ``profile`` — Enable per-phase timing breakdown (default False).
        * ``keep_events`` — Store the full per-cancer-event list in the
          result (default True).  Set to False to save RAM / disk I/O when
          only summary statistics are needed; the GUI's Cancer Events tab
          and Research Questions refinement will not be available.
    seed : int, optional
        Root seed for the Monte Carlo random streams.  ``None`` (default)
        draws the root seed from OS entropy, so every run is an independent
        realisation.  Give an integer to make a run reproducible: the same
        input files, configuration and seed give the same output.

    Returns
    -------
    stats : dict
        Aggregated simulation results (cases, deaths, QALYs lost,
        modality breakdown, cancer breakdown, heritable risk, etc.).
    iter_cases_list : list of int
        Per-iteration cancer case counts.
    iter_deaths_list : list of int
        Per-iteration cancer death counts.
    all_events : list of dict
        Every cancer event (age, sex, modality, region, cancer type).
    exam_data : list of dict
        Final processed exam records (post-merge, post-filter).
    """
    # Unpack simulation config (with safe defaults)
    _cfg = sim_config or {}
    cfg = _SimConfig.from_dict(sim_config)

    cohort = _prepare_cohort(file_list, cfg, population, on_progress)
    if cohort.error is not None:
        return {"error": cohort.error}, [], [], [], []
    _full_cached = cohort.full_cached
    ea = cohort.ea
    exam_data = cohort.exam_data
    total = cohort.total
    ages = cohort.ages
    n_cpu = cohort.n_cpu
    population = cohort.population
    _ea_ratio = cohort.ea_ratio
    _as_ratio = cohort.as_ratio
    _use_mixed_kernels = cohort.use_mixed_kernels
    _pop_counts = cohort.pop_counts
    _file_pop_counts = cohort.file_pop_counts
    _ct_no_dose = cohort.ct_no_dose
    _n_acquisition_rows = cohort.n_acquisition_rows
    _acq_rows_by_modality = cohort.acq_rows_by_modality
    _acq_rows_by_file = cohort.acq_rows_by_file
    _file_labels = cohort.file_labels
    _excluded_no_organ_dose = cohort.excluded_no_organ_dose
    _participant_flow = cohort.participant_flow
    _excluded_by_modality = dict(cohort.excluded_by_modality or {})
    _excluded_dose_sv_by_modality = dict(cohort.excluded_dose_sv_by_modality or {})
    # Exclusion-ledger count tables ({reason key: CohortByFile}, see
    # participant_flow.build_excluded_cohorts): restored from the full
    # cache on the warm path, built here from the raw dropped exams on the
    # cold path -- before the arrays are built and cached, so the tables
    # ride along in the cache and the raw lists can be released.
    _excluded_cohorts = cohort.excluded_cohorts
    if _excluded_cohorts is None and cohort.excluded_exams:
        try:
            from .participant_flow import build_excluded_cohorts
            _excluded_cohorts = build_excluded_cohorts(
                cohort.excluded_exams, file_list,
                group_label=_cfg.get("_year_group"))
        except Exception:
            _excluded_cohorts = None
        cohort.excluded_cohorts = _excluded_cohorts
    cohort.excluded_exams = None

    # 3. Precompute risk kernels and survival curves once
    if on_progress:
        on_progress("precompute", 0, 3, "Preparing kernel cache...")
    configure_simulation({
        **_cfg,
        "population": population,
        "risk_model":  risk_model,
        **min_latency_params,
    })
    if _use_mixed_kernels:
        precompute_kernels(max_age=MAX_ATTAINED_AGE_ICRP152, population="Euro-American")
        precompute_kernels(max_age=MAX_ATTAINED_AGE_ICRP152, population="Asian")
    else:
        precompute_kernels(max_age=MAX_ATTAINED_AGE_ICRP152, population=population)

    # 4. Vectorized engine setup
    rpg = get_vectorized_rpg_2pop() if _use_mixed_kernels else get_vectorized_rpg()
    rpg_ct_to_idx = rpg[4]  # ct_to_idx dict

    ea, exam_data, ages, n_cpu = _build_exam_arrays_and_cache(
        cohort, rpg_ct_to_idx, cfg, file_list, on_progress)

    if on_progress:
        _pop_label = (f"EA({_ea_ratio:.0%}) + Asian({_as_ratio:.0%}) per-patient kernels"
                      if _use_mixed_kernels else population)
        on_progress("precompute", 1, 3,
                    f"Kernel cache ready for {len(CANCER_TO_ORGAN_DOSE_KEY)} mapped cancer types "
                    f"({risk_model}, {_pop_label})")

    if on_progress:
        on_progress("precompute", 2, 3, "Building population split cache...")

    # Auto-detect GPU backend
    use_gpu = False
    n_gpu = 0
    if cp is not None:
        try:
            _ = cp.array([0.0])
            use_gpu = True
            n_gpu = max(0, int(cp.cuda.runtime.getDeviceCount()))
        except Exception:
            pass

    # Note: Iterations run sequentially (GPU is bottleneck, not CPU)
    # Each iteration processes all exams in one GPU batch for maximum utilization

    if on_progress:
        if use_gpu:
            backend_label = f"GPU (CuPy, {n_gpu} device{'s' if n_gpu != 1 else ''})"
        else:
            backend_label = f"CPU ({n_cpu} threads)"
        on_progress("prepare_iterations", 0, 1,
                     f"Vectorized engine: {total:,} exams, "
                     f"{len(rpg[3])} solid types, backend={backend_label} (1 iteration = 1 year)")

    # 5. Monte Carlo iteration loop with 95% CI convergence
    monitor = _ConvergenceMonitor(ages, ea["ed_sv"], cfg)

    # Track timing for throughput calculation
    _iteration_start_time = time.perf_counter()

    # Final per-modality/per-file examination counts after merge and dose filtering.
    _exams_by_modality = _count_by_modality(exam_data)
    _exams_by_file = _count_by_file(exam_data, _file_labels)

    # Per-input-file cohort profiles for the GUI's Patient Cohort and By Body
    # Region tabs.  Built here, from the post-merge, post-filter, post-sort
    # exam set, because this is the last point at which every exam of *this*
    # run is guaranteed alive: a per_year_average batch drops exam_data before
    # loading the next year, and the profiles are what lets those tabs still
    # show every file of every year (see file_cohort's module docstring).
    # A few kB per file, so keeping them costs nothing next to exam_data.
    try:
        from .file_cohort import build_file_cohort
        _file_cohort = build_file_cohort(exam_data, file_list,
                                         group_label=_cfg.get("_year_group"))
    except Exception:
        _file_cohort = None
    if on_progress and _file_cohort is not None:
        on_progress("file_cohort", _file_cohort)
    if on_progress and _excluded_cohorts:
        on_progress("excluded_cohorts", _excluded_cohorts)

    # Weight-adjustment summary for the Summary tab's Adjustments view,
    # built while this run's exams are alive (a per_year_average batch
    # drops them before the next year).
    try:
        from .analysis import summarize_weight_adjustment
        _weight_adjustment = summarize_weight_adjustment(exam_data)
    except Exception:
        _weight_adjustment = None

    stats = {
        "iterations": 0, "acquisitions": 0,
        "acquisition_rows": _n_acquisition_rows,
        "acquisition_rows_by_modality": dict(_acq_rows_by_modality),
        "examinations_by_modality": dict(_exams_by_modality),
        # Per-modality count of exams dropped upstream by the CT/Dental-CBCT
        # no-organ-dose filter (see _excluded_no_organ_dose). Empty on the
        # warm-cache path and when no CT doses were missing. Reported in
        # format_summary section 2 alongside the risk-modelled exclusions.
        "excluded_no_organ_dose_by_modality": dict(_excluded_no_organ_dose),
        # Exams dropped at source because they carry no risk model (see
        # participant_flow.py): Non-Dose rows and exams with no organ-dose
        # model, per modality, plus the effective dose those exams carried
        # (e.g. male mammography).  Flat {modality: n} tables so year_batch
        # averages them per year like examinations_by_modality and
        # format_summary section 2 can read them with
        # _exams_by_modality_for_summary.
        "excluded_non_dose_by_modality": dict(
            _excluded_by_modality.get("non_dose") or {}),
        "excluded_no_dose_model_by_modality": dict(
            _excluded_by_modality.get("no_dose_model") or {}),
        "excluded_dose_sv_by_modality": _sum_count_tables(
            _excluded_dose_sv_by_modality.values()),
        # {reason key: CohortByFile} count tables of the excluded exams.
        # Not a numeric tree: year_batch merges it explicitly (sum of
        # counts) like _file_cohort; rendered by the By Body Region tab.
        "_excluded_cohorts": _excluded_cohorts,
        # Per-input-file breakdown, keyed by disambiguated basename in file
        # order.  Only rendered by format_summary when there is more than
        # one file -- a single-file run's numbers are identical to the
        # totals above, so the section would add nothing.
        "acquisition_rows_by_file": dict(_acq_rows_by_file),
        "examinations_by_file": dict(_exams_by_file),
        # Mergeable per-file count tables; see file_cohort.py.  Not a numeric
        # tree, so year_batch excludes it from the cross-year average and
        # merges it explicitly instead.
        "_file_cohort": _file_cohort,
        "total_sv": 0.0, "cancer_cases": 0, "cancer_deaths": 0, "expected_deaths": 0.0,
        "expected_incidence": 0.0,
        "QALY_lost": 0.0, "life_lost": 0.0, "total_detriment": 0.0,
        "modality_stats": {}, "cancer_breakdown": {},
        "device_stats": {}, "year_stats": {},
        "heritable_risk": 0.0, "heritable_detriment": 0.0, "rem_subtype_breakdown": {},
        # STROBE/RECORD participant-flow ledger: built on the cold path,
        # restored from the v4 full cache on the warm path (see
        # participant_flow.py).  Rendered by analysis.format_summary.
        "_participant_flow": _participant_flow,
        # Per-region weight-adjustment counts and doses; summed across
        # years by year_batch (analysis.merge_weight_adjustments).
        "_weight_adjustment": _weight_adjustment,
        "risk_model": str(risk_model).upper(), "population": population,
        "sim_config": {
            "ddref_solid": cfg.ddref_solid, "ddref_leuk": cfg.ddref_leuk,
            "use_golikov": cfg.use_golikov, "use_mataac": cfg.use_mataac,
            "convergence_pct": cfg.convergence_pct,
            "precision_stop_enabled": cfg.precision_stop_enabled,
            "precision_target_rel_se": cfg.precision_target_rel_se,
            "icrp152_strict": cfg.icrp152_strict,
            "background_radiation_msv_per_year": cfg.background_rad,
            "QALY_mode": cfg.QALY_mode,
            "QALY_discount_rate": cfg.QALY_discount_rate,
            "population": population,
            "risk_model": str(risk_model).upper(),
            "skip_preprocess_cache": cfg.skip_preprocess_cache,
        },
    }
    # Precompute patient-id aggregates and source counts once, while
    # exam_data is still alive, and cache them on stats.  The live Summary
    # tab renders format_summary with exam_data=None (to avoid blocking the
    # worker thread), so without this cache unique_patients would silently
    # fall back to unique_examinations during the simulation -- making the
    # two counts identical mid-run even though they differ at the end.
    # The cached form mirrors what format_summary itself stores: pid_counts
    # plus none/nan tallies, with the heavy pid_to_dose/pid_to_exams dicts
    # dropped (only the count is needed for the header).
    try:
        from .analysis import _build_patient_aggregates
        _pid_counts, _, _, _pid_none_n, _pid_nan_n = _build_patient_aggregates(exam_data)
        stats["_pid_aggregates"] = (_pid_counts, {}, {}, _pid_none_n, _pid_nan_n)
        from collections import Counter as _Counter
        # When exam_data is a ColumnarExamsList, read the _person_id_source
        # column directly instead of iterating N proxies -- same counts, no
        # per-row dict.get overhead.  None values are mapped to "unknown" to
        # match the per-row expression's ``ex.get("_person_id_source", "unknown")``
        # semantics (a missing key -> default "unknown"; a present-but-None
        # value -> "unknown" here too, since the columnar path can't tell the
        # two apart once the column exists).
        if hasattr(exam_data, "_cols"):
            _src_col = exam_data._cols.get("_person_id_source") or [None] * len(exam_data)
            stats["_pid_source_counts"] = _Counter(
                (s if s is not None else "unknown") for s in _src_col)
        else:
            stats["_pid_source_counts"] = _Counter(
                ex.get("_person_id_source", "unknown") for ex in exam_data
            )
        from .analysis import build_patients_by_modality
        stats["_patients_by_modality"] = build_patients_by_modality(exam_data)
    except Exception:
        pass
    # Cohort-weighted ICRP 152 baseline incidence, computed while exam_data
    # is alive so the live Summary (format_summary with exam_data=None) can
    # still resolve it.  Own statement on purpose: a failure must raise, not
    # be swallowed by the diagnostics block above.
    from .analysis import _cohort_weighted_baseline_incidence_with_n
    _bl = _cohort_weighted_baseline_incidence_with_n(exam_data)
    if _bl is not None:
        stats["_baseline_incidence_cohort"] = {
            "value": _bl[0], "n_patients": _bl[2], "source": _bl[1],
        }
    iter_cases_list = []
    iter_deaths_list = []
    # Per-iteration QALYs and life-years lost for the Summary tab distributions.
    iter_QALY_list = []
    iter_life_list = []

    # Buffer per-iteration events and flush to temp chunks to bound RAM use.
    # Skipped entirely when keep_events is False.
    spool = _EventSpool(cfg.keep_events)

    # Live event-payload throttle (C) and per-phase profiling accumulators (A).
    _PROGRESS_EVENT_EVERY = 10
    _prof_acc = {"sampling_s": 0.0, "gpu_iter_s": 0.0, "host_collect_s": 0.0}

    # Reset persistent CuPy generators so each run is an independent MC realisation.
    _vec_invalidate_persistent_cp_rng(ns=MAIN_NS)

    # SeedSequence() without arguments uses OS-level entropy (/dev/urandom on Unix, CryptGenRandom on Windows)
    # This is cryptographically secure and truly stochastic.  An explicit
    # ``seed`` makes the run reproducible.
    run_seed_sequence = np.random.SeedSequence() if seed is None else np.random.SeedSequence(int(seed))

    def _collect_and_report(p, completed):
        """Merge one iteration result and stream to GUI (caller's thread only)."""
        _t_h = time.perf_counter() if cfg.profile else 0.0
        _merge_partial(stats, p)
        ic = p["cases"]
        id_ = p["deaths"]
        spool.add(p["events"])
        stats["acquisitions"] += total
        stats["iterations"] = completed
        iter_cases_list.append(ic)
        iter_deaths_list.append(id_)
        iter_QALY_list.append(float(p.get("QALY", 0.0) or 0.0))
        iter_life_list.append(float(p.get("life_lost", 0.0) or 0.0))
        monitor.update(p, iter_cases_list, iter_QALY_list)
        if on_progress:
            # Stream events only every N iterations to avoid saturating the
            # worker→UI queue.  The full event set is delivered at the end.
            _send_events = (
                completed % _PROGRESS_EVENT_EVERY == 0
                or monitor.converged
                or completed >= n_iterations
            )
            
            # Send columnar events directly; the GUI converts only the visible subset.
            progress_events = None
            if _send_events and p["events"] and len(p["events"].get("cancer_type", [])) > 0:
                progress_events = EventColumns(**p["events"])

            on_progress("iteration", stats, completed, n_iterations, ic, id_,
                         f"Year cases: {ic}, deaths: {id_}{monitor.status_str()}",
                         progress_events)
        if cfg.profile:
            _prof_acc["host_collect_s"] += time.perf_counter() - _t_h
            _prof_acc["sampling_s"] += float(p.get("_prof_sampling", 0.0) or 0.0)
            _prof_acc["gpu_iter_s"] += float(p.get("_prof_iter", 0.0) or 0.0)

    # Run up to max_parallel iterations concurrently; plan_iteration_slots assigns workers to devices/slots.
    _desired_per_device = cfg.gpu_iters_per_device if use_gpu else max(1, int(_cfg.get("cpu_workers") or os.cpu_count() or 1))
    _blas_ctrl_main = None
    if not use_gpu:
        try:
            import threadpoolctl
            _blas_ctrl_main = threadpoolctl.threadpool_limits(limits=1, user_api="blas")
        except Exception:
            _blas_ctrl_main = None
    with _vec_gpu_run_section(use_gpu):
        try:
            try:
                slot_plan = _vec_plan_iteration_slots(
                    _desired_per_device, ea["N"], len(rpg[3]), bool(use_gpu)
                )
            except Exception:
                # Defensive: never let slot planning crash the run — fall back to
                # a single-slot CPU-style plan so the run still completes.
                slot_plan = [(None, 0)]
            if not slot_plan:
                slot_plan = [(None, 0)]

            max_parallel = len(slot_plan)

            _iteration_task = functools.partial(
                _run_one_iteration,
                ea=ea, exam_data=exam_data, rpg=rpg,
                min_latency_params=min_latency_params,
                population=population, use_gpu=use_gpu,
                max_parallel=max_parallel, cfg=cfg,
            )

            # Precompute the iteration-invariant risk arrays once, on the main thread,
            # so every worker of the first wave reuses them instead of each lazily
            # recomputing them under the slot lock.  The lazy fallback inside
            # vectorized_iteration remains (research_questions.py relies on it).
            try:
                _vec_precompute_risk_invariants(
                    ea, rpg, get_cancer_params_cached,
                    cfg.ddref_solid, cfg.ddref_leuk,
                    use_gpu=use_gpu,
                )
            except Exception:
                pass

            # Surface the effective parallelism to the progress UI so the user can
            # see whether multi-GPU actually kicked in.
            if on_progress:
                _dev_counts: dict = {}
                for _d, _ in slot_plan:
                    _dev_counts[_d] = _dev_counts.get(_d, 0) + 1
                if len(_dev_counts) == 1 and next(iter(_dev_counts)) is None:
                    _sched_label = f"CPU ({max_parallel} concurrent iteration{'s' if max_parallel != 1 else ''})"
                else:
                    _parts = [
                        f"GPU{d}:{n}" for d, n in sorted(_dev_counts.items(),
                                                         key=lambda kv: -1 if kv[0] is None else kv[0])
                        if d is not None
                    ]
                    _sched_label = (
                        f"{max_parallel} concurrent iters across "
                        f"{len([d for d in _dev_counts if d is not None])} GPU(s) "
                        f"[{', '.join(_parts)}]"
                    )
                on_progress("prepare_iterations", 0, 1,
                            f"Scheduler: {_sched_label}")

            # Keep a fixed-size queue of futures to overlap GPU/CPU work with result merging.
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                inflight = deque()  # (iteration_idx, future) pairs
                it = 0

                def _submit_one_iteration(it_idx: int):
                    child = run_seed_sequence.spawn(1)[0]
                    seed = int(child.generate_state(1, dtype=np.uint64)[0])
                    _dev_id, _slot_idx = slot_plan[it_idx % max_parallel]
                    return executor.submit(_iteration_task, (it_idx, seed, _dev_id, _slot_idx))

                # Prime the pipeline: submit up to max_parallel iterations
                while it < n_iterations and len(inflight) < max_parallel:
                    inflight.append((it, _submit_one_iteration(it)))
                    it += 1

                # Main loop: collect oldest, report, submit next
                while inflight:
                    collect_it, fut = inflight.popleft()
                    p = fut.result()
                    _collect_and_report(p, collect_it + 1)

                    if monitor.converged:
                        # Cancel any remaining in-flight futures (Python 3.9+)
                        for _, remaining_fut in inflight:
                            remaining_fut.cancel()
                        break

                    # Submit next iteration if budget remains
                    if it < n_iterations:
                        inflight.append((it, _submit_one_iteration(it)))
                        it += 1

            if monitor.converged and on_progress:
                patients_simulated = stats["iterations"] * total
                _reason_msg = monitor.stop_reason_message()
                on_progress("iteration", stats, stats["iterations"], n_iterations,
                             iter_cases_list[-1], iter_deaths_list[-1],
                             f"Converged after {stats['iterations']:,} years "
                             f"({patients_simulated:,} patients). {_reason_msg}",
                             [])
            elif not monitor.converged:
                # Loop reached n_iterations without the stability+precision
                # criterion being met.
                monitor.stop_reason = "max_iterations"

            # Calculate throughput (acquisitions per second)
            _iteration_end_time = time.perf_counter()
            _total_time = _iteration_end_time - _iteration_start_time
            if _total_time > 0:
                stats["throughput"] = stats["acquisitions"] / _total_time
            else:
                stats["throughput"] = 0.0

            # Per-phase wall-clock breakdown (worker + serial host time).
            if cfg.profile:
                _n_done = max(1, int(stats.get("iterations", 0)) or 1)
                prof = {
                    "sampling_s": _prof_acc["sampling_s"],
                    "gpu_iter_s": _prof_acc["gpu_iter_s"],
                    "host_collect_s": _prof_acc["host_collect_s"],
                    "wall_s": _total_time,
                    "iterations": int(stats.get("iterations", 0)),
                    "gpu_iters_per_device": int(_desired_per_device),
                    "max_parallel": int(max_parallel),
                }
                stats["profile"] = prof
                _msg = (
                    "PROFILE  "
                    f"wall={_total_time:.1f}s  iters={prof['iterations']}  "
                    f"sampling={prof['sampling_s']:.1f}s  "
                    f"gpu_iter={prof['gpu_iter_s']:.1f}s  "
                    f"host_collect={prof['host_collect_s']:.1f}s  "
                    f"(per-iter host {prof['host_collect_s']/_n_done*1e3:.1f}ms)  "
                    f"parallel={max_parallel}"
                )
                print(f"[{_msg}]", flush=True)
                if on_progress:
                    try:
                        on_progress("init", stats["iterations"], stats["iterations"], _msg)
                    except Exception:
                        pass

            # Attach per-iteration QALYs / life-years series for the Summary tab.
            stats["iter_QALY_list"] = iter_QALY_list
            stats["iter_life_list"] = iter_life_list

            # Convergence metadata: iteration count, stop reason, the
            # stability + precision thresholds, and the final rel. SE of
            # the annual cancer count and annual QALYs.  Read by the
            # per-year precision table (year_batch) and the single-run
            # summary line (analysis.format_summary).
            stats["_convergence"] = monitor.metadata(
                stats.get("iterations", monitor.n_iter_done), n_iterations)

            # Merge flushed chunks or in-memory events into a single EventColumns.
            # Returns empty columns when keep_events is False.
            concatenated = spool.collect()
        finally:
            # Restore BLAS threading so downstream analysis is not penalised.
            if _blas_ctrl_main is not None:
                try:
                    _blas_ctrl_main.unregister()
                except Exception:
                    pass
            spool.cleanup()

    has_events = cfg.keep_events and any(len(v) > 0 for v in concatenated.values())
    if has_events:
        for key in _EVENT_COLUMN_KEYS:
            if concatenated[key]:
                concatenated[key] = np.concatenate(concatenated[key])
            else:
                concatenated[key] = getattr(EventColumns.empty(), key)
        event_columns = EventColumns(**concatenated)
    else:
        event_columns = _empty_event_columns()

    all_events = event_columns

    return stats, iter_cases_list, iter_deaths_list, all_events, exam_data


# 3. ENTRY POINT & GUI LAUNCHER

def start_gui(year_groups, n_iterations, min_latency_params, output_dir, population,
              run_mode: str = "per_year_average",
              risk_model: str = "REIC", sim_config: dict = None):
    """Launch the results GUI, wiring the (possibly multi-year) engine.

    Parameters
    ----------
    year_groups : dict[str, list[str]] or list[str]
        Input files grouped into user-named year buckets, e.g.
        ``{"2021": [...], "2022": [...]}``.  A plain list is also accepted
        and treated as a single unnamed year.
    run_mode : {"per_year_average", "combined"}
        See :func:`year_batch.run_yearly_simulation`.

    Notes
    -----
    ``launch_results_gui`` uses its ``file_list`` argument for exactly one thing:
    forwarding it into the engine callable.  The engine installed here ignores
    that copy and uses ``year_groups`` captured by closure, so the flattened
    list is passed only to satisfy the positional signature.
    """
    from .year_batch import run_yearly_simulation
    # Imported here (not at module level) so importing main.py -- and
    # therefore the whole package -- does not force tkinter/matplotlib to
    # load for callers that only need the engine, e.g. run_simulation().
    from .gui import start_gui as launch_results_gui

    if not isinstance(year_groups, dict):
        year_groups = {"All files": list(year_groups or [])}
        run_mode = "combined"

    def _engine(file_list, n_iterations, min_latency_params, population,
                on_progress=None, risk_model="REIC", sim_config=None):
        return run_yearly_simulation(
            year_groups, n_iterations, min_latency_params, population,
            run_mode=run_mode,
            on_progress=on_progress, risk_model=risk_model, sim_config=sim_config,
        )

    all_files = list(dict.fromkeys(
        f for files in year_groups.values() for f in files
    ))
    return launch_results_gui(all_files, n_iterations, min_latency_params, output_dir,
                          population, _engine, risk_model=risk_model,
                          backend_info_getter=get_backend_info,
                          sim_config=sim_config)


def _install_stall_dump_watchdog():
    """Diagnostic: dump every thread's stack on a fixed interval.

    Enabled only when ``DOSE2QALY_STALL_DUMP=<seconds>`` is set in the
    environment; a no-op otherwise (zero overhead on normal runs).  When an
    indefinite freeze occurs the repeated, identical dumps reveal exactly
    which line each thread (notably the background simulation worker) is
    blocked on -- something a ``Ctrl+C`` on the Tk main loop cannot show.
    Output goes to stderr and to ``<temp>/dose2qaly_stall_dump.log``.
    """
    _sec = os.environ.get("DOSE2QALY_STALL_DUMP", "").strip()
    if not _sec:
        return
    try:
        interval = float(_sec)
    except ValueError:
        return
    if interval <= 0:
        return
    import traceback as _traceback
    import datetime as _datetime

    log_path = os.path.join(tempfile.gettempdir(), "dose2qaly_stall_dump.log")

    def _dump_loop():
        while True:
            time.sleep(interval)
            ts = _datetime.datetime.now().isoformat(timespec="seconds")
            frames = sys._current_frames()
            names = {t.ident: t.name for t in threading.enumerate()}
            lines = [f"\n===== Dose2QALY stall dump {ts} "
                     f"({len(frames)} threads) ====="]
            for tid, frame in frames.items():
                lines.append(f"--- Thread {names.get(tid, '?')} (id={tid}) ---")
                lines.extend(s.rstrip("\n")
                             for s in _traceback.format_stack(frame))
            text = "\n".join(lines) + "\n"
            try:
                sys.stderr.write(text)
                sys.stderr.flush()
            except Exception:
                pass
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(text)
            except OSError:
                pass

    threading.Thread(target=_dump_loop, name="dose2qaly-stall-watchdog",
                     daemon=True).start()
    try:
        sys.stderr.write(f"[watchdog] stall dump every {interval}s "
                         f"-> {log_path}\n")
        sys.stderr.flush()
    except Exception:
        pass


if __name__ == "__main__":
    # Launch the GUI: config → load/preprocess → Monte Carlo → display.
    mp.freeze_support()
    _install_stall_dump_watchdog()

    _pkg_warnings = check_optional_packages()
    if _pkg_warnings:
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb
            _warn_root = _tk.Tk()
            _warn_root.withdraw()
            for _msg in _pkg_warnings:
                _mb.showinfo("Compute Backend Notice", _msg)
            _warn_root.destroy()
            del _tk, _mb, _warn_root
        except Exception:
            for _msg in _pkg_warnings:
                print(f"\nNOTICE: {_msg}\n", file=sys.stderr)
        del _pkg_warnings

    from .config import show_config_dialog
    cfg = show_config_dialog()
    if cfg is None:
        sys.exit(0)

    start_gui(
        year_groups=cfg.get("year_groups") or {"All files": cfg["file_list"]},
        run_mode=cfg.get("run_mode", "per_year_average"),
        n_iterations=cfg["n_iterations"],
        min_latency_params=cfg["min_latency_params"],
        output_dir=cfg["output_dir"],
        population=cfg["population"],
        risk_model=cfg["risk_model"],
        sim_config={
            "ddref_solid": cfg["ddref_solid"],
            "ddref_leuk": cfg["ddref_leuk"],
            "use_golikov": cfg["use_golikov"],
            "use_mataac": cfg["use_mataac"],
            "convergence_pct": cfg["convergence_pct"],
            "precision_stop_enabled": cfg.get("precision_stop_enabled", True),
            "precision_target_rel_se": cfg.get("precision_target_rel_se", 0.01),
            "min_iterations_enabled": cfg.get("min_iterations_enabled", False),
            "min_iterations": cfg.get("min_iterations", DEFAULT_MIN_ITERATIONS),
            "icrp152_strict": cfg.get("icrp152_strict", False),
            "background_radiation_msv_per_year": cfg.get("background_radiation_msv_per_year", BACKGROUND_MSV_PER_YEAR),
            "needed_cols": cfg.get("needed_cols"),  # None → excel_io falls back to DEFAULT_NEEDED_COLS
            # Optional performance / profiling settings.
            "gpu_iters_per_device": cfg.get("gpu_iters_per_device", 1),
            "cpu_workers": cfg.get("cpu_workers", None),  # None → auto
            "profile": cfg.get("profile", False),
        },
    )
