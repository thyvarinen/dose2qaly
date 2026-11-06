"""
Dose2QALY Vectorised Simulation Engine
======================================================
Batch NumPy / optional CuPy GPU implementation of the per-exam MC loop.
``main.run_simulation`` builds the per-exam arrays with
:func:`build_exam_arrays`, then calls the iteration kernels here; results
flow back to ``main`` for aggregation and to ``analysis`` for display.

Key operations:
  1. Precomputed per-exam risk tables for O(1) lookup.
  2. Bernoulli draws via a single (N, n_types) matrix comparison.
  3. Only the rare cancer-hit events (~0.01%) fall back to Python.
"""
from __future__ import annotations

import multiprocessing as _mp
import os
from typing import Any, Dict

import numpy as np

# Set CuPy environment variables for performance before importing CuPy
os.environ['CUPY_CUDA_PERFORMANCE'] = '1'
os.environ['CUPY_FAST_MATH'] = '1'
os.environ['CUPY_CUDA_GRAPH'] = '1'

from .risk_models import (
    _require_population,
    MAX_ATTAINED_AGE_ICRP152,
    OTHER_SOLID_REMAINDER_SITES,
    REMAINDER_ORGAN_MULTIPLICITY,
    HERITABLE_DETRIMENT_PER_GY,
    sample_cancer_latency_bulk,
    sample_lifespans_vectorized,
    _discount_interval,
    _SIM_CONFIG,
)
from .region_resolver import radiography_reporting_region

_TOTAL_REMAINDER_MULT: int = sum(REMAINDER_ORGAN_MULTIPLICITY.values())
_HERITABLE_RISK_COEFF: float = 20.0 / 10000.0  # ICRP 152 Table 3.7: heritable nominal risk per Gy
# ICRP 152 page 26 Table footnote: average years of life lost per radiogenic cancer.
_ICRP_AVG_LIFE_LOST_YEARS: float = 15.0


def _reporting_reg(name: str) -> str:
    """Strip the orientation suffix from a radiography region label.

    Dental-imaging subtypes (Panoramic/Kefalometry/Intraoral/Dental, and
    Dental CBCT) are passed straight through, keeping their own name --
    they are NOT collapsed into one flat "Dental imaging" bucket here, so
    ``modality_stats[...]["regions"]`` keeps per-subtype granularity for
    callers (research questions' modality/region breakdown) that group
    them into the "Dental imaging" pseudo-modality while still wanting to
    see each subtype separately.  The GUI's Summary tab already re-folds
    them into one "Dental imaging" row for its own display
    (``gui._compute_modality_region_rows``), so this is safe either way.
    """
    return radiography_reporting_region(name)


_HD_BY_SEX_NP: np.ndarray = np.array([          # heritable detriment per Gy by sex index M=0,F=1,U=2
    float(HERITABLE_DETRIMENT_PER_GY.get("M", 0.0)),
    float(HERITABLE_DETRIMENT_PER_GY.get("F", 0.0)),
    float(HERITABLE_DETRIMENT_PER_GY.get("U", 0.0)),
], dtype=np.float64)

# Cache namespaces: each namespace gets its own independent set of caches
# (device arrays / slot rings / RNGs) so two MC workloads can run
# concurrently on the GPU without sharing state.
MAIN_NS = "main"

_CPU_SOLID_VECS: dict = {}  # CPU-side k/det cache, keyed by namespace

# ---------------------------------------------------------------------------
# Optional CuPy GPU
# ---------------------------------------------------------------------------
try:
    # Spawned pool workers re-import this module; a CUDA context per worker
    # costs ~400 MB commit for nothing (same guard as main.py's cupy import).
    if _mp.parent_process() is not None:
        raise ImportError("GPU is never used in spawned worker processes")
    import cupy as _cp
    _cp.array([0.0])  # test allocation
    _HAS_GPU = True

    # Fused kernel: clip(risk * surv, 0, 1) -> risk, then (rand < risk) -> hit.
    # Replaces three separate kernel launches with one HBM round-trip.
    _FUSED_BERN_KERNEL = _cp.ElementwiseKernel(
        in_params='T risk_in, T surv, T rand',
        out_params='T risk_out, bool hit',
        operation='''
            T v = risk_in * surv;
            if (v < (T)0) v = (T)0;
            if (v > (T)1) v = (T)1;
            risk_out = v;
            hit = rand < v;
        ''',
        name='mc_fused_bern',
    )

    # Auto-configure the GPU memory pool from the visible free VRAM.
    mempool = _cp.get_default_memory_pool()
    free_mem = _cp.cuda.Device().mem_info[0]

    # Cap pool at 92% of free VRAM (leaves headroom for driver / OS).
    # ``size=`` keyword works on both CuPy 12.x and 13.x.
    mem_limit = int(free_mem * 0.92)
    mempool.set_limit(size=mem_limit)

    # Scale lifespan chunk size by available VRAM.
    if free_mem >= 8_589_934_592:  # >= 8GB
        _LIFESPAN_CHUNK = 1_000_000
    elif free_mem >= 4_294_967_296:  # >= 4GB
        _LIFESPAN_CHUNK = 500_000
    elif free_mem >= 2_147_483_648:  # >= 2GB
        _LIFESPAN_CHUNK = 250_000
    else:
        _LIFESPAN_CHUNK = 100_000  # Conservative for low-memory GPUs

    _mem_pool = mempool
    _stream = _cp.cuda.Stream()
except Exception:
    _cp = None
    _HAS_GPU = False
    _mem_pool = None
    _stream = None
    _FUSED_BERN_KERNEL = None    # fused Bernoulli kernel unavailable on CPU
    _LIFESPAN_CHUNK = 250_000

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SEX_TO_IDX = {"M": 0, "F": 1, "U": 2}
_SEX_CODES = ("M", "F", "U")  # reverse lookup for the hot loop
_USE_FLOAT32 = False  # float64 Monte Carlo draws and risk arrays: float32
# uniforms from ``Generator.random(dtype=float32)`` have a granularity of
# 2^-24 (~6e-8), which rounds per-exam hit probabilities of order ~1e-6
# upward (measured on the synthetic cohort: +0.05 % expected hits overall,
# up to +1.2 % for radiography); float64 removes that bias.
# Sized so an 8GB+ GPU skips chunking entirely.
_ITERATION_CHUNK = _LIFESPAN_CHUNK * 3 if _HAS_GPU else 250_000

# REM (remainder) cancer subtype split (ICRP 152 Table 4.4 / "other solid").
# Constant across iterations; hoisted to module level so the per-iteration
# loop does not rebuild the tuple/dict every MC year.
_REM_SUBTYPES = ("Brain", "Kidney", "Pancreas", "other")
_rem_subtype_to_idx = {s: i for i, s in enumerate(_REM_SUBTYPES)}


# ---------------------------------------------------------------------------
# Persistent CuPy RNG registry
# ---------------------------------------------------------------------------
# Reuse one Generator per (namespace, kind, device, slot) across iterations.
# Creating a fresh CuPy RNG twice per iteration was ~15% of wall time; reusing
# streams is still an independent, valid MC realisation.  Keyed by namespace
# so the main run and the refine batch keep independent generators.  Cleared
# at run start by invalidate_persistent_cp_rng().
_PERSISTENT_CP_RNG: dict = {}


def _get_persistent_cp_rng(kind: str, dev_id, slot_idx, seed, ns: str = MAIN_NS):
    """Return a reusable per-(namespace, kind, device, slot) CuPy Generator.

    Created once (seeded from ``seed`` on first use) then reused; callers just
    invoke ``.random(...)`` and the stream advances.  Concurrent in-flight
    iterations always own distinct ``slot_idx`` values, so no two threads ever
    share a Generator.  ``ns`` separates the main run's generators from the
    refine batch's so the two can overlap on the GPU without crosstalk.
    """
    key = (ns, kind, dev_id, int(slot_idx))
    g = _PERSISTENT_CP_RNG.get(key)
    if g is None:
        g = _cp.random.default_rng(seed)
        _PERSISTENT_CP_RNG[key] = g
    return g


def invalidate_persistent_cp_rng(ns=None) -> None:
    """Drop cached persistent CuPy generators (called at run start).

    ``ns=None`` clears every namespace; otherwise only keys whose namespace
    matches ``ns``.
    """
    if ns is None:
        _PERSISTENT_CP_RNG.clear()
    else:
        for key in [k for k in _PERSISTENT_CP_RNG if k[0] == ns]:
            del _PERSISTENT_CP_RNG[key]


def _to_host(arr, on_gpu: bool = True):
    """Transfer a GPU array to host NumPy; identity on CPU or when disabled."""
    if on_gpu and _cp is not None:
        return _cp.asnumpy(arr)
    return arr


def _upload_many(gc, xp, attrs: list, values: list):
    """Batch ``setattr(gc, attr, xp.asarray(value))`` for repetitive uploads."""
    for attr, value in zip(attrs, values):
        setattr(gc, attr, xp.asarray(value))


def _bincount_breakdown(idx, died_mask, n_buckets, names, has_deaths):
    """Build ``{name: {'developed': d, 'deaths': k}}`` from per-hit index arrays.
    Only emits keys with at least one hit so the dict matches the schema the
    rest of the pipeline expects."""
    dev = np.bincount(idx, minlength=n_buckets)
    die = (np.bincount(idx[died_mask], minlength=n_buckets)
           if has_deaths else np.zeros(n_buckets, dtype=np.int64))
    out: Dict[str, Dict[str, int]] = {}
    for i, nm in enumerate(names):
        d = int(dev[i]); k = int(die[i])
        if d or k:
            out[nm] = {"developed": d, "deaths": k}
    return out


# ---------------------------------------------------------------------------
# Columnar exam array builder
# ---------------------------------------------------------------------------
def build_exam_arrays(exam_data, rpg_ct_to_idx, on_progress=None):
    """Convert exam_data list-of-dicts or dict-of-lists to columnar NumPy arrays.

    Remainder-subsite doses -> ``rem_mean_dose``.
    Bone / Skin / leukaemia -> separate 1-D arrays.
    Everything else -> ``dose_solid`` (N x n_solid).
    """
    def _emit_1pct(cur, total, last_pct, detail):
        if total <= 0 or on_progress is None:
            return last_pct
        pct = min(100, int(cur * 100 / total))
        if pct != last_pct:
            on_progress("init", cur, total, detail)
            return pct
        return last_pct

    is_columnar = isinstance(exam_data, dict)
    if is_columnar:
        exam_cols = exam_data
        N = len(exam_cols.get("modality", []))
    else:
        N = len(exam_data)
        
    n_solid = len(rpg_ct_to_idx)
    total_mult = _TOTAL_REMAINDER_MULT

    # Dose/exam storage stays float32 regardless of _USE_FLOAT32: F-2 widens
    # only the Monte Carlo draw and risk arrays (fdtype_risk), so dose-derived
    # totals are bit-identical to the float32 baseline.
    fdtype = np.float32
    dose_solid    = np.zeros((N, n_solid), dtype=fdtype)
    dose_leuk     = np.zeros(N, dtype=fdtype)
    dose_bone     = np.zeros(N, dtype=fdtype)
    dose_skin     = np.zeros(N, dtype=fdtype)
    dose_gonad    = np.zeros(N, dtype=fdtype)
    rem_mean_dose = np.zeros(N, dtype=fdtype)
    age_int       = np.empty(N, dtype=np.int32)
    sex_idx       = np.empty(N, dtype=np.int32)
    ed_sv         = np.empty(N, dtype=fdtype)
    is_nondose    = np.empty(N, dtype=bool)
    prior_dose_sv = np.empty(N, dtype=fdtype)
    frailty       = np.empty(N, dtype=fdtype)
    # True for exams whose patient age is > 94 (ICRP 152 REIC integration
    # ceiling). Used downstream to zero out their surv_correction.
    over_age_limit = np.zeros(N, dtype=bool)
    mod_idx_arr   = np.empty(N, dtype=np.int32)
    reg_idx_arr   = np.empty(N, dtype=np.int32)
    device_idx_arr = np.empty(N, dtype=np.int32)
    year_idx_arr   = np.empty(N, dtype=np.int32)
    pop_idx       = np.empty(N, dtype=np.int32)
    pid_int       = np.empty(N, dtype=np.int32)

    # Precompute the per-cancer-type column indices and the Remainder
    # multiplicity list once so the hot loop uses ints instead of dict lookups.
    _solid_keys: list = []           # [(name, col_idx), ...]
    for _ct, _idx in rpg_ct_to_idx.items():
        if _ct not in OTHER_SOLID_REMAINDER_SITES:
            _solid_keys.append((_ct, int(_idx)))
    _rem_keys_with_mult = [
        (ct, float(REMAINDER_ORGAN_MULTIPLICITY.get(ct, 1)))
        for ct in sorted(OTHER_SOLID_REMAINDER_SITES)
    ]
    _total_mult_f = float(total_mult)

    # Local aliases for the hot loop.
    _max_age = MAX_ATTAINED_AGE_ICRP152
    _SEX = _SEX_TO_IDX
    _np_int = np.integer

    # Phase 1: scalar field extraction via list comprehensions or dictionary lookups
    if is_columnar:
        _mods_raw  = [str(m) if m is not None else "Unknown" for m in exam_cols.get("modality", [])]
        _regs_raw  = [str(r) if r is not None else "Unknown" for r in exam_cols.get("region", [])]
        _pops_raw  = exam_cols.get("_population", [None] * N)
        _ages_raw  = [a if a is not None else 0 for a in exam_cols.get("age", [])]
        _sex_raw   = [s if s is not None else "U" for s in exam_cols.get("sex_code", [])]
        _ed_raw    = [e if e is not None else 0.0 for e in exam_cols.get("ed_msv", [])]
        _prior_raw = [p if p is not None else 0.0 for p in exam_cols.get("_prior_cumulative_dose_sv", [])]
        _frail_raw = [f if f is not None else 1.0 for f in exam_cols.get("_mataac_frailty_factor", [])]
        _gonad_raw = [g if g is not None else 0.0 for g in exam_cols.get("_gonad_dose_gy", [])]
        _pids_raw  = exam_cols.get("_person_id")
        if _pids_raw is None:
            _pids_raw = exam_cols.get("_patient_id", [None] * N)
        _cds_raw   = exam_cols.get("_cancer_doses", [None] * N)
        _dev_raw   = [str(d) if d else "" for d in exam_cols.get("device", [""] * N)]
        _eqp_raw   = [str(e) if e else "" for e in exam_cols.get("equipment", [""] * N)]
        _date_raw  = exam_cols.get("date_performed", [None] * N)
    else:
        _mods_raw  = [str(e.get("modality", "Unknown")) for e in exam_data]
        _regs_raw  = [str(e.get("region", "Unknown")) for e in exam_data]
        _pops_raw  = [e.get("_population") for e in exam_data]
        _ages_raw  = [e.get("age", 0) or 0 for e in exam_data]
        _sex_raw   = [e.get("sex_code", "U") for e in exam_data]
        _ed_raw    = [e.get("ed_msv", 0) or 0 for e in exam_data]
        _prior_raw = [e.get("_prior_cumulative_dose_sv", 0) or 0 for e in exam_data]
        _frail_raw = [e.get("_mataac_frailty_factor", 1.0) or 1.0 for e in exam_data]
        _gonad_raw = [e.get("_gonad_dose_gy", 0.0) for e in exam_data]
        _pids_raw  = [e.get("_person_id", e.get("_patient_id")) for e in exam_data]
        _cds_raw   = [e.get("_cancer_doses") or None for e in exam_data]
        _dev_raw   = [str(e.get("device", "") or "") for e in exam_data]
        _eqp_raw   = [str(e.get("equipment", "") or "") for e in exam_data]
        _date_raw  = [e.get("date_performed") or e.get("_exam_date") for e in exam_data]

    # Phase 2: vectorised numeric coercion + clipping.
    _age_f = np.array(_ages_raw, dtype=np.float64)
    age_int_tmp = _age_f.astype(np.int64)  # int64 catches overflow before clip
    np.maximum(age_int_tmp, 0, out=age_int_tmp)
    over_age_limit = age_int_tmp > _max_age
    np.minimum(age_int_tmp, _max_age, out=age_int_tmp)
    age_int[:] = age_int_tmp.astype(np.int32, copy=False)

    ed_sv[:] = np.array(_ed_raw, dtype=fdtype) * 0.001  # mSv -> Sv
    prior_dose_sv[:] = np.array(_prior_raw, dtype=fdtype)
    frailty[:] = np.array(_frail_raw, dtype=fdtype)
    dose_gonad[:] = np.array(_gonad_raw, dtype=fdtype)

    sex_idx[:] = np.array([_SEX.get(s, 2) for s in _sex_raw], dtype=np.int32)
    # Per-exam populations are resolved upstream (main._prepare_cohort), so
    # anything that is not a concrete label here is a bug -- never silently
    # map an unknown tag onto the Euro-American kernels.
    def _pop_idx_of(p):
        if p == "Euro-American":
            return 0
        if p == "Asian":
            return 1
        raise ValueError(
            f"Unrecognised _population {p!r}; expected Euro-American or Asian"
        )
    pop_idx[:] = np.array([_pop_idx_of(p) for p in _pops_raw], dtype=np.int32)
    is_nondose[:] = np.array([r.startswith("Non-Dose") for r in _regs_raw], dtype=bool)

    # Phase 3: intern modality / region / patient-id strings -> integer indices.
    mod_set: dict = {}
    reg_set: dict = {}
    for _m in _mods_raw:
        if _m not in mod_set:
            mod_set[_m] = len(mod_set)
    for _r in _regs_raw:
        if _r not in reg_set:
            reg_set[_r] = len(reg_set)
    mod_idx_arr[:] = np.array([mod_set[m] for m in _mods_raw], dtype=np.int32)
    reg_idx_arr[:] = np.array([reg_set[r] for r in _regs_raw], dtype=np.int32)

    # Intern device (equipment) and year strings into integer indices.
    # Use equipment as the device identifier (unique room/model).  Fall back
    # to device if equipment is empty.  Both empty → "Unknown".
    _dev_keys = []
    for _d, _e in zip(_dev_raw, _eqp_raw):
        _key = _e.strip() if _e else ""
        if not _key:
            _key = _d.strip() if _d else ""
        _dev_keys.append(_key if _key else "Unknown")
    device_set: dict = {}
    for _k in _dev_keys:
        if _k not in device_set:
            device_set[_k] = len(device_set)
    device_idx_arr[:] = np.array([device_set[k] for k in _dev_keys], dtype=np.int32)

    # Extract year from exam date.  Handles pandas.Timestamp, datetime,
    # numpy.datetime64, and strings.
    _year_keys = []
    for _d in _date_raw:
        _yr = 0
        if _d is not None:
            _yattr = getattr(_d, "year", None)
            if _yattr is not None:
                try:
                    _yr = int(_yattr)
                except (TypeError, ValueError):
                    pass
            if _yr == 0:
                _s = str(_d)[:4]
                if _s.isdigit():
                    _yr = int(_s)
        _year_keys.append(_yr if _yr > 0 else 0)
    year_set: dict = {}
    for _k in _year_keys:
        if _k not in year_set:
            year_set[_k] = len(year_set)
    year_idx_arr[:] = np.array([year_set[k] for k in _year_keys], dtype=np.int32)

    # Patient-id interning: missing / empty / NaN ids get a per-row
    # synthetic key so they don't accidentally cluster together. The NaN
    # check is critical -- without it every NaN-id row would collapse to
    # pid_idx=0 and form one giant super-patient.
    pid_set: dict = {}
    _last_pct = -1
    for _i, _pid_raw in enumerate(_pids_raw):
        _is_missing = (
            _pid_raw is None
            or _pid_raw == ""
            or (isinstance(_pid_raw, float) and _pid_raw != _pid_raw)  # NaN check
        )
        _pid_key = ("__row__", _i) if _is_missing else _pid_raw
        _pid_idx = pid_set.get(_pid_key)
        if _pid_idx is None:
            _pid_idx = len(pid_set)
            pid_set[_pid_key] = _pid_idx
        pid_int[_i] = _pid_idx
        _last_pct = _emit_1pct(_i + 1, N, _last_pct,
                               f"Building vectorized exam arrays: {_i + 1:,} / {N:,}")

    # Build unique cancer-dose objects once; upstream interning already collapsed most duplicates.
    cd_seen: dict = {}
    cd_unique_objs: list = []
    cd_idx_per_exam = np.full(N, -1, dtype=np.int32)
    _last_pct = -1
    for _i, _cd in enumerate(_cds_raw):
        if _cd:
            _cd_id = id(_cd)
            _u = cd_seen.get(_cd_id)
            if _u is None:
                _u = len(cd_unique_objs)
                cd_seen[_cd_id] = _u
                cd_unique_objs.append(_cd)
            cd_idx_per_exam[_i] = _u
        _last_pct = _emit_1pct(_i + 1, N, _last_pct,
                               f"Building vectorized exam arrays: {_i + 1:,} / {N:,}")
    # Risk-modelled exam: organ-dose model gives >=1 cancer-site dose
    # (cd_idx_per_exam >= 0) AND the region is not a Non-Dose placeholder.
    # Computed before the _perm permutation so it can be permuted with the
    # other arrays; ``scans``/``scans_dose`` keep their old meaning.
    is_modelled = (cd_idx_per_exam >= 0) & ~is_nondose
    # Compute the patient-id permutation early and build the columnar store directly in sorted order.
    if N > 1:
        _perm = np.argsort(pid_int, kind="stable")
        _already_sorted = bool(np.array_equal(_perm, np.arange(N)))
    else:
        _perm = np.arange(N, dtype=np.intp)
        _already_sorted = True

    # Build columnar dict of lists in sorted order.
    from .columnar_exams import _ALL_EXAM_FIELDS, ColumnarExamsList
    if is_columnar:
        cols = dict(exam_cols)
        if not _already_sorted:
            cols_sorted = {}
            for k, col in cols.items():
                if isinstance(col, np.ndarray):
                    cols_sorted[k] = col[_perm]
                else:
                    cols_sorted[k] = [col[i] for i in _perm]
            cols = cols_sorted
    else:
        keys = set(_ALL_EXAM_FIELDS)
        if N > 0:
            # Collect all possible keys in one pass so the fill loop can skip "if key in cols" checks.
            for ex in exam_data:
                keys.update(ex.keys())
        sorted_refs = exam_data if _already_sorted else [exam_data[i] for i in _perm]
        cols = {k: [None] * N for k in keys}
        _pid_col = cols.get("_patient_id")
        _age_col = cols.get("age")
        _last_pct = -1
        _progress_step = max(1, N // 100)
        for _si, ex in enumerate(sorted_refs):
            for k, v in ex.items():
                cols[k][_si] = v
            # Normalize missing / empty / NaN patient IDs and None ages in the
            # same pass so we don't loop over the data again.
            if _pid_col is not None:
                pid = _pid_col[_si]
                if pid is None or pid == "" or (isinstance(pid, float) and pid != pid):
                    _pid_col[_si] = ("__row__", _si)
            if _age_col is not None and _age_col[_si] is None:
                _age_col[_si] = 0
            if (_si + 1) % _progress_step == 0 or _si == N - 1:
                _last_pct = _emit_1pct(_si + 1, N, _last_pct,
                                       f"Building vectorized exam arrays: {_si + 1:,} / {N:,}")
    exam_data_sorted = ColumnarExamsList(cols, N)

    # Free the temporary list references — they may carry MB of overhead.
    del _mods_raw, _regs_raw, _pops_raw, _ages_raw, _sex_raw, _ed_raw
    del _prior_raw, _frail_raw, _gonad_raw, _pids_raw, _cds_raw, _age_f, age_int_tmp
    mod_list = [""] * len(mod_set)
    for m, idx in mod_set.items():
        mod_list[idx] = m
    reg_list = [""] * len(reg_set)
    for r, idx in reg_set.items():
        reg_list[idx] = r

    # Cancer-dose extraction: read each unique dict once into a float64
    # table, then scatter to all exams in one vectorised gather.
    n_unique_cd = len(cd_unique_objs)
    if n_unique_cd > 0:
        u_solid = np.zeros((n_unique_cd, n_solid), dtype=np.float64)
        u_leuk  = np.zeros(n_unique_cd, dtype=np.float64)
        u_bone  = np.zeros(n_unique_cd, dtype=np.float64)
        u_skin  = np.zeros(n_unique_cd, dtype=np.float64)
        u_rem   = np.zeros(n_unique_cd, dtype=np.float64)
        for _u, _cd in enumerate(cd_unique_objs):
            _v = _cd.get("Leukaemia")
            if _v is not None:
                u_leuk[_u] = _v
            _v = _cd.get("Bone")
            if _v is not None:
                u_bone[_u] = _v
            _v = _cd.get("Skin")
            if _v is not None:
                u_skin[_u] = _v
            for _ct, _ci in _solid_keys:
                _v = _cd.get(_ct)
                if _v is not None:
                    u_solid[_u, _ci] = _v
            _rs = 0.0
            for _ct, _wt in _rem_keys_with_mult:
                _v = _cd.get(_ct)
                if _v is not None:
                    _rs += _v * _wt
            if _rs:
                u_rem[_u] = _rs / _total_mult_f
        # Vectorised scatter — exam slots with cd_idx == -1 (no cancer-dose
        # dict) keep their pre-allocated zeros.
        _valid_mask = cd_idx_per_exam >= 0
        if _valid_mask.any():
            _vi = cd_idx_per_exam[_valid_mask]
            dose_solid[_valid_mask]    = u_solid[_vi].astype(fdtype, copy=False)
            dose_leuk[_valid_mask]     = u_leuk[_vi].astype(fdtype, copy=False)
            dose_bone[_valid_mask]     = u_bone[_vi].astype(fdtype, copy=False)
            dose_skin[_valid_mask]     = u_skin[_vi].astype(fdtype, copy=False)
            rem_mean_dose[_valid_mask] = u_rem[_vi].astype(fdtype, copy=False)

    # Permute the numeric arrays into patient-id order.
    if not _already_sorted:
        dose_solid    = dose_solid[_perm]
        dose_leuk     = dose_leuk[_perm]
        dose_bone     = dose_bone[_perm]
        dose_skin     = dose_skin[_perm]
        dose_gonad    = dose_gonad[_perm]
        rem_mean_dose = rem_mean_dose[_perm]
        age_int       = age_int[_perm]
        sex_idx       = sex_idx[_perm]
        ed_sv         = ed_sv[_perm]
        is_nondose    = is_nondose[_perm]
        is_modelled   = is_modelled[_perm]
        prior_dose_sv = prior_dose_sv[_perm]
        frailty       = frailty[_perm]
        over_age_limit = over_age_limit[_perm]
        mod_idx_arr   = mod_idx_arr[_perm]
        reg_idx_arr   = reg_idx_arr[_perm]
        device_idx_arr = device_idx_arr[_perm]
        year_idx_arr   = year_idx_arr[_perm]
        pop_idx       = pop_idx[_perm]

    del cols
    
    # CRITICAL: Compute patient changes BEFORE deleting pid_int
    # Data is now sorted by patient ID, so consecutive pid changes = patient boundaries
    pid_sorted = pid_int[_perm] if not _already_sorted else pid_int
    _patient_change_base   = np.empty(N, dtype=bool)
    _patient_change_base[0] = True
    _patient_change_base[1:] = pid_sorted[1:] != pid_sorted[:-1]
    _where_patient_change  = np.where(_patient_change_base)[0]
    _mataac_patient_ids    = (np.cumsum(_patient_change_base.astype(np.int32)) - 1).astype(np.int32)
    
    del pid_int, pid_set, pid_sorted

    # -- Iteration-invariant precomputations (built once, reused every iter) --
    _n_reg_local   = len(reg_list)
    _n_mod_local   = len(mod_list)
    _n_combined    = _n_mod_local * _n_reg_local
    _combined_idx  = mod_idx_arr * _n_reg_local + reg_idx_arr
    _ed_sv_masked  = np.where(is_nondose, 0.0, ed_sv)
    _dose_mask_bool = ~is_nondose
    _scans_per_bin = np.bincount(_combined_idx, minlength=_n_combined).reshape(_n_mod_local, _n_reg_local)
    # Dose-bearing exam count per bin: "Non-Dose*" placeholder rows carry no
    # dose and no risk model, so they must not dilute any per-exam mean.
    # ``_scans_per_bin`` stays the full exam volume (what "Scans / year"
    # reports); this is the denominator for per-exam dose / detriment.
    _scans_dose_per_bin = np.bincount(
        _combined_idx[_dose_mask_bool], minlength=_n_combined
    ).reshape(_n_mod_local, _n_reg_local)
    _dose_per_bin  = np.bincount(_combined_idx, weights=_ed_sv_masked, minlength=_n_combined).reshape(_n_mod_local, _n_reg_local)
    # Risk-modelled exam counts and dose per (modality x region) bin. The
    # modelled mask is a stricter subset of _dose_mask_bool (it additionally
    # requires a non-empty _cancer_doses dict), so these are <= the dose
    # figures above. Used as the denominator/numerator for per-exam figures.
    _scans_model_per_bin = np.bincount(
        _combined_idx[is_modelled], minlength=_n_combined
    ).reshape(_n_mod_local, _n_reg_local)
    _dose_model_per_bin_2d = np.bincount(
        _combined_idx, weights=np.where(is_modelled, ed_sv, 0.0),
        minlength=_n_combined,
    ).reshape(_n_mod_local, _n_reg_local)
    # Heritable risk/detriment depend only on (dose_gonad, sex_idx, constants) — all invariant
    _heritable_det  = dose_gonad * _HD_BY_SEX_NP[sex_idx]
    _heritable_risk = dose_gonad * _HERITABLE_RISK_COEFF
    _heritable_det_total  = float(_heritable_det[_dose_mask_bool].sum())
    _heritable_risk_total = float(_heritable_risk[_dose_mask_bool].sum())
    _age_int_f64    = age_int.astype(np.float32)

    # Build device_list and year_list from their sets.
    device_list = [""] * len(device_set)
    for d, idx in device_set.items():
        device_list[idx] = d
    year_list = [0] * len(year_set)
    for y, idx in year_set.items():
        year_list[idx] = y

    # Combined indices for per-modality-device and per-modality-year bincounts.
    _n_dev = len(device_list)
    _n_yr  = len(year_list)
    _dev_combined_idx = mod_idx_arr * _n_dev + device_idx_arr
    _yr_combined_idx  = mod_idx_arr * _n_yr + year_idx_arr

    return {
        "dose_solid": dose_solid, "dose_leuk": dose_leuk,
        "dose_bone": dose_bone, "dose_skin": dose_skin,
        "dose_gonad": dose_gonad,
        "rem_mean_dose": rem_mean_dose,
        "age_int": age_int, "sex_idx": sex_idx, "ed_sv": ed_sv,
        "is_nondose": is_nondose,
        "is_modelled": is_modelled,
        "over_age_limit": over_age_limit,
        "prior_dose_sv": prior_dose_sv,
        "frailty": frailty,
        "mod_idx": mod_idx_arr, "reg_idx": reg_idx_arr,
        "device_idx": device_idx_arr, "year_idx": year_idx_arr,
        "pop_idx": pop_idx,
        "mod_list": mod_list, "modality_list": mod_list,
        "reg_list": reg_list, "region_list": reg_list,
        "device_list": device_list,
        "year_list": year_list,
        "num_modalities": len(mod_list),
        "num_regions": len(reg_list),
        "num_devices": _n_dev,
        "num_years": _n_yr,
        "num_patients": len(_where_patient_change),
        "N": N,
        # Per-exam survival correction: Mataac frailty, zeroed for age > MAX_ATTAINED_AGE_ICRP152.
        # frailty is 1.0 whenever Mataac is off (including under icrp152_strict), so under ICRP 152
        # reference assumptions this reduces to the age cap alone, which is itself ICRP-conformant.
        "surv_correction": (
            frailty * (~over_age_limit).astype(fdtype)
        ),
        "patient_change_base":   _patient_change_base,
        "where_patient_change":  _where_patient_change,
        "mataac_patient_ids":    _mataac_patient_ids,
        "ct_range": np.arange(len(rpg_ct_to_idx)),
        "all_type_names": list(rpg_ct_to_idx.keys()) + ["Remainder", "Leukaemia", "Bone", "Skin"],
        "dose_leuk_sq":      dose_leuk ** 2,
        "combined_idx":      _combined_idx,
        "n_combined":        _n_combined,
        "dev_combined_idx":  _dev_combined_idx,
        "yr_combined_idx":   _yr_combined_idx,
        "n_dev_combined":    _n_mod_local * _n_dev,
        "n_yr_combined":     _n_mod_local * _n_yr,
        "ed_sv_dose_masked": _ed_sv_masked,
        "scans_per_bin":     _scans_per_bin,
        "scans_dose_per_bin": _scans_dose_per_bin,
        "dose_per_bin_2d":   _dose_per_bin,
        "scans_model_per_bin": _scans_model_per_bin,
        "dose_model_per_bin_2d": _dose_model_per_bin_2d,
        "total_dose_sv":     float(_ed_sv_masked.sum()),
        "heritable_det":     _heritable_det,
        "heritable_risk":    _heritable_risk,
        "heritable_det_total":  _heritable_det_total,
        "heritable_risk_total": _heritable_risk_total,
        "age_int_f64":       _age_int_f64,
        "exam_perm":         _perm,
        "exam_data_sorted":  exam_data_sorted,
    }



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GPU-resident cache for static arrays (transferred once, reused every iteration)
# ---------------------------------------------------------------------------
# Caches are namespaced so independent device arrays / slot rings / RNGs
# can run concurrently on the GPU.  Within a
# namespace each CUDA device gets its own _GPUCache.
class _GPUCache:
    """GPU copies of iteration-invariant arrays plus per-slot scratch buffers.
    Parallel iterations claim a slot via ``iteration_idx % n_slots``.
    """
    __slots__ = ('rpg_s', 'rpg_l1', 'rpg_l2', 'bone_b', 'skin_b',
                 'd_solid', 'd_leuk', 'd_leuk_sq', 'd_bone', 'd_skin', 'd_rem',
                 'age_i', 'sex_i', 'pop_idx', 'is_nd', 'k_solid_v', 'det_solid_v',
                 'mod_idx', 'reg_idx',
                 'surv_correction',
                 'patient_change_base', 'ct_range',
                 'mataac_patient_ids',
                 'heritable_det',
                 'slots', 'n_slots', 'N_cached', 'n_solid_cached',
                 'ready', 'ea_token')
    def __init__(self):
        self.ready = False
        self.slots = None
        self.n_slots = 0
        self.N_cached = 0
        self.n_solid_cached = 0
        self.ea_token = None

import threading as _threading  # for per-device cache-init lock
import contextlib as _contextlib

_gpu_cache = _GPUCache()  # alias for the (MAIN_NS, 0) cache
# Per-(namespace, device) cache registry: each CUDA device gets its own
# static-data cache and slot ring (CuPy arrays are device-bound).
_gpu_caches: dict = {(MAIN_NS, 0): _gpu_cache}
# Per-(namespace, device) locks serialize first-touch allocation across
# worker threads.
_gpu_cache_locks: dict = {}
_gpu_caches_registry_lock = _threading.Lock()
_gpu_cache_ever_invalidated = False

# Per-namespace RLocks serialising whole GPU *runs* (a main simulation's
# iteration loop, a refine batch, a cache invalidation) that share one
# namespace's caches.  year_batch overlaps year i's digests -- including GPU
# rare-cell refinement -- with year i+1's simulation on another thread; the
# main and refine namespaces have separate locks so the two runs overlap
# instead of alternating their exam arrays on the same cache mid-run.
_GPU_RUN_LOCKS: dict = {}


def _gpu_run_lock(ns: str = MAIN_NS) -> "_threading.RLock":
    """Fetch (or lazily create) the per-namespace GPU run lock."""
    lk = _GPU_RUN_LOCKS.get(ns)
    if lk is not None:
        return lk
    with _gpu_caches_registry_lock:
        lk = _GPU_RUN_LOCKS.get(ns)
        if lk is None:
            lk = _threading.RLock()
            _GPU_RUN_LOCKS[ns] = lk
    return lk


@_contextlib.contextmanager
def gpu_run_section(use_gpu: bool, ns: str = MAIN_NS):
    """Hold the namespace's GPU run lock for the duration of a GPU run; no-op on CPU."""
    if not use_gpu:
        yield
        return
    with _gpu_run_lock(ns):
        yield


def _current_device_id() -> int:
    """Active CUDA device id, or 0 when CuPy isn't available."""
    if _cp is None:
        return 0
    try:
        return int(_cp.cuda.Device().id)
    except Exception:
        return 0


def _get_gpu_cache(ns: str = MAIN_NS) -> "_GPUCache":
    """GPU cache for the active CUDA device in namespace ``ns`` (lazily created)."""
    global _gpu_cache
    dev_id = _current_device_id()
    key = (ns, dev_id)
    gc = _gpu_caches.get(key)
    if gc is None:
        with _gpu_caches_registry_lock:
            gc = _gpu_caches.get(key)
            if gc is None:
                gc = _GPUCache()
                _gpu_caches[key] = gc
                _gpu_cache_locks[key] = _threading.Lock()
                if ns == MAIN_NS and dev_id == 0:
                    _gpu_cache = gc
    return gc


def _get_gpu_cache_lock(ns: str = MAIN_NS, dev_id: int = 0) -> "_threading.Lock":
    """Fetch (or lazily create) the per-(namespace, device) cache-init lock."""
    key = (ns, dev_id)
    lk = _gpu_cache_locks.get(key)
    if lk is not None:
        return lk
    with _gpu_caches_registry_lock:
        lk = _gpu_cache_locks.get(key)
        if lk is None:
            lk = _threading.Lock()
            _gpu_cache_locks[key] = lk
    return lk


def _gpu_device_count() -> int:
    """Return the number of usable CUDA devices, or 0 if GPU is unavailable."""
    if _cp is None or not _HAS_GPU:
        return 0
    try:
        return max(0, int(_cp.cuda.runtime.getDeviceCount()))
    except Exception:
        return 0


# CPU-side per-slot scratch-buffer cache. Mirrors the GPU slot ring;
# avoids per-iteration reallocation of ``all_risk`` / ``rand_mat`` /
# ``cancer_hits``. Keyed on (N, n_solid); invalidated each new simulation.
# Namespaced for independent CPU slot rings.
class _CPUCache:
    """CPU-side scratch-buffer ring (lightweight; only NumPy arrays).

    Holds per-slot ``all_risk``, ``rand_mat``, and ``cancer_hits`` buffers
    for the CPU path.  Populated lazily by ``_ensure_cpu_slots``.
    """
    __slots__ = ('slots', 'n_slots', 'N_cached', 'n_solid_cached')

    def __init__(self):
        self.slots = None
        self.n_slots = 0
        self.N_cached = 0
        self.n_solid_cached = 0


_cpu_caches: dict = {}  # keyed by namespace


def _make_slot_buffers(xp, N: int, n_all: int, fdtype, *, include_stream: bool = False):
    """Allocate one set of per-iteration scratch buffers.

    Returns a dict with ``all_risk``, ``rand_mat``, ``cancer_hits``,
    ``surv_correction_bc`` (broadcast view, filled later), and
    ``hit_exam_idx_host`` / ``hit_type_idx_host`` (reusable host buffers
    for GPU→CPU transfers).  When ``include_stream`` is True also adds
    ``stream`` and ``rng`` for GPU slots.
    """
    slot = {
        'all_risk':             xp.empty((N, n_all), dtype=fdtype),
        'rand_mat':             xp.empty((N, n_all), dtype=fdtype),
        'cancer_hits':          xp.empty((N, n_all), dtype=xp.bool_),
        'surv_correction_bc':   None,   # filled once after surv_correction is known
        'hit_exam_idx_host':    np.empty(N, dtype=np.int64),   # over-allocated; sliced on use
        'hit_type_idx_host':    np.empty(N, dtype=np.int64),
    }
    if include_stream:
        slot['stream'] = xp.cuda.Stream(non_blocking=True)
        slot['rng']    = xp.random.default_rng()
    return slot


def _ensure_cpu_slots(n_slots: int, N: int, n_solid: int, ns: str = MAIN_NS) -> None:
    """Allocate ``n_slots`` sets of per-iteration CPU scratch buffers.
    Auto-shrinks ``n_slots`` to fit inside a safe percentage of available system RAM.
    """
    n_all = n_solid + 4
    fdtype = np.float32 if _USE_FLOAT32 else np.float64

    # Calculate memory footprint per slot in bytes
    itemsize = 4 if fdtype == np.float32 else 8
    slot_bytes = 2 * (N * n_all) * itemsize + (N * n_all) * 1 + 2 * N * 8

    # Query available system RAM (fallback to 8 GB if psutil is missing or fails)
    try:
        import psutil
        available_ram = psutil.virtual_memory().available
    except Exception:
        available_ram = 8 * 1024 * 1024 * 1024

    # Allow at most 10% of available system RAM for all scratch slots combined (minimum 1 slot)
    max_scratch_ram = max(256 * 1024 * 1024, int(available_ram * 0.10))
    max_allowed_slots = max(1, int(max_scratch_ram // slot_bytes))
    effective_n_slots = max(1, min(n_slots, max_allowed_slots))

    cc = _cpu_caches.get(ns)
    if cc is None:
        cc = _CPUCache()
        _cpu_caches[ns] = cc
    if (cc.slots is not None and
            cc.n_slots >= effective_n_slots and
            len(cc.slots) >= effective_n_slots and
            cc.N_cached == N and
            cc.n_solid_cached == n_solid):
        return

    cc.slots = []
    cc.n_slots = 0
    for _ in range(effective_n_slots):
        cc.slots.append(_make_slot_buffers(np, N, n_all, fdtype))
    cc.n_slots = effective_n_slots
    cc.N_cached = N
    cc.n_solid_cached = n_solid


def _ensure_gpu_slots(n_slots: int, N: int, n_solid: int, ns: str = MAIN_NS) -> None:
    """Allocate ``n_slots`` GPU scratch-buffer slots on the current device.
    Reallocates if (N, n_solid) or slot count changed, and auto-shrinks
    n_slots to fit available VRAM.
    """
    if _cp is None:
        return
    gc = _get_gpu_cache(ns)
    dev_id = _current_device_id()
    with _get_gpu_cache_lock(ns, dev_id):
        _ensure_gpu_slots_locked(gc, n_slots, N, n_solid)


def _ensure_gpu_slots_locked(gc, n_slots: int, N: int, n_solid: int) -> None:
    """Body of ``_ensure_gpu_slots``; must hold the device lock."""
    xp = _cp
    if (gc.slots is not None and
            gc.n_slots >= n_slots and
            len(gc.slots) >= n_slots and
            gc.N_cached == N and
            gc.n_solid_cached == n_solid):
        return
    xp = _cp
    n_all = n_solid + 4
    # Per-slot footprint: 2 * N * n_all * itemsize (risk + rand_mat)
    # + N * n_all (bool hits).
    _itemsize = 4 if _USE_FLOAT32 else 8
    bytes_per_slot = (
        2 * N * n_all * _itemsize
        + N * n_all
    )
    # Cap n_slots so the total fits in the pool's free budget (with a 200 MB safety margin).
    if _mem_pool is not None:
        try:
            pool_limit = _mem_pool.get_limit()
            pool_used = _mem_pool.used_bytes()
            free_budget = pool_limit - pool_used - (200 * 1024 * 1024)
            if free_budget > 0:
                max_fit = max(1, int(free_budget // bytes_per_slot))
                if max_fit < n_slots:
                    n_slots = max_fit
        except Exception:
            pass
    # Free transient pool blocks before the big allocation to maximise contiguous space.
    try:
        _mem_pool.free_all_blocks()  # type: ignore[union-attr]
    except Exception:
        pass
    gc.slots = []
    gc.n_slots = 0
    for _ in range(n_slots):
        gc.slots.append(_make_slot_buffers(
            xp, N, n_all,
            xp.float32 if _USE_FLOAT32 else xp.float64,
            include_stream=True))
    gc.n_slots = n_slots
    gc.N_cached = N
    gc.n_solid_cached = n_solid

def plan_iteration_slots(n_slots_requested: int, N: int, n_solid: int,
                         use_gpu: bool, cache_ns: str = MAIN_NS):
    """Build the per-(device, slot) execution plan for concurrent MC iterations.

    Returns a list of ``(device_id, slot_idx)`` tuples (length ≥ 1).
    ``device_id`` is None on CPU runs. Pre-allocates all scratch buffers so
    worker threads never race on first-touch.

    CPU: allocates ``req`` scratch-buffer slots and returns the
    flattened list ``[(None, 0), (None, 1), ...]`` so the caller can
    use a ThreadPoolExecutor with as many workers as it requested.
    GPU: allocates a slot ring on every visible device and returns the
    flattened Cartesian product. Per-device slot count auto-shrinks to fit VRAM.
    """
    req = max(1, int(n_slots_requested))
    on_gpu = bool(use_gpu) and _HAS_GPU and _cp is not None
    if not on_gpu:
        _ensure_cpu_slots(req, N, n_solid, ns=cache_ns)
        return [(None, i) for i in range(req)]
    n_gpu = _gpu_device_count()
    if n_gpu <= 0:
        _ensure_cpu_slots(req, N, n_solid, ns=cache_ns)
        return [(None, i) for i in range(req)]
    plan: list = []
    cur = _cp.cuda.Device().id if _cp is not None else 0
    try:
        for dev_id in range(n_gpu):
            # Pin device so _ensure_gpu_slots allocates on the right one.
            _cp.cuda.Device(dev_id).use()
            _ensure_gpu_slots(req, N, n_solid, ns=cache_ns)
            gc = _get_gpu_cache(cache_ns)
            eff = int(gc.n_slots) if gc.slots is not None else 0
            eff = max(1, eff)
            for s in range(eff):
                plan.append((dev_id, s))
    finally:
        try:
            _cp.cuda.Device(cur).use()
        except Exception:
            pass
    if not plan:
        plan = [(0, 0)]
    return plan


def _gpu_cache_ea_token(ea) -> tuple:
    """Identity of the exam-array set a GPU cache was uploaded from."""
    return (id(ea), int(ea["N"]), id(ea["is_nondose"]))


def _ensure_gpu_cache(ea, rpg, gpc_fn, ns: str = MAIN_NS):
    """Upload static exam + kernel arrays to GPU once per device.

    Re-uploads when ``ea`` is not the exam-array set the cache was built
    from: main.run_simulation only invalidates the cache when it rebuilds
    the exam arrays, so a run that hits the preprocess disk cache would
    otherwise reuse the previous dataset's device arrays.
    """
    gc = _get_gpu_cache(ns)
    token = _gpu_cache_ea_token(ea)
    if gc.ready and gc.ea_token == token:
        return
    dev_id = _current_device_id()
    with _get_gpu_cache_lock(ns, dev_id):
        if gc.ready and gc.ea_token == token:
            return
        if gc.ready:
            # Stale: drop slot ring too so _ensure_gpu_slots re-sizes for N.
            gc.ready = False
            gc.slots = None
            gc.n_slots = 0
            gc.N_cached = 0
            gc.n_solid_cached = 0
        _ensure_gpu_cache_locked(gc, ea, rpg, gpc_fn)


def _ensure_gpu_cache_locked(gc, ea, rpg, gpc_fn):
    """Body of ``_ensure_gpu_cache``; must hold the device lock."""
    xp = _cp
    # Per-call stream on the active device (avoids cross-device queueing).
    local_stream = xp.cuda.Stream(non_blocking=True)
    with local_stream:
        rpg_solid, rpg_l1, rpg_l2, ct_list, _, _, bone_beta, skin_beta = rpg
        _upload_many(gc, xp, ["rpg_s", "rpg_l1", "rpg_l2", "bone_b", "skin_b"],
                     [rpg_solid, rpg_l1, rpg_l2, bone_beta, skin_beta])
        _default_pop_idx = np.zeros(ea["N"], dtype=np.int32)
        _upload_many(gc, xp, [
            "d_solid", "d_leuk", "d_bone", "d_skin", "d_rem", "age_i",
            "sex_i", "is_nd", "mod_idx", "reg_idx", "pop_idx",
        ], [
            ea["dose_solid"], ea["dose_leuk"], ea["dose_bone"], ea["dose_skin"],
            ea["rem_mean_dose"], ea["age_int"], ea["sex_idx"], ea["is_nondose"],
            ea["mod_idx"], ea["reg_idx"],
            ea.get("pop_idx", _default_pop_idx),
        ])
        gc.k_solid_v   = xp.asarray([gpc_fn(ct)[0] for ct in ct_list])
        gc.det_solid_v = xp.asarray([gpc_fn(ct)[3] for ct in ct_list])
        # Upload surv_correction in the same dtype the risk/rand buffers use.
        gc.surv_correction     = xp.asarray(
            ea["surv_correction"],
            dtype=xp.float32 if _USE_FLOAT32 else xp.float64)
        gc.ct_range = xp.arange(len(ct_list))
        _upload_many(gc, xp, [
            "patient_change_base", "mataac_patient_ids",
            "d_leuk_sq", "heritable_det",
        ], [
            ea["patient_change_base"], ea["mataac_patient_ids"],
            ea["dose_leuk_sq"], ea["heritable_det"],
        ])
    local_stream.synchronize()
    gc.ea_token = _gpu_cache_ea_token(ea)
    gc.ready = True

def _reset_namespace_caches(ns: str) -> None:
    """Drop one namespace's GPU/CPU caches (caller holds the namespace run lock)."""
    global _gpu_cache, _cpu_caches, _CPU_SOLID_VECS, _gpu_caches
    # The registry lock keeps this mutation from racing another
    # namespace's lazy _get_gpu_cache insertion.
    with _gpu_caches_registry_lock:
        # Drop GPU caches for this namespace across all devices.
        for key in [k for k in _gpu_caches if k[0] == ns]:
            del _gpu_caches[key]
        _cpu_caches.pop(ns, None)
        _CPU_SOLID_VECS.pop(ns, None)
        # Rebuild the (MAIN_NS, 0) alias entry whenever MAIN_NS is reset.
        if ns == MAIN_NS:
            _gpu_cache = _GPUCache()
            _gpu_caches[(MAIN_NS, 0)] = _gpu_cache


def _gpu_cache_reclaim_tail() -> None:
    """gc + free_all_blocks; skipped on the very first (cold-start) call.

    ``free_all_blocks`` only releases unused blocks, so it is safe to run
    while another namespace has kernels in flight.
    """
    global _gpu_cache_ever_invalidated
    if not _gpu_cache_ever_invalidated:
        _gpu_cache_ever_invalidated = True
        return  # cold start: nothing to reclaim
    # Drop orphaned CuPy ndarrays so free_all_blocks() can reclaim them.
    try:
        import gc as _gc
        _gc.collect()
    except Exception:
        pass
    if _mem_pool is not None:
        # Free blocks on every visible CUDA device.
        try:
            cur = _cp.cuda.Device().id if _cp is not None else 0
        except Exception:
            cur = 0
        try:
            n_dev = _gpu_device_count() if _cp is not None else 0
            for _d in range(max(1, n_dev)):
                try:
                    if _cp is not None and n_dev > 0:
                        _cp.cuda.Device(_d).use()
                    _mem_pool.free_all_blocks()
                except Exception:
                    pass
        finally:
            try:
                if _cp is not None:
                    _cp.cuda.Device(cur).use()
            except Exception:
                pass


def invalidate_gpu_cache(ns=None):
    """Reset GPU / CPU per-slot caches so the next simulation starts fresh.

    ``ns=None`` (default) resets every namespace: each namespace's run lock
    is acquired in sorted order, then the gc / ``free_all_blocks`` reclaim
    pass runs once.  ``ns="x"`` resets only that namespace's caches under its
    own run lock (so it neither waits for nor wipes another namespace), then
    runs the same reclaim tail.  Aggressively returns VRAM to the pool on
    subsequent calls; the very first call (cold start) skips the GC pass to
    save ~200 ms.
    """
    global _gpu_cache, _CPU_SOLID_VECS, _cpu_caches, _gpu_caches
    global _gpu_cache_ever_invalidated
    if ns is None:
        # Reset every namespace. Acquire each namespace's run lock in
        # sorted order over the union of currently-present namespaces.
        ns_set = {MAIN_NS}
        with _gpu_caches_registry_lock:
            ns_set.update(k[0] for k in _gpu_caches)
            ns_set.update(_cpu_caches)
            ns_set.update(_CPU_SOLID_VECS)
        for n in sorted(ns_set):
            with _gpu_run_lock(n):
                _reset_namespace_caches(n)
        _gpu_cache_reclaim_tail()
    else:
        with _gpu_run_lock(ns):
            _reset_namespace_caches(ns)
        _gpu_cache_reclaim_tail()


# ---------------------------------------------------------------------------
# Precompute iteration-invariant risk arrays (once per run)
# ---------------------------------------------------------------------------
def _get_backend_arrays(ea, rpg, get_cancer_params_cached_fn, use_gpu: bool, ns: str = MAIN_NS) -> Dict[str, Any]:
    """Return backend-resident arrays common to precompute/invariant and iteration paths.

    Collapses the duplicated CPU-vs-GPU branch, array-pull and solid-type
    lethality/detriment vector caching from ``precompute_risk_invariants`` and
    ``vectorized_iteration`` into a single helper.  Callers unpack the returned
    dict into local variables.
    """
    (rpg_solid, rpg_l1, rpg_l2, ct_list, _,
     rem_idx, bone_beta, skin_beta) = rpg
    gpc = get_cancer_params_cached_fn
    two_pop = rpg_solid.ndim == 4
    on_gpu = use_gpu and _HAS_GPU and _cp is not None
    n_solid = len(ct_list)

    out: Dict[str, Any] = {
        "two_pop": two_pop,
        "on_gpu": on_gpu,
        "n_solid": n_solid,
        "ct_list": ct_list,
        "rem_idx": rem_idx,
    }
    if on_gpu:
        xp = _cp
        gc = _get_gpu_cache(ns)
        out.update({
            "xp": xp,
            "rpg_s": gc.rpg_s, "rpg_l1d": gc.rpg_l1, "rpg_l2d": gc.rpg_l2,
            "bone_b": gc.bone_b, "skin_b": gc.skin_b,
            "d_solid": gc.d_solid, "d_leuk": gc.d_leuk, "d_leuk_sq": gc.d_leuk_sq,
            "d_bone": gc.d_bone, "d_skin": gc.d_skin, "d_rem": gc.d_rem,
            "age_i": gc.age_i, "sex_i": gc.sex_i, "is_nd": gc.is_nd,
            "k_solid_v_g": gc.k_solid_v, "det_solid_v_g": gc.det_solid_v,
            "ct_range": gc.ct_range, "heritable_det": gc.heritable_det,
            "surv_correction": gc.surv_correction,
            "pop_i": gc.pop_idx if two_pop else None,
            "patient_change_base": gc.patient_change_base,
            "gc": gc,
        })
    else:
        xp = np
        _csv = _CPU_SOLID_VECS.get(ns)
        if _csv is None:
            _csv = {"k": None, "det": None, "ct_list": None}
            _CPU_SOLID_VECS[ns] = _csv
        if _csv["ct_list"] is not ct_list:
            _csv["k"] = np.asarray([gpc(ct)[0] for ct in ct_list])
            _csv["det"] = np.asarray([gpc(ct)[3] for ct in ct_list])
            _csv["ct_list"] = ct_list
        out.update({
            "xp": xp,
            "rpg_s": rpg_solid, "rpg_l1d": rpg_l1, "rpg_l2d": rpg_l2,
            "bone_b": bone_beta, "skin_b": skin_beta,
            "d_solid": ea["dose_solid"], "d_leuk": ea["dose_leuk"], "d_leuk_sq": ea["dose_leuk_sq"],
            "d_bone": ea["dose_bone"], "d_skin": ea["dose_skin"], "d_rem": ea["rem_mean_dose"],
            "age_i": ea["age_int"], "sex_i": ea["sex_idx"], "is_nd": ea["is_nondose"],
            "k_solid_v_g": _csv["k"], "det_solid_v_g": _csv["det"],
            "ct_range": ea["ct_range"], "heritable_det": ea["heritable_det"],
            "surv_correction": ea["surv_correction"],
            "pop_i": ea["pop_idx"] if two_pop else None,
            "patient_change_base": ea["patient_change_base"],
            "gc": None,
        })
    return out


def precompute_risk_invariants(
    ea, rpg, get_cancer_params_cached_fn,
    ddref_solid, ddref_leuk,
    use_gpu: bool = False,
    cache_ns: str = MAIN_NS,
):
    """Precompute the per-exam risk probability arrays that don't change
    between MC iterations and cache them on ``ea["_risk_inv"]``.

    The RPG attained-age axis is pre-collapsed (run_simulation step 4a),
    so ``solid_risk / rem_risk / leuk_risk / bone_risk / skin_risk`` and
    their derived ``exp_deaths`` / ``detriment_arr`` depend only on
    (dose, sex, age, pop). Hoisting them here saves ~30-60% per-iteration
    work on CPU and ~20-40% on GPU with deterministic output.

    Parameters
    ----------
    ea : dict
        Columnar arrays from ``build_exam_arrays``.
    rpg : tuple
        From ``get_vectorized_rpg()`` or ``get_vectorized_rpg_2pop()``.
    get_cancer_params_cached_fn : callable
        ``_get_cancer_params_cached`` from the main module.
    ddref_solid, ddref_leuk : float
        Dose-rate effectiveness factors.
    use_gpu : bool
        If True and CuPy is available, precompute on GPU.

    Returns
    -------
    dict
        The cache dict (same reference as ``ea["_risk_inv"]``).
    """
    N = ea["N"]
    gpc = get_cancer_params_cached_fn
    on_gpu = use_gpu and _HAS_GPU and _cp is not None
    if on_gpu:
        _ensure_gpu_cache(ea, rpg, gpc, ns=cache_ns)

    _b = _get_backend_arrays(ea, rpg, gpc, use_gpu, ns=cache_ns)
    two_pop = _b["two_pop"]
    n_solid = _b["n_solid"]
    rem_idx = _b["rem_idx"]
    rpg_s = _b["rpg_s"]
    rpg_l1d = _b["rpg_l1d"]
    rpg_l2d = _b["rpg_l2d"]
    bone_b = _b["bone_b"]
    skin_b = _b["skin_b"]
    d_solid = _b["d_solid"]
    d_leuk = _b["d_leuk"]
    d_leuk_sq = _b["d_leuk_sq"]
    d_bone = _b["d_bone"]
    d_skin = _b["d_skin"]
    d_rem = _b["d_rem"]
    age_i = _b["age_i"]
    sex_i = _b["sex_i"]
    is_nd = _b["is_nd"]
    k_solid_v_g = _b["k_solid_v_g"]
    det_solid_v_g = _b["det_solid_v_g"]
    ct_range = _b["ct_range"]
    heritable_det = _b["heritable_det"]
    surv_correction = _b["surv_correction"]
    pop_i = _b["pop_i"]
    patient_change_base = _b["patient_change_base"]

    # RPG lookup by (pop, sex, age) per exam. The attained-age axis is
    # pre-collapsed to its value at max_a so no per-iteration dedup is needed.
    if two_pop:
        rpg_lookup = rpg_s[pop_i[None, :], ct_range[:, None],
                           sex_i[None, :], age_i[None, :]]    # (n_solid, N)
        rpg_rem_lk = rpg_s[pop_i, rem_idx, sex_i, age_i]       # (N,)
        l1_lk      = rpg_l1d[pop_i, sex_i, age_i]              # (N,)
        l2_lk      = rpg_l2d[pop_i, sex_i, age_i]              # (N,)
    else:
        rpg_lookup = rpg_s[ct_range[:, None], sex_i[None, :],
                           age_i[None, :]]
        rpg_rem_lk = rpg_s[rem_idx, sex_i, age_i]
        l1_lk      = rpg_l1d[sex_i, age_i]
        l2_lk      = rpg_l2d[sex_i, age_i]

    # Per-exam risk components (Bernoulli probabilities).
    solid_risk = d_solid * rpg_lookup.T / ddref_solid                   # (N, n_solid)
    rem_risk   = d_rem * rpg_rem_lk / ddref_solid                        # (N,)
    leuk_risk  = (d_leuk * l1_lk + d_leuk_sq * l2_lk) / ddref_leuk      # (N,)
    if two_pop:
        bone_risk = d_bone * bone_b[pop_i, sex_i]
        skin_risk = d_skin * skin_b[pop_i, sex_i]
    else:
        bone_risk = d_bone * bone_b[sex_i]
        skin_risk = d_skin * skin_b[sex_i]

    # Zero out Non-Dose exams.
    solid_risk[is_nd] = 0.0
    rem_risk[is_nd]   = 0.0
    leuk_risk[is_nd]  = 0.0
    bone_risk[is_nd]  = 0.0
    skin_risk[is_nd]  = 0.0

    # Expected deaths and detriment (with heritable + survival correction).

    k_rem,  _, _, det_rem_f  = gpc("Remainder")
    k_leuk, _, _, det_leuk_f = gpc("Leukaemia")
    k_bone, _, _, det_bone_f = gpc("Bone")
    k_skin, _, _, det_skin_f = gpc("Skin")

    exp_deaths = (
        (solid_risk * k_solid_v_g).sum(axis=1)
        + rem_risk * k_rem + leuk_risk * k_leuk
        + bone_risk * k_bone + skin_risk * k_skin
    )
    detriment_arr = (
        (solid_risk * det_solid_v_g).sum(axis=1)
        + rem_risk * det_rem_f + leuk_risk * det_leuk_f
        + bone_risk * det_bone_f + skin_risk * det_skin_f
    )
    detriment_arr = detriment_arr + heritable_det
    exp_deaths   *= surv_correction
    detriment_arr *= surv_correction

    # Chunk boundaries for large datasets (iteration-invariant grouping).
    use_chunking = N > _ITERATION_CHUNK
    chunk_boundaries = None
    if use_chunking:
        change_indices = np.where(_to_host(patient_change_base, on_gpu))[0]
        chunk_boundaries = [0]
        current_size = 0
        for i in range(1, len(change_indices)):
            chunk_size_cb = int(change_indices[i] - change_indices[i - 1])
            if current_size + chunk_size_cb > _ITERATION_CHUNK:
                chunk_boundaries.append(int(change_indices[i - 1]))
                current_size = 0
            current_size += chunk_size_cb
        chunk_boundaries.append(N)

    # Analytical aggregate scalars (iteration-invariant). Used as control
    # variates in place of per-iteration MC reductions.
    per_exam_total = solid_risk.sum(axis=1) + rem_risk + leuk_risk + bone_risk + skin_risk
    all_risk_sum  = float((per_exam_total * surv_correction).sum())
    exp_deaths_sum = float(exp_deaths.sum())
    detriment_sum  = float(detriment_arr.sum())

    # Per-cancer-type (k_lethality, q_min, l_rel) arrays. Indexed by
    # hit_type_idx in the event loop -- replaces per-hit gpc() dict lookups.
    all_type_names = ea["all_type_names"]
    n_all_types = len(all_type_names)
    k_leth_per_type = np.zeros(n_all_types, dtype=np.float64)
    q_min_per_type  = np.zeros(n_all_types, dtype=np.float64)
    l_rel_per_type  = np.zeros(n_all_types, dtype=np.float64)
    for _t_idx, _site_name in enumerate(all_type_names):
        _k, _qm, _lr, _det = gpc(_site_name)
        k_leth_per_type[_t_idx] = _k
        q_min_per_type[_t_idx]  = _qm
        l_rel_per_type[_t_idx]  = _lr

    # Iteration-invariant aggregations (host numpy, deterministic)
    # These bincount reductions over combined_idx / dev_combined_idx /
    # yr_combined_idx depend only on (dose, sex, age, pop) and the index
    # arrays, never on the per-iteration random draws, so hoisting them
    # here makes them identical run-to-run on every backend.  They are
    # computed on the host with numpy because cupy.bincount uses float
    # atomics whose summation order, and so rounding, varies between runs.
    # Formulas mirror the CPU branch of
    # vectorized_iteration exactly so the CPU result stays bit-identical.
    _ed_host  = _to_host(exp_deaths, on_gpu)
    _det_host = _to_host(detriment_arr, on_gpu)
    _cidx_np   = ea["combined_idx"]
    _is_nd_np  = ea["is_nondose"]
    _surv_np   = ea["surv_correction"]
    _hd_np     = ea["heritable_det"]
    _dg_np     = ea["dose_gonad"]
    _ed_sv_masked_np = ea["ed_sv_dose_masked"]
    n_combined = ea["n_combined"]
    n_mod = len(ea["mod_list"])
    n_reg = len(ea["reg_list"])

    _dose_mask_f_np = (~_is_nd_np).astype(np.float64)
    # Risk-modelled mask: stricter subset of the dose mask (additionally
    # requires a non-empty _cancer_doses dict). Old snapshots reassembled by
    # verify_6x.py predate is_modelled, so fall back to ~is_nondose there.
    _is_modelled_np = ea.get("is_modelled")
    if _is_modelled_np is None:
        _is_modelled_np = ~_is_nd_np
    _model_mask_f_np = _is_modelled_np.astype(np.float64)
    _det_dm_np = np.where(~_is_nd_np, _det_host, 0.0)
    ed_all  = np.bincount(_cidx_np, weights=_ed_host, minlength=n_combined).reshape(n_mod, n_reg)
    det_all = np.bincount(_cidx_np, weights=_det_dm_np, minlength=n_combined).reshape(n_mod, n_reg)
    _herit_det_per_ex  = _hd_np * _surv_np * _dose_mask_f_np
    _herit_risk_per_ex = _dg_np * _HERITABLE_RISK_COEFF * _surv_np * _dose_mask_f_np
    herit_det_per_mod  = np.bincount(
        _cidx_np, weights=_herit_det_per_ex, minlength=n_combined,
    ).reshape(n_mod, n_reg).sum(axis=1)
    herit_risk_per_mod = np.bincount(
        _cidx_np, weights=_herit_risk_per_ex, minlength=n_combined,
    ).reshape(n_mod, n_reg).sum(axis=1)
    heritable_det_scaled  = float(_herit_det_per_ex.sum())
    heritable_risk_scaled = float(_herit_risk_per_ex.sum())

    _risk_inv = {
        "solid_risk": solid_risk, "rem_risk": rem_risk,
        "leuk_risk": leuk_risk, "bone_risk": bone_risk, "skin_risk": skin_risk,
        "exp_deaths": exp_deaths, "detriment_arr": detriment_arr,
        "use_chunking": use_chunking, "chunk_boundaries": chunk_boundaries,
        "n_solid": n_solid,
        "all_risk_sum":   all_risk_sum,
        "exp_deaths_sum": exp_deaths_sum,
        "detriment_sum":  detriment_sum,
        "k_leth_per_type": k_leth_per_type,
        "q_min_per_type":  q_min_per_type,
        "l_rel_per_type":  l_rel_per_type,
        "ed_all": ed_all, "det_all": det_all,
        "herit_det_per_mod": herit_det_per_mod,
        "herit_risk_per_mod": herit_risk_per_mod,
        "heritable_det_scaled": heritable_det_scaled,
        "heritable_risk_scaled": heritable_risk_scaled,
    }

    # Per-device / per-year aggregations require the dev/yr index arrays that
    # build_exam_arrays produces but the verify_6x.py snapshot does not carry;
    # compute them only when the keys are present so that minimal ea dicts
    # (used by verify_6x.py and other direct callers) work.
    if "dev_combined_idx" in ea and "yr_combined_idx" in ea:
        _dev_cidx_np = ea["dev_combined_idx"]
        _yr_cidx_np  = ea["yr_combined_idx"]
        n_dev_combined = ea["n_dev_combined"]
        n_yr_combined  = ea["n_yr_combined"]
        n_dev = len(ea["device_list"])
        n_yr  = len(ea["year_list"])
        _risk_inv["dev_scans"]      = np.bincount(_dev_cidx_np, minlength=n_dev_combined).reshape(n_mod, n_dev)
        _risk_inv["dev_scans_dose"] = np.bincount(_dev_cidx_np, weights=_dose_mask_f_np, minlength=n_dev_combined).reshape(n_mod, n_dev)
        _risk_inv["dev_dose"]       = np.bincount(_dev_cidx_np, weights=_ed_sv_masked_np, minlength=n_dev_combined).reshape(n_mod, n_dev)
        _risk_inv["dev_ed"]         = np.bincount(_dev_cidx_np, weights=_ed_host, minlength=n_dev_combined).reshape(n_mod, n_dev)
        _risk_inv["dev_det"]        = np.bincount(_dev_cidx_np, weights=_det_dm_np, minlength=n_dev_combined).reshape(n_mod, n_dev)
        _risk_inv["dev_scans_model"] = np.bincount(_dev_cidx_np, weights=_model_mask_f_np, minlength=n_dev_combined).reshape(n_mod, n_dev)
        _risk_inv["dev_dose_model"]  = np.bincount(_dev_cidx_np, weights=np.where(_is_modelled_np, _ed_sv_masked_np, 0.0), minlength=n_dev_combined).reshape(n_mod, n_dev)
        _risk_inv["yr_scans"]      = np.bincount(_yr_cidx_np, minlength=n_yr_combined).reshape(n_mod, n_yr)
        _risk_inv["yr_scans_dose"] = np.bincount(_yr_cidx_np, weights=_dose_mask_f_np, minlength=n_yr_combined).reshape(n_mod, n_yr)
        _risk_inv["yr_dose"]       = np.bincount(_yr_cidx_np, weights=_ed_sv_masked_np, minlength=n_yr_combined).reshape(n_mod, n_yr)
        _risk_inv["yr_ed"]         = np.bincount(_yr_cidx_np, weights=_ed_host, minlength=n_yr_combined).reshape(n_mod, n_yr)
        _risk_inv["yr_det"]        = np.bincount(_yr_cidx_np, weights=_det_dm_np, minlength=n_yr_combined).reshape(n_mod, n_yr)
        _risk_inv["yr_scans_model"] = np.bincount(_yr_cidx_np, weights=_model_mask_f_np, minlength=n_yr_combined).reshape(n_mod, n_yr)
        _risk_inv["yr_dose_model"]  = np.bincount(_yr_cidx_np, weights=np.where(_is_modelled_np, _ed_sv_masked_np, 0.0), minlength=n_yr_combined).reshape(n_mod, n_yr)

    ea["_risk_inv"] = _risk_inv
    return ea["_risk_inv"]


def vectorized_iteration(
    ea, exam_data, rpg, min_latency_params,
    seed, iteration_idx,
    get_cancer_params_cached_fn,
    ddref_solid=None, ddref_leuk=None,
    population=None, use_gpu=False,
    n_slots: int = 1,
    slot_idx=None,
    cache_ns: str = MAIN_NS,
):
    """Run one MC iteration across all exams using batched array ops.

    Parameters
    ----------
    ea : dict
        Columnar arrays from ``build_exam_arrays``.
    exam_data : list[dict]
        Original exam dicts (used only for rare-event detail).
    rpg : tuple
        From ``get_vectorized_rpg()``.
    min_latency_params : dict
        Cancer latency parameters.
    seed : int
        RNG seed for this iteration.
    iteration_idx : int
        Iteration number (for event metadata).
    get_cancer_params_cached_fn : callable
        ``_get_cancer_params_cached(site)`` from the main module.
    ddref_solid, ddref_leuk : float
        Dose-rate effectiveness factors.
    population : str
        Population model name.
    use_gpu : bool
        If True and CuPy is available, run on GPU.
    n_slots : int
        Number of in-flight iterations; sizes the per-slot scratch-buffer
        ring. Must be >= the caller's ``max_workers``.
    slot_idx : int or None
        Explicit slot index for the multi-GPU path.

    Returns
    -------
    dict
        Partial-stats dict compatible with ``_merge_partial``.

    Notes
    -----
    REIC (ICRP 152 para 51) integrates cancer-free survival S(a|e) up to
    age 94 into the lifetime risk probability, so the Bernoulli draw
    already accounts for competing mortality.  No additional gating by
    individual lifespan is applied — that would double-apply survival
    and reduce counts by ~50%.

    Death / life-years-lost bookkeeping uses a SEPARATE, per-hit natural
    (competing-mortality) lifespan, sampled only for exams that actually
    register a cancer hit, conditioned on surviving cancer-free to that
    hit's own onset age (exposure age + latency).  This guarantees
    lifespan >= onset by construction, so a fatal hit can never show 0
    life-years-lost — a whole-cohort, exposure-age-rooted lifespan draw
    (independent of onset) could land before onset even for a hit the
    case-fatality draw marked as fatal.
    """
    if ddref_solid is None:
        ddref_solid = _SIM_CONFIG["ddref_solid"]
    if ddref_leuk is None:
        ddref_leuk = _SIM_CONFIG["ddref_leuk"]
    if population is None:
        population = _SIM_CONFIG["population"]
    N = ea["N"]
    gpc = get_cancer_params_cached_fn

    on_gpu = use_gpu and _HAS_GPU and _cp is not None

    # Ensure caches / slots, then pull the common backend-resident arrays.
    if on_gpu:
        _ensure_gpu_cache(ea, rpg, gpc, ns=cache_ns)
        _ensure_gpu_slots(max(1, n_slots), ea["N"], len(rpg[3]), ns=cache_ns)
        gc = _get_gpu_cache(cache_ns)
        _n_dev_slots = max(1, int(gc.n_slots))
        if slot_idx is not None:
            _s = int(slot_idx) % _n_dev_slots
        else:
            _s = int(iteration_idx) % _n_dev_slots
        slot = gc.slots[_s]
    else:
        _ensure_cpu_slots(max(1, n_slots), ea["N"], len(rpg[3]), ns=cache_ns)
        _n_cpu_slots = max(1, int(_cpu_caches[cache_ns].n_slots))
        if slot_idx is not None:
            _s_cpu = int(slot_idx) % _n_cpu_slots
        else:
            _s_cpu = int(iteration_idx) % _n_cpu_slots
        slot = _cpu_caches[cache_ns].slots[_s_cpu]

    _b = _get_backend_arrays(ea, rpg, gpc, use_gpu, ns=cache_ns)
    two_pop = _b["two_pop"]
    n_solid = _b["n_solid"]
    xp = _b["xp"]
    surv_correction = _b["surv_correction"]
    gc = _b["gc"]

    # Mataac frailty is a deterministic survival correction applied via ``surv_correction``.
    use_chunking = N > _ITERATION_CHUNK

    # NOTE: a per-exam, exposure-age-rooted lifespan draw is not needed: it is only ever
    # consumed below for cancer hits, and drawing it from exposure age
    # (independent of the cancer's own onset age) could land BEFORE onset
    # even for a fatal hit, producing a "died" event with 0 life lost. See
    # the per-hit conditional lifespan draw further down, which samples the
    # natural lifespan starting FROM each hit's onset age instead, so
    # lifespan >= onset always holds by construction.

    # Iteration-invariant risk arrays. precompute_risk_invariants is normally
    # called once before the iteration loop; we populate lazily on cache miss.
    _inv_cached = ea.get("_risk_inv")
    if _inv_cached is None:
        _inv_cached = precompute_risk_invariants(
            ea, rpg, gpc, ddref_solid, ddref_leuk,
            use_gpu=use_gpu,
            cache_ns=cache_ns,
        )
    solid_risk       = _inv_cached["solid_risk"]
    rem_risk         = _inv_cached["rem_risk"]
    leuk_risk        = _inv_cached["leuk_risk"]
    bone_risk        = _inv_cached["bone_risk"]
    skin_risk        = _inv_cached["skin_risk"]
    use_chunking     = _inv_cached["use_chunking"]
    chunk_boundaries = _inv_cached["chunk_boundaries"]
    n_solid          = _inv_cached["n_solid"]
    # Mataac frailty multiplier (per-exam survival correction).
    surv_correction = gc.surv_correction if on_gpu else ea["surv_correction"]
    if slot['surv_correction_bc'] is None:
        slot['surv_correction_bc'] = surv_correction[:, None]

    # Bernoulli draw for stochastic cancer occurrence.
    n_all = n_solid + 4
    fdtype_risk = xp.float32 if _USE_FLOAT32 else xp.float64

    if use_chunking:
        # Pre-allocated cancer_hits buffer on the active device.
        cancer_hits = slot['cancer_hits']

        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1]
            chunk_size = end - start

            # Assemble all_risk for this chunk.
            all_risk_chunk = xp.empty((chunk_size, n_all), dtype=fdtype_risk)
            all_risk_chunk[:, :n_solid] = solid_risk[start:end]
            all_risk_chunk[:, n_solid] = rem_risk[start:end]
            all_risk_chunk[:, n_solid + 1] = leuk_risk[start:end]
            all_risk_chunk[:, n_solid + 2] = bone_risk[start:end]
            all_risk_chunk[:, n_solid + 3] = skin_risk[start:end]

            # Per-chunk RNG: (seed + i) gives independent streams.
            if on_gpu:
                chunk_seed = (seed + i) & 0xFFFF_FFFF
                _cp_rng_chunk = _cp.random.default_rng(chunk_seed)
                rand_mat_chunk = _cp_rng_chunk.random((chunk_size, n_all), dtype=fdtype_risk)
            else:
                chunk_rng = np.random.default_rng(seed + i)
                rand_mat_chunk = chunk_rng.random((chunk_size, n_all), dtype=fdtype_risk)

            # surv × clip × Bernoulli in one fused kernel on GPU,
            # otherwise the equivalent NumPy ops in sequence.
            if on_gpu and _FUSED_BERN_KERNEL is not None:
                _FUSED_BERN_KERNEL(
                    all_risk_chunk,
                    slot['surv_correction_bc'][start:end],
                    rand_mat_chunk,
                    all_risk_chunk,
                    cancer_hits[start:end],
                )
            else:
                all_risk_chunk *= surv_correction[start:end, None]
                xp.clip(all_risk_chunk, 0.0, 1.0, out=all_risk_chunk)
                cancer_hits[start:end] = rand_mat_chunk < all_risk_chunk

        expected_incidence = _inv_cached["all_risk_sum"]
    else:
        # Non-chunked: process all N exams in one pass using the persistent
        # slot buffers.
        all_risk   = slot['all_risk']
        rand_mat   = slot['rand_mat']
        cancer_hits = slot['cancer_hits']
        all_risk[:, :n_solid]    = solid_risk
        all_risk[:, n_solid]     = rem_risk
        all_risk[:, n_solid + 1] = leuk_risk
        all_risk[:, n_solid + 2] = bone_risk
        all_risk[:, n_solid + 3] = skin_risk

        if on_gpu:
            # Persistent per-slot generator: reuse the stream across iterations
            # instead of paying ~0.9 ms to rebuild XORWOW every MC year.
            _cp_rng_mc = _get_persistent_cp_rng('mc', _current_device_id(), _s, seed, ns=cache_ns)
            _cp_rng_mc.random(size=(N, n_all), dtype=fdtype_risk, out=rand_mat)
            # surv × clip × Bernoulli in one fused kernel on GPU.
            if _FUSED_BERN_KERNEL is not None:
                _FUSED_BERN_KERNEL(
                    all_risk, slot['surv_correction_bc'], rand_mat,
                    all_risk, cancer_hits,
                )
            else:
                # Fallback if the fused kernel did not compile.
                all_risk *= slot['surv_correction_bc']
                xp.clip(all_risk, 0.0, 1.0, out=all_risk)
                xp.less(rand_mat, all_risk, out=cancer_hits)
        else:
            all_risk *= slot['surv_correction_bc']
            np.clip(all_risk, 0.0, 1.0, out=all_risk)
            rng = np.random.default_rng(seed)
            rng.random(size=(N, n_all), dtype=fdtype_risk, out=rand_mat)
            np.less(rand_mat, all_risk, out=cancer_hits)

    # Transfer only the hit indices to CPU (avoids shipping the ~99.99%-zero
    # cancer_hits mask). xp.nonzero + GPU-side dedup keeps PCIe traffic low.
    if on_gpu:
        hit_exam_idx_g, hit_type_idx_g = xp.nonzero(cancer_hits)
        if hit_exam_idx_g.size > 1:
            _hit_pid_g   = gc.mataac_patient_ids[hit_exam_idx_g]
            _composite_g = _hit_pid_g * xp.int64(n_all) + hit_type_idx_g
            _, _first_idx_g = xp.unique(_composite_g, return_index=True)
            if _first_idx_g.size < hit_exam_idx_g.size:
                _first_idx_g.sort()
                hit_exam_idx_g = hit_exam_idx_g[_first_idx_g]
                hit_type_idx_g = hit_type_idx_g[_first_idx_g]
        # Transfer filtered indices to reusable host buffers (sized N; fall back to a fresh allocation if n_hits > N).
        n_hits = int(hit_exam_idx_g.size)
        _host_cap = slot['hit_exam_idx_host'].shape[0]
        if n_hits <= _host_cap:
            hit_exam_idx = slot['hit_exam_idx_host'][:n_hits]
            hit_type_idx = slot['hit_type_idx_host'][:n_hits]
            hit_exam_idx_g.get(out=hit_exam_idx)
            hit_type_idx_g.get(out=hit_type_idx)
        else:
            hit_exam_idx = hit_exam_idx_g.get()
            hit_type_idx = hit_type_idx_g.get()
    else:
        hit_exam_idx, hit_type_idx = np.nonzero(cancer_hits)
        # CPU-side dedup of multi-exam patients.
        if hit_exam_idx.size > 1:
            _mataac_pid_arr = ea["mataac_patient_ids"]
            _hit_pid   = _mataac_pid_arr[hit_exam_idx]
            _composite = _hit_pid * np.int64(n_all) + hit_type_idx
            _, _first_idx = np.unique(_composite, return_index=True)
            if _first_idx.size < hit_exam_idx.size:
                _first_idx.sort()
                hit_exam_idx = hit_exam_idx[_first_idx]
                hit_type_idx = hit_type_idx[_first_idx]

    all_type_names = ea["all_type_names"]

    # Rare-event (cancer hit) setup. cancer_hits is already False at
    # is_nondose rows, so no extra masking is required.
    ev_cancer_type: list = []
    ev_age: list = []
    ev_sex: list = []
    ev_modality: list = []
    ev_region: list = []
    ev_protocol: list = []
    ev_ed_msv: list = []
    ev_died: list = []
    ev_exam_index: list = []
    ev_age_at_death: list = []
    ev_projected_lifetime: list = []
    ev_QALY_lost: list = []
    ev_life_lost_years: list = []
    exam_cancer_count = np.zeros(N, dtype=np.int32)
    exam_death_count  = np.zeros(N, dtype=np.int32)
    total_QALY      = 0.0
    total_life_lost = 0.0
    cancer_bd: Dict[str, Dict[str, int]] = {}
    rem_subtype_bd: Dict[str, Dict[str, int]] = {}
    # Per-modality cancer breakdown (used by the GUI to split developed
    # counts by imaging modality).
    modality_cancer_bd: Dict[str, Dict[str, Dict[str, int]]] = {}

    # Group hits by exam index. hit_exam_idx is already sorted ascending
    # (np.nonzero returns row-major order), so np.unique gives the groups.
    if hit_exam_idx.size > 0:
        unique_exams, group_starts = np.unique(hit_exam_idx, return_index=True)
        # One shared RNG for all hits this iteration; bulk-draw all variates
        # in four calls instead of per-hit dispatch.
        hit_rng = np.random.default_rng(seed + 0x7F3C_A100)

        _n_hits_total = int(hit_exam_idx.size)
        rand_latency_inverse_cdf = hit_rng.random(size=_n_hits_total)
        rand_latency_jitter      = hit_rng.random(size=_n_hits_total)
        rand_death_draw          = hit_rng.random(size=_n_hits_total)
        rand_remainder_draw      = hit_rng.random(size=_n_hits_total)

        _hit_type_idx_np = np.asarray(hit_type_idx, dtype=np.int64)
        _hit_sites_list  = [all_type_names[int(t)] for t in _hit_type_idx_np]
        _hit_ages_np     = ea["age_int_f64"][hit_exam_idx]
        _hit_sex_idx_np  = ea["sex_idx"][hit_exam_idx]
        _hit_sexes_list  = [_SEX_CODES[int(s)] for s in _hit_sex_idx_np]
        _hit_sexes_arr   = np.asarray(_hit_sexes_list, dtype="U1")
        # Per-hit populations: from the exam's own tag in two-population
        # mode, otherwise the single bound run population (concrete label —
        # the mixed run-level label never reaches the engine's population-
        # dependent calls).
        if two_pop:
            _hit_pop_idx = ea["pop_idx"][hit_exam_idx]
            _hit_pops_arr = np.where(
                _hit_pop_idx == 1, "Asian", "Euro-American")
        else:
            _hit_pop_idx = None
            _hit_pops_arr = np.full(
                _n_hits_total, _require_population(population), dtype=object)

        # Bulk latency sampling (one pass instead of n_hits calls).
        latency_bulk = sample_cancer_latency_bulk(
            _hit_sites_list, _hit_ages_np, min_latency_params,
            rand_latency_inverse_cdf, rand_latency_jitter,
            sexes=_hit_sexes_list, populations=_hit_pops_arr,
        )

        # Per-type cancer parameters indexed by hit_type.
        lethality_bulk = _inv_cached["k_leth_per_type"][_hit_type_idx_np]
        q_min_bulk     = _inv_cached["q_min_per_type"][_hit_type_idx_np]
        l_rel_bulk     = _inv_cached["l_rel_per_type"][_hit_type_idx_np]

        # Bulk death-Bernoulli draw.  A fatal cancer is taken to end life at
        # its occurrence (no modelled post-diagnosis survival lag).
        died_bulk = rand_death_draw < lethality_bulk

        # Onset per hit (used by "full" QALY mode below and by the
        # conditional lifespan draw immediately following).
        onset_bulk = _hit_ages_np + latency_bulk

        # Per-hit conditional lifespan draw: sample this patient's natural
        # (competing-mortality) lifespan STARTING FROM the hit's own onset
        # age rather than from exposure age. sample_lifespans_vectorized
        # only searches for a death age >= the given start age, so this
        # guarantees lifespan_bulk >= onset_bulk by construction -- a fatal
        # hit can never end up with 0 life-years-lost (age_death capped at
        # a lifespan that is shorter than onset). Reuses hit_rng (no new
        # seed plumbing needed) and is drawn only for the hit subset,
        # avoiding a full-cohort per-exam lifespan draw.
        if two_pop:
            lifespan_bulk = np.empty(_n_hits_total, dtype=np.float64)
            for _pop_val, _pop_name in ((0, "Euro-American"), (1, "Asian")):
                _pmask = _hit_pop_idx == _pop_val
                if not _pmask.any():
                    continue
                lifespan_bulk[_pmask] = sample_lifespans_vectorized(
                    onset_bulk[_pmask], _hit_sexes_arr[_pmask],
                    population=_pop_name, rng=hit_rng,
                )
        else:
            lifespan_bulk = sample_lifespans_vectorized(
                onset_bulk, _hit_sexes_arr,
                population=population, rng=hit_rng,
            )

        # Severity component per hit (ICRP 152 §78): the non-fatal weight
        # used one tier down in ICRP's own nested detriment construction
        # (nominal risk -> severity adjustment -> a SECOND severity adjustment
        # that treats q_severity itself as the "non-fatal" weight against a
        # fatal weight of 1). Verified empirically against ICRP 152 Table 4.4
        # sensitivity ratios (k->1, q_min->0) for Lung/Colon/Breast, which only
        # match a nested (k + q_severity*(1-k)) structure, not the naive
        # q_severity alone. Used as-is only in "icrp152" mode below, where it
        # is applied to ICRP's flat population-average YLL rather than to any
        # sampled per-patient outcome.
        q_severity_bulk = lethality_bulk + q_min_bulk * (1.0 - lethality_bulk)

        # QALYs accounting
        # "icrp152" mode is the ICRP 152 §3.2.3 population-level construction:
        #   QALYs lost = q_severity · l_rel · 15 yr      (ICRP 152 page 26)
        # where l_rel is the §3.2.3 relative-YLL factor and 15 yr is the
        # all-cancer average YLL, so (l_rel · 15) = the site-specific average
        # YLL.  l_rel is REQUIRED here to shape that single average per site.
        # This mode never samples fatal/non-fatal for QALY purposes -- it
        # reproduces ICRP's own aggregate formula, so using the averaged
        # q_severity here is correct by construction, not an approximation.
        #
        # "full" mode (default) is the PER-PATIENT analogue: it swaps ICRP's
        # flat (l_rel * 15)-yr site-average YLL for each patient's actual
        # Years-of-Life-Lost -- the gap from the modeled cancer onset age to
        # that patient's own competing-mortality-adjusted lifespan.  Because
        # this mode already simulates each cancer down to an individual
        # fatal/non-fatal outcome (died_bulk, also used for the reported fatal
        # -cancer counts), the severity weight is applied INDIVIDUALLY too:
        # 1 for a simulated death, q_severity_bulk for a simulated survival
        # (the nested ICRP 152 detriment structure -- see below). Averaging
        # with q_severity here instead would double-average the same
        # stochastic outcome (once as a hard draw for the death counts, once
        # as its own expectation for the severity weight) while leaving the
        # two statistics decorrelated across iterations. The realised-outcome
        # weighting keeps a single cancer's fatality status and its severity
        # weight consistent, and preserves the same population-level mean
        # (E[q_individual] = k_lethal + q_severity*(1-k_lethal) by
        # construction, the nested ICRP detriment severity factor, not
        # q_severity alone).  It optionally
        # discounts the YLL stream exponentially (WHO-CHOICE / GBD 2019
        # convention; the 3% default recommended by Finnish HTA / NICE is
        # opt-in via sim_config["QALY_discount_rate"] = 0.03).  Full mode
        # deliberately DROPS l_rel: the site-specific onset-age timing that
        # l_rel encodes is already captured by the latency-driven onset and
        # the lifespan, so multiplying by l_rel would double-count it.
        _QALY_mode = str(_SIM_CONFIG["QALY_mode"]).lower()
        _QALY_r    = float(_SIM_CONFIG.get("QALY_discount_rate", 0.0))

        if _QALY_mode == "full":
            # Per-patient, age-specific Years of Life Lost: the gap from the
            # cancer onset age to the patient's OWN competing-mortality-adjusted
            # lifespan, computed for EVERY cancer (the per-age generalisation of
            # ICRP's flat l_rel*15-yr site average).
            #   t = 0        → exposure (discount reference)
            #   t = onset    → cancer occurrence
            #   t = lifespan → expected death age absent the cancer
            exposure_ages_bulk = _hit_ages_np
            # YLL = onset → lifespan, present-valued; _discount_interval clamps
            # to >= 0, so a cancer arising after the expected death loses nothing.
            years_of_life_lost = _discount_interval(onset_bulk - exposure_ages_bulk, lifespan_bulk - exposure_ages_bulk, _QALY_r)
            # Individual severity weight from the SAME Bernoulli draw already
            # used for the fatal-cancer-count statistics (died_bulk): 1 for a
            # simulated death, q_severity_bulk for a simulated non-fatal
            # cancer. Its expectation is k_lethal + q_severity*(1-k_lethal),
            # the nested ICRP 152 detriment severity factor (nominal risk ->
            # severity adjustment -> severity adjustment again), not the
            # single-tier q_severity alone -- see the note above.
            q_individual_bulk = np.where(died_bulk, 1.0, q_severity_bulk)
            QALY_event_bulk = q_individual_bulk * years_of_life_lost
        else:  # "icrp152" — ICRP 152 §3.2.3 site-average YLL (population-level, not per-patient)
            QALY_event_bulk = q_severity_bulk * l_rel_bulk * _ICRP_AVG_LIFE_LOST_YEARS
    else:
        # Empty fallbacks for the zero-hit case.
        unique_exams = np.empty(0, dtype=hit_exam_idx.dtype)
        group_starts = np.empty(0, dtype=np.int64)
        hit_rng = None
        lifespan_bulk    = np.empty(0, dtype=np.float64)
        latency_bulk     = np.empty(0, dtype=np.float64)
        lethality_bulk   = np.empty(0, dtype=np.float64)
        q_min_bulk       = np.empty(0, dtype=np.float64)
        l_rel_bulk       = np.empty(0, dtype=np.float64)
        died_bulk        = np.empty(0, dtype=bool)
        QALY_event_bulk  = np.empty(0, dtype=np.float64)
        rand_remainder_draw = np.empty(0, dtype=np.float64)

    # Columnar views (avoids exam.get() in the hot loop).
    ages_f64_np    = ea["age_int_f64"]
    sex_idx_np     = ea["sex_idx"]
    mod_idx_arr_np = ea["mod_idx"]
    reg_idx_arr_np = ea["reg_idx"]
    ed_sv_arr_np   = ea["ed_sv"]
    ea_mod_list    = ea["mod_list"]
    ea_reg_list    = ea["reg_list"]
    n_unique_hits  = len(unique_exams)

    # Reported-site mapping: cancer-type idx -> display site name idx.
    # Cached on ``ea`` since all_type_names is iteration-invariant.
    _reported_meta = ea.get("_reported_site_meta")
    if _reported_meta is None or _reported_meta[0] is not all_type_names:
        def _resolve_reported_site(_nm):
            if _nm == "Leukaemia":
                return "Leukaemia"
            if _nm == "Remainder":
                return "Other solid"
            return _nm.replace("_", " ")
        _all_reported_per_type = [_resolve_reported_site(n) for n in all_type_names]
        _unique_reported_sites = sorted(set(_all_reported_per_type))
        _reported_to_idx = {s: i for i, s in enumerate(_unique_reported_sites)}
        _type_to_reported_idx = np.array(
            [_reported_to_idx[_all_reported_per_type[t]] for t in range(len(all_type_names))],
            dtype=np.int32,
        )
        _reported_meta = (all_type_names, _unique_reported_sites, _type_to_reported_idx, _all_reported_per_type)
        ea["_reported_site_meta"] = _reported_meta
    _all_type_names_ref, unique_reported_sites, type_to_reported_idx, type_display_names = _reported_meta
    n_reported_sites = len(unique_reported_sites)

    # Per-hit index arrays for vectorised aggregation after the loop.
    if hit_exam_idx.size > 0:
        reported_idx_per_hit = type_to_reported_idx[_hit_type_idx_np]
        # Remainder hits set this inside the loop; non-Remainder stay at -1.
        rem_subtype_idx_per_hit = np.full(hit_exam_idx.size, -1, dtype=np.int32)
    else:
        reported_idx_per_hit = np.empty(0, dtype=np.int32)
        rem_subtype_idx_per_hit = np.empty(0, dtype=np.int32)

    for gi in range(n_unique_hits):
        e_idx = int(unique_exams[gi])
        _tstart = group_starts[gi]
        _tend   = group_starts[gi + 1] if gi + 1 < n_unique_hits else hit_exam_idx.size
        t_indices = hit_type_idx[_tstart:_tend]

        exam = exam_data[e_idx]
        age  = float(ages_f64_np[e_idx])
        sex  = _SEX_CODES[int(sex_idx_np[e_idx])]

        # Per-exam fields read once. mod/reg/ed_msv are columnar; only
        # protocol still needs an exam.get().
        exam_mod       = ea_mod_list[int(mod_idx_arr_np[e_idx])]
        exam_reg       = ea_reg_list[int(reg_idx_arr_np[e_idx])]
        exam_proto     = str(exam.get("protocol", ""))
        exam_ed_msv    = float(ed_sv_arr_np[e_idx]) * 1000.0

        # flat_idx = _tstart + local_i indexes into the bulk per-hit arrays.
        for local_i, t_idx in enumerate(t_indices):
            flat_idx = _tstart + local_i

            site = all_type_names[t_idx]

            # Resolve Remainder to a concrete subtype by inverse-CDF sampling
            # against the per-exam organ-dose distribution.
            if site == "Remainder":
                cdoses = exam.get("_cancer_doses", {})
                rw: dict = {}
                for rct in sorted(OTHER_SOLID_REMAINDER_SITES):
                    d = cdoses.get(rct, 0.0)
                    if d > 0:
                        w = d * REMAINDER_ORGAN_MULTIPLICITY.get(rct, 1)
                        rw[rct] = w
                # Add individual remainder organ doses if available
                indiv = exam.get("_remainder_individual_doses", {})
                if indiv:
                    rw.pop("Remainder", None)
                    for organ, dose in indiv.items():
                        if dose > 0:
                            rw[organ] = dose
                if rw:
                    names = list(rw.keys())
                    probs = np.fromiter(
                        (rw[nm] for nm in names), dtype=np.float64, count=len(names),
                    )
                    # Inverse-CDF sampling with the pre-drawn uniform.
                    _cdf_rem = np.cumsum(probs)
                    _tot_rem = float(_cdf_rem[-1])
                    if _tot_rem > 0.0:
                        _k_rem = int(np.searchsorted(
                            _cdf_rem, rand_remainder_draw[flat_idx] * _tot_rem, side="right",
                        ))
                        if _k_rem >= len(names):
                            _k_rem = len(names) - 1
                        chosen = names[_k_rem]
                    else:
                        chosen = names[0]
                    # All Remainder subtypes report as "Other solid" per ICRP 152/103.
                    reported_site = "Other solid"
                    rem_subtype = chosen if chosen in ("Brain", "Kidney", "Pancreas") else "other"
                else:
                    reported_site = "Other solid"
                    rem_subtype = "other"
                rem_subtype_idx_per_hit[flat_idx] = _rem_subtype_to_idx[rem_subtype]
            else:
                rem_subtype = None
                reported_site = type_display_names[t_idx]

            # Per-hit values from the pre-drawn bulk arrays.
            latency    = float(latency_bulk[flat_idx])
            onset      = age + latency
            died       = bool(died_bulk[flat_idx])
            ls         = float(lifespan_bulk[flat_idx])
            age_death  = min(onset, ls) if died else ls
            life_lost  = max(0.0, ls - age_death) if died else 0.0
            QALY_event = float(QALY_event_bulk[flat_idx])
            total_QALY += QALY_event
            total_life_lost += life_lost
            # Breakdown bookkeeping happens vectorised after the loop.

            # Columnar appends
            ev_cancer_type.append(reported_site)
            ev_age.append(age)
            ev_sex.append(sex)
            ev_modality.append(exam_mod)
            ev_region.append(exam_reg)
            ev_protocol.append(exam_proto)
            ev_ed_msv.append(exam_ed_msv)
            ev_died.append(died)
            ev_exam_index.append(e_idx)
            ev_age_at_death.append(age_death)
            ev_projected_lifetime.append(ls)
            ev_QALY_lost.append(QALY_event)
            ev_life_lost_years.append(life_lost)

    # Aggregate breakdowns vectorised (bincount over per-hit index arrays).
    if hit_exam_idx.size > 0:
        has_deaths = bool(died_bulk.any())

        # Per-exam cancer & death counts.
        np.add.at(exam_cancer_count, hit_exam_idx, 1)
        if has_deaths:
            np.add.at(exam_death_count, hit_exam_idx[died_bulk], 1)

        # Per-reported-site breakdown.
        cancer_bd.update(_bincount_breakdown(
            reported_idx_per_hit, died_bulk, n_reported_sites,
            unique_reported_sites, has_deaths,
        ))

        # Per-modality × per-site breakdown (2D bincount).
        _mod_idx_per_hit = mod_idx_arr_np[hit_exam_idx].astype(np.int64, copy=False)
        _mcb_flat = _mod_idx_per_hit * n_reported_sites + reported_idx_per_hit
        _n_mod_local = len(ea_mod_list)
        _mcb_dev = np.bincount(
            _mcb_flat, minlength=_n_mod_local * n_reported_sites,
        ).reshape(_n_mod_local, n_reported_sites)
        _mcb_die = (np.bincount(
                        _mcb_flat, weights=died_bulk.astype(np.int64),
                        minlength=_n_mod_local * n_reported_sites,
                    ).astype(np.int64).reshape(_n_mod_local, n_reported_sites)
                    if has_deaths else np.zeros_like(_mcb_dev))
        _nz_mod, _nz_site = np.nonzero(_mcb_dev | _mcb_die)
        for _m, _s in zip(_nz_mod.tolist(), _nz_site.tolist()):
            _md = modality_cancer_bd.setdefault(ea_mod_list[_m], {})
            _md[unique_reported_sites[_s]] = {
                "developed": int(_mcb_dev[_m, _s]),
                "deaths":    int(_mcb_die[_m, _s]),
            }

        # Remainder-subtype breakdown (only Remainder hits).
        _rem_mask = rem_subtype_idx_per_hit >= 0
        if _rem_mask.any():
            _rem_died = died_bulk[_rem_mask]
            rem_subtype_bd.update(_bincount_breakdown(
                rem_subtype_idx_per_hit[_rem_mask], _rem_died,
                len(_REM_SUBTYPES), _REM_SUBTYPES,
                bool(_rem_died.any()) if has_deaths else False,
            ))


    # Per-modality / per-region aggregation via bincount over combined_idx.
    mod_list = ea["mod_list"]
    reg_list = ea["reg_list"]
    n_mod = len(mod_list)
    n_reg = len(reg_list)
    modality_stats: Dict[str, dict] = {}

    n_combined      = ea["n_combined"]
    combined_idx_np = ea["combined_idx"]
    scans_all = ea["scans_per_bin"]
    scans_dose_all = ea["scans_dose_per_bin"]
    dose_all  = ea["dose_per_bin_2d"]
    # Risk-modelled exam counts / dose per (modality x region). Fall back to
    # the dose arrays when the snapshot has no ``is_modelled`` column.
    scans_model_all = ea.get("scans_model_per_bin", scans_dose_all)
    dose_model_all  = ea.get("dose_model_per_bin_2d", dose_all)

    # exp_deaths is pre-zeroed at non-dose rows; detriment_arr is not, so it
    # needs masking (heritable_det can be non-zero for non-dose exams).
    # ed_all / det_all / heritable aggregates are iteration-invariant and
    # were precomputed on host in precompute_risk_invariants (deterministic
    # on every backend); read them from the cache instead of recomputing.
    ed_all             = _inv_cached["ed_all"]
    det_all            = _inv_cached["det_all"]
    herit_det_per_mod  = _inv_cached["herit_det_per_mod"]
    herit_risk_per_mod = _inv_cached["herit_risk_per_mod"]

    # Cases and deaths: rare-event arrays are CPU-only.
    cases_w     = exam_cancer_count.astype(np.float64)
    cases_all   = np.bincount(combined_idx_np, weights=cases_w, minlength=n_combined).reshape(n_mod, n_reg)
    deaths_w    = exam_death_count.astype(np.float64)
    deaths_all  = np.bincount(combined_idx_np, weights=deaths_w, minlength=n_combined).reshape(n_mod, n_reg)

    for m_val, mod_name in enumerate(mod_list):
        ms: dict = {
            "scans": int(scans_all[m_val].sum()),
            "scans_dose": int(scans_dose_all[m_val].sum()),
            "scans_model": int(scans_model_all[m_val].sum()),
            "dose_sv": float(dose_all[m_val].sum()),
            "dose_sv_model": float(dose_model_all[m_val].sum()),
            "cases": int(cases_all[m_val].sum()),
            "deaths": int(deaths_all[m_val].sum()),
            "expected_deaths": float(ed_all[m_val].sum()),
            "detriment": float(det_all[m_val].sum()),
            "QALY": 0.0, "life_lost": 0.0,
            "regions": {},
            # Per-modality cancer breakdown and heritable totals (used by
            # the GUI's ICRP-152 Table 3.9 comparison split-by-modality view).
            "cancer_bd": modality_cancer_bd.get(mod_name, {}),
            "heritable_risk": float(herit_risk_per_mod[m_val]),
            "heritable_detriment": float(herit_det_per_mod[m_val]),
        }
        for r_val, reg_name in enumerate(reg_list):
            if scans_all[m_val, r_val] == 0:
                continue
            ms["regions"][reg_name] = {
                "scans": int(scans_all[m_val, r_val]),
                "scans_dose": int(scans_dose_all[m_val, r_val]),
                "scans_model": int(scans_model_all[m_val, r_val]),
                "dose_sv": float(dose_all[m_val, r_val]),
                "dose_sv_model": float(dose_model_all[m_val, r_val]),
                "cases": int(cases_all[m_val, r_val]),
                "deaths": int(deaths_all[m_val, r_val]),
                "expected_deaths": float(ed_all[m_val, r_val]),
            }
        # Merge orientation-specific radiography regions into combined
        # reporting regions (e.g. "Chest PA" + "Chest AP" + "Chest Lat"
        # → "Chest") via the module-level ``_reporting_reg`` helper.  Dental
        # subtypes (Panoramic/Kefalometry/Intraoral/Dental, Dental CBCT) are
        # left as-is here -- they keep their own name so per-subtype counts
        # survive into ``ms["regions"]``; grouping them into the
        # "Dental imaging" pseudo-modality is done by callers that need
        # that view (e.g. research_questions.py, gui.py's Summary tab). The
        # risk calculation above used the full region labels for correct
        # dose coefficients; this only affects the reporting/stats view.
        merged_regions: dict = {}
        for reg_name, rv in ms["regions"].items():
            rpt_reg = _reporting_reg(reg_name)
            if rpt_reg == reg_name:
                merged_regions[reg_name] = rv
            else:
                acc = merged_regions.setdefault(rpt_reg, {
                    "scans": 0, "scans_dose": 0, "scans_model": 0,
                    "dose_sv": 0.0, "dose_sv_model": 0.0, "cases": 0,
                    "deaths": 0, "expected_deaths": 0.0,
                })
                acc["scans"] += rv["scans"]
                acc["scans_dose"] += rv["scans_dose"]
                acc["scans_model"] += rv["scans_model"]
                acc["dose_sv"] += rv["dose_sv"]
                acc["dose_sv_model"] += rv["dose_sv_model"]
                acc["cases"] += rv["cases"]
                acc["deaths"] += rv["deaths"]
                acc["expected_deaths"] += rv["expected_deaths"]
        ms["regions"] = merged_regions
        modality_stats[mod_name] = ms

    # Per-modality-per-device and per-modality-per-year aggregation.
    device_list = ea["device_list"]
    year_list   = ea["year_list"]
    n_dev = len(device_list)
    n_yr  = len(year_list)
    dev_combined_idx = ea["dev_combined_idx"]
    yr_combined_idx  = ea["yr_combined_idx"]
    n_dev_combined   = ea["n_dev_combined"]
    n_yr_combined    = ea["n_yr_combined"]

    # Iteration-invariant device/year aggregates (scans / dose / ed / det)
    # were precomputed on host in precompute_risk_invariants; read them from
    # the cache.  Cases/deaths weights are CPU-only (rare-event arrays), so
    # those still use np.bincount with the CPU index arrays.
    dev_scans      = _inv_cached["dev_scans"]
    dev_scans_dose = _inv_cached["dev_scans_dose"]
    dev_dose       = _inv_cached["dev_dose"]
    dev_ed         = _inv_cached["dev_ed"]
    dev_det        = _inv_cached["dev_det"]
    dev_scans_model = _inv_cached.get("dev_scans_model", dev_scans_dose)
    dev_dose_model  = _inv_cached.get("dev_dose_model", dev_dose)
    dev_cases  = np.bincount(dev_combined_idx, weights=cases_w, minlength=n_dev_combined).reshape(n_mod, n_dev)
    dev_deaths = np.bincount(dev_combined_idx, weights=deaths_w, minlength=n_dev_combined).reshape(n_mod, n_dev)

    yr_scans      = _inv_cached["yr_scans"]
    yr_scans_dose = _inv_cached["yr_scans_dose"]
    yr_dose       = _inv_cached["yr_dose"]
    yr_ed         = _inv_cached["yr_ed"]
    yr_det        = _inv_cached["yr_det"]
    yr_scans_model = _inv_cached.get("yr_scans_model", yr_scans_dose)
    yr_dose_model  = _inv_cached.get("yr_dose_model", yr_dose)
    yr_cases  = np.bincount(yr_combined_idx, weights=cases_w, minlength=n_yr_combined).reshape(n_mod, n_yr)
    yr_deaths = np.bincount(yr_combined_idx, weights=deaths_w, minlength=n_yr_combined).reshape(n_mod, n_yr)

    device_stats: Dict[str, dict] = {}
    for m_val, mod_name in enumerate(mod_list):
        for d_val, dev_name in enumerate(device_list):
            if dev_scans[m_val, d_val] == 0:
                continue
            device_stats.setdefault(mod_name, {})[dev_name] = {
                "scans": int(dev_scans[m_val, d_val]),
                "scans_dose": int(round(float(dev_scans_dose[m_val, d_val]))),
                "scans_model": int(round(float(dev_scans_model[m_val, d_val]))),
                "dose_sv": float(dev_dose[m_val, d_val]),
                "dose_sv_model": float(dev_dose_model[m_val, d_val]),
                "cases": int(dev_cases[m_val, d_val]),
                "deaths": int(dev_deaths[m_val, d_val]),
                "expected_deaths": float(dev_ed[m_val, d_val]),
                "detriment": float(dev_det[m_val, d_val]),
            }

    year_stats: Dict[str, dict] = {}
    for m_val, mod_name in enumerate(mod_list):
        for y_val, yr in enumerate(year_list):
            if yr_scans[m_val, y_val] == 0:
                continue
            year_stats.setdefault(mod_name, {})[yr] = {
                "scans": int(yr_scans[m_val, y_val]),
                "scans_dose": int(round(float(yr_scans_dose[m_val, y_val]))),
                "scans_model": int(round(float(yr_scans_model[m_val, y_val]))),
                "dose_sv": float(yr_dose[m_val, y_val]),
                "dose_sv_model": float(yr_dose_model[m_val, y_val]),
                "cases": int(yr_cases[m_val, y_val]),
                "deaths": int(yr_deaths[m_val, y_val]),
                "expected_deaths": float(yr_ed[m_val, y_val]),
                "detriment": float(yr_det[m_val, y_val]),
            }

    # Deterministic expected incidence = sum of all per-exam Bernoulli probabilities.
    if not use_chunking:
        expected_incidence = _inv_cached["all_risk_sum"]

    # Heritable detriment / risk totals with surv_correction applied
    # (iteration-invariant; precomputed on host in precompute_risk_invariants).
    heritable_det_scaled  = _inv_cached["heritable_det_scaled"]
    heritable_risk_scaled = _inv_cached["heritable_risk_scaled"]

    # Convert collected columnar event lists to numpy arrays
    events_col = {
        "cancer_type": np.array(ev_cancer_type, dtype=object),
        "age_at_exposure": np.array(ev_age, dtype=np.float32),
        "sex": np.array(ev_sex, dtype=object),
        "modality": np.array(ev_modality, dtype=object),
        "region": np.array(ev_region, dtype=object),
        "protocol": np.array(ev_protocol, dtype=object),
        "ed_msv": np.array(ev_ed_msv, dtype=np.float32),
        "died": np.array(ev_died, dtype=bool),
        "exam_index": np.array(ev_exam_index, dtype=np.int64),
        "age_at_death": np.array(ev_age_at_death, dtype=np.float32),
        "projected_lifetime": np.array(ev_projected_lifetime, dtype=np.float32),
        "QALY_lost": np.array(ev_QALY_lost, dtype=np.float32),
        "life_lost_years": np.array(ev_life_lost_years, dtype=np.float32),
        "iteration": np.full(len(ev_cancer_type), int(iteration_idx), dtype=np.int32),
    }

    # Return MC aggregates plus analytical sums cached from iteration-invariant arrays.
    return {
        "dose_sv": ea["total_dose_sv"],
        "cases": int(exam_cancer_count.sum()),
        "deaths": int(exam_death_count.sum()),
        "cases_analytical":  _inv_cached["all_risk_sum"],
        "deaths_analytical": _inv_cached["exp_deaths_sum"],
        "expected_deaths":   _inv_cached["exp_deaths_sum"],
        "expected_incidence": expected_incidence,
        "detriment":         _inv_cached["detriment_sum"],
        "QALY": total_QALY,
        "life_lost": total_life_lost,
        "modality": modality_stats,
        "device_stats": device_stats,
        "year_stats": year_stats,
        "cancer_bd": cancer_bd,
        "rem_subtype_bd": rem_subtype_bd,
        "events": events_col,
        "heritable_risk": heritable_risk_scaled,
        "heritable_detriment": heritable_det_scaled,
    }
