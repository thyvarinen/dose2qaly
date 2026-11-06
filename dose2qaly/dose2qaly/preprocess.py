"""Per-exam preprocessing helpers used by ``main.run_simulation``.

This module owns the post-load / pre-iteration preprocessing pipeline
that turns raw exam-row dose dicts into the annotated structures the
vectorized iteration engine consumes.

Pipeline (called in this order from ``run_simulation``):

  1. ``load_and_preprocess``           -- per-file load + row dispatch
                                            via :func:`exam_doses._compute_exam_batch`.
  2. ``analyze_population_and_ct_doses`` -- single O(N) scan that resolves
                                            the population mix, per-file
                                            mix report, and CT dose
                                            coverage / missing-dose alerts.
  3. ``apply_golikov_and_cumulative_dose`` -- toggles the Golikov body-size
                                              correction off (when
                                              ``use_golikov=False``),
                                              annotates each exam with
                                              prior cumulative dose, and
                                              applies the Mataac frailty
                                              factor when enabled.

Public API
----------
* ``load_and_preprocess``             -- per-file load + exam-row dispatch.
* ``analyze_population_and_ct_doses``   -- population mix + missing-dose detection.
* ``apply_golikov_and_cumulative_dose`` -- Golikov toggle + cumulative dose.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from . import region_resolver as region_resolver
from .ct_doses import _EXCEL_CT_COLUMN_TO_ORGAN
from .risk_models import (
    mataac_survival_correction,
    get_organ_dose_for_cancer,
    normalize_sex_code,
    HERITABLE_GONAD_DOSE_KEY,
)
from .exam_doses import (
    _ColumnRowView,
    _compute_exam_batch,
    compute_exam_organ_doses,
    _build_cancer_dose_map,
    _parse_population,
    set_patient_id_salt,
    new_run_patient_id_salt,
)

if TYPE_CHECKING:
    import pandas as pd

# Canonical list of organ names expected from the CT dose-tracking system.
# Shown in the missing-CT-doses alert when a CT row has no organ doses.
# CBCT (dental or not) is never flagged by this alert -- see the loop below.
_CT_EXPECTED_ORGANS: List[str] = sorted(set(_EXCEL_CT_COLUMN_TO_ORGAN.values()))

# Minimum total-row estimate at which the loader spins up a shared
# ProcessPoolExecutor for the per-row dose computation.  Below this the spawn
# overhead dominates, so the calling thread runs ``_process_single_row`` inline.
_PARALLEL_MIN_ROWS = 10_000

# measured ~150 MB imports + batch slice + pickled results headroom
_WORKER_COMMIT_BYTES = 400 * 1024 ** 2


def _preprocess_pool_workers(total_rows: int) -> int:
    """Worker count for the per-row dose pool: CPUs, capped by rows (>=5k rows per
    worker) and by available RAM (~400 MB commit per spawned worker)."""
    n = max(1, mp.cpu_count() or 1)
    if total_rows > 0:
        n = min(n, max(1, total_rows // 5_000))
    try:
        import psutil
        n = min(n, max(1, int(psutil.virtual_memory().available * 0.5 // _WORKER_COMMIT_BYTES)))
    except Exception:
        pass
    return n


def _make_preprocess_pool(n_workers: int, salt) -> ProcessPoolExecutor:
    # Children inherit the environment: pin BLAS to one thread BEFORE they import
    # numpy (OpenBLAS otherwise allocates per-thread buffers, ~380 MB per worker
    # at 12 threads vs 8 MB at 1). Parent's numpy is already loaded -- unaffected.
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    return ProcessPoolExecutor(max_workers=n_workers,
                               initializer=set_patient_id_salt,
                               initargs=(salt,))


def _emit_1pct_progress(
    phase: str,
    cur: int,
    total: int,
    last_pct: int,
    detail: str,
    on_progress,
) -> int:
    """Emit on_progress only when the integer percentage changes.

    Returns the new ``last_pct`` so callers can keep a running counter.
    This keeps the GUI update rate bounded (at most 100 calls per phase)
    even for multi-million-row datasets.
    """
    if total <= 0 or on_progress is None:
        return last_pct
    pct = min(100, int(cur * 100 / total))
    if pct != last_pct:
        on_progress(phase, cur, total, detail)
        return pct
    return last_pct


def _load_one_file(fp: str, needed_cols: List[str], on_progress,
                   _estimated_total: int, _shared_rows: list, _shared_last_pct: list,
                   _progress_lock) -> "pd.DataFrame":
    """Load a single input file (Excel) and emit row progress.

    Checks the per-file raw-Excel cache
    (keyed on this file's own content + size + ``needed_cols``, shared cache
    format/directory) before parsing, and saves a fresh parse for next time.
    ``load_and_preprocess`` discards the DataFrame immediately after
    preprocessing each file, so without this a large multi-file run would
    re-parse every input file on every run regardless of whether it had
    changed.
    """
    from .excel_io import _read_excel_streaming

    try:
        from .preprocess_cache import load_raw_excel_cache_one as _load_raw_one
        _cached_df = _load_raw_one(fp, needed_cols)
    except Exception:
        _cached_df = None

    if _cached_df is not None:
        df = _cached_df
        if _progress_lock is not None and on_progress and _estimated_total > 0:
            with _progress_lock:
                _shared_rows[0] += len(df)
                pct = min(100, int(_shared_rows[0] * 100 / _estimated_total))
                if pct != _shared_last_pct[0]:
                    _shared_last_pct[0] = pct
                    on_progress("init", _shared_rows[0], _estimated_total,
                                f"Loaded from cache: {_shared_rows[0]:,} / {_estimated_total:,}")
        return df

    df = _read_excel_streaming(
        fp,
        needed_cols=needed_cols,
        on_progress=on_progress,
        total_rows=_estimated_total,
        rows_processed_ref=_shared_rows,
        last_pct_ref=_shared_last_pct,
        progress_lock=_progress_lock,
    )

    try:
        from .preprocess_cache import save_raw_excel_cache_one as _save_raw_one
        _save_raw_one(fp, df, needed_cols)
    except Exception:
        pass

    return df


def _emit_file_load_done(on_progress, file_results: dict, file_list: list,
                         fi: int, n_rows: int) -> None:
    """Emit the per-file completion progress used by both loaders."""
    if not on_progress:
        return
    _n_done = sum(1 for v in file_results.values() if v is not None)
    on_progress("file_load", _n_done, len(file_list),
                f"Loaded + preprocessed {os.path.basename(file_list[fi])}: {n_rows:,} rows")


def _emit_postprocess_progress(on_progress, total: int, elapsed: float,
                               message: str,
                               next_phase: str = "Analysing population mix and CT dose coverage...") -> None:
    """Emit the standard post-load progress triplet used by all preprocess paths."""
    if not on_progress:
        return
    on_progress("init", total, total,
                f"{message}: {total:,} exams in {elapsed:.1f}s")
    on_progress("unknown_regions", region_resolver.get_unknown_descriptions())
    on_progress("init", total, total, next_phase)


def _process_single_row(col_data: dict, i: int, fi: int) -> dict:
    """Process a single exam row: build dict, compute doses, attach metadata.

    Uses the zero-allocation ``_ColumnRowView`` (the same view the parallel
    ``_compute_exam_batch`` worker uses) instead of materialising a full
    per-row dict and rebuilding ``list(col_data.keys())`` on every row.
    The dose dispatcher then touches only the columns the row's modality
    actually needs.
    """
    row = _ColumnRowView(col_data, i)
    result = compute_exam_organ_doses(row)
    result["protocol"] = str(row.get("Protocol", "") or "")
    _dev = row.get("Device")
    _eqp = row.get("Equipment")
    _dte = row.get("Date Performed")
    result["device"]         = "" if _dev is None else str(_dev)
    result["equipment"]      = "" if _eqp is None else str(_eqp)
    result["date_performed"] = _dte
    result["_file_index"] = fi
    return result


def _merge_unknown_descs(unk) -> None:
    """Merge worker-reported unknown region descriptions into the module cache.

    Deduplicates by the lowercased description so the same unknown reported by
    multiple workers/files is recorded once.  Must be called under a lock when
    invoked from multiple threads (the multi-file loaders do, since their
    ThreadPool workers each drive batches through the shared process pool).
    """
    for _d, _m in unk:
        _key = str(_d or "").strip().lower()
        if _key and _key not in region_resolver._UNKNOWN_DESCRIPTION_CACHE:
            region_resolver._UNKNOWN_DESCRIPTION_CACHE.add(_key)
            region_resolver._UNKNOWN_DESCRIPTIONS.append((_d, _m))


def _preprocess_col_data(col_data: dict, n_rows: int, fi: int, executor,
                         on_batch_done) -> list:
    """Preprocess one file's column-major data into exam dicts.

    ``executor`` is a shared ProcessPoolExecutor (or None → sequential in the
    calling thread).  ``on_batch_done(n_rows_in_batch, unknown_descs)`` is
    called after every completed batch (sequential mode: once per row with
    an empty list) so callers can emit progress / merge unknowns.

    In parallel mode the per-row work is dispatched to ``_compute_exam_batch``,
    so the per-row computation is byte-identical to sequential mode.
    """
    results: list = []
    if executor is None:
        for i in range(n_rows):
            results.append(_process_single_row(col_data, i, fi))
            on_batch_done(1, [])
        return results

    n_workers = executor._max_workers
    batch = max(5_000, n_rows // (n_workers * 4))
    _pool_failed = False

    def _warn_pool(exc):
        nonlocal _pool_failed
        if not _pool_failed:
            print(f"[preprocess] worker pool failed ({type(exc).__name__}); "
                  f"finishing this file's rows in-process", flush=True)
            _pool_failed = True

    futures = []
    try:
        for s in range(0, n_rows, batch):
            e = min(s + batch, n_rows)
            _slice = {k: v[s:e] for k, v in col_data.items()}
            _args = (_slice, [fi] * (e - s), 0, e - s)
            futures.append((executor.submit(_compute_exam_batch, _args), e - s, _args))
    except (MemoryError, BrokenProcessPool, OSError) as exc:
        _warn_pool(exc)
        # Submit itself died: recompute every batch inline (results list is
        # still empty -- nothing to duplicate).
        for s in range(0, n_rows, batch):
            e = min(s + batch, n_rows)
            _slice = {k: v[s:e] for k, v in col_data.items()}
            batch_results, unk_descs = _compute_exam_batch(
                (_slice, [fi] * (e - s), 0, e - s))
            results.extend(batch_results)
            on_batch_done(e - s, unk_descs)
        return results
    for fut, n, _args in futures:
        try:
            batch_results, unk_descs = fut.result()
        except (MemoryError, BrokenProcessPool, OSError) as exc:
            _warn_pool(exc)
            batch_results, unk_descs = _compute_exam_batch(_args)
        results.extend(batch_results)
        on_batch_done(n, unk_descs)
    return results
def _validate_input_columns(df):
    """One-pass validation of the per-file sex and Population columns.

    Returns ``(bad_sex, bad_pop)`` — sorted lists of unrecognised distinct
    values.  A bad value must not reach a worker, where it would raise
    inside ``normalize_sex_code`` / ``_parse_population`` without the file
    context; the caller aggregates the results across files and raises one
    ``ValueError`` naming every file.
    """
    bad_sex = []
    sex_col = ("Gender" if "Gender" in df.columns
               else "Sex" if "Sex" in df.columns else None)
    if sex_col is not None:
        for v in df[sex_col].unique():
            try:
                normalize_sex_code(v)
            except ValueError:
                bad_sex.append(v)
    bad_pop = []
    if "Population" in df.columns:
        for v in df["Population"].unique():
            try:
                _parse_population(v)
            except ValueError:
                bad_pop.append(v)
    return sorted(bad_sex, key=str), sorted(bad_pop, key=str)

def load_and_preprocess(
    file_list: list,
    needed_cols=None,
    on_progress=None,
) -> Tuple[List[dict], int, List[int]]:
    """Load every input file and preprocess its rows into exam dicts.

    Each file (Excel; per-file raw-Excel cache aware) is
    loaded and preprocessed in a single pass.  Per-row work runs on a
    shared process pool when the dataset is large enough, else in the
    loading thread.  A header-only pre-check reports any expected
    columns absent from the input files.

    Returns
    -------
    (exam_data, total, row_file_index)
        Exam dicts in file order, the total row count, and the per-row
        file-index list.
    """
    from .excel_io import DEFAULT_NEEDED_COLS, report_missing_headers

    if needed_cols is None:
        needed_cols = DEFAULT_NEEDED_COLS

    report_missing_headers(list(file_list), frozenset(needed_cols), on_progress)

    # One salt per run, shared across all threads (same process, so the
    # module global set here is visible to every worker thread) and across
    # the shared process pool (workers get it via the initializer).
    _salt = new_run_patient_id_salt()
    set_patient_id_salt(_salt)
    _n_threads = min(len(file_list), max(1, mp.cpu_count() or 1))
    _t0 = time.perf_counter()
    region_resolver.clear_unknown_descriptions()
    _estimated_total = _estimate_total_rows(file_list)
    _shared_rows = [0]
    _shared_last_pct = [-1]
    _progress_lock = threading.Lock()

    _pool = None
    if _estimated_total == 0 or _estimated_total >= _PARALLEL_MIN_ROWS:
        _pool = _make_preprocess_pool(
            _preprocess_pool_workers(_estimated_total), _salt)

    def _on_batch(n, unk):
        with _progress_lock:
            _shared_rows[0] += n
            _merge_unknown_descs(unk)
            if _estimated_total > 0:
                _shared_last_pct[0] = _emit_1pct_progress(
                    "init", _shared_rows[0], _estimated_total, _shared_last_pct[0],
                    f"Preprocessing: {_shared_rows[0]:,} / {_estimated_total:,}",
                    on_progress,
                )

    _bad_sex: dict = {}  # filepath -> sorted list of unrecognised sex values
    _bad_pop: dict = {}  # filepath -> sorted list of unrecognised population values

    def _load_and_prep(fi_fp):
        """Load one file and preprocess its rows."""
        fi, fp = fi_fp
        # 1. Load (streaming with progress when possible)
        df = _load_one_file(fp, needed_cols, on_progress, _estimated_total,
                            _shared_rows, _shared_last_pct, _progress_lock)
        n_rows = len(df)
        bad_sex, bad_pop = _validate_input_columns(df)
        if bad_sex or bad_pop:
            with _progress_lock:
                if bad_sex:
                    _bad_sex[fp] = bad_sex
                if bad_pop:
                    _bad_pop[fp] = bad_pop
            return fi, [], n_rows
        # 2. Convert once
        col_data = df.to_dict(orient="list")
        del df
        # 3. Preprocess every row (shared process pool when available)
        results = _preprocess_col_data(col_data, n_rows, fi, _pool, _on_batch)
        del col_data
        return fi, results, n_rows

    # Run all files in parallel
    _file_results: dict = {}
    try:
        with ThreadPoolExecutor(max_workers=_n_threads) as _tex:
            _futs = {
                _tex.submit(_load_and_prep, (fi, fpath)): fpath
                for fi, fpath in enumerate(file_list)
            }
            for _fut in as_completed(_futs):
                fi, results, n_rows = _fut.result()
                _file_results[fi] = (results, n_rows)
                _emit_file_load_done(on_progress, _file_results, file_list, fi, n_rows)
    finally:
        if _pool is not None:
            _pool.shutdown(wait=True)

    if _bad_sex or _bad_pop:
        parts = []
        if _bad_sex:
            details = "; ".join(
                f"{os.path.basename(str(fp))}: {vals}"
                for fp, vals in sorted(_bad_sex.items(), key=lambda kv: str(kv[0]))
            )
            parts.append(f"unrecognised sex value(s): {details}")
        if _bad_pop:
            details = "; ".join(
                f"{os.path.basename(str(fp))}: {vals}"
                for fp, vals in sorted(_bad_pop.items(), key=lambda kv: str(kv[0]))
            )
            parts.append(f"unrecognised Population value(s): {details}")
        raise ValueError(f"Invalid input file(s) — {'; '.join(parts)}")

    # Merge results in file order
    exam_data: list = []
    row_file_index: list = []
    for fi in range(len(file_list)):
        results, n_rows = _file_results[fi]
        exam_data.extend(results)
        row_file_index.extend([fi] * n_rows)

    total = len(exam_data)
    _emit_postprocess_progress(
        on_progress, total, time.perf_counter() - _t0,
        "Load+preprocess done",
    )
    return exam_data, total, row_file_index


def _estimate_total_rows(file_list: List[str]) -> int:
    """Fast total-row estimate using metadata.

    Returns 0 if metadata cannot be read.  This is only used for progress
    reporting; the actual simulation uses the real row counts after loading.
    """
    from .excel_io import _estimate_file_rows
    total = 0
    for fp in file_list:
        try:
            total += _estimate_file_rows(fp)
        except Exception:
            pass
    return max(0, total)
def _merge_exam_group(group: List[dict]) -> dict:
    """Combine one examination's acquisition rows into a single exam dict.

    ``group`` holds the per-acquisition exam dicts that share a
    ``(patient, date, modality, protocol)`` key.  Absorbed organ dose,
    effective dose and gonad dose are additive across acquisitions, so the
    merged exam reproduces what the dose-tracking system would have reported
    had it emitted one row for the whole study.  The highest-effective-dose
    acquisition (the main diagnostic series) is the template, so the merged
    exam keeps that series' region / dose source / metadata; the low-dose
    scout and bolus-tracking series only contribute their dose.  Cancer-site
    doses, modelled types and the gonad dose are rebuilt from the summed
    organ doses with the same canonical helper the per-row path uses, so the
    result is identical to a natively combined row.
    """
    rep = max(group, key=lambda e: float(e.get("ed_msv", 0.0) or 0.0))
    merged = dict(rep)

    def _sum_dicts(field: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for e in group:
            d = e.get(field)
            if not d:
                continue
            for k, v in d.items():
                fv = float(v or 0.0)
                if fv:
                    out[k] = out.get(k, 0.0) + fv
        return out

    organ_doses = _sum_dicts("organ_doses_gy")
    sex_code = rep.get("sex_code", "U")
    cancer_doses, modeled_types = _build_cancer_dose_map(organ_doses, sex_code)
    _gkey = HERITABLE_GONAD_DOSE_KEY.get(sex_code)
    gonad_dose = float(get_organ_dose_for_cancer(_gkey, organ_doses)) if _gkey else 0.0

    merged["organ_doses_gy"] = organ_doses
    merged["_cancer_doses"] = cancer_doses
    merged["_modeled_types"] = modeled_types
    merged["_gonad_dose_gy"] = gonad_dose
    merged["_remainder_individual_doses"] = _sum_dicts("_remainder_individual_doses")
    merged["ed_msv"] = sum(float(e.get("ed_msv", 0.0) or 0.0) for e in group)
    merged["ed_msv_no_golikov"] = sum(
        float(e.get("ed_msv_no_golikov", 0.0) or 0.0) for e in group
    )
    merged["_golikov_no_weight"] = any(e.get("_golikov_no_weight") for e in group)
    merged["_golikov_eligible"] = any(e.get("_golikov_eligible") for e in group)
    merged["_dose_quantity_missing"] = all(e.get("_dose_quantity_missing") for e in group)
    return merged


def merge_acquisition_series(exam_data, on_progress=None):
    """Merge multi-acquisition sub-series of one examination into one exam.

    Dose-tracking exports split a single clinical examination into one row
    per acquisition series -- e.g. a contrast CT emits PreMonitoring and
    Monitoring (bolus-tracking) rows alongside the diagnostic phase, and a
    plain radiograph emits one row per projection (PA + lateral).  Treating
    each row as its own exam inflates per-modality scan counts and dilutes
    the average dose per examination, even though total dose and per-Sv risk
    are unaffected (dose is additive and risk is ~linear at these levels).

    Rows are grouped by ``_patient_id`` (the ``A#`` accession number) and
    summed via :func:`_merge_exam_group`.  ``A#`` is unique per examination,
    so it alone is sufficient as the merge key.  A row with no ``A#``
    stays solo so unrelated rows can never collapse together.

    Returns a new exam-data list (one entry per examination); the original
    list is returned unchanged when nothing is mergeable.
    """
    if not exam_data:
        return exam_data

    groups: Dict[object, List[int]] = {}
    order: List[object] = []
    total = len(exam_data)
    _last_pct = -1
    for i, ex in enumerate(exam_data):
        pid = ex.get("_patient_id")
        pid_s = "" if pid is None else str(pid).strip()
        if not pid_s:
            key: object = ("__solo__", i)        # no accession id -> never merges
        else:
            key = pid_s
        bucket = groups.get(key)
        if bucket is None:
            groups[key] = [i]
            order.append(key)
        else:
            bucket.append(i)
        _last_pct = _emit_1pct_progress(
            "init", i + 1, total, _last_pct,
            f"Grouping acquisition series: {i + 1:,} / {total:,}",
            on_progress,
        )

    n_multi = sum(1 for k in order if len(groups[k]) > 1)
    if n_multi == 0:
        return exam_data

    merged_data = []
    for key in order:
        idxs = groups[key]
        if len(idxs) == 1:
            merged_data.append(exam_data[idxs[0]])
        else:
            merged_data.append(_merge_exam_group([exam_data[i] for i in idxs]))

    if on_progress:
        n_before = len(exam_data)
        n_after = len(merged_data)
        on_progress("init", n_after, n_after,
                    f"Combined {n_before:,} acquisition rows into {n_after:,} "
                    f"examinations ({n_multi:,} multi-series studies merged)")
    return merged_data


def analyze_population_and_ct_doses(exam_data, file_list, population, total, on_progress=None):
    """Single O(N) scan: population mix, per-file report, CT dose coverage, auto-detect."""
    _pop_counts = {"Euro-American": 0, "Asian": 0, "unspecified": 0}
    _file_pop = [{"Euro-American": 0, "Asian": 0, "unspecified": 0} for _ in file_list]
    _ct_no_dose: Dict[str, dict] = {}
    _last_pct = -1
    for i, ex in enumerate(exam_data):
        ep = ex.get("_population")
        fi = ex.get("_file_index", 0)
        pid = ex.get("_patient_id")
        if ep == "Euro-American":
            _pop_counts["Euro-American"] += 1
            _file_pop[fi]["Euro-American"] += 1
        elif ep == "Asian":
            _pop_counts["Asian"] += 1
            _file_pop[fi]["Asian"] += 1
        else:
            _pop_counts["unspecified"] += 1
            _file_pop[fi]["unspecified"] += 1
        # CBCT is excluded from this alert entirely: dental CBCT gets its
        # dose from the DAP x dental-coefficient model (see exam_doses.py),
        # and non-dental CBCT (extremity/MSK CBCT, unlocalized CBCT) has no
        # organ-dose model by design -- neither case is a data problem worth
        # surfacing here. Only genuine CT rows are checked.
        if ex.get("modality") == "CT" and not ex.get("organ_doses_gy"):
            region = ex.get("region") or "Unknown"
            if region not in _ct_no_dose:
                _ct_no_dose[region] = {"patient_ids": set(), "n_exams": 0, "expected_organs": _CT_EXPECTED_ORGANS}
            if pid is not None:
                _ct_no_dose[region]["patient_ids"].add(pid)
            _ct_no_dose[region]["n_exams"] += 1
        _last_pct = _emit_1pct_progress(
            "init", i + 1, total, _last_pct,
            f"Analysing population mix: {i + 1:,} / {total:,}",
            on_progress,
        )

    if on_progress and len(file_list) > 0:
        _file_pop_parts = []
        for fi, fpath in enumerate(file_list):
            fname = os.path.basename(fpath)
            fp = _file_pop[fi]
            _tagged = fp["Euro-American"] + fp["Asian"]
            if _tagged > 0:
                ea_pct = round(100.0 * fp["Euro-American"] / _tagged)
                _file_pop_parts.append(f"{fname}: {ea_pct}% EuroAmerican, {100 - ea_pct}% Asian")
            else:
                _file_pop_parts.append(f"{fname}: {population} (config default)")
        on_progress("init", total, total,
                    "Risk models used: " + " | ".join(_file_pop_parts))

    _n_tagged = _pop_counts["Euro-American"] + _pop_counts["Asian"]
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
        if on_progress:
            _pop_desc = (f"blending EA({_ea_ratio:.0%}) + AS({_as_ratio:.0%})"
                         if _use_mixed_kernels else f"using {population}")
            on_progress("init", total, total,
                        f"Population from Excel: {_pop_counts['Euro-American']:,} Euro-American, "
                        f"{_pop_counts['Asian']:,} Asian, "
                        f"{_pop_counts['unspecified']:,} unspecified \u2192 {_pop_desc}")

    return population, _ea_ratio, _as_ratio, _use_mixed_kernels, _ct_no_dose, _pop_counts, _file_pop


def apply_golikov_and_cumulative_dose(exam_data, total, cfg_use_golikov, cfg_use_mataac, on_progress=None):
    """Apply Golikov toggle and compute per-patient cumulative dose + Mataac frailty."""
    if not cfg_use_golikov:
        n_swapped = 0
        n_organ_reverted = 0
        _last_pct = -1
        for i, ex in enumerate(exam_data):
            no_g = ex.get("ed_msv_no_golikov")
            if no_g is not None and no_g > 0 and no_g != ex.get("ed_msv"):
                ex["ed_msv"] = no_g
                n_swapped += 1
            gf = float(ex.get("_golikov_factor", 1.0) or 1.0)
            if gf > 0.0 and gf != 1.0:
                inv_gf = 1.0 / gf
                # Build fresh dose dicts so a Golikov re-run cannot corrupt shared interned objects.
                od = ex.get("organ_doses_gy")
                if od:
                    ex["organ_doses_gy"] = {_k: _v * inv_gf for _k, _v in od.items()}
                cd = ex.get("_cancer_doses")
                if cd:
                    ex["_cancer_doses"] = {_k: _v * inv_gf for _k, _v in cd.items()}
                rid = ex.get("_remainder_individual_doses")
                if rid:
                    ex["_remainder_individual_doses"] = {_k: _v * inv_gf for _k, _v in rid.items()}
                gonad = ex.get("_gonad_dose_gy", 0.0) or 0.0
                if gonad:
                    ex["_gonad_dose_gy"] = gonad * inv_gf
                n_organ_reverted += 1
            _last_pct = _emit_1pct_progress(
                "init", i + 1, total, _last_pct,
                f"Reverting Golikov weight adjustment: {i + 1:,} / {total:,}",
                on_progress,
            )
        if on_progress and (n_swapped or n_organ_reverted):
            on_progress("init", total, total,
                        f"Golikov weight adjustment DISABLED — "
                        f"reverted {n_swapped:,} ED and {n_organ_reverted:,} "
                        f"organ-dose exams to unweighted (reference-phantom) values")

    if on_progress:
        on_progress("init", total, total,
                    "Computing per-patient cumulative doses and frailty weights...")

    # Single-pass extraction of all per-exam fields needed downstream:
    # one loop costs ~N Python iterations instead of ~4 x N plus the
    # extra dict.get calls of separate comprehensions.
    def _pid_or_synth(_pid, _i):
        if _pid is None or _pid == "":
            return f"__row__{_i}"
        if isinstance(_pid, float) and _pid != _pid:  # NaN
            return f"__row__{_i}"
        return str(_pid)
    _pids_list = [None] * total
    _dates_list = [None] * total
    _ages  = np.empty(total, dtype=np.float64)
    _eds   = np.empty(total, dtype=np.float64)
    # Fields only needed when Mataac is active; allocated here to avoid a
    # second pass through exam_data below.
    _raw_dates = [None] * total if cfg_use_mataac else None
    _mods = [None] * total if cfg_use_mataac else None
    _sex_codes = [None] * total if cfg_use_mataac else None
    _populations = [None] * total if cfg_use_mataac else None
    _n_acc_fallback = 0  # counted here to avoid a separate pass below
    for i, ex in enumerate(exam_data):
        _pid = ex.get("_person_id")
        if _pid is None:
            _pid = ex.get("_patient_id")
        _pids_list[i] = _pid_or_synth(_pid, i)
        _d = ex.get("_exam_date")
        _dates_list[i] = str(_d or "")
        _ages[i] = float(ex.get("age", 0) or 0)
        _eds[i] = max(0.0, ex.get("ed_msv", 0.0)) / 1000.0
        if cfg_use_mataac:
            _raw_dates[i] = _d
            _mod = ex.get("Acquisition Modality")
            if not _mod:
                _mod = ex.get("modality")
            _mods[i] = str(_mod or "")
            _sex_codes[i] = ex.get("sex_code", "U")
            _populations[i] = ex["_population"]
        _src = ex.get("_person_id_source")
        if _src in ("accession_fallback", "none"):
            _n_acc_fallback += 1
    _pids  = np.array(_pids_list)
    _dates = np.array(_dates_list)

    # lexsort: last key = primary → sort by pid, then date, then age within each pid
    _perm = np.lexsort((_ages, _dates, _pids))
    _pids_s = _pids[_perm]
    _eds_s  = _eds[_perm]

    # Find patient boundaries: first occurrence of each pid in sorted order
    _, _boundaries, _counts = np.unique(_pids_s, return_index=True, return_counts=True)
    n_patients = len(_counts)

    # Cumulative dose within each patient group
    _cum_dose = np.zeros(total, dtype=np.float64)
    _total_dose = np.zeros(total, dtype=np.float64)
    for _b, _c in zip(_boundaries, _counts):
        _end = _b + _c
        _grp_eds = _eds_s[_b:_end]
        _total_dose[_b:_end] = _grp_eds.sum()
        _cum_dose[_b:_end] = np.cumsum(_grp_eds) - _grp_eds  # prior dose = cumsum excluding self

    # Scatter back to original order (vectorized)
    _inv = np.empty(total, dtype=np.intp)
    _inv[_perm] = np.arange(total)
    _prior = _cum_dose[_inv]
    _total = _total_dose[_inv]
    for _i in range(total):
        exam_data[_i]["_prior_cumulative_dose_sv"] = float(_prior[_i])
        exam_data[_i]["_total_patient_dose_sv"] = float(_total[_i])

    if cfg_use_mataac:
        # Mataac (2024) competing-mortality adjustment: heavily-imaged patients have elevated
        # all-cause mortality, reducing their chance of surviving to realise a radiation-induced
        # cancer.  The correction adds an empirical excess hazard to the ICRP cancer-free
        # survival product for the patient's cumulative-dose stratum
        # (risk_models.mataac_survival_correction).  The returned factor multiplies each exam's
        # incidence, deaths, and detriment via surv_correction in vectorized.py.
        #
        # Cumulative-effective-dose strata / observed 10-yr survival
        # (Mataac et al., Eur J Radiol 2024;178:111629):
        #   <10 mSv -> 0.846 (reference: standard ICRP survival, factor = 1)
        #   10-50   -> 0.739
        #   50-100  -> 0.480
        #   >=100   -> 0.367
        #
        # STRATIFICATION VARIABLE.  The stratum must be measured the same way the mortality
        # table was built, or the table is being indexed with the wrong variable.  In the
        # source cohort the CED is the effective dose from CT examinations accrued by one
        # patient within ONE CALENDAR YEAR (dose data collected January-December 2013; vital
        # status followed 2013-2023 with year 0 = the accrual year), and cohort entry requires
        # at least one CT in that year.  We therefore key the stratum on the patient's CT-only
        # effective dose within each calendar year, NOT on the whole-record total across the
        # available imaging history.  Three consequences, all of them deliberate:
        #   * No look-ahead.  A stratum built from the whole record would charge an early exam
        #     with the mortality of a patient whose dose is only accrued years later - person
        #     time in which the record itself proves the patient survived.  A calendar-year CED
        #     is fixed before its own follow-up window opens, exactly as in the source cohort.
        #   * No accrual-window inflation.  Reaching 50 mSv within twelve months marks acute,
        #     intensive investigation; reaching it across nine years need not.  Accruing over
        #     the longer window would promote patients into strata whose observed mortality
        #     they do not share.
        #   * Cohort membership.  Patients with no CT in a given year get zero CED for that
        #     year and fall in the reference stratum, because they are not represented anywhere
        #     in the source cohort - not even in its lowest dose group.  This is conservative:
        #     it withholds a risk-reducing correction rather than extrapolating one.
        # The factor derived from the year's CED is then applied to every examination of that
        # patient in that year regardless of modality, because it represents the patient's
        # clinical condition rather than a property of the individual examination.
        #
        # NO AGE RESTRICTION.  The yearly CT CED is the only criterion; the <10 mSv stratum is
        # left unadjusted (no net excess hazard over the ICRP reference; see risk_models).
        # The strata are defined by dose alone - Mataac reports no age-restricted analysis, and
        # the death counts the excess hazard is built from already contain the groups' own
        # under-30s (Table 1 gives 52.5 +/- 17.6, 50.1 +/- 15.4 and 52.9 +/- 14.2 y for B, C and
        # D, i.e. ~10%, ~10% and ~5% of each group below 30 under the same Gaussian
        # reconstruction used to standardise the baseline).  An age gate would therefore index
        # the table with a different population from the one that built it, and the paper's own
        # Table 2 points the other way: in every group the deceased were YOUNGER than the living
        # (group B 40.8 +/- 14.0 vs 56.7 +/- 16.9 y; the split reproduces Table 1's group means
        # exactly, so it is not a transcription slip).  External data agree: of 18-35 year-olds
        # scanned >15 times, 46% (32/70) died over a mean 5.4 y, mostly cancer and CF/lung
        # transplant patients (Zondervan et al., J Am Coll Radiol 2011;8:501-7, verified against
        # the abstract, PMID 21723488).  A cut would also be a step, not
        # a taper: the correction is near flat in age below ~60 because the excess is additive
        # and dwarfs the baseline hazard (group D gives 0.483 at age 5 and 0.483 at age 30), so
        # any threshold puts a discontinuity of up to 2x at an age with no data behind it.
        # Caveat carried in the manuscript rather than in code: that same additive, age-invariant
        # excess is an extrapolation at paediatric ages, where a multiplicative form would give a
        # far smaller absolute excess.  It is left unrestricted because the >=10 mSv yearly CT
        # CED requirement is itself the frailty selector at any age.
        MATAAC_MIN_DOSE_SV = 0.010  # adjustment applied only at yearly CT CED >= 10 mSv

        # Calendar year of each exam, from the shared "Date Performed" parser
        # (excel_io.parse_exam_dates) so the year the stratum is keyed on can never
        # disagree with the year reported in year_stats or used by "Auto-Sort by Year",
        # which read the same column through the same helper.  A multi-file run mixes
        # encodings freely -- datetime objects, ISO strings, locale strings and bare
        # Excel serial day numbers all show up -- and every one of them has to survive,
        # since a row the parser gives up on drops out of its patient-year and loses
        # the correction.  Anything still unparseable is counted and warned about below.
        # _raw_dates was pre-extracted in the merged loop above.
        _years: list = [None] * total
        try:
            from .excel_io import parse_exam_dates as _parse_dates
            _yr = _parse_dates(_raw_dates).dt.year.to_numpy(dtype="float64")
            _years = [None if _v != _v else int(_v) for _v in _yr]
        except Exception:
            # pandas unavailable or an unexpected dtype: fall back to per-row parsing.
            # The serial branch is not optional here either -- str(43013)[:4] is "4301",
            # which would fragment one true year into ten-day buckets rather than fail.
            from datetime import date as _date, timedelta as _timedelta
            _epoch = _date(1899, 12, 30)
            for _i, _d in enumerate(_raw_dates):
                _y = getattr(_d, "year", None)
                if _y is None:
                    _s = str(_d or "").strip()
                    try:
                        _n = float(_s)
                    except ValueError:
                        _n = None
                    if _n is not None and 20000.0 < _n < 80000.0:
                        _y = (_epoch + _timedelta(days=int(_n))).year
                    elif len(_s) >= 4 and _s[:4].isdigit():
                        _y = int(_s[:4])
                _years[_i] = _y

        # Rows whose date cannot be recovered get a per-row sentinel: they cannot be
        # pooled into a calendar year without guessing, so each forms its own index
        # period.  That leaves them in the reference stratum unless the single exam
        # alone clears 10 mSv, which is the conservative direction.
        n_undated = sum(1 for _y in _years if _y is None)

        # CT-only effective dose per (patient, calendar year) = the source cohort's CED.
        # Uses the pre-extracted _mods and _eds arrays to avoid re-reading from dicts.
        _ct_ced: dict = {}
        _index_keys = []
        for _i in range(total):
            _y = _years[_i]
            _key = (_pids[_i], _y if _y is not None else f"__undated_{_i}")
            _index_keys.append(_key)
            if _mods[_i] == "CT":
                _ct_ced[_key] = _ct_ced.get(_key, 0.0) + _eds[_i]

        # Pre-compute rounded ages as int array to avoid N Python round()+int()
        # calls in the apply loop below (measured ~89 K calls / ~0.1 s on 200 K exams).
        _ages_int = np.round(_ages).astype(np.int64)
        _n_frail = 0
        for _i in range(total):
            _ced = _ct_ced.get(_index_keys[_i], 0.0)
            ex = exam_data[_i]
            ex["_mataac_index_ced_sv"] = _ced
            if _ced >= MATAAC_MIN_DOSE_SV:
                _factor = mataac_survival_correction(
                    int(_ages_int[_i]),
                    _sex_codes[_i],
                    _ced,
                    _populations[_i],
                )
                ex["_mataac_frailty_factor"] = _factor
                if _factor < 1.0:
                    _n_frail += 1
            else:
                ex["_mataac_frailty_factor"] = 1.0
        n_ct_years = len(_ct_ced)
    else:
        n_undated = 0
        n_ct_years = 0
        for ex in exam_data:
            ex["_mataac_frailty_factor"] = 1.0
            ex["_mataac_index_ced_sv"] = 0.0

    # Summary stats: use the already-scattered _prior array instead of
    # re-reading from dicts (avoids 2 × N dict.get calls + 2 temp arrays).
    n_with_prior = int((_prior > 0).sum())
    n_frail = _n_frail if cfg_use_mataac else 0
    if on_progress:
        mataac_status = "enabled" if cfg_use_mataac else "DISABLED"
        _undated_note = (
            f", {n_undated:,} exams with unparseable dates" if n_undated else ""
        )
        on_progress("init", total, total,
                    f"Cumulative dose: {n_patients:,} patients, "
                    f"{n_with_prior:,} exams with prior dose, "
                    f"{n_frail:,} with Mataac frailty ({mataac_status}; "
                    f"{n_ct_years:,} patient-years with CT{_undated_note})")

    # A stable per-person key is what makes both the cumulative prior dose and the
    # Mataac stratum mean anything.  When an export carries no "Patient MRN",
    # exam_doses falls back to the accession number (see
    # ``_person_id_source``), which is unique per EXAMINATION: every patient then looks
    # like a one-exam patient, prior cumulative dose is always zero, and the Mataac CED
    # collapses from "this patient's CT dose this calendar year" to "this one exam's
    # dose", so only single exams above 10 mSv are ever adjusted.  The run completes
    # normally and simply under-applies the correction, which is why it needs saying
    # out loud -- and in a multi-file run only *some* files may be affected.
    # _n_acc_fallback was counted in the merged extraction loop above.
    if _n_acc_fallback:
        _pct = 100.0 * _n_acc_fallback / max(1, total)
        _warn = (f"WARNING: {_n_acc_fallback:,} of {total:,} exams ({_pct:.1f}%) have no "
                 f"\"Patient MRN\" -- each falls back to its own "
                 f"accession number, so they cannot be grouped into patients. Prior "
                 f"cumulative dose reads 0 and the Mataac stratum sees one exam at a "
                 f"time instead of a patient-year")
        if _pct >= 99.9:
            _warn += (f". Every exam in this run is its own patient "
                      f"({n_patients:,} \"patients\" for {total:,} exams)")
        if on_progress:
            on_progress("init", total, total, _warn)
        else:
            print(f"[{_warn}]", flush=True)

    # Undated rows silently disable the correction for the exams they belong
    # to: without a calendar year each one becomes its own index period, so
    # only a single exam above 10 mSv can still reach a frail stratum.  With
    # one input file that shows up as an obviously wrong frailty count, but a
    # multi-file or per-year batch reports one line per year and a single file
    # whose date column uses an unrecognised encoding is easy to miss -- the
    # run then quietly over-estimates risk for that year only, which is worse
    # than failing.  Warn separately, and loudly when a whole run is affected.
    if cfg_use_mataac and n_undated:
        _pct = 100.0 * n_undated / max(1, total)
        _warn = (f"WARNING: {n_undated:,} of {total:,} exams ({_pct:.1f}%) have an "
                 f"unparseable \"Date Performed\" -- the Mataac frailty correction "
                 f"cannot assign them a calendar-year CT dose and leaves them "
                 f"unadjusted (risk is over-estimated for those exams)")
        if _pct >= 99.9:
            _warn += ". No exam in this run has a usable date; check the date column."
        if on_progress:
            on_progress("init", total, total, _warn)
        else:
            print(f"[{_warn}]", flush=True)

    for ex in exam_data:
        ex.setdefault("_prior_cumulative_dose_sv", 0.0)
        ex.setdefault("_total_patient_dose_sv", 0.0)
        ex.setdefault("_mataac_frailty_factor", 1.0)
        ex.setdefault("_mataac_index_ced_sv", 0.0)

    # Intern content-identical dose dicts and low-cardinality strings to save ~40% per-exam memory.
    # Safe because the COW revert above guarantees no later in-place mutation.
    from .preprocess_cache import (
        _dedup_dose_dicts_inplace,
        _intern_low_cardinality_strings_inplace,
    )
    _dedup_dose_dicts_inplace(exam_data)
    _intern_low_cardinality_strings_inplace(exam_data)
