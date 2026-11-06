"""
Persistent preprocessing caches for Monte Carlo simulation.

Two on-disk caches let re-runs with the same Excel files and
configuration skip expensive work:

- A raw-Excel cache keyed per input file that stores each file's parsed
  DataFrame so a cold run does not re-pay the Excel parse.
- A full cache keyed on all input files plus the source hashes of the
  modules that affect preprocessing; it stores the complete
  post-``build_exam_arrays`` state (exam_data + columnar ``ea`` arrays)
  so a warm hit goes straight to the iteration loop.

This mirrors the pattern used in risk_models._kernel_cache_*.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import gzip
import sys
from typing import Dict, List, Optional, Tuple

from .constants import DEFAULT_OUTPUT_DIR

# When set by the GUI before simulation starts, patient-data caches
# (raw Excel cache and preprocess cache) are stored under this directory
# instead of next to the package source files.  This keeps patient data
# in a user-chosen, explicitly-known location.
_CACHE_OUTPUT_DIR: Optional[str] = None


def set_cache_output_dir(path: Optional[str]) -> None:
    """Set the directory used for patient-data cache files.

    Called once by the GUI before ``run_simulation`` starts, using the
    output directory selected in the config dialog.  When set, the raw
    Excel cache and preprocess cache are written under
    ``<path>/.raw_excel_cache`` and ``<path>/.preprocess_cache``
    respectively.  When ``None`` (the default), caches fall back to
    ``constants.DEFAULT_OUTPUT_DIR``.
    """
    global _CACHE_OUTPUT_DIR
    _CACHE_OUTPUT_DIR = path


# Optional zstd compression: ~3-5× faster decompression than gzip at the
# same on-disk size on the 1 M-exam radiography blob (1.32 s → ~0.3 s).
# Falls back to gzip transparently when ``zstandard`` is not installed.
# Format detection on load is via the magic bytes in the file header
# (zstd: 0x28b52ffd LE; gzip: 0x1f8b), so existing gzip caches keep
# working after this module change without bumping the format version.
try:
    import zstandard as _zstd  # type: ignore
    _HAS_ZSTD = True
except Exception:
    _zstd = None
    _HAS_ZSTD = False

# File-format magic bytes — used on cache load to dispatch
# between zstd and gzip without bumping the cache format version.
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_GZIP_MAGIC = b"\x1f\x8b"

# Low-cardinality string fields on each exam dict.  Each takes <100 unique
# values across millions of exams in real datasets, but the Excel reader
# allocates a *separate* Python string object per row.  Calling
# ``sys.intern`` on each before pickling lets pickle's memo emit one copy
# per unique value, shrinking the pickle by ~100-300 MB on 1 M exams and
# speeding ``pickle.load`` proportionally (the decoder's main cost is
# string + dict object construction).  Numeric fields are NOT interned —
# pickle handles small ints natively and floats have no intern table.
_INTERNABLE_STRING_FIELDS = (
    "modality",
    "region",
    "dose_source",
    "protocol",
    "sex_code",
    "_population",
    "description",
    "debug_description",
    "orientation_inferred",
    "orientation_detected",
    "orientation_resolved",
)

def _preprocess_cache_dir() -> str:
    """Return (creating if needed) the on-disk preprocessing-cache directory.

    When ``set_cache_output_dir`` has been called, the cache is stored
    under ``<output_dir>/.preprocess_cache``; otherwise it falls back
    to ``constants.DEFAULT_OUTPUT_DIR``.
    """
    base = _CACHE_OUTPUT_DIR or DEFAULT_OUTPUT_DIR
    d = os.path.join(base, ".preprocess_cache")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def _hash_file_content(filepath: str) -> str:
    """Return a hex hash of the file content (first 4 MB for speed)."""
    h = hashlib.blake2b(digest_size=16)
    try:
        with open(filepath, "rb") as f:
            # Read first 4 MB — enough to detect content changes while
            # keeping the hash fast for large Excel files.
            chunk = f.read(4 * 1024 * 1024)
            h.update(chunk)
    except OSError:
        h.update(filepath.encode())  # Fallback to path hash on read error
    return h.hexdigest()


def _hash_module_source(module_name: str) -> str:
    """Return a hex hash of the module's source file."""
    h = hashlib.blake2b(digest_size=16)
    try:
        import importlib.util as _iu
        spec = _iu.find_spec(module_name)
        if spec and spec.origin:
            with open(spec.origin, "rb") as f:
                h.update(f.read())
    except Exception:
        pass
    return h.hexdigest()


# Modules whose source controls the cached exam_data.  The full cache
# uses this base set and adds ``vectorized``
# (which owns ``build_exam_arrays``).  Any new dose-source module must
# be added here so an edit there invalidates the cache instead of
# yielding a false hit with stale results.
# The package name is derived rather than hard-coded:
# ``_hash_module_source`` returns a constant hash for an unresolvable
# name, so a name that went stale after a rename would silently stop
# the cache from invalidating on source edits.
_PKG = __name__.rpartition(".")[0]
_PREPROCESS_FINGERPRINT_MODULES = (
    f"{_PKG}.excel_io",              # column whitelist (usecols push-down) --
                                                      # controls which fields (e.g. Body Part
                                                      # Examined) even reach the resolver
    f"{_PKG}.preprocess",            # merge_acquisition_series, cumulative-dose / MATAAC grouping
    f"{_PKG}.exam_doses",            # organ dose dispatch
    f"{_PKG}.region_resolver",       # region inference
    f"{_PKG}.risk_models",           # cancer-to-organ / DDREF
    f"{_PKG}.projection_orientation",# projection inference
    f"{_PKG}.radiography_doses",     # radiography coefficients
    f"{_PKG}.fluoro_doses",          # fluoro coefficients
    f"{_PKG}.ct_doses",              # CT coefficients
    f"{_PKG}.dental_doses",                # dental dose mapping
    f"{_PKG}.mammography_doses",                # mammo MGD helpers
)


def _build_preprocess_fingerprint(
    version_prefix: str,
    file_list: List[str],
    use_golikov: bool,
    use_mataac: bool,
    extra_modules: Tuple[str, ...] = (),
) -> str:
    """Compute a fingerprint over Excel content + module sources + flags.

    ``version_prefix`` selects the cache's format version tag
    (``"full_v{N}|"`` for the full cache).  ``extra_modules`` lets the
    full cache add ``vectorized`` to the base module list.
    """
    h = hashlib.blake2b(digest_size=24)
    h.update(version_prefix.encode())
    for fp in sorted(file_list):
        h.update(_hash_file_content(fp).encode())
        h.update(b"|")
    for _mod in (*_PREPROCESS_FINGERPRINT_MODULES, *extra_modules):
        h.update(_hash_module_source(_mod).encode())
        h.update(b"|")
    h.update(f"golikov={int(use_golikov)}|".encode())
    h.update(f"mataac={int(use_mataac)}|".encode())
    return h.hexdigest()



# 
# Raw-Excel DataFrame cache
# 
# Caches each file's own ``pd.read_excel`` output so a cold run after a
# module edit doesn't re-pay the Calamine parse -- and, since the cache is
# keyed PER FILE (not on the whole file_list together), editing/replacing
# one file out of a multi-file cohort (e.g. one of 7 large monthly
# extracts) does not force a full re-parse of the other, unchanged
# files.  Keyed only on that file's own bytes + size + ``needed_cols`` (no
# module hashes) since a file's parsed DataFrame is a pure function of its
# own content, independent of the other files in the run.  Stored as
# pickle.gz or zstd when available.

_RAW_EXCEL_CACHE_FORMAT_VERSION = 3  # Increment when the raw Excel cache schema changes.
# Entries are keyed per file.  Entries written by earlier schema versions
# use a different filename pattern, so they are ignored (and eventually
# orphaned) rather than colliding with current entries.


def _raw_excel_cache_dir() -> str:
    """Return (creating if needed) the on-disk raw-Excel cache directory.

    When ``set_cache_output_dir`` has been called, the cache is stored
    under ``<output_dir>/.raw_excel_cache``; otherwise it falls back
    to ``constants.DEFAULT_OUTPUT_DIR``.
    """
    base = _CACHE_OUTPUT_DIR or DEFAULT_OUTPUT_DIR
    d = os.path.join(base, ".raw_excel_cache")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def _raw_excel_fingerprint_one(filepath: str, needed_cols=None) -> str:
    """Hash of ONE Excel file's content + on-disk size + column whitelist.

    No module hashes: the cached DataFrame is purely the output of
    ``pd.read_excel`` on these bytes and has no dependency on anything
    in the application code path.  Size is included so two files that
    share their first 4 MB (the content-hash window) but differ later
    still invalidate correctly.

    ``needed_cols`` is folded in because the loader pushes it down to the
    parser via ``usecols`` — a cache saved with one column whitelist must
    not be returned for a run that requests a different set, or the
    downstream organ-dose calculation would read stale / missing columns.

    Deliberately keyed on this file alone (not the whole file list it
    happens to run alongside): the parsed DataFrame is a pure function of
    this file's own bytes, so a multi-file cohort where only one file
    changes should re-parse only that file.
    """
    h = hashlib.blake2b(digest_size=24)
    h.update(f"rawxl1_v{_RAW_EXCEL_CACHE_FORMAT_VERSION}|".encode())
    h.update(_hash_file_content(filepath).encode())
    try:
        h.update(f"|sz={os.path.getsize(filepath)}".encode())
    except OSError:
        h.update(b"|sz=?")
    # Column whitelist (order-independent): None means "all columns".
    if needed_cols is None:
        h.update(b"|cols=*|")
    else:
        h.update(("|cols=" + ",".join(sorted(str(c) for c in needed_cols)) + "|").encode())
    return h.hexdigest()


def _raw_excel_cache_path_one(fingerprint: str) -> str:
    return os.path.join(
        _raw_excel_cache_dir(),
        f"raw_excel_file_v{_RAW_EXCEL_CACHE_FORMAT_VERSION}_{fingerprint}.pkl.gz",
    )


def load_raw_excel_cache_one(filepath: str, needed_cols=None):
    """Return the cached DataFrame for ONE Excel file, or ``None`` on miss.

    Any failure — missing file, corrupt payload, pickle version skew —
    is treated as a miss, never raised.  Callers fall through to a
    fresh Excel parse of that file.
    """
    try:
        path = _raw_excel_cache_path_one(_raw_excel_fingerprint_one(filepath, needed_cols))
        if not os.path.exists(path):
            return None
        blob = _decompress_from_file(path)
        payload = pickle.loads(blob)
        if not isinstance(payload, dict) or payload.get("version") != _RAW_EXCEL_CACHE_FORMAT_VERSION:
            return None
        return payload.get("df")
    except Exception:
        return None


def save_raw_excel_cache_one(filepath: str, df, needed_cols=None) -> None:
    """Persist one Excel file's parsed DataFrame, keyed on its own content.

    Writes to a ``.tmp`` sibling and atomically renames so a crashed
    save never leaves a torn file at the final path.  Silent on any
    error: the cache is strictly best-effort.
    """
    tmp = None
    try:
        path = _raw_excel_cache_path_one(_raw_excel_fingerprint_one(filepath, needed_cols))
        tmp = path + ".tmp"
        payload = {"version": _RAW_EXCEL_CACHE_FORMAT_VERSION, "df": df}
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        _compress_to_file(blob, tmp)
        os.replace(tmp, path)
    except Exception:
        # Best-effort — never break a working simulation because the
        # cache directory is read-only / full / etc.
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _compress_to_file(blob: bytes, path: str) -> None:
    """Write ``blob`` to ``path`` using zstd if available, else gzip.

    Caller is responsible for atomic rename — this writes the whole blob
    in one shot so a failure leaves the partial file at ``path`` for the
    caller's cleanup.
    """
    if _HAS_ZSTD:
        # zstd level=3: ~same ratio as gzip-6 but 3-5× faster; threads=-1 uses all cores.
        cctx = _zstd.ZstdCompressor(level=3, threads=-1)
        with open(path, "wb") as f:
            with cctx.stream_writer(f) as writer:
                writer.write(blob)
    else:
        with gzip.open(path, "wb") as f:
            f.write(blob)


def _decompress_from_file(path: str) -> bytes:
    """Decompress a cache file by sniffing its magic bytes.

    Supports both zstd-compressed and gzip-compressed payloads regardless
    of the file extension.  Raises ``ValueError`` if the format is
    unrecognised so the caller's outer try/except can treat the cache as
    corrupt and recompute.
    """
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(0)
        if head[:4] == _ZSTD_MAGIC:
            if not _HAS_ZSTD:
                # The cache was saved by a build with zstd installed but
                # this run does not have the library.  Treat as a miss so
                # the caller recomputes; the next save will fall back to
                # gzip and stay within the available codec set.
                raise ValueError("zstd cache present but zstandard not installed")
            # ``stream_reader`` is required (NOT ``decompress``): the
            # writer in ``_compress_to_file`` uses ``threads=-1`` which
            # produces a multi-frame zstd stream, and
            # ``ZstdDecompressor.decompress`` only handles single-frame
            # input.  ``stream_reader`` walks all frames transparently.
            dctx = _zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                return reader.read()
        if head[:2] == _GZIP_MAGIC:
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                return gz.read()
        raise ValueError(f"unknown cache format: magic={head[:4]!r}")


def _intern_low_cardinality_strings_inplace(exam_data: List[Dict]) -> None:
    """Run ``sys.intern`` on every low-cardinality string field of every
    exam, so pickle's memo emits one copy per unique value.

    Each field listed in ``_INTERNABLE_STRING_FIELDS`` typically has
    <100 distinct values across the entire dataset.  Without interning,
    the Excel reader produces 1 M independent string objects (one per
    row) which pickle stores as 1 M independent references — bloating
    the cache by hundreds of MB and slowing ``pickle.load`` by the
    object-construction cost.

    Interning is in-place: the exam dicts now hold references to the
    canonical interned strings.  This is safe because Python strings are
    immutable, and downstream consumers compare with ``==`` (which is
    short-circuited to ``is`` for interned strings → faster on read
    too).  Idempotent: calling again is a no-op.
    """
    intern = sys.intern
    # Only iterate over fields that are actually present in the first exam,
    # so a dataset that carries 6 of 11 internable fields does 6 × N dict
    # lookups instead of 11 × N (saving ~5 × N no-op .get → None + isinstance
    # calls, which on a 1 M-exam input is ~5 M avoided calls).
    _present = [k for k in _INTERNABLE_STRING_FIELDS
                if isinstance(exam_data[0].get(k), str)] if exam_data else []
    for ex in exam_data:
        for k in _present:
            v = ex.get(k)
            if type(v) is str:
                ex[k] = intern(v)


def _dedup_dose_dicts_inplace(exam_data: List[Dict]) -> Tuple[int, int]:
    """Replace duplicate dose-dict references in ``exam_data`` with shared ones.

    Optimisation rationale (verified against ICRP 152 Radiography 1 M-exam
    inputs): real datasets typically contain only a handful of unique
    ``organ_doses_gy`` / ``_cancer_doses`` / ``_remainder_individual_doses``
    dicts (one per modality+sex+age+dose bin), but the upstream Excel-load
    path produces a *fresh* dict per exam.  pickle's memo system only
    de-duplicates objects that are *identical Python objects*, so the
    resulting cache file contains 1 M independent copies of the same
    ~4 unique dose tables — bloating the on-disk size and forcing
    ``pickle.load`` to allocate millions of redundant Python dicts (the
    measured bottleneck: 5.5 s of 8.7 s on a 1 M-exam warm load).

    Mutating the references in place before pickling collapses those
    1 M copies into ~4 shared references, which pickle's memo then stores
    once.  After load, ``id()``-based fast paths in ``_intern_dose_dicts``
    and ``_apply_golikov_revert`` (the only mutator) take advantage of the
    sharing; ``_apply_golikov_revert`` detaches a copy on first mutation
    so per-exam factors stay independent.

    Returns ``(n_organ_unique, n_cancer_unique)`` for diagnostics.

    Notes
    -----
    * Uses ``tuple(d.items())`` as the canonical key — same approach as
      ``_intern_dose_dicts``; CPython 3.7+ insertion order is
      deterministic for content-identical dicts produced by
      ``compute_exam_organ_doses`` and ``_compute_patient_cumulative_doses``.
    * Mutates ``exam_data`` in place: idempotent on subsequent calls
      (already-shared dicts hit the same key on the first iteration).
    * Skips ``_remainder_individual_doses`` if every exam's value is
      empty (common for radiography datasets).
    """
    proxies_organ:   Dict[tuple, dict] = {}
    proxies_cancer:  Dict[tuple, dict] = {}
    proxies_rem:     Dict[tuple, dict] = {}
    proxies_modeled: Dict[tuple, list] = {}
    # Shared empty-dict for exams with no remainder doses; pickle's memo collapses all references to one.
    _shared_empty_rem: dict = {}
    for ex in exam_data:
        d = ex.get("organ_doses_gy")
        if d:
            key = tuple(d.items())
            shared = proxies_organ.get(key)
            if shared is None:
                proxies_organ[key] = d
            else:
                ex["organ_doses_gy"] = shared
        d = ex.get("_cancer_doses")
        if d:
            key = tuple(d.items())
            shared = proxies_cancer.get(key)
            if shared is None:
                proxies_cancer[key] = d
            else:
                ex["_cancer_doses"] = shared
        d = ex.get("_remainder_individual_doses")
        if d:
            # Non-empty remainder: dedupe by content like organ/cancer.
            key = tuple(d.items())
            shared = proxies_rem.get(key)
            if shared is None:
                proxies_rem[key] = d
            else:
                ex["_remainder_individual_doses"] = shared
        elif "_remainder_individual_doses" in ex:
            # Replace empty remainder bucket with the shared singleton for pickle memoisation.
            ex["_remainder_individual_doses"] = _shared_empty_rem
        # ``_modeled_types`` has very few distinct values (~50 vs N exams); sharing one list per
        # distinct content is safe (read-only downstream) and recovers the bulk of its footprint.
        m = ex.get("_modeled_types")
        if m:
            mkey = tuple(m)
            shared = proxies_modeled.get(mkey)
            if shared is None:
                proxies_modeled[mkey] = m
            else:
                ex["_modeled_types"] = shared
    return len(proxies_organ), len(proxies_cancer)


def _serialize_ct_no_dose(ct_no_dose: Dict) -> Dict:
    """Convert ``ct_no_dose`` ``patient_ids`` sets to lists for pickling."""
    return {
        k: {
            "patient_ids": list(v.get("patient_ids", [])),
            "n_exams": v["n_exams"],
            "expected_organs": v["expected_organs"],
        }
        for k, v in ct_no_dose.items()
    }


def _deserialize_ct_no_dose(blob_ct_no_dose: Dict) -> Dict:
    """Convert ``patient_ids`` lists back to sets after unpickling."""
    return {
        k: {
            "patient_ids": set(v["patient_ids"]),
            "n_exams": v["n_exams"],
            "expected_organs": v["expected_organs"],
        }
        for k, v in blob_ct_no_dose.items()
    }



# FULL CACHE (post-build_exam_arrays) — caches the complete preprocessing
# state including columnar `ea` arrays.  A warm hit skips all preprocessing
# (~11 s on 1 M exams) and goes straight to the iteration loop.

# Increment when the full (post-build_exam_arrays) cache schema changes.
# Caches written by earlier format versions may still contain dental rows
# (the non-CBCT dental-radiography exclusion runs in main.run_simulation
# before ea is built), Non-Dose rows or exams without an organ-dose model
# (dropped at source since v6), Dental CBCT rows labelled "Radiography"
# (relabeled "CBCT" at source since v7), or lack the persisted STROBE/RECORD
# participant-flow ledger / exclusion ledger, so they must be rejected.
_FULL_CACHE_FORMAT_VERSION = 9


def _preprocess_full_cache_fingerprint(
    file_list: List[str],
    use_golikov: bool,
    use_mataac: bool,
    needed_cols=None,
    merge_series: bool = True,
    population: str = "Euro-American",
) -> str:
    """Fingerprint for the post-build_exam_arrays cache.

    Adds ``vectorized`` (which owns ``build_exam_arrays``) and
    ``participant_flow`` (which owns the source-drop partitions that decide
    what reaches ``ea``) to the base module set.  Any logic change there
    invalidates the full cache.
    """
    base_fp = _build_preprocess_fingerprint(
        f"full_v{_FULL_CACHE_FORMAT_VERSION}|",
        file_list, use_golikov, use_mataac,
        extra_modules=(f"{_PKG}.vectorized", f"{_PKG}.participant_flow"),
    )
    h = hashlib.blake2b(digest_size=24)
    h.update(base_fp.encode())
    if needed_cols is None:
        h.update(b"|cols=ALL|")
    else:
        h.update(("|cols=" + ",".join(sorted(str(c) for c in needed_cols)) + "|").encode())
    h.update(f"merge={int(bool(merge_series))}|".encode())
    # Rows without a Population tag take the start-up population, so the
    # cached pop_idx / exam_data_sorted depend on it.
    h.update(f"pop={population}|".encode())
    return h.hexdigest()


def _preprocess_full_cache_path(fingerprint: str) -> str:
    """On-disk path for the full-state cache file."""
    return os.path.join(
        _preprocess_cache_dir(),
        f"preprocess_full_v{_FULL_CACHE_FORMAT_VERSION}_{fingerprint}.pkl.gz",
    )


def save_preprocess_full_cache(
    file_list: List[str],
    ea: dict,
    pop_counts: Dict[str, int],
    file_pop: List[Dict[str, int]],
    ct_no_dose: Dict,
    use_golikov: bool,
    use_mataac: bool,
    unknown_descriptions: Optional[List[Tuple[str, str]]] = None,
    n_acquisition_rows: Optional[int] = None,
    acq_rows_by_modality: Optional[Dict[str, int]] = None,
    acq_rows_by_file: Optional[Dict[str, int]] = None,
    needed_cols=None,
    merge_series: bool = True,
    excluded_no_organ_dose_by_modality: Optional[Dict[str, int]] = None,
    participant_flow: Optional[dict] = None,
    population: str = "Euro-American",
    excluded_by_modality: Optional[Dict[str, Dict[str, int]]] = None,
    excluded_dose_sv_by_modality: Optional[Dict[str, Dict[str, float]]] = None,
    excluded_cohorts: Optional[dict] = None,
) -> Optional[str]:
    """Persist the FULL post-build_exam_arrays state.

    ``ea`` is the dict returned by ``build_exam_arrays`` and contains both
    the columnar numpy arrays AND ``exam_data_sorted`` (the post-mutation
    list of exam dicts).  Pickle's memo collapses shared dose-dict
    references inside ``exam_data_sorted`` so the file size stays close
    to exam_data alone despite carrying more state.

    ``excluded_no_organ_dose_by_modality`` is the per-modality count of
    exams dropped upstream by the CT/Dental-CBCT no-organ-dose filter
    (not in the simulation at all); persisted so a warm-cache hit can
    still report the exclusion breakdown in summary section 2.

    ``participant_flow`` is the STROBE/RECORD stage ledger built by
    ``participant_flow.build_participant_flow``; persisted so the warm
    path restores it without rebuilding exam_data.

    ``excluded_by_modality`` / ``excluded_dose_sv_by_modality`` are the
    ``{exclusion key: {modality: n | Sv}}`` counters and ``excluded_cohorts``
    the ``{exclusion key: CohortByFile}`` tables of the Non-Dose /
    no-organ-dose-model exams dropped at source (participant_flow.py); the
    exams themselves are not in ``ea``, so these are the only record of them.

    Returns the cache file path on success, ``None`` on failure.  Never
    raises — a save failure must not break the simulation.
    """
    fp = _preprocess_full_cache_fingerprint(
        file_list, use_golikov, use_mataac,
        needed_cols=needed_cols, merge_series=merge_series,
        population=population,
    )
    path = _preprocess_full_cache_path(fp)
    tmp = path + ".tmp"
    try:
        ct_no_dose_serializable = _serialize_ct_no_dose(ct_no_dose)
        # Dose dicts are already shared via MappingProxyType; only string
        # interning is needed here (new fields from the cumulative-dose pass
        # may have created fresh un-interned strings).  Idempotent + cheap.
        try:
            _intern_low_cardinality_strings_inplace(ea.get("exam_data_sorted") or [])
        except Exception:
            # Interning is purely a size optimisation — never fail the
            # save over it.
            pass
        blob = {
            "format_version": _FULL_CACHE_FORMAT_VERSION,
            "file_list": file_list,
            "ea": ea,
            "pop_counts": pop_counts,
            "file_pop": file_pop,
            "ct_no_dose": ct_no_dose_serializable,
            "use_golikov": use_golikov,
            "use_mataac": use_mataac,
            "unknown_descriptions": list(unknown_descriptions or []),
            # Marker tells the loader that dose dicts are already shared
            # so the engine's ``_intern_dose_dicts`` (if it runs at all)
            # can take its id-only fast path.
            "dose_dicts_interned": True,
            "n_acquisition_rows": n_acquisition_rows,
            "acq_rows_by_modality": acq_rows_by_modality,
            "acq_rows_by_file": acq_rows_by_file,
            "merge_series": bool(merge_series),
            "needed_cols": (None if needed_cols is None else sorted(str(c) for c in needed_cols)),
            "excluded_no_organ_dose_by_modality": dict(excluded_no_organ_dose_by_modality or {}),
            "participant_flow": participant_flow,
            "excluded_by_modality": {
                str(k): dict(v or {}) for k, v in (excluded_by_modality or {}).items()},
            "excluded_dose_sv_by_modality": {
                str(k): dict(v or {}) for k, v in (excluded_dose_sv_by_modality or {}).items()},
            "excluded_cohorts": excluded_cohorts,
        }
        pickled = pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)
        _compress_to_file(pickled, tmp)
        os.replace(tmp, path)
        return path
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None


def load_preprocess_full_cache(
    file_list: List[str],
    use_golikov: bool,
    use_mataac: bool,
    needed_cols=None,
    merge_series: bool = True,
    population: str = "Euro-American",
) -> Optional[Tuple[dict, Dict[str, int], List[Dict[str, int]], Dict, List[Tuple[str, str]], Optional[int], Optional[Dict[str, int]], Optional[Dict[str, int]], Optional[Dict[str, int]], Optional[dict], Optional[dict], Optional[dict], Optional[dict]]]:
    """Load the full post-build_exam_arrays state if cache hit.

    Returns ``(ea, pop_counts, file_pop, ct_no_dose, unknown_descriptions, n_acquisition_rows, acq_rows_by_modality, acq_rows_by_file, excluded_no_organ_dose_by_modality, participant_flow, excluded_by_modality, excluded_dose_sv_by_modality, excluded_cohorts)``
    on hit, ``None`` on miss or corruption. ``n_acquisition_rows`` is the original
    acquisition series count before merge; falls back to ``None`` when absent from the cache.
    ``acq_rows_by_modality`` is the per-modality breakdown of acquisition rows
    before merge; falls back to ``None`` when absent from the cache.  ``acq_rows_by_file``
    is the analogous per-input-file breakdown, keyed by disambiguated
    basename in file order; also ``None`` when absent from the cache.
    ``excluded_no_organ_dose_by_modality`` is the per-modality count of exams
    dropped upstream by the CT/Dental-CBCT no-organ-dose filter; ``None`` for
    old caches (caller falls back to an empty dict).
    ``participant_flow`` is the STROBE/RECORD stage ledger (see
    ``participant_flow.build_participant_flow``); ``None`` when absent from the cache.
    ``excluded_by_modality`` / ``excluded_dose_sv_by_modality`` /
    ``excluded_cohorts`` are the source-drop exclusion ledger (see
    ``save_preprocess_full_cache``); ``None`` when absent.

    On a hit, the caller MUST skip ``_apply_golikov_revert``,
    ``_compute_patient_cumulative_doses``, ``_intern_dose_dicts`` AND
    ``build_exam_arrays``: every output of those four steps is already
    embedded in the returned ``ea`` dict.  The mutations they perform
    were captured at save time.
    """
    global _dose_dicts_interned_on_load
    _dose_dicts_interned_on_load = False
    fp = _preprocess_full_cache_fingerprint(
        file_list, use_golikov, use_mataac,
        needed_cols=needed_cols, merge_series=merge_series,
        population=population,
    )
    path = _preprocess_full_cache_path(fp)
    if not os.path.exists(path):
        return None
    try:
        raw = _decompress_from_file(path)
        blob = pickle.loads(raw)
        if blob.get("format_version") != _FULL_CACHE_FORMAT_VERSION:
            return None
        if blob.get("file_list") != file_list:
            return None
        if blob.get("use_golikov") != use_golikov or blob.get("use_mataac") != use_mataac:
            return None
        if blob.get("merge_series", True) != bool(merge_series):
            return None
        if blob.get("needed_cols") != (None if needed_cols is None else sorted(str(c) for c in needed_cols)):
            return None
        ct_no_dose = _deserialize_ct_no_dose(blob["ct_no_dose"])
        unknown_descriptions = [
            tuple(item) for item in blob.get("unknown_descriptions", [])
        ]
        _dose_dicts_interned_on_load = bool(blob.get("dose_dicts_interned", False))
        n_acquisition_rows = blob.get("n_acquisition_rows")  # None when absent from the cache
        acq_rows_by_modality = blob.get("acq_rows_by_modality")  # None when absent from the cache
        acq_rows_by_file = blob.get("acq_rows_by_file")  # None when absent from the cache
        excluded_no_organ_dose_by_modality = blob.get("excluded_no_organ_dose_by_modality")  # None when absent from the cache
        participant_flow = blob.get("participant_flow")  # None when absent from the cache
        return (
            blob["ea"],
            blob["pop_counts"],
            blob["file_pop"],
            ct_no_dose,
            unknown_descriptions,
            n_acquisition_rows,
            acq_rows_by_modality,
            acq_rows_by_file,
            excluded_no_organ_dose_by_modality,
            participant_flow,
            blob.get("excluded_by_modality"),
            blob.get("excluded_dose_sv_by_modality"),
            blob.get("excluded_cohorts"),
        )
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        _dose_dicts_interned_on_load = False
        return None


__all__ = [
    "save_preprocess_full_cache",
    "load_preprocess_full_cache",
    "set_cache_output_dir",
]
