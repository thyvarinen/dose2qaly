"""Multi-year batch orchestration for the Dose2QALY engine.

The setup dialog (``config.show_config_dialog``) lets the user group input
files into named year buckets, e.g. ``{"2021": [a.xlsx, b.xlsx],
"2022": [c.xlsx]}``.  This module turns those buckets into one of three
run modes:

* ``"per_year_average"`` — run every year as its own independent simulation,
  then report the mean annual burden across years plus the full per-year
  breakdown (``stats["_year_batches"]``).  This is the default.  Cumulative
  per-iteration totals (cancer cases, dose, detriment, ...) are converted to
  per-iteration rates before averaging so every year gets equal weight,
  regardless of how many MC iterations it happened to run before converging
  -- see ``_to_per_iteration_rates`` / ``_scale_rates_to_mean_iterations``.
  The Summary tab's per-iteration distributions (cancer-case count, death
  count, QALYs lost, life-years lost) are **pooled across every successful
  year** rather than averaged or taken from one year -- each year's MC
  iterations are lightweight scalars, so concatenating them costs nothing
  and lets those histograms reflect the whole batch's iteration sample
  instead of just the last year processed -- see ``_pooled_cl`` and
  friends below.  The same pooling is done **per modality** (see
  ``_per_modality_breakdown`` / ``_cancer_bd_by_modality`` /
  ``_iter_lists_by_modality``) so that filtering the Summary tab to one
  modality -- which cannot be answered from ``modality_stats`` alone, since
  that tree has no per-cancer-type breakdown -- still reflects every year
  instead of silently falling back to whichever year's raw events happen to
  be the last one alive (see ``gui._compute_modality_filtered_results``).
* ``"combined"``         — pool every file from every year into one
  population and run once (identical to the pre-batch behaviour).

Design notes
------------
* **Same return contract as** :func:`main.run_simulation` — a 5-tuple
  ``(stats, iter_cases_list, iter_deaths_list, all_events, exam_data)`` — so
  :func:`run_yearly_simulation` is a drop-in replacement anywhere
  ``run_simulation`` is passed as a callable (notably into
  ``gui.start_gui``).
* **Bounded memory across the batch.**  Only one year's heavy per-exam
  payload (``exam_data`` / event columns) is alive at a time -- the retained
  payload is dropped before the next year's files are touched -- while the
  per-year ``stats`` dicts kept for *all* years hold scalars, small
  breakdown dicts and the Research-Questions sweep.  See
  :func:`run_yearly_simulation` for the explicit reclaim steps.
  Anything the GUI must show for *every* year therefore has to be reduced to
  such a summary while that year's data is still alive.  ``_file_cohort``
  (:mod:`.file_cohort`) is the newest of these: a few kB of count tables per
  input file, computed per year and merged after the loop, which is what
  lets the Patient Cohort and By Body Region tabs cover the whole archive
  instead of only the one year whose ``exam_data`` happened to survive.
  With ``overlap_year_digests`` (the default) one extra year's payload is
  alive while its digests finish in the background, and that payload is
  released as soon as that background thread ends (joined before the next
  year is queued), so the bound becomes "at most two years' payloads
  briefly, then back to one" rather than "one".
* **The user's "year" buckets are unrelated to** ``stats["year_stats"]``,
  which the engine derives from each exam's "Date Performed" calendar year
  inside a single run.  Batch results therefore never touch ``year_stats``.
"""
from __future__ import annotations

import calendar
import gc
import os
import threading
import warnings
from collections import Counter
from typing import Optional

import numpy as np

from .columns import EventColumns
from .main import run_simulation, _disambiguated_basenames
from .vectorized import invalidate_gpu_cache, invalidate_persistent_cp_rng, MAIN_NS


# Keys that must never be blended across year batches
# Timing dicts, per-iteration arrays and GUI-side columnar caches are not
# numeric trees; ``year_stats`` is a per-run calendar-year breakdown whose
# buckets differ between years and would be meaningless once averaged.
_EXCLUDE_FROM_AVERAGE = {
    "profile", "iter_QALY_list", "iter_life_list",
    "_exam_columns", "_event_columns", "throughput", "error",
    "year_stats",
    # Not a numeric tree either: per_region_age/per_region are lists of
    # dicts and cohort_totals holds numpy arrays, so the generic
    # dict-of-numbers recursion in _average_numeric_tree would either take
    # one year's copy verbatim or misbehave.  Combined properly (across
    # every year, not just the one _average_numeric_tree would pick) by
    # research_questions.pool_region_age_sweeps() -- see the call after
    # the per-year loop below.
    "_rq_sweep_det",
    # Per-input-file cohort count tables (file_cohort.CohortByFile).  Not a
    # numeric tree, and averaging would be the wrong operation regardless:
    # these are patient head-counts answering "who was imaged?", so the whole
    # archive's age pyramid is the SUM over years, not their mean.  Merged
    # explicitly after the per-year loop below.
    "_file_cohort",
    # {exclusion key: CohortByFile} ledger of the exams dropped at source
    # (participant_flow.build_excluded_cohorts) -- same reasoning and the
    # same explicit merge as _file_cohort.  The flat per-modality counters
    # (excluded_non_dose_by_modality, ...) ARE numeric trees and go through
    # the generic per-year mean like examinations_by_modality.
    "_excluded_cohorts",
    # Per-modality per-iteration series (lists, not numeric-tree leaves) --
    # pooled explicitly after the per-year loop below, mirroring
    # iter_QALY_list/iter_life_list.  ``_cancer_bd_by_modality`` is a pure
    # numeric tree (counts only) and is instead handled generically via
    # _CUMULATIVE_NESTED_TREES below.
    "_iter_lists_by_modality",
    # Per-year patient-id aggregates (a 5-tuple, not a numeric tree):
    # _average_numeric_tree would take vals[0] -- the FIRST processed year's
    # patient table -- and silently blend it with the last year's exam_data.
    # Not pooled here either; the per-year tables are now UNIONED by
    # _cross_year_patient_totals (valid only because the batch pins one
    # pseudonymisation salt -- see run_yearly_simulation) for the
    # cross-year distinct-patient head-count, and format_summary
    # recomputes the detail-year diagnostics from the retained exam_data.
    "_pid_aggregates",
    # Per-year cached cohort baseline incidence (a dict of value /
    # n_patients / source, not a numeric tree): pooled below as a
    # patient-count-weighted mean with n_patients summed.
    "_baseline_incidence_cohort",
    # Per-year sets of distinct patient ids by modality -- not a numeric
    # tree; UNIONED by _cross_year_patients_by_modality below.
    "_patients_by_modality",
    # STROBE/RECORD participant-flow ledger (dict of stage dicts with
    # patient-id sets): not a numeric tree; UNIONED by
    # participant_flow.union_participant_flows below (valid under the
    # batch's one pinned pseudonymisation salt).
    "_participant_flow",
    # Per-region weight-adjustment counts and doses: a SUM over years
    # (analysis.merge_weight_adjustments after the per-year loop), and
    # factor_max must not be averaged.
    "_weight_adjustment",
    # Per-year row-source counts (a Counter, i.e. a dict subclass) so the
    # generic numeric leaf branch would average it into floats.  These are
    # dataset-size counts, so the right cross-year operation is a SUM --
    # done explicitly by _pool_pid_source_counts after the per-year loop.
    "_pid_source_counts",
    # Risk Analysis tab digest (analysis.build_risk_curve_digest): custom
    # duck-typed column objects, not a numeric tree.  Deliberately NOT
    # combined into one batch-wide object here -- adv_ui.py recomputes and
    # equal-weight-averages the curve on demand per filter combination (see
    # analysis.compute_batch_averaged_risk_curve), reading each year's own
    # digest straight out of ``_year_batches`` instead.
    "_risk_curve_digest",
    # Per-year stratified-breakdown digest (analysis.stratified_digest):
    # a pid-free snapshot of the sex/age/procedure/frequent-flyer
    # accumulators.  Not a numeric tree (nested dicts/lists); combined
    # explicitly by analysis.merge_stratified_digests after the per-year
    # loop, so format_summary can render sections 4/5 as true cross-year
    # averages instead of the detail year's view.
    "_strat_digest",
}

_VALID_RUN_MODES = ("per_year_average", "combined")

# Cumulative-vs-rate classification for per_year_average
# Every value below is a SUM accumulated once per completed MC iteration
# (see main._merge_partial), so a year that ran more iterations than another
# has an inflated total that has nothing to do with its underlying per-year
# risk.  Averaging these totals directly across years (a "ratio of means",
# sum(values) / sum(iterations)) implicitly weights each year by its own
# iteration count instead of giving every year equal weight, which is what
# "average across years" promises.  These keys are therefore divided by that
# year's own ``iterations`` before the cross-year average, then the resulting
# equal-weighted per-iteration rate is scaled back up by the batch's mean
# iteration count so every downstream consumer (``format_summary``, the GUI
# tabs) that still divides by ``stats["iterations"]`` recovers the correct
# equal-weighted mean.
_CUMULATIVE_TOP_LEVEL = {
    "acquisitions", "total_sv", "cancer_cases", "cancer_deaths",
    "expected_deaths", "expected_incidence", "QALY_lost", "life_lost",
    "total_detriment", "heritable_risk", "heritable_detriment",
}
# Nested trees whose every numeric leaf is likewise a per-iteration
# accumulation (see main._merge_partial's modality_stats / cancer_breakdown /
# device_stats / rem_subtype_breakdown updates).  ``year_stats`` is excluded
# on purpose -- it is already dropped from the blend entirely, see
# ``_EXCLUDE_FROM_AVERAGE``.
_CUMULATIVE_NESTED_TREES = {
    "modality_stats", "cancer_breakdown", "device_stats", "rem_subtype_breakdown",
    # {modality: {cancer_type: {"developed": n, "deaths": n}}} -- one extra
    # nesting level versus cancer_breakdown, but _scale_tree/_average_numeric_tree
    # recurse to arbitrary depth so no special-casing is needed here.
    "_cancer_bd_by_modality",
}


def _scale_tree(d: dict, factor: float) -> dict:
    """Return a copy of ``d`` with every numeric leaf multiplied by ``factor``."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _scale_tree(v, factor)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v * factor
        else:
            out[k] = v
    return out


def _to_per_iteration_rates(stats: dict) -> dict:
    """Divide one year's cumulative totals by that year's iteration count.

    Turns "total accumulated over N iterations" into "mean per iteration"
    for every key in :data:`_CUMULATIVE_TOP_LEVEL` /
    :data:`_CUMULATIVE_NESTED_TREES`, so that a subsequent equal-weight
    average across years is a true mean-of-rates rather than a
    sum(values)/sum(iterations) ratio biased toward the years that happened
    to run more MC iterations.  Everything else (dataset-level counts,
    metadata, ``iterations`` itself) passes through unchanged.
    """
    it = float(stats.get("iterations", 0) or 0)
    if it <= 0:
        return dict(stats)
    out = dict(stats)
    for k in _CUMULATIVE_TOP_LEVEL:
        v = out.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v / it
    for k in _CUMULATIVE_NESTED_TREES:
        v = out.get(k)
        if isinstance(v, dict):
            out[k] = _scale_tree(v, 1.0 / it)
    return out


def _scale_rates_to_mean_iterations(avg_stats: dict) -> dict:
    """Undo :func:`_to_per_iteration_rates` using the *averaged* iterations.

    After averaging equal-weighted per-iteration rates across years, scale
    the cumulative keys back up by the batch's mean iteration count so that
    every existing consumer -- which divides these totals by
    ``stats["iterations"]`` to display a per-iteration figure -- keeps
    working unmodified and recovers the equal-weighted mean.
    """
    # Consumers read ``int(stats["iterations"])``; scaling by the fractional
    # mean while they divide by its truncation would inflate every rate by
    # ``mean_it / int(mean_it)`` (e.g. 131,900 exams/yr showing as 131,919),
    # so the stored count and the scale factor are the same integer.
    mean_it = float(int(float(avg_stats.get("iterations", 0) or 0)))
    if mean_it <= 0:
        return avg_stats
    out = dict(avg_stats)
    out["iterations"] = int(mean_it)
    for k in _CUMULATIVE_TOP_LEVEL:
        v = out.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v * mean_it
    for k in _CUMULATIVE_NESTED_TREES:
        v = out.get(k)
        if isinstance(v, dict):
            out[k] = _scale_tree(v, mean_it)
    return out


# Partial calendar years


def _compute_year_coverage(exam_data, year: "str | int") -> float:
    """Fraction of the calendar year covered by this year's exam dates.

    ``coverage = ((max_date - min_date).days + 1) / days_in_calendar_year``,
    clipped to ``(0, 1]``.  A full calendar year of exams therefore reads
    1.0; the archive's partial 2025 (~71 % of the year) reads ~0.71.  When
    no exam carries a parseable ``Date Performed`` (all NaT) the coverage
    is taken to be 1.0 (so the year is treated as complete rather than
    silently dropped) and a ``warnings.warn`` is emitted -- this is the
    "warn via mc_warnings" surface, since the package's own warning ring
    (see ``mc_warnings``).
    """
    from .analysis import _ensure_exam_columns

    try:
        _ec = _ensure_exam_columns(exam_data)
        dates = _ec.date_performed if _ec is not None else None
    except Exception:
        dates = None
    if dates is None:
        dates = np.array([], dtype="datetime64[ns]")
    dates = np.asarray(dates, dtype="datetime64[ns]")
    valid = dates[~np.isnat(dates)] if dates.size else dates
    if valid.size == 0:
        warnings.warn(
            f"Year {year}: no parseable 'Date Performed' values; partial-year "
            f"coverage assumed to be 1.0 (full calendar year).",
            RuntimeWarning, stacklevel=2,
        )
        return 1.0
    d_min = valid.min()
    d_max = valid.max()
    try:
        cal_year = int(year)
    except (TypeError, ValueError):
        cal_year = int(np.datetime64(d_min, "Y").astype(int)) + 1970
    days_in_year = 366 if calendar.isleap(cal_year) else 365
    span = int((d_max - d_min).astype("timedelta64[D]").astype(int)) + 1
    coverage = span / float(days_in_year)
    if coverage <= 0.0:
        return 1.0
    return min(coverage, 1.0)


def _scale_rate_dict_for_partial(rate_dict: dict, factor: float) -> dict:
    """Multiply a per-iteration rate dict's extensive quantities by ``factor``.

    Scales every key in :data:`_CUMULATIVE_TOP_LEVEL` and every numeric leaf
    of :data:`_CUMULATIVE_NESTED_TREES` -- the extensive quantities a partial
    year under-contributed to.  Intensities (per-100k, mean dose, shares)
    are NOT scaled.  ``factor`` is ``1 / coverage`` for a partial year.
    """
    if factor == 1.0:
        return rate_dict
    out = dict(rate_dict)
    for k in _CUMULATIVE_TOP_LEVEL:
        v = out.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v * factor
    for k in _CUMULATIVE_NESTED_TREES:
        v = out.get(k)
        if isinstance(v, dict):
            out[k] = _scale_tree(v, factor)
    return out


def _scale_sweep_for_partial(sweep, factor: float) -> None:
    """Scale a per-year Region x Age sweep's extensive per-year cell fields.

    Multiplies ``scans_per_year`` / ``dose_sv_per_year`` /
    ``modelled_scans_per_year`` / ``modelled_dose_sv_per_year`` (the
    :data:`_PER_YEAR_CELL_KEYS`) in every
    cell of ``sweep["per_region_age"]`` and ``sweep["per_region"]`` by
    ``factor``.  Intensities (``*_per_100k``, ``mean_dose_msv``,
    ``leukq_share_*``) are left untouched -- they are already normalised by
    the cell's own scan count and must not be annualised.
    """
    if factor == 1.0 or not sweep:
        return
    from .research_questions import _PER_YEAR_CELL_KEYS

    for _key in ("per_region_age", "per_region"):
        cells = sweep.get(_key) or []
        for c in cells:
            for k in _PER_YEAR_CELL_KEYS:
                v = c.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    c[k] = v * factor
                elif isinstance(v, np.ndarray):
                    c[k] = v * factor


# Averaging

def _average_numeric_tree(dicts: list, n_years: int = None) -> dict:
    """Recursively average a list of same-shaped nested dicts of numbers.

    Works for ``modality_stats``, ``device_stats``, ``cancer_breakdown`` and
    friends without hardcoding each shape.

    Design decision: every numeric leaf is divided by the number of years in
    the batch, *not* by the number of years that happen to contain that key.
    A year with zero exams of a given modality genuinely contributed 0 to that
    modality's burden and must dilute the average accordingly; dividing by only
    the years that "had" the key would overstate the typical annual burden.
    ``n_years`` is therefore threaded through the recursion rather than
    re-derived from each sub-list, which would silently restore the wrong
    denominator for any nested key (e.g. a modality present in one year only).

    Parameters
    ----------
    dicts : list of dict
        One stats dict (or sub-dict) per year.  Years missing a sub-key are
        simply absent from the list at that nesting level.
    n_years : int, optional
        Batch size to divide by.  Defaults to ``len(dicts)`` at the top level
        and is then held constant for every nested level.

    Returns
    -------
    dict
        Same shape as the inputs, with numeric leaves averaged.
    """
    if n_years is None:
        n_years = len(dicts)
    if not dicts or n_years <= 0:
        return {}
    all_keys: set = set()
    for d in dicts:
        all_keys.update(d.keys())
    out: dict = {}
    for k in all_keys:
        if k in _EXCLUDE_FROM_AVERAGE:
            continue
        vals = [d[k] for d in dicts if k in d]
        if all(isinstance(v, dict) for v in vals):
            out[k] = _average_numeric_tree(vals, n_years=n_years)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = sum(vals) / n_years
        else:
            # Non-numeric leaf (shouldn't normally occur inside a stats tree).
            out[k] = vals[0]
    return out


def _pool_pid_source_counts(per_year_stats: dict) -> "Counter":
    """Sum every year's ``_pid_source_counts`` Counter into one batch total.

    Row-source counts (``patient_mrn`` /
    ``accession_fallback`` / ...) are dataset-size tallies answering "how
    many rows resolved their patient id each way", so the cross-year
    operation is a SUM -- mirroring the ``_file_cohort`` reasoning -- not
    the mean :func:`_average_numeric_tree` would apply (and which would
    also leak floats, since ``Counter`` is a ``dict`` subclass).  Years
    without a ``_pid_source_counts`` key contribute nothing.
    """
    out: Counter = Counter()
    for s in per_year_stats.values():
        sc = s.get("_pid_source_counts") if isinstance(s, dict) else None
        if not sc:
            continue
        for k, v in sc.items():
            out[k] = out.get(k, 0) + int(v or 0)
    return out


def _cross_year_patient_totals(per_year_stats: dict) -> "tuple[Optional[int], Optional[int]]":
    """Distinct patients across the whole batch, and how many appear in >1 year.

    Unions the per-year ``_pid_aggregates`` patient tables (valid only
    because the batch pins one pseudonymisation salt -- see
    run_yearly_simulation).  Exams with a missing/NaN id cannot be
    matched across years, so they are summed, one patient each, exactly
    as ``_build_patient_aggregates`` counts them within a year.  Returns
    ``(None, None)`` when any year lacks a usable table, so callers fall
    back to the per-year sum.
    """
    seen: dict = {}
    missing = 0
    for s in per_year_stats.values():
        if not isinstance(s, dict):
            return (None, None)
        agg = s.get("_pid_aggregates")
        if not (isinstance(agg, tuple) and len(agg) == 5):
            return (None, None)
        pid_counts, _, _, none_n, nan_n = agg
        for pid in pid_counts:
            seen[pid] = seen.get(pid, 0) + 1
        missing += int(none_n or 0) + int(nan_n or 0)
    multi = sum(1 for n in seen.values() if n > 1)
    return (len(seen) + missing, multi)


def _cross_year_patients_by_modality(per_year_stats: dict) -> "Optional[dict]":
    """Union each modality's per-year patient-id sets across the batch.

    ``stats["_patients_by_modality"]`` per year is
    ``{"sets": {modality: set}, "missing": {modality: int}}`` (see
    ``analysis.build_patients_by_modality``); unioning the sets is valid
    for the same reason as ``_cross_year_patient_totals`` -- the batch
    pins one pseudonymisation salt.  ``missing`` counts are summed (an
    unidentifiable exam cannot be matched across years).  Returns None
    when any year lacks a well-formed dict."""
    sets: dict = {}
    missing: dict = {}
    for s in per_year_stats.values():
        if not isinstance(s, dict):
            return None
        pbm = s.get("_patients_by_modality")
        if not (isinstance(pbm, dict)
                and isinstance(pbm.get("sets"), dict)
                and isinstance(pbm.get("missing"), dict)):
            return None
        for mod, ids in pbm["sets"].items():
            if not isinstance(ids, (set, frozenset)):
                return None
            sets.setdefault(mod, set()).update(ids)
        for mod, n in pbm["missing"].items():
            missing[mod] = missing.get(mod, 0) + int(n or 0)
    return {"sets": sets, "missing": missing}


def _rel_se_of_list(values: list) -> "float | None":
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


# Per-modality breakdown (for the Summary tab's modality filter)

def _per_modality_breakdown(ev, iterations: int) -> dict:
    """Split one year's cancer events into per-modality detail.

    ``gui._compute_modality_filtered_results`` needs, for a single selected
    modality: a cancer-type breakdown (for the pie chart / table) and
    per-iteration case/death/QALY/life-years-lost series (for the
    histograms).  ``stats["modality_stats"]`` has none of this -- it only
    carries modality-level scalars (dose, QALY, detriment, ...) -- so the
    GUI would otherwise have to recompute it from raw events, which in a
    ``per_year_average`` batch are only the last processed year's (see the
    module docstring).  Computing it here, once per year while that year's
    events are still alive, lets the result be pooled across the whole
    batch exactly like ``iter_QALY_list``/``iter_life_list`` are.

    Returns ``{modality: {"cancer_breakdown": {cancer_type: {"developed":
    n, "deaths": n}}, "iter_cases": [...], "iter_deaths": [...],
    "iter_qaly": [...], "iter_life": [...]}}``, the last four each of length
    ``iterations``.  Empty if there are no events or no completed
    iterations.
    """
    if ev is None or len(ev) == 0 or iterations <= 0:
        return {}
    if not isinstance(ev, EventColumns):
        ev = EventColumns.from_events(ev)
    mods = np.asarray(ev.modality)
    ctypes = np.asarray(ev.cancer_type)
    died = np.asarray(ev.died).astype(bool)
    iters = np.asarray(ev.iteration)
    qaly = np.asarray(ev.QALY_lost, dtype=float)
    life = np.asarray(ev.life_lost_years, dtype=float)

    valid_iter = (iters >= 0) & (iters < iterations)
    out: dict = {}
    for mod in sorted({m for m in mods.tolist() if m is not None}):
        mmask = (mods == mod)
        cbd: dict = {}
        for ct in {c for c in ctypes[mmask].tolist() if c is not None}:
            ctmask = mmask & (ctypes == ct)
            cbd[ct] = {
                "developed": int(ctmask.sum()),
                "deaths": int((ctmask & died).sum()),
            }
        m_valid = mmask & valid_iter
        if m_valid.any():
            bin_idx = iters[m_valid].astype(np.int64)
            iter_cases = np.bincount(bin_idx, minlength=iterations).tolist()
            iter_deaths = np.bincount(
                bin_idx, weights=died[m_valid].astype(np.int64), minlength=iterations
            ).tolist()
            iter_qaly = np.bincount(bin_idx, weights=qaly[m_valid], minlength=iterations).tolist()
            iter_life = np.bincount(bin_idx, weights=life[m_valid], minlength=iterations).tolist()
        else:
            iter_cases = [0] * iterations
            iter_deaths = [0] * iterations
            iter_qaly = [0.0] * iterations
            iter_life = [0.0] * iterations
        out[mod] = {
            "cancer_breakdown": cbd,
            "iter_cases": iter_cases, "iter_deaths": iter_deaths,
            "iter_qaly": iter_qaly, "iter_life": iter_life,
        }
    return out


def _pool_iter_lists_by_modality(per_year_stats: dict, years_sorted: list,
                                 years_to_average: "list | None" = None,
                                 partial_scaling: "dict | None" = None,
                                 common_length: "int | None" = None) -> dict:
    """Concatenate every year's per-modality per-iteration series.

    A modality absent from a given year (no exams of that type that year)
    contributes that year's iteration count worth of zeros -- the same
    "absent = genuine 0" convention as :func:`_average_numeric_tree` -- so
    the pooled distribution is not silently biased toward the years that
    happened to use a given modality.

    ``years_to_average`` / ``partial_scaling`` mirror the top-level pooled
    per-iteration lists: only years in ``years_to_average`` are pooled, and
    a "scale"d partial year's per-iteration values are multiplied by its
    ``partial_scaling`` factor (1/coverage) so the per-modality
    distributions match the annualised means.  Both default to "all years,
    no scaling" so a caller that passes nothing gets the unfiltered pooled
    behaviour.

    ``common_length`` truncates every pooled year to its first L elements
    (L = min iteration count across the pooled years) so the pooled
    distribution gives every year equal weight; ``None`` concatenates the
    full series.
    """
    if years_to_average is None:
        years_to_average = list(per_year_stats.keys())
    partial_scaling = partial_scaling or {}
    all_mods: set = set()
    for y in years_to_average:
        s = per_year_stats.get(y)
        if s is None:
            continue
        all_mods.update((s.get("_iter_lists_by_modality") or {}).keys())
    out: dict = {}
    for mod in sorted(all_mods):
        cl, dl, ql, ll = [], [], [], []
        for year in years_sorted:
            if year not in years_to_average:
                continue
            s = per_year_stats.get(year)
            if s is None:
                continue
            yiters = int(s.get("iterations", 0) or 0)
            keep = common_length if common_length is not None else yiters
            entry = (s.get("_iter_lists_by_modality") or {}).get(mod)
            factor = float(partial_scaling.get(year, 1.0) or 1.0)
            if entry is not None:
                if factor == 1.0:
                    cl.extend(entry["iter_cases"][:keep])
                    dl.extend(entry["iter_deaths"][:keep])
                    ql.extend(entry["iter_qaly"][:keep])
                    ll.extend(entry["iter_life"][:keep])
                else:
                    cl.extend([float(v) * factor for v in entry["iter_cases"][:keep]])
                    dl.extend([float(v) * factor for v in entry["iter_deaths"][:keep]])
                    ql.extend([float(v) * factor for v in entry["iter_qaly"][:keep]])
                    ll.extend([float(v) * factor for v in entry["iter_life"][:keep]])
            else:
                cl.extend([0] * keep); dl.extend([0] * keep)
                ql.extend([0.0] * keep); ll.extend([0.0] * keep)
        out[mod] = {"iter_cases": cl, "iter_deaths": dl, "iter_qaly": ql, "iter_life": ll}
    return out


def _pool_cell_iter_series(per_year_stats: dict, years_sorted: list,
                           years_to_average=None, partial_scaling=None,
                           common_length=None):
    """Concatenate every year's sparse per-cell simulated outcome series.

    ``research_questions.per_iteration_cell_series`` stores
    ``{cell_key: {"pos": int32, "cases"/"deaths"/"qaly": f32, ...}}`` per
    year; this shifts each year's ``pos`` by the running pooled-iteration
    offset — the same layout as ``_pool_iter_lists_by_modality`` and the
    top-level pooled lists (only ``years_to_average``; a "scale"d partial
    year's values are multiplied by its factor) — and concatenates the
    sparse rows.  The pooled ``pos`` then indexes into the pooled lists, so
    percentiles over the densified series describe exactly the pooled
    "single simulated year" distribution.

    A year whose digest lacks the series (``None``) is recorded in
    ``missing`` — its pooled positions then carry no events, so consumers
    must treat the store as partial (render "n/a") rather than a silent
    zero year.  A genuine empty store (``{}``, a year with no events) is
    NOT missing.  ``raw_mods`` is unioned across years.

    ``common_length`` keeps only each year's first L iterations
    (L = min iteration count across the pooled years; positions >= L are
    dropped and the offset advances by L) so the pooled series gives
    every year equal weight; ``None`` keeps the full series.

    Returns ``(store, total_pooled_length, missing_years)``.
    """
    if years_to_average is None:
        years_to_average = list(per_year_stats.keys())
    partial_scaling = partial_scaling or {}
    store: dict = {}
    missing: list = []
    offset = 0
    for year in years_sorted:
        if year not in years_to_average:
            continue
        s = per_year_stats.get(year)
        if s is None:
            continue
        yiters = int(s.get("iterations", 0) or 0)
        keep = common_length if common_length is not None else yiters
        series = s.get("_sim_cell_series")
        if series is None:
            missing.append(year)
        elif series:
            factor = float(partial_scaling.get(year, 1.0) or 1.0)
            for key, e in series.items():
                pos = np.asarray(e["pos"], dtype=np.int64)
                keep_mask = pos < keep
                if not keep_mask.any():
                    continue
                dst = store.setdefault(key, {
                    "pos": [], "cases": [], "deaths": [], "qaly": [],
                    "raw_mods": set()})
                dst["pos"].append(pos[keep_mask] + offset)
                for f in ("cases", "deaths", "qaly"):
                    v = np.asarray(e[f], dtype=np.float32)[keep_mask]
                    dst[f].append(v if factor == 1.0 else v * factor)
                dst["raw_mods"].update(e.get("raw_mods") or ())
        offset += keep
    out = {}
    for key, d in store.items():
        out[key] = {
            "pos": np.concatenate(d["pos"]).astype(np.int32),
            "cases": np.concatenate(d["cases"]).astype(np.float32),
            "deaths": np.concatenate(d["deaths"]).astype(np.float32),
            "qaly": np.concatenate(d["qaly"]).astype(np.float32),
            "raw_mods": tuple(sorted(d["raw_mods"])),
        }
    return out, offset, missing


# Orchestrator


def _compute_year_digests(stats, ed, ev, min_latency_params, on_progress,
                           year, i, n_years, n_iterations) -> None:
    """Compute one year's post-run digests, mutating ``stats`` in place.

    Extracted from :func:`run_yearly_simulation` so it can run on a background
    thread and overlap with the next year's file loading / preprocessing /
    MC run (``overlap_year_digests``).  Every block keeps its existing
    try/except so a single digest failing never aborts the year; the whole
    body is additionally wrapped in an outer guard so a background thread
    can never die with an unhandled exception, and a ``finally`` ensures the
    two keys the post-loop combiners unconditionally read are present.

    ``on_progress`` is ``None`` when called from the background thread (the
    GUI's queue.Queue-based callback is safe cross-thread, but emitting
    "year X: computing digests" status from a thread that is *supposed* to
    be invisible behind the next year's loading would just clutter the
    status line); the foreground (non-overlap) path passes it through.
    """
    try:
        # Partial-calendar-year coverage: computed now while this year's
        # exam_data (carrying ``ExamColumns.date_performed``) is still alive,
        # so the post-loop averager can annualise / exclude partial years.
        # See ``_compute_year_coverage`` and the ``partial_year_mode``
        # config keys.  Stored on the per-year stats as ``_coverage``.
        try:
            stats["_coverage"] = _compute_year_coverage(ed, year)
        except Exception:
            stats["_coverage"] = 1.0
        # Snapshot this year's unique-patient count while ``ed`` (this
        # year's exam_data) is still in memory -- it is discarded before
        # the next year loads (see the module docstring's memory note),
        # so this is the only chance to compute it.  Unlike
        # examinations_by_modality/acquisition_rows, patient identity is
        # not already a scalar on ``stats`` and needs a full pass over
        # exam_data (same logic ``format_summary`` uses for a single
        # run).  This lets the Summary tab report per_year_average
        # dataset totals (Acquisition series / Unique examinations /
        # Unique patients) as a true sum-then-average across the whole
        # batch, instead of only the last processed year.
        if callable(on_progress):
            on_progress("init", n_iterations, n_iterations,
                        f"Year {year} ({i}/{n_years}): summarising cohort counts...")
        try:
            # Reuse the patient-id aggregates run_simulation already
            # cached on stats (it builds them once while exam_data is
            # alive); only fall back to a fresh pass when that cache is
            # missing or malformed.  The 5-tuple shape mirrors what
            # _build_patient_aggregates returns.
            _agg = stats.get("_pid_aggregates")
            if isinstance(_agg, tuple) and len(_agg) == 5:
                _pid_counts, _, _, _pid_none_n, _pid_nan_n = _agg
            else:
                from .analysis import _build_patient_aggregates
                _pid_counts, _, _, _pid_none_n, _pid_nan_n = _build_patient_aggregates(ed)
            stats["_unique_patients"] = len(_pid_counts) + _pid_none_n + _pid_nan_n
        except Exception:
            stats["_unique_patients"] = 0
        # Same reasoning for the examination count: a direct count of
        # distinct A# accession numbers (see analysis._count_unique_examinations),
        # not the acquisitions/iterations quotient, so the cross-year
        # total is tied to "#A" exactly like the single-year summary.
        try:
            from .analysis import _count_unique_examinations
            stats["_unique_examinations"] = _count_unique_examinations(ed) or 0
        except Exception:
            stats["_unique_examinations"] = 0
        # Dose-weighted mean age at exposure for this year (used by the
        # Summary tab's detriment-coefficient context line).  Computed
        # now while this year's exam_data is alive; the batch average is
        # the exam-dose-weighted mean of the per-year values (weights =
        # each year's total_sv), combined after the per-year loop below.
        try:
            from .analysis import _ensure_exam_columns
            _ec = _ensure_exam_columns(ed)
            if _ec is not None and len(_ec) > 0:
                _ages = np.asarray(_ec.age, dtype=np.float64)
                _eds = np.asarray(_ec.ed_msv, dtype=np.float64)
                _valid = ~np.isnan(_ages)
                _w = _eds * _valid
                _wsum = float(_w.sum())
                stats["_dose_weighted_mean_age"] = (
                    float((_ages * _w).sum() / _wsum) if _wsum > 0 else None)
            else:
                stats["_dose_weighted_mean_age"] = None
        except Exception:
            stats["_dose_weighted_mean_age"] = None
        # Same reasoning for the Research Questions tab (the Region x
        # Age dose-reduction sweep): compute it now, with this year's
        # exam_data/cancer_events still alive.  The RQ tab ALWAYS uses the
        # analytical ICRP-152 baseline (deterministic_baseline=True), so
        # only that sweep is computed per year; pool_region_age_sweeps()
        # below then pools every year's sweep as ONE exam-weighted cohort
        # (annualised by dividing by the number of year groups) instead of
        # gui.py silently recomputing one from only the last processed
        # year.
        if callable(on_progress):
            on_progress("init", n_iterations, n_iterations,
                        f"Year {year} ({i}/{n_years}): Region x Age dose sweep (analytical ICRP 152 baseline)...")
        try:
            from .research_questions import compute_region_age_dose_sweep
            stats["_rq_sweep_det"] = compute_region_age_dose_sweep(
                stats, ed, cancer_events=ev, deterministic_baseline=True,
                min_latency_params=min_latency_params)
        except Exception:
            stats["_rq_sweep_det"] = None
        # Drop the deterministic-baseline cache entry for this year's
        # exam_data now that the sweep has consumed it: the cache is keyed
        # by id(), and once this year's exam_data is freed the next year's
        # ColumnarExamsList can reuse the address and hit a stale entry.
        try:
            from .research_questions import discard_deterministic_baseline_cache
            discard_deterministic_baseline_cache(ed, stats)
        except Exception:
            pass
        # Per-modality cancer-type breakdown + per-iteration series (see
        # _per_modality_breakdown), computed now while this year's events
        # are alive so the Summary tab's modality filter can be answered
        # for the whole batch, not just the last processed year.
        try:
            stats["_iter_lists_by_modality"] = _per_modality_breakdown(
                ev, int(stats.get("iterations", 0) or 0))
        except Exception:
            stats["_iter_lists_by_modality"] = {}
        stats["_cancer_bd_by_modality"] = {
            mod: entry["cancer_breakdown"]
            for mod, entry in stats["_iter_lists_by_modality"].items()
        }
        # Research-Questions per-cell simulated annual outcome series
        # (per (modality, region, age-decade) iteration -> cases/deaths/
        # QALYs), computed while this year's events are alive.  Pooled by
        # concatenation across years, they back the "central 95% range of
        # simulated annual outcomes" the RQ tab reports (percentiles of the
        # pooled per-iteration series -- the Summary tab's quantity).
        try:
            from .research_questions import per_iteration_cell_series
            stats["_sim_cell_series"] = per_iteration_cell_series(
                ev, int(stats.get("iterations", 0) or 0), ed)
        except Exception:
            import traceback
            print(f"[year_batch] RQ cell series failed for {year}:\n"
                  f"{traceback.format_exc()}", flush=True)
            stats["_sim_cell_series"] = None
        # Risk Analysis tab digest: just the exam/event columns
        # compute_filtered_risk_curve's "Simulation" curve needs (age,
        # sex, modality, region, dose; age-at-exposure, sex, modality,
        # region, cancer type, died, QALY_lost for the sampled-YLL
        # detriment basis), computed now while this year's
        # data is alive.  Lets adv_ui.AdvancedRiskAnalysisPanel recompute
        # and equal-weight-average that curve for ANY sex/modality/
        # region/age-range filter across every year of the batch,
        # instead of showing only the last processed year (see
        # analysis.compute_batch_averaged_risk_curve).
        if callable(on_progress):
            on_progress("init", n_iterations, n_iterations,
                        f"Year {year} ({i}/{n_years}): risk-curve and stratified digests...")
        try:
            from .analysis import build_risk_curve_digest, _ensure_exam_columns, _ensure_event_columns
            stats["_risk_curve_digest"] = build_risk_curve_digest(
                _ensure_exam_columns(ed), _ensure_event_columns(ev))
            if stats["_risk_curve_digest"] is not None:
                # The Risk Analysis batch combiner weights each year by its
                # OWN iteration count -- record it while the year's stats
                # are still alive.
                stats["_risk_curve_digest"]["iterations"] = int(
                    stats.get("iterations", 0) or 0)
        except Exception:
            import traceback
            print(f"[year_batch] risk-curve digest failed for {year}:\n"
                  f"{traceback.format_exc()}", flush=True)
            stats["_risk_curve_digest"] = None
        # Stratified-breakdown digest (sex/age/procedure/frequent-flyer):
        # a pid-free snapshot computed now while this year's exam_data
        # and events are alive, so format_summary can render sections 4/5
        # as true cross-year averages instead of the detail year's view.
        try:
            from .analysis import (
                stratified_digest, _stratified_accumulate,
                _ensure_exam_columns as _sec, _ensure_event_columns as _evc)
            _strat_acc = _stratified_accumulate(
                _sec(ed), _evc(ev), ev, None)
            stats["_strat_digest"] = stratified_digest(
                _strat_acc, int(stats.get("iterations", 0) or 0))
        except Exception:
            stats["_strat_digest"] = None
    except Exception:
        # Belt-and-braces: the per-block guards above already make this
        # near-impossible, but a background thread must never die with an
        # unhandled exception (the join would still succeed, but the year
        # would silently be missing digests).  The finally below keeps the
        # two keys the post-loop combiners unconditionally read present.
        pass
    finally:
        stats.setdefault("_iter_lists_by_modality", {})
        stats.setdefault("_cancer_bd_by_modality", {})
        stats.setdefault("_sim_cell_series", None)


def _wait_for_digest_thread(p, on_progress, n_iterations):
    """Join the background digest thread, surfacing its latest status message.

    Used by ``run_yearly_simulation._finalize`` while the main thread waits
    for the previous year's digests to finish.  The digest thread records its
    latest ``on_progress`` message on ``p["status"]`` (see the overlap-thread
    setup in ``run_yearly_simulation``); this polls it every 0.5 s and emits a
    ``Waiting for ...`` progress line that changes only when the recorded
    status changes, so the GUI shows what the digest is actually doing
    instead of a static "Waiting for year X digests to finish...".  When
    ``on_progress`` is not callable, falls back to a plain blocking join.
    """
    t = p["thread"]
    if t is None:
        return
    if not callable(on_progress):
        t.join()
        return
    year = p["year"]
    last_msg = None
    while t.is_alive():
        t.join(0.5)
        status = p.get("status") or ""
        if status:
            msg = f"Waiting for digests: {status}".rstrip()
        else:
            msg = f"Waiting for year {year} digests to finish..."
        if msg != last_msg:
            on_progress("init", n_iterations, n_iterations, msg)
            last_msg = msg


def run_yearly_simulation(
    year_groups: dict,
    n_iterations,
    min_latency_params,
    population="Euro-American",
    run_mode: str = "per_year_average",
    on_progress=None,
    risk_model: str = "REIC",
    sim_config: "dict | None" = None,
):
    """Run the simulation across one or more user-defined year groups.

    Parameters
    ----------
    year_groups : dict[str, list[str]]
        e.g. ``{"2021": [...], "2022": [...]}``.  Insertion order is not
        significant; years are always processed sorted so progress messages
        and the retained "last year" are reproducible.
    n_iterations : int
        Hard maximum MC years per simulated year group.
    min_latency_params : dict
        Passed straight through to :func:`main.run_simulation`.
    population : str
        Default population when the input data does not specify one.
    run_mode : {"per_year_average", "combined"}
        See the module docstring.  Default ``"per_year_average"``.
    on_progress : callable, optional
        GUI callback, forwarded unchanged to the engine.
    risk_model : {"REIC", "LAR"}
        Risk integral type.
    sim_config : dict, optional
        Engine settings; a shallow copy is taken so per-batch overrides do not
        leak back to the caller.

    Returns
    -------
    tuple
        ``(stats, iter_cases_list, iter_deaths_list, all_events, exam_data)``,
        matching :func:`main.run_simulation`.
    """
    if run_mode != "per_year_average":
        return _run_yearly_simulation_impl(
            year_groups, n_iterations, min_latency_params, population,
            run_mode, on_progress, risk_model, sim_config,
        )
    # Pin one pseudonymisation salt for the whole per_year_average batch so a
    # patient's token is identical in every year's run and the per-year
    # ``_pid_aggregates`` tables can be unioned into a cross-year
    # distinct-patient count (see _cross_year_patient_totals).  Unpinned in
    # ``finally`` so later single/combined runs in this process go back to
    # fresh random salts.
    from .exam_doses import pin_patient_id_salt
    pin_patient_id_salt(os.urandom(16))
    try:
        return _run_yearly_simulation_impl(
            year_groups, n_iterations, min_latency_params, population,
            run_mode, on_progress, risk_model, sim_config,
        )
    finally:
        pin_patient_id_salt(None)


def _run_yearly_simulation_impl(
    year_groups: dict,
    n_iterations,
    min_latency_params,
    population="Euro-American",
    run_mode: str = "per_year_average",
    on_progress=None,
    risk_model: str = "REIC",
    sim_config: "dict | None" = None,
):
    """Implementation of :func:`run_yearly_simulation` (see its docstring)."""
    sim_config = dict(sim_config or {})
    year_groups = {str(y): list(files or []) for y, files in (year_groups or {}).items()}

    if run_mode == "combined":
        # De-duplicate: the same file may legitimately sit in two buckets.
        all_files = list(dict.fromkeys(
            f for files in year_groups.values() for f in files
        ))
        stats, cl, dl, ev, ed = _run_one(
            all_files, n_iterations, min_latency_params,
            population, on_progress, risk_model, sim_config,
        )
        if isinstance(stats, dict):
            stats["_run_mode"] = "combined"
            stats["_detail_files"] = list(all_files)
            # One pooled run means the cohort profiles all carry the same
            # (empty) year label; re-file them under the bucket each file
            # actually came from so the year-group entries in the GUI's
            # source dropdown work in this mode too.
            try:
                from .file_cohort import assign_year_groups
                assign_year_groups(stats.get("_file_cohort"), year_groups)
                for _xc in (stats.get("_excluded_cohorts") or {}).values():
                    assign_year_groups(_xc, year_groups)
            except Exception:
                pass
            # Unlike per_year_average, "combined" pools every year group's
            # files into ONE simulation, so there is no per-year run to sum
            # -- the totals already computed above already are the true
            # totals.  Recording how many year groups contributed lets
            # format_summary show a "(avg .../year)" alongside each total
            # (dividing by this count), the same courtesy per_year_average
            # gets, without pretending combined mode ran anything per-year.
            stats["_n_years"] = len([y for y, files in year_groups.items() if files]) or 1
            # Map disambiguated basename -> year label so the Summary tab's
            # "by input file" breakdown can show year-bin names instead of
            # file names.  Disambiguation must match main.run_simulation's
            # (_disambiguated_basenames on the same all_files list).
            _labels = _disambiguated_basenames(all_files)
            _f2y: dict = {}
            for yr, files in year_groups.items():
                for f in (files or []):
                    _f2y[f] = str(yr)
            stats["_file_to_year"] = {
                _labels[i]: _f2y.get(all_files[i], str(all_files[i]))
                for i in range(len(all_files))
            }
        return stats, cl, dl, ev, ed

    if run_mode != "per_year_average":
        raise ValueError(
            f"Unknown run_mode: {run_mode!r} (expected one of {_VALID_RUN_MODES})"
        )

    # per_year_average
    years_sorted = sorted(y for y, files in year_groups.items() if files)
    n_years = len(years_sorted)
    if n_years == 0:
        return {"error": "No input files in any year"}, [], [], [], []

    # Bypass the full preprocess cache: the batch pins one pseudonymisation
    # salt across every year (see run_yearly_simulation), so a cached year
    # would carry tokens written under a different salt and break the
    # cross-year patient union.
    sim_config["skip_preprocess_cache"] = True
    # Run each finished year's post-run digest block on a background thread so
    # it overlaps with the NEXT year's file loading/preprocessing/MC run,
    # instead of blocking between years.  Disabled only by explicitly setting
    # ``overlap_year_digests: False`` in the config dict.
    overlap = bool(sim_config.get("overlap_year_digests", True))

    if callable(on_progress):
        on_progress("init", n_iterations, n_iterations,
                    "Per-year batch: preprocess cache bypassed so patient IDs stay comparable across years")

    per_year_stats: dict = {}
    failed_years: dict = {}
    last_year = None
    last_ev, last_ed = [], []
    # Per-iteration distributions, kept as per-year segments (one scalar per
    # completed MC iteration, so the whole batch's segments are negligible in
    # memory) and pooled AFTER the partial-year decision below -- so a partial
    # year under "scale" is annualised in the pooled distribution that feeds
    # the 2.5-97.5 percentile "95% range of simulated years", and an
    # "exclude"d year is not pooled at all.  See the module docstring.
    pooled_segments: dict = {}

    # ``pending`` holds the previous year's (year, stats, cl, dl, ev, ed,
    # thread) awaiting finalisation.  It is finalised (thread joined, stats
    # filed into per_year_stats, per-iteration series pooled) at the top of
    # the next iteration -- AFTER the next year's _run_one has already
    # started, which is what makes the digests overlap with the next load.
    pending = None   # type: ignore[var-annotated]

    def _finalize(p):
        if p["thread"] is not None:
            _wait_for_digest_thread(p, on_progress, n_iterations)
        per_year_stats[p["year"]] = p["stats"]
        pooled_segments[p["year"]] = {
            "cl": list(p["cl"] or []),
            "dl": list(p["dl"] or []),
            "qaly": list(p["stats"].get("iter_QALY_list") or []),
            "life": list(p["stats"].get("iter_life_list") or []),
        }

    for i, year in enumerate(years_sorted, start=1):
        files = year_groups[year]
        if callable(on_progress):
            # Reuses the existing ("init", total, total, message) call shape
            # already emitted by main.run_simulation, so no GUI-side handler
            # needs to learn a new one.
            on_progress("init", n_iterations, n_iterations,
                        f"Starting year {year} ({i}/{n_years})...")

        # Run the next year while the previous year's digests (if any) are
        # still finishing on the background thread.  _run_one's own
        # preprocessing uses a ProcessPoolExecutor, so the main thread mostly
        # waits and the digest thread gets the GIL for its Python work.
        stats, cl, dl, ev, ed = _run_one(
            files, n_iterations, min_latency_params, population,
            _make_progress_forwarder(on_progress, year, i, n_years),
            risk_model, sim_config, year_group=year,
        )

        # Finalise the previous year now that this year's run has started
        # (the overlap window is the time between _run_one returning for year
        # N-1 and this point for year N).  Its payload is then unreferenced.
        if pending is not None:
            _finalize(pending)
            pending = None
            gc.collect()

        if not isinstance(stats, dict) or "error" in stats:
            msg = stats.get("error", "unknown error") if isinstance(stats, dict) \
                else "engine returned no stats"
            failed_years[year] = msg
        else:
            # Snapshot THIS year's active latency on the main thread before
            # the next run_simulation reconfigures it: the background digest
            # thread reads _LATENCY_PARAMS via expected_onset_pmf, and
            # configure_simulation does _LATENCY_PARAMS.clear() then repopulate,
            # so a thread that runs during that window would silently use the
            # wrong latency.  The dict copy is taken while the main thread
            # still holds the post-_run_one (i.e. post-configure_simulation)
            # state, which is exactly the latency this year was simulated with.
            from .risk_models import _LATENCY_PARAMS
            _lat = dict(_LATENCY_PARAMS)
            p = {"year": year, "stats": stats, "cl": cl, "dl": dl,
                 "ev": ev, "ed": ed, "thread": None, "status": ""}
            if overlap and i < n_years:
                # Overlap: run this year's digests on a background thread
                # while the next year's _run_one loads/preprocesses/runs.
                # The digest's on_progress messages are recorded on
                # p["status"] (not emitted directly) so they do not
                # interleave with the next year's; _finalize surfaces them
                # while the main thread is waiting for this thread to join.
                if callable(on_progress):
                    on_progress("init", n_iterations, n_iterations,
                                f"Year {year} ({i}/{n_years}): computing digests in the background while the next year loads...")
                t = threading.Thread(
                    target=_compute_year_digests,
                    args=(stats, ed, ev, _lat,
                          lambda _phase, _d, _t, m, _p=p: _p.__setitem__("status", str(m)),
                          year, i, n_years, n_iterations, sim_config),
                    name=f"year-digests-{year}", daemon=True)
                # Only the thread holds the heavy per-exam payload; the main
                # thread drops its references so they are released when the
                # thread finishes.  The per-exam detail is only retained for
                # the LAST year (it belongs to the last processed year),
                # which is handled by the pending/finalize
                # logic below, not by keeping ev/ed alive here.
                p["ev"] = p["ed"] = None
                t.start()
                p["thread"] = t
            else:
                # Last year, or overlap disabled: run digests inline so the
                # status messages flow to the GUI and the payload is consumed
                # before the loop moves on.
                _compute_year_digests(stats, ed, ev, _lat, on_progress,
                                      year, i, n_years, n_iterations)
            pending = p

        # Reclaim RAM/VRAM now instead of waiting for the next
        # run_simulation() call's lazy rebuild.  This is also a *correctness*
        # requirement, not just hygiene: main.run_simulation only calls
        # invalidate_gpu_cache() when it has to rebuild the exam arrays, so a
        # year that hits the full-preprocess disk cache would otherwise find
        # _GPUCache.ready still True and reuse the previous year's
        # device-side exam arrays.
        del stats, cl, dl, ev, ed
        if callable(on_progress):
            on_progress("init", n_iterations, n_iterations,
                        f"Year {year} ({i}/{n_years}): releasing memory before next year...")
        invalidate_gpu_cache(ns=MAIN_NS)
        invalidate_persistent_cp_rng(ns=MAIN_NS)
        gc.collect()

    if pending is not None:
        _finalize(pending)
        last_year = pending["year"]
        last_ev, last_ed = pending["ev"], pending["ed"]
        pending = None

    if not per_year_stats:
        return ({"error": "All years failed: " + "; ".join(
            f"{y}: {msg}" for y, msg in failed_years.items())},
            [], [], [], [])

    # Failed years are excluded from the batch rather than counted as
    # zero-burden years: they represent missing data, not a quiet year.  So the
    # averaging denominator (and ``_n_years``) is the number of years that
    # actually produced results; ``_n_years_attempted`` keeps the difference
    # visible.
    # Partial calendar years
    # A year whose exam dates cover only part of the calendar year (e.g. the
    # archive's 2025 covers ~71%) would be averaged in as a full year and bias
    # every annual total down.  ``partial_year_mode`` (config.DEFAULTS) decides
    # how a year with coverage < ``partial_year_min_coverage`` is treated:
    #   "scale"   - annualise: multiply that year's extensive quantities by
    #               1/coverage before the cross-year average;
    #   "exclude" - drop it from the annual averages and sweeps (kept in
    #               ``_year_batches`` detail, recorded below);
    #   "asis"    - leave it untouched (no annualisation or exclusion).
    _partial_mode = str(sim_config.get("partial_year_mode", "scale") or "scale").lower()
    _partial_min = float(sim_config.get("partial_year_min_coverage", 0.95) or 0.95)
    _year_coverage = {y: float(s.get("_coverage", 1.0) or 1.0)
                      for y, s in per_year_stats.items()}
    _partial_scaling: dict = {}      # {year: factor} for scaled years
    _partial_excluded: list = []     # [year, ...] for excluded years
    _years_to_average = list(per_year_stats.keys())
    if _partial_mode == "exclude":
        _years_to_average = [
            y for y in _years_to_average
            if _year_coverage.get(y, 1.0) >= _partial_min
        ]
        _partial_excluded = [
            y for y in per_year_stats.keys()
            if _year_coverage.get(y, 1.0) < _partial_min
        ]
    # Convert each year's cumulative totals to per-iteration rates before
    # averaging, so a year that happened to run more MC iterations does not
    # get more weight in the cross-year mean -- see _to_per_iteration_rates.
    # Partial-year "scale" annualisation is applied here (one place): the
    # extensive per-iteration rates and the per-year sweep cells' extensive
    # fields are multiplied by 1/coverage so the partial year contributes a
    # full-year-equivalent to the cross-year mean.
    rate_dicts = []
    for _y in _years_to_average:
        _s = per_year_stats[_y]
        _rd = _to_per_iteration_rates(_s)
        _cov = _year_coverage.get(_y, 1.0)
        if _partial_mode == "scale" and _cov < _partial_min and _cov > 0.0:
            _factor = 1.0 / _cov
            _rd = _scale_rate_dict_for_partial(_rd, _factor)
            # Scale this year's analytical sweep in place (it is read again
            # below for the cross-year pooled sweep and kept on _year_batches).
            _scale_sweep_for_partial(_s.get("_rq_sweep_det"), _factor)
            _partial_scaling[_y] = _factor
        rate_dicts.append(_rd)
    avg_stats = _average_numeric_tree(rate_dicts)
    avg_stats = _scale_rates_to_mean_iterations(avg_stats)
    avg_stats["_run_mode"] = "per_year_average"
    avg_stats["_year_batches"] = per_year_stats       # full per-year detail
    avg_stats["_year_batch_failures"] = failed_years  # {} when none failed
    avg_stats["_n_years"] = len(_years_to_average)
    avg_stats["_n_years_attempted"] = n_years
    avg_stats["_detail_year"] = last_year
    avg_stats["_year_coverage"] = _year_coverage
    avg_stats["_partial_year_scaling"] = _partial_scaling
    avg_stats["_partial_years_excluded"] = _partial_excluded
    # Achieved-precision report per year: iterations, coverage, partial-year
    # handling, the relative SE of each year's per-iteration annual
    # outcome (cases/deaths/QALYs/life-years), and the convergence
    # metadata.  One row per year (including excluded ones, flagged
    # ``excluded``) so the supplementary precision table covers the whole
    # batch.  rel_se is std(ddof=1)/sqrt(n)/mean over the per-iteration
    # list, None when n<2 or mean<=0.
    _precision_by_year: dict = {}
    for _y in sorted(per_year_stats.keys(), key=str):
        _s = per_year_stats[_y]
        _seg = pooled_segments.get(_y, {})
        _conv = _s.get("_convergence") or {}
        _prec = {
            "iterations": int(_s.get("iterations", 0) or 0),
            "coverage": float(_s.get("_coverage", 1.0) or 1.0),
            "scale_factor": float(_partial_scaling.get(_y, 1.0) or 1.0),
            "excluded": _y not in _years_to_average,
            "rel_se_cases": _rel_se_of_list(_seg.get("cl")),
            "rel_se_deaths": _rel_se_of_list(_seg.get("dl")),
            "rel_se_qaly": _rel_se_of_list(_seg.get("qaly")),
            "rel_se_life": _rel_se_of_list(_seg.get("life")),
        }
        _prec.update({
            "stop_reason": _conv.get("stop_reason"),
            "min_iterations": _conv.get("min_iterations"),
            "max_iterations": _conv.get("max_iterations"),
            "stability_threshold": _conv.get("stability_threshold"),
            "stability_window": _conv.get("stability_window"),
            "precision_target_rel_se": _conv.get("precision_target_rel_se"),
        })
        _precision_by_year[_y] = _prec
    avg_stats["_precision_by_year"] = _precision_by_year
    # Combine every successful year's analytical Research Questions sweep
    # (computed above, per year, while that year's exam_data was still
    # alive) into ONE exam-weighted pooled cohort, annualised by dividing
    # by the number of year groups -- see pool_region_age_sweeps() and the
    # module note at its definition.  The per-year sweeps are also kept on
    # ``_year_batches`` so the GUI can re-pool a modality subset on demand
    # (see research_questions.filter_region_age_sweep); this is just the
    # unfiltered "All modalities" fast path.  Excluded partial years are
    # dropped from the pooled sweep too (still kept on ``_year_batches``
    # for the detail view).
    try:
        from .research_questions import pool_region_age_sweeps
        _sweeps = [per_year_stats[y].get("_rq_sweep_det")
                   for y in _years_to_average
                   if per_year_stats[y].get("_rq_sweep_det")]
        avg_stats["_rq_sweep_det_avg"] = (
            pool_region_age_sweeps(_sweeps) if _sweeps else None)
    except Exception:
        avg_stats["_rq_sweep_det_avg"] = None
    # Combine every successful year's per-input-file cohort profiles into one
    # archive-wide set.  This is what makes the Patient Cohort and By Body
    # Region tabs honest in this mode: they draw on exam_data, which is
    # only the last processed year's, so without these profiles "All files
    # (combined)" would show one year and quietly drop the rest.  The profiles are
    # counts, so merging is addition and the combined pyramid is the true
    # whole-archive one -- see file_cohort.py.
    try:
        from .file_cohort import merge_file_cohorts
        avg_stats["_file_cohort"] = merge_file_cohorts(
            s.get("_file_cohort") for s in per_year_stats.values()
        )
    except Exception:
        avg_stats["_file_cohort"] = None
    # Same for the exclusion ledger: counts of the exams dropped at source,
    # so the By Body Region tab's "Excluded examinations" covers every year.
    try:
        from .participant_flow import merge_excluded_cohorts
        avg_stats["_excluded_cohorts"] = merge_excluded_cohorts(
            s.get("_excluded_cohorts") for s in per_year_stats.values()
        )
    except Exception:
        avg_stats["_excluded_cohorts"] = None

    # Sum every year's row-source counts into one batch total.  These are
    # dataset-size tallies (not rates), so the cross-year operation is a
    # sum -- see _pool_pid_source_counts.  ``_pid_aggregates`` is NOT put on
    # avg_stats at all: it is a per-year patient table that cannot be
    # averaged or summed meaningfully.  It CAN now be UNIONED for the
    # cross-year distinct-patient head-count (valid only because the batch
    # pins one pseudonymisation salt -- see run_yearly_simulation), which
    # is exactly what ``_unique_patients_total`` below is; it is still not
    # averaged, and format_summary recomputes the detail-year diagnostics
    # from the retained ``exam_data`` instead.
    avg_stats["_pid_source_counts"] = _pool_pid_source_counts(per_year_stats)
    from .analysis import merge_weight_adjustments
    avg_stats["_weight_adjustment"] = merge_weight_adjustments(
        ys.get("_weight_adjustment") for ys in per_year_stats.values())

    # Pool the per-year cached cohort baseline incidence as a
    # patient-count-weighted mean (n_patients summed), so the combined
    # Summary resolves the same cohort-weighted value the single-year runs
    # produced.
    _bl_num = 0.0
    _bl_n = 0
    _bl_source = "icrp152"
    for _s in per_year_stats.values():
        _b = _s.get("_baseline_incidence_cohort")
        if not _b:
            continue
        _w = int(_b.get("n_patients") or 0)
        if _w <= 0:
            continue
        _bl_num += float(_b["value"]) * _w
        _bl_n += _w
        _bl_source = _b.get("source", _bl_source)
    avg_stats["_baseline_incidence_cohort"] = (
        {"value": _bl_num / _bl_n, "n_patients": _bl_n, "source": _bl_source}
        if _bl_n > 0 else None)
    avg_stats["_unique_patients_total"], avg_stats["_patients_multi_year"] = \
        _cross_year_patient_totals(per_year_stats)
    avg_stats["_patients_by_modality"] = _cross_year_patients_by_modality(
        per_year_stats)

    # Union the per-year STROBE/RECORD participant-flow ledgers (exams
    # summed, patient ids unioned under the batch's pinned salt, missing-id
    # counts summed).  None when any year's ledger is missing.
    try:
        from .participant_flow import union_participant_flows
        avg_stats["_participant_flow"] = union_participant_flows(
            {y: s.get("_participant_flow") for y, s in per_year_stats.items()})
    except Exception:
        avg_stats["_participant_flow"] = None

    # Merge every year's stratified-breakdown digest into one cross-year
    # average (see analysis.merge_stratified_digests).  Lets format_summary
    # render sections 4/5 as true per-year means instead of the detail year's
    # view, even when the live GUI calls it with exam_data=None.
    try:
        from .analysis import merge_stratified_digests
        avg_stats["_strat_digest_avg"] = merge_stratified_digests(
            [s.get("_strat_digest") for s in per_year_stats.values()
             if s.get("_strat_digest")])
    except Exception:
        avg_stats["_strat_digest_avg"] = None

    # Dose-weighted mean age at exposure, averaged across years: the
    # exam-dose-weighted mean of each year's own ``_dose_weighted_mean_age``
    # (weights = that year's total_sv), so a year that delivered more dose
    # counts more in the cohort-wide mean age.  None when no year produced
    # a usable value (or all weights are zero).
    _dwma_num = 0.0
    _dwma_den = 0.0
    for _s in per_year_stats.values():
        _a = _s.get("_dose_weighted_mean_age")
        if _a is None:
            continue
        _w = float(_s.get("total_sv", 0.0) or 0.0)
        if _w > 0:
            _dwma_num += float(_a) * _w
            _dwma_den += _w
    avg_stats["_dose_weighted_mean_age"] = (
        _dwma_num / _dwma_den if _dwma_den > 0 else None)

    # The retained per-exam payload (last_ev/last_ed) belongs to the last
    # *processed* year only, so the file list its ``_file_index`` values
    # reference is that year's bucket -- not the flattened all-files list the
    # GUI receives.  Carry it on the stats so anything working from raw
    # exam_data can map file indices to names.  Charts that must cover the
    # whole batch use ``_file_cohort`` above instead.
    avg_stats["_detail_files"] = list(year_groups[last_year]) if last_year else []

    # Per-iteration distributions are pooled across the years that survive
    # the partial-year decision (see the module docstring and
    # ``pooled_segments`` above), with a "scale"d partial year annualised
    # (each per-iteration value multiplied by 1/coverage) and an "exclude"d
    # year dropped entirely -- so the 2.5-97.5 percentile "95% range of
    # simulated years" matches the annualised/excluded means.  cl/dl become
    # floats when scaled; np.percentile and the histogram plots accept them.
    pooled_cl: list = []
    pooled_dl: list = []
    pooled_qaly: list = []
    pooled_life: list = []
    _pooled_iter_scaling: dict = {}
    # Common pooling length: a raw concatenation would weight each year by
    # its own iteration count (an MC-convergence detail, not signal), so
    # the pooled mean would differ from the equal-weight annual mean that
    # _CUMULATIVE_TOP_LEVEL produces for section 6.  Every surviving year
    # therefore contributes only its FIRST L iterations, where L is the
    # smallest iteration count across the pooled years -- the per-iteration
    # draws are iid, so a prefix is an unbiased subsample.
    _pooled_iter_counts: dict = {}
    for _y in _years_to_average:
        _s = per_year_stats.get(_y)
        if _s is None or pooled_segments.get(_y) is None:
            continue
        _pooled_iter_counts[_y] = int(_s.get("iterations", 0) or 0)
    _common_L = min(_pooled_iter_counts.values()) if _pooled_iter_counts else 0
    for _y in _years_to_average:
        _seg = pooled_segments.get(_y)
        if _seg is None:
            continue
        _factor = float(_partial_scaling.get(_y, 1.0) or 1.0)
        _pooled_iter_scaling[_y] = _factor
        if _factor == 1.0:
            pooled_cl.extend(_seg["cl"][:_common_L])
            pooled_dl.extend(_seg["dl"][:_common_L])
            pooled_qaly.extend(_seg["qaly"][:_common_L])
            pooled_life.extend(_seg["life"][:_common_L])
        else:
            pooled_cl.extend([float(v) * _factor for v in _seg["cl"][:_common_L]])
            pooled_dl.extend([float(v) * _factor for v in _seg["dl"][:_common_L]])
            pooled_qaly.extend([float(v) * _factor for v in _seg["qaly"][:_common_L]])
            pooled_life.extend([float(v) * _factor for v in _seg["life"][:_common_L]])
    avg_stats["_pooled_iter_scaling"] = _pooled_iter_scaling
    avg_stats["_pooled_common_length"] = _common_L
    avg_stats["_pooled_iter_counts"] = _pooled_iter_counts
    avg_stats["iter_QALY_list"] = pooled_qaly
    avg_stats["iter_life_list"] = pooled_life

    # Per-modality per-iteration series, pooled the same way (see
    # _pool_iter_lists_by_modality).  The cancer-type breakdown counterpart,
    # ``_cancer_bd_by_modality``, is a pure numeric tree and was already
    # averaged above via _CUMULATIVE_NESTED_TREES.
    avg_stats["_iter_lists_by_modality"] = _pool_iter_lists_by_modality(
        per_year_stats, years_sorted,
        years_to_average=_years_to_average, partial_scaling=_partial_scaling,
        common_length=_common_L)

    # Per-cell simulated annual outcome series (Research Questions tab):
    # pooled the same way so the "central 95% range of simulated annual
    # outcomes" covers every processed year.  ``_sim_cell_series_missing``
    # lists years whose digest lacked the series — consumers render the
    # affected ranges as "n/a" rather than reading those years as zero.
    (_cell_store, _cell_len, _cell_missing) = _pool_cell_iter_series(
        per_year_stats, years_sorted,
        years_to_average=_years_to_average, partial_scaling=_partial_scaling,
        common_length=_common_L)
    avg_stats["_sim_cell_series"] = _cell_store
    avg_stats["_sim_cell_series_len"] = _cell_len
    avg_stats["_sim_cell_series_missing"] = _cell_missing

    return avg_stats, pooled_cl, pooled_dl, last_ev, last_ed


def _make_progress_forwarder(on_progress, year: str, index: int, n_years: int):
    """Return a callback that forwards engine progress, tagged with the year.

    The engine's ``on_progress(stage, *args)`` argument shapes differ per
    stage, so nothing is reordered or dropped here.  For stages that carry a
    human-readable detail string the string is prefixed with the year label
    (same argument count, same positions) so the GUI status line shows which
    year of the batch is running.
    """
    if not callable(on_progress) or n_years <= 1:
        return on_progress

    tag = f"[{year} {index}/{n_years}] "

    def _forward(phase, *args):
        if phase in ("init", "precompute", "prepare_iterations", "file_load"):
            detail_idx = 3
        elif phase == "iteration":
            detail_idx = 5
        else:
            detail_idx = None
        if detail_idx is not None and len(args) > detail_idx and isinstance(args[detail_idx], str):
            args = list(args)
            args[detail_idx] = tag + args[detail_idx]
            args = tuple(args)
        return on_progress(phase, *args)

    return _forward


def _run_one(files, n_iterations, min_latency_params, population,
             on_progress, risk_model, sim_config, year_group=None):
    """Run one simulation and normalise the result to a 5-tuple.

    ``main.run_simulation`` returns a 4-tuple on some early-failure paths
    (e.g. ``{"error": "No data"}, [], [], []``), so the length is padded here
    to keep every caller's unpacking safe.

    ``year_group`` is threaded through ``sim_config`` purely as a label: it
    tags this run's per-input-file cohort profiles (see file_cohort.py) with
    the year bucket they came from, so the GUI can offer "just 2021" next to
    the individual files.  It has no effect on the simulation itself.
    """
    if not files:
        return {"error": "No files for this year"}, [], [], [], []
    if year_group is not None:
        sim_config = {**(sim_config or {}), "_year_group": str(year_group)}
    result = run_simulation(
        files, n_iterations, min_latency_params, population,
        on_progress=on_progress, risk_model=risk_model, sim_config=sim_config,
    )
    if not isinstance(result, tuple):
        return result, [], [], [], []
    parts = list(result[:5])
    while len(parts) < 5:
        parts.append([])
    return tuple(parts)
