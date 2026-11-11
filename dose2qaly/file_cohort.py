"""Per-input-file cohort aggregates that survive a multi-year batch.

Why this exists
---------------
``year_batch.run_yearly_simulation`` in ``per_year_average`` mode drops each
year's ``exam_data`` before loading the next one, so by the time the GUI draws
the Patient Cohort / By Body Region tabs only the *last processed year's*
exams are still in memory.  Those tabs' "All files (combined)" option was
therefore a lie: it combined one year's files and silently discarded every
other year, and the per-file dropdown could only offer that same year's files.

Keeping every year's ``exam_data`` alive would throw away the whole point of
the batch's bounded-memory design (a year of a real archive is hundreds of MB
of per-exam detail).  So each run instead contributes a *cohort profile*: a
few small integer count tables per input file, computed once while that year's
exam_data is still alive.  Every chart on those tabs is a function of counts
over the same handful of categorical cross-products, so the profiles are a
lossless substitute for them -- and they merge by plain addition, which is
exactly what "combine the files" means.

Cost: the dominant table is age x sex x modality, i.e.
``(21 age bins) x (3 sexes) x n_modalities`` int64 = ~5 kB for ten
modalities.  A 20-file, 5-year archive therefore costs on the order of 100 kB
in place of the many GB its ``exam_data`` occupied, and nothing has to be
recomputed when the user switches files in the dropdown.

Identity
--------
Files are keyed by their normalised absolute path, **not** by the
``_file_index`` stored on each exam.  Those indices are only meaningful
relative to the file list of the run that produced them, so two years of a
batch both use index 0 for entirely different files; merging on them would
silently add unrelated cohorts together.

Age binning
-----------
``N_AGE_BINS`` five-year bins covering 0-99 (the last one labelled "95+"),
matching ``analysis.plot_age_histogram``, plus one extra row
(``UNKNOWN_AGE_BIN``) for exams whose age is missing or negative.  The pyramid
ignores that row -- it has no bar to sit in -- while the sex pie and modality
mix include it, which reproduces exactly what those three charts do today when
handed raw exam data.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

AGE_BIN_WIDTH = 5
N_AGE_BINS = 20                 # 0-4, 5-9, ..., 95+
UNKNOWN_AGE_BIN = N_AGE_BINS    # extra row: age missing / negative
N_AGE_ROWS = N_AGE_BINS + 1

# Column order of the sex axis in every ``age_sex_mod`` table.
SEX_ORDER = ("M", "F", "U")
N_SEX = len(SEX_ORDER)

# NOTE: dental subtypes (Dental Panoramic, Dental_intraoral, Kefalometry,
# Dental CBCT, …) are NOT collapsed here — they pass through as-is so the
# By Body Region plot can break the dental panel down by subtype.  The MC
# stats path (vectorized.py) still collapses them to "Dental imaging" for
# modality_stats/region_stats, which is a separate display.


def file_key(path) -> str:
    """Return the identity a file is merged on across runs.

    ``normcase`` matters on Windows, where the same file routinely reaches us
    as both ``D:\\Data\\2021.xlsx`` and ``d:/data/2021.xlsx`` depending on
    whether it came from the config dialog or a recent-files list.
    """
    p = str(path or "")
    try:
        return os.path.normcase(os.path.abspath(p))
    except Exception:
        return os.path.normcase(p)


class FileCohort:
    """Count tables describing one input file's exams.

    Attributes
    ----------
    age_sex_mod : dict[str, np.ndarray]
        ``{modality: int64 array of shape (N_AGE_ROWS, N_SEX)}``.  Row
        ``UNKNOWN_AGE_BIN`` holds exams with no usable age.
    mod_region : dict[str, dict[str, int]]
        ``{modality: {reporting_region: count}}`` -- the same shape and the
        same region collapsing as :func:`analysis.region_by_modality`.
    population : dict[str, int]
        ``{population_tag: count}``, for the sex pie's cohort caption.
    pop_age_sex_mod : dict[str, dict[str, np.ndarray]]
        ``{population_tag: {modality: int64 array of shape
        (N_AGE_ROWS, N_SEX)}}`` -- the *joint* population x modality x age x
        sex table the per-population age pyramids need.  ``age_sex_mod`` is
        its sum over populations and ``population`` its marginal; neither of
        those carries which population each age-bin count belongs to.
    n_exams : int
        Total exams, i.e. the sum of every ``age_sex_mod`` table.
    region_debug : dict[tuple, int]
        ``{(modality, raw_region, description, inferred, detected, chosen):
        count}`` -- the per-description breakdown the By Body Region tab's
        text box shows.  Mergeable by addition across files/years so the
        text can cover the whole archive, not just the last processed year.
    """

    __slots__ = ("age_sex_mod", "mod_region", "population", "pop_age_sex_mod",
                 "n_exams", "region_debug")

    def __init__(self, age_sex_mod=None, mod_region=None, population=None,
                 pop_age_sex_mod=None, n_exams: int = 0, region_debug=None):
        self.age_sex_mod: Dict[str, np.ndarray] = age_sex_mod or {}
        self.mod_region: Dict[str, Dict[str, int]] = mod_region or {}
        self.population: Dict[str, int] = population or {}
        self.pop_age_sex_mod: Dict[str, Dict[str, np.ndarray]] = \
            pop_age_sex_mod or {}
        self.n_exams = int(n_exams)
        self.region_debug: Dict[tuple, int] = region_debug or {}

    # merging
    def copy(self) -> "FileCohort":
        """Deep-ish copy: a new FileCohort whose per-modality/per-population
        arrays and dicts are independent copies, so merging into one does
        not mutate the other."""
        return FileCohort(
            age_sex_mod={m: a.copy() for m, a in self.age_sex_mod.items()},
            mod_region={m: dict(r) for m, r in self.mod_region.items()},
            population=dict(self.population),
            pop_age_sex_mod={p: {m: a.copy() for m, a in mods.items()}
                             for p, mods in self.pop_age_sex_mod.items()},
            n_exams=self.n_exams,
            region_debug=dict(self.region_debug),
        )

    def add(self, other: "FileCohort") -> "FileCohort":
        """Accumulate ``other`` into this cohort in place and return self."""
        if other is None:
            return self
        for mod, arr in other.age_sex_mod.items():
            cur = self.age_sex_mod.get(mod)
            if cur is None:
                self.age_sex_mod[mod] = arr.copy()
            else:
                cur += arr
        for mod, regions in other.mod_region.items():
            cur_r = self.mod_region.setdefault(mod, {})
            for reg, cnt in regions.items():
                cur_r[reg] = cur_r.get(reg, 0) + cnt
        for pop, cnt in other.population.items():
            self.population[pop] = self.population.get(pop, 0) + cnt
        for pop, mods in other.pop_age_sex_mod.items():
            cur_p = self.pop_age_sex_mod.setdefault(pop, {})
            for mod, arr in mods.items():
                cur = cur_p.get(mod)
                if cur is None:
                    cur_p[mod] = arr.copy()
                else:
                    cur += arr
        for key, cnt in other.region_debug.items():
            self.region_debug[key] = self.region_debug.get(key, 0) + cnt
        self.n_exams += other.n_exams
        return self

    # derived views used by the charts
    def modalities(self) -> List[str]:
        """Modality names present, most exams first (ties broken by name)."""
        totals = {m: int(a.sum()) for m, a in self.age_sex_mod.items()}
        return sorted(totals, key=lambda m: (-totals[m], m))

    def filtered(self, modality: Optional[str] = None) -> "FileCohort":
        """Return a copy restricted to a single modality (None = unchanged)."""
        if modality is None:
            return self
        out = FileCohort()
        arr = self.age_sex_mod.get(modality)
        if arr is not None:
            out.age_sex_mod[modality] = arr.copy()
            out.n_exams = int(arr.sum())
        regions = self.mod_region.get(modality)
        if regions:
            out.mod_region[modality] = dict(regions)
        # Filter region_debug to the selected modality.
        for key, cnt in self.region_debug.items():
            if key[0] == modality:
                out.region_debug[key] = cnt
        # Population is not tracked per modality; scale-free captions would be
        # wrong either way, so report the whole file's mix rather than
        # inventing a per-modality one.  The joint table, by contrast, CAN be
        # filtered exactly (it has a modality axis), so keep only the selected
        # modality's table inside each population.
        out.population = dict(self.population)
        for pop, mods in self.pop_age_sex_mod.items():
            arr = mods.get(modality)
            if arr is not None:
                out.pop_age_sex_mod.setdefault(pop, {})[modality] = arr.copy()
        return out


    def mod_region_dict(self) -> Dict[str, Dict[str, int]]:
        """Return ``{modality: {region: count}}`` — same shape as
        :func:`analysis.region_by_modality` — directly from the count
        tables, with no per-exam expansion.
        """
        return {m: dict(r) for m, r in self.mod_region.items()}

    def to_compact_exam_columns(self):
        """Build a minimal :class:`~columns.ExamColumns` from the count
        tables, for the Patient Cohort charts (age pyramid, sex pie,
        modality mix).

        Only ``age``, ``sex_code``, ``modality`` and ``population`` are
        populated — the count tables carry the age x sex x modality joint
        distribution and the population marginal, but not region jointly
        with age.  ``region`` is filled with ``"Unknown"`` and dose/patient
        fields are zeroed, which is harmless because those charts never
        read them.  Use :meth:`mod_region_dict` directly for the By Body
        Region tab instead.

        Ages are set to each bin's *start* (e.g. 60 for 60-64); the pyramid
        only reads bin membership via ``age // bin_width``, so this is
        exact.  Unknown-age exams get ``-1``, which the pyramid filters out
        — matching its existing behaviour on raw data.

        This is a **transient** allocation for the currently selected view
        (one file, one year group, or combined), not a retained copy of
        raw data.  Its size is bounded by the selected view's exam count,
        not the whole archive's.
        """
        from .columns import ExamColumns

        n = self.n_exams
        _empty = ExamColumns(
            age=np.zeros(0, dtype=np.float64),
            sex_code=np.empty(0, dtype="U1"),
            modality=np.empty(0, dtype=object),
            region=np.empty(0, dtype=object),
            protocol=np.empty(0, dtype=object),
            ed_msv=np.zeros(0, dtype=np.float64),
            patient_id=np.empty(0, dtype=object),
            total_patient_dose_sv=np.zeros(0, dtype=np.float64),
            population=np.empty(0, dtype=object),
            mataac_frailty_factor=np.ones(0, dtype=np.float64),
        )
        if n == 0:
            return _empty

        # The joint pop x modality x age x sex table, when present and
        # consistent, lets ``population`` be emitted in lockstep with the
        # age/sex/modality chunks below -- so a per-population age pyramid
        # is meaningful.  The ``population`` marginal alone cannot do that:
        # padding it alongside the age/sex chunks would attach each exam to
        # an arbitrary population tag.  Cohorts without the joint table
        # keep the marginal path.
        joint_total = sum(int(a.sum())
                          for mods in self.pop_age_sex_mod.values()
                          for a in mods.values())
        use_joint = bool(self.pop_age_sex_mod) and joint_total == n

        mod_chunks, age_chunks, sex_chunks, pop_chunks = [], [], [], []

        def _emit_table(mod, arr, pop=None):
            for ai in range(N_AGE_ROWS):
                row = arr[ai]
                for si, sex in enumerate(SEX_ORDER):
                    c = int(row[si])
                    if c <= 0:
                        continue
                    mod_chunks.append(np.full(c, mod, dtype=object))
                    if pop is not None:
                        pop_chunks.append(np.full(c, pop, dtype=object))
                    if ai == UNKNOWN_AGE_BIN:
                        age_chunks.append(np.full(c, -1.0, dtype=np.float64))
                    else:
                        age_chunks.append(
                            np.full(c, float(ai * AGE_BIN_WIDTH), dtype=np.float64))
                    sex_chunks.append(np.full(c, sex, dtype="U1"))

        if use_joint:
            for pop, mods in self.pop_age_sex_mod.items():
                for mod, arr in mods.items():
                    _emit_table(mod, arr, pop)
        else:
            for mod, arr in self.age_sex_mod.items():
                _emit_table(mod, arr)

        if mod_chunks:
            modality = np.concatenate(mod_chunks)
            age = np.concatenate(age_chunks)
            sex_code = np.concatenate(sex_chunks)
        else:
            return _empty

        # Population marginal — the sex pie reads it for the cohort caption.
        if use_joint:
            population = np.concatenate(pop_chunks)
        else:
            pop_chunks = []
            for pop, cnt in self.population.items():
                if cnt > 0:
                    pop_chunks.append(np.full(cnt, pop, dtype=object))
            if pop_chunks:
                population = np.concatenate(pop_chunks)
                # Pad/truncate to match n (should already match, but guard anyway).
                if len(population) < n:
                    population = np.concatenate([
                        population,
                        np.full(n - len(population), "Euro-American", dtype=object),
                    ])
                elif len(population) > n:
                    population = population[:n]
            else:
                population = np.full(n, "Euro-American", dtype=object)

        return ExamColumns(
            age=age,
            sex_code=sex_code,
            modality=modality,
            region=np.full(n, "Unknown", dtype=object),
            protocol=np.full(n, "", dtype=object),
            ed_msv=np.zeros(n, dtype=np.float64),
            patient_id=np.full(n, "", dtype=object),
            total_patient_dose_sv=np.zeros(n, dtype=np.float64),
            population=population,
            mataac_frailty_factor=np.ones(n, dtype=np.float64),
        )


class CohortByFile:
    """Every input file's :class:`FileCohort`, indexed by year group too.

    The primary store is keyed by ``(year group, file)`` rather than by file
    alone.  The config dialog lets the same file sit in two year buckets, and
    ``per_year_average`` then genuinely simulates it twice -- once in each
    year -- so a file-keyed store would have to choose between reporting that
    file's exams once (wrong for the year totals) or twice (wrong for the
    file's own row).  Keeping the cells separate lets every view be exact:

    * :meth:`for_file`  sums a file's cells across the groups it appears in;
    * :meth:`for_group` sums one group's cells;
    * :meth:`combined`  sums every cell, so it always equals the sum of the
      individual file rows *and* the sum of the year-group rows.

    ``files`` and ``group_order`` preserve first-seen order so the GUI lists
    files and years the way the user supplied them, across the whole batch
    rather than within one year.
    """

    __slots__ = ("files", "labels", "group_order", "cells")

    def __init__(self):
        self.files: List[str] = []                          # merge keys, first-seen order
        self.labels: Dict[str, str] = {}                    # merge key -> display path
        self.group_order: List[str] = []                    # year-group labels, first-seen
        self.cells: Dict[tuple, FileCohort] = {}            # (group, key) -> profile

    # construction
    def _slot(self, path, group: str = "") -> FileCohort:
        key = file_key(path)
        if key not in self.labels:
            self.files.append(key)
            self.labels[key] = str(path)
        if group and group not in self.group_order:
            self.group_order.append(group)
        cell = self.cells.get((group, key))
        if cell is None:
            cell = FileCohort()
            self.cells[(group, key)] = cell
        return cell

    def merge(self, other: "CohortByFile") -> "CohortByFile":
        """Accumulate another run's profiles into this one, in place."""
        if other is None:
            return self
        for (group, key), cell in other.cells.items():
            self._slot(other.labels.get(key, key), group).add(cell)
        return self

    # queries
    def combined(self) -> FileCohort:
        """Sum every cell: the whole archive, every file of every year."""
        out = FileCohort()
        for cell in self.cells.values():
            out.add(cell)
        return out

    def for_file(self, key: str) -> FileCohort:
        """Sum one file's cells across every year group it appears in."""
        out = FileCohort()
        for (_group, k), cell in self.cells.items():
            if k == key:
                out.add(cell)
        return out

    def for_group(self, group_label: str) -> FileCohort:
        """Sum one year group's files."""
        out = FileCohort()
        for (group, _k), cell in self.cells.items():
            if group == group_label:
                out.add(cell)
        return out


    def display_labels(self) -> Dict[str, str]:
        """``{merge key: unique human-readable label}``.

        Basenames, with a ``" (2)"``, ``" (3)"`` ... suffix when two files in
        different directories share one -- the same disambiguation
        ``main._disambiguated_basenames`` applies to the Summary tab, so the
        two per-file listings name the same file the same way.
        """
        out: Dict[str, str] = {}
        seen: Dict[str, int] = {}
        for key in self.files:
            path = self.labels.get(key, key)
            base = os.path.basename(str(path)) or str(path)
            n = seen.get(base, 0) + 1
            seen[base] = n
            out[key] = base if n == 1 else f"{base} ({n})"
        return out

    def __len__(self) -> int:
        return len(self.files)

    def __bool__(self) -> bool:
        return bool(self.files)

    # GUI selector support
    def selector_choices(self) -> List[tuple]:
        """Return ``[(label, kind, key), ...]`` for a source dropdown.

        ``kind`` is ``"all"``, ``"group"`` or ``"file"``.  ``key`` is the
        merge key for ``"file"`` entries, the group label for ``"group"``
        entries, and ``None`` for ``"all"``.  The GUI prepends its own
        "All files (combined)" entry (which maps to ``kind="all"``) so this
        method returns only the year-group and per-file entries, in the
        order the user supplied them.
        """
        labels = self.display_labels()
        choices: List[tuple] = []
        # Year groups first, in first-seen order.
        for group in self.group_order:
            choices.append((f"Year: {group}", "group", group))
        # Then individual files, in first-seen order.
        for key in self.files:
            choices.append((labels.get(key, key), "file", key))
        return choices

    def select(self, kind: str, key=None) -> FileCohort:
        """Resolve a selector choice to a :class:`FileCohort`.

        * ``kind="all"``   → :meth:`combined`
        * ``kind="group"`` → :meth:`for_group` (``key`` = group label)
        * ``kind="file"``  → :meth:`for_file`  (``key`` = merge key)
        """
        if kind == "all":
            return self.combined()
        if kind == "group":
            return self.for_group(key or "")
        if kind == "file":
            return self.for_file(key or "")
        return self.combined()


# building from exam data

def _exam_arrays(exam_data):
    """Extract the columns the profiles need, whatever the container.

    Handles the three exam representations that reach this module:
    ``ExamColumns`` (GUI cache), ``ColumnarExamsList`` (the engine's own
    dict-of-lists, which is what ``run_simulation`` returns) and a plain list
    of exam dicts.  Returns ``None`` when there is nothing to count.

    Returns a 10-tuple:
    ``(ages, sexes, mods, regs, pops, fidx,
       desc, orient_inf, orient_det, orient_cho)`` where the last four are
    the per-exam debug fields used by the By Body Region text box.
    ``ExamColumns`` does not carry those fields, so they come back as empty
    strings (the region_debug table is then empty, which is harmless).
    """
    n = len(exam_data) if exam_data is not None else 0
    if n == 0:
        return None

    if hasattr(exam_data, "age") and hasattr(exam_data, "sex_code"):
        # ExamColumns — no debug fields, fill with empties.
        _empty = np.full(n, "", dtype=object)
        return (
            np.asarray(exam_data.age, dtype=np.float64),
            np.asarray(exam_data.sex_code, dtype="U1"),
            np.asarray(exam_data.modality, dtype=object),
            np.asarray(exam_data.region, dtype=object),
            np.asarray(exam_data.population, dtype=object),
            np.asarray(exam_data.file_index, dtype=np.int64),
            _empty, _empty, _empty, _empty,
        )

    cols = getattr(exam_data, "_cols", None)
    if cols is not None:
        # ColumnarExamsList
        def _col(name, default):
            v = cols.get(name)
            return v if v is not None else [default] * n

        return (
            np.asarray([float(a or 0.0) for a in _col("age", 0.0)], dtype=np.float64),
            np.asarray([str(s or "U")[:1].upper() for s in _col("sex_code", "U")], dtype="U1"),
            np.asarray([m or "Unknown" for m in _col("modality", "Unknown")], dtype=object),
            np.asarray([r or "Unknown" for r in _col("region", "Unknown")], dtype=object),
            np.asarray([p or "Euro-American" for p in _col("_population", "Euro-American")], dtype=object),
            np.asarray([int(f or 0) for f in _col("_file_index", 0)], dtype=np.int64),
            np.asarray([str(d or "") for d in _col("debug_description", "")], dtype=object),
            np.asarray([str(v or "") for v in _col("orientation_inferred", "")], dtype=object),
            np.asarray([str(v or "") for v in _col("orientation_detected", "")], dtype=object),
            np.asarray([str(v or "") for v in _col("orientation_resolved", "")], dtype=object),
        )

    # List of dicts.
    ages = np.empty(n, dtype=np.float64)
    sexes = np.empty(n, dtype="U1")
    mods = np.empty(n, dtype=object)
    regs = np.empty(n, dtype=object)
    pops = np.empty(n, dtype=object)
    fidx = np.zeros(n, dtype=np.int64)
    desc = np.empty(n, dtype=object)
    oinf = np.empty(n, dtype=object)
    odet = np.empty(n, dtype=object)
    ocho = np.empty(n, dtype=object)
    for i, ex in enumerate(exam_data):
        try:
            ages[i] = float(ex.get("age", 0) or 0)
        except (TypeError, ValueError):
            ages[i] = -1.0
        sx = str(ex.get("sex_code") or ex.get("sex") or "U").strip().upper()[:1]
        sexes[i] = sx if sx in ("M", "F") else "U"
        mods[i] = ex.get("modality") or ex.get("Acquisition Modality") or "Unknown"
        regs[i] = ex.get("region") or "Unknown"
        pops[i] = ex.get("_population") or ex.get("population") or "Euro-American"
        _fi = ex.get("_file_index")
        fidx[i] = int(_fi) if _fi is not None else 0
        desc[i] = str(ex.get("debug_description", "") or ex.get("protocol", "") or ex.get("description", "") or "[no description]").strip()
        oinf[i] = str(ex.get("orientation_inferred", "") or "").strip()
        odet[i] = str(ex.get("orientation_detected", "") or "").strip()
        ocho[i] = str(ex.get("orientation_resolved", "") or "").strip()
    return ages, sexes, mods, regs, pops, fidx, desc, oinf, odet, ocho


def _reporting_regions(regions: np.ndarray) -> np.ndarray:
    """Collapse raw regions to reporting regions, one lookup per *unique* name.

    ``radiography_reporting_region`` is a non-trivial string routine and a
    real archive has millions of exams over a few hundred distinct region
    names, so it is called once per name rather than once per exam.
    """
    from .region_resolver import radiography_reporting_region as _rpt_reg

    uniq, inverse = np.unique(regions.astype(str), return_inverse=True)
    mapped = np.array(
        [_rpt_reg(u) for u in uniq],
        dtype=object,
    )
    return mapped[inverse]


def build_file_cohort(exam_data, file_list, group_label: Optional[str] = None) -> CohortByFile:
    """Summarise ``exam_data`` into one :class:`FileCohort` per input file.

    Parameters
    ----------
    exam_data
        ``ExamColumns``, ``ColumnarExamsList`` or a list of exam dicts.  Each
        exam's ``_file_index`` selects its entry in ``file_list``.
    file_list : list of str
        The paths ``_file_index`` indexes into -- i.e. the file list of the
        run that produced ``exam_data``, not the GUI's flattened all-years
        list.  Exams whose index falls outside it are dropped rather than
        misattributed.
    group_label : str, optional
        The year-group name this run belongs to, recorded so the GUI can
        offer "show me just 2021" alongside the individual files.

    Notes
    -----
    Everything is one pass of ``np.bincount`` over integer codes, so the cost
    is linear in the exam count with no Python-level loop over exams.
    """
    out = CohortByFile()
    paths = [str(p) for p in (file_list or [])]
    group = "" if group_label is None else str(group_label)
    # Register every input file up front, so a file that contributed no exams
    # at all (everything filtered out) is still listed rather than vanishing.
    for p in paths:
        out._slot(p, group)

    arrays = _exam_arrays(exam_data)
    if arrays is None or not paths:
        return out
    ages, sexes, mods, regs, pops, fidx, desc, oinf, odet, ocho = arrays

    n_files = len(paths)
    keep = (fidx >= 0) & (fidx < n_files)
    if not keep.all():
        ages, sexes, mods, regs, pops, fidx, desc, oinf, odet, ocho = (
            ages[keep], sexes[keep], mods[keep], regs[keep], pops[keep], fidx[keep],
            desc[keep], oinf[keep], odet[keep], ocho[keep],
        )
    if fidx.size == 0:
        return out

    # Age row: five-year bins, everything at or above the top bin folded into
    # it ("95+"), anything unusable parked in the unknown row.
    with np.errstate(invalid="ignore"):
        valid_age = np.isfinite(ages) & (ages >= 0)
    age_row = np.where(
        valid_age,
        np.clip((np.nan_to_num(ages, nan=0.0) // AGE_BIN_WIDTH).astype(np.int64),
                0, N_AGE_BINS - 1),
        UNKNOWN_AGE_BIN,
    )

    sex_col = np.full(fidx.shape, 2, dtype=np.int64)   # default "U"
    sex_col[sexes == "M"] = 0
    sex_col[sexes == "F"] = 1

    mod_names, mod_code = np.unique(mods.astype(str), return_inverse=True)
    n_mod = len(mod_names)

    # age x sex x modality, per file, in one bincount.
    flat = ((fidx * n_mod + mod_code) * N_AGE_ROWS + age_row) * N_SEX + sex_col
    counts = np.bincount(flat, minlength=n_files * n_mod * N_AGE_ROWS * N_SEX)
    counts = counts.reshape(n_files, n_mod, N_AGE_ROWS, N_SEX).astype(np.int64)

    # modality x reporting region, per file.
    reg_names, reg_code = np.unique(_reporting_regions(regs).astype(str), return_inverse=True)
    n_reg = len(reg_names)
    reg_flat = (fidx * n_mod + mod_code) * n_reg + reg_code
    reg_counts = np.bincount(reg_flat, minlength=n_files * n_mod * n_reg)
    reg_counts = reg_counts.reshape(n_files, n_mod, n_reg)

    # population tag, per file.
    pop_names, pop_code = np.unique(pops.astype(str), return_inverse=True)
    n_pop = len(pop_names)
    pop_counts = np.bincount(fidx * n_pop + pop_code,
                             minlength=n_files * n_pop).reshape(n_files, n_pop)

    # population x modality x age x sex, per file -- the *joint* table the
    # per-population age pyramids need (``pop_counts`` above is only a
    # marginal over the same exams).
    joint_flat = (((fidx * n_pop + pop_code) * n_mod + mod_code)
                  * N_AGE_ROWS + age_row) * N_SEX + sex_col
    joint_counts = np.bincount(
        joint_flat, minlength=n_files * n_pop * n_mod * N_AGE_ROWS * N_SEX)
    joint_counts = joint_counts.reshape(
        n_files, n_pop, n_mod, N_AGE_ROWS, N_SEX).astype(np.int64)

    # region_debug: per-file counts of (modality, raw_region, description,
    # orientation_inferred, orientation_detected, orientation_resolved) tuples.
    # These are what the By Body Region text box shows.  Encoded as a single
    # integer code via np.unique on a 2-D array, then bincounted per file.
    raw_reg_names, raw_reg_code = np.unique(regs.astype(str), return_inverse=True)
    desc_names, desc_code = np.unique(desc.astype(str), return_inverse=True)
    inf_names, inf_code = np.unique(oinf.astype(str), return_inverse=True)
    det_names, det_code = np.unique(odet.astype(str), return_inverse=True)
    cho_names, cho_code = np.unique(ocho.astype(str), return_inverse=True)
    # Combine the 6 codes into one via row-wise unique.
    _combo = np.stack([mod_code, raw_reg_code, desc_code,
                       inf_code, det_code, cho_code], axis=1)
    _combo_rows, combo_inverse = np.unique(_combo, axis=0, return_inverse=True)
    n_combo = len(_combo_rows)
    combo_flat = fidx * n_combo + combo_inverse
    combo_counts = np.bincount(combo_flat, minlength=n_files * n_combo)
    combo_counts = combo_counts.reshape(n_files, n_combo)

    for fi, path in enumerate(paths):
        fc = out._slot(path, group)
        mod_totals = counts[fi].sum(axis=(1, 2))
        for mi in np.nonzero(mod_totals)[0]:
            fc.age_sex_mod[str(mod_names[mi])] = counts[fi, mi].copy()
            nz = np.nonzero(reg_counts[fi, mi])[0]
            if nz.size:
                fc.mod_region[str(mod_names[mi])] = {
                    str(reg_names[ri]): int(reg_counts[fi, mi, ri]) for ri in nz
                }
        fc.n_exams = int(mod_totals.sum())
        for pi in np.nonzero(pop_counts[fi])[0]:
            fc.population[str(pop_names[pi])] = int(pop_counts[fi, pi])
            per_mod = joint_counts[fi, pi]
            nz_mods = np.nonzero(per_mod.sum(axis=(1, 2)))[0]
            if nz_mods.size:
                fc.pop_age_sex_mod[str(pop_names[pi])] = {
                    str(mod_names[mi]): per_mod[mi].copy() for mi in nz_mods
                }
        # region_debug for this file
        nz = np.nonzero(combo_counts[fi])[0]
        if nz.size:
            fc.region_debug = {
                (
                    str(mod_names[_combo_rows[ui, 0]]),
                    str(raw_reg_names[_combo_rows[ui, 1]]),
                    str(desc_names[_combo_rows[ui, 2]]),
                    str(inf_names[_combo_rows[ui, 3]]),
                    str(det_names[_combo_rows[ui, 4]]),
                    str(cho_names[_combo_rows[ui, 5]]),
                ): int(combo_counts[fi, ui])
                for ui in nz
            }

    return out


def assign_year_groups(cohort: Optional[CohortByFile], year_groups: dict) -> Optional[CohortByFile]:
    """Re-file ungrouped cells under the year group each file belongs to.

    ``run_mode="combined"`` pools every bucket's files into a single
    simulation, so ``build_file_cohort`` has only one group label to give
    them all.  The bucket membership is still known to the caller, though, so
    the year-group dropdown can be restored afterwards.

    A file listed in two buckets is attributed to the first one: combined mode
    de-duplicates the file list, so its exams were loaded exactly once and
    counting them under both years would inflate the archive total.
    """
    if cohort is None or not year_groups:
        return cohort
    owner: Dict[str, str] = {}
    for group, files in year_groups.items():
        for path in (files or []):
            owner.setdefault(file_key(path), str(group))
    if not owner:
        return cohort

    regrouped: Dict[tuple, FileCohort] = {}
    order: List[str] = []
    for (group, key), cell in cohort.cells.items():
        new_group = group or owner.get(key, "")
        if new_group and new_group not in order:
            order.append(new_group)
        existing = regrouped.get((new_group, key))
        if existing is None:
            regrouped[(new_group, key)] = cell
        else:
            existing.add(cell)
    cohort.cells = regrouped
    cohort.group_order = order
    return cohort


def merge_file_cohorts(cohorts) -> Optional[CohortByFile]:
    """Combine several runs' :class:`CohortByFile` into one (or None)."""
    out: Optional[CohortByFile] = None
    for c in cohorts or ():
        if c is None:
            continue
        if out is None:
            out = CohortByFile()
        out.merge(c)
    return out
