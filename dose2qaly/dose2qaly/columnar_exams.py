"""Columnar exam data structure for zero-copy preprocessing pipeline.

Replaces list-of-dicts with dict-of-lists to eliminate ~40 ms of dict.get
overhead in build_exam_arrays when processing 250k exams.

The columnar layout allows build_exam_arrays to extract entire columns with
a single dict lookup instead of N dict.get calls per field.
"""
from __future__ import annotations

from typing import Any, Dict, List



# All fields returned by compute_exam_organ_doses
_ALL_EXAM_FIELDS = {
    "modality",
    "region",
    "reporting_region",
    "sex_code",
    "protocol",
    "description",
    "debug_description",
    "orientation_inferred",
    "orientation_detected",
    "orientation_resolved",
    "organ_doses_gy",
    "ed_msv",
    "ed_msv_no_golikov",
    "dose_source",
    "_golikov_factor",
    "_golikov_no_weight",
    "_golikov_eligible",
    "_dose_quantity_missing",
    "_cancer_doses",
    "_modeled_types",
    "_gonad_dose_gy",
    "_patient_id",
    "_person_id",
    "_person_id_source",
    "_exam_date",
    "_remainder_individual_doses",
    "_population",
    "device",
    "equipment",
    "date_performed",
    "_file_index",  # added by _process_single_row / _compute_exam_batch
}


class ExamProxy:
    """Lightweight, mutable proxy behaving like an exam dictionary.
    
    Acts as a view into ColumnarExamsList at a specific index. Replaces
    standard Python dictionaries to avoid massive object-overhead.
    """
    __slots__ = ("_list", "_idx")
    
    def __init__(self, col_list: ColumnarExamsList, idx: int):
        self._list = col_list
        self._idx = idx
        
    def get(self, key: str, default: Any = None) -> Any:
        """Return this row's value for column ``key``, or ``default`` when
        the column is absent or this row's entry is ``None``."""
        cols = self._list._cols
        if key not in cols:
            return default
        val = cols[key][self._idx]
        return val if val is not None else default
        
    def __getitem__(self, key: str) -> Any:
        cols = self._list._cols
        if key not in cols:
            raise KeyError(key)
        val = cols[key][self._idx]
        if val is None:
            raise KeyError(key)
        return val
        
    def __setitem__(self, key: str, value: Any) -> None:
        cols = self._list._cols
        if key not in cols:
            # Dynamically add the column to support post-processed annotations
            cols[key] = [None] * self._list._N
        cols[key][self._idx] = value
        # Any mutation invalidates the cached ExamColumns (see columns.py).
        # Plain assignment to a declared slot always works even if the slot
        # was never set (e.g. an old pickle unpickled into this class), so no
        # try/except is needed here.
        self._list._exam_columns = None
        
    def __contains__(self, key: str) -> bool:
        cols = self._list._cols
        return key in cols and cols[key][self._idx] is not None
        
    def keys(self):
        """Column names holding a non-``None`` value for this row."""
        return [k for k, v in self._list._cols.items() if v[self._idx] is not None]
        
    def items(self):
        """``(key, value)`` pairs for columns with a non-``None`` value in
        this row."""
        return [(k, v[self._idx]) for k, v in self._list._cols.items() if v[self._idx] is not None]
        
    def __repr__(self) -> str:
        return f"ExamProxy({dict(self.items())})"


class ColumnarExamsList:
    """A memory-efficient virtual sequence of exam dictionaries.
    
    Stores exam data as columns (lists or arrays) and instantiates lightweight
    ExamProxy objects on the fly, eliminating hundreds of MBs of dict overhead.

    Caches the ``ExamColumns`` built from it on ``_exam_columns`` so repeated
    ``ExamColumns.from_exam_data`` calls (the per-year post-run block rebuilds
    it >=5 times over the same data) return in O(1).  Any ``ExamProxy`` mutation
    clears the cache.  Reads use ``getattr(self, "_exam_columns", None)`` so an
    old pickle unpickled into this class (which lacks the slot) does not raise.
    """
    __slots__ = ("_cols", "_N", "_exam_columns")
    
    def __init__(self, cols: Dict[str, List[Any]], N: int):
        self._cols = cols
        self._N = N
        self._exam_columns = None
        
    def __len__(self) -> int:
        return self._N
        
    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._N)
            return [ExamProxy(self, i) for i in range(start, stop, step)]
        if idx < 0:
            idx += self._N
        if idx < 0 or idx >= self._N:
            raise IndexError("Index out of range")
        return ExamProxy(self, idx)
        
    def __iter__(self):
        for i in range(self._N):
            yield ExamProxy(self, i)

    @classmethod
    def from_dicts(cls, exams) -> "ColumnarExamsList":
        """Normalise exam data to a ``ColumnarExamsList``.

        Callers accept exam data in three shapes and this converts them all
        once, so the rest of the function only handles the columnar case:

        - an existing ``ColumnarExamsList`` is returned unchanged;
        - a dict of columns ``{name: [values]}`` is wrapped directly (``N``
          is the length of the first column, 0 for an empty dict);
        - any other sequence of row dicts is converted: columns are the
          union of keys in first-appearance order, and a key absent from a
          row becomes ``None`` in that row's column.  ``ExamProxy`` treats
          ``None`` as absent, so ``.get(k, d)`` and ``in`` behave exactly
          as on the original dict.

        Empty input returns ``ColumnarExamsList({}, 0)``.
        """
        if getattr(exams, "_cols", None) is not None:
            return exams
        if isinstance(exams, dict):
            cols = dict(exams)
            n = len(next(iter(cols.values()))) if cols else 0
            return cls(cols, n)
        rows = list(exams)
        if not rows:
            return cls({}, 0)
        keys: Dict[str, None] = {}
        for row in rows:
            for k in (row.keys() if hasattr(row, "keys") else row):
                if k not in keys:
                    keys[k] = None
        cols = {k: [] for k in keys}
        for row in rows:
            for k in keys:
                cols[k].append(row.get(k))
        return cls(cols, len(rows))

