"""
Columnar data structures for Monte Carlo analysis optimization.

Provides ExamColumns and EventColumns classes that store data in NumPy arrays
for efficient vectorized operations, avoiding repeated Python dict.get() calls.
Preprocessed exams and Monte Carlo events are converted into these tables so
``analysis``, ``vectorized``, ``year_batch`` and ``research_questions`` can
consume whole columns at once.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np


class ExamColumns:
    """Columnar storage for exam_data using NumPy arrays.

    Stores frequently-accessed exam fields as dense NumPy arrays for
    efficient vectorized operations. Built once in a single pass over
    exam_data and reused across multiple analysis functions.

    Attributes
    ----------
    age : np.ndarray[float64]
        Patient age at exposure
    sex_code : np.ndarray[U1] (single-character unicode)
        Sex code: 'M', 'F', or 'U' (unknown)
    modality : np.ndarray[object]
        Modality string
    region : np.ndarray[object]
        Region string
    protocol : np.ndarray[object]
        Protocol/description string
    ed_msv : np.ndarray[float64]
        Effective dose in mSv
    patient_id : np.ndarray[object]
        Patient identifier (can be None, NaN, or string/int)
    total_patient_dose_sv : np.ndarray[float64]
        Cumulative patient dose in Sv (engine annotation)
    population : np.ndarray[object]
        Population string (Euro-American, Asian, etc.)
    mataac_frailty_factor : np.ndarray[float64]
        MATAAC frailty adjustment factor
    mataac_index_ced_sv : np.ndarray[float64]
        The MATAAC stratification variable: CT-only effective dose (Sv)
        accrued by this patient in the calendar year of this exam.  This is
        the quantity Mataac et al. grouped on, so it is also what any
        dose-group breakdown must bin by.
    device : np.ndarray[object]
        Short device/model name (e.g. "DigitalDiagnost").  Empty string
        when the input file has no "Device" column.  Used by the
        device-renewal sub-study only; the risk model ignores it.
    equipment : np.ndarray[object]
        Unique equipment/room identifier (e.g.
        "HOSP-ROOM-VENDOR_MODEL_1").  Empty string when
        the input file has no "Equipment" column.
    date_performed : np.ndarray[datetime64[ns]]
        Calendar time of the exam.  NaT when missing / unparseable.
        Used by the device-renewal sub-study to detect the moment a
        given room's equipment identifier changes.
    file_index : np.ndarray[int32]
        Index into the file list of the input file each exam originated
        from (set during preprocessing as ``_file_index``).  Used by the
        Patient Cohort age pyramid to filter the chart by input file.
        All zeros when the source data did not record it (e.g. a single
        file), which is harmless: every exam maps to the same (only) file.
    """

    __slots__ = (
        "age",
        "sex_code",
        "modality",
        "region",
        "protocol",
        "ed_msv",
        "patient_id",
        "person_id",
        "total_patient_dose_sv",
        "population",
        "mataac_frailty_factor",
        "mataac_index_ced_sv",
        "device",
        "equipment",
        "date_performed",
        "file_index",
        "_n",
    )

    def __init__(
        self,
        age: np.ndarray,
        sex_code: np.ndarray,
        modality: np.ndarray,
        region: np.ndarray,
        protocol: np.ndarray,
        ed_msv: np.ndarray,
        patient_id: np.ndarray,
        total_patient_dose_sv: np.ndarray,
        population: np.ndarray,
        mataac_frailty_factor: np.ndarray,
        mataac_index_ced_sv: Optional[np.ndarray] = None,
        device: Optional[np.ndarray] = None,
        equipment: Optional[np.ndarray] = None,
        date_performed: Optional[np.ndarray] = None,
        person_id: Optional[np.ndarray] = None,
        file_index: Optional[np.ndarray] = None,
    ):
        """Build a columnar exam table from per-field arrays, one row per
        exam.  ``ed_msv`` is the per-exam effective dose in mSv,
        ``total_patient_dose_sv`` the running cumulative effective dose in
        Sv, ``age`` in years, ``mataac_frailty_factor`` a unitless survival
        multiplier.  Optional fields default to empty/zero arrays of the
        same length (``person_id`` defaults to ``patient_id``); every array
        must be the same length, which is ``len(exam)``."""
        self.age = age
        self.sex_code = sex_code
        self.modality = modality
        self.region = region
        self.protocol = protocol
        self.ed_msv = ed_msv
        self.patient_id = patient_id
        self.total_patient_dose_sv = total_patient_dose_sv
        self.population = population
        self.mataac_frailty_factor = mataac_frailty_factor
        n = len(age)
        self.mataac_index_ced_sv = (
            mataac_index_ced_sv
            if mataac_index_ced_sv is not None
            else np.zeros(n, dtype=np.float64)
        )
        self.person_id = (
            person_id if person_id is not None else patient_id
        )
        self.device = (
            device if device is not None else np.full(n, "", dtype=object)
        )
        self.equipment = (
            equipment if equipment is not None else np.full(n, "", dtype=object)
        )
        self.date_performed = (
            date_performed
            if date_performed is not None
            else np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        )
        self.file_index = (
            file_index
            if file_index is not None
            else np.zeros(n, dtype=np.int32)
        )
        self._n = n

    def __len__(self) -> int:
        return self._n

    def subset(self, mask):
        """Return a new ExamColumns containing only rows where mask is True."""
        return ExamColumns(
            age=self.age[mask],
            sex_code=self.sex_code[mask],
            modality=self.modality[mask],
            region=self.region[mask],
            protocol=self.protocol[mask],
            ed_msv=self.ed_msv[mask],
            patient_id=self.patient_id[mask],
            total_patient_dose_sv=self.total_patient_dose_sv[mask],
            population=self.population[mask],
            mataac_frailty_factor=self.mataac_frailty_factor[mask],
            mataac_index_ced_sv=self.mataac_index_ced_sv[mask],
            device=self.device[mask],
            equipment=self.equipment[mask],
            date_performed=self.date_performed[mask],
            person_id=self.person_id[mask],
            file_index=self.file_index[mask],
        )

    @classmethod
    def from_exam_data(cls, exam_data: List[Dict[str, Any]]) -> "ExamColumns":
        """Build ExamColumns from a list of exam dicts in a single pass.

        Parameters
        ----------
        exam_data : list of dict
            Raw exam data from simulation

        Returns
        -------
        ExamColumns
            Columnar representation of the exam data

        Caching
        -------
        If ``exam_data`` is a :class:`ColumnarExamsList` (or any object exposing
        ``_exam_columns``), an existing ``ExamColumns`` of matching
        length is returned in O(1) instead of rebuilding.  This avoids the
        >=5 redundant rebuilds the per-year post-run block performs over the
        same data.  A plain ``list`` is never cached (ids are reused, so an
        id-keyed dict would be unsafe); only objects whose type declares
        ``_exam_columns`` in ``__slots__`` get the cache written back.  Reads
        use ``getattr(..., None)`` so an old pickle unpickled into a class
        that lacks the slot does not raise.
        """
        cached = getattr(exam_data, "_exam_columns", None)
        if cached is not None and len(cached) == len(exam_data):
            return cached
        n = len(exam_data)
        age = np.zeros(n, dtype=np.float64)
        sex_code = np.empty(n, dtype="U1")
        modality = np.empty(n, dtype=object)
        region = np.empty(n, dtype=object)
        protocol = np.empty(n, dtype=object)
        ed_msv = np.zeros(n, dtype=np.float64)
        patient_id = np.empty(n, dtype=object)
        person_id = np.empty(n, dtype=object)
        total_patient_dose_sv = np.zeros(n, dtype=np.float64)
        population = np.empty(n, dtype=object)
        mataac_frailty_factor = np.ones(n, dtype=np.float64)
        mataac_index_ced_sv = np.zeros(n, dtype=np.float64)
        device = np.empty(n, dtype=object)
        equipment = np.empty(n, dtype=object)
        file_index = np.zeros(n, dtype=np.int32)
        _date_raw: list = [None] * n

        for i, ex in enumerate(exam_data):
            age[i] = float(ex.get("age", 0) or 0)
            sx = str(ex.get("sex_code") or ex.get("sex") or "U").strip().upper()[:1]
            sex_code[i] = sx if sx in ("M", "F") else "U"
            modality[i] = ex.get("Acquisition Modality") or ex.get("modality") or "Unknown"
            region[i] = ex.get("region") or "Unknown"
            protocol[i] = str(ex.get("Protocol") or ex.get("protocol") or "").strip() or str(ex.get("Acquisition Modality") or ex.get("modality") or "Unknown")
            ed_msv[i] = float(ex.get("ed_msv", 0.0) or 0.0)
            patient_id[i] = ex.get("_patient_id")
            person_id[i] = ex.get("_person_id", ex.get("_patient_id"))
            total_patient_dose_sv[i] = float(ex.get("_total_patient_dose_sv", 0.0) or 0.0)
            population[i] = ex.get("_population") or ex.get("population") or "Euro-American"
            mataac_frailty_factor[i] = float(ex.get("_mataac_frailty_factor", 1.0) or 1.0)
            mataac_index_ced_sv[i] = float(ex.get("_mataac_index_ced_sv", 0.0) or 0.0)
            device[i]    = str(ex.get("Device") or ex.get("device") or "")
            equipment[i] = str(ex.get("Equipment") or ex.get("equipment") or "")
            _fi = ex.get("_file_index")
            file_index[i] = int(_fi) if _fi is not None else 0
            _date_raw[i] = ex.get("Date Performed") or ex.get("date_performed")

        try:
            # Shared parser: a bare ``pd.to_datetime`` reads an Excel serial day
            # number as nanoseconds since the epoch and returns 1970 instead of
            # failing, which would put a whole serial-dated file into one bogus
            # year here while preprocess keyed the Mataac stratum on the real one.
            from .excel_io import parse_exam_dates as _parse_dates
            date_performed = _parse_dates(_date_raw).to_numpy()
        except Exception:
            date_performed = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

        built = cls(
            age=age,
            sex_code=sex_code,
            modality=modality,
            region=region,
            protocol=protocol,
            ed_msv=ed_msv,
            patient_id=patient_id,
            total_patient_dose_sv=total_patient_dose_sv,
            population=population,
            mataac_frailty_factor=mataac_frailty_factor,
            mataac_index_ced_sv=mataac_index_ced_sv,
            device=device,
            equipment=equipment,
            date_performed=date_performed,
            person_id=person_id,
            file_index=file_index,
        )
        # Write back to the source object's cache slot when its type declares
        # ``_exam_columns`` in ``__slots__`` (i.e. a ColumnarExamsList).  Plain
        # lists get no caching -- an id()-keyed dict would be unsafe because
        # ids are reused.  Assignment to a declared slot always works even if
        # the slot was never set (e.g. an old pickle), so no guard is needed.
        if hasattr(type(exam_data), "__slots__") and \
                "_exam_columns" in getattr(type(exam_data), "__slots__", ()):
            exam_data._exam_columns = built
        return built


class EventColumns:
    """Columnar storage for cancer_events using NumPy arrays.

    Stores frequently-accessed event fields as dense NumPy arrays for
    efficient vectorized operations. Built once in a single pass over
    cancer_events and reused across multiple analysis functions.

    Attributes
    ----------
    cancer_type : np.ndarray[object]
        Cancer type string
    modality : np.ndarray[object]
        Modality string
    region : np.ndarray[object]
        Region string
    protocol : np.ndarray[object]
        Protocol/description string
    age_at_exposure : np.ndarray[float64]
        Age at exposure (years)
    sex : np.ndarray[U1]
        Sex code: 'M', 'F', or 'U'
    died : np.ndarray[bool_]
        Whether the patient died from this cancer
    exam_index : np.ndarray[int64]
        Index into exam_data for the associated exam
    iteration : np.ndarray[int32]
        Monte Carlo iteration index for each event
    ed_msv : np.ndarray[float64]
        Effective dose in mSv (from associated exam)
    QALY_lost : np.ndarray[float64]
        QALYs lost
    life_lost_years : np.ndarray[float64]
        Life years lost
    """

    __slots__ = (
        "cancer_type",
        "modality",
        "region",
        "protocol",
        "age_at_exposure",
        "sex",
        "died",
        "exam_index",
        "iteration",
        "ed_msv",
        "QALY_lost",
        "life_lost_years",
        "age_at_death",
        "projected_lifetime",
        "_n",
    )

    def __init__(
        self,
        cancer_type: np.ndarray,
        modality: np.ndarray,
        region: np.ndarray,
        protocol: np.ndarray,
        age_at_exposure: np.ndarray,
        sex: np.ndarray,
        died: np.ndarray,
        exam_index: np.ndarray,
        iteration: np.ndarray,
        ed_msv: np.ndarray,
        QALY_lost: np.ndarray,
        life_lost_years: np.ndarray,
        age_at_death: np.ndarray,
        projected_lifetime: np.ndarray,
    ):
        """Build a columnar cancer-event table from per-field arrays, one
        row per simulated cancer event.  ``ed_msv`` is the exam's effective
        dose in mSv, ``age_at_exposure``/``age_at_death``/``life_lost_years``
        and ``projected_lifetime`` are in years, ``QALY_lost`` in
        severity-weighted QALYs, ``died`` a boolean fatality flag,
        ``iteration`` the Monte Carlo iteration index and ``exam_index``
        the row of the source exam.  All arrays must be the same length."""
        self.cancer_type = cancer_type
        self.modality = modality
        self.region = region
        self.protocol = protocol
        self.age_at_exposure = age_at_exposure
        self.sex = sex
        self.died = died
        self.exam_index = exam_index
        self.iteration = iteration
        self.ed_msv = ed_msv
        self.QALY_lost = QALY_lost
        self.life_lost_years = life_lost_years
        self.age_at_death = age_at_death
        self.projected_lifetime = projected_lifetime
        self._n = len(cancer_type)

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        """Yield each event as a dict so callers written for list-of-dicts events still work."""
        for i in range(self._n):
            yield {
                "cancer_type": self.cancer_type[i],
                "modality": self.modality[i],
                "region": self.region[i],
                "protocol": self.protocol[i],
                "age_at_exposure": self.age_at_exposure[i],
                "age": self.age_at_exposure[i],
                "sex": self.sex[i],
                "died": self.died[i],
                "exam_index": self.exam_index[i],
                "iteration": self.iteration[i],
                "ed_msv": self.ed_msv[i],
                "QALY_lost": self.QALY_lost[i],
                "life_lost_years": self.life_lost_years[i],
                "age_at_death": self.age_at_death[i],
                "projected_lifetime": self.projected_lifetime[i],
            }

    def subset(self, mask):
        """Return a new EventColumns containing only rows where mask is True."""
        return EventColumns(
            cancer_type=self.cancer_type[mask],
            modality=self.modality[mask],
            region=self.region[mask],
            protocol=self.protocol[mask],
            age_at_exposure=self.age_at_exposure[mask],
            sex=self.sex[mask],
            died=self.died[mask],
            exam_index=self.exam_index[mask],
            iteration=self.iteration[mask],
            ed_msv=self.ed_msv[mask],
            QALY_lost=self.QALY_lost[mask],
            life_lost_years=self.life_lost_years[mask],
            age_at_death=self.age_at_death[mask],
            projected_lifetime=self.projected_lifetime[mask],
        )

    @classmethod
    def empty(cls) -> "EventColumns":
        """Return an empty EventColumns with correctly-typed arrays.

        Single source of truth for the empty-event-columns construction,
        shared by ``main.py``, ``analysis.py`` and ``research_questions.py``.
        """
        return cls(
            cancer_type=np.array([], dtype=object),
            modality=np.array([], dtype=object),
            region=np.array([], dtype=object),
            protocol=np.array([], dtype=object),
            age_at_exposure=np.array([], dtype=np.float64),
            sex=np.array([], dtype="U1"),
            died=np.array([], dtype=bool),
            exam_index=np.array([], dtype=np.int64),
            iteration=np.array([], dtype=np.int32),
            ed_msv=np.array([], dtype=np.float64),
            QALY_lost=np.array([], dtype=np.float64),
            life_lost_years=np.array([], dtype=np.float64),
            age_at_death=np.array([], dtype=np.float64),
            projected_lifetime=np.array([], dtype=np.float64),
        )

    @classmethod
    def from_events(
        cls, cancer_events: List[Dict[str, Any]], exam_data: Optional[List[Dict[str, Any]]] = None
    ) -> "EventColumns":
        """Build EventColumns from a list of event dicts in a single pass.

        Parameters
        ----------
        cancer_events : list of dict
            Raw cancer event data from simulation
        exam_data : list of dict, optional
            Exam data for looking up exam-specific fields like ed_msv

        Returns
        -------
        EventColumns
            Columnar representation of the event data
        """
        n = len(cancer_events)
        cancer_type = np.empty(n, dtype=object)
        modality = np.empty(n, dtype=object)
        region = np.empty(n, dtype=object)
        protocol = np.empty(n, dtype=object)
        age_at_exposure = np.zeros(n, dtype=np.float64)
        sex = np.empty(n, dtype="U1")
        died = np.zeros(n, dtype=bool)
        exam_index = np.full(n, -1, dtype=np.int64)
        iteration = np.zeros(n, dtype=np.int32)
        ed_msv = np.zeros(n, dtype=np.float64)
        QALY_lost = np.zeros(n, dtype=np.float64)
        life_lost_years = np.zeros(n, dtype=np.float64)
        age_at_death = np.zeros(n, dtype=np.float64)
        projected_lifetime = np.zeros(n, dtype=np.float64)

        for i, ev in enumerate(cancer_events):
            cancer_type[i] = str(ev.get("cancer_type", "Unknown"))
            modality[i] = ev.get("modality") or "Unknown"
            region[i] = ev.get("region") or "Unknown"
            protocol[i] = str(ev.get("protocol") or "").strip() or str(ev.get("modality") or "Unknown")
            age_at_exposure[i] = float(ev.get("age_at_exposure", ev.get("age", 0)) or 0)
            sx = str(ev.get("sex") or "U").strip().upper()[:1]
            sex[i] = sx if sx in ("M", "F") else "U"
            died[i] = bool(ev.get("died", False))
            exam_index[i] = int(ev.get("exam_index", -1) or -1)
            iteration[i] = int(ev.get("iteration", 0) or 0)
            ed_msv[i] = float(ev.get("ed_msv", 0.0) or 0.0)
            QALY_lost[i] = float(ev.get("QALY_lost", 0.0) or 0.0)
            life_lost_years[i] = float(
                ev.get("life_lost_years", ev.get("life_lost", ev.get("Life_Lost", ev.get("life_years_lost", 0.0)))) or 0.0
            )
            age_at_death[i] = float(ev.get("age_at_death") or ev.get("age_at_exposure", ev.get("age", 0)) or 0.0)
            projected_lifetime[i] = float(ev.get("projected_lifetime") or ev.get("age_at_exposure", ev.get("age", 0)) or 0.0)

        return cls(
            cancer_type=cancer_type,
            modality=modality,
            region=region,
            protocol=protocol,
            age_at_exposure=age_at_exposure,
            sex=sex,
            died=died,
            exam_index=exam_index,
            iteration=iteration,
            ed_msv=ed_msv,
            QALY_lost=QALY_lost,
            life_lost_years=life_lost_years,
            age_at_death=age_at_death,
            projected_lifetime=projected_lifetime,
        )


def is_event_columns(obj) -> bool:
    """Return True if ``obj`` is a columnar EventColumns instance.

    Shared by ``analysis.py`` and ``research_questions.py`` so both use the
    same type check.
    """
    return hasattr(obj, "region") and hasattr(obj, "age_at_exposure")
