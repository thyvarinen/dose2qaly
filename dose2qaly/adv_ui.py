"""Advanced analysis panels for the Monte Carlo results GUI.

Provides a Tkinter sub-panel that is embedded as a tab inside
``gui.start_gui``:

1. **AdvancedRiskAnalysisPanel** — Bayesian risk curves, filtered by
   sex / age / modality, with DDREF sensitivity.

All panels use matplotlib embedded in Tkinter via FigureCanvasTkAgg.
"""
from __future__ import annotations

import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .analysis import (
    union_risk_analysis_filter_values,
    compute_filtered_risk_curve,
    compute_batch_averaged_risk_curve,
    compute_batch_averaged_bayesian_curve,
    batch_bayesian_curve_supported,
    compute_combined_bayesian_curve,
    plot_filtered_risk_analysis,
    format_filtered_risk_summary,
    summarize_mataac_strata,
    _ensure_exam_columns,
    _ensure_event_columns,
)

def _detail_payload_scope(stats: dict, fallback_iterations: int) -> tuple:
    """``(iterations, note, payload_stats)`` for the per-exam payload held here.

    A ``per_year_average`` batch drops each year's ``exam_data``/``events``
    before loading the next one, so ``year_batch.run_yearly_simulation``
    returns only the LAST PROCESSED year's payload (``last_ed`` / ``last_ev``)
    -- while ``stats["iterations"]`` is the batch-wide MEAN iteration count
    (see ``_scale_rates_to_mean_iterations``).  Pairing those two is a unit
    mismatch, not just a labelling problem:
    :func:`analysis.compute_filtered_risk_curve` multiplies the dose
    denominator by the iteration count it is handed, but the event list it
    counts cases from is already fixed at whatever the detail year actually
    ran.  Feeding it the batch mean therefore scales risk/Sv by exactly
    ``year_iters / mean_iters`` -- e.g. a year that converged at 40
    iterations inside a batch averaging 80 reads half its true risk/Sv.

    Returns that year's own iteration count, a note naming the year so the
    report cannot be mistaken for a batch-wide figure, and the stats dict
    that DESCRIBES the retained payload -- the detail year's own, not the
    averaged one.  Every cumulative scalar the report mixes with the event
    list (``total_sv``, ``heritable_risk``, ``heritable_detriment``; see
    ``year_batch._CUMULATIVE_TOP_LEVEL``) has to be read from that dict for
    the same reason, or a one-year numerator is divided by a nine-year
    denominator.  Any other run returns ``stats`` itself and an empty note.
    """
    stats = stats or {}
    iterations = int(stats.get("iterations", fallback_iterations) or fallback_iterations)
    if stats.get("_run_mode") != "per_year_average":
        return iterations, "", stats
    detail_year = stats.get("_detail_year")
    year_stats = ((stats.get("_year_batches") or {}).get(detail_year) or {})
    year_iters = int(year_stats.get("iterations", 0) or 0)
    if len(stats.get("_year_batches") or {}) <= 1:
        return (year_iters or iterations), "", (year_stats or stats)
    n_years = stats.get("_n_years", "?")
    note = (
        f"SINGLE YEAR ONLY \u2014 this tab reflects year group '{detail_year}', "
        f"not the {n_years}-year average shown on the Summary tab.  The "
        f"per-exam detail it filters is dropped for every earlier year to "
        f"bound memory, so run a batch containing only that year group to "
        f"inspect a different one."
    )
    return (year_iters or iterations), note, (year_stats or stats)


def _filter_events_by_visible_cancers(events, visible):
    """Filter events to visible cancer types.

    Supports both columnar EventColumns and list-of-dicts event formats.
    Returns the same type as the input.
    """
    if not visible:
        return events
    # Columnar fast path
    if hasattr(events, "cancer_type") and hasattr(events, "subset"):
        ct = events.cancer_type
        mask = np.array([c in visible for c in ct], dtype=bool)
        if mask.all():
            return events
        return events.subset(mask)
    # List-of-dicts fallback
    return [ev for ev in events if ev.get("cancer_type", "") in visible]


class AdvancedRiskAnalysisPanel:
    """Filterable analysis tab embedded in the results window.

    The user picks a sex / modality / region / age-window subset (plus
    risk-model and DDREF overrides) and the panel recomputes the comparison
    outputs for that sub-cohort without rerunning the simulation.
    """
    def __init__(self, parent, n_iterations: int):
        self.n_iterations = n_iterations
        self.frame = ttk.Frame(parent)
        self.sex_var = tk.StringVar(value="All")
        self.mod_var = tk.StringVar(value="All")
        self.region_var = tk.StringVar(value="All")
        self.min_age_var = tk.StringVar(value="0")
        self.max_age_var = tk.StringVar(value="100")
        self.ddref_solid_var = tk.StringVar(value="2.0")
        self.ddref_leuk_var = tk.StringVar(value="1.0")
        self.compare_golikov_var = tk.BooleanVar(value=False)
        self.compare_model_var = tk.BooleanVar(value=False)
        self.compare_mataac_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Radiation sensitivity analysis ready")
        self._results_getter = None
        self._summary_refresh_callback = None
        self._visible_cancers_getter = None
        self._build_ui()

    def _build_ui(self):
        """Build the filter row, parameter inputs and output area of the panel."""
        # Row 1: Filters
        row1 = ttk.Frame(self.frame)
        row1.pack(fill=tk.X, pady=(3, 0))

        ttk.Label(row1, text="Sex:").pack(side=tk.LEFT, padx=(5, 2))
        self.sex_cb = ttk.Combobox(row1, textvariable=self.sex_var, width=8, state="readonly")
        self.sex_cb["values"] = ["All"]
        self.sex_cb.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row1, text="Modality:").pack(side=tk.LEFT, padx=(0, 2))
        self.mod_cb = ttk.Combobox(row1, textvariable=self.mod_var, width=12, state="readonly")
        self.mod_cb["values"] = ["All"]
        self.mod_cb.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row1, text="Region:").pack(side=tk.LEFT, padx=(0, 2))
        self.region_cb = ttk.Combobox(row1, textvariable=self.region_var, width=14, state="readonly")
        self.region_cb["values"] = ["All"]
        self.region_cb.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row1, text="Age:").pack(side=tk.LEFT, padx=(0, 2))
        self.min_age_entry = ttk.Entry(row1, textvariable=self.min_age_var, width=5)
        self.min_age_entry.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(row1, text="–").pack(side=tk.LEFT)
        self.max_age_entry = ttk.Entry(row1, textvariable=self.max_age_var, width=5)
        self.max_age_entry.pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(row1, text="DDREF Solid:").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Entry(row1, textvariable=self.ddref_solid_var, width=5).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(row1, text="DDREF Leuk:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Entry(row1, textvariable=self.ddref_leuk_var, width=5).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(row1, text="Run Analysis", command=self.refresh).pack(side=tk.LEFT, padx=5)

        # Row 2: Comparison toggles + status
        row2 = ttk.Frame(self.frame)
        row2.pack(fill=tk.X, pady=(1, 3))

        ttk.Checkbutton(
            row2,
            text="Compare with vs without weight adjustment (patients < 70 kg)",
            variable=self.compare_golikov_var,
            command=self.refresh,
        ).pack(side=tk.LEFT, padx=(5, 10))

        # The BEIR VII LAR model is not installed yet, so the comparison is
        # shown but cannot be switched on.
        self.compare_model_var.set(False)
        ttk.Checkbutton(
            row2,
            text="Compare risk models (LAR vs REIC) \u2014 not yet available",
            variable=self.compare_model_var,
            state="disabled",
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Checkbutton(
            row2,
            text="Compare with vs without competing-mortality adjustment",
            variable=self.compare_mataac_var,
            command=self.refresh,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row2, textvariable=self.status_var).pack(side=tk.LEFT, padx=(10, 5))

        mid = ttk.Frame(self.frame)
        mid.pack(fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(12, 4), dpi=90)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.text = scrolledtext.ScrolledText(self.frame, font=("Consolas", 9), wrap=tk.WORD, height=10)
        self.text.pack(fill=tk.BOTH, expand=True)

        self.sex_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        self.mod_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        self.region_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        self.min_age_entry.bind("<FocusOut>", lambda _e: self.refresh())
        self.max_age_entry.bind("<FocusOut>", lambda _e: self.refresh())
        self.min_age_entry.bind("<Return>", lambda _e: self.refresh())
        self.max_age_entry.bind("<Return>", lambda _e: self.refresh())
        
    def _parse_float(self, value: str, default: float) -> float:
        try:
            return float(value)
        except ValueError:
            return default

    def set_results_getter(self, getter):
        """Register a callable returning the latest results dict so the panel
        can read stats without holding a direct reference."""
        self._results_getter = getter

    def set_visible_cancers_getter(self, getter):
        """Register a callable that returns the set of visible cancer types."""
        self._visible_cancers_getter = getter

    def set_summary_refresh_callback(self, callback):
        """Register a callback so Bayesian recalculation also refreshes the Summary tab."""
        self._summary_refresh_callback = callback

    def _batch_digests(self, stats: dict):
        """``{year: digest}`` for a ``per_year_average`` batch, else ``None``.

        Each digest is a shallow copy carrying the digest's own iteration
        count (``"iterations"`` -- set by ``year_batch``; falls back to the
        year stats' value for digests recorded before the key existed).
        Returns ``None`` unless at least one year group produced a digest,
        so callers can fall back to the retained detail-year payload.
        """
        if (stats or {}).get("_run_mode") != "per_year_average":
            return None
        digests = {}
        for year, ystats in (stats.get("_year_batches") or {}).items():
            d = ystats.get("_risk_curve_digest")
            if not d:
                continue
            d = dict(d)
            if not d.get("iterations"):
                d["iterations"] = int(ystats.get("iterations", 0) or 0)
            digests[year] = d
        return digests or None

    def update_filter_values(self, exam_data: list):
        """Repopulate the modality/region filter comboboxes from the current
        exam data (or the per-year batch digests when they cover every
        year)."""
        sources = [exam_data]
        if self._results_getter is not None:
            _stats = (self._results_getter() or {}).get("stats") or {}
            _digests = self._batch_digests(_stats)
            if _digests:
                # The digests cover every successful year, so the union is
                # complete without the retained detail-year payload.
                sources = [d.get("exam_columns") for d in _digests.values()]
        values = union_risk_analysis_filter_values(sources)
        self.sex_cb["values"] = values.get("sexes", ["All"])
        self.mod_cb["values"] = values.get("modalities", ["All"])
        self.region_cb["values"] = values.get("regions", ["All"])
        if self.sex_var.get() not in self.sex_cb["values"]:
            self.sex_var.set("All")
        if self.mod_var.get() not in self.mod_cb["values"]:
            self.mod_var.set("All")
        if self.region_var.get() not in self.region_cb["values"]:
            self.region_var.set("All")

    def refresh(self):
        """Pull the latest results through the registered getter and refresh
        the whole panel; reports a status line when nothing is available
        yet."""
        if self._results_getter is None:
            self.status_var.set("No results getter configured")
            return
        results = self._results_getter() or {}
        stats = results.get("stats")
        if not stats:
            self.status_var.set("Waiting for simulation data...")
            return
        exam_data = results.get("exam_data") or []
        events = results.get("events") or []
        exam_columns = _ensure_exam_columns(exam_data)
        event_columns = _ensure_event_columns(events)
        # Filter events by visible cancer types
        if self._visible_cancers_getter is not None and event_columns is not None:
            visible = self._visible_cancers_getter()
            event_columns = _filter_events_by_visible_cancers(event_columns, visible)
        if exam_columns is None or len(exam_columns) == 0:
            # In a per_year_average batch the retained payload is only the
            # last processed year and may legitimately be absent while the
            # per-year digests still carry every year's curve inputs.
            if self._batch_digests(stats) is None:
                self.status_var.set("No exam data available yet")
                return
        self.status_var.set("Computing risk analysis...")
        self.frame.update_idletasks()
        try:
            self._refresh_analysis(stats, exam_data, events, exam_columns, event_columns)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[Risk Analysis] Error in refresh: {exc}\n{tb}", flush=True)
            self.status_var.set(f"Error: {exc}")
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, f"Error during risk analysis:\n{exc}\n\n{tb}")

    def _refresh_analysis(self, stats, exam_data, events, exam_columns, event_columns):
        """Recompute the analysis for the currently selected filters/model
        parameters and repopulate the panel's outputs."""
        selected_model = str(stats.get("risk_model", "REIC") or "REIC").upper()
        comparison_model = "LAR" if selected_model == "REIC" else "REIC"
        population = str(stats.get("population", "Euro-American") or "Euro-American")
        ddref_solid = self._parse_float(self.ddref_solid_var.get(), 2.0)
        ddref_leuk = self._parse_float(self.ddref_leuk_var.get(), 1.0)
        min_age = self._parse_float(self.min_age_var.get(), 0.0)
        max_age = self._parse_float(self.max_age_var.get(), 100.0)

        filters = {
            "sex": self.sex_var.get(),
            "modality": self.mod_var.get(),
            "region": self.region_var.get(),
            "min_age": min_age,
            "max_age": max_age,
        }

        cmp_golikov = self.compare_golikov_var.get()
        cmp_model = self.compare_model_var.get()
        cmp_mataac = self.compare_mataac_var.get()
        ddref_changed = (ddref_solid != 2.0 or ddref_leuk != 1.0)
        has_adjustments = ddref_changed or cmp_golikov or cmp_model or cmp_mataac

        # The retained per-exam payload (exam_data/events above) is one
        # year only in a per_year_average batch -- see _detail_payload_scope.
        # The Bayesian comparison overlay needs full exam objects (model
        # comparison, Mataac frailty fields) that the batch does not retain
        # across years, so it is ALWAYS computed from that one year,
        # regardless of what the Simulation curve below does.
        detail_iterations, detail_note, payload_stats = _detail_payload_scope(
            stats, self.n_iterations)

        # Simulation curve (always: default DDREF, standard dose, no
        # scaling).  In a per_year_average batch this is instead a true
        # equal-weighted average across every year group that has a Risk
        # Analysis digest (see year_batch's per-year loop /
        # analysis.compute_batch_averaged_risk_curve) -- unlike the
        # Bayesian overlay, it needs only age/sex/modality/region/dose and
        # a handful of event fields, all of which the digest retains.
        digests = self._batch_digests(stats)
        n_year_groups = len(stats.get("_year_batches") or {})
        # Apply the visible-cancer filter to each year's digest events, so a
        # cancer type hidden on the Cancer Events tab is excluded from the
        # batch-averaged curves exactly as it is from the single-year path.
        if digests and self._visible_cancers_getter is not None:
            _visible = self._visible_cancers_getter()
            if _visible:
                digests = {y: {**d, "event_columns": _filter_events_by_visible_cancers(
                    d.get("event_columns"), _visible)} for y, d in digests.items()}
        n_batch_digests = len(digests or {})

        _sim_cfg = stats.get("sim_config") or {}
        # Sampled-YLL detriment: unless the ICRP 152 reference-assumptions
        # switch is on, "full" QALY mode already carries each event's own
        # YLL (QALY_lost), so the Risk Analysis detriment weights events by
        # QALY_lost/15 instead of the l_rel x 15 y site average.
        _sampled_yll = (not bool(_sim_cfg.get("icrp152_strict", False))
                        and str(_sim_cfg.get("QALY_mode", "full")).lower() == "full")

        sim_scope_note = ""
        if n_batch_digests > 0:
            sim_curve = compute_batch_averaged_risk_curve(
                digests, filters, bin_width=5, sampled_yll=_sampled_yll)
            sim_years_used = sim_curve.get("_averaged_across_years", n_batch_digests)
            sim_iterations_label = f"{sim_years_used}yr pooled"
            # Heritable terms must be on the SAME "per one simulated year"
            # basis as the averaged curve.  stats["heritable_risk"] is the
            # equal-weighted per-iteration rate scaled UP to the batch's
            # mean iteration count (year_batch._scale_rates_to_mean_iterations);
            # dividing by stats["iterations"] (that same mean) undoes
            # exactly that rescale and recovers the per-simulated-year rate.
            _batch_iters = float(stats.get("iterations", 0) or 0)
            _heritable_risk = (float(stats.get("heritable_risk", 0.0) or 0.0) / _batch_iters
                              if _batch_iters > 0 else 0.0)
            _heritable_detriment = (float(stats.get("heritable_detriment", 0.0) or 0.0) / _batch_iters
                                    if _batch_iters > 0 else 0.0)
            # The averaged curve's own dose is already self-consistent with
            # its cases/deaths (both equal-weighted across the same years);
            # substituting any single-cohort total here would reintroduce
            # exactly the scope mismatch this averaging exists to fix.
            _total_dose_sv = None
            if n_batch_digests < n_year_groups:
                sim_scope_note = (
                    f"Simulation curve/headline: equal-weighted average "
                    f"across {sim_years_used} of {n_year_groups} year "
                    f"group(s) -- the rest predate the Risk Analysis "
                    f"per-year digest and are excluded rather than treated "
                    f"as a zero year."
                )
        else:
            sim_curve = compute_filtered_risk_curve(
                exam_data, events, detail_iterations, filters, bin_width=5,
                exam_columns=exam_columns, event_columns=event_columns,
                sampled_yll=_sampled_yll)
            sim_iterations_label = f"{int(detail_iterations):,} iter"
            _heritable_risk = float(payload_stats.get("heritable_risk", 0.0) or 0.0)
            _heritable_detriment = float(payload_stats.get("heritable_detriment", 0.0) or 0.0)
            # When filters are at their defaults, use the exact simulation
            # total dose so the headline matches the Summary tab (the
            # simulation accumulates dose in float32 internally while the
            # curve is recomputed in float64, a ~0.1% difference) -- but
            # only when payload_stats and the curve describe the SAME
            # cohort, which this branch (single run, or a batch with no
            # digest) guarantees.
            _filters_default = (
                filters.get("sex") == "All"
                and filters.get("modality") == "All"
                and filters.get("region") == "All"
                and filters.get("min_age") == 0.0
                and filters.get("max_age") == 100.0
            )
            _total_dose_sv = (
                float(payload_stats.get("total_sv", 0.0) or 0.0)
                if _filters_default else None
            )
            sim_scope_note = detail_note

        sim_curve["selected_model"] = selected_model

        # Combined Bayesian curve (only if any adjustment is active).
        # Averaged across the batch whenever the digest can support every
        # active comparison, so it describes the same cohort as the
        # Simulation curve above; otherwise the detail year.
        bayes_curve = None
        bayes_is_batch = False
        if has_adjustments:
            _bayes_kwargs = dict(
                ddref_solid=ddref_solid, ddref_leuk=ddref_leuk,
                use_no_golikov_dose=cmp_golikov,
                compare_model=cmp_model,
                selected_model=selected_model,
                comparison_model=comparison_model,
                population=population,
                compare_mataac=cmp_mataac,
                sampled_yll=_sampled_yll,
            )
            bayes_is_batch = (
                n_batch_digests > 0
                and batch_bayesian_curve_supported(
                    digests, compare_model=cmp_model,
                    use_no_golikov_dose=cmp_golikov)
            )
            if bayes_is_batch:
                bayes_curve = compute_batch_averaged_bayesian_curve(
                    digests, filters, bin_width=5,
                    **{k: v for k, v in _bayes_kwargs.items() if k != "compare_model"})
            else:
                bayes_curve = compute_combined_bayesian_curve(
                    exam_data, events, detail_iterations, filters, bin_width=5,
                    exam_columns=exam_columns, event_columns=event_columns,
                    **_bayes_kwargs,
                )

        plot_filtered_risk_analysis(self.ax1, self.ax2, sim_curve, bayes_curve)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.canvas.flush_events()

        # All-years-combined Mataac strata summary, shown with the
        # competing-mortality comparison.
        # Strata are per-calendar-year by construction (the digest carries
        # the exam's own-year cumulative CT dose), so summing across year
        # groups is correct; counted over ALL exams, not the active filters.
        _strata = None
        _strata_scope = ""
        if cmp_mataac and _sim_cfg.get("use_mataac", True):
            if n_batch_digests > 0:
                _src = [d.get("exam_columns") for d in digests.values()]
                _ys = sorted(str(y) for y in digests)
                _span = f"{_ys[0]}\u2013{_ys[-1]}" if len(_ys) > 1 else _ys[0]
                _strata_scope = (f"all modalities, {n_batch_digests} year "
                                 f"group(s) ({_span})")
            elif exam_columns is not None and len(exam_columns) > 0:
                _src = [exam_columns]
                _strata_scope = "all modalities, this run"
            else:
                _src = []
            _strata = summarize_mataac_strata(_src)

        adjustments = {
            "ddref_solid": ddref_solid, "ddref_leuk": ddref_leuk,
            "ddref_changed": ddref_changed,
            "golikov": cmp_golikov,
            "model_compare": cmp_model,
            "selected_model": selected_model,
            "comparison_model": comparison_model,
            "mataac": cmp_mataac,
            "mataac_dose_groups": (bayes_curve.get("mataac_dose_groups")
                                   if bayes_curve else None),
            # True when both curves cover the same cohort, so an inversion
            # cannot be explained away as a scope mismatch.
            "bayes_scope_matches_sim": bool(bayes_is_batch or n_batch_digests == 0),
            "sim_use_mataac": bool(_sim_cfg.get("use_mataac", True)),
            "sim_use_golikov": bool(_sim_cfg.get("use_golikov", True)),
        }
        report = format_filtered_risk_summary(
            sim_curve, bayes_curve, filters, adjustments,
            heritable_risk=_heritable_risk,
            heritable_detriment=_heritable_detriment,
            total_dose_sv=_total_dose_sv,
            mataac_strata=_strata, strata_scope=_strata_scope)
        _notes = []
        if sim_scope_note:
            _notes.append(sim_scope_note)
        # Only say anything about the Bayesian overlay's scope when it can
        # differ from the Simulation curve's (i.e. a genuine batch that
        # averaged that curve): flag the single-year fallback, and confirm the
        # match when both are the batch average, since the whole point of
        # saying so is that the two curves are then comparable.
        if has_adjustments and n_batch_digests > 0:
            if bayes_is_batch:
                _notes.append(
                    "Bayesian comparison curve: equal-weighted average across the "
                    "same year groups as the Simulation curve.")
            elif detail_note:
                _what = ("risk-model comparison" if cmp_model else
                         "weight-adjustment comparison" if cmp_golikov else
                         "competing-mortality comparison")
                _notes.append(
                    f"Bayesian comparison curve: year group "
                    f"'{stats.get('_detail_year')}' only, so it is not directly "
                    f"comparable to the {stats.get('_n_years', '?')}-year average "
                    f"Simulation curve above. To make the {_what} cover every "
                    f"year, run the years as one pool (Simulation setup: "
                    f"\"Combine all years into one run\").")
        if _notes:
            report = "\n\n".join(_notes) + "\n\n" + report
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, report)

        if n_batch_digests > 0:
            _yrs = sorted(str(y) for y in digests)
            _span = f"{_yrs[0]}\u2013{_yrs[-1]}" if len(_yrs) > 1 else _yrs[0]
            _total_exams = sum(
                len(d.get("exam_columns")) for d in digests.values()
                if d.get("exam_columns") is not None)
            _scope = (f"{n_batch_digests} years ({_span}), "
                      f"{_total_exams:,} exams")
            if n_batch_digests < n_year_groups:
                _scope += (f" [{n_year_groups - n_batch_digests} year "
                           f"group(s) had no digest]")
            parts = [f"Risk analysis: {selected_model}, {_scope}"]
        else:
            parts = [f"Risk analysis: {selected_model}, {sim_iterations_label}, {len(exam_data):,} exams"]
        if sim_scope_note and n_batch_digests == 0:
            parts.append(f"year '{stats.get('_detail_year')}' only")
        if has_adjustments and detail_note and n_batch_digests > 0 and not bayes_is_batch:
            parts.append(f"Bayesian: year '{stats.get('_detail_year')}' only")
        if ddref_changed:
            parts.append(f"DDREF {ddref_solid:.1f}/{ddref_leuk:.1f}")
        if cmp_golikov:
            parts.append("no Golikov")
        if cmp_model:
            parts.append(f"vs {comparison_model}")
        if cmp_mataac:
            parts.append("no Mataac")
        self.status_var.set(" | ".join(parts))

        if self._summary_refresh_callback:
            try:
                self._summary_refresh_callback(bayes_curve, ddref_solid, ddref_leuk)
            except Exception as cb_exc:
                print(f"[Risk Analysis] Summary refresh callback error: {cb_exc}", flush=True)


