"""Tkinter results GUI for the Dose2QALY Monte Carlo simulation.

This module builds the post-simulation results window with tabbed views:
Summary, By Modality, By Body Region, By Cancer Type, Cancer Events,
Patient Cohort, Research Questions, Risk Analysis.

It runs the simulation in a background thread and streams live updates
to the GUI via a queue.  All matplotlib charts inherit the Okabe-Ito
colour-blind-safe palette from ``style.py``.

Key design choices
------------------
* **Threading model** — the heavy NumPy / CuPy work stays on a worker
  thread; Tkinter widgets are updated from the main thread via a polling
  loop (``_poll_worker_queue``).
* **Lazy tab refresh** — only the currently-visible tab redraws on each
  iteration update, so hidden tabs do not stall the event loop.
* **Paned windows** — the Summary tab uses a vertical PanedWindow so
  users can drag the sash to balance text summary vs. chart height.

Entry point
-----------
``start_gui(...)`` is called from ``main.py`` once ``run_simulation``
has returned the final stats, exam data, and cancer events.
"""
from __future__ import annotations

from collections import deque
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .analysis import (
    modality_name,
    region_by_modality,
    plot_all_regions_by_modality,
    _ensure_exam_columns,
    plot_age_histogram,
    plot_cancer_pie,
    plot_cancer_frequency_vs_burden,
    plot_iteration_histogram,
    plot_QALY_histogram,
    plot_sex_breakdown,
    plot_modality_mix,
    filter_cancer_breakdown,
    format_summary,
    format_adjustments,
    plot_sankey_diagram,
    plot_cancer_age_histogram,
    batch_completion_totals,
    resolve_cohort_examinations,
    resolve_modelled_examinations,
)
from .columns import ExamColumns, EventColumns
from .adv_ui import AdvancedRiskAnalysisPanel
from .style import (
    apply_style as _apply_style,
    SEMANTIC as _SEMANTIC,
    QUARTILE_RAMP as _QUARTILE_RAMP,
    QUARTILE_RAMP_LIGHT as _QUARTILE_RAMP_LIGHT,
)
from .research_questions import (
    compute_region_age_dose_sweep,
    rescale_region_age_sweep,
    filter_region_age_sweep,
    pool_region_age_sweeps,
    plot_region_age_sweep,
    compute_annual_benefit,
    plot_annual_QALY_bar,
    compute_annual_burden,
    plot_annual_burden_bar,
    plot_burden_vs_dose_small_multiples,
    format_burden_vs_dose_by_modality_summary,
    format_region_age_sweep_summary,
    attach_simulated_ranges,
    set_cell_interval,
    CELL_INTERVAL_COHORT,
    apply_min_scans,
    SIM_ANNUAL_RANGE_LABEL,
    DEFAULT_MIN_SCANS,
    _concat_event_columns,
)
from .columns import EventColumns as _EventColumnsForEmpty

from .preprocess_cache import set_cache_output_dir
from .constants import BACKGROUND_MSV_PER_YEAR


# Shared GUI helpers — tooltips + UX metadata

# Bounds on the annual-benefit "Dose change" field.
#
# -100 % is total dose elimination; below that the reduced dose would be
# negative, which is meaningless rather than merely extreme.  The upper bound
# is a judgement call: the risk model extrapolates linearly (plus the
# leukaemia quadratic term), so nothing breaks numerically at large
# increases, but an LNT extrapolation to many times the observed dose is not
# something this model can support.  +200 % (a tripling) is generous for the
# "what if we got sloppy" question the field exists to answer.
_DOSE_CHANGE_MIN = -100.0
_DOSE_CHANGE_MAX = 200.0


def _parse_dose_change(text: str) -> tuple[float | None, str]:
    """Parse the "Dose change" field into a signed percentage.

    Returns ``(value, "")`` on success or ``(None, message)`` describing what
    is wrong, so the caller can show it verbatim.  Accepts a leading "+",
    surrounding whitespace and a trailing "%", and tolerates the comma
    decimal separator used in Finnish locales.
    """
    raw = (text or "").strip().rstrip("%").strip().replace(",", ".")
    if not raw:
        return None, "Enter a dose change, e.g. -10 for a 10% reduction."
    try:
        val = float(raw)
    except ValueError:
        return None, (f"{text!r} is not a number.  Enter a signed "
                      "percentage such as -10, -17.5 or +20.")
    if not (_DOSE_CHANGE_MIN <= val <= _DOSE_CHANGE_MAX):
        return None, (f"Dose change must be between {_DOSE_CHANGE_MIN:g}% "
                      f"and {_DOSE_CHANGE_MAX:g}%.  {val:g}% is outside "
                      "the range this model can be extrapolated over.")
    if val == 0.0:
        return None, ("A 0% dose change has no effect, so there is nothing "
                      "to plot.  Use a negative value for a reduction or a "
                      "positive one for an increase.")
    return val, ""


class _ToolTip:
    """Lightweight floating hint, similar to a native status-bar tooltip.

    One instance is enough for the whole window — we re-use the same
    Toplevel for tab-strip hovers and Treeview-header hovers so at most
    one hint is ever on screen.  ``show`` with the same text only
    reposition the popup instead of destroying and rebuilding it, which
    keeps ``<Motion>`` handlers cheap.
    """

    __slots__ = ("widget", "tipwindow", "_current_text")

    def __init__(self, widget: tk.Widget):
        self.widget = widget
        self.tipwindow: tk.Toplevel | None = None
        self._current_text: str = ""

    def show(self, text: str, x_root: int, y_root: int) -> None:
        """Show the tooltip with ``text`` near the given root-window
        coordinates; an unchanged text just follows the cursor."""
        text = str(text or "").strip()
        if not text:
            self.hide()
            return
        # Already showing this exact text → just follow the cursor.
        if self.tipwindow is not None and self._current_text == text:
            try:
                self.tipwindow.wm_geometry(f"+{int(x_root) + 14}+{int(y_root) + 20}")
            except Exception:
                self.hide()
            return
        self.hide()
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{int(x_root) + 14}+{int(y_root) + 20}")
        try:
            tw.wm_attributes("-topmost", True)
        except Exception:
            pass
        label = tk.Label(
            tw, text=text,
            background="#FFFFE0", foreground="#222",
            relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 9),
            padx=7, pady=4,
            justify=tk.LEFT, wraplength=460,
        )
        label.pack(ipadx=1)
        self.tipwindow = tw
        self._current_text = text

    def hide(self) -> None:
        """Destroy the tooltip window and forget the shown text."""
        if self.tipwindow is not None:
            try:
                self.tipwindow.destroy()
            except Exception:
                pass
        self.tipwindow = None
        self._current_text = ""


# One-line hover hints keyed by visible tab label.
_TAB_HINTS: dict[str, str] = {
    "Summary":              "Headline numbers · per-year cancer & death histograms · cancer-type pie.",
    "By Modality":          "Dose and risk broken down by imaging modality (CT · MG · XA · …).",
    "By Body Region":       "Dose and risk broken down by anatomical region (chest · abdomen · head · …).",
    "By Cancer Type":       "Per-cancer summary, per-Sv detriment table, and the outcome flow diagram (Sankey).",
    "Cancer Events":        "Age distribution of every simulated cancer (all input files) · plus a capped, debugging-only sample of individual event rows.",
    "Patient Cohort":       "Who was exposed: age pyramid, sex split, modality mix.",
    "Research Questions":   "Dose-reduction benefit per region & 10-year age group · cancers / fatal cancers / QALY vs examination dose.",
    "Risk Analysis":        "Filter by sex / age / modality · Bayesian DDREF sensitivity · subgroup risk.",
    "Per-Year Comparison":  "Multi-year batches only: one row per year group plus the averaged row.",
}


# Column-header tooltips keyed by internal column ID.
_MOD_TREE_TOOLTIPS: dict[str, str] = {
    "mod":    "Imaging modality (e.g. CT, MG, XA).",
    "scans":  "Average number of examinations of this modality per simulated year.",
    "dose":   "Collective effective dose delivered by this modality's risk-modelled exams per simulated year (Sv) — sum of per-exam effective doses for exams with an organ-dose model.",
    "edps":   "Mean effective dose per exam (mSv) = collective dose / number of risk-modelled exams (exams with no organ-dose model or Non-Dose regions are excluded from the denominator).",
    "cases":  "Average radiation-induced cancers attributed to this modality per simulated year (ICRP 152 REIC or LAR, cohort-sampled).",
    "deaths": "Average radiation-induced cancer deaths attributed to this modality per simulated year (cases x ICRP 152 Table 3.9 lethality k).",
    "det":    "ICRP 152 cancer + heritable detriment per simulated year — cases weighted by lethality k, minimum QOL q, and relative years of life lost l_rel (Table 3.9).",
    "crpv":   "Cancer-incidence risk per collective Sv (%). 100 x cases / collective dose; lower = fewer induced cancers per Sv of population dose.",
    "drpv":   "Cancer-death risk per collective Sv (%). 100 x deaths / collective dose; sampled value shown with deterministic REIC expectation in parentheses.",
    "dps":    "Detriment caused per exam, expressed in hours of disability-adjusted life lost (detriment x 365.25 x 24 / risk-modelled exams; exams with no organ-dose model or Non-Dose regions are excluded from the denominator). A clinically meaningful per-exam harm unit.",
}

# Row cap for the Cancer Events "Table" subtab.
#
# The table is a DEBUGGING view: one row per simulated Monte Carlo draw, kept
# so a developer can eyeball that ages, organ doses and outcomes look sane.
# It is not a sample of anything -- a row is a draw, not a patient -- so there
# is no analytical reason to hold the whole stream in a second copy just to
# render it.  A full run can produce hundreds of thousands of events, and the
# tab was retaining every one of them (a duplicate of ``results["events"]``,
# already the memory high-water mark) purely to fill a Treeview that stops
# painting at 10,000 rows anyway.
#
# Beyond this cap the tab stops accumulating entirely -- events keep flowing
# into every statistic, breakdown and chart unchanged; only the per-row copy
# stops.  The Age Distribution subtab therefore reads ``results["events"]``
# (every event, every input file / year group), never this store.
MAX_EVENT_TABLE_ROWS = 1000


def _head_events(evts, n: int):
    """First ``n`` events of a columnar bundle or a list, without copying more.

    ``EventColumns.subset`` takes any numpy index, so a slice gives a view-like
    head without materialising a boolean mask over the whole stream -- which
    matters here precisely because the stream is the thing we are refusing to
    hold on to.
    """
    if n <= 0:
        return _EventColumnsForEmpty.empty() if isinstance(evts, EventColumns) else []
    if isinstance(evts, EventColumns):
        return evts if len(evts) <= n else evts.subset(slice(0, n))
    return list(evts)[:n]


_EV_TREE_TOOLTIPS: dict[str, str] = {
    "cancer":    "Cancer type that manifested in this simulated event.",
    "age_exp":   "Patient age at the exposure that induced this cancer.",
    "age_death": "Age at death if the cancer was fatal; otherwise the patient's modelled life-end age.",
    "life_lost": "Years of life lost relative to the patient's expected lifespan.",
    "sex":       "Patient sex (M / F / U).",
    "mod":       "Imaging modality that delivered the dose.",
    "region":    "Body region imaged.",
    "proto":     "Protocol / exam description.",
    "ed":        "Whole-body effective dose of the exam (mSv).",
    "outcome":   "Whether the simulated patient survived or died from the cancer.",
}


def _bind_header_tooltips(tree: ttk.Treeview,
                          tooltips: dict[str, str],
                          holder: _ToolTip) -> None:
    """Show a column-specific tooltip when the cursor hovers a tree header.

    Uses ``tree.identify_region`` + ``tree.identify_column`` so the hit
    detection automatically follows the user's current column widths
    and horizontal scroll position.  ``last_col`` short-circuits the
    common case of the mouse drifting inside a single header cell, so
    we only rebuild the Toplevel when the column actually changes.
    """
    cols = list(tree["columns"])
    last_col: list[str | None] = [None]

    def _on_motion(event):
        if tree.identify_region(event.x, event.y) != "heading":
            if last_col[0] is not None:
                holder.hide()
                last_col[0] = None
            return
        colid = tree.identify_column(event.x)  # "#1", "#2", ...
        try:
            idx = int(str(colid).lstrip("#")) - 1
        except ValueError:
            return
        if not (0 <= idx < len(cols)):
            return
        name = cols[idx]
        hint = tooltips.get(name)
        if not hint:
            if last_col[0] is not None:
                holder.hide()
                last_col[0] = None
            return
        try:
            header = str(tree.heading(name, "text") or name)
        except Exception:
            header = str(name)
        holder.show(f"{header} — {hint}", event.x_root, event.y_root)
        last_col[0] = name

    def _on_leave(_event):
        holder.hide()
        last_col[0] = None

    tree.bind("<Motion>", _on_motion, add="+")
    tree.bind("<Leave>",  _on_leave,  add="+")


def _safe_iterations(stats: dict) -> int:
    try:
        return max(1, int((stats or {}).get("iterations", 0) or 0))
    except Exception:
        return 1


def _per_iteration_value(value, iterations: int) -> float:
    try:
        return float(value or 0.0) / max(1, int(iterations or 0))
    except Exception:
        return 0.0


def _format_backend_summary(backend_info: dict | None) -> str:
    info = backend_info or {}
    backend = str(info.get("backend", "CPU") or "CPU")
    cpu_available = int(info.get("cpu_available", 1) or 1)
    gpu_available = int(info.get("gpu_available", 0) or 0)
    cpu_in_use = int(info.get("cpu_in_use", 0) or 0)
    gpu_in_use = int(info.get("gpu_in_use", 0) or 0)
    return (
        f"Backend: {backend} | "
        f"CPU available: {cpu_available}, in use: {cpu_in_use} | "
        f"GPU available: {gpu_available}, in use: {gpu_in_use}"
    )


def _show_missing_ct_dialog(parent, missing_summary, continue_event, continue_result,
                             suppress_state=None):
    """Modal dialog reporting missing CT organ doses; blocks worker via threading.Event.

    ``suppress_state``, when given, is a mutable dict (typically
    ``{"suppress": False}`` owned by the caller). If the user ticks
    "Don't show this again for this simulation" and continues, the dict is
    updated in place so later ``missing_ct_doses`` events in the same
    simulation run (e.g. when a batch moves on to another input file) are
    auto-continued without popping this dialog up again.
    """
    dlg = tk.Toplevel(parent)
    dlg.title("Missing CT Organ Doses")
    _place_window(dlg, 700, 540)
    dlg.resizable(True, True)
    try:
        dlg.transient(parent)
        dlg.grab_set()
    except Exception:
        pass

    header = ttk.Frame(dlg, padding=(12, 10, 12, 4))
    header.pack(fill=tk.X)
    ttk.Label(
        header,
        text="\u26a0  Missing CT Organ Doses Detected",
        font=("Segoe UI", 11, "bold"),
        foreground="#b85c00",
    ).pack(anchor=tk.W)
    ttk.Label(
        header,
        text=(
            "The following CT regions have exams with no organ dose data (no per-organ dose columns found).\n"
            "If you continue, these exams will be EXCLUDED from the simulation so they do not\n"
            "dilute per-Sv and per-scan risk averages."
        ),
        font=("Segoe UI", 9),
        wraplength=660,
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(2, 0))

    ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=6)

    frame_body = ttk.Frame(dlg, padding=(12, 0, 12, 4))
    frame_body.pack(fill=tk.BOTH, expand=True)
    txt = scrolledtext.ScrolledText(
        frame_body, font=("Consolas", 9), wrap=tk.WORD, height=14, state=tk.NORMAL
    )
    txt.pack(fill=tk.BOTH, expand=True)

    total_exams = sum(v["n_exams"] for v in missing_summary.values())
    total_patients = sum(v["n_patients"] for v in missing_summary.values())
    txt.insert(
        tk.END,
        f"Total: {total_exams:,} CT exams missing organ doses "
        f"across {total_patients:,} patients\n\n",
    )
    for region, info in sorted(missing_summary.items(), key=lambda x: -x[1]["n_exams"]):
        n_ex = info["n_exams"]
        n_pt = info["n_patients"]
        organs = info.get("expected_organs", [])
        txt.insert(tk.END, f"  Region : {region}\n")
        txt.insert(tk.END, f"    Exams without organ dose : {n_ex:,}\n")
        txt.insert(tk.END, f"    Unique patients affected : {n_pt:,}\n")
        if organs:
            shown = ", ".join(organs[:10])
            if len(organs) > 10:
                shown += f", \u2026 (+{len(organs) - 10} more)"
            txt.insert(tk.END, f"    Missing organ doses       : {shown}\n")
        txt.insert(tk.END, "\n")
    txt.config(state=tk.DISABLED)

    dont_ask_var = tk.BooleanVar(value=False)
    if suppress_state is not None:
        ttk.Checkbutton(
            dlg,
            text="Don't show this again for this simulation",
            variable=dont_ask_var,
        ).pack(anchor=tk.W, padx=12, pady=(0, 2))

    ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=4)
    btn_frame = ttk.Frame(dlg, padding=(12, 4, 12, 10))
    btn_frame.pack(fill=tk.X)
    ttk.Label(
        btn_frame,
        text="Do you want to continue the simulation?",
        font=("Segoe UI", 9, "bold"),
    ).pack(side=tk.LEFT, padx=(0, 20))

    def _on_continue():
        continue_result[0] = True
        if suppress_state is not None and dont_ask_var.get():
            suppress_state["suppress"] = True
        dlg.destroy()
        if continue_event:
            continue_event.set()

    def _on_cancel():
        continue_result[0] = False
        dlg.destroy()
        if continue_event:
            continue_event.set()

    ttk.Button(btn_frame, text="Continue", command=_on_continue).pack(side=tk.RIGHT, padx=(6, 0))
    ttk.Button(btn_frame, text="Cancel", command=_on_cancel).pack(side=tk.RIGHT)
    dlg.protocol("WM_DELETE_WINDOW", _on_cancel)
    dlg.focus_set()


def _show_missing_columns_note(parent, missing_per_file, startup_population=None):
    """Non-blocking info window for the ``missing_columns`` progress event.

    Mirrors the ``[NOTE]`` text ``excel_io.report_missing_headers`` writes to
    stderr plus the per-column consequence lines; informational only (no
    continue/cancel decision), so unlike ``_show_missing_ct_dialog`` it does
    not grab or block the worker.
    """
    from .missing_columns import format_consequences, format_per_file_summary

    dlg = tk.Toplevel(parent)
    dlg.title("Missing Columns")
    _place_window(dlg, 700, 340)
    dlg.resizable(True, True)
    try:
        dlg.transient(parent)
    except Exception:
        pass

    header = ttk.Frame(dlg, padding=(12, 10, 12, 4))
    header.pack(fill=tk.X)
    ttk.Label(
        header,
        text="Some expected columns are absent from the input file(s)",
        font=("Segoe UI", 11, "bold"),
        foreground="#b85c00",
    ).pack(anchor=tk.W)
    ttk.Label(
        header,
        text="Columns listed below are absent; the consequences are:",
        font=("Segoe UI", 9),
        wraplength=660,
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(2, 0))

    ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=6)

    frame_body = ttk.Frame(dlg, padding=(12, 0, 12, 4))
    frame_body.pack(fill=tk.BOTH, expand=True)
    txt = scrolledtext.ScrolledText(
        frame_body, font=("Consolas", 9), wrap=tk.WORD, height=10, state=tk.NORMAL
    )
    txt.pack(fill=tk.BOTH, expand=True)
    txt.insert(tk.END, format_per_file_summary(missing_per_file) or "(none)")
    consequences = format_consequences(
        missing_per_file, startup_population=startup_population)
    if consequences:
        txt.insert(tk.END, f"\n\n{consequences}")
    txt.config(state=tk.DISABLED)

    btn_frame = ttk.Frame(dlg, padding=(12, 4, 12, 10))
    btn_frame.pack(fill=tk.X)
    ttk.Button(btn_frame, text="OK", command=dlg.destroy).pack(side=tk.RIGHT)
    dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)


# Multi-year batch banner

def _format_batch_note(stats: dict) -> str:
    """One-paragraph description of a multi-year batch run.

    Returns an empty string for ordinary single-population runs, so callers
    can insert the result unconditionally.  Failed years are never hidden.
    """
    if not isinstance(stats, dict):
        return ""
    mode = stats.get("_run_mode")
    if mode != "per_year_average":
        return ""
    n = stats.get("_n_years", "?")
    years = ", ".join(sorted(stats.get("_year_batches") or {}))
    lines = [f"AVERAGE ACROSS {n} YEAR GROUP(S): {years}",
             "Dose, cancers, deaths, detriment and QALYs below are per-year "
             "means only (see the Per-Year Comparison tab).  Dataset size "
             "(acquisitions/examinations/patients) is shown both as the "
             "total across all years and as a per-year average."]
    detail_year = stats.get("_detail_year")
    if detail_year:
        lines.append(
            f"Per-exam detail (Cancer Events / Patient Cohort / Dose "
            f"Reduction) is from "
            f"'{detail_year}' only \u2014 to inspect one year in full, run a "
            f"batch that contains only that year group.  Two tabs are the "
            f"exception: the "
            f"Research Questions tab's Region \u00d7 Age dose-reduction sweep "
            f"and burden-vs-examination-dose curves are computed from "
            f"every year group pooled as one cohort (analytical ICRP 152 "
            f"baseline), per modality as well as overall "
            f"(see the notes at the top of each sub-tab); and the Risk "
            f"Analysis tab's \"Simulation\" curve/headline is likewise a "
            f"true equal-weighted average across every year group for any "
            f"filter combination \u2014 only its optional Bayesian "
            f"comparison overlay (DDREF / model / Mataac / Golikov "
            f"adjustments) remains '{detail_year}'-only, since it needs "
            f"per-exam fields the batch does not retain across years."
        )
    failed = stats.get("_year_batch_failures") or {}
    if failed:
        lines.append("EXCLUDED \u2014 {} year(s) failed: {}".format(
            len(failed), "; ".join(f"{y}: {msg}" for y, msg in failed.items())))
    return "\n".join(lines)


# Slide-sized figure export
# FigureCanvasTkAgg resizes the figure's size-in-inches to the widget's pixel
# size, so on a big / high-DPI screen the live Summary-tab figures can be 30-40
# inches wide while their fonts stay fixed in points. Exported that way the
# panels are physically huge with relatively tiny text, and once shrunk onto a
# PowerPoint slide the text is unreadable. These helpers re-render the figure
# at a fixed slide-friendly size with scaled fonts onto a throwaway Agg canvas
# so the Tk canvas never draws a mismatched-size image, then restore every
# piece of on-screen state. They are used only by the Summary-tab save
# buttons (``_save_subplot`` / ``_save_full_figure_as_shown`` below).

_EXPORT_PANEL_W = 8.0     # inches per axes panel
_EXPORT_PANEL_H = 5.0
_EXPORT_FONT_SCALE = 1.35


def _export_text_artists(fig):
    """Yield every non-tick text artist on ``fig`` whose fontsize the slide
    export scales and restores (tick labels are handled separately via
    ``tick_params`` because the locator re-creates them on resize).

    Legend texts are deliberately left out: the Summary legends are already
    sized for slides on screen (``analysis._SUMMARY_LEGEND_FS``), and
    scaling them further makes the three-line legend cover the histogram
    peak in the narrow export panel."""
    for ax in fig.axes:
        for txt in ax.texts:
            yield txt
        yield ax.title
        for name in ("_left_title", "_right_title"):
            t = getattr(ax, name, None)
            if t is not None and t.get_text():
                yield t
        if ax.xaxis.label is not None:
            yield ax.xaxis.label
        if ax.yaxis.label is not None:
            yield ax.yaxis.label
    for txt in fig.texts:
        yield txt
    supt = getattr(fig, "_suptitle", None)
    if supt is not None and supt.get_text():
        yield supt


def _apply_slide_font_scale(fig, font_scale):
    """Scale every text artist on ``fig`` by ``font_scale``.

    Returns ``(text_saved, tick_saved)`` for :func:`_restore_slide_fonts`.
    Tick labels are scaled via ``tick_params`` (robust to the locator
    re-creating ticks after the resize); the per-label bold the plotting
    functions set is re-applied here so ticks the locator yields at the
    export size stay bold.
    """
    text_saved = [(a, a.get_fontsize()) for a in _export_text_artists(fig)]
    for artist, sz in text_saved:
        artist.set_fontsize(sz * font_scale)
    tick_saved = []
    for ax in fig.axes:
        for axname, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            labels = axis.get_ticklabels()
            if not labels:
                continue
            old = labels[0].get_fontsize()
            tick_saved.append((ax, axname, old))
            ax.tick_params(axis=axname, labelsize=old * font_scale)
            for lbl in labels:
                lbl.set_fontweight("bold")
    return text_saved, tick_saved


def _restore_slide_fonts(text_saved, tick_saved):
    for artist, sz in text_saved:
        artist.set_fontsize(sz)
    for ax, axname, old in tick_saved:
        ax.tick_params(axis=axname, labelsize=old)


def _shrink_overflowing_titles(fig, renderer, title_sizes):
    """Shrink any visible axes title that overflows its axes width.

    Long titles such as "Contribution to Total QALY Burden by Cancer Type" or
    "Cancer Count Distribution (per 100,000 examinations)" would overflow an
    8-inch panel at the scaled fontsize; each is shrunk by the overflow ratio
    but never below its original (unscaled) size.
    """
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        title = ax.title
        if title is None or not title.get_text():
            continue
        tw = title.get_window_extent(renderer).width
        aw = ax.get_window_extent(renderer).width
        if aw <= 0 or tw <= 0.98 * aw:
            continue
        unscaled = title_sizes.get(id(title))
        if unscaled is None:
            continue
        current = title.get_fontsize()
        new = max(current * (0.98 * aw / tw), unscaled)
        if new < current:
            title.set_fontsize(new)


def export_axes_for_slides(fig, ax, path, *, panel_w=_EXPORT_PANEL_W,
                           panel_h=_EXPORT_PANEL_H,
                           font_scale=_EXPORT_FONT_SCALE, dpi=300):
    """Export a single Axes of a Tk-embedded figure to a slide-sized file.

    Re-renders ``ax`` onto a throwaway Agg canvas at
    ``(n_axes * panel_w, panel_h)`` inches with all fonts scaled by
    ``font_scale``, hides every other axes so a neighbour cannot bleed into
    the crop, and saves the axes' tight bbox. Every piece of on-screen state
    (canvas, figure size, fonts, tick sizes, axes visibility, constrained
    layout flag) is restored afterwards, so the live Tk figure is untouched.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    orig_canvas = fig.canvas
    orig_size = fig.get_size_inches()
    n_axes = len(fig.axes)
    export_size = (n_axes * panel_w, panel_h)
    _was_cl = fig.get_constrained_layout()

    text_saved, tick_saved = _apply_slide_font_scale(fig, font_scale)
    title_objs = {id(a.title) for a in fig.axes}
    title_sizes = {id(a): sz for a, sz in text_saved if id(a) in title_objs}

    vis_saved = [(other, other.get_visible())
                 for other in fig.axes if other is not ax]

    tmp = FigureCanvasAgg(fig)
    try:
        fig.set_constrained_layout(False)
        fig.set_size_inches(*export_size, forward=False)
        for other, _ in vis_saved:
            other.set_visible(False)

        tmp.draw()
        renderer = tmp.get_renderer()
        _shrink_overflowing_titles(fig, renderer, title_sizes)
        tmp.draw()
        renderer = tmp.get_renderer()

        tight_bbox = ax.get_tightbbox(renderer)
        if tight_bbox is None:
            fig.savefig(path, dpi=dpi, facecolor="white")
        else:
            # matplotlib only applies ``pad_inches`` when ``bbox_inches`` is
            # ``"tight"``; an explicit Bbox is saved as-is, so pad it here.
            bbox_in = tight_bbox.transformed(fig.dpi_scale_trans.inverted())
            bbox_in = bbox_in.padded(0.15)
            fig.savefig(path, bbox_inches=bbox_in, dpi=dpi,
                        pad_inches=0.15, facecolor="white")
    finally:
        _restore_slide_fonts(text_saved, tick_saved)
        for other, v in vis_saved:
            other.set_visible(v)
        fig.set_size_inches(*orig_size, forward=False)
        fig.set_constrained_layout(_was_cl)
        fig.set_canvas(orig_canvas)
        orig_canvas.draw_idle()



def export_figure_as_shown(fig, path, *, dpi=300, tight=False):
    """Save ``fig`` exactly as it appears on screen (same inch size, same
    constrained layout, same fonts) at higher resolution."""
    kw = dict(dpi=dpi, facecolor=fig.get_facecolor())
    if tight:
        kw.update(bbox_inches="tight", pad_inches=0.15)
    fig.canvas.draw()          # make sure layout is solved at the current size
    fig.savefig(path, **kw)


def export_axes_as_shown(fig, ax, path, *, dpi=300, pad_in=0.12):
    """Crop a single panel out of the live figure without altering it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer)
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(pad_in)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox,
                facecolor=fig.get_facecolor())


def _rng(lo, hi) -> str:
    """'[lo, hi]' or 'n/a' when no simulated series was recorded."""
    try:
        if not (np.isfinite(float(lo)) and np.isfinite(float(hi))):
            return "n/a"
    except (TypeError, ValueError):
        return "n/a"
    return f"[{lo:.3f}, {hi:.3f}]"


# Moved out of start_gui: only used by _modality_filename_prefix.
_MODALITY_ABBREV = {
    "ct": "ct",
    "radiography": "radio",
    "mammography": "mammo",
    "dental cbct": "dentalcbct",
    "dental imaging": "dentalcbct",
}


def _attach_toolbar(canvas, parent, use_before=True):
    """Pack a compact matplotlib NavigationToolbar above ``canvas``.

    Uses Tk's ``before=`` argument so the toolbar appears on top of
    an already-packed canvas widget without having to rearrange the
    existing pack order.  Returns the toolbar so callers can style
    or hide it if they wish.

    Pass ``use_before=False`` when the canvas is not a pack sibling of
    ``parent`` -- e.g. it lives in a nested size-controlling frame, so
    ``before=`` would raise -- and ``parent`` is instead an empty frame
    positioned where the toolbar should go.
    """
    try:
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        if use_before:
            toolbar.pack(side=tk.TOP, fill=tk.X,
                         before=canvas.get_tk_widget())
        else:
            toolbar.pack(side=tk.TOP, fill=tk.X)
    except Exception:
        # Matplotlib fallback: auto-packed toolbar still provides zoom/pan/export.
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
    return toolbar


def _place_window(win, w, h, min_w=400, min_h=300, max_screen_frac=(0.96, 0.90)):
    """Open ``win`` at ``w``x``h`` clamped to the screen, centred, with a minsize."""
    win.update_idletasks()
    scr_w, scr_h = win.winfo_screenwidth(), win.winfo_screenheight()
    w = min(w, int(scr_w * max_screen_frac[0]))
    h = min(h, int(scr_h * max_screen_frac[1]))
    win.geometry(f"{w}x{h}+{max(0, (scr_w - w) // 2)}+{max(0, (scr_h - h) // 4)}")
    win.minsize(min(min_w, w), min(min_h, h))
    return w, h


def _fit_window_to_content(win, min_w=900, min_h=600,
                           max_screen_frac=(0.96, 0.90)):
    """Size ``win`` to what its children actually request, then clamp.

    Hard-coded ``geometry()`` strings go stale: a window holding a
    figure canvas plus a text summary plus a toolbar needs the sum of
    their heights, and that sum changes whenever the figure size, the
    text widget's row count or the font changes.  When the geometry is
    too small the canvas is the widget that gives (it is the one packed
    with ``expand=True``), so the figure arrives squashed and the user
    has to drag the window before the plot is readable.

    Asking Tk for the requested size instead means the window is right
    by construction.  The result is clamped to a fraction of the screen
    so a large figure cannot open a window taller than the display, and
    ``minsize`` keeps it from being dragged down to nothing.
    """
    win.update_idletasks()
    scr_w, scr_h = win.winfo_screenwidth(), win.winfo_screenheight()
    w = min(win.winfo_reqwidth(), int(scr_w * max_screen_frac[0]))
    h = min(win.winfo_reqheight(), int(scr_h * max_screen_frac[1]))
    w, h = max(w, min(min_w, scr_w)), max(h, min(min_h, scr_h))
    # Centre horizontally, and sit high vertically -- centring a tall
    # window puts its title bar off the top of the screen.
    win.geometry(f"{w}x{h}+{max(0, (scr_w - w) // 2)}"
                 f"+{max(0, (scr_h - h) // 4)}")
    win.minsize(min(min_w, w), min(min_h, h))
    return w, h


def _modality_filename_prefix(modality_region_map):
    """Build a filename-friendly prefix from the modalities shown in
    a chart, e.g. "ct" or "radio_mammo_dentalcbct", so the exported
    file name identifies which modalities it covers.
    """
    parts = []
    for mod in (modality_region_map or {}):
        key = str(mod).strip().lower()
        abbrev = _MODALITY_ABBREV.get(key)
        if abbrev is None:
            abbrev = "".join(ch for ch in key if ch.isalnum())
        if abbrev and abbrev not in parts:
            parts.append(abbrev)
    return "_".join(parts) if parts else "all"


def _save_subplot(fig, ax, default_name):
    """Save a single subplot (axes) from a multi-axes figure to a file,
    re-rendered at slide-friendly size/fonts via
    :func:`export_axes_for_slides`."""
    from tkinter import filedialog
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        initialfile=default_name,
        filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
    )
    if not path:
        return
    export_axes_for_slides(fig, ax, path)



def _save_subplot_as_shown(fig, ax, default_name):
    """Save a single subplot exactly as it looks on screen
    (:func:`export_axes_as_shown`)."""
    from tkinter import filedialog
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        initialfile=default_name,
        filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
    )
    if not path:
        return
    export_axes_as_shown(fig, ax, path)


def _save_full_figure_as_shown(fig, default_name):
    """Save the whole figure exactly as it looks on screen
    (:func:`export_figure_as_shown`)."""
    from tkinter import filedialog
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        initialfile=default_name,
        filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
    )
    if not path:
        return
    export_figure_as_shown(fig, path)


def _compute_modality_filtered_results(mod_code, s, ev, ed):
    """Recompute Summary-tab inputs restricted to one modality.

    Returns ``(filtered_stats, filtered_events, filtered_cl, filtered_dl,
    filtered_exam_data)``.  Scalar totals come from
    ``modality_stats[mod]`` (already accumulated across iterations);
    ``cancer_breakdown`` and the per-iteration case/death/QALY/life lists
    come from ``stats["_cancer_bd_by_modality"]`` /
    ``stats["_iter_lists_by_modality"]`` when present (a
    ``per_year_average`` batch, pooled across every year by
    year_batch.py), otherwise are recomputed from ``ev`` so the
    distributions reflect only the selected modality.
    """
    iterations = _safe_iterations(s)
    ms = ((s or {}).get("modality_stats", {}) or {}).get(mod_code, {}) or {}

    # Filter events (normalised to EventColumns) to the selected modality.
    if ev is not None and len(ev) and not isinstance(ev, EventColumns):
        ev = EventColumns.from_events(list(ev))
    if ev is None or len(ev) == 0:
        filtered_ev = _EventColumnsForEmpty.empty()
    else:
        mod_arr = np.asarray(ev.modality)
        mask = (mod_arr == mod_code)
        if not mask.any():
            filtered_ev = _EventColumnsForEmpty.empty()
        else:
            idx = np.nonzero(mask)[0]
            filtered_ev = EventColumns(
                cancer_type=ev.cancer_type[idx],
                modality=ev.modality[idx],
                region=ev.region[idx],
                protocol=ev.protocol[idx],
                age_at_exposure=ev.age_at_exposure[idx],
                sex=ev.sex[idx],
                died=ev.died[idx],
                exam_index=ev.exam_index[idx],
                iteration=ev.iteration[idx],
                ed_msv=ev.ed_msv[idx],
                QALY_lost=ev.QALY_lost[idx],
                life_lost_years=ev.life_lost_years[idx],
                age_at_death=ev.age_at_death[idx],
                projected_lifetime=ev.projected_lifetime[idx],
            )


    # Recompute cancer_breakdown and per-iteration lists from the
    # filtered events -- UNLESS this is a "per_year_average" batch,
    # where ``ev`` is only the last processed year's (see
    # year_batch.py's memory note) and year_batch.py has already
    # precomputed the same breakdown pooled across every year (see
    # _per_modality_breakdown / _pool_iter_lists_by_modality there).
    # Prefer that so the modality filter doesn't silently revert to a
    # single year while the unfiltered ("All modalities") view already
    # covers the whole batch.
    _by_mod_cbd = (s or {}).get("_cancer_bd_by_modality")
    _by_mod_iters = (s or {}).get("_iter_lists_by_modality")
    if _by_mod_cbd is not None and _by_mod_iters is not None:
        cancer_bd = dict(_by_mod_cbd.get(mod_code) or {})
        _entry = _by_mod_iters.get(mod_code) or {}
        filtered_cl = list(_entry.get("iter_cases") or ([0] * max(0, iterations)))
        filtered_dl = list(_entry.get("iter_deaths") or ([0] * max(0, iterations)))
        filtered_qaly = list(_entry.get("iter_qaly") or ([0.0] * max(0, iterations)))
        filtered_life = list(_entry.get("iter_life") or ([0.0] * max(0, iterations)))
    else:
        cancer_bd = {}
        filtered_cl = [0] * max(0, iterations)
        filtered_dl = [0] * max(0, iterations)
        filtered_qaly = [0.0] * max(0, iterations)
        filtered_life = [0.0] * max(0, iterations)
        if len(filtered_ev) > 0:
            iters = np.asarray(filtered_ev.iteration)
            died = np.asarray(filtered_ev.died)
            ctypes = np.asarray(filtered_ev.cancer_type)
            qaly = np.asarray(filtered_ev.QALY_lost, dtype=float)
            life = np.asarray(filtered_ev.life_lost_years, dtype=float)
            for i in range(len(filtered_ev)):
                ct = ctypes[i]
                rec = cancer_bd.setdefault(ct, {"developed": 0, "deaths": 0})
                rec["developed"] += 1
                if bool(died[i]):
                    rec["deaths"] += 1
            if iterations > 0:
                valid = (iters >= 0) & (iters < iterations)
                if valid.any():
                    bin_idx = iters[valid].astype(np.int64)
                    filtered_cl = np.bincount(bin_idx, minlength=iterations).tolist()
                    death_w = died[valid].astype(np.int64)
                    filtered_dl = np.bincount(bin_idx, weights=death_w,
                                              minlength=iterations).tolist()
                    filtered_qaly = np.bincount(bin_idx, weights=qaly[valid].astype(float),
                                                minlength=iterations).tolist()
                    filtered_life = np.bincount(bin_idx, weights=life[valid].astype(float),
                                                minlength=iterations).tolist()

    # Filter exam_data to the modality.
    filtered_ed = [ex for ex in (ed or []) if str(ex.get("modality", "") or "") == mod_code]

    # Build a filtered stats dict: start from the global stats so all
    # structural keys (sim_config, _run_mode, etc.) survive, then override
    # the modality-specific scalars.
    filtered_s = dict(s)
    filtered_s["cancer_breakdown"] = cancer_bd
    filtered_s["cancer_cases"] = float(sum(v["developed"] for v in cancer_bd.values()))
    filtered_s["cancer_deaths"] = float(sum(v["deaths"] for v in cancer_bd.values()))
    filtered_s["QALY_lost"] = float(ms.get("QALY", 0.0) or 0.0)
    filtered_s["life_lost"] = float(ms.get("life_lost", 0.0) or 0.0)
    filtered_s["total_detriment"] = float(ms.get("detriment", 0.0) or 0.0)
    filtered_s["total_sv"] = float(ms.get("dose_sv", 0.0) or 0.0)
    filtered_s["expected_deaths"] = float(ms.get("expected_deaths", 0.0) or 0.0)
    # acquisitions / acquisition_rows: keep consistent with modality_stats
    # so "Unique examinations" reads as this modality's exam count.
    filtered_s["acquisitions"] = float(ms.get("scans", 0) or 0)
    filtered_s["acquisition_rows"] = len(filtered_ed)
    # expected_incidence is a deterministic REIC total that is not tracked
    # per modality; scale it by the modality's dose share so the display
    # stays roughly consistent with the filtered cohort.
    total_dose = float(s.get("total_sv", 0.0) or 0.0)
    if total_dose > 0:
        filtered_s["expected_incidence"] = (
            float(s.get("expected_incidence", 0.0) or 0.0)
            * (filtered_s["total_sv"] / total_dose)
        )
    else:
        filtered_s["expected_incidence"] = 0.0
    # Heritable detriment is not attributed per modality; drop it from the
    # filtered view so the cancer-only totals stay internally consistent.
    filtered_s["heritable_risk"] = 0.0
    filtered_s["heritable_detriment"] = 0.0
    # Per-iteration QALY / life lists for the QALYs tab.
    filtered_s["iter_QALY_list"] = filtered_qaly
    filtered_s["iter_life_list"] = filtered_life
    # Patient aggregates must be recomputed for the filtered cohort.
    filtered_s.pop("_pid_aggregates", None)
    filtered_s.pop("_pid_source_counts", None)
    filtered_s["_summary_modality_filter"] = mod_code
    # Restrict the per-modality structures to the selected modality so
    # section 1's "Risk-modelled exams" line and sections 2/3's tables
    # describe only this modality (otherwise the global modality_stats
    # would leak every modality into the filtered view). Shallow copies
    # only -- never mutate the original ``s``.
    filtered_s["modality_stats"] = {mod_code: ms}
    _glob_exams = (s or {}).get("examinations_by_modality") or {}
    filtered_s["examinations_by_modality"] = (
        {mod_code: _glob_exams[mod_code]} if mod_code in _glob_exams else {}
    )
    _glob_excl = (s or {}).get("excluded_no_organ_dose_by_modality") or {}
    filtered_s["excluded_no_organ_dose_by_modality"] = (
        {mod_code: _glob_excl[mod_code]} if mod_code in _glob_excl else {}
    )
    _yb = (s or {}).get("_year_batches")
    if isinstance(_yb, dict):
        filtered_s["_year_batches"] = {
            yr: {
                **ys,
                "examinations_by_modality": (
                    {mod_code: (ys.get("examinations_by_modality") or {})[mod_code]}
                    if mod_code in (ys.get("examinations_by_modality") or {})
                    else {}
                ),
                "excluded_no_organ_dose_by_modality": (
                    {mod_code: (ys.get("excluded_no_organ_dose_by_modality") or {})[mod_code]}
                    if mod_code in (ys.get("excluded_no_organ_dose_by_modality") or {})
                    else {}
                ),
            }
            for yr, ys in _yb.items()
        }
    return filtered_s, filtered_ev, filtered_cl, filtered_dl, filtered_ed


def _compute_modality_region_rows(stats, selected_modality):
    """Build the per-region rows for the Summary tab's modality table:
    collapse dental subtypes into "Dental imaging", then compute
    per-region scans/dose/detriment per year."""
    modality_stats = ((stats or {}).get("modality_stats", {}) or {}).get(selected_modality, {}) or {}
    region_stats = modality_stats.get("regions", {}) or {}
    iterations = _safe_iterations(stats)
    # Collapse dental dose subtypes (Dental CBCT / Panoramic / Intraoral /
    # Kefalometry) into a single "Dental imaging" reporting category.
    if any(_is_dental_region(r) for r in region_stats):
        collapsed = {}
        dental_acc = {}
        for region, region_values in region_stats.items():
            if _is_dental_region(region):
                for k, v in (region_values or {}).items():
                    try:
                        dental_acc[k] = dental_acc.get(k, 0) + v
                    except TypeError:
                        dental_acc.setdefault(k, v)
            else:
                collapsed[region] = region_values
        if dental_acc:
            collapsed["Dental imaging"] = dental_acc
        region_stats = collapsed
    rows = []
    for region, region_values in sorted(region_stats.items(), key=lambda x: x[1].get("scans", 0), reverse=True):
        scans = _per_iteration_value(region_values.get("scans", 0), iterations)
        # Risk-modelled exams only in the per-exam denominator (organ-dose
        # model gives >=1 cancer-site dose, region not Non-Dose) — see
        # vectorized.scans_model_per_bin. Fall back to scans_dose then
        # scans for results without the modelled mask.
        scans_model = _per_iteration_value(
            region_values.get("scans_model",
                              region_values.get("scans_dose", region_values.get("scans", 0))),
            iterations)
        _has_model = "scans_model" in region_values
        dose_sv_model = _per_iteration_value(region_values.get("dose_sv_model", 0.0), iterations)
        dose_sv_all = _per_iteration_value(region_values.get("dose_sv", 0.0), iterations)
        dose_sv = dose_sv_model if _has_model else dose_sv_all
        ed_per_scan_msv = (dose_sv * 1000.0 / scans_model) if scans_model > 0 else 0.0
        cases = _per_iteration_value(region_values.get("cases", 0), iterations)
        deaths = _per_iteration_value(region_values.get("deaths", 0), iterations)
        expected_deaths = _per_iteration_value(region_values.get("expected_deaths", 0.0), iterations)
        crpv = (cases / dose_sv * 100) if dose_sv > 0 else 0.0
        drpv = (deaths / dose_sv * 100) if dose_sv > 0 else 0.0
        drpv_expected = (expected_deaths / dose_sv * 100) if dose_sv > 0 else 0.0
        rows.append((
            region,
            f"{scans:,.0f}",
            f"{dose_sv:.3f}",
            f"{ed_per_scan_msv:.3f}",
            f"{cases:.4f}",
            f"{deaths:.4f}",
            "-",
            f"{crpv:.4f}",
            _format_sim_expected(drpv, drpv_expected),
            "-",
        ))
    return rows


def _format_sim_expected(sim_value, expected_value):
    return f"{float(sim_value):.4f} ({float(expected_value):.4f})"


def _is_dental_region(region):
    r = str(region or "").strip().lower()
    return r.startswith("dental") or r in ("kefalometry", "intraoral")


def _region_debug_text(filtered):
    """Assemble the region-resolution debug text (per-region exam
    counts, orientation summaries, combo folding) for display."""
    if not filtered:
        return "No exam data yet."

    from .region_resolver import (
        radiography_reporting_region as _rpt_reg,
        is_combo_region as _is_combo,
        CT_OTHER_COMBINATIONS as _OTHER_COMBOS,
        RARE_COMBO_THRESHOLD as _RARE_THRESHOLD,
    )
    lines = []

    def _append_region_block(region_label, region_exams, show_orientation,
                             fold_rare_desc=False):
        """Append one region's block to the debug ``lines``: exam count,
        orientation distribution when combined, and description-level
        detail (rare descriptions folded when ``fold_rare_desc``)."""
        lines.append(f"[{region_label}] ({len(region_exams):,})")
        # Orientation distribution summary for combined radiography regions.
        if show_orientation:
            ori_counts = {}
            for ex in region_exams:
                full_reg = str(ex.get("region", "") or "")
                ori = full_reg
                if ori and ori != region_label:
                    ori_counts[ori] = ori_counts.get(ori, 0) + 1
            if ori_counts:
                ori_parts = [f"{c}x {o}" for o, c in
                             sorted(ori_counts.items(), key=lambda x: (-x[1], x[0]))]
                lines.append("  Orientation distribution: " + ", ".join(ori_parts))
        desc_counts = {}
        for ex in region_exams:
            desc = str(ex.get("debug_description", "") or ex.get("protocol", "") or ex.get("description", "") or "[no description]").strip()
            inferred = str(ex.get("orientation_inferred", "") or "").strip()
            detected = str(ex.get("orientation_detected", "") or "").strip()
            chosen = str(ex.get("orientation_resolved", "") or "").strip()
            if not show_orientation:
                inferred = detected = chosen = ""
            key = (desc, inferred, detected, chosen)
            desc_counts[key] = desc_counts.get(key, 0) + 1
        # Fold rare description variants (count < threshold) into a single
        # "Other (rare variants)" line when fold_rare_desc is set.  Used
        # for the Extremities bucket, which has ~30 raw text variants
        # many at n=1-4.
        folded_descs = []
        kept_items = []
        for key, count in desc_counts.items():
            if fold_rare_desc and count < _RARE_THRESHOLD:
                folded_descs.append((key, count))
            else:
                kept_items.append((key, count))
        for (desc, inferred, detected, chosen), count in sorted(kept_items, key=lambda item: (-item[1], item[0][0].lower())):
            parts = [f"{count:>5}x", desc]
            if inferred:
                parts.append(f"text={inferred}")
            if detected:
                parts.append(f"detected={detected}")
            if chosen:
                parts.append(f"chosen={chosen}")
            lines.append(" | ".join(parts))
        if folded_descs:
            _folded_total = sum(c for _, c in folded_descs)
            lines.append(f"{_folded_total:>5}x Other (rare variants, "
                         f"{len(folded_descs)} descriptions, each <{_RARE_THRESHOLD} exams)")
        lines.append("")

    # Dental imaging is reported as a single cross-modality category (CBCT,
    # panoramic, intraoral, cephalometric) rather than split per modality.
    modality_groups = {}
    dental_exams = []
    for ex in filtered:
        if _is_dental_region(ex.get("region")):
            dental_exams.append(ex)
            continue
        mod = str(ex.get("modality", "Unknown") or "Unknown")
        modality_groups.setdefault(mod, []).append(ex)

    for mod in sorted(modality_groups.keys()):
        exams = modality_groups[mod]
        # Projection orientation (text/detected/chosen) is only meaningful
        # for projection radiography; suppress it for CT / CBCT / CT-PT.
        _show_orientation = "CT" not in str(mod).strip().upper()
        lines.append("=" * 100)
        lines.append(f"{modality_name(mod)} ({len(exams):,})")
        lines.append("=" * 100)
        # Group by reporting_region (orientation-stripped) so that
        # "Chest PA", "Chest AP", "Chest Lat" all appear under "Chest".
        region_groups = {}
        for ex in exams:
            rpt_reg = _rpt_reg(str(ex.get("region", "Unknown") or "Unknown"))
            region_groups.setdefault(rpt_reg, []).append(ex)
        # Fold rare multi-region combos (e.g. "Neck, Pelvis" with <30
        # scans) into "Other combinations" so the text isn't cluttered
        # with n=1-13 combo labels.
        _combo_groups = {}
        _other_combos_exams = []
        for _reg, _exams in region_groups.items():
            if _is_combo(_reg) and len(_exams) < _RARE_THRESHOLD:
                _other_combos_exams.extend(_exams)
            else:
                _combo_groups[_reg] = _exams
        if _other_combos_exams:
            _combo_groups[_OTHER_COMBOS] = _other_combos_exams
        region_groups = _combo_groups
        for region in sorted(region_groups.keys(), key=lambda key: (-len(region_groups[key]), key)):
            _fold_desc = (region == "Extremities")
            _append_region_block(region, region_groups[region], _show_orientation,
                                 fold_rare_desc=_fold_desc)

    if dental_exams:
        lines.append("=" * 100)
        lines.append(f"Dental imaging (all modalities) ({len(dental_exams):,})")
        lines.append("=" * 100)
        # Break down by dose subtype so the dental coefficients stay visible.
        subtype_groups = {}
        for ex in dental_exams:
            subtype = str(ex.get("region", "Unknown") or "Unknown")
            subtype_groups.setdefault(subtype, []).append(ex)
        for subtype in sorted(subtype_groups.keys(), key=lambda key: (-len(subtype_groups[key]), key)):
            _append_region_block(subtype, subtype_groups[subtype], False)

    return "\n".join(lines).rstrip()


def _region_debug_text_from_cohort(fc):
    """Generate the By Body Region debug text from a FileCohort's
    ``region_debug`` count table.

    Produces the same format as :func:`_region_debug_text` but works on
    the mergeable count tables, so it covers every file of every year —
    not just the last processed year's raw exam_data.  Falls back to
    ``"No exam data yet."`` when the cohort has no region_debug entries
    (e.g. ExamColumns input, which does not carry the debug fields).
    """
    rd = fc.region_debug if fc is not None else None
    if not rd:
        return "No exam data yet."

    from .region_resolver import (
        radiography_reporting_region as _rpt_reg,
        is_combo_region as _is_combo,
        CT_OTHER_COMBINATIONS as _OTHER_COMBOS,
        RARE_COMBO_THRESHOLD as _RARE_THRESHOLD,
    )
    lines = []

    def _append_region_block(region_label, entries, show_orientation,
                             fold_rare_desc=False):
        """Append one region's block to ``lines`` from (raw_region, desc,
        inferred, detected, chosen, count) entries; orientation summary
        included when the region combines projections."""
        # entries: list of (raw_region, desc, inferred, detected, chosen, count)
        total = sum(e[5] for e in entries)
        lines.append(f"[{region_label}] ({total:,})")
        if show_orientation:
            ori_counts = {}
            for raw_reg, _, _, _, _, c in entries:
                if raw_reg and raw_reg != region_label:
                    ori_counts[raw_reg] = ori_counts.get(raw_reg, 0) + c
            if ori_counts:
                ori_parts = [f"{c}x {o}" for o, c in
                             sorted(ori_counts.items(), key=lambda x: (-x[1], x[0]))]
                lines.append("  Orientation distribution: " + ", ".join(ori_parts))
        desc_counts = {}
        for _, desc, inferred, detected, chosen, c in entries:
            if not show_orientation:
                inferred = detected = chosen = ""
            key = (desc, inferred, detected, chosen)
            desc_counts[key] = desc_counts.get(key, 0) + c
        # Fold rare description variants (count < threshold) into a single
        # "Other (rare variants)" line when fold_rare_desc is set.
        folded_descs = []
        kept_items = []
        for key, count in desc_counts.items():
            if fold_rare_desc and count < _RARE_THRESHOLD:
                folded_descs.append((key, count))
            else:
                kept_items.append((key, count))
        for (desc, inferred, detected, chosen), count in sorted(kept_items, key=lambda item: (-item[1], item[0][0].lower())):
            parts = [f"{count:>5}x", desc]
            if inferred:
                parts.append(f"text={inferred}")
            if detected:
                parts.append(f"detected={detected}")
            if chosen:
                parts.append(f"chosen={chosen}")
            lines.append(" | ".join(parts))
        if folded_descs:
            _folded_total = sum(c for _, c in folded_descs)
            lines.append(f"{_folded_total:>5}x Other (rare variants, "
                         f"{len(folded_descs)} descriptions, each <{_RARE_THRESHOLD} exams)")
        lines.append("")

    # Group by modality (non-dental) vs dental.
    modality_groups = {}   # mod -> list of (raw_reg, desc, inf, det, cho, cnt)
    dental_entries = []
    for (mod, raw_reg, desc, inf, det, cho), cnt in rd.items():
        entry = (raw_reg, desc, inf, det, cho, cnt)
        if _is_dental_region(raw_reg):
            dental_entries.append(entry)
        else:
            modality_groups.setdefault(mod, []).append(entry)

    for mod in sorted(modality_groups.keys()):
        entries = modality_groups[mod]
        total = sum(e[5] for e in entries)
        _show_orientation = "CT" not in str(mod).strip().upper()
        lines.append("=" * 100)
        lines.append(f"{modality_name(mod)} ({total:,})")
        lines.append("=" * 100)
        # Group by reporting_region (orientation-stripped).
        region_groups = {}
        for raw_reg, desc, inf, det, cho, cnt in entries:
            rpt_reg = _rpt_reg(str(raw_reg or "Unknown"))
            region_groups.setdefault(rpt_reg, []).append((raw_reg, desc, inf, det, cho, cnt))
        # Fold rare multi-region combos (e.g. "Neck, Pelvis" with <30
        # scans) into "Other combinations".
        _combo_groups = {}
        _other_combos_entries = []
        for _reg, _entries in region_groups.items():
            _reg_total = sum(e[5] for e in _entries)
            if _is_combo(_reg) and _reg_total < _RARE_THRESHOLD:
                _other_combos_entries.extend(_entries)
            else:
                _combo_groups[_reg] = _entries
        if _other_combos_entries:
            _combo_groups[_OTHER_COMBOS] = _other_combos_entries
        region_groups = _combo_groups
        for region in sorted(region_groups.keys(),
                             key=lambda key: (-sum(e[5] for e in region_groups[key]), key)):
            _fold_desc = (region == "Extremities")
            _append_region_block(region, region_groups[region], _show_orientation,
                                 fold_rare_desc=_fold_desc)

    if dental_entries:
        total = sum(e[5] for e in dental_entries)
        lines.append("=" * 100)
        lines.append(f"Dental imaging (all modalities) ({total:,})")
        lines.append("=" * 100)
        subtype_groups = {}
        for raw_reg, desc, inf, det, cho, cnt in dental_entries:
            subtype_groups.setdefault(raw_reg, []).append((raw_reg, desc, inf, det, cho, cnt))
        for subtype in sorted(subtype_groups.keys(),
                              key=lambda key: (-sum(e[5] for e in subtype_groups[key]), key)):
            _append_region_block(subtype, subtype_groups[subtype], False)

    return "\n".join(lines).rstrip()


def _excluded_examinations_text(excluded_cohorts, kind: str = "all", key=None,
                                modality: "str | None" = None) -> str:
    """Render the By Body Region tab's "Excluded examinations" section.

    ``excluded_cohorts`` is ``stats["_excluded_cohorts"]`` -- the
    ``{exclusion key: CohortByFile}`` ledger of the exams dropped at source
    because they carry no risk model (participant_flow.py).  Each table's
    ``region_debug`` reads ``(modality, raw region, reason, "", "", "") ->
    n``, so this lists, per exclusion reason, every modality / region with
    its count and the exact reason attached when the exam was dropped.
    ``kind`` / ``key`` select the same source (all / year group / file) as
    the analysed-cohort text above it; ``modality`` applies the tab's
    modality filter.  Empty string when there is nothing to show.
    """
    if not excluded_cohorts:
        return ""
    from .participant_flow import EXCLUSION_LABELS, EXCLUSION_ORDER
    keys = [k for k in EXCLUSION_ORDER if k in excluded_cohorts]
    keys += [k for k in excluded_cohorts if k not in keys]
    blocks: list = []
    grand_total = 0
    for ek in keys:
        cohort = excluded_cohorts.get(ek)
        if cohort is None or len(cohort) == 0:
            continue
        fc = cohort.select(kind, key)
        if modality is not None:
            fc = fc.filtered(modality)
        rd = fc.region_debug or {}
        if not rd:
            continue
        # (modality, region) -> {reason: n}
        table: dict = {}
        for (mod, raw_reg, reason, _i, _d, _c), cnt in rd.items():
            cell = table.setdefault((str(mod), str(raw_reg or "Unknown")), {})
            cell[str(reason)] = cell.get(str(reason), 0) + int(cnt)
        total = sum(sum(v.values()) for v in table.values())
        if total == 0:
            continue
        grand_total += total
        label = EXCLUSION_LABELS.get(ek, ek)
        lines = ["-" * 100, f"{label} ({total:,})", "-" * 100]
        by_mod: dict = {}
        for (mod, _r), cell in table.items():
            by_mod[mod] = by_mod.get(mod, 0) + sum(cell.values())
        for mod in sorted(by_mod, key=lambda m: (-by_mod[m], m)):
            lines.append(f"{modality_name(mod)} ({by_mod[mod]:,})")
            rows = [((m, r), cell) for (m, r), cell in table.items() if m == mod]
            rows.sort(key=lambda it: (-sum(it[1].values()), it[0][1].lower()))
            for (_m, reg), cell in rows:
                for reason, n in sorted(cell.items(), key=lambda it: (-it[1], it[0])):
                    lines.append(f"{n:>8,}x  {reg:<34s} {reason}")
            lines.append("")
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    head = [
        "=" * 100,
        f"EXCLUDED EXAMINATIONS -- not risk-modelled, not in the analysed cohort ({grand_total:,})",
        "=" * 100,
        "These examinations were removed from the analysed cohort before the simulation and are",
        "not counted in the region figures above, in Table 1 or in any per-100 000 denominator.",
        "",
    ]
    return "\n".join(head + blocks).rstrip()


def _upd_cancer_per_sv(cancer_per_sv_text, s, iterations, visible):
    """Rebuild the per-collective-dose cancer table: totals divided by
    the summed annual effective dose (person Sv)."""
    # Calculate total effective dose across all modalities
    total_dose_sv = 0.0
    modality_stats = (s or {}).get("modality_stats", {}) or {}
    for mod, ms in modality_stats.items():
        dose_sv = _per_iteration_value(ms.get("dose_sv", 0.0), iterations)
        total_dose_sv += dose_sv

    # Import risk model parameters for detriment calculation
    from .risk_models import lookup_cancer_params

    # Format cancers per Sv text as ICRP Table 3.9 format
    lines = []
    lines.append("=" * 100)
    lines.append("Radiation Detriment (ICRP Table 3.9 Format)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Total effective dose per iteration: {total_dose_sv:.6f} Sv")
    lines.append("")
    lines.append(f"{'Organ/tissue':<20} {'Nominal risk':<18} {'Lethality':<12} {'Radiation':<18} {'Relative':<15}")
    lines.append(f"{'':<20} {'coefficient *':<18} {'fraction':<12} {'detriment #':<18} {'radiation':<15}")
    lines.append(f"{'':<20} {'(R)':<18} {'(k)':<12} {'(D)':<18} {'detriment':<15}")
    lines.append("-" * 100)

    total_detriment = 0.0
    detriment_values = {}
    nominal_risk_values = {}
    lethality_values = {}
    organ_names = {}  # Store mapping from cancer type to organ name

    for ct, cv in sorted(s.get("cancer_breakdown", {}).items(),
                          key=lambda x: x[1]["developed"], reverse=True):
        if ct not in visible:
            continue
        developed = _per_iteration_value(cv.get("developed", 0), iterations)

        cancers_per_sv = (developed / total_dose_sv) if total_dose_sv > 0 else 0

        # Convert to cases per 10,000 persons per Sv (ICRP Table 3.9 format)
        nominal_risk = cancers_per_sv * 10000

        # Calculate radiation detriment using ICRP 103/152 Table 3.9 formula:
        # D = R × [k + q×(1-k)] × l, where q = k + q_min×(1-k)
        _param_key = {"Other solid": "Remainder"}.get(ct, ct)
        params = lookup_cancer_params(_param_key)
        k = params.get("k_lethality", 0.5)
        q_min = params.get("q_min", 0.1)
        l_rel = params.get("l_rel", 1.0)

        q = k + q_min * (1 - k)
        detriment_factor = k + q * (1 - k)
        detriment = nominal_risk * detriment_factor * l_rel

        detriment_values[ct] = detriment
        nominal_risk_values[ct] = nominal_risk
        lethality_values[ct] = k
        total_detriment += detriment

        # Map cancer type to ICRP 152 Table 3.9 naming convention
        organ_name = ct
        if ct == "Leukaemia":
            organ_name = "Leukaemia"
        elif ct == "Remainder":
            organ_name = "Other solid"

        organ_names[ct] = organ_name
        lines.append(f"{organ_name:<20} {nominal_risk:<18.3f} {k:<12.2f} {detriment:<18.3f} {'':<15}")

    # Second pass: add relative detriment using stored organ names
    # lines[0-5] = banner + dose header; [6-8] = 3-row column header; [9] = separator
    lines_with_rel = lines[:10]

    # ICRP 152 Table 3.9 order: Bone surface and Skin before Breast; Leukaemia then Other solid last
    icrp_order = [
        "Oesophagus", "Stomach", "Colon", "Liver", "Lung",
        "Bone", "Skin", "Breast", "Ovary", "Bladder",
        "Thyroid", "Leukaemia", "Remainder"
    ]

    def get_icrp_order(ct):
        """Get ICRP Table 3.9 order index for sorting"""
        try:
            return icrp_order.index(ct)
        except ValueError:
            return 999  # Put unknown types at the end

    for ct in sorted(detriment_values.keys(), key=get_icrp_order):
        if ct not in organ_names:
            continue
        organ_name = organ_names[ct]
        nominal_val = nominal_risk_values[ct]
        lethality_val = lethality_values[ct]
        detriment_val = detriment_values[ct]
        rel_detriment = (detriment_val / total_detriment) if total_detriment > 0 else 0
        lines_with_rel.append(f"{organ_name:<20} {nominal_val:<18.3f} {lethality_val:<12.2f} {detriment_val:<18.3f} {rel_detriment:<15.3f}")

    # Add heritable (Gonads) risk as a separate row (ICRP 152 Table 3.9)
    heritable_risk = _per_iteration_value(s.get("heritable_risk", 0.0), iterations)
    heritable_detriment = _per_iteration_value(s.get("heritable_detriment", 0.0), iterations)
    if heritable_risk > 0 or heritable_detriment > 0:
        heritable_nominal = (heritable_risk / total_dose_sv * 10000) if total_dose_sv > 0 else 0
        heritable_det = (heritable_detriment / total_dose_sv * 10000) if total_dose_sv > 0 else 0
        total_detriment += heritable_det
        rel_heritable_det = (heritable_det / total_detriment) if total_detriment > 0 else 0
        lines_with_rel.append(f"{'Gonads':<20} {heritable_nominal:<18.3f} {'0.80':<12} {heritable_det:<18.3f} {rel_heritable_det:<15.3f}")
        lines_with_rel.append(f"{'(heritable)':<20} {'':<18} {'':<12} {'':<18} {'':<15}")

    # Add total row
    lines_with_rel.append("-" * 100)
    total_nominal = sum(nominal_risk_values.values()) + (heritable_nominal if heritable_risk > 0 or heritable_detriment > 0 else 0)
    lines_with_rel.append(f"{'Total':<20} {total_nominal:<18.3f} {'':<12} {total_detriment:<18.3f} {'1.000':<15}")

    lines_with_rel.append("=" * 100)
    lines_with_rel.append("* Cases per 10,000 persons per Sv (ICRP 152 Table 3.9 R column convention)")
    lines_with_rel.append("# Cases per 10,000 persons per Sv (ICRP 152 Table 3.9 D column convention)")

    cancer_per_sv_text.delete("1.0", tk.END)
    cancer_per_sv_text.insert(tk.END, "\n".join(lines_with_rel))


def _show_5050_comparison(results):
    """Recompute nominal risk using 50/50 sex AND 50/50 population split."""
    s = results.get("stats")
    exam_data_local = results.get("exam_data") or []
    if not s or not s.get("iterations") or not exam_data_local:
        messagebox.showwarning("No Data", "Run the simulation first to generate data.")
        return
    import numpy as np
    from .risk_models import (
        get_vectorized_rpg, get_vectorized_rpg_2pop, _RPG_CANCER_LIST,
    )
    rpg      = get_vectorized_rpg()
    rpg_2pop = get_vectorized_rpg_2pop()
    if not rpg or not _RPG_CANCER_LIST:
        messagebox.showwarning("No Kernels", "Kernels not precomputed. Run the simulation first.")
        return

    two_pop  = rpg_2pop is not None
    sim_pop  = (s.get("sim_config") or {}).get("population", "Euro-American")
    _age_dim = 121

    # Patient population split (Euro-American vs Asian).  Only the totals
    # are needed -- they decide which populations the Table 3.8
    # comparison averages over (has_ea / has_as below) -- so a single
    # light pass over the exam list suffices (the kernel-weighted Table
    # 3.8 maths uses the ICRP 152 Table A.7 reference age profile,
    # not the patient age distribution).
    n_as = sum(1 for ex in exam_data_local
               if (ex.get("_population") or sim_pop) == "Asian")
    n_ea = len(exam_data_local) - n_as

    # Build ICRP 152 Table A.7 reference age weights (equal within each 5-year bin).
    # These are used instead of actual patient counts in the 'balanced' kernel sums
    # so the comparison standardises to the true ICRP demographic, not patient demographics.
    _A7 = [
        ( 0,  4,  68640, 81767), ( 5,  9,  68504, 75689), (10, 14,  63852, 77326),
        (15, 19,  61919, 93319), (20, 24,  67185, 91193), (25, 29,  74019, 78216),
        (30, 34,  79117, 68370), (35, 39,  73816, 74757), (40, 44,  68990, 70443),
        (45, 49,  69559, 58195), (50, 54,  59602, 52664), (55, 59,  48570, 49365),
        (60, 64,  43652, 41958), (65, 69,  42536, 31838), (70, 74,  40006, 23070),
        (75, 79,  30888, 16822), (80, 84,  23487,  9005), (85, 89,  15658,  6003),
    ]
    _icrp_ea_raw = np.zeros(_age_dim, dtype=float)
    _icrp_as_raw = np.zeros(_age_dim, dtype=float)
    for _lo, _hi, _ea_tot, _as_tot in _A7:
        _n_y = _hi - _lo + 1
        for _age in range(_lo, min(_hi + 1, _age_dim)):
            _icrp_ea_raw[_age] = _ea_tot / _n_y
            _icrp_as_raw[_age] = _as_tot / _n_y

    # Determine which populations are available
    has_ea = n_ea > 0
    has_as = n_as > 0 and two_pop

    from .risk_models import (
        solid_kernel_perGy as _skpg,
        leukaemia_err_ear_kernels_perGy as _leuk_err_ear_k,
        get_female_breast_EAR_kernel_perGy as _breast_k,
        _baseline_survival_curve as _surv_fn,
        MAX_ATTAINED_AGE_ICRP152 as _MAX_A,
        DDREF_SOLID_CANCERS as _T38_DDREF_SOLID,
        DDREF_LEUKAEMIA as _T38_DDREF_LEUK,
    )

    def _aler_err_ear(ct, sex_key, pop_str, age_weights):
        """Compute ALER_ERR and ALER_EAR (cases per 100 per Gy) using
        separate ERR-only and EAR-only kernels × survival, averaged over
        the supplied per-age weights.

        Population-specific cancer-free survival is recomputed from the
        cached log-prefix arrays (_CANCER_FREE_SURV_PREFIX[(sex, pop)])
        rather than read from the shared module-level _SURV_CACHE.  The
        shared cache is overwritten by whichever call to
        precompute_kernels ran last, so indexing it by (sex, e) alone
        silently used the wrong population's survival and inflated
        every EA solid-cancer row by ~5-10 %.

        Leukaemia is evaluated per ICRP 152 paragraph 138 convention
        (REIC at 0.1 Gy × 10), giving an annual-rate contribution of
        L1 + 0.1·L2 per Gy rather than L1 + L2 (which would be the LQ
        excess at 1 Gy).  Solid-cancer kernels are already linear in
        dose, so the 0.1 Gy × 10 convention leaves them unchanged apart
        from the small nonlinear-survival correction (not applied here;
        the correction is ~0.7 % and is captured in Table 3.9)."""
        w_sum = float(age_weights.sum())
        if w_sum == 0:
            return 0.0, 0.0
        acc_err = acc_ear = 0.0
        for e in range(len(age_weights)):
            w = float(age_weights[e])
            if w == 0:
                continue
            ages_arr = np.arange(max(1, e), _MAX_A + 1, dtype=int)
            if ages_arr.size == 0:
                continue
            surv = _surv_fn(ages_arr, e, sex_key, pop_str)
            if surv.size == 0:
                continue
            n = min(surv.size, ages_arr.size)
            if n == 0:
                continue
            if ct == "Leukaemia":
                # ICRP 152 para 42 / 51: leukaemia has no native ERR model,
                # so its ERR projection is EAR_LSS / baseline_LSS applied to
                # the TARGET-population baseline (ERR transfer), while the
                # EAR projection is the population-independent EAR kernel.
                # Both are the per-Gy slope L1 + 0.1·L2 at the 0.1 Gy
                # reference dose (para 138).
                K_err, K_ear = _leuk_err_ear_k(sex_key, e, pop_str)
                acc_err += w * float(np.sum(K_err[ages_arr[:n]] * surv[:n]))
                acc_ear += w * float(np.sum(K_ear[ages_arr[:n]] * surv[:n]))
            elif ct == "Breast" and sex_key == "F":
                K_ear = _breast_k(float(e))
                acc_err += 0.0
                acc_ear += w * float(np.sum(K_ear[ages_arr[:n]] * surv[:n]))
            else:
                K_err = _skpg(ct, sex_key, e, pop_str, force_model="ERR")
                K_ear = _skpg(ct, sex_key, e, pop_str, force_model="EAR")
                acc_err += w * float(np.sum(K_err[ages_arr[:n]] * surv[:n]))
                acc_ear += w * float(np.sum(K_ear[ages_arr[:n]] * surv[:n]))
        return 100.0 * acc_err / w_sum, 100.0 * acc_ear / w_sum

    # Table 3.8: working-age (18-64) nominal risk coefficients
    # (cases per 10 000 persons per Gy).  Standardised to the ICRP 152
    # Table A.7 reference demographic restricted to age at exposure
    # 18-64 (ICRP 152 para 93 / Fig 4.3 working-age population), then:
    #   nominal_c = (w_ERR·REIC_ERR + w_EAR·REIC_EAR) / DDREF_c
    # sex-averaged (50/50) and population-averaged (50/50).
    # Transfer weights: ICRP 152 para 99 (Thyroid 100:0, Breast 0:100,
    # Lung 30:70, all other solids + leukaemia 50:50).  DDREF = 2 for
    # solids, 1 for leukaemia (para 64/100).  Compared with the ICRP 152
    # recalculated values (Table 3.8, correction of programming errors).
    _T38_W = {"Oesophagus": (0.5, 0.5), "Stomach": (0.5, 0.5),
              "Colon": (0.5, 0.5), "Liver": (0.5, 0.5), "Lung": (0.3, 0.7),
              "Bladder": (0.5, 0.5), "Thyroid": (1.0, 0.0),
              "Breast": (0.0, 1.0), "Ovary": (0.5, 0.5),
              "Remainder": (0.5, 0.5), "Leukaemia": (0.5, 0.5)}
    _T38_DDREF = {ct: _T38_DDREF_SOLID for ct in _T38_W}
    _T38_DDREF["Leukaemia"] = _T38_DDREF_LEUK
    _T38_DISP = {"Remainder": "Other solid"}
    # ICRP 152 Table 3.8 recalculated values (per 10 000 per Gy).
    _T38_REF = {"Oesophagus": 13, "Stomach": 64, "Colon": 55, "Liver": 25,
                "Lung": 117, "Breast": 66, "Ovary": 9, "Bladder": 38,
                "Thyroid": 10, "Leukaemia": 44, "Other solid": 98}
    # Not recalculated (§): fixed nominal risks (ICRP 60 bone, ICRP 59 skin,
    # heritable gonads) — model-independent reference rows.
    _T38_FIXED = [("Bone surface", 5), ("Skin", 670), ("Gonads (heritable)", 12)]
    _T38_ORDER = ["Oesophagus", "Stomach", "Colon", "Liver", "Lung",
                  "Breast", "Ovary", "Bladder", "Thyroid",
                  "Leukaemia", "Remainder"]
    _t38_pops = [p for p in ("EA", "AS")
                 if (p == "EA" and has_ea) or (p == "AS" and has_as)]
    if not _t38_pops:
        _t38_pops = ["EA" if sim_pop == "Euro-American" else "AS"]

    def _t38_working_age_weights(pop_key):
        raw = (_icrp_ea_raw if pop_key == "EA" else _icrp_as_raw).astype(float).copy()
        mask = np.zeros_like(raw)
        mask[18:65] = 1.0          # age at exposure 18-64 inclusive
        return raw * mask

    def _t38_nominal(ct):
        w_err, w_ear = _T38_W[ct]
        ddref = _T38_DDREF[ct]
        acc = 0.0
        npop = 0
        for pop_key in _t38_pops:
            pop_str = "Euro-American" if pop_key == "EA" else "Asian"
            wts = _t38_working_age_weights(pop_key)
            if wts.sum() <= 0:
                continue
            per_sex = []
            for sex_key in ("M", "F"):
                if ct in ("Breast", "Ovary") and sex_key == "M":
                    per_sex.append(0.0)
                    continue
                err, ear = _aler_err_ear(ct, sex_key, pop_str, wts)  # cases/100/Gy
                per_sex.append((w_err * err + w_ear * ear) * 100.0)  # -> /10 000/Gy
            acc += 0.5 * (per_sex[0] + per_sex[1])
            npop += 1
        return (acc / npop) / ddref if npop else 0.0

    # Per-organ rows: (display, simulation value, ICRP 152 value, is_fixed).
    t38_rows = []
    for ct in _T38_ORDER:
        disp = _T38_DISP.get(ct, ct)
        t38_rows.append((disp, _t38_nominal(ct), float(_T38_REF[disp]), False))
    for disp, rec in _T38_FIXED:
        # Not recalculated (§): the simulation uses the same fixed nominal
        # risk, so its value mirrors the ICRP reference.
        t38_rows.append((disp, float(rec), float(rec), True))

    # Table 3.8 text (working-age nominal risk: Simulation vs ICRP 152)
    lines3 = [
        "Table 3.8. Nominal risk coefficients for the working-age population",
        "           (cases per 10,000 persons per Gy)",
        f"Age at exposure 18-64 y; ICRP 152 Table A.7 reference demographic; "
        f"populations: {', '.join(_t38_pops)}",
        "",
        f"{'Organ/tissue':<22}{'Simulation':>12}{'ICRP 152':>12}{'Deviation':>14}",
        "-" * 60,
    ]
    _t38_sim_tot = 0.0
    _t38_ref_tot = 0.0
    for _disp, _sim, _rv, _is_fixed in t38_rows:
        _sim_r = float(round(_sim))
        _t38_sim_tot += _sim_r
        _t38_ref_tot += _rv
        if _is_fixed:
            lines3.append(f"{_disp:<22}{_sim_r:>12.0f}{_rv:>12.0f}{'':>14}")
        else:
            _dev = (_sim_r / _rv - 1.0) * 100.0 if _rv else 0.0
            lines3.append(f"{_disp:<22}{_sim_r:>12.0f}{_rv:>12.0f}{_dev:>+13.1f}%")
    _t38_tot_dev = (_t38_sim_tot / _t38_ref_tot - 1.0) * 100.0 if _t38_ref_tot else 0.0
    lines3.append("-" * 60)
    lines3.append(f"{'Total':<22}{_t38_sim_tot:>12.0f}{_t38_ref_tot:>12.0f}{_t38_tot_dev:>+13.1f}%")
    lines3 += [
        "",
        "ICRP 152 = recalculated working-age coefficients (Table 3.8, with the",
        "  correction of Publication 103 programming errors; see paras 71-72).",
        "Bone surface (ICRP 60), Skin (ICRP 59), and Gonads/heritable are not",
        "  LSS-modelled and were not recalculated (§); shown as reference rows.",
    ]

    win = tk.Toplevel()
    win.title("ICRP 152 Comparison \u2013 Table 3.8 (working-age)")
    _place_window(win, 760, 560)
    txt = scrolledtext.ScrolledText(win, font=("Consolas", 9), wrap=tk.NONE)
    txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    txt.insert(tk.END, "\n".join(lines3))
    txt.config(state=tk.DISABLED)
    ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)
    win.lift()
    win.focus_set()


def _db_strip_dental_imaging(sweep):
    """Return *sweep* with the "Dental imaging" pseudo-modality removed.

    Dental CBCT is keyed under the CBCT modality in the sweep (see
    ``_region_age_baseline``), so it is retained.  Only the
    Panoramic/Kefalometry/Intraoral/Dental subtypes grouped under
    "Dental imaging" are dropped.  Used for the two burden-vs-dose
    PLOTS only -- the text table describes the whole cohort and gets
    the unstripped sweep.  Returns the sweep unchanged when
    there is no "Dental imaging" modality or the input is empty.
    """
    if not sweep:
        return sweep
    mod_map = sweep.get("modality_region_map") or {}
    if "Dental imaging" not in mod_map:
        return sweep
    keep = {m for m in mod_map.keys() if m != "Dental imaging"}
    if not keep:
        return sweep
    return filter_region_age_sweep(sweep, keep)


def _format_life_lost(x: float) -> str:
    """Format life-years-lost for the events table.

    Fatal cancers now always carry a strictly positive ``life_lost``
    (the per-hit lifespan draw is conditioned on lifespan >= onset), but
    that gap can be a small fraction of a year when the patient's
    competing-mortality lifespan draw lands only slightly beyond the
    cancer onset age. The default ``.1f`` formatting rounds anything
    below 0.05 to "0.0", which reads as an implausible zero. Switch to
    scientific notation in that range so the true (small, nonzero) value
    is visible instead of being rounded away.
    """
    if 0.0 < x < 0.05:
        return f"{x:.2e}"
    return f"{x:.1f}"


def _show_annual_benefit(root, _rq_cache, rq_annual_red_var):
    """Open the annual-benefit window for a user-typed dose reduction:
    reads the signed change from the entry, reports bad input, and
    plots the simulated-benefit summary."""
    sweep = _rq_cache.get("sweep")
    if sweep is None:
        messagebox.showinfo(
            "No data",
            "Run a simulation first, then the plot will be available.")
        return
    # Signed dose change typed beside the button.  Bad input is
    # reported rather than silently replaced with a default -- showing a
    # figure for a percentage the user did not ask for is worse than
    # refusing to draw one.
    _chg_pct, _chg_err = _parse_dose_change(rq_annual_red_var.get())
    if _chg_pct is None:
        messagebox.showwarning("Dose change", _chg_err)
        return
    benefit = compute_annual_benefit(sweep, target_pct=_chg_pct)
    burden = compute_annual_burden(sweep)
    _is_incr = _chg_pct > 0.0
    _chg_txt = f"{abs(_chg_pct):g}% dose {'increase' if _is_incr else 'reduction'}"
    win = tk.Toplevel(root)
    win.title(
        f"Annual QALYs "
        f"{'lost to' if _is_incr else 'gained from'} a {_chg_txt}")
    win.transient(root)
    # Geometry is set by _fit_window_to_content once the canvas, summary
    # text and toolbar are packed, so the figure opens at full size.

    # Build a lookup of annual burden by (modality, region) from the burden dict.
    burden_by_mod_region = {
        (r.get("modality", ""), r["region"]): r
        for r in (burden.get("per_region") or [])
    }

    # Text summary at top.
    cohort = benefit.get("cohort_total", {})
    pr = benefit.get("per_region", [])

    # Per-modality QALY-lost totals (grouping the per-region-and-modality
    # rows currently displayed) so the reader can see, at a glance, which
    # modalities contribute how much of the baseline burden within the
    # selected set of modalities/regions shown below.
    mod_QALY_lost: dict[str, float] = {}
    for r in pr:
        mod = r.get("modality", "") or ""
        b = burden_by_mod_region.get((mod, r["region"]), {})
        QALY_lost = r.get("QALY_lost_annual", b.get("QALY_annual", 0.0))
        mod_QALY_lost[mod] = mod_QALY_lost.get(mod, 0.0) + QALY_lost
    total_selected_QALY_lost = sum(mod_QALY_lost.values())

    # For an increase every "saved"/"prevented" figure comes back
    # negative, so the wording flips with the sign rather than
    # reporting a negative number of cancers prevented.  Because the
    # wording (and hence the label length) depends on the sign, the
    # labels are padded to a common width instead of with hand-counted
    # runs of spaces, which silently misalign the values.
    _lbl_qaly = f"Total QALYs {'lost' if _is_incr else 'gained'}/yr:"
    _lbl_cases = f"Radiation-induced cancers {'caused' if _is_incr else 'prevented'}/yr:"
    _lbl_deaths = f"Fatal cancers {'caused' if _is_incr else 'prevented'}/yr:"
    _LBL_W = 41

    summary_lines = [
        f"Expected annual {'harm' if _is_incr else 'benefit'} from a {_chg_txt}",
        f"  Intervals: {SIM_ANNUAL_RANGE_LABEL} "
        "(point estimates are analytical).",
        f"  {'Total QALYs lost/yr (baseline):':<{_LBL_W}}{cohort.get('QALY_lost_annual', 0):.3f} {_rng(cohort.get('QALY_lost_annual_ci_lo'), cohort.get('QALY_lost_annual_ci_hi'))}",
        f"  {_lbl_qaly:<{_LBL_W}}{abs(cohort.get('QALY_saved_annual', 0)):.3f} {_rng(abs(cohort.get('QALY_saved_annual_ci_lo', 0)), abs(cohort.get('QALY_saved_annual_ci_hi', 0)))}",
        f"  {_lbl_cases:<{_LBL_W}}{abs(cohort.get('cases_saved_annual', 0)):.3f} {_rng(abs(cohort.get('cases_saved_annual_ci_lo', 0)), abs(cohort.get('cases_saved_annual_ci_hi', 0)))}",
        f"  {_lbl_deaths:<{_LBL_W}}{abs(cohort.get('deaths_saved_annual', 0)):.3f} {_rng(abs(cohort.get('deaths_saved_annual_ci_lo', 0)), abs(cohort.get('deaths_saved_annual_ci_hi', 0)))}",
        f"  {'Total collective effective dose/yr:':<{_LBL_W}}{cohort.get('dose_sv_annual', 0):.4f} person Sv",
        f"  {'Total examinations/yr:':<{_LBL_W}}{cohort.get('total_scans', 0):,.0f}",
        f"  Total QALYs lost/yr in the selected modalities: {total_selected_QALY_lost:.3f}",
        "",
        "Contribution by modality (share of the selected modalities' QALYs lost):",
    ]
    for mod, val in sorted(mod_QALY_lost.items(), key=lambda kv: kv[1], reverse=True):
        pct = (val / total_selected_QALY_lost * 100.0) if total_selected_QALY_lost > 0 else 0.0
        summary_lines.append(f"  {mod[:14]:<14} {val:>10.3f}  ({pct:>5.1f}%)")
    summary_lines.append("")
    summary_lines.append("By modality and region (sorted by annual QALYs gained):")
    summary_lines.append(
        f"  {'#':>2} {'Modality':<14} {'Region':<30} {'Person Sv/yr':>14} {'QALYs lost':>14} {'QALYs gained':>15} {'Cancers prev.':>14} {'Fatal prev.':>12} {'Exams/yr':>11}"
    )
    summary_lines.append(
        f"  {'─'*2} {'─'*14} {'─'*30} {'─'*14} {'─'*14} {'─'*15} {'─'*14} {'─'*12} {'─'*11}"
    )
    for rank, r in enumerate(pr, 1):
        mod = r.get("modality", "") or ""
        reg = r["region"]
        b = burden_by_mod_region.get((mod, reg), {})
        QALY_lost = r.get("QALY_lost_annual", b.get("QALY_annual", 0.0))
        dose_sv = r.get("dose_sv_annual", b.get("dose_sv_annual", 0.0))
        ci_lo = r.get("QALY_saved_annual_ci_lo", 0.0)
        ci_hi = r.get("QALY_saved_annual_ci_hi", 0.0)
        summary_lines.append(
            f"  {rank:>2} {mod[:14]:<14} {reg[:30]:<30} {dose_sv:>14.4f} {QALY_lost:>14.3f} "
            f"{r['QALY_saved_annual']:>15.3f} {r['cases_saved_annual']:>14.3f} "
            f"{r['deaths_saved_annual']:>12.3f} {r['total_scans']:>11,.0f}"
        )
        summary_lines.append(
            f"     {'':<14} {'':<30} {'':<14} {'95% range:':>14} "
            f"{_rng(r.get('QALY_lost_annual_ci_lo'), r.get('QALY_lost_annual_ci_hi'))} "
            f"{_rng(ci_lo, ci_hi)} "
            f"{_rng(r.get('cases_saved_annual_ci_lo'), r.get('cases_saved_annual_ci_hi'))} "
            f"{_rng(r.get('deaths_saved_annual_ci_lo'), r.get('deaths_saved_annual_ci_hi'))}"
        )
    summary_lines.append("")
    summary_lines.append(
        "Primary metric: annual QALYs gained (combines cancer probability,"
        " lethality, age at exposure, latency, and remaining life expectancy)."
    )
    summary_lines.append(
        "Secondary outcomes: radiation-induced cancers prevented and"
        " fatal cancers prevented (clinically intuitive supporting endpoints)."
    )
    summary_lines.append("")
    summary_lines.append(
        "Note: Annual QALYs gained depend on local examination volumes"
        " and age distribution.  Other hospitals should not copy this"
        " ranking directly — they should recalculate with their own"
        " dose archive.  The per-examination burden shown in the heatmap"
        " above is the transferable metric."
    )
    summary_text = scrolledtext.ScrolledText(
        win, font=("Consolas", 9), wrap=tk.NONE, height=16)
    summary_text.pack(fill=tk.X, padx=4, pady=(4, 2))
    summary_text.insert(tk.END, "\n".join(summary_lines))
    summary_text.configure(state="disabled")

    # Two side-by-side panels with a long region-label column need a
    # wide aspect; 14.6x9.4 is what the layout was tuned against.  The
    # canvas dpi is low so that 14.6x9.4 in maps to ~934x602 px rather
    # than 1460x940 and the window still fits a 1366-wide display
    # alongside the summary text and toolbar -- the same trick as the
    # population pyramid tab.  It costs nothing in export quality:
    # this function does all its own layout from rendered pixel
    # extents, so it is dpi-invariant, and the toolbar's save path
    # re-renders at 300 dpi.
    bar_fig = Figure(figsize=(14.6, 9.4), dpi=64)
    plot_annual_QALY_bar(benefit, bar_fig)
    bar_canvas = FigureCanvasTkAgg(bar_fig, master=win)
    bar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    _prefix = _modality_filename_prefix(benefit.get("modality_region_map"))
    # %g, and no int(), so a fractional change like -17.5 does not
    # truncate to a filename claiming 17 %.  "." is not filename-safe
    # in a suffix position, so use "p".
    _pct = f"{abs(_chg_pct):g}".replace(".", "p")
    _verb = "lost" if _is_incr else "gained"
    bar_canvas.get_default_filename = (
        lambda: f"{_prefix}_annual_qalys_{_verb}_{_pct}pct_stacked.png"
    )
    _attach_toolbar(bar_canvas, win)
    _fit_window_to_content(win)


def _show_annual_burden(root, _rq_cache):
    """Open the annual-burden window computed by
    ``compute_annual_burden`` on the cached sweep."""
    sweep = _rq_cache.get("sweep")
    if sweep is None:
        messagebox.showinfo(
            "No data",
            "Run a simulation first, then the plot will be available.")
        return
    burden = compute_annual_burden(sweep)
    win = tk.Toplevel(root)
    win.title("Annual QALYs lost (baseline)")
    win.transient(root)

    cohort = burden.get("cohort_total", {})
    pr = burden.get("per_region", [])

    # Per-modality QALY totals (grouping the per-region-and-modality
    # rows currently displayed) so the reader can see, at a glance, which
    # modalities contribute how much of the burden within the selected
    # set of modalities/regions shown below.
    mod_QALY: dict[str, float] = {}
    for r in pr:
        mod = r.get("modality", "") or ""
        mod_QALY[mod] = mod_QALY.get(mod, 0.0) + r.get("QALY_annual", 0.0)
    total_selected_QALY = sum(mod_QALY.values())

    _BL_LBL_W = 41

    summary_lines = [
        "Expected annual QALY burden (baseline, no dose reduction)",
        f"  Intervals: {SIM_ANNUAL_RANGE_LABEL} "
        "(point estimates are analytical).",
        f"  {'Total QALYs lost/yr:':<{_BL_LBL_W}}{cohort.get('QALY_annual', 0):.3f} {_rng(cohort.get('QALY_annual_ci_lo'), cohort.get('QALY_annual_ci_hi'))}",
        f"  {'Radiation-induced cancers/yr:':<{_BL_LBL_W}}{cohort.get('cases_annual', 0):.3f} {_rng(cohort.get('cases_annual_ci_lo'), cohort.get('cases_annual_ci_hi'))}",
        f"  {'Fatal radiation-induced cancers/yr:':<{_BL_LBL_W}}{cohort.get('deaths_annual', 0):.3f} {_rng(cohort.get('deaths_annual_ci_lo'), cohort.get('deaths_annual_ci_hi'))}",
        f"  {'Total collective effective dose/yr:':<{_BL_LBL_W}}{cohort.get('dose_sv_annual', 0):.4f} person Sv",
        f"  {'Total examinations/yr:':<{_BL_LBL_W}}{cohort.get('total_scans', 0):,.0f}",
        f"  Total QALYs lost/yr in the selected modalities: {total_selected_QALY:.3f}",
        "",
        "Contribution by modality (share of the selected modalities' QALYs lost):",
    ]
    for mod, val in sorted(mod_QALY.items(), key=lambda kv: kv[1], reverse=True):
        pct = (val / total_selected_QALY * 100.0) if total_selected_QALY > 0 else 0.0
        summary_lines.append(f"  {mod[:14]:<14} {val:>10.3f}  ({pct:>5.1f}%)")
    summary_lines.append("")
    summary_lines.append("By modality and region (sorted by annual QALYs lost):")
    summary_lines.append(
        f"  {'#':>2} {'Modality':<14} {'Region':<30} {'Person Sv/yr':>14} {'QALYs lost':>14} {'Cancers':>14} {'Fatal':>12} {'Exams/yr':>11}"
    )
    summary_lines.append(
        f"  {'─'*2} {'─'*14} {'─'*30} {'─'*14} {'─'*14} {'─'*14} {'─'*12} {'─'*11}"
    )
    for rank, r in enumerate(pr, 1):
        mod = r.get("modality", "") or ""
        summary_lines.append(
            f"  {rank:>2} {mod[:14]:<14} {r['region'][:30]:<30} {r.get('dose_sv_annual', 0):>14.4f} "
            f"{r['QALY_annual']:>14.3f} {r['cases_annual']:>14.3f} {r['deaths_annual']:>12.3f} "
            f"{r['total_scans']:>11,.0f}"
        )
        summary_lines.append(
            f"     {'':<14} {'':<30} {'95% range:':>14} "
            f"{_rng(r.get('QALY_annual_ci_lo'), r.get('QALY_annual_ci_hi'))} "
            f"{_rng(r.get('cases_annual_ci_lo'), r.get('cases_annual_ci_hi'))} "
            f"{_rng(r.get('deaths_annual_ci_lo'), r.get('deaths_annual_ci_hi'))}"
        )
    summary_lines.append("")
    summary_lines.append(
        "Note: Annual QALYs lost depend on local examination volumes"
        " and age distribution.  The per-examination burden shown in the"
        " heatmap is the transferable metric."
    )
    summary_text = scrolledtext.ScrolledText(
        win, font=("Consolas", 9), wrap=tk.NONE, height=16)
    summary_text.pack(fill=tk.X, padx=4, pady=(4, 2))
    summary_text.insert(tk.END, "\n".join(summary_lines))
    summary_text.configure(state="disabled")

    # Same figure geometry as the annual-effect window (see there).
    bar_fig = Figure(figsize=(14.6, 9.4), dpi=64)
    plot_annual_burden_bar(burden, bar_fig)
    bar_canvas = FigureCanvasTkAgg(bar_fig, master=win)
    bar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    _prefix = _modality_filename_prefix(burden.get("modality_region_map"))
    bar_canvas.get_default_filename = (
        lambda: f"{_prefix}_annual_qalys_lost_baseline_stacked.png"
    )
    _attach_toolbar(bar_canvas, win)
    _fit_window_to_content(win)


def _year_batch_row(label: str, ys: dict) -> tuple:
    """Format one table row from a single year's stats dict."""
    iters = float(ys.get("iterations", 0) or 0)
    exams = sum((ys.get("examinations_by_modality") or {}).values())
    if not exams and iters:
        exams = float(ys.get("acquisitions", 0) or 0) / iters
    return (
        label,
        f"{exams:,.0f}",
        f"{iters:,.0f}",
        f"{float(ys.get('total_sv', 0.0)) / (iters or 1.0):,.2f}",
        f"{float(ys.get('cancer_cases', 0.0)) / (iters or 1.0):,.3f}",
        f"{float(ys.get('cancer_deaths', 0.0)) / (iters or 1.0):,.3f}",
        f"{float(ys.get('total_detriment', 0.0)) / (iters or 1.0):,.4f}",
        f"{float(ys.get('QALY_lost', 0.0)) / (iters or 1.0):,.2f}",
    )


def start_gui(file_list, n_iterations, min_latency_params, output_dir, population, run_simulation,
              risk_model: str = "REIC", backend_info_getter=None, sim_config: dict = None):
    """Launch the Dose2QALY results GUI.

    Creates a Tkinter window with tabs for setup, progress, results,
    plots, cancer events, frequent-flyer analysis, and research questions.
    Runs ``run_simulation`` in a background thread and streams live updates.
    """
    # Install the centralised matplotlib style once before any Figure is
    # constructed so every chart inherits fonts, grid style and the
    # Okabe-Ito colour-blind-safe default palette.
    _apply_style()

    root = tk.Tk()
    root.title("Dose2QALY")
    _place_window(root, 1300, 850, min_w=900, min_h=600)
    try:
        root.state('zoomed')
    except Exception:
        root.attributes('-zoomed', True)

    top = ttk.Frame(root, padding=5)
    top.pack(fill=tk.X)
    status_var = tk.StringVar(value="Starting...")
    progress_var = tk.DoubleVar(value=0.0)
    detail_var = tk.StringVar(value="Preparing GUI...")
    backend_var = tk.StringVar(value="Backend: detecting...")
    ttk.Label(top, textvariable=status_var, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=5)
    ttk.Label(top, textvariable=detail_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=12)
    ttk.Progressbar(top, variable=progress_var, maximum=100, length=350).pack(side=tk.RIGHT, padx=5)
    ttk.Label(root, textvariable=backend_var, font=("Segoe UI", 9)).pack(anchor=tk.W, padx=10, pady=(0, 4))

    try:
        backend_info = backend_info_getter() if callable(backend_info_getter) else None
    except Exception:
        backend_info = None
    backend_var.set(_format_backend_summary(backend_info))

    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Shared hover tooltip instance reused by the tab strip and tree headers.
    _gui_tooltip = _ToolTip(root)

    def _on_tab_strip_motion(event):
        """Show a one-line description for the tab currently under the cursor."""
        try:
            idx = nb.index(f"@{event.x},{event.y}")
        except Exception:
            _gui_tooltip.hide()
            return
        try:
            label = str(nb.tab(idx, "text"))
        except Exception:
            _gui_tooltip.hide()
            return
        hint = _TAB_HINTS.get(label)
        if hint:
            _gui_tooltip.show(f"{label} — {hint}", event.x_root, event.y_root)
        else:
            _gui_tooltip.hide()

    def _on_tab_strip_leave(_event):
        _gui_tooltip.hide()

    nb.bind("<Motion>", _on_tab_strip_motion, add="+")
    nb.bind("<Leave>",  _on_tab_strip_leave,  add="+")

    # Track currently visible tab for selective updates during simulation
    current_tab_index = tk.IntVar(value=0)
    # Canonical top-level tab order; used by name for live-update dispatch.
    tab_names = [
        "Summary",
        "By Modality",
        "By Body Region",
        "By Cancer Type",
        "Cancer Events",
        "Patient Cohort",
        "Research Questions",
        "Risk Analysis",
    ]


    # Lazily refresh the Cancer Events Age Distribution histogram on first visit.
    def _on_tab_change(event):
        idx = nb.index(nb.select())
        current_tab_index.set(idx)
        try:
            tab_text = str(nb.tab(idx, "text"))
            if tab_text == "Cancer Events":
                # Check if Age Distribution subtab is selected
                try:
                    ev_idx = ev_sub_nb.index(ev_sub_nb.select())
                    if str(ev_sub_nb.tab(ev_idx, "text")) == "Age Distribution" and not ev_age_histogram_drawn:
                        _upd_age_histogram()
                except Exception:
                    pass
        except Exception:
            pass

    nb.bind("<<NotebookTabChanged>>", _on_tab_change)

    t1 = ttk.Frame(nb); nb.add(t1, text="Summary")
    summary_top = ttk.Frame(t1)
    summary_top.pack(fill=tk.BOTH, expand=True)

    # Modality filter for the whole Summary tab.  Defaults to "All
    # modalities"; selecting one re-renders the text summary, the
    # per-iteration cancer/death distributions, the cancer-type pie, the
    # QALY/life-year distributions and the frequency-vs-burden charts using
    # only that modality's exams and events.
    summary_modality_bar = ttk.Frame(summary_top)
    summary_modality_bar.pack(fill=tk.X, pady=(2, 0))
    ttk.Label(summary_modality_bar, text="Modality:").pack(side=tk.LEFT, padx=(0, 6))
    summary_mod_view_var = tk.StringVar(value="All modalities")
    summary_mod_view_cb = ttk.Combobox(summary_modality_bar, textvariable=summary_mod_view_var,
                                       width=28, state="readonly")
    summary_mod_view_cb["values"] = ["All modalities"]
    summary_mod_view_cb.pack(side=tk.LEFT)

    # Per-100,000-examinations projection toggle.  Re-expresses the headline
    # averages (cancers/deaths/heritable/detriment/QALYs/life-years, with
    # their 95% sampled ranges) and the four per-iteration distribution
    # charts (cancer/death/QALY/life-years-lost) as a linear rescale to a
    # fixed cohort size -- the same per_n_scans convention
    # research_questions uses for the Region x Age dose-reduction sweep.
    # Honours the Modality dropdown above: with a modality selected, "100,000
    # examinations" means 100,000 examinations of THAT modality.  Off by
    # default so the tab keeps reporting this cohort's own raw numbers.
    summary_per100k_var = tk.BooleanVar(value=False)
    summary_per100k_cb = ttk.Checkbutton(
        summary_modality_bar, text="Per 100,000 examinations",
        variable=summary_per100k_var, command=lambda: _render_summary())
    summary_per100k_cb.pack(side=tk.LEFT, padx=(16, 0))

    # Cancer-visibility filter pinned above the split so toggles don't lose scroll position.
    summary_cancer_filter_bar = ttk.Frame(summary_top)
    summary_cancer_filter_bar.pack(fill=tk.X, pady=(2, 4))
    ttk.Label(summary_cancer_filter_bar, text="Show cancers:").pack(side=tk.LEFT, padx=(0, 6))

    # Resizable split: text summary on top, charts on bottom.
    summary_paned = ttk.PanedWindow(summary_top, orient=tk.VERTICAL)
    summary_paned.pack(fill=tk.BOTH, expand=True)

    # Text summary on the left; on the right a notebook with the
    # selection-of-examinations flow diagram (the manuscript's STROBE-style
    # figure, drawn from stats["_participant_flow"]) and the adjustments
    # (competing-mortality strata and weight adjustment).
    summary_text_pane = ttk.PanedWindow(summary_paned, orient=tk.HORIZONTAL)
    summary_text_frame = ttk.Frame(summary_text_pane)
    summary_text = scrolledtext.ScrolledText(summary_text_frame, font=("Consolas", 9),
                                             wrap=tk.NONE, width=78, height=20)
    summary_text.pack(fill=tk.BOTH, expand=True)
    summary_text_pane.add(summary_text_frame, weight=3)

    # The diagram keeps a readable, fixed font: the figure is drawn at a
    # fixed width and grows in height to fit the ledger, and sits inside
    # a scrollable Tk canvas instead of being squeezed into the pane.
    summary_side_nb = ttk.Notebook(summary_text_pane)
    summary_flow_frame = ttk.Frame(summary_side_nb)
    summary_side_nb.add(summary_flow_frame, text="Participant flow")
    summary_adj_frame = ttk.Frame(summary_side_nb)
    summary_side_nb.add(summary_adj_frame, text="Adjustments")
    summary_adj_text = scrolledtext.ScrolledText(
        summary_adj_frame, font=("Consolas", 9), wrap=tk.NONE)
    summary_adj_text.pack(fill=tk.BOTH, expand=True)
    summary_flow_save_bar = ttk.Frame(summary_flow_frame)
    summary_flow_save_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 2))
    summary_flow_scroll = tk.Canvas(summary_flow_frame, highlightthickness=0, bd=0,
                                    background="white")
    summary_flow_vsb = ttk.Scrollbar(summary_flow_frame, orient=tk.VERTICAL,
                                     command=summary_flow_scroll.yview)
    summary_flow_hsb = ttk.Scrollbar(summary_flow_frame, orient=tk.HORIZONTAL,
                                     command=summary_flow_scroll.xview)
    summary_flow_scroll.configure(yscrollcommand=summary_flow_vsb.set,
                                  xscrollcommand=summary_flow_hsb.set)
    summary_flow_vsb.pack(side=tk.RIGHT, fill=tk.Y)
    summary_flow_hsb.pack(side=tk.BOTTOM, fill=tk.X)
    summary_flow_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    summary_flow_fig = Figure(figsize=(5.6, 5.0), dpi=90)
    summary_flow_ax = summary_flow_fig.add_axes([0.01, 0.01, 0.98, 0.98])
    summary_flow_ax.set_axis_off()
    summary_flow_canvas = FigureCanvasTkAgg(summary_flow_fig, master=summary_flow_scroll)
    summary_flow_widget = summary_flow_canvas.get_tk_widget()
    summary_flow_scroll.create_window((0, 0), window=summary_flow_widget, anchor="nw")

    def _flow_sync_scrollregion(_event=None):
        """Size the Tk widget to the figure (which plot_participant_flow may
        have grown) and update the scroll region to match."""
        w_in, h_in = summary_flow_fig.get_size_inches()
        w_px, h_px = int(round(w_in * summary_flow_fig.dpi)), int(round(h_in * summary_flow_fig.dpi))
        summary_flow_widget.configure(width=w_px, height=h_px)
        summary_flow_scroll.configure(scrollregion=(0, 0, w_px, h_px))

    def _flow_on_mousewheel(event):
        if event.state & 0x1:   # Shift: horizontal
            summary_flow_scroll.xview_scroll(int(-event.delta / 120), "units")
        else:
            summary_flow_scroll.yview_scroll(int(-event.delta / 120), "units")

    for _w in (summary_flow_scroll, summary_flow_widget):
        _w.bind("<MouseWheel>", _flow_on_mousewheel)
        _w.bind("<Shift-MouseWheel>", _flow_on_mousewheel)
    _flow_sync_scrollregion()

    ttk.Label(summary_flow_save_bar, text="Selection of examinations:",
              font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 4))
    ttk.Button(summary_flow_save_bar, text="Save figure", width=12,
               command=lambda: _save_full_figure_as_shown(summary_flow_fig,
                                                          "participant_flow")).pack(side=tk.LEFT, padx=2)
    summary_text_pane.add(summary_side_nb, weight=2)
    summary_paned.add(summary_text_pane, weight=2)

    summary_chart_pane = ttk.Frame(summary_paned)
    # constrained_layout handles subplot spacing automatically on every
    # redraw, so the refresh path does not need a per-update tight_layout
    # call (which re-solves the whole GridSpec and stalls the Tk event loop
    # for ~10-40 ms on large figures).
    #
    # The charts live in a Notebook so the (cancer + death) count
    # distributions and the (QALYs + life-years) distributions each get
    # their own tab in the SAME pane location.
    summary_chart_nb = ttk.Notebook(summary_chart_pane)
    summary_chart_nb.pack(fill=tk.BOTH, expand=True)

    # "Cancers" tab — per-year cancer & death count distributions plus the
    # cancer-type pie.
    summary_cancers_tab = ttk.Frame(summary_chart_nb)
    summary_chart_nb.add(summary_cancers_tab, text="Cancers")
    summary_iter_fig = Figure(figsize=(12, 3.2), dpi=90, constrained_layout=True)
    summary_iter_ax1 = summary_iter_fig.add_subplot(131)
    summary_iter_ax2 = summary_iter_fig.add_subplot(132)
    summary_iter_ax3 = summary_iter_fig.add_subplot(133)
    summary_iter_canvas = FigureCanvasTkAgg(summary_iter_fig, master=summary_cancers_tab)

    # Save bars are packed BEFORE the canvases: Tk's packer allocates space
    # in pack order, so a canvas packed first with expand=True swallows the
    # whole pane when the figure's requested height exceeds it and the bar
    # packed afterwards is squeezed to zero height (it was invisible on the
    # taller "Frequency vs burden" figure).
    summary_cancers_save_bar = ttk.Frame(summary_cancers_tab)
    summary_cancers_save_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 2))
    ttk.Label(summary_cancers_save_bar, text="Save subfigure:",
              font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 4))
    ttk.Button(summary_cancers_save_bar, text="Cases", width=10,
               command=lambda: _save_subplot(summary_iter_fig, summary_iter_ax1,
                                             "cancer_cases_distribution")).pack(side=tk.LEFT, padx=2)
    ttk.Button(summary_cancers_save_bar, text="Deaths", width=10,
               command=lambda: _save_subplot(summary_iter_fig, summary_iter_ax2,
                                             "cancer_deaths_distribution")).pack(side=tk.LEFT, padx=2)
    ttk.Button(summary_cancers_save_bar, text="Cancer types", width=12,
               command=lambda: _save_subplot(summary_iter_fig, summary_iter_ax3,
                                             "cancer_type_distribution")).pack(side=tk.LEFT, padx=2)
    summary_iter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # "QALYs" tab — per-year QALYs lost (overall) & life-years-lost (fatal
    # only) distributions; the continuous analogues of the count panels.
    summary_QALY_tab = ttk.Frame(summary_chart_nb)
    summary_chart_nb.add(summary_QALY_tab, text="QALYs")
    summary_QALY_fig = Figure(figsize=(12, 3.2), dpi=90, constrained_layout=True)
    summary_QALY_ax1 = summary_QALY_fig.add_subplot(121)
    summary_QALY_ax2 = summary_QALY_fig.add_subplot(122)
    summary_QALY_canvas = FigureCanvasTkAgg(summary_QALY_fig, master=summary_QALY_tab)

    summary_QALY_save_bar = ttk.Frame(summary_QALY_tab)
    summary_QALY_save_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 2))
    ttk.Label(summary_QALY_save_bar, text="Save subfigure:",
              font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 4))
    ttk.Button(summary_QALY_save_bar, text="QALYs", width=18,
               command=lambda: _save_subplot(summary_QALY_fig, summary_QALY_ax1,
                                             "qalys_lost_distribution")).pack(side=tk.LEFT, padx=2)
    ttk.Button(summary_QALY_save_bar, text="Life-years lost (fatal)", width=20,
               command=lambda: _save_subplot(summary_QALY_fig, summary_QALY_ax2,
                                             "life_years_lost_distribution_fatal_only")).pack(side=tk.LEFT, padx=2)
    summary_QALY_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # "Frequency vs burden" tab — two aligned horizontal bar charts comparing
    # cancer frequency (%) with QALYL contribution (%) per cancer type.
    summary_freq_burden_tab = ttk.Frame(summary_chart_nb)
    summary_chart_nb.add(summary_freq_burden_tab, text="Frequency vs burden")
    summary_freq_burden_fig = Figure(figsize=(8.4, 5), dpi=90, constrained_layout=True)
    summary_freq_burden_ax1 = summary_freq_burden_fig.add_subplot(121)
    summary_freq_burden_ax2 = summary_freq_burden_fig.add_subplot(122)
    summary_freq_burden_canvas = FigureCanvasTkAgg(summary_freq_burden_fig, master=summary_freq_burden_tab)

    summary_freq_burden_save_bar = ttk.Frame(summary_freq_burden_tab)
    summary_freq_burden_save_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 2))
    ttk.Label(summary_freq_burden_save_bar, text="Save:",
              font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 4))
    ttk.Button(summary_freq_burden_save_bar, text="Full figure", width=16,
               command=lambda: _save_full_figure_as_shown(summary_freq_burden_fig,
                                                          "cancer_frequency_vs_qaly_burden")).pack(side=tk.LEFT, padx=2)
    ttk.Button(summary_freq_burden_save_bar, text="Frequency", width=12,
               command=lambda: _save_subplot_as_shown(summary_freq_burden_fig, summary_freq_burden_ax1,
                                                      "cancer_frequency")).pack(side=tk.LEFT, padx=2)
    ttk.Button(summary_freq_burden_save_bar, text="QALY burden", width=14,
               command=lambda: _save_subplot_as_shown(summary_freq_burden_fig, summary_freq_burden_ax2,
                                                      "cancer_qalyl_burden")).pack(side=tk.LEFT, padx=2)
    summary_freq_burden_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    summary_paned.add(summary_chart_pane, weight=1)

    t2 = ttk.Frame(nb); nb.add(t2, text="By Modality")
    mod_filter_bar = ttk.Frame(t2)
    mod_filter_bar.pack(fill=tk.X, padx=5, pady=(5, 2))
    ttk.Label(mod_filter_bar, text="View:").pack(side=tk.LEFT, padx=(0, 6))
    mod_view_var = tk.StringVar(value="All modalities")
    mod_view_cb = ttk.Combobox(mod_filter_bar, textvariable=mod_view_var, width=28, state="readonly")
    mod_view_cb["values"] = ["All modalities"]
    mod_view_cb.pack(side=tk.LEFT)

    ttk.Label(mod_filter_bar, text="Device:").pack(side=tk.LEFT, padx=(12, 6))
    mod_device_var = tk.StringVar(value="All devices")
    mod_device_cb = ttk.Combobox(mod_filter_bar, textvariable=mod_device_var, width=24, state="readonly")
    mod_device_cb["values"] = ["All devices"]
    mod_device_cb.pack(side=tk.LEFT)

    ttk.Label(mod_filter_bar, text="Year:").pack(side=tk.LEFT, padx=(12, 6))
    mod_year_var = tk.StringVar(value="All years")
    mod_year_cb = ttk.Combobox(mod_filter_bar, textvariable=mod_year_var, width=12, state="readonly")
    mod_year_cb["values"] = ["All years"]
    mod_year_cb.pack(side=tk.LEFT)

    # Visual summary bar chart for ED/scan and detriment/scan.
    mod_fig = Figure(figsize=(12, 3.38), dpi=90, constrained_layout=True)
    mod_ax_ed  = mod_fig.add_subplot(121)
    mod_ax_det = mod_fig.add_subplot(122)
    mod_canvas = FigureCanvasTkAgg(mod_fig, master=t2)
    mod_canvas.get_tk_widget().pack(fill=tk.X, expand=False, padx=5, pady=(2, 0))

    # Quartile-shading legend for the detriment-per-scan table rows.
    _MOD_Q_COLORS = [
        (_QUARTILE_RAMP_LIGHT[0], "Low (Q1)"),
        (_QUARTILE_RAMP_LIGHT[1], "Q2"),
        (_QUARTILE_RAMP_LIGHT[2], "Q3"),
        (_QUARTILE_RAMP_LIGHT[3], "High (Q4)"),
    ]
    mod_legend_bar = ttk.Frame(t2)
    mod_legend_bar.pack(fill=tk.X, padx=5, pady=(2, 0))
    ttk.Label(mod_legend_bar, text="Row shading — Detriment / exam:",
              font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(0, 8))
    for _color, _name in _MOD_Q_COLORS:
        _swatch = tk.Frame(mod_legend_bar, width=16, height=14,
                           background=_color,
                           highlightbackground="#888", highlightthickness=1)
        _swatch.pack(side=tk.LEFT, padx=(6, 2))
        _swatch.pack_propagate(False)
        ttk.Label(mod_legend_bar, text=_name,
                  font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 4))
    ttk.Label(mod_legend_bar,
              text="  (lower is better — modalities with the lowest detriment per individual exam are greenest)",
              font=("Segoe UI", 8), foreground="#555").pack(side=tk.LEFT, padx=(8, 0))

    mod_tree = ttk.Treeview(t2, columns=(
        "mod", "scans", "dose", "edps", "cases", "deaths", "det", "crpv", "drpv", "dps",
    ), show="headings")
    for c, h, w in [
        ("mod", "Modality", 200), ("scans", "Exams / year", 80), ("dose", "Collective dose / year (person Sv)", 130),
        ("edps", "ED/exam (mSv)", 95), ("cases", "Cancers / year", 90), ("deaths", "Deaths / year", 90), ("det", "Detriment / year", 110),
        ("crpv", "CancerRisk/Sv%", 100), ("drpv", "DeathRisk/Sv%", 100),
        ("dps", "Detriment caused/exam (hours)", 170),
    ]:
        mod_tree.heading(c, text=h)
        mod_tree.column(c, width=w, anchor=tk.W if c == "mod" else tk.E)
    # Quartile-based row shading by detriment/scan rank; tags are assigned in _upd_mod.
    mod_tree.tag_configure("mod_q1", background=_MOD_Q_COLORS[0][0])  # green   — lowest quartile
    mod_tree.tag_configure("mod_q2", background=_MOD_Q_COLORS[1][0])  # yellow  — second
    mod_tree.tag_configure("mod_q3", background=_MOD_Q_COLORS[2][0])  # orange  — third
    mod_tree.tag_configure("mod_q4", background=_MOD_Q_COLORS[3][0])  # red     — highest
    mod_tree.pack(fill=tk.BOTH, expand=True)
    _bind_header_tooltips(mod_tree, _MOD_TREE_TOOLTIPS, _gui_tooltip)

    t3 = ttk.Frame(nb); nb.add(t3, text="By Cancer Type")
    cancer_sub_nb = ttk.Notebook(t3)
    cancer_sub_nb.pack(fill=tk.BOTH, expand=True)

    # -- Subtab 1: Cancer Summary (cancers/year, deaths/year, lethality) --
    cancer_summary_frame = ttk.Frame(cancer_sub_nb)
    cancer_sub_nb.add(cancer_summary_frame, text="Summary")
    cancer_text = scrolledtext.ScrolledText(cancer_summary_frame, font=("Consolas", 9),
                                           wrap=tk.NONE, width=78, height=30)
    cancer_text.pack(fill=tk.BOTH, expand=True)

    # -- Subtab 2: Cancers per Sv --
    cancer_per_sv_frame = ttk.Frame(cancer_sub_nb)
    cancer_sub_nb.add(cancer_per_sv_frame, text="Cancers per Sv")
    cancer_per_sv_btn_frame = ttk.Frame(cancer_per_sv_frame)
    cancer_per_sv_btn_frame.pack(fill=tk.X, pady=(2, 0))
    ttk.Button(cancer_per_sv_btn_frame, text="Compare with ICRP152 values",
               command=lambda: _show_5050_comparison(results)).pack(side=tk.LEFT, padx=4)
    cancer_per_sv_text = scrolledtext.ScrolledText(cancer_per_sv_frame, font=("Consolas", 9),
                                                   wrap=tk.NONE, width=78, height=30)
    cancer_per_sv_text.pack(fill=tk.BOTH, expand=True)

    t5 = ttk.Frame(nb); nb.add(t5, text="By Body Region")
    reg_sub_nb = ttk.Notebook(t5)
    reg_sub_nb.pack(fill=tk.BOTH, expand=True)

    # -- Subtab 1: Region Statistics (existing content) --
    reg_stats_frame = ttk.Frame(reg_sub_nb)
    reg_sub_nb.add(reg_stats_frame, text="Region Statistics")
    reg_filter_bar = ttk.Frame(reg_stats_frame)
    reg_filter_bar.pack(fill=tk.X, padx=5, pady=(5, 2))
    ttk.Label(reg_filter_bar, text="Modality:").pack(side=tk.LEFT, padx=(0, 6))
    reg_view_var = tk.StringVar(value="All modalities")
    reg_view_cb = ttk.Combobox(reg_filter_bar, textvariable=reg_view_var, width=28, state="readonly")
    reg_view_cb["values"] = ["All modalities"]
    reg_view_cb.pack(side=tk.LEFT, padx=(0, 12))
    ttk.Label(reg_filter_bar, text="Source:").pack(side=tk.LEFT, padx=(0, 4))
    reg_source_var = tk.StringVar(value="All files (combined)")
    reg_source_cb = ttk.Combobox(reg_filter_bar, textvariable=reg_source_var,
                                 width=28, state="readonly")
    reg_source_cb["values"] = ["All files (combined)"]
    reg_source_cb.pack(side=tk.LEFT)
    # Selector map for the region source dropdown (label -> (kind, key)).
    reg_source_map = {}
    reg_fig = Figure(figsize=(12, 6), dpi=85)
    reg_canvas = FigureCanvasTkAgg(reg_fig, master=reg_stats_frame)
    reg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    reg_text = scrolledtext.ScrolledText(reg_stats_frame, font=("Consolas", 9), wrap=tk.NONE, height=14)
    reg_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))

    # -- Subtab 2: Unrecognized Regions --
    unrec_frame = ttk.Frame(reg_sub_nb)
    reg_sub_nb.add(unrec_frame, text="Unrecognized Regions")
    unrec_label = ttk.Label(unrec_frame, text="Exam descriptions that could not be mapped to a known region (assigned to 'Unknown/unspecified region'):")
    unrec_label.pack(fill=tk.X, padx=5, pady=(5, 2))
    unrec_tree = ttk.Treeview(unrec_frame, columns=("modality", "description"), show="headings")
    unrec_tree.heading("modality", text="Modality")
    unrec_tree.heading("description", text="Description")
    unrec_tree.column("modality", width=120, anchor=tk.W)
    unrec_tree.column("description", width=600, anchor=tk.W)
    unrec_sb = ttk.Scrollbar(unrec_frame, orient=tk.VERTICAL, command=unrec_tree.yview)
    unrec_tree.configure(yscrollcommand=unrec_sb.set)
    unrec_sb.pack(side=tk.RIGHT, fill=tk.Y)
    unrec_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))
    unrec_count_var = tk.StringVar(value="0 unrecognized descriptions")
    ttk.Label(unrec_frame, textvariable=unrec_count_var).pack(fill=tk.X, padx=5, pady=(0, 5))

    t6 = ttk.Frame(nb); nb.add(t6, text="Patient Cohort")

    # Fixed 65 / 35 vertical split
    # The pyramid (plus its toolbar and buttons) gets the top 65%, the
    # sex-breakdown and modality-mix pair the bottom 35%.  The pyramid is
    # the tab's headline chart and is held at a 14:8 aspect, so it converts
    # extra height into a bigger chart, whereas the two charts below are
    # short and wide and just gain whitespace.
    #
    # place(relheight=...) rather than pack(expand=True) or grid row
    # weights: both of those distribute only the space left *after* every
    # child's requested size is satisfied.  The bottom canvas asks for its
    # figure's natural 3.6 in x 90 dpi = 324 px, which is far more than the
    # pyramid section's controls ask for, so the leftover-based split lands
    # nowhere near 65/35 (and inverts on a short window).  place ignores
    # requested sizes, making the ratio exact at every window size.
    _T6_TOP_FRAC = 0.65
    t6_top = ttk.Frame(t6)
    t6_top.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=_T6_TOP_FRAC)
    t6_bottom = ttk.Frame(t6)
    t6_bottom.place(relx=0.0, rely=_T6_TOP_FRAC,
                    relwidth=1.0, relheight=1.0 - _T6_TOP_FRAC)

    # Age pyramid (top, rebuilt by plot_age_histogram which calls
    #    fig.clear() and subplots(1, N); keeping this figure separate
    #    from the sex + modality-mix figure avoids those clears stomping
    #    on the other two charts).
    # The pyramid is laid out for a 14:8 canvas *per population panel*;
    # that ratio is fixed here and preserved at every window size by
    # _fit_pyramid_box below, with the figure widening to 14 in per panel
    # when several populations are drawn side by side.
    _PYRAMID_PANEL_W = 14.0
    _PYRAMID_FIGH = 8.0
    # Below this per-panel width the age labels and per-bar counts start
    # colliding, so the chart stops shrinking and the tab scrolls/clips
    # instead.
    _PYRAMID_PANEL_MIN_W = 760

    # Selection/view state for the pyramid controls (documented with the
    # dropdowns below); it has to exist before the figure is created
    # because the figure's inch geometry depends on ``n_panels``.
    pyramid_state = {
        "ed_view": None,        # fallback ExamColumns (last year's raw data)
        "detail_files": [],
        "cohort": None,         # CohortByFile (compact, all files/years)
        "selector_map": {},     # dropdown label -> (kind, key)
        "n_panels": 1,          # population panels in the current pyramid
    }

    def _pyramid_geometry():
        """(fig width in, fig height in, aspect, min box width px) for the
        current panel count -- 14 in wide and 760 px minimum per panel."""
        n = max(1, int(pyramid_state["n_panels"]))
        figw = _PYRAMID_PANEL_W * n
        return figw, _PYRAMID_FIGH, figw / _PYRAMID_FIGH, _PYRAMID_PANEL_MIN_W * n

    _figw, _figh, _aspect, _min_w = _pyramid_geometry()
    chart_fig = Figure(figsize=(_figw, _figh), dpi=75,
                       constrained_layout=True)

    # Responsive, aspect-preserving sizing
    # Packing the canvas fill=BOTH/expand=True on its own is not enough:
    # FigureCanvasTkAgg resizes the *figure* to whatever the widget is
    # given, so on a maximised window the pyramid was stretched to the full
    # tab width while sharing the tab's height with the sex/modality-mix
    # chart below -- roughly 4.7:1 instead of 14:8, an aspect at which the
    # value labels and the modality legend are crushed against the bars.
    #
    # So the chart gets an expanding *area* and is centred inside it at the
    # largest 14:8 rectangle that fits.  It therefore grows and shrinks with
    # the window (matching the charts below it) but is never distorted.
    chart_outer = ttk.Frame(t6_top)
    chart_outer.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))
    # Empty frame the toolbar is packed into, above the chart.
    chart_tb_holder = ttk.Frame(chart_outer)
    chart_tb_holder.pack(fill=tk.X)
    chart_area = ttk.Frame(chart_outer)
    chart_area.pack(fill=tk.BOTH, expand=True)
    chart_box = ttk.Frame(chart_area)
    # place() rather than pack() so the box can be centred at an explicit
    # size that is independent of both its parent's and its child's size.
    chart_box.place(relx=0.5, rely=0.5, anchor="center",
                    width=_min_w,
                    height=int(_min_w / _aspect))
    # Stop the child canvas from dictating the frame's size, so the
    # width/height set by _fit_pyramid_box is what actually applies.
    chart_box.pack_propagate(False)
    chart_canvas = FigureCanvasTkAgg(chart_fig, master=chart_box)
    chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _fit_pyramid_box(event=None):
        """Centre the pyramid at the largest 14:8-per-panel box it can hold."""
        avail_w = chart_area.winfo_width()
        avail_h = chart_area.winfo_height()
        if avail_w < 2 or avail_h < 2:
            return
        figw, figh, aspect, min_w = _pyramid_geometry()
        w = max(min_w, min(avail_w, int(avail_h * aspect)))
        h = max(2, int(round(w / aspect)))
        if (w, h) != (chart_box.winfo_width(), chart_box.winfo_height()):
            chart_box.place_configure(width=w, height=h)
        # Scale the figure's *dpi*, holding its inch size at 14x8 per panel,
        # so the whole chart zooms uniformly.  Letting the inch size grow
        # instead (what FigureCanvasTkAgg does by default) keeps every font
        # at a fixed point size while the axes expand around it, so text
        # effectively shrinks as the window is enlarged -- the opposite of
        # what resizing should do.  Driving dpi keeps fonts, line widths and
        # legend swatches at a constant fraction of the chart.
        chart_fig.set_dpi(max(40.0, w / figw))
        # FigureCanvasTkAgg's own resize handler derives the inch size from
        # (widget px / dpi), which lands back on 14x8 per panel given the dpi
        # above -- but only once it has seen the new dpi.  Setting it here
        # too keeps the invariant true even mid-resize, so anything reading
        # the figure's inch size is never handed a stale value.
        chart_fig.set_size_inches(figw, figh, forward=False)
        chart_canvas.draw_idle()

    chart_area.bind("<Configure>", _fit_pyramid_box)

    _pyramid_btn_frame = ttk.Frame(t6_top)
    _pyramid_btn_frame.pack(pady=(2, 4))

    def _export_pyramid(path, dpi=None):
        """Save the pyramid at a fixed publication size and resolution.

        The on-screen preview varies its dpi to fill the window, which would
        otherwise leak straight into the export: at ``bbox_inches="tight"``
        the saved pixel dimensions are (figure inches x save dpi), and the
        figure's inch size is what the resize handler leaves behind.  Pinning
        both to 14x8 in per panel here makes every export byte-for-byte
        reproducible no matter how the user had sized the window.
        """
        old_dpi = chart_fig.get_dpi()
        old_size = tuple(chart_fig.get_size_inches())
        try:
            figw, figh, _aspect, _min_w = _pyramid_geometry()
            chart_fig.set_dpi(100.0)
            chart_fig.set_size_inches(figw, figh, forward=False)
            chart_fig.savefig(path, dpi=dpi, bbox_inches="tight",
                              facecolor="white")
        finally:
            chart_fig.set_dpi(old_dpi)
            chart_fig.set_size_inches(*old_size, forward=False)
            chart_canvas.draw_idle()

    def _save_embedded_pyramid_png():
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Save age pyramid as PNG",
            initialfile="patient_age_pyramid.png",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if path:
            # 600 dpi over 14 in = ~8400 px wide: comfortably above the
            # 300 dpi journal floor even after a figure is scaled down to a
            # single column, and enough that the raster has no visible
            # softness at 100% zoom.
            _export_pyramid(path, dpi=600)
            messagebox.showinfo("Saved", f"PNG saved to:\n{path}")

    def _save_embedded_pyramid_pdf():
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Save age pyramid as PDF",
            initialfile="patient_age_pyramid.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            # No dpi: PDF output is vector, so it is resolution-independent.
            _export_pyramid(path)
            messagebox.showinfo("Saved", f"PDF saved to:\n{path}")

    ttk.Button(_pyramid_btn_frame, text="Save PNG (600 DPI)",
               command=_save_embedded_pyramid_png).pack(side=tk.LEFT, padx=4)
    ttk.Button(_pyramid_btn_frame, text="Save PDF (vector)",
               command=_save_embedded_pyramid_pdf).pack(side=tk.LEFT, padx=4)

    # Age pyramid modality + input-file controls
    # The pyramid always breaks each male/female bar down into stacked
    # modality segments, so the only control needed is the filter that
    # narrows it to a single modality (which renders as a one-segment
    # stack).  There is no plain/stacked toggle: the stacked view shows the
    # same totals and labels as the plain one plus the modality mix, so the
    # plain variant carried no information of its own.
    #
    # A second filter narrows the pyramid to a single input file.  When
    # several files were loaded the chart defaults to the *combination* of
    # every file (the whole cohort), and the dropdown lists each input file
    # by basename so the per-file age distribution can be inspected
    # individually.  The dropdown's values are rebuilt whenever the exam
    # data changes (see ``_upd_charts``): only files that actually
    # contributed exams to the current view are offered, so a per-year run
    # that only retained the last year's detail never shows the other
    # years' files as empty choices.  (``pyramid_state`` itself lives up by
    # the figure constants, since the figure's geometry already depends on
    # its panel count.)

    def _pyramid_modality_names():
        """Modality names for the dropdown — all modalities in the cohort."""
        cohort = pyramid_state.get("cohort")
        if cohort is not None and len(cohort) > 0:
            return cohort.combined().modalities()
        ed_view = pyramid_state.get("ed_view")
        if ed_view is None:
            return []
        cols = ed_view if isinstance(ed_view, ExamColumns) else ExamColumns.from_exam_data(ed_view)
        if len(cols) == 0:
            return []
        return sorted(set(str(m) for m in cols.modality if m))

    def _pyramid_file_choices():
        """Return ``[(label, kind, key), ...]`` for the source dropdown.

        When a :class:`CohortByFile` is available, choices come from its
        year-group and per-file entries — covering every file of every year,
        not just the last processed year's ``exam_data``.  Otherwise falls
        back to listing files present in the raw ``ed_view`` by
        ``_file_index``.
        """
        cohort = pyramid_state.get("cohort")
        if cohort is not None and len(cohort) > 0:
            return cohort.selector_choices()
        # Fallback: raw ExamColumns path (single run or no cohort built).
        ed_view = pyramid_state.get("ed_view")
        if ed_view is None:
            return []
        cols = ed_view if isinstance(ed_view, ExamColumns) else ExamColumns.from_exam_data(ed_view)
        if len(cols) == 0:
            return []
        present = set(int(fi) for fi in cols.file_index)
        detail_files = pyramid_state.get("detail_files") or []
        choices = []
        seen_labels = set()
        for fi in sorted(present):
            if 0 <= fi < len(detail_files):
                path = detail_files[fi]
            else:
                continue
            label = os.path.basename(str(path)) or str(path)
            base = label
            _n = 2
            while label in seen_labels:
                label = f"{base} ({_n})"
                _n += 1
            seen_labels.add(label)
            choices.append((label, "file_idx", fi))
        return choices

    def _resolve_cohort_view():
        """Resolve the current dropdown selection to an ExamColumns for plotting.

        Uses the compact :class:`CohortByFile` when available (the whole
        archive's files and year groups), falling back to filtering the raw
        ``ed_view`` by ``_file_index`` for single-run mode where no cohort
        was built.
        """
        cohort = pyramid_state.get("cohort")
        file_sel = pyramid_file_var.get()
        if cohort is not None and len(cohort) > 0:
            kind_key = pyramid_state.get("selector_map", {}).get(
                file_sel, ("all", None))
            fc = cohort.select(kind_key[0], kind_key[1])
            return fc.to_compact_exam_columns()
        # Fallback: filter raw ExamColumns by file_index.
        ed_view = pyramid_state.get("ed_view")
        if ed_view is None:
            return None
        if file_sel and file_sel != "All files (combined)":
            cols = ed_view if isinstance(ed_view, ExamColumns) else ExamColumns.from_exam_data(ed_view)
            target_fi = pyramid_state.get("file_label_to_index", {}).get(file_sel)
            if target_fi is not None and len(cols) > 0:
                mask = cols.file_index == target_fi
                if mask.any():
                    ed_view = cols.subset(mask)
                else:
                    return None
        return ed_view

    def _redraw_age_pyramid():
        # plot_age_histogram clears the figure and creates its own axes (one
        # per population when the selection mixes several), so any earlier
        # axes is a stale handle — always work on fresh axes from chart_fig.
        ed_view = _resolve_cohort_view()
        chart_fig.clear()
        ax = chart_fig.add_subplot(111)
        if ed_view is None or (hasattr(ed_view, "__len__") and len(ed_view) == 0):
            ax.set_title("No exams for this selection")
            ax.axis("off")
            pyramid_state["n_panels"] = 1
            _fit_pyramid_box()
            chart_canvas.draw_idle()
            return
        n = plot_age_histogram(
            ax, ed_view,
            modality_filter=pyramid_modality_var.get(),
        )
        pyramid_state["n_panels"] = max(1, int(n or 1))
        # Refit now so the box aspect and figure dpi match the new panel
        # count immediately, rather than waiting for the next <Configure>.
        _fit_pyramid_box()
        chart_canvas.draw_idle()

    def _apply_pyramid_modality_filter(ed_view):
        """Narrow an ExamColumns to the pyramid's selected modality.

        Mirrors the filtering ``plot_age_histogram`` does internally, so the
        sex-distribution pie and the modality-mix bar chart stay consistent
        with the pyramid when a single modality is picked.  Returns the
        original view unchanged when "All modalities" is selected, and
        ``None`` when the filter selects zero exams (so the caller can show
        the "No data" state instead of an empty chart).
        """
        modality_filter = pyramid_modality_var.get()
        if not modality_filter or modality_filter == "All modalities":
            return ed_view
        if ed_view is None or (hasattr(ed_view, "__len__") and len(ed_view) == 0):
            return ed_view
        # ed_view may be a raw list of dicts (single-run fallback) or an
        # ExamColumns; normalise so we can mask the modality array.
        ec = _ensure_exam_columns(ed_view)
        if ec is None or len(ec) == 0:
            return None
        mod_arr = np.asarray([m or "Unknown" for m in ec.modality], dtype=object)
        mask = mod_arr == modality_filter
        if not mask.any():
            return None
        return ec.subset(mask)

    def _redraw_cohort_extra():
        """Redraw the sex pie + modality mix for the current selection."""
        ed_view = _resolve_cohort_view()
        ed_view = _apply_pyramid_modality_filter(ed_view)
        if ed_view is None or (hasattr(ed_view, "__len__") and len(ed_view) == 0):
            cohort_sex_ax.clear()
            cohort_sex_ax.set_title("Sex distribution: No data")
            cohort_sex_ax.axis("off")
            cohort_mod_ax.clear()
            cohort_mod_ax.set_title("Modality Mix: No data")
            cohort_mod_ax.axis("off")
            cohort_extra_canvas.draw_idle()
            return
        plot_sex_breakdown(cohort_sex_ax, ed_view)
        modality_split_state["ed_view"] = ed_view
        plot_modality_mix(cohort_mod_ax, ed_view, split_by_sex=modality_split_state["on"])
        cohort_extra_canvas.draw_idle()

    def _on_pyramid_modality_selected(event=None):
        _redraw_age_pyramid()
        _redraw_cohort_extra()

    def _on_pyramid_file_selected(event=None):
        _redraw_age_pyramid()
        _redraw_cohort_extra()

    pyramid_control_frame = ttk.Frame(t6_top)
    pyramid_control_frame.pack(pady=(0, 4))
    ttk.Label(pyramid_control_frame, text="Modality:").pack(side=tk.LEFT, padx=(0, 4))
    pyramid_modality_var = tk.StringVar(value="All modalities")
    pyramid_modality_combo = ttk.Combobox(
        pyramid_control_frame, textvariable=pyramid_modality_var,
        values=["All modalities"], state="readonly", width=20,
    )
    pyramid_modality_combo.pack(side=tk.LEFT, padx=(0, 8))
    pyramid_modality_combo.bind("<<ComboboxSelected>>", _on_pyramid_modality_selected)
    ttk.Label(pyramid_control_frame, text="Input file:").pack(side=tk.LEFT, padx=(0, 4))
    pyramid_file_var = tk.StringVar(value="All files (combined)")
    pyramid_file_combo = ttk.Combobox(
        pyramid_control_frame, textvariable=pyramid_file_var,
        values=["All files (combined)"], state="readonly", width=28,
    )
    pyramid_file_combo.pack(side=tk.LEFT, padx=(0, 8))
    pyramid_file_combo.bind("<<ComboboxSelected>>", _on_pyramid_file_selected)

    # Sex breakdown + modality mix (bottom, side-by-side)
    # Sex split + modality mix alongside the age distribution so
    # reviewers asking "who was exposed?" get answers on one screen.
    cohort_extra_fig = Figure(figsize=(12, 3.6), dpi=90, constrained_layout=True)
    cohort_sex_ax = cohort_extra_fig.add_subplot(121)
    cohort_mod_ax = cohort_extra_fig.add_subplot(122)

    # Modality mix defaults to one colour per modality; this toggle
    # re-renders it as a male/female/unknown stacked breakdown on demand,
    # instead of always showing the (busier) sex-split version.
    modality_split_state = {"on": False, "ed_view": None}

    def _redraw_modality_mix():
        ed_view = modality_split_state["ed_view"]
        if ed_view is None:
            return
        plot_modality_mix(cohort_mod_ax, ed_view, split_by_sex=modality_split_state["on"])
        cohort_extra_canvas.draw_idle()

    def _toggle_modality_split():
        modality_split_state["on"] = not modality_split_state["on"]
        modality_split_btn.config(
            text="Show modality mix (all)" if modality_split_state["on"]
            else "Split modality mix by sex"
        )
        _redraw_modality_mix()

    modality_split_btn = ttk.Button(t6_bottom, text="Split modality mix by sex",
                                     command=_toggle_modality_split)
    modality_split_btn.pack(pady=(0, 2))

    cohort_extra_canvas = FigureCanvasTkAgg(cohort_extra_fig, master=t6_bottom)
    cohort_extra_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))

    t7 = ttk.Frame(nb); nb.add(t7, text="Cancer Events")
    
    # Subtabs for Cancer Events: Age Distribution, then Table
    # Age Distribution comes FIRST and is the tab that opens by default: it
    # aggregates every event of every input file / year group, so it is the
    # only view here that supports a conclusion.  The Table behind it is a
    # capped debugging sample -- see MAX_EVENT_TABLE_ROWS.
    ev_sub_nb = ttk.Notebook(t7)
    ev_sub_nb.pack(fill=tk.BOTH, expand=True)

    # -- Subtab 1: Age Distribution Histogram (all events) --
    ev_age_frame = ttk.Frame(ev_sub_nb)
    ev_sub_nb.add(ev_age_frame, text="Age Distribution")

    # Filter bar for the age pyramid: cancer type + outcome (lethality)
    ev_age_filter_bar = ttk.Frame(ev_age_frame)
    ev_age_filter_bar.pack(fill=tk.X, pady=3, padx=5)
    ttk.Label(ev_age_filter_bar, text="Cancer type:").pack(side=tk.LEFT, padx=(0, 3))
    ev_age_type_var = tk.StringVar(value="All")
    ev_age_type_cb = ttk.Combobox(ev_age_filter_bar, textvariable=ev_age_type_var, width=20, state="readonly")
    ev_age_type_cb["values"] = ["All"]
    ev_age_type_cb.pack(side=tk.LEFT, padx=(0, 10))
    ttk.Label(ev_age_filter_bar, text="Outcome:").pack(side=tk.LEFT, padx=(0, 3))
    ev_age_out_var = tk.StringVar(value="All")
    ev_age_out_cb = ttk.Combobox(ev_age_filter_bar, textvariable=ev_age_out_var, width=12, state="readonly")
    ev_age_out_cb["values"] = ["All", "Lethal", "Non-lethal"]
    ev_age_out_cb.pack(side=tk.LEFT, padx=(0, 10))
    ev_age_scope_var = tk.StringVar(value="")
    ttk.Label(ev_age_filter_bar, textvariable=ev_age_scope_var,
              foreground="#555555").pack(side=tk.LEFT, padx=(10, 0))

    ev_age_fig = Figure(figsize=(12, 5), dpi=90, constrained_layout=True)
    ev_age_ax = ev_age_fig.add_subplot(111)
    ev_age_canvas = FigureCanvasTkAgg(ev_age_fig, master=ev_age_frame)
    ev_age_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    _attach_toolbar(ev_age_canvas, ev_age_frame)

    # -- Subtab 2: Table (capped debugging sample) --
    ev_table_frame = ttk.Frame(ev_sub_nb)
    ev_sub_nb.add(ev_table_frame, text=f"Table (first {MAX_EVENT_TABLE_ROWS:,})")

    ev_warn_frame = ttk.Frame(ev_table_frame)
    ev_warn_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
    ttk.Label(
        ev_warn_frame,
        text=(f"Debugging sample \u2014 the first {MAX_EVENT_TABLE_ROWS:,} simulated "
              f"events only. Later events are counted everywhere else but are "
              f"deliberately not stored row by row."),
        foreground="#B35C00", font=("Segoe UI", 9, "bold"),
        wraplength=1400, justify=tk.LEFT,
    ).pack(anchor=tk.W)
    ttk.Label(
        ev_warn_frame,
        text=("Each row is one Monte Carlo draw, not a confirmed patient outcome. Rows are not a "
              "sample of anybody's record and carry no individual meaning: never "
              "read a row as a prediction, diagnosis or risk statement about a "
              "real person. Use the Age Distribution tab, which aggregates every "
              "event from every input file, for anything you intend to conclude."),
        foreground="#444444", wraplength=1400, justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(1, 3))

    ev_filter_bar = ttk.Frame(ev_table_frame)
    ev_filter_bar.pack(fill=tk.X, pady=3, padx=5)
    ttk.Label(ev_filter_bar, text="Modality:").pack(side=tk.LEFT, padx=(0, 3))
    ev_mod_var = tk.StringVar(value="All")
    ev_mod_cb = ttk.Combobox(ev_filter_bar, textvariable=ev_mod_var, width=15, state="readonly")
    ev_mod_cb["values"] = ["All"]
    ev_mod_cb.pack(side=tk.LEFT, padx=(0, 10))
    ttk.Label(ev_filter_bar, text="Outcome:").pack(side=tk.LEFT, padx=(0, 3))
    ev_out_var = tk.StringVar(value="All")
    ev_out_cb = ttk.Combobox(ev_filter_bar, textvariable=ev_out_var, width=12, state="readonly")
    ev_out_cb["values"] = ["All", "Death", "Survived"]
    ev_out_cb.pack(side=tk.LEFT, padx=(0, 10))
    ttk.Label(ev_filter_bar, text="Rows stored:").pack(side=tk.LEFT, padx=(10, 3))
    ev_live_count_var = tk.StringVar(value="0")
    ttk.Label(ev_filter_bar, textvariable=ev_live_count_var).pack(side=tk.LEFT)
    ev_tree = ttk.Treeview(ev_table_frame, columns=(
        "cancer", "age_exp", "age_death", "life_lost", "sex", "mod", "region", "proto", "ed", "outcome",
    ), show="headings")
    for c, h, w in [
        ("cancer", "Cancer", 120), ("age_exp", "Age at exposure", 95),
        ("age_death", "Age at death", 85), ("life_lost", "Life lost (y)", 85), ("sex", "Sex", 40),
        ("mod", "Modality", 60), ("region", "Region", 100), ("proto", "Protocol", 150),
        ("ed", "ED mSv", 70), ("outcome", "Outcome", 80),
    ]:
        ev_tree.heading(c, text=h)
        ev_tree.column(c, width=w, anchor=tk.E if c in ("age_exp", "age_death", "life_lost", "ed") else tk.W)
    ev_sb = ttk.Scrollbar(ev_table_frame, orient=tk.VERTICAL, command=ev_tree.yview)
    ev_tree.configure(yscrollcommand=ev_sb.set)
    ev_sb.pack(side=tk.RIGHT, fill=tk.Y)
    ev_tree.pack(fill=tk.BOTH, expand=True)
    _bind_header_tooltips(ev_tree, _EV_TREE_TOOLTIPS, _gui_tooltip)

    # Capped debugging sample of events, for the Table subtab only.  Every
    # other view (Age Distribution included) reads the full stream in
    # ``results["events"]``, so capping this costs no analysis.
    all_events_store = _EventColumnsForEmpty.empty()
    ev_events_seen = 0        # visible events streamed so far, capped or not
    visible_cancer_vars = {}
    default_hidden_cancers = set()
    ev_age_histogram_drawn = False  # Flag to track if histogram has been drawn

    t8 = ttk.Frame(nb); nb.add(t8, text="Risk Analysis")
    advanced_panel = AdvancedRiskAnalysisPanel(t8, n_iterations)
    advanced_panel.frame.pack(fill=tk.BOTH, expand=True)

    # Sankey / flow diagram as a sub-tab of "By Cancer Type".
    t11 = ttk.Frame(cancer_sub_nb)
    cancer_sub_nb.add(t11, text="Flow diagram")
    sankey_fig = Figure(figsize=(14, 7), dpi=90)
    sankey_canvas = FigureCanvasTkAgg(sankey_fig, master=t11)
    sankey_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Sankey hover tooltip: show the full box label when the cursor is over a region.
    def _on_sankey_hover(event):
        """On mouse move, hit-test the Sankey boxes and update the figure's
        info text with the hovered box's value."""
        try:
            fig = sankey_fig
            ax = getattr(fig, "_sankey_ax", None)
            hits = getattr(fig, "_sankey_box_hits", None) or []
            if ax is None or event.inaxes is not ax or not hits:
                return
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            tooltip = getattr(fig, "_sankey_tooltip", None)
            for (bx, by, bw, bh, label) in hits:
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    if tooltip is None:
                        tooltip = ax.annotate(
                            "", xy=(x, y), xytext=(12, 12),
                            textcoords="offset pixels",
                            bbox=dict(boxstyle="round,pad=0.4",
                                      facecolor="#FFFFE0",
                                      edgecolor="#555555", alpha=0.95),
                            fontsize=8, ha="left", va="bottom",
                            visible=False, zorder=100,
                        )
                        fig._sankey_tooltip = tooltip
                    tooltip.xy = (x, y)
                    tooltip.set_text(label)
                    tooltip.set_visible(True)
                    sankey_canvas.draw_idle()
                    return
            if tooltip is not None and tooltip.get_visible():
                tooltip.set_visible(False)
                sankey_canvas.draw_idle()
        except Exception:
            pass  # never let a stray motion event crash the GUI

    sankey_canvas.mpl_connect("motion_notify_event", _on_sankey_hover)

    # Research Questions tab
    # Single sub-tab: dose-reduction benefit per imaging region × 10-year
    # age group (cancers / fatal cancers / QALYs avoided per N examinations
    # under an LNT projection of the simulation's per-imaging risk).
    tRQ = ttk.Frame(nb); nb.add(tRQ, text="Research Questions")
    rq_nb = ttk.Notebook(tRQ)
    rq_nb.pack(fill=tk.BOTH, expand=True)

    # — Region x Age dose-reduction sub-tab ----------------------------
    # Answers the planning-meeting question:
    #   "How many cancers / deaths / QALYs are avoided per 100 000
    #    examinations in each anatomical region (heart, chest, head,
    #    extremities, ...) when dose is reduced in 10% increments?"
    # Controls:
    #   * Per-N normaliser- 1 000 / 10 000 / 100 000 examinations
    #   * Region          - "All (small multiples)" or one region focused
    #   * Highlight level - which reduction % the text summary ranks by
    rq_frame = ttk.Frame(rq_nb)
    rq_nb.add(rq_frame, text="Region x Age dose reduction")
    rq_ctrl_bar = ttk.Frame(rq_frame)
    rq_ctrl_bar.pack(fill=tk.X, padx=5, pady=(5, 2))

    ttk.Label(rq_ctrl_bar, text="     Per:").pack(side=tk.LEFT, padx=(8, 4))
    rq_pern_var = tk.StringVar(value="100,000")
    rq_pern_cb = ttk.Combobox(
        rq_ctrl_bar, textvariable=rq_pern_var,
        values=["1,000", "10,000", "100,000", "1,000,000"],
        width=10, state="readonly")
    rq_pern_cb.pack(side=tk.LEFT)
    ttk.Label(rq_ctrl_bar, text="examinations").pack(side=tk.LEFT, padx=(2, 0))

    # Heatmap unit toggle -- only changes the Region x Age heatmap's unit
    # (QALYs per N exams vs QALY hours per exam), not the sweep itself, so
    # the redraw reuses the cached sweep matrix.
    ttk.Label(rq_ctrl_bar, text="     Show:").pack(side=tk.LEFT, padx=(8, 4))
    rq_value_mode_var = tk.StringVar(value="per_n")
    rq_value_mode_box = ttk.Frame(rq_ctrl_bar)
    rq_value_mode_box.pack(side=tk.LEFT)
    rq_value_mode_rb_n = ttk.Radiobutton(rq_value_mode_box, text="QALYs per N exams", variable=rq_value_mode_var, value="per_n")
    rq_value_mode_rb_h = ttk.Radiobutton(rq_value_mode_box, text="QALY hours per exam", variable=rq_value_mode_var, value="hours_per_exam")
    rq_value_mode_rb_n.pack(anchor=tk.W); rq_value_mode_rb_h.pack(anchor=tk.W)

    # Blank-below-floor toggle -- like the unit radios it only changes how
    # the Region x Age heatmap draws the cached sweep, not the sweep
    # itself, so the redraw is cheap.
    rq_hide_small_var = tk.BooleanVar(value=True)
    rq_hide_small_cb = ttk.Checkbutton(
        rq_ctrl_bar, variable=rq_hide_small_var,
        text="Blank cells with <")
    rq_hide_small_cb.pack(side=tk.LEFT, padx=(10, 0))
    rq_min_scans_var = tk.StringVar(value=f"{DEFAULT_MIN_SCANS:g}")
    rq_min_scans_entry = ttk.Entry(
        rq_ctrl_bar, textvariable=rq_min_scans_var,
        width=5, justify=tk.RIGHT)
    rq_min_scans_entry.pack(side=tk.LEFT, padx=(2, 0))
    ttk.Label(rq_ctrl_bar, text="exams").pack(side=tk.LEFT, padx=(2, 0))
    try:
        _rq_hide_tip = _ToolTip(rq_hide_small_cb)
        _rq_hide_tip_txt = (
            "Region x age cells with fewer than this many examinations "
            "in total (all year groups pooled) are marked * in the text "
            "summary and \u2014 when ticked \u2014 "
            "shown as grey n/a in the heatmap.  Untick to draw their "
            "values anyway.  Type a number and press Enter.")
        for _w in (rq_hide_small_cb, rq_min_scans_entry):
            _w.bind(
                "<Enter>",
                lambda e: _rq_hide_tip.show(_rq_hide_tip_txt, e.x_root, e.y_root),
                add="+")
            _w.bind(
                "<Leave>", lambda e: _rq_hide_tip.hide(), add="+")
    except Exception:
        pass

    # Modality filter for the sweep; populated lazily after simulation.
    # Multi-select Listbox so the user can combine modalities (e.g. CT + MG).
    ttk.Label(rq_ctrl_bar, text="     Modality:").pack(
        side=tk.LEFT, padx=(8, 4))
    _RQ_MOD_SENTINEL = "All modalities"
    rq_modality_frame = ttk.Frame(rq_ctrl_bar)
    rq_modality_frame.pack(side=tk.LEFT)
    rq_modality_lb = tk.Listbox(
        rq_modality_frame, selectmode=tk.MULTIPLE, height=4,
        exportselection=False, width=24)
    rq_modality_lb.pack(side=tk.LEFT, fill=tk.Y)
    rq_modality_sb = ttk.Scrollbar(
        rq_modality_frame, orient=tk.VERTICAL, command=rq_modality_lb.yview)
    rq_modality_lb.configure(yscrollcommand=rq_modality_sb.set)
    rq_modality_sb.pack(side=tk.LEFT, fill=tk.Y)
    # Insert the "All modalities" sentinel as the first entry, selected by default.
    rq_modality_lb.insert(tk.END, _RQ_MOD_SENTINEL)
    rq_modality_lb.selection_set(0)
    # Tracks the selection (by label) as of the end of the previous
    # refresh, so the exclusivity logic below can tell which entry the
    # user just clicked rather than only seeing the resulting set.
    _rq_mod_prior_labels = {_RQ_MOD_SENTINEL}

    ttk.Label(rq_ctrl_bar, text="     Region:").pack(
        side=tk.LEFT, padx=(8, 4))
    rq_region_frame = ttk.Frame(rq_ctrl_bar)
    rq_region_frame.pack(side=tk.LEFT)
    rq_region_lb = tk.Listbox(
        rq_region_frame, selectmode=tk.MULTIPLE, height=4,
        exportselection=False, width=22)
    rq_region_lb.pack(side=tk.LEFT, fill=tk.Y)
    rq_region_scroll = ttk.Scrollbar(
        rq_region_frame, orient=tk.VERTICAL,
        command=rq_region_lb.yview)
    rq_region_scroll.pack(side=tk.LEFT, fill=tk.Y)
    rq_region_lb.configure(yscrollcommand=rq_region_scroll.set)

    # (The "Highlight @ %" dropdown sits beside the "Annual effect of dose change"
    # button, where a reduction percentage has an effect.  It is
    # deliberately not offered on this tab: neither
    # format_region_age_sweep_summary nor plot_region_age_sweep takes a
    # dose-change level, so the control would change nothing here.)
    #
    # (The Research Questions tab ALWAYS uses the analytical ICRP-152
    # baseline, whose cells are never flagged for refinement, so this tab
    # offers no deterministic-baseline checkbox.)

    # "Annual effect of dose change" and the dose change it assumes, stacked in one
    # box so the control is visibly attached to the button it
    # parameterises -- it applies to that figure only, not to the rest of
    # the tab, which a lone dropdown on the control bar would imply.
    rq_annual_box = ttk.Frame(rq_ctrl_bar)
    rq_annual_box.pack(side=tk.RIGHT, padx=(8, 0))
    rq_annual_btn = ttk.Button(
        rq_annual_box, text="Annual effect of dose change")
    rq_annual_btn.pack(side=tk.TOP)
    rq_annual_red_row = ttk.Frame(rq_annual_box)
    rq_annual_red_row.pack(side=tk.TOP, pady=(3, 0))
    ttk.Label(rq_annual_red_row, text="Dose change:").pack(side=tk.LEFT)
    # A free-text signed percentage, not a preset list: any value is priced
    # exactly.  compute_annual_benefit falls back to each cell's baseline
    # and stored leukaemia-quadratic share when the value is not one of the
    # levels the sweep evaluated, so -17 % is as valid as -10 %.
    #
    # Signed, because a positive value is meaningful: it models a dose
    # INCREASE and shows the harm added rather than the benefit gained.
    # Default -10 % (a reduction) is the common case.
    rq_annual_red_var = tk.StringVar(value="-10")
    rq_annual_red_entry = ttk.Entry(
        rq_annual_red_row, textvariable=rq_annual_red_var,
        width=6, justify=tk.RIGHT)
    rq_annual_red_entry.pack(side=tk.LEFT, padx=(3, 0))
    ttk.Label(rq_annual_red_row, text="%").pack(side=tk.LEFT, padx=(2, 0))
    try:
        _rq_red_tip = _ToolTip(rq_annual_red_entry)
        _rq_red_tip_txt = (
            "Signed dose change applied to the annual-benefit figure.\n"
            "Negative reduces dose (-10 = a 10 % cut, the usual case); "
            "positive increases it (+17) to show the harm that adds.\n"
            f"Any value from {_DOSE_CHANGE_MIN:g} to {_DOSE_CHANGE_MAX:g} % "
            "is priced exactly, not only the simulated levels.")
        rq_annual_red_entry.bind(
            "<Enter>",
            lambda e: _rq_red_tip.show(_rq_red_tip_txt, e.x_root, e.y_root),
            add="+")
        rq_annual_red_entry.bind(
            "<Leave>", lambda e: _rq_red_tip.hide(), add="+")
    except Exception:
        pass

    rq_burden_btn = ttk.Button(
        rq_ctrl_bar, text="Annual burden", width=14)
    rq_burden_btn.pack(side=tk.RIGHT, padx=(8, 0))

    ttk.Label(rq_frame,
              text=("Linear-no-threshold projection of the simulation's "
                    "per-examination risk.  Baseline (0%) shown for reference; "
                    "negative dose change -> positive avoided counts."),
              font=("Segoe UI", 8), foreground="#555").pack(
        fill=tk.X, padx=6, pady=(2, 0))

    rq_text = scrolledtext.ScrolledText(
        rq_frame, font=("Consolas", 9), wrap=tk.NONE, height=14)
    rq_text.pack(fill=tk.X, padx=4, pady=(2, 2))

    # The heatmap's cell size is fixed (see plot_region_age_sweep), so
    # its total height grows with however many modalities/regions are
    # selected rather than being squeezed to fit a fixed frame -- a
    # scrollable container lets the whole figure be viewed (by
    # scrolling) instead of distorting cell dimensions to fit at
    # once.  Plain pack() (no fill/expand) on the canvas widget keeps
    # it at its natural size instead of matplotlib's default
    # auto-resize-on-configure behaviour, which would otherwise
    # stretch the figure to fill the frame.
    rq_plot_outer = ttk.Frame(rq_frame)
    rq_plot_outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
    rq_scrollbar = ttk.Scrollbar(rq_plot_outer, orient=tk.VERTICAL)
    rq_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    rq_scroll_canvas = tk.Canvas(
        rq_plot_outer, highlightthickness=0,
        yscrollcommand=rq_scrollbar.set)
    rq_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    rq_scrollbar.config(command=rq_scroll_canvas.yview)

    rq_fig = Figure(figsize=(10, 15.4), dpi=100, constrained_layout=True)
    rq_canvas = FigureCanvasTkAgg(rq_fig, master=rq_scroll_canvas)
    rq_canvas_widget = rq_canvas.get_tk_widget()
    rq_canvas_window = rq_scroll_canvas.create_window(
        (0, 0), window=rq_canvas_widget, anchor="nw")

    def _rq_sync_scroll_size() -> None:
        """Resize the scroll canvas's window/scrollregion to match the
        figure's actual (possibly grown) pixel dimensions."""
        w_px = int(round(rq_fig.get_figwidth() * rq_fig.dpi))
        h_px = int(round(rq_fig.get_figheight() * rq_fig.dpi))
        rq_canvas_widget.config(width=w_px, height=h_px)
        rq_scroll_canvas.itemconfig(rq_canvas_window, width=w_px, height=h_px)
        rq_scroll_canvas.configure(scrollregion=(0, 0, w_px, h_px))

    def _rq_on_mousewheel(event) -> None:
        rq_scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    rq_scroll_canvas.bind("<Enter>", lambda _e: rq_scroll_canvas.bind_all(
        "<MouseWheel>", _rq_on_mousewheel))
    rq_scroll_canvas.bind("<Leave>", lambda _e: rq_scroll_canvas.unbind_all(
        "<MouseWheel>"))

    _rq_toolbar = _attach_toolbar(rq_canvas, rq_frame)
    # Override save for publication-quality output: 600 dpi, tight bbox,
    # white background, minimal padding.
    if _rq_toolbar is not None:
        _rq_orig_save = _rq_toolbar.save_figure
        def _rq_save_figure(*args, **kwargs):
            # NB: ``rq_canvas``, not ``_rq_canvas`` -- the latter was never
            # defined, so every Save from this toolbar raised NameError.
            _orig_dpi = rq_canvas.figure.dpi
            rq_canvas.figure.dpi = 600
            try:
                _rq_orig_save(*args, **kwargs)
            finally:
                rq_canvas.figure.dpi = _orig_dpi
        _rq_toolbar.save_figure = _rq_save_figure
    # Cached sweep shared by the metric/region/target callbacks to avoid recomputation.
    _rq_cache: dict = {
        "sweep": None,
        "sweep_key": None,
        "last_modality": "All",  # tracks previous combobox selection for save/restore
    }

    # — Burden vs collective dose sub-tab ------------------------------
    # Per-modality small multiples of the burden-vs-dose curve and a
    # by-modality text companion, both built from the cached Region x Age
    # sweep (_rq_cache["sweep"]).
    #
    # This view reads the Region x Age sweep (_rq_cache["sweep"]), so it is
    # refreshed after that sweep is (re)built in _upd_research_questions.
    # If the sweep is not yet available the tab shows a "open Region x Age
    # first" message rather than computing one implicitly.
    db_frame = ttk.Frame(rq_nb)
    rq_nb.add(db_frame, text="Burden vs collective dose")

    ttk.Label(db_frame,
              text=("Per-modality burden of the current imaging practice "
                    "against collective effective dose (person-Sv). Built "
                    "from the Region \u00d7 Age sweep: open that sub-tab "
                    "first if this is empty."),
              font=("Segoe UI", 8), foreground="#555", wraplength=900,
              justify=tk.LEFT).pack(fill=tk.X, padx=6, pady=(2, 0))

    # Optional manual extrapolation target (collective dose, person-Sv).
    db_extrap_bar = ttk.Frame(db_frame)
    db_extrap_bar.pack(fill=tk.X, padx=6, pady=(2, 0))
    ttk.Label(db_extrap_bar,
              text="Extrapolate current imaging practice to collective dose:"
              ).pack(side=tk.LEFT)
    db_extrap_var = tk.StringVar(value="")
    db_extrap_entry = ttk.Entry(db_extrap_bar, textvariable=db_extrap_var, width=10)
    db_extrap_entry.pack(side=tk.LEFT, padx=(6, 2))
    ttk.Label(db_extrap_bar, text="person-Sv").pack(side=tk.LEFT)
    db_extrap_apply_btn = ttk.Button(db_extrap_bar, text="Apply", width=8)
    db_extrap_apply_btn.pack(side=tk.LEFT, padx=(8, 0))
    ttk.Label(db_extrap_bar,
              text="   (blank = current practice; press Apply or Enter)",
              font=("Segoe UI", 8), foreground="#555").pack(side=tk.LEFT)

    db_text = scrolledtext.ScrolledText(
        db_frame, font=("Consolas", 9), wrap=tk.NONE, height=13)
    db_text.pack(fill=tk.X, padx=4, pady=(2, 2))

    # Same scrollable container as the heatmap above: the figure keeps its
    # own aspect ratio (a FigureCanvasTkAgg packed fill=BOTH would resize the
    # *figure* to the widget and squash the three panels).
    db_plot_outer = ttk.Frame(db_frame)
    db_plot_outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
    db_scrollbar = ttk.Scrollbar(db_plot_outer, orient=tk.VERTICAL)
    db_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    # Horizontal scrollbar: the small-multiples figure is three panels wide,
    # so it can exceed the visible pane width.  Without this the right-hand
    # panel (and the shared legend) would be unreachable, since the scroll
    # region already spans the full figure width.
    db_hscrollbar = ttk.Scrollbar(db_plot_outer, orient=tk.HORIZONTAL)
    db_hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    db_scroll_canvas = tk.Canvas(
        db_plot_outer, highlightthickness=0,
        yscrollcommand=db_scrollbar.set,
        xscrollcommand=db_hscrollbar.set)
    db_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    db_scrollbar.config(command=db_scroll_canvas.yview)
    db_hscrollbar.config(command=db_scroll_canvas.xview)

    # The per-modality small-multiples figure lives inside an inner frame
    # held by the scroll canvas, so one scroll region covers it.
    db_plot_inner = ttk.Frame(db_scroll_canvas)
    db_canvas_window = db_scroll_canvas.create_window(
        (0, 0), window=db_plot_inner, anchor="nw")

    # Per-modality burden figure: three side-by-side panels (QALY burden,
    # cancer burden, fatal cancer burden) sharing one collective-dose
    # x-axis, with per-modality lines and a shared external legend.  Wide
    # enough that each of the three panels keeps its own y-tick labels and
    # y-label; the function solves its own margins from rendered pixel
    # extents, so it re-renders correctly at the 600-dpi export size.
    ttk.Label(
        db_plot_inner,
        text=("Per-modality burden \u2014 QALY, cancer and fatal-cancer "
              "panels; each panel's y-axis is auto-scaled."),
        font=("Segoe UI", 8), foreground="#555", justify=tk.LEFT
    ).pack(side=tk.TOP, fill=tk.X, pady=(2, 2))
    db_sm_fig = Figure(figsize=(18.0, 6.5), dpi=100)
    db_sm_canvas = FigureCanvasTkAgg(db_sm_fig, master=db_plot_inner)
    db_sm_canvas_widget = db_sm_canvas.get_tk_widget()
    db_sm_canvas_widget.pack(side=tk.TOP, fill=tk.X, padx=(0, 0), pady=(0, 0))

    def _db_sync_scroll_size() -> None:
        # The scroll region must cover the figure plus its label; let Tk
        # compute the packed height via update_idletasks, then read the
        # inner frame's requested size.
        db_plot_inner.update_idletasks()
        w_px = int(round(db_sm_fig.get_figwidth() * db_sm_fig.dpi))
        h_px = int(round(db_plot_inner.winfo_reqheight()))
        db_scroll_canvas.itemconfig(db_canvas_window, width=w_px)
        db_scroll_canvas.configure(scrollregion=(0, 0, w_px, h_px))

    def _db_on_mousewheel(event) -> None:
        db_scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    db_scroll_canvas.bind("<Enter>", lambda _e: db_scroll_canvas.bind_all(
        "<MouseWheel>", _db_on_mousewheel))
    db_scroll_canvas.bind("<Leave>", lambda _e: db_scroll_canvas.unbind_all(
        "<MouseWheel>"))

    # A toolbar on the small-multiples canvas so it can be zoomed,
    # panned and saved independently.  Same 600-dpi export override.
    _db_sm_toolbar = _attach_toolbar(db_sm_canvas, db_frame)
    if _db_sm_toolbar is not None:
        _db_sm_orig_save = _db_sm_toolbar.save_figure
        def _db_sm_save_figure(*args, **kwargs):
            _orig_dpi = db_sm_canvas.figure.dpi
            db_sm_canvas.figure.dpi = 600
            try:
                _db_sm_orig_save(*args, **kwargs)
            finally:
                db_sm_canvas.figure.dpi = _orig_dpi
        _db_sm_toolbar.save_figure = _db_sm_save_figure

    # ``drawing`` guards against reentrant redraws.  The view reads the
    # Region x Age sweep directly (no per-modality curves cache, no
    # modality selection, no per-N normaliser), so the cache only needs the
    # guard flag -- the sweep itself lives in _rq_cache["sweep"].
    _db_cache: dict = {"drawing": False}

    # Final tab order
    # Arrange the tabs in user-facing order.
    # ``Notebook.insert(pos, existing_widget)`` moves the tab, so this is cheap and idempotent.
    _tab_order = [t1, t2, t5, t3, t7, t6, tRQ, t8]
    assert len(_tab_order) == len(tab_names), (
        "tab_names and _tab_order must describe the same tabs"
    )
    for _pos, _tw in enumerate(_tab_order):
        nb.insert(_pos, _tw)

    # Attach matplotlib toolbars to all charts except the borderless Sankey canvas.
    _attach_toolbar(summary_iter_canvas, summary_cancers_tab)
    _attach_toolbar(summary_QALY_canvas, summary_QALY_tab)
    _attach_toolbar(mod_canvas, t2)
    _attach_toolbar(reg_canvas, reg_stats_frame)
    # The pyramid canvas lives inside chart_box, not directly in t6, so the
    # toolbar goes into its own frame above it (see _attach_toolbar).
    _attach_toolbar(chart_canvas, chart_tb_holder, use_before=False)
    # Separate toolbar for the modality-mix chart so it can be zoomed/saved independently.
    # Must be t6_bottom, not t6: _attach_toolbar packs with before=<canvas>,
    # which requires the toolbar and the canvas to be pack siblings.
    _attach_toolbar(cohort_extra_canvas, t6_bottom)

    results = {"ready": False, "stats": None, "exam_data": None,
               "events": _EventColumnsForEmpty.empty(), "cases_l": [], "deaths_l": [],
               "file_cohort": None, "excluded_cohorts": None, "missing_columns": {}}
    # Set by the "Don't show this again for this simulation" checkbox in
    # _show_missing_ct_dialog; once True, later missing_ct_doses events in
    # this run (e.g. a batch moving on to the next input file) auto-continue
    # instead of popping the dialog up again.
    _missing_ct_suppress_state = {"suppress": False}
    ui_q = queue.Queue()
    modality_filter_map = {"All modalities": None}
    device_filter_map = {"All devices": None}
    year_filter_map = {"All years": None}
    region_modality_filter_map = {"All modalities": None}
    summary_modality_filter_map = {"All modalities": None}

    advanced_panel.set_results_getter(lambda: results)
    advanced_panel.set_visible_cancers_getter(lambda: _visible_cancers_set())

    def _summary_per100k_scale(ed, s):
        """Scale factor for the Summary tab's "per 100,000 examinations"
        toggle, or ``None`` when it is off / the cohort size is unknown.

        ``ed``/``s`` must already be modality-filtered (i.e. the same
        values passed to ``format_summary``) so the projection is against
        that filtered examination count, not the whole cohort.
        """
        if not summary_per100k_var.get():
            return None
        n = resolve_modelled_examinations(s, s.get("iterations", 0) or 0, resolve_cohort_examinations(s, ed))
        return (100_000.0 / n) if n > 0 else None

    def _on_bayesian_summary_refresh(bayes_curve, ddref_solid, ddref_leuk):
        """Called by AdvancedRiskAnalysisPanel after Bayesian recalculation.
        Updates the Summary tab cancer pie chart and text to reflect the new DDREF."""
        s = results.get("stats")
        if not s:
            return
        cl = results.get("cases_l") or []
        dl = results.get("deaths_l") or []
        ev = results.get("events") or []
        ed = results.get("exam_data") or []
        # Honour the Summary-tab modality dropdown so the DDREF sensitivity
        # view matches the rest of the tab.
        mod_code = summary_modality_filter_map.get(summary_mod_view_var.get())
        if mod_code is not None:
            s, ev, cl, dl, ed = _compute_modality_filtered_results(mod_code, s, ev, ed)
        bg_rad = (s.get("sim_config") or {}).get("background_radiation_msv_per_year", BACKGROUND_MSV_PER_YEAR)
        _per100k_scale = _summary_per100k_scale(ed, s)
        base_summary = format_summary(
            s,
            visible_cancers=_visible_cancers_set(),
            background_msv_per_year=bg_rad,
            exam_data=ed,
            cancer_events=ev,
            iter_cases_list=cl,
            iter_deaths_list=dl,
            event_columns=ev,
            per_100k_examinations=(_per100k_scale is not None),
        )
        from .risk_models import DDREF_SOLID_CANCERS, DDREF_LEUKAEMIA
        base_summary += "\n"
        if ddref_solid != DDREF_SOLID_CANCERS or ddref_leuk != DDREF_LEUKAEMIA:
            base_summary += (
                f"--- Bayesian sensitivity active: "
                f"DDREF_solid={ddref_solid:.2f}, DDREF_leuk={ddref_leuk:.2f} ---\n"
                f"    (Default: DDREF_solid={DDREF_SOLID_CANCERS:.2f}, "
                f"DDREF_leuk={DDREF_LEUKAEMIA:.2f}; "
                f"pie chart rescaled below)\n"
            )
            ddref_solid_ratio = DDREF_SOLID_CANCERS / max(0.01, ddref_solid)
            ddref_leuk_ratio = DDREF_LEUKAEMIA / max(0.01, ddref_leuk)
            adjusted_bd = {}
            for ct, cv in s.get("cancer_breakdown", {}).items():
                is_leuk = ct.lower() in ("leukaemia", "leukemia")
                ratio = ddref_leuk_ratio if is_leuk else ddref_solid_ratio
                adjusted_bd[ct] = {
                    "developed": cv.get("developed", 0) * ratio,
                    "deaths": cv.get("deaths", 0) * ratio,
                }
            filtered_bd = filter_cancer_breakdown(adjusted_bd, _visible_cancers_set())
        else:
            filtered_bd = filter_cancer_breakdown(s.get("cancer_breakdown", {}), _visible_cancers_set())
        summary_text.delete("1.0", tk.END)
        summary_text.insert(tk.END, base_summary)
        _upd_summary_iter(cl, dl, filtered_bd, is_final=True, events=ev, stats=s,
                          scale=_per100k_scale)

    advanced_panel.set_summary_refresh_callback(_on_bayesian_summary_refresh)

    def _visible_cancers_set():
        selected = {ct for ct, var in visible_cancer_vars.items() if var.get()}
        return selected if selected else {ct for ct in visible_cancer_vars.keys() if ct not in default_hidden_cancers}

    def _filter_visible_events(evts):
        """Return ``evts`` restricted to the currently visible cancer types
        (all events when nothing is hidden); handles both EventColumns
        and plain event dicts."""
        visible = _visible_cancers_set()
        if isinstance(evts, EventColumns):
            if not visible:
                return evts
            cancer_types = evts.cancer_type
            mask = np.array([ct in visible for ct in cancer_types], dtype=bool)
            if not mask.any():
                return _EventColumnsForEmpty.empty()
            idx = np.nonzero(mask)[0]
            # Build a proper EventColumns subset so downstream code can rely on
            # the columnar attributes instead of dict lookups.
            return EventColumns(
                cancer_type=evts.cancer_type[idx],
                modality=evts.modality[idx],
                region=evts.region[idx],
                protocol=evts.protocol[idx],
                age_at_exposure=evts.age_at_exposure[idx],
                sex=evts.sex[idx],
                died=evts.died[idx],
                exam_index=evts.exam_index[idx],
                iteration=evts.iteration[idx],
                ed_msv=evts.ed_msv[idx],
                QALY_lost=evts.QALY_lost[idx],
                life_lost_years=evts.life_lost_years[idx],
                age_at_death=evts.age_at_death[idx],
                projected_lifetime=evts.projected_lifetime[idx],
            )
        else:
            return [ev for ev in evts if ev.get("cancer_type", "") in visible]

    def _rebuild_cancer_visibility_controls(cancer_bd):
        existing = set(visible_cancer_vars.keys())
        incoming = list(sorted((cancer_bd or {}).keys()))
        if set(incoming) == existing:
            return
        for child in summary_cancer_filter_bar.winfo_children()[1:]:
            child.destroy()
        visible_cancer_vars.clear()
        for ct in incoming:
            var = tk.BooleanVar(value=(ct not in default_hidden_cancers))
            visible_cancer_vars[ct] = var
            cb = ttk.Checkbutton(summary_cancer_filter_bar, text=ct, variable=var,
                                 command=lambda: _refresh_visible_cancer_views())
            cb.pack(side=tk.LEFT, padx=(0, 6))

    def _visible_iter_case_counts(all_events, total_iterations):
        """Per-iteration cancer counts limited to visible cancer types;
        fast path uses the precomputed counts when nothing is hidden."""
        visible = _visible_cancers_set()
        per_iter = [0] * max(0, total_iterations)
        if not visible or total_iterations <= 0:
            return per_iter
        if not all_events:
            return per_iter
        # Fast-path: when no cancers are hidden (default case), skip the expensive
        # iteration over all events and use the precomputed iteration counts directly.
        # The worker thread already provides cases_l which is the per-iteration count.
        # This avoids O(events) work on the GUI thread during live updates.
        all_cancer_types = set(visible_cancer_vars.keys())
        if visible == all_cancer_types - default_hidden_cancers:
            # No cancers hidden - return empty list to signal use of precomputed counts
            return []
            
        if not isinstance(all_events, EventColumns):
            all_events = EventColumns.from_events(list(all_events))
        cancer_types = all_events.cancer_type
        iterations = all_events.iteration
        for i in range(len(all_events)):
            if cancer_types[i] in visible:
                idx = iterations[i]
                if 0 <= idx < total_iterations:
                    per_iter[idx] += 1
        return per_iter

    def _rebuild_summary_modality_filter_values(s):
        """Populate the Summary-tab modality dropdown from modality_stats."""
        summary_modality_filter_map.clear()
        summary_modality_filter_map["All modalities"] = None
        for mod in sorted((s or {}).get("modality_stats", {}).keys()):
            summary_modality_filter_map[modality_name(mod)] = mod
        current = summary_mod_view_var.get()
        values = list(summary_modality_filter_map.keys())
        summary_mod_view_cb["values"] = values
        if current not in summary_modality_filter_map:
            summary_mod_view_var.set("All modalities")


    def _draw_summary_flow():
        """Redraw the selection-of-examinations diagram from the UNFILTERED
        stats (the modality dropdown does not apply to the flow ledger).
        The renderer grows the figure height to fit the ledger at a fixed
        font; the Tk widget and scroll region are then resized to match."""
        from .participant_flow import plot_participant_flow
        s = results.get("stats") or {}
        flow = s.get("_participant_flow")
        yb = s.get("_year_batches") or {}
        period = None
        if yb:
            labels = sorted(str(k) for k in yb)
            period = labels[0] if len(labels) == 1 else f"{labels[0]}\u2013{labels[-1]}"
        n_years = s.get("_n_years") or (flow or {}).get("n_years")
        plot_participant_flow(summary_flow_ax, flow, n_years=n_years, period=period,
                              fit="grow")
        _flow_sync_scrollregion()
        summary_flow_canvas.draw_idle()

    def _draw_summary_adjustments():
        """Fill the Adjustments view from the UNFILTERED stats (the
        modality dropdown does not apply to the adjustments)."""
        s = results.get("stats") or {}
        is_final = bool(results.get("ready"))
        summary_adj_text.delete("1.0", tk.END)
        summary_adj_text.insert(tk.END, format_adjustments(
            s,
            exam_columns=s.get("_exam_columns") if is_final else None,
            exam_data=(results.get("exam_data") or None) if is_final else None))

    def _render_summary():
        """Render the Summary tab text + charts, honouring the modality
        dropdown, the per-100,000-examinations toggle and the
        cancer-visibility checkboxes."""
        s = results.get("stats") or {}
        if not s:
            return
        _draw_summary_flow()
        _draw_summary_adjustments()
        ev = results.get("events") or []
        cl = results.get("cases_l") or []
        dl = results.get("deaths_l") or []
        ed = results.get("exam_data") or []
        mod_code = summary_modality_filter_map.get(summary_mod_view_var.get())
        mod_active = mod_code is not None
        if mod_active:
            s, ev, cl, dl, ed = _compute_modality_filtered_results(mod_code, s, ev, ed)
        visible_cancers = _visible_cancers_set()
        filtered_bd = filter_cancer_breakdown(s.get("cancer_breakdown", {}), visible_cancers)
        visible_cl = _visible_iter_case_counts(ev, len(cl)) if ev and cl else cl
        if cl and visible_cl == []:
            visible_cl = cl
        bg_rad = (s.get("sim_config") or {}).get("background_radiation_msv_per_year", BACKGROUND_MSV_PER_YEAR)
        is_final = bool(results.get("ready"))
        # exam_data is only handed to format_summary once the run is final
        # (see the comment below); the per-100k scale must be resolved from
        # the same (gated) exam_data so the chart projection always matches
        # the cohort size implied by the text block.
        _per100k_scale = _summary_per100k_scale(ed if is_final else None, s)
        summary_text.delete("1.0", tk.END)
        if is_final:
            _batch_note = _format_batch_note(s)
            if _batch_note:
                summary_text.insert(tk.END, _batch_note + "\n\n")
        # When a modality filter is active, drop the cached global exam_columns
        # so format_summary rebuilds them from the filtered exam_data.
        _exam_cols = None if mod_active else s.get("_exam_columns")
        summary_text.insert(tk.END, format_summary(
            s,
            visible_cancers=visible_cancers,
            background_msv_per_year=bg_rad,
            exam_data=ed if is_final else None,
            cancer_events=ev if is_final else None,
            iter_cases_list=cl,
            iter_deaths_list=dl,
            exam_columns=(_exam_cols if is_final else None),
            event_columns=ev if is_final else None,
            per_100k_examinations=(_per100k_scale is not None),
        ))
        _upd_summary_iter(visible_cl, dl, filtered_bd, is_final=is_final, events=ev, stats=s,
                          scale=_per100k_scale)


    def _rebuild_modality_filter_values(s):
        """Repopulate the modality filter dropdown from ``s``'s
        ``modality_stats`` keys, preserving the current selection when it
        is still valid."""
        modality_filter_map.clear()
        modality_filter_map["All modalities"] = None
        for mod in sorted((s or {}).get("modality_stats", {}).keys()):
            modality_filter_map[modality_name(mod)] = mod
        current = mod_view_var.get()
        values = list(modality_filter_map.keys())
        mod_view_cb["values"] = values
        if current not in modality_filter_map:
            mod_view_var.set("All modalities")

        # Rebuild device filter from device_stats.
        device_filter_map.clear()
        device_filter_map["All devices"] = None
        dev_stats = (s or {}).get("device_stats") or {}
        for mod, devs in dev_stats.items():
            for dev in sorted(devs.keys()):
                if dev:
                    device_filter_map[dev] = dev
        cur_dev = mod_device_var.get()
        dev_values = list(device_filter_map.keys())
        mod_device_cb["values"] = dev_values
        if cur_dev not in device_filter_map:
            mod_device_var.set("All devices")

        # Rebuild year filter from year_stats.
        year_filter_map.clear()
        year_filter_map["All years"] = None
        yr_stats = (s or {}).get("year_stats") or {}
        for mod, yrs in yr_stats.items():
            for yr in sorted(yrs.keys()):
                if yr is not None:
                    year_filter_map[str(yr)] = yr
        cur_yr = mod_year_var.get()
        yr_values = list(year_filter_map.keys())
        mod_year_cb["values"] = yr_values
        if cur_yr not in year_filter_map:
            mod_year_var.set("All years")

    def _rebuild_region_filter_values(exam_data, exam_columns=None):
        if len(reg_view_cb["values"]) > 1:
            return
        region_modality_filter_map.clear()
        region_modality_filter_map["All modalities"] = None

        # Use columnar data if available (efficient path)
        if exam_columns is not None:
            modalities = {str(exam_columns.modality[i] or "").strip() for i in range(len(exam_columns)) if str(exam_columns.modality[i] or "").strip()}
        else:
            # Dict-based processing for non-columnar exam_data
            modalities = {str(ex.get("modality", "") or "").strip() for ex in (exam_data or []) if str(ex.get("modality", "") or "").strip()}

        for mod in sorted(modalities):
            region_modality_filter_map[modality_name(mod)] = mod
        current = reg_view_var.get()
        values = list(region_modality_filter_map.keys())
        reg_view_cb["values"] = values
        if current not in region_modality_filter_map:
            reg_view_var.set("All modalities")

    def _filtered_region_exam_data(exam_data):
        selected_label = reg_view_var.get()
        selected_modality = region_modality_filter_map.get(selected_label)
        if selected_modality is None:
            return list(exam_data or [])
        return [ex for ex in (exam_data or []) if str(ex.get("modality", "") or "") == selected_modality]


    def _upd_mod(s):
        """Rebuild the By-Modality treeview rows from stats ``s`` under the
        current modality/device/year filters."""
        import numpy as _np
        for i in mod_tree.get_children():
            mod_tree.delete(i)
        selected_label = mod_view_var.get()
        selected_modality = modality_filter_map.get(selected_label)
        selected_device = device_filter_map.get(mod_device_var.get())
        selected_year = year_filter_map.get(mod_year_var.get())
        iterations = _safe_iterations(s)

        mod_ax_ed.clear()
        mod_ax_det.clear()

        # Quartile palette shared by row backgrounds and chart bars
        _Q_TAGS   = ("mod_q1", "mod_q2", "mod_q3", "mod_q4")
        _Q_COLORS = tuple(_QUARTILE_RAMP)

        def _quartile_rank(val, qs):
            if val <= 0:
                return -1
            if val <= qs[0]:
                return 0
            if val <= qs[1]:
                return 1
            if val <= qs[2]:
                return 2
            return 3

        def _style_mod_ax(ax):
            # Cosmetic pass over an already-drawn panel: 35% larger title and
            # bold title / axis-label / tick text.
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=12, fontweight="bold")
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=11, fontweight="bold")
            for _lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                _lbl.set_fontweight("bold")
                _lbl.set_fontsize(11)

        def _bar_chart(ax, labels, values, title, xlabel, ranks):
            y = _np.arange(len(labels))
            colors = [_Q_COLORS[r] if r >= 0 else _SEMANTIC["reference"] for r in ranks]
            ax.barh(y, values, color=colors, edgecolor="white", height=0.72)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)
            x_max = max(values) if values else 1.0
            if x_max <= 0:
                x_max = 1.0
            ax.set_xlim(0, x_max * 1.18)
            for i, v in enumerate(values):
                if v > 0:
                    ax.text(v + x_max * 0.015, i, f"{v:.3g}",
                            va="center", ha="left", fontsize=9.5,
                            fontweight="bold", color="#2C3E50")

        def _build_entry(display, stats_dict):
            """Build one summary row dict (scans, ED/exam, collective dose,
            detriment, detriment-per-exam) for the modality table from
            ``stats_dict`` and the iteration count."""
            scans = _per_iteration_value(stats_dict.get("scans", 0), iterations)
            # Per-exam means divide by risk-modelled exams only (organ-dose
            # model gives >=1 cancer-site dose, region not Non-Dose) — see
            # vectorized.scans_model_per_bin. Fall back to scans_dose then
            # scans for results without the modelled mask. "Exams / year"
            # stays the full scans volume.
            scans_model = _per_iteration_value(
                stats_dict.get("scans_model",
                                stats_dict.get("scans_dose", stats_dict.get("scans", 0))),
                iterations)
            _has_model = "scans_model" in stats_dict
            dose_sv_model = _per_iteration_value(stats_dict.get("dose_sv_model", 0.0), iterations)
            dose_sv_all = _per_iteration_value(stats_dict.get("dose_sv", 0.0), iterations)
            # Displayed collective dose uses the modelled dose (fallback to
            # the full dose when the modelled mask is absent): the tables
            # describe modelled exams only.
            dose_sv = dose_sv_model if _has_model else dose_sv_all
            cases = _per_iteration_value(stats_dict.get("cases", 0), iterations)
            deaths = _per_iteration_value(stats_dict.get("deaths", 0), iterations)
            expected_deaths = _per_iteration_value(stats_dict.get("expected_deaths", 0.0), iterations)
            detriment = _per_iteration_value(stats_dict.get("detriment", 0.0), iterations)
            ed_per_scan_msv = (dose_sv * 1000.0 / scans_model) if scans_model > 0 else 0
            dps = (detriment / scans_model * 365.25 * 24) if scans_model > 0 else 0
            crpv = (cases / dose_sv * 100) if dose_sv > 0 else 0
            drpv = (deaths / dose_sv * 100) if dose_sv > 0 else 0
            drpv_expected = (expected_deaths / dose_sv * 100) if dose_sv > 0 else 0
            return {
                "display": display,
                "scans": scans, "scans_model": scans_model, "dose_sv": dose_sv,
                "cases": cases, "deaths": deaths,
                "detriment": detriment,
                "ed_per_scan_msv": ed_per_scan_msv, "dps": dps,
                "crpv": crpv, "drpv": drpv, "drpv_expected": drpv_expected,
            }

        def _render_entries(entries, col_header):
            """Render the built entries into ``mod_tree`` with the column
            headings and quartile-coloured detriment-per-exam cells."""
            mod_tree.heading("mod", text=col_header)
            mod_tree.heading("dose", text="Collective dose / year (person Sv)")
            mod_tree.heading("edps", text="ED/exam (mSv)")
            mod_tree.heading("det", text="Detriment / year")
            mod_tree.heading("dps", text="Detriment caused/exam (hours)")

            _pos_dps = [e["dps"] for e in entries if e["dps"] > 0]
            dps_qs = list(_np.percentile(_pos_dps, [25, 50, 75])) if _pos_dps else [0, 0, 0]

            for e in entries:
                rank = _quartile_rank(e["dps"], dps_qs)
                tag = (_Q_TAGS[rank],) if rank >= 0 else ()
                mod_tree.insert("", tk.END, tags=tag, values=(
                    e["display"], f"{e['scans']:,.0f}", f"{e['dose_sv']:.3f}",
                    f"{e['ed_per_scan_msv']:.3f}",
                    f"{e['cases']:.4f}", f"{e['deaths']:.4f}", f"{e['detriment']:.6f}",
                    f"{e['crpv']:.4f}",
                    _format_sim_expected(e['drpv'], e['drpv_expected']),
                    f"{e['dps']:.4f}",
                ))

            if entries:
                labels = [e["display"] for e in entries]
                ed_vals = [e["ed_per_scan_msv"] for e in entries]
                dps_vals = [e["dps"] for e in entries]
                _pos_ed = [v for v in ed_vals if v > 0]
                ed_qs = list(_np.percentile(_pos_ed, [25, 50, 75])) if _pos_ed else [0, 0, 0]
                ed_ranks  = [_quartile_rank(v, ed_qs)  for v in ed_vals]
                dps_ranks = [_quartile_rank(v, dps_qs) for v in dps_vals]
                _bar_chart(mod_ax_ed,  labels, ed_vals,
                           "Effective dose per exam",
                           "mSv / exam", ed_ranks)
                _bar_chart(mod_ax_det, labels, dps_vals,
                           "Expected QALY lost per exam (hours)",
                           "hours / exam", dps_ranks)
                _style_mod_ax(mod_ax_ed)
                _style_mod_ax(mod_ax_det)
            else:
                for _ax, _title in ((mod_ax_ed, "Effective dose per exam"),
                                    (mod_ax_det, "Expected QALY lost per exam (hours)")):
                    _ax.set_title(_title, fontsize=9, fontweight="bold")
                    _ax.text(0.5, 0.5, "No data", ha="center", va="center",
                             transform=_ax.transAxes, fontsize=10, color="#888")
                    _ax.set_xticks([]); _ax.set_yticks([])
                    _style_mod_ax(_ax)
            mod_canvas.draw_idle()

        # No modality selected: show per-modality summary
        if selected_modality is None:
            entries = []
            if selected_device is not None:
                # Per-modality rows filtered by selected device.
                dev_stats = (s or {}).get("device_stats") or {}
                for mod in sorted(s.get("modality_stats", {}).keys(),
                                  key=lambda m: s["modality_stats"][m].get("cases", 0), reverse=True):
                    ds = dev_stats.get(mod, {}).get(selected_device)
                    if ds is None or ds.get("scans", 0) == 0:
                        continue
                    entries.append(_build_entry(modality_name(mod), ds))
                _render_entries(entries, "Modality")
            elif selected_year is not None:
                # Per-modality rows filtered by selected year.
                yr_stats = (s or {}).get("year_stats") or {}
                for mod in sorted(s.get("modality_stats", {}).keys(),
                                  key=lambda m: s["modality_stats"][m].get("cases", 0), reverse=True):
                    ys = yr_stats.get(mod, {}).get(selected_year)
                    if ys is None or ys.get("scans", 0) == 0:
                        continue
                    entries.append(_build_entry(modality_name(mod), ys))
                _render_entries(entries, "Modality")
            else:
                # Standard per-modality summary.
                for mod, ms in sorted(s.get("modality_stats", {}).items(),
                                      key=lambda x: x[1].get("cases", 0), reverse=True):
                    entries.append(_build_entry(modality_name(mod), ms))
                _render_entries(entries, "Modality")
            return

        # Modality selected: drill-down view
        if selected_device is not None:
            # Show per-device breakdown within selected modality.
            dev_stats = ((s or {}).get("device_stats") or {}).get(selected_modality, {}) or {}
            entries = []
            for dev in sorted(dev_stats.keys(), key=lambda d: dev_stats[d].get("scans", 0), reverse=True):
                ds = dev_stats[dev]
                if ds.get("scans", 0) == 0:
                    continue
                entries.append(_build_entry(str(dev), ds))
            _render_entries(entries, "Device")
            return

        if selected_year is not None:
            # Show per-year breakdown within selected modality.
            yr_stats = ((s or {}).get("year_stats") or {}).get(selected_modality, {}) or {}
            entries = []
            for yr in sorted(yr_stats.keys(), reverse=True):
                ys = yr_stats[yr]
                if ys.get("scans", 0) == 0:
                    continue
                entries.append(_build_entry(str(yr), ys))
            _render_entries(entries, "Year")
            return

        # Modality selected, no device/year: per-region drill-down
        mod_tree.heading("mod", text="Region")
        mod_tree.heading("dose", text="Collective dose / year (person Sv)")
        mod_tree.heading("edps", text="Expected ED/exam (mSv)")
        mod_tree.heading("det", text="Detriment / year")
        mod_tree.heading("dps", text="Detriment caused/exam (hours)")

        ms_data = ((s or {}).get("modality_stats", {}) or {}).get(selected_modality, {}) or {}
        region_stats = ms_data.get("regions", {}) or {}
        reg_entries = []
        for region, rv in sorted(region_stats.items(), key=lambda x: x[1].get("scans", 0), reverse=True):
            scans = _per_iteration_value(rv.get("scans", 0), iterations)
            # Risk-modelled exams in the per-exam denominator (fallback to
            # scans_dose then scans for results without the mask).
            scans_model = _per_iteration_value(
                rv.get("scans_model", rv.get("scans_dose", rv.get("scans", 0))), iterations)
            _has_model = "scans_model" in rv
            dose_sv_model = _per_iteration_value(rv.get("dose_sv_model", 0.0), iterations)
            dose_sv_all = _per_iteration_value(rv.get("dose_sv", 0.0), iterations)
            dose_sv = dose_sv_model if _has_model else dose_sv_all
            ed_per_scan_msv = (dose_sv * 1000.0 / scans_model) if scans_model > 0 else 0
            reg_entries.append({
                "display": str(region),
                "scans": scans, "dose_sv": dose_sv,
                "ed_per_scan_msv": ed_per_scan_msv,
            })

        _pos_ed = [e["ed_per_scan_msv"] for e in reg_entries if e["ed_per_scan_msv"] > 0]
        ed_qs = list(_np.percentile(_pos_ed, [25, 50, 75])) if _pos_ed else [0, 0, 0]
        ed_by_region = {e["display"]: e["ed_per_scan_msv"] for e in reg_entries}

        for row in _compute_modality_region_rows(s, selected_modality):
            region_name = str(row[0])
            ed_val = ed_by_region.get(region_name, 0.0)
            rank = _quartile_rank(ed_val, ed_qs)
            tag = (_Q_TAGS[rank],) if rank >= 0 else ()
            mod_tree.insert("", tk.END, tags=tag, values=row)

        if reg_entries:
            labels = [e["display"] for e in reg_entries]
            ed_vals = [e["ed_per_scan_msv"] for e in reg_entries]
            scan_vals = [e["scans"] for e in reg_entries]
            ed_ranks = [_quartile_rank(v, ed_qs) for v in ed_vals]
            _pos_scans = [v for v in scan_vals if v > 0]
            scan_qs = list(_np.percentile(_pos_scans, [25, 50, 75])) if _pos_scans else [0, 0, 0]
            scan_ranks = [_quartile_rank(v, scan_qs) for v in scan_vals]
            _mod_disp = modality_name(selected_modality)
            _bar_chart(mod_ax_ed,  labels, ed_vals,
                       f"Effective dose per exam \u2014 {_mod_disp}",
                       "mSv / exam", ed_ranks)
            _bar_chart(mod_ax_det, labels, scan_vals,
                       f"Exam volume \u2014 {_mod_disp}",
                       "exams / year", scan_ranks)
            _style_mod_ax(mod_ax_ed)
            _style_mod_ax(mod_ax_det)
        else:
            for _ax, _title in ((mod_ax_ed, "Effective dose per exam"),
                                (mod_ax_det, "Exam volume")):
                _ax.set_title(_title, fontsize=9, fontweight="bold")
                _ax.text(0.5, 0.5, "No region data", ha="center", va="center",
                         transform=_ax.transAxes, fontsize=10, color="#888")
                _ax.set_xticks([]); _ax.set_yticks([])
                _style_mod_ax(_ax)
        mod_canvas.draw_idle()

    def _upd_cancer(s):
        """Rebuild the Cancer Type Breakdown text block from stats ``s``,
        restricted to visible cancer types."""
        iterations = _safe_iterations(s)
        visible = _visible_cancers_set()
        
        # Format cancer summary text
        lines = []
        lines.append("=" * 100)
        lines.append("Cancer Type Breakdown")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"{'Cancer Type':<30} {'Cancers / year':<20} {'Deaths / year':<20} {'Lethality %':<15}")
        lines.append("-" * 100)
        
        rem_bd = s.get("rem_subtype_breakdown", {})
        for ct, cv in sorted(s.get("cancer_breakdown", {}).items(),
                              key=lambda x: x[1]["developed"], reverse=True):
            if ct not in visible:
                continue
            developed = _per_iteration_value(cv.get("developed", 0), iterations)
            deaths = _per_iteration_value(cv.get("deaths", 0), iterations)
            lethality = (deaths / developed * 100) if developed > 0 else 0
            lines.append(f"{ct:<30} {developed:<20.4f} {deaths:<20.4f} {lethality:<15.1f}")
            if ct == "Other solid" and rem_bd:
                for rct in ("Brain", "Kidney", "Pancreas", "other"):
                    rcv = rem_bd.get(rct)
                    if not rcv:
                        continue
                    rdev = _per_iteration_value(rcv.get("developed", 0), iterations)
                    rdeath = _per_iteration_value(rcv.get("deaths", 0), iterations)
                    rleth = (rdeath / rdev * 100) if rdev > 0 else 0
                    pct = (rdev / developed * 100) if developed > 0 else 0
                    label = f"  \u21b3 {rct}" if rct != "other" else "  \u21b3 other (\u00d710 organs)"
                    lines.append(f"{label:<30} {rdev:<20.4f} {rdeath:<20.4f} {rleth:<15.1f}  ({pct:.1f}%)")

        heritable_risk_total = _per_iteration_value(s.get("heritable_risk", 0.0), iterations)
        if heritable_risk_total > 0:
            lines.append(f"{'Gonads (heritable)':<30} {heritable_risk_total:<20.4f} {'—':<20} {'n/a':<15}  *heritable")

        lines.append("=" * 100)
        
        cancer_text.delete("1.0", tk.END)
        cancer_text.insert(tk.END, "\n".join(lines))
        
        # Format cancers per Sv text
        _upd_cancer_per_sv(cancer_per_sv_text, s, iterations, visible)


    def _upd_summary_iter(cl, dl, cancer_bd=None, is_final=False, events=None, stats=None,
                          scale=None):
        """``scale``, when set, projects the four per-iteration distribution
        charts (cancers/deaths/QALYs/life-years) to a fixed examination
        count (see ``_summary_per100k_scale``) instead of this cohort's own
        per-iteration values."""
        summary_iter_ax1.clear(); summary_iter_ax2.clear(); summary_iter_ax3.clear()
        if events is None:
            events = results.get("events") or []
        visible_cl = _visible_iter_case_counts(events, len(cl)) if cl else cl
        # Fast-path: if _visible_iter_case_counts returns an empty list, it means
        # no cancers are hidden, so we can use the precomputed cl directly.
        # This avoids O(events) iteration on the GUI thread during live updates.
        if cl and visible_cl == []:
            visible_cl = cl
        _scale_kwargs = (
            {"scale": scale, "per_label": "per 100,000 examinations"}
            if scale else {}
        )
        plot_iteration_histogram(summary_iter_ax1, summary_iter_ax2, visible_cl, dl, **_scale_kwargs)
        if cancer_bd:
            plot_cancer_pie(summary_iter_ax3, cancer_bd, _visible_cancers_set())
        else:
            summary_iter_ax3.set_title("Cancer Type Distribution")
        summary_iter_canvas.draw_idle()
        # QALYs and frequency-vs-burden panels require
        # event/QALYL data that is only available after the simulation
        # completes.  Skip them during live updates to avoid redundant
        # redraws with incomplete data.
        if not is_final:
            return
        # QALYs tab: per-iteration QALYs lost (overall) & life-years-lost
        # (fatal only) distributions, read from the final stats.
        summary_QALY_ax1.clear(); summary_QALY_ax2.clear()
        _st = stats if stats is not None else (results.get("stats") or {})
        plot_QALY_histogram(
            summary_QALY_ax1, summary_QALY_ax2,
            _st.get("iter_QALY_list") or [],
            _st.get("iter_life_list") or [],
            **_scale_kwargs,
        )
        summary_QALY_canvas.draw_idle()
        # Frequency vs burden tab: two aligned horizontal bar charts
        # comparing cancer frequency (%) with QALYL contribution (%).
        if cancer_bd:
            _ev_cols = events if events is not None else results.get("events")
            plot_cancer_frequency_vs_burden(
                summary_freq_burden_ax1, summary_freq_burden_ax2,
                cancer_bd, _ev_cols, _visible_cancers_set(),
            )
        else:
            summary_freq_burden_ax1.clear()
            summary_freq_burden_ax2.clear()
            summary_freq_burden_ax1.set_title("No cancer data yet")
            summary_freq_burden_ax2.set_title("No cancer data yet")
        summary_freq_burden_canvas.draw_idle()


    def _db_read_extrapolation():
        """Read the extrapolation box. Returns a positive float, or None when blank or invalid."""
        raw = str(db_extrap_var.get() or "")
        raw = raw.replace("\u00a0", "").replace(" ", "").replace(",", ".").strip()
        if raw == "":
            return None
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if not (0.0 < value < 1.0e6):
            messagebox.showwarning(
                "Extrapolation",
                "Enter a positive collective dose in person-Sv (for example 50), "
                "or leave the box blank.")
            return None
        return value

    def _db_draw(sweep):
        """Redraw the by-modality summary + both figures from the Region x
        Age sweep.

        No recomputation: the sweep is already built and cached in
        _rq_cache by _upd_research_questions, which calls this after its
        own successful refresh.  The text companion describes the WHOLE
        cohort (all exams, including zero-dose modalities like "Dental
        imaging"), so it gets the sweep unstripped; the two figures plot
        only the dosed modalities and get the stripped sweep.
        """
        if _db_cache.get("drawing"):
            return
        _db_cache["drawing"] = True
        try:
            # Strip the "Dental imaging" pseudo-modality from the two
            # PLOTS: those exams (Panoramic/Kefalometry/Intraoral/Dental)
            # have no meaningful dose model for this comparison.  Dental
            # CBCT is keyed under "CBCT" and is retained.  The
            # table intentionally keeps it (whole-cohort participant
            # characteristics + a zero-dose burden row).
            plot_sweep = _db_strip_dental_imaging(sweep) if sweep else sweep
            db_text.delete("1.0", tk.END)
            _ex_dose = _db_read_extrapolation()
            db_text.insert(tk.END,
                           format_burden_vs_dose_by_modality_summary(
                               sweep or {}, extrapolate_to_dose_sv=_ex_dose,
                               patients_by_modality=(
                                   results.get("stats") or {}
                               ).get("_patients_by_modality")))
            plot_burden_vs_dose_small_multiples(
                plot_sweep or {}, db_sm_fig, extrapolate_to_dose_sv=_ex_dose)
            db_sm_canvas.draw_idle()
            _db_sync_scroll_size()
        finally:
            _db_cache["drawing"] = False

    def _upd_dose_burden():
        """Refresh the "Burden vs collective dose" sub-tab.

        The view reads the Region x Age sweep (_rq_cache["sweep"]), which
        _upd_research_questions builds and caches before calling this.  If
        the sweep is not yet available (the user has not opened the Region
        x Age sub-tab, or the run has no dosed examinations), show the
        companion "open Region x Age first" message rather than computing
        a sweep implicitly -- the sweep is expensive and is owned by the
        Region x Age tab's controls (per-N, modality filter).
        """
        try:
            sweep = _rq_cache.get("sweep")
            _db_draw(sweep)
        except Exception:
            import traceback
            db_text.delete("1.0", tk.END)
            db_text.insert(tk.END, "Analysis failed:\n" + traceback.format_exc())

    db_extrap_apply_btn.configure(command=_upd_dose_burden)
    db_extrap_entry.bind("<Return>", lambda _e: _upd_dose_burden())

    def _upd_research_questions(stats: dict, exam_data: list,
                                 cancer_events: list | None = None):
        """Refresh the Region × Age dose-reduction sub-tab from the latest results.

        ``cancer_events`` is required to derive per-cell QALYs lost — the
        engine only tracks the global ``QALY_lost`` total, so without
        events the QALYs columns would render as all zeros.
        """
        ce = cancer_events if cancer_events is not None else (
            results.get("events") or [])
        # The "Burden vs collective dose" sub-tab reads the Region x
        # Age sweep, so it is refreshed AFTER the sweep is (re)built and
        # cached below -- the call sits at the end of the successful
        # sweep path so a sweep failure cannot leave it stale.
        # Dose-reduction benefit per region x 10-year age group
        try:
            # Resolve the per-N normaliser from the dropdown ("100,000" -> 100000).
            try:
                pern = int(str(rq_pern_var.get() or "100,000").replace(",", "").strip())
            except ValueError:
                pern = 100_000
            # Refresh the modality picker so the listbox always reflects
            # the modalities actually present in the latest simulation.
            # Labels are "<code> — <human label>" (em-dash)
            # and we split on the separator to recover the canonical code
            # used by exam_data / cancer_events.
            mod_codes = sorted((stats or {}).get("modality_stats", {}).keys())
            # Add "Dental imaging" pseudo-modality if any exams have a
            # dental-imaging subtype region (dental panoramic, intraoral,
            # kefalometry — but NOT Dental CBCT, which already carries
            # the "CBCT" modality).
            # Each subtype keeps its own reporting_region value (see
            # exam_doses.py), so membership is checked against the set
            # rather than a single collapsed "Dental imaging" string.
            from .region_resolver import DENTAL_IMAGING_SUBTYPE_REGIONS as _dental_subtypes
            # In a per-year-average batch the last processed year's
            # exam_data is the only one still in memory, so deriving
            # "Dental imaging" membership from it would miss dental exams
            # in earlier years.  The pooled sweep's modality_region_map
            # already carries every modality seen in any year, so use it
            # instead; the exam_data scan stays for single/combined runs.
            _rq_is_batch = (stats or {}).get("_run_mode") == "per_year_average"
            if _rq_is_batch:
                _has_dental_imaging = "Dental imaging" in (
                    ((stats or {}).get("_rq_sweep_det_avg") or {})
                    .get("modality_region_map") or {})
            else:
                _has_dental_imaging = any(
                    str(ex.get("reporting_region", "") or "") in _dental_subtypes
                    for ex in (exam_data or [])
                )
            if _has_dental_imaging:
                mod_codes = mod_codes + ["Dental imaging"]
            mod_options = [_RQ_MOD_SENTINEL] + [
                f"{m} — {modality_name(m)}" if modality_name(m) and modality_name(m) != m else m
                for m in mod_codes
            ]
            # Current selection right after the click that triggered this
            # refresh (or unchanged, if this refresh was triggered by
            # something else, e.g. the per-N combobox).
            cur_mod_sel = {rq_modality_lb.get(i) for i in rq_modality_lb.curselection()}
            # Exclusivity rule vs. "All modalities":
            #  - if the user just clicked "All modalities", it becomes the
            #    sole selection (clears every specific modality);
            #  - if "All modalities" was active and the user just clicked a
            #    specific modality, drop the sentinel so only the clicked
            #    modality/modalities remain;
            #  - otherwise (adding/removing specific modalities while the
            #    sentinel is already inactive), leave the selection as-is.
            _newly_selected = cur_mod_sel - _rq_mod_prior_labels
            if _RQ_MOD_SENTINEL in _newly_selected:
                cur_mod_sel = {_RQ_MOD_SENTINEL}
            elif _RQ_MOD_SENTINEL in cur_mod_sel and any(
                    lbl != _RQ_MOD_SENTINEL for lbl in _newly_selected):
                cur_mod_sel = cur_mod_sel - {_RQ_MOD_SENTINEL}
            if not cur_mod_sel:
                cur_mod_sel = {_RQ_MOD_SENTINEL}
            prev_mod_sel = cur_mod_sel
            rq_modality_lb.delete(0, tk.END)
            for opt in mod_options:
                rq_modality_lb.insert(tk.END, opt)
            # Restore selections if still valid; otherwise default to "All".
            restored_any = False
            for i, opt in enumerate(mod_options):
                if opt in prev_mod_sel:
                    rq_modality_lb.selection_set(i)
                    restored_any = True
            if not restored_any:
                rq_modality_lb.selection_set(0)  # "All modalities"
                prev_mod_sel = {_RQ_MOD_SENTINEL}
            _rq_mod_prior_labels.clear()
            _rq_mod_prior_labels.update(prev_mod_sel)
            # Read selected modalities from the Listbox.
            sel_indices = rq_modality_lb.curselection()
            sel_labels = [rq_modality_lb.get(i) for i in sel_indices]
            # If "All modalities" is selected (either alone or alongside
            # others), treat as no filter — it means "all".
            if not sel_labels or _RQ_MOD_SENTINEL in sel_labels:
                modality_filter = None
            else:
                modality_filter = [
                    lbl.split(" — ", 1)[0] for lbl in sel_labels
                ]
            # Cache the sweep by the parameters that actually change its
            # contents: per-N, modality and underlying exam/event data.
            # Region / metric / target only change how the same sweep is
            # filtered and plotted, so they are cheap.
            _sweep_key = (
                pern, modality_filter,
                id(exam_data or []), len(ce),
            )
            # "Average across years" batch: exam_data/cancer_events above are
            # only the last processed year (see year_batch.py's memory
            # note), so recomputing the sweep from them would silently
            # mislabel one year's numbers as "average across years".
            # year_batch.py instead runs compute_region_age_dose_sweep()
            # once per year -- with deterministic_baseline=True, since the
            # Research Questions tab ALWAYS uses the analytical ICRP-152
            # baseline -- while that year's data is still alive, and keeps
            # the result on the year's own stats as ``_rq_sweep_det``.  The
            # modality filter is applied to each year's sweep and the
            # results re-pooled here, which is bit-for-bit what a per-year
            # run with that filter would have produced: every cell is
            # keyed by modality and computed from its own exams alone (see
            # research_questions.filter_region_age_sweep).
            # ``_rq_sweep_det_avg`` is the already-pooled unfiltered sweep,
            # used as the "All modalities" fast path.  Only the per-N
            # normaliser is applied afterwards, as a pure linear rescale
            # (rescale_region_age_sweep).
            _rq_key = "_rq_sweep_det"
            _rq_avg_sweep = (stats or {}).get("_rq_sweep_det_avg")
            _rq_batch_note = ""
            if _rq_is_batch and _rq_avg_sweep:
                if modality_filter is None:
                    sweep = _rq_avg_sweep
                else:
                    sweep = _rq_cache.get("sweep")
                    if sweep is None or _rq_cache.get("sweep_key") != _sweep_key:
                        # Every year that produced a sweep takes part, even
                        # one with no examinations of the selected modality:
                        # that year genuinely performed none, and dropping
                        # it would shrink the pooling denominator and
                        # overstate the typical annual burden -- the same
                        # "absent = genuine 0" convention the unfiltered
                        # pool already uses.
                        sweep = pool_region_age_sweeps([
                            filter_region_age_sweep(y[_rq_key], modality_filter)
                            for y in ((stats or {}).get("_year_batches") or {}).values()
                            if y.get(_rq_key)
                        ])
                if not sweep or not sweep.get("per_region_age"):
                    rq_text.delete("1.0", tk.END)
                    rq_text.insert(tk.END,
                        "No examinations of the selected modality/modalities "
                        "in any year group of this batch.\n\n"
                        "Widen the selection, or run a batch containing only "
                        "that year group.")
                    rq_fig.clear()
                    rq_canvas.draw_idle()
                    _rq_sync_scroll_size()
                    # The burden sub-tab reads the sweep; with no cells there
                    # is nothing to plot, so drop the stale cache and refresh
                    # it to the empty/message state.
                    _rq_cache["sweep"] = None
                    _rq_cache["sweep_key"] = None
                    _upd_dose_burden()
                    return
                sweep = rescale_region_age_sweep(sweep, pern)
                # Central 95% range of simulated annual outcomes: pooled
                # per-iteration series covering every processed year (the
                # same quantity the Summary tab calls the "95% range of
                # simulated years").  Rewrites the per-cell intervals on a
                # sweep copy; analytical point estimates are untouched.
                sweep = attach_simulated_ranges(
                    sweep, stats,
                    iter_cases=results.get("cases_l"),
                    iter_deaths=results.get("deaths_l"))
                _rq_cache["sweep"] = sweep
                _rq_cache["sweep_key"] = _sweep_key
                _n_rq_years = sweep.get("_averaged_across_years", 1)
                _interval_sentence = (
                    "the interval around each is the analytical 95% range "
                    f"for a cohort of {pern:,} examinations with the cell's "
                    "recorded dose/age/sex mix (chance variation only; it "
                    "does not reflect how few examinations the cell rests "
                    "on). The 'Between-yr 95 % CI' row is the "
                )
                _rq_batch_note = (
                    f"POOLED ACROSS {_n_rq_years} YEAR GROUP(S) \u2014 point "
                    "estimates are the analytical ICRP 152 expectation of "
                    "all year groups pooled as one cohort (exam-weighted), "
                    "annualised by dividing by the number of year groups "
                    "(no Monte Carlo sampling error); " + _interval_sentence
                    + "Student-t interval of the multi-year mean from the "
                    "between-year SD of the annual expectations "
                    "(cohort-composition uncertainty only; no risk-model "
                    "or dosimetric uncertainty) and is the interval to "
                    "report for cell values.\n\n"
                )
            elif _rq_is_batch:
                # Batch, but no year produced an analytical sweep (the
                # ICRP-152 integral failed for every year).  Do NOT fall
                # back to recomputing from the last year's exam_data --
                # that would silently mislabel one year's numbers as the
                # pooled cohort.
                rq_text.delete("1.0", tk.END)
                rq_text.insert(tk.END,
                    "The analytical ICRP 152 baseline could not be computed "
                    "for any year group of this batch.\n\nIt is evaluated "
                    "per year while that year's per-exam detail is still "
                    "in memory, and every year's evaluation failed.  Run a "
                    "batch containing only one year group to inspect it in "
                    "isolation.")
                rq_fig.clear()
                rq_canvas.draw_idle()
                _rq_sync_scroll_size()
                _rq_cache["sweep"] = None
                _rq_cache["sweep_key"] = None
                _upd_dose_burden()
                return
            else:
                sweep = _rq_cache.get("sweep")
                if sweep is None or _rq_cache.get("sweep_key") != _sweep_key:
                    sweep = compute_region_age_dose_sweep(
                        stats or {}, exam_data or [], cancer_events=ce,
                        per_n_scans=pern,
                        modality=modality_filter,
                        deterministic_baseline=True,
                    )
                    # Central 95% range of simulated annual outcomes from
                    # this run's own per-iteration series (the Summary
                    # tab's "95% range of simulated years").
                    sweep = attach_simulated_ranges(
                        sweep, stats or {}, events=ce,
                        exam_data=exam_data,
                        iter_cases=results.get("cases_l"),
                        iter_deaths=results.get("deaths_l"))
                    _rq_cache["sweep"] = sweep
                    _rq_cache["sweep_key"] = _sweep_key
            # Refresh the region picker so it always lists the regions
            # actually seen in this simulation.  Preserve the user's prior
            # selections if still valid.
            # Regions are now (modality, region) tuples; display as
            # "modality — region" for clarity.
            region_pairs = list(sweep.get("regions") or [])
            region_options = []
            for item in region_pairs:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    mod, reg = item
                    if mod:
                        region_options.append(f"{mod} — {reg}")
                    else:
                        region_options.append(reg)
                else:
                    region_options.append(str(item))
            prev_sel = {rq_region_lb.get(i) for i in rq_region_lb.curselection()}
            rq_region_lb.delete(0, tk.END)
            for r in region_options:
                rq_region_lb.insert(tk.END, r)
            for i, r in enumerate(region_options):
                if r in prev_sel:
                    rq_region_lb.selection_set(i)
            selected = [rq_region_lb.get(i) for i in rq_region_lb.curselection()]
            # Build region_filter as a set of (modality, region) tuples
            # so filtering is modality-aware.
            if selected:
                region_filter = []
                for s in selected:
                    if " — " in s:
                        mod_part, reg_part = s.split(" — ", 1)
                        region_filter.append((mod_part, reg_part))
                    else:
                        region_filter.append(("", s))
            else:
                region_filter = None

            # No reduction level is passed: this summary tabulates the
            # baseline burden per age decade and region and does not depend
            # on one.  The figure likewise collapses the dose-reduction
            # dimension, since under LNT every level is a scalar multiple
            # of any other.
            # Re-flag the small-cell floor from the tab's threshold entry.
            # Cheap post-transform (flags only, numbers untouched); stored
            # back so the burden sub-tab sees the same flags as the summary
            # and heatmap.
            sweep = apply_min_scans(sweep, _rq_min_scans_value())
            # Display-only interval mode tag (cell_interval): the text
            # summary and the burden sub-tab read the same tagged sweep.
            sweep = set_cell_interval(sweep, CELL_INTERVAL_COHORT)
            _rq_cache["sweep"] = sweep
            rq_text.delete("1.0", tk.END)
            if _rq_batch_note:
                rq_text.insert(tk.END, _rq_batch_note)
            rq_text.insert(tk.END, format_region_age_sweep_summary(
                sweep, region_filter=region_filter))
            plot_region_age_sweep(rq_fig, sweep,
                                  region_filter=region_filter,
                                  value_mode=rq_value_mode_var.get(),
                                  hide_small_cells=rq_hide_small_var.get())
            rq_canvas.draw_idle()
            _rq_sync_scroll_size()
            # Now that the sweep is built and cached, refresh the sibling
            # "Burden vs collective dose" sub-tab, which reads it directly.
            _upd_dose_burden()
        except Exception:
            import traceback
            _rq_cache["sweep"] = None
            _rq_cache["sweep_key"] = None
            rq_text.delete("1.0", tk.END)
            rq_text.insert(tk.END, "Analysis failed:\n" + traceback.format_exc())
            # Refresh the burden sub-tab too so it shows the empty/message
            # state rather than a stale figure from a previous run.
            _upd_dose_burden()

    def _resolve_region_view():
        """Resolve the By Body Region source dropdown to a FileCohort or
        raw exam_data list, applying the modality filter.

        Returns something :func:`region_by_modality` accepts: a
        :class:`FileCohort` when the compact cohort is available, or a list
        of exam dicts / ExamColumns otherwise.
        """
        cohort = results.get("file_cohort")
        if cohort is None:
            _s = results.get("stats")
            if isinstance(_s, dict):
                cohort = _s.get("_file_cohort")
        if cohort is not None and len(cohort) > 0:
            sel = reg_source_var.get()
            kind_key = reg_source_map.get(sel, ("all", None))
            fc = cohort.select(kind_key[0], kind_key[1])
            # Apply the modality filter via FileCohort.filtered, which is
            # exact and avoids expanding to per-exam rows.
            mod_label = reg_view_var.get()
            mod_code = region_modality_filter_map.get(mod_label)
            if mod_code is not None:
                fc = fc.filtered(mod_code)
            return fc
        # Fallback: raw exam_data path.
        ed = results.get("exam_data") or []
        return _filtered_region_exam_data(ed)

    def _rebuild_region_source_choices():
        """Populate the By Body Region source dropdown from the cohort."""
        cohort = results.get("file_cohort")
        if cohort is None:
            _s = results.get("stats")
            if isinstance(_s, dict):
                cohort = _s.get("_file_cohort")
        if cohort is None or len(cohort) == 0:
            reg_source_cb["values"] = ["All files (combined)"]
            reg_source_map.clear()
            return
        choices = cohort.selector_choices()
        reg_source_map.clear()
        for label, kind, key in choices:
            reg_source_map[label] = (kind, key)
        values = ["All files (combined)"] + [label for label, _, _ in choices]
        reg_source_cb["values"] = values
        if reg_source_var.get() not in values:
            reg_source_var.set("All files (combined)")

    def _upd_regions(ed):
        # Rebuild the source dropdown in case a new simulation produced
        # a different set of files/year groups.
        _rebuild_region_source_choices()
        view = _resolve_region_view()
        rbm = region_by_modality(view)
        plot_all_regions_by_modality(reg_fig, rbm)
        reg_canvas.draw_idle()
        # Generate the region debug text from the same source the figures
        # use.  When a FileCohort is available (multi-file / multi-year
        # runs), the text is built from the cohort's region_debug count
        # table, which covers every file of every year — not just the last
        # processed one.  Falls back to the raw exam_data path for single
        # runs that never built a cohort, or when the cohort has no
        # region_debug entries (e.g. ExamColumns input).
        if hasattr(view, "region_debug"):
            _text = _region_debug_text_from_cohort(view)
        else:
            _text = _region_debug_text(_filtered_region_exam_data(ed))
        # Excluded examinations (Non-Dose / no organ-dose model): dropped
        # from the analysed cohort at source, so they are absent from the
        # view above; listed here from the exclusion ledger with the exact
        # reason, under the same source / modality selection.
        try:
            _xc = results.get("excluded_cohorts")
            if _xc is None:
                _s = results.get("stats")
                if isinstance(_s, dict):
                    _xc = _s.get("_excluded_cohorts")
            _kind, _key = reg_source_map.get(reg_source_var.get(), ("all", None))
            _xtext = _excluded_examinations_text(
                _xc, _kind, _key,
                modality=region_modality_filter_map.get(reg_view_var.get()))
        except Exception:
            _xtext = ""
        if _xtext:
            _text = (_text.rstrip() + "\n\n\n" + _xtext) if _text else _xtext
        reg_text.delete("1.0", tk.END)
        reg_text.insert(tk.END, _text)

    def _upd_unknown_regions(unknown_list):
        """Populate the Unrecognized Regions subtab from a list of (desc, modality) tuples."""
        for item in unrec_tree.get_children():
            unrec_tree.delete(item)
        if not unknown_list:
            unrec_count_var.set("0 unrecognized descriptions")
            return
        for desc_text, mod in unknown_list:
            unrec_tree.insert("", tk.END, values=(str(mod), str(desc_text)))
        unrec_count_var.set(f"{len(unknown_list):,} unrecognized descriptions")

    def _upd_charts(ed, ev, s):
        """Refresh the chart tabs (Patient Cohort etc.) for the current
        ``ed``/``ev``/``s`` data, preferring the cached columnar exam
        table when present."""
        # Prefer the cached columnar exam data to avoid re-converting the
        # list of dicts every time the Patient Cohort tab is refreshed.
        exam_cols = s.get("_exam_columns") if isinstance(s, dict) else None
        ed_view = exam_cols if exam_cols is not None else ed
        # The file list whose indices match the exams' ``_file_index``
        # values.  Attached to stats by the batch runner; falls back to the
        # GUI's own flattened file_list for direct (non-batched) runs where
        # that list IS the index basis.
        detail_files = (s.get("_detail_files") if isinstance(s, dict) else None) or file_list
        # Compact per-file cohort profiles (see file_cohort.py).  When
        # available, the dropdown offers every file of every year and the
        # charts draw from the count tables instead of the last year's raw
        # exam_data.  Falls back to the raw ed_view path for single runs
        # that never built a cohort.
        cohort = results.get("file_cohort")
        if cohort is None and isinstance(s, dict):
            cohort = s.get("_file_cohort")
        pyramid_state["ed_view"] = ed_view
        pyramid_state["detail_files"] = list(detail_files)
        pyramid_state["cohort"] = cohort if (cohort is not None and len(cohort) > 0) else None
        # Refresh the modality dropdown's choices, keeping the current
        # selection if it's still valid (e.g. across live-update ticks).
        mod_names = _pyramid_modality_names()
        new_values = ["All modalities"] + mod_names
        if tuple(pyramid_modality_combo["values"]) != tuple(new_values):
            pyramid_modality_combo["values"] = new_values
            if pyramid_modality_var.get() not in new_values:
                pyramid_modality_var.set("All modalities")
        # Refresh the input-file dropdown's choices.  Default to the
        # combined view ("All files (combined)") so a multi-file run opens
        # showing the whole cohort; keep the current selection if it is
        # still among the files present in the new view.
        file_choices = _pyramid_file_choices()
        # Build the selector map: label -> (kind, key) for cohort mode,
        # or label -> file_index for the fallback path.
        selector_map = {}
        file_label_to_index = {}
        for label, kind, key in file_choices:
            if kind == "file_idx":
                file_label_to_index[label] = key
            else:
                selector_map[label] = (kind, key)
        pyramid_state["selector_map"] = selector_map
        pyramid_state["file_label_to_index"] = file_label_to_index
        file_values = ["All files (combined)"] + [label for label, _, _ in file_choices]
        if tuple(pyramid_file_combo["values"]) != tuple(file_values):
            pyramid_file_combo["values"] = file_values
            if pyramid_file_var.get() not in file_values:
                pyramid_file_var.set("All files (combined)")
        # Draw all three charts from the current selection.
        _redraw_age_pyramid()
        _redraw_cohort_extra()


    def _apply_ev_filter(*_args):
        """Refill the events table applying the modality and outcome filter
        dropdowns."""
        mod_f = ev_mod_var.get()
        out_f = ev_out_var.get()
        for i in ev_tree.get_children():
            ev_tree.delete(i)
        count = 0

        if isinstance(all_events_store, EventColumns) and len(all_events_store) > 0:
            # Render directly from the columnar store (no dict conversion).
            n = len(all_events_store)
            ct_arr = all_events_store.cancer_type
            age_arr = all_events_store.age_at_exposure
            sex_arr = all_events_store.sex
            mod_arr = all_events_store.modality
            reg_arr = all_events_store.region
            proto_arr = all_events_store.protocol
            ed_arr = all_events_store.ed_msv
            died_arr = all_events_store.died
            ad_arr = all_events_store.age_at_death
            pl_arr = all_events_store.projected_lifetime
            ll_arr = all_events_store.life_lost_years
            for i in range(n):
                if mod_f != "All" and str(mod_arr[i]) != mod_f:
                    continue
                outcome_str = "Death" if died_arr[i] else "Survived"
                if out_f != "All" and outcome_str != out_f:
                    continue
                if count >= MAX_EVENT_TABLE_ROWS:  # store is already capped; belt and braces
                    break
                age_exp = float(age_arr[i])
                age_death = float(ad_arr[i]) if ad_arr[i] is not None else None
                projected_lifetime = float(pl_arr[i]) if pl_arr[i] is not None else None
                life_lost = float(ll_arr[i]) if ll_arr[i] is not None else 0.0
                ad_val = age_death
                pl_val = projected_lifetime
                if ad_val is not None:
                    age_death_text = f"{ad_val:.1f}"
                    if pl_val is not None and abs(pl_val - ad_val) > 0.05:
                        age_death_text = f"{age_death_text} ({pl_val:.1f})"
                elif pl_val is not None:
                    age_death_text = f"({pl_val:.1f})"
                else:
                    age_death_text = ""
                ev_tree.insert("", tk.END, values=(
                    str(ct_arr[i]), f"{age_exp:.1f}",
                    age_death_text,
                    _format_life_lost(life_lost), str(sex_arr[i]), str(mod_arr[i]),
                    str(reg_arr[i]), str(proto_arr[i])[:40],
                    f"{float(ed_arr[i]):.3f}",
                    outcome_str,
                ))
                count += 1
        else:
            for ev in all_events_store:
                if mod_f != "All" and ev.get("modality", "") != mod_f:
                    continue
                outcome_str = "Death" if ev.get("died") else "Survived"
                if out_f != "All" and outcome_str != out_f:
                    continue
                if count >= MAX_EVENT_TABLE_ROWS:  # store is already capped; belt and braces
                    break
                age_exp = ev.get("age_at_exposure", ev.get("age", 0))
                age_death = ev.get("age_at_death")
                projected_lifetime = ev.get("projected_lifetime")
                life_lost = ev.get("life_lost_years", 0.0)
                ad_val = None
                pl_val = None
                if age_death not in (None, ""):
                    try:
                        ad_val = float(age_death)
                    except Exception:
                        ad_val = None
                if projected_lifetime not in (None, ""):
                    try:
                        pl_val = float(projected_lifetime)
                    except Exception:
                        pl_val = None
                if ad_val is not None:
                    age_death_text = f"{ad_val:.1f}"
                    if pl_val is not None and abs(pl_val - ad_val) > 0.05:
                        age_death_text = f"{age_death_text} ({pl_val:.1f})"
                elif pl_val is not None:
                    age_death_text = f"({pl_val:.1f})"
                else:
                    age_death_text = ""
                ev_tree.insert("", tk.END, values=(
                    ev.get("cancer_type", ""), f"{float(age_exp):.1f}",
                    age_death_text,
                    _format_life_lost(float(life_lost)), ev.get("sex", ""), ev.get("modality", ""),
                    ev.get("region", ""), str(ev.get("protocol", ""))[:40],
                    f"{ev.get('ed_msv', 0):.3f}",
                    outcome_str,
                ))
                count += 1


    def _upd_events(evts, append=False):
        """Merge a new batch of events into the events table: filter to
        visible cancer types, cap at MAX_EVENT_TABLE_ROWS, and refresh
        the displayed rows."""
        nonlocal all_events_store, ev_events_seen
        visible_evts = _filter_visible_events(evts)
        if not isinstance(visible_evts, EventColumns):
            # Normalise to the columnar bundle once at entry so the store
            # and every table consumer below only ever see EventColumns.
            visible_evts = EventColumns.from_events(list(visible_evts or []))
        n_new = len(visible_evts)
        if not append:
            ev_events_seen = n_new
            all_events_store = _head_events(visible_evts, MAX_EVENT_TABLE_ROWS)
        else:
            ev_events_seen += n_new
            # Stop accumulating once the debugging sample is full: the rest of
            # the stream is already counted everywhere it matters, and holding
            # a second copy of it here is the whole cost this cap removes.
            room = MAX_EVENT_TABLE_ROWS - len(all_events_store)
            if room > 0:
                head = _head_events(visible_evts, room)
                all_events_store = _concat_event_columns([all_events_store, head])
        # Modality picker describes what the table can actually show, so it
        # stops learning new values at the same point the table does.
        if len(all_events_store) > 0:
            mods_seen = sorted(set(str(m) for m in all_events_store.modality))
        else:
            mods_seen = []
        ev_mod_cb["values"] = ["All"] + sorted(set(mods_seen) - {"All"})
        _n_stored = len(all_events_store)
        if _n_stored >= MAX_EVENT_TABLE_ROWS and ev_events_seen > _n_stored:
            ev_live_count_var.set(
                f"{_n_stored:,} of {ev_events_seen:,} (capped)")
        else:
            ev_live_count_var.set(f"{_n_stored:,}")
        _apply_ev_filter()
        # The histogram deliberately does NOT read the capped store: it is
        # redrawn from the full cumulative stream, and only when its subtab is
        # on screen (the outer tab being visible is not enough).
        if _ev_age_subtab_selected():
            _upd_age_histogram()

    def _ev_age_subtab_selected() -> bool:
        """True when the Age Distribution subtab is the one on screen."""
        try:
            return str(ev_sub_nb.tab(ev_sub_nb.index(ev_sub_nb.select()),
                                     "text")) == "Age Distribution"
        except Exception:
            return False

    def _upd_age_histogram(events=None):
        """Update the cancer age distribution histogram with current data.

        Reads the FULL cumulative event stream by default -- every event of
        every input file and year group -- never the Table subtab's capped
        debugging sample.  That is the whole reason the cap is safe.
        """
        nonlocal ev_age_histogram_drawn
        if events is None:
            events = _filter_visible_events(results.get("events") or [])
        if not isinstance(events, EventColumns):
            events = EventColumns.from_events(list(events or []))
        ev_age_scope_var.set(
            f"{len(events):,} events (all input files)" if events else "")
        if not events:
            ev_age_histogram_drawn = False
            return

        # Populate cancer type combobox from the event data
        cancer_types_seen = sorted(set(str(ct) for ct in events.cancer_type))
        ev_age_type_cb["values"] = ["All"] + cancer_types_seen

        # Apply cancer type filter
        type_f = ev_age_type_var.get()
        if type_f and type_f != "All":
            mask = np.array([str(ct) == type_f for ct in events.cancer_type], dtype=bool)
            if not mask.any():
                ev_age_histogram_drawn = False
                return
            events = EventColumns(
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

        # Apply outcome (lethality) filter
        out_f = ev_age_out_var.get()
        if out_f and out_f != "All":
            if out_f == "Lethal":
                mask = events.died.astype(bool)
            else:
                mask = ~events.died.astype(bool)
            if not mask.any():
                ev_age_histogram_drawn = False
                return
            events = EventColumns(
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

        plot_cancer_age_histogram(ev_age_ax, events, _visible_cancers_set())
        ev_age_canvas.draw_idle()
        ev_age_histogram_drawn = True

    def _on_ev_subtab_change(_event=None):
        """Draw the histogram when its subtab is opened.

        The outer notebook's handler only fires when the Cancer Events tab
        itself is (re)selected, so without this switching between the two
        subtabs left the histogram stale -- and now that the live path skips
        it unless it is on screen, stale would mean empty.
        """
        if _ev_age_subtab_selected():
            _upd_age_histogram()

    ev_sub_nb.bind("<<NotebookTabChanged>>", _on_ev_subtab_change)

    def _refresh_visible_cancer_views():
        s = results.get("stats") or {}
        ev = results.get("events") or []
        ed = results.get("exam_data") or []
        if not s:
            return
        visible_cancers = _visible_cancers_set()
        filtered_bd = filter_cancer_breakdown(s.get("cancer_breakdown", {}), visible_cancers)
        # Summary tab: routed through _render_summary so it honours the
        # Summary-tab modality dropdown as well as the cancer checkboxes.
        _render_summary()
        _upd_cancer(s)
        # _upd_charts does not use events currently, so pass the raw list
        # rather than filtering (the filter would be wasted work).
        _upd_charts(ed, ev, {**s, "cancer_breakdown": filtered_bd})
        _upd_events(ev)
        _upd_mod(s)
        _upd_research_questions(s, ed, cancer_events=ev)
        _upd_sankey(s, visible_cancers=visible_cancers)

    ev_mod_cb.bind("<<ComboboxSelected>>", _apply_ev_filter)
    ev_out_cb.bind("<<ComboboxSelected>>", _apply_ev_filter)
    ev_age_type_cb.bind("<<ComboboxSelected>>", lambda _e: _upd_age_histogram())
    ev_age_out_cb.bind("<<ComboboxSelected>>", lambda _e: _upd_age_histogram())
    mod_view_cb.bind("<<ComboboxSelected>>", lambda _event: _upd_mod(results.get("stats") or {}))
    mod_device_cb.bind("<<ComboboxSelected>>", lambda _event: _upd_mod(results.get("stats") or {}))
    mod_year_cb.bind("<<ComboboxSelected>>", lambda _event: _upd_mod(results.get("stats") or {}))
    summary_mod_view_cb.bind("<<ComboboxSelected>>", lambda _event: _render_summary())
    reg_view_cb.bind("<<ComboboxSelected>>", lambda _event: _upd_regions(results.get("exam_data") or []))
    reg_source_cb.bind("<<ComboboxSelected>>", lambda _event: _upd_regions(results.get("exam_data") or []))
    def _refresh_rq_from_results(_event=None):
        _upd_research_questions(
            results.get("stats") or {},
            results.get("exam_data") or [],
            cancer_events=results.get("events") or [],
        )
    # Control widgets — same refresh callback rebuilds the sweep from
    # the latest simulation results.  The per-N dropdown forces a full
    # recompute (different normaliser), but the region filter is cheap
    # because it operates on the cached sweep matrix.  The annual-benefit
    # reduction dropdown is deliberately NOT bound here: it only affects
    # the pop-out figure, which reads it when opened.
    rq_pern_cb.bind("<<ComboboxSelected>>", _refresh_rq_from_results)
    rq_region_lb.bind("<<ListboxSelect>>", _refresh_rq_from_results)

    rq_modality_lb.bind("<<ListboxSelect>>", _refresh_rq_from_results)

    # Heatmap unit toggle reuses the cached sweep (the value mode is not
    # part of the sweep key), so this is a cheap redraw, not a recompute.
    rq_value_mode_rb_n.configure(command=_refresh_rq_from_results)
    rq_value_mode_rb_h.configure(command=_refresh_rq_from_results)
    rq_hide_small_cb.configure(command=_refresh_rq_from_results)

    # Small-cell exam floor entry: the value drives the WHOLE tab (heatmap
    # blanking, '*' markers) via apply_min_scans on the
    # resolved sweep in _upd_research_questions.  Invalid input is
    # rejected with a bell and the last valid value restored.
    _rq_min_scans_state = {"last_valid": float(DEFAULT_MIN_SCANS)}

    def _rq_min_scans_value() -> float:
        return _rq_min_scans_state["last_valid"]

    def _on_rq_min_scans_change(_event=None):
        txt = rq_min_scans_var.get().strip()
        try:
            v = float(txt)
            if not (v >= 0.0) or v != v:
                raise ValueError
        except ValueError:
            rq_min_scans_entry.bell()
            rq_min_scans_var.set(f"{_rq_min_scans_state['last_valid']:g}")
            return
        if v == _rq_min_scans_state["last_valid"]:
            return
        _rq_min_scans_state["last_valid"] = v
        _refresh_rq_from_results()

    rq_min_scans_entry.bind("<Return>", _on_rq_min_scans_change)
    rq_min_scans_entry.bind("<FocusOut>", _on_rq_min_scans_change)

    # The "Burden vs collective dose" sub-tab is refreshed by
    # _upd_dose_burden(), called at the end of _upd_research_questions once
    # the Region x Age sweep is cached.


    rq_annual_btn.configure(command=lambda: _show_annual_benefit(root, _rq_cache, rq_annual_red_var))


    rq_burden_btn.configure(command=lambda: _show_annual_burden(root, _rq_cache))


    def _poll_worker_queue():
        """Drain queued worker-thread messages on the Tk loop: lightweight
        progress updates are applied inline; heavy visual updates ("live",
        "done") break out early so Tk can render before the next message."""
        # Process messages from the worker thread.  Heavy visual updates
        # ("live", "done") break out of the loop so the tkinter event loop
        # can actually *render* the changes before we process the next one.
        rendered_heavy = False
        try:
            while True:
                msg = ui_q.get_nowait()
                try:
                    k = msg[0]
                    if k == "progress":
                        phase = msg[1] if len(msg) > 1 else ""
                        if phase == "file_load":
                            cur = msg[2] if len(msg) > 2 else 0
                            tot = msg[3] if len(msg) > 3 else 1
                            name = msg[4] if len(msg) > 4 else ""
                            status_var.set(f"Loading input files {cur}/{tot}")
                            detail_var.set(name or "Reading Excel data...")
                            progress_var.set(2.0 + 13.0 * cur / max(1, tot))
                        elif phase == "init":
                            cur = msg[2] if len(msg) > 2 else 0
                            tot = msg[3] if len(msg) > 3 else 1
                            detail = msg[4] if len(msg) > 4 else "Building exam dose data..."
                            pct = (cur / max(1, tot)) * 100.0
                            status_var.set(f"Initializing data {pct:.1f}%")
                            detail_var.set(f"{detail} ({cur:,}/{tot:,})")
                            progress_var.set(15.0 + 35.0 * cur / max(1, tot))
                        elif phase == "precompute":
                            cur = msg[2] if len(msg) > 2 else 0
                            tot = msg[3] if len(msg) > 3 else 1
                            detail = msg[4] if len(msg) > 4 else "Precomputing risk kernels..."
                            status_var.set("Initializing risk model")
                            detail_var.set(detail)
                            progress_var.set(50.0 + 10.0 * cur / max(1, tot))
                        elif phase == "prepare_iterations":
                            detail = msg[4] if len(msg) > 4 else "Preparing Monte Carlo years..."
                            status_var.set("Preparing simulation")
                            detail_var.set(detail)
                            progress_var.set(62.0)
                        else:
                            ic = msg[3] if len(msg) > 3 else 0
                            it = msg[4] if len(msg) > 4 else n_iterations
                            detail = msg[5] if len(msg) > 5 else "Running Monte Carlo iterations..."
                            status_var.set(f"Iteration {ic}/{it}")
                            detail_var.set(detail)
                            progress_var.set(65.0 + 35.0 * ic / max(1, it))
                    elif k == "live":
                        s = msg[1]
                        cl = msg[2] if len(msg) > 2 else []
                        dl = msg[3] if len(msg) > 3 else []
                        ev_live = msg[4] if len(msg) > 4 else []
                        if isinstance(s, dict):
                            results["stats"] = s
                            results["cases_l"] = cl
                            results["deaths_l"] = dl
                            if ev_live:
                                # Keep the live history as a columnar bundle so
                                # the GUI never converts the full stream to
                                # dicts.  Only the visible subset is converted
                                # when the Cancer Events tab is rendered.
                                results["events"] = _concat_event_columns(
                                    [results["events"], ev_live]
                                )
                            _rebuild_modality_filter_values(s)
                            _rebuild_region_filter_values(results.get("exam_data") or [])
                            _rebuild_cancer_visibility_controls(s.get("cancer_breakdown", {}))
                            _rebuild_summary_modality_filter_values(s)

                            # Only update the currently visible tab during simulation.
                            # Dispatch by tab NAME, not by integer index:
                            # index-based dispatch breaks silently when tabs
                            # are reordered or folded into each other.
                            # Looking up the label via ``nb.tab(idx, "text")``
                            # is O(1) and is resilient to any tab shuffle.
                            visible_idx = current_tab_index.get()
                            try:
                                visible_name = str(nb.tab(visible_idx, "text"))
                            except Exception:
                                visible_name = ""

                            if visible_name == "Summary":
                                # Live summary skips exam_data/events to avoid
                                # blocking the worker thread; _render_summary
                                # honours the modality dropdown too.
                                _render_summary()
                            elif visible_name == "By Modality":
                                _upd_mod(s)
                            elif visible_name == "By Cancer Type":
                                # Refresh both cancer summary tables and the Sankey sub-tab in one shot.
                                _upd_cancer(s)
                                _upd_sankey(s)
                            elif visible_name == "By Body Region":
                                _upd_regions(results.get("exam_data") or [])
                            elif visible_name == "Patient Cohort":
                                _upd_charts(results.get("exam_data") or [], results.get("events") or [], s)
                            elif visible_name == "Cancer Events":
                                if ev_live:
                                    _upd_events(ev_live, append=True)
                            elif visible_name == "Research Questions":
                                _upd_research_questions(
                                    s,
                                    results.get("exam_data") or [],
                                    cancer_events=results.get("events") or [],
                                )
                            elif visible_name == "Risk Analysis":
                                advanced_panel.update_filter_values(results.get("exam_data") or [])
                        # Break after heavy visual update so tkinter renders it
                        rendered_heavy = True
                        break
                    elif k == "unknown_regions":
                        unknown_list = msg[1] if len(msg) > 1 else []
                        _upd_unknown_regions(unknown_list)
                    elif k == "missing_ct_doses":
                        _missing_summary = msg[1] if len(msg) > 1 else {}
                        _cont_event = msg[2] if len(msg) > 2 else None
                        _cont_result = msg[3] if len(msg) > 3 else [True]
                        if _missing_ct_suppress_state["suppress"]:
                            _cont_result[0] = True
                            if _cont_event:
                                _cont_event.set()
                        else:
                            _show_missing_ct_dialog(
                                root, _missing_summary, _cont_event, _cont_result,
                                suppress_state=_missing_ct_suppress_state,
                            )
                    elif k == "missing_columns":
                        # {file: [missing column names]} from the header
                        # pre-check; informational, non-blocking.
                        _mc = msg[1] if len(msg) > 1 else {}
                        if _mc:
                            results["missing_columns"] = _mc
                            _show_missing_columns_note(
                                root, _mc, startup_population=population)
                    elif k == "exam_data":
                        ed = msg[1] if len(msg) > 1 else []
                        results["exam_data"] = ed
                        _rebuild_modality_filter_values(results.get("stats") or {})
                        _rebuild_region_filter_values(ed)
                        advanced_panel.update_filter_values(ed)
                        # Defer heavy matplotlib rendering so the message queue
                        # can drain first.
                        root.after_idle(lambda e=ed: _upd_regions(e))
                        root.after_idle(lambda e=ed: _upd_charts(
                            e, results.get("events") or [], results.get("stats") or {}))
                    elif k == "file_cohort":
                        # Compact per-input-file count tables (see
                        # file_cohort.py).  Stored on results so the Patient
                        # Cohort and By Body Region tabs can draw every file
                        # of every year, not just the last processed year's
                        # exam_data.  The "done" handler also pulls this
                        # from stats["_file_cohort"], so this early event
                        # only matters for live updates before completion.
                        fc = msg[1] if len(msg) > 1 else None
                        if fc is not None:
                            results["file_cohort"] = fc
                    elif k == "excluded_cohorts":
                        # {exclusion key: CohortByFile} ledger of the exams
                        # dropped at source (participant_flow.py); the By
                        # Body Region tab lists them as excluded.  The
                        # "done" handler re-pulls the cross-year merge from
                        # stats["_excluded_cohorts"].
                        xc = msg[1] if len(msg) > 1 else None
                        if xc:
                            results["excluded_cohorts"] = xc
                    elif k == "done":
                        s, cl, dl, ev, ed = msg[1], msg[2], msg[3], msg[4], msg[5]
                        if not isinstance(ev, EventColumns):
                            # Normalise once at entry so results["events"] and
                            # every consumer below only ever see EventColumns.
                            try:
                                ev = EventColumns.from_events(list(ev or []))
                            except Exception:
                                ev = _EventColumnsForEmpty.empty()
                        results.update({"ready": True, "stats": s, "exam_data": ed,
                                        "events": ev, "cases_l": cl, "deaths_l": dl})
                        # Pull the merged per-file cohort profiles off the
                        # final stats dict (year_batch attaches the
                        # cross-year merge there).  Falls back to whatever
                        # a "file_cohort" progress event stored earlier.
                        _fc = (s or {}).get("_file_cohort") or results.get("file_cohort")
                        results["file_cohort"] = _fc
                        results["excluded_cohorts"] = (
                            (s or {}).get("_excluded_cohorts") or results.get("excluded_cohorts"))
                        # New simulation invalidates the cached sweep.
                        _rq_cache["sweep"] = None
                        _rq_cache["sweep_key"] = None
                        # The burden-vs-dose sub-tab reads the Region x Age
                        # sweep, which is cleared below (_rq_cache["sweep"]
                        # = None), so it needs no separate cache reset here;
                        # its next refresh will show the empty/message state
                        # until the Region x Age tab rebuilds the sweep.
                        progress_var.set(100)
                        # MC iterations are done, but the heavy matplotlib
                        # tab renders (Summary + visible tab, then deferred
                        # off-screen tabs) still have to run before the result
                        # is fully ready.  Show a rendering status here and
                        # flip to "Simulation complete!" only once the final
                        # render finishes (see the rendered_heavy branch below).
                        status_var.set("Rendering results...")
                        # "years" here means MC-simulated years (iterations),
                        # not the calendar year groups from the setup dialog.
                        # For an "average across years" batch, ``ed`` is only
                        # the last processed year and ``s['iterations']`` is
                        # the per-year MEAN -- batch_completion_totals sums
                        # each year's own count instead so this reports the
                        # true batch totals, not a partial/averaged figure.
                        _tot_ex, _tot_it = batch_completion_totals(s, ed)
                        detail_var.set(f"Processed {_tot_ex:,} examinations across {_tot_it:,} years")
                        # Flush tk so the status update appears before the heavy chart redraws begin.
                        try:
                            root.update_idletasks()
                        except Exception:
                            pass
                        # Rebuild shared filters and cancer-visibility controls up front.
                        _rebuild_modality_filter_values(s)
                        _rebuild_region_filter_values(ed)
                        _rebuild_cancer_visibility_controls(s.get("cancer_breakdown", {}))
                        _rebuild_summary_modality_filter_values(s)
                        visible_cancers = _visible_cancers_set()

                        # Render the visible tab synchronously and defer off-screen tabs to idle time.
                        visible_idx = current_tab_index.get()
                        try:
                            visible_name = str(nb.tab(visible_idx, "text"))
                        except Exception:
                            visible_name = ""

                        def _do_summary():
                            # Honours the Summary-tab modality dropdown and
                            # cancer-visibility checkboxes in one place.
                            _render_summary()

                        def _do_modality():
                            _upd_mod(s)

                        def _do_cancer():
                            _upd_cancer(s)
                            _upd_sankey(s, visible_cancers=visible_cancers)

                        def _do_regions():
                            _upd_regions(ed)

                        def _do_charts():
                            _upd_charts(ed, ev, s)

                        def _do_events():
                            _upd_events(ev)

                        def _do_advanced():
                            advanced_panel.update_filter_values(ed)
                            advanced_panel.refresh()

                        def _do_research_questions():
                            _upd_research_questions(s, ed, cancer_events=ev)

                        def _do_year_batches():
                            _upd_year_batches(s)

                        # Map tab labels to their updater(s).  Same labels
                        # the live-update branch uses just above.
                        _tab_updaters = {
                            "Summary":            _do_summary,
                            "By Modality":        _do_modality,
                            "By Cancer Type":     _do_cancer,
                            "By Body Region":     _do_regions,
                            "Patient Cohort":     _do_charts,
                            "Cancer Events":      _do_events,
                            "Research Questions": _do_research_questions,
                            "Risk Analysis":      _do_advanced,
                            # No-op unless the run produced per-year batches.
                            "Per-Year Comparison": _do_year_batches,
                        }
                        # Render Summary first, then the currently visible tab (no-op if they are the same).
                        try:
                            _do_summary()
                        except Exception:
                            import traceback
                            traceback.print_exc()
                        if visible_name and visible_name != "Summary":
                            _vis_fn = _tab_updaters.get(visible_name)
                            if _vis_fn is not None:
                                try:
                                    _vis_fn()
                                except Exception:
                                    import traceback
                                    traceback.print_exc()
                        # Schedule off-screen tab updates on the tk idle queue so the UI stays interactive.
                        _deferred_tabs = [
                            (name, fn) for name, fn in _tab_updaters.items()
                            if name not in ("Summary", visible_name)
                        ]
                        # Wrap each deferred updater so one failure doesn't cancel the rest of the queue.
                        def _safe_run(fn, label):
                            try:
                                fn()
                            except Exception:
                                import traceback
                                traceback.print_exc()
                                print(f"[deferred update] {label} failed", flush=True)
                        for _name, _fn in _deferred_tabs:
                            root.after(0, lambda f=_fn, n=_name: _safe_run(f, n))
                        rendered_heavy = True
                        break
                    elif k == "error":
                        results["ready"] = True
                        status_var.set("Error!")
                        detail_var.set(str(msg[1]))
                        messagebox.showerror("Error", str(msg[1]), parent=root)
                        break
                except Exception as _poll_err:
                    import traceback
                    traceback.print_exc()
                    print(f"[_poll_worker_queue] Error processing message {msg[0] if msg else '?'}: {_poll_err}", flush=True)
        except queue.Empty:
            pass
        if rendered_heavy:
            try:
                root.update_idletasks()
            except Exception:
                pass
        if not results["ready"]:
            root.after(100, _poll_worker_queue)
        elif results["ready"] and rendered_heavy and results.get("stats"):
            # Final render is done -- now the result is truly ready.
            status_var.set("Simulation complete!")
            try:
                root.update_idletasks()
            except Exception:
                pass
            # Show completion dialog after the final render
            root.after(50, lambda: messagebox.showinfo("Complete", "Simulation completed!", parent=root))

    def _upd_sankey(s, visible_cancers=None):
        try:
            if visible_cancers is None:
                visible_cancers = _visible_cancers_set()
            plot_sankey_diagram(sankey_fig, s, visible_cancers)
            sankey_canvas.draw_idle()
        except Exception:
            pass

    # Per-Year Comparison tab: created on first use and only for batch runs,
    # so single-run simulations never see an extra (empty) tab.
    _year_batch_tab = {"frame": None, "tree": None, "note": None}

    _YEAR_BATCH_COLUMNS = (
        ("year",      "Year group",        130, tk.W),
        ("exams",     "Examinations",      110, tk.E),
        ("iters",     "MC iterations",     100, tk.E),
        ("dose",      "Collective dose (person Sv/yr)", 190, tk.E),
        ("cases",     "Cancer cases",      110, tk.E),
        ("deaths",    "Cancer deaths",     110, tk.E),
        ("detriment", "Detriment",         110, tk.E),
        ("qaly",      "QALY lost",         110, tk.E),
    )


    def _upd_year_batches(s):
        """Fill the Per-Year Comparison table from ``stats['_year_batches']``.

        Cases / deaths / detriment / QALY are shown per MC year (divided by
        that year group's completed iteration count) so year groups that
        converged after a different number of iterations stay comparable.
        """
        batches = (s or {}).get("_year_batches") or {}
        if not batches:
            return
        if _year_batch_tab["frame"] is None:
            frame = ttk.Frame(nb)
            nb.add(frame, text="Per-Year Comparison")
            note = ttk.Label(frame, font=("Segoe UI", 9), justify=tk.LEFT,
                             wraplength=1150)
            note.pack(fill=tk.X, padx=8, pady=(8, 6))
            tree_frame = ttk.Frame(frame)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            tree = ttk.Treeview(tree_frame,
                                columns=[c[0] for c in _YEAR_BATCH_COLUMNS],
                                show="headings", height=16)
            for key, heading, width, anchor in _YEAR_BATCH_COLUMNS:
                tree.heading(key, text=heading)
                tree.column(key, width=width, anchor=anchor, stretch=False)
            vscroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                                    command=tree.yview)
            tree.configure(yscrollcommand=vscroll.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vscroll.pack(side=tk.RIGHT, fill=tk.Y)
            tree.tag_configure("average", font=("Segoe UI", 9, "bold"))
            _year_batch_tab.update({"frame": frame, "tree": tree, "note": note})

        tree = _year_batch_tab["tree"]
        tree.delete(*tree.get_children())
        for year in sorted(batches):
            tree.insert("", tk.END, values=_year_batch_row(year, batches[year] or {}))
        if s.get("_run_mode") == "per_year_average":
            tree.insert("", tk.END,
                        values=_year_batch_row("MEAN (all years)", s),
                        tags=("average",))
        _note = "To inspect one year in full, run a batch that contains only that year group."
        failed = s.get("_year_batch_failures") or {}
        if failed:
            _note += "\nEXCLUDED \u2014 {} year(s) failed: {}".format(
                len(failed), "; ".join(f"{y}: {msg}" for y, msg in failed.items()))
        _year_batch_tab["note"].configure(text=_note)

    def _worker():
        """Background worker: runs the simulation off the UI thread and posts
        throttled progress/snapshot messages to ``ui_q`` (progress capped
        at 20 Hz, heavy snapshots at 0.5 Hz, both slowing after 30 s)."""
        try:
            # GUI feedback throttling: cap progress updates at 20 Hz (4 Hz after 30s),
            # snapshots at 0.5 Hz, and use bounded pending-event buffers to avoid unbounded growth.
            _MIN_ITER_INTERVAL_S_FAST = 0.05   # 20 Hz cap for progress messages (first 30s)
            _MIN_ITER_INTERVAL_S_SLOW = 0.25   # 4 Hz cap for progress messages (after 30s)
            _LIVE_INTERVAL_S_FAST     = 2.0    # 0.5 Hz cap for heavy snapshot (first 30s)
            _LIVE_INTERVAL_S_SLOW     = 10.0   # 0.1 Hz cap for heavy snapshot (after 30s)
            _SLOW_UPDATE_THRESHOLD_S = 30.0    # Time before switching to slower updates
            live_window = 500
            last_live      = [0.0]
            last_iter_tick = [0.0]
            first_iter_time = [None]  # Track when first iteration starts
            # Bounded ring buffers for per-year case/death history.
            live_cases     = deque(maxlen=live_window)
            live_deaths    = deque(maxlen=live_window)
            # Event buffer accumulates EventColumns between live ticks and is cleared after each flush.
            pending_event_columns: list = []

            def _fast_snapshot(obj):
                """Shallow-ish snapshot of a dict-of-dicts stats tree.

                Rebuilds dict / list containers so the UI sees an immutable
                point-in-time view, but lets immutable scalars pass by
                reference (ints / floats / strings / tuples are never
                duplicated in CPython anyway).  Roughly 5-10× faster than
                ``copy.deepcopy`` on the ``stats`` payload because it skips
                the copy module's memo table and ``__deepcopy__`` dispatch.
                """
                if isinstance(obj, dict):
                    return {k: _fast_snapshot(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_fast_snapshot(v) for v in obj]
                return obj

            def _ui_progress_callback(phase, *args):
                """Map engine progress phases to ``ui_q`` messages for the poller:
                progress phases forward as-is; data phases forward their payload."""
                if phase in ("file_load", "init", "precompute", "prepare_iterations"):
                    ui_q.put(("progress", phase, *args))
                    return
                if phase == "exam_data":
                    ui_q.put(("exam_data", args[0] if args else []))
                    return
                if phase == "unknown_regions":
                    ui_q.put(("unknown_regions", args[0] if args else []))
                    return
                if phase == "missing_ct_doses":
                    ui_q.put(("missing_ct_doses", args[0] if args else {},
                               args[1] if len(args) > 1 else None,
                               args[2] if len(args) > 2 else [True]))
                    # The dialog (or the suppress state) always answers.
                    return True
                # ``missing_columns`` and ``header_check`` carry the same
                # {file: [columns]} payload from the header pre-check.
                if phase in ("missing_columns", "header_check"):
                    ui_q.put(("missing_columns", args[0] if args else {}))
                    return
                if phase == "file_cohort":
                    # CohortByFile object; the poller stores it on results.
                    ui_q.put(("file_cohort", args[0] if args else None))
                    return
                if phase == "excluded_cohorts":
                    # {exclusion key: CohortByFile}; the poller stores it on results.
                    ui_q.put(("excluded_cohorts", args[0] if args else None))
                    return
                if phase != "iteration":
                    # Any other phase is not a stats payload — never feed it
                    # to the live-stats path.
                    print(f"[gui] unhandled progress phase {phase!r}", file=sys.stderr)
                    return
                data = args[0] if args else None
                ic = args[1] if len(args) > 1 else 0
                it = args[2] if len(args) > 2 else n_iterations
                iter_cases = args[3] if len(args) > 3 else None
                iter_deaths = args[4] if len(args) > 4 else None
                detail = args[5] if len(args) > 5 else "Running Monte Carlo years..."
                iter_events = args[6] if len(args) > 6 else []

                # Track first iteration time for adaptive update frequency
                if first_iter_time[0] is None and isinstance(data, dict):
                    first_iter_time[0] = time.perf_counter()

                # Always accumulate cheap per-year history and pending events; flushing is throttled separately.
                if isinstance(data, dict):
                    if iter_cases is not None:
                        live_cases.append(iter_cases)
                    if iter_deaths is not None:
                        live_deaths.append(iter_deaths)
                    if isinstance(iter_events, EventColumns):
                        pending_event_columns.append(iter_events)

                # Determine update interval based on elapsed time
                now_perf = time.perf_counter()
                if first_iter_time[0] is not None and (now_perf - first_iter_time[0]) >= _SLOW_UPDATE_THRESHOLD_S:
                    min_iter_interval = _MIN_ITER_INTERVAL_S_SLOW
                    live_interval = _LIVE_INTERVAL_S_SLOW
                else:
                    min_iter_interval = _MIN_ITER_INTERVAL_S_FAST
                    live_interval = _LIVE_INTERVAL_S_FAST

                # Throttle progress updates; always emit the final iteration and convergence messages.
                is_final = (
                    (isinstance(ic, (int, float)) and isinstance(it, (int, float)) and ic >= it)
                    or ("Converged" in str(detail))
                )
                if is_final or (now_perf - last_iter_tick[0]) >= min_iter_interval:
                    last_iter_tick[0] = now_perf
                    ui_q.put(("progress", "iteration", data, ic, it, detail))

                # Throttle the heavy live channel independently; only it performs the snapshot copy.
                if isinstance(data, dict):
                    now_wall = time.time()
                    if now_wall - last_live[0] > live_interval:
                        last_live[0] = now_wall
                        # Concatenate pending EventColumns once per live tick; the UI converts only the visible subset to dicts.
                        if pending_event_columns:
                            events_to_send = _concat_event_columns(pending_event_columns)
                            pending_event_columns.clear()
                        else:
                            events_to_send = None
                        ui_q.put((
                            "live",
                            _fast_snapshot(data),
                            list(live_cases),
                            list(live_deaths),
                            events_to_send,
                        ))

            result = run_simulation(
                file_list, n_iterations, min_latency_params,
                population, on_progress=_ui_progress_callback, risk_model=risk_model,
                sim_config=sim_config,
            )
            if isinstance(result, tuple) and len(result) >= 4:
                s, cl, dl, ev = result[0], result[1], result[2], result[3]
                ed = result[4] if len(result) > 4 else []
            else:
                s, cl, dl, ev, ed = result, [], [], [], []

            # Build columnar data structures for efficient analysis
            if isinstance(s, dict) and ed:
                try:
                    exam_cols = ExamColumns.from_exam_data(ed)
                    s["_exam_columns"] = exam_cols
                except Exception:
                    pass  # Fall back to dict-based processing if column building fails
            if isinstance(s, dict) and ev:
                if isinstance(ev, EventColumns):
                    s["_event_columns"] = ev
                else:
                    try:
                        event_cols = EventColumns.from_events(ev)
                        s["_event_columns"] = event_cols
                    except Exception:
                        pass  # Fall back to dict-based processing if column building fails

            ui_q.put(("done", s, cl, dl, ev, ed))
        except Exception as e:
            import traceback
            traceback.print_exc()
            ui_q.put(("error", str(e)))

    root.after(100, _poll_worker_queue)
    set_cache_output_dir(output_dir)
    threading.Thread(target=_worker, daemon=True).start()
    root.mainloop()