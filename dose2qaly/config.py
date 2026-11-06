"""
Startup configuration window for Dose2QALY.

Opens a dialog that lets the user:
  1. Select one or more Excel input files (defaults to user home directory),
     grouped into named *year* buckets (e.g. 2021, 2022, 2023)
  2. Choose how the batch is run: average across years, or all years
     pooled into one combined population
  3. Configure all simulation parameters (defaults come from DEFAULTS below)
  4. Launch the simulation or cancel

The default output directory is ``~/dose2qaly_output`` (see
``constants.DEFAULT_OUTPUT_DIR``).  Results and all patient-data caches
(raw Excel cache, preprocess cache with per-patient cumulative doses and
frailty strata) are stored under this directory, so it must be a private,
access-controlled location.

Returns a config dict consumed by start_gui(), or None if cancelled.
"""

import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox, simpledialog


# Default Excel column list
# Single source of truth is excel_io.NEEDED_COLS_ORDERED.  Re-exported as a
# list because callers in this module mutate/copy it via list operations
# (e.g. the "Input File Columns" tab editor).
from .excel_io import NEEDED_COLS_ORDERED as _NEEDED_COLS_ORDERED, COLUMN_GUIDE
DEFAULT_NEEDED_COLS: list[str] = list(_NEEDED_COLS_ORDERED)


def _default_columns_text() -> str:
    """Default column list grouped by modality, with '#' headings.

    The headings are ignored by the parser; they only make the list
    readable.  A column used by several modalities is listed once, under
    the first group that uses it.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for group in dict.fromkeys(g for g, _, _ in COLUMN_GUIDE):
        new = [c for g, cols, _ in COLUMN_GUIDE if g == group
               for c in cols if c not in seen]
        if not new:
            continue
        seen.update(new)
        lines.extend([f"# {group}", *new, ""])
    return "\n".join(lines).rstrip() + "\n"

# Canonical DDREF values -- single source of truth is risk_models.py; do not
# re-declare 2.0 / 1.0 as bare literals elsewhere in this module.
from .risk_models import DDREF_SOLID_CANCERS, DDREF_LEUKAEMIA
from .constants import BACKGROUND_MSV_PER_YEAR, DEFAULT_OUTPUT_DIR


# Default values
DEFAULTS = {
    "n_iterations":       100_000,
    "population":         "Euro-American",
    "risk_model":         "REIC",
    "ddref_solid":        DDREF_SOLID_CANCERS,
    "ddref_leuk":         DDREF_LEUKAEMIA,
    "latency_leukaemia":  5.0,
    "latency_solid":      5.0,
    "use_golikov":        True,
    "use_mataac":         True,
    "min_iterations":     1000,
    "min_iterations_enabled": False,
    "convergence_pct":    1.0,
    # Precision-based stopping rule (in addition to the per-bin CI
    # half-width stability rule): once the stability window is reached,
    # also require the relative SE of the annual cancer count and the
    # annual QALYs (std(ddof=1)/sqrt(n)/mean over the per-iteration
    # lists) to fall at or below ``precision_target_rel_se``.  A None
    # rel SE (n<2 or mean<=0) means the criterion is not met while
    # enabled.  Off disables the precision half and keeps the
    # stability-only behaviour.
    "precision_stop_enabled":     True,
    "precision_target_rel_se":     0.01,
    "icrp152_strict":     False,
    # Run each finished year's post-run digest block on a background thread so
    # it overlaps with the NEXT year's file loading/preprocessing/MC run,
    # instead of blocking between years. Not exposed in GUI; set via config dict.
    "overlap_year_digests": True,
    # Single source of truth for the default multi-year run mode; the radio
    # buttons read their initial value from here so the two cannot drift.
    "run_mode":           "per_year_average",
    "background_radiation_msv_per_year": BACKGROUND_MSV_PER_YEAR,
    "output_dir":         DEFAULT_OUTPUT_DIR,
    # Partial calendar years (e.g. an archive whose last year covers only
    # ~71% of the year) are currently averaged in as full years, biasing
    # annual totals down.  ``partial_year_mode`` controls how a year whose
    # exam-date coverage falls below ``partial_year_min_coverage`` is
    # treated in the cross-year average:
    #   "scale"   - multiply that year's extensive quantities by 1/coverage
    #               before averaging (annualise);
    #   "exclude" - drop it from the annual averages and sweeps (kept in
    #               ``_year_batches`` detail, recorded in
    #               ``_partial_years_excluded``);
    #   "asis"    - leave it untouched (no annualisation or exclusion).
    "partial_year_mode":           "scale",
    "partial_year_min_coverage":    0.95,
}

INITIAL_DIR = os.path.expanduser("~")


def _scan_files_for_missing_cols(
    file_list: list[str], wanted: list[str]
) -> dict[str, list[str]]:
    """Read the header of each Excel file and report missing columns.

    Returns a dict ``{file_path: [missing_col, ...]}`` containing only the
    files that are missing one or more of ``wanted``.  Files whose headers
    cannot be read are reported with the special value ``["<could not read
    headers>"]`` so the caller can still warn the user.

    Delegates to :func:`excel_io.check_excel_headers` so the report is
    context-aware: modality-specific columns (e.g. CT organ doses) are only
    flagged when that modality is present in the file's first 500 rows.
    """
    from .excel_io import check_excel_headers

    return check_excel_headers(file_list, wanted, context_aware=True)


def show_config_dialog() -> dict | None:
    """Show the startup configuration window.

    Returns a dict with all parameters if the user clicks *Run*,
    or ``None`` if the window is closed / cancelled.
    """
    result = {"cancelled": True}

    root = tk.Tk()
    root.title("Dose2QALY — Simulation Setup")
    root.resizable(True, True)

    # Named fonts used throughout this dialog.  Scaling them (below) resizes
    # every widget at once, so the whole dialog fits a smaller screen without
    # the user resizing or scrolling.
    fonts = {
        "ui":         tkfont.Font(root=root, family="Segoe UI", size=9),
        "small":      tkfont.Font(root=root, family="Segoe UI", size=8),
        "small_bold": tkfont.Font(root=root, family="Segoe UI", size=8, weight="bold"),
        "bold":       tkfont.Font(root=root, family="Segoe UI", size=9, weight="bold"),
        "section":    tkfont.Font(root=root, family="Segoe UI", size=10, weight="bold"),
        "run":        tkfont.Font(root=root, family="Segoe UI", size=11, weight="bold"),
        "header":     tkfont.Font(root=root, family="Segoe UI", size=14, weight="bold"),
        "mono":       tkfont.Font(root=root, family="Consolas", size=9),
        "mono_text":  tkfont.Font(root=root, family="Consolas", size=10),
    }
    tk_named = [tkfont.nametofont(n, root=root) for n in ("TkDefaultFont", "TkTextFont", "TkHeadingFont")]
    _base_sizes = [(f, f.cget("size")) for f in [*fonts.values(), *tk_named]]

    def _scale_fonts(factor: float) -> None:
        for f, size in _base_sizes:   # Tk named fonts may be negative (= pixels); keep the sign
            sign = -1 if size < 0 else 1
            f.configure(size=sign * max(7, int(round(abs(size) * factor))))

    # Provisional geometry so the window chrome can be measured; the final
    # size is set after every widget exists (see the fit block below).
    root.update_idletasks()
    scr_w, scr_h = root.winfo_screenwidth(), root.winfo_screenheight()
    avail_w, avail_h = scr_w - 40, scr_h - 90          # window chrome + taskbar margin
    root.geometry(f"{min(1150, avail_w)}x{min(760, avail_h)}+0+0")
    root.minsize(min(720, avail_w), min(480, avail_h))

    # Wrapping labels re-flow to their actual width on <Configure>.  ``width=1``
    # stops the label's own requested width from following its wraplength, so
    # the surrounding widgets -- not the help text -- decide how wide a column
    # is (the labels are packed/gridded to fill their cell anyway).
    def _autowrap(label, pad=6, initial=440):
        label.configure(wraplength=initial, width=1)
        label.bind("<Configure>", lambda e: label.configure(wraplength=max(120, e.width - pad)))

    # Style
    style = ttk.Style()
    style.configure("Header.TLabel", font=fonts["header"])
    style.configure("Section.TLabelframe.Label", font=fonts["section"])
    style.configure("Run.TButton", font=fonts["run"])

    # Title
    ttk.Label(root, text="Simulation Setup",
              style="Header.TLabel").pack(pady=(8, 4))

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Bottom action row (reserved first)
    # pack() serves earlier-packed widgets first, so this container is packed
    # against the bottom edge *before* the notebook asks for space.  Otherwise
    # a tall Simulation Setup tab pushes "Run Simulation" off-screen on short
    # displays instead of shrinking the notebook.
    action_area = ttk.Frame(main_frame)
    action_area.pack(side=tk.BOTTOM, fill=tk.X)

    # Notebook (tabbed layout)
    # Tab 1: Simulation Setup (files, parameters, output dir)
    # Tab 2: Input File Columns    (editable list pushed down to pd.read_excel)
    notebook = ttk.Notebook(main_frame)
    notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 8))

    tab_setup_outer = ttk.Frame(notebook)
    tab_columns = ttk.Frame(notebook, padding=8)
    notebook.add(tab_setup_outer, text="  Simulation Setup  ")
    notebook.add(tab_columns, text="  Input File Columns  ")

    # The setup tab is a two-column grid (files on the left, parameters on the
    # right); named fonts scale to fit the screen, so the canvas below is only
    # a fallback for tiny displays.  Everything still parents to ``tab_setup``.
    setup_canvas = tk.Canvas(tab_setup_outer, borderwidth=0, highlightthickness=0)
    setup_scroll = ttk.Scrollbar(tab_setup_outer, orient=tk.VERTICAL,
                                 command=setup_canvas.yview)
    setup_canvas.configure(yscrollcommand=setup_scroll.set)
    setup_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    tab_setup = ttk.Frame(setup_canvas, padding=8)
    setup_window = setup_canvas.create_window((0, 0), window=tab_setup, anchor="nw")

    # Two-column layout: left holds files/year/output/mode, right the parameters.
    tab_setup.columnconfigure(0, weight=1)   # left: files etc. — gets the extra width
    tab_setup.columnconfigure(1, weight=0)
    tab_setup.rowconfigure(0, weight=1)
    left_col = ttk.Frame(tab_setup);  left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    right_col = ttk.Frame(tab_setup); right_col.grid(row=0, column=1, sticky="nsew")

    def _sync_setup_scrollregion(_event=None):
        setup_canvas.configure(scrollregion=setup_canvas.bbox("all"))
        # Keep the inner frame as wide as the canvas AND at least as tall, so
        # the file list grows on big windows instead of collapsing.
        setup_canvas.itemconfigure(
            setup_window,
            width=setup_canvas.winfo_width(),
            height=max(setup_canvas.winfo_height(), tab_setup.winfo_reqheight()),
        )
        # The scrollbar is only a fallback for tiny screens: show it only
        # when the setup content is actually taller than the canvas.
        if tab_setup.winfo_reqheight() > setup_canvas.winfo_height():
            if not setup_scroll.winfo_ismapped():
                setup_scroll.pack(side=tk.RIGHT, fill=tk.Y, before=setup_canvas)
        else:
            if setup_scroll.winfo_ismapped():
                setup_scroll.pack_forget()

    tab_setup.bind("<Configure>", _sync_setup_scrollregion)
    setup_canvas.bind("<Configure>", _sync_setup_scrollregion)

    def _on_setup_mousewheel(event):
        setup_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    # Bound only while the pointer is over the setup tab, so the wheel keeps
    # working normally in the Input File Columns text box.
    setup_canvas.bind("<Enter>",
                      lambda e: setup_canvas.bind_all("<MouseWheel>", _on_setup_mousewheel))
    setup_canvas.bind("<Leave>",
                      lambda e: setup_canvas.unbind_all("<MouseWheel>"))

    # 1. FILE SELECTION (GROUPED BY YEAR)
    # Each "year" is a user-named bucket of input files.  ``year_groups`` is
    # the source of truth; the listbox below always shows exactly one bucket
    # and ``file_list_var`` mirrors it.  Files are moved between the two by
    # ``_save_current_year_files`` / ``_load_year_into_listbox`` on every
    # bucket switch, so nothing leaks between years.
    year_groups: dict[str, list[str]] = {"Year 1": []}
    file_list_var: list[str] = []
    current_year_var = tk.StringVar(value="Year 1")
    # Tk gives no "old value" on <<ComboboxSelected>>, so the year the listbox
    # is currently showing is tracked separately.
    _shown_year = ["Year 1"]
    run_mode_var = tk.StringVar(value=DEFAULTS["run_mode"])

    def _save_current_year_files() -> None:
        """Copy the listbox contents back into the bucket it belongs to."""
        year_groups[_shown_year[0]] = list(file_list_var)

    def _load_year_into_listbox(year: str) -> None:
        """Repopulate the listbox from ``year_groups[year]``."""
        _shown_year[0] = year
        file_listbox.delete(0, tk.END)
        file_list_var.clear()
        for p in year_groups.get(year, []):
            file_list_var.append(p)
            file_listbox.insert(tk.END, p)
        file_frame.configure(text=f"  Input files for {year} (.xlsx)  ")

    def _refresh_year_widgets() -> None:
        """Sync both year comboboxes with the current bucket labels."""
        labels = sorted(year_groups.keys())
        year_combo.configure(values=labels)

    def _on_year_selected(_event=None) -> None:
        new_year = current_year_var.get()
        if new_year == _shown_year[0]:
            return
        _save_current_year_files()
        _load_year_into_listbox(new_year)

    def _add_year() -> None:
        label = simpledialog.askstring(
            "New Year", "Label for this year (e.g. 2023):", parent=root)
        if label is None:
            return
        label = label.strip()
        if not label:
            messagebox.showwarning("Invalid label",
                                   "The year label cannot be empty.",
                                   parent=root)
            return
        if label in year_groups:
            messagebox.showwarning("Duplicate year",
                                   f"A year group named {label!r} already exists.",
                                   parent=root)
            return
        _save_current_year_files()
        year_groups[label] = []
        _refresh_year_widgets()
        current_year_var.set(label)
        _load_year_into_listbox(label)

    def _remove_year() -> None:
        if len(year_groups) <= 1:
            messagebox.showwarning(
                "Cannot remove",
                "At least one year group must remain.  Use 'Clear All' to "
                "empty this year's file list instead.",
                parent=root,
            )
            return
        victim = _shown_year[0]
        if not messagebox.askyesno(
            "Remove year",
            f"Remove year group {victim!r} and its "
            f"{len(year_groups.get(victim, []))} file(s) from the run?",
            parent=root,
        ):
            return
        year_groups.pop(victim, None)
        _refresh_year_widgets()
        next_year = sorted(year_groups.keys())[0]
        current_year_var.set(next_year)
        _load_year_into_listbox(next_year)

    def _all_files() -> list[str]:
        """Flattened, de-duplicated union of every year's files.

        Used by the validations and header pre-scans, which sanity-check the
        union of everything that will be loaded regardless of grouping.
        """
        _save_current_year_files()
        return list(dict.fromkeys(
            f for files in year_groups.values() for f in files
        ))

    def _auto_sort_by_year() -> None:
        """Replace all year groups with one bucket per detected calendar year.

        Scans every input file's "Date Performed" column (regardless of which
        manual bucket it currently sits in) and re-groups the files so that
        each resulting bucket is named after the single calendar year its
        exams fall in.  Any file whose exams span more than one calendar
        year -- or whose date column is missing/unreadable -- aborts the
        whole operation with an error dialog and leaves the existing year
        groups untouched.
        """
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .excel_io import get_file_calendar_years

        files = _all_files()
        if not files:
            messagebox.showinfo(
                "Auto-Sort by Year",
                "Add input files first, then use Auto-Sort by Year.",
                parent=root,
            )
            return

        if not messagebox.askyesno(
            "Auto-Sort by Year",
            f"This scans the \"Date Performed\" column of all {len(files)} "
            "input file(s) and replaces the current year groups with one "
            "bucket per calendar year.  Each file must contain exams from "
            "exactly one calendar year.\n\nContinue?",
            parent=root,
        ):
            return

        top = tk.Toplevel(root)
        top.title("Scanning files...")
        top.transient(root)
        top.grab_set()
        top.resizable(False, False)
        lbl = ttk.Label(top, text="Scanning...", font=fonts["ui"],
                        justify=tk.CENTER, padding=20)
        lbl.pack()
        pbar = ttk.Progressbar(top, orient=tk.HORIZONTAL, length=350,
                               mode="determinate", maximum=len(files))
        pbar.pack(pady=(0, 16), padx=20)
        top.update_idletasks()
        tx = root.winfo_x() + (root.winfo_width() - top.winfo_width()) // 2
        ty = root.winfo_y() + (root.winfo_height() - top.winfo_height()) // 2
        tx = max(0, min(tx, scr_w - top.winfo_width()))
        ty = max(0, min(ty, scr_h - top.winfo_height()))
        top.geometry(f"+{tx}+{ty}")

        file_year: dict[str, int] = {}
        multi_year_files: dict[str, list[int]] = {}
        unreadable_files: list[str] = []

        # Each file only needs its "Date Performed" column read (see
        # get_file_calendar_years' column push-down), and that per-file read
        # is I/O-bound / releases the GIL in the calamine/zip parsing paths,
        # so scanning files concurrently gives a near-linear speed-up over
        # doing them one at a time.  GUI updates still happen on the main thread as each
        # future completes.
        n_workers = min(len(files), max(1, mp.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(get_file_calendar_years, fp): fp for fp in files}
            for i, fut in enumerate(as_completed(futures), start=1):
                fp = futures[fut]
                lbl.configure(text=f"Scanning file {i}/{len(files)}:\n{os.path.basename(fp)}")
                pbar["value"] = i
                top.update()
                try:
                    years = fut.result()
                except Exception:
                    years = None
                if not years:
                    unreadable_files.append(fp)
                elif len(years) > 1:
                    multi_year_files[fp] = years
                else:
                    file_year[fp] = years[0]

        top.destroy()

        if unreadable_files:
            messagebox.showerror(
                "Auto-Sort by Year",
                "Could not determine a calendar year for the following "
                "file(s) -- their \"Date Performed\" column is missing or "
                "unreadable.  Auto-sort was cancelled; no year groups were "
                "changed:\n\n"
                + "\n".join(os.path.basename(f) for f in unreadable_files),
                parent=root,
            )
            return

        if multi_year_files:
            details = "\n".join(
                f"{os.path.basename(f)}: {', '.join(str(y) for y in ys)}"
                for f, ys in multi_year_files.items()
            )
            messagebox.showerror(
                "Auto-Sort by Year",
                "The following file(s) contain exams from more than one "
                "calendar year, so they cannot be auto-sorted into a single "
                "year group.  Auto-sort was cancelled; no year groups were "
                "changed:\n\n" + details,
                parent=root,
            )
            return

        # Every file maps to exactly one calendar year -- rebuild the buckets.
        new_groups: dict[str, list[str]] = {}
        for fp, year in file_year.items():
            new_groups.setdefault(str(year), []).append(fp)

        year_groups.clear()
        year_groups.update(new_groups)
        _refresh_year_widgets()
        first_year = sorted(year_groups.keys())[0]
        current_year_var.set(first_year)
        _load_year_into_listbox(first_year)

        messagebox.showinfo(
            "Auto-Sort by Year",
            f"Sorted {len(files)} file(s) into {len(year_groups)} year "
            "group(s): " + ", ".join(sorted(year_groups.keys())),
            parent=root,
        )

    year_frame = ttk.LabelFrame(left_col, text="  Year Groups  ",
                                style="Section.TLabelframe", padding=8)
    year_frame.pack(fill=tk.X, pady=(0, 8))

    year_row = ttk.Frame(year_frame)
    year_row.pack(fill=tk.X)
    ttk.Label(year_row, text="Editing year:", font=fonts["ui"]).pack(
        side=tk.LEFT, padx=(0, 6))
    year_combo = ttk.Combobox(year_row, textvariable=current_year_var,
                              values=["Year 1"], state="readonly", width=14,
                              font=fonts["ui"])
    year_combo.pack(side=tk.LEFT, padx=(0, 8))
    year_combo.bind("<<ComboboxSelected>>", _on_year_selected)
    ttk.Button(year_row, text="Add Year…", command=_add_year).pack(
        side=tk.LEFT, padx=(0, 6))
    ttk.Button(year_row, text="Remove Year", command=_remove_year).pack(
        side=tk.LEFT, padx=(0, 6))
    ttk.Button(year_row, text="Auto-Sort by Year", command=_auto_sort_by_year).pack(
        side=tk.LEFT)

    _year_help = ttk.Label(
        year_frame,
        text=("Add one group per calendar year and put that year's file(s) in "
              "it.  The file list below always shows the selected year only.  "
              "\"Auto-Sort by Year\" replaces all groups automatically, based "
              "on each file's \"Date Performed\" column -- it fails if any "
              "file mixes more than one calendar year."),
        font=fonts["small"], foreground="#666666", justify=tk.LEFT,
    )
    _year_help.pack(fill=tk.X, anchor=tk.W, pady=(4, 0))
    _autowrap(_year_help, initial=440)

    file_frame = ttk.LabelFrame(left_col, text="  Input files for Year 1 (.xlsx)  ",
                                style="Section.TLabelframe", padding=8)
    file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

    list_frame = ttk.Frame(file_frame)
    list_frame.pack(fill=tk.BOTH, expand=True)

    file_listbox = tk.Listbox(list_frame, height=4, font=fonts["mono"],
                              selectmode=tk.EXTENDED)
    file_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                command=file_listbox.yview)
    file_listbox.configure(yscrollcommand=file_scroll.set)
    file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    file_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    btn_frame = ttk.Frame(file_frame)
    btn_frame.pack(fill=tk.X, pady=(6, 0))

    def _add_files():
        paths = filedialog.askopenfilenames(
            title="Select Data files",
            initialdir=INITIAL_DIR,
            filetypes=[
                ("Data files", "*.xlsx *.xls"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ],
        )
        for p in paths:
            if p and p not in file_list_var:
                file_list_var.append(p)
                file_listbox.insert(tk.END, p)

    def _remove_selected():
        sel = list(file_listbox.curselection())
        for idx in reversed(sel):
            file_listbox.delete(idx)
            del file_list_var[idx]

    def _clear_all():
        file_listbox.delete(0, tk.END)
        file_list_var.clear()

    ttk.Button(btn_frame, text="Add Files…", command=_add_files).pack(
        side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_frame, text="Remove Selected", command=_remove_selected).pack(
        side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_frame, text="Clear All", command=_clear_all).pack(
        side=tk.LEFT, padx=(0, 6))

    # 3. OUTPUT DIRECTORY
    # Packed directly under the input files in the left column, so the
    # patient-data warning is on screen without scrolling.
    out_frame = ttk.LabelFrame(left_col, text="  Output Directory  ",
                               style="Section.TLabelframe", padding=8)
    out_frame.pack(fill=tk.X, pady=(0, 8))

    out_row = ttk.Frame(out_frame)
    out_row.pack(fill=tk.X, pady=(0, 6))

    out_var = tk.StringVar(value=DEFAULTS["output_dir"])
    out_entry = ttk.Entry(out_row, textvariable=out_var, width=30,
                          font=fonts["mono"])
    out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

    def _browse_output():
        d = filedialog.askdirectory(initialdir=out_var.get() or INITIAL_DIR,
                                    title="Select output directory")
        if d:
            out_var.set(d)

    ttk.Button(out_row, text="Browse…", command=_browse_output).pack(
        side=tk.LEFT)

    out_warn = tk.Label(
        out_frame,
        text=(
            "\u26a0  WARNING: Patient data (raw Excel copies, preprocessed "
            "exam records with per-patient cumulative doses) is cached in "
            "this directory. Ensure input data is pseudonymised. Choose a "
            "secure, access-controlled location \u2014 not a shared or "
            "network drive. The cache only speeds up re-runs; deleting it "
            "does not change results."
        ),
        font=fonts["small_bold"],
        foreground="#cc0000",
        justify=tk.LEFT,
    )
    out_warn.pack(fill=tk.X, anchor=tk.W)
    _autowrap(out_warn, initial=440)

    # 1b. RUN MODE
    mode_frame = ttk.LabelFrame(left_col, text="  Run Mode  ",
                                style="Section.TLabelframe", padding=8)
    mode_frame.pack(fill=tk.X, pady=(0, 8))

    ttk.Radiobutton(
        mode_frame,
        text="Average across years\n(each year runs on its own; results are "
             "the mean annual burden)",
        variable=run_mode_var, value="per_year_average",
    ).pack(anchor=tk.W)

    ttk.Radiobutton(
        mode_frame,
        text="Combine all years into one run\n(every file pooled into a "
             "single population)",
        variable=run_mode_var, value="combined",
    ).pack(anchor=tk.W)

    # 2. SIMULATION PARAMETERS
    param_frame = ttk.LabelFrame(right_col, text="  Simulation Parameters  ",
                                 style="Section.TLabelframe", padding=8)
    param_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

    # Helper to build a labelled row
    row_idx = [0]

    def _add_row(label_text, widget_factory):
        r = row_idx[0]
        ttk.Label(param_frame, text=label_text, font=fonts["ui"]).grid(
            row=r, column=0, sticky=tk.W, padx=(0, 12), pady=2)
        widget = widget_factory(param_frame)
        widget.grid(row=r, column=1, sticky=tk.W, pady=2)
        row_idx[0] += 1
        return widget

    def _add_separator():
        r = row_idx[0]
        ttk.Separator(param_frame, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky=tk.EW, pady=3)
        row_idx[0] += 1

    # Max iterations
    iter_var = tk.IntVar(value=DEFAULTS["n_iterations"])
    _add_row("Max number of iterations:", lambda p: ttk.Spinbox(
        p, from_=100, to=10_000_000, textvariable=iter_var, width=14,
        font=fonts["ui"]))

    # Population
    pop_var = tk.StringVar(value=DEFAULTS["population"])
    _add_row("Default population (if not in input data):", lambda p: ttk.Combobox(
        p, textvariable=pop_var, values=["Euro-American", "Asian"],
        state="readonly", width=18, font=fonts["ui"]))

    # Risk model
    # Only REIC is implemented.  LAR (BEIR VII) is shown dimmed as a
    # placeholder so the intended second option is visible but cannot be
    # chosen; a Combobox cannot grey out a single entry, hence radio buttons.
    model_var = tk.StringVar(value=DEFAULTS["risk_model"])

    def _risk_model_widget_factory(p):
        frame = ttk.Frame(p)
        ttk.Radiobutton(frame, text="REIC (ICRP model)",
                        variable=model_var, value="REIC").pack(side=tk.LEFT)
        ttk.Radiobutton(frame, text="LAR (BEIR VII) \u2014 not yet available",
                        variable=model_var, value="LAR",
                        state="disabled").pack(side=tk.LEFT, padx=(12, 0))
        return frame

    _add_row("Risk model:", _risk_model_widget_factory)

    _add_separator()

    # DDREF
    ddref_s_var = tk.DoubleVar(value=DEFAULTS["ddref_solid"])
    _add_row(f"DDREF — solid cancers [ICRP 152 ref: {DDREF_SOLID_CANCERS:g}]:", lambda p: ttk.Spinbox(
        p, from_=1.0, to=5.0, increment=0.1, textvariable=ddref_s_var,
        width=10, font=fonts["ui"]))

    ddref_l_var = tk.DoubleVar(value=DEFAULTS["ddref_leuk"])
    _add_row(f"DDREF — leukaemia [ICRP 152 ref: {DDREF_LEUKAEMIA:g}]:", lambda p: ttk.Spinbox(
        p, from_=1.0, to=5.0, increment=0.1, textvariable=ddref_l_var,
        width=10, font=fonts["ui"]))

    _add_separator()

    # Minimum latency
    # ICRP 152 reference detriment uses 5 y for ALL cancer sites (p.61).
    # Alternatives from §4.2 para 97 / Table 4.1 / Fig. 4.7:
    #   solid cancers: 5–10 y      leukaemia: 2–5 y
    # are available here for sensitivity analysis.
    lat_leuk_var = tk.DoubleVar(value=DEFAULTS["latency_leukaemia"])
    _add_row("Min latency — leukaemia (yr) [ICRP 152 ref: 5]:", lambda p: ttk.Spinbox(
        p, from_=0.0, to=20.0, increment=0.5, textvariable=lat_leuk_var,
        width=10, font=fonts["ui"]))

    lat_solid_var = tk.DoubleVar(value=DEFAULTS["latency_solid"])
    _add_row("Min latency — solid cancers (yr) [ICRP 152 ref: 5]:", lambda p: ttk.Spinbox(
        p, from_=0.0, to=20.0, increment=0.5, textvariable=lat_solid_var,
        width=10, font=fonts["ui"]))

    _add_separator()

    # Toggles
    golikov_var = tk.BooleanVar(value=DEFAULTS["use_golikov"])
    golikov_cb = _add_row("Weight adjustment (Golikov and Druzhinina):", lambda p: ttk.Checkbutton(
        p, variable=golikov_var, text="Enabled"))

    mataac_var = tk.BooleanVar(value=DEFAULTS["use_mataac"])
    mataac_cb = _add_row("Competing mortality correction (Mataac et al.):", lambda p: ttk.Checkbutton(
        p, variable=mataac_var, text="Enabled"))

    # Saved states so we can restore when ICRP 152 is unchecked
    _saved_golikov = [DEFAULTS["use_golikov"]]
    _saved_mataac  = [DEFAULTS["use_mataac"]]

    def _toggle_icrp152():
        if icrp152_var.get():
            # Save current states, then force off and disable
            _saved_golikov[0] = golikov_var.get()
            _saved_mataac[0]  = mataac_var.get()
            golikov_var.set(False)
            mataac_var.set(False)
            golikov_cb.configure(state="disabled")
            mataac_cb.configure(state="disabled")
        else:
            # Restore saved states and re-enable
            golikov_var.set(_saved_golikov[0])
            mataac_var.set(_saved_mataac[0])
            golikov_cb.configure(state="normal")
            mataac_cb.configure(state="normal")

    icrp152_var = tk.BooleanVar(value=DEFAULTS["icrp152_strict"])
    _add_row("ICRP 152 reference assumptions:", lambda p: ttk.Checkbutton(
        p, variable=icrp152_var, command=_toggle_icrp152,
        text="Enabled (turns off both adjustments above)"))

    _add_separator()

    # Minimum iterations before convergence stop
    min_iter_enabled_var = tk.BooleanVar(value=DEFAULTS["min_iterations_enabled"])
    min_iter_var = tk.IntVar(value=DEFAULTS["min_iterations"])
    min_iter_spin = None

    def _toggle_min_iter_state():
        if min_iter_spin is not None:
            min_iter_spin.configure(state="normal" if min_iter_enabled_var.get() else "disabled")

    def _min_iter_widget_factory(p):
        nonlocal min_iter_spin
        frame = ttk.Frame(p)
        ttk.Checkbutton(
            frame,
            variable=min_iter_enabled_var,
            command=_toggle_min_iter_state,
            text="Custom minimum",
        ).pack(side=tk.LEFT)
        min_iter_spin = ttk.Spinbox(
            frame,
            from_=1,
            to=10_000_000,
            textvariable=min_iter_var,
            width=14,
            font=fonts["ui"],
            state="disabled",
        )
        min_iter_spin.pack(side=tk.LEFT, padx=(8, 6))
        _toggle_min_iter_state()
        return frame

    _add_row("Minimum iterations before stop:", _min_iter_widget_factory)

    _add_separator()

    # Convergence threshold
    conv_var = tk.StringVar(value=f"{DEFAULTS['convergence_pct']:.1f}%")
    # Labels describe the trade-off (tighter CI half-width costs more
    # iterations) rather than endorsing a value; the default is marked as the
    # default, not as advice.
    conv_options = [
        "0.1%  —  tightest CI, most iterations",
        "0.5%  —  tighter CI, more iterations",
        "1.0%  —  default",
        "2.0%  —  wider CI, fewer iterations",
        "5.0%  —  wide CI, few iterations",
        "10.0%  —  widest CI, fewest iterations",
    ]
    _add_row("Convergence threshold:", lambda p: ttk.Combobox(
        p, textvariable=conv_var, values=conv_options,
        width=34, font=fonts["ui"]))

    _add_separator()

    # Precision-based stopping rule
    # In addition to the per-bin CI half-width stability rule, stop once
    # the relative SE of the annual cancer count and annual QALYs falls at
    # or below the target.  A checkbox + a percent field, mirroring the
    # min-iterations row's widget style.
    precision_stop_enabled_var = tk.BooleanVar(value=DEFAULTS["precision_stop_enabled"])
    precision_target_var = tk.DoubleVar(value=DEFAULTS["precision_target_rel_se"] * 100.0)
    precision_target_spin = None

    def _toggle_precision_target_state():
        if precision_target_spin is not None:
            precision_target_spin.configure(
                state="normal" if precision_stop_enabled_var.get() else "disabled")

    def _precision_widget_factory(p):
        nonlocal precision_target_spin
        frame = ttk.Frame(p)
        ttk.Checkbutton(
            frame,
            variable=precision_stop_enabled_var,
            command=_toggle_precision_target_state,
            text="Enabled",
        ).pack(side=tk.LEFT)
        precision_target_spin = ttk.Spinbox(
            frame,
            from_=0.1, to=100.0, increment=0.1,
            textvariable=precision_target_var,
            width=10,
            font=fonts["ui"],
        )
        precision_target_spin.pack(side=tk.LEFT, padx=(8, 6))
        ttk.Label(frame, text="%").pack(side=tk.LEFT)
        _toggle_precision_target_state()
        return frame

    _add_row("Precision stop (rel. SE of cancers & QALYs):", _precision_widget_factory)

    _add_separator()

    # Background radiation dose
    bg_rad_var = tk.DoubleVar(value=DEFAULTS["background_radiation_msv_per_year"])
    _add_row("Natural background radiation (mSv/year):", lambda p: ttk.Spinbox(
        p, from_=0.1, to=20.0, increment=0.1, textvariable=bg_rad_var,
        width=10, font=fonts["ui"]))
    # Add explanatory label
    _bg_note = ttk.Label(param_frame, text="Used only for the 'equivalent time of natural background radiation' line. Default 2.4 = UNSCEAR worldwide average from all natural sources incl. radon.",
              font=fonts["small"], foreground="#666666", justify=tk.LEFT)
    _bg_note.grid(row=row_idx[0], column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
    _autowrap(_bg_note, initial=440)
    row_idx[0] += 1

    # 3b. EXCEL COLUMNS TAB
    # Editable column list pushed to pd.read_excel(usecols=...).  Restricting to
    # needed columns. Missing columns are silently skipped.
    cols_intro = ttk.Label(
        tab_columns,
        text=(
            "Only the columns listed below are read from each Excel file. "
            "The table shows what each column is used for, by modality; "
            "the list under it is what is actually loaded (one name per "
            "line, '#' lines are headings). If your files use other header "
            "names, edit the list. A column missing from a file is treated "
            "as empty and you are warned before the run starts."
        ),
        font=fonts["ui"],
        foreground="#444444",
        justify=tk.LEFT,
    )
    cols_intro.pack(fill=tk.X, anchor=tk.W, pady=(0, 6))
    _autowrap(cols_intro, initial=700)

    cols_paned = ttk.PanedWindow(tab_columns, orient=tk.VERTICAL)
    cols_paned.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

    # Reference table: one parent row per modality group, one child row per
    # column (or per set of interchangeable column names).
    guide_frame = ttk.LabelFrame(
        cols_paned, text="  What each column is used for  ",
        style="Section.TLabelframe", padding=8,
    )
    cols_paned.add(guide_frame, weight=3)

    guide_tree = ttk.Treeview(
        guide_frame, columns=("used_for",), show="tree headings",
        selectmode="none",
    )
    guide_tree.heading("#0", text="Column", anchor=tk.W)
    guide_tree.heading("used_for", text="Used for", anchor=tk.W)
    guide_tree.column("#0", width=260, minwidth=160, stretch=False)
    guide_tree.column("used_for", width=420, minwidth=200, stretch=True)
    guide_scroll = ttk.Scrollbar(
        guide_frame, orient=tk.VERTICAL, command=guide_tree.yview,
    )
    guide_tree.configure(yscrollcommand=guide_scroll.set)
    guide_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    guide_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    guide_tree.tag_configure("group", font=fonts["bold"])

    for group in dict.fromkeys(g for g, _, _ in COLUMN_GUIDE):
        parent = guide_tree.insert(
            "", tk.END, text=group, open=True, tags=("group",),
        )
        for g, cols, used_for in COLUMN_GUIDE:
            if g != group:
                continue
            if len(cols) > 6:
                label = f"{cols[0]} \u2026 {cols[-1]}"
            else:
                label = " / ".join(cols)
            guide_tree.insert(parent, tk.END, text=label, values=(used_for,))

    cols_frame = ttk.LabelFrame(
        cols_paned, text="  Columns to load from each Excel file  ",
        style="Section.TLabelframe", padding=8,
    )
    cols_paned.add(cols_frame, weight=2)

    cols_text_frame = ttk.Frame(cols_frame)
    cols_text_frame.pack(fill=tk.BOTH, expand=True)

    cols_text = tk.Text(
        cols_text_frame, height=8, width=60,
        font=fonts["mono_text"], undo=True, wrap=tk.NONE,
    )
    cols_scroll = ttk.Scrollbar(
        cols_text_frame, orient=tk.VERTICAL, command=cols_text.yview,
    )
    cols_text.configure(yscrollcommand=cols_scroll.set)
    cols_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    cols_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _populate_columns(text: str) -> None:
        cols_text.delete("1.0", tk.END)
        cols_text.insert("1.0", text)

    _populate_columns(_default_columns_text())

    def _parse_column_text() -> list[str]:
        raw = cols_text.get("1.0", tk.END)
        seen: set[str] = set()
        out: list[str] = []
        for line in raw.splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    cols_btn_frame = ttk.Frame(cols_frame)
    cols_btn_frame.pack(fill=tk.X, pady=(6, 0))

    ttk.Button(
        cols_btn_frame, text="Reset to Defaults",
        command=lambda: _populate_columns(_default_columns_text()),
    ).pack(side=tk.LEFT, padx=(0, 6))

    def _check_columns_against_files() -> None:
        """Show which of the listed columns are present/missing in each file."""
        files_to_scan = _all_files()
        if not files_to_scan:
            messagebox.showinfo(
                "No files",
                "Add at least one Excel file on the Simulation Setup tab first.",
                parent=root,
            )
            return
        wanted = _parse_column_text()
        if not wanted:
            messagebox.showwarning(
                "No columns",
                "The column list is empty.  Add at least one column name "
                "(or click Reset to Defaults).",
                parent=root,
            )
            return
        try:
            missing_per_file = _scan_files_for_missing_cols(files_to_scan, wanted)
        except Exception as exc:
            messagebox.showerror(
                "Could not read file headers",
                f"Header pre-scan failed:\n\n{exc}",
                parent=root,
            )
            return
        if not missing_per_file:
            messagebox.showinfo(
                "All columns present",
                f"All {len(wanted)} listed columns were found in every file.",
                parent=root,
            )
            return
        from .missing_columns import format_per_file_summary as _fmt_missing
        body = (
            "Some listed columns are missing from your files.\n\n"
            + _fmt_missing(missing_per_file)
            + "\n\nMissing columns are tolerated (treated as absent), "
              "but related dose calculations may be skipped for the\n"
              "affected modality rows."
        )
        messagebox.showwarning(
            "Missing columns",
            body,
            parent=root,
        )

    ttk.Button(
        cols_btn_frame, text="Check Against Selected Files",
        command=_check_columns_against_files,
    ).pack(side=tk.LEFT)

    # 4. RUN / CANCEL BUTTONS
    action_frame = ttk.Frame(action_area)
    action_frame.pack(fill=tk.X, pady=(4, 0))

    def _on_run_clicked():
        """Validate the file list and year groups, capture the current
        settings, and start the simulation run.  Every validation failure
        is reported in a dialog and aborts the run."""
        # Capture the year currently being edited before validating anything.
        all_files = _all_files()
        if not all_files:
            messagebox.showwarning("No files",
                                   "Please add at least one Excel input file.",
                                   parent=root)
            return
        active_groups = {y: list(f) for y, f in year_groups.items() if f}
        # Validate iterations
        try:
            n = iter_var.get()
            if n < 1:
                raise ValueError
        except (tk.TclError, ValueError):
            messagebox.showwarning("Invalid iterations",
                                   "Max iterations must be a positive integer.",
                                   parent=root)
            return
        result["cancelled"] = False
        result["year_groups"] = active_groups
        result["run_mode"] = run_mode_var.get()
        # Flattened union for the validations below (and any caller that
        # expects a plain file list).
        result["file_list"] = all_files
        result["n_iterations"] = iter_var.get()
        result["population"] = pop_var.get()
        result["risk_model"] = model_var.get()
        result["ddref_solid"] = ddref_s_var.get()
        result["ddref_leuk"] = ddref_l_var.get()
        result["min_latency_params"] = {
            "Leukaemia": lat_leuk_var.get(),
            "Solid": lat_solid_var.get(),
        }
        result["use_golikov"] = golikov_var.get()
        result["use_mataac"] = mataac_var.get()
        result["icrp152_strict"] = icrp152_var.get()
        result["min_iterations_enabled"] = min_iter_enabled_var.get()
        if min_iter_enabled_var.get():
            try:
                min_iterations = int(min_iter_var.get())
                if min_iterations < 1:
                    raise ValueError
            except (tk.TclError, ValueError, TypeError):
                messagebox.showwarning(
                    "Invalid minimum iterations",
                    "Minimum iterations must be a positive integer.",
                    parent=root,
                )
                return
            result["min_iterations"] = min_iterations
        else:
            result["min_iterations"] = DEFAULTS["min_iterations"]
        # Parse convergence threshold (e.g. "1.0%  —  default" → 1.0)
        try:
            raw = conv_var.get().strip()
            conv_str = raw.split("%")[0].strip() if "%" in raw else raw
            conv_pct = float(conv_str)
            if conv_pct <= 0 or conv_pct > 100:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showwarning("Invalid threshold",
                                   "Convergence threshold must be a positive number (e.g. 1.0%).",
                                   parent=root)
            return
        result["convergence_pct"] = conv_pct
        result["precision_stop_enabled"] = precision_stop_enabled_var.get()
        try:
            precision_pct = float(precision_target_var.get())
            if precision_pct <= 0 or precision_pct > 100:
                raise ValueError
        except (tk.TclError, ValueError, TypeError):
            messagebox.showwarning(
                "Invalid precision target",
                "Precision target must be a positive number (e.g. 1.0%).",
                parent=root,
            )
            return
        result["precision_target_rel_se"] = precision_pct / 100.0
        result["background_radiation_msv_per_year"] = bg_rad_var.get()
        result["output_dir"] = out_var.get()

        # Excel column push-down + missing-column pre-flight check
        wanted_cols = _parse_column_text()
        if not wanted_cols:
            messagebox.showwarning(
                "No columns to load",
                "The Input File Columns tab is empty.  Add at least one column "
                "name, or click 'Reset to Defaults'.",
                parent=root,
            )
            notebook.select(tab_columns)
            return
        try:
            missing_per_file = _scan_files_for_missing_cols(
                all_files, wanted_cols
            )
        except Exception as exc:
            if not messagebox.askyesno(
                "Header pre-scan failed",
                f"Could not pre-read column headers:\n\n{exc}\n\n"
                "Continue anyway?",
                parent=root,
            ):
                return
            missing_per_file = {}
        if missing_per_file:
            from .missing_columns import format_per_file_summary as _fmt_missing
            body = (
                "These expected columns are missing from your file(s).\n\n"
                + _fmt_missing(missing_per_file)
                + "\n\nMissing columns are tolerated (treated as absent), "
                  "but any dose calculation that depends on them\n"
                  "will be skipped for the affected modality rows.\n\n"
                  "Continue with the simulation?"
            )
            if not messagebox.askyesno(
                "Missing columns — continue?",
                body,
                parent=root,
            ):
                notebook.select(tab_columns)
                return
        result["needed_cols"] = wanted_cols
        root.destroy()

    def _cancel():
        root.destroy()

    ttk.Button(action_frame, text="Cancel", command=_cancel).pack(
        side=tk.RIGHT, padx=(6, 0))
    run_btn = ttk.Button(action_frame, text="  Run Simulation  ",
                         command=_on_run_clicked, style="Run.TButton")
    run_btn.pack(side=tk.RIGHT)

    # Enter key runs the simulation
    root.bind("<Return>", lambda _e: _on_run_clicked())
    root.bind("<Escape>", lambda _e: _cancel())

    # Handle window close button (X)
    root.protocol("WM_DELETE_WINDOW", _cancel)

    # Now that every widget exists, size the window to the setup tab's
    # content (plus the title/notebook/button chrome).  If that does not fit
    # the screen, shrink every named font proportionally until it does, so a
    # small display shows the whole dialog at open without resizing or
    # scrolling.  Paddings do not scale, hence the small safety margin and
    # the re-measure loop; 0.7 is the readability floor, below which the
    # canvas scrollbar takes over.
    root.update_idletasks()
    chrome_w = root.winfo_width() - setup_canvas.winfo_width()
    chrome_h = root.winfo_height() - setup_canvas.winfo_height()
    factor = 1.0
    for _ in range(4):
        root.update_idletasks()
        need_w = tab_setup.winfo_reqwidth() + chrome_w
        need_h = tab_setup.winfo_reqheight() + chrome_h
        if (need_w <= avail_w and need_h <= avail_h) or factor <= 0.7:
            break
        factor = max(0.7, factor * min(avail_w / need_w, avail_h / need_h) * 0.97)
        _scale_fonts(factor)
    w, h = min(need_w, avail_w), min(need_h, avail_h)
    root.geometry(f"{w}x{h}+{max(0, (scr_w - w) // 2)}+{max(0, (scr_h - h) // 2)}")

    root.mainloop()

    if result.get("cancelled", True):
        return None
    return result
