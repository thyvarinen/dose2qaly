"""dose2qaly — Monte Carlo lifetime cancer-risk simulation
for diagnostic medical-imaging exams.

Implements ICRP 152 LAR / REIC integration plus per-exam Monte Carlo
sampling of cancer outcomes for radiology workflow datasets (CT,
mammography, radiography, dental CBCT).

Public API (lazily re-exported -- see ``_LAZY_ATTRS`` below)
--------------------------------------------------------------
* ``compute_exam_organ_doses``   per-exam organ-absorbed-dose dict (Gy).
* ``compute_lifetime_risk``      ICRP 152 LAR / REIC integral per cancer site.
* ``run_simulation``             full MC pipeline driver.
* ``run_yearly_simulation``      multi-year batch driver (averaged / single / combined).
* ``start_gui``                  launch the Tkinter GUI.
* ``get_backend_info``           query NumPy / vectorisation backend.

Why lazy?
---------
``import dose2qaly.<any submodule>`` always executes this
file first (that is how Python packages work), so if this file eagerly
imported ``main`` (which pulls in ``gui`` -> tkinter/matplotlib and
``excel_io`` -> pandas) every caller paid that cost even when they only
wanted e.g. ``compute_exam_organ_doses``.  PEP 562 module ``__getattr__``
below defers each symbol's real import until it is first *used*, not
merely referenced by name.  For callers: each sub-import happens exactly
once, attribute lookups return the same object, and unknown names raise
``AttributeError``.
"""

# Maps a public attribute name -> (relative module, attribute name in that
# module).  Add new re-exports here rather than importing eagerly above.
_LAZY_ATTRS = {
    "compute_exam_organ_doses": (".exam_doses", "compute_exam_organ_doses"),
    "compute_lifetime_risk":    (".risk", "compute_lifetime_risk"),
    "run_simulation":           (".main", "run_simulation"),
    "start_gui":                (".main", "start_gui"),
    # Defined in mc_warnings; main.py itself only re-exports it, so we can
    # skip loading main.py entirely for this one symbol.
    "get_backend_info":         (".mc_warnings", "get_backend_info"),
    "run_yearly_simulation":    (".year_batch", "run_yearly_simulation"),
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str):
    """PEP 562 lazy attribute lookup -- see module docstring."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    # Cache on the package module so repeated access skips __getattr__.
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
