"""Shared parsing utilities for dose modules.

This module hosts the single ``_to_float`` implementation shared by the
dose modules (``exam_doses``, ``radiography_doses``, ``ct_doses``,
``fluoro_doses``) — one superset of NaN handling, comma-decimal
support, and int/float fast paths, so all dose lookups parse Excel
cells the same way.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def to_float(x) -> Optional[float]:
    """Parse a value to ``float``, returning ``None`` on failure or non-finite.

    Handles:
    * ``None`` → ``None``
    * ``float`` (incl. ``np.float64``) → ``float(x)`` if finite, else ``None``
    * ``int`` / ``np.integer`` → ``float(x)``
    * strings (incl. comma-decimal ``"1,5"``) → parsed ``float`` if finite
    * anything else that ``float(str(x))`` can parse

    Non-finite values (``inf``, ``nan``) always return ``None`` — the
    strictest behaviour of the four former implementations, which all
    callers already guard against with ``is None or <= 0``.
    """
    if x is None:
        return None
    # Fast path: pandas numeric cells are already float (np.float64
    # subclasses float).  ``float(x)`` round-trips identically to
    # ``float(str(x))`` for finite floats and normalises np.float64 ->
    # python float, so this is identical while skipping the str()/replace()
    # on the common case.
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (int, np.integer)):
        return float(x)
    try:
        v = float(str(x).replace(",", "."))
        return v if math.isfinite(v) else None
    except Exception:
        return None
