"""Cross-cutting numeric constants shared by multiple modules.

Each value here is the single declaration of a constant used at several
call sites.  New cross-cutting constants should be added here rather than
re-declared at each use site.
"""
from __future__ import annotations

import os

# Wald 95% normal-approximation z-score: half-width = Z_95 * SE.
# Used for Monte Carlo convergence (main.py), confidence bands on filtered
# risk curves (analysis.py), and the "refine rare cells" stopping rule
# (``research_questions`` imports it as ``CI_Z``).
Z_95: float = 1.96

# Natural background effective dose used ONLY for the plain-language
# "equivalent time of natural background" comparison in the Summary tab
# (config default, engine fallback, GUI fallback).  2.4 mSv/yr is the
# UNSCEAR worldwide average annual effective dose from natural sources
# (cosmic 0.39 + external terrestrial 0.48 + inhalation, mainly radon,
# 1.26 + ingestion 0.29; typical range 1-13 mSv/yr): UNSCEAR 2008 Report,
# Annex B, Table 12.  For reference, Finland: 5.1 mSv/yr natural incl.
# 4 mSv radon; external (cosmic + terrestrial) alone 0.8 (STUK-A263, 2020).
BACKGROUND_MSV_PER_YEAR: float = 2.4

# Default output directory for results and the patient-data caches
# (.raw_excel_cache, .preprocess_cache).  Deliberately NOT relative to the
# package: after ``pip install`` the package lives in site-packages, which
# is the wrong place for pseudonymised patient records and may be read-only.
# A per-user folder is writable, private to the account, and survives
# reinstalling the package.  The GUI lets the user override it.
DEFAULT_OUTPUT_DIR: str = os.path.join(os.path.expanduser("~"), "dose2qaly_output")
