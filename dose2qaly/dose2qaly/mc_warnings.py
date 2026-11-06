"""CuPy / GPU backend diagnostics shared by the Monte Carlo modules.

Holds the user-facing notice text shown when the GPU backend is missing or
non-functional, plus :func:`get_backend_info`, the probe the GUI status bar
uses to report which backend the simulation will run on.  ``main`` imports
these at startup; the simulation itself lives in ``vectorized`` and
``main``.
"""
from __future__ import annotations

import multiprocessing as _mp
from typing import Dict

# CuPy / GPU backend diagnostics
# User-facing notice text and probe helpers.  ``main.py`` imports these on
# startup to tell the user which backend will be used; the simulation engine
# also calls :func:`get_backend_info` to populate the GUI status bar.


# Wording note: these notices state only what can be verified from the machine
# itself (which backend will be used, and how to change it).  They deliberately
# avoid quantified speed claims, since the CPU/GPU difference here depends on
# dataset size, iteration count, CPU core count and GPU model, and is not
# measured by this program.
CUPY_NOT_INSTALLED_WARNING = (
    "The GPU backend is not available — CuPy is not installed.\n\n"
    "The simulation will run on the CPU backend instead.  Both backends\n"
    "evaluate the same risk model and convergence criterion; the GPU backend\n"
    "is typically faster, but the actual difference depends on your dataset\n"
    "size, iteration count and hardware.\n\n"
    "To make the GPU backend available:\n"
    "  1. Install the NVIDIA CUDA Toolkit (if not already installed):\n"
    "     https://developer.nvidia.com/cuda-downloads\n\n"
    "  2. Install CuPy matching your CUDA version, e.g. for CUDA 12.x:\n"
    "     pip install cupy-cuda12x\n\n"
    "  3. Restart this application."
)
CUPY_NOT_FUNCTIONAL_WARNING = (
    "CuPy is installed but cannot access the GPU.\n"
    "The simulation will run on the CPU backend instead.\n\n"
    "Possible causes and fixes:\n"
    "  • NVIDIA CUDA Toolkit is missing or the wrong version:\n"
    "    https://developer.nvidia.com/cuda-downloads\n\n"
    "  • GPU drivers need updating:\n"
    "    https://www.nvidia.com/drivers\n\n"
    "  • CuPy version does not match your CUDA version.\n"
    "    Reinstall with the correct variant, e.g.:\n"
    "    pip uninstall cupy  &&  pip install cupy-cuda12x"
)


def check_optional_packages() -> list:
    """Check for optional packages and return a list of user-facing warning strings.

    Only packages that would meaningfully change performance or functionality
    are checked here.  Required packages (numpy, pandas) are not listed because
    their absence causes an immediate ImportError before this function is reached.
    """
    warnings_out = []

    cupy_installed = False
    gpu_functional = False
    try:
        import cupy as _cp_chk
        cupy_installed = True
        _test = _cp_chk.array([1.0])
        _ = float(_test[0])          # force sync to confirm CUDA is live
        gpu_functional = True
    except ImportError:
        pass
    except Exception:
        pass

    if not cupy_installed:
        warnings_out.append(CUPY_NOT_INSTALLED_WARNING)
    elif not gpu_functional:
        warnings_out.append(CUPY_NOT_FUNCTIONAL_WARNING)

    return warnings_out


def get_backend_info() -> Dict[str, object]:
    """Detect available CPU cores and GPU devices for GUI display.

    Reports what ``run_simulation`` will actually use:
    - If CuPy GPU is functional: all GPUs in use, 0 CPUs
    - Otherwise: all CPU cores in use, 0 GPUs
    """
    cpu_available = 1
    try:
        cpu_available = max(1, int(_mp.cpu_count()))
    except Exception:
        cpu_available = 1

    try:
        import cupy as _cp_chk
    except Exception:
        _cp_chk = None

    gpu_functional = False
    gpu_available_count = 0
    if _cp_chk is not None:
        try:
            _ = _cp_chk.array([0.0])
            gpu_functional = True
            gpu_available_count = max(0, int(_cp_chk.cuda.runtime.getDeviceCount()))
        except Exception:
            gpu_available_count = 0

    if gpu_functional and gpu_available_count > 0:
        return {
            "backend": "GPU",
            "cpu_available": cpu_available,
            "gpu_available": gpu_available_count,
            "cpu_in_use": 0,
            "gpu_in_use": gpu_available_count,
        }
    return {
        "backend": "CPU",
        "cpu_available": cpu_available,
        "gpu_available": gpu_available_count,
        "cpu_in_use": cpu_available,
        "gpu_in_use": 0,
    }
