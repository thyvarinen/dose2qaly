"""Entry point: ``python -m dose2qaly`` launches the GUI.

Delegates to :mod:`dose2qaly.main`.  This thin wrapper only
exists so that the package is callable via ``python -m`` (the canonical
way to launch a Python package); ``main.py`` carries the full engine
logic and its own ``if __name__ == "__main__":`` GUI launcher.
"""
import runpy


def main():
    # Re-execute main.py with __name__ == "__main__" so the GUI block at the
    # bottom of main.py runs as if the file had been invoked directly.
    runpy.run_module("dose2qaly.main", run_name="__main__")


if __name__ == "__main__":
    main()
