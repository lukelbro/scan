"""
Backward-compatibility shim.

Old code that does ``import e11scan.e11scan as e11scan`` will continue
to work.  All real logic now lives in main.py, core.py, and fitting.py.
"""
# Re-export everything from the new architecture
from .main import Scan, ScanMD, _ScanSubset

# Backward-compatible lowercase names
scan = Scan
scanmd = ScanMD
scan_base = _ScanSubset  # Approximation — old scan_base is now _ScanSubset

# Also re-export the fitting classes for any code that imported them directly
from .fitting import FitResult, fit_gaussian, fit_double_gaussian, fit_rabi
