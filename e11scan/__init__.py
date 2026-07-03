# New modular architecture
from .main import Scan, ScanMD, _ScanSubset

# Backward-compatible lowercase aliases
scan = Scan
scanmd = ScanMD

# Legacy re-export for old e11scan.e11scan import paths
from . import main as _main
from . import core as _core
from . import fitting as _fitting

__all__ = ['Scan', 'ScanMD', 'scan', 'scanmd', '_ScanSubset']