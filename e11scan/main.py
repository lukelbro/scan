"""
User-facing Scan and ScanMD classes.

Backward-compatible API: sc.x, sc.y, sc.gauss.p0, sc.update_function(), etc.
Internally delegates all computation to core.py and fitting.py.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from colorama import Fore

from .core import (
    load_raw_hdf5,
    load_trace,
    evaluate_windows,
    evaluate_raw_function,
    calculate_signal_error,
    process_signal,
    apply_basic_filter,
    apply_stability_filter,
    recompute_windows,
)
from .fitting import fit_gaussian, fit_double_gaussian, fit_rabi


class Scan:
    """Process and analyse oscilloscope scan data from HDF5 files.

    Example::

        sc = Scan('data.h5', function='-(a0 - a1)/((a0 - a1) + (a0 - a2))')
        plt.plot(sc.x, sc.y)
        plt.plot(sc.x, sc.gauss.fit)
    """

    def __init__(
        self,
        filename: str,
        function: str,
        experiment: str = None,
        power: float = None,
        efield: int = None,
        detuning: float = None,
        scanfreq: float = None,
        averages: int = None,
    ):
        # Public attributes (backward-compatible)
        self.filename = filename
        self.function = function
        self.power = power
        self.efield = efield
        self.detuning = detuning
        self.scanfreq = scanfreq
        self.averages = averages

        # Load raw HDF5 data
        raw = load_raw_hdf5(filename)
        self._raw_df = raw['df']
        self.f = raw['f']
        self.dset = raw['dset']
        self.t0 = raw['t0']
        self.dt = raw['dt']
        self.tt = raw['tt']
        self.windows = raw['windows']
        self._windowsind = raw['windowsind']
        self.metadata = raw['metadata']
        self.timestamp = raw['metadata']['timestamp']
        self.numloops = raw['metadata']['numloops']
        self.numtraces = raw['metadata']['numtraces']

        # Resolve experiment type
        if experiment is not None:
            if experiment not in ('microwave', 'volt', 'time'):
                print(
                    f'{Fore.RED}WARNING{Fore.RESET}: New type of experiment '
                    f'{Fore.RED}{experiment}{Fore.RESET} using settings for generic'
                )
                self.experiment = 'generic'
            else:
                self.experiment = experiment
        elif raw['metadata']['experiment_type'] is not None:
            self.experiment = raw['metadata']['experiment_type']
            if self.experiment == 'generic':
                print(
                    f'{Fore.RED}WARNING{Fore.RED}: New type of experiment '
                    f'{Fore.RED}{self.experiment}{Fore.RESET} using settings for generic'
                )
        else:
            raise ValueError(
                'Could not deduce experiment type, please set experiment type manually'
            )

        # Run the full processing pipeline
        self._rebuild()

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _rebuild(self):
        """Recompute everything from raw data. Single source of truth."""
        # Start from a clean copy of raw data
        self.df = self._raw_df.copy()

        # Check for multi-dimensional scan
        x2 = list(set(self.df['v1']))
        if len(x2) > 1:
            print('Multidimensional scan detected use scanmd')

        # Evaluate user function → adds 'signal' column
        self.df = evaluate_windows(self.df, self.function)

        # Calculate error if averages specified
        self.df = calculate_signal_error(
            self.df, self.numloops, self.averages
        )

        # Group, average, baseline subtract
        result = process_signal(
            self.df, self.experiment, self.numloops, self.averages
        )
        self.x = result['x']
        self.y = result['y']
        self.xall = result['xall']
        self.yall = result['yall']
        self.error = result['error']
        self.baseValue = result['baseValue']
        self.variance = result['variance']
        self.range = result['range']

        # Store originals for set_range()
        self._x_original = self.x.copy()
        self._y_original = self.y.copy()

        # Invalidate fit caches (lazy evaluation)
        self._gauss_cache = None
        self._gaussgauss_cache = None
        self._rabi_cache = None

    # ------------------------------------------------------------------
    # Lazy-cached fit properties
    # ------------------------------------------------------------------

    @property
    def gauss(self):
        """Gaussian fit result (computed on first access, then cached)."""
        if self._gauss_cache is None:
            self._gauss_cache = fit_gaussian(self.x, self.y, self.error)
        return self._gauss_cache

    @property
    def gaussgauss(self):
        """Double Gaussian fit result (computed on first access, then cached)."""
        if self._gaussgauss_cache is None:
            self._gaussgauss_cache = fit_double_gaussian(
                self.x, self.y, self.error
            )
        return self._gaussgauss_cache

    @property
    def rabi(self):
        """Rabi oscillation fit result (computed on first access, then cached)."""
        if self._rabi_cache is None:
            self._rabi_cache = fit_rabi(self.x, self.y, self.error)
        return self._rabi_cache

    # ------------------------------------------------------------------
    # Window index property (backward-compatible setter)
    # ------------------------------------------------------------------

    @property
    def windowsind(self):
        return self._windowsind

    @windowsind.setter
    def windowsind(self, new_windows: dict):
        if not isinstance(new_windows, dict):
            raise ValueError("Value must be a dict.")

        # Update window indices and times
        for key in new_windows:
            ind = new_windows[key]
            self._windowsind[key] = ind
            self.windows[key] = self.tt[ind]

        # Recompute a0, a1, a2 from raw traces with new windows
        self._raw_df = recompute_windows(
            self._raw_df, self.f, self._windowsind, self.tt
        )

        # Full rebuild
        self._rebuild()

    # ------------------------------------------------------------------
    # Public mutation methods
    # ------------------------------------------------------------------

    def update_function(self, function: str):
        """Change the evaluation function and recompute."""
        self.function = function
        self._rebuild()

    def update_experiment(self, experiment: str):
        """Change the experiment type and recompute."""
        self.experiment = experiment
        self._rebuild()

    def set_range(self, range_pair):
        """Select a subset of y based on index range [start, end]."""
        start = range_pair[0]
        end = range_pair[1]
        self.x = self._x_original.copy()[start:end]
        self.y = self._y_original.copy()[start:end]

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def basic_filter(self, m: float):
        """Remove outlier data points and recompute."""
        self.df = apply_basic_filter(self.df, m)
        # Re-run processing from filtered df (don't touch _raw_df)
        self.df = calculate_signal_error(self.df, self.numloops, self.averages)
        result = process_signal(
            self.df, self.experiment, self.numloops, self.averages
        )
        self.x = result['x']
        self.y = result['y']
        self.xall = result['xall']
        self.yall = result['yall']
        self.error = result['error']
        self.baseValue = result['baseValue']
        self.variance = result['variance']
        self.range = result['range']
        self._x_original = self.x.copy()
        self._y_original = self.y.copy()
        self._gauss_cache = None
        self._gaussgauss_cache = None
        self._rabi_cache = None

    def remove_unstable(self, m: float, customfunction: str = 'a0-a1'):
        """Remove unstable data points and recompute."""
        self.df = apply_stability_filter(self.df, m, customfunction)
        self.df = calculate_signal_error(self.df, self.numloops, self.averages)
        result = process_signal(
            self.df, self.experiment, self.numloops, self.averages
        )
        self.x = result['x']
        self.y = result['y']
        self.xall = result['xall']
        self.yall = result['yall']
        self.error = result['error']
        self.baseValue = result['baseValue']
        self.variance = result['variance']
        self.range = result['range']
        self._x_original = self.x.copy()
        self._y_original = self.y.copy()
        self._gauss_cache = None
        self._gaussgauss_cache = None
        self._rabi_cache = None

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def get_stability(self, customfunction: str = 'a0-a1') -> pd.Series:
        """Evaluate a custom function on the raw DataFrame."""
        return evaluate_raw_function(self.df, customfunction)

    def plot_stability(self, hline=None, customfunction: str = 'a0-a1'):
        """Plot signal stability over measurement number."""
        plt.clf()
        stability = self.get_stability(customfunction)
        plt.scatter(
            np.linspace(0, stability.shape[0], stability.shape[0]),
            stability, s=1,
        )
        plt.xlabel('measurement number')
        if hline is not None:
            plt.hlines(hline, 0, stability.shape[0])
        plt.ylabel(customfunction)

    # ------------------------------------------------------------------
    # Trace access
    # ------------------------------------------------------------------

    def trace(self, ind: int):
        """Get raw oscilloscope trace (time array, signal array)."""
        return load_trace(self.f, ind)

    def plot_trace(self, ind: int):
        """Plot an oscilloscope trace with window markers."""
        tt, signal = self.trace(ind)
        plt.plot(tt, signal)
        plt.xlabel('time')
        plt.ylabel('signal')

        minval = np.min(signal)
        maxval = np.max(signal)
        color = {
            'A': 'tab:pink', 'B': 'tab:green', 'C': 'tab:red',
            'D': 'tab:orange', 'E': 'tab:purple', 'F': 'tab:olive',
        }
        for key in self.windows:
            plt.vlines(
                self.windows[key], minval, maxval,
                label=key, color=[str(color[key])], linewidth=0.5,
            )
        plt.legend()

    # ------------------------------------------------------------------
    # Background subtraction
    # ------------------------------------------------------------------

    def load_background(self, model, splitpoint: int):
        """Subtract background using a predictive model and recompute."""
        from .core import recompute_windows_with_background

        self._raw_df = recompute_windows_with_background(
            self._raw_df, self.f, self._windowsind, model, splitpoint
        )
        self._rebuild()

    # ------------------------------------------------------------------
    # Legacy aliases
    # ------------------------------------------------------------------

    def load_database(self):
        """Reload raw data from HDF5 and rebuild."""
        raw = load_raw_hdf5(self.filename)
        self._raw_df = raw['df']
        self._rebuild()

    def filter(self, m: float):
        """Legacy filter method."""
        self.load_database()
        self.basic_filter(m)


class ScanMD(Scan):
    """Multi-dimensional scan: splits data by v1 into per-v1 Scan objects."""

    def __init__(
        self,
        filename: str,
        function: str,
        experiment: str = None,
        **kwargs,
    ):
        # Use Scan's init but override build step
        # We need to replicate the init without calling _rebuild
        # because ScanMD has its own build logic.
        self.filename = filename
        self.function = function
        self.power = kwargs.get('power')
        self.efield = kwargs.get('efield')
        self.detuning = kwargs.get('detuning')
        self.scanfreq = kwargs.get('scanfreq')
        self.averages = kwargs.get('averages')

        raw = load_raw_hdf5(filename)
        self._raw_df = raw['df']
        self.f = raw['f']
        self.dset = raw['dset']
        self.t0 = raw['t0']
        self.dt = raw['dt']
        self.tt = raw['tt']
        self.windows = raw['windows']
        self._windowsind = raw['windowsind']
        self.metadata = raw['metadata']
        self.timestamp = raw['metadata']['timestamp']
        self.numloops = raw['metadata']['numloops']
        self.numtraces = raw['metadata']['numtraces']

        # Resolve experiment type
        if experiment is not None:
            if experiment not in ('microwave', 'volt', 'time'):
                print(
                    f'{Fore.RED}WARNING{Fore.RESET}: New type of experiment '
                    f'{Fore.RED}{experiment}{Fore.RESET} using settings for generic'
                )
                self.experiment = 'generic'
            else:
                self.experiment = experiment
        elif raw['metadata']['experiment_type'] is not None:
            self.experiment = raw['metadata']['experiment_type']
            if self.experiment == 'generic':
                print(
                    f'{Fore.RED}WARNING{Fore.RED}: New type of experiment '
                    f'{Fore.RED}{self.experiment}{Fore.RESET} using settings for generic'
                )
        else:
            raise ValueError(
                'Could not deduce experiment type, please set experiment type manually'
            )

        # Build the multi-dimensional structure
        self._build_md()

    def _build_md(self):
        """Build the per-v1 subsets."""
        self.df = self._raw_df.copy()
        self.df = evaluate_windows(self.df, self.function)
        self.df = calculate_signal_error(
            self.df, self.numloops, self.averages
        )

        # Split by v1
        self.x2 = sorted(self.df['v1'].unique())
        self.sets = []
        for val in self.x2:
            df_subset = self.df[self.df['v1'] == val].copy()
            sc = _ScanSubset(
                experiment=self.experiment,
                df=df_subset,
                x2=val,
                function=self.function,
                filename=self.filename,
                averages=self.averages,
                numloops=self.numloops,
            )
            self.sets.append(sc)

    def load_background(self, model, splitpoint: int):
        """Subtract background using a predictive model and rebuild MD."""
        from .core import recompute_windows_with_background

        self._raw_df = recompute_windows_with_background(
            self._raw_df, self.f, self._windowsind, model, splitpoint
        )
        self._build_md()


class _ScanSubset:
    """Lightweight container for a single v1 subset of a ScanMD.

    Holds processed x, y, error, fits, etc. without file handles.
    This replaces the old scan_base objects used in scanmd.generate_sets().
    """

    def __init__(
        self,
        experiment: str,
        df: pd.DataFrame,
        x2: float,
        function: str,
        filename: str,
        averages: int = None,
        numloops: int = 1,
    ):
        self.experiment = experiment
        self.df = df
        self.x2 = x2
        self.function = function
        self.filename = filename
        self.averages = averages
        self.numloops = numloops

        # Process the subset
        result = process_signal(df, experiment, numloops, averages)
        self.x = result['x']
        self.y = result['y']
        self.xall = result['xall']
        self.yall = result['yall']
        self.error = result['error']
        self.baseValue = result['baseValue']
        self.variance = result['variance']
        self.range = result['range']

        # Lazy fit caches
        self._gauss_cache = None
        self._gaussgauss_cache = None
        self._rabi_cache = None

    @property
    def gauss(self):
        if self._gauss_cache is None:
            self._gauss_cache = fit_gaussian(self.x, self.y, self.error)
        return self._gauss_cache

    @property
    def gaussgauss(self):
        if self._gaussgauss_cache is None:
            self._gaussgauss_cache = fit_double_gaussian(
                self.x, self.y, self.error
            )
        return self._gaussgauss_cache

    @property
    def rabi(self):
        if self._rabi_cache is None:
            self._rabi_cache = fit_rabi(self.x, self.y, self.error)
        return self._rabi_cache

    def get_stability(self, customfunction: str = 'a0-a1') -> pd.Series:
        return evaluate_raw_function(self.df, customfunction)

    def plot_stability(self, hline=None, customfunction: str = 'a0-a1'):
        plt.clf()
        stability = self.get_stability(customfunction)
        plt.scatter(
            np.linspace(0, stability.shape[0], stability.shape[0]),
            stability, s=1,
        )
        plt.xlabel('measurement number')
        if hline is not None:
            plt.hlines(hline, 0, stability.shape[0])
        plt.ylabel(customfunction)

    def remove_unstable(self, m: float, customfunction: str = 'a0-a1'):
        """Remove unstable points and reprocess the subset."""
        self.df = apply_stability_filter(self.df, m, customfunction)
        result = process_signal(
            self.df, self.experiment, self.numloops, self.averages
        )
        self.x = result['x']
        self.y = result['y']
        self.xall = result['xall']
        self.yall = result['yall']
        self.error = result['error']
        self.baseValue = result['baseValue']
        self.variance = result['variance']
        self.range = result['range']
        self._gauss_cache = None
        self._gaussgauss_cache = None
        self._rabi_cache = None
