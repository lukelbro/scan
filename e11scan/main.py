"""
User-facing Scan and ScanMD classes.

Backward-compatible API: sc.x, sc.y, sc.gauss.p0, sc.update_function(), etc.
Internally delegates all computation to core.py and fitting.py.
"""
from typing import Optional

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

# ---------------------------------------------------------------------------
# Pipeline step registry: maps step names to pure (df, **kwargs) -> df
# ---------------------------------------------------------------------------
_PIPELINE_STEPS = {
    'drop_rows': lambda df, indices: df.drop(indices, axis=0),
    'filter_rows': lambda df, rows: df.iloc[rows[0]:rows[1]].copy(),
    'basic_filter': lambda df, m: apply_basic_filter(df, m),
    'remove_unstable': lambda df, threshold, customfunction='a0-a1': (
        apply_stability_filter(df, threshold, customfunction)
    ),
}


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

        # Declarative pipeline: ordered list of (step_name, kwargs)
        self.pipeline: list[tuple[str, dict]] = []

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
        """Recompute from pristine raw data by replaying ``self.pipeline`` in order.

        ``evaluate_windows`` is called lazily — right before the first
        step that requires the ``signal`` column.
        """
        df = self._raw_df.copy()
        windows_evaluated = False

        for step_name, kwargs in self.pipeline:
            if step_name in ('basic_filter', 'remove_unstable') and not windows_evaluated:
                df = evaluate_windows(df, self.function)
                windows_evaluated = True

            handler = _PIPELINE_STEPS[step_name]
            df = handler(df, **kwargs)

        # Ensure windows are evaluated even if no signal-dependent steps were run
        if not windows_evaluated:
            df = evaluate_windows(df, self.function)

        # Compute error, group, average, baseline subtract
        df = calculate_signal_error(df, self.numloops, self.averages)
        result = process_signal(
            df, self.experiment, self.numloops, self.averages
        )
        self.df = df
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

        # Invalidate lazy fit caches
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
    # Pipeline helpers
    # ------------------------------------------------------------------

    _STEP_ARG_NAMES = {
        'drop_rows': ['indices'],
        'filter_rows': ['rows'],
        'basic_filter': ['m'],
        'remove_unstable': ['threshold', 'customfunction'],
    }

    def _resolve_step_args(self, name, args, kwargs) -> dict:
        """Map positional convenience args onto the handler's kwarg names."""
        resolved = dict(zip(self._STEP_ARG_NAMES[name], args))
        resolved.update(kwargs)
        return resolved

    def _normalize_step(self, name, args):
        """Used by ``set_pipeline`` to accept bare values or dicts uniformly."""
        if isinstance(args, dict):
            return (name, args)
        if isinstance(args, tuple) and name in ('remove_unstable',):
            return (name, self._resolve_step_args(name, args, {}))
        return (name, self._resolve_step_args(name, (args,), {}))

    def _replace_or_remove_step(self, name, value, arg_name, extra_kwargs=None):
        """Remove any existing step of *name*, then re-add it unless *value* is None."""
        # Remember the position of the existing step (if any)
        existing_pos = None
        for i, (n, kw) in enumerate(self.pipeline):
            if n == name:
                existing_pos = i
                break
        self.pipeline = [(n, kw) for n, kw in self.pipeline if n != name]
        if value is not None:
            kwargs = {arg_name: value, **(extra_kwargs or {})}
            if existing_pos is not None:
                self.pipeline.insert(existing_pos, (name, kwargs))
            else:
                self.pipeline.append((name, kwargs))
        self._rebuild()

    # ------------------------------------------------------------------
    # Pipeline public API
    # ------------------------------------------------------------------

    def set_pipeline(self, steps: list):
        """Replace the entire filter pipeline and rebuild.

        Parameters
        ----------
        steps : list of (name, args)
            Each entry is ``(step_name, kwargs_dict)`` or
            ``(step_name, positional_value)`` — e.g.
            ``('drop_rows', [50, 150])`` or
            ``('drop_rows', {'indices': [50, 150]})``.
        """
        for name, args in steps:
            if name not in _PIPELINE_STEPS:
                raise ValueError(
                    f"Unknown pipeline step '{name}'. "
                    f"Valid: {list(_PIPELINE_STEPS)}"
                )
        self.pipeline = [self._normalize_step(name, args) for name, args in steps]
        self._rebuild()

    def add_step(self, name: str, *args, order: int = None, **kwargs):
        """Append (or insert) a single pipeline step and rebuild.

        Parameters
        ----------
        name : str
            One of ``'drop_rows'``, ``'filter_rows'``, ``'basic_filter'``,
            ``'remove_unstable'``.
        order : int, optional
            Position to insert at.  Defaults to the end.
        """
        if name not in _PIPELINE_STEPS:
            raise ValueError(
                f"Unknown pipeline step '{name}'. "
                f"Valid: {list(_PIPELINE_STEPS)}"
            )
        step_kwargs = self._resolve_step_args(name, args, kwargs)
        step = (name, step_kwargs)
        if order is None:
            self.pipeline.append(step)
        else:
            self.pipeline.insert(order, step)
        self._rebuild()

    def clear_pipeline(self):
        """Remove all pipeline steps and restore the pristine, unfiltered dataset."""
        self.pipeline = []
        self._rebuild()

    # ------------------------------------------------------------------
    # Backward-compatible convenience wrappers
    # ------------------------------------------------------------------

    def filter_rows(self, rows: Optional[tuple]):
        """Select a contiguous range of rows to keep.

        Parameters
        ----------
        rows : tuple of (start, end) or None
            ``(start, end)`` passed to ``DataFrame.iloc[start:end]``.
            ``None`` disables the row range and restores the full dataset.
        """
        self._replace_or_remove_step('filter_rows', rows, arg_name='rows')

    def filter_drop_rows(self, indices):
        """Drop specific rows by their DataFrame index.

        Parameters
        ----------
        indices : list, tuple, or None
            Row indices to exclude from the dataset.
            ``None`` disables the drop and restores those rows.
        """
        self._replace_or_remove_step(
            'drop_rows',
            list(indices) if indices is not None else None,
            arg_name='indices',
        )

    def basic_filter(self, m: Optional[float]):
        """Apply (or disable) the basic outlier filter.

        Parameters
        ----------
        m : float or None
            Relative spacing threshold.  ``None`` disables the filter
            and restores previously removed points.
        """
        self._replace_or_remove_step('basic_filter', m, arg_name='m')

    def remove_unstable(self, threshold: Optional[float], customfunction: str = 'a0-a1'):
        """Apply (or disable) the stability filter.

        Parameters
        ----------
        threshold : float or None
            Maximum allowed value of *customfunction*.  ``None`` disables
            the filter and restores previously removed points.
        customfunction : str
            Expression evaluated on the raw window columns.
        """
        self._replace_or_remove_step(
            'remove_unstable', threshold, arg_name='threshold',
            extra_kwargs={'customfunction': customfunction},
        )

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
        """Reload raw data from HDF5, reset all filters, and rebuild."""
        raw = load_raw_hdf5(self.filename)
        self._raw_df = raw['df']
        self.pipeline = []
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
        self.x2 = x2
        self.function = function
        self.filename = filename
        self.averages = averages
        self.numloops = numloops

        # Pristine raw data for this subset (already has 'signal' column)
        self._raw_df = df.copy()

        # Declarative pipeline: ordered list of (step_name, kwargs)
        self.pipeline: list[tuple[str, dict]] = []

        # Lazy fit caches
        self._gauss_cache = None
        self._gaussgauss_cache = None
        self._rabi_cache = None

        # Run the full processing pipeline
        self._rebuild()

    def _rebuild(self):
        """Recompute from pristine subset data by replaying ``self.pipeline``."""
        df = self._raw_df.copy()
        windows_evaluated = False

        for step_name, kwargs in self.pipeline:
            if step_name in ('basic_filter', 'remove_unstable') and not windows_evaluated:
                df = evaluate_windows(df, self.function)
                windows_evaluated = True

            handler = _PIPELINE_STEPS[step_name]
            df = handler(df, **kwargs)

        if not windows_evaluated:
            df = evaluate_windows(df, self.function)

        # Compute error, group, average, baseline subtract
        df = calculate_signal_error(df, self.numloops, self.averages)
        result = process_signal(
            df, self.experiment, self.numloops, self.averages
        )
        self.df = df
        self.x = result['x']
        self.y = result['y']
        self.xall = result['xall']
        self.yall = result['yall']
        self.error = result['error']
        self.baseValue = result['baseValue']
        self.variance = result['variance']
        self.range = result['range']

        # Invalidate lazy fit caches
        self._gauss_cache = None
        self._gaussgauss_cache = None
        self._rabi_cache = None

    # ------------------------------------------------------------------
    # Lazy-cached fit properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pipeline helpers (subset-specific)
    # ------------------------------------------------------------------

    _STEP_ARG_NAMES = {
        'drop_rows': ['indices'],
        'filter_rows': ['rows'],
        'basic_filter': ['m'],
        'remove_unstable': ['threshold', 'customfunction'],
    }

    def _resolve_step_args(self, name, args, kwargs) -> dict:
        resolved = dict(zip(self._STEP_ARG_NAMES[name], args))
        resolved.update(kwargs)
        return resolved

    def _normalize_step(self, name, args):
        if isinstance(args, dict):
            return (name, args)
        if isinstance(args, tuple) and name in ('remove_unstable',):
            return (name, self._resolve_step_args(name, args, {}))
        return (name, self._resolve_step_args(name, (args,), {}))

    def _replace_or_remove_step(self, name, value, arg_name, extra_kwargs=None):
        """Remove any existing step of *name*, then re-add it unless *value* is None."""
        existing_pos = None
        for i, (n, kw) in enumerate(self.pipeline):
            if n == name:
                existing_pos = i
                break
        self.pipeline = [(n, kw) for n, kw in self.pipeline if n != name]
        if value is not None:
            kwargs = {arg_name: value, **(extra_kwargs or {})}
            if existing_pos is not None:
                self.pipeline.insert(existing_pos, (name, kwargs))
            else:
                self.pipeline.append((name, kwargs))
        self._rebuild()

    # ------------------------------------------------------------------
    # Pipeline public API
    # ------------------------------------------------------------------

    def set_pipeline(self, steps: list):
        """Replace the entire filter pipeline and rebuild."""
        for name, args in steps:
            if name not in _PIPELINE_STEPS:
                raise ValueError(
                    f"Unknown pipeline step '{name}'. "
                    f"Valid: {list(_PIPELINE_STEPS)}"
                )
        self.pipeline = [self._normalize_step(name, args) for name, args in steps]
        self._rebuild()

    def add_step(self, name: str, *args, order: int = None, **kwargs):
        """Append (or insert) a single pipeline step and rebuild."""
        if name not in _PIPELINE_STEPS:
            raise ValueError(
                f"Unknown pipeline step '{name}'. "
                f"Valid: {list(_PIPELINE_STEPS)}"
            )
        step_kwargs = self._resolve_step_args(name, args, kwargs)
        step = (name, step_kwargs)
        if order is None:
            self.pipeline.append(step)
        else:
            self.pipeline.insert(order, step)
        self._rebuild()

    def clear_pipeline(self):
        """Remove all pipeline steps and restore the pristine, unfiltered dataset."""
        self.pipeline = []
        self._rebuild()

    # ------------------------------------------------------------------
    # Backward-compatible convenience wrappers
    # ------------------------------------------------------------------

    def filter_rows(self, rows: Optional[tuple]):
        """Select a contiguous range of rows to keep."""
        self._replace_or_remove_step('filter_rows', rows, arg_name='rows')

    def filter_drop_rows(self, indices):
        """Drop specific rows by their DataFrame index."""
        self._replace_or_remove_step(
            'drop_rows',
            list(indices) if indices is not None else None,
            arg_name='indices',
        )

    def basic_filter(self, m: Optional[float]):
        """Apply (or disable) the basic outlier filter."""
        self._replace_or_remove_step('basic_filter', m, arg_name='m')

    def remove_unstable(self, threshold: Optional[float], customfunction: str = 'a0-a1'):
        """Apply (or disable) the stability filter."""
        self._replace_or_remove_step(
            'remove_unstable', threshold, arg_name='threshold',
            extra_kwargs={'customfunction': customfunction},
        )
