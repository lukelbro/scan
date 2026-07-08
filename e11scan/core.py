"""
Pure functions for HDF5 loading, signal evaluation, and processing.

Every function takes data in and returns data out — no side effects,
no hidden state, no file handle management.
"""
import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HDF5 Loading
# ---------------------------------------------------------------------------

def load_raw_hdf5(filename: str) -> dict:
    """Load raw data from an HDF5 file and return all metadata + data.

    Returns a dict with keys:
        df          — DataFrame with columns v0, v1, a0, a1, a2
        f           — open h5py.File handle (caller must close)
        dset        — reference to f['analysis']
        t0          — oscilloscope start time
        dt          — oscilloscope time step
        tt          — full time axis array
        windows     — dict of window times {A: t, B: t, ...}
        windowsind  — dict of window indices {A: i, B: i, ...}
        metadata    — dict with timestamp, numloops, numtraces, experiment_type
    """
    try:
        f = h5py.File(filename, 'r', swmr=True)
        dset = f['analysis']
    except Exception:
        raise ValueError('File or dataset not found')

    # Oscilloscope time axis
    t0 = f['osc_0'].attrs['t0']
    dt = f['osc_0'].attrs['dt']
    signal_0 = f['osc_0'][0]
    tt = np.linspace(t0, t0 + dt * len(signal_0), len(signal_0))

    # Window locations — time values and array indices
    windows = {}
    windowsind = {}
    for window in 'ABCDEF':
        windows[window] = dset.attrs[window] + t0
        windowsind[window] = int(np.abs(tt - windows[window]).argmin())

    # Sort window pairs so lower index comes first
    for pair in [('A', 'B'), ('C', 'D'), ('E', 'F')]:
        if windowsind[pair[0]] > windowsind[pair[1]]:
            windowsind[pair[0]], windowsind[pair[1]] = (
                windowsind[pair[1]], windowsind[pair[0]]
            )

    # Metadata
    timestamp = np.datetime64(f.attrs['timestamp'])
    numloops = f.attrs['v0_loops']
    numtraces = len(f['osc_0'])

    # Experiment type deduction
    experiment_type = None
    if 'var 0' in f.attrs:
        raw_type = f.attrs['var 0']
        if raw_type == 'microwaves (GHz)':
            experiment_type = 'microwave'
        else:
            experiment_type = 'generic'

    # Build DataFrame
    df = pd.DataFrame.from_records(dset, columns=dset.dtype.fields.keys())

    metadata = {
        'timestamp': timestamp,
        'numloops': numloops,
        'numtraces': numtraces,
        'experiment_type': experiment_type,
    }

    # Guard: if both channels exist they must have the same number of traces
    has_osc_1 = 'osc_1' in f
    if has_osc_1 and len(f['osc_1']) != len(f['osc_0']):
        raise ValueError(
            f"osc_0 has {len(f['osc_0'])} traces but osc_1 has "
            f"{len(f['osc_1'])}. Both channels must have the same "
            f"number of rows."
        )

    return {
        'df': df,
        'f': f,
        'dset': dset,
        't0': t0,
        'dt': dt,
        'tt': tt,
        'windows': windows,
        'windowsind': windowsind,
        'metadata': metadata,
        'has_osc_1': has_osc_1,
    }


def load_trace(f, ind: int, channel: str = 'osc_0'):
    """Load a single oscilloscope trace from an open HDF5 file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.
    ind : int
        Trace index.
    channel : str
        HDF5 dataset name.  Default ``'osc_0'``.  Pass ``'osc_1'`` to read
        the secondary channel.

    Returns
    -------
    (tt, signal) where tt is the time axis and signal is the trace.
    """
    if channel not in f:
        raise ValueError(f"Channel '{channel}' not found in HDF5 file.")
    if ind > len(f[channel]):
        raise ValueError(
            f"Number of scans in '{channel}' is {len(f[channel])}, "
            f"{ind} is outside range"
        )
    signal = f[channel][ind]
    t0 = f[channel].attrs['t0']
    dt = f[channel].attrs['dt']
    tt = np.linspace(t0, t0 + dt * len(signal), len(signal))
    return tt, signal


# ---------------------------------------------------------------------------
# Signal Evaluation
# ---------------------------------------------------------------------------

def evaluate_windows(df: pd.DataFrame, function_str: str) -> pd.DataFrame:
    """Evaluate the user function on window columns and add a 'signal' column.

    Uses the same string-replacement + eval() approach as the original code
    to guarantee identical behaviour, but with a safer namespace.
    """
    df = df.copy()
    function_string = function_str
    function_string = function_string.replace('a0', "df['a0']")
    function_string = function_string.replace('a1', "df['a1']")
    function_string = function_string.replace('a2', "df['a2']")

    # Build a safe namespace with only what the expression needs
    namespace = {'df': df, 'np': np, 'abs': abs}
    df['signal'] = eval(function_string, {"__builtins__": {}}, namespace)
    return df


def evaluate_raw_function(df: pd.DataFrame, function_str: str) -> pd.Series:
    """Evaluate a user function directly on DataFrame columns.

    Used by filters that need to compute expressions like 'a0-a1' on the
    current DataFrame without adding a 'signal' column.

    Returns a Series with the evaluation result.
    """
    function_string = function_str
    function_string = function_string.replace('a0', "df['a0']")
    function_string = function_string.replace('a1', "df['a1']")
    function_string = function_string.replace('a2', "df['a2']")

    namespace = {'df': df, 'np': np, 'abs': abs}
    return eval(function_string, {"__builtins__": {}}, namespace)


# ---------------------------------------------------------------------------
# Error Calculation
# ---------------------------------------------------------------------------

def calculate_signal_error(
    df: pd.DataFrame, numloops: int, averages: int = None,
) -> pd.DataFrame:
    """Add an 'error' column to the DataFrame if averages are specified.

    Uses Bernoulli statistics: sigma = sqrt(0.01/averages) / sqrt(numloops).
    """
    df = df.copy()
    if averages is not None:
        df['error'] = np.sqrt(0.01 / averages) * 1.0 / np.sqrt(numloops)
    return df


# ---------------------------------------------------------------------------
# Signal Processing Pipeline
# ---------------------------------------------------------------------------

def process_signal(
    df: pd.DataFrame,
    experiment: str,
    numloops: int = 1,
    averages: int = None,
) -> dict:
    """Group by v0, average, subtract baseline, compute ranges.

    Returns a dict with keys:
        x, y, xall, yall, error, baseValue, variance, range, df_mean
    """
    dfmean = df.groupby(['v0']).mean()
    dfmean = dfmean.sort_values('v0')

    # Baseline subtraction
    if experiment == 'microwave':
        baseValue = dfmean['signal'].take(np.arange(0, 10)).mean()
    elif experiment == 'volt':
        baseValue = 0
    elif experiment == 'time':
        baseValue = dfmean['signal'].take([0]).values
    elif experiment == 'generic':
        baseValue = 0
    else:
        raise ValueError(f'Experiment {experiment} not recognised')

    y = dfmean['signal'].values - baseValue
    x = dfmean['signal'].keys()  # v0 index values
    variance = df.groupby(['v0'])['signal'].var()

    yall = df['signal'].values - baseValue
    xall = df['v0'].values

    # Error from DataFrame if present
    error = np.array([0])
    if averages is not None and 'error' in dfmean.columns:
        error = np.array(dfmean['error'])

    # Range calculation
    ranges = df.groupby(['v0']).agg({'signal': ['min', 'max', 'mean']})
    ranges = ranges.sort_values('v0')
    signalmin = ranges['signal']['min']
    signalmax = ranges['signal']['max']
    signalmean = ranges['signal']['mean']
    signal_range = [np.abs(signalmin - signalmean), np.abs(signalmax - signalmean)]

    return {
        'x': x,
        'y': y,
        'xall': xall,
        'yall': yall,
        'error': error,
        'baseValue': baseValue,
        'variance': variance,
        'range': signal_range,
        'df_mean': dfmean,
    }


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_basic_filter(df: pd.DataFrame, m: float) -> pd.DataFrame:
    """Remove outlier measurements based on relative spacing.

    For each v0 group, if the gap between the first two sorted signal values
    is more than m times the gap between the second and third, drop the
    extreme point.  Symmetrically for the third point.
    """
    df = df.copy()
    idrop = []
    for v0 in np.array(df['v0']):
        dfi = df[df['v0'] == v0]
        signal = dfi['signal'].sort_values()
        if len(signal) < 3:
            continue
        d1 = np.abs(signal.iloc[0] - signal.iloc[1])
        d2 = np.abs(signal.iloc[1] - signal.iloc[2])

        if d1 > m * d2:
            idrop.append(signal.index[0])
        if d2 > m * d1:
            idrop.append(signal.index[2])

    idrop = list(set(idrop))
    if idrop:
        df = df.drop(idrop, axis=0)
    return df


def apply_stability_filter(
    df: pd.DataFrame,
    threshold: float,
    customfunction: str = 'a0-a1',
) -> pd.DataFrame:
    """Remove measurements where the custom function exceeds a threshold.

    The customfunction is evaluated on the current DataFrame and any
    value above the threshold is dropped.
    """
    df = df.copy()
    stability = evaluate_raw_function(df, customfunction)
    idrop = [
        index for index, value in stability.items() if value > threshold
    ]
    idrop = list(set(idrop))
    if idrop:
        df = df.drop(idrop, axis=0)
    return df


# ---------------------------------------------------------------------------
# Window Recomputation
# ---------------------------------------------------------------------------

def recompute_windows(
    df: pd.DataFrame,
    f,
    windowsind: dict,
    tt: np.ndarray,
) -> pd.DataFrame:
    """Recompute a0, a1, a2 columns from raw traces using new window indices.

    Used when the user adjusts window boundaries via the windowsind setter.
    """
    df = df.copy()
    indA = windowsind['A']
    indB = windowsind['B']
    indC = windowsind['C']
    indD = windowsind['D']
    indE = windowsind['E']
    indF = windowsind['F']

    for i in range(len(df['a0'])):
        tt_arr, signal = load_trace(f, i)
        df.loc[i, 'a0'] = np.mean(signal[slice(*sorted([indA, indB]))])
        df.loc[i, 'a1'] = np.mean(signal[slice(*sorted([indC, indD]))])
        df.loc[i, 'a2'] = np.mean(signal[slice(*sorted([indE, indF]))])

    return df


def recompute_windows_with_background(
    df: pd.DataFrame,
    f,
    windowsind: dict,
    model,
    splitpoint: int,
) -> pd.DataFrame:
    """Recompute a0, a1, a2 with background subtraction using a model.

    The model must have a .predict() method that takes a 2D array.
    """
    df = df.copy()
    indA = windowsind['A']
    indB = windowsind['B']
    indC = windowsind['C']
    indD = windowsind['D']
    indE = windowsind['E']
    indF = windowsind['F']

    for i in range(len(df['a0'])):
        _, signal = load_trace(f, i)
        background_reference = signal[:splitpoint]
        background_predicted = model.predict(np.array([background_reference]))[0]
        background_predicted = np.concatenate(
            (background_reference, background_predicted)
        )
        signal = signal - background_predicted
        df.loc[i, 'a0'] = np.mean(signal[indA:indB])
        df.loc[i, 'a1'] = np.mean(signal[indC:indD])
        df.loc[i, 'a2'] = np.mean(signal[indE:indF])

    return df
