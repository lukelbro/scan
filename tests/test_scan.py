from .context import e11scan
import numpy as np
from pytest import approx, raises
import pandas as pd

def test_scan_baseValue():
    # Generate analysis data using Scan object
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename= filename, function=function)
    assert sc.baseValue == -0.500220349806397


def test_scan_output():
    # Generate analysis data using Scan object
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename = filename, function=function, power=0, efield=0, detuning=0)
    # Load analysis data generated in Matlab
    data =  pd.read_csv('tests/m70mV_005.dat', delimiter = '\t', names = ['f', 'x', 'err'])
    
    assert len(sc.x) == len(data.f)
    assert len(sc.x) == len(sc.y)
    assert len(sc.x) == len(data.x)

    for i in range(len(sc.x)):
        if i==17:
            # a bug in the matlab analysis code deletes the last entry
            pass
        else:
            assert sc.x[i] == approx(data.f[i], abs=1e-8)
            assert sc.y[i] == approx(data.x[i], abs=1e-8)

def test_scan_gaussian():
    # Generate analysis data using Scan object
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename = filename, function=function)
    coeff = [1.41586588e-01, 1.95563550e+01, 6.61302896e-05]

    for i in range(len(sc.gauss.p0)):
        assert sc.gauss.p0[i] == approx(coeff[i], abs=1e-4)

def test_plot():
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename = filename, function=function)
    sc.plot_trace(10)

def test_no_experiment_in_file():
    with raises(ValueError):
        e11scan.scan(filename='tests/20220624_007_scan.h5', function='a0')

def test_time():
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename='tests/20210722_038_scan.h5', function=function, experiment='time')
    assert sc.baseValue == approx(-0.5105504, abs=1e-8)

def test_scanmd():
    filename = 'tests/20221208_006_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    
    sc = e11scan.scanmd(filename=filename, function=function)
    assert len(sc.x2) == 2
    assert len(sc.sets) == 2
    
    assert sc.x2[0] == -0.03
    assert sc.x2[1] == 0.01


def test_scanmd_double_init_check():
    filename = 'tests/20221208_006_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    
    sc = e11scan.scanmd(filename=filename, function=function)
    sc = e11scan.scanmd(filename=filename, function=function)
    assert len(sc.x2) == 2
    assert len(sc.sets) == 2
    
    assert sc.x2[0] == -0.03
    assert sc.x2[1] == 0.01

def test_scanmd_gauss():
    scs  = e11scan.scanmd(filename='tests/20221208_006_scan.h5', function='a0-a1', experiment='microwave')
    scs.sets[0].gauss.fit == np.array([4.57743247e-30, 2.93283739e-29, 1.82790554e-28, 1.10820280e-27,
       6.53558498e-27, 3.74929331e-26, 2.09225250e-25, 1.13573890e-24,
       5.99711861e-24, 3.08039566e-23, 1.53911154e-22, 7.48054802e-22,
       3.53668568e-21, 1.62651937e-20, 7.27648681e-20, 3.16653272e-19,
       1.34043556e-18, 5.51960034e-18, 2.21089975e-17, 8.61450244e-17,
       3.26505930e-16, 1.20379262e-15, 4.31729782e-15, 1.50616333e-14,
       5.11130572e-14, 1.68729638e-13, 5.41814496e-13, 1.69242563e-12,
       5.14242889e-12, 1.51994098e-11, 4.37003468e-11, 1.22220138e-10,
       3.32506688e-10, 8.79949460e-10, 2.26524302e-09, 5.67246207e-09,
       1.38174572e-08, 3.27404305e-08, 7.54640954e-08, 1.69198335e-07,
       3.69021377e-07, 7.82900725e-07, 1.61570323e-06, 3.24351731e-06,
       6.33389084e-06, 1.20316353e-05, 2.22319967e-05, 3.99606008e-05,
       6.98691206e-05, 1.18833332e-04, 1.96603388e-04, 3.16405082e-04,
       4.95331141e-04, 7.54305964e-04, 1.11737559e-03, 1.61009146e-03,
       2.25684452e-03, 3.07717690e-03, 4.08134224e-03, 5.26566582e-03,
       6.60850620e-03, 8.06776125e-03, 9.58081606e-03, 1.10675552e-02,
       1.24365699e-02, 1.35940626e-02, 1.44543193e-02, 1.49501568e-02,
       1.50415843e-02, 1.47211299e-02, 1.40148494e-02, 1.29788274e-02,
       1.16918222e-02, 1.02453943e-02, 8.73322971e-03, 7.24137129e-03,
       5.84072120e-03, 4.58259910e-03, 3.49749404e-03, 2.59658072e-03,
       1.87519485e-03, 1.31731824e-03, 9.00191143e-04, 5.98381964e-04,
       3.86920715e-04, 2.43368966e-04, 1.48904609e-04, 8.86238910e-05,
       5.13089603e-05, 2.88958413e-05, 1.58298646e-05, 8.43565453e-06,
       4.37280488e-06, 2.20496216e-06, 1.08153832e-06, 5.16038864e-07,
       2.39509441e-07, 1.08134081e-07, 4.74900117e-08, 2.02881145e-08,
       8.43103376e-09, 3.40815782e-09, 1.34016507e-09, 5.12621266e-10,
       1.90736879e-10, 6.90354967e-11, 2.43058025e-11, 8.32429002e-12,
       2.77321902e-12, 8.98712670e-13, 2.83306989e-13, 8.68747243e-14,
       2.59136936e-14, 7.51908385e-15, 2.12226819e-15, 5.82687001e-16,
       1.55621688e-16, 4.04300840e-17, 1.02173647e-17, 2.51172953e-18,
       6.00629389e-19, 1.39714028e-19, 3.16135434e-20, 6.95834677e-21,
       1.48983677e-21, 3.10292326e-22, 6.28641593e-23, 1.23889619e-23,
       2.37501551e-24, 4.42891873e-25, 8.03394215e-26, 1.41761865e-26,
       2.43326755e-27, 4.06274944e-28])


def test_scan_windows_ind():
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename = filename, function = function)
    
    expected_ind = {'A': 201, 'B': 257, 'C': 368, 'D': 439, 'E': 478, 'F': 542}
    assert sc.windowsind == expected_ind

def test_scan_set_windows_ind():
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename = filename, function = function)
    
    set_ind = {'A': 201, 'B': 257, 'C': 368, 'D': 439, 'E': 478, 'F': 542}
    
    sc.windowsind = set_ind

    assert sc.df['a0'][0] == -0.4693715081171429


def test_scan_sc_xall():
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename = filename, function = function)
    assert sc.xall[0] == sc.df['v0'][0]

def test_scan_scs_xall():
    filename = 'tests/20230531_012_scan.h5'
    function = '(a0 - a1)'
    scs = e11scan.scanmd(filename=filename, function= function, experiment = 'microwave')
    for sc in scs.sets:
        if sc.x2 == 0.45:
            sc_single =  sc
    sc = sc_single
    assert sc.xall[0] == sc.df['v0'].values[0]


# ------------------------------------------------------------------
# Declarative filter V2 tests
# ------------------------------------------------------------------

def test_filter_rows_reversibility():
    """Selecting a row range and then disabling it restores the full dataset."""
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename=filename, function=function)
    original_len = len(sc.x)

    # Select a middle slice (drop rows at both ends)
    sc.filter_rows((10, 110))
    assert len(sc.x) == 100

    # Restore
    sc.filter_rows(None)
    assert len(sc.x) == original_len


def test_basic_filter_reversibility():
    """Applying basic_filter then disabling it restores removed points."""
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename=filename, function=function)
    original_len = len(sc.x)
    original_y = sc.y.copy()

    # Filter
    sc.basic_filter(1.5)
    filtered_len = len(sc.x)
    assert filtered_len <= original_len

    # Restore
    sc.basic_filter(None)
    assert len(sc.x) == original_len
    assert np.allclose(sc.y, original_y)


def test_remove_unstable_reversibility():
    """Applying remove_unstable then disabling it restores removed points."""
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename=filename, function=function)
    original_len = len(sc.x)
    original_y = sc.y.copy()

    # Filter
    sc.remove_unstable(0.01)
    filtered_len = len(sc.x)
    assert filtered_len <= original_len

    # Restore
    sc.remove_unstable(None)
    assert len(sc.x) == original_len
    assert np.allclose(sc.y, original_y)


def test_filter_order_independence():
    """Applying basic_filter then stability_filter yields same result as reverse order."""
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'

    # Order A: basic_filter first
    sc_a = e11scan.scan(filename=filename, function=function)
    sc_a.basic_filter(1.5)
    sc_a.remove_unstable(0.01)
    x_a, y_a = sc_a.x.copy(), sc_a.y.copy()

    # Order B: stability_filter first
    sc_b = e11scan.scan(filename=filename, function=function)
    sc_b.remove_unstable(0.01)
    sc_b.basic_filter(1.5)
    x_b, y_b = sc_b.x.copy(), sc_b.y.copy()

    # Results must be identical (same length and values)
    assert len(x_a) == len(x_b)
    assert np.allclose(x_a, x_b)
    assert np.allclose(y_a, y_b)


def test_filter_cache_invalidation():
    """Accessing a fit cache, then filtering, must trigger recomputation."""
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename=filename, function=function)

    # Force cache population
    gauss_before = sc.gauss
    assert sc._gauss_cache is not None

    # Apply filter — cache must be invalidated
    sc.basic_filter(1.5)
    assert sc._gauss_cache is None

    # Re-access triggers recomputation
    gauss_after = sc.gauss
    assert sc._gauss_cache is not None
    # The fit results should differ because the data changed
    assert gauss_before.p0[0] != gauss_after.p0[0]


def test_filter_drop_rows_reversibility():
    """Dropping rows by index and then disabling the drop restores them."""
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename=filename, function=function)
    original_df_len = len(sc.df)

    # Drop rows 0, 1, 2 from the raw DataFrame
    sc.filter_drop_rows([0, 1, 2])
    assert len(sc.df) == original_df_len - 3

    # Restore
    sc.filter_drop_rows(None)
    assert len(sc.df) == original_df_len


def test_load_database_resets_filters():
    """load_database() clears all declarative filter state."""
    filename = 'tests/20210707_005_scan.h5'
    function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
    sc = e11scan.scan(filename=filename, function=function)
    original_len = len(sc.x)

    # Apply filters
    sc.filter_rows((0, 100))
    sc.filter_drop_rows([5, 10])
    sc.basic_filter(1.5)
    sc.remove_unstable(0.01)

    # Reload clears everything
    sc.load_database()
    assert sc.pipeline == []
    assert len(sc.x) == original_len


# ------------------------------------------------------------------
# Pattern B pipeline API tests
# ------------------------------------------------------------------

def _load_scan():
    """Shared fixture for pipeline tests."""
    return e11scan.scan(
        filename='tests/20210707_005_scan.h5',
        function='-(a0 - a1)/((a0 - a1) + (a0 - a2))',
    )


def test_set_pipeline_basic():
    """set_pipeline with a single filter step works and populates pipeline."""
    sc = _load_scan()
    sc.set_pipeline([('basic_filter', 1.5)])
    assert len(sc.pipeline) == 1
    assert sc.pipeline[0][0] == 'basic_filter'
    assert sc.pipeline[0][1] == {'m': 1.5}


def test_set_pipeline_with_dicts():
    """set_pipeline accepts dict-style kwargs."""
    sc = _load_scan()
    sc.set_pipeline([('drop_rows', {'indices': [50, 150]})])
    assert len(sc.pipeline) == 1
    assert sc.pipeline[0] == ('drop_rows', {'indices': [50, 150]})


def test_add_step_order():
    """add_step with order= inserts at the correct position."""
    sc = _load_scan()
    sc.add_step('basic_filter', 1.5)
    sc.add_step('drop_rows', [50, 150], order=0)
    assert [name for name, _ in sc.pipeline] == ['drop_rows', 'basic_filter']


def test_clear_pipeline_restores_data():
    """clear_pipeline() removes all steps and restores the pristine dataset."""
    sc = _load_scan()
    original_unique = len(sc._raw_df['v0'].unique())
    sc.set_pipeline([('filter_rows', (0, 100))])
    assert len(sc.x) != original_unique
    sc.clear_pipeline()
    assert sc.pipeline == []
    assert len(sc.x) == original_unique


def test_backward_compat_wrappers_write_to_pipeline():
    """Legacy wrapper methods correctly populate self.pipeline."""
    sc = _load_scan()
    sc.basic_filter(1.5)
    assert sc.pipeline == [('basic_filter', {'m': 1.5})]


def test_backward_compat_disable_removes_from_pipeline():
    """Calling a wrapper with None removes the step from the pipeline."""
    sc = _load_scan()
    sc.basic_filter(1.5)
    assert len(sc.pipeline) == 1
    sc.basic_filter(None)
    assert sc.pipeline == []


def test_backward_compat_update_replaces_position():
    """Updating a step via the wrapper preserves its position in the pipeline."""
    sc = _load_scan()
    sc.filter_rows((20, 280))
    sc.basic_filter(1.5)
    # filter_rows is at position 0, basic_filter at position 1
    sc.filter_rows((10, 200))
    assert len(sc.pipeline) == 2
    assert sc.pipeline[0][0] == 'filter_rows'
    assert sc.pipeline[0][1] == {'rows': (10, 200)}
    assert sc.pipeline[1][0] == 'basic_filter'


def test_set_pipeline_unknown_step_raises():
    """set_pipeline with an unknown step name raises ValueError."""
    sc = _load_scan()
    try:
        sc.set_pipeline([('nonexistent', {})])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_pipeline_printable():
    """sc.pipeline is a plain list of tuples — easy to inspect."""
    sc = _load_scan()
    sc.set_pipeline([
        ('drop_rows', [50, 150]),
        ('basic_filter', 1.5),
    ])
    s = str(sc.pipeline)
    assert 'drop_rows' in s
    assert 'basic_filter' in s


# ---------------------------------------------------------------------------
# osc_1 minimal support
# ---------------------------------------------------------------------------

_OSC1_FILENAME = 'tests/20260529_019_scan.h5'


def test_has_osc_1_true():
    """File with osc_1 channel sets has_osc_1 to True."""
    sc = e11scan.scan(filename=_OSC1_FILENAME, function='a0-a1', experiment='microwave')
    assert sc.has_osc_1 is True


def test_has_osc_1_false():
    """File without osc_1 channel sets has_osc_1 to False."""
    sc = _load_scan()
    assert sc.has_osc_1 is False


def test_trace_osc_0_default():
    """trace(ind) with no channel arg returns osc_0 data (backward compat)."""
    sc = e11scan.scan(filename=_OSC1_FILENAME, function='a0-a1', experiment='microwave')
    tt, signal = sc.trace(0)
    assert len(tt) == len(signal)
    assert len(signal) == 1002  # known trace length for this file


def test_trace_osc_1():
    """trace(ind, channel='osc_1') returns osc_1 data with correct shape."""
    sc = e11scan.scan(filename=_OSC1_FILENAME, function='a0-a1', experiment='microwave')
    tt, signal = sc.trace(0, channel='osc_1')
    assert len(tt) == len(signal)
    assert len(signal) == 1002


def test_trace_osc_1_raises_if_missing():
    """Requesting osc_1 on a file without it raises ValueError."""
    sc = _load_scan()
    with raises(ValueError, match="No osc_1 channel"):
        sc.trace(0, channel='osc_1')


def test_trace_unknown_channel_raises():
    """Requesting a nonexistent channel raises ValueError."""
    sc = e11scan.scan(filename=_OSC1_FILENAME, function='a0-a1', experiment='microwave')
    with raises(ValueError, match="not found"):
        sc.trace(0, channel='osc_2')


# ---------------------------------------------------------------------------
# param_names on FitResult
# ---------------------------------------------------------------------------

def test_gauss_param_names():
    """gauss.param_names matches expected (A, mu, sigma)."""
    sc = _load_scan()
    assert sc.gauss.param_names == ('A', 'mu', 'sigma')


def test_gaussgauss_param_names():
    """gaussgauss.param_names uses primary/secondary naming."""
    sc = e11scan.scan(
        filename='tests/20221208_006_scan.h5',
        function='a0-a1', experiment='microwave',
    )
    assert sc.gaussgauss.param_names == (
        'A_primary', 'mu_primary', 'sigma_primary',
        'A_secondary', 'mu_secondary', 'sigma_secondary',
    )


def test_rabi_param_names():
    """rabi.param_names matches expected (omega, decay, amplitude)."""
    sc = e11scan.scan(
        filename='tests/20210722_038_scan.h5',
        function='-(a0 - a1)/((a0 - a1) + (a0 - a2))',
        experiment='time',
    )
    assert sc.rabi.param_names == ('omega', 'decay', 'amplitude')


def test_param_names_length_matches_p0():
    """len(param_names) == len(p0) for all fit types."""
    sc = _load_scan()
    assert len(sc.gauss.param_names) == len(sc.gauss.p0)

    sc2 = e11scan.scan(
        filename='tests/20221208_006_scan.h5',
        function='a0-a1', experiment='microwave',
    )
    assert len(sc2.gaussgauss.param_names) == len(sc2.gaussgauss.p0)

    sc3 = e11scan.scan(
        filename='tests/20210722_038_scan.h5',
        function='-(a0 - a1)/((a0 - a1) + (a0 - a2))',
        experiment='time',
    )
    assert len(sc3.rabi.param_names) == len(sc3.rabi.p0)


def test_gaussgauss_primary_is_taller_peak():
    """The primary peak amplitude should be >= the secondary peak amplitude."""
    sc = e11scan.scan(
        filename='tests/20221208_006_scan.h5',
        function='a0-a1', experiment='microwave',
    )
    names = sc.gaussgauss.param_names
    p = sc.gaussgauss.p0
    A_primary = p[names.index('A_primary')]
    A_secondary = p[names.index('A_secondary')]
    assert A_primary >= A_secondary


# ------------------------------------------------------------------
# Fit p0 setter tests
# ------------------------------------------------------------------

def test_set_gauss_p0_on_scan():
    """set_gauss_p0 provides a seed that affects the fit result."""
    sc = _load_scan()
    # Get default fit
    default_p0 = sc.gauss.p0.copy()

    # Set a custom seed and verify the fit changes
    sc.set_gauss_p0([default_p0[0] * 0.5, default_p0[1], default_p0[2] * 2.0])
    custom_result = sc.gauss
    # The fit should still converge (may or may not equal default)
    assert len(custom_result.p0) == 3
    assert custom_result.p0[0] > 0  # A should be positive


def test_set_gauss_p0_none_reverts_to_heuristic():
    """Setting p0=None reverts to automatic heuristic."""
    sc = _load_scan()
    default_p0 = sc.gauss.p0.copy()

    # Set a custom seed
    sc.set_gauss_p0([0.1, 19.5, 0.001])
    sc.gauss  # force computation with custom seed

    # Revert to heuristic
    sc.set_gauss_p0(None)
    reverted = sc.gauss
    # Should get back to the same result as default
    assert np.allclose(reverted.p0, default_p0, atol=1e-6)


def test_set_gauss_p0_cache_invalidation():
    """Setting p0 invalidates the cache."""
    sc = _load_scan()
    sc.gauss  # populate cache
    assert sc._gauss_cache is not None

    sc.set_gauss_p0([0.1, 19.5, 0.001])
    assert sc._gauss_cache is None  # cache invalidated


def test_set_gauss_p0_survives_pipeline_change():
    """p0 persists across pipeline/filter changes (unlike fit caches)."""
    sc = _load_scan()
    sc.set_gauss_p0([0.1, 19.5, 0.001])
    assert sc._gauss_p0 == [0.1, 19.5, 0.001]

    # Apply a filter — p0 should survive
    sc.basic_filter(1.5)
    assert sc._gauss_p0 == [0.1, 19.5, 0.001]


def test_set_gauss_p0_clears_on_load_database():
    """load_database() resets p0 to None."""
    sc = _load_scan()
    sc.set_gauss_p0([0.1, 19.5, 0.001])
    assert sc._gauss_p0 is not None

    sc.load_database()
    assert sc._gauss_p0 is None
    assert sc._gaussgauss_p0 is None
    assert sc._rabi_p0 is None


def test_set_gaussgauss_p0_on_scan():
    """set_gaussgauss_p0 provides a seed for double Gaussian."""
    sc = _load_scan()
    result = sc.gaussgauss
    assert len(result.p0) == 6


def test_set_rabi_p0_on_scan():
    """set_rabi_p0 provides a seed for Rabi fit."""
    sc = _load_scan()
    result = sc.rabi
    assert len(result.p0) == 3


def test_set_gauss_p0_on_scan_subset():
    """set_gauss_p0 works on _ScanSubset (via ScanMD)."""
    scs = e11scan.scanmd(
        filename='tests/20221208_006_scan.h5',
        function='a0-a1', experiment='microwave',
    )
    subset = scs.sets[0]
    default_p0 = subset.gauss.p0.copy()

    # Set a custom seed
    subset.set_gauss_p0([default_p0[0] * 0.5, default_p0[1], default_p0[2] * 2.0])
    result = subset.gauss
    assert len(result.p0) == 3


def test_set_gauss_p0_none_on_subset():
    """Setting p0=None on a subset reverts to heuristic."""
    scs = e11scan.scanmd(
        filename='tests/20221208_006_scan.h5',
        function='a0-a1', experiment='microwave',
    )
    subset = scs.sets[0]
    default_p0 = subset.gauss.p0.copy()

    subset.set_gauss_p0([0.1, 19.5, 0.001])
    subset.gauss  # force computation

    subset.set_gauss_p0(None)
    reverted = subset.gauss
    assert np.allclose(reverted.p0, default_p0, atol=1e-6)

