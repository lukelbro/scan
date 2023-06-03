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
        assert sc.gauss.p0[i] == approx(coeff[i], abs=1e-8)

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
