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

    for i in range(len(sc.gauss.p0())):
        assert sc.gauss.p0()[i] == approx(coeff[i], abs=1e-8)

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

