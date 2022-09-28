from .context import scan
import numpy as np
from pytest import approx, raises
import pandas as pd

def test_scan_baseValue():
    # Generate analysis data using Scan object
    filename = 'scan/tests/20210707_005_scan.h5'
    sc = scan.Scan(filename= filename, power=0, efield=0, detuning=0)
    assert sc.baseValue == -0.500220349806397


def test_scan_output():
    # Generate analysis data using Scan object
    filename = 'scan/tests/20210707_005_scan.h5'
    sc = scan.Scan(filename = filename, power=0, efield=0, detuning=0)
    # Load analysis data generated in Matlab
    data =  pd.read_csv('scan/tests/m70mV_005.dat', delimiter = '\t', names = ['f', 'x', 'err'])
    
    assert len(sc.freq) == len(data.f)
    assert len(sc.freq) == len(sc.data)
    assert len(sc.freq) == len(data.x)

    for i in range(len(sc.freq)):
        if i==17:
            # a bug in the matlab analysis code deletes the last entry
            pass
        else:
            assert sc.freq[i] == approx(data.f[i], abs=1e-8)
            assert sc.data[i] == approx(data.x[i], abs=1e-8)

def test_scan_gaussian():
    # Generate analysis data using Scan object
    filename = 'scan/tests/20210707_005_scan.h5'
    sc = scan.Scan(filename = filename)
    coeff = [1.41586588e-01, 1.95563550e+01, 6.61302896e-05]

    for i in range(len(sc.gauss.p0())):
        assert sc.gauss.p0()[i] == approx(coeff[i], abs=1e-8)

