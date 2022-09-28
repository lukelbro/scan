from dataclasses import dataclass, field
from typing import List
import h5py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from functools import cache, singledispatch

@dataclass
class Scan:
    """Data Structure object for H5 scan files from the E11 lab."""
    filename: str
    power: float = 0
    efield: int = 0
    detuning: float = 0
    scanfreq: float = 0
    scantype: str = 'default'

    freq: np.ndarray = np.array([0]) # Array of frequencies from h5 file
    data: np.ndarray = np.array([0]) # Array of data point from h5 file
    error: np.ndarray = np.array([0])
    baseValue: float = 0

    def __post_init__(self):
        # Load data from hdf5 file
        try:
            f = h5py.File(self.filename, 'r')
            dset = f['analysis']
        except:
            raise ValueError('File or dataset not found')
            
        # Load data into Pandas data frame
        df = pd.DataFrame.from_records(dset, columns=dset.dtype.fields.keys())
        # Generate signal data from windows
        df['signal'] =  -(df['a0'] - df['a1'])/((df['a0'] - df['a1']) + (df['a0'] - df['a2']))
        df['error'] = np.sqrt(np.abs(df['signal']) * (1 - np.abs(df['signal']))/100)
        df['error2'] = df['error']**2
        
        #Group data points by frequency (v0) and calculate mean
        dfmean = df.groupby(['v0']).mean()
        dfmean['error_final'] = np.sqrt(dfmean['error2'])/3
        dfmean = dfmean.sort_values('v0')

        # Calculate baseline value from first 10 data points
        if self.scantype == 'default':
            self.baseValue = dfmean['signal'].take(np.arange(0,10)).mean()
        elif self.scantype == 'time':
            self.baseValue = list(dfmean['signal'])[0]
            # Calculate error using bernoulli trials (assuming three loops of 100)
            df['signal'] = df['signal'] - self.baseValue
            df['error'] = np.sqrt(np.abs(df['signal']) * (1 - np.abs(df['signal']))/100)
            df['error2'] = df['error']**2
            dfmean = df.groupby(['v0']).mean()
            dfmean['error_final'] = np.sqrt(dfmean['error2'])/3
            dfmean = dfmean.sort_values('v0')
            self.baseValue = 0
        else:
            raise ValueError(f'Scan type {self.scantype} is not recognised')

        self.data = np.array((dfmean['signal'] - self.baseValue).to_list())
        self.freq = np.array(dfmean['signal'].keys().to_list())
        self.error = np.array(dfmean['error_final'].to_list())
        self._freq_orignal = self.freq.copy()
        self._data_orignal = self.data.copy()
        self._error_orignal = self.error.copy()

        self.gauss = Gauss(self)
        self.rabi = Rabi(self)
    
    def set_range(self, range):
        "Select subset of data based on values of freq"
        start = range[0]
        end = range[1]

        self.freq = self._freq_orignal.copy()[start:end]
        self.data = self._data_orignal.copy()[start:end]
        self.error = self._error_orignal.copy()[start:end]


    def savefile(self, dir):
        return

    def get_dataframe(self):
        """Auxillary function for unit testing.

        Returns:
            Pandas DataFrame: data frame where signal is analysis column. Baseline value has
            not been subtracted and frequency values have not been grouped/averaged.
        """
        # Load data from hdf5 file
        f = h5py.File(self.filename, 'r')
        dset = f['analysis']
        # Load data into Pandas data frame
        df = pd.DataFrame.from_records(dset, columns=dset.dtype.fields.keys())
        # Generate signal data from windows
        df['signal'] =  -(df['a0'] - df['a1'])/((df['a0'] - df['a1']) + (df['a0'] - df['a2']))
        return df


class abstract_fitting:
    def __init__(self):
        self._p0 : np.ndarray
        self._varMatrix : np.ndarray
        self.scan : Scan
        self.guess : np.ndarray
        self.bounds = 0
        self.sigma = []
        self.fitdone = False
    
    @staticmethod
    def func(x):
        return
    
    def fit(self, userGuess=None):
        # Fit gaussian function
        if userGuess != None:
            self.guess = userGuess
        scan = self.scan
        p = self.guess
        if len(self.sigma) >  0:
            self._p0, self._varMatrix = curve_fit(self.func, scan.freq, scan.data, p0=p, absolute_sigma=True, sigma=scan.error)
        else:
            if self.bounds == 0:
                self._p0, self._varMatrix = curve_fit(self.func, scan.freq, scan.data, p0=p, absolute_sigma=False)
            else:
                self._p0, self._varMatrix = curve_fit(self.func, scan.freq, scan.data, p0=p, bounds=self.bounds, absolute_sigma=False)  
    def p0(self):
        self.fit()
        return self._p0
    
    def varMatrix(self):
        self.fit()
        return self._varMatrix


class Gauss(abstract_fitting):
    def __init__(self, scan):
        super().__init__()
        self.guess = [np.max(scan.freq), scan.freq[np.argmax(scan.data)], 0.0001]
        self.scan = scan

    @staticmethod
    def func(x, *p):
        A, mu, sigma = p
        return A * np.exp( - (x-mu)**2/(2.*sigma**2))

class Rabi(abstract_fitting):
    def __init__(self, scan):
        super().__init__()
        self.scan = scan
        self.sigma = self.scan.error
        fftGuess = self.fft_guess()
        self.guess = [fftGuess["omega"], 0, fftGuess["amp"], fftGuess["offset"]]
        
        #self.bounds = [[0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]]
    
    def fft_guess(self):
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = self.scan.freq
        yy = self.scan.data
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
        return {"amp": guess_amp, "omega": guess_freq, "offset": guess_offset}
        
    @staticmethod
    def func(x, *p):
        omega, decay, a, c = p
        return -a*np.cos(x*omega)*np.exp(-decay*x)+c
    

if __name__ == '__main__':
    filepath = 'analysis/07/21/20210721_009/20210721_009_scan.h5'
    sc = Scan(filepath)
    print(sc.rabi.p0())
    filepath = 'analysis/07/22/20210722_038/20210722_038_scan.h5'
    sc = Scan(filepath)
    sc.set_range([0,47])
    print(sc.rabi.p0()[0])