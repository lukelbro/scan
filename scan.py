from dataclasses import dataclass, field
from typing import List
from xml.sax.saxutils import XMLFilterBase
import h5py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from functools import cache, singledispatch
from  colorama import Fore

@dataclass
class Scan:
    """Data Structure object for H5 scan files from the E11 lab."""
    filename: str
    function: str

    experiment: str = 'generic' # 'generic', 'volt', 'microwave', 'time'
    power: float = None
    efield: int = None
    detuning: float = None
    scanfreq: float = None
    timestamp: np.datetime64 = None

    x: np.ndarray = np.array([0]) # Array of frequencies from h5 file
    y: np.ndarray = np.array([0]) # Array of data point from h5 file
    error: np.ndarray = np.array([0])
    baseValue: float = None


    def __post_init__(self):
        # Load data from hdf5 file
        try:
            self.f = h5py.File(self.filename, 'r')
            self.dset = self.f['analysis']
        except:
            raise ValueError('File or dataset not found')
        
        # Read locations of windows
        self.windows = {}
        windows = ['A', 'B', 'C', 'D', 'E', 'F']
        for window in windows:
            self.windows[window] = self.dset.attrs[window]

        # Read time stamp
        self.timestamp = np.datetime64(self.f.attrs['timestamp'])
        
        # Read number of loops
        self.numloops = self.f.attrs['v0_loops']

        # Read number of measurments
        self.nummeasurements = self.f.attrs['v0_num']

        # read experiment type from metadata
        if self.f.attrs['var 0'] == 'microwaves (GHz)':
            self.experiment = 'microwave'
        else:
            self.experiment = 'generic'
            print(f'{Fore.RED}WARNING{Fore.RED}: New type of experiment {Fore.RED}{self.experiment}{Fore.RESET} using settings for generic')
     
        # Load data into Pandas data frame
        self.df = pd.DataFrame.from_records(self.dset, columns=self.dset.dtype.fields.keys())

        # Generate signal data from windows
        self.evaluate_windows()
        
        # Group data points by x (v0) and calculate mean, and apply baseline if appropiate
        self.process_signal()

        # load fitting routines (does not do fit now)
        self.gauss = Gauss(self)
        self.rabi = Rabi(self)

    def evaluate_windows(self):
        # convert input string i.e a0 + a1 into something python can evaluate
        functionString = Scan.function_parser(self.function)
        # evaluate windows
        self.df['signal'] = eval(functionString)

    def function_parser(function):
        functionString = function
        functionString = functionString.replace('a0', "self.df['a0']")
        functionString = functionString.replace('a1', "self.df['a1']")
        functionString = functionString.replace('a2', "self.df['a2']")
        return functionString

    def process_signal(self):
        dfmean = self.df.groupby(['v0']).mean()
        dfmean = dfmean.sort_values('v0')
        # Apply baseline
        if self.experiment == 'microwave':
            self.baseValue = dfmean['signal'].take(np.arange(0,10)).mean()
        elif self.experiment == 'time':
            self.baseValue = dfmean['signal'].take(0)
        elif self.experiment == 'generic':
            self.baseValue = 0
        else:
            raise ValueError(f'Experiment {self.experiment} not recognised')
    
        self.y = np.array((dfmean['signal'] - self.baseValue).to_list())
        self.x = np.array(dfmean['signal'].keys().to_list())
        self._x_orignal = self.x.copy()
        self._y_orignal = self.y.copy()

    def update_function(self, function):
        pass

    def set_range(self, range):
        "Select subset of y based on values of x"
        start = range[0]
        end = range[1]

        self.x = self._x_orignal.copy()[start:end]
        self.y = self._y_orignal.copy()[start:end]
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
            self._p0, self._varMatrix = curve_fit(self.func, scan.x, scan.y, p0=p, absolute_sigma=True, sigma=scan.error)
        else:
            if self.bounds == 0:
                self._p0, self._varMatrix = curve_fit(self.func, scan.x, scan.y, p0=p, absolute_sigma=False)
            else:
                self._p0, self._varMatrix = curve_fit(self.func, scan.x, scan.y, p0=p, bounds=self.bounds, absolute_sigma=False)  
    def p0(self):
        self.fit()
        return self._p0
    
    def varMatrix(self):
        self.fit()
        return self._varMatrix

class Gauss(abstract_fitting):
    def __init__(self, scan):
        super().__init__()
        self.guess = [np.max(scan.x), scan.x[np.argmax(scan.y)], 0.0001]
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
        tt = self.scan.x
        yy = self.scan.y
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