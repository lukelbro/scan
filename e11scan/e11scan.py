from dataclasses import dataclass, field
from typing import List
from xml.sax.saxutils import XMLFilterBase
import h5py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from functools import cache, singledispatch
from colorama import Fore
from matplotlib import pyplot as plt

@dataclass
class scan_base:
    """Data Structure object for H5 scan files from the E11 lab."""
    filename: str 
    function: str
    
    
    experiment: str = None # 'generic', 'volt', 'microwave', 'time'
    power: float = None
    efield: int = None
    detuning: float = None
    scanfreq: float = None
    timestamp: np.datetime64 = None
    x2 : float = None
    df: pd.DataFrame = None
    error: np.ndarray = np.array([0])
    range: list = None
    averages: int = None

    def __post_init__(self):
        self.filterTracker = {'filterBool' : False, 'filters' : {'basicm': None, 'stablem': None }}
        

    def evaluate_windows(self):
        # convert input string i.e a0 + a1 into something python can evaluate
        functionString = scan.function_parser(self.function)
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
        elif self.experiment == 'volt':
            self.baseValue = 0
        elif self.experiment == 'time':
            self.baseValue = dfmean['signal'].take([0]).values
        elif self.experiment == 'generic':
            self.baseValue = 0
        else:
            raise ValueError(f'Experiment {self.experiment} not recognised')
    
        self.y = np.array((dfmean['signal'] - self.baseValue).to_list())
        self.x = np.array(dfmean['signal'].keys().to_list())
        self._x_orignal = self.x.copy()
        self._y_orignal = self.y.copy()

        if self.averages != None:
            self.error = np.array(dfmean['error'])
        
        # Not an efficient solution
        ranges = self.df.groupby(['v0']).agg({'signal': [np.min, np.max, np.mean]})
        ranges = ranges.sort_values('v0')
        signalmin = ranges['signal']['amin']
        signalmax = ranges['signal']['amax']
        signalmean = ranges['signal']['mean']

        self.range = [np.abs(signalmin - signalmean), np.abs(signalmax - signalmean)]

        # load fitting routines (does not do fit now)
        self.gauss = Gauss(self)
        self.rabi = Rabi(self)
    
    def filter_manager(self, customfunction = 'a0-a1'):
        if self.filterTracker['filterBool'] == False:
            self.df_spare = self.df.copy()
            self.filterTracker['filterBool'] = True
        
        self.df = self.df_spare.copy()
        df = self.df

        ids = []

        for fname in self.filterTracker['filters'].keys():
            m = self.filterTracker['filters'][fname]
            if m != None:
                if fname == 'basicm':
                    ids += self.__basic_filter(m)
                if fname == 'stablem':
                    ids += self.__remove_unstable(m, customfunction=customfunction)
                
        
        ids = list(set(ids))
        
        self.df.drop(ids, axis=0, inplace=True)
        self.process_signal()
        
    def basic_filter(self, m):
        self.filterTracker['filters']['basicm'] = m
        self.filter_manager()

    def __basic_filter(self, m):
        df = self.df
        idrop = []
        for v0 in np.array(df['v0']):
            dfi = df[df['v0'] == v0]
            signal = dfi['signal']
            signal.sort_values()
            d1 = np.abs(signal.iloc[0] - signal.iloc[1])
            d2 = np.abs(signal.iloc[1] - signal.iloc[2])

            if d1 > m*d2:
                idrop.append(signal.index[0])

            if d2 > m*d1:  
                idrop.append(signal.index[2])        
        return idrop

    def remove_unstable(self, m, customfunction = 'a0-a1'):
        self.filterTracker['filters']['stablem'] = m
        self.filter_manager(customfunction)

    def __remove_unstable(self, threshold,  customfunction = 'a0-a1'):
        functionstring = scan_base.function_parser(customfunction)
        stability = eval(functionstring)
        idrop = []
        
        for index, value in stability.items():
            if value > threshold:
                idrop.append(index)
        return idrop
        
    def plot_stability(self, hline = None, customfunction = 'a0-a1'):
        """Plots the stability of the signal from a0 - a1

        Args:
            customfunction (str, optional): Option for custom function. Defaults to 'a0-a1'.
        """
        plt.clf()
        functionstring = scan_base.function_parser(customfunction)
        stability = eval(functionstring)

        plt.scatter(np.linspace(0, stability.shape[0], stability.shape[0]), stability, s=1)
        plt.xlabel('measurement number')
        if hline != None:
            plt.hlines(hline, 0, stability.shape[0])
        plt.ylabel(customfunction)

    def trace(self, ind):
        if ind>len(self.f['osc_0']):
            raise ValueError(f"Number of scans is {len(self.f['osc_0'])}, {ind} is outside range")
        signal = self.f['osc_0'][ind]
        t0 = self.f['osc_0'].attrs['t0']
        dt = self.f['osc_0'].attrs['dt']
        tt = np.linspace(t0, t0+dt*len(signal), len(signal))
    
        return tt, signal

    def update_function(self, function):
        self.function = function
        self.evaluate_windows()
        self.process_signal()
    
    def update_experiment(self, experiment):
        self.experiment = experiment
        self.evaluate_windows()
        self.process_signal()

    def set_range(self, range):
        "Select subset of y based on values of x"
        start = range[0]
        end = range[1]
        self.x = self._x_orignal.copy()[start:end]
        self.y = self._y_orignal.copy()[start:end]
        #self.error = self._error_orignal.copy()[start:end]


class scan(scan_base):
    def __post_init__(self):
        super().__post_init__()
        self.x =  np.array([0]) # Array of frequencies from h5 file
        self.y = np.array([0]) # Array of data point from h5 file
        self.error = np.array([0])
        self.baseValue = None

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

        # read experiment type from metadata
        if self.experiment == None:
            try: 
                self.f.attrs['var 0']
            except:
                raise ValueError('Could not deduce experiment type, please set experiment type manually')
            if self.f.attrs['var 0'] == 'microwaves (GHz)':
                self.experiment = 'microwave'
            else:
                self.experiment = 'generic'
                print(f'{Fore.RED}WARNING{Fore.RED}: New type of experiment {Fore.RED}{self.experiment}{Fore.RESET} using settings for generic')
        else:
            if self.experiment not in ['microwave', 'volt', 'time']:
                print(f'{Fore.RED}WARNING{Fore.RESET}: New type of experiment {Fore.RED}{self.experiment}{Fore.RESET} using settings for generic')
                self.experiment = 'generic'
        
        
        self.build_database()


    def build_database(self):
        self.df = pd.DataFrame.from_records(self.dset, columns=self.dset.dtype.fields.keys())
        # Check if data is multidimensional.
        x2 =  list(set(self.df['v1']))
        if len(x2) > 1:
            print('Multidimensional scan detected use scanmd')
    
        # Generate signal data from windows
        self.evaluate_windows()
        # Calculate error on db
        self.calculate_signal_error()
        # Group data points by x (v0) and calculate mean, and apply baseline if appropiate
        self.process_signal()

    def filter(self, m):
        self.load_database()
        self.evaluate_windows()
        self.basic_filter(m)
        self.build_database()


    def calculate_signal_error(self):
        """Calculate error of data points using standard error and error propogation
        """
        # Check if averages have been given
        if self.averages != None:
            df = self.df
            df['error'] = np.sqrt(0.25/self.averages) * 1/(np.sqrt(self.numloops))
            # 0.25 comes from the maximum error from bernoulli trials (so an overestimate of the error - seeems reasonable-ish)
    
    def plot_trace(self, ind):
        tt, signal  = self.trace(ind)
        plt.plot(tt, signal)
        plt.xlabel('time')
        plt.ylabel('signal')

        minval = np.min(signal)
        maxval = np.max(signal)
        color = {'A': 'tab:pink', 'B':'tab:green', 'C': 'tab:red', 'D':'tab:orange', 'E':'tab:purple', 'F':'tab:olive'}
        for key in self.windows.keys():
            plt.vlines(self.windows[key]+self.f['osc_0'].attrs['t0'], minval, maxval, label=key, color=[str(color[key])])
        plt.legend()

class scanmd(scan):

    def __post_init__(self):
        super().__post_init__()
        
    def build_database(self):
        self.df = pd.DataFrame.from_records(self.dset, columns=self.dset.dtype.fields.keys())
        # Generate signal data from windows
        self.evaluate_windows()

        self.calculate_signal_error()
        # Check if data is multidimensional.
        self.x2 =  list(set(self.df['v1']))
        self.x2.sort()
        self.sets = []
        
        for val in self.x2:
            dfval = self.df[self.df['v1'] == val]
            sc = scan_base(experiment = self.experiment, df=dfval, x2 = val, function = self.function, filename = self.filename, averages=self.averages)
            # Group data points by x (v0) and calculate mean, and apply baseline if appropiate
            sc.process_signal()
            self.sets.append(sc)
    
class abstract_fitting:
    def __init__(self):
        self._p0 : np.ndarray
        self._varMatrix : np.ndarray
        self.scan : scan
        self.guess : np.ndarray
        self.bounds = 0
        self.sigma = []
        
    @staticmethod
    def func(x):
        return
    
    def perform_fit(self, userGuess=None):
        # Fit gaussian function
        if userGuess != None:
            self.guess = userGuess
        scan = self.scan
        p = self.guess
        if len(self.sigma) >  1:
            self._p0, self._varMatrix = curve_fit(self.func, scan.x, scan.y, p0=p, absolute_sigma=True, sigma=scan.error)
        else:
            if self.bounds == 0:
                self._p0, self._varMatrix = curve_fit(self.func, scan.x, scan.y, p0=p, absolute_sigma=True)
            else:
                self._p0, self._varMatrix = curve_fit(self.func, scan.x, scan.y, p0=p, bounds=self.bounds, absolute_sigma=True)  
    @property
    def p0(self):
        self.perform_fit()
        return self._p0
    
    @property
    def varMatrix(self):
        self.perform_fit()
        return self._varMatrix
    
    @property
    def fit(self):
        """
        Return y values predicted by the fitted model for the scan.x values.
        Returns:
            np.array : y values predicted by model
        """
        self.perform_fit()
        x = self.scan.x
        y = self.func(x, *self.p0)
        return y

class Gauss(abstract_fitting):
    def __init__(self, scan):
        super().__init__()
        self.guess = [np.max(scan.x), scan.x[np.argmax(scan.y)], 0.0001]
        self.scan = scan
        self.sigma = self.scan.error

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
    function = 'a0'
    filepath = 'tests/20210707_005_scan.h5'
    #sc = scan(filepath, function)
    #print(sc.rabi.p0())
    #sc.set_range([0,47])
    #print(sc.rabi.p0()[0])
    filepath = 'tests/20221208_006_scan.h5'
    
    scs = scanmd(filepath, function)
    #sc.remove_unstable(-0.02)
    sc = scs.sets[-1]
    sc.plot_stability(0.012,'(a1-a0)+(a2-a0)')
    sc.remove_unstable(0.012, customfunction='(a1-a0)+(a2-a0)')
