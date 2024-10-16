# Scan Data Object
My personal library for working with data from the E11 lab. The data format used in the lab is HDF5. All the data from an experiment is included in this file, including metadata. The files can be read for analysis using the [h5py](https://www.h5py.org) library. A description of the Scan object is below as well as a reference section can be found below that outlines the structure of these files.

How to use Scan object:
```
from e11scan import scan
filename = 'tests/20210707_005_scan.h5'
function = '-(a0 - a1)/((a0 - a1) + (a0 - a2))'
sc = scan(filename = filename, function = function)
```
The scan object automatically processes the data by grouping and averaging measurements made at the same point and subtracting a baseline value. The baseline that is subtracted depends on the type of the experiment. The experiment type is read from the metadata in the h5py file.

Experiment Types
- microwave:
    - Baseline processing: baseline calculated from the signal of the lowest 10 microwave frequencies.
- volt:
    - No baseline subtracted
- time: 
    - Baseline calculated from the signal measured for the lowest time value.
- generic: 
    - No baseline subtracted

The experiment type used can also be set manually:
```
scan(filename = filename, function = function, experiment = 'experiment')
```

Change the evaluation function:
```
>>> function = 'a0 - a1'
>>> sc.update_function(function)
```

Accessing data:
- Independent variable: `sc.x`
- Dependent variable: `sc.y`
- Without averaging:
    - Independent variable: `sc.xall`
    - Dependent variable: `sc.yall`

Access an individual trace:
```
times, signal = sc.trace(index)
```

Access locations of windows:
```
>>> sc.windows
{'A': 1.0063636363636365e-07,
 'B': 1.288636363636364e-07,
 'C': 1.840909090909091e-07,
 'D': 2.196818181818182e-07,
 'E': 2.3931818181818186e-07,
 'F': 2.7122727272727274e-07}
```
The location of the windows are in reference to the first time on the scope i.e. ``self.f['osc_0'].attrs['t0']``.

Plot a trace with location of windows:
```
>>> sc.plot_trace(index)
```
![trace_example.png](trace_example.png)

Plot the stability of the signal:
```
>>> sc.plot_stability()
```
![stability_example.png](stability_example.png)

By default, the function `a0 - a1` is used. A custom function can also be used e.g `sc.plot_stability(customfunction = a0-a2)`

## Fitting Routines
Basic fitting routines for fitting oscillatory data and Gaussian peaks are included. The best fit values `p0` and variance matrix `varMatrix` can be accessed through a scan object i.e.

### Gaussian
$$
A  e^{- \frac{(x-\mu)^2}{2 \sigma^2}}
$$

```
>>> sc = scan(filename, function)
>>> sc.Gauss.p0
array([8.90789113e-03, 1.95563556e+01, 6.64935890e-05])

>> sc.varMatrix
array([[ 2.82771663e-08,  6.16810598e-13, -1.40790216e-10],
       [ 6.16810598e-13,  2.10294996e-12, -4.60657391e-15],
       [-1.40790216e-10, -4.60657391e-15,  2.10293986e-12]])
```
`p0()` and `varMatrix()` returns fit parameters in the order: `[A, mu, sigma]`

It is also possible to return the values predicted by the model.
```
>> sc = scan(filename, function)
>> plt.plot(sc.x, sc.gauss.fit)
```
![gauss_fit_example.png](gauss_fit_example.png)

### Rabi

$$
A (1 - \cos(\omega t)) e^{-t/T_{2}}/2
$$

The fitting routine is performed subtracting the first element of scan.y from y, such that the oscillation starts at the origin.

`p0()` is returned in the order: `[omega, t2, A]`

```
>>> sc = scan(filename, function)
>>> plt.errorbar(sc.x, (sc.y-sc.y[0])/sc.rabi.p0[2], sc.error, ls='none', marker='x', markersize=3, elinewidth=1,alpha=0.7)
>>> plt.plot(sc.x, (sc.rabi.func(sc.x, *sc.rabi.p0))/sc.rabi.p0[2], label='fit')
```
![rabi_fit_example.png](rabi_fit_example.png)

### Interpolation
All fitting routines can be used to interpolate the data. The interpolated fit can be accessed using `sc.gauss.fit_interpolated(interpolation = 100)` or `sc.rabi.fit_interpolated(interpolation=100)`. Where the interpolation is the factor by which the number of points is increased. The default interpolation is 100.

```
>>> x,y = sc.gauss_fit_interpolated()
```



## Two Dimensional Scans
It is possible to perform measurements in two dimensions - with a range of values in both `v0` and `v1`. For this type of analysis use the `scanmd` object. This object builds a list (`scanmd.sets`) of `scan` objects for each dataset associated with independent values of `v1`.  Each `scan` object contains the values of `v0` accessible through `scanmd.sets[0].x` as well as the associated `v1` value accessible through `scanmd.set[0].x2`.

```
>>> scs = scanmd(filename = 'filename_of_twodimensional_dataset', function= 'a0-a1')

>>> for sc in scs.sets:
>>>    plt.plot(sc.x, sc.y, label = sc.x2)
```

## Changing windows
The position of the windows can be changed by defining a new set of indices for the window locations. An example of checking the current window locations, changing the window locations, and viewing the updated locations is shown below. The new window locations are not written to file.

```
>>> sc.windowsind
{'A': 201, 'B': 257, 'C': 368, 'D': 439, 'E': 478, 'F': 542}

>>> sc.plot_trace(0) # Check current windows

>>> windows = {'A': 210, 'B': 247} # Define new windows

>>> sc.windowsind  = windows

>>> sc.plot_trace(0) # Check new windows
```




## Error Calculation
For a full calculation of the error the distribution functions of the states must be known. If the states do not overlap then a calculation of the standard error of each data point can be estimated. To calculate the standard error the standard deviation of each measurement must be known.

An estimation of the standard deviation of each measurement can be made by treating the measurements, $x_i$, as Bernoulli trial (only valid if the distributions do not overlap):
$$
\sigma_\mathrm{x_i} = \sqrt{\frac{pq}{n}}
$$
where $p$ is the probability of success, $q = 1 - p$, and $n$ is the number of averages on the oscilloscope. The error is then calculated by setting $p$ equal to the signal calculated using data from the oscilloscope and the windows. This is only valid if the value of the signal has been correctly normalized. If this is not the case then an overestimate of the error can be made by setting $p=0.5$.

If standard error can then be calculated by considering the number of loops:



##  Reference: File structure
A file can be loaded using 
```
f = h5py.File('FILENAME', 'r')
```
The file is also accessible from the Scan object:
```
>>> sc = scan(filename = 'FILENAME', function = 'a0')
>>> sc.f
<HDF5 file "20210707_005_scan.h5" (mode r)>

```
The metadata associated with the file is stored as a proxy object accessible using key value pairs through `f.attrs`. The metadata available depends on which oscilloscope was used.


Example (agilent):
```
>>> sc.f.attrs.keys()
<KeysViewHDF5 ['run_ID', 'timestamp', 'v0_loops', 'v0_num', 'v0_options', 'v0_repeats', 'v1_loops', 'v1_num', 'v1_options', 'v1_repeats', 'var 0', 'var 1']>
```
| `key`     |Description|
|--------------|--------------------------------------------------------|
| `run_ID`     | Date of file and measurement number i.e. `20210707_005` |
| `timestamp`  | i.e. `2021-07-07 15:04:14`                              |
| `v0_loops`   | Number of measurement loops on `v0`                    |
| `v0_num`     | Number of measurements on `v0`                        |
| `v0_options` |                                                        |
| `v0_repeats` |                                                        |
| `v1_loops`   |                                                        |
| `v1_num`     |                                                        |
| `v1_options` |                                                        |
| `v1_repeats` |                                                        |                                                |
| `var 0`      | Type of measurement i.e. `microwaves (GHz)`             |
| `var 1`      |                                                        |

Alternative Example (lecroy)
| `key`                | Description                                               |
|----------------------|-----------------------------------------------------------|
| `num_rows`           |                                                           |
| `run_id`             | Date of file and measurement number i.e.   `20210707_005` |
| `scope_VISA`         | Visa name i.e `lecroy`                                    |
| `scope_averages`     | Number of averages used by scope                          |
| `scope_channels`     | Channel number                                            |
| `scope_max_points`   | Number of points                                          |
| `scope_noise filter` |                                                           |
| `scope_timeout (ms)` |                                                           |
| `timestamp`          | i.e `2021-07-07 15:04:14`                                 |
| `v0_VISA`            | Extra visa info i.e. `COM 4`                              |
| `v0_hardware`        |                                                           |
| `v0_loops`           | Number of measurement loops on `v0`                       |
| `v0_options`         |                                                           |
| `v0_repeats`         |                                                           |
| `v1_hardware`        |                                                           |
| `v1_loops`           |                                                           |
| `v1_options`         |                                                           |
| `v1_repeats`         |                                                           |

Each file has two datasets:
- `analysis`: Data after applying windows
- `osc_0`: Data before applying windows 

Each of these datasets also has an associated `attrs` object containing useful metadata as well the data itself.

### analysis
The metadata associated with analysis includes the location of the windows as well as the function used to perform analysis in the e11 control software.

```
>>> sc.f['analysis'].attrs.keys()
<KeysViewHDF5 ['A', 'B', 'C', 'D', 'E', 'F', 'a0 ', 'a1', 'a2', 'f']>
```

The data can be read into a pandas data frame using:

```
>>> import pandas as pd
>>> df = pd.DataFrame.from_records(sc.f['analysis'], columns = sc.f['analysis'
].dtype.fields.keys())
```

The dataframe then has columns: `v0, v1, w0, w1, a0, a1, a2`.

### osc_0
This contains the time of arrival data stored on the oscilloscope. The metadata includes the `t0` and `dt` time step information.

```
>>> sc.f['osc_0'].attrs.keys()
<KeysViewHDF5 ['dt', 't0']>
```

The individual scans can be loaded into a dataframe using:
```
pd.DataFrame.from_records(sc.f['osc_0'])
```
Or accessed using:
```
sc.f['osc_0'][index]
```



# Instillation
```
pip install git+https://github.com/lukelbro/scan
```












