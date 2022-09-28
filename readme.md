# Scan Data Object
My personal library for working with data from the E11 lab. The data format used in the lab is HDF5. All the data from an experiment is included in this file, including metadata. The files can be read for analysis using the [h5py](https://www.h5py.org) library. A reference section can be found below that outlines the structure of these files.

How to use Scan object
- load data
- chose function
- change windows
- change function
- plot scans with window locations (interactive?)

Experiment Types
- Microwave: Baseline processing (first 10 values)
- Generic: Baseline processing


Fitting functions
##  Reference: File structure
A file can be loaded using 
```
f = h5py.File('FILENAME', 'r')
```
The metadata associated with the file is stored as a proxy object accessible using key value pairs through `f.attrs` i.e:
```
>>> f.attrs.keys()
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
| `v1_repeats` |                                                        |
| `var 0`      | Type of measurement i.e. `microwaves (GHz)`             |
| `var 1`      |                                                        |



Each file has two datasets:
- `analysis`: Data after applying windows
- `osc_0`: Data before applying windows 

Each of these datasets also has an associated `attrs` object containing useful metadata as well the data itself.

### analysis
The metadata associated with analysis includes the location of the windows as well as the function used to perform analysis in the e11 control software.

```
>>> f['analysis].attrs.keys()
<KeysViewHDF5 ['A', 'B', 'C', 'D', 'E', 'F', 'a0 ', 'a1', 'a2', 'f']>
```

The data can be read into a pandas data frame using:

```
>>> import pandas as pd
>>> df = pd.DataFrame.from_records(f['analysis'], columns=f['analysis'
].dtype.fields.keys())
>>>
```

The dataframe then has columns: `v0, v1, w0, w1, a0, a1, a2`.

### osc_0
This contains the time of arrival data stored on the oscilloscope. The metadata includes the `t0` and `dt` time step information.

```
>>> f['osc_0'].attrs.keys()
<KeysViewHDF5 ['dt', 't0']>
```

The individual scans can be loaded into a dataframe using:
```
pd.DataFrame.from_records(f['osc_0'])
```














