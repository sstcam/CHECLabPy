# CHECLabPy [![Build Status](https://travis-ci.org/cta-chec/CHECLabPy.svg?branch=master)](https://travis-ci.org/cta-chec/CHECLabPy)
Python scripts for reduction and analysis of CHEC lab data

These set of python scripts provide a standard approach for reading,
plotting and reducing the CHEC tio files for lab testing and comissioning.

To contribute a new charge-extraction/waveform-reducer method please read the
waveform_reducers section.

The dl1 files are stored as a `pandas.DataFrame` in HDF5 format. A `DataFrame`
is an object that acts as a table. It is compatible with numpy methods and
allows easy category searching. Learn about `pandas.DataFrame` at:
https://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe

There are also executables created to convert the DataFrame into other formats
like csv and ROOT TTree.

See tutorial.ipynb for instructions on the CHEC calibration and reduction 
flow and how to use the CHECLabPy software.

## Installation

### Downloading and updating

#### Non-contributor
If you wish to just use the software, and not contribute: 
* To Download: `git clone https://github.com/cta-chec/CHECLabPy.git`
* To Update: `git pull`

#### Contributor
1. Create a fork of https://github.com/cta-chec/CHECLabPy to your GitHub 
account
2. `git clone https://github.com/YOURGITHUBACCOUNT/CHECLabPy.git`
3. `cd CHECLabPy`
4. `git remote add upstream https://github.com/cta-chec/CHECLabPy.git`
* To Update: `git fetch upstream && git checkout master && 
git merge upstream/master && git push origin master`

### Installing
To install, run `python setup.py develop`

## Contributing
The "master" branch is meant to always be a clean, up-to-date copy of the 
cta-chec/CHECLabPy:master. You should always create a branch when developing 
some new code (unless it is a very small change). Generally make a new 
branch for each new feature, so that you can make pull-requests for each one 
separately and not mix code from each. Remember that `git checkout` switches 
between branches, `git checkout -b` creates a new branch, and `git branch` on 
it’s own will tell you which branches are available and which one you are 
currently on.

**1. Create a new branch**
```bash
git checkout -b branch_name
```

**2. Make your additions to the code on that branch**
**3. Commit your changes locally**
```bash
git add some_changed_file.py another_file.py
git commit
```
**4. Push your changes to your GitHub account**
```bash
git push -u origin branch_name
```
**5. On GitHub, create a pull request into cta-chec/CHECLabPy:master**

## Layout

### core
This module contains the core funtionality for CHECLabPy, such as base
classes and file io. It is advised not to change the contents of this
directory, it should not be necessary.

### waveform_reducers
This directory contains all the waveform reducer methods. A set of default
reduction methods are defined in `CHECLabPy.core.base_reducer.WaveformReducer`.

To create a new waveform reducer simply create a new file in this directory,
containing a class that inherits from `WaveformReducer`. The
`WaveformReducerFactory` will automatically find any `WaveformReducer` inside
this directory, and add it as an option to the extract_dl1.py executable.

In the new waveform reducer you may override the default
`CHECLabPy.core.base_reducer.WaveformReducer` method. The storage of parameters
extracted from the reducer is very flexible, therefore users can define new
parameters to return from the reducer in the dict, and they will be stored in
the dl1 file.

The default parameters that will always exist in the dl1 file are:
```
    iev                     uint32    Event Number
    pixel                   uint32    Pixel Number
    t_cpu_ns                uint64    CPU Time (ns)
    t_cpu_sec               uint64    CPU Time (seconds)
    t_tack                  uint64    Tack Time
    first_cell_id           uint16    First Cell ID (TARGET Storage Array)
    baseline_start_mean     float32   Average of the first 20 samples
    baseline_start_rms      float32   RMS of the first 20 samples
    baseline_end_mean       float32   Average of the last 20 samples
    baseline_end_rms        float32   RMS of the last 20 samples
    baseline_subtracted     float32   Baseline subtracted from each waveform
    t_event                 uint16    Time of pulse obtained from user or
                                      average wf, used in charge extaction
    charge                  float32   Extracted charge
    t_pulse                 float32   Time of pulse
    amp_pulse               float32   Amplitude at t_pulse
    fwhm                    float32   FWHM of pulse
    tr                      float32   Rise time of pulse
    waveform_mean           float32   Average of full waveform
    waveform_rms            float32   RMS of full waveform
    saturation_coeff        float32   Coefficient that can be used for 
                                      saturation investigations and recovery
```

If the chosen reducer does not define one of these parameters, then they are
still included in the output dl1 file as "0".

The following metadata is also included inside the file:
```
    input_path
    n_events
    n_pixels
    n_samples
    n_cells
    start_time      CPU time of the first event
    end_time        CPU time of the last event 
    version         Camera version (used for obtaining the correct mapping)
    reducer         Name of the WaveformReducer used
    configuration   Additional configuration that was passed from the cmdline
```

### data
Contains data that can be used for the algorithms in the module
(such as a reference pulse shape).

### plotting
Defines some classes for various plots, including a conveniant method for
plotting camera images (which uses the TargetCalib mapping)

### utils
Contains some useful functions and classes that may be used in multiple
scripts to process the data and waveforms.
