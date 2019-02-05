# CHECLabPy [![Build Status](https://travis-ci.org/cta-chec/CHECLabPy.svg?branch=master)](https://travis-ci.org/cta-chec/CHECLabPy)
Python scripts for reduction and analysis of CHEC lab data

These set of python scripts provide a standard approach for reading,
plotting and reducing the CHEC tio files for lab testing and comissioning.
 
Refer to https://forge.in2p3.fr/projects/gct/wiki/Installing_CHEC_Software for 
intstructions for preparing your environment and installing the 
TARGET libraries. It is not a requirement to install the TARGET libraries 
to use this package (unless you wish to read R0/R1 tio files).

To set up TC_CONFIG_PATH for the Transfer Functions, download
svn.in2p3.fr/cta/Sandbox/justuszorn/CHECDevelopment/CHECS/Operation to a
directory, and `export TC_CONFIG_PATH=...`.

The dl1 files are stored as a `pandas.DataFrame` in HDF5 format. A `DataFrame`
is an object that acts as a table. It is compatible with numpy methods and
allows easy category searching. Learn about `pandas.DataFrame` at:
https://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe

There are also executables created to convert the DataFrame into other formats
like csv and ROOT TTree.

See the examples/tutorials for instructions on the CHEC calibration and reduction 
flow and how to use the CHECLabPy software.

## Installation

### Downloading and updating

#### Prerequisites
It is recommended to use a conda environment running Python3.5 (or above).
Instructions on how to setup such a conda environment can be found in
https://forge.in2p3.fr/projects/gct/wiki/Installing_CHEC_Software. The
required python packages for CHECLabPy (which can be installed using
`conda install ...` or `pip install ...`) are:
* astropy
* scipy
* numpy
* matplotlib
* tqdm
* pandas

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

### scripts
Contains the useful scripts for processessing the data. The most important script is extract_dl1.py, which reduces the waveforms into a TIO file to produce a dl1.h5 file, with parameters extracted per pixel and per event.

### examples
Contains some examples of how to use functionality of CHECLabPy. There are also some tutorial Jupyter notebooks included in this directory.

### CHECLabPy/core
This module contains the core funtionality for CHECLabPy, such as base
classes and file io. It is advised not to change the contents of this
directory, it should not be necessary.

### CHECLabPy/waveform_reducers
This directory contains all the waveform reducer methods. See the tutorial on 
waveform reducers on how to contribute a new waveform reducer class.

### CHECLabPy/spectrum_fitters
This directory contains all the fitting methods for SPE spectra.

### CHECLabPy/plotting
Defines some classes for various plots, including a conveniant method for
plotting camera images (which uses the TargetCalib mapping)

### CHECLabPy/utils
Contains some useful functions and classes that may be used in multiple
scripts to process the data and waveforms.
