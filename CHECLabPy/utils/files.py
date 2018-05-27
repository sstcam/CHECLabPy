import os
import numpy as np
import pandas as pd


def create_directory(directory):
    if directory:
        if not os.path.exists(directory):
            print("Creating directory: {}".format(directory))
            os.makedirs(directory)


def read_runlist_old(path):
    df = pd.read_csv(
        path, header=None, delimiter=' ', index_col=0, comment='#',
        names=['run', 'attenuation', 'illumination', 'n_events', 'fw'],
        dtype={'run': np.int32, 'attenuation': np.float32,
               'illumination': np.float32, 'n_events': np.int32, 'fw': str}
    )
    return df


def read_runlist(path):
    df = pd.read_csv(
        path, delimiter=' ', index_col=0, comment='#'
    )
    return df


def open_runlist_r1(path, open_readers=True, max_events=None):
    from CHECLabPy.core.io import ReaderR1
    df = read_runlist(path)
    input_dir = os.path.dirname(path)
    input_run_path = os.path.join(input_dir, "Run{:05d}_r1.tio")
    df['path'] = [input_run_path.format(i) for i in df.index]
    if open_readers:
        df['reader'] = [ReaderR1(fp, max_events) for fp in df['path'].values]
    return df


def open_runlist_dl1(path, open_readers=True):
    from CHECLabPy.core.io import DL1Reader
    df = read_runlist(path)
    input_dir = os.path.dirname(path)
    input_run_path = os.path.join(input_dir, "Run{:05d}_dl1.h5")
    df['path'] = [input_run_path.format(i) for i in df.index]
    if open_readers:
        df['reader'] = [DL1Reader(fp) for fp in df['path'].values]
    return df
