import numpy as np
import pandas as pd
import os
from CHECLabPy.utils.files import create_directory
from CHECLabPy import __version__
from collections import defaultdict


class HDF5Writer:
    def __init__(self, path):
        create_directory(os.path.dirname(path))
        print("Creating HDF5 file: {}".format(path))

        self.metadata = defaultdict(lambda: defaultdict(dict))
        self.df_list = defaultdict(list)
        self.df_list_n_bytes = defaultdict(int)
        self.n_bytes = defaultdict(int)

        self.store = pd.HDFStore(
            path, mode='w', complevel=9, complib='blosc:blosclz'
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    @staticmethod
    def _prepare_before_append(df):
        return df

    def _append_to_store(self, key, expectedrows=None):
        if self.df_list[key]:
            df = pd.concat(self.df_list[key], ignore_index=True)
            df = self._prepare_before_append(df)

            kwargs = dict(index=False, data_columns=True)
            if expectedrows is not None:
                kwargs['expectedrows'] = expectedrows
            self.store.append(key, df, **kwargs)
            self.n_bytes[key] += df.memory_usage(index=True, deep=True).sum()
            self.df_list[key] = []
            self.df_list_n_bytes[key] = 0

    def append(self, df, key='data', expectedrows=None):
        self.df_list[key].append(df)
        memory_usage = df.memory_usage(index=True, deep=True).sum()
        self.df_list_n_bytes[key] += memory_usage
        if self.df_list_n_bytes[key] >= 0.5E9:
            self._append_to_store(key, expectedrows)

    def write(self, **kwargs):
        for key, value in kwargs.items():
            self.store[key] = value
            self.n_bytes[key] = value.memory_usage(index=True, deep=True).sum()

    def add_mapping(self, mapping):
        self.store['mapping'] = mapping
        self.store.get_storer('mapping').attrs.metadata = mapping.metadata

    def add_metadata(self, key='data', name='metadata', **kwargs):
        self.metadata[key][name].update(kwargs)

    def _save_metadata(self):
        print("Saving metadata to HDF5 file")
        for key, submeta in self.metadata.items():
            attrs = self.store.get_storer(key).attrs
            for name, d in submeta.items():
                setattr(attrs, name, d)

    def finish(self):
        total_bytes = 0
        for key in self.df_list.keys():
            self._append_to_store(key)
            self.add_metadata(key=key, n_bytes=self.n_bytes[key])
            total_bytes += self.n_bytes[key]
        self.add_metadata(
            total_bytes=total_bytes,
            version=__version__,
        )
        self._save_metadata()
        self.store.close()
