import numpy as np
import pandas as pd
import os
from CHECLabPy.utils.files import create_directory
from CHECLabPy import __version__
from collections import defaultdict


class HDF5Writer:
    def __init__(self, path):
        """
        Helper class to write dataframes and metadata to a HDF5 file

        Parameters
        ----------
        path : str
            Path to store the HDF5
        """
        create_directory(os.path.dirname(path))
        print("Creating HDF5 file: {}".format(path))

        self.keys = set()
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
        """
        Perform some operations on the dataframe before it is appended to the
        HDFStore

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        df : pd.DataFrame
        """
        return df

    def _append_to_store(self, key, expectedrows=None):
        """
        Concanate the dataframes stored to the list of dataframes, and append
        the resulting dataframe to the HDFStore (i.e. the table in the file)

        Parameters
        ----------
        key : str
            Name of the HDFStore to append to (i.e. the table)
        expectedrows : int or None
            Total number of rows that are expected for the table in the end.
            Not required, but can speed up the file filling/generation
        """
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
        """
        Append a dataframe to the list of dataframes. Each HDFStore
        (i.e. table) has its own list of dataframes. When the amount of
        memory held in the dataframe list exceeds 0.5GB, the dataframes are
        stored to the file and the list is emptied.

        Using this method is advised when one is storing a large amount of
        data that can be split in chunks (e.g. per event), to avoid trying to
        hold the entire data in memory before it is stored to file.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to append to the file
        key : str
            Name of the HDFStore to append to (i.e. the table)
        expectedrows : int or None
            Total number of rows that are expected for the table in the end.
            Not required, but can speed up the file filling/generation
        """
        self.keys.add(key)
        self.df_list[key].append(df)
        memory_usage = df.memory_usage(index=True, deep=True).sum()
        self.df_list_n_bytes[key] += memory_usage
        if self.df_list_n_bytes[key] >= 0.5E9:
            self._append_to_store(key, expectedrows)

    def write(self, **kwargs):
        """
        Write a dataframe directly to the file. Multiple dataframes can be
        written with one call of this command:

        >>> df1 = pd.DataFrame(dict(x=np.arange(10), y=np.arange(10)*2))
        >>> df2 = pd.DataFrame(dict(x=np.arange(10), y=np.arange(10)*5))
        >>> path = "/path/to/file.h5"
        >>> with HDF5Writer(path) as writer:
        >>>    writer.write(times2=df1, times5=df2)

        This command will overwrite any other table/HDFStore with the same name

        Parameters
        ----------
        kwargs
            Named arguments where the key corresponds to the name of the
            HDFStore/table, and value corresponds to the dataframe
        """
        for key, value in kwargs.items():
            self.keys.add(key)
            self.store[key] = value
            self.n_bytes[key] = value.memory_usage(index=True, deep=True).sum()

    def add_mapping(self, mapping):
        """
        Add the CHECLabPy mapping dataframe to the file.

        Parameters
        ----------
        mapping : pd.DataFrame
        """
        self.store['mapping'] = mapping
        self.store.get_storer('mapping').attrs.metadata = mapping.metadata

    def add_metadata(self, key='data', name='metadata', **kwargs):
        """
        Add some metadata, which will be stored to the file as it is closed

        >>> path = "/path/to/file.h5"
        >>> nev = 2000
        >>> with HDF5Writer(path) as writer:
        >>>    writer.add_metadata(key='data', name='metadata', n_events=nev)

        >>> path = "/path/to/file.h5"
        >>> metadata = dict(
        >>>    n_events=2000,
        >>>    n_pixels=2048,
        >>> )
        >>> with HDF5Writer(path) as writer:
        >>>    writer.add_metadata(key='data', name='metadata', **metadata)

        Parameters
        ----------
        key : str
            HDFStore to store the metadata to (e.g. data)
        name : str
            Name for the type of metadata (e.g. metadata, config, mcheader...)
        kwargs
            Named arguments containing the metadata to store to the location
            specified by the other parameters.

        Returns
        -------

        """
        if name in ['info', 'mapping']:
            raise ValueError(
                "The name '{}' is reserved, "
                "please choose a different name".format(name)
            )
        self.metadata[key][name].update(kwargs)

    def _save_metadata(self):
        print("Saving metadata to HDF5 file")
        for key, submeta in self.metadata.items():
            if key not in self.store:
                self.store[key] = pd.DataFrame()
            attrs = self.store.get_storer(key).attrs
            for name, d in submeta.items():
                setattr(attrs, name, d)

    def finish(self):
        """
        Finish creating the file by appending remaining dataframes and saving
        the metadata. This is method is automatically called when HDF5Writer
        is used in a context manager (i.e. `with HDF5Writer(path) as writer:`)
        """
        total_bytes = 0
        for key in self.keys:
            self._append_to_store(key)
            self.add_metadata(key=key, n_bytes=self.n_bytes[key])
            total_bytes += self.n_bytes[key]
        self.add_metadata(
            total_bytes=total_bytes,
            version=__version__,
        )
        self._save_metadata()
        self.store.close()
