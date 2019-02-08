import pandas as pd
import warnings
import os
from CHECLabPy import __version__
from packaging.version import parse
from collections import defaultdict


class HDF5Reader:
    def __init__(self, path):
        """
        Reader for generic HDF5 files produced by CHECLabPy.core.io.HDF5Writer

        Parameters
        ----------
        path : str
            Path to the HDF5 file
        """

        print("Opening HDF5 file: {}".format(path))
        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist: {}".format(path))
        self.path = path
        self.store = self.store = pd.HDFStore(
            path, mode='r', complevel=9, complib='blosc:blosclz'
        )

        self._generate_contents_list()

        if parse(self.version).release[0] < parse(__version__).release[0]:
            warnings.warn(
                "WARNING: HDF5 file created with older version of CHECLabPy",
                UserWarning
            )
        elif parse(self.version).release[0] > parse(__version__).release[0]:
            warnings.warn(
                "WARNING: HDF5 file created with newer version of CHECLabPy",
                UserWarning
            )

    def _generate_contents_list(self):
        """
        Generate a list of dataframes and metadata in file
        """
        self.dataframe_keys = []
        self.metadata_keys = defaultdict(list)
        for key in self.store.keys():
            key = key[1:]
            try:
                n_bytes = self.store.get_storer(key).attrs.metadata['n_bytes']
            except KeyError:
                n_bytes = 0
            if not n_bytes == 0:
                self.dataframe_keys.append(key)
            attrs = self.store.get_storer(key).attrs
            for subattr in dir(attrs):
                if subattr.startswith("_"):
                    continue
                if isinstance(getattr(attrs, subattr), dict):
                    self.metadata_keys[key].append(subattr)
        self.metadata_keys = dict(self.metadata_keys)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.store.close()

    @property
    def version(self):
        """CHECLabPy version used to create the file"""
        try:
            return self.get_metadata()['version']
        except KeyError:
            return "1.0.0"

    def read(self, key, force=False):
        """
        Load the entire DataFrame into memory

        Parameters
        ----------
        key : str
            Name of the table to read from the file
        force : bool
            Force the loading of a DataFrame into memory even if it is of a
            large size

        Returns
        -------
        df : `pandas.DataFrame`

        """
        print("Reading entire DataFrame ({}) from HDF5 file into memory"
              .format(key))
        if key not in self.dataframe_keys:
            raise KeyError(
                "No DataFrame in file with key: {}. Available keys: {}"
                .format(key, self.dataframe_keys)
            )
        n_bytes = self.get_n_bytes(key)
        if (n_bytes > 8E9) and not force:
            raise MemoryError(
                "DataFrame is larger than 8GB, set force=True to proceed with "
                "loading the entire DataFrame into memory"
            )
        if n_bytes > 8E9:
            warnings.warn("WARNING: DataFrame is larger than 8GB", UserWarning)
        return self.store[key]

    def get_mapping(self):
        """
        Obtain the CHECLabPy camera pixel mapping dataframe from the file

        Returns
        -------
        mapping : pd.DataFrame
        """
        if 'mapping' not in self.store:
            raise KeyError("No Mapping stored in HDF5 file")
        mapping = self.store['mapping']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            mapping.metadata = self.store.get_storer('mapping').attrs.metadata
        return mapping

    def get_metadata(self, key='data', name='metadata'):
        """
        Obtain metadata from the file

        Parameters
        ----------
        key : str
            HDFStore to which the metadata was attached to
        name : str
            Name of the metadata table

        Returns
        -------
        dict
        """
        if key not in self.metadata_keys:
            raise KeyError(
                "No metadata in file with key: {}. Available keys: {}"
                .format(key, list(self.metadata_keys.keys()))
            )
        elif name not in self.metadata_keys[key]:
            msg = ("No metadata with key and name: ({}: {}). "
                   "Available names for key: {}"
                   .format(key, name, self.metadata_keys[key]))
            raise KeyError(msg)
        return getattr(self.store.get_storer(key).attrs, name)

    def get_n_rows(self, key):
        """
        Get number of rows in the DataFrame

        Parameters
        ----------
        key : str
            Name of the table in the file

        Returns
        -------
        int
        """
        return self.store.get_storer(key).nrows

    def get_n_bytes(self, key):
        """
        Get number of bytes in the DataFrame

        Parameters
        ----------
        key : str
            Name of the table in the file

        Returns
        -------
        int
        """
        return self.get_metadata(key)["n_bytes"]

    def get_columns(self, key):
        """
        Get columns in the DataFrame

        Parameters
        ----------
        key : str
            Name of the table in the file

        Returns
        -------
        int
        """
        return self.store.get_storer(key).attrs.non_index_axes[0][1]

    def select(self, key, **kwargs):
        """
        Use the pandas.HDFStore.select method to select a subset of the
        DataFrame.

        Parameters
        ----------
        key : str
            Name of the table to read from the file
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        df : `pandas.DataFrame`

        """
        return self.store.select(key, **kwargs)

    def select_column(self, key, column, **kwargs):
        """
        Use the pandas.HDFStore.select_column method to obtain a single
        column

        Parameters
        ----------
        key : str
            Name of the table to read from the file
        column : str
            The column you wish to obtain
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        `pandas.Series`

        """
        return self.store.select_column(key, column, **kwargs)

    def select_columns(self, key, columns, **kwargs):
        """
        Use the pandas.HDFStore.select_column method to obtain a list of
        columns as numpy arrays

        Parameters
        ----------
        key : str
            Name of the table to read from the file
        columns : list
            A list of the columns you wish to obtain
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        values : list
            List of numpy arrays containing the values for all of the columns

        """
        values = []
        for c in columns:
            values.append(self.select_column(key, c, **kwargs))
        return values

    def iterate_over_chunks(self, key, chunksize=None, **kwargs):
        """
        Loop over the DataFrame in chunks, therefore avoiding loading the
        entire table into memory. The chunksize is automatically defined to
        be approximately 2GB of memory.

        Parameters
        ----------
        key : str
            Name of the table to read from the file
        chunksize : int
            Size of the chunk. By default it is set to a number of rows that
            is approximately equal to 2GB of memory.
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        df : `pandas.DataFrame`

        """
        if not chunksize:
            chunksize = self.get_n_rows(key) / self.get_n_bytes(key) * 2E9
        for df in self.store.select(key, chunksize=chunksize, **kwargs):
            yield df
