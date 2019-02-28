import pandas as pd
import warnings
import os
from CHECLabPy import __version__
from packaging.version import parse
from collections import defaultdict
from CHECLabPy.core.io import HDF5Reader, HDF5Writer


class HDF5Appender(HDF5Reader, HDF5Writer):
    def __init__(self, path):
        """
        Appender for generic HDF5 files produced by
        CHECLabPy.core.io.HDF5Writer

        Parameters
        ----------
        path : str
            Path to the HDF5 file
        """

        print("Opening HDF5 file: {}".format(path))
        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist: {}".format(path))
        self.path = path

        self.store = pd.HDFStore(
            path, mode='r+', complevel=9, complib='blosc:blosclz'
        )

        self.keys = set()
        self.metadata = defaultdict(lambda: defaultdict(dict))
        self.df_list = defaultdict(list)
        self.df_list_n_bytes = defaultdict(int)
        self.n_bytes = defaultdict(int)

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def _save_metadata(self):
        print("Saving metadata to HDF5 file")
        for key, submeta in self.metadata.items():
            if key not in self.store:
                self.store[key] = pd.DataFrame()
            attrs = self.store.get_storer(key).attrs
            for name, d in submeta.items():
                if hasattr(attrs, name):
                    old_d = getattr(attrs, name)
                    if isinstance(old_d, dict):
                        d = {**getattr(attrs, name), **d}
                setattr(attrs, name, d)

    def finish(self):
        """
        Finish creating the file by appending remaining dataframes and saving
        the metadata. This is method is automatically called when HDF5Writer
        is used in a context manager (i.e. `with HDF5Writer(path) as writer:`)
        """
        total_bytes = self.get_metadata()['total_bytes']
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
