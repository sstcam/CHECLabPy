import numpy as np
import pandas as pd
import warnings
import json
from CHECLabPy.core.io.hdf5_reader import HDF5Reader


def deprecate(msg):
    warnings.simplefilter('always', DeprecationWarning)  # turn off filter
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
    warnings.simplefilter('default', DeprecationWarning)  # reset filter


class DL1Reader(HDF5Reader):
    def __init__(self, path):
        """
        Reader for the HDF5 DL1 Files. Contains some conveniance methods on
        top of the HDF5Reader that are specific for DL1 files.

        Parameters
        ----------
        path : str
            Path to the HDF5 DL1 file.
        """
        super().__init__(path)
        self._monitor = None

    def __getitem__(self, iev):
        start = iev * self.n_pixels
        stop = (iev + 1) * self.n_pixels
        df = self.select(start=start, stop=stop)
        return df

    @property
    def metadata(self):
        return self.get_metadata('data')

    @property
    def config(self):
        return self.get_metadata('data', 'config')

    @property
    def columns(self):
        return self.get_column_names('data')

    @property
    def n_rows(self):
        return self.get_n_rows('data')

    @property
    def n_bytes(self):
        return self.get_n_bytes('data')

    @property
    def is_mc(self):
        return self.metadata['is_mc']

    @property
    def n_events(self):
        return self.metadata['n_events']

    @property
    def n_modules(self):
        return self.metadata['n_modules']

    @property
    def mapping(self):
        return self.get_mapping()

    @property
    def tc_mapping(self):
        """
        Obtain the TargetCalib mapping class
        """
        from target_calib import CameraConfiguration
        version = self.mapping.metadata.version
        camera_config = CameraConfiguration(version)
        return camera_config.GetMapping(self.n_modules == 1)

    @property
    def monitor(self):
        deprecate(
            "Accessing the monitor information via the _dl1.h5 file has been "
            "deprecated. Instead the monitor information is kept seperate and "
            "can be matched to the dl1 information afterwards. See the new "
            "tutorial on monitor data for more information."
        )
        if not self._monitor:
            if 'monitor_metadata' in self.store:
                self._monitor = DeprecatedMonitorReader(self.store)
            else:
                msg = ("Monitor information has not been "
                       "stored inside this DL1 file")
                raise AttributeError(msg)
        return self._monitor

    @property
    def n_pixels(self):
        return self.metadata['n_pixels']

    @property
    def n_samples(self):
        return self.metadata['n_samples']

    @property
    def camera_version(self):
        return self.metadata['camera_version']

    def get_sn(self, tm):
        """
        Get the SN of the TARGET module in a slot

        Parameters
        ----------
        tm : int
            Slot number for the TARGET module

        Returns
        -------
        int
            Serial number of the TM
        """
        if tm >= self.n_modules:
            raise IndexError("Requested TM out of range: {}".format(tm))
        return self.get_metadata('data', 'sn')["TM{:02d}".format(tm)]

    def load_entire_table(self, force=False):
        """
        Load the entire DataFrame into memory

        Parameters
        ----------
        force : bool
            Force the loading of a DataFrame into memory even if it is of a
            large size

        Returns
        -------
        df : `pandas.DataFrame`

        """
        return self.read('data', force=force)

    def select(self, **kwargs):
        return super().select('data', **kwargs)

    def select_column(self, column, **kwargs):
        return super().select_column('data', column, **kwargs)

    def select_columns(self, columns, **kwargs):
        values = []
        for c in columns:
            values.append(self.select_column(c, **kwargs))
        return values

    def iterate_over_rows(self):
        """
        Loop over the each row in the file

        Returns
        -------
        row : `pandas.Series`

        """
        for row in self.iterate_over_chunks(1):
            yield row

    def iterate_over_events(self):
        """
        Loop over the each event in the file

        Returns
        -------
        df : `pandas.DataFrame`

        """
        for df in self.iterate_over_chunks(self.metadata['n_pixels']):
            yield df

    def iterate_over_chunks(self, chunksize=None, **kwargs):
        """
        Loop over the DataFrame in chunks, therefore avoiding loading the
        entire table into memory. The chunksize is automatically defined to
        be approximately 2GB of memory.

        Parameters
        ----------
        chunksize : int
            Size of the chunk. By default it is set to a number of rows that
            is approximately equal to 2GB of memory.
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        df : `pandas.DataFrame`

        """
        yield from super().iterate_over_chunks('data', chunksize, **kwargs)

    def get_matched_monitor_column(self, column_name):
        """
        Match a monitor column to the waveform events

        Parameters
        ----------
        column_name : str
            Column name from the monitor DataFrame

        Returns
        -------
        pd.Series

        """
        key = self.monitor.get_key_for_column(column_name)
        if key == "monitor_camera":
            event_key = "monitor_camera_index"
        elif key == "monitor_tm":
            event_key = "monitor_tm_index"
        elif key == "monitor_pixel":
            event_key = "monitor_pixel_index"
        else:
            msg = "Case not handled for key: {}".format(key)
            raise KeyError(msg)

        imonitor = self.select_column(event_key)
        column = self.monitor.select_column(column_name, key)[imonitor]

        return column


class DeprecatedMonitorReader:
    def __init__(self, store):
        self.store = store

    @property
    def metadata(self):
        return self.store.get_storer("monitor_metadata").attrs.metadata

    @property
    def columns_camera(self):
        storer = self.store.get_storer("monitor_camera")
        return storer.attrs.non_index_axes[0][1]

    @property
    def columns_tm(self):
        storer = self.store.get_storer("monitor_tm")
        return storer.attrs.non_index_axes[0][1]

    @property
    def columns_pixel(self):
        storer = self.store.get_storer("monitor_pixel")
        return storer.attrs.non_index_axes[0][1]

    @property
    def n_bytes(self):
        return self.metadata['n_bytes']

    @property
    def n_events(self):
        return self.metadata['n_events']

    @property
    def n_modules(self):
        return self.metadata['n_modules']

    def _load_entire_table(self, key, force=False):
        """
        Load the entire DataFrame into memory

        Parameters
        ----------
        force : bool
            Force the loading of a DataFrame into memory even if it is of a
            large size

        Returns
        -------
        df : `pandas.DataFrame`

        """
        print("Loading entire DataFrame from HDF5 file")
        if (self.n_bytes > 8E9) and not force:
            raise MemoryError("DataFrame is larger than 8GB, "
                              "set force=True to proceed with loading the "
                              "entire DataFrame into memory")
        if self.n_bytes > 8E9:
            warnings.warn("WARNING: DataFrame is larger than 8GB", UserWarning)
        return self.store[key]

    def load_entire_table_camera(self, force=False):
        """
        Load the entire DataFrame into memory (monitor_camera)

        Parameters
        ----------
        force : bool
            Force the loading of a DataFrame into memory even if it is of a
            large size

        Returns
        -------
        df : `pandas.DataFrame`

        """
        return self._load_entire_table("monitor_camera", force)

    def load_entire_table_tm(self, force=False):
        """
        Load the entire DataFrame into memory (monitor_tm)

        Parameters
        ----------
        force : bool
            Force the loading of a DataFrame into memory even if it is of a
            large size

        Returns
        -------
        df : `pandas.DataFrame`

        """
        return self._load_entire_table("monitor_tm", force)

    def load_entire_table_pixel(self, force=False):
        """
        Load the entire DataFrame into memory (monitor_pixel)

        Parameters
        ----------
        force : bool
            Force the loading of a DataFrame into memory even if it is of a
            large size

        Returns
        -------
        df : `pandas.DataFrame`

        """
        return self._load_entire_table("monitor_pixel", force)

    def get_key_for_column(self, column):
        """
        Search the monitor dataframes to find which contains the requested
        column

        Parameters
        ----------
        column : str
            The column you wish to obtain

        Returns
        -------
        str

        """
        if column in self.columns_camera:
            return "monitor_camera"
        elif column in self.columns_tm:
            return "monitor_tm"
        elif column in self.columns_pixel:
            return "monitor_pixel"
        else:
            msg = "Column does not exist in monitor dataframes"
            raise KeyError(msg)

    def select_column(self, column, key=None, **kwargs):
        """
        Use the pandas.HDFStore.select_column method to obtain a single
        column

        Parameters
        ----------
        column : str
            The column you wish to obtain
        key : str
            Force function to look for column in this dataframe
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        `pandas.Series`

        """
        if not key:
            key = self.get_key_for_column(column)
        return self.store.select_column(key, column, **kwargs)

    def select_columns(self, columns, key=None, **kwargs):
        """
        Use the pandas.HDFStore.select_column method to obtain a list of
        columns as numpy arrays

        Parameters
        ----------
        columns : list
            A list of the columns you wish to obtain
        key : str
            Force function to look for columns in this dataframe
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        values : list
            List of numpy arrays containing the values for all of the columns

        """
        values = []
        for c in columns:
            values.append(self.select_column(c, key, **kwargs))
        return values

    def select_column_as_array(self, column, key=None, **kwargs):
        """
        Return select_column transformed into a numpy array

        Parameters
        ----------
        column : str
            The column you wish to obtain
        key : str
            Force function to look for column in this dataframe
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        ndarray

        """
        column = self.select_column(column, key, **kwargs).values
        n_idevice = column.size // self.n_events
        return np.reshape(column, (self.n_events, n_idevice))

    def get_average(self, column):
        """
        Obtain the average for a column

        Parameters
        ----------
        column : str
            The column you wish to obtain

        Returns
        -------
        ndarray

        """
        key = self.get_key_for_column(column)
        idevice, col_values = self.select_columns(['idevice', column], key)
        df = pd.DataFrame({"idevice": idevice, column: col_values})
        return df.groupby('idevice').mean()[column].values

    def print_columns(self):
        """
        Print each column heading and which DataFrame they are located inside
        """
        columns_camera = self.columns_camera
        columns_tm = self.columns_tm
        columns_pixel = self.columns_pixel
        remove_ = ['imon', 'idevice', 't_cpu']
        columns = dict(
            monitor_camera=[c for c in columns_camera if c not in remove_],
            monitor_tm=[c for c in columns_tm if c not in remove_],
            monitor_pixel=[c for c in columns_pixel if c not in remove_],
        )
        print(json.dumps(columns, indent=4))
