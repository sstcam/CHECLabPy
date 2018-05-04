import numpy as np
import pandas as pd
from astropy.io import fits
import warnings
import os
from os import remove
from abc import ABC, abstractmethod
from CHECLabPy.utils.files import create_directory
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping


class TIOReader:
    """
    Reader for the R0 and R1 tio files
    """
    def __init__(self, path, max_events=None):
        try:
            from target_io import WaveformArrayReader
            from target_calib import CameraConfiguration
        except ModuleNotFoundError:
            msg = ("Cannot find TARGET libraries, please follow installation "
                   "instructions from https://forge.in2p3.fr/projects/gct/"
                   "wiki/Installing_CHEC_Software")
            raise ModuleNotFoundError(msg)

        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist: {}".format(path))
        self.path = path

        self._reader = WaveformArrayReader(self.path, 2, 1)

        self.is_r1 = self._reader.fR1
        self.n_events = self._reader.fNEvents
        self.run_id = self._reader.fRunID
        self.n_pixels = self._reader.fNPixels
        self.n_modules = self._reader.fNModules
        self.n_tmpix = self.n_pixels // self.n_modules
        self.n_samples = self._reader.fNSamples

        self._camera_config = CameraConfiguration(self._reader.fCameraVersion)
        self.tc_mapping = self._camera_config.GetMapping(self.n_modules == 1)

        self._pixel = self._PixelWaveforms(self)

        self.n_cells = self._camera_config.GetNCells()
        self.camera_version = self._camera_config.GetVersion()
        self.reference_pulse_path = self._camera_config.GetReferencePulsePath()

        self.current_tack = None
        self.current_cpu_ns = None
        self.current_cpu_s = None

        self.first_cell_ids = np.zeros(self.n_pixels, dtype=np.uint16)

        if self.is_r1:
            self.samples = np.zeros((self.n_pixels, self.n_samples),
                                    dtype=np.float32)
            self.get_tio_event = self._reader.GetR1Event
        else:
            self.samples = np.zeros((self.n_pixels, self.n_samples),
                                    dtype=np.uint16)
            self.get_tio_event = self._reader.GetR0Event

        if max_events and max_events < self.n_events:
            self.n_events = max_events

    def _get_event(self, iev):
        self.index = iev
        self.get_tio_event(iev, self.samples, self.first_cell_ids)
        self.current_tack = self._reader.fCurrentTimeTack
        self.current_cpu_ns = self._reader.fCurrentTimeNs
        self.current_cpu_s = self._reader.fCurrentTimeSec
        return self.samples

    def __iter__(self):
        for iev in range(self.n_events):
            yield self._get_event(iev)

    def __getitem__(self, iev):
        return np.copy(self._get_event(iev))

    class _PixelWaveforms:
        def __init__(self, tio_reader):
            self.reader = tio_reader

        def __getitem__(self, p):
            if not isinstance(p, list) and not isinstance(p, np.ndarray):
                p = [p]

            n_events = self.reader.n_events
            n_pixels = len(p)
            n_samples = self.reader.n_samples
            waveforms = np.zeros((n_events, n_pixels, n_samples))

            for iev, wf in enumerate(self.reader):
                waveforms[iev] = wf[p]

            return waveforms

    @property
    def pixel(self):
        return self._pixel

    @property
    def mapping(self):
        return get_clp_mapping_from_tc_mapping(self.tc_mapping)

    def get_sn(self, tm):
        if tm >= self.n_modules:
            raise IndexError("Requested TM out of range: {}".format(tm))
        return self._reader.GetSN(tm)

    @staticmethod
    def is_compatible(filepath):
        try:
            h = fits.getheader(filepath, 0)
            if 'EVENT_HEADER_VERSION' not in h:
                return False
        except IOError:
            return False
        return True


class ReaderR1(TIOReader):
    """
    Reader for the R1 tio files
    """
    def __init__(self, path, max_events=None):
        super().__init__(path, max_events)
        if not self.is_r1:
            raise IOError("This script is only setup to read *_r1.tio files!")


class ReaderR0(TIOReader):
    """
    Reader for the R0 tio files
    """
    def __init__(self, path, max_events=None):
        super().__init__(path, max_events)
        if self.is_r1:
            raise IOError("This script is only setup to read *_r0.tio files!")


class DL1Writer:
    """
    Writer for the HDF5 DL1 Files
    """
    def __init__(self, dl1_path, totalrows, monitor_path=None):
        print("Creating HDF5 file: {}".format(dl1_path))
        create_directory(os.path.dirname(dl1_path))
        if os.path.exists(dl1_path):
            remove(dl1_path)

        self.totalrows = totalrows
        self.metadata = {}
        self.n_bytes = 0
        self.df_list = []
        self.df_list_n_bytes = 0
        self.monitor = None

        self.store = pd.HDFStore(
            dl1_path, complevel=9, complib='blosc:blosclz'
        )

        if monitor_path:
            self.monitor = MonitorWriter(monitor_path, self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def _append_to_file(self):
        if self.df_list:
            df = pd.concat(self.df_list, ignore_index=True)

            default_c = [
                't_event',
                't_pulse',
                'amp_pulse',
                'charge',
                'fwhm',
                'tr',
                'baseline_start_mean',
                'baseline_start_rms',
                'baseline_end_mean',
                'baseline_end_rms',
                'baseline_subtracted',
                'waveform_mean',
                'waveform_rms',
                'saturation_coeff'
            ]
            for column in default_c:
                if column not in df:
                    df[column] = 0.

            df_float = df.select_dtypes(
                include=['float']
            ).apply(pd.to_numeric, downcast='float')
            df[df_float.columns] = df_float
            df['iev'] = df['iev'].astype(np.uint32)
            df['pixel'] = df['pixel'].astype(np.uint32)
            df['first_cell_id'] = df['first_cell_id'].astype(np.uint16)
            df['t_tack'] = df['t_tack'].astype(np.uint64)
            df['t_event'] = df['t_event'].astype(np.uint16)

            df = df.sort_values(["iev", "pixel"])
            self.store.append('data', df, index=False, data_columns=True,
                              expectedrows=self.totalrows)
            self.n_bytes += df.memory_usage(index=True, deep=True).sum()
            self.df_list = []
            self.df_list_n_bytes = 0

    def append_event(self, df_ev):
        self.df_list.append(df_ev)
        self.df_list_n_bytes += df_ev.memory_usage(index=True, deep=True).sum()
        if self.monitor:
            self.monitor.match_to_data_events(df_ev)
        if self.df_list_n_bytes >= 0.5E9:
            self._append_to_file()

    def add_metadata(self, **kwargs):
        self.metadata = dict(**self.metadata, **kwargs)

    def _save_metadata(self):
        print("Saving data metadata to HDF5 file")
        self.store.get_storer('data').attrs.metadata = self.metadata

    def add_mapping(self, mapping):
        self.store['mapping'] = mapping
        self.store.get_storer('mapping').attrs.metadata = mapping.metadata

    def finish(self):
        self._append_to_file()
        # cols = self.store.get_storer('data').attrs.non_index_axes[0][1]
        # self.store.create_table_index('data', columns=['iev'], kind='full')
        if self.monitor:
            camera_version = self.metadata['camera_version']
            self.monitor.add_metadata(camera_version=camera_version)
            self.monitor.finish()
        self.add_metadata(n_bytes=self.n_bytes)
        self._save_metadata()
        self.store.close()


class MonitorWriter:
    """
    Read the monitor ASCII file and create the monitor DataFrame in the
    DL1 file
    """

    def __init__(self, monitor_path, dl1_writer):
        print("WARNING MonitorWriter assumes monitor timestamps are in "
              "MPIK time. This needs fixing once they have been converted "
              "to unix/UTC")

        print("Reading monitor information from: {}".format(monitor_path))
        if not os.path.exists(monitor_path):
            FileNotFoundError("Cannot find monitor file: {}"
                              .format(monitor_path))

        self.store = dl1_writer.store

        self.supported = [
            "TM_T_PRI",
            "TM_T_AUX",
            "TM_T_PSU",
            "TM_T_SIPM"
        ]

        self.metadata = {}
        self.n_bytes = 0
        self.df_list = []
        self.df_list_n_bytes = 0

        self.n_modules = 32
        self.n_tmpix = 64
        self.empty_df = pd.DataFrame(dict(
            imon=0,
            t_cpu=pd.to_datetime(0, unit='ns'),
            module=np.arange(self.n_modules),
            **dict.fromkeys(self.supported, np.nan)
        ))
        self.first = True
        self.eof = False
        self.aeof = False
        self.t_delta_max = pd.Timedelta(0)

        self.monitor_it = self._get_next_monitor_event(monitor_path)
        try:
            self.monitor_ev = next(self.monitor_it)
        except StopIteration:
            print("WARNING: No monitor events found in file")
            self.monitor_ev = self.empty_df.copy()
            self.eof = True
        self.next_monitor_ev = self.monitor_ev.copy()

    def _get_next_monitor_event(self, monitor_path):
        imon = 0
        t_cpu = 0
        start_time = 0
        df = self.empty_df.copy()
        with open(monitor_path) as file:
            for line in file:
                if line:
                    try:
                        data = line.replace('\n', '').split(" ")

                        t_cpu = pd.to_datetime(
                            "{} {}".format(data[0], data[1]),
                            format="%Y-%m-%d %H:%M:%S:%f"
                        )
                        # TODO: store monitor ASCII with UTC timestamps
                        t_cpu -= pd.Timedelta(1, unit='h')

                        if 'Monitoring Event Done' in line:
                            if not start_time:
                                start_time = t_cpu
                            df.loc[:, 'imon'] = imon
                            df.loc[:, 't_cpu'] = t_cpu
                            self.append_monitor_event(df)
                            yield df
                            imon += 1
                            df = self.empty_df.copy()
                            continue

                        if len(data) < 6:
                            continue

                        device = data[2]
                        measurement = data[3]
                        key = device + "_" + measurement
                        if key in self.supported:
                            device_id = int(data[4])
                            value = float(data[5])

                            df.loc[device_id, key] = value

                    except ValueError:
                        print("ValueError from line: {}".format(line))

            metadata = dict(
                input_path=monitor_path,
                n_events=imon,
                start_time=start_time,
                end_time=t_cpu,
                n_modules=self.n_modules,
                n_tmpix=self.n_tmpix
            )
            self.add_metadata(**metadata)

    def match_to_data_events(self, data_ev):
        t_cpu_data = data_ev.loc[0, 't_cpu']
        t_cpu_next_monitor = self.next_monitor_ev.loc[0, 't_cpu']
        delta = t_cpu_next_monitor - t_cpu_data
        if self.first and delta > pd.Timedelta(5, unit='m'):
            print("WARNING: events are >5 minutes before start of monitor "
                  "file, are you sure it is the correct monitor file?")
            self.first = False
        if self.t_delta_max < delta:
            self.t_delta_max = delta

        # Get next monitor event until the times match
        while (t_cpu_data > t_cpu_next_monitor) and not self.eof:
            try:
                self.monitor_ev = self.next_monitor_ev.copy()
                self.next_monitor_ev = next(self.monitor_it)
                t_cpu_next_monitor = self.next_monitor_ev.loc[0, 't_cpu']
            except StopIteration:
                self.eof = True
                # Use last monitor event for t_delta_max seconds
                imon = self.monitor_ev.loc[0, 'imon']
                t_cpu = self.monitor_ev.loc[0, 't_cpu']
                self.next_monitor_ev = self.empty_df.copy()
                self.next_monitor_ev.loc[:, 'imon'] = imon + 1
                self.next_monitor_ev.loc[:, 't_cpu'] = t_cpu + self.t_delta_max
                t_cpu_next_monitor = self.next_monitor_ev.loc[0, 't_cpu']
        if self.eof and (t_cpu_data > t_cpu_next_monitor) and not self.aeof:
            # Add empty monitor event to file
            print("WARNING: End of monitor events reached, "
                  "setting new monitor items to NaN")
            self.monitor_ev = self.next_monitor_ev.copy()
            self.append_monitor_event(self.monitor_ev)
            self.aeof = True

        imon = self.monitor_ev.loc[0, 'imon']
        module = data_ev.loc[:, 'pixel'] // self.n_tmpix
        data_ev['monitor_index'] = imon * self.n_modules + module

    def _append_to_file(self):
        if self.df_list:
            df = pd.concat(self.df_list, ignore_index=True)

            df_float = df.select_dtypes(
                include=['float']
            ).apply(pd.to_numeric, downcast='float')
            df[df_float.columns] = df_float
            df['imon'] = df['imon'].astype(np.uint32)
            df['module'] = df['module'].astype(np.uint8)

            self.store.append('monitor', df, index=False, data_columns=True)
            self.n_bytes += df.memory_usage(index=True, deep=True).sum()
            self.df_list = []
            self.df_list_n_bytes = 0

    def append_monitor_event(self, df_ev):
        self.df_list.append(df_ev)
        self.df_list_n_bytes += df_ev.memory_usage(index=True, deep=True).sum()
        if self.df_list_n_bytes >= 0.5E9:
            self._append_to_file()

    def add_metadata(self, **kwargs):
        self.metadata = dict(**self.metadata, **kwargs)

    def _save_metadata(self):
        print("Saving monitor metadata to HDF5 file")
        self.store.get_storer('monitor').attrs.metadata = self.metadata

    def finish(self):
        # Finish processing monitor file
        for _ in self.monitor_it:
            pass
        self._append_to_file()
        self.add_metadata(n_bytes=self.n_bytes)
        self._save_metadata()


class HDFStoreReader(ABC):
    """
    Base class for reading from HDFStores
    """
    @abstractmethod
    def __init__(self):
        self.store = None
        self.key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.store.close()

    @property
    def metadata(self):
        return self.store.get_storer(self.key).attrs.metadata

    @property
    def columns(self):
        return self.store.get_storer(self.key).attrs.non_index_axes[0][1]

    @property
    def n_rows(self):
        return self.store.get_storer(self.key).nrows

    @property
    def n_bytes(self):
        return self.metadata['n_bytes']

    @property
    def n_events(self):
        return self.metadata['n_events']

    @property
    def n_modules(self):
        return self.metadata['n_modules']

    @property
    def mapping(self):
        df = self.store['mapping']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            df.metadata = self.store.get_storer('mapping').attrs.metadata
        if 'size' not in df.metadata:
            print("WARNING: This file is outdated, please re-generate it "
                  "using scripts/extract_dl1.py")
            df_row = df.loc[df['row'] == df.metadata.n_rows/2]
            x = df_row['xpix'].values
            df.metadata['size'] = np.min(np.diff(np.sort(x)))
        return df

    @property
    def tc_mapping(self):
        from target_calib import CameraConfiguration
        version = self.mapping.metadata.version
        camera_config = CameraConfiguration(version)
        return camera_config.GetMapping(self.n_modules == 1)

    def get_sn(self, tm):
        if tm >= self.n_modules:
            raise IndexError("Requested TM out of range: {}".format(tm))
        return self.metadata["TM{}_SN".format(tm)]

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
        print("Loading entire DataFrame from HDF5 file")
        if (self.n_bytes > 8E9) and not force:
            raise MemoryError("DataFrame is larger than 8GB, "
                              "set force=True to proceed with loading the "
                              "entire DataFrame into memory")
        if self.n_bytes > 8E9:
            print("WARNING: DataFrame is larger than 8GB")
        return self.store[self.key]

    def select(self, **kwargs):
        """
        Use the pandas.HDFStore.select method to select a subset of the
        DataFrame.

        Parameters
        ----------
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        df : `pandas.DataFrame`

        """
        return self.store.select(self.key, **kwargs)

    def select_column(self, column, **kwargs):
        """
        Use the pandas.HDFStore.select_column method to obtain a single
        column

        Parameters
        ----------
        column : str
            The column you wish to obtain
        kwargs
            Arguments to pass to pandas.HDFStore.select

        Returns
        -------
        `pandas.Series`

        """
        return self.store.select_column(self.key, column, **kwargs)

    def select_columns(self, columns, **kwargs):
        """
        Use the pandas.HDFStore.select_column method to obtain a list of
        columns as numpy arrays

        Parameters
        ----------
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
            values.append(self.select_column(c, **kwargs))
        return values

    def iterate_over_rows(self):
        """
        Loop over the each event in the file, therefore avoiding loading the
        entire table into memory.

        Returns
        -------
        row : `pandas.Series`

        """
        for row in self.iterate_over_chunks(1):
            yield row

    def iterate_over_events(self):
        """
        Loop over the each event in the file, therefore avoiding loading the
        entire table into memory.

        Returns
        -------
        df : `pandas.DataFrame`

        """
        for df in self.iterate_over_chunks(self.metadata['n_pixels']):
            yield df

    def iterate_over_chunks(self, chunksize=None):
        """
        Loop over the DataFrame in chunks, therefore avoiding loading the
        entire table into memory. The chunksize is automatically defined to
        be approximately 2GB of memory.

        Parameters
        ----------
        chunksize : int
            Size of the chunk. By default it is set to a number of rows that
            is approximately equal to 2GB of memory.

        Returns
        -------
        df : `pandas.DataFrame`

        """
        if not chunksize:
            chunksize = self.n_rows / self.n_bytes * 2E9
        for df in self.store.select(self.key, chunksize=chunksize):
            yield df


class DL1Reader(HDFStoreReader):
    """
    Reader for the HDF5 DL1 Files
    """
    def __init__(self, path):
        super().__init__()
        print("Opening HDF5 file: {}".format(path))
        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist: {}".format(path))
        self.store = pd.HDFStore(
            path, mode='r', complevel=9, complib='blosc:blosclz'
        )
        self.path = path
        self.key = 'data'
        self._monitor = None

    def __getitem__(self, iev):
        start = iev * self.n_pixels
        stop = (iev + 1) * self.n_pixels
        df = self.select(start=start, stop=stop)
        return df

    @property
    def monitor(self):
        if not self._monitor:
            if 'monitor' in self.store:
                self._monitor = MonitorReader(self.store)
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
    def version(self):
        return self.metadata['camera_version']

    @staticmethod
    def is_compatible(filepath):
        try:
            kwargs = dict(mode='r', complevel=9, complib='blosc:blosclz')
            with pd.HDFStore(filepath, **kwargs) as store:
                if 'data' not in store:
                    return False
        except IOError:
            return False
        return True

    def get_monitor_column(self, monitor_index, column_name):
        """
        Get a column from the monitor column corresponding to the
        monitor_index of the currect 'data' DataFrame.

        Parameters
        ----------
        monitor_index : ndarray
            The indicis of the monitor rows requested
        column_name : str
            Column name from the monitor DataFrame

        Returns
        -------

        """
        try:
            column = self.monitor.select_column(column_name)[monitor_index]
        except AttributeError:
            raise AttributeError("No monitor information was included in "
                                 "the creation of this file")
        return column


class MonitorReader(HDFStoreReader):
    def __init__(self, store):
        super().__init__()
        self.key = 'monitor'
        self.store = store

    def iterate_over_events(self):
        for df in self.iterate_over_chunks(self.metadata['n_modules']):
            yield df
