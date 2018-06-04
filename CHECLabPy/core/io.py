import numpy as np
import pandas as pd
from astropy.io import fits
import warnings
import os
from os import remove
import json
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
            self.monitor = MonitorWriter(monitor_path, self.store)

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
        if self.monitor:
            self.monitor.match_to_data_events(df_ev)
        self.df_list.append(df_ev)
        self.df_list_n_bytes += df_ev.memory_usage(index=True, deep=True).sum()
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

    def __init__(self, monitor_path, store):
        print("WARNING: MonitorWriter assumes monitor timestamps are in "
              "MPIK time. This needs fixing once they have been converted "
              "to unix/UTC")

        print("Reading monitor information from: {}".format(monitor_path))
        if not os.path.exists(monitor_path):
            FileNotFoundError("Cannot find monitor file: {}"
                              .format(monitor_path))

        self.store = store

        self.supported_camera = [
            "CAM_CameraState",
            "CAM_EventBuildingStatus",
            "CAM_EventRate",
            "CAM_TotalPacketsReceived",
            "CAM_BadPacketsReceived",
            "CH_POWER",
            "CH_PUMP_1",
            "CH_T_AMBIENT",
            "CH_T_SET",
            "CH_T_WATER_IN",
            "CH_T_WATER_OUT",
            "DACQ_T_DACQ1",
            "DACQ_T_DACQ2"
        ]
        self.supported_tm = [
            "TM_T_PRI",
            "TM_T_AUX",
            "TM_T_PSU",
            "TM_T_SIPM",
            "TM_SiPM_I",
            "TM_SiPM_V"
        ]
        self.supported_pixel = []

        self.metadata = {}
        self.n_bytes = 0
        self.df_camera_list = []
        self.df_tm_list = []
        self.df_pixel_list = []
        self.df_list_n_bytes = 0
        self.t_cpu_list = []

        self.n_modules = 32
        self.n_tmpix = 64
        self.n_pixels = self.n_modules * self.n_tmpix
        self.first = True
        self.eof = False
        self.aeof = False
        self.t_delta_max = pd.Timedelta(0)

        self._load_monitor_data(monitor_path)
        self.t_delta_max = np.diff(self.t_cpu_list).max()

    def _get_new_dfs(self, imon, t_cpu):
        df_camera = pd.DataFrame(dict(
            imon=imon,
            t_cpu=t_cpu,
            idevice=0,
            **dict.fromkeys(self.supported_camera, np.nan)
        ), index=np.array([0]))
        df_tm = pd.DataFrame(dict(
            imon=imon,
            t_cpu=t_cpu,
            idevice=np.arange(self.n_modules),
            **dict.fromkeys(self.supported_tm, np.nan)
        ))
        df_pixel = pd.DataFrame(dict(
            imon=imon,
            t_cpu=t_cpu,
            idevice=np.arange(self.n_pixels),
            **dict.fromkeys(self.supported_pixel, np.nan)
        ))
        return df_camera, df_tm, df_pixel

    def _load_monitor_data(self, monitor_path):
        imon = 0
        null_time = pd.to_datetime(0, unit='ns')
        start_time = null_time
        t_cpu = null_time
        df_camera, df_tm, df_pixel = self._get_new_dfs(0, null_time)
        with open(monitor_path) as file:
            for line in file:
                if line and line != '\n':
                    try:
                        data = line.replace('\n', '').split(" ")

                        t_cpu = pd.to_datetime(
                            "{} {}".format(data[0], data[1]),
                            format="%Y-%m-%d %H:%M:%S:%f"
                        )
                        # TODO: store monitor ASCII with UTC timestamps
                        t_cpu = (t_cpu.tz_localize("Europe/Berlin")
                                 .tz_convert("UTC")
                                 .tz_localize(None))

                        if 'Monitoring Event Done' in line:
                            if imon > 0:
                                self._append_monitor_event(
                                    df_camera, df_tm, df_pixel
                                )
                            if not start_time:
                                start_time = t_cpu
                            self.current_t_cpu = t_cpu
                            new_dfs = self._get_new_dfs(imon, t_cpu)
                            df_camera, df_tm, df_pixel = new_dfs
                            imon += 1
                            continue

                        # if len(data) < 6:
                        #     continue

                        device = data[2]
                        measurement = data[3]
                        key = device + "_" + measurement
                        if key in self.supported_camera:
                            value = float(data[4])
                            df_camera.loc[0, key] = value
                        elif key in self.supported_tm:
                            idevice = int(data[4])
                            value = float(data[5])
                            df_tm.loc[idevice, key] = value
                        elif key in self.supported_pixel:
                            idevice = int(data[4])
                            value = float(data[5])
                            df_pixel.loc[idevice, key] = value

                    except ValueError:
                        print("ValueError from line: {}".format(line))

        metadata = dict(
            input_path=monitor_path,
            start_time=start_time,
            end_time=t_cpu,
            n_modules=self.n_modules,
            n_tmpix=self.n_tmpix,
            n_pixels=self.n_pixels
        )
        self.add_metadata(**metadata)

    def _append_monitor_event(self, df_camera, df_tm, df_pixel):
        self.t_cpu_list.append(self.current_t_cpu)
        self.df_camera_list.append(df_camera)
        self.df_tm_list.append(df_tm)
        self.df_pixel_list.append(df_pixel)
        n_bytes_camera = df_camera.memory_usage(index=True, deep=True).sum()
        n_bytes_tm = df_tm.memory_usage(index=True, deep=True).sum()
        n_bytes_pixel = df_pixel.memory_usage(index=True, deep=True).sum()
        self.df_list_n_bytes += (n_bytes_camera + n_bytes_tm + n_bytes_pixel)
        if self.df_list_n_bytes >= 0.5E9:
            self._append_to_file()

    def _append_to_file(self):
        type_list = [
            ("df_camera_list", 'monitor_camera'),
            ("df_tm_list", 'monitor_tm'),
            ("df_pixel_list", 'monitor_pixel'),
        ]
        for attr, key in type_list:
            df_list = getattr(self, attr)
            if df_list:
                df = pd.concat(df_list, ignore_index=True)

                df_float = df.select_dtypes(
                    include=['float']
                ).apply(pd.to_numeric, downcast='float')
                df[df_float.columns] = df_float
                df['imon'] = df['imon'].astype(np.uint32)
                df['idevice'] = df['idevice'].astype(np.uint16)

                self.store.append(key, df, index=False, data_columns=True)
                self.n_bytes += df.memory_usage(index=True, deep=True).sum()
                setattr(self, attr, [])
                self.df_list_n_bytes = 0

    def add_metadata(self, **kwargs):
        self.metadata = dict(**self.metadata, **kwargs)

    def _save_metadata(self):
        print("Saving monitor metadata to HDF5 file")
        self.store['monitor_metadata'] = pd.DataFrame()
        storer = self.store.get_storer('monitor_metadata')
        storer.attrs.metadata = self.metadata

    def finish(self):
        self._append_to_file()
        n_events = self.store.select_column("monitor_camera", "imon").max() + 1
        self.add_metadata(
            n_bytes=self.n_bytes,
            n_events=n_events
        )
        self._save_metadata()

    def _set_data_monitor_index(self, data_ev, imon):
        pix = data_ev.loc[:, 'pixel']
        tm = pix // self.n_tmpix
        data_ev['monitor_camera_index'] = imon
        data_ev['monitor_tm_index'] = imon * self.n_modules + tm
        data_ev['monitor_pixel_index'] = imon * self.n_pixels + pix

    def match_to_data_events(self, data_ev):
        t_cpu_data = data_ev.loc[0, 't_cpu']
        if self.first:
            delta = self.t_cpu_list[0] - t_cpu_data
            if delta > pd.Timedelta(5, unit='m'):
                print("WARNING: events are >5 minutes before start of monitor "
                      "file, are you sure it is the correct monitor file?")
            self.first = False

        if not self.eof:
            for imon, t_cpu_monitor in enumerate(self.t_cpu_list):
                if t_cpu_data < t_cpu_monitor:
                    self._set_data_monitor_index(data_ev, imon)
                    break
                self.eof = True

        if self.eof:
            t_limit = self.t_cpu_list[-1] + self.t_delta_max * 5
            if t_cpu_data <= t_limit:
                imon = len(self.t_cpu_list) - 1
                self._set_data_monitor_index(data_ev, imon)
            else:
                imon = len(self.t_cpu_list)
                if not self.aeof:
                    # Add empty monitor event to file
                    print("WARNING: End of monitor events reached, "
                          "setting new monitor items to NaN")
                    t_cpu = t_limit
                    dfs = self._get_new_dfs(imon, t_cpu)
                    self._append_monitor_event(*dfs)
                    self.aeof = True
                self._set_data_monitor_index(data_ev, imon)


class DL1Reader:
    """
    Reader for the HDF5 DL1 Files
    """
    def __init__(self, path):
        print("Opening HDF5 file: {}".format(path))
        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist: {}".format(path))
        self.store = pd.HDFStore(
            path, mode='r', complevel=9, complib='blosc:blosclz'
        )
        self.path = path
        self.key = 'data'
        self._monitor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.store.close()

    def __getitem__(self, iev):
        start = iev * self.n_pixels
        stop = (iev + 1) * self.n_pixels
        df = self.select(start=start, stop=stop)
        return df

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
            df_row = df.loc[df['row'] == df.metadata['n_rows']/2]
            x = df_row['xpix'].values
            df.metadata['size'] = np.min(np.diff(np.sort(x)))
        return df

    @property
    def tc_mapping(self):
        from target_calib import CameraConfiguration
        version = self.mapping.metadata.version
        camera_config = CameraConfiguration(version)
        return camera_config.GetMapping(self.n_modules == 1)

    @property
    def monitor(self):
        if not self._monitor:
            if 'monitor_metadata' in self.store:
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

    def get_first_n_events(self, n_events):
        """
        Obtain the first N events from the DL1 file

        Parameters
        ----------
        n_events : int

        Returns
        -------
        df : `pandas.DataFrame`

        """
        n_rows = n_events * self.metadata['n_pixels']
        return next(self.iterate_over_chunks(n_rows))

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


class MonitorReader:
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
            print("WARNING: DataFrame is larger than 8GB")
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
        remove = ['imon', 'idevice', 't_cpu']
        columns = dict(
            monitor_camera=[c for c in columns_camera if c not in remove],
            monitor_tm=[c for c in columns_tm if c not in remove],
            monitor_pixel=[c for c in columns_pixel if c not in remove],
        )
        print(json.dumps(columns, indent=4))
