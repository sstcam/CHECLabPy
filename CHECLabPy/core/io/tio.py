import numpy as np
from astropy.io import fits
import warnings
import os
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping
import pandas as pd


class TIOReader:
    def __init__(self, path, max_events=None):
        """
        Utilies TargetIO to read R0 and R1 tio files. Enables easy access to
        the waveforms for anaylsis.

        Waveforms can be read from the file by either indexing this reader or
        iterating over it:

        >>> path = "/path/to/file_r0.tio"
        >>> reader = TIOReader(path)
        >>> wf = reader[3]  # Obtain the waveforms for the third event

        >>> path = "/path/to/file_r0.tio"
        >>> reader = TIOReader(path)
        >>> wfs = reader[:10]  # Obtain the waveforms for the first 10 events

        >>> path = "/path/to/file_r0.tio"
        >>> reader = TIOReader(path)
        >>> for wf in reader:  # Iterate over all events in the file
        >>>    print(wf)

        Parameters
        ----------
        path : str
            Path to the _r0.tio or _r1.tio file
        max_events : int
            Maximum number of events to read from the file
        """
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

        self.n_cells = self._camera_config.GetNCells()
        self.camera_version = self._camera_config.GetVersion()
        self.reference_pulse_path = self._camera_config.GetReferencePulsePath()

        self.current_tack = None
        self.current_cpu_ns = None
        self.current_cpu_s = None

        self.first_cell_ids = np.zeros(self.n_pixels, dtype=np.uint16)
        self.stale = np.zeros(self.n_pixels, dtype=np.uint8)

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
        try:  # TODO: Remove try in future version
            self.get_tio_event(iev, self.samples, self.first_cell_ids,
                               self.stale)
        except TypeError:
            warnings.warn(
                "This call to WaveformArrayReader has been deprecated. "
                "Please update TargetIO",
                SyntaxWarning
            )
            self.get_tio_event(iev, self.samples, self.first_cell_ids)
        self.current_tack = self._reader.fCurrentTimeTack
        self.current_cpu_ns = self._reader.fCurrentTimeNs
        self.current_cpu_s = self._reader.fCurrentTimeSec
        return self.samples

    def __iter__(self):
        for iev in range(self.n_events):
            yield self._get_event(iev)

    def __getitem__(self, iev):
        if isinstance(iev, slice):
            ev_list = [self[ii] for ii in range(*iev.indices(self.n_events))]
            return np.array(ev_list)
        elif isinstance(iev, list):
            ev_list = [self[ii] for ii in iev]
            return np.array(ev_list)
        elif isinstance(iev, int):
            if iev < 0:
                iev += self.n_events
            if iev < 0 or iev >= len(self):
                raise IndexError("The requested event ({}) is out of range"
                                 .format(iev))
            return np.copy(self._get_event(iev))
        else:
            raise TypeError("Invalid argument type")

    def __len__(self):
        return self.n_events

    @property
    def t_cpu(self):
        return pd.to_datetime(
            np.int64(self.current_cpu_s * 1E9) + np.int64(self.current_cpu_ns),
            unit='ns'
        )

    @property
    def mapping(self):
        return get_clp_mapping_from_tc_mapping(self.tc_mapping)

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
