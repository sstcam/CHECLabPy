import os
import numpy as np
from CHECLabPy.core import child_subclasses
from abc import ABC, abstractmethod


class WaveformReader(ABC):
    """
    Base class for waveform-file readers to define some common interface,
    making it easier to use them flexibly in extract_dl1.py
    """
    def __init__(self, path, max_events=None):
        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist: {}".format(path))
        self.path = path
        self.max_events = max_events

        self.run_id = 0
        self.n_modules = 0
        self.n_pixels = 0
        self.n_superpixels_per_module = 0
        self.n_samples = 0
        self.n_cells = 0
        self.camera_version = ''
        self.reference_pulse_path = None

        self.index = 0
        self.current_tack = 0
        self.first_cell_ids = 0

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

    @abstractmethod
    def _get_event(self, iev):
        pass

    @staticmethod
    @abstractmethod
    def is_compatible(path):
        pass

    @property
    @abstractmethod
    def n_events(self):
        pass

    def get_sn(self, tm):
        return np.nan

    def get_sipm_temp(self, tm):
        return np.nan

    def get_primary_temp(self, tm):
        return np.nan

    def get_sp_dac(self, tm, sp):
        return np.nan

    def get_sp_hvon(self, tm, sp):
        return np.nan

    @classmethod
    def from_path(cls, path, max_events=None):
        """
        Factory method to obtain the correct waveform-file reader class for
        the given file

        Parameters
        ----------
        path : str
            Path to the waveform file
        max_events : int
            Maximum number of events to read from the file

        Returns
        -------

        """
        for subclass in child_subclasses(cls):
            if subclass.is_compatible(path):
                return subclass(path, max_events)
