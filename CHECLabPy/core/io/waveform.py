import os
import numpy as np
import pandas as pd
from CHECLabPy.core import child_subclasses
from abc import ABC, abstractmethod


class Waveform(np.ndarray):
    def __new__(cls, input_array, iev, is_r1=False,
                first_cell_id=0, stale=False, missing_packets=False, t_tack=0,
                t_cpu_container=0, mc_true=None):
        obj = np.asarray(input_array).view(cls)
        obj.iev = iev
        obj.is_r1 = is_r1
        obj.first_cell_id = first_cell_id
        obj.stale = stale
        obj.missing_packets = missing_packets
        obj.t_tack = t_tack
        obj._t_cpu_container = t_cpu_container
        obj.mc_true = mc_true
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.iev = getattr(obj, 'iev', None)
        self.is_r1 = getattr(obj, 'is_r1', None)
        self.first_cell_id = getattr(obj, 'first_cell_id', None)
        self.stale = getattr(obj, 'stale', None)
        self.missing_packets = getattr(obj, 'missing_packets', None)
        self.t_tack = getattr(obj, 't_tack', None)
        self._t_cpu_container = getattr(obj, '_t_cpu_container', None)
        self.mc_true = getattr(obj, 'mc_true', None)

    @property
    def t_cpu(self):
        t_cpu_s, t_cpu_ns = self._t_cpu_container
        return pd.to_datetime(
            np.int64(t_cpu_s * 1E9) + np.int64(t_cpu_ns),
            unit='ns'
        )


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

    def __iter__(self):
        for iev in range(self.n_events):
            yield self._get_event(iev)

    def __getitem__(self, iev):
        if isinstance(iev, int):
            if iev < 0:
                iev += self.n_events
            if iev < 0 or iev >= len(self):
                raise IndexError(
                    "The requested event ({}) is out of range".format(iev)
                )
            return self._get_event(iev)
        elif isinstance(iev, slice):
            ev_list = [self[ii] for ii in range(*iev.indices(self.n_events))]
            return np.array(ev_list)
        elif isinstance(iev, list):
            ev_list = [self[ii] for ii in iev]
            return np.array(ev_list)
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
