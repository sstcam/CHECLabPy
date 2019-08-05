from astropy.io import fits
from CHECLabPy.core.io.waveform import WaveformReader, Waveform
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping
import numpy as np
import gzip


class TIOReader(WaveformReader):
    def __init__(self, path, max_events=None,
                 skip_events=2, skip_end_events=1):
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
        super().__init__(path, max_events)

        try:
            from target_io import WaveformArrayReader
            from target_calib import CameraConfiguration
        except ModuleNotFoundError:
            msg = ("Cannot find TARGET libraries, please follow installation "
                   "instructions from https://forge.in2p3.fr/projects/gct/"
                   "wiki/Installing_CHEC_Software")
            raise ModuleNotFoundError(msg)

        self._reader = WaveformArrayReader(
            self.path, skip_events, skip_end_events
        )

        self.is_r1 = self._reader.fR1
        self._n_events = self._reader.fNEvents
        self.run_id = self._reader.fRunID
        self.n_pixels = self._reader.fNPixels
        self.n_superpixels_per_module = self._reader.fNSuperpixelsPerModule
        self.n_modules = self._reader.fNModules
        self.n_tmpix = self.n_pixels // self.n_modules
        self.n_samples = self._reader.fNSamples

        self._camera_config = CameraConfiguration(self._reader.fCameraVersion)
        self.tc_mapping = self._camera_config.GetMapping(self.n_modules == 1)

        self.n_cells = self._camera_config.GetNCells()
        self.camera_version = self._camera_config.GetVersion()
        self.reference_pulse_path = self._camera_config.GetReferencePulsePath()

        if self.is_r1:
            self.dtype = np.float32
            self.get_tio_event = self._reader.GetR1Event
        else:
            self.dtype = np.uint16
            self.get_tio_event = self._reader.GetR0Event

        if max_events and max_events < self._n_events:
            self._n_events = max_events

    def _get_event(self, iev):
        samples = np.zeros((self.n_pixels, self.n_samples), self.dtype)
        first_cell_id = np.zeros(self.n_pixels, dtype=np.uint16)
        stale = np.zeros(self.n_pixels, dtype=np.uint8)
        self.get_tio_event(iev, samples, first_cell_id, stale)
        t_tack, t_cpu_s, t_cpu_ns = self._reader.GetTimestamps(iev)
        waveform = Waveform(
            input_array=samples,
            iev=iev,
            is_r1=self.is_r1,
            first_cell_id=first_cell_id,
            stale=stale,
            t_tack=t_tack,
            t_cpu_container=(t_cpu_s, t_cpu_ns),
        )
        return waveform

    @staticmethod
    def is_compatible(path):
        with open(path, 'rb') as f:
            marker_bytes = f.read(1024)

        # if file is gzip, read the first 4 bytes with gzip again
        if marker_bytes[0] == 0x1f and marker_bytes[1] == 0x8b:
            with gzip.open(path, 'rb') as f:
                marker_bytes = f.read(1024)

        if b'FITS' not in marker_bytes:
            return False

        try:
            h = fits.getheader(path, 0)
            if 'EVENT_HEADER_VERSION' not in h:
                return False
        except IOError:
            return False
        return True

    @property
    def n_events(self):
        return self._n_events

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

    def get_sipm_temp(self, tm):
        if tm >= self.n_modules:
            raise IndexError("Requested TM out of range: {}".format(tm))
        return self._reader.GetSiPMTemp(tm)

    def get_primary_temp(self, tm):
        if tm >= self.n_modules:
            raise IndexError("Requested TM out of range: {}".format(tm))
        return self._reader.GetPrimaryTemp(tm)

    def get_sp_dac(self, tm, sp):
        if tm >= self.n_modules:
            raise IndexError("Requested TM out of range: {}".format(tm))
        if sp >= self.n_superpixels_per_module:
            raise IndexError("Requested SP out of range: {}".format(sp))
        return self._reader.GetSPDAC(tm, sp)

    def get_sp_hvon(self, tm, sp):
        if tm >= self.n_modules:
            raise IndexError("Requested TM out of range: {}".format(tm))
        if sp >= self.n_superpixels_per_module:
            raise IndexError("Requested SP out of range: {}".format(sp))
        return self._reader.GetSPHVON(tm, sp)


class ReaderR1(TIOReader):
    """
    Reader for the R1 tio files
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_r1:
            raise IOError("This script is only setup to read *_r1.tio files!")


class ReaderR0(TIOReader):
    """
    Reader for the R0 tio files
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_r1:
            raise IOError("This script is only setup to read *_r0.tio files!")
