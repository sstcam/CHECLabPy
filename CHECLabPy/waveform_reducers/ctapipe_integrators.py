from CHECLabPy.core.reducer import WaveformReducer, column
from CHECLabPy.utils.mapping import get_ctapipe_camera_geometry


class CtapipeLocalPeakIntegrator(WaveformReducer):
    """
    Utilises ctapipe's LocalPeakIntegrator to define an integration window for
    each pixel, based on the maximum of the waveform in that pixel.
    """
    def __init__(self, n_pixels, n_samples, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)

        try:
            from ctapipe.image.extractor import LocalPeakWindowSum
        except ImportError:
            msg = ("ctapipe not found. Please either install ctapipe or "
                   "disable the columns from WaveformReducer {} ({})"
                   .format(self.__class__.__name__, self.columns))
            raise ImportError(msg)

        self.window_size = self.kwargs.get("window_size", 8)
        self.window_shift = self.kwargs.get("window_shift", 4)
        self.integrator = LocalPeakWindowSum(
            window_shift=self.window_shift,
            window_width=self.window_size
        )

    def _prepare(self, waveforms):
        super()._prepare(waveforms)
        charge, pulse_time = self.integrator(waveforms[None, ...])

        self.t = pulse_time[0]
        self.charge = charge[0]

    @column
    def t_local(self):
        """
        Time corresponding to the maximum peak in the waveform, at which the
        charge is extracted.
        """
        return self.t

    @column
    def charge_local(self):
        """
        Charge extracted from the integration window defined locally.
        """
        return self.charge


class CtapipeNeighbourPeakIntegrator(WaveformReducer):
    """
    Utilises ctapipe's NeighbourPeakIntegrator to define an integration window
    for each pixel, based on the maximum of the average waveform across the
    neighbouring pixels.
    """
    def __init__(self, n_pixels, n_samples, mapping=None, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)

        if mapping is None:
            raise ValueError("A mapping must be passed "
                             "to CtapipeNeighbourPeakIntegrator")

        try:
            from ctapipe.image.extractor import NeighborPeakWindowSum
        except ImportError:
            msg = ("ctapipe not found. Please either install ctapipe or "
                   "disable the columns from WaveformReducer {} ({})"
                   .format(self.__class__.__name__, self.columns))
            raise ImportError(msg)

        camera = get_ctapipe_camera_geometry(mapping)

        self.window_size = self.kwargs.get("window_size", 8)
        self.window_shift = self.kwargs.get("window_shift", 4)
        self.integrator = NeighborPeakWindowSum(
            window_shift=self.window_shift,
            window_width=self.window_size,
            lwt=0,
        )
        self.integrator.neighbors = camera.neighbor_matrix_where

    def _prepare(self, waveforms):
        super()._prepare(waveforms)
        charge, pulse_time = self.integrator(waveforms[None, ...])

        self.t = pulse_time[0]
        self.charge = charge[0]

    @column
    def t_nn(self):
        """
        Time corresponding to the maximum peak in the average waveform across
        the neighbouring pixels, at which the charge is extracted.
        """
        return self.t

    @column
    def charge_nn(self):
        """
        Charge extracted from the integration window defined from the
        neighbours.
        """
        return self.charge
