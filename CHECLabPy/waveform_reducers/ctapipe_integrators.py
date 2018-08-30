from CHECLabPy.core.base_reducer import WaveformReducer
from CHECLabPy.utils.mapping import get_ctapipe_camera_geometry


class CtapipeNeighbourPeakIntegrator(WaveformReducer):
    def __init__(self, n_pixels, n_samples, plot=False,
                 mapping=None, **kwargs):
        super().__init__(n_pixels, n_samples, plot, **kwargs)

        if mapping is None:
            raise ValueError("A mapping must be passed "
                             "to CtapipeNeighbourPeakIntegrator")

        from ctapipe.image.charge_extractors import NeighbourPeakIntegrator

        camera = get_ctapipe_camera_geometry(mapping)

        self.window_size = self.kwargs.get("window_size", 8)
        self.window_shift = self.kwargs.get("window_shift", 4)
        self.integrator = NeighbourPeakIntegrator(
            window_shift=self.window_shift,
            window_width=self.window_size
        )
        self.integrator.neighbours = camera.neighbor_matrix_where

    def _get_charge(self, waveforms):
        extract = self.integrator.extract_charge
        charge, peakpos, window = extract(waveforms[None, ...])

        params = dict(
            charge=charge[0],
            ctapipe_peakpos=peakpos[0],
        )
        return params


class CtapipeLocalPeakIntegrator(WaveformReducer):
    def __init__(self, n_pixels, n_samples, plot=False, **kwargs):
        super().__init__(n_pixels, n_samples, plot, **kwargs)

        from ctapipe.image.charge_extractors import LocalPeakIntegrator

        self.window_size = self.kwargs.get("window_size", 8)
        self.window_shift = self.kwargs.get("window_shift", 4)
        self.integrator = LocalPeakIntegrator(
            window_shift=self.window_shift,
            window_width=self.window_size
        )

    def _get_charge(self, waveforms):
        extract = self.integrator.extract_charge
        charge, peakpos, window = extract(waveforms[None, ...])

        params = dict(
            charge=charge[0],
            ctapipe_peakpos=peakpos[0],
        )
        return params
