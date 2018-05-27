from CHECLabPy.core.base_reducer import WaveformReducer
import numpy as np


class CtapipeNeighbourPeakIntegrator(WaveformReducer):
    def __init__(self, n_pixels, n_samples, plot=False,
                 mapping=None, **kwargs):
        super().__init__(n_pixels, n_samples, plot, **kwargs)

        if mapping is None:
            raise ValueError("A mapping must be passed "
                             "to CtapipeNeighbourPeakIntegrator")

        from ctapipe.image.charge_extractors import NeighbourPeakIntegrator
        from ctapipe.instrument import TelescopeDescription
        from astropy import units as u

        foclen = 2.283 * u.m
        pix_pos = np.vstack([
            mapping['xpix'].values,
            mapping['xpix'].values
        ]) * u.m
        camera = TelescopeDescription.guess(*pix_pos, foclen).camera

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
            ctapipe_peakpos=peakpos,
        )
        return params
