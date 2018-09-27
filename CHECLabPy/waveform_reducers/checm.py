from CHECLabPy.core.base_reducer import WaveformReducer
from scipy.signal import general_gaussian
import numpy as np


class CHECMReducer(WaveformReducer):
    def __init__(self, n_pixels, n_samples, plot=False, **kwargs):
        super().__init__(n_pixels, n_samples, plot, **kwargs)

        from ctapipe.image.charge_extractors import AverageWfPeakIntegrator

        self.clean_waveform = kwargs.get("clean_waveform", True)
        self.window_size = self.kwargs.get("window_size", 8)
        self.window_shift = self.kwargs.get("window_shift", 4)
        self.integrator = AverageWfPeakIntegrator(
            window_shift=self.window_shift,
            window_width=self.window_size
        )

        self.kernel = general_gaussian(10, p=1.0, sig=32)

    def _clean(self, waveforms):

        # Subtract initial baseline
        baseline_sub = waveforms - np.mean(waveforms[:, :32], axis=1)[:, None]

        # Obtain waveform with pulse masked
        baseline_sub_b = baseline_sub[None, ...]
        window, _ = self.integrator.get_window_from_waveforms(waveforms[None, ...])
        windowed = np.ma.array(baseline_sub_b, mask=window[0])
        no_pulse = np.ma.filled(windowed, 0)[0]

        # Get smooth baseline (no pulse)
        smooth_flat = np.convolve(no_pulse.ravel(), self.kernel, "same")
        smooth_baseline = np.reshape(smooth_flat, waveforms.shape)
        no_pulse_std = np.std(no_pulse, axis=1)
        smooth_baseline_std = np.std(smooth_baseline, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            smooth_baseline *= (no_pulse_std / smooth_baseline_std)[:, None]
            smooth_baseline[~np.isfinite(smooth_baseline)] = 0

        # Get smooth waveform
        smooth_wf = baseline_sub  # self.wf_smoother.apply(baseline_sub)

        # Subtract smooth baseline
        cleaned = smooth_wf - smooth_baseline

        return cleaned

    def _get_charge(self, waveforms):
        if self.clean_waveform:
            waveforms = self._clean(waveforms)

        extract = self.integrator.extract_charge
        charge, peakpos, window = extract(waveforms[None, ...])

        params = dict(
            charge=charge[0],
            ctapipe_peakpos=peakpos[0],
        )
        return params
