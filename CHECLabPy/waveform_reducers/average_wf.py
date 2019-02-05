from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np


class AverageWF(WaveformReducer):
    """
    Defines a window about the maximum of the average waveform across all
    pixels to extract charge.
    """
    def __init__(self, n_pixels, n_samples, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)
        self.window_size = self.kwargs.get("window_size", 8)
        self.window_shift = self.kwargs.get("window_shift", 4)

    def _prepare(self, waveforms):
        super()._prepare(waveforms)
        if "t_extract" in self.kwargs: # TODO: Get rid of t_extract? replace with its own extractor...
            t = self.kwargs['t_extract']
        else:
            avg = np.mean(self.waveforms, axis=0)
            t = np.argmax(avg)
            n_samples = self.waveforms.shape[1]
            if t < 10:
                t = 10
            elif t > n_samples - 10:
                t = n_samples - 10
        self.t = t

    @column
    def t_averagewf(self):
        """
        Sample corresponding to the maximum of the average waveform.
        The integration window is defined around this time.
        """
        return self.t

    @column
    def charge_averagewf(self):
        """
        Charge extracted from each pixel using the integration window defined
        from the average waveform.
        """
        start = self.t - self.window_shift
        end = start + self.window_size

        charge = np.sum(self.waveforms[:, start:end], axis=1)
        return charge
