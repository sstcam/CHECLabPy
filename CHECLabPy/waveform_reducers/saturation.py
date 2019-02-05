from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np


class Saturation(WaveformReducer):
    """
    Obtains a coefficient from the waveform that attempts to characterise a
    property that continues to increase with illumination even in the
    saturated regime of the electronics.
    """
    def __init__(self, n_pixels, n_samples, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)
        self.window_shift = self.kwargs.get("window_shift", 4)

    @column
    def saturation_sum(self):
        """
        Sum of the waveform samples from the start of the pulse to the end of
        the waveform.
        """
        if "t_event" in self.kwargs:
            t_event = self.kwargs['t_event']
        else:
            avg = np.mean(self.waveforms, axis=0)
            t_event = np.argmax(avg)
            n_samples = self.waveforms.shape[1]
            if t_event < 10:
                t_event = 10
            elif t_event > n_samples - 10:
                t_event = n_samples - 10
        start = t_event - self.window_shift

        coeff = np.sum(self.waveforms[:, start:], axis=1)
        return coeff
