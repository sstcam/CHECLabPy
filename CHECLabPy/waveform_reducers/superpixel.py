from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np


class SPAmplitude(WaveformReducer):
    """
    Values corresponding to the superpixel-averaged waveform
    """
    def __init__(self, n_pixels, n_samples, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)
        self.n_superpixels = n_pixels // 4

    def _prepare(self, waveforms):
        super()._prepare(waveforms)
        self.sp_wf = waveforms.reshape(
            (self.n_superpixels, 4, self.n_samples)
        ).sum(1)

    @column
    def sp_argmax(self):
        """
        Sample corresponding to the maximum of the superpixel-averaged waveform

        """
        return np.repeat(self.sp_wf.argmax(1), 4)

    @column
    def sp_max(self):
        """
        Maximum of the superpixel-averaged waveform
        """
        return np.repeat(self.sp_wf.max(1), 4)
