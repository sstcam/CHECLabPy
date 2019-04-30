from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np


class Common(WaveformReducer):
    """
    Extracts common information about the waveform
    """
    @column
    def baseline_start_mean(self):
        """
        Mean of the first 20 samples.
        """
        return np.mean(self.waveforms[:, 0:20], axis=1)

    @column
    def baseline_start_rms(self):
        """
        Standard deviation of the first 20 samples.
        """
        return np.std(self.waveforms[:, 0:20], axis=1)

    @column
    def baseline_end_mean(self):
        """
        Mean of the last 20 samples.
        """
        return np.mean(self.waveforms[:, -20:], axis=1)

    @column
    def baseline_end_rms(self):
        """
        Standard deviation of the last 20 samples.
        """
        return np.std(self.waveforms[:, -20:], axis=1)

    @column
    def waveform_mean(self):
        """
        Mean of all samples in the waveform.
        """
        return np.mean(self.waveforms, axis=1)

    @column
    def waveform_rms(self):
        """
        Standard deviation of all samples in the waveform.
        """
        return np.std(self.waveforms, axis=1)

    @column
    def waveform_max(self):
        """
        Maxima of the waveform.
        """
        return np.max(self.waveforms, axis=1)
