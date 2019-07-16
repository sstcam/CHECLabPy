from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np
from CHECLabPy.utils.mapping import get_ctapipe_camera_geometry
from scipy.ndimage import correlate1d


class SlidingWindow(WaveformReducer):
    """
    Extract the charge from a waveform using a window whose position is
    defined by the maximum returned when sliding across the average waveform
    """

    def __init__(self, n_pixels, n_samples, window_size=8, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)
        self.window_size = window_size
        self.window = np.ones(window_size)
        self.origin = -(self.window.size//2)

        try:
            from ctapipe.image.extractor import sum_samples_around_peak, \
                extract_pulse_time_around_peak
        except ImportError:
            msg = ("ctapipe not found. Please either install ctapipe or "
                   "disable the columns from WaveformReducer {} ({})"
                   .format(self.__class__.__name__, self.columns))
            raise ImportError(msg)

        self.sum_samples_around_peak = sum_samples_around_peak
        self.extract_pulse_time_around_peak = extract_pulse_time_around_peak

    def _prepare(self, waveforms):
        super()._prepare(waveforms)
        avg_wf = np.mean(self.waveforms, axis=0)
        self._peak_index = correlate1d(
            avg_wf, self.window, mode='constant', origin=self.origin
        ).argmax()

    @column
    def t_sliding(self):
        """
        Pulse time extracted using the weighted average of the samples in the
        window
        """
        return self.extract_pulse_time_around_peak(
            self.waveforms, self._peak_index, self.window_size, 0
        )

    @column
    def charge_sliding(self):
        """
        Charge extracted from the sliding window.
        """
        return self.sum_samples_around_peak(
            self.waveforms, self._peak_index, self.window_size, 0
        )


class SlidingWindowLocal(SlidingWindow):
    """
    Same as `SlidingWindow`, except done per-pixel
    """

    def _prepare(self, waveforms):
        WaveformReducer._prepare(self, waveforms)
        self.windowed_sum = correlate1d(
            self.waveforms, self.window, mode='constant', origin=self.origin
        )

    @column
    def t_sliding_local(self):
        """
        Pulse time extracted from the maximum sliding window per pixel
        """
        peak_index = self.windowed_sum.argmax(1)
        return self.extract_pulse_time_around_peak(
            self.waveforms, peak_index, self.window_size, 0
        )

    @column
    def charge_sliding_local(self):
        """
        Charge extracted from the maximum sliding window per pixel
        """
        return self.windowed_sum.max(1)


class SlidingWindowNeighbour(SlidingWindow):
    """
    Same as `SlidingWindow`, except calculated according to the pixel neighbors
    """
    def __init__(self, n_pixels, n_samples, mapping=None, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)

        if mapping is None:
            raise ValueError("A mapping must be passed "
                             "to SlidingWindowNeighbour")

        try:
            from ctapipe.image.extractor import neighbor_average_waveform
        except ImportError:
            msg = ("ctapipe not found. Please either install ctapipe or "
                   "disable the columns from WaveformReducer {} ({})"
                   .format(self.__class__.__name__, self.columns))
            raise ImportError(msg)

        camera = get_ctapipe_camera_geometry(mapping)
        self.neighbor_func = neighbor_average_waveform
        self.neighbors = camera.neighbor_matrix_where

    def _prepare(self, waveforms):
        WaveformReducer._prepare(self, waveforms)
        avg_wfs = self.neighbor_func(self.waveforms, self.neighbors, 0)
        self._peak_index = correlate1d(
            avg_wfs, self.window, mode='constant', origin=self.origin
        ).argmax(1)

    @column
    def t_sliding_nn(self):
        """
        Pulse time extracted using the weighted average of the samples in the
        window defined by the pixel neighbors
        """
        return self.extract_pulse_time_around_peak(
            self.waveforms, self._peak_index, self.window_size, 0
        )

    @column
    def charge_sliding_nn(self):
        """
        Charge extracted from the sliding window defined by the pixel neighbors
        """
        return self.sum_samples_around_peak(
            self.waveforms, self._peak_index, self.window_size, 0
        )
