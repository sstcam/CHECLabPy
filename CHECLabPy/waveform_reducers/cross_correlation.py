from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np
from scipy import interpolate
from CHECLabPy.utils.mapping import get_ctapipe_camera_geometry
from scipy.ndimage import correlate1d
from numba import guvectorize, float64, float32, int64


@guvectorize(
    [
        (float64[:], int64, float64[:]),
        (float32[:], int64, float64[:]),
    ],
    '(s),()->()',
    nopython=True,
)
def extract_pulse_time(waveforms, peak_index, ret):
    n_samples = waveforms.size
    y0 = waveforms[peak_index - 1]
    y1 = waveforms[peak_index]
    y2 = waveforms[peak_index + 1]

    # Quadratic peak interpolation
    a = y0 - y2
    b = y0 - 2 * y1 + y2
    if b == 0:
        ret[0] = peak_index
        return
    t = 0.5 * a / b
    t_pulse = t + peak_index
    if not 0 <= t_pulse < n_samples:
        ret[0] = peak_index
        return
    ret[0] = t_pulse


class CrossCorrelation(WaveformReducer):
    """
    Performs a cross correlation on the waveform with a reference pulse shape
    to extract charge. The cross correlation result acts as a sliding
    integration window that is weighted according to the pulse shape. The
    maximum of the cross correlation result is the point at which the
    reference pulse best matches the waveform. An unbiased
    extraction time is chosed from the maximum of the average of the cross
    correlation result across all pixels.
    """

    def __init__(self, n_pixels, n_samples, reference_pulse_path='', **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)

        if not reference_pulse_path:
            raise ValueError(
                "Missing argument to WaveformReducer/WaveformReducerChain, "
                "please specify reference_pulse_path. "
                "This can be obtained from TIOReader.reference_pulse_path or "
                "SimtelReader.reference_pulse_path."
            )

        ref = self._load_reference_pulse(reference_pulse_path)
        self.reference_pulse, self.y_1pe = ref
        self.cc_origin = (self.reference_pulse.argmax() -
                          self.reference_pulse.size // 2)
        self._cc = np.zeros((2048, 128))
        self._peak_index = np.zeros(2048, dtype=np.int64)
        self._charge = np.zeros(2048)

    @staticmethod
    def _load_reference_pulse(path):
        file = np.loadtxt(path)
        time_slice = 1E-9
        refx = file[:, 0]
        refy = file[:, 1]
        f = interpolate.interp1d(refx, refy, kind=3)
        max_sample = int(refx[-1] / time_slice)
        x = np.linspace(0, max_sample * time_slice, max_sample + 1)
        y = f(x)

        # Create 1p.e. pulse shape
        y_1pe = y / np.trapz(y)

        # Make maximum of cc result == 1
        origin = y.argmax() - y.size // 2
        y = y / correlate1d(
            y_1pe[None, :], y, mode='constant', origin=origin
        ).max()

        return y, y_1pe

    def get_pulse_height(self, charge):
        return charge * self.y_1pe.max()

    def _prepare(self, waveforms):
        super()._prepare(waveforms)
        self._cc = correlate1d(
            waveforms, self.reference_pulse,
            mode='constant', origin=self.cc_origin
        )
        self._peak_index = self._cc.mean(0).argmax()
        self._charge = self._cc[:, self._peak_index]

    @column
    def t_cc(self):
        """
        Pulse time extracted from the cross correlation.
        """
        return extract_pulse_time(self._cc, self._peak_index)

    @column
    def charge_cc(self):
        """
        Charge extracted from the cross correlation.
        """
        return self._charge

    @column
    def height_cc(self):
        """
        Height of a reference pulse with a cross correlation charge equal to
        that extracted.
        """
        return self.get_pulse_height(self._charge)


class CrossCorrelationLocal(CrossCorrelation):
    """
    Same as `CrossCorrelation`, , except uses local-peak-finding
    """
    def __init__(self, n_pixels, n_samples, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)
        self._pa = np.arange(n_pixels)

    def _prepare(self, waveforms):
        WaveformReducer._prepare(self, waveforms)
        self._cc = correlate1d(
            waveforms, self.reference_pulse,
            mode='constant', origin=self.cc_origin
        )

        self._peak_index = self._cc.argmax(1)
        self._charge = self._cc[self._pa, self._peak_index]

    @column
    def t_cc_local(self):
        """
        Time of maximum of the cross correlation result in each pixel.
        """
        return extract_pulse_time(self._cc, self._peak_index)

    @column
    def charge_cc_local(self):
        """
        Charge extracted from the cross correlation at the local maximum.
        """
        return self._charge

    @column
    def height_cc_local(self):
        """
        Height of a reference pulse with a cross correlation charge equal to
        that extracted.
        """
        return self.get_pulse_height(self._charge)


class CrossCorrelationNeighbour(CrossCorrelation):
    """
    Same as `CrossCorrelation`, except uses neighbour-peak-finding
    """
    def __init__(self, n_pixels, n_samples, mapping=None, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)
        self._pa = np.arange(n_pixels)

        if mapping is None:
            raise ValueError("A mapping must be passed "
                             "to CrossCorrelationNeighbour")

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
        self._cc = correlate1d(
            waveforms, self.reference_pulse,
            mode='constant', origin=self.cc_origin
        )
        avg_wfs = self.neighbor_func(self._cc[None, :], self.neighbors, 0)[0]
        self._peak_index = avg_wfs.argmax(1)
        self._charge = self._cc[self._pa, self._peak_index]

    @column
    def t_cc_nn(self):
        """
        Time of maximum of the average cross correlation result across the
        neighbouring pixels.
        """
        return extract_pulse_time(self._cc, self._peak_index)

    @column
    def charge_cc_nn(self):
        """
        Charge extracted from the cross correlation at the neighbour maximum.
        """
        return self._charge

    @column
    def height_cc_nn(self):
        """
        Height of a reference pulse with a cross correlation charge equal to
        that extracted.
        """
        return self.get_pulse_height(self._charge)
