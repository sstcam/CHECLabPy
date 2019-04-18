from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np
from scipy import interpolate
from CHECLabPy.utils.mapping import get_ctapipe_camera_geometry
from numba import njit, prange, float64, float32


@njit([
    float64[:, :](float64[:, :], float64[:]),
    float64[:, :](float32[:, :], float64[:]),
], parallel=True, nogil=True)
def correlate1d(waveforms, ref_pulse):
    n_pixels, n_samples = waveforms.shape
    ref_t_start = ref_pulse.size // 2
    ref_t_end = ref_t_start + n_samples
    cc_res = np.zeros((n_pixels, n_samples))
    for ipix in prange(n_pixels):
        for t in range(n_samples):
            start = ref_t_start - t
            end = ref_t_end - t
            cc_res[ipix, t] = np.sum(waveforms[ipix] * ref_pulse[start:end])
    return cc_res


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
        self.reference_pulse = np.pad(self.reference_pulse, n_samples, 'constant')
        self.cc = None
        self.t = None
        self.charge = None

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

        # Put pulse in center so result peak time matches with input peak
        pad = y.size - 2 * np.argmax(y)
        if pad > 0:
            y = np.pad(y, (pad, 0), mode='constant')
        else:
            y = np.pad(y, (0, -pad), mode='constant')

        # Create 1p.e. pulse shape
        y_1pe = y / np.trapz(y)

        # Make maximum of cc result == 1
        y_pad = np.pad(y, y.size, 'constant')
        y = y / correlate1d(y_1pe[None, :], y_pad).max()

        return y, y_1pe

    def get_pulse_height(self, charge):
        return charge * self.y_1pe.max()

    def get_reference_pulse_at_t(self, t):
        ref_pad = np.pad(self.reference_pulse, self.n_samples, 'constant')
        ref_t_start = ref_pad.size // 2
        ref_t_end = ref_t_start + self.n_samples
        if t > self.n_samples:
            raise IndexError
        start = ref_t_start - t
        end = ref_t_end - t
        return ref_pad[start:end]

    def _prepare(self, waveforms):
        super()._prepare(waveforms)
        self.cc = correlate1d(waveforms, self.reference_pulse)
        if "t_extract" in self.kwargs:
            t = self.kwargs['t_extract']
        else:
            avg = np.mean(self.cc, axis=0)
            t = np.argmax(avg)
            if t < 10:
                t = 10
            elif t > self.n_samples - 10:
                t = self.n_samples - 10
        self.t = t
        self.charge = self.cc[:, self.t]

    @column
    def t_cc(self):
        """
        Sample corresponding to the maximum of the average cross correlation
        result. The charge is extracted at this sample.
        """
        return self.t

    @column
    def charge_cc(self):
        """
        Charge extracted from the cross correlation.
        """
        return self.charge

    @column
    def height_cc(self):
        """
        Height of a reference pulse with a cross correlation charge equal to
        that extracted.
        """
        return self.get_pulse_height(self.charge)


class CrossCorrelationLocal(CrossCorrelation):
    """
    Same as `CrossCorrelation`, except combined with ctapipe's
    `LocalPeakIntegrator` to instead use the maximum sample in each pixel for
    charge extraction.
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

        self.integrator = LocalPeakWindowSum(
            window_shift=0,
            window_width=1,
        )

    def _prepare(self, waveforms):
        WaveformReducer._prepare(self, waveforms)
        self.cc = correlate1d(waveforms, self.reference_pulse)
        charge, pulse_time = self.integrator(self.cc[None, ...])

        self.t = pulse_time[0]
        self.charge = charge[0]

    @column
    def t_cc_local(self):
        """
        Time of maximum of the cross correlation result in each pixel.
        """
        return self.t

    @column
    def charge_cc_local(self):
        """
        Charge extracted from the cross correlation at the local maximum.
        """
        return self.charge

    @column
    def height_cc_local(self):
        """
        Height of a reference pulse with a cross correlation charge equal to
        that extracted.
        """
        return self.get_pulse_height(self.charge)


class CrossCorrelationNeighbour(CrossCorrelation):
    """
    Same as `CrossCorrelation`, except combined with ctapipe's
    `NeighbourPeakIntegrator` to instead use the maximum sample from the
    average of the neighbouring pixels for charge extraction.
    """
    def __init__(self, n_pixels, n_samples, mapping=None, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)

        if mapping is None:
            raise ValueError("A mapping must be passed "
                             "to CrossCorrelationNeighbour")

        try:
            from ctapipe.image.extractor import NeighborPeakWindowSum
        except ImportError:
            msg = ("ctapipe not found. Please either install ctapipe or "
                   "disable the columns from WaveformReducer {} ({})"
                   .format(self.__class__.__name__, self.columns))
            raise ImportError(msg)

        camera = get_ctapipe_camera_geometry(mapping)

        self.integrator = NeighborPeakWindowSum(
            window_shift=0,
            window_width=1,
            lwt=0,
        )
        self.integrator.neighbors = camera.neighbor_matrix_where

    def _prepare(self, waveforms):
        WaveformReducer._prepare(self, waveforms)
        self.cc = correlate1d(waveforms, self.reference_pulse)
        charge, pulse_time = self.integrator(self.cc[None, ...])

        self.t = pulse_time[0]
        self.charge = charge[0]

    @column
    def t_cc_nn(self):
        """
        Time of maximum of the average cross correlation result across the
        neighbouring pixels.
        """
        return self.t

    @column
    def charge_cc_nn(self):
        """
        Charge extracted from the cross correlation at the neighbour maximum.
        """
        return self.charge

    @column
    def height_cc_nn(self):
        """
        Height of a reference pulse with a cross correlation charge equal to
        that extracted.
        """
        return self.get_pulse_height(self.charge)
