from CHECLabPy.core.base_reducer import WaveformReducer
import numpy as np
from scipy import interpolate
from scipy.ndimage import correlate1d
from CHECLabPy.utils.mapping import get_ctapipe_camera_geometry


class CrossCorrelation(WaveformReducer):
    """
    Extractor which uses the result of the cross correlation of the waveforms
    with a reference pulse. The cross correlation results acts as a sliding
    integration window that is weighted according to the pulse shape. The
    maximum of the cross correlation result is the point at which the
    reference pulse best matches the waveform. To choose an unbiased
    extraction time I average the cross correlation result across all pixels
    and take the maximum as the peak time.
    """

    def __init__(self, n_pixels, n_samples, plot=False,
                 reference_pulse_path='', **kwargs):
        super().__init__(n_pixels, n_samples, plot, **kwargs)

        ref = self.load_reference_pulse(reference_pulse_path)
        self.reference_pulse, self.y_1pe = ref
        self.cc = None

    @staticmethod
    def load_reference_pulse(path):
        file = np.loadtxt(path)
        print("Loaded reference pulse: {}".format(path))
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
        y = y / correlate1d(y_1pe, y).max()

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

    def _apply_cc(self, waveforms):
        cc = correlate1d(waveforms, self.reference_pulse)
        return cc

    def _set_t_event(self, waveforms):
        self.cc = self._apply_cc(waveforms)
        super()._set_t_event(self.cc)

    def _get_charge(self, waveforms):
        charge = self.cc[:, self.t_event]
        cc_height = self.get_pulse_height(charge)

        params = dict(
            charge=charge,
            cc_height=cc_height,
        )
        return params


class CrossCorrelationPlus1(CrossCorrelation):
    def _get_charge(self, waveforms):
        charge = self.cc[:, self.t_event+1]
        cc_height = self.get_pulse_height(charge)

        params = dict(
            charge=charge,
            cc_height=cc_height,
        )
        return params


class CrossCorrelationMinus1(CrossCorrelation):
    def _get_charge(self, waveforms):
        charge = self.cc[:, self.t_event-1]
        cc_height = self.get_pulse_height(charge)

        params = dict(
            charge=charge,
            cc_height=cc_height,
        )
        return params


class CrossCorrelationNeighbour(CrossCorrelation):
    def __init__(self, n_pixels, n_samples, plot=False,
                 mapping=None, **kwargs):
        super().__init__(n_pixels, n_samples, plot, **kwargs)

        if mapping is None:
            raise ValueError("A mapping must be passed "
                             "to CtapipeNeighbourPeakIntegrator")

        from ctapipe.image.charge_extractors import NeighbourPeakIntegrator

        camera = get_ctapipe_camera_geometry(mapping)

        self.integrator = NeighbourPeakIntegrator(
            window_shift=0,
            window_width=1,
        )
        self.integrator.neighbours = camera.neighbor_matrix_where

    def _get_charge(self, waveforms):
        extract = self.integrator.extract_charge
        charge, peakpos, window = extract(self.cc[None, ...])
        cc_height = self.get_pulse_height(charge[0])

        params = dict(
            charge=charge[0],
            cc_height=cc_height,
            ctapipe_peakpos=peakpos[0],
        )
        return params


class CrossCorrelationLocal(CrossCorrelation):
    def __init__(self, n_pixels, n_samples, plot=False, **kwargs):
        super().__init__(n_pixels, n_samples, plot, **kwargs)

        from ctapipe.image.charge_extractors import LocalPeakIntegrator

        self.integrator = LocalPeakIntegrator(
            window_shift=0,
            window_width=1,
        )

    def _get_charge(self, waveforms):
        extract = self.integrator.extract_charge
        charge, peakpos, window = extract(self.cc[None, ...])
        cc_height = self.get_pulse_height(charge[0])

        params = dict(
            charge=charge[0],
            cc_height=cc_height,
            ctapipe_peakpos=peakpos[0],
        )
        return params
