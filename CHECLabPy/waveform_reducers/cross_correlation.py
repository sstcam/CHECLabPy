from CHECLabPy.core.base_reducer import WaveformReducer
from CHECLabPy.data import get_file
import numpy as np
from scipy import interpolate
from scipy.ndimage import correlate1d


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = self.kwargs.get("reference_pulse_path",
                               get_file("checs_reference_pulse_lei.txt"))
        file = np.loadtxt(path, delimiter=', ')
        refx = file[:, 0]
        refy = file[:, 1] - file[:, 1][0]
        f = interpolate.interp1d(refx, refy, kind=3)
        x = np.linspace(0, 77e-9, 76)
        y = f(x)

        # Put pulse in center so result peak time matches with input peak
        pad = y.size - 2 * np.argmax(y)
        if pad > 0:
            y = np.pad(y, (pad, 0), mode='constant')
        else:
            y = np.pad(y, (0, -pad), mode='constant')

        # Create 1p.e. pulse shape
        self.y_1pe = y / np.trapz(y)

        # Make maximum of cc result == 1
        y = y / correlate1d(self.y_1pe, y).max()

        self.reference_pulse = y
        self.cc = None

    def get_pulse_height(self, charge):
        return charge * self.y_1pe.max()

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
