from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange, float64, int64


@njit((float64[:, :], int64, int64), parallel=True)
def obtain_pulse_timing(waveforms, window_start, window_end):
    n_pixels, n_samples = waveforms.shape

    t_pulse_arr = np.zeros(n_pixels)
    h_pulse_arr = np.zeros(n_pixels)
    fwhm_arr = np.zeros(n_pixels)
    rise_time_arr = np.zeros(n_pixels)
    t_l = np.zeros(n_pixels)
    t_r = np.zeros(n_pixels)
    t10 = np.zeros(n_pixels)
    t90 = np.zeros(n_pixels)

    for pixel in prange(n_pixels):
        wf = waveforms[pixel]
        baseline = np.mean(wf[:10])

        #######################
        # __Obtain Pulse Time__
        #######################
        peakpos = np.argmax(wf[window_start:window_end]) + window_start

        y0 = wf[peakpos - 1]
        y1 = wf[peakpos]
        y2 = wf[peakpos + 1]

        # Quadratic peak interpolation
        a = y0 - y2
        b = y0 - 2 * y1 + y2
        t = 0.5 * a / b
        h = y1 - 0.25 * (y0 - y2) * t
        h_pulse = h - baseline
        t_pulse = t + peakpos
        i_pulse = int(np.round(t_pulse))

        ###############################
        # __Obtain FWHM and Rise Time__
        ###############################
        half_max = baseline + h_pulse / 2
        _10percent = baseline + 0.1 * h_pulse
        _90percent = baseline + 0.9 * h_pulse

        fwhm_r = 0
        fwhm_l = 0
        rt_90 = 0
        rt_10 = 0

        # Forward loop over samples
        for xi in range(i_pulse, n_samples - 1):
            xj = xi + 1
            yi = wf[xi]
            yj = wf[xj]
            if yi >= half_max >= yj:
                fwhm_r = xi + (half_max - yi) * (xj - xi) / (yj - yi)
                break

        # Backward loop over samples
        for xi in range(i_pulse, 0, -1):
            xj = xi - 1
            yi = wf[xi]
            yj = wf[xj]
            if yi >= half_max >= yj:
                if not fwhm_l:  # Only set on first time
                    fwhm_l = xi + (half_max - yi) * (xj - xi) / (yj - yi)
            if yi >= _90percent >= yj:
                if not rt_90:  # Only set on first time
                    rt_90 = xi + (_90percent - yi) * (xj - xi) / (yj - yi)
            if yi >= _10percent >= yj:
                if not rt_10:  # Only set on first time
                    rt_10 = xi + (_10percent - yi) * (xj - xi) / (yj - yi)
            if fwhm_l and rt_90 and rt_10:
                break

        if fwhm_r and fwhm_l:
            fwhm = fwhm_r - fwhm_l
        else:
            fwhm = np.nan

        if rt_90 and rt_10:
            rise_time = rt_90 - rt_10
        else:
            rise_time = np.nan

        t_pulse_arr[pixel] = t_pulse
        h_pulse_arr[pixel] = h_pulse
        fwhm_arr[pixel] = fwhm
        rise_time_arr[pixel] = rise_time

        t_l[pixel] = fwhm_l
        t_r[pixel] = fwhm_r
        t10[pixel] = rt_10
        t90[pixel] = rt_90

    return (t_pulse_arr, h_pulse_arr, fwhm_arr,
            rise_time_arr, t_l, t_r, t10, t90)


class Timing(WaveformReducer):
    """
    Obtains timing characteristics of the waveform.
    """

    def __init__(self, n_pixels, n_samples, plot_timing=False, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)
        self.window_size = self.kwargs.get("window_size", 8)
        self.window_shift = self.kwargs.get("window_shift", 4)

        self.plot = plot_timing

        self._t_pulse = None
        self._t_pulse_amp = None
        self._t_event = None
        self._window_start = None
        self._window_end = None

        self._t_l = None
        self._t_r = None
        self._t10 = None
        self._t90 = None

    @staticmethod
    def interpolate(x, x1, x2, y1, y2):
        y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        isinf = np.isinf(y)
        y[isinf] = y1[isinf]
        return y

    @staticmethod
    def interpolate_peak(y0, y1, y2):
        with np.errstate(divide='ignore', invalid='ignore'):
            a = y0 - y2
            b = y0 - 2 * y1 + y2
            xm = 0.5 * a / b
            ym = y1 - 0.25 * (y0 - y2) * xm
        return xm, ym

    def _prepare(self, waveforms):
        super()._prepare(waveforms)

        if "t_extract" in self.kwargs:
            t_extract = self.kwargs['t_extract']
        else:
            avg = np.mean(waveforms, axis=0)
            t_extract = np.argmax(avg)
            n_samples = waveforms.shape[1]
            if t_extract < 10:
                t_extract = 10
            elif t_extract > n_samples - 10:
                t_extract = n_samples - 10
        window_start = t_extract - self.window_shift
        window_end = window_start + self.window_size

        params = obtain_pulse_timing(waveforms, window_start, window_end)
        t_pulse, h_pulse, fwhm, rise_time, t_l, t_r, t10, t90 = params

        self._t_pulse = t_pulse
        self._t_pulse_amp = h_pulse
        self._t_event = t_extract
        self._fwhm = fwhm
        self._rise_time = rise_time
        self._window_start = window_start
        self._window_end = window_end

        self._t_l = t_l
        self._t_r = t_r
        self._t10 = t10
        self._t90 = t90

    @column
    def t_pulse(self):
        """
        Time of the pulse in each pixel, defined as the interpolated maximum
        of the pulse.
        """
        return self._t_pulse

    @column
    def t_pulse_amp(self):
        """
        Interpolated amplitude of the pulse at the time t_pulse.
        """
        return self._t_pulse_amp

    @column
    def t_event(self):
        """
        Sample corresponding to the maximum amplitude in the waveform.
        """
        return self._t_event

    @column
    def fwhm(self):
        """
        Full Width at Half Maximum of the pulse.
        """
        return self._fwhm

    @column
    def t_rise(self):
        """
        Rise time of the pulse (from 10% of the maximum to 90%)
        """
        return self._rise_time

    def _post(self):
        if self.plot:
            ipulse = np.argmax(self.waveforms[:, self.t_event])
            ilow = np.argmin(self.waveforms[:, self.t_event])

            fig = plt.figure(figsize=(13, 5))
            ax_p = fig.add_subplot(1, 2, 1)
            ax_p.set_title("Large Pulse")
            ax_p.plot(self.waveforms[ipulse])
            ax_p.axvline(self._t_event, c='black', ls=':', label="t_event")
            ax_p.axvline(self._window_start, c='blue', ls=':', label="window")
            ax_p.axvline(self._window_end, c='blue', ls=':')
            ax_p.axvline(self._t_pulse[ipulse], c='black', ls='-',
                         label="t_pulse")
            ax_p.plot(self._t_pulse[ipulse], self._t_pulse_amp[ipulse], ".")
            ax_p.axvline(self._t_l[ipulse], c='red', label="FWHM")
            ax_p.axvline(self._t_r[ipulse], c='red')
            ax_p.axvline(self._t10[ipulse], c='blue', label="tr")
            ax_p.axvline(self._t90[ipulse], c='blue')
            ax_l = fig.add_subplot(1, 2, 2)
            ax_l.set_title("Low/No Pulse")
            ax_l.plot(self.waveforms[ilow])
            ax_l.axvline(self._t_event, c='black', ls=':')
            ax_l.axvline(self._window_start, c='blue', ls=':')
            ax_l.axvline(self._window_end, c='blue', ls=':')
            ax_l.axvline(self._t_pulse[ilow], c='black', ls='-')
            ax_l.plot(self._t_pulse[ilow], self._t_pulse_amp[ilow], ".")
            ax_l.axvline(self._t_l[ilow], c='red')
            ax_l.axvline(self._t_r[ilow], c='red')
            ax_l.axvline(self._t10[ilow], c='blue')
            ax_l.axvline(self._t90[ilow], c='blue')
            ax_p.legend(loc="upper right")
            plt.pause(1)
            plt.close("all")
