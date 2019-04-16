from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np


class Timing(WaveformReducer):
    """
    Obtains timing characteristics of the waveform.
    """
    def __init__(self, n_pixels, n_samples, plot_timing=False, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)
        self.window_size = self.kwargs.get("window_size", 8)
        self.window_shift = self.kwargs.get("window_shift", 4)

        self.pixel_arange = np.arange(n_pixels)
        self.ind = np.indices((n_pixels, n_samples))[1]
        self.r_ind = self.ind[:, ::-1]

        self.plot = plot_timing

        self.t_pulse = None
        self.t_pulse_amp = None
        self.t_event = None
        self.window_start = None
        self.window_end = None

        self.t_l = np.zeros(n_pixels)
        self.t_r = np.zeros(n_pixels)
        self.t10 = np.zeros(n_pixels)
        self.t90 = np.zeros(n_pixels)

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

        start = window_start - 3
        end = window_end + 3
        t1 = np.argmax(waveforms[:, start:end], 1) + start
        y0 = waveforms[self.pixel_arange, t1 - 1]
        y1 = waveforms[self.pixel_arange, t1]
        y2 = waveforms[self.pixel_arange, t1 + 1]
        xm, ym = self.interpolate_peak(y0, y1, y2)
        peak_time = xm + t1
        peak_amp = ym
        isinf = np.isinf(peak_time)
        with np.errstate(divide='ignore', invalid='ignore'):
            outofrange = (peak_time >= end) | (peak_time < start)
        mask = isinf | outofrange
        peak_time[mask] = t_extract
        peak_amp[mask] = waveforms[mask, t_extract]

        self.t_pulse = peak_time
        self.t_pulse_amp = peak_amp
        self.t_event = t_extract
        self.window_start = window_start
        self.window_end = window_end

    @column
    def t_pulse(self):
        """
        Time of the pulse in each pixel, defined as the interpolated maximum
        of the pulse.
        """
        return self.t_pulse

    @column
    def t_pulse_amp(self):
        """
        Interpolated amplitude of the pulse at the time t_pulse.
        """
        return self.t_pulse_amp

    @column
    def t_event(self):
        """
        Sample corresponding to the maximum amplitude in the waveform.
        """
        return self.t_event

    @column
    def fwhm(self):
        """
        Full Width at Half Maximum of the pulse.
        """
        waveforms = self.waveforms

        bad_wf_mask = np.zeros(self.n_pixels, dtype=np.bool)

        reversed_ = waveforms[:, ::-1]
        peak_time_i = np.ones(waveforms.shape) * self.t_pulse[:, None]
        mask_before = np.ma.masked_less(self.ind, peak_time_i).mask
        mask_after = np.ma.masked_greater(self.r_ind, peak_time_i).mask
        masked_bef = np.ma.masked_array(waveforms, mask_before)
        masked_aft = np.ma.masked_array(reversed_, mask_after)

        half_max = self.t_pulse_amp / 2
        d_l = np.diff(np.sign(half_max[:, None] - masked_aft))
        d_r = np.diff(np.sign(half_max[:, None] - masked_bef))
        d_l_am = np.argmax(d_l, axis=1)
        d_r_am = np.argmax(d_r, axis=1)
        t_l2 = self.r_ind[0, d_l_am + 1]
        t_l1 = self.r_ind[0, d_l_am]
        t_r1 = self.ind[0, d_r_am]
        t_r2 = self.ind[0, d_r_am + 1]
        w_l2 = waveforms[self.pixel_arange, t_l2]
        w_l1 = waveforms[self.pixel_arange, t_l1]
        w_r1 = waveforms[self.pixel_arange, t_r1]
        w_r2 = waveforms[self.pixel_arange, t_r2]
        with np.errstate(divide='ignore', invalid='ignore'):
            t_l = self.interpolate(half_max, w_l2, w_l1, t_l2, t_l1)
            t_r = self.interpolate(half_max, w_r2, w_r1, t_r2, t_r1)
            where = ((t_l > t_r) | (t_l < 0) | (t_r < 0) |
                     (t_l > self.n_samples) | (t_r > self.n_samples))
            bad_wf_mask = bad_wf_mask | where

        t_l[bad_wf_mask] = np.nan
        t_r[bad_wf_mask] = np.nan
        self.t_l = t_l
        self.t_r = t_r

        fwhm = t_r - t_l

        return fwhm

    @column
    def t_rise(self):
        """
        Rise time of the pulse (from 10% of the maximum to 90%)
        """
        waveforms = self.waveforms

        bad_wf_mask = np.zeros(self.n_pixels, dtype=np.bool)

        reversed_ = waveforms[:, ::-1]
        peak_time_i = np.ones(waveforms.shape) * self.t_pulse[:, None]
        mask_after = np.ma.masked_greater(self.r_ind, peak_time_i).mask
        masked_aft = np.ma.masked_array(reversed_, mask_after)

        _10percent = 0.1 * self.t_pulse_amp
        _90percent = 0.9 * self.t_pulse_amp
        d10 = np.diff(np.sign(_10percent[:, None] - masked_aft))
        d90 = np.diff(np.sign(_90percent[:, None] - masked_aft))
        t10_2 = self.r_ind[0, np.argmax(d10, axis=1) + 1]
        t10_1 = self.r_ind[0, np.argmax(d10, axis=1)]
        t90_2 = self.r_ind[0, np.argmax(d90, axis=1) + 1]
        t90_1 = self.r_ind[0, np.argmax(d90, axis=1)]
        w10_2 = waveforms[self.pixel_arange, t10_2]
        w10_1 = waveforms[self.pixel_arange, t10_1]
        w90_2 = waveforms[self.pixel_arange, t90_2]
        w90_1 = waveforms[self.pixel_arange, t90_1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t10 = self.interpolate(_10percent, w10_2, w10_1, t10_2, t10_1)
            t90 = self.interpolate(_90percent, w90_2, w90_1, t90_2, t90_1)
            where = ((t10 > t90) | (t10 < 0) | (t90 < 0) |
                     (t10 > self.n_samples) | (t90 > self.n_samples))
            bad_wf_mask = bad_wf_mask | where

        t10[bad_wf_mask] = np.nan
        t90[bad_wf_mask] = np.nan
        self.t10 = t10
        self.t90 = t90

        rise_time = t90 - t10

        return rise_time

    def _post(self):
        if self.plot:
            from matplotlib import pyplot as plt
            ipulse = np.argmax(self.waveforms[:, self.t_event])
            ilow = np.argmin(self.waveforms[:, self.t_event])

            fig = plt.figure(figsize=(13, 5))
            ax_p = fig.add_subplot(1, 2, 1)
            ax_p.set_title("Large Pulse")
            ax_p.plot(self.waveforms[ipulse])
            ax_p.axvline(self.t_event, c='black', ls=':', label="t_event")
            ax_p.axvline(self.window_start, c='blue', ls=':', label="window")
            ax_p.axvline(self.window_end, c='blue', ls=':')
            ax_p.axvline(self.t_pulse[ipulse], c='black', ls='-',
                         label="t_pulse")
            ax_p.plot(self.t_pulse[ipulse], self.t_pulse_amp[ipulse], ".")
            ax_p.axvline(self.t_l[ipulse], c='red', label="FWHM")
            ax_p.axvline(self.t_r[ipulse], c='red')
            ax_p.axvline(self.t10[ipulse], c='blue', label="tr")
            ax_p.axvline(self.t90[ipulse], c='blue')
            ax_l = fig.add_subplot(1, 2, 2)
            ax_l.set_title("Low/No Pulse")
            ax_l.plot(self.waveforms[ilow])
            ax_l.axvline(self.t_event, c='black', ls=':')
            ax_l.axvline(self.window_start, c='blue', ls=':')
            ax_l.axvline(self.window_end, c='blue', ls=':')
            ax_l.axvline(self.t_pulse[ilow], c='black', ls='-')
            ax_l.plot(self.t_pulse[ilow], self.t_pulse_amp[ilow], ".")
            ax_l.axvline(self.t_l[ilow], c='red')
            ax_l.axvline(self.t_r[ilow], c='red')
            ax_l.axvline(self.t10[ilow], c='blue')
            ax_l.axvline(self.t90[ilow], c='blue')
            ax_p.legend(loc="upper right")
            plt.pause(1)
            plt.close("all")
