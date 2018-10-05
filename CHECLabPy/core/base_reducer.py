import numpy as np
from matplotlib import pyplot as plt


class WaveformReducer:
    """
    Base class for the Waveform Reducers.
    This class implements the default methods for reducing the waveform.
    """
    def __init__(self, n_pixels, n_samples, plot=False, **kwargs):
        self.kwargs = kwargs

        self.extract_charge_only = kwargs.get("extract_charge_only", False)
        self.window_size = self.kwargs.get("window_size", 8)
        self.window_shift = self.kwargs.get("window_shift", 4)

        self.t_event = 0
        self.window_start = 0
        self.window_end = 0

        self.n_pixels = n_pixels
        self.n_samples = n_samples
        self.pixel_arange = np.arange(n_pixels)
        self.ind = np.indices((n_pixels, n_samples))[1]
        self.r_ind = self.ind[:, ::-1]

        self.plot = plot

    def process(self, waveforms):
        """
        Method to process the waveforms into their reduced parameters per pixel

        Parameters
        ----------
        waveforms : ndarray
            Numpy array of shape (npixels, nsamples)

        Returns
        -------
        params : dict
            Dictionary containing the reduced parameters from the waveforms
        """

        self._set_t_event(waveforms)
        params = dict(t_event=self.t_event)

        params.update(self._get_charge(waveforms))
        if not self.extract_charge_only:
            params.update(self._get_baseline(waveforms))
            params.update(self._get_timing(waveforms))
            params.update(self._get_saturation(waveforms))

        return params

    def _set_t_event(self, waveforms):
        if "t_event" in self.kwargs:
            t_event = self.kwargs['t_event']
        else:
            avg = np.mean(waveforms, axis=0)
            t_event = np.argmax(avg)
            n_samples = waveforms.shape[1]
            if t_event < 10:
                t_event = 10
            elif t_event > n_samples - 10:
                t_event = n_samples - 10
        self.t_event = t_event
        self.window_start = self.t_event - self.window_shift
        self.window_end = self.window_start + self.window_size

    def _get_charge(self, waveforms):
        start = self.window_start
        end = self.window_end
        charge = np.sum(waveforms[:, start:end], axis=1)

        params = dict(
            charge=charge,
        )
        return params

    def _get_baseline(self, waveforms):
        baseline_start_mean = np.mean(waveforms[:, 0:20], axis=1)
        baseline_start_rms = np.std(waveforms[:, 0:20], axis=1)

        baseline_end_mean = np.mean(waveforms[:, -20:], axis=1)
        baseline_end_rms = np.std(waveforms[:, -20:], axis=1)

        waveform_mean = np.mean(waveforms, axis=1)
        waveform_rms = np.std(waveforms, axis=1)

        params = dict(
            baseline_start_mean=baseline_start_mean,
            baseline_start_rms=baseline_start_rms,
            baseline_end_mean=baseline_end_mean,
            baseline_end_rms=baseline_end_rms,
            waveform_mean=waveform_mean,
            waveform_rms=waveform_rms
        )
        return params

    def _get_saturation(self, waveforms):
        saturation_coeff = np.sum(waveforms[:, self.window_start:], axis=1)

        params = dict(
            saturation_coeff=saturation_coeff
        )
        return params

    @staticmethod
    def interpolate(x, x1, x2, y1, y2):
        y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        isinf = np.isinf(y)
        y[isinf] = y1[isinf]
        return y

    @staticmethod
    def interpolate_peak(y0, y1, y2):
        a = y0 - y2
        b = y0 - 2 * y1 + y2
        xm = 0.5 * a / b
        ym = y1 - 0.25 * (y0 - y2) * xm
        return xm, ym

    def interpolate_wf_peak(self, waveforms):
        if "t_event" in self.kwargs:
            t_event = self.kwargs['t_event']
        else:
            avg = np.mean(waveforms, axis=0)
            t_event = np.argmax(avg)
            n_samples = waveforms.shape[1]
            if t_event < 10:
                t_event = 10
            elif t_event > n_samples - 10:
                t_event = n_samples - 10
        window_start = t_event - self.window_shift
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
        outofrange = (peak_time >= end) | (peak_time < start)
        mask = isinf | outofrange
        peak_time[mask] = t_event
        peak_amp[mask] = waveforms[mask, t_event]
        return peak_time, peak_amp, t_event, window_start, window_end

    def _get_timing(self, waveforms):
        n_pixels, n_samples = waveforms.shape

        ind = self.ind
        r_ind = self.r_ind

        bad_wf_mask = np.zeros(n_pixels, dtype=np.bool)

        with np.errstate(divide='ignore', invalid='ignore'):
            t_max, amp_max, t_event, window_start, window_end = \
                self.interpolate_wf_peak(waveforms)

        t_pulse = t_max
        amp_pulse = amp_max

        reversed_ = waveforms[:, ::-1]
        peak_time_i = np.ones(waveforms.shape) * t_pulse[:, None]
        mask_before = np.ma.masked_less(ind, peak_time_i).mask
        mask_after = np.ma.masked_greater(r_ind, peak_time_i).mask
        masked_bef = np.ma.masked_array(waveforms, mask_before)
        masked_aft = np.ma.masked_array(reversed_, mask_after)
        half_max = amp_pulse / 2
        d_l = np.diff(np.sign(half_max[:, None] - masked_aft))
        d_r = np.diff(np.sign(half_max[:, None] - masked_bef))
        d_l_am = np.argmax(d_l, axis=1)
        d_r_am = np.argmax(d_r, axis=1)
        t_l2 = r_ind[0, d_l_am + 1]
        t_l1 = r_ind[0, d_l_am]
        t_r1 = ind[0, d_r_am]
        t_r2 = ind[0, d_r_am + 1]
        w_l2 = waveforms[self.pixel_arange, t_l2]
        w_l1 = waveforms[self.pixel_arange, t_l1]
        w_r1 = waveforms[self.pixel_arange, t_r1]
        w_r2 = waveforms[self.pixel_arange, t_r2]
        with np.errstate(divide='ignore', invalid='ignore'):
            t_l = self.interpolate(half_max, w_l2, w_l1, t_l2, t_l1)
            t_r = self.interpolate(half_max, w_r2, w_r1, t_r2, t_r1)
            where = ((t_l > t_r) | (t_l < 0) | (t_r < 0) |
                     (t_l > n_samples) | (t_r > n_samples))
            bad_wf_mask = bad_wf_mask | where

        _10percent = 0.1 * amp_pulse
        _90percent = 0.9 * amp_pulse
        d10 = np.diff(np.sign(_10percent[:, None] - masked_aft))
        d90 = np.diff(np.sign(_90percent[:, None] - masked_aft))
        t10_2 = r_ind[0, np.argmax(d10, axis=1) + 1]
        t10_1 = r_ind[0, np.argmax(d10, axis=1)]
        t90_2 = r_ind[0, np.argmax(d90, axis=1) + 1]
        t90_1 = r_ind[0, np.argmax(d90, axis=1)]
        w10_2 = waveforms[self.pixel_arange, t10_2]
        w10_1 = waveforms[self.pixel_arange, t10_1]
        w90_2 = waveforms[self.pixel_arange, t90_2]
        w90_1 = waveforms[self.pixel_arange, t90_1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t10 = self.interpolate(_10percent, w10_2, w10_1, t10_2, t10_1)
            t90 = self.interpolate(_90percent, w90_2, w90_1, t90_2, t90_1)
            where = ((t10 > t90) | (t10 < 0) | (t90 < 0) |
                     (t10 > n_samples) | (t90 > n_samples))
            bad_wf_mask = bad_wf_mask | where

        t_l[bad_wf_mask] = np.nan
        t_r[bad_wf_mask] = np.nan
        t10[bad_wf_mask] = np.nan
        t90[bad_wf_mask] = np.nan
        fwhm = t_r - t_l
        rise_time = t90 - t10

        if self.plot:
            ipulse = np.argmax(waveforms[:, t_event])
            ilow = np.argmin(waveforms[:, t_event])

            fig = plt.figure(figsize=(13, 5))
            ax_p = fig.add_subplot(1, 2, 1)
            ax_p.set_title("Large Pulse")
            ax_p.plot(waveforms[ipulse])
            ax_p.axvline(t_event, c='black', ls=':', label="t_event")
            ax_p.axvline(window_start, c='blue', ls=':', label="window")
            ax_p.axvline(window_end, c='blue', ls=':')
            ax_p.axvline(t_pulse[ipulse], c='black', ls='-', label="t_pulse")
            ax_p.plot(t_pulse[ipulse], amp_pulse[ipulse], ".")
            ax_p.axvline(t_l[ipulse], c='red', label="FWHM")
            ax_p.axvline(t_r[ipulse], c='red')
            ax_p.axvline(t10[ipulse], c='blue', label="tr")
            ax_p.axvline(t90[ipulse], c='blue')
            ax_l = fig.add_subplot(1, 2, 2)
            ax_l.set_title("Low/No Pulse")
            ax_l.plot(waveforms[ilow])
            ax_l.axvline(t_event, c='black', ls=':')
            ax_l.axvline(window_start, c='blue', ls=':')
            ax_l.axvline(window_end, c='blue', ls=':')
            ax_l.axvline(t_pulse[ilow], c='black', ls='-')
            ax_l.plot(t_pulse[ilow], amp_pulse[ilow], ".")
            ax_l.axvline(t_l[ilow], c='red')
            ax_l.axvline(t_r[ilow], c='red')
            ax_l.axvline(t10[ilow], c='blue')
            ax_l.axvline(t90[ilow], c='blue')
            ax_p.legend(loc="upper right")
            plt.pause(1)
            plt.close("all")

        params = dict(
            t_pulse=t_pulse,
            amp_pulse=amp_pulse,
            fwhm=fwhm,
            tr=rise_time
        )
        return params
