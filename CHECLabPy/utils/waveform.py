"""
This module contains the utilities that are specific to operations on
waveforms
"""
import numpy as np
from tqdm import tqdm


class BaselineSubtractor:
    """
    Keep track of the baseline for the previous `n_base` events, therefore
    keeping a "rolling average" of the baseline to subtract.
    """
    def __init__(self, source, n_base=50, n_base_samples=16):
        n_pixels = source.n_pixels
        self.n_base = n_base
        self.n_base_samples = n_base_samples
        self.iev = 0

        print("Creating initial baseline from first {} events".format(n_base))
        self.baseline_waveforms = np.zeros((n_base, n_pixels, n_base_samples))
        for waveforms in source:
            ev = source.index
            if ev >= n_base:
                break
            self.baseline_waveforms[ev] = waveforms[:, :n_base_samples]
        self.baseline = np.mean(self.baseline_waveforms, axis=(0, 2))
        print("Baseline Created")

    def update_baseline(self, waveforms):
        entry = self.iev % self.n_base
        self.baseline_waveforms[entry] = waveforms[:, :self.n_base_samples]
        self.baseline = np.mean(self.baseline_waveforms, axis=(0, 2))
        self.iev += 1

    def subtract(self, waveforms):
        self.update_baseline(waveforms)
        return waveforms - self.baseline[:, None]


def shift_array(arr, num, fill_value=0):
    result = np.empty_like(arr)
    if num > 0:
        result[:, :num] = fill_value
        result[:, num:] = arr[:, :-num]
    elif num < 0:
        result[:, num:] = fill_value
        result[:, :num] = arr[:, -num:]
    else:
        result[:] = arr[:]
    return result


def shift_waveform(waveforms, t_shift, fill_value=0):
    """
    Shift the waveforms so that the maximum of average wf lays at `t_shift`

    Parameters
    ----------
    waveforms : ndarray
    t_shift : int
        Time bin to shift to
    fill_value : int
        Value to fill the empty samples

    Returns
    -------
    shifted : ndarray
        `waveforms` shifted
    """
    avg_wf_tevent = np.argmax(np.mean(waveforms, 0))
    shift = t_shift - avg_wf_tevent
    return shift_array(waveforms, shift, fill_value)


def get_average_wf(source, t_shift):
    """
    Loop over the file to get average waveform across all events and pixels
    (while shifting each event so the max of the event-averages all match)

    Parameters
    ----------
    source : `core.file_handling.Reader`
    t_shift : int
        Time bin to shift to

    Returns
    -------
    average_wf : ndarray
        One dimenstional array of shape (n_samples) containing the average
        waveform across all events and pixels
    """
    n_events = source.n_events
    n_pixels = source.n_pixels
    n_samples = source.n_samples

    baseline_subtractor = BaselineSubtractor(source)

    desc = "Processing events"
    all_wf = np.zeros((n_events, n_pixels, n_samples))
    for waveforms in tqdm(source, total=n_events, desc=desc):
        iev = source.index

        waveforms_bs = baseline_subtractor.subtract(waveforms)

        all_wf[iev] = shift_waveform(waveforms_bs, t_shift)

    average_wf = np.mean(all_wf, axis=(0, 1))
    return average_wf
