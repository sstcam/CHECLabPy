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

        self.baseline_waveforms = np.zeros((n_base, n_pixels, n_base_samples))
        for waveforms in source:
            ev = source.index
            if ev >= n_base:
                break
            self.baseline_waveforms[ev] = waveforms[:, :n_base_samples]
        self.baseline = np.mean(self.baseline_waveforms, axis=(0, 2))

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
    n_samples = source.n_samples

    baseline_subtractor = BaselineSubtractor(source)

    desc = "Processing events"
    average_wf = np.zeros(n_samples)
    n = 0
    for waveforms in tqdm(source, total=n_events, desc=desc):
        waveforms_bs = baseline_subtractor.subtract(waveforms)

        wf = shift_waveform(waveforms_bs, t_shift)

        average_wf += wf.mean(0)
        n += 1

    average_wf /= n
    return average_wf


def get_average_wf_per_pixel(source, t_shift):
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
    average_wf = np.zeros((n_pixels, n_samples))
    n = 0
    for waveforms in tqdm(source, total=n_events, desc=desc):
        waveforms_bs = baseline_subtractor.subtract(waveforms)

        wf = shift_waveform(waveforms_bs, t_shift)

        average_wf += wf
        n += 1

    average_wf /= n
    return average_wf


def obtain_dead_pixel_list_r1(source, threshold_percent=0.1, max_events=100):
    """
    From an R1 file, obtain a list of pixels that are considered dead as the
    peak height of their average waveform is less that the threshold.

    Parameters
    ----------
    source : `CHECLabPy.core.io.TIOReader`
    threshold_percent : float
        Percentage of the average height that should be used as a threshold
        to define a dead pixel
    max_events : int
        Max number of events to loop over

    Returns
    -------
    list
    """
    n_events = source.n_events if source.n_events < max_events else max_events
    n_pixels = source.n_pixels
    n_samples = source.n_samples
    array = np.zeros((n_events, n_pixels, n_samples))
    for waveforms in source:
        iev = source.index
        if iev >= n_events:
            break
        array[iev] = waveforms
    avg_wf = np.mean(array, axis=0)
    peak_height = np.max(avg_wf, axis=1)
    dead = np.where(peak_height < peak_height.mean() * threshold_percent)[0]
    print("Dead Pixels: {}".format(dead))
    return dead
