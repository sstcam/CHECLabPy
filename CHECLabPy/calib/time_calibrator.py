import pandas as pd
from CHECLabPy.calib import get_calib_data
from numba import njit, prange, float64, float32
from math import floor, ceil


@njit([
    float64[:, :](float64[:, :], float64[:]),
    float32[:, :](float32[:, :], float64[:]),
])
def apply_t_cor(waveforms, t_cor):
    n_pixels, n_samples = waveforms.shape
    shifted = waveforms.copy()
    for ipix in prange(n_pixels):
        correction = t_cor[ipix]
        for isample in prange(n_samples):
            x = isample - correction
            x1 = floor(x)
            x2 = ceil(x)
            if x1 < 0:
                x1 = 0
                x2 = 1
            elif x2 >= n_samples:
                x1 = n_samples - 2
                x2 = n_samples - 1
            y1 = waveforms[ipix, x1]
            y2 = waveforms[ipix, x2]
            shifted[ipix, isample] = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return shifted


class TimeCalibrator:
    def __init__(self):
        path = get_calib_data("time_correction.dat")
        print(f"[TimeCalibrator] Loading timing corrections: {path}")
        df = pd.read_csv(path, sep='\t')
        self.t_cor = df['t_cor'].values

    def __call__(self, waveforms):
        return apply_t_cor(waveforms, self.t_cor)
