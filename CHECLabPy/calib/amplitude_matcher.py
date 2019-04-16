import numpy as np
from enum import IntEnum
from numba import njit, prange
from CHECLabPy.waveform_reducers.average_wf import AverageWF
from CHECLabPy.core.io import ReaderR0


class STAGE(IntEnum):
    SWEEP = 0
    COMB = 1
    FINISH = 2


class AmplitudeMatcher:
    def __init__(self, calibrator, amplitude_aim, dead_list):
        self.calibrator = calibrator
        n_pixels = calibrator.n_pixels
        n_samples = calibrator.n_samples

        self.waveform_reducer = AverageWF(n_pixels, n_samples)

        self.amplitude_aim = amplitude_aim
        self.dead_mask = np.zeros(n_pixels, dtype=np.bool)
        self.dead_mask[dead_list] = True

        n_superpixels = n_pixels // 4
        self.stage = np.zeros(n_superpixels, dtype=np.int16)
        self.dead_sp_mask = self.dead_mask.reshape((n_superpixels, 4)).all(1)
        self.current_dac = np.zeros(n_superpixels, dtype=np.int16)
        self.previous_dac = np.zeros(n_superpixels, dtype=np.int16)
        self.previous_rmse = np.zeros(n_superpixels)

        self.sweep_steps = np.linspace(0, 255, 10, dtype=np.int16)
        self.sweep_min = np.zeros(n_superpixels)
        self.sweep_min_index = np.zeros(n_superpixels, dtype=np.int)
        self.sweep_current_index = 0
        self.comb_iteration = np.zeros(n_superpixels, dtype=np.int)
        self.comb_end = np.zeros(n_superpixels, dtype=np.int16)
        self.final = False

    def _extract_amplitude(self, r0_waveforms, fci):
        r1_waveforms = self.calibrator(r0_waveforms, fci)

        amplitude = np.max(r1_waveforms, axis=1)
        # TODO switch to integration - but need conversion to mV
        # self.waveform_reducer._prepare(r1_waveforms)
        # amplitude = self.waveform_reducer.amplitude_averagewf

        # TODO Apply illumination profile correction

        return amplitude

    def _extract_amplitude_array(self, r0_path):
        reader = ReaderR0(r0_path, max_events=100)
        n_events = reader.n_events
        n_pixels = reader.n_pixels
        amplitude = np.zeros((n_events, n_pixels))
        for wfs in reader:
            iev = reader.index
            fci = reader.first_cell_ids
            amplitude[iev] = self._extract_amplitude(wfs, fci)
        return amplitude

    @staticmethod
    @njit
    def _calculate_rmse_nb(amplitude, amplitude_aim, dead):
        n_events, n_pixels = amplitude.shape
        n_superpixels = n_pixels // 4
        sum_ = np.zeros(n_superpixels)
        n = np.zeros(n_superpixels)
        for iev in range(n_events):
            for ipix in range(n_pixels):
                if not dead[ipix]:
                    sum_[ipix//4] += (amplitude[iev, ipix] - amplitude_aim)**2
                    n[ipix // 4] += 1
        return np.sqrt(sum_ / n)

    def _calculate_rmse(self, amplitude):
        return self._calculate_rmse_nb(
            amplitude, self.amplitude_aim, self.dead_mask
        )

    @staticmethod
    @njit(parallel=True)
    def _get_next_dac_nb(
            current_dac, current_rmse,
            previous_dac, previous_rmse,
            sweep_steps, current_index,
            min_index, min_rmse,
            dead_sp, stage, comb_iteration, comb_end
    ):
        next_dac = current_dac.copy()
        n_superpixels = next_dac.size
        for isp in prange(n_superpixels):
            if stage[isp] == STAGE.FINISH:
                continue

            if dead_sp[isp]:
                next_dac[isp] = 255
                stage[isp] = STAGE.FINISH
                continue

            if stage[isp] == STAGE.SWEEP:
                if current_index[isp] == 0:
                    min_rmse[isp] = current_rmse[isp]
                if current_rmse[isp] < min_rmse[isp]:
                    min_rmse[isp] = current_rmse[isp]
                    min_index[isp] = current_index[isp]

                next_index = current_index[isp] + 1
                max_index = sweep_steps.size - 1
                if next_index <= max_index:
                    next_dac[isp] = sweep_steps[next_index]
                    current_index[isp] = next_index
                    continue
                else:
                    istart = min_index - 1
                    istart = istart if istart < 0 else 0
                    iend = min_index + 1
                    iend = iend if iend > max_index else max_index
                    next_dac[isp] = sweep_steps[istart]
                    comb_end[isp] = sweep_steps[iend]
                    stage[isp] += 1
                    continue

            if stage[isp] == STAGE.COMB:
                if comb_iteration[isp] > 0:
                    if previous_rmse[isp] <= current_rmse[isp]:
                        next_dac[isp] = previous_dac[isp]
                        stage[isp] += 1
                        if stage[isp] == STAGE.FINISH:
                            continue

                next_dac[isp] += 1
                if next_dac[isp] > comb_end[isp]:
                    next_dac[isp] = comb_end[isp]
                    stage[isp] = STAGE.FINISH
                    continue

                comb_iteration[isp] += 1
                previous_dac[isp] = current_dac[isp]
                previous_rmse[isp] = current_rmse[isp]

        return next_dac

    def _get_next_dac(self, current_rmse):
        next_dac = self._get_next_dac_nb(
            self.current_dac, current_rmse,
            self.previous_dac, self.previous_rmse,
            self.sweep_steps, self.sweep_current_index,
            self.sweep_min_index, self.sweep_min,
            self.dead_sp_mask, self.stage, self.comb_iteration, self.comb_end
        )
        return next_dac.reshape((32, 16))

    def get_first_dac(self):
        return self.current_dac.reshape((32, 16))

    def process(self, r0_path):
        if self.final:
            return self.current_dac, True

        amplitude = self._extract_amplitude_array(r0_path)
        current_rmse = self._calculate_rmse(amplitude)
        next_dac = self._get_next_dac(current_rmse)
        self.final = (self.stage == STAGE.FINISH).all()
        self.current_dac = next_dac

        return next_dac, False
