import numpy as np
from enum import IntEnum
from numba import njit, prange
from CHECLabPy.waveform_reducers.average_wf import AverageWF
from CHECLabPy.core.io import ReaderR0


class STAGE(IntEnum):
    STEP10 = 0
    STEP1 = 1
    FINISH = 2


class AmplitudeMatcher:
    def __init__(self, calibrator, starting_dacs, amplitude_aim, dead_list):
        self.calibrator = calibrator
        n_pixels = calibrator.n_pixels
        n_samples = calibrator.n_samples

        self.starting_dacs = starting_dacs.ravel()

        self.waveform_reducer = AverageWF(n_pixels, n_samples)

        self.amplitude_aim = amplitude_aim
        self.dead_mask = np.zeros(n_pixels, dtype=np.bool)
        self.dead_mask[dead_list] = True

        n_superpixels = n_pixels // 4
        self.final = False
        self.stage = np.zeros(n_superpixels, dtype=np.int16)
        self.dead_sp_mask = self.dead_mask.reshape((n_superpixels, 4)).all(1)
        self.current_dac = starting_dacs.copy()
        self.previous_dac = np.zeros(n_superpixels, dtype=np.int16)
        self.previous_rmse = np.zeros(n_superpixels)
        self.iteration = 0

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
    def _calculate_average_and_rmse_nb(amplitude, amplitude_aim, dead):
        n_events, n_pixels = amplitude.shape
        n_superpixels = n_pixels // 4
        sum_average = np.zeros(n_superpixels)
        sum_rmse = np.zeros(n_superpixels)
        n = np.zeros(n_superpixels)
        for iev in range(n_events):
            for ipix in range(n_pixels):
                amp = amplitude[iev, ipix]
                if not dead[ipix]:
                    sum_average[ipix // 4] += amp
                    sum_rmse[ipix // 4] += (amp - amplitude_aim) ** 2
                    n[ipix // 4] += 1
        average = sum_average / n
        direction = np.sign(average - amplitude_aim)
        rmse = np.sqrt(sum_rmse / n)
        return average, direction, rmse

    def _calculate_average_and_rmse(self, amplitude):
        return self._calculate_average_and_rmse_nb(
            amplitude, self.amplitude_aim, self.dead_mask
        )

    @staticmethod
    @njit(parallel=True)
    def _get_next_dac_nb(
            dac, rmse, direction,
            previous_dac, previous_rmse,
            dead_sp, stage, iteration
    ):
        next_dac = dac.copy()
        n_superpixel = next_dac.size
        for isp in prange(n_superpixel):
            if stage[isp] == STAGE.FINISH:
                continue

            if dead_sp[isp]:
                next_dac[isp] = 255
                stage[isp] = STAGE.FINISH
                continue

            if iteration == 0:
                previous_rmse = rmse[isp]

            if previous_rmse[isp] < rmse[isp]:
                next_dac[isp] = previous_dac[isp]
                stage[isp] += 1
                if stage[isp] == STAGE.FINISH:
                    continue

            if stage[isp] == STAGE.STEP10:
                next_dac[isp] = dac[isp] + 10 * direction[isp]
            elif stage[isp] == STAGE.STEP1:
                next_dac[isp] = dac[isp] + 1 * direction[isp]

            if next_dac[isp] > 255:
                next_dac[isp] = 255
                stage[isp] = STAGE.FINISH
                continue
            elif next_dac[isp] < 10:
                next_dac[isp] = 255
                stage[isp] = STAGE.FINISH
                continue

    def _get_next_dac(self, rmse, direction):
        next_dac = self._get_next_dac_nb(
            self.current_dac, rmse, direction,
            self.previous_dac, self.previous_rmse,
            self.dead_sp_mask, self.stage, self.iteration
        )

        self.iteration += 1
        self.previous_dac = self.current_dac
        self.previous_rmse = rmse
        self.current_dac = next_dac

        return next_dac

    def process(self, r0_path):
        if self.final:
            return self.current_dac, True

        amplitude = self._extract_amplitude_array(r0_path)
        average, direction, rmse = self._calculate_average_and_rmse(amplitude)
        next_dac = self._get_next_dac(rmse, direction)
        self.final = (self.stage == STAGE.FINISH).all()

        print(f"Amplitude = {average[0]}, "
              f"RMSE = {rmse[0]}, "
              f"DAC = {next_dac[0]}")

        return next_dac.reshape((32, 16)), average, False
