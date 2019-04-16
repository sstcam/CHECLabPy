import numpy as np
from enum import IntEnum
from CHECLabPy.waveform_reducers.average_wf import AverageWF
from CHECLabPy.core.io import ReaderR0


class STAGE(IntEnum):
    STEP10 = 0
    STEP5 = 1
    STEP1 = 2
    FINISH = 3


class GLOBAL_STAGE(IntEnum):
    FIRST_PASS = 0
    RESET = 1
    SECOND_PASS = 2
    FINISH = 3


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
        self.global_stage = GLOBAL_STAGE.FIRST_PASS
        self.ready_for_final2 = False
        self.stage = np.zeros(n_superpixels, dtype=np.int16)
        self.dead_sp_mask = self.dead_mask.reshape((n_superpixels, 4)).all(1)
        self.current_dac = self.starting_dacs.copy()
        self.previous_dac = self.starting_dacs.copy()
        self.previous_rmse = np.zeros(n_superpixels)
        self.previous_direction = np.zeros(n_superpixels, dtype=np.int16)
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
    def _get_next_dac_nb(
            dac, rmse, direction,
            previous_dac, previous_rmse, previous_direction,
            dead_sp, stage, iteration
    ):
        next_dac = dac.copy()
        n_superpixel = next_dac.size
        for isp in range(n_superpixel):
            if stage[isp] == STAGE.FINISH:
                continue

            if dead_sp[isp]:
                next_dac[isp] = 255
                stage[isp] = STAGE.FINISH
                continue

            if iteration == 0:
                previous_rmse[isp] = rmse[isp]

            if previous_rmse[isp] < rmse[isp]:
                next_dac[isp] = previous_dac[isp]
                direction[isp] = previous_direction[isp]
                stage[isp] += 1
                if stage[isp] == STAGE.FINISH:
                    continue
            else:
                previous_dac[isp] = dac[isp]
                previous_rmse[isp] = rmse[isp]
                previous_direction[isp] = direction[isp]

            if stage[isp] == STAGE.STEP10:
                next_dac[isp] += 10 * direction[isp]
            elif stage[isp] == STAGE.STEP5:
                next_dac[isp] += 5 * direction[isp]
            elif stage[isp] == STAGE.STEP1:
                next_dac[isp] += 1 * direction[isp]

            if next_dac[isp] > 255:
                next_dac[isp] = 255
                stage[isp] = STAGE.FINISH
                continue
            elif next_dac[isp] < 10:
                next_dac[isp] = 255
                stage[isp] = STAGE.FINISH
                continue

        return next_dac

    def _get_next_dac(self, rmse, direction):
        next_dac = self._get_next_dac_nb(
            self.current_dac, rmse, direction,
            self.previous_dac, self.previous_rmse, self.previous_direction,
            self.dead_sp_mask, self.stage, self.iteration
        )

        self.iteration += 1
        self.current_dac = next_dac

        return next_dac

    def process(self, r0_path):
        amplitude = self._extract_amplitude_array(r0_path)
        average_amplitude_pix = np.mean(amplitude, axis=0)
        average, direction, rmse = self._calculate_average_and_rmse(amplitude)

        if self.global_stage == GLOBAL_STAGE.RESET:
            self.iteration = 0
            self.stage[:] = STAGE.STEP5
            self.global_stage += 1

        if self.global_stage == GLOBAL_STAGE.FINISH:
            return self.current_dac, average_amplitude_pix, True

        next_dac = self._get_next_dac(rmse, direction)
        finished = (self.stage == STAGE.FINISH).all()
        if finished:
            self.global_stage += 1

        return next_dac.reshape((32, 16)), average_amplitude_pix, False
