import numpy as np
from enum import IntEnum
from CHECLabPy.waveform_reducers.average_wf import AverageWF
from CHECLabPy.core.io import ReaderR0


class STAGE(IntEnum):
    STEP10 = 0
    STEP5 = 1
    STEP1 = 2
    FINISH = 3


class GLOBALSTAGE(IntEnum):
    FIRST_PASS = 0
    RESET = 1
    SECOND_PASS = 2
    FINISH = 3


class AmplitudeMatcher:
    def __init__(self, calibrator, amplitude_aim,
                 starting_dacs=None, bad_hv=None, illumination_profile=None):
        """
        Logic and waveform processessing for amplitude matching

        Parameters
        ----------
        calibrator : CHECLabPy.calib.WaveformCalibrator
            Class for performing the online TargetCalib waveform calibration
        amplitude_aim : float
            Target amplitude for matching
        starting_dacs : ndarray
            Starting dac value for each superpixel
        bad_hv : ndarray
            Bad HV superpixel mask, of shape (n_superpixels)
        illumination_profile : CHECLabPy.calib.IlluminationProfile
            Class for handling the illumination profile
        """
        self.calibrator = calibrator
        n_pixels = calibrator.n_pixels
        n_samples = calibrator.n_samples
        n_superpixels = n_pixels // 4

        self.amplitude_aim = amplitude_aim

        if starting_dacs is None:
            starting_dacs = np.full(n_superpixels, 100)
        self.starting_dacs = starting_dacs.ravel()

        if bad_hv is None:
            self.sp_mask = np.zeros(n_superpixels, dtype=np.bool)
        else:
            if bad_hv.size == n_pixels:
                bad_hv.reshape((n_superpixels, 4)).all(1)
            self.sp_mask = bad_hv.ravel()

        self.illumination_profile = illumination_profile

        self.waveform_reducer = AverageWF(n_pixels, n_samples)

        self.global_stage = GLOBALSTAGE.FIRST_PASS
        self.ready_for_final2 = False
        self.stage = np.zeros(n_superpixels, dtype=np.int16)
        self.current_dac = self.starting_dacs.copy()
        self.previous_dac = self.starting_dacs.copy()
        self.previous_distance = np.zeros(n_superpixels)
        self.iteration = 0

    def _extract_amplitude(self, r0_waveforms, fci):
        r1_waveforms = self.calibrator(r0_waveforms, fci)
        amplitude = np.max(r1_waveforms, axis=1)
        if self.illumination_profile:
            amplitude = self.illumination_profile.unfold(amplitude)
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

    def _calculate_average(self, amplitude):
        aim = self.amplitude_aim
        n_events, n_pixels = amplitude.shape
        n_superpixels = n_pixels // 4
        average = amplitude.reshape((n_events, n_superpixels, 4)).mean((0, 2))
        direction = np.sign(average - aim)
        distance = ((average - aim) / aim) ** 2
        return average, direction, distance

    @staticmethod
    def _get_next_dac_nb(
            dac, distance, direction,
            previous_dac, previous_distance,
            sp_mask, stage, iteration
    ):
        next_dac = dac.copy()
        n_superpixel = next_dac.size
        for isp in range(n_superpixel):
            if stage[isp] == STAGE.FINISH:
                continue

            if sp_mask[isp]:
                next_dac[isp] = 255
                stage[isp] = STAGE.FINISH
                continue

            if iteration == 0:
                previous_distance[isp] = distance[isp]

            if previous_distance[isp] < distance[isp]:
                next_dac[isp] = previous_dac[isp]
                stage[isp] += 1
                continue

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

    def _get_next_dac(self, distance, direction):
        next_dac = self._get_next_dac_nb(
            self.current_dac, distance, direction,
            self.previous_dac, self.previous_distance,
            self.sp_mask, self.stage, self.iteration
        )

        self.previous_dac = self.current_dac
        self.previous_distance = distance
        self.iteration += 1
        self.current_dac = next_dac

        return next_dac

    def process(self, r0_path):
        """
        Process the R0 file, and obtain the next DAC values

        Parameters
        ----------
        r0_path : str
            Path to the R0 file

        Returns
        -------
        next_dac : ndarray
            The next DAC values to set, in shape (32, 16)
        """
        amplitude = self._extract_amplitude_array(r0_path)
        average_amplitude_pix = np.mean(amplitude, axis=0)
        average, direction, distance = self._calculate_average(amplitude)

        if self.global_stage == GLOBALSTAGE.RESET:
            self.iteration = 0
            self.stage[:] = STAGE.STEP1
            self.global_stage += 1

        if self.global_stage == GLOBALSTAGE.FINISH:
            return self.current_dac, average_amplitude_pix, True

        next_dac = self._get_next_dac(distance, direction)
        finished = (self.stage == STAGE.FINISH).all()
        if finished:
            self.global_stage += 1

        return next_dac.reshape((32, 16)), average_amplitude_pix, False
