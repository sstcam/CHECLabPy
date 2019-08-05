import numpy as np


class WaveformCalibrator:
    def __init__(self, pedestal_path, n_pixels, n_samples, sn_list=None):
        if sn_list is None:
            from target_calib import Calibrator
            self.calibrator = Calibrator(pedestal_path)
        else:
            from target_calib import CalibratorMultiFile
            self.calibrator = CalibratorMultiFile(pedestal_path, sn_list)
        self.n_pixels = n_pixels
        self.n_samples = n_samples

    def __call__(self, waveforms, fci):
        calibrated_wfs = waveforms.astype(np.float32, copy=True)
        self.calibrator.ApplyEvent(waveforms, fci, calibrated_wfs)
        calibrated_wfs.r1 = True
        return calibrated_wfs

    @classmethod
    def from_tio_reader(cls, pedestal_path, tio_reader):
        sn_list = [tio_reader.get_sn(tm) for tm in range(tio_reader.n_modules)]
        n_pixels = tio_reader.n_pixels
        n_samples = tio_reader.n_samples
        return cls(pedestal_path, n_pixels, n_samples, sn_list)
