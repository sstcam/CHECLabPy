import numpy as np


def get_waveform_calibrator_args(tio_reader):
    sn_list = [tio_reader.get_sn(tm) for tm in range(tio_reader.n_modules)]
    return tio_reader.n_pixels, tio_reader.n_samples, sn_list


class WaveformCalibrator:
    def __init__(self, pedestal_path, n_pixels, n_samples, sn_list):
        from target_calib import CalibratorMultiFile
        self.calibrator = CalibratorMultiFile(pedestal_path, sn_list)
        self.calibrated_wfs = np.zeros((n_pixels, n_samples), dtype=np.float32
        )
        self.n_pixels = n_pixels
        self.n_samples = n_samples

    def __call__(self, waveforms, fci):
        self.calibrator.ApplyEvent(waveforms, fci, self.calibrated_wfs)
        return self.calibrated_wfs

    @classmethod
    def from_tio_reader(cls, pedestal_path, tio_reader):
        args = get_waveform_calibrator_args(tio_reader)
        return cls(pedestal_path, *args)
