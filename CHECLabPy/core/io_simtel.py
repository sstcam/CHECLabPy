from ctapipe.calib import HESSIOR1Calibrator
from ctapipe.io import HESSIOEventSource, EventSeeker
from target_calib import CameraConfiguration
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping


class ReaderSimtel:
    def __init__(self, path, max_events=None):
        kwargs = dict(input_url=path, max_events=max_events)
        reader = HESSIOEventSource(**kwargs)
        self.seeker = EventSeeker(reader)

        first_event = self.seeker[0]
        tels = list(first_event.r0.tels_with_data)
        self.tel = tels[0]
        shape = first_event.r0.tel[self.tel].waveform.shape
        _, self.n_pixels, self.n_samples = shape
        self.index = 0

        n_modules = 32
        camera_version = "1.1.0"
        self.camera_config = CameraConfiguration(camera_version)
        tc_mapping = self.camera_config.GetMapping(n_modules == 1)
        self.mapping = get_clp_mapping_from_tc_mapping(tc_mapping)
        pix_x = first_event.inst.subarray.tel[tels[0]].camera.pix_x.value
        pix_y = first_event.inst.subarray.tel[tels[0]].camera.pix_y.value
        self.mapping['xpix'] = pix_x
        self.mapping['ypix'] = pix_y
        self.reference_pulse_path = self.camera_config.GetReferencePulsePath()

        self.r1 = HESSIOR1Calibrator()

        self.mc_true = None

    @property
    def n_events(self):
        return len(self.seeker)

    def __iter__(self):
        for event in self.seeker:
            self.index = event.count
            self.r1.calibrate(event)
            waveforms = event.r1.tel[self.tel].waveform[0]
            self.mc_true = event.mc.tel[self.tel].photo_electron_image
            yield waveforms

    def __getitem__(self, iev):
        event = self.seeker[iev]
        self.index = event.count
        self.r1.calibrate(event)
        waveforms = event.r1.tel[self.tel].waveform[0]
        return waveforms
