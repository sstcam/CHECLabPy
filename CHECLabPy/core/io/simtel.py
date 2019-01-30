from target_calib import CameraConfiguration
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping
import pandas as pd


class SimtelReader:
    def __init__(self, path, max_events=None):
        from ctapipe.calib import HESSIOR1Calibrator
        from ctapipe.io import SimTelEventSource, EventSeeker
        self.path = path
        reader = SimTelEventSource(input_url=path, max_events=max_events)
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

        self.gps_time = None
        self.mc_true = None
        self.mc = None
        self.pointing = None

    @property
    def n_events(self):
        return len(self.seeker)

    def __iter__(self):
        for event in self.seeker:
            self.index = event.count
            self.gps_time = event.trig.gps_time
            self.obs_id = event.r0.obs_id

            self.r1.calibrate(event)
            waveforms = event.r1.tel[self.tel].waveform[0]
            self.mc_true = event.mc.tel[self.tel].photo_electron_image

            self.mc = dict(
                iev=self.index,
                obs_id=self.obs_id,
                t_cpu=self.t_cpu,
                energy=event.mc.energy.value,
                alt=event.mc.alt.value,
                az=event.mc.az.value,
                core_x=event.mc.core_x.value,
                core_y=event.mc.core_y.value,
                h_first_int=event.mc.h_first_int.value,
                x_max=event.mc.x_max.value,
                shower_primary_id=event.mc.shower_primary_id
            )

            self.pointing = dict(
                iev=self.index,
                obs_id=self.obs_id,
                t_cpu=self.t_cpu,
                azimuth_raw=event.mc.tel[self.tel].azimuth_raw,
                altitude_raw=event.mc.tel[self.tel].altitude_raw,
                azimuth_cor=event.mc.tel[self.tel].azimuth_cor,
                altitude_cor=event.mc.tel[self.tel].altitude_cor,
            )

            yield waveforms

    def __getitem__(self, iev):
        event = self.seeker[iev]
        self.index = event.count
        self.r1.calibrate(event)
        waveforms = event.r1.tel[self.tel].waveform[0]
        return waveforms

    @property
    def t_cpu(self):
        return pd.to_datetime(self.gps_time.value, unit='s')
