from CHECLabPy.core.io.waveform import WaveformReader
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping
import pandas as pd
import struct
import gzip


class SimtelReader(WaveformReader):
    def __init__(self, path, max_events=None):
        """
        Reads simtelarray files utilising the SimTelEventSource from ctapipe

        Parameters
        ----------
        path : str
            Path to the simtel file
        max_events : int
            Maximum number of events to read from the file
        """
        super().__init__(path, max_events)

        try:
            from ctapipe.io import SimTelEventSource, EventSeeker
        except ModuleNotFoundError:
            msg = "Cannot find ctapipe installation"
            raise ModuleNotFoundError(msg)

        try:
            from target_calib import CameraConfiguration
        except ModuleNotFoundError:
            msg = ("Cannot find TARGET libraries, please follow installation "
                   "instructions from https://forge.in2p3.fr/projects/gct/"
                   "wiki/Installing_CHEC_Software")
            raise ModuleNotFoundError(msg)

        self.path = path
        reader = SimTelEventSource(input_url=path, max_events=max_events)
        self.seeker = EventSeeker(reader)

        first_event = self.seeker[0]
        tels = list(first_event.r0.tels_with_data)
        self.tel = tels[0]
        shape = first_event.r0.tel[self.tel].waveform.shape
        _, self.n_pixels, self.n_samples = shape
        self.n_modules = self.n_pixels // 64
        self.index = 0

        n_modules = 32
        camera_version = "1.1.0"
        self._camera_config = CameraConfiguration(camera_version)
        tc_mapping = self._camera_config.GetMapping(n_modules == 1)
        self.mapping = get_clp_mapping_from_tc_mapping(tc_mapping)
        pix_x = first_event.inst.subarray.tel[tels[0]].camera.pix_x.value
        pix_y = first_event.inst.subarray.tel[tels[0]].camera.pix_y.value
        self.mapping['xpix'] = pix_x
        self.mapping['ypix'] = pix_y
        self.reference_pulse_path = self._camera_config.GetReferencePulsePath()
        self.camera_version = self._camera_config.GetVersion()

        self.gps_time = None
        self.mc_true = None
        self.mc = None
        self.pointing = None
        self.mcheader = None

    def _get_event(self, iev):
        event = self.seeker[iev]
        self._fill_event_containers(event)
        return event.r1.tel[self.tel].waveform[0]

    @staticmethod
    def is_compatible(path):
        # read the first 4 bytes
        with open(path, 'rb') as f:
            marker_bytes = f.read(4)

        # if file is gzip, read the first 4 bytes with gzip again
        if marker_bytes[0] == 0x1f and marker_bytes[1] == 0x8b:
            with gzip.open(path, 'rb') as f:
                marker_bytes = f.read(4)

        # check for the simtel magic marker
        int_marker, = struct.unpack('I', marker_bytes)
        return int_marker == 3558836791 or int_marker == 931798996

    def __iter__(self):
        for event in self.seeker:
            self._fill_event_containers(event)
            yield event.r1.tel[self.tel].waveform[0]

    def _fill_event_containers(self, event):
        self.index = event.count
        self.run_id = event.r0.obs_id
        self.gps_time = event.trig.gps_time

        self.mc_true = event.mc.tel[self.tel].photo_electron_image

        self.mc = dict(
            iev=self.index,
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
            t_cpu=self.t_cpu,
            azimuth_raw=event.mc.tel[self.tel].azimuth_raw,
            altitude_raw=event.mc.tel[self.tel].altitude_raw,
            azimuth_cor=event.mc.tel[self.tel].azimuth_cor,
            altitude_cor=event.mc.tel[self.tel].altitude_cor,
        )

        if self.mcheader is None:
            mch = event.mcheader
            self.mcheader = dict(
                corsika_version=mch.corsika_version,
                simtel_version=mch.simtel_version,
                energy_range_min=mch.energy_range_min.value,
                energy_range_max=mch.energy_range_max.value,
                prod_site_B_total=mch.prod_site_B_total.value,
                prod_site_B_declination=mch.prod_site_B_declination.value,
                prod_site_B_inclination=mch.prod_site_B_inclination.value,
                prod_site_alt=mch.prod_site_alt.value,
                spectral_index=mch.spectral_index,
                shower_prog_start=mch.shower_prog_start,
                shower_prog_id=mch.shower_prog_id,
                detector_prog_start=mch.detector_prog_start,
                detector_prog_id=mch.detector_prog_id,
                num_showers=mch.num_showers,
                shower_reuse=mch.shower_reuse,
                max_alt=mch.max_alt.value,
                min_alt=mch.min_alt.value,
                max_az=mch.max_az.value,
                min_az=mch.min_az.value,
                diffuse=mch.diffuse,
                max_viewcone_radius=mch.max_viewcone_radius.value,
                min_viewcone_radius=mch.min_viewcone_radius.value,
                max_scatter_range=mch.max_scatter_range.value,
                min_scatter_range=mch.min_scatter_range.value,
                core_pos_mode=mch.core_pos_mode,
                injection_height=mch.injection_height.value,
                atmosphere=mch.atmosphere,
                corsika_iact_options=mch.corsika_iact_options,
                corsika_low_E_model=mch.corsika_low_E_model,
                corsika_high_E_model=mch.corsika_high_E_model,
                corsika_bunchsize=mch.corsika_bunchsize,
                corsika_wlen_min=mch.corsika_wlen_min.value,
                corsika_wlen_max=mch.corsika_wlen_max.value,
                corsika_low_E_detail=mch.corsika_low_E_detail,
                corsika_high_E_detail=mch.corsika_high_E_detail,
            )

    @property
    def n_events(self):
        return len(self.seeker)

    @property
    def t_cpu(self):
        return pd.to_datetime(self.gps_time.value, unit='s')
