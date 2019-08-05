from CHECLabPy.core.io.waveform import WaveformReader, Waveform
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping, \
    get_row_column
import numpy as np
import pandas as pd
import struct
import gzip


class SimtelWaveform(Waveform):
    @property
    def t_cpu(self):
        return self._t_cpu_container


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
            from ctapipe.coordinates import EngineeringCameraFrame
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
        reader = SimTelEventSource(
            input_url=path, max_events=max_events, back_seekable=True
        )
        self.seeker = EventSeeker(reader)

        first_event = self.seeker[0]
        tels = list(first_event.r0.tels_with_data)
        self.tel = tels[0]
        shape = first_event.r0.tel[self.tel].waveform.shape
        _, self.n_pixels, self.n_samples = shape
        self.n_modules = self.n_pixels // 64

        n_modules = 32
        camera_version = "1.1.0"
        self._camera_config = CameraConfiguration(camera_version)
        tc_mapping = self._camera_config.GetMapping(n_modules == 1)
        self.mapping = get_clp_mapping_from_tc_mapping(tc_mapping)
        n_rows = self.mapping.metadata['n_rows']
        n_columns = self.mapping.metadata['n_columns']
        camera_geom = first_event.inst.subarray.tel[tels[0]].camera
        engineering_frame = EngineeringCameraFrame(n_mirrors=2)
        engineering_geom = camera_geom.transform_to(engineering_frame)
        pix_x = engineering_geom.pix_x.value
        pix_y = engineering_geom.pix_y.value
        row, col = get_row_column(pix_x, pix_y)
        camera_2d = np.zeros((n_rows, n_columns), dtype=np.int)
        camera_2d[row, col] = np.arange(self.n_pixels, dtype=np.int)
        self.pixel_order = camera_2d[self.mapping['row'], self.mapping['col']]

        self.reference_pulse_path = self._camera_config.GetReferencePulsePath()
        self.camera_version = self._camera_config.GetVersion()

        self._iev = None
        self._t_cpu = None
        self.mc = None
        self.pointing = None
        self.mcheader = None

    def _build_waveform(self, event):
        self._fill_event_containers(event)
        samples = event.r1.tel[self.tel].waveform[0][self.pixel_order]
        mc_true = event.mc.tel[self.tel].photo_electron_image[self.pixel_order]
        waveform = SimtelWaveform(
            samples,
            iev=self._iev,
            is_r1=True,
            mc_true=mc_true,
            t_cpu_container=self._t_cpu
        )
        return waveform

    def _get_event(self, iev):
        event = self.seeker[iev]
        return self._build_waveform(event)

    def __iter__(self):
        for event in self.seeker:
            yield self._build_waveform(event)

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

    def _fill_event_containers(self, event):
        self._iev = event.count
        self._t_cpu = pd.to_datetime(event.trig.gps_time.value, unit='s')
        self.run_id = event.r0.obs_id

        self.mc = dict(
            iev=self._iev,
            t_cpu=self._t_cpu,
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
            iev=self._iev,
            t_cpu=self._t_cpu,
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
