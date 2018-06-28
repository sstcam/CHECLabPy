"""
Executable for processing the R1 waveforms, and storing the reduced parameters
into a HDF5 file, openable as a `pandas.DataFrame`.
"""
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from CHECLabPy.core.io import DL1Writer
from CHECLabPy.core.factory import WaveformReducerFactory
from ctapipe.calib import HESSIOR1Calibrator
from ctapipe.io import HESSIOEventSource, EventSeeker
from target_calib import CameraConfiguration
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping


class BaselineSubtractor:
    """
    Keep track of the baseline for the previous `n_base` events, therefore
    keeping a "rolling average" of the baseline to subtract.
    """
    def __init__(self, source, n_base=50, n_base_samples=16):
        self.n_base = n_base
        self.n_base_samples = n_base_samples
        self.iev = 0

        first_event = source[0]
        tels = list(first_event.r0.tels_with_data)
        _, n_pixels, n_samples = first_event.r0.tel[tels[0]].waveform.shape
        r1 = HESSIOR1Calibrator()

        self.baseline_waveforms = np.zeros((n_base, n_pixels, n_base_samples))
        for event in source:
            r1.calibrate(event)
            waveforms = event.r1.tel[tels[0]].waveform[0]
            ev = event.count
            if ev >= n_base:
                break
            self.baseline_waveforms[ev] = waveforms[:, :n_base_samples]
        self.baseline = np.mean(self.baseline_waveforms, axis=(0, 2))

    def update_baseline(self, waveforms):
        entry = self.iev % self.n_base
        self.baseline_waveforms[entry] = waveforms[:, :self.n_base_samples]
        self.baseline = np.mean(self.baseline_waveforms, axis=(0, 2))
        self.iev += 1

    def subtract(self, waveforms):
        self.update_baseline(waveforms)
        return waveforms - self.baseline[:, None]


def main():
    description = ('Reduce a *_r1.tio file into a *_dl1.hdf5 file containing '
                   'various parameters extracted from the waveforms')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--files', dest='input_paths', nargs='+',
                        help='path to the TIO r1 run files')
    parser.add_argument('-m', '--monitor', dest='monitor', action='store',
                        help='path to the monitor file (OPTIONAL)')
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        help='path to store the output HDF5 dl1 file '
                             '(OPTIONAL, will be automatically set if '
                             'not specified)')
    parser.add_argument('-n', '--maxevents', dest='max_events', action='store',
                        help='Number of events to process', type=int)
    parser.add_argument('-r', '--reducer', dest='reducer', action='store',
                        default='AverageWF',
                        choices=WaveformReducerFactory.subclass_names,
                        help='WaveformReducer to use')
    parser.add_argument('-c', '--config', dest='configuration',
                        help="""Configuration to pass to the waveform reducer
                        (Usage: '{"window_shift":6, "window_size":6}') """)
    parser.add_argument('-p', '--plot', dest='plot', action='store_true',
                        help="Plot stages for waveform reducers")
    args = parser.parse_args()
    if args.configuration:
        config = json.loads(args.configuration)
        config_string = args.configuration
    else:
        config = {}
        config_string = ""

    input_paths = args.input_paths
    n_files = len(input_paths)
    for i_path, input_path in enumerate(input_paths):
        print("PROGRESS: Reducing file {}/{}".format(i_path + 1, n_files))

        kwargs = dict(input_url=input_path, max_events=args.max_events)
        reader = HESSIOEventSource(**kwargs)
        seeker = EventSeeker(reader)

        n_events = len(seeker)

        first_event = seeker[0]
        tels = list(first_event.r0.tels_with_data)
        _, n_pixels, n_samples = first_event.r0.tel[tels[0]].waveform.shape
        n_modules = 32
        n_cells = 1
        pixel_array = np.arange(n_pixels)
        camera_version = "1.1.0"
        camera_config = CameraConfiguration(camera_version)
        tc_mapping = camera_config.GetMapping(n_modules == 1)
        mapping = get_clp_mapping_from_tc_mapping(tc_mapping)
        if 'reference_pulse_path' not in config:
            reference_pulse_path = camera_config.GetReferencePulsePath()
            config['reference_pulse_path'] = reference_pulse_path

        kwargs = dict(
            n_pixels=n_pixels,
            n_samples=n_samples,
            plot=args.plot,
            mapping=mapping,
            **config
        )
        reducer = WaveformReducerFactory.produce(args.reducer, **kwargs)
        baseline_subtractor = BaselineSubtractor(seeker)

        input_path = reader.input_url
        output_path = args.output_path
        if not output_path:
            output_path = input_path.replace(".simtel.gz", "_dl1.h5")
            output_path = output_path.replace("run", "Run")

        r1 = HESSIOR1Calibrator()

        with DL1Writer(output_path, n_events*n_pixels, args.monitor) as writer:
            t_cpu = 0
            start_time = 0
            desc = "Processing events"
            for event in tqdm(seeker, total=n_events, desc=desc):
                iev = event.count

                r1.calibrate(event)
                waveforms = event.r1.tel[tels[0]].waveform[0]
                mc_true = event.mc.tel[tels[0]].photo_electron_image

                t_cpu = pd.to_datetime(event.trig.gps_time.value, unit='s')

                if not start_time:
                    start_time = t_cpu

                waveforms_bs = baseline_subtractor.subtract(waveforms)
                bs = baseline_subtractor.baseline

                params = reducer.process(waveforms_bs)

                df_ev = pd.DataFrame(dict(
                    iev=iev,
                    pixel=pixel_array,
                    first_cell_id=0,
                    t_cpu=t_cpu,
                    t_tack=0,
                    baseline_subtracted=bs,
                    **params,
                    mc_true=mc_true
                ))
                writer.append_event(df_ev)

            sn_dict = {}
            for tm in range(n_modules):
                sn_dict['TM{:02d}_SN'.format(tm)] = "NaN"

            metadata = dict(
                source="CHECLabPy",
                date_generated=pd.datetime.now(),
                input_path=input_path,
                n_events=n_events,
                n_modules=n_modules,
                n_pixels=n_pixels,
                n_samples=n_samples,
                n_cells=n_cells,
                start_time=start_time,
                end_time=t_cpu,
                camera_version=camera_version,
                reducer=reducer.__class__.__name__,
                configuration=config_string,
                **sn_dict
            )

            writer.add_metadata(**metadata)
            writer.add_mapping(mapping)


if __name__ == '__main__':
    main()
