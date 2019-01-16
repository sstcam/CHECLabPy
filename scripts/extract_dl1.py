"""
Executable for processing the R1 waveforms, and storing the reduced parameters
into a HDF5 file, readable as a `pandas.DataFrame`.
"""
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from tqdm import tqdm
from CHECLabPy.core.io import ReaderR1, DL1Writer
from CHECLabPy.core.chain import WaveformReducerChain
from CHECLabPy.utils.waveform import BaselineSubtractor


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
    parser.add_argument('-c', '--config', dest='config_path',
                        help="Path to config file")
    args = parser.parse_args()

    input_paths = args.input_paths
    n_files = len(input_paths)
    for i_path, input_path in enumerate(input_paths):
        print("PROGRESS: Reducing file {}/{}".format(i_path + 1, n_files))
        reader = ReaderR1(input_path, args.max_events)
        n_events = reader.n_events
        n_modules = reader.n_modules
        n_pixels = reader.n_pixels
        n_samples = reader.n_samples
        n_cells = reader.n_cells
        pixel_array = np.arange(n_pixels)
        camera_version = reader.camera_version
        mapping = reader.mapping
        reference_pulse_path = reader.reference_pulse_path

        kwargs = dict(
            n_pixels=n_pixels,
            n_samples=n_samples,
            mapping=mapping,
            reference_pulse_path=reference_pulse_path,
            config_path=args.config_path,
        )
        chain = WaveformReducerChain(**kwargs)
        baseline_subtractor = BaselineSubtractor(reader)

        input_path = reader.path
        output_path = args.output_path
        if not output_path:
            output_path = input_path.rsplit('_r1', 1)[0] + "_dl1.h5"

        with DL1Writer(output_path, n_events*n_pixels, args.monitor) as writer:
            t_cpu = 0
            start_time = 0
            desc = "Processing events"
            for waveforms in tqdm(reader, total=n_events, desc=desc):
                iev = reader.index

                t_tack = reader.current_tack
                t_cpu_sec = reader.current_cpu_s
                t_cpu_ns = reader.current_cpu_ns
                t_cpu = pd.to_datetime(
                    np.int64(t_cpu_sec * 1E9) + np.int64(t_cpu_ns),
                    unit='ns'
                )
                fci = reader.first_cell_ids

                if not start_time:
                    start_time = t_cpu

                waveforms_bs = baseline_subtractor.subtract(waveforms)
                bs = baseline_subtractor.baseline

                params = chain.process(waveforms_bs)

                df_ev = pd.DataFrame(dict(
                    iev=iev,
                    pixel=pixel_array,
                    first_cell_id=fci,
                    t_cpu=t_cpu,
                    t_tack=t_tack,
                    baseline_subtracted=bs,
                    **params
                ))
                writer.append_event(df_ev)

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
            )
            config = chain.config
            config.pop('mapping', None)

            writer.add_metadata(**metadata)
            writer.add_config(**config)
            writer.add_sn(n_modules, reader.get_sn)
            writer.add_mapping(mapping)


if __name__ == '__main__':
    main()
