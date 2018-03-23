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
from CHECLabPy.core.file_handling import ReaderR1, DL1Writer
from CHECLabPy.core.factory import WaveformReducerFactory
from CHECLabPy.utils.waveform import BaselineSubtractor


def main():
    description = ('Reduce a *_r1.tio file into a *_dl1.hdf5 file containing '
                   'various parameters extracted from the waveforms')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
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

    source = ReaderR1(args.input_path, args.max_events)
    n_events = source.n_events
    n_pixels = source.n_pixels
    n_samples = source.n_samples
    n_cells = source.n_cells
    pixel_array = np.arange(n_pixels)

    kwargs = dict(
        n_pixels=n_pixels,
        n_samples=n_samples,
        plot=args.plot,
        **config
    )
    reducer = WaveformReducerFactory.produce(args.reducer, **kwargs)
    baseline_subtractor = BaselineSubtractor(source)

    input_path = source.path
    output_path = args.output_path
    if not output_path:
        output_path = input_path.rsplit('_r1', 1)[0] + "_dl1.h5"

    with DL1Writer(output_path, n_events*n_pixels, args.monitor) as writer:
        t_cpu = 0
        start_time = 0
        desc = "Processing events"
        for waveforms in tqdm(source, total=n_events, desc=desc):
            iev = source.index

            t_tack = source.reader.fCurrentTimeTack
            t_cpu_sec = source.reader.fCurrentTimeSec
            t_cpu_ns = source.reader.fCurrentTimeNs
            t_cpu = pd.to_datetime(
                np.int64(t_cpu_sec * 1E9) + np.int64(t_cpu_ns),
                unit='ns'
            )
            fci = source.first_cell_ids

            if not start_time:
                start_time = t_cpu

            waveforms_bs = baseline_subtractor.subtract(waveforms)
            bs = baseline_subtractor.baseline

            params = reducer.process(waveforms_bs)

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
            input_path=input_path,
            n_events=n_events,
            n_pixels=n_pixels,
            n_samples=n_samples,
            n_cells=n_cells,
            start_time=start_time,
            end_time=t_cpu,
            version="0.0.0",
            reducer=reducer.__class__.__name__,
            configuration=config_string
        )

        writer.add_metadata(**metadata)


if __name__ == '__main__':
    main()
