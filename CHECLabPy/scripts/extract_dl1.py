"""
Executable for processing the R1 waveforms, and storing the reduced parameters
into a HDF5 file, readable as a `pandas.DataFrame`.
"""
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from tqdm import tqdm
from CHECLabPy.core.io import SimtelReader, WaveformReader, HDF5Writer
from CHECLabPy.core.chain import WaveformReducerChain
from CHECLabPy.utils.waveform import BaselineSubtractor


class DL1Writer(HDF5Writer):
    @staticmethod
    def _prepare_before_append(df):
        df_float = df.select_dtypes(
            include=['float']
        ).apply(pd.to_numeric, downcast='float')
        df[df_float.columns] = df_float
        df['iev'] = df['iev'].astype(np.uint32)
        if 'first_cell_id' in df.columns:
            df['first_cell_id'] = df['first_cell_id'].astype(np.uint16)
        if 't_tack' in df.columns:
            df['t_tack'] = df['t_tack'].astype(np.uint64)
        if 'mc_true' in df.columns:
            df['mc_true'] = df['mc_true'].astype(np.uint32)
        if 'pixel' in df.columns:
            df['pixel'] = df['pixel'].astype(np.uint32)
            df = df.sort_values(["iev", "pixel"])
        return df


def main():
    description = ('Reduce a waveform file into a *_dl1.h5 file containing '
                   'various parameters extracted from the waveforms')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--files', dest='input_paths', nargs='+',
                        help='path to the file containing waveforms '
                             '(TIO or simtel)')
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        help='path to store the output HDF5 dl1 file '
                             '(OPTIONAL, will be automatically set if '
                             'not specified)')
    parser.add_argument('-n', '--maxevents', dest='max_events', action='store',
                        help='Number of events to process', type=int)
    parser.add_argument('-c', '--config', dest='config_path',
                        help="Path to config file. If no path is given, "
                             "then the default columns will be stored.")
    args = parser.parse_args()

    input_paths = args.input_paths
    n_files = len(input_paths)
    for i_path, input_path in enumerate(input_paths):
        print("PROGRESS: Reducing file {}/{}".format(i_path + 1, n_files))
        reader = WaveformReader.from_path(input_path, args.max_events)
        n_events = reader.n_events
        n_modules = reader.n_modules
        n_pixels = reader.n_pixels
        n_samples = reader.n_samples
        pixel_array = np.arange(n_pixels)

        is_mc = isinstance(reader, SimtelReader)

        kwargs = dict(
            n_pixels=n_pixels,
            n_samples=n_samples,
            mapping=reader.mapping,
            reference_pulse_path=reader.reference_pulse_path,
            config_path=args.config_path,
        )
        chain = WaveformReducerChain(**kwargs)
        baseline_subtractor = BaselineSubtractor(reader)

        input_path = reader.path
        output_path = args.output_path
        if not output_path:
            output_path = (
                input_path.replace('_r1', '_dl1').replace('.tio', '.h5')
            )
            if is_mc:
                output_path = input_path.replace('.simtel.gz', '_dl1.h5')

        with DL1Writer(output_path) as writer:
            t_cpu = 0
            start_time = 0
            desc = "Processing events"
            for waveforms in tqdm(reader, total=n_events, desc=desc):
                iev = reader.index
                t_cpu = reader.t_cpu

                if not start_time:
                    start_time = t_cpu

                waveforms_bs = baseline_subtractor.subtract(waveforms)
                bs = baseline_subtractor.baseline

                params = dict(
                    iev=iev,
                    pixel=pixel_array,
                    t_cpu=t_cpu,
                    t_tack=reader.current_tack,
                    first_cell_id=reader.first_cell_ids,
                    baseline_subtracted=bs,
                    **chain.process(waveforms_bs),
                )
                if is_mc:
                    params['mc_true'] = reader.mc_true

                writer.append(
                    pd.DataFrame(params), key='data',
                    expectedrows=n_events*n_pixels
                )
                if is_mc:
                    writer.append(
                        pd.DataFrame([reader.mc]), key='mc',
                        expectedrows=n_events
                    )
                    writer.append(
                        pd.DataFrame([reader.pointing]), key='pointing',
                        expectedrows=n_events
                    )

            sn_dict = {}
            for tm in range(n_modules):
                sn_dict['TM{:02d}'.format(tm)] = reader.get_sn(tm)

            metadata = dict(
                source="CHECLabPy",
                run_id=reader.run_id,
                is_mc=is_mc,
                date_generated=pd.datetime.now(),
                input_path=input_path,
                n_events=n_events,
                n_modules=n_modules,
                n_pixels=n_pixels,
                n_samples=n_samples,
                start_time=start_time,
                end_time=t_cpu,
                camera_version=reader.camera_version,
                n_cells=reader.n_cells,
            )
            config = chain.config
            config.pop('mapping', None)

            writer.add_mapping(reader.mapping)
            writer.add_metadata(name='metadata', **metadata)
            writer.add_metadata(name='config', **config)
            writer.add_metadata(name='sn', **sn_dict)
            if is_mc:
                writer.add_metadata(
                    key='mc', name='mcheader', **reader.mcheader
                )


if __name__ == '__main__':
    main()
