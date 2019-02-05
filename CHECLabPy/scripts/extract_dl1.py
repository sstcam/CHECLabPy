"""
Executable for processing the R1 waveforms, and storing the reduced parameters
into a HDF5 file, readable as a `pandas.DataFrame`.
"""
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from tqdm import tqdm
from CHECLabPy.core.io import ReaderR1, HDF5Writer
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
        df['pixel'] = df['pixel'].astype(np.uint32)
        df['first_cell_id'] = df['first_cell_id'].astype(np.uint16)
        df['t_tack'] = df['t_tack'].astype(np.uint64)
        df = df.sort_values(["iev", "pixel"])
        return df


def main():
    description = ('Reduce a *_r1.tio file into a *_dl1.h5 file containing '
                   'various parameters extracted from the waveforms')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--files', dest='input_paths', nargs='+',
                        help='path to the TIO r1 run files')
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
        reader = ReaderR1(input_path, args.max_events)
        n_events = reader.n_events
        n_modules = reader.n_modules
        n_pixels = reader.n_pixels
        n_samples = reader.n_samples
        n_cells = reader.n_cells
        expectedrows = n_events * n_pixels
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
            output_path = (
                input_path.replace('_r1', '_dl1').replace('.tio', '.h5')
            )

        with DL1Writer(output_path) as writer:
            t_cpu = 0
            start_time = 0
            desc = "Processing events"
            for waveforms in tqdm(reader, total=n_events, desc=desc):
                iev = reader.index

                t_tack = reader.current_tack
                t_cpu = reader.t_cpu
                fci = reader.first_cell_ids

                if not start_time:
                    start_time = t_cpu

                waveforms_bs = baseline_subtractor.subtract(waveforms)
                bs = baseline_subtractor.baseline

                params = chain.process(waveforms_bs)

                df = pd.DataFrame(dict(
                    iev=iev,
                    pixel=pixel_array,
                    first_cell_id=fci,
                    t_cpu=t_cpu,
                    t_tack=t_tack,
                    baseline_subtracted=bs,
                    **params
                ))
                writer.append(df, key='data', expectedrows=expectedrows)

            sn_dict = {}
            for tm in range(n_modules):
                sn_dict['TM{:02d}_SN'.format(tm)] = reader.get_sn(tm)

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

            writer.add_mapping(mapping)
            writer.add_metadata(name='metadata', **metadata)
            writer.add_metadata(name='sn', **sn_dict)
            writer.add_metadata(name='config', **config)


if __name__ == '__main__':
    main()
