"""
Executable for processing the R1 waveforms, and storing the reduced parameters
into a HDF5 file, readable as a `pandas.DataFrame`.
"""
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from tqdm import tqdm
from CHECLabPy.core.io import SimtelReader, HDF5Writer
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
        if 'pixel' in df.columns:
            df['pixel'] = df['pixel'].astype(np.uint32)
            df = df.sort_values(["iev", "pixel"])
        else:
            df = df.sort_values(["iev"])
        return df


def main():
    description = ('Reduce a simtel.gz file into a *_dl1.h5 file containing '
                   'various parameters extracted from the waveforms')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--files', dest='input_paths', nargs='+',
                        help='path to the simtel run files')
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
        reader = SimtelReader(input_path, args.max_events)
        n_events = reader.n_events
        n_pixels = reader.n_pixels
        n_samples = reader.n_samples
        expectedrows = n_events * n_pixels
        expectedrows_mc = n_events
        pixel_array = np.arange(n_pixels)
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
            output_path = input_path.replace('.simtel.gz', '.h5')

        with DL1Writer(output_path) as writer:
            t_cpu = 0
            start_time = 0
            desc = "Processing events"
            for waveforms in tqdm(reader, total=n_events, desc=desc):
                iev = reader.index
                t_cpu = reader.t_cpu
                mc_true = reader.mc_true

                if not start_time:
                    start_time = t_cpu

                waveforms_bs = baseline_subtractor.subtract(waveforms)
                bs = baseline_subtractor.baseline

                params = chain.process(waveforms_bs)

                df = pd.DataFrame(dict(
                    iev=iev,
                    pixel=pixel_array,
                    t_cpu=t_cpu,
                    baseline_subtracted=bs,
                    mc_true=mc_true,
                    **params
                ))
                writer.append(
                    df, key='data', expectedrows=expectedrows
                )
                writer.append(
                    pd.DataFrame([reader.mc]), key='mc',
                    expectedrows=expectedrows_mc
                )
                writer.append(
                    pd.DataFrame([reader.pointing]), key='pointing',
                    expectedrows=expectedrows_mc
                )

            metadata = dict(
                source="CHECLabPy",
                obs_id=reader.obs_id,
                date_generated=pd.datetime.now(),
                input_path=input_path,
                n_events=n_events,
                n_pixels=n_pixels,
                n_samples=n_samples,
                start_time=start_time,
                end_time=t_cpu,
            )
            config = chain.config
            config.pop('mapping', None)

            writer.add_mapping(mapping)
            writer.add_metadata(name='metadata', **metadata)
            writer.add_metadata(name='config', **config)
            writer.add_metadata(name='mcheader', **reader.mcheader)


if __name__ == '__main__':
    main()
