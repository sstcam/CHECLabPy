"""
Executable for processing the R1 waveforms, and storing the reduced parameters
into a HDF5 file, readable as a `pandas.DataFrame`.
"""
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from CHECLabPy.core.io import WaveformReader, HDF5Writer
from CHECLabPy.core.chain import WaveformReducerChain
from CHECLabPy.utils.waveform import BaselineSubtractor
from CHECLabPy.calib import TimeCalibrator


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

    @classmethod
    def write_dl1_file(cls, reader, dl1_extractor, output_path=None):
        if not output_path:
            output_path = reader.path.replace('_r1', '_dl1').replace('.tio',
                                                                     '.h5')
            if reader.is_mc:
                output_path = reader.path.replace('.simtel.gz', '_dl1.h5')

        with cls(output_path) as writer:
            n_events_processed = 0
            n_events_skipped = 0
            start_time = None
            end_time = None
            desc = "Processing events"
            for waveforms in tqdm(reader, total=reader.n_events, desc=desc):
                if dl1_extractor.skip_event(waveforms):
                    n_events_skipped += 1
                    continue

                end_time = waveforms.t_cpu
                if start_time is None:
                    start_time = waveforms.t_cpu

                dl1 = dl1_extractor(waveforms)
                writer.append(
                    pd.DataFrame(dl1), key='data',
                    expectedrows=reader.n_events * reader.n_pixels
                )
                if reader.is_mc:
                    writer.append(
                        pd.DataFrame([reader.mc]), key='mc',
                        expectedrows=reader.n_events
                    )
                    writer.append(
                        pd.DataFrame([reader.pointing]), key='pointing',
                        expectedrows=reader.n_events
                    )

                n_events_processed += 1

            meta = dl1_extractor.metadata
            meta['n_events'] = n_events_processed
            meta['n_skipped'] = n_events_skipped
            meta['start_time'] = start_time
            meta['end_time'] = end_time

            writer.add_mapping(reader.mapping)
            add_meta = writer.add_metadata
            add_meta(name='metadata', **meta)
            add_meta(name='config', **dl1_extractor.config)
            add_meta(name='sn', **dl1_extractor.sn)
            add_meta(name='sipm_temp', **dl1_extractor.sipm_temp)
            add_meta(name='primary_temp', **dl1_extractor.primary_temp)
            add_meta(name='dac', **dl1_extractor.dac)
            add_meta(name='hvon', **dl1_extractor.hvon)
            if reader.is_mc:
                add_meta(key='mc', name='mcheader', **reader.mcheader)


class DL1Extractor:
    def __init__(self, reader, config_path):
        self.source = "CHECLabPy"
        self.reader = reader
        self.pixel_array = np.arange(self.reader.n_pixels)

        self.time_calibrator = None
        if not self.reader.is_mc:
            self.time_calibrator = TimeCalibrator()

        self.reducer_kwargs = dict(
            n_pixels=self.reader.n_pixels,
            n_samples=self.reader.n_samples,
            mapping=self.reader.mapping,
            reference_pulse_path=self.reader.reference_pulse_path,
            config_path=config_path,
        )
        self.chain = WaveformReducerChain(**self.reducer_kwargs)
        self.baseline_subtractor = BaselineSubtractor(self.reader)

        # Module metadata dicts
        self.config = self.chain.config.copy()
        self.config.pop('mapping', None)
        self.sn = {}
        self.sipm_temp = {}
        self.primary_temp = {}
        self.dac = {}
        self.hvon = {}
        for tm in range(self.reader.n_modules):
            tm_str = f'TM{tm:02}'
            self.sn[tm_str] = self.reader.get_sn(tm)
            self.sipm_temp[tm_str] = self.reader.get_sipm_temp(tm)
            self.primary_temp[tm_str] = self.reader.get_primary_temp(tm)
            for sp in range(self.reader.n_superpixels_per_module):
                tm_sp_str = f'TM{tm:02}_SP{sp:02}'
                self.dac[tm_sp_str] = self.reader.get_sp_dac(tm, sp)
                self.hvon[tm_sp_str] = self.reader.get_sp_hvon(tm, sp)

    @staticmethod
    def skip_event(waveforms):
        return waveforms.stale or waveforms.missing_packets

    @property
    def metadata(self):
        return dict(
            source=self.source,
            run_id=self.reader.run_id,
            is_mc=self.reader.is_mc,
            date_generated=pd.datetime.now(),
            input_path=self.reader.path,
            n_modules=self.reader.n_modules,
            n_pixels=self.reader.n_pixels,
            n_superpixels_per_module=self.reader.n_superpixels_per_module,
            n_samples=self.reader.n_samples,
            camera_version=self.reader.camera_version,
            n_cells=self.reader.n_cells,
        )

    def __call__(self, waveforms):

        # Calibrate timing offsets between pixels
        waveforms_tc = waveforms
        if self.time_calibrator is not None:
            waveforms_tc = self.time_calibrator(waveforms)

        # Subtract waveform baselines per pixel
        waveforms_bs = self.baseline_subtractor.subtract(waveforms_tc)
        bs = self.baseline_subtractor.baseline

        params = dict(
            iev=waveforms.iev,
            pixel=self.pixel_array,
            t_cpu=waveforms.t_cpu,
            t_tack=waveforms.t_tack,
            first_cell_id=waveforms.first_cell_id,
            baseline_subtracted=bs,
            **self.chain.process(waveforms_bs),
        )
        if waveforms.mc_true is not None:
            params['mc_true'] = waveforms.mc_true

        return params


def main():
    parser = argparse.ArgumentParser(
        description='Reduce a waveform file into a *_dl1.h5 file containing '
                    'various parameters extracted from the waveforms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--files', dest='input_paths', nargs='+',
        help='path to the file containing waveforms (TIO or simtel)'
    )
    parser.add_argument(
        '-o', '--output', dest='output_path', action='store',
        help='path to store the output HDF5 dl1 file (OPTIONAL, will be '
             'automatically set if not specified)'
    )
    parser.add_argument(
        '-n', '--maxevents', dest='max_events', action='store', type=int,
        help='Number of events to process'
    )
    parser.add_argument(
        '-c', '--config', dest='config_path',
        help="Path to config file. If no path is given, then the default "
             "columns will be stored."
    )
    args = parser.parse_args()

    input_paths = args.input_paths
    n_files = len(input_paths)
    for i_path, input_path in enumerate(input_paths):
        print("PROGRESS: Reducing file {}/{}".format(i_path + 1, n_files))
        reader = WaveformReader.from_path(input_path, args.max_events)
        dl1_extractor = DL1Extractor(reader, args.config_path)
        DL1Writer.write_dl1_file(reader, dl1_extractor, args.output_path)


if __name__ == '__main__':
    main()
