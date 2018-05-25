import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import yaml
import json
import os
from numpy.polynomial.polynomial import polyfit
from CHECLabPy.core.io import ReaderR1
from CHECLabPy.core.factory import WaveformReducerFactory
from CHECLabPy.utils.waveform import BaselineSubtractor
from CHECLabPy.utils.files import open_runlist_r1
from CHECLabPy.plotting.setup import Plotter


class Scatter(Plotter):
    def __init__(self, x_label="", y_label=""):
        super().__init__()
        self.x_label = x_label
        self.y_label = y_label

    def add(self, x, y, xerr=None, yerr=None, m=None, c=None, label='', **kw):
        color = self.ax._get_lines.get_next_color()
        (_, caps, _) = self.ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                                        mew=1, capsize=3, elinewidth=0.7,
                                        markersize=3,
                                        color=color, label=label, **kw)

        for cap in caps:
            cap.set_markeredgewidth(0.7)

        if m is not None and c is not None:
            y_lin = m * x + c
            self.ax.plot(x, y_lin, color=color)

    def set_log_x(self):
        self.ax.set_xscale('log')
        self.ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, _: '{:g}'.format(x)))

    def set_log_y(self):
        self.ax.set_yscale('log')
        self.ax.get_yaxis().set_major_formatter(
            FuncFormatter(lambda y, _: '{:g}'.format(y)))

    def finish(self):
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)


def main():
    description = ('Obtain the correction factor for a charge extraction '
                   'approach defined in CHECLabPy with respect to the '
                   'CrossCorrelation method')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the runlist.txt file for '
                                            'a dynamic range run')
    parser.add_argument('-r', '--reducer', dest='reducer', action='store',
                        default='AverageWF',
                        choices=WaveformReducerFactory.subclass_names,
                        help='WaveformReducer to use')
    parser.add_argument('-c', '--config', dest='configuration',
                        help="""Configuration to pass to the waveform reducer
                        (Usage: '{"window_shift":6, "window_size":6}') """)
    args = parser.parse_args()
    config = {}
    if args.configuration:
        config = json.loads(args.configuration)

    output_dir = os.path.dirname(args.input_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: {}".format(output_dir))

    df_runs = open_runlist_r1(args.input_path, open_readers=False)
    df_runs = df_runs.loc[(df_runs['pe_expected'] > 50) &
                          (df_runs['pe_expected'] < 100)]
    df_runs['reader'] = [ReaderR1(fp, 100) for fp in df_runs['path'].values]
    n_rows = df_runs.index.size

    first_reader = df_runs.iloc[0]['reader']
    n_pixels = first_reader.n_pixels
    n_samples = first_reader.n_samples
    mapping = first_reader.mapping
    if 'reference_pulse_path' not in config:
        config['reference_pulse_path'] = first_reader.reference_pulse_path
    pixel_aray = np.arange(n_pixels)

    reducer_dict = dict()
    reducers = ['CrossCorrelation', args.reducer]
    print("Investigating reducers: {}".format(reducers))
    print("Reducer Config: {}".format(config))
    for reducer_name in reducers:
        reducer = WaveformReducerFactory.produce(
            reducer_name,
            n_pixels=n_pixels,
            n_samples=n_samples,
            extract_charge_only=True,
            mapping=mapping,
            **config
        )
        reducer_dict[reducer_name] = reducer

    df_list = []

    desc0 = "Looping over runs"
    it = enumerate(df_runs.iterrows())
    for i, (_, row) in tqdm(it, total=n_rows, desc=desc0):
        reader = row['reader']
        attenuation = row['fw_atten']
        baseline_subtractor = BaselineSubtractor(reader)
        n_events = reader.n_events
        desc1 = "Processing events"

        a = np.zeros((2048, n_samples))

        for waveforms in tqdm(reader, total=n_events, desc=desc1):
            iev = reader.index
            waveforms_bs = baseline_subtractor.subtract(waveforms)
            for reducer_name, reducer in reducer_dict.items():
                charge = reducer.process(waveforms_bs)['charge']
                df_list.append(pd.DataFrame(dict(
                    transmission=1/attenuation,
                    iev=iev,
                    pixel=pixel_aray,
                    reducer=reducer_name,
                    charge=charge,
                )))
    df = pd.concat(df_list)

    poi = 1920
    df_p = df.loc[df['pixel'] == poi]

    m_dict = {}
    p_scatter = Scatter("Transmission", "Charge (mV)")
    for reducer_name in reducers:
        df_r = df_p.loc[df_p['reducer'] == reducer_name]
        f = dict(charge=['mean', 'std'])
        df_agg = df_r.groupby('transmission').agg(f)
        x = df_agg.index.values
        y = df_agg['charge']['mean'].values
        yerr = df_agg['charge']['std'].values

        xp = df_r['transmission'].values
        yp = df_r['charge'].values
        c, m = polyfit(x, y, np.arange(1, 2))
        label = reducer_name + " (m={:.3f}, c(fixed)={:.3f})".format(m, c)

        p_scatter.add(x, y, yerr=yerr, m=m, c=c, fmt='o', label=label)

        m_dict[reducer_name] = m

    p_scatter.add_legend(loc='upper left')
    path = os.path.join(output_dir, "charge_corrections.pdf".format(poi))
    p_scatter.save(path)

    ref_reducer = 'CrossCorrelation'
    norm_gradient_dict = {n: float("{:.3f}".format(m_dict[ref_reducer]/v))
                          for n, v in m_dict.items()}
    print(norm_gradient_dict)
    path = os.path.join(output_dir, "charge_corrections.yml")
    with open(path, 'w') as file:
        yaml.safe_dump(norm_gradient_dict, file, default_flow_style=False)


if __name__ == '__main__':
    main()
