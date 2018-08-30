import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from numpy.polynomial.polynomial import polyfit, polyval
from CHECLabPy.utils.files import open_runlist_dl1
from CHECLabPy.plotting.camera import CameraImage
from CHECLabPy.utils.resolutions import ChargeResolution, ChargeStatistics
from matplotlib import pyplot as plt


class Calibrator:
    def __init__(self, calib_path):
        """
        Class to handle the calibration of the true charge and measured
        charge
        """
        store_spe = pd.HDFStore(calib_path)
        df_calib = store_spe['calibration_coeff']
        self.fw_m = df_calib['fw_m'].values
        self.ff_c = df_calib['ff_c'].values
        self.ff_m = df_calib['ff_m'].values

    def calibrate_true(self, pixel, transmission):
        m = self.fw_m[pixel]
        return transmission * m

    def calibrate_measured(self, pixel, mv):
        c = self.ff_c[pixel]
        m = self.ff_m[pixel]
        return (mv - c) / m


def main():
    description = 'Extract the charge resolution from a dynamic range dataset'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the runlist.txt file for '
                                            'a dynamic range run')
    parser.add_argument('-c', '--calib', dest='calib_path', action='store',
                        required=True,
                        help='path to the calibration file that '
                             'is a result of extract_cr_calibration')
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        help='path to store the output HDF5 file '
                             '(OPTIONAL, will be automatically set if '
                             'not specified)')
    args = parser.parse_args()

    input_path = args.input_path
    calib_path = args.calib_path
    output_path = args.output_path

    df_runs = open_runlist_dl1(input_path)
    df_runs['transmission'] = 1/df_runs['fw_atten']
    n_runs = df_runs.index.size

    dead = [677, 293, 27, 1925]

    calibrator = Calibrator(calib_path)
    cr = ChargeResolution()
    cs = ChargeStatistics()

    if not output_path:
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "charge_res.h5")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: {}".format(output_dir))
    if os.path.exists(output_path):
        os.remove(output_path)

    with pd.HDFStore(output_path) as store:
        desc0 = "Looping over files"
        it = enumerate(df_runs.iterrows())
        for i, (_, row) in tqdm(it, total=n_runs, desc=desc0):
            reader = row['reader']
            transmission = row['transmission']
            # for df in reader.iterate_over_chunks():
            df = reader.get_first_n_events(1000)
            df = df.loc[~df['pixel'].isin(dead)]
            pixel = df['pixel'].values
            true = calibrator.calibrate_true(pixel, transmission)
            measured = df['charge'].values
            measured = calibrator.calibrate_measured(pixel, measured)

            # df_bias = pd.DataFrame(dict(pixel=pixel, true=true, measured=measured))
            # df_bias['bias'] = df_bias.groupby(['pixel', 'true']).transform('mean')['measured']
            # df_bias['withoutbias'] = df_bias['measured'] / df_bias['bias'] * df_bias['true']
            # measured = df_bias['withoutbias'].values

            cr.add(pixel, true, measured)
            cs.add(pixel, true, measured)
            reader.store.close()
        df_pixel, df_camera = cr.finish()
        store['charge_resolution_pixel'] = df_pixel
        store['charge_resolution_camera'] = df_camera
        df_pixel, df_camera = cs.finish()
        store['charge_statistics_pixel'] = df_pixel
        store['charge_statistics_camera'] = df_camera

    print("Created charge resolution file: {}".format(output_path))


if __name__ == '__main__':
    main()
