import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from numpy.polynomial.polynomial import polyfit, polyval

from CHECLabPy.core.io import DL1Reader
from CHECLabPy.utils.files import open_runlist_dl1
from CHECLabPy.plotting.camera import CameraImage
from CHECLabPy.utils.resolutions import ChargeResolution, ChargeStatistics
from matplotlib import pyplot as plt
from CHECLabPy.plotting.setup import Plotter
from IPython import embed


class SPEHandler:
    def __init__(self, df_runs, spe_path, illumination_path, dead):
        """
        Class to handle the calibration of the true charge and measured
        charge from the SPE spectrum
        """
        store_spe = pd.HDFStore(spe_path)
        df_spe = store_spe['coeff_pixel']

        meta_spe = store_spe.get_storer('metadata').attrs.metadata
        n_spe_illuminations = meta_spe['n_illuminations']
        spe_files = meta_spe['files']
        n_pixels = meta_spe['n_pixels']
        mapping = store_spe['mapping']

        dead_mask = np.zeros(n_pixels, dtype=np.bool)
        dead_mask[dead] = True

        self.spe_transmission = []
        pattern = '(.+?)/Run(.+?)_dl1.h5'
        for path in spe_files:
            try:
                reg_exp = re.search(pattern, path)
                if reg_exp:
                    run = int(reg_exp.group(2))
                    self.spe_transmission.append(df_runs.loc[run]['transmission'])
            except AttributeError:
                print("Problem with Regular Expression, "
                      "{} does not match patten {}".format(path, pattern))

        self.pix_lambda = np.zeros((n_spe_illuminations, n_pixels))
        for ill in range(n_spe_illuminations):
            key = "lambda_" + str(ill)
            lambda_ = df_spe[['pixel', key]].sort_values('pixel')[key].values
            self.pix_lambda[ill] = lambda_

        self.c, self.m = polyfit(self.spe_transmission, self.pix_lambda, 1)

        illumination_corrections = np.loadtxt(illumination_path)
        self.c_corr = self.c / illumination_corrections
        self.m_corr = self.m / illumination_corrections
        self.c_avg = np.mean(self.c_corr[~dead_mask])
        self.m_avg = np.mean(self.m_corr[~dead_mask])
        self.c_pix = self.c_avg * illumination_corrections
        self.m_pix = self.m_avg * illumination_corrections

        # p = self.SPEHandlerPlotter(self)
        # p.plot_cm()
        # p.save("spehandler.pdf")
        # exit()

    def calibrate_true(self, pixel, transmission):
        m = self.m_pix[pixel]
        return transmission * m

    class SPEHandlerPlotter(Plotter):
        def __init__(self, handler):
            super().__init__()
            self.handler = handler

        def plot_cm(self):
            x = self.handler.spe_transmission
            y = self.handler.pix_lambda
            c = self.handler.c_pix
            m = self.handler.m_pix

            c[:] = 0

            # self.ax.plot(x, y, 'x')
            self.ax.plot(x, polyval(x, (c, m)).T)

            c = self.handler.c_avg
            m = self.handler.m_avg

            self.ax.set_xlabel("Transmission")
            self.ax.set_ylabel("Flat-fielded True Illumination (p.e.)")


def main():
    description = 'Extract the charge resolution from a dynamic range dataset'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the runlist.txt file for '
                                            'a dynamic range run')
    parser.add_argument('-s', '--spe', dest='spe_path', action='store',
                        required=True, help='path to the spe file to use for '
                                            'the calibration of the measured '
                                            'and true charges')
    parser.add_argument('-i', '--illcorr', dest='illumination_path',
                        action='store', required=True,
                        help='path to the txt file containing the '
                             'illumination correction for each pixel')
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        help='path to store the output HDF5 file '
                             '(OPTIONAL, will be automatically set if '
                             'not specified)')
    args = parser.parse_args()

    input_path = args.input_path
    spe_path = args.spe_path
    illumination_path = args.illumination_path
    output_path = args.output_path

    if not output_path:
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "calibration_coeff.h5")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: {}".format(output_dir))
    if os.path.exists(output_path):
        os.remove(output_path)

    df_runs = open_runlist_dl1(input_path, False)
    df_runs['transmission'] = 1/df_runs['fw_atten']
    n_runs = df_runs.index.size

    dead = [677, 293, 27, 1925]

    spe_handler = SPEHandler(df_runs, spe_path, illumination_path, dead)

    illumination = df_runs['transmission'] * spe_handler.m_avg
    df_runs_ff = df_runs.loc[(illumination >= 40) & (illumination <= 90)].copy()
    df_runs_ff['reader'] = [DL1Reader(fp) for fp in df_runs_ff['path'].values]
    n_runs = df_runs_ff.index.size

    cs = ChargeStatistics()

    desc0 = "Looping over files"
    it = enumerate(df_runs_ff.iterrows())
    for i, (_, row) in tqdm(it, total=n_runs, desc=desc0):
        reader = row['reader']
        transmission = row['transmission']
        df = reader.get_first_n_events(1000)
        pixel = df['pixel'].values
        true = spe_handler.calibrate_true(pixel, transmission)
        measured = df['charge'].values
        cs.add(pixel, true, measured)
        reader.store.close()
    df_pixel, df_camera = cs.finish()

    d_list = []
    for pix in np.unique(df_pixel['pixel']):
        df = df_pixel.loc[df_pixel['pixel'] == pix]
        true = df['amplitude']
        measured = df['mean']
        measured_std = df['std']
        ff_c, ff_m = polyfit(true, measured, 1, w=1 / measured_std)
        fw_m = spe_handler.m_pix[pix]
        d_list.append(dict(
            pixel=pix,
            fw_m=fw_m,
            ff_c=ff_c,
            ff_m=ff_m
        ))
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(true, measured, 'x')
        # ax.plot(true, polyval(true, (ff_c, ff_m)).T)
        # ax.set_xlabel("True Charge (p.e.)")
        # ax.set_ylabel("Measured Charge (mV)")
        # plt.show()

    df_calib = pd.DataFrame(d_list)
    df_calib = df_calib.sort_values('pixel')

    with pd.HDFStore(output_path) as store:
        store['calibration_coeff'] = df_calib

    print("Created calibration coefficient file: {}".format(output_path))


if __name__ == '__main__':
    main()
