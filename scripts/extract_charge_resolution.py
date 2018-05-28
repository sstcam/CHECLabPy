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


class SPEHandler:
    def __init__(self, df_runs, spe_path):
        """
        Class to handle the calibration of the true charge and measured
        charge from the SPE spectrum
        """
        store_spe = pd.HDFStore(spe_path)
        df_spe = store_spe['coeff_pixel']

        self.baseline = df_spe['eped'].values
        self.mvperpe = df_spe['spe'].values

        meta_spe = store_spe.get_storer('metadata').attrs.metadata
        n_spe_illuminations = meta_spe['n_illuminations']
        spe_files = meta_spe['files']
        n_pixels = meta_spe['n_pixels']

        spe_transmission = []
        pattern = '(.+?)/Run(.+?)_dl1.h5'
        for path in spe_files:
            try:
                reg_exp = re.search(pattern, path)
                if reg_exp:
                    run = int(reg_exp.group(2))
                    spe_transmission.append(df_runs.loc[run]['transmission'])
            except AttributeError:
                print("Problem with Regular Expression, "
                      "{} does not match patten {}".format(path, pattern))
        np.array(spe_transmission)

        pix_lambda = np.zeros((n_spe_illuminations, n_pixels))
        for ill in range(n_spe_illuminations):
            key = "lambda_" + str(ill)
            lambda_ = df_spe[['pixel', key]].sort_values('pixel')[key].values
            pix_lambda[ill] = lambda_

        self.c, self.m = polyfit(spe_transmission, pix_lambda, 1)

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(spe_transmission, pix_lambda, 'x')
        # ax.plot(spe_transmission,
        #         polyval(spe_transmission, (self.c, self.m)).T)
        #
        # dead = [677, 293, 27, 1925]
        # mask = np.zeros(n_pixels, dtype=np.bool)
        # mask[dead] = True
        # mm = np.ma.masked_array(self.m, mask=mask)
        # cm = np.ma.masked_array(self.c, mask=mask)
        # im_m = CameraImage.from_camera_version("1.1.0")
        # im_c = CameraImage.from_camera_version("1.1.0")
        # im_m.image = mm
        # im_c.image = cm
        # im_m.add_colorbar()
        # im_c.add_colorbar()
        # plt.show()

    def calibrate_true(self, pixel, transmission):
        m = self.m[pixel]
        c = self.c[pixel]
        return transmission * m + c

    def calibrate_measured(self, pixel, mv):
        # TODO: include offset?
        return (mv - self.baseline[pixel]) / self.mvperpe[pixel]


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
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        help='path to store the output HDF5 file '
                             '(OPTIONAL, will be automatically set if '
                             'not specified)')
    args = parser.parse_args()

    input_path = args.input_path
    spe_path = args.spe_path
    output_path = args.output_path

    df_runs = open_runlist_dl1(input_path)
    df_runs['transmission'] = 1/df_runs['fw_atten']
    n_runs = df_runs.index.size

    dead = [677, 293, 27, 1925]

    spe_handler = SPEHandler(df_runs, spe_path)
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
            true = spe_handler.calibrate_true(pixel, transmission)
            measured = df['charge'].values
            measured = spe_handler.calibrate_measured(pixel, measured)
            cr.add(pixel, true, measured)
            cs.add(pixel, true, measured)
            reader.store.close()
        df_pixel, df_camera = cr.finish()
        store['charge_resolution_pixel'] = df_pixel
        store['charge_resolution_camera'] = df_camera
        df_pixel, df_camera = cs.finish()
        store['charge_statistics_pixel'] = df_pixel
        store['charge_statistics_camera'] = df_camera


if __name__ == '__main__':
    main()
