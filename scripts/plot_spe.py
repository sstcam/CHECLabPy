"""
Executable for plotting the extractd fit parameters from extract_spe
"""
import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.plotting.spe import SpectrumFitPlotter
from CHECLabPy.plotting.camera import CameraImage


class SPECamera:
    def __init__(self, mapping, output_dir):
        self.camera = CameraImage.from_mapping(mapping)
        self.camera.add_colorbar()
        self.output_dir = output_dir

    def produce(self, df):
        pixel = df['pixel'].values
        columns = df.columns
        d = self.output_dir
        for c in columns:
            if c == 'pixel':
                continue
            image = df[c].values[pixel]
            self.camera.image = image
            self.camera.ax.set_title(c)
            output_path = os.path.join(d, "camera_{}.pdf".format(c))
            self.camera.save(output_path)


class SPEHist(Plotter):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def produce(self, df):
        pixel = df['pixel'].values
        columns = df.columns
        d = self.output_dir
        for c in columns:
            self.fig, self.ax = self.create_figure()
            if c == 'pixel':
                continue
            values = df[c].values[pixel]
            mean = np.mean(values)
            std = np.std(values)
            label = "Mean = {:.3g}, Stddev = {:.3g}".format(mean, std)
            try:
                self.ax.hist(values, bins='fd', label=label)
            except ValueError:
                self.ax.hist(values, bins='doane', label=label)
            self.add_legend()
            self.ax.set_title(c)
            output_path = os.path.join(d, "hist_{}.pdf".format(c))
            self.save(output_path)


def main():
    description = 'Plot the contents of the spe HDF5 file'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        help='path to the input spe HDF5 file')
    parser.add_argument('-p', '--pixel', dest='plot_pixel', action='store',
                        default=0, type=int,
                        help='Pixel to plot the spectrum for')
    args = parser.parse_args()

    input_path = args.input_path
    plot_pixel = args.plot_pixel

    output_dir = os.path.join(os.path.splitext(input_path)[0], "spe")

    store = pd.HDFStore(input_path)
    df_pixel_coeff = store['coeff_pixel']
    df_pixel_initial = store['initial_pixel']
    df_pixel_array = store['array_pixel']
    df_camera_coeff = store['coeff_camera']
    df_camera_initial = store['initial_camera']
    df_camera_array = store['array_camera']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mapping = store['mapping']
        mapping.metadata = store.get_storer('mapping').attrs.metadata

    metadata = store.get_storer('metadata').attrs.metadata

    p_camera = SPECamera(mapping, output_dir)
    p_camera.produce(df_pixel_coeff)
    plt.close("all")

    p_hist = SPEHist(output_dir)
    p_hist.produce(df_pixel_coeff)
    plt.close("all")

    p_spectrum_pixel = SpectrumFitPlotter()
    p_spectrum_pixel.plot_from_df_pixel(df_pixel_coeff, df_pixel_initial,
                                        df_pixel_array, metadata, plot_pixel)
    p_spectrum_pixel.save(os.path.join(output_dir, "spectrum_pixel.pdf"))

    p_spectrum_camera = SpectrumFitPlotter()
    p_spectrum_camera.plot_from_df_camera(df_camera_coeff, df_camera_initial,
                                          df_camera_array, metadata)
    p_spectrum_camera.save(os.path.join(output_dir, "spectrum_camera.pdf"))


if __name__ == '__main__':
    main()
