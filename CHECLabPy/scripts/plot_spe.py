"""
Executable for plotting the extracted fit parameters from extract_spe
"""
import argparse
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from CHECLabPy.core.io import HDF5Reader
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.plotting.spe import SpectrumFitPlotter
from CHECLabPy.plotting.camera import CameraImage


class SPECamera(Plotter):
    def __init__(self, mapping):
        super().__init__()

        self.fig = plt.figure(figsize=(8, 3))
        self.ax_values = self.fig.add_subplot(1, 2, 1)
        self.ax_errors = self.fig.add_subplot(1, 2, 2)
        self.ci_values = CameraImage.from_mapping(mapping, ax=self.ax_values)
        self.ci_errors = CameraImage.from_mapping(mapping, ax=self.ax_errors)
        self.ci_values.add_colorbar("Fit Values", pad=0.1)
        self.ci_errors.add_colorbar("Fit Errors", pad=0.1)

    def set_image(self, values, errors):
        self.ci_values.image = values
        self.ci_errors.image = errors


class SPEHist(Plotter):
    def plot(self, values):
        mean = np.mean(values)
        std = np.std(values)
        label = "Mean = {:.3g}, Stddev = {:.3g}".format(mean, std)
        self.ax.hist(values, bins='auto', label=label)
        self.add_legend()


def main():
    parser = argparse.ArgumentParser(
        description='Plot the contents of the SPE HDF5 file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--file', dest='input_path',
        help='path to the input SPE HDF5 file'
    )
    parser.add_argument(
        '-o', '--output', dest='output_dir',
        help='directory to save plots'
    )
    parser.add_argument(
        '-p', '--pixel', dest='plot_pixel', type=int,
        help='Pixel to plot the spectrum for'
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir
    plot_pixel = args.plot_pixel

    with HDF5Reader(input_path) as reader:
        df_values = reader.read('values')
        df_errors = reader.read('errors')
        df_arrays = reader.read('arrays')
        mapping = reader.get_mapping()
        metadata = reader.get_metadata()

    columns = df_values.columns
    for column in columns:
        p_camera = SPECamera(mapping)
        p_camera.set_image(df_values[column], df_errors[column])
        p_camera.fig.suptitle(column)
        p_camera.save(join(output_dir, f"camera_{column}.pdf"))

        p_hist = SPEHist()
        p_hist.plot(df_values[column])
        p_hist.fig.suptitle(column)
        p_hist.save(join(output_dir, f"hist_{column}.pdf"))

    n_illuminations = metadata['n_illuminations']
    fitter_name = metadata['fitter']
    initial = metadata['initial']
    pixel_values = df_values.loc[plot_pixel].to_dict()
    pixel_errors = df_errors.loc[plot_pixel].to_dict()
    pixel_arrays = df_arrays.loc[plot_pixel]
    p_spectrum_pixel = SpectrumFitPlotter(n_illuminations)
    p_spectrum_pixel.plot(
        pixel_arrays['charge_hist_x'],
        pixel_arrays['charge_hist_y'],
        pixel_arrays['charge_hist_edges'],
        pixel_arrays['fit_x'],
        pixel_arrays['fit_y'],
        pixel_values,
        pixel_errors,
        initial,
    )
    p_spectrum_pixel.fig.suptitle(
        f"{fitter_name}, {n_illuminations} Illuminations, Pixel={plot_pixel}",
        x=0.75
    )
    p_spectrum_pixel.save(join(output_dir, f"spectrum_p{plot_pixel}.pdf"))


if __name__ == '__main__':
    main()
