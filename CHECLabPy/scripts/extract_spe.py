"""
Executable for extracting the Single-Photoelectron spectrum from DL1 files
"""
import argparse
import pandas as pd
from pandas.errors import PerformanceWarning
from tqdm import tqdm, trange
from multiprocessing import Pool, Manager
import warnings
from CHECLabPy.core.io import DL1Reader, HDF5Writer
from CHECLabPy.core.factory import SpectrumFitterFactory
from CHECLabPy.plotting.spe import SpectrumFitPlotter


class PixelFitApplier:
    def __init__(self, fitter):
        self.fitter = fitter

        manager = Manager()
        self.pixel_values = manager.dict()
        self.pixel_errors = manager.dict()
        self.pixel_arrays = manager.dict()

        self.charges = None

    def _apply_pixel(self, pixel):
        n_illuminations = self.fitter.n_illuminations
        pixel_charges = [self.charges[i][:, pixel] for i in range(n_illuminations)]
        self.fitter.apply(*pixel_charges)

        charge_hist_x, charge_hist_y, charge_hist_edges = self.fitter.charge_histogram
        fit_x, fit_y = self.fitter.fit_result_curve

        pixel_values = dict(self.fitter.fit_result_values)
        pixel_values["pixel"] = pixel
        self.pixel_values[pixel] = pixel_values
        pixel_errors = dict(self.fitter.fit_result_errors)
        pixel_errors["pixel"] = pixel
        self.pixel_errors[pixel] = pixel_errors
        self.pixel_arrays[pixel] = dict(
            pixel=pixel,
            charge_hist_x=charge_hist_x,
            charge_hist_y=charge_hist_y,
            charge_hist_edges=charge_hist_edges,
            fit_x=fit_x,
            fit_y=fit_y,
        )

    def multiprocess(self, n_processes, *charges):
        self.charges = charges
        _, n_pixels = charges[0].shape
        print(f"Multiprocessing pixel SPE fit (n_processes = {n_processes})")
        with Pool(n_processes) as pool:
            pool.map(self._apply_pixel, trange(n_pixels))

    def process(self, *charges):
        self.charges = charges
        _, n_pixels = charges[0].shape
        for pixel in trange(n_pixels):
            self._apply_pixel(pixel)


def main():
    parser = argparse.ArgumentParser(
        description='Extract and fit the Single-Photoelectron spectrum '
                    'from N dl1 files simultaneously',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--files', dest='input_paths', nargs='+',
        help='paths to the input dl1 run files'
    )
    parser.add_argument(
        '-o', '--output', dest='output_path', required=True,
        help='path to store the output HDF5 file containing the SPE fit result'
    )
    parser.add_argument(
        '-C', '--charge_col_name', dest='charge_col_name', required=True,
        help='The column name of the charge to be used in the fit.'
    )
    parser.add_argument(
        '-s', '--fitter', dest='fitter', default='SiPMGentileFitter',
        choices=SpectrumFitterFactory.subclass_names, help='SpectrumFitter to use'
    )
    parser.add_argument(
        '-c', '--config', dest='config',
        help='Path to SpectrumFitter configuration YAML file')
    parser.add_argument(
        '-p', '--pixel', dest='plot_pixel', type=int,
        help='Enter plot mode, and plot the spectrum and fit for the pixel'
    )
    parser.add_argument(
        '--n_processes', dest='n_processes', type=int, default=1,
        help='Multi-process each pixel in parallel'
    )

    args = parser.parse_args()

    input_paths = args.input_paths
    output_path = args.output_path
    charge_col_name = args.charge_col_name
    fitter_name = args.fitter
    config_path = args.config
    plot_pixel = args.plot_pixel
    n_processes = args.n_processes

    # Get charges
    charges = []
    mapping = None
    for path in input_paths:
        with DL1Reader(path) as reader:
            charges.append(reader.select_column(charge_col_name).values.reshape(
                (reader.n_events, reader.n_pixels)
            ))
            if mapping is None:
                mapping = reader.mapping
    n_illuminations = len(charges)

    # Create fitter class
    fitter = SpectrumFitterFactory.produce(
        product_name=fitter_name,
        n_illuminations=n_illuminations,
    )
    if config_path is not None:
        fitter.load_config(config_path)
    initial = {param.name: param.initial for param in fitter.parameters}
    lambda_initial = initial.pop("lambda_")
    for i in range(n_illuminations):
        initial[f"lambda_{i}"] = lambda_initial

    # Plot mode
    if plot_pixel is not None:
        pixel_charges = [charges[i][:, plot_pixel] for i in range(n_illuminations)]
        fitter.apply(*pixel_charges)
        p_fit = SpectrumFitPlotter(n_illuminations)
        p_fit.plot(
            *fitter.charge_histogram, *fitter.fit_result_curve,
            fitter.fit_result_values, fitter.fit_result_errors, initial
        )
        p_fit.fig.suptitle(
            f"{fitter_name}, {n_illuminations} Illuminations, Pixel={plot_pixel}",
            x=0.75
        )
        p_fit.show()
        exit()

    pixel_fitter = PixelFitApplier(fitter)
    if n_processes > 1:
        pixel_fitter.multiprocess(n_processes, *charges)
    else:
        pixel_fitter.process(*charges)

    df_values = pd.DataFrame(list(pixel_fitter.pixel_values.values()))
    df_errors = pd.DataFrame(list(pixel_fitter.pixel_errors.values()))
    df_arrays = pd.DataFrame(list(pixel_fitter.pixel_arrays.values()))
    df_values = df_values.set_index('pixel')
    df_errors = df_errors.set_index('pixel')
    df_arrays = df_arrays.set_index('pixel')

    with HDF5Writer(output_path) as writer:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PerformanceWarning)
            writer.write(
                values=df_values,
                errors=df_errors,
                arrays=df_arrays,
            )
        writer.add_mapping(mapping)
        writer.add_metadata(
            files=input_paths,
            fitter=fitter.__class__.__name__,
            n_illuminations=n_illuminations,
            n_pixels=charges[0].shape,
            initial=initial,
        )


if __name__ == '__main__':
    main()
