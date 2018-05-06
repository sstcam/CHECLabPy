"""
Executable for extracting the Single-Photoelectron spectrum and fit from dl1
files
"""
import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter as Formatter
from functools import partial
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from tqdm import tqdm, trange
from multiprocessing import Pool, Manager
import warnings
from CHECLabPy.core.io import DL1Reader
from CHECLabPy.core.factory import SpectrumFitterFactory
from CHECLabPy.plotting.spe import SpectrumFitPlotter


class SpectrumFitProcessor:
    def __init__(self, fitter, *readers, dead_pixels=None):
        """
        Processes the spectrum to obtain the fit parameters for each pixel,
        utilising all cpu cores via the multiprocessing package.

        Parameters
        ----------
        fitter : `CHECLabPy.core.spectrum_fitter.SpectrumFitter`
        readers : list[`CHECLabPy.core.io.DL1Reader`]
            The readers for each SPE illumination
        dead_pixels : list
            List of dead pixels to skip
        """
        self.fitter = fitter
        self.dead_pixels = dead_pixels if dead_pixels is not None else []
        self.n_readers = len(readers)
        self.n_pixels = 1#readers[0].n_pixels
        self.pixels = []
        self.charges = []
        desc = "Obtaining charge columns from readers"
        for reader in tqdm(readers, desc=desc):
            pixel, charge = reader.select_columns(['pixel', 'charge'])
            self.pixels.append(pixel.values)
            self.charges.append(charge.values)

        self.x = np.linspace(self.fitter.range[0], self.fitter.range[1], 1000)
        self.f_hist = partial(
            self.fitter.get_histogram_summed,
            bins=self.fitter.nbins,
            range_=self.fitter.range
        )
        self.f_fit = partial(self.fitter.get_fit_summed, x=self.x)

        manager = Manager()
        self.array = manager.dict()
        self.coeff = manager.dict()
        self.initial = manager.dict()

    def apply_pixel(self, pixel):
        """
        Obtain the fit parameters for a pixel

        Parameters
        ----------
        pixel : int
        """
        if pixel in self.dead_pixels:
            return
        pixel_charges = self.get_pixel_charges(pixel)
        self.fitter.apply(*pixel_charges)

        hist, edges, between = self.f_hist(pixel_charges)
        fit = self.f_fit(**self.fitter.coeff)
        initial = self.f_fit(**self.fitter.p0)
        self.array[pixel] = dict(
            pixel=pixel,
            hist=hist,
            edges=edges,
            between=between,
            fit_x=self.x,
            fit=fit,
            initial=initial
        )

        coeff = self.fitter.coeff.copy()
        coeff['pixel'] = pixel
        coeff['chi2'] = self.fitter.chi2
        coeff['rchi2'] = self.fitter.reduced_chi2
        coeff['p_value'] = self.fitter.p_value
        self.coeff[pixel] = coeff

        initial = self.fitter.p0.copy()
        self.fitter.coeff = initial
        initial['pixel'] = pixel
        initial['chi2'] = self.fitter.chi2
        initial['rchi2'] = self.fitter.reduced_chi2
        initial['p_value'] = self.fitter.p_value
        self.initial[pixel] = initial

    def get_pixel_charges(self, pixel):
        """
        Get the charges for every illumination for a single pixel.

        Parameters
        ----------
        pixel : int

        Returns
        -------
        list[ndarray]
        """
        return [self.get_pixel_charge(r, pixel) for r in range(self.n_readers)]

    def get_pixel_charge(self, reader_i, pixel):
        """
        Get the charge values of an illumination for a single pixel.

        Parameters
        ----------
        reader_i : int
        pixel : int

        Returns
        -------
        ndarray
        """
        assert (pixel < self.n_pixels) & (pixel >= 0)
        return self.charges[reader_i][self.pixels[reader_i] == pixel]

    def process(self):
        """
        Traditional process where each pixel is fitted in series
        """
        for i in trange(self.n_pixels):
            self.apply_pixel(i)

    def multiprocess(self):
        """
        Multi-process, where every cpu core is utilised to process pixels in
        parallel for a faster result
        """
        with Pool() as pool:
            pool.map(self.apply_pixel, trange(self.n_pixels))

    def get_df_result(self):
        """
        Extract of the fit results per pixel as DataFrames

        Returns
        -------
        df_coeff : `pd.DataFrame`
            DataFrame containing the fit coefficients for each pixel
        df_initial : `pd.DataFrame`
            DataFrame containing the initial fit coefficients for each pixel
        df_array : `pd.DataFrame`
            DataFrame containing the arrays of the histograms and fits for
            each pixel
        """
        df_coeff = pd.DataFrame(list(self.coeff.values()))
        df_initial = pd.DataFrame(list(self.initial.values()))
        df_array = pd.DataFrame(list(self.array.values()))
        df_coeff = df_coeff.set_index('pixel', drop=False).sort_index()
        df_initial = df_initial.set_index('pixel', drop=False).sort_index()
        df_array = df_array.set_index('pixel', drop=False).sort_index()
        return df_coeff, df_initial, df_array

    def get_df_result_camera(self):
        """
        Calculate and extract the fit results for the entire camera

        Returns
        -------
        df_coeff : `pd.DataFrame`
            DataFrame containing the fit coefficients for all pixels
        df_initial : `pd.DataFrame`
            DataFrame containing the initial fit coefficients for all pixels
        df_array : `pd.DataFrame`
            DataFrame containing the arrays of the histograms and fits for
            all pixels
        """
        self.fitter.apply(*self.charges)
        hist, edges, between = self.f_hist(self.charges)
        fit = self.f_fit(**self.fitter.coeff)
        initial = self.f_fit(**self.fitter.p0)
        array = dict(
            hist=hist,
            edges=edges,
            between=between,
            fit_x=self.x,
            fit=fit,
            initial=initial
        )

        coeff = self.fitter.coeff.copy()
        coeff['chi2'] = self.fitter.chi2
        coeff['rchi2'] = self.fitter.reduced_chi2
        coeff['p_value'] = self.fitter.p_value

        initial = self.fitter.p0.copy()
        self.fitter.coeff = initial
        initial['chi2'] = self.fitter.chi2
        initial['rchi2'] = self.fitter.reduced_chi2
        initial['p_value'] = self.fitter.p_value

        df_coeff = pd.DataFrame(list(coeff.values()))
        df_initial = pd.DataFrame(list(initial.values()))
        df_array = pd.DataFrame(list(array.values()))
        return df_coeff, df_initial, df_array


def main():
    description = ('Extract and fit the Single-Photoelectron spectrum '
                   'from N dl1 files simultaneously')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--files', dest='input_paths', nargs='+',
                        help='path to the input dl1 run files')
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        help='path to store the output HDF5 file '
                             '(OPTIONAL, will be automatically set if '
                             'not specified)')
    parser.add_argument('-s', '--fitter', dest='fitter', action='store',
                        default='GentileFitter',
                        choices=SpectrumFitterFactory.subclass_names,
                        help='SpectrumFitter to use')
    parser.add_argument('-p', '--pixel', dest='plot_pixel', action='store',
                        default=None, type=int,
                        help='Enter plot mode, and plot the spectrum and fit '
                             'for the pixel specified. "-1" speciefies the '
                             'entire camera')
    args = parser.parse_args()

    input_paths = args.input_paths
    output_path = args.output_path
    fitter_str = args.fitter
    plot_pixel = args.plot_pixel

    readers = [DL1Reader(path) for path in input_paths]
    kwargs = dict(product_name=fitter_str, n_illuminations=len(readers))
    fitter = SpectrumFitterFactory.produce(**kwargs)

    fit_processor = SpectrumFitProcessor(fitter, *readers)
    if plot_pixel is not None:
        p_fit = SpectrumFitPlotter()
        if plot_pixel == -1:
            charges = fit_processor.charges
            p_fit.plot_from_fitter(fitter, charges)
        else:
            charges = fit_processor.get_pixel_charges(plot_pixel)
            p_fit.plot_from_fitter(fitter, charges)

        if not output_path:
            name = '_spe_fit_p{}.pdf'.format(plot_pixel)
            if len(input_paths) == 1:
                output_path = input_paths[0].replace('_dl1.h5', name)
            else:
                output_dir = os.path.dirname(input_paths[0])
                output_path = os.path.join(output_dir, name)

        p_fit.save(output_path)
        exit()

    # fit_processor.process()
    fit_processor.multiprocess()

    if not output_path:
        if len(input_paths) == 1:
            output_path = input_paths[0].replace('_dl1.h5', '_spe.h5')
        else:
            output_dir = os.path.dirname(input_paths[0])
            output_path = os.path.join(output_dir, "spe.h5")
    if os.path.exists(output_path):
        os.remove(output_path)

    print("Created HDFStore file: {}".format(output_path))
    with pd.HDFStore(output_path) as store:
        df_coeff, df_initial, df_array = fit_processor.get_df_result()
        store['coeff_pixel'] = df_coeff
        store['initial_pixel'] = df_initial
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PerformanceWarning)
            store['array_pixel'] = df_array

        df_coeff, df_initial, df_array = fit_processor.get_df_result_camera()
        store['coeff_camera'] = df_coeff
        store['initial_camera'] = df_initial
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PerformanceWarning)
            store['array_camera'] = df_array

        metadata = dict(
            files=input_paths,
            fitter=fitter.__class__.__name__,
            n_illuminations=fit_processor.n_readers,
            n_pixels=fit_processor.n_pixels
        )
        store['metadata'] = pd.DataFrame()
        store.get_storer('metadata').attrs.metadata = metadata

        store['mapping'] = readers[0].mapping
        mapping_meta = readers[0].mapping.metadata
        store.get_storer('mapping').attrs.metadata = mapping_meta


if __name__ == '__main__':
    main()
