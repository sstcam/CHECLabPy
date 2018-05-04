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
from tqdm import trange
from multiprocessing import Pool, Manager
from CHECLabPy.core.io import DL1Reader
from CHECLabPy.core.factory import SpectrumFitterFactory


class SpectrumFitProcessor:
    def __init__(self, fitter, *readers):
        """
        Processes the spectrum to obtain the fit parameters for each pixel,
        utilising all cpu cores via the multiprocessing package.

        Parameters
        ----------
        fitter : `CHECLabPy.core.spectrum_fitter.SpectrumFitter`
        readers : list[`CHECLabPy.core.io.DL1Reader`]
            The readers for each SPE illumination
        """
        self.fitter = fitter
        self.n_readers = len(readers)
        self.n_pixels = readers[0].n_pixels
        self.pixels = []
        self.charges = []
        for reader in readers:
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
        initial['pixel'] = pixel
        initial['chi2'] = np.nan
        initial['rchi2'] = np.nan
        initial['p_value'] = np.nan
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
        initial['chi2'] = np.nan
        initial['rchi2'] = np.nan
        initial['p_value'] = np.nan

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
    args = parser.parse_args()

    input_paths = args.input_paths
    output_path = args.output_path
    fitter_str = args.fitter

    readers = [DL1Reader(path) for path in input_paths]
    kwargs = dict(product_name=fitter_str, n_illuminations=len(readers))
    fitter = SpectrumFitterFactory.produce(**kwargs)

    fit_processor = SpectrumFitProcessor(fitter, *readers)
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
        store['array_pixel'] = df_array

        df_coeff, df_initial, df_array = fit_processor.get_df_result_camera()
        store['coeff_camera'] = df_coeff
        store['initial_camera'] = df_initial
        store['array_camera'] = df_array

        # TODO: metadata & mapping

    # embed()

    # pixel, charge = reader.select_columns(['pixel', 'charge'])
    # # df = pd.DataFrame(dict(pixel=pixel, charge=charge))
    #
    # groupby = df.groupby('pixel')
    # n_pixels = reader.n_pixels
    # desc = "Fitting pixels"
    # # for pixel, df_group in tqdm(groupby, total=n_pixels, desc=desc):
    # #     pixel_charge = df_group['charge'].values
    # #     fitter.apply(pixel_charge)
    #
    # for p in trange(n_pixels, desc=desc):
    #     pixel_charge = charge[pixel == p]
    #     fitter.apply(pixel_charge)
    #     # d = fitter.coeff.copy()
    #     # d['pixel'] = d



    # plt.plot(fitter.between, fitter.hist[0])
    # plt.plot(fitter.fit_x, fitter.fit[0])
    # fitter.coeff = fitter.initial
    # plt.plot(fitter.fit_x, fitter.fit[0])
    # plt.show()
    #
    # fit_processor(897)
    # fitter = fit_processor.fitter
    # fig_fit = plt.figure(figsize=(13, 6))
    # fig_fit.suptitle("Single Fit")
    # ax = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    # ax_t = plt.subplot2grid((3, 2), (0, 1), rowspan=3)
    # x = np.linspace(fitter.range[0], fitter.range[1], 1000)
    # hist = fitter.hist_summed
    # edges = fitter.edges
    # between = fitter.between
    # coeff = fitter.coeff.copy()
    # coeffl = fitter.coeff_names.copy()
    # initial = fitter.p0.copy()
    # fit = fitter.get_fit_summed(x, **coeff)
    # init = fitter.get_fit_summed(x, **initial)
    # rc2 = fitter.reduced_chi2
    # pval = fitter.p_value
    # ax.hist(between, bins=edges, weights=hist, histtype='step', label="Hist")
    # ax.plot(x, fit, label="Fit")
    # ax.plot(x, init, label="Initial")
    # ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
    # ax_t.axis('off')
    # td = [[initial[i], '%.3f' % coeff[i]] for i in coeffl]
    # td.append(["", '%.3g' % rc2])
    # td.append(["", '%.3g' % pval])
    # tr = coeffl
    # tr.append("Reduced Chi^2")
    # tr.append("P-Value")
    # tc = ['Initial', 'Fit']
    # table = ax_t.table(cellText=td, rowLabels=tr, colLabels=tc, loc='center')
    # table.set_fontsize(6)
    # plt.show()

        # t_cpu = 0
        # start_time = 0
        # desc = "Processing events"
        # for waveforms in tqdm(reader, total=n_events, desc=desc):
        #     iev = reader.index
        #
        #     t_tack = reader.current_tack
        #     t_cpu_sec = reader.current_cpu_s
        #     t_cpu_ns = reader.current_cpu_ns
        #     t_cpu = pd.to_datetime(
        #         np.int64(t_cpu_sec * 1E9) + np.int64(t_cpu_ns),
        #         unit='ns'
        #     )
        #     fci = reader.first_cell_ids
        #
        #     if not start_time:
        #         start_time = t_cpu
        #
        #     waveforms_bs = baseline_subtractor.subtract(waveforms)
        #     bs = baseline_subtractor.baseline
        #
        #     params = reducer.process(waveforms_bs)
        #
        #     df_ev = pd.DataFrame(dict(
        #         iev=iev,
        #         pixel=pixel_array,
        #         first_cell_id=fci,
        #         t_cpu=t_cpu,
        #         t_tack=t_tack,
        #         baseline_subtracted=bs,
        #         **params
        #     ))
        #     writer.append_event(df_ev)
        #
        # sn_dict = {}
        # for tm in range(n_modules):
        #     sn_dict['TM{:02d}_SN'.format(tm)] = reader.get_sn(tm)
        #
        # metadata = dict(
        #     source="CHECLabPy",
        #     date_generated=pd.datetime.now(),
        #     input_path=input_path,
        #     n_events=n_events,
        #     n_modules=n_modules,
        #     n_pixels=n_pixels,
        #     n_samples=n_samples,
        #     n_cells=n_cells,
        #     start_time=start_time,
        #     end_time=t_cpu,
        #     camera_version=camera_version,
        #     reducer=reducer.__class__.__name__,
        #     configuration=config_string,
        #     **sn_dict
        # )
        #
        # writer.add_metadata(**metadata)
        # writer.add_mapping(mapping)

if __name__ == '__main__':
    main()
