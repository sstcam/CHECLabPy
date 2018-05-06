import numpy as np
from matplotlib import pyplot as plt
from CHECLabPy.plotting.setup import Plotter


class SpectrumFitPlotter(Plotter):
    def __init__(self, talk=False):
        super().__init__(talk=talk)
        self.fig = plt.figure(figsize=(13, 6))
        self.ax = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
        self.ax_t = plt.subplot2grid((3, 2), (0, 1), rowspan=3)

    def plot(self, hist, edges, between, fit_x, fit, initial, coeff,
             coeff_initial, fitter_name, n_illuminations):

        self.ax.hist(between, bins=edges, weights=hist, histtype='step',
                     label="Hist")
        self.ax.plot(fit_x, fit, label="Fit")
        self.ax.plot(fit_x, initial, label="Initial")
        self.ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
        self.ax_t.axis('off')
        columns = ['Initial', 'Fit']
        rows = list(coeff.keys())
        cells = [['%.3g' % coeff_initial[i], '%.3g' % coeff[i]] for i in rows]
        table = self.ax_t.table(cellText=cells, rowLabels=rows,
                                colLabels=columns, loc='center')
        table.set_fontsize(6)
        title = "{}, {} Illuminations".format(fitter_name, n_illuminations)
        self.fig.suptitle(title, x=0.75)

    def plot_from_fitter(self, fitter, charges):
        fitter.apply(*charges)
        fitter_name = fitter.__class__.__name__
        n_illuminations = fitter.n_illuminations
        x = np.linspace(fitter.range[0], fitter.range[1], 1000)
        hist, edges, between = fitter.get_histogram_summed(
            charges, fitter.nbins, fitter.range
        )
        coeff = fitter.coeff.copy()
        coeffl = fitter.coeff_names.copy()
        coeff_initial = fitter.p0.copy()
        fit = fitter.get_fit_summed(x, **coeff)
        initial = fitter.get_fit_summed(x, **coeff_initial)

        coeff['chi2'] = fitter.chi2
        coeff['rchi2'] = fitter.reduced_chi2
        coeff['p_value'] = fitter.p_value
        fitter.coeff = coeff_initial
        coeff_initial['chi2'] = fitter.chi2
        coeff_initial['rchi2'] = fitter.reduced_chi2
        coeff_initial['p_value'] = fitter.p_value

        self.plot(hist, edges, between, x, fit, initial, coeff, coeff_initial,
                  fitter_name, n_illuminations)
