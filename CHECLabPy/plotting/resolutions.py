import numpy as np
from scipy.stats import binned_statistic as bs
from matplotlib.ticker import FuncFormatter
from CHECLabPy.plotting.setup import Plotter
from IPython import embed


def sum_errors(array):
    return np.sqrt(np.sum(np.power(array, 2))) / array.size


def bin_points(x, y, yerr):
    bins = np.geomspace(0.1, x.max(), 100)
    x_b = bs(x, x, 'mean', bins=bins)[0]
    y_b = bs(x, y, 'mean', bins=bins)[0]
    yerr_b = bs(x, yerr, sum_errors, bins=bins)[0]
    valid = ~np.isnan(x_b)
    x_b = x_b[valid]
    y_b = y_b[valid]
    yerr_b = yerr_b[valid]
    return x_b, y_b, yerr_b


class ChargeResolutionPlotter(Plotter):
    def __init__(self):
        super().__init__()
        self.df_pixel = None
        self.df_camera = None

    def set_store(self, store):
        self.df_pixel = store['charge_resolution_pixel']
        self.df_camera = store['charge_resolution_camera']
        self.df_pixel = self.df_pixel.loc[self.df_pixel['true'] > 0.001]
        self.df_camera = self.df_camera.loc[self.df_camera['true'] > 0.001]

    def _plot(self, x, y, yerr, label=''):
        color = self.ax._get_lines.get_next_color()
        (_, caps, _) = self.ax.errorbar(
            x, y, yerr=yerr, mew=1, capsize=3, elinewidth=0.7,
            markersize=3, color=color, label=label
        )
        for cap in caps:
            cap.set_markeredgewidth(0.7)

    def plot_pixel(self, pixel, label=''):
        df_pixel = self.df_pixel.loc[self.df_pixel['pixel'] == pixel]
        x = df_pixel['true']
        y = df_pixel['rmse']
        yerr = 1 / np.sqrt(x)

        x, y, yerr = bin_points(x, y, yerr)

        self._plot(x, y, yerr, label)

    def plot_camera(self, label=''):
        x = self.df_camera['true']
        y = self.df_camera['rmse']
        yerr = 1 / np.sqrt(x)

        x, y, yerr = bin_points(x, y, yerr)

        self._plot(x, y, yerr, label)

    def finish(self):
        self.ax.set_xlabel("True Charge (p.e.)")
        self.ax.set_ylabel("Charge Resolution / True")
        self.ax.set_xscale('log')
        self.ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, _: '{:g}'.format(x)))
        self.ax.set_yscale('log')
        self.ax.get_yaxis().set_major_formatter(
            FuncFormatter(lambda y, _: '{:g}'.format(y)))

    @staticmethod
    def limit_curves(npe, n_nsb, n_add, enf, sigma2):
        """
        Equation for calculating the Goal and Requirement curves, as defined
        in SCI-MC/121113.
        https://portal.cta-observatory.org/recordscentre/Records/SCI/
        SCI-MC/measurment_errors_system_performance_1YQCBC.pdf

        Parameters
        ----------
        npe : ndarray
            Number of photoeletrons (variable).
        n_nsb : float
            Number of NSB photons.
        n_add : float
            Number of photoelectrons from additional noise sources.
        enf : float
            Excess noise factor.
        sigma2 : float
            Percentage ofmultiplicative errors.
        """
        return (np.sqrt((n_nsb + n_add) + np.power(enf, 2) * npe +
                        np.power(sigma2 * npe, 2)) / npe).astype(float)

    @staticmethod
    def requirement(true):
        """
        CTA requirement curve.

        Parameters
        ----------
        true : ndarray
            Number of photoeletrons
        """
        n_nsb = np.sqrt(4.0 + 3.0)
        n_add = 0
        enf = 1.2
        sigma2 = 0.1
        defined_npe = 1000

        lc = ChargeResolutionPlotter.limit_curves
        requirement = lc(true, n_nsb, n_add, enf, sigma2)
        requirement[true > defined_npe] = np.nan

        return requirement

    @staticmethod
    def goal(true):
        """
        CTA goal curve.

        Parameters
        ----------
        true : ndarray
            Number of photoeletrons
        """
        n_nsb = 2
        n_add = 0
        enf = 1.1152
        sigma2 = 0.05
        defined_npe = 2000

        lc = ChargeResolutionPlotter.limit_curves
        goal = lc(true, n_nsb, n_add, enf, sigma2)
        goal[true > defined_npe] = np.nan

        return goal

    @staticmethod
    def poisson(true):
        """
        Poisson limit curve.

        Parameters
        ----------
        true : ndarray
            Number of photoeletrons
        """
        poisson = np.sqrt(true) / true
        return poisson

    def plot_requirement(self, true):
        requirement = self.requirement(true)
        self.ax.plot(true, requirement, '--', color='red', label="Requirement")

    def plot_goal(self, true):
        goal = self.goal(true)
        self.ax.plot(true, goal, '--', color='green', label="Goal")

    def plot_poisson(self, true):
        poisson = self.poisson(true)
        self.ax.plot(true, poisson, '--', color='grey', label="Poisson")



class ChargeResolutionWRRPlotter(ChargeResolutionPlotter):
    def _plot(self, x, y, yerr, label=''):
        y = y / self.requirement(x)
        super()._plot(x, y, yerr, label)

    def plot_requirement(self, true):
        requirement = self.requirement(true)
        requirement /= self.requirement(true)
        self.ax.plot(true, requirement, '--', color='red', label="Requirement")

    def plot_goal(self, true):
        goal = self.goal(true)
        goal /= self.requirement(true)
        self.ax.plot(true, goal, '--', color='green', label="Goal")

    def plot_poisson(self, true):
        poisson = self.poisson(true)
        poisson /= self.requirement(true)
        self.ax.plot(true, poisson, '--', color='grey', label="Poisson")

    def finish(self):
        self.ax.set_xlabel("True Charge (p.e.)")
        self.ax.set_ylabel("(Charge Resolution / True) / Requirement")
        self.ax.set_xscale('log')
        self.ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, _: '{:g}'.format(x)))


class ChargeMeanPlotter(Plotter):
    def __init__(self):
        super().__init__()
        self.df_pixel = None
        self.df_camera = None
        self.x_min = None
        self.x_max = None

    def set_store(self, store):
        self.df_pixel = store['charge_statistics_pixel']
        self.df_camera = store['charge_statistics_camera']

    def _plot(self, x, y, yerr, label=''):
        y /= x
        yerr /= x
        yerr = None
        color = self.ax._get_lines.get_next_color()

        (_, caps, _) = self.ax.errorbar(
            x, y, yerr=yerr, mew=1, capsize=3, elinewidth=0.7,
            markersize=3, color=color, label=label
        )
        for cap in caps:
            cap.set_markeredgewidth(0.7)

        if self.x_min is None or self.x_min > x.min():
            self.x_min = x.min()
        if self.x_max is None or self.x_max < x.max():
            self.x_max = x.max()

    def plot_pixel(self, pixel, label=''):
        df_pixel = self.df_pixel.loc[self.df_pixel['pixel'] == pixel]
        x = df_pixel['amplitude']
        y = df_pixel['mean']
        yerr = df_pixel['std']
        x, y, yerr = bin_points(x, y, yerr)
        self._plot(x, y, yerr, label)

    def plot_camera(self, label=''):
        x = self.df_camera['amplitude']
        y = self.df_camera['mean']
        yerr = self.df_camera['std']
        x, y, yerr = bin_points(x, y, yerr)
        self._plot(x, y, yerr, label)


    def finish(self):
        p = np.linspace(self.x_min, self.x_max, 1000)
        self.ax.plot(p, p/p, '--', color='grey')

        self.ax.set_xlabel("True Charge (p.e.)")
        self.ax.set_ylabel("Charge Average / True")
        self.ax.set_xscale('log')
        self.ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, _: '{:g}'.format(x)))
        # self.ax.set_yscale('log')
        # self.ax.get_yaxis().set_major_formatter(
        #     FuncFormatter(lambda y, _: '{:g}'.format(y)))
