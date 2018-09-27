import matplotlib as mpl
# mpl.use('pgf')
from matplotlib import pyplot as plt
import os
import numpy as np
from CHECLabPy.utils.files import create_directory


class Plotter:
    def __init__(self, ax=None, talk=False):
        """
        Base class for plotting classes to define common appearance

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Optionally place the plot on a pre-existing axes
        talk : bool
            Configure appearance to be appropriate for a presentation
        """
        # sns.set_style("white")
        # sns.set_style("ticks")
        rc = {
            "font.family": [u"Helvetica"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "lines.markeredgewidth": 1
        }
        if talk:
            talk_rc = {
                "font.family": [u"Helvetica"],
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "font.size": 12,
                "axes.titlesize": 12,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
                "lines.markeredgewidth": 1
            }
            rc = {**rc, **talk_rc}
        mpl.rcParams.update(rc)
        # sns.set_context("talk", rc=rc)

        if ax:
            self.ax = ax
            self.fig = ax.figure
        else:
            self.fig, self.ax = self.create_figure()

    @staticmethod
    def figsize(scale=0.9):
        fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0 / 72.27  # Convert pt to inch
        golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
        fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
        fig_height = fig_width * golden_mean  # height in inches
        fig_size = [fig_width, fig_height]
        return fig_size

    def create_figure(self):
        fig = plt.figure(figsize=self.figsize())
        ax = fig.add_subplot(1, 1, 1)

        # fmt = mpl.ticker.StrMethodFormatter("{x}")
        # ax.xaxis.set_major_formatter(fmt)
        # ax.yaxis.set_major_formatter(fmt)
        return fig, ax

    def add_legend(self, loc="upper right", **kwargs):
        self.ax.legend(loc=loc, **kwargs)

    @staticmethod
    def create_directory(directory):
        create_directory(directory)

    def finish(self):
        pass

    def save(self, output_path):
        self.finish()
        output_dir = os.path.dirname(output_path)
        self.create_directory(output_dir)
        # self.fig.tight_layout()
        self.fig.savefig(output_path, bbox_inches='tight')
        print("Figure saved to: {}".format(output_path))
