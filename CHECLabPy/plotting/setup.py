import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import numpy as np
from CHECLabPy.utils.files import create_directory


class Plotter:
    def __init__(self, ax=None, sidebyside=False, talk=False):
        """
        Base class for plotting classes to define common appearance

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Optionally place the plot on a pre-existing axes
        sidebyside : bool
            Resize figure so that it can be placed side-by-side
        talk : bool
            Configure appearance to be appropriate for a presentation
        """
        self.sidebyside = sidebyside

        rc = {  # setup matplotlib to use latex for output
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6,
            "axes.prop_cycle": plt.cycler(color=plt.cm.Dark2.colors),

            # Set x axis
            "xtick.labelsize": 8,
            "xtick.direction": 'in',
            "xtick.major.size": 3,
            "xtick.major.width": 0.5,
            "xtick.minor.size": 1.5,
            "xtick.minor.width": 0.5,
            "xtick.minor.visible": True,
            "xtick.top": True,

            # Set y axis
            "ytick.labelsize": 8,
            "ytick.direction": 'in',
            "ytick.major.size": 3,
            "ytick.major.width": 0.5,
            "ytick.minor.size": 1.5,
            "ytick.minor.width": 0.5,
            "ytick.minor.visible": True,
            "ytick.right": True,

            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.,

            "savefig.bbox": 'tight',
            "savefig.pad_inches": 0.05,

            "figure.figsize": self.get_figsize(),
            "lines.markeredgewidth": 1,
        }

        if talk:
            talk_rc = {
                "font.family": "Latin Modern Roman",
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

        if ax:
            self.ax = ax
            self.fig = ax.figure
        else:
            self.fig = self.create_figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    @staticmethod
    def golden_figsize(scale=0.9):
        fig_width_pt = 421.10046  # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0 / 72.27  # Convert pt to inch
        golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
        fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
        fig_height = fig_width * golden_mean  # height in inches
        fig_size = [fig_width, fig_height]
        return fig_size

    def get_figsize(self):
        if self.sidebyside:
            return self.golden_figsize(0.6)
        else:
            return self.golden_figsize(0.9)

    def create_figure(self):
        fig = plt.figure(figsize=self.get_figsize())
        return fig

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
        self.fig.savefig(output_path, bbox_inches='tight')
        print("Figure saved to: {}".format(output_path))
        self.close()

    def show(self, *args, **kwargs):
        plt.show(*args, **kwargs)

    def close(self):
        plt.close(self.fig)
