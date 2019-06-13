"""
Plot waveform
"""
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.utils.mapping import get_tm_mapping
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm


class WaveformPlotter(Plotter):
    def __init__(self, title="", units="", tunits="ns", ax=None, talk=False):
        """
        Create a plotter for a waveform

        Parameters
        ----------
        title : str
        units : str
            Y axis units
        tunits : str
            X axis (time) units
        ax : `matplotlib.axes.Axes`
            Optionally place the plot on a pre-existing axes
        talk : bool
            Configure appearance to be appropriate for a presentation
        """
        super().__init__(ax=ax, talk=talk)
        self.title = title
        self.units = units
        self.tunits = tunits

    def add(self, x, data, label=""):
        if x is None:
            self.ax.plot(data, label=label)
        else:
            self.ax.plot(x, data, label=label)

    def finish(self):
        self.ax.set_title(self.title)
        x_label = "Time"
        if self.tunits:
            x_label += " ({})".format(self.tunits)
        self.ax.set_xlabel(x_label)
        y_label = "Amplitude"
        if self.units:
            y_label += " ({})".format(self.units)
        self.ax.set_ylabel(y_label)
        self.ax.xaxis.set_major_locator(MultipleLocator(16))


class CameraTMWaveformPlotter(Plotter):
    def __init__(self, mapping, title):
        super().__init__()
        self.tm_mapping = get_tm_mapping(mapping)
        n_rows = self.tm_mapping.metadata['n_rows']
        n_cols = self.tm_mapping.metadata['n_columns']

        self.fig = plt.figure(figsize=(8.27, 11.69 - 3.5))
        self.fig.subplots_adjust(hspace=0.0, wspace=0.0)
        self.fig.suptitle(title, fontsize=14)

        self.ax_dict = dict()
        self.asic_color = ['firebrick', 'dodgerblue', 'darkgreen', 'orchid']

        self.fig.text(
            0.06, 0.5, "mV", rotation=90, fontsize=10,
            transform=self.fig.transFigure
        )
        self.fig.text(
            0.5, 0.06, "Sample", fontsize=10,
            transform=self.fig.transFigure
        )
        self.fig.text(
            0.25 + 4 * 0.07, 0.905, "Cam-Mean(-)", fontsize=10,
            transform=self.fig.transFigure
        )
        self.fig.text(
            0.30 + 5 * 0.07, 0.905, "TM-Mean(--)", fontsize=10,
            transform=self.fig.transFigure
        )
        for icol, col in enumerate(self.asic_color):
            self.fig.text(
                0.25 + icol * 0.07, 0.905, "ASIC %i" % icol, color=col,
                fontsize=10, transform=self.fig.transFigure
            )

        for _, df_row in self.tm_mapping.iterrows():
            tm = int(df_row['slot'])
            row = int(df_row['row'])
            col = int(df_row['col'])

            ax_col = col
            ax_row = n_rows - 1 - row

            ax = plt.subplot2grid(
                (n_rows, n_cols), (ax_row, ax_col), fig=self.fig
            )
            ax.xaxis.set_minor_locator(AutoMinorLocator(8))
            ax.xaxis.set_major_locator(MultipleLocator(16))

            # Add label
            ax.text(
                0.05, 0.95, f'{tm}',
                va='top', fontsize=6, transform=ax.transAxes
            )

            ax.tick_params(axis='both', which='major', labelsize=5)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            self.ax_dict[tm] = ax

    def plot_waveforms(self, waveforms, thresh_fraction=0.75):
        n_pix, n_samples = waveforms.shape
        n_tmpix = 64
        n_asicpix = 16
        n_tm = n_pix // n_tmpix
        mapping = self.tm_mapping
        n_cols = mapping.metadata['n_columns']

        avg_wf = waveforms.mean(axis=0)
        tm_avg_wf = waveforms.reshape((n_tm, n_tmpix, n_samples)).mean(1)
        thresh = avg_wf.max() * thresh_fraction
        min_ = waveforms.min()
        max_ = waveforms.max()

        self.fig.text(
            0.32, 0.885,
            f"[Lists indicate pixels with "
            f"amplitudes <{thresh_fraction:.2f} x cam. mean]",
            fontsize=8, transform=self.fig.transFigure
        )

        for tm, ax in self.ax_dict.items():
            outliers = ''
            for tmpix in range(n_tmpix):
                pixel = tm * n_tmpix + tmpix
                asic = tmpix // n_asicpix
                color = self.asic_color[asic]
                wf = waveforms[pixel]
                ax.plot(wf, color=color, lw=0.2, alpha=0.5)
                if wf.max() < thresh:
                    outliers += f'[{tmpix}|{pixel}]\n'

            ax.plot(avg_wf, color='black', ls='-', lw=0.5, alpha=0.9)
            ax.plot(tm_avg_wf[tm], color='black', ls='--', lw=0.5, alpha=0.9)

            ax.text(
                0.05, 0.85, outliers,
                va='top', fontsize=4, transform=ax.transAxes
            )

            ax.set_xlim(0, n_samples)
            ax.set_ylim(min_, max_)

            # Plot axis on subplot
            xticks = np.arange(0, n_samples+32, 32)
            yticks = ax.get_yticks()
            row = mapping.loc[mapping['slot'] == tm]['row'].iloc[0]
            col = mapping.loc[mapping['slot'] == tm]['col'].iloc[0]
            mapping_row = mapping.loc[mapping['row'] == row]
            mapping_col = mapping.loc[mapping['col'] == col]
            if col == mapping_row['col'].min():
                if row == 0:
                    ax.set_yticks(yticks[:-2])
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.yaxis.set_visible(False)
            if row == mapping_col['row'].min():
                if col == n_cols-1:
                    ax.set_xticks(xticks[1:])
                elif col < n_cols-2:
                    ax.set_xticks(xticks[:-1])
                else:
                    ax.set_xticks(xticks)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.xaxis.set_visible(False)
            [
                ax.axvline(x=x, ls='--', color='grey', lw=0.5, alpha=0.2)
                for x in xticks
            ]
            [
                ax.axhline(y=y, ls='--', color='grey', lw=0.5, alpha=0.2)
                for y in xticks
            ]


class CameraPixelWaveformPlotter(Plotter):
    def __init__(self, mapping):
        super().__init__()

        mpl.rcParams['axes.linewidth'] = 0.1

        self.mapping = mapping
        n_rows = self.mapping.metadata['n_rows']
        n_cols = self.mapping.metadata['n_columns']
        n_rows_tm = n_rows // 6
        n_cols_tm = n_cols // 6
        n_tmpix = self.mapping.index.size // self.mapping.metadata['n_modules']

        self.fig = plt.figure(figsize=(8.27, 11.69 - 3.5))
        self.fig.subplots_adjust(hspace=0.0, wspace=0.0)

        self.ax_dict = dict()

        total = self.mapping.index.size
        desc = "Creating pixel axes"
        for pixel, df_row in tqdm(self.mapping.iterrows(), total=total, desc=desc):
            row = int(df_row['row'])
            col = int(df_row['col'])

            ax_col = col
            ax_row = n_rows - 1 - row

            ax = plt.subplot2grid(
                (n_rows, n_cols), (ax_row, ax_col), fig=self.fig
            )

            # Add label
            ax.text(
                0.05, 0.95, f'{pixel}\n{pixel//n_tmpix},{pixel%n_tmpix}',
                va='top', fontsize=1, transform=ax.transAxes
            )

            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

            # Outlines
            if col % n_cols_tm != 0:
                ax.spines['left'].set_visible(False)
            if col % n_cols_tm != n_cols_tm - 1:
                ax.spines['right'].set_visible(False)
            if row % n_rows_tm != 0:
                ax.spines['bottom'].set_visible(False)
            if row % n_rows_tm != n_rows_tm - 1:
                ax.spines['top'].set_visible(False)
            ax.axhline(y=0, ls=':', color='grey', lw=0.1, alpha=0.2)

            self.ax_dict[pixel] = ax

    def plot_waveforms(self, waveforms):
        n_pix, n_samples = waveforms.shape

        avg_wf = waveforms.mean(axis=0)
        min_ = waveforms.min()
        max_ = waveforms.max()

        norm = mpl.colors.Normalize(
            vmin=waveforms.max(1).min(),
            vmax=waveforms.max(1).max(),
        )

        desc = "Plotting pixels"
        for pixel, ax in tqdm(self.ax_dict.items(), desc=desc):
            color = mpl.cm.viridis(norm(waveforms[pixel].max()))
            ax.set_facecolor(color)
            ax.plot(waveforms[pixel], color='white', lw=0.2, alpha=1)
            ax.plot(avg_wf, color='black', ls='--', lw=0.2, alpha=0.5)

            ax.set_xlim(0, n_samples)
            ax.set_ylim(min_, max_)

    def highlight_pixels(self, pixel_mask, color='red'):
        n_rows = self.mapping.metadata['n_rows']
        n_cols = self.mapping.metadata['n_columns']
        n_rows_tm = n_rows // 6
        n_cols_tm = n_cols // 6

        for pixel, true in enumerate(pixel_mask):
            ax = self.ax_dict[pixel]

            if true:
                for key, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(1)
            else:
                # Reset
                for key, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(0.1)
                row = int(self.mapping.iloc[pixel]['row'])
                col = int(self.mapping.iloc[pixel]['col'])
                if col % n_cols_tm != 0:
                    ax.spines['left'].set_visible(False)
                if col % n_cols_tm != n_cols_tm - 1:
                    ax.spines['right'].set_visible(False)
                if row % n_rows_tm != 0:
                    ax.spines['bottom'].set_visible(False)
                if row % n_rows_tm != n_rows_tm - 1:
                    ax.spines['top'].set_visible(False)
