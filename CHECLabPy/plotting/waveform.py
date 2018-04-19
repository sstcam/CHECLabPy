"""
Plot waveform
"""
from matplotlib.ticker import MultipleLocator
from CHECLabPy.plotting.setup import Plotter


class WaveformPlotter(Plotter):
    def __init__(self, title="", units="", tunits="ns", talk=False):
        super().__init__(talk=talk)
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
