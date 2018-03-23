"""
Plot camera image using just TargetCalib and python
"""
import numpy as np
from target_calib import MappingCHEC
from CHECLabPy.plotting.setup import Plotter


class CameraPlotter(Plotter):
    def __init__(self, version):
        super().__init__()
        self.mapping = MappingCHEC()  # TODO update when version implemented

        self.row = np.array(self.mapping.GetRowVector())
        self.col = np.array(self.mapping.GetColumnVector())
        self.nrows = self.row.max() + 1
        self.ncols = self.col.max() + 1

        self.data = np.ma.zeros((self.nrows, self.ncols))
        self.image = self.ax.imshow(self.data, origin='lower')
        self.fig.colorbar(self.image)
        self.ax.axis('off')

    @staticmethod
    def figsize(scale=1.5):
        super(CameraPlotter, CameraPlotter).figsize(scale)

    def set(self, data):
        self.data = np.ma.zeros((self.nrows, self.ncols))
        self.data[self.row, self.col] = data
        self.data[0:8, 40:48] = np.ma.masked
        self.data[0:8, 0:8] = np.ma.masked
        self.data[40:48, 0:8] = np.ma.masked
        self.data[40:48, 40:48] = np.ma.masked
        self.image.set_data(self.data)
        self.image.autoscale()

    def annotate(self):
        axl = self.mapping.fOTUpCol_l
        ayl = self.mapping.fOTUpRow_l
        adx = self.mapping.fOTUpCol_u - axl
        ady = self.mapping.fOTUpRow_u - ayl
        self.ax.arrow(axl, ayl, adx, ady, head_width=1, head_length=1, fc='r',
                      ec='r')
        text = "ON-Telescope UP"
        self.ax.text(axl, ayl, text, fontsize=8, color='r', ha='center',
                     va='bottom')

    def plot(self, data):
        self.set(data)
