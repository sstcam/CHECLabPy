"""
Plot camera image using just TargetCalib and python
"""
import numpy as np
from CHECLabPy.plotting.setup import Plotter


class CameraPlotter(Plotter):
    def __init__(self, mapping):
        """
        Plot values in a camera image

        Parameters
        ----------
        mapping : pandas.DataFrame
            The mapping for the pixels stored in a pandas DataFrame. Can be
            obtained from either of these options:

            CHECLabPy.io.Reader.mapping
            CHECLabPy.io.ReaderR0.mapping
            CHECLabPy.io.ReaderR1.mapping
            CHECLabPy.io.DL1Reader.mapping
        """
        super().__init__()
        self.mapping = mapping

        self.row = self.mapping['row'].values
        self.col = self.mapping['col'].values
        self.n_rows = self.mapping.metadata['n_rows']
        self.n_cols = self.mapping.metadata['n_columns']

        self.data = np.ma.zeros((self.n_rows, self.n_cols))
        self.image = self.ax.imshow(self.data, origin='lower')
        self.fig.colorbar(self.image)
        self.ax.axis('off')

    @staticmethod
    def figsize(scale=1.5):
        super(CameraPlotter, CameraPlotter).figsize(scale)

    def set(self, data):
        self.data = np.ma.zeros((self.n_rows, self.n_cols))
        self.data[self.row, self.col] = data
        if (self.n_rows == 48) & (self.n_cols == 48):
            self.data[0:8, 40:48] = np.ma.masked
            self.data[0:8, 0:8] = np.ma.masked
            self.data[40:48, 0:8] = np.ma.masked
            self.data[40:48, 40:48] = np.ma.masked
        self.image.set_data(self.data)
        self.image.autoscale()

    def annotate(self):
        axl = self.mapping.metadata['fOTUpCol_l']
        ayl = self.mapping.metadata['fOTUpRow_l']
        adx = self.mapping.metadata['fOTUpCol_u'] - axl
        ady = self.mapping.metadata['fOTUpRow_u'] - ayl
        self.ax.arrow(axl, ayl, adx, ady, head_width=1, head_length=1, fc='r',
                      ec='r')
        text = "ON-Telescope UP"
        self.ax.text(axl, ayl, text, fontsize=8, color='r', ha='center',
                     va='bottom')

    def plot(self, data):
        self.set(data)
