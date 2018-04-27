"""
Plot camera image using just TargetCalib and python
"""
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping


class CameraPlotter(Plotter):
    def __init__(self, mapping, talk=False):
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
        super().__init__(talk=talk)
        print("CameraPlotter is deprecated, consider switching to CameraImage")
        self.mapping = mapping

        self.row = self.mapping['row'].values
        self.col = self.mapping['col'].values
        self.n_rows = self.mapping.metadata['n_rows']
        self.n_cols = self.mapping.metadata['n_columns']

        self.data = np.ma.zeros((self.n_rows, self.n_cols))
        self.image = self.ax.imshow(self.data, origin='lower')
        self.colorbar = self.fig.colorbar(self.image)
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


class CameraImage(Plotter):
    def __init__(self, xpix, ypix, size, ax=None, talk=False):
        """
        Create a camera-image plot

        Parameters
        ----------
        xpix : ndarray
            The X positions of the pixels/superpixels/TMs
        ypix : ndarray
            The Y positions of the pixels/superpixels/TMs
        size : float
            The size of the pixels/superpixels/TMs
        ax : `matplotlib.axes.Axes`
            Optionally place the plot on a pre-existing axes
        talk : bool
            Plot with presentation formatting
        """
        super().__init__(ax=ax, talk=talk)

        self._image = None
        self._mapping = None
        self.colorbar = None

        self.xpix = xpix
        self.ypix = ypix

        assert self.xpix.size == self.ypix.size
        self.n_pixels = self.xpix.size

        patches = []
        for xx, yy in zip(self.xpix, self.ypix):
            rr = size + 0.0001  # extra size to pixels to avoid aliasing
            poly = Rectangle(
                (xx - rr / 2., yy - rr / 2.),
                width=rr,
                height=rr,
                fill=True,
            )
            patches.append(poly)

        self.pixels = PatchCollection(patches, linewidth=0)
        self.ax.add_collection(self.pixels)
        self.pixels.set_array(np.zeros(self.n_pixels))

        self.ax.set_aspect('equal', 'datalim')
        self.ax.set_xlabel("X position (m)")
        self.ax.set_ylabel("Y position (m)")
        self.ax.autoscale_view()
        self.ax.axis('off')

    @staticmethod
    def figsize(scale=1.5):
        super(CameraPlotter, CameraPlotter).figsize(scale)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        assert val.size == self.n_pixels

        self.pixels.set_array(val)
        self.pixels.changed()
        self.pixels.autoscale()
        self.ax.figure.canvas.draw()

    def add_colorbar(self, label=''):
        self.colorbar = self.ax.figure.colorbar(self.pixels, label=label)

    def annotate_on_telescope_up(self):
        """
        Add an arrow indicating where "ON-Telescope-UP" is
        """
        if self._mapping is not None:
            axl = self._mapping.metadata['fOTUpX_l']
            ayl = self._mapping.metadata['fOTUpY_l']
            adx = self._mapping.metadata['fOTUpX_u'] - axl
            ady = self._mapping.metadata['fOTUpY_u'] - ayl
            text = "ON-Telescope UP"
            self.ax.arrow(axl, ayl, adx, ady, head_width=0.01,
                          head_length=0.01, fc='r', ec='r')
            self.ax.text(axl, ayl, text, fontsize=4, color='r',
                         ha='center', va='bottom')
        else:
            print("Cannot annotate, no mapping attached to class")

    def add_pixel_text(self, values, fmt=None, size=3):
        assert values.size == self.n_pixels
        for pix in range(self.n_pixels):
            pos_x = self.xpix[pix]
            pos_y = self.ypix[pix]
            val = values[pix]
            if fmt:
                val = fmt.format(val)
            self.ax.text(pos_x, pos_y, val, fontsize=size,
                         color='w', ha='center')

    @classmethod
    def from_mapping(cls, mapping, talk=False):
        """
        Generate the class from a CHECLabPy mapping dataframe

        Parameters
        ----------
        mapping : `pandas.DataFrame`
            The mapping for the pixels stored in a pandas DataFrame. Can be
            obtained from either of these options:

            CHECLabPy.io.TIOReader.mapping
            CHECLabPy.io.ReaderR0.mapping
            CHECLabPy.io.ReaderR1.mapping
            CHECLabPy.io.DL1Reader.mapping
            CHECLabPy.utils.mapping.get_clp_mapping_from_tc_mapping
        talk : bool
            Plot with presentation formatting

        Returns
        -------
        `CameraImage`

        """
        xpix = mapping['xpix'].values
        ypix = mapping['ypix'].values
        size = mapping.metadata['size']
        image = cls(xpix, ypix, size, talk=talk)
        image._mapping = mapping
        return image

    @classmethod
    def from_tc_mapping(cls, tc_mapping, talk=False):
        """
        Generate the class using the TargetCalib Mapping Class
        Parameters
        ----------
        tc_mapping : `target_calib.Mapping`
        talk : bool
            Plot with presentation formatting

        Returns
        -------
        `CameraImage`

        """
        mapping = get_clp_mapping_from_tc_mapping(tc_mapping)
        return cls.from_mapping(mapping, talk=talk)

    @classmethod
    def from_camera_version(cls, camera_version, single=False, talk=False):
        """
        Generate the class using the camera version (required TargetCalib)

        Parameters
        ----------
        camera_version : str
            Version of the camera (e.g. "1.0.1" corresponds to CHEC-S)
        single : bool
            Designate if it is just a single module you wish to plot
        talk : bool
            Plot with presentation formatting

        Returns
        -------
        `CameraImage`

        """
        from target_calib import CameraConfiguration
        config = CameraConfiguration(camera_version)
        tc_mapping = config.GetMapping(single)
        return cls.from_tc_mapping(tc_mapping, talk=talk)


class CameraImageImshow(Plotter):
    def __init__(self, row, col, n_rows, n_cols, ax=None, talk=False):
        """
        Create a camera-image plot using imshow (essentially a 2D histogram,
        therefore missing module gaps)

        Parameters
        ----------
        row : ndarray
            The row for each pixel/superpixel/TM
        col : ndarray
            The row for each pixel/superpixel/TM
        n_rows : int
            The number of rows of pixel/superpixel/TM on the camera
        n_cols : int
            The number of rows of pixel/superpixel/TM on the camera
        ax : `matplotlib.axes.Axes`
            Optionally place the plot on a pre-existing axes
        talk : bool
            Plot with presentation formatting
        """
        super().__init__(ax=ax, talk=talk)

        self._image = None
        self._mapping = None
        self.colorbar = None

        self.row = row
        self.col = col
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_pixels = self.row.size

        assert self.row.size == self.col.size

        data = np.ma.zeros((self.n_rows, self.n_cols))
        self.mask(data)
        self.camera = self.ax.imshow(data, origin='lower')
        self.ax.axis('off')

    @staticmethod
    def figsize(scale=1.5):
        super(CameraPlotter, CameraPlotter).figsize(scale)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        assert val.size == self.n_pixels

        self._image = val
        data = np.ma.zeros((self.n_rows, self.n_cols))
        self.mask(data)
        data[self.row, self.col] = val
        self.camera.set_data(data)
        self.camera.autoscale()

    def add_colorbar(self, label=''):
        self.colorbar = self.fig.colorbar(self.camera, label=label)

    def mask(self, data):
        if (self.n_rows == 48) & (self.n_cols == 48):
            data[0:8, 40:48] = np.ma.masked
            data[0:8, 0:8] = np.ma.masked
            data[40:48, 0:8] = np.ma.masked
            data[40:48, 40:48] = np.ma.masked
        elif (self.n_rows == 24) & (self.n_cols == 24):
            data[0:4, 20:24] = np.ma.masked
            data[0:4, 0:4] = np.ma.masked
            data[20:24, 0:4] = np.ma.masked
            data[20:24, 20:24] = np.ma.masked
        elif (self.n_rows == 6) & (self.n_cols == 6):
            data[0:1, 5:6] = np.ma.masked
            data[0:1, 0:1] = np.ma.masked
            data[5:6, 0:1] = np.ma.masked
            data[5:6, 5:6] = np.ma.masked

    def annotate_on_telescope_up(self):
        """
        Add an arrow indicating where "ON-Telescope-UP" is
        """
        if self._mapping is not None:
            axl = self._mapping.metadata['fOTUpCol_l']
            ayl = self._mapping.metadata['fOTUpRow_l']
            adx = self._mapping.metadata['fOTUpCol_u'] - axl
            ady = self._mapping.metadata['fOTUpRow_u'] - ayl
            text = "ON-Telescope UP"
            self.ax.arrow(axl, ayl, adx, ady, head_width=1, head_length=1,
                          fc='r', ec='r')
            self.ax.text(axl, ayl, text, fontsize=8, color='r',
                         ha='center', va='bottom')
        else:
            print("Cannot annotate, no mapping attached to class")

    def add_pixel_text(self, values, fmt=None, size=3):
        assert values.size == self.n_pixels
        for pix in range(self.n_pixels):
            pos_x = self.col[pix]
            pos_y = self.row[pix]
            val = values[pix]
            if fmt:
                val = fmt.format(val)
            self.ax.text(pos_x, pos_y, val, fontsize=size,
                         color='w', ha='center')

    @classmethod
    def from_mapping(cls, mapping, talk=False):
        """
        Generate the class using a CHECLabPy mapping dataframe

        Parameters
        ----------
        mapping : `pandas.DataFrame`
            The mapping for the pixels stored in a pandas DataFrame. Can be
            obtained from either of these options:

            CHECLabPy.io.TIOReader.mapping
            CHECLabPy.io.ReaderR0.mapping
            CHECLabPy.io.ReaderR1.mapping
            CHECLabPy.io.DL1Reader.mapping
            CHECLabPy.utils.mapping.get_clp_mapping_from_tc_mapping
        talk : bool
            Plot with presentation formatting

        Returns
        -------
        `CameraImage`

        """
        row = mapping['row'].values
        col = mapping['col'].values
        n_rows = mapping.metadata['n_rows']
        n_cols = mapping.metadata['n_columns']
        image = cls(row, col, n_rows, n_cols, talk=talk)
        image._mapping = mapping
        return image

    @classmethod
    def from_tc_mapping(cls, tc_mapping, talk=False):
        """
        Generate the class using the TargetCalib Mapping Class
        Parameters
        ----------
        tc_mapping : `target_calib.Mapping`
        talk : bool
            Plot with presentation formatting

        Returns
        -------
        `CameraImage`

        """
        mapping = get_clp_mapping_from_tc_mapping(tc_mapping)
        return cls.from_mapping(mapping, talk=talk)

    @classmethod
    def from_camera_version(cls, camera_version, single=False, talk=False):
        """
        Generate the class using the camera version (required TargetCalib)

        Parameters
        ----------
        camera_version : str
            Version of the camera (e.g. "1.0.1" corresponds to CHEC-S)
        single : bool
            Designate if it is just a single module you wish to plot
        talk : bool
            Plot with presentation formatting

        Returns
        -------
        `CameraImage`

        """
        from target_calib import CameraConfiguration
        config = CameraConfiguration(camera_version)
        tc_mapping = config.GetMapping(single)
        return cls.from_tc_mapping(tc_mapping, talk=talk)
