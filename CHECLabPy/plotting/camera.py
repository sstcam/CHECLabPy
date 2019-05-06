from CHECLabPy.plotting.setup import Plotter
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
from CHECLabPy.utils.mapping import get_clp_mapping_from_tc_mapping
import os
from copy import copy


class CameraImage(Plotter):
    def __init__(self, xpix, ypix, size, cmap=None, **kwargs):
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
        kwargs
            Arguments passed to `CHECLabPy.plottong.setup.Plotter`
        """
        super().__init__(**kwargs)

        rc = {
            "xtick.direction": 'out',
            "ytick.direction": 'out',
        }
        mpl.rcParams.update(rc)

        self._image = None
        self._mapping = None
        self.colorbar = None
        self.autoscale = True

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

        self.pixels = PatchCollection(patches, linewidth=0, cmap=cmap)
        self.ax.add_collection(self.pixels)
        self.pixels.set_array(np.zeros(self.n_pixels))

        self.ax.set_aspect('equal', 'datalim')
        self.ax.set_xlabel("X position (m)")
        self.ax.set_ylabel("Y position (m)")
        self.ax.autoscale_view()
        self.ax.axis('off')

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        assert val.size == self.n_pixels

        self._image = val

        self.pixels.set_array(np.ma.masked_invalid(val))
        self.pixels.changed()
        if self.autoscale:
            self.pixels.autoscale()  # Updates the colorbar
        self.ax.figure.canvas.draw()

    def save(self, output_path):
        super().save(output_path)
        if output_path.endswith('.pdf'):
            try:
                self.crop(output_path)
            except ModuleNotFoundError:
                pass

    @staticmethod
    def crop(path):
        from PyPDF2 import PdfFileWriter, PdfFileReader
        with open(path, "rb") as in_f:
            input1 = PdfFileReader(in_f)
            output = PdfFileWriter()

            num_pages = input1.getNumPages()
            for i in range(num_pages):
                page = input1.getPage(i)
                page.cropBox.lowerLeft = (100, 20)
                page.cropBox.upperRight = (340, 220)
                output.addPage(page)

            pdf_path = os.path.splitext(path)[0] + "_cropped.pdf"
            with open(pdf_path, "wb") as out_f:
                output.write(out_f)
                print("Cropped figure saved to: {}".format(pdf_path))

    def add_colorbar(self, label='', pad=-0.2, ax=None, **kwargs):
        if ax is None:
            ax = self.ax
        self.colorbar = self.ax.figure.colorbar(
            self.pixels, label=label, pad=pad, ax=ax, **kwargs
        )

    def set_limits_minmax(self, zmin, zmax):
        """
        Set the color scale limits from min to max
        """
        self.pixels.set_clim(zmin, zmax)
        self.autoscale = False

    def set_z_log(self):
        self.pixels.norm = LogNorm()
        self.pixels.autoscale()

    def reset_limits(self):
        """
        Reset to auto color scale limits
        """
        self.autoscale = True
        self.pixels.autoscale()

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

    def add_text_to_pixel(self, pixel, value, size=3, color='w', **kwargs):
        """
        Add a text label to a single pixel

        Parameters
        ----------
        pixel : int
        value : str or float
        size : int
            Font size
        color : str
            Color of the text
        kwargs
            Named arguments to pass to matplotlib.pyplot.text
        """
        pos_x = self.xpix[pixel]
        pos_y = self.ypix[pixel]
        self.ax.text(pos_x, pos_y, value,
                     fontsize=size, color=color, ha='center', **kwargs)

    def add_pixel_text(self, values, size=3, color='w', **kwargs):
        """
        Add a text label to each pixel

        Parameters
        ----------
        values : ndarray
        size : int
            Font size
        color : str
            Color of the text
        kwargs
            Named arguments to pass to matplotlib.pyplot.text
        """
        assert values.size == self.n_pixels
        for pixel in range(self.n_pixels):
            self.add_text_to_pixel(pixel, values[pixel], size, color, **kwargs)

    def highlight_pixels(self, pixels, color='g', linewidth=0.5, alpha=0.75):
        """
        Highlight the given pixels with a colored line around them

        Parameters
        ----------
        pixels : index-like
            The pixels to highlight.
            Can either be a list or array of integers or a
            boolean mask of length number of pixels
        color: a matplotlib conform color
            the color for the pixel highlighting
        linewidth: float
            linewidth of the highlighting in points
        alpha: 0 <= alpha <= 1
            The transparency
        """

        lw_array = np.zeros_like(self.image)
        lw_array[pixels] = linewidth
        pixel_highlighting = copy(self.pixels)
        pixel_highlighting.set_facecolor('none')
        pixel_highlighting.set_linewidth(lw_array)
        pixel_highlighting.set_alpha(alpha)
        pixel_highlighting.set_edgecolor(color)
        self.ax.add_collection(pixel_highlighting)
        return pixel_highlighting

    def annotate_tm_edge_label(self):
        """
        Annotate each of the TMs on the top and bottom of the camera
        """
        if self._mapping is not None:
            kw = dict(fontsize=6, color='black', ha='center')
            m = self._mapping
            pix_size = self._mapping.metadata['size']
            f_tm_top = lambda g: m.ix[m.ix[g.index]['row'].idxmax(), 'slot']
            f_tm_bottom = lambda g: m.ix[m.ix[g.index]['row'].idxmin(), 'slot']
            tm_top = np.unique(m.groupby('col').agg(f_tm_top)['slot'])
            tm_bottom = np.unique(m.groupby('col').agg(f_tm_bottom)['slot'])
            for tm in tm_top:
                df = m.loc[m['slot'] == tm]
                ypix = df['ypix'].max() + pix_size * 0.7
                xpix = df['xpix'].mean()
                tm_txt = "TM{:02d}".format(tm)
                self.ax.text(xpix, ypix, tm_txt, va='bottom', **kw)
            for tm in tm_bottom:
                df = m.loc[m['slot'] == tm]
                ypix = df['ypix'].min() - pix_size * 0.7
                xpix = df['xpix'].mean()
                tm_txt = "TM{:02d}".format(tm)
                self.ax.text(xpix, ypix, tm_txt, va='top', **kw)
        else:
            print("Cannot annotate, no mapping attached to class")

    def annotate_led_flasher(self):
        """
        Annotate each of the LED flashers in the four corners of the camera
        """
        if self._mapping is not None:
            pix_size = self._mapping.metadata['size']
            axl = self._mapping.metadata['fOTUpX_l']
            ayl = self._mapping.metadata['fOTUpY_l'] + 2 * pix_size

            dxl = [1, -1, 1, -1]
            dyl = [1, 1, -1, -1]
            for i, (dx, dy) in enumerate(zip(dxl, dyl)):
                x = axl * dx
                y = ayl * dy
                self.ax.add_patch(Circle((x, y), radius=0.01, color='red'))
                self.ax.text(x, y, f"{i}", fontsize=7, color='white',
                             ha='center', va='center')
        else:
            print("Cannot annotate, no mapping attached to class")

    @classmethod
    def from_mapping(cls, mapping, **kwargs):
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
        kwargs
            Arguments passed to `CHECLabPy.plottong.setup.Plotter`

        Returns
        -------
        `CameraImage`

        """
        xpix = mapping['xpix'].values
        ypix = mapping['ypix'].values
        size = mapping.metadata['size']
        image = cls(xpix, ypix, size, **kwargs)
        image._mapping = mapping
        return image

    @classmethod
    def from_tc_mapping(cls, tc_mapping, **kwargs):
        """
        Generate the class using the TargetCalib Mapping Class
        Parameters
        ----------
        tc_mapping : `target_calib.Mapping`
        kwargs
            Arguments passed to `CHECLabPy.plottong.setup.Plotter`

        Returns
        -------
        `CameraImage`

        """
        mapping = get_clp_mapping_from_tc_mapping(tc_mapping)
        return cls.from_mapping(mapping, **kwargs)

    @classmethod
    def from_camera_version(cls, camera_version, single=False, **kwargs):
        """
        Generate the class using the camera version (required TargetCalib)

        Parameters
        ----------
        camera_version : str
            Version of the camera (e.g. "1.0.1" corresponds to CHEC-S)
        single : bool
            Designate if it is just a single module you wish to plot
        kwargs
            Arguments passed to `CHECLabPy.plottong.setup.Plotter`

        Returns
        -------
        `CameraImage`

        """
        from target_calib import CameraConfiguration
        config = CameraConfiguration(camera_version)
        tc_mapping = config.GetMapping(single)
        return cls.from_tc_mapping(tc_mapping, **kwargs)


class CameraImageImshow(Plotter):
    def __init__(self, row, col, n_rows, n_cols, **kwargs):
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
        kwargs
            Arguments passed to `CHECLabPy.plottong.setup.Plotter`
        """
        super().__init__(**kwargs)

        self._image = None
        self._mapping = None
        self.colorbar = None
        self.autoscale = True

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
        if self.autoscale:
            self.camera.autoscale()

    def add_colorbar(self, label=''):
        self.colorbar = self.fig.colorbar(self.camera, label=label)

    def set_limits_minmax(self, zmin, zmax):
        """
        Set the color scale limits from min to max
        """
        self.camera.set_clim(zmin, zmax)
        self.autoscale = False

    def reset_limits(self):
        """
        Reset to auto color scale limits
        """
        self.autoscale = True
        self.camera.autoscale()

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

    def add_text_to_pixel(self, pixel, value, size=3, color='w', **kwargs):
        """
        Add a text label to a single pixel

        Parameters
        ----------
        pixel : int
        value : str or float
        size : int
            Font size
        color : str
            Color of the text
        kwargs
            Named arguments to pass to matplotlib.pyplot.text
        """
        pos_x = self.col[pixel]
        pos_y = self.row[pixel]
        self.ax.text(pos_x, pos_y, value,
                     fontsize=size, color=color, ha='center', **kwargs)

    def add_pixel_text(self, values, size=3, color='w', **kwargs):
        """
        Add a text label to each pixel

        Parameters
        ----------
        values : ndarray
        size : int
            Font size
        color : str
            Color of the text
        kwargs
            Named arguments to pass to matplotlib.pyplot.text
        """
        assert values.size == self.n_pixels
        for pixel in range(self.n_pixels):
            self.add_text_to_pixel(pixel, values[pixel], size, color, **kwargs)

    def annotate_tm_edge_label(self):
        """
        Annotate each of the TMs on the top and bottom of the camera
        """
        if self._mapping is not None:
            kw = dict(fontsize=6, color='black', ha='center')
            m = self._mapping
            n_rows = m.metadata['n_rows']
            f_tm_top = lambda g: m.ix[m.ix[g.index]['row'].idxmax(), 'slot']
            f_tm_bottom = lambda g: m.ix[m.ix[g.index]['row'].idxmin(), 'slot']
            tm_top = np.unique(m.groupby('col').agg(f_tm_top)['slot'])
            tm_bottom = np.unique(m.groupby('col').agg(f_tm_bottom)['slot'])
            for tm in tm_top:
                df = m.loc[m['slot'] == tm]
                row = df['row'].max() + n_rows/48 + 0.4
                col = df['col'].mean()
                tm_txt = "TM{:02d}".format(tm)
                self.ax.text(col, row, tm_txt, va='bottom', **kw)
            for tm in tm_bottom:
                df = m.loc[m['slot'] == tm]
                row = df['row'].min() - n_rows/48 - 0.4
                col = df['col'].mean()
                tm_txt = "TM{:02d}".format(tm)
                self.ax.text(col, row, tm_txt, va='top', **kw)
        else:
            print("Cannot annotate, no mapping attached to class")

    @classmethod
    def from_mapping(cls, mapping, **kwargs):
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
        kwargs
            Arguments passed to `CHECLabPy.plottong.setup.Plotter`

        Returns
        -------
        `CameraImage`

        """
        row = mapping['row'].values
        col = mapping['col'].values
        n_rows = mapping.metadata['n_rows']
        n_cols = mapping.metadata['n_columns']
        image = cls(row, col, n_rows, n_cols, **kwargs)
        image._mapping = mapping
        return image

    @classmethod
    def from_tc_mapping(cls, tc_mapping, **kwargs):
        """
        Generate the class using the TargetCalib Mapping Class
        Parameters
        ----------
        tc_mapping : `target_calib.Mapping`
        kwargs
            Arguments passed to `CHECLabPy.plottong.setup.Plotter`

        Returns
        -------
        `CameraImage`

        """
        mapping = get_clp_mapping_from_tc_mapping(tc_mapping)
        return cls.from_mapping(mapping, **kwargs)

    @classmethod
    def from_camera_version(cls, camera_version, single=False, **kwargs):
        """
        Generate the class using the camera version (required TargetCalib)

        Parameters
        ----------
        camera_version : str
            Version of the camera (e.g. "1.0.1" corresponds to CHEC-S)
        single : bool
            Designate if it is just a single module you wish to plot
        kwargs
            Arguments passed to `CHECLabPy.plottong.setup.Plotter`

        Returns
        -------
        `CameraImage`

        """
        from target_calib import CameraConfiguration
        config = CameraConfiguration(camera_version)
        tc_mapping = config.GetMapping(single)
        return cls.from_tc_mapping(tc_mapping, **kwargs)
