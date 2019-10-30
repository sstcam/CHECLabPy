import numpy as np
import pandas as pd
from CHECLabPy.calib import get_calib_data
from CHECLabPy.utils.files import sort_file_list, create_directory
from glob import glob
import os
from datetime import datetime


N_PIXELS = 2048
N_SUPERPIXELS = N_PIXELS // 4


class PixelMasks:
    def __init__(self, path=None):
        """
        Class to handle the three types of pixel masks:
            - Dead pixels
            - Low pixels
            - Bad-HV superpixels

        A pixel is set to True if it is masked/dead/low/bad...

        The latest pixel mask file is loaded by default

        Parameters
        ----------
        path : str
            OPTIONAL specify a particular pixel mask file to load
        """
        self.dead = np.zeros(N_PIXELS, dtype=np.bool)
        self.low = np.zeros(N_PIXELS, dtype=np.bool)
        self.bad_hv = np.zeros(N_SUPERPIXELS, dtype=np.bool)
        self._load_file(path=path, merge=False)

    def _load_file(self, path=None, merge=False):
        if path is None:
            paths = sort_file_list(glob(get_calib_data("pixel_masks/*")))
            path = paths[-1]  # Get latest file
        print(f"Reading pixel masks from: {path} (MERGE: {merge})")
        df = pd.read_csv(path, sep='\t')
        dead = df['dead'].values
        low = df['low'].values
        bad_hv = df['bad_hv'].values.reshape((N_SUPERPIXELS, 4)).any(1)
        if merge:
            self.dead = np.logical_or(self.dead, dead)
            self.low = np.logical_or(self.low, low)
            self.bad_hv = np.logical_or(self.bad_hv, bad_hv)
        else:
            self.dead = dead
            self.low = low
            self.bad_hv = bad_hv

    def _save_to_file(self, path):
        create_directory(os.path.dirname(path))
        print(f"Saving pixel masks to: {path}")
        bad_hv = np.repeat(self.bad_hv, 4)
        df = pd.DataFrame(dict(
            pixel=np.arange(self.dead.size),
            dead=self.dead,
            low=self.low,
            bad_hv=bad_hv,
        ))
        df.to_csv(path, sep='\t', index=False)

    def commit(self):
        """
        Commit the current mappings to a new file
        """
        now = datetime.now().strftime("d%y%m%dT%H%M")
        path = get_calib_data(f"pixel_masks/{now}.dat")
        self._save_to_file(path)

    @property
    def all_mask(self):
        return np.logical_or.reduce([
            self.dead, self.low, np.repeat(self.bad_hv, 4)
        ])

    @staticmethod
    def convert_superpixels_to_pixels(superpixel_list):
        superpixels = np.repeat(superpixel_list, 4)
        pixels = superpixels * 4 + np.arange(superpixels.size) % 4
        return pixels

    @staticmethod
    def convert_module_to_pixels(module_list):
        modules = np.repeat(module_list, 64)
        pixels = modules * 64 + np.arange(modules.size) % 64
        return pixels

    def plot(self):
        from CHECLabPy.plotting.camera import CameraImage
        ci = CameraImage.from_camera_version("1.1.0")
        ci.pixels.set_linewidth(0.2)
        ci.pixels.set_edgecolor('black')
        ci.add_pixel_text(np.arange(2048), color='red', size=1)
        image = np.full(2048, np.nan)
        image[self.dead] = 1
        image[self.low] = 2
        image[np.repeat(self.bad_hv, 4)] = 0.5
        ci.image = image
        ci.show()
