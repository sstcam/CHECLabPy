import numpy as np
import pandas as pd


class ChargeResolution:
    def __init__(self):
        """
        Calculates the charge resolution with an efficient, low-memory
        interative approach, allowing the contribution of data/events
        without reading the entire dataset into memory.

        Utilises Pandas DataFrames, and makes no assumptions on the order of
        the data, and does not require the true charge to be integer (as may
        be the case for lab measurements where an average illumination
        is used).

        A list is filled with a dataframe for each contribution, and only
        amalgamated into a single dataframe (reducing memory) once the memory
        of the list is becomming large (or at the end of the filling),
        reducing the time required to produce the output.

        Attributes
        ----------
        self._df_list : list
        self._df : pd.DataFrame
        self._n_bytes : int
            Monitors the number of bytes being held in memory
        """
        self._df_list = []
        self._df = pd.DataFrame()
        self._n_bytes = 0

    @staticmethod
    def rmse_abs(sum_, n):
        return np.sqrt(sum_ / n)

    @staticmethod
    def rmse(true, sum_, n):
        return ChargeResolution.rmse_abs(sum_, n) / np.abs(true)

    @staticmethod
    def charge_res_abs(true, sum_, n):
        return np.sqrt((sum_ / n) + true)

    @staticmethod
    def charge_res(true, sum_, n):
        return ChargeResolution.charge_res_abs(true, sum_, n) / np.abs(true)

    def add(self, pixel, true, measured):
        """
        Contribute additional values to the Charge Resolution

        Parameters
        ----------
        pixel : ndarray
            1D array containing the pixel for each entry
        true : ndarray
            1D array containing the true charge for each entry
        measured : ndarray
            1D array containing the measured charge for each entry
        """
        diff2 = np.power(measured - true, 2)
        df = pd.DataFrame(dict(
            pixel=pixel,
            true=true,
            sum=diff2,
            n=np.uint32(1)
        ))
        self._df_list.append(df)
        self._n_bytes += df.memory_usage(index=True, deep=True).sum()
        if self._n_bytes > 0.5E9:
            self.amalgamate()

    def amalgamate(self):
        """
        Concatenate the dataframes inside the list, and sum together
        values per pixel and true charge in order to reduce memory use.
        """
        self._df = pd.concat([self._df, *self._df_list], ignore_index=True)
        self._df = self._df.groupby(['pixel', 'true']).sum().reset_index()
        self._n_bytes = 0
        self._df_list = []

    def finish(self):
        """
        Perform the final amalgamation, and calculate the charge resolution
        from the resulting sums

        Returns
        -------
        df_pixel : pd.DataFrame
            Dataframe containing the charge resolution per pixel
        df_camera : pd.DataFrame
            Dataframe containing the charge resolution for the entire camera
        """
        self.amalgamate()
        df_pixel = self._df.copy()
        true = df_pixel['true'].values
        sum_ = df_pixel['sum'].values
        n = df_pixel['n'].values
        df_pixel['rmse'] = self.rmse(true, sum_, n)
        df_pixel['rmse_abs'] = self.rmse_abs(sum_, n)
        df_pixel['charge_resolution'] = self.charge_res(true, sum_, n)
        df_pixel['charge_resolution_abs'] = self.charge_res_abs(true, sum_, n)
        df_camera = self._df.copy().groupby('true').sum().reset_index()
        df_camera = df_camera.drop(columns='pixel')
        true = df_camera['true'].values
        sum_ = df_camera['sum'].values
        n = df_camera['n'].values
        df_camera['rmse'] = self.rmse(true, sum_, n)
        df_camera['rmse_abs'] = self.rmse_abs(sum_, n)
        df_camera['charge_resolution'] = self.charge_res(true, sum_, n)
        df_camera['charge_resolution_abs'] = self.charge_res_abs(true, sum_, n)
        return df_pixel, df_camera


class ChargeStatistics:
    def __init__(self):
        """
        Calculates the charge statistics with an efficient, low-memory
        interative approach, allowing the contribution of data/events
        without reading the entire dataset into memory.

        Utilises Pandas DataFrames, and makes no assumptions on the order of
        the data.

        A list is filled with a dataframe for each contribution, and only
        amalgamated into a single dataframe (reducing memory) once the memory
        of the list is becomming large (or at the end of the filling),
        reducing the time required to produce the output.

        Attributes
        ----------
        self._df_list : list
        self._df : pd.DataFrame
        self._n_bytes : int
            Monitors the number of bytes being held in memory
        """
        self._df_list = []
        self._df = pd.DataFrame()
        self._n_bytes = 0

    def add(self, pixel, amplitude, charge):
        """
        Contribute additional values to the statistics

        Parameters
        ----------
        pixel : ndarray
            1D array containing the pixel for each entry
        amplitude : ndarray
            1D array containing the input amplitude for each entry
        charge : ndarray
            1D array containing the measured charge for each entry
        """
        df = pd.DataFrame(dict(
            pixel=pixel,
            amplitude=amplitude,
            sum=charge,
            sum2=charge**2,
            n=np.uint32(1)
        ))
        self._df_list.append(df)
        self._n_bytes += df.memory_usage(index=True, deep=True).sum()
        if self._n_bytes > 0.5E9:
            self.amalgamate()

    def amalgamate(self):
        """
        Concatenate the dataframes inside the list, and sum together
        values per pixel and true charge in order to reduce memory use.
        """
        self._df = pd.concat([self._df, *self._df_list], ignore_index=True)
        self._df = self._df.groupby(['pixel', 'amplitude']).sum().reset_index()
        self._n_bytes = 0
        self._df_list = []

    def finish(self):
        """
        Perform the final amalgamation, and calculate the charge statistics
        from the resulting sums

        Returns
        -------
        df_pixel : pd.DataFrame
            Dataframe containing the charge statistics per pixel
        df_camera : pd.DataFrame
            Dataframe containing the charge statistics for the entire camera
        """
        self.amalgamate()
        df = self._df.copy()
        sum_ = df['sum'].values
        sum2 = df['sum2'].values
        n = df['n'].values
        mean = sum_ / n
        std = np.sqrt((sum2 / n) - (mean**2))
        df['mean'] = mean
        df['std'] = std
        df_camera = self._df.copy().groupby('amplitude').sum().reset_index()
        df_camera = df_camera.drop(columns='pixel')
        sum_ = df_camera['sum'].values
        sum2 = df_camera['sum2'].values
        n = df_camera['n'].values
        mean = sum_ / n
        std = np.sqrt((sum2 / n) - (mean**2))
        df_camera['mean'] = mean
        df_camera['std'] = std
        return df, df_camera
