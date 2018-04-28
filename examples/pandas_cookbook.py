"""
A selection of methods for a Pandas DataFrame. Please refer to the Pandas
documentation for a more desciptive guide to the methods:
https://pandas.pydata.org/pandas-docs/stable/dsintro.html
"""

import numpy as np
import pandas as pd
from CHECLabPy.core.io import DL1Reader


PATH = "/Users/Jason/Software/CHECLabPy/refdata/Run17473_dl1.h5"


def get_numpy():
    """
    Pandas dataframe columns are essentially numpy arrays.
    """
    r = DL1Reader(PATH)
    df = r.load_entire_table()
    charge_numpy_array = df['charge'].values
    print(type(charge_numpy_array))


def get_numpy_mean():
    """
    Pandas dataframe columns are essentially numpy array, and therefore can
    be operated on by any of the numpy methods.
    """
    r = DL1Reader(PATH)
    df = r.load_entire_table()
    charge_mean = np.mean(df['charge'])
    print(charge_mean)


def get_table_mean():
    """
    Pandas also has its own methods for obtaining many statistical results,
    which can be applied to the entire table at once efficiently.
    """
    r = DL1Reader(PATH)
    df = r.load_entire_table()
    mean_series = df.mean()
    print(mean_series)


def select_subset():
    """
    A subset of the DataFrame can be selected to produce a new DataFrame
    """
    r = DL1Reader(PATH)
    df = r.load_entire_table()
    df['tm'] = df['pixel'] // 64
    df_tm4 = df.loc[df['tm'] == 4]
    print(df_tm4)


def get_mean_per_tm():
    """
    The Pandas groupby method can be used to calculate statistics per group
    """
    r = DL1Reader(PATH)
    df = r.load_entire_table()
    df['tm'] = df['pixel'] // 64
    df_mean = df.groupby('tm').mean().reset_index()
    # reset_index() restores the tm column,
    # otherwise it will remain as the index
    print(df_mean)


def get_multiple_statistics():
    """
    The `aggregate` method allows multiple operations to be performed at once
    """
    r = DL1Reader(PATH)
    df = r.load_entire_table()
    df['tm'] = df['pixel'] // 64
    df_stats = df[['tm', 'charge']].groupby('tm').agg(['mean', 'min', 'max'])
    print(df_stats)
    print(df_stats['charge']['mean'])


def apply_different_statistic_to_different_column():
    """
    Passing a dict to `aggregate` allows you to specify a different operation
    depending on the column
    """
    r = DL1Reader(PATH)
    df = r.load_entire_table()
    df['tm'] = df['pixel'] // 64
    f = dict(pixel='first', charge='std')
    df_stats = df[['tm', 'pixel', 'charge']].groupby('tm').agg(f)
    print(df_stats)


def apply_custom_function():
    """
    Any function can be passed to the `apply` method, including numpy functions

    You will notice that the numpy std method produces a different result to
    the pandas result. Thats because by default numpy calculates the sample
    standard deviation, whereas pandas includes the Bessel correction by
    default to correct for the bias in the estimation of the
    population variance.
    """
    r = DL1Reader(PATH)
    df = r.load_entire_table()
    df['tm'] = df['pixel'] // 64
    df_pd_std = df[['tm', 'charge']].groupby('tm').std()['charge']
    df_np_std = df[['tm', 'charge']].groupby('tm').apply(np.std)['charge']
    df_comparison = pd.DataFrame(dict(pd=df_pd_std, np=df_np_std))
    print(df_comparison)


def apply_custom_function_agg():
    """
    One can also apply a custom function inside the agg approach
    """
    r = DL1Reader(PATH)
    df = r.load_entire_table()
    df['tm'] = df['pixel'] // 64
    f_camera_first_half = lambda g: df.loc[g.index].iloc[0]['tm'] < 32/2
    f = dict(pixel=f_camera_first_half, charge='std')
    df_stats = df[['tm', 'pixel', 'charge']].groupby('tm').agg(f)
    df_stats = df_stats.rename(columns={'pixel': 'camera_first_half'})
    print(df_stats)


def get_running_mean():
    """
    For very large files it may not be possible to utilise pandas statistical
    methods which assume the entire dataset is loaded into memory. This is the
    approach I often use to calculate a running statistic (including
    charge resolution)

    One should be careful with using this approach to calculate the standard
    deviation as it which can lead to numerical instability and
    arithmetic overflow when dealing with large values.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    class RunningStats:
        def __init__(self):
            self._df_list = []
            self._df = pd.DataFrame()
            self._n_bytes = 0

        def add(self, df_e):
            pixel = df_e['pixel'].values
            tm = pixel // 64
            charge = df_e['charge'].values

            df = pd.DataFrame(dict(
                tm=tm,
                sum=charge,
                sum2=charge**2,
                n=np.uint32(1)
            ))
            self._df_list.append(df)
            self._n_bytes += df.memory_usage(index=True, deep=True).sum()
            if self._n_bytes > 0.5E9:
                self.amalgamate()

        def amalgamate(self):
            self._df = pd.concat([self._df, *self._df_list], ignore_index=True)
            self._df = self._df.groupby(['tm']).sum().reset_index()
            self._n_bytes = 0
            self._df_list = []

        def finish(self):
            self.amalgamate()
            df = self._df.copy()
            sum_ = df['sum'].values
            sum2 = df['sum2'].values
            n = df['n'].values
            mean = sum_/n
            std = np.sqrt((sum2 / n) - (mean**2))
            df['mean'] = mean
            df['std'] = std
            return df

    r = DL1Reader(PATH)
    stats = RunningStats()
    for df_ev in r.iterate_over_events():
        stats.add(df_ev)
    df_rs = stats.finish()

    df_pd = r.load_entire_table()
    df_pd['tm'] = df_pd['pixel'] // 64
    df_stats = df_pd[['tm', 'charge']].groupby('tm').agg(['mean', np.std])
    df_rs['pd_mean'] = df_stats['charge']['mean']
    df_rs['pd_std'] = df_stats['charge']['std']
    print(df_rs)


if __name__ == '__main__':
    get_numpy()
    get_numpy_mean()
    get_table_mean()
    select_subset()
    get_mean_per_tm()
    get_multiple_statistics()
    apply_different_statistic_to_different_column()
    apply_custom_function()
    apply_custom_function_agg()
    get_running_mean()
