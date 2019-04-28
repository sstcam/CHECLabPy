from CHECLabPy.calib import get_calib_data
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class SiPMDataSheetConverter:
    def __init__(self):
        path = get_calib_data("sipm_data_sheet.csv")
        self.df = pd.read_csv(path)

    def __call__(self, *, vover=None, opct=None, pde=None, gain=None):
        values = dict(vover=vover, opct=opct, pde=pde, gain=gain)
        n_set = sum(x is not None for x in values.values())
        if n_set != 1:
            raise ValueError(
                "SiPMDataSheetConverter.convert requires a single "
                "named value (e.g. opct=0.25)"
            )
        for key, value in values.items():
            if value is not None:
                interpolated = dict()
                x = self.df[key].values

                for column in self.df.columns:
                    f = interp1d(x, self.df[column].values)
                    interpolated[column] = f(value)
                return interpolated
