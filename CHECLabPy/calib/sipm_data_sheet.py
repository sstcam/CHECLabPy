from CHECLabPy.calib import get_calib_data
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class SiPMDataSheetConverter:
    def __init__(self):
        """
        Convert between SiPM characteristics using the S12642 data sheet
        """
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

    def plot(self):
        from matplotlib import pyplot as plt
        x = self.df['vover'].values
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ln1 = ax1.plot(x, self.df['gain'], color='green', label="Gain")
        ln2 = ax2.plot(x, self.df['pde'], color='blue', label="PDE")
        ln3 = ax2.plot(x, self.df['opct'], color='red', label="OPCT")
        ax1.set_xlabel("Vover (V)")
        ax1.set_ylabel("Gain")
        ax2.set_ylabel("PDE, OPCT")
        ax1.set_ylim(0, 4e6)
        ax2.set_ylim(0, 1)
        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')
        plt.show()
