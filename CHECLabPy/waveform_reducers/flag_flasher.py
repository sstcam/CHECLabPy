from CHECLabPy.core.reducer import WaveformReducer, column
import numpy as np
import os
import time

class FlagFlashers(WaveformReducer):
    """
    Looks into an event waveforms for a flasher event.
    Returns -1 if the event is not due to a flasher,
    else it returns the number of the flashing unit.
    It assumes (as of 28.10.19) the patterns used as reference
    in the first two ASTRI campaigns.
    """
    def __init__(self, n_pixels, n_samples, **kwargs):
        super().__init__(n_pixels, n_samples, **kwargs)

    def _prepare(self, waveforms):
        super()._prepare(waveforms)
        peaks=np.max(self.waveforms,axis=1)
        unit_vect=np.asarray([np.average(peaks[64*5:64*6]),np.average(peaks[64*23:64*24]),np.average(peaks[64*8:64*9]),np.average(peaks[64*26:64*27])]) # average pulse height for TMs 5,23,8,26
        if (unit_vect<30).any(): # they should be all well above 30 mV on average, for a flasher event
            unit=-1
        else:
            unit=np.argmax(unit_vect)
        self.unit=unit

    @column
    def flasher_unit(self):
        """
        Finds the "active" flashing unit
        """

        return self.unit
