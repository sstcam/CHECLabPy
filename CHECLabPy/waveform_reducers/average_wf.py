from CHECLabPy.core.base_reducer import WaveformReducer


class AverageWF(WaveformReducer):
    """
    Extractor which defines a window about the maximum of the average
    waveform across all pixels. No cleaning is applied to the waveforms. The
    window size and shift is configurable in the initialization.

    Uses the default implementation of WaveformReducer
    """
