from CHECLabPy.waveform_reducers.cross_correlation import CrossCorrelation


class CrossCorrelationCE(CrossCorrelation):
    """
    Same as CrossCorrelation, but purposefully not performing anything other
    than charge extraction to save on processing time.
    """

    def process(self, waveforms):
        self._set_t_event(waveforms)
        params = dict(t_event=self.t_event)

        params.update(self._get_charge(waveforms))

        return params
