
class AmplitudeCalibrator:
    def __init__(self, ff, pedestal, conversion):
        """
        Calibrator of the amplitude extracted from the waveform into
        photoelectrons

        Parameters
        ----------
        ff : ndarray
            Flat-field coefficient for each pixel
        pedestal : ndarray
            Pedestal coefficient for each pixel
        conversion : float
            Nominal conversion value of "extracted units"
            (typically mV or mVns) per photoelectrons or photons.
            Single value for entire camera.
        """
        self.ff = ff
        self.pedestal = pedestal
        self.conversion = conversion

    def __call__(self, values, pixels):
        """
        Calibrate the extracted amplitudes

        Parameters
        ----------
        values : float or ndarray
            Amplitude values to calibrate
        pixels : int or ndarray
            Pixel corresponding to the value. Must be same shape as values.

        Returns
        -------
        calibrated : float or ndarray
            Calibrated photoelectron values. Same shape as values

        """
        return ((values - self.pedestal[pixels]) /
                self.conversion * self.ff[pixels])
