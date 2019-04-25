
class AmplitudeCalibrator:
    def __init__(self, ff, pedestal, mv2pe):
        """
        Calibrator of the amplitude extracted from the waveform into
        photoelectrons

        Parameters
        ----------
        ff : ndarray
            Flat-field coefficient for each pixel
        pedestal : ndarray
            Pedestal for each pixel
        mv2pe : float
            Nominal conversion value from "extracted units"
            (typically mV or mVns) into photoelectrons.
            This value is what you divide by to convert into photoelectrons.
            Single value for entire camera.
        """
        self.ff = ff
        self.pedestal = pedestal
        self.mv2pe = mv2pe

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
        return (values - self.pedestal[pixels]) / self.mv2pe * self.ff[pixels]
