import numpy as np
from numba import njit
from math import exp, sqrt
from CHECLabPy.stats.pdf import poisson, normal_pdf, poisson_logpmf
from CHECLabPy.core.spectrum_fitter import SpectrumFitter, SpectrumParameter, \
    SpectrumParameterCollection


class MAPMFitter(SpectrumFitter):
    def __init__(self, n_illuminations, config_path=None):
        """
        SpectrumFitter which uses the MAPM fitting formula
        """
        super().__init__(n_illuminations, config_path)

        self.parameters = SpectrumParameterCollection([
            SpectrumParameter("eped", 0, (-10, 10)),
            SpectrumParameter("eped_sigma", 9, (0, 20)),
            SpectrumParameter("spe", 25, (0, 40)),
            SpectrumParameter("spe_sigma", 2, (0, 20)),
            SpectrumParameter("lambda_", 0.7, (0, 6), multi=True),
        ], n_illuminations, config_path)
        self.n_bins = 100
        self.range = (-40, 150)

    @staticmethod
    @njit(fastmath=True)
    def _get_spectra(n_illuminations, data_x, lookup, *parameter_values):
        spectra = []
        for i in range(n_illuminations):
            spectrum = calculate_spectrum(data_x, lookup[i], *parameter_values)
            spectra.append(spectrum)
        return spectra

    @staticmethod
    @njit(fastmath=True)
    def _get_likelihood(n_illuminations, data_x, data_y, lookup, *parameter_values):
        likelihood = 0
        for i in range(n_illuminations):
            spectrum = calculate_spectrum(data_x, lookup[i], *parameter_values)
            spectrum *= np.trapz(data_y[i], data_x) / np.trapz(spectrum, data_x)
            likelihood += np.nansum(-2 * poisson_logpmf(data_y[i], spectrum))
        return likelihood


@njit(fastmath=True)
def calculate_spectrum(data_x, lookup, *parameter_values):
    return mapm_spe(
        x=data_x,
        eped=parameter_values[lookup["eped"]],
        eped_sigma=parameter_values[lookup["eped_sigma"]],
        spe=parameter_values[lookup["spe"]],
        spe_sigma=parameter_values[lookup["spe_sigma"]],
        lambda_=parameter_values[lookup["lambda_"]],
    )


@njit(fastmath=True, parallel=True)
def mapm_spe(x, eped, eped_sigma, spe, spe_sigma, lambda_):
    """
    Fit for the SPE spectrum of a MAPM

    Parameters
    ----------
    x : ndarray
        The x values to evaluate at
    eped : float
        Distance of the zeroth peak from the origin
    eped_sigma : float
        Sigma of the zeroth peak, represents electronic noise of the system
    spe : float
        Signal produced by 1 photo-electron
    spe_sigma : float
        Spread in the number of photo-electrons incident on the MAPMT
    lambda_ : float
        Poisson mean (average illumination in p.e.)

    Returns
    -------
    spectrum : ndarray
        The y values of the total spectrum.
    """
    # Obtain pedestal peak
    p_ped = exp(-lambda_)
    spectrum = p_ped * normal_pdf(x, eped, eped_sigma)

    pk_max = 0

    # Loop over the possible total number of cells fired
    for k in range(1, 100):
        p = poisson(k, lambda_)  # Probability to get k avalanches

        # Skip insignificant probabilities
        if p > pk_max:
            pk_max = p
        elif p < 1e-4:
            break

        # Combine spread of pedestal and pe peaks
        pe_sigma = sqrt(k * spe_sigma ** 2 + eped_sigma ** 2)

        # Evaluate probability at each value of x
        spectrum += p * normal_pdf(x, eped + k * spe, pe_sigma)

    return spectrum
