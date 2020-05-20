import numpy as np
from numba import njit
from math import exp, pow, sqrt
from CHECLabPy.stats.pdf import binom, poisson, normal_pdf, poisson_logpmf
from CHECLabPy.core.spectrum_fitter import SpectrumFitter, SpectrumParameter, \
    SpectrumParameterCollection


class SiPMGentileFitter(SpectrumFitter):
    def __init__(self, n_illuminations):
        """
        SpectrumFitter which uses the SiPM fitting formula from Gentile 2010
        http://adsabs.harvard.edu/abs/2010arXiv1006.3263G
        """
        super().__init__(n_illuminations)

        self.parameters = SpectrumParameterCollection([
            SpectrumParameter("eped", 0, (-10, 10)),
            SpectrumParameter("eped_sigma", 9, (0, 20)),
            SpectrumParameter("spe", 25, (0, 40)),
            SpectrumParameter("spe_sigma", 2, (0, 20)),
            SpectrumParameter("opct", 0.4, (0, 1)),
            SpectrumParameter("lambda_", 0.7, (0, 6), multi=True),
        ], n_illuminations)
        self.n_bins = 100
        self.range = (-30, 200)

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
    return sipm_gentile_spe(
        x=data_x,
        eped=parameter_values[lookup["eped"]],
        eped_sigma=parameter_values[lookup["eped_sigma"]],
        spe=parameter_values[lookup["spe"]],
        spe_sigma=parameter_values[lookup["spe_sigma"]],
        opct=parameter_values[lookup["opct"]],
        lambda_=parameter_values[lookup["lambda_"]],
    )


@njit(fastmath=True)
def sipm_gentile_spe(x, eped, eped_sigma, spe, spe_sigma, opct, lambda_):
    """
    Fit for the SPE spectrum of a SiPM

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
    opct : float
        Optical crosstalk probability
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
        pk = 0
        for j in range(1, k+1):
            pj = poisson(j, lambda_)  # Probability for j initial fired cells

            # Skip insignificant probabilities
            if pj < 1e-4:
                continue

            # Sum the probability from the possible combinations which result
            # in a total of k fired cells to get the total probability of k
            # fired cells
            pk += pj * pow(1-opct, j) * pow(opct, k-j) * binom(k-1, j-1)

        # Skip insignificant probabilities
        if pk > pk_max:
            pk_max = pk
        elif pk < 1e-4:
            break

        # Combine spread of pedestal and pe peaks
        pe_sigma = sqrt(k * spe_sigma ** 2 + eped_sigma ** 2)

        # Evaluate probability at each value of x
        spectrum += pk * normal_pdf(x, eped + k * spe, pe_sigma)

    return spectrum
