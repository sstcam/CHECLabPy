import numpy as np
from numba import jit, prange
from math import lgamma, exp, sqrt, log
from CHECLabPy.core.spectrum_fitter import SpectrumFitter


class MAPMFitter(SpectrumFitter):
    def __init__(self, n_illuminations, config_path=None):
        """
        SpectrumFitter which uses the MAPM fitting formula

        Parameters
        ----------
        n_illuminations : int
            Number of illuminations to fit simultaneously
        """
        super().__init__(n_illuminations, config_path)

        self.nbins = 100
        self.range = [-40, 150]

        self.add_parameter("norm", None, 0, 100000, fix=True, multi=True)
        self.add_parameter("eped", 0, -10, 10)
        self.add_parameter("eped_sigma", 9, 2, 20)
        self.add_parameter("spe", 25, 5, 30)
        self.add_parameter("spe_sigma", 2, 1, 20)
        self.add_parameter("lambda_", 0.2, 0.001, 6, multi=True)

    def _prepare_params(self, p0, limits, fix):
        for i in range(self.n_illuminations):
            norm = 'norm{}'.format(i)
            if p0[norm] is None:
                p0[norm] = np.trapz(self.hist[i], self.between)

    @staticmethod
    def _fit(x, **kwargs):
        return mapm_spe_fit(x, **kwargs)


SQRT2PI = np.sqrt(2.0 * np.pi)


@jit(nopython=True, fastmath=True, parallel=True)
def poisson(k, mu):
    return exp(k * log(mu) - mu - lgamma(k + 1))


@jit(nopython=True, fastmath=True, parallel=True)
def normal_pdf(x, mean=0, std_deviation=1):
    u = (x - mean) / std_deviation
    return np.exp(-0.5 * u ** 2) / (SQRT2PI * std_deviation)


@jit(fastmath=True, parallel=True)
def mapm_nb(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_):
    # Obtain pedestal peak
    p_ped = exp(-lambda_)
    ped_signal = norm * p_ped * normal_pdf(x, eped, eped_sigma)

    pe_signal = 0
    found = False

    # Loop over the possible total number of cells fired
    for k in prange(1, 250):
        p = poisson(k, lambda_)  # Probability to get k avalanches

        # Skip insignificant probabilities
        if (not found) & (p < 1e-5):
            continue
        if found & (p < 1e-5):
            break
        found = True

        # Combine spread of pedestal and pe peaks
        pe_sigma = sqrt(k * spe_sigma ** 2 + eped_sigma ** 2)

        # Evaluate probability at each value of x
        pe_signal += norm * p * normal_pdf(x, eped + k * spe, pe_sigma)

    return ped_signal + pe_signal


def mapm_spe_fit(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, **kwargs):
    """
    Fit for the SPE spectrum of a MAPM

    Parameters
    ----------
    x : 1darray
        The x values to evaluate at
    norm : float
        Integral of the zeroth peak in the distribution, represents p(0)
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
    signal : ndarray
        The y values of the total signal.
    """
    return mapm_nb(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_)
