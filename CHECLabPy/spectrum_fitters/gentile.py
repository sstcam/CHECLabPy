import numpy as np
from numba import njit, prange, vectorize, int64, float64
from math import lgamma, exp, pow, sqrt, log, pi
from CHECLabPy.core.spectrum_fitter import SpectrumFitter


class GentileFitter(SpectrumFitter):
    def __init__(self, n_illuminations, config_path=None):
        """
        SpectrumFitter which uses the SiPM fitting formula from Gentile 2010
        http://adsabs.harvard.edu/abs/2010arXiv1006.3263G

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
        self.add_parameter("spe", 25, 15, 40)
        self.add_parameter("spe_sigma", 2, 1, 20)
        self.add_parameter("lambda_", 0.7, 0.001, 3, multi=True)
        self.add_parameter("opct", 0.4, 0.01, 0.8)
        self.add_parameter("pap", 0.09, 0.01, 0.8)
        self.add_parameter("dap", 0.5, 0, 0.8)

    def _prepare_params(self, p0, limits, fix):
        for i in range(self.n_illuminations):
            norm = 'norm{}'.format(i)
            if p0[norm] is None:
                p0[norm] = np.trapz(self.hist[i], self.between)

    @staticmethod
    def _fit(x, **kwargs):
        return sipm_spe_fit(x, **kwargs)


SQRT2PI = sqrt(2.0 * pi)


@vectorize([float64(int64, int64)], fastmath=True)
def binom(n, k):
    """
    Obtain the binomial coefficient, using a definition that is mathematically
    equivalent but numerically stable to avoid arithmetic overflow.

    The result of this method is "n choose k", the number of ways choose an
    (unordered) subset of k elements from a fixed set of n elements.

    Source: https://en.wikipedia.org/wiki/Binomial_coefficient
    """
    return exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1))


@vectorize([float64(int64, float64)], fastmath=True)
def poisson(k, mu):
    """
    Obtain the poisson PMF, using a definition that is mathematically
    equivalent but numerically stable to avoid arithmetic overflow.

    The result is the probability of observing k events for an average number
    of events per interval, lambda_.

    Source: https://en.wikipedia.org/wiki/Poisson_distribution
    """
    return exp(k * log(mu) - mu - lgamma(k + 1))


@vectorize([float64(float64, float64, float64)], fastmath=True)
def normal_pdf(x, mean=0, std_deviation=1):
    """
    Obtain the normal PDF.

    The result is the probability of obseving a value at a position x, for a
    normal distribution described by a mean m and a standard deviation s.

    Source: https://stackoverflow.com/questions/10847007/using-the-gaussian-probability-density-function-in-c
    """
    u = (x - mean) / std_deviation
    return exp(-0.5 * u ** 2) / (SQRT2PI * std_deviation)


@njit(fastmath=True, parallel=True)
def sipm_nb(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap):
    sap = spe_sigma  # Assume the sigma of afterpulses is the same

    # Obtain pedestal peak
    p_ped = exp(-lambda_)
    ped_signal = norm * p_ped * normal_pdf(x, eped, eped_sigma)

    pe_signal = np.zeros(x.size)
    found = False

    # Loop over the possible total number of cells fired
    for k in prange(1, 250):
        pk = 0
        for j in prange(1, k+1):
            pj = poisson(j, lambda_)  # Probability for j initial fired cells

            # Skip insignificant probabilities
            if pj < 1e-4:
                continue

            # Sum the probability from the possible combinations which result
            # in a total of k fired cells to get the total probability of k
            # fired cells
            pk += pj * pow(1-opct, j) * pow(opct, k-j) * binom(k-1, j-1)

        # Skip insignificant probabilities
        if (not found) & (pk < 1e-4):
            continue
        if found & (pk < 1e-4):
            break
        found = True

        # Consider probability of afterpulses
        papk = pow(1 - pap, k)
        p0ap = pk * papk
        pap1 = pk * (1-papk) * papk

        # Combine spread of pedestal and pe (and afterpulse) peaks
        pe_sigma = sqrt(k * spe_sigma ** 2 + eped_sigma ** 2)
        ap_sigma = sqrt(k * sap ** 2 + eped_sigma ** 2)

        # Evaluate probability at each value of x
        pe_signal += norm * (
                p0ap * normal_pdf(x, eped + k * spe, pe_sigma) +
                pap1 * normal_pdf(x, eped + k * spe * (1 - dap), ap_sigma)
        )

    return ped_signal + pe_signal


def sipm_spe_fit(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_,
                 opct, pap, dap, **kwargs):
    """
    Fit for the SPE spectrum of a SiPM

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
    opct : float
        Optical crosstalk probability
    pap : float
        Afterpulse probability
    dap : float
        The first distance of the after-pulse Gaussians from the main peaks

    Returns
    -------
    signal : ndarray
        The y values of the total signal.
    """
    return sipm_nb(x,  norm, eped, eped_sigma, spe, spe_sigma, lambda_,
                   opct, pap, dap)
