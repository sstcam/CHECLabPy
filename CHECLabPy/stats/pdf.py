from numba import vectorize, int64, float64, float32
from math import lgamma, exp, sqrt, log, pi, isnan

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


@vectorize([float64(float32, float64, float64)], fastmath=True)
def normal_pdf(x, mean, std_deviation):
    """
    Obtain the normal PDF.

    The result is the probability of obseving a value at a position x, for a
    normal distribution described by a mean m and a standard deviation s.

    Source: https://stackoverflow.com/questions/10847007/using-the-gaussian-probability-density-function-in-c
    """
    u = (x - mean) / std_deviation
    return exp(-0.5 * u ** 2) / (SQRT2PI * std_deviation)


@vectorize([float64(float32, float64)], fastmath=True)
def xlogy(x, y):
    if x == 0 and not isnan(y):
        return 0
    else:
        return x * log(y)


@vectorize([float64(float32, float64)], fastmath=True)
def poisson_logpmf(k, mu):
    return xlogy(k, mu) - lgamma(k + 1) - mu
