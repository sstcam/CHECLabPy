from CHECLabPy.stats.pdf import binom, poisson, normal_pdf, xlogy, poisson_logpmf
import scipy.special as scipy_special
import scipy.stats as scipy_stats
import numpy as np
from numpy.testing import assert_allclose


def test_binom():
    n = np.arange(100)
    k = np.arange(100)
    assert_allclose(binom(n, k), scipy_special.binom(n, k))


def test_poisson():
    k = np.arange(1, 100)
    mu = np.arange(1, 100)
    assert_allclose(poisson(k, mu), scipy_stats.poisson.pmf(k, mu))


def test_normal_pdf():
    x = np.linspace(-10, 10, 100, dtype=np.float32)
    mean = 0
    std = 5
    assert_allclose(normal_pdf(x, mean, std), scipy_stats.norm.pdf(x, mean, std))


def test_xlogy():
    x = np.arange(100, dtype=np.float32)
    y = np.arange(100, dtype=np.float32)
    assert_allclose(xlogy(x, y), scipy_special.xlogy(x, y))


def test_poisson_pmf():
    k = np.arange(100, dtype=np.float32)
    mu = np.arange(100, dtype=np.float32)
    assert_allclose(poisson_logpmf(k, mu), scipy_stats.poisson.logpmf(k, mu), rtol=1e-5)
