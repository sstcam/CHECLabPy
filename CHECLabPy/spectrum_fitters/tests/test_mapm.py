from CHECLabPy.spectrum_fitters.mapm import mapm_spe, \
    calculate_spectrum, MAPMFitter, SpectrumParameter
import numpy as np
from numpy.testing import assert_allclose
from numba import typed


def test_mapm_spe():
    x = np.linspace(-1, 20, 1000, dtype=np.float32)
    y = mapm_spe(x, 0., 0.2, 1., 0.1, 1.)
    np.testing.assert_allclose(np.trapz(y, x), 1, rtol=1e-3)


def test_calculate_spectrum():
    x = np.linspace(-1, 20, 1000, dtype=np.float32)
    parameter_values = [0., 0.2, 1., 0.1, 1.]
    lookup = typed.Dict()
    lookup['eped'] = 0
    lookup['eped_sigma'] = 1
    lookup['spe'] = 2
    lookup['spe_sigma'] = 3
    lookup['lambda_'] = 4
    y = calculate_spectrum(x, lookup, *parameter_values)
    np.testing.assert_allclose(np.trapz(y, x), 1, rtol=1e-3)


def test_mapm_fitter():
    # Define SPE
    params = dict(
        eped=-0.5,
        eped_sigma=0.2,
        spe=2,
        spe_sigma=0.1,
    )
    lambda_values = [0.5, 0.7, 0.9]

    # Get charges
    random = np.random.RandomState(2)
    pdf_x = np.linspace(-5, 50, 10000, dtype=np.float32)
    pdf_y = []
    charges = []
    for lambda_ in lambda_values:
        pdf = mapm_spe(pdf_x, lambda_=lambda_, **params)
        pdf /= pdf.sum()
        charge = random.choice(pdf_x, 30000, p=pdf)
        pdf_y.append(pdf)
        charges.append(charge)

    # Create Fitter class
    n_illuminations = len(lambda_values)
    fitter = MAPMFitter(n_illuminations=n_illuminations)

    # Update Fit Parameters
    spectrum_parameter_list = [
        SpectrumParameter("eped", 0, (-10, 10)),
        SpectrumParameter("eped_sigma", 0.5, (0.01, 1)),
        SpectrumParameter("spe", 1, (0.1, 5)),
        SpectrumParameter("spe_sigma", 0.5, (0.01, 1)),
        SpectrumParameter("lambda_", 0.7, (0.001, 3), multi=True),
    ]
    fitter.parameters.update_parameters(spectrum_parameter_list)
    fitter.range = (-1, 7)
    fitter.n_bins = 1000

    fitter.apply(*charges)
    parameter_values = fitter.fit_result_values

    rtol = 1e-2
    assert_allclose(parameter_values["eped"], params["eped"], rtol=rtol)
    assert_allclose(parameter_values["eped_sigma"], params["eped_sigma"], rtol=rtol)
    assert_allclose(parameter_values["spe"], params["spe"], rtol=rtol)
    assert_allclose(parameter_values["spe_sigma"], params["spe_sigma"], rtol=rtol)
    assert_allclose(parameter_values["lambda_0"], lambda_values[0], rtol=rtol)
    assert_allclose(parameter_values["lambda_1"], lambda_values[1], rtol=rtol)
    assert_allclose(parameter_values["lambda_2"], lambda_values[2], rtol=rtol)
