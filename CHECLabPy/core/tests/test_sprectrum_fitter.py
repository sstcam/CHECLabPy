from CHECLabPy.core.spectrum_fitter import SpectrumFitter, SpectrumParameter, \
    SpectrumParameterCollection
from CHECLabPy.stats.pdf import normal_pdf, poisson_logpmf
import numpy as np


class TestSpectrumFitter(SpectrumFitter):
    def __init__(self, n_illuminations, config_path=None):
        super().__init__(n_illuminations, config_path)
        self.parameters = SpectrumParameterCollection([
            SpectrumParameter("mean", 0, (-10, 10), multi=True),
            SpectrumParameter("std", 0.5, (0.1, 1)),
        ], n_illuminations, config_path)
        self.n_bins = 2000
        self.range = (-10, 10)

    @staticmethod
    def _get_spectra(n_illuminations, data_x, lookup, *parameter_values):
        spectra = []
        for i in range(n_illuminations):
            spectrum = normal_pdf(
                data_x,
                parameter_values[lookup[i]['mean']],
                parameter_values[lookup[i]['std']],
            )
            spectra.append(spectrum)
        return spectra

    @staticmethod
    def _get_likelihood(n_illuminations, data_x, data_y, lookup, *parameter_values):
        likelihood = 0
        for i in range(n_illuminations):
            spectrum = normal_pdf(
                data_x,
                parameter_values[lookup[i]['mean']],
                parameter_values[lookup[i]['std']],
            )
            likelihood += np.nansum(-2 * poisson_logpmf(data_y[i], spectrum))
        return likelihood


def test_spectrum_parameter():
    param = SpectrumParameter("test", 1, (0, 2))
    assert param.name == "test"
    assert param.initial == 1
    assert param.limits == (0, 2)
    assert not param.fixed
    assert not param.multi
    param = SpectrumParameter("test", 1, (0, 2), fixed=True, multi=True)
    assert param.fixed
    assert param.multi


def test_spectrum_parameter_collection():
    test1_param = SpectrumParameter("test1_", 1, (0, 2), fixed=True)
    test2_param = SpectrumParameter("test2_", 2, (1, 3), multi=True)
    spectrum_parameter_list = [test1_param, test2_param]
    parameters = SpectrumParameterCollection(spectrum_parameter_list, 2)
    assert parameters.spectrum_parameter_list == spectrum_parameter_list
    assert parameters.test1_ == test1_param
    assert parameters.test2_ == test2_param
    assert parameters.test1_.name == "test1_"
    assert parameters.test1_.initial == 1
    assert parameters.test1_.limits == (0, 2)
    assert parameters.test1_.fixed is True
    assert parameters.test1_.multi is False
    assert parameters.test2_.name == "test2_"
    assert parameters.test2_.initial == 2
    assert parameters.test2_.limits == (1, 3)
    assert parameters.test2_.fixed is False
    assert parameters.test2_.multi is True
    assert parameters.parameter_names == ["test1_", "test2_0", "test2_1"]
    assert parameters.minuit_kwargs == dict(
        test1_=1,
        test2_0=2,
        test2_1=2,
        limit_test1_=(0, 2),
        limit_test2_0=(1, 3),
        limit_test2_1=(1, 3),
        fix_test1_=True,
        fix_test2_0=False,
        fix_test2_1=False,
    )
    assert parameters.lookup == [
        dict(test1_=0, test2_=1),
        dict(test1_=0, test2_=2),
    ]
    assert parameters.lookup_typed[0]["test1_"] == 0
    assert parameters.lookup_typed[0]["test2_"] == 1
    assert parameters.lookup_typed[1]["test1_"] == 0
    assert parameters.lookup_typed[1]["test2_"] == 2

    test1_param = SpectrumParameter("test1_", 5, (4, 6), fixed=False)
    test2_param = SpectrumParameter("test2_", 7, (6, 8), multi=True)
    spectrum_parameter_list = [test1_param, test2_param]
    parameters.update_parameters(spectrum_parameter_list)
    assert parameters.test1_.name == "test1_"
    assert parameters.test1_.initial == 5
    assert parameters.test1_.limits == (4, 6)
    assert parameters.test1_.fixed is False
    assert parameters.test1_.multi is False
    assert parameters.test2_.name == "test2_"
    assert parameters.test2_.initial == 7
    assert parameters.test2_.limits == (6, 8)
    assert parameters.test2_.fixed is False
    assert parameters.test2_.multi is True

    # Test interation over SpectrumParameterCollection
    for param_obj, param_list in zip(parameters, spectrum_parameter_list):
        assert param_obj.name == param_list.name


def test_spectrum_fitter():
    mean0 = 3
    mean1 = 1
    std = 0.6
    random = np.random.RandomState(1)
    charges = [
        random.normal(mean0, std, 10000),
        random.normal(mean1, std, 10000),
    ]
    n_illuminations = len(charges)
    fitter = TestSpectrumFitter(n_illuminations)
    fitter.apply(*charges)

    parameter_values = fitter.fit_result_values
    np.testing.assert_allclose(parameter_values['mean0'], mean0, rtol=1e-2)
    np.testing.assert_allclose(parameter_values['mean1'], mean1, rtol=1e-2)
    np.testing.assert_allclose(parameter_values['std'], std, rtol=1e-2)

    parameter_errors = fitter.fit_result_errors
    assert parameter_errors['mean0'] < 0.1
    assert parameter_errors['mean1'] < 0.1
    assert parameter_errors['std'] < 0.1

    hist_x, hist_y, hist_edges = fitter.charge_histogram
    assert hist_x.size == fitter.n_bins
    assert len(hist_y) == len(charges)
    assert hist_y[0].size == fitter.n_bins
    assert hist_y[1].size == fitter.n_bins
    assert hist_edges.size == fitter.n_bins + 1

    fit_x, fit_y = fitter.fit_result_curve
    assert len(fit_y) == len(charges)
    np.testing.assert_allclose(fit_x[fit_y[0].argmax()], mean0, rtol=1e-2)
    np.testing.assert_allclose(fit_x[fit_y[1].argmax()], mean1, rtol=1e-2)
