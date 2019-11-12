from abc import abstractmethod
import iminuit
from iminuit.iminuit_warnings import HesseFailedWarning
import numpy as np
from numba import typed, types
import yaml
import warnings
from functools import partial


class SpectrumParameter:
    def __init__(self, name: str, inital: float, limits: tuple,
                 fixed: bool = False, multi: bool = False):
        """
        Paramters defining a spectrum description

        TODO: Can be converted into a dataclass once 3.7 is adopted as
            minimum Python version
        """
        self.name = name
        self.initial = inital
        self.limits = limits
        self.fixed = fixed
        self.multi = multi


class SpectrumParameterCollection:
    def __init__(self, spectrum_parameter_list, n_illuminations, config_path=None):
        """
        Class to handle the bookkeeping of the SpectrumParameters for a
        SpectrumFitter

        Parameters
        ----------
        spectrum_parameter_list : list
        n_illuminations : int
        config_path : str
        """
        self.spectrum_parameter_list = spectrum_parameter_list
        self.n_illuminations = n_illuminations
        self._lookup_typed = None
        for param in spectrum_parameter_list:
            setattr(self, param.name, param)

        # self.load_config(config_path)
        self._prepare_parameters()

    def __len__(self):
        return self.spectrum_parameter_list.__len__()

    def __iter__(self):
        return self.spectrum_parameter_list.__iter__()

    @property
    def lookup_typed(self):
        """
        Numba typed version of `self.lookup`

        TODO: Unikely to be necessary in future versions of numba (0.47)

        Returns
        -------
        lookup_typed : numba.typed.List[numba.typed.Dict]
        """
        if self._lookup_typed is None:
            self._lookup_typed = typed.List()
            for i in range(self.n_illuminations):
                self._lookup_typed.append(typed.Dict.empty(
                    key_type=types.unicode_type,
                    value_type=types.intp
                ))
                for key, value in self.lookup[i].items():
                    self._lookup_typed[i][key] = value
        return self._lookup_typed

    def _prepare_parameters(self):
        self.parameter_names = []
        self.minuit_kwargs = {}
        self._lookup_typed = None  # Reset
        self.lookup = [dict() for _ in range(self.n_illuminations)]

        lookup_i = 0
        for param in self.spectrum_parameter_list:
            if not param.multi:
                name = param.name
                self.parameter_names.append(name)
                self.minuit_kwargs[name] = param.initial
                self.minuit_kwargs["limit_" + name] = param.limits
                self.minuit_kwargs["fix_" + name] = param.fixed
                for i in range(self.n_illuminations):
                    self.lookup[i][param.name] = lookup_i
                lookup_i += 1
            else:
                for i in range(self.n_illuminations):
                    name = param.name + str(i)
                    self.parameter_names.append(name)
                    self.minuit_kwargs[name] = param.initial
                    self.minuit_kwargs["limit_" + name] = param.limits
                    self.minuit_kwargs["fix_" + name] = param.fixed
                    self.lookup[i][param.name] = lookup_i
                    lookup_i += 1

    def update_parameters(self, spectrum_parameter_list):
        """
        Update the existing SpectrumParameters attached to this class with new
        values.

        Parameters
        ----------
        spectrum_parameter_list : list
        """
        for param in spectrum_parameter_list:
            getattr(self, param.name).initial = param.initial
            getattr(self, param.name).limits = param.limits
            getattr(self, param.name).fixed = param.fixed
            getattr(self, param.name).multi = param.multi

        self._prepare_parameters()

    # def load_config(self, path):
    #     """
    #     Load a YAML configuration file to set initial fitting parameters
    #
    #     Parameters
    #     ----------
    #     path : str
    #     """
    #     print("Loading SpectrumFitter configuration from: {}".format(path))
    #     with open(path, 'r') as file:
    #         d = yaml.safe_load(file)
    #         if d is None:
    #             return
    #         self.n_bins = d.pop('nbins', self.n_bins)
    #         self.range = d.pop('range', self.range)
    #         for c in self.parameter_names:
    #             if 'initial' in d:
    #                 ini = c
    #                 self.initial[ini] = d['initial'].pop(c, self.initial[ini])
    #                 if self.initial[ini] is not None:
    #                     self.initial[ini] = float(self.initial[ini])
    #             if 'limits' in d:
    #                 lim = "limit_" + c
    #                 list_ = d['limits'].pop(c, self.limits[lim])
    #                 if isinstance(list_, list):
    #                     self.limits[lim] = tuple([float(l) for l in list_])
    #                 else:
    #                     self.limits[lim] = list_
    #             if 'fix' in d:
    #                 fix = "fix_" + c
    #                 self.fix[fix] = d['fix'].pop(c, self.fix[fix])
    #         if 'initial' in d and not d['initial']:
    #             d.pop('initial')
    #         if 'limits' in d and not d['limits']:
    #             d.pop('limits')
    #         if 'fix' in d and not d['fix']:
    #             d.pop('fix')
    #         if d:
    #             print("WARNING: Unused SpectrumFitter config parameters:")
    #             print(d)
    #
    # def save_config(self, path):
    #     """
    #     Save the configuration of the fit. If the fit has already been
    #     performed, the fit coefficients will be included as the initial
    #     coefficients
    #
    #     Parameters
    #     ----------
    #     path : str
    #         Path to save the configuration file to
    #     """
    #     print("Writing SpectrumFitter configuration to: {}".format(path))
    #     initial = dict()
    #     limits = dict()
    #     fix = dict()
    #     coeff_dict = self.coeff if self.coeff else self.initial
    #     for c, val in coeff_dict.items():
    #         initial[c] = val
    #     for c, val in self.limits.items():
    #         limits[c.replace("limit_", "")] = val
    #     for c, val in self.fix.items():
    #         fix[c.replace("fix_", "")] = val
    #     data = dict(
    #         n_bins=self.n_bins,
    #         range=self.range,
    #         initial=initial,
    #         limits=limits,
    #         fix=fix
    #     )
    #     with open(path, 'w') as outfile:
    #         yaml.safe_dump(data, outfile, default_flow_style=False)


class SpectrumFitter:
    def __init__(self, n_illuminations, config_path=None):
        """
        Base class for fitters of Single-Photoelectron spectra. Built to
        flexibly handle any number of illuminations simultaneously.

        Parameters
        ----------
        n_illuminations : int
            Number of illuminations to fit simultaneously
        config_path : str
            Path to JAML config file
        """
        self.n_illuminations = n_illuminations

        self.parameters = None
        self.n_bins = 100
        self.range = (-10, 100)

        self.charge_hist_x = None
        self.charge_hist_y = None
        self.charge_hist_y_typed = None
        self.charge_hist_edges = None
        self.fit_result_values = None
        self.fit_result_errors = None

        # if config_path:
        #     self.load_config(config_path)

    @property
    def fit_result_curve(self):
        """
        Obtain the curve resulting from the fit result

        Returns
        -------
        fit_x : ndarray
        fit_y : list[ndarray]
            Y values for each illumination
        """
        fit_x = np.linspace(
            self.range[0], self.range[1], self.n_bins*10, dtype=np.float32
        )
        fit_y = self._get_spectra(
            self.n_illuminations,
            fit_x,
            self.parameters.lookup_typed,
            *self.fit_result_values.values()
        )
        for i in range(self.n_illuminations):
            hist_area = np.trapz(self.charge_hist_y[i], self.charge_hist_x)
            fit_y[i] *= hist_area / np.trapz(fit_y[i], fit_x)
        return fit_x, fit_y

    @property
    def charge_histogram(self):
        """
        Obtain the histagram variables for the latest application

        Returns
        -------
        charge_hist_x : ndarray
        charge_hist_y : list[ndarray]
        charge_hist_edges : ndarray
        """
        return self.charge_hist_x, self.charge_hist_y, self.charge_hist_edges

    def apply(self, *charges):
        """
        Create charge histogram and fit it with a model of the spectrum

        Resulting parameters values are found in `self.fit_result_values`

        Parameters
        ----------
        charges : list[ndarray]
            A list of the charges to fit. Should have a length equal to the
            self.n_illuminations.
        """
        assert len(charges) == self.n_illuminations
        self.charge_hist_y = []
        self.charge_hist_y_typed = typed.List()
        for i in range(self.n_illuminations):
            hist, edges = np.histogram(
                charges[i], bins=self.n_bins, range=self.range
            )
            between = (edges[1:] + edges[:-1]) / 2

            self.charge_hist_x = between.astype(np.float32)
            self.charge_hist_y.append(hist.astype(np.float32))
            self.charge_hist_y_typed.append(hist.astype(np.float32))
            self.charge_hist_edges = edges.astype(np.float32)

        m0 = iminuit.Minuit(
            self._minimize_function, **self.parameters.minuit_kwargs,
            print_level=0, pedantic=False, throw_nan=True, errordef=1,
            forced_parameters=self.parameters.parameter_names
        )
        m0.migrad()
        self.fit_result_values = m0.values

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', HesseFailedWarning)
            m0.hesse()
        self.fit_result_errors = m0.errors

    def _minimize_function(self, *parameter_values):
        try:
            return self._get_likelihood(
                self.n_illuminations,
                self.charge_hist_x,
                self.charge_hist_y_typed,
                self.parameters.lookup_typed,
                *parameter_values,
            )
        except ZeroDivisionError:
            return np.inf

    @staticmethod
    @abstractmethod
    def _get_spectra(n_illuminations, data_x, lookup, *parameter_values):
        """
        Abstract method to be defined by the SpectrumFitter subclass

        Define how the spectrum and likelihood is calculated, and combined for
        each illumination

        Parameters
        ----------
        n_illuminations : int
            Number of illuminations to fit simultaneously
        data_x : ndarray
            Datapoints to calculate the spectra at
        lookup : numba.typed.List[numba.typed.Dict]
            Lookup for position of each spectrum parameter in the
            parameters tuple
        parameter_values: tuple
            Values for each parameter of the spectrum

        Returns
        -------
        spectra : list[ndarray]
            The calculated spectrum for each illumination corresponding to the
            specified parameter_values
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_likelihood(n_illuminations, data_x, data_y, lookup, *parameter_values):
        """
        Abstract method to be defined by the SpectrumFitter subclass

        Define how the spectrum and likelihood is calculated, and combined for
        each illumination

        Parameters
        ----------
        n_illuminations : int
            Number of illuminations to fit simultaneously
        data_x : ndarray
            Datapoints to calculate the spectra at
        data_y : list[ndarray]
            The Y values to compare against the calculated spectrum to
            calculate the likelihood.
            Shape: (n_illuminations, data_x.size)
        lookup : numba.typed.List[numba.typed.Dict]
            Lookup for position of each spectrum parameter in the
            parameters tuple
        parameter_values: tuple
            Values for each parameter in this iteration of the iminuit minimization

        Returns
        -------
        likelihood : float
            The likelihood to be minimised, resulting from the comparison
            between the calculated spectrum and data_y.
        """
        pass
