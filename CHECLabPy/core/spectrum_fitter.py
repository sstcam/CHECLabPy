from abc import abstractmethod
import iminuit
import numpy as np
from scipy.stats.distributions import poisson
from scipy.stats import chisquare


class SpectrumFitter:
    def __init__(self, n_illuminations):
        """
        Base class for fitters of Single-Photoelectron spectra. Built to
        flexibly handle any number of illuminations simultaneously.

        Parameters
        ----------
        n_illuminations : int
            Number of illuminations to fit simultaneously
        """
        self.hist = None
        self.edges = None
        self.between = None
        self.coeff = None
        self.p0 = None

        self.nbins = 100
        self.range = [-10, 100]

        self.coeff_names = []
        self.multi_coeff = []
        self.initial = dict()
        self.limits = dict()
        self.fix = dict()

        self.n_illuminations = n_illuminations

    @property
    def fit_x(self):
        """
        Default X coordinates for the fit

        Returns
        -------
        ndarray
        """
        return np.linspace(self.edges[0], self.edges[-1], 10*self.edges.size)

    @property
    def fit(self):
        """
        Curve for the current fit result

        Returns
        -------
        ndarray
        """
        return self.fit_function(x=self.fit_x, **self.coeff)

    @property
    def n_coeff(self):
        """
        Number of free parameters in the fit

        Returns
        -------
        int
        """
        return len(self.coeff) - sum(self.fix.values())

    @property
    def chi2(self):
        """
        Chi-squared statistic

        Returns
        -------
        float
        """
        h = np.hstack(self.hist)
        f = np.hstack(self.fit_function(x=self.between, **self.coeff))
        b = h >= 5
        h = h[b]
        f = f[b]
        chi2 = np.sum(np.power(h - f, 2)/f)
        return chi2

    @property
    def dof(self):
        """
        Degrees of freedom based on the histogram and the number of free
        parameters

        Returns
        -------
        int
        """
        h = np.hstack(self.hist)
        n = h[h >= 5].size
        m = self.n_coeff
        dof = n - 1 - m
        return dof

    @property
    def reduced_chi2(self):
        """
        Reduced Chi-Squared statistic

        Returns
        -------
        float
        """
        return self.chi2 / self.dof

    @property
    def p_value(self):
        """
        The probability value for the resulting fit of obtaining a spectrum
        equal to or more extreme than what was actually measured.

        In this context, a high p-value indicates a good fit.

        Returns
        -------
        float
        """
        h = np.hstack(self.hist)
        f = np.hstack(self.fit_function(x=self.between, **self.coeff))
        b = h >= 5
        h = h[b]
        f = f[b]
        return chisquare(h, f, self.n_coeff).pvalue

    def add_parameter(self, name, inital, lower, upper,
                      fix=False, multi=False):
        """
        Add a new parameter for this particular fit function

        Parameters
        ----------
        name : str
            Name of the parameter
        inital : float
            Initial value for the parameter
        lower : float
            Lower limit for the parameter
        upper : float
            Upper limit for the parameter
        fix : bool
            Specify if the parameter should be fixed
        multi : bool
            Specify if the parameter should be duplicated for additional
            illuminations
        """
        if not multi:
            self.coeff_names.append(name)
            self.initial[name] = inital
            self.limits["limit_" + name] = (lower, upper)
            self.fix["fix_" + name] = fix
        else:
            self.multi_coeff.append(name)
            for i in range(self.n_illuminations):
                name_i = name + str(i)
                self.coeff_names.append(name_i)
                self.initial[name_i] = inital
                self.limits["limit_" + name_i] = (lower, upper)
                self.fix["fix_" + name_i] = fix
        ds = "minimize_function(" + ", ".join(self.coeff_names) + ")"
        self.minimize_function.__func__.__doc__ = ds

    def apply(self, *spectrum):
        """
        Fit the spectra

        Parameters
        ----------
        spectrum : list[ndarray]
            A list of the spectra to fit. Should have a length equal to the
            self.n_illuminations.
        """
        assert len(spectrum) == self.n_illuminations
        bins = self.nbins
        range_ = self.range
        self.hist = []
        for i in range(self.n_illuminations):
            h, e, b = self.get_histogram(spectrum[i], bins, range_)
            self.hist.append(h)
            self.edges = e
            self.between = b

        self._perform_fit()

    @staticmethod
    def get_histogram(spectrum, bins, range_):
        """
        Obtain a histogram for the spectrum.

        Look at `np.histogram` documentation for further info on Parameters.

        Parameters
        ----------
        spectrum : ndarray
        bins
        range_

        Returns
        -------
        hist : ndarray
            The histogram
        edges : ndarray
            Edges of the histogram
        between : ndarray
            X values of the middle of each bin
        """
        hist, edges = np.histogram(spectrum, bins=bins, range=range_)
        between = (edges[1:] + edges[:-1]) / 2

        # zero = between[np.argmax(hist)]
        # spectrum -= zero
        #
        # hist, edges = np.histogram(spectrum, bins=bins, range=range_)
        # between = (edges[1:] + edges[:-1]) / 2

        return hist, edges, between

    def get_histogram_summed(self, spectra, bins, range_):
        """
        Get the histogram including the spectra from all the illuminations.

        Look at `np.histogram` documentation for further info on Parameters.

        Parameters
        ----------
        spectra : list
        bins
        range_

        Returns
        -------
        hist : ndarray
            The histogram
        edges : ndarray
            Edges of the histogram
        between : ndarray
            X values of the middle of each bin
        """
        spectra_stacked = np.hstack(spectra)
        hist, edge, between = self.get_histogram(spectra_stacked, bins, range_)
        return hist, edge, between

    def get_fit_summed(self, x, **coeff):
        """
        Get the summed fit for all the illuminations.

        Parameters
        ----------
        x : ndarray
            X values for the fit
        coeff
            The fit coefficients to apply to the fit function.

        Returns
        -------
        ndarray
        """
        return np.sum(self.fit_function(x, **coeff), 0)

    def _perform_fit(self):
        """
        Run iminuit on the fit function to find the best fit
        """
        self.coeff = {}
        self.p0 = self.initial.copy()
        limits = self.limits.copy()
        fix = self.fix.copy()
        self.prepare_params(self.p0, limits, fix)

        m0 = iminuit.Minuit(self.minimize_function, **self.p0, **limits, **fix,
                            print_level=0, pedantic=False, throw_nan=False,
                            forced_parameters=self.coeff_names)
        m0.migrad()

        self.coeff = m0.values

    def prepare_params(self, p0, limits, fix):
        """
        Apply some automation to the contents of initial, limits, and fix
        dictionaries.

        Parameters
        ----------
        p0 : dict
            Initial values dict
        limits : dict
            Dict containing the limits for each parameter
        fix : dict
            Dict containing which parameters should be fixed
        """
        pass

    def minimize_function(self, *args):
        """
        Function which calculates the likelihood to be minimised.

        Parameters
        ----------
        args
            The values to apply to the fit function.

        Returns
        -------
        likelihood : float
        """
        kwargs = dict(zip(self.coeff_names, args))
        x = self.between
        y = self.hist
        p = self.fit_function(x, **kwargs)
        like = [-2 * poisson._logpmf(y[i], p[i])
                for i in range(self.n_illuminations)]
        like = np.hstack(like)
        return np.nansum(like)

    def fit_function(self, x, **kwargs):
        """
        Function which applies the parameters for each illumination and
        returns the resulting curves.

        Parameters
        ----------
        x : ndarray
            X values
        kwargs
            The values to apply to the fit function

        Returns
        -------

        """
        p = []
        for i in range(self.n_illuminations):
            for coeff in self.multi_coeff:
                kwargs[coeff] = kwargs[coeff + str(i)]
            p.append(self._fit(x, **kwargs))
        return p

    @staticmethod
    @abstractmethod
    def _fit(x, **kwargs):
        """
        Define the low-level function to be used in the fit

        Parameters
        ----------
        x : ndarray
            X values
        kwargs
            The values to apply to the fit function

        Returns
        -------
        ndarray
        """
        pass
