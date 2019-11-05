"""
This module contains objects to characterize the pure-component adsorption
loading surface from experimental or simulated data. These will be fed into the
IAST functions in pyiast.py. A loading surface is a function of both temperature
and pressure where an isotherm is only a function of pressure.
"""
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range

__author__ = 'Andrew Sun'
__all__ = [
    "ModelIsotherm", "InterpolatorIsotherm", "plot_isotherm", "_MODELS",
    "_MODEL_PARAMS", "_VERSION", "LangmuirIsotherm", "SipsIsotherm",
    "QuadraticIsotherm"
]
# last line includes depreciated classes

import scipy.optimize
from scipy.interpolate import interp1d
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd

# ! version
_VERSION = "1.4.3a1"

# ! list of models implemented in pyIAST
_MODELS = [
    "Langmuir", "Quadratic", "BET", "Henry", "TemkinApprox", "DSLangmuir"
]

# ! dictionary of parameters involved in each model
_MODEL_PARAMS = {
    "Langmuir": {
        "M": np.nan,
        "K_a": np.nan,
        "K_b": np.nan
    },
    "Quadratic": {
        "M": np.nan,
        "Ka_a": np.nan,
        "Ka_b": np.nan,
        "Kb_a": np.nan,
        "Kb_b": np.nan
    },
    "BET": {
        "M": np.nan,
        "Ka_a": np.nan,
        "Ka_b": np.nan,
        "Kb_a": np.nan,
        "Kb_b": np.nan
    },
    "DSLangmuir": {
        "M1": np.nan,
        "K1_a": np.nan,
        "K1_b": np.nan,
        "M2": np.nan,
        "K2_a": np.nan,
        "K2_b": np.nan
    },
    "TemkinApprox": {
        "M": np.nan,
        "K_a": np.nan,
        "K_b": np.nan,
        "theta": np.nan
    },
    "Henry": {
        "KH_a": np.nan,
        "KH_b": np.nan
    }
}

def arrhenius(a, b, temperature):
    return np.exp(a + b / temperature)
    

def get_default_guess_params(model, df, pressure_key, loading_key, temperature_key):
    """
    Get dictionary of default parameters for starting guesses in data fitting
    routine.

    The philosophy behind the default starting guess is that (1) the saturation
    loading is close to the highest loading observed in the data, and (2) the
    default assumption is a Langmuir isotherm.

    Reminder: pass your own guess via `param_guess` in instantiation if these
    default guesses do not lead to a converged set of parameters.

    :param model: String name of analytical model
    :param df: DataFrame adsorption isotherm data
    :param pressure_key: String key for pressure column in df
    :param loading_key: String key for loading column in df
    """
    # guess saturation loading to 10% more than highest loading
    saturation_loading = 1.1 * df[loading_key].max()
    # use linear regression for loading/(saturation*pressure)
    # and temperature to find initial K_a and K_b
    #   pressure point (but not zero)
    df_nonzero = df[df[loading_key] != 0.0]
    k_init = df_nonzero[loading_key] / df_nonzero[pressure_key] / saturation_loading
    X = np.vstack([np.ones(df_nonzero.shape[0]), 1 / df_nonzero[temperature_key]]).T
    params = np.linalg.pinv(X) @ np.log(k_init)
    arrhenius_a = params[0]
    arrhenius_b = params[1]

    if model == "Langmuir":
        return {"M": saturation_loading, "K_a": arrhenius_a, "K_b": arrhenius_b}

    if model == "Quadratic":
        # Quadratic = Langmuir when Kb = Ka^2. This is our default assumption.
        # Also, M is half of the saturation loading in the Quadratic model.
        return {
            "M": saturation_loading / 2.0,
            "Ka_a": arrhenius_a,
            "Ka_b": arrhenius_b,
            "Kb_a": arrhenius_a ** 2.0,
            "Kb_b": arrhenius_b
        }

    if model == "BET":
        # BET = Langmuir when Kb = 0.0. This is our default assumption.
        return {
            "M": saturation_loading,
            "Ka_a": arrhenius_a,
            "Ka_b": arrhenius_b,
            "Kb_a": arrhenius_a * 0.01,
            "Kb_b": arrhenius_b
        }

    if model == "DSLangmuir":
        return {
            "M1": 0.5 * saturation_loading,
            "K1_a": 0.4 * arrhenius_a,
            "K1_b": arrhenius_b,
            "M2": 0.5 * saturation_loading,
            "K2_a": 0.6 * arrhenius_a,
            "K2_b": arrhenius_b
        }

    if model == "Henry":
        return {"KH_a": saturation_loading * arrhenius_a, "KH_b": arrhenius_b}

    if model == "TemkinApprox":
        # equivalent to Langmuir model if theta = 0.0
        return {"M": saturation_loading, "K_a": arrhenius_a, "K_b": arrhenius_b, "theta": 0.0}


class LoadingSurface:
    """
    Class to characterize pure-component isotherm data with an analytical model.
    Data fitting is done during instantiation.

    Models supported are as follows. Here, :math:`L` is the gas uptake,
    :math:`P` is pressure (fugacity technically).

    * Langmuir isotherm model

    .. math::

        L(P) = M\\frac{KP}{1+KP},

    * Quadratic isotherm model

    .. math::

        L(P) = M \\frac{(K_a + 2 K_b P)P}{1+K_aP+K_bP^2}

    * Brunauer-Emmett-Teller (BET) adsorption isotherm

    .. math::

        L(P) = M\\frac{K_A P}{(1-K_B P)(1-K_B P+ K_A P)}

    * Dual-site Langmuir (DSLangmuir) adsorption isotherm

    .. math::

        L(P) = M_1\\frac{K_1 P}{1+K_1 P} +  M_2\\frac{K_2 P}{1+K_2 P}

    * Asymptotic approximation to the Temkin Isotherm
    (see DOI: 10.1039/C3CP55039G)

    .. math::

        L(P) = M\\frac{KP}{1+KP} + M \\theta (\\frac{KP}{1+KP})^2 (\\frac{KP}{1+KP} -1)

    * Henry's law. Only use if your data is linear, and do not necessarily trust
      IAST results from Henry's law if the result required an extrapolation
      of your data; Henry's law is unrealistic because the adsorption sites
      will saturate at higher pressures.

    .. math::

        L(P) = K_H P

    """



    def __init__(self,
                 df,
                 loading_key=None,
                 pressure_key=None,
                 temperature_key=None,
                 model=None,
                 param_guess=None,
                 optimization_method="Nelder-Mead"):
        """
        Instantiation. A `ModelIsotherm` class is instantiated by passing it the
        pure-component adsorption isotherm in the form of a Pandas DataFrame.
        The least squares data fitting is done here.

        :param df: DataFrame pure-component adsorption isotherm data
        :param loading_key: String key for loading column in df
        :param pressure_key: String key for pressure column in df
        :param param_guess: Dict starting guess for model parameters in the
            data fitting routine
        :param optimization_method: String method in SciPy minimization function
            to use in fitting model to data.
            See [here](http://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize).

        :return: self
        :rtype: ModelIsotherm
        """
        if model is None:
            raise Exception("Specify a model to fit to the pure-component"
                            " isotherm data. e.g. model=\"Langmuir\"")
        if model not in _MODELS:
            raise Exception("Model %s not an option in pyIAST. See viable"
                            "models with pyiast._MODELS" % model)

        #: Name of analytical model to fit to pure-component isotherm data
        #: adsorption isotherm
        self.model = model

        #: Pandas DataFrame on which isotherm was fit
        self.df = df
        if None in [loading_key, pressure_key]:
            raise Exception(
                "Pass loading_key and pressure_key, the names of the loading and"
                " pressure columns in the DataFrame, to the constructor.")
        #: name of column in `df` that contains loading
        self.loading_key = loading_key
        #: name of column in `df` that contains pressure
        self.pressure_key = pressure_key
        #: name of column in `df` that contains temperature
        self.temperature_key = temperature_key

        # ! root mean square error in fit
        self.rmse = np.nan

        # ! Dictionary of parameters as a starting point for data fitting
        self.param_guess = get_default_guess_params(model, df, pressure_key,
                                                    loading_key, temperature_key)

        # Override defaults if user provides param_guess dictionary
        if param_guess is not None:
            for param, guess_val in param_guess.items():
                if param not in list(self.param_guess.keys()):
                    raise Exception("%s is not a valid parameter"
                                    " in the %s model." % (param, model))
                self.param_guess[param] = guess_val

        # ! Dictionary of identified model parameters
        # initialize params as nan
        self.params = copy.deepcopy(_MODEL_PARAMS[model])

        # fit model to isotherm data in self.df
        self._fit(optimization_method)

    def loading(self, pressure, temperature):
        """
        Given stored model parameters, compute loading at pressure P.

        :param pressure: Float or Array pressure (in corresponding units as df
            in instantiation)
        :param temperature: Float or Array temperature (in corresponding units as df
            in instantiation), temperature and pressure should be in the same shape
            if both are arrays
        :return: predicted loading at pressure P (in corresponding units as df
            in instantiation) using fitted model params in `self.params`.
        :rtype: Float or Array
        """
        if type(pressure) == np.array and type(temperature) == np.array and \
            pressure.shape != temperature.shape:
            raise ValueError("Shape of pressure and temperature must match!")

        if self.model == "Langmuir":
            return self.params["M"] \
                * arrhenius(self.params["K_a"], self.params["K_b"], temperature) * pressure / \
                   (1.0 + arrhenius(self.params["K_a"], self.params["K_b"], temperature)* pressure)

        if self.model == "Quadratic":
            return self.params["M"] * (
                arrhenius(self.params["Ka_a"], self.params["Ka_b"], temperature) \
                + 2.0 * arrhenius(self.params["Kb_a"], self.params["Kb_b"], temperature) * pressure
            ) * pressure / (1.0 + arrhenius(self.params["Ka_a"], self.params["Ka_b"], temperature) * pressure +
                            arrhenius(self.params["Kb_a"], self.params["Kb_b"], temperature) * pressure**2)

        if self.model == "BET":
            return self.params["M"] * arrhenius(self.params["Ka_a"], self.params["Ka_b"], temperature)* pressure / (
                (1.0 - arrhenius(self.params["Kb_a"], self.params["Kb_b"], temperature) * pressure) *
                (1.0 - arrhenius(self.params["Kb_a"], self.params["Kb_b"], temperature) * pressure +
                 arrhenius(self.params["Ka_a"], self.params["Ka_b"], temperature) * pressure))

        if self.model == "DSLangmuir":
            # K_i P
            k1p = arrhenius(self.params["K1_a"], self.params["K1_b"], temperature) * pressure
            k2p = arrhenius(self.params["K2_a"], self.params["K2_b"], temperature) * pressure
            return self.params["M1"] * k1p / (1.0 + k1p) + \
                   self.params["M2"] * k2p / (1.0 + k2p)

        if self.model == "Henry":
            return arrhenius(self.params["KH_a"], self.params["KH_b"], temperature) * pressure

        if self.model == "TemkinApprox":
            langmuir_fractional_loading = arrhenius(self.params["K_a"], self.params["K_b"], temperature) * pressure / \
                                          (1.0 + arrhenius(self.params["K_a"], self.params["K_b"], temperature) * pressure)
            return self.params["M"] * (langmuir_fractional_loading + \
                     self.params["theta"] * langmuir_fractional_loading ** 2 * \
                     (langmuir_fractional_loading - 1))

    

    def _fit(self, optimization_method):
        """
        Fit model to data using nonlinear optimization with least squares loss
            function. Assigns params to self.

        :param K_guess: float guess Langmuir constant (units: 1/pressure)
        :param M_guess: float guess saturation loading (units: loading)
        """
        # parameter names (cannot rely on order in Dict)
        param_names = [param for param in self.params.keys()]
        # guess
        guess = np.array([self.param_guess[param] for param in param_names])

        def residual_sum_of_squares(params_):
            """
            Residual Sum of Squares between model and data in df
            :param params_: Array of parameters
            """
            # change params to those in x
            for i in range(len(param_names)):
                self.params[param_names[i]] = params_[i]

            return np.sum((self.df[self.loading_key].values - self.loading(
                self.df[self.pressure_key].values, self.df[self.temperature_key].values))**2)

        # minimize RSS
        opt_res = scipy.optimize.minimize(
            residual_sum_of_squares, guess, method=optimization_method)
        if not opt_res.success:
            print((opt_res.message))
            print(("\n\tDefault starting guess for parameters:",
                   self.param_guess))
            raise Exception("""Minimization of RSS for %s isotherm fitting
            failed. Try a different starting point in the nonlinear optimization
            by passing a dictionary of parameter guesses, param_guess, to the
            constructor""" % self.model)

        # assign params
        for j in range(len(param_names)):
            self.params[param_names[j]] = opt_res.x[j]

        self.rmse = np.sqrt(opt_res.fun / self.df.shape[0])

    def spreading_pressure(self, pressure, temperature):
        """
        Calculate reduced spreading pressure at a bulk gas pressure P.

        The reduced spreading pressure is an integral involving the isotherm
        :math:`L(P)`:

        .. math::

            \\Pi(p) = \\int_0^p \\frac{L(\\hat{p})}{ \\hat{p}} d\\hat{p},

        which is computed analytically, as a function of the model isotherm
        parameters.

        :param pressure: float pressure (in corresponding units as df in
            instantiation)
        :return: spreading pressure, :math:`\\Pi`
        :rtype: Float
        """
        if self.model == "Langmuir":
            #if 1.0 + arrhenius(self.params["K_a"], self.params["K_b"], temperature) * pressure <= 0:
            #    print("Warning:", temperature, pressure)
            return self.params["M"] * np.log(1.0 + arrhenius(self.params["K_a"], self.params["K_b"], temperature) * pressure)

        if self.model == "Quadratic":
            return self.params["M"] * np.log(1.0 + arrhenius(self.params["Ka_a"], self.params["Ka_b"], temperature) * pressure
                                             + arrhenius(self.params["Kb_a"], self.params["Kb_b"], temperature) * pressure**2)

        if self.model == "BET":
            return self.params["M"] * np.log(
                (1.0 - arrhenius(self.params["Kb_a"], self.params["Kb_b"], temperature) * pressure \
                    + arrhenius(self.params["Ka_a"], self.params["Ka_b"], temperature) *
                 pressure) / (1.0 - arrhenius(self.params["Kb_a"], self.params["Kb_b"], temperature) * pressure))

        if self.model == "DSLangmuir":
            return self.params["M1"] * np.log(
                1.0 + arrhenius(self.params["K1_a"], self.params["K1_b"], temperature) * pressure) +\
                   self.params["M2"] * np.log(
                       1.0 + arrhenius(self.params["K2_a"], self.params["K2_b"], temperature) * pressure)

        if self.model == "Henry":
            return arrhenius(self.params["KH_a"], self.params["KH_b"], temperature) * pressure

        if self.model == "TemkinApprox":
            one_plus_kp = 1.0 + arrhenius(self.params["K_a"], self.params["K_b"], temperature) * pressure
            return self.params["M"] * (
                np.log(one_plus_kp) + self.params["theta"] *
                (2.0 * arrhenius(self.params["K_a"], self.params["K_b"], temperature) * pressure + 1.0) /
                (2.0 * one_plus_kp**2))

    def print_params(self):
        """
        Print identified model parameters
        """
        print(("%s identified model parameters:" % self.model))
        for param, val in self.params.items():
            print(("\t%s = %f" % (param, val)))
        print(("RMSE = ", self.rmse))

    def get_isotherm(self, temperature):
        """
        Returns an adsorption isotherm at TEMPERATURE.
        """
        return NestedIsotherm(self, temperature)


class NestedIsotherm:
        
    def __init__(self, loadingsurface, temperature):
        self.loadingsurface = loadingsurface
        self.temperature = temperature
        self.df = loadingsurface.df.loc[loadingsurface.df[loadingsurface.temperature_key] == temperature]
        self.pressure_key = loadingsurface.pressure_key
        self.loading_key = loadingsurface.loading_key

    def loading(self, pressure):
        return self.loadingsurface.loading(pressure, self.temperature)

    def spreading_pressure(self, pressure):
        return self.loadingsurface.spreading_pressure(pressure, self.temperature)