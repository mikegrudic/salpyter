"""
Routines for analyzing the IMF data from the simulations
"""

import numpy as np
from .norm_functions import powerlaw_integral, chabrier_imf_norm, normal
from .default_imf_params import *


def powerlaw_imf(logm, params, logmmin=-np.inf, logmmax=None):
    """
    Simple single-power-law (e.g. Salpeter) IMF

    Parameters
    ----------
    logm: array_like
        Array of log10(mass) values at which to evaluate the IMF
    params: array_like
        Shape (1,) array-like containing the IMF slope (Salpeter value = -1.35)
    logmmin: float, optional
        Low-mass cutoff
    logmmax: float, optional
        High-mass cutoff

    Returns
    -------
    imf: array-like
        Value of the IMF normalized to integrate over logm
    """
    slope = params[0]  # -1.35 = salpeter
    mmin, mmax = 10**logmmin, 10**logmmax
    norm = powerlaw_integral(mmin, mmax, slope - 1) / np.log(10)  # slope-1 and log10 to convert from m to logm function
    m = 10**logm
    imf = m**slope / norm
    imf[(m > mmax * (1 + 1e-15)) ^ (m < (1 - 1e-15) * mmin)] = 0.0
    return imf


def imf_with_bounds_params(logm, params, imf0=powerlaw_imf):
    """
    Given another base IMF model, implements an IMF that adds lower and upper bounds as the last 2 additional parameters

    Parameters
    ----------
    logm: array_like
        Array of log10(mass) values at which to evaluate the IMF
    params: array_like
        Shape (n_params+2,) array where the initial parameters are for the base IMF and the  last 2  are the low- and
        high-mass cutoff
    imf0: function, optional
        function implmenting the base IMF

    Returns
    -------
    imf: array-like
        Value of the IMF normalized to integrate over logm
    """
    logmmin, logmmax = params[-2:]
    imf = imf0(logm, params[:-2], logmmin, logmmax)
    imf[(logm > logmmax) ^ (logm < logmmin)] = 0.0
    return imf


# def piecewise_powerlaw_imf(logm, params, logmmin=-np.inf, logmmax=4):
#     """Piecewise-power-law (e.g. Scalo, Kroupa) IMF"""
#     if len(params) == 1:
#         return powerlaw_imf(logm, params, logmmin, logmmax)

#     if len(params) % 2 == 0:
#         raise ValueError("Piecewise power-law IMF must have an odd number of parameters.")
#     slopes = params[::2]  # parameters: slope 1, logm12, slope2, logm23, ... slopeN
#     logmbreaks = params[1::2]  # segment break masses
#     mmin, mmax = 10**logmmin, 10**logmmax
#     norm = 0

#     #    imf_value = np.heaviside(logm - logmmin) * np.heaviside(logmmax - logm)
#     imf_value = np.ones_like(logm)
#     m0 = 1.0
#     for i, s in enumerate(slopes):  # loop over segments
#         if logmbreaks[i]
#         #norm += powerlaw_integral(mmin, mmax, s - 1) / np.log(10)

#         # imf_value *= np.heaviside

#     m = 10**logm
#     return m**slope / norm


def chabrier_imf(logm, params, logmmin=-np.inf, logmmax=4):
    """Returns the value of the Chabrier IMF form as a distribution in log m

    Parameters
    ----------
    logm: array_like
        Array of log10(mass) values at which to evaluate the IMF
    params: array_like
        Shape (4,) array of parameters: [log m_peak, log sigma, slope, log m_break]
    logmmin: float, optional
        Low-mass cutoff
    logmmax: float, optional
        High-mass cutoff

    Returns
    -------
    imf: array-like
        Value of the IMF normalized to integrate over logm
    """
    logm0, logsigma, alpha, logmbreak = params
    sigma = np.exp(logsigma)
    imf = normal(logm, logm0, sigma)

    m, mbreak = 10**logm, 10**logmbreak
    imf[logm > logmbreak] = normal(logmbreak, logm0, sigma) * (m[logm > logmbreak] / mbreak) ** alpha
    norm = chabrier_imf_norm(params, logmmin, logmmax)

    return imf / norm  # chabrier_imf_norm(params, logmmin, logmmax)


def chabrier_smooth_imf(logm, params, logmmin=-np.inf, logmmax=4):
    """
    Chabrier IMF constrained to have a smooth break between the lognormal and power-law part

    Parameters
    ----------
    logm: array_like
        Array of log10(mass) values at which to evaluate the IMF
    params: array_like
        Shape (3,) array of parameters: [log m_peak, log sigma, slope]
    logmmin: float, optional
        Low-mass cutoff
    logmmax: float, optional
        High-mass cutoff

    Returns
    -------
    imf: array-like
        Value of the IMF normalized to integrate over logm
    """

    logm0, logsigma, alpha = params
    sigma = np.exp(logsigma)
    logmbreak = logm0 - alpha * sigma * sigma * np.log(10.0)  # condition for smooth transition
    params_chabrier = logm0, np.log(sigma), alpha, logmbreak
    return chabrier_imf(logm, params_chabrier, logmmin, logmmax)


def imf_plus_lognormal(logm, params, imf0=chabrier_smooth_imf, logmmin=-np.inf, logmmax=4, cutoff=False):
    """Sum of any IMF and a lognormal peak

    Parameters
    ----------
    logm: array_like
        Array of log10(mass) values at which to evaluate the IMF
    params: array_like
        Shape (n_params,) array of parameters. The first values will be the parameters of the base IMF.  If cutoff is
        True, the last 3 parameters specify:
          1. The high-mass cutoff of the original IMF
          2. the log of the fraction of stars in the lognormal component,
          3. the log of the lognormal's peak mass
          4. the log of the lognormal's sigma
        If cutoff is False, the high-mass cutoff parameter above is omitted.
    imf0: function, optional
        IMF function for the base IMF
    logmmin: float, optional
        low-mass cutoff
    logmmax: float, optional
        high-mass cutoff
    cutoff: boolean, optional
        Whether to include a high-mass cutoff for the base IMF in the parameters


    Returns
    -------
    imf: array-like
        Value of the IMF normalized to integrate over logm
    """
    xmin, xmax = logmmin, logmmax
    if cutoff:
        params0 = params[:-4]
        log_mcut, log_fpeak, logm0_peak, logsigma_peak = params[-4:]
        imf1 = imf0(logm, params0, xmin, min(xmax, log_mcut))
        imf1[logm > log_mcut] = 0.0
    else:
        params0 = params[:-3]
        log_fpeak, logm0_peak, logsigma_peak = params[-3:]
        imf1 = imf0(logm, params0, xmin, xmax)

    imf2 = normal(logm, logm0_peak, np.exp(logsigma_peak), xmin=xmin, xmax=xmax)
    wt = 10**log_fpeak
    wt1 = 1 / (1 + wt)
    wt2 = 1 - wt1

    imf = wt1 * imf1 + wt2 * imf2
    return imf


def chabrier_smooth_lognormal_imf(logm, params, logmmin=-np.inf, logmmax=4):
    """
    IMF consisting of a chabrier_smooth IMF component plus a lognormal component

    Parameters
    ----------
    logm: array_like
        Array of log10(mass) values at which to evaluate the IMF
    params: array_like
        Shape (6,) array of parameters: [log m_peak, log sigma, slope, log f_peak, log m_peak, log sigma_peak]
    logmmin: float, optional
        Low-mass cutoff
    logmmax: float, optional
        High-mass cutoff

    Returns
    -------
    imf: array-like
        Value of the IMF normalized to integrate over logm
    """
    return imf_plus_lognormal(logm, params, chabrier_smooth_imf, logmmin, logmmax)


def chabrier_smooth_cutoff_lognormal_imf(logm, params, logmmin=-np.inf, logmmax=4):
    """IMF consisting of a chabrier_smooth IMF component *with a sharp high-mass cutoff* plus a lognormal component

    Parameters
    ----------
    logm: array_like
        Array of log10(mass) values at which to evaluate the IMF
    params: array_like
        Shape (7,) array of parameters: [log m_peak, log sigma, slope, log mmax, log f_peak, log m_peak, log sigma_peak]
    logmmin: float, optional
        Low-mass cutoff
    logmmax: float, optional
        High-mass cutoff

    Returns
    -------
    imf: array-like
        Value of the IMF normalized to integrate over logm
    """
    return imf_plus_lognormal(logm, params, chabrier_smooth_imf, logmmin, logmmax, cutoff=True)
