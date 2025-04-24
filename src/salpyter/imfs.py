"""
Routines for analyzing the IMF data from the simulations
"""

import numpy as np
from .norm_functions import powerlaw_integral, chabrier_imf_norm, normal
from .default_imf_params import *


def powerlaw_imf(logm, params, logmmin=-np.inf, logmmax=None):
    """Simple single-power-law (i.e. Salpeter) IMF allowing the maximum mass to be a free parameter"""
    slope = params[0]  # -1.35 = salpeter
    mmin, mmax = 10**logmmin, 10**logmmax
    norm = powerlaw_integral(mmin, mmax, slope - 1) / np.log(10)  # slope-1 and log10 to convert from m to logm function
    m = 10**logm
    imf = m**slope / norm
    imf[(m > mmax) ^ (m < mmin)] = 0.0
    return imf


def imf_with_bounds_params(logm, params, imf0=powerlaw_imf):
    """IMF that adds lower and upper bounds to an IMF as the last 2 parameters"""
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
    """Returns the value of the Chabrier IMF form as a distribution in log m"""
    logm0, logsigma, alpha, logmbreak = params
    sigma = np.exp(logsigma)
    imf = normal(logm, logm0, sigma)

    m, mbreak = 10**logm, 10**logmbreak
    imf[logm > logmbreak] = normal(logmbreak, logm0, sigma) * (m[logm > logmbreak] / mbreak) ** alpha
    norm = chabrier_imf_norm(params, logmmin, logmmax)

    return imf / norm  # chabrier_imf_norm(params, logmmin, logmmax)


def chabrier_smooth_imf(logm, params, logmmin=-np.inf, logmmax=4):
    """Version of the Chabrier IMF constrained to have a smooth break between
    the lognormal and power-law parts"""
    logm0, logsigma, alpha = params
    sigma = np.exp(logsigma)
    logmbreak = logm0 - alpha * sigma * sigma * np.log(10.0)  # condition for smooth transition
    params_chabrier = logm0, np.log(sigma), alpha, logmbreak
    return chabrier_imf(logm, params_chabrier, logmmin, logmmax)


def imf_plus_lognormal(logm, params, imf0=chabrier_smooth_imf, logmmin=-np.inf, logmmax=4, cutoff=False):
    """Sum of any IMF and a lognormal peak"""
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
    return imf_plus_lognormal(logm, params, chabrier_smooth_imf, logmmin, logmmax)


def chabrier_smooth_cutoff_lognormal_imf(logm, params, logmmin=-np.inf, logmmax=4):
    return imf_plus_lognormal(logm, params, chabrier_smooth_imf, logmmin, logmmax, cutoff=True)
