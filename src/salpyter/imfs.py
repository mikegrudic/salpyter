"""
Routines for analyzing the IMF data from the simulations
"""

import numpy as np
from .norm_functions import powerlaw_integral, chabrier_imf_norm, normal

CHABRIER_DEFAULT_PARAMS = [np.log10(0.25), np.log(0.55), -1.3, 0.0]  # log mc, log sigma, high-mass slope, log m_break
CHABRIER_SMOOTH_DEFAULT_PARAMS = [np.log10(0.25), np.log(0.55), -1.3]  # log mc, log sigma, high-mass slope

DEFAULT_IMF_PARAMS = {
    "powerlaw": [-1.3],
    "powerlaw_mmax": [-1.3, 4],
    "chabrier": CHABRIER_DEFAULT_PARAMS,
    "chabrier_smooth": CHABRIER_SMOOTH_DEFAULT_PARAMS,
    "chabrier_smooth_lognormal": [np.log10(0.08), np.log(0.69), -1.0, -1.0, 3, 0.0],
    "chabrier_smooth_cutoff_lognormal": [np.log10(0.08), np.log(0.69), -1.0, 2, -1.0, 3, 0.0],
}


DEFAULT_IMF_PARAMS_BOUNDS = {
    # log mc, log sigma, high-mass slope
    "chabrier_smooth": [[-2, 2], [-2, 2], [-10, 2]],
    "powerlaw": [[-10, 2]],
    "powerlaw_mmax": [[-10, 2], [-2, 4]],
}

p0 = DEFAULT_IMF_PARAMS_BOUNDS["chabrier_smooth"]
# + log m_break
DEFAULT_IMF_PARAMS_BOUNDS["chabrier"] = p0 + [[-3, 3]]
# + fraction in lognormal part, log mc of lognormal part, log sigma of lognormal part
DEFAULT_IMF_PARAMS_BOUNDS["chabrier_smooth_lognormal"] = p0 + [[-10, 6], [-1, 4], [-2, 1]]
# + high-mass cutoff of lognormal part, fraction in lognormal part, log mc of lognormal part, log sigma of lognormal part
DEFAULT_IMF_PARAMS_BOUNDS["chabrier_smooth_cutoff_lognormal"] = p0 + [[0, 4], [-10, 6], [-1, 4], [-2, 1]]


def powerlaw_imf(logm, params, logmmin=-np.inf, logmmax=None):
    """Simple single-power-law (i.e. Salpeter) IMF allowing the maximum mass to be a free parameter"""
    slope = params[0]  # -1.35 = salpeter
    mmin, mmax = 10**logmmin, 10**logmmax
    norm = powerlaw_integral(mmin, mmax, slope - 1) / np.log(10)  # slope-1 and log10 to convert from m to logm function
    m = 10**logm
    imf = m**slope / norm
    imf[(m > mmax) ^ (m < mmin)] = 0.0
    return imf


def powerlaw_bounds_imf(logm, params, logmmin=-np.inf, logmmax=4):
    """Single power law with fixed max and min mass"""
    return imf_with_bounds_params(logm, params, powerlaw_imf)


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

    # logmgrid = np.linspace(logmmin, logmmax, 10000)
    # imf2 = normal(logmgrid, logm0, sigma)
    # imf2[logmgrid > logmbreak] = (
    #     normal(logmbreak, logm0, sigma) * ((10**logmgrid)[logmgrid > logmbreak] / mbreak) ** alpha
    # )
    # norm2 = np.trapz(imf2, logmgrid)

    # if np.abs(np.log10(norm / norm2)) > 0.01:
    #     print(params, norm, norm2)
    #     from matplotlib import pyplot as plt

    #     plt.plot(logmgrid, imf2)
    #     plt.yscale("log")
    #     plt.savefig("imf.png")
    #     exit()

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
