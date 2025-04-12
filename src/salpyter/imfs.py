"""
Routines for analyzing the IMF data from the simulations
"""

import numpy as np
from scipy.special import erf

CHABRIER_DEFAULT_PARAMS = (np.log10(0.08), np.log(0.69), -1.3, 0.0)
CHABRIER_SMOOTH_DEFAULT_PARAMS = (np.log10(0.08), np.log(0.69), -1.3)

DEFAULT_IMF_PARAMS = {
    "chabrier": CHABRIER_DEFAULT_PARAMS,
    "chabrier_smooth": CHABRIER_SMOOTH_DEFAULT_PARAMS,
    "chabrier_smooth_lognormal": (np.log10(0.08), np.log(0.69), -1.0, -1.0, 3, 0.0),
    "chabrier_smooth_cutoff_lognormal": (np.log10(0.08), np.log(0.69), -1.0, 2, -1.0, 3, 0.0),
}


def normal_cdf(X1, X2):
    """Returns the integral of a normal distribution from X1 to X2"""
    integral = 0.5 * (1 + erf(X2 / np.sqrt(2)))
    if X1 > -np.inf:
        integral -= 0.5 * (1 + erf(X1 / np.sqrt(2)))
    return integral


def normal(x, mu, sigma, xmin=-np.inf, xmax=np.inf):
    """Returns the value of the normal distribution"""
    funcval = (2 * np.pi) ** -0.5 / sigma * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    # print(xmin, xmax)
    if xmin > -np.inf or xmax < np.inf:
        # print("normal")
        # print(xmin, mu, xmax, sigma, normal_cdf(xmin - mu, xmax - mu))
        funcval /= normal_cdf((xmin - mu) / sigma, (xmax - mu) / sigma)
    return funcval


def chabrier_imf_norm(params, logmmin=-np.inf, logmmax=2):
    """Returns the integral of a Chabrier IMF with the given parameters - assumes
    the lognormal part is already normalized"""
    logm0, logsigma, alpha, logmbreak = params
    sigma = np.exp(logsigma)

    lognormal_norm = normal_cdf((logmmin - logm0) / sigma, (logmbreak - logm0) / sigma)

    mbreak, mmax = 10**logmbreak, 10**logmmax
    if mmax < mbreak:
        mbreak = mmax

    powerlaw_norm = (
        normal(logmbreak, logm0, sigma)
        * mbreak**-alpha
        * powerlaw_integral(mbreak, mmax, alpha - 1)  # alpha-1 because the measure is dlog10(m)
        / np.log(10.0)
    )
    return lognormal_norm + powerlaw_norm


def powerlaw_integral(xmin, xmax, alpha):
    """Returns the integral of x^alpha from xmin to xmax"""
    if alpha == -1:
        return np.log(xmax / xmin)
    return (xmax ** (1 + alpha) - xmin ** (1 + alpha)) / (1 + alpha)


def chabrier_imf(logm, params, logmmin=-np.inf, logmmax=4):
    """Returns the value of the Chabrier IMF form as a distribution in log m"""
    logm0, logsigma, alpha, logmbreak = params
    sigma = np.exp(logsigma)
    imf = normal(logm, logm0, sigma)
    m = 10**logm
    mbreak = 10**logmbreak
    imf[logm > logmbreak] = normal(logmbreak, logm0, sigma) * (m[logm > logmbreak] / mbreak) ** alpha
    return imf / chabrier_imf_norm(params, logmmin, logmmax)


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
