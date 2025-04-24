"""Mathmatical helper functions for computing normalizations of IMF segments"""

import numpy as np
from scipy.special import erf


def normal_cdf(X1, X2):
    """Returns the integral of a normal distribution from X1 to X2"""
    integral = 0.5 * (1 + erf(X2 / np.sqrt(2)))
    if X1 > -np.inf:
        integral -= 0.5 * (1 + erf(X1 / np.sqrt(2)))
    return integral


def normal(x, mu, sigma, xmin=-np.inf, xmax=np.inf):
    """Returns the value of the normal distribution"""
    funcval = (2 * np.pi) ** -0.5 / sigma * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    if xmin > -np.inf or xmax < np.inf:
        funcval /= normal_cdf((xmin - mu) / sigma, (xmax - mu) / sigma)
    return funcval


def chabrier_imf_norm(params, logmmin=-np.inf, logmmax=4):
    """Returns the integral of a Chabrier IMF with the given parameters"""
    logm0, logsigma, slope, logmbreak = params
    sigma = np.exp(logsigma)

    if logmmin < logmbreak:
        lognormal_norm = normal_cdf((logmmin - logm0) / sigma, (min(logmmax, logmbreak) - logm0) / sigma)
    else:
        lognormal_norm = 0  # lognormal part outside of the range

    mbreak, mmax, mmin = 10**logmbreak, 10**logmmax, 10**logmmin

    # mbreak = max(min(mbreak, mmax), mmin)
    # logmbreak = np.log10(mbreak)

    if logmmax > logmbreak:
        powerlaw_norm = (
            normal(logmbreak, logm0, sigma)
            * mbreak**-slope
            * powerlaw_integral(max(mmin, mbreak), mmax, slope - 1)  # alpha-1 because the measure is dlog10(m)
            / np.log(10.0)
        )
    else:
        powerlaw_norm = 0
    # print(lognormal_norm, powerlaw_norm)
    return lognormal_norm + powerlaw_norm


def powerlaw_integral(xmin, xmax, slope):
    """Returns the integral of x^alpha from xmin to xmax with respect to x

    Note that when using this to normalize integrals w.r.t. log10(m) you need a factor of ln(10)
    """
    if slope == -1:
        return np.log(xmax / xmin)
    return (xmax ** (1 + slope) - xmin ** (1 + slope)) / (1 + slope)
