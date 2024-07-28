"""
Routines for analyzing the IMF data from the simulations
"""

import numpy as np
from scipy.special import erf
from matplotlib import pyplot as plt


def normal_left_integral(X1, X2):
    """Returns the integral of a normal distribution from X1 to X2"""
    integral = 0.5 * (1 + erf(X2 / np.sqrt(2)))
    if X2 > 0:
        integral -= 0.5 * (1 + erf(X1 / np.sqrt(2)))
    return integral


def normal(x, mu, sigma):
    """Returns the value of the normal distribution"""
    return (2 * np.pi) ** -0.5 / sigma * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def chabrier_imf_norm(params, logmmin=0, logmmax=np.inf):
    """Returns the integral of a Chabrier IMF with the given parameters - assumes
    the lognormal part is already normalized"""

    logm0, logsigma, logmbreak, alpha = params
    sigma = np.exp(logsigma)

    lognormal_norm = normal_left_integral(
        (logmmin - logm0) / sigma, (logmbreak - logm0) / sigma
    )

    mbreak, mmax = 10**logmbreak, 10**logmmax
    if mmax < mbreak:
        mbreak = mmax

    powerlaw_norm = (
        normal(logmbreak, logm0, sigma)
        * mbreak**-alpha
        * powerlaw_integral(
            mbreak, mmax, alpha - 1
        )  # alpha-1 because the measure is dlog10(m)
        / np.log(10.0)
    )
    # print(sigma, logmmin, logmbreak, lognormal_norm, powerlaw_norm)
    return lognormal_norm + powerlaw_norm


def powerlaw_integral(xmin, xmax, alpha):
    """Returns the integral of x^alpha from xmin to xmax"""
    if alpha == -1:
        return np.log(xmax / xmin)
    return (xmax ** (1 + alpha) - xmin ** (1 + alpha)) / (1 + alpha)


def chabrier_imf(logm, params):
    """Returns the value of the Chabrier IMF form as a distribution in log m"""
    logm0, logsigma, logmbreak, alpha = params
    sigma = np.exp(logsigma)
    imf = normal(logm, logm0, sigma)
    m = 10**logm
    mbreak = 10**logmbreak
    imf[logm > logmbreak] = (
        normal(logmbreak, logm0, sigma) * (m[logm > logmbreak] / mbreak) ** alpha
    )
    return imf / chabrier_imf_norm(params, logm.min(), logm.max())


CHABRIER_DEFAULT_PARAMS = (
    np.log10(0.08),
    np.log(0.69),
    0.0,
    -1.3,
)
CHABRIER_SMOOTH_DEFAULT_PARAMS = (
    np.log10(0.08),
    np.log(0.69),
    -1.3,
)


def chabrier_smooth_imf(logm, params):
    """Version of the Chabrier IMF constrained to have a smooth break between
    the lognormal and power-law parts"""
    logm0, logsigma, alpha = params
    sigma = np.exp(logsigma)
    logmbreak = logm0 - alpha * sigma * sigma * np.log(10.0)
    params_chabrier = logm0, np.log(sigma), logmbreak, alpha
    return chabrier_imf(logm, params_chabrier)


def chabrier_smooth_lognormal_peak_imf(logm, params, imf0=chabrier_smooth_imf):
    """Sum of any IMF and a lognormal peak"""
    params0 = params[:-4]
    log_mmax1, log_fpeak, logm0_peak, logsigma_peak = params[-4:]

    imf1 = imf0(logm.clip(-np.inf, log_mmax1), params0) * (logm < log_mmax1)
    imf2 = normal(logm, logm0_peak, np.exp(logsigma_peak))
    wt = np.exp(log_fpeak)
    wt1 = 1 / (1 + wt)
    wt2 = 1 - wt1
    imf = wt1 * imf1 + wt2 * imf2
    return imf

def test_chabrier_imf():
    mgrid = np.logspace(-3, np.log10(120.0), 100001)
    logm = np.log10(mgrid)
    params = CHABRIER_DEFAULT_PARAMS

    params = [-0.25999083, -0.69767731, -0.70363855]
    imf = chabrier_smooth_imf(logm, params)
    # chabrier_imf_norm(params)
    plt.loglog(mgrid, imf)
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    test_chabrier_imf()
