"""Function to evaluate the log-likelihood of a certain IMF given a set of samples"""

import numpy as np
from .imfs import *
from scipy.optimize import minimize


def imf_lnprob(params, masses, model="chabrier"):
    """Computes the posterior likelihood of a given IMF model given
    the stellar masses
    """
    logm = np.log10(masses)
    match model.lower():
        case "chabrier":
            imf_func = chabrier_imf
        case "chabrier_smooth":
            imf_func = chabrier_smooth_imf
        case "chabrier_smooth_lognormal":
            imf_func = chabrier_smooth_lognormal_peak_imf
        case _:
            raise ValueError("IMF model not implemented!")

    imf_val = imf_func(logm, params)
    if np.any(imf_val <= 0):
        return -np.inf
    if not np.all(np.isfinite(np.log(imf_val))):
        return -np.inf
    return np.log(imf_val).sum()


def imf_mostlikely_params(masses, model="chabrier"):
    """Return the most likely IMF parameters"""
    p0 = DEFAULT_IMF_PARAMS[model]

    def lossfunc(p):
        return -imf_lnprob(p, masses, model)

    sol = minimize(lossfunc, p0, method="Nelder-Mead")
    if sol.success:
        return sol.x
    else:
        raise (ValueError("Could not successfully maximize IMF likelihood"))
