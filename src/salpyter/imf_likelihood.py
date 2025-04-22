"""Function to evaluate the log-likelihood of a certain IMF given a set of samples"""

import numpy as np
from . import imfs
from scipy.optimize import minimize
import emcee

DEFAULT_LOGMMIN = -2
DEFAULT_LOGMMAX = 4


def imf_lnprob(params, masses, model="chabrier", logmmin=None, logmmax=None):
    """Computes the posterior likelihood of a given IMF model given
    the stellar masses
    """
    imf_func = getattr(imfs, model.lower() + "_imf")

    logm = np.log10(masses)

    if logmmin is None:
        logmmin = logm.min()
    if logmmax is None:
        logmmax = logm.max()

    imf_val = imf_func(logm, params, logmmin, logmmax)
    if np.any(imf_val <= 0):
        return -np.inf
    if not np.all(np.isfinite(np.log(imf_val))):
        return -np.inf
    return np.log(imf_val).sum()


def imf_default_params(model="chabrier"):
    """Convenience method to access default IMF parameters"""

    if "_bounds" in model:  # 'imf + upper and lower bounds' model
        params = imfs.DEFAULT_IMF_PARAMS[model.replace("_bounds", "")] + [DEFAULT_LOGMMIN, DEFAULT_LOGMMAX]
        #
    else:
        params = imfs.DEFAULT_IMF_PARAMS[model]
    return params


def imf_default_bounds(model="chabrier"):
    """Convenience method to access default IMF parameters"""

    if "_bounds" in model:  # 'imf + upper and lower bounds' model
        bounds = imfs.DEFAULT_IMF_PARAMS_BOUNDS[model.replace("_bounds", "")] + [[-4, 4], [-4, 4]]
        # print(bounds)
    else:
        bounds = imfs.DEFAULT_IMF_PARAMS_BOUNDS[model]
    return bounds


def imf_mostlikely_params(masses, model="chabrier", bounds=None, p0=None):
    """Return the most likely IMF parameters"""
    if p0 is None:
        p0 = list(imf_default_params(model))

    if "chabrier" in model and "smooth" not in model:
        # start by fitting the simplest model and using those parameters in the guess for the
        # more-complex model
        p0[:3] = imf_mostlikely_params(
            masses, "chabrier_smooth", (bounds[:3] if bounds is not None else None), p0[:3]
        ).x

    def lossfunc(p):
        return -imf_lnprob(p, masses, model)

    if bounds is None:
        bounds = imf_default_bounds(model)

    sol = minimize(lossfunc, p0, bounds=bounds, method="Nelder-Mead")
    return sol


def imf_lnprob_samples(
    masses, model="chabrier", p0=None, bounds=None, nwalkers=100, chainlength=1000, logmmin=None, logmmax=None
):
    """Returns samples from the likelihood distribution of IMF parameters"""

    ndim = len(imf_default_params(model))
    if bounds is None:
        bounds = imf_default_bounds(model)

    bounds = np.array(bounds)

    def lnprob(params):
        if np.any(params < bounds[:, 0]):
            return -np.inf
        if np.any(params > bounds[:, 1]):
            return -np.inf
        return imf_lnprob(params, masses, model, logmmin, logmmax)

    if p0 is None:  # initial guess
        p0 = imf_mostlikely_params(masses, model).x
        if lnprob(p0) == -np.inf:
            p0 = imf_default_params(model)

    for i, b in enumerate(bounds):  # clip to bounds
        p0[i] = p0[i].clip(b[0], b[1])

    if lnprob(p0) == -np.inf:
        raise (ValueError("lnprob is negative infinity at p0 - find a better guess."))

    nwalkers, ndim = 100, len(p0)
    p0 = np.array(p0) + 0.01 * np.random.normal(size=(nwalkers, ndim))

    # print(p0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    state = sampler.run_mcmc(p0, chainlength // 10)
    sampler.reset()
    sampler.run_mcmc(state, chainlength)
    flat_samples = sampler.get_chain(flat=True, thin=1000)
    return flat_samples
