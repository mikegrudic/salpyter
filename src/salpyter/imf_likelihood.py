"""Function to evaluate the log-likelihood of a certain IMF given a set of samples"""

import numpy as np
from . import imfs
from scipy.optimize import minimize
import emcee
from .default_imf_params import *


def imf_lnprob(params, masses, model=DEFAULT_MODEL, logmmin=None, logmmax=None):
    """Computes the posterior likelihood of a given IMF model given
    the stellar masses

    Parameters
    ----------
    params: array_like
        IMF parameters
    masses: array_like
        array of sample masses
    model: string, optional
        name of the IMF model function  (default chabrier)
    logmin: float, optional
        Lower bound of the IMF mass range
    logmmax: float, optional
        Upper bound of the IMF mass range

    Returns
    -------
    lnprob: float
        log-likelihood value
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


def imf_mostlikely_params(masses, model=DEFAULT_MODEL, bounds=None, p0=None):
    """Estimates the most-likely set of IMF parameters for a given sample and model

    Parameters
    ----------
    masses: array_like
        array of sample masses
    model: string, optional
        name of the IMF model function (default chabrier)
    bounds: array_like, optional
        Shape (n_params, 2) array of upper and lower bounds for parameters; if None will use defaults specified in default_imf_params.py
    p0: array_like, optional
        Initial parameter guess; if not provided will use default IMF parameters

    Returns
    -------
    pmax: array_like
        Shape (n_params) array of the maximum-likelihood IMF parameters given the data
    """

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
    masses,
    model=DEFAULT_MODEL,
    p0=None,
    bounds=None,
    nwalkers: int = 100,
    chainlength: int = 1000,
    logmmin=None,
    logmmax=None,
):
    """Calls emcee and returns samples from the likelihood distribution of IMF parameters

    Parameters
    ----------
    masses: array_like
        array of sample masses
    model: string, optional
        name of the IMF model function  (default chabrier)
    p0: array_like, optional
        Initial guess for the most-likely parameters
    bounds: array_like, optional
        Shape (n_params, 2) array of upper and lower bounds for parameters; if None will use defaults specified in default_imf_params.py
    nwalkers: int, optional
        Number of Monte Carlo walkers (should be much more than the dimensionality of your parameter space)
    chainlength: int, optional
        Length of Monte Carlo walk chain (should be enough to converge to the actual distribution)

    Returns
    -------
    samples: array_like
        Shape (N, N_params) array of samples from the posterior likelihood distribution of the IMF model given the data
    """

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
        # print(p0, b, imf_mostlikely_params(masses, model).x)
        p0[i] = np.clip(p0[i],b[0], b[1])

    if lnprob(p0) == -np.inf:
        raise (ValueError(f"lnprob is negative infinity at p0={p0} - find a better guess."))

    nwalkers, ndim = 100, len(p0)
    p0 = np.array(p0) + 0.01 * np.random.normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    state = sampler.run_mcmc(p0, chainlength // 10)
    sampler.reset()
    sampler.run_mcmc(state, chainlength)
    flat_samples = sampler.get_chain(flat=True, thin=1000)
    return flat_samples
