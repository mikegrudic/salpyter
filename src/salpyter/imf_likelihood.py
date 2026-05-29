"""Function to evaluate the log-likelihood of a certain IMF given a set of samples"""

import numpy as np
from . import imfs
from scipy.optimize import minimize
from scipy.special import erf
import emcee
from .default_imf_params import *


_LN10 = np.log(10.0)
_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _chabrier_smooth_lnprob_vec(params_batch, logm, m, logmmin, logmmax, lo, hi):
    """Vectorized lnprob for the ``chabrier_smooth`` IMF.

    Computes log p(masses | params) for many parameter vectors at once. Equivalent
    to looping ``imf_lnprob(params, masses, "chabrier_smooth", logmmin, logmmax)``
    over the rows of ``params_batch``, but with all the numpy/Python overhead
    amortized across walkers.

    Parameters
    ----------
    params_batch: (W, 3) array of [logm0, logsigma, alpha]
    logm: (N,) array of log10(masses)
    m: (N,) array of masses (== 10**logm, precomputed)
    logmmin, logmmax: scalar mass-range bounds
    lo, hi: (3,) parameter bounds

    Returns
    -------
    (W,) array of log-probabilities, with -inf for walkers outside bounds or
    where the IMF is non-positive at any sample.
    """
    params_batch = np.atleast_2d(params_batch)
    in_bounds = ~(
        np.any(params_batch < lo, axis=1) | np.any(params_batch > hi, axis=1)
    )

    logm0 = params_batch[:, 0]
    logsigma = params_batch[:, 1]
    alpha = params_batch[:, 2]
    sigma = np.exp(logsigma)
    inv_sigma = 1.0 / sigma
    # Smooth-break condition (matches chabrier_smooth_imf)
    logmbreak = logm0 - alpha * sigma * sigma * _LN10
    mbreak = 10.0**logmbreak

    # Lognormal part, shape (W, N)
    z = (logm[None, :] - logm0[:, None]) * inv_sigma[:, None]
    lognormal = _INV_SQRT_2PI * inv_sigma[:, None] * np.exp(-0.5 * z * z)

    # Normal at the break point, shape (W,)
    z_break = (logmbreak - logm0) * inv_sigma
    normal_at_break = _INV_SQRT_2PI * inv_sigma * np.exp(-0.5 * z_break * z_break)

    # Power-law part, shape (W, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        powerlaw = normal_at_break[:, None] * (m[None, :] / mbreak[:, None]) ** alpha[:, None]

    mask = logm[None, :] > logmbreak[:, None]
    imf_pre = np.where(mask, powerlaw, lognormal)

    # --- Normalization (vectorized chabrier_imf_norm) ---
    # Lognormal-part integral: nonzero only where logmmin < logmbreak.
    upper_cap = np.minimum(logmmax, logmbreak)
    X1 = (logmmin - logm0) * inv_sigma
    X2 = (upper_cap - logm0) * inv_sigma
    lognormal_norm = np.where(
        logmmin < logmbreak,
        0.5 * (erf(X2 / _SQRT2) - erf(X1 / _SQRT2)),
        0.0,
    )

    # Power-law-part integral: nonzero only where logmmax > logmbreak.
    mmin = 10.0**logmmin
    mmax = 10.0**logmmax
    xmin_pl = np.maximum(mmin, mbreak)
    # powerlaw_integral(xmin, xmax, alpha-1) = (xmax^alpha - xmin^alpha) / alpha,
    # except when alpha == 0 it is log(xmax/xmin). Vectorized via np.where.
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_alpha = np.where(alpha != 0, alpha, 1.0)
        pl_integral = np.where(
            alpha != 0,
            (mmax**alpha - xmin_pl**alpha) / safe_alpha,
            np.log(mmax / xmin_pl),
        )
        powerlaw_norm = np.where(
            logmmax > logmbreak,
            normal_at_break * mbreak ** (-alpha) * pl_integral / _LN10,
            0.0,
        )
    norm = lognormal_norm + powerlaw_norm

    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log(imf_pre / norm[:, None])  # (W, N)

    finite = np.isfinite(lg).all(axis=1)
    sums = lg.sum(axis=1)
    return np.where(in_bounds & finite, sums, -np.inf)


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
    lo, hi = bounds[:, 0], bounds[:, 1]

    # Hoist per-call work out of the MCMC inner loop: the masses, the IMF function
    # lookup, and the default mass bounds are fixed across all ~1e6 lnprob evaluations.
    imf_func = getattr(imfs, model.lower() + "_imf")
    # Flatten so the vectorized path can rely on logm being 1D; matches the
    # scalar imf_lnprob behavior (it just sums over all entries).
    logm = np.log10(np.asarray(masses).ravel())
    lmin = logm.min() if logmmin is None else logmmin
    lmax = logm.max() if logmmax is None else logmmax

    # The chabrier_smooth model has a hand-vectorized lnprob that evaluates all
    # walkers in one call, collapsing ~100x of the per-step numpy overhead.
    use_vec = model.lower() == "chabrier_smooth"

    if use_vec:
        m_arr = 10.0**logm

        def lnprob(params):
            return _chabrier_smooth_lnprob_vec(params, logm, m_arr, lmin, lmax, lo, hi)

        def lnprob_scalar(params):
            return lnprob(params)[0]
    else:

        def lnprob(params):
            if np.any(params < lo) or np.any(params > hi):
                return -np.inf
            with np.errstate(invalid="ignore", divide="ignore"):
                lg = np.log(imf_func(logm, params, lmin, lmax))
            if not np.isfinite(lg).all():
                return -np.inf
            return lg.sum()

        lnprob_scalar = lnprob

    if p0 is None:  # initial guess
        p0 = imf_mostlikely_params(masses, model).x
        if lnprob_scalar(p0) == -np.inf:
            p0 = imf_default_params(model)

    for i, b in enumerate(bounds):  # clip to bounds
        # print(p0, b, imf_mostlikely_params(masses, model).x)
        p0[i] = np.clip(p0[i],b[0], b[1])

    if lnprob_scalar(p0) == -np.inf:
        raise (ValueError(f"lnprob is negative infinity at p0={p0} - find a better guess."))

    nwalkers, ndim = 100, len(p0)
    p0 = np.array(p0) + 0.01 * np.random.normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, vectorize=use_vec)
    state = sampler.run_mcmc(p0, chainlength // 10)
    sampler.reset()
    sampler.run_mcmc(state, chainlength)
    flat_samples = sampler.get_chain(flat=True, thin=100)
    return flat_samples
