"""IMF log-likelihood and MAP estimation (jax branch).

``imf_lnprob`` is the JAX-differentiable log-probability used by both the MAP
estimator (L-BFGS-B with autodiff gradients) and the NUTS sampler.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import imfs
from .default_imf_params import (
    DEFAULT_MODEL,
    imf_default_bounds,
    imf_default_params,
)


_MODEL_TO_FUNC = {
    "chabrier_smooth": imfs.chabrier_smooth_imf,
    "chabrier": imfs.chabrier_imf,
    "chabrier_smooth_bounds": imfs.chabrier_smooth_bounds_imf,
    "chabrier_smooth_exp_bounds": imfs.chabrier_smooth_exp_bounds_imf,
    "chabrier_exp_bounds": imfs.chabrier_exp_bounds_imf,
}


def _resolve_imf_func(model):
    fn = _MODEL_TO_FUNC.get(model.lower())
    if fn is None:
        raise NotImplementedError(
            f"jax salpyter supports {sorted(_MODEL_TO_FUNC)}; got {model!r}"
        )
    return fn


def _bootstrap_p0(masses, model, logmmin=None, logmmax=None):
    """Pick a sensible starting point for the MAP optimizer.

    Without bootstrapping, the bounded model's MAP from
    ``imf_default_params`` lands in a wrong local mode (narrow lognormal +
    loose bounds) that has ~150 lower log-likelihood than the global mode
    (wide lognormal + bounds pinned at the data extrema). L-BFGS-B can't
    cross between modes because the gradient kicks in only when bounds
    cross the data extrema. So we first fit the 3-param chabrier_smooth
    model (well-behaved, no bound parameters), then extend its solution
    with bound values right at the data extrema. Same trick the master
    branch uses for chabrier-from-chabrier_smooth.
    """
    lower = model.lower()
    if lower == "chabrier_smooth_bounds":
        base = imf_mostlikely_params(
            masses, "chabrier_smooth", logmmin=logmmin, logmmax=logmmax,
        ).x
        logm = np.log10(np.asarray(masses).ravel())
        # Set bounds just outside the data extrema; the small margin keeps the
        # data strictly inside [logmmin, logmmax] (the boundary in
        # chabrier_smooth_bounds_imf uses >=/<=, but the margin avoids any
        # floating-point edge case at the L-BFGS-B starting evaluation).
        margin = 1e-3
        return list(base) + [float(logm.min()) - margin, float(logm.max()) + margin]
    if lower == "chabrier":
        base = imf_mostlikely_params(
            masses, "chabrier_smooth", logmmin=logmmin, logmmax=logmmax,
        ).x
        # Use the chabrier_smooth break point as the default free logmbreak.
        logm0, logsigma, alpha = base
        sigma = float(np.exp(logsigma))
        logmbreak = float(logm0) - float(alpha) * sigma * sigma * float(np.log(10.0))
        return list(base) + [logmbreak]
    if lower == "chabrier_smooth_exp_bounds":
        # Same shape bootstrap as the hard-bounds version, but initialize the
        # exp cutoffs *outside* the data range so they don't suppress data
        # at the starting point. exp(-1) cutoff at logmmin sits right at data
        # min, so we put logmmin a decade below to keep the cutoff out of the
        # data range during the first MAP evaluation.
        base = imf_mostlikely_params(
            masses, "chabrier_smooth", logmmin=logmmin, logmmax=logmmax,
        ).x
        logm = np.log10(np.asarray(masses).ravel())
        return list(base) + [float(logm.min()) - 1.0, float(logm.max()) + 1.0]
    if lower == "chabrier_exp_bounds":
        base = imf_mostlikely_params(
            masses, "chabrier_smooth", logmmin=logmmin, logmmax=logmmax,
        ).x
        logm0, logsigma, alpha = base
        sigma = float(np.exp(logsigma))
        logmbreak = float(logm0) - float(alpha) * sigma * sigma * float(np.log(10.0))
        logm = np.log10(np.asarray(masses).ravel())
        return list(base) + [logmbreak, float(logm.min()) - 1.0, float(logm.max()) + 1.0]
    return imf_default_params(model)


def imf_lnprob(params, masses, model=DEFAULT_MODEL, logmmin=None, logmmax=None):
    """Total log-probability of ``masses`` under ``model`` with ``params``.

    Returns a scalar JAX array, safe to feed to ``jax.grad``.
    """
    imf_fn = _resolve_imf_func(model)
    logm = jnp.log10(jnp.asarray(masses).ravel())
    if logmmin is None:
        logmmin = jnp.min(logm)
    if logmmax is None:
        logmmax = jnp.max(logm)
    imf_val = imf_fn(logm, jnp.asarray(params), logmmin, logmmax)
    return jnp.sum(jnp.log(imf_val))


def imf_mostlikely_params(
    masses,
    model=DEFAULT_MODEL,
    bounds=None,
    p0=None,
    logmmin=None,
    logmmax=None,
):
    """Maximum-a-posteriori IMF parameters via Nelder-Mead.

    Returns the ``scipy.optimize.OptimizeResult`` from ``minimize``.

    Notes
    -----
    Uses Nelder-Mead despite JAX giving us analytic gradients, because
    L-BFGS-B's initial Hessian-identity step takes ~unit-length jumps in
    the gradient direction. On well-fit datasets, the line search bounds
    the chabrier_smooth IMF off into the underflow regime (extreme alpha
    + narrow sigma -> IMF identically 0 at data points -> NaN gradient),
    and L-BFGS-B exits with ABNORMAL on the very first iteration. Nelder-
    Mead is gradient-free and only evaluates the function, so it sails
    through these regions without issue. The few hundred lnprob calls
    Nelder-Mead needs are negligible vs the NUTS sampling cost.
    """
    if p0 is None:
        p0 = _bootstrap_p0(masses, model, logmmin=logmmin, logmmax=logmmax)
    p0 = np.asarray(p0, dtype=np.float64)
    if bounds is None:
        bounds = imf_default_bounds(model)

    imf_fn = _resolve_imf_func(model)
    logm = jnp.log10(jnp.asarray(masses).ravel())
    lmin = jnp.min(logm) if logmmin is None else jnp.asarray(logmmin, dtype=jnp.float64)
    lmax = jnp.max(logm) if logmmax is None else jnp.asarray(logmmax, dtype=jnp.float64)

    @jax.jit
    def neg_lnprob_jit(p):
        imf_val = imf_fn(logm, p, lmin, lmax)
        return -jnp.sum(jnp.log(imf_val))

    def neg_lp(p_np):
        return float(neg_lnprob_jit(jnp.asarray(p_np, dtype=jnp.float64)))

    sol = minimize(neg_lp, p0, bounds=bounds, method="Nelder-Mead")
    return sol
