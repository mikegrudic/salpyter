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
}


def _resolve_imf_func(model):
    fn = _MODEL_TO_FUNC.get(model.lower())
    if fn is None:
        raise NotImplementedError(
            f"jax salpyter supports {sorted(_MODEL_TO_FUNC)}; got {model!r}"
        )
    return fn


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
    """Maximum-a-posteriori IMF parameters via L-BFGS-B with autodiff gradients.

    Faster and more robust than the master-branch Nelder-Mead estimator because
    L-BFGS-B uses the analytic JAX gradient.

    Returns the ``scipy.optimize.OptimizeResult`` from ``minimize``.
    """
    if p0 is None:
        p0 = imf_default_params(model)
    p0 = np.asarray(p0, dtype=np.float64)
    if bounds is None:
        bounds = imf_default_bounds(model)

    imf_fn = _resolve_imf_func(model)
    logm = jnp.log10(jnp.asarray(masses).ravel())
    lmin = jnp.min(logm) if logmmin is None else jnp.asarray(logmmin, dtype=jnp.float64)
    lmax = jnp.max(logm) if logmmax is None else jnp.asarray(logmmax, dtype=jnp.float64)

    @jax.jit
    def neg_lnprob_and_grad(p):
        def fn(q):
            imf_val = imf_fn(logm, q, lmin, lmax)
            return -jnp.sum(jnp.log(imf_val))

        return jax.value_and_grad(fn)(p)

    def fun_and_grad(p_np):
        val, grad = neg_lnprob_and_grad(jnp.asarray(p_np, dtype=jnp.float64))
        return float(val), np.asarray(grad, dtype=np.float64)

    sol = minimize(fun_and_grad, p0, jac=True, bounds=bounds, method="L-BFGS-B")
    return sol
