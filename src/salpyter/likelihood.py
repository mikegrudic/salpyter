"""IMF log-likelihood and MAP estimation (jax branch).

``imf_lnprob`` is the JAX-differentiable log-probability used by both the MAP
estimator (L-BFGS-B with autodiff gradients) and the NUTS sampler.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .default_imf_params import DEFAULT_MODEL
from .model import IMFModel, _REGISTRY


def _resolve_model(model):
    """Return the :class:`IMFModel` for ``model``.

    Accepts either a string registered in the model registry or an
    ``IMFModel`` instance directly (so composed-on-the-fly models like
    ``piecewise(...)`` work without registration).
    """
    if isinstance(model, IMFModel):
        return model
    if isinstance(model, str):
        try:
            return _REGISTRY[model.lower()]
        except KeyError:
            raise NotImplementedError(
                f"unknown model {model!r}; registered: {sorted(_REGISTRY)}"
            )
    raise TypeError(f"model must be a string or IMFModel, got {type(model).__name__}")


def _resolve_imf_func(model):
    return _resolve_model(model).imf_fn


# Backward-compatible dict view for any callers that import this directly.
_MODEL_TO_FUNC = {name: m.imf_fn for name, m in _REGISTRY.items()}


def _bootstrap_p0(masses, model, logmmin=None, logmmax=None):
    """Pick a sensible starting point for the MAP optimizer.

    Each registered :class:`IMFModel` carries its own ``bootstrap_fn`` (set
    via the ``@imf_model`` decorator in ``imfs.py``); we just look up the
    model and call it. Without a bootstrap, the bounded models' MAPs from
    ``default_params`` land in the wrong local mode (narrow lognormal + loose
    bounds, ~150 lower log-likelihood than the global mode of wide lognormal
    + bounds at the data extrema).
    """
    resolved = _resolve_model(model)
    if resolved.bootstrap_fn is not None:
        return resolved.bootstrap_fn(masses, logmmin, logmmax)
    return list(resolved.default_params)


def imf_lnprob(params, masses, model=DEFAULT_MODEL, logmmin=None, logmmax=None):
    """Total log-probability of ``masses`` under ``model`` with ``params``.

    Returns a scalar JAX array, safe to feed to ``jax.grad``.
    """
    imf_fn = _resolve_model(model).imf_fn
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
        resolved = _resolve_model(model)
        bounds = [list(b) for b in resolved.default_bounds]

    imf_fn = _resolve_model(model).imf_fn
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
