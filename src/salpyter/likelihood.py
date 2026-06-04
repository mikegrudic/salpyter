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
        name = model.lower()
        if name in _REGISTRY:
            return _REGISTRY[name]
        # Convention: <base>_bounds is automatically synthesized as
        # ``<base>.truncate()`` with a data-extrema bootstrap. Lets the user
        # write ``imf_lnprob_samples(masses, model="chabrier_bounds")`` for any
        # base model without hand-defining a bounded variant.
        if name.endswith("_bounds"):
            base_name = name[: -len("_bounds")]
            if base_name in _REGISTRY:
                return _make_truncate_alias(_REGISTRY[base_name], name)
        # Convention: <base>_lognormal is automatically synthesized as the
        # mixture ``<base> + lognormal`` — i.e. the base IMF with an additional
        # lognormal "bump" component, mixed via a logit weight. Use to fit an
        # excess of stars (e.g. a brown-dwarf hump) on top of any registered base.
        if name.endswith("_lognormal") and name != "lognormal":
            base_name = name[: -len("_lognormal")]
            if base_name in _REGISTRY:
                return _make_lognormal_mixture_alias(_REGISTRY[base_name], name)
        raise NotImplementedError(
            f"unknown model {model!r}; registered: {sorted(_REGISTRY)}"
        )
    raise TypeError(f"model must be a string or IMFModel, got {type(model).__name__}")


def _make_truncate_alias(base, name):
    """Build, register, and return ``base.truncate()`` under ``name``.

    The synthesized model adds two parameters ``(logmmin, logmmax)`` to the
    base model and zeros the IMF outside that range. A bootstrap_fn is
    attached so the MAP optimizer starts with the base-model MAP plus
    cutoffs at the data extrema — without it the optimizer would settle in
    a wrong local mode (narrow base + loose cutoffs, ~150 lower log-likelihood
    than the global mode of wide base + tight cutoffs).
    """
    import dataclasses

    bounded = base.truncate()

    def bootstrap(masses, logmmin, logmmax):
        # Lazy-import to avoid the `imf_mostlikely_params` self-reference at
        # module load time.
        base_x = imf_mostlikely_params(
            masses, base, logmmin=logmmin, logmmax=logmmax,
        ).x
        logm = np.log10(np.asarray(masses).ravel())
        return list(base_x) + [float(logm.min()) - 1e-3, float(logm.max()) + 1e-3]

    bounded = dataclasses.replace(bounded, name=name, bootstrap_fn=bootstrap)
    _REGISTRY[name] = bounded
    return bounded


def _make_lognormal_mixture_alias(base, name):
    """Build, register, and return ``base + (unbounded) lognormal`` under ``name``.

    Synthesizes a mixture model that adds an extra lognormal "bump" to the
    base IMF. The new parameters are ``(logm0, logsigma, logit_w)`` for the
    lognormal location/width and the mixture logit weight (sigmoid(logit_w)
    is the base weight, 1-sigmoid(logit_w) is the lognormal weight).
    Duplicate component names (e.g. "logm0" in chabrier_smooth + lognormal)
    get the standard ``_1`` / ``_2`` suffix from ``__add__``.

    The lognormal component is always normalized over its natural support
    ``(-inf, +inf)`` — i.e. its erf-normalization integral spans the whole
    real line, not ``[data_logm_min, data_logm_max]``. This keeps a high-mass
    bump from having its tail above the data folded back in, which would
    otherwise make a bump centered above the data act as a tail rather than
    a localized excess. The base is normalized over whatever range it
    natively uses (e.g. its own sampled cutoffs for ``_bounds`` bases).

    Bootstrap fits the base alone, then seeds the lognormal at ``log10(max(masses))``
    with width 0.2 dex (``logsigma = log(0.2)``) and ``logit_w = 4``
    (≈98% base / 2% bump). Pitched toward catching a *narrow* high-mass
    excess — the lnL landscape has a broad-bump local mode at σ ≈ 0.5 dex
    and a narrow-tail global mode at σ ≈ 0.1-0.2 dex; seeding narrow puts
    Nelder-Mead in the right basin. The optimizer is free to widen if the
    bump actually is broad.
    """
    import dataclasses

    import jax.numpy as jnp_local  # avoid shadowing the module-level jnp alias

    if "lognormal" not in _REGISTRY:
        raise RuntimeError(
            "_lognormal suffix requires the 'lognormal' base model to be "
            "registered, but it isn't."
        )
    lognormal = _REGISTRY["lognormal"]

    # Wrap the lognormal's imf_fn so it ignores the outer ``[logmmin, logmmax]``
    # passed by ``__add__`` and instead normalizes over the full real line. This
    # is what makes the bump "unbounded".
    _lognormal_natural_imf = lognormal.imf_fn

    def _lognormal_unbounded_imf_fn(logm, params, logmmin=-jnp_local.inf, logmmax=4.0):
        del logmmin, logmmax
        return _lognormal_natural_imf(logm, params, -jnp_local.inf, jnp_local.inf)

    unbounded_lognormal = dataclasses.replace(
        lognormal, imf_fn=_lognormal_unbounded_imf_fn,
    )
    mixture = base + unbounded_lognormal

    # Capture an unparameterized mixture for the multi-start MAP calls inside
    # bootstrap; the registered, named version is built below.
    mixture_unnamed = mixture

    def bootstrap(masses, logmmin, logmmax):
        base_x = imf_mostlikely_params(
            masses, base, logmmin=logmmin, logmmax=logmmax,
        ).x
        logm_max = float(np.log10(np.asarray(masses).ravel().max()))
        # The (logsigma_lognormal, logmmax_base) sub-landscape is multi-modal:
        # a "broad bump" basin sits at σ ≈ 0.5 dex with the base cutoff near the
        # data max, and a "narrow tail" basin at σ ≈ 0.1 dex with the base
        # cutoff pulled in. One bootstrap seed only finds one basin, so try a
        # few logsigma starts and return the one whose MAP has the lowest -lnL.
        best_p, best_fun = None, np.inf
        for logsigma_seed in (float(np.log(0.2)), float(np.log(0.5))):
            p_seed = list(base_x) + [logm_max, logsigma_seed, 4.0]
            sol = imf_mostlikely_params(
                masses, mixture_unnamed, p0=p_seed,
                logmmin=logmmin, logmmax=logmmax,
            )
            if sol.fun < best_fun:
                best_fun = sol.fun
                best_p = sol.x
        return list(best_p)

    mixture = dataclasses.replace(mixture, name=name, bootstrap_fn=bootstrap)
    _REGISTRY[name] = mixture
    return mixture


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


def imf_log_slope(logm, params, model=DEFAULT_MODEL, logmmin=-jnp.inf, logmmax=4.0):
    r"""Local logarithmic slope of the IMF, :math:`\Gamma = d \log_{10}\xi / d \log_{10} m`.

    Computed via ``jax.grad`` of ``log(imf_fn(logm, params, ...))`` w.r.t. its
    ``logm`` argument and vmapped across the input grid. Since the IMF
    functions return :math:`dN/d\log_{10} m` (per-log10-mass), this returns
    :math:`\Gamma` in the dN/dlog10m convention — Salpeter is :math:`-1.35`.

    Parameters
    ----------
    logm : array_like
        log10(mass) at which to evaluate the slope.
    params : array_like
        IMF parameters for ``model``.
    model : str or IMFModel
        Model name (registry lookup) or an ``IMFModel`` instance.
    logmmin, logmmax : float
        Mass-range bounds passed to the underlying IMF function. For models
        that take their cutoffs from ``params`` (e.g. ``chabrier_smooth_bounds``)
        these are ignored.

    Returns
    -------
    jnp.ndarray
        Same shape as ``logm``.

    Notes
    -----
    For hard-bounded models the IMF is exactly zero outside the support, so
    :math:`\Gamma` is undefined (or ``nan``) there — evaluate only at points
    where the IMF is positive.
    """
    fn = _resolve_model(model).imf_fn
    params_arr = jnp.asarray(params)
    lmin_arr = jnp.asarray(logmmin)
    lmax_arr = jnp.asarray(logmmax)

    def _scalar_log_imf(x):
        return jnp.log(fn(jnp.atleast_1d(x), params_arr, lmin_arr, lmax_arr)[0])

    # d ln(imf) / d log10(m) == d log10(imf) / d log10(m) * ln(10), so divide
    # the autodiff result by ln(10) to get the log10/log10 slope.
    logm_arr = jnp.atleast_1d(jnp.asarray(logm))
    slope_ln = jax.vmap(jax.grad(_scalar_log_imf))(logm_arr)
    return slope_ln / jnp.log(10.0)


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

    # scipy's Nelder-Mead defaults to maxfev = maxiter = N*200, which is enough
    # for 3-4 param models but cuts off too early on 8-D mixture models like
    # ``chabrier_smooth_bounds_lognormal``. Bump to N*1000 so high-D MAPs reach
    # tolerance; cheap-to-compute lnprob makes the cost negligible.
    n_dim = len(p0)
    sol = minimize(
        neg_lp, p0, bounds=bounds, method="Nelder-Mead",
        options={"maxiter": 1000 * n_dim, "maxfev": 1000 * n_dim},
    )
    # Attach the model's parameter names so callers can introspect the result
    # without re-resolving the model. ``sol.x`` is still the bare array; ``sol.params``
    # is the {name: value} dict for the typical "what fit did I get?" lookup.
    names = tuple(_resolve_model(model).param_names)
    sol.param_names = names
    sol.params = {name: float(v) for name, v in zip(names, sol.x)}
    return sol
