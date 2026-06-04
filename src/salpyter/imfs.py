"""Differentiable JAX IMF functions, each decorated with ``@imf_model`` to
auto-register a callable :class:`~salpyter.model.IMFModel` under the function's
public name.

Each underlying IMF takes ``logm`` (log10 mass), a ``params`` vector, and scalar
mass-range bounds ``logmmin``/``logmmax`` defining the normalization range. It
returns the IMF value (normalized to integrate to 1 over [logmmin, logmmax]
with respect to log10(m)).

All functions are pure JAX and safe to ``jit``, ``grad``, and ``vmap``. After
decoration, ``chabrier_smooth_imf`` (etc.) is an ``IMFModel`` instance that is
*still callable* via ``__call__``, so direct calls like
``chabrier_smooth_imf(logm, params)`` keep working.
"""

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erf

from .default_imf_params import DEFAULT_IMF_PARAMS, DEFAULT_IMF_PARAMS_BOUNDS
from .model import imf_model

_LN10 = jnp.log(10.0)
_SQRT2 = jnp.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / jnp.sqrt(2.0 * jnp.pi)


# --------------------------------------------------------------------------- #
# Bootstrap functions for the bounded models.                                 #
# --------------------------------------------------------------------------- #
# These move the per-model MAP-initialization rules from likelihood._bootstrap_p0
# onto the IMFModel objects themselves. They use lazy imports of
# imf_mostlikely_params to avoid a circular import (likelihood -> imfs -> ...).


def _bootstrap_chabrier(masses, logmmin, logmmax):
    from .likelihood import imf_mostlikely_params
    base = imf_mostlikely_params(masses, "chabrier_smooth", logmmin=logmmin, logmmax=logmmax).x
    logm0, logsigma, alpha = [float(x) for x in base]
    sigma = float(np.exp(logsigma))
    logmbreak = logm0 - alpha * sigma * sigma * float(np.log(10.0))
    return [logm0, logsigma, alpha, logmbreak]


def _bootstrap_chabrier_smooth_bounds(masses, logmmin, logmmax):
    from .likelihood import imf_mostlikely_params
    base = imf_mostlikely_params(masses, "chabrier_smooth", logmmin=logmmin, logmmax=logmmax).x
    logm = np.log10(np.asarray(masses).ravel())
    return [float(x) for x in base] + [float(logm.min()) - 1e-3, float(logm.max()) + 1e-3]


def _bootstrap_chabrier_smooth_exp_bounds(masses, logmmin, logmmax):
    from .likelihood import imf_mostlikely_params
    base = imf_mostlikely_params(masses, "chabrier_smooth", logmmin=logmmin, logmmax=logmmax).x
    logm = np.log10(np.asarray(masses).ravel())
    return [float(x) for x in base] + [float(logm.min()) - 1.0, float(logm.max()) + 1.0]


def _bootstrap_chabrier_exp_bounds(masses, logmmin, logmmax):
    from .likelihood import imf_mostlikely_params
    base = imf_mostlikely_params(masses, "chabrier_smooth", logmmin=logmmin, logmmax=logmmax).x
    logm0, logsigma, alpha = [float(x) for x in base]
    sigma = float(np.exp(logsigma))
    logmbreak = logm0 - alpha * sigma * sigma * float(np.log(10.0))
    logm = np.log10(np.asarray(masses).ravel())
    return [logm0, logsigma, alpha, logmbreak, float(logm.min()) - 1.0, float(logm.max()) + 1.0]


@imf_model(
    name="chabrier_smooth",
    param_names=("logm0", "logsigma", "alpha"),
    default_params=tuple(DEFAULT_IMF_PARAMS["chabrier_smooth"]),
    default_bounds=tuple(tuple(b) for b in DEFAULT_IMF_PARAMS_BOUNDS["chabrier_smooth"]),
)
def chabrier_smooth_imf(logm, params, logmmin=-jnp.inf, logmmax=4.0):
    """Chabrier IMF with a smooth high-mass break.

    A lognormal at low mass that transitions smoothly into a power-law tail.
    The break point ``logmbreak`` is set by the continuity-of-derivative
    condition (same as the master-branch ``chabrier_smooth_imf``).

    Parameters
    ----------
    logm : array_like
        log10(mass) at which to evaluate the IMF.
    params : array_like, shape (3,)
        ``[logm0, logsigma, alpha]`` where ``logm0`` is the log10 of the
        lognormal peak mass, ``logsigma`` is the log of the (log10-units)
        lognormal width, and ``alpha`` is the high-mass slope.
    logmmin, logmmax : float
        log10 mass-range bounds defining the normalization integral.

    Returns
    -------
    jnp.ndarray
        Same shape as ``logm``. The IMF normalized so the integral over
        log10(m) in [logmmin, logmmax] is 1.
    """
    logm = jnp.asarray(logm)
    params = jnp.asarray(params)
    logm0 = params[0]
    logsigma = params[1]
    alpha = params[2]

    sigma = jnp.exp(logsigma)
    inv_sigma = 1.0 / sigma
    # Smooth-break condition: the powerlaw matches the lognormal's value and
    # log-derivative at logmbreak.
    logmbreak = logm0 - alpha * sigma * sigma * _LN10
    mbreak = 10.0**logmbreak

    z = (logm - logm0) * inv_sigma
    lognormal = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z * z)

    z_break = (logmbreak - logm0) * inv_sigma
    normal_at_break = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z_break * z_break)

    m = 10.0**logm
    powerlaw = normal_at_break * (m / mbreak) ** alpha

    imf_pre = jnp.where(logm > logmbreak, powerlaw, lognormal)

    # Normalization. The lognormal contributes only on [logmmin, min(logmmax, logmbreak)];
    # the power-law contributes only on [max(logmmin, logmbreak), logmmax].
    upper_cap = jnp.minimum(logmmax, logmbreak)
    X1 = (logmmin - logm0) * inv_sigma
    X2 = (upper_cap - logm0) * inv_sigma
    lognormal_norm = jnp.where(
        logmmin < logmbreak,
        0.5 * (erf(X2 / _SQRT2) - erf(X1 / _SQRT2)),
        0.0,
    )

    mmin = 10.0**logmmin
    mmax = 10.0**logmmax
    xmin_pl = jnp.maximum(mmin, mbreak)
    # powerlaw_integral(xmin, xmax, alpha-1) = (xmax**alpha - xmin**alpha) / alpha,
    # for alpha != 0. NUTS proposes alpha continuously so alpha == 0 has measure 0.
    pl_integral = (mmax**alpha - xmin_pl**alpha) / alpha
    powerlaw_norm = jnp.where(
        logmmax > logmbreak,
        normal_at_break * mbreak ** (-alpha) * pl_integral / _LN10,
        0.0,
    )
    norm = lognormal_norm + powerlaw_norm

    return imf_pre / norm


@imf_model(
    name="chabrier_smooth_bounds",
    param_names=("logm0", "logsigma", "alpha", "logmmin", "logmmax"),
    default_params=tuple(DEFAULT_IMF_PARAMS["chabrier_smooth_bounds"]),
    default_bounds=tuple(tuple(b) for b in DEFAULT_IMF_PARAMS_BOUNDS["chabrier_smooth_bounds"]),
    bootstrap_fn=_bootstrap_chabrier_smooth_bounds,
)
def chabrier_smooth_bounds_imf(logm, params, logmmin=-jnp.inf, logmmax=4.0):
    """Chabrier-smooth IMF with sampled low/high-mass cutoffs (5 parameters).

    The last two parameters ``logmmin``, ``logmmax`` are the mass-range cutoffs
    of the IMF support. The IMF is set to zero outside ``[logmmin, logmmax]``
    (matching the master-branch ``imf_with_bounds_params`` behavior), and the
    normalization integral is over the same range.

    The ``logmmin``/``logmmax`` keyword arguments are accepted for signature
    parity with the other IMF functions but are ignored; the cutoffs are taken
    from ``params``.

    Parameters
    ----------
    params : array_like, shape (5,)
        ``[logm0, logsigma, alpha, logmmin, logmmax]``.
    """
    del logmmin, logmmax  # cutoffs come from params, not kwargs
    logm = jnp.asarray(logm)
    params = jnp.asarray(params)
    logm0 = params[0]
    logsigma = params[1]
    alpha = params[2]
    logmmin = params[3]
    logmmax = params[4]

    sigma = jnp.exp(logsigma)
    inv_sigma = 1.0 / sigma
    logmbreak = logm0 - alpha * sigma * sigma * _LN10
    mbreak = 10.0**logmbreak

    z = (logm - logm0) * inv_sigma
    lognormal = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z * z)

    z_break = (logmbreak - logm0) * inv_sigma
    normal_at_break = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z_break * z_break)

    m = 10.0**logm
    powerlaw = normal_at_break * (m / mbreak) ** alpha

    imf_pre = jnp.where(logm > logmbreak, powerlaw, lognormal)
    # Zero outside the sampled support; this is what makes the bounded model
    # distinct from chabrier_smooth — data points outside [logmmin, logmmax]
    # contribute log(0) ≈ -691 per point under the lnprob clip, strongly
    # rejecting proposals whose support doesn't cover the data.
    inside = (logm >= logmmin) & (logm <= logmmax)
    imf_pre = jnp.where(inside, imf_pre, 0.0)

    upper_cap = jnp.minimum(logmmax, logmbreak)
    X1 = (logmmin - logm0) * inv_sigma
    X2 = (upper_cap - logm0) * inv_sigma
    lognormal_norm = jnp.where(
        logmmin < logmbreak,
        0.5 * (erf(X2 / _SQRT2) - erf(X1 / _SQRT2)),
        0.0,
    )

    mmin = 10.0**logmmin
    mmax = 10.0**logmmax
    xmin_pl = jnp.maximum(mmin, mbreak)
    pl_integral = (mmax**alpha - xmin_pl**alpha) / alpha
    powerlaw_norm = jnp.where(
        logmmax > logmbreak,
        normal_at_break * mbreak ** (-alpha) * pl_integral / _LN10,
        0.0,
    )
    norm = lognormal_norm + powerlaw_norm

    return imf_pre / norm


# Number of log-mass grid points used by the *_exp_bounds models for numerical
# normalization. 501 spans ~10 decades at 0.02-decade resolution, plenty to
# resolve the exp cutoff transition (which happens over ~1 decade).
_EXP_NORM_NGRID = 501
# Margin (in log10 mass) on each side of [logmmin, logmmax] for the
# normalization integration domain. 5 decades is enough for the cutoff to
# decay to exp(-1e5), entirely negligible.
_EXP_NORM_MARGIN = 5.0


def _chabrier_smooth_shape_unnorm(logm, logm0, logsigma, alpha):
    """chabrier_smooth unnormalized shape (smooth-break condition)."""
    sigma = jnp.exp(logsigma)
    inv_sigma = 1.0 / sigma
    logmbreak = logm0 - alpha * sigma * sigma * _LN10
    mbreak = 10.0**logmbreak

    z = (logm - logm0) * inv_sigma
    lognormal = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z * z)

    z_break = (logmbreak - logm0) * inv_sigma
    normal_at_break = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z_break * z_break)
    m = 10.0**logm
    powerlaw = normal_at_break * (m / mbreak) ** alpha
    return jnp.where(logm > logmbreak, powerlaw, lognormal)


def _chabrier_shape_unnorm(logm, logm0, logsigma, alpha, logmbreak):
    """chabrier unnormalized shape (free logmbreak, no smooth condition)."""
    sigma = jnp.exp(logsigma)
    inv_sigma = 1.0 / sigma
    mbreak = 10.0**logmbreak

    z = (logm - logm0) * inv_sigma
    lognormal = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z * z)

    z_break = (logmbreak - logm0) * inv_sigma
    normal_at_break = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z_break * z_break)
    m = 10.0**logm
    powerlaw = normal_at_break * (m / mbreak) ** alpha
    return jnp.where(logm > logmbreak, powerlaw, lognormal)


def _schechter_cutoff(logm, logmmin, logmmax):
    """Smooth Schechter-style cutoff: exp(-m_min/m - m/m_max) in log space.

    Equal to ~1 well inside [logmmin, logmmax], exp(-1) ≈ 0.37 at each cutoff
    itself, decays exponentially outside. Smooth everywhere (NUTS-safe).
    """
    return jnp.exp(-(10.0 ** (logmmin - logm)) - (10.0 ** (logm - logmmax)))


def _exp_bounds_norm(shape_fn, cutoff_fn, logmmin_p, logmmax_p):
    """Numerical normalization integral over a wide log-mass grid."""
    grid_logm = jnp.linspace(
        logmmin_p - _EXP_NORM_MARGIN,
        logmmax_p + _EXP_NORM_MARGIN,
        _EXP_NORM_NGRID,
    )
    return jnp.trapezoid(shape_fn(grid_logm) * cutoff_fn(grid_logm), grid_logm)


@imf_model(
    name="chabrier_smooth_exp_bounds",
    param_names=("logm0", "logsigma", "alpha", "logmmin", "logmmax"),
    default_params=tuple(DEFAULT_IMF_PARAMS["chabrier_smooth_exp_bounds"]),
    default_bounds=tuple(tuple(b) for b in DEFAULT_IMF_PARAMS_BOUNDS["chabrier_smooth_exp_bounds"]),
    bootstrap_fn=_bootstrap_chabrier_smooth_exp_bounds,
)
def chabrier_smooth_exp_bounds_imf(logm, params, logmmin=-jnp.inf, logmmax=4.0):
    """chabrier_smooth IMF with Schechter-style exponential mass cutoffs.

    Replaces the hard ``imf=0 outside [logmmin, logmmax]`` step in
    ``chabrier_smooth_bounds_imf`` with the smooth Schechter-like factor
    ``exp(-m_min/m - m/m_max)``. The cutoff is ~1 well inside ``[m_min, m_max]``,
    exp(-1) ≈ 0.37 at each cutoff itself, and decays exponentially outside.
    Because the integral is no longer analytic, normalization is computed
    numerically by trapezoidal integration over a 501-point log-mass grid.

    The ``logmmin``/``logmmax`` keyword arguments are accepted for signature
    parity with the other IMF functions but are ignored; the cutoffs are taken
    from ``params``.

    Parameters
    ----------
    params : array_like, shape (5,)
        ``[logm0, logsigma, alpha, logmmin, logmmax]``.
    """
    del logmmin, logmmax
    logm = jnp.asarray(logm)
    params = jnp.asarray(params)
    logm0 = params[0]
    logsigma = params[1]
    alpha = params[2]
    logmmin_p = params[3]
    logmmax_p = params[4]

    def shape(lm):
        return _chabrier_smooth_shape_unnorm(lm, logm0, logsigma, alpha)

    def cutoff(lm):
        return _schechter_cutoff(lm, logmmin_p, logmmax_p)

    return shape(logm) * cutoff(logm) / _exp_bounds_norm(shape, cutoff, logmmin_p, logmmax_p)


@imf_model(
    name="chabrier_exp_bounds",
    param_names=("logm0", "logsigma", "alpha", "logmbreak", "logmmin", "logmmax"),
    default_params=tuple(DEFAULT_IMF_PARAMS["chabrier_exp_bounds"]),
    default_bounds=tuple(tuple(b) for b in DEFAULT_IMF_PARAMS_BOUNDS["chabrier_exp_bounds"]),
    bootstrap_fn=_bootstrap_chabrier_exp_bounds,
)
def chabrier_exp_bounds_imf(logm, params, logmmin=-jnp.inf, logmmax=4.0):
    """chabrier IMF (free logmbreak) with Schechter-style exponential cutoffs.

    Same idea as ``chabrier_smooth_exp_bounds_imf`` but with a free high-mass
    break (chabrier rather than chabrier_smooth shape).

    Parameters
    ----------
    params : array_like, shape (6,)
        ``[logm0, logsigma, alpha, logmbreak, logmmin, logmmax]``.
    """
    del logmmin, logmmax
    logm = jnp.asarray(logm)
    params = jnp.asarray(params)
    logm0 = params[0]
    logsigma = params[1]
    alpha = params[2]
    logmbreak = params[3]
    logmmin_p = params[4]
    logmmax_p = params[5]

    def shape(lm):
        return _chabrier_shape_unnorm(lm, logm0, logsigma, alpha, logmbreak)

    def cutoff(lm):
        return _schechter_cutoff(lm, logmmin_p, logmmax_p)

    return shape(logm) * cutoff(logm) / _exp_bounds_norm(shape, cutoff, logmmin_p, logmmax_p)


# --------------------------------------------------------------------------- #
# Base components for piecewise composition.                                  #
# --------------------------------------------------------------------------- #


@imf_model(
    name="lognormal",
    param_names=("logm0", "logsigma"),
    default_params=(float(np.log10(0.25)), float(np.log(0.55))),
    # logm0 spans (-4, 4) so any astrophysically plausible peak (1e-4 to 1e4 Msun)
    # is inside the bound — matches the cutoff-bound convention used by
    # chabrier_smooth_bounds and friends. logsigma stays at (-2, 2), giving widths
    # in [exp(-2), exp(2)] ≈ [0.14, 7.4] dex.
    default_bounds=((-4.0, 4.0), (-2.0, 2.0)),
)
def lognormal_imf(logm, params, logmmin=-jnp.inf, logmmax=4.0):
    """Lognormal IMF in dN/d(log10 m) units.

    A Gaussian in log10(m) with peak at ``logm0`` and width ``exp(logsigma)``
    (in log10-mass units), normalized to integrate to 1 over
    ``[logmmin, logmmax]``. This is the standalone shape that sits below the
    high-mass power-law in the Chabrier composition.

    Parameters
    ----------
    params : array_like, shape (2,)
        ``[logm0, logsigma]`` — peak log10(mass) and log of the (log10-units)
        width.
    """
    logm = jnp.asarray(logm)
    params = jnp.asarray(params)
    logm0 = params[0]
    logsigma = params[1]
    sigma = jnp.exp(logsigma)
    inv_sigma = 1.0 / sigma
    z = (logm - logm0) * inv_sigma
    shape = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z * z)
    X1 = (logmmin - logm0) * inv_sigma
    X2 = (logmmax - logm0) * inv_sigma
    norm = 0.5 * (erf(X2 / _SQRT2) - erf(X1 / _SQRT2))
    return shape / norm


@imf_model(
    name="powerlaw",
    param_names=("slope",),
    default_params=tuple(DEFAULT_IMF_PARAMS["powerlaw"]),
    default_bounds=tuple(tuple(b) for b in DEFAULT_IMF_PARAMS_BOUNDS["powerlaw"]),
)
def powerlaw_imf(logm, params, logmmin=-jnp.inf, logmmax=4.0):
    """Single power-law IMF in dN/d(log10 m) units.

    With slope ``s``, the IMF value is ``m^s`` normalized to integrate to 1
    over ``[logmmin, logmmax]``. The dN/dm convention has slope ``s - 1``;
    Salpeter is ``s = -1.35`` here (corresponding to dN/dm ∝ m^-2.35).

    Parameters
    ----------
    params : array_like, shape (1,)
        ``[slope]`` in dN/d(log10 m) units.
    """
    logm = jnp.asarray(logm)
    params = jnp.asarray(params)
    slope = params[0]
    mmin = 10.0**logmmin
    mmax = 10.0**logmmax
    # ∫ m^s d(log10 m) = (mmax^s - mmin^s) / (s * ln 10).
    norm = (mmax**slope - mmin**slope) / (slope * _LN10)
    return (10.0**logm) ** slope / norm


def _schechter_cutoff_fn(logm, params):
    """``exp(-m_min/m - m/m_max)``; params = [logmmin, logmmax]."""
    return jnp.exp(-(10.0 ** (params[0] - logm)) - (10.0 ** (logm - params[1])))


def _schechter_support_hint(params):
    return params[0] - _EXP_NORM_MARGIN, params[1] + _EXP_NORM_MARGIN
