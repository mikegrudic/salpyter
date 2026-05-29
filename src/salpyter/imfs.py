"""Differentiable JAX IMF functions.

Each IMF takes ``logm`` (log10 mass), a ``params`` vector, and scalar mass-range
bounds ``logmmin``/``logmmax`` defining the normalization range. It returns the
IMF value (normalized to integrate to 1 over [logmmin, logmmax] with respect to
log10(m)).

All functions are pure JAX and safe to ``jit``, ``grad``, and ``vmap``.
"""

import jax.numpy as jnp
from jax.scipy.special import erf

_LN10 = jnp.log(10.0)
_SQRT2 = jnp.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / jnp.sqrt(2.0 * jnp.pi)


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


def chabrier_imf(logm, params, logmmin=-jnp.inf, logmmax=4.0):
    """Chabrier IMF with a free high-mass break (4 parameters).

    Same lognormal-plus-powerlaw form as ``chabrier_smooth_imf`` but the break
    point ``logmbreak`` is sampled independently rather than derived from the
    smooth-derivative condition.

    Parameters
    ----------
    params : array_like, shape (4,)
        ``[logm0, logsigma, alpha, logmbreak]``.
    """
    logm = jnp.asarray(logm)
    params = jnp.asarray(params)
    logm0 = params[0]
    logsigma = params[1]
    alpha = params[2]
    logmbreak = params[3]

    sigma = jnp.exp(logsigma)
    inv_sigma = 1.0 / sigma
    mbreak = 10.0**logmbreak

    z = (logm - logm0) * inv_sigma
    lognormal = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z * z)

    z_break = (logmbreak - logm0) * inv_sigma
    normal_at_break = _INV_SQRT_2PI * inv_sigma * jnp.exp(-0.5 * z_break * z_break)

    m = 10.0**logm
    powerlaw = normal_at_break * (m / mbreak) ** alpha

    imf_pre = jnp.where(logm > logmbreak, powerlaw, lognormal)

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


def chabrier_smooth_bounds_imf(logm, params, logmmin_unused=-jnp.inf, logmmax_unused=4.0):
    """Chabrier-smooth IMF with sampled low/high-mass cutoffs (5 parameters).

    The last two parameters ``logmmin``, ``logmmax`` are the mass-range cutoffs
    of the IMF support. The IMF is set to zero outside ``[logmmin, logmmax]``
    (matching the master-branch ``imf_with_bounds_params`` behavior), and the
    normalization integral is over the same range.

    The ``logmmin_unused``/``logmmax_unused`` arguments are accepted for
    signature parity with the other IMF functions; the cutoffs are taken from
    ``params``.

    Parameters
    ----------
    params : array_like, shape (5,)
        ``[logm0, logsigma, alpha, logmmin, logmmax]``.
    """
    del logmmin_unused, logmmax_unused
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
