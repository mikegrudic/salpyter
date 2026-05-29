"""IMF sampling (jax branch).

* :func:`imf_samples` — draw masses from a known IMF via rejection sampling.
  (Same approach as master; no need to JAX-ify the sampler.)
* :func:`imf_lnprob_samples` — draw posterior samples of IMF parameters from
  data, using NUTS via ``blackjax``. Replaces the master-branch emcee path.
"""

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from .default_imf_params import (
    DEFAULT_IMF_PARAMS,
    DEFAULT_MODEL,
    imf_default_bounds,
)
from .likelihood import _resolve_imf_func, imf_mostlikely_params


def imf_samples(num_samples, imf, params=None, logmmin=-3, logmmax=2):
    """Draw ``num_samples`` masses from an IMF via rejection sampling.

    Parameters
    ----------
    num_samples : int
    imf : str or callable
        Model name (e.g. ``"chabrier_smooth"``) or a JAX-callable IMF function
        with signature ``(logm, params, logmmin, logmmax) -> jnp.ndarray``.
    params : array_like, optional
        IMF parameters; defaults to ``DEFAULT_IMF_PARAMS[imf]`` if ``imf`` is a string.
    logmmin, logmmax : float
        log10 mass range to draw from.

    Returns
    -------
    np.ndarray
        Shape ``(num_samples,)`` of masses (linear, not log).
    """
    if isinstance(imf, str):
        imf_fn = _resolve_imf_func(imf)
        if params is None:
            params = DEFAULT_IMF_PARAMS[imf]
    else:
        imf_fn = imf
    params = jnp.asarray(params)

    samples = np.empty(0)
    N = num_samples
    while len(samples) < num_samples:
        x = np.random.rand(N)
        logm_proposals = logmmin + (logmmax - logmmin) * x
        imf_vals = np.asarray(imf_fn(jnp.asarray(logm_proposals), params, logmmin, logmmax))
        ymax = imf_vals.max()
        y = ymax * np.random.rand(N)
        accepted = 10 ** logm_proposals[y < imf_vals]
        samples = np.concatenate([samples, accepted])
        N *= 2
    return samples[:num_samples]


def imf_lnprob_samples(
    masses,
    model=DEFAULT_MODEL,
    p0=None,
    bounds=None,  # accepted for API parity with master; not used here
    num_warmup: int = 500,
    num_samples: int = 1000,
    seed: int = 0,
    logmmin=None,
    logmmax=None,
    target_acceptance: float = 0.8,
):
    """Posterior samples of IMF parameters via NUTS (blackjax).

    Parameters
    ----------
    masses : array_like
        Stellar masses.
    model : str
        IMF model name. Only ``"chabrier_smooth"`` is supported in the MVP.
    p0 : array_like, optional
        NUTS starting position. If ``None``, runs L-BFGS-B first to find the MAP.
    bounds : ignored
        Present only for source-level API parity with the emcee version. NUTS
        runs in unconstrained coordinates with a flat improper prior.
    num_warmup : int
        Number of window-adaptation steps. Step size and the inverse mass
        matrix are tuned during this phase; the samples are discarded.
    num_samples : int
        Number of post-warmup samples to draw (returned).
    seed : int
        PRNG seed for the NUTS chain.
    logmmin, logmmax : float, optional
        log10 mass-range bounds defining the IMF normalization. Default to the
        data min/max.
    target_acceptance : float
        Window-adaptation target acceptance rate.

    Returns
    -------
    samples : np.ndarray, shape (num_samples, n_params)
    """
    imf_fn = _resolve_imf_func(model)

    masses_arr = jnp.asarray(masses)
    logm = jnp.log10(masses_arr.ravel())
    lmin = jnp.min(logm) if logmmin is None else jnp.asarray(logmmin, dtype=jnp.float64)
    lmax = jnp.max(logm) if logmmax is None else jnp.asarray(logmmax, dtype=jnp.float64)

    # Match the master-branch uniform prior on bounds. NUTS needs a smooth
    # log-prior so its leapfrog has gradients; a hard indicator (log p = -inf
    # outside) would also stall the sampler. A quadratic barrier outside the
    # box gives zero contribution inside and a strong restoring force outside,
    # so it behaves like a uniform prior in practice while staying gradient-safe.
    bounds_arr = np.asarray(bounds if bounds is not None else imf_default_bounds(model))
    lo = jnp.asarray(bounds_arr[:, 0], dtype=jnp.float64)
    hi = jnp.asarray(bounds_arr[:, 1], dtype=jnp.float64)

    def lnprob(p):
        imf_val = imf_fn(logm, p, lmin, lmax)
        # Guard against -inf when the proposed IMF is non-positive at any
        # sample mass. Using jnp.log(jnp.clip(...)) keeps the gradient finite,
        # which matters for NUTS leapfrog stability.
        log_imf = jnp.log(jnp.clip(imf_val, 1e-300, None))
        ll = jnp.sum(log_imf)
        # Smooth uniform prior on bounds (quadratic penalty outside box).
        over = jax.nn.relu(p - hi)
        under = jax.nn.relu(lo - p)
        log_prior = -1e6 * jnp.sum(over * over + under * under)
        return ll + log_prior

    if p0 is None:
        sol = imf_mostlikely_params(masses, model, logmmin=lmin, logmmax=lmax)
        p0 = sol.x
    p0_arr = jnp.asarray(p0, dtype=jnp.float64)

    key = jax.random.PRNGKey(seed)
    warmup_key, sample_key = jax.random.split(key)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        lnprob,
        target_acceptance_rate=target_acceptance,
    )
    (state, tuned_params), _ = warmup.run(warmup_key, p0_arr, num_steps=num_warmup)

    nuts = blackjax.nuts(lnprob, **tuned_params)

    def one_step(state, key):
        new_state, _info = nuts.step(key, state)
        return new_state, new_state.position

    keys = jax.random.split(sample_key, num_samples)
    _, positions = jax.lax.scan(one_step, state, keys)

    return np.asarray(positions)
