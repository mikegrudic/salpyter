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
)
from .likelihood import _resolve_imf_func, _resolve_model, imf_mostlikely_params


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
    bounds=None,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
    logmmin=None,
    logmmax=None,
    target_acceptance: float = 0.8,
    perturbation_scale=None,
):
    """Posterior samples of IMF parameters via multi-chain NUTS (blackjax).

    Runs ``num_chains`` independent NUTS chains in parallel via ``jax.vmap``.
    Each chain gets its own randomly perturbed starting point (so different
    chains can land in different modes of a multimodal posterior), runs its
    own window-adaptation warmup (so each chain tunes step size + mass matrix
    to its local geometry), then draws ``num_samples // num_chains`` samples.
    Output is the concatenation, shape ``(num_samples_total, ndim)``.

    Parameters
    ----------
    masses : array_like
        Stellar masses.
    model : str
        IMF model name.
    p0 : array_like, optional
        Base NUTS starting position (length ``ndim``). If ``None``, runs the
        MAP estimator first. Each chain starts at ``p0`` + a per-chain
        Gaussian perturbation.
    bounds : array_like, optional
        Per-parameter ``(lo, hi)``. Used for the smooth quadratic prior
        barrier and to clip the per-chain initial perturbations.
    num_warmup : int
        Window-adaptation steps per chain. Adapted state is discarded.
    num_samples : int
        Total number of post-warmup samples returned (across all chains).
        Each chain runs ``num_samples // num_chains`` steps. If not divisible,
        the actual total is rounded down.
    num_chains : int
        Number of NUTS chains (default 1). Multi-chain is implemented via
        ``jax.vmap`` but **does not parallelize on CPU** — XLA's CPU backend
        serializes the vmapped axis, so 4 chains takes ~4x single-chain wall
        time. Reach for multi-chain only when you suspect the posterior is
        multimodal and you want different chains to land in different modes.
    seed : int
        PRNG seed.
    logmmin, logmmax : float, optional
        log10 mass-range bounds for the IMF normalization. Default to data
        min/max.
    target_acceptance : float
        Window-adaptation target acceptance rate.
    perturbation_scale : array_like or float, optional
        Std-dev of the Gaussian perturbation applied to ``p0`` per chain.
        Defaults to ``0.1 * (hi - lo)`` per parameter — wide enough to seed
        cross-mode exploration of typical bounded-IMF posteriors, narrow
        enough that most chains start in a reasonable-likelihood region.
        Ignored when ``num_chains == 1``.

    Returns
    -------
    samples : np.ndarray, shape (num_samples_total, ndim)
    """
    resolved = _resolve_model(model)
    imf_fn = resolved.imf_fn

    masses_arr = jnp.asarray(masses)
    logm = jnp.log10(masses_arr.ravel())
    lmin = jnp.min(logm) if logmmin is None else jnp.asarray(logmmin, dtype=jnp.float64)
    lmax = jnp.max(logm) if logmmax is None else jnp.asarray(logmmax, dtype=jnp.float64)

    # Match the master-branch uniform prior on bounds. NUTS needs a smooth
    # log-prior so its leapfrog has gradients; a hard indicator (log p = -inf
    # outside) would also stall the sampler. A quadratic barrier outside the
    # box gives zero contribution inside and a strong restoring force outside,
    # so it behaves like a uniform prior in practice while staying gradient-safe.
    if bounds is not None:
        bounds_arr = np.asarray(bounds)
    else:
        bounds_arr = np.asarray([list(b) for b in resolved.default_bounds])
    lo = jnp.asarray(bounds_arr[:, 0], dtype=jnp.float64)
    hi = jnp.asarray(bounds_arr[:, 1], dtype=jnp.float64)

    def lnprob(p):
        imf_val = imf_fn(logm, p, lmin, lmax)
        log_imf = jnp.log(jnp.clip(imf_val, 1e-300, None))
        ll = jnp.sum(log_imf)
        over = jax.nn.relu(p - hi)
        under = jax.nn.relu(lo - p)
        log_prior = -1e6 * jnp.sum(over * over + under * under)
        return ll + log_prior

    if p0 is None:
        sol = imf_mostlikely_params(masses, model, logmmin=lmin, logmmax=lmax)
        p0 = sol.x
    p0_arr = jnp.asarray(p0, dtype=jnp.float64)
    ndim = p0_arr.shape[0]

    num_chains = max(1, int(num_chains))
    samples_per_chain = max(1, num_samples // num_chains)

    # Build per-chain starting points.
    rng = jax.random.PRNGKey(seed)
    init_key, warmup_key, sample_key = jax.random.split(rng, 3)

    if num_chains == 1:
        p0_chains = p0_arr[None, :]
    else:
        if perturbation_scale is None:
            scale = 0.1 * (hi - lo)
        else:
            scale = jnp.broadcast_to(jnp.asarray(perturbation_scale, dtype=jnp.float64), (ndim,))
        noise = jax.random.normal(init_key, (num_chains, ndim))
        p0_chains = p0_arr[None, :] + noise * scale[None, :]
        # Clip into the box minus a small margin so warmup doesn't start
        # already in the barrier region.
        margin = 0.01 * (hi - lo)
        p0_chains = jnp.clip(p0_chains, (lo + margin)[None, :], (hi - margin)[None, :])

    warmup_keys = jax.random.split(warmup_key, num_chains)
    sample_keys = jax.random.split(sample_key, num_chains)

    def run_chain(wkey, skey, init_pos):
        warmup = blackjax.window_adaptation(
            blackjax.nuts, lnprob, target_acceptance_rate=target_acceptance,
        )
        (state, tuned), _ = warmup.run(wkey, init_pos, num_steps=num_warmup)
        nuts = blackjax.nuts(lnprob, **tuned)

        def step(s, k):
            ns, _ = nuts.step(k, s)
            return ns, ns.position

        keys = jax.random.split(skey, samples_per_chain)
        _, positions = jax.lax.scan(step, state, keys)
        return positions  # (samples_per_chain, ndim)

    if num_chains == 1:
        all_samples = run_chain(warmup_keys[0], sample_keys[0], p0_chains[0])
    else:
        # vmap across chains: each chain runs an independent warmup + sampling.
        all_samples = jax.vmap(run_chain)(warmup_keys, sample_keys, p0_chains)
        # Shape (num_chains, samples_per_chain, ndim) -> (total, ndim).
        all_samples = all_samples.reshape(-1, ndim)

    return np.asarray(all_samples)
