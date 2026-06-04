"""Bayesian evidence (marginal likelihood) for IMF model comparison.

Two estimators are exposed:

* :func:`imf_log_evidence_laplace` — Gaussian approximation at the MAP.
* :func:`imf_log_evidence_bridge` — Meng-Wong bridge sampling from posterior
  samples plus a Gaussian proposal fit to them.

Both target

.. math::

    \\log Z \\;=\\; \\log p(D \\mid M) \\;=\\; \\log \\int p(D \\mid \\theta, M)\\,
    \\pi(\\theta)\\, d\\theta

with :math:`\\pi(\\theta)` taken to be uniform on the model's parameter
``default_bounds`` (or a user-supplied ``bounds``). This matches the implicit
prior of :func:`salpyter.sampling.imf_lnprob_samples`, whose quadratic-barrier
"prior" behaves like a uniform-on-the-box prior inside the box.

Method comparison
-----------------
Laplace is one MAP + one Hessian. Cheap, good when the posterior is roughly
Gaussian, breaks down hard on:

* Models whose MAP sits at the edge of the likelihood support — most notably
  the auto-generated ``*_bounds`` / ``truncate()`` variants, whose
  ``logmmin``/``logmmax`` parameters live against the data extrema.
* Multimodal or strongly skewed posteriors.

Bridge sampling reuses NUTS samples and is much more robust. It still assumes
a single-mode Gaussian proposal overlaps the posterior well; for multimodal
posteriors prefer tempered SMC (not implemented here).

Non-uniform priors
------------------
If you want :math:`\\log Z` under some other prior :math:`\\pi'`, write

.. math::

    \\log Z[\\pi'] = \\log Z[\\text{uniform}] + \\log \\mathbb{E}_{p(\\theta\\mid D,\\pi=\\text{uniform})}
                    \\!\\left[ \\frac{\\pi'(\\theta)}{\\pi_{\\text{uniform}}(\\theta)} \\right]

so the uniform-prior :math:`\\log Z` returned here is the natural building
block — multiply by posterior-mean of the prior ratio (computed from the
NUTS samples) for any reweighted prior.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp

from .default_imf_params import DEFAULT_MODEL
from .likelihood import _resolve_model, imf_mostlikely_params
from .sampling import imf_lnprob_samples


def _resolve_bounds(model, bounds):
    resolved = _resolve_model(model)
    if bounds is None:
        bounds_arr = np.asarray([list(b) for b in resolved.default_bounds], dtype=np.float64)
    else:
        bounds_arr = np.asarray(bounds, dtype=np.float64)
    if bounds_arr.shape != (resolved.ndim, 2):
        raise ValueError(
            f"bounds shape {bounds_arr.shape} doesn't match model ndim {resolved.ndim}"
        )
    return resolved, bounds_arr


def _log_prior_volume(bounds):
    return float(np.sum(np.log(bounds[:, 1] - bounds[:, 0])))


def _make_loglike(resolved, masses, logmmin, logmmax):
    """Return a JIT-compiled scalar log-likelihood log p(D | θ)."""
    logm = jnp.log10(jnp.asarray(masses, dtype=jnp.float64).ravel())
    lmin = jnp.min(logm) if logmmin is None else jnp.asarray(logmmin, dtype=jnp.float64)
    lmax = jnp.max(logm) if logmmax is None else jnp.asarray(logmmax, dtype=jnp.float64)

    @jax.jit
    def loglike(p):
        imf_val = resolved.imf_fn(logm, p, lmin, lmax)
        return jnp.sum(jnp.log(jnp.clip(imf_val, 1e-300, None)))

    return loglike


def imf_log_evidence_laplace(
    masses,
    model=DEFAULT_MODEL,
    bounds=None,
    logmmin=None,
    logmmax=None,
    p0=None,
):
    """log p(D | M) via Gaussian Laplace approximation at the MAP.

    .. math::

        \\log Z \\;\\approx\\; \\log L(\\theta_\\star) \\;-\\; \\log V
        \\;+\\; \\tfrac{d}{2}\\log(2\\pi) \\;-\\; \\tfrac{1}{2}\\log\\det(-H)

    where :math:`\\theta_\\star` is the MAP, :math:`V = \\prod_i (b_i^{hi}-b_i^{lo})`
    is the prior box volume, and :math:`H` is the Hessian of the log-likelihood at
    :math:`\\theta_\\star` (the uniform log-prior is constant inside the box, so it
    contributes nothing to the curvature).

    Caveats
    -------
    * If the MAP sits at a boundary of the likelihood's support (true for the
      auto-generated ``*_bounds`` models, whose ``logmmin``/``logmmax`` parameters
      pin against the data extrema), :math:`H` is not negative-definite and the
      returned ``log_evidence`` is ``nan``. Use :func:`imf_log_evidence_bridge`.

    Parameters
    ----------
    masses : array_like
    model : str or IMFModel
    bounds : array_like of shape (ndim, 2), optional
        Per-parameter (lo, hi) defining the uniform prior. Defaults to the
        model's ``default_bounds``.
    logmmin, logmmax : float, optional
        log10 mass-range bounds passed into the IMF normalization. Default to
        data extrema.
    p0 : array_like, optional
        Starting point for the MAP optimizer.

    Returns
    -------
    dict
        ``log_evidence`` — Laplace estimate of ``log Z``.
        ``log_likelihood`` — log L at the MAP.
        ``log_prior_volume`` — ``log V``.
        ``map_params`` — MAP parameter vector.
        ``hessian`` — full Hessian of the log-likelihood at the MAP.
        ``log_det_neg_hessian`` — ``log det(-H)`` or ``nan`` if non-PSD.
    """
    resolved, bounds_arr = _resolve_bounds(model, bounds)
    logV = _log_prior_volume(bounds_arr)
    d = resolved.ndim

    sol = imf_mostlikely_params(
        masses, model, bounds=bounds_arr.tolist(), p0=p0,
        logmmin=logmmin, logmmax=logmmax,
    )
    theta_star = jnp.asarray(sol.x, dtype=jnp.float64)

    loglike = _make_loglike(resolved, masses, logmmin, logmmax)
    ll_star = float(loglike(theta_star))
    H = np.asarray(jax.hessian(loglike)(theta_star), dtype=np.float64)
    neg_H = -H
    # log det(-H) via slogdet — handles sign cleanly.
    sign, logdet = np.linalg.slogdet(neg_H)
    if sign <= 0 or not np.isfinite(logdet):
        log_evidence = float("nan")
        log_det_neg_H = float("nan")
    else:
        log_det_neg_H = float(logdet)
        log_evidence = ll_star - logV + 0.5 * d * math.log(2 * math.pi) - 0.5 * log_det_neg_H

    return dict(
        log_evidence=log_evidence,
        log_likelihood=ll_star,
        log_prior_volume=logV,
        map_params=np.asarray(sol.x, dtype=np.float64),
        hessian=H,
        log_det_neg_hessian=log_det_neg_H,
    )


def imf_log_evidence_bridge(
    masses,
    model=DEFAULT_MODEL,
    bounds=None,
    logmmin=None,
    logmmax=None,
    posterior_samples=None,
    num_proposal_samples: int = 2000,
    max_iter: int = 500,
    tol: float = 1e-6,
    seed: int = 0,
    num_warmup: int = 500,
    num_samples: int = 2000,
    num_chains: int = 1,
    target_acceptance: float = 0.8,
):
    """log p(D | M) via Meng-Wong optimal-bridge sampling.

    The Meng-Wong identity gives

    .. math::

        Z = \\frac{\\mathbb{E}_q\\!\\left[\\alpha(\\theta)\\, \\tilde p(\\theta)\\right]}
                  {\\mathbb{E}_p\\!\\left[\\alpha(\\theta)\\, q(\\theta)\\right]}

    where :math:`\\tilde p = L\\pi` is the unnormalized posterior, :math:`q` is a
    normalized proposal, :math:`p = \\tilde p / Z` is the posterior, and the
    optimal bridge :math:`\\alpha \\propto (s_1 \\tilde p + s_2 Z q)^{-1}` reduces
    the identity to a fixed-point iteration in :math:`Z`. We iterate in
    log-space for numerical stability.

    Parameters
    ----------
    posterior_samples : array_like, optional
        Shape ``(n_post, ndim)`` of posterior draws. If ``None``, calls
        :func:`salpyter.sampling.imf_lnprob_samples` with the kwargs prefixed
        ``num_warmup``/``num_samples``/``num_chains``/``target_acceptance``/``seed``.
    num_proposal_samples : int
        How many samples to draw from the Gaussian proposal fit to the
        posterior.
    max_iter, tol : int, float
        Fixed-point iteration controls. Convergence is on ``|Δ log r|``.

    Returns
    -------
    dict
        ``log_evidence`` — bridge estimate of ``log Z``.
        ``log_evidence_se`` — first-order standard-error estimate (in nats).
        ``num_iterations`` — fixed-point iterations actually run.
        ``converged`` — bool.
        ``num_proposal_in_box`` — how many proposal draws had finite
        unnormalized posterior (used as a proposal-quality diagnostic).
        ``posterior_samples`` — passed-through or newly drawn samples.
        ``proposal_mean`` / ``proposal_cov`` — fit Gaussian.
    """
    resolved, bounds_arr = _resolve_bounds(model, bounds)
    lo = bounds_arr[:, 0]
    hi = bounds_arr[:, 1]
    logV = _log_prior_volume(bounds_arr)
    d = resolved.ndim

    if posterior_samples is None:
        posterior_samples = imf_lnprob_samples(
            masses,
            model=model,
            bounds=bounds_arr.tolist(),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
            logmmin=logmmin,
            logmmax=logmmax,
            target_acceptance=target_acceptance,
            return_dict=False,
        )
    post = np.asarray(posterior_samples, dtype=np.float64)
    if post.ndim != 2 or post.shape[1] != d:
        raise ValueError(
            f"posterior_samples shape {post.shape} doesn't match model ndim {d}"
        )
    n_post = post.shape[0]

    mu = post.mean(axis=0)
    cov = np.cov(post, rowvar=False)
    if cov.ndim == 0:  # ndim == 1 case
        cov = cov.reshape(1, 1)
    # Tiny ridge for numerical PSD safety on near-degenerate axes (e.g. when a
    # parameter is effectively pinned by the data).
    cov = cov + 1e-12 * np.eye(d)
    chol, _ = cho_factor(cov, lower=True)
    logdet_cov = 2.0 * float(np.sum(np.log(np.diag(chol))))

    rng = np.random.default_rng(seed + 1)
    # X = mu + L z, z ~ N(0, I), L = chol(cov)
    L = np.tril(chol)
    z = rng.standard_normal((num_proposal_samples, d))
    prop = mu[None, :] + z @ L.T

    loglike = _make_loglike(resolved, masses, logmmin, logmmax)
    log_lik_v = jax.jit(jax.vmap(loglike))

    log_lik_post = np.asarray(log_lik_v(jnp.asarray(post)), dtype=np.float64)
    log_lik_prop = np.asarray(log_lik_v(jnp.asarray(prop)), dtype=np.float64)

    # Uniform log-prior: -logV inside the box, -inf outside.
    in_box_post = np.all((post >= lo) & (post <= hi), axis=1)
    in_box_prop = np.all((prop >= lo) & (prop <= hi), axis=1)
    log_unnorm_post_at_post = np.where(in_box_post, log_lik_post - logV, -np.inf)
    log_unnorm_post_at_prop = np.where(in_box_prop, log_lik_prop - logV, -np.inf)

    # Analytic log q evaluated at both sample sets.
    def log_q(x):
        delta = x - mu
        # solve_triangular wants column-major; .T handles it.
        sol_tri = solve_triangular(L, delta.T, lower=True).T
        return -0.5 * (d * math.log(2.0 * math.pi) + logdet_cov + np.sum(sol_tri * sol_tri, axis=1))

    log_q_post = log_q(post)
    log_q_prop = log_q(prop)

    # l1 = log π̃(post) - log q(post),  l2 = log π̃(prop) - log q(prop).
    l1 = log_unnorm_post_at_post - log_q_post
    l2 = log_unnorm_post_at_prop - log_q_prop

    n1 = n_post
    n2 = num_proposal_samples
    log_s1 = math.log(n1 / (n1 + n2))
    log_s2 = math.log(n2 / (n1 + n2))

    # Initial guess: importance-sampling estimate (= mean exp(l2) under the
    # proposal), in log-space.
    log_r = float(logsumexp(l2) - math.log(n2)) if np.isfinite(l2).any() else 0.0
    converged = False
    iters_run = 0
    for it in range(max_iter):
        iters_run = it + 1
        # Numerator: log E_q[ exp(l2) / (s1 exp(l2) + s2 r) ]
        #          = logsumexp(l2 - logaddexp(log_s1 + l2, log_s2 + log_r)) - log n2
        denom_num = np.logaddexp(log_s1 + l2, log_s2 + log_r)
        log_num = float(logsumexp(l2 - denom_num) - math.log(n2))
        # Denominator: log E_p[ 1 / (s1 exp(l1) + s2 r) ]
        #            = logsumexp(- logaddexp(log_s1 + l1, log_s2 + log_r)) - log n1
        denom_den = np.logaddexp(log_s1 + l1, log_s2 + log_r)
        log_den = float(logsumexp(-denom_den) - math.log(n1))
        log_r_new = log_num - log_den
        if not math.isfinite(log_r_new):
            break
        if abs(log_r_new - log_r) < tol:
            log_r = log_r_new
            converged = True
            break
        log_r = log_r_new

    # First-order relative SE (Frühwirth-Schnatter, eq. 4):
    #     V(log r) ≈ Var_q[f_2] / (n2 E_q[f_2]^2) + Var_p[f_1] / (n1 E_p[f_1]^2)
    # with f_2 = exp(l2) / (s1 exp(l2) + s2 r), f_1 = 1 / (s1 exp(l1) + s2 r).
    # Compute on a stable scale (subtract maxima before exponentiating).
    denom_num = np.logaddexp(log_s1 + l2, log_s2 + log_r)
    denom_den = np.logaddexp(log_s1 + l1, log_s2 + log_r)
    log_f2 = l2 - denom_num
    log_f1 = -denom_den
    finite_f2 = np.isfinite(log_f2)
    finite_f1 = np.isfinite(log_f1)
    if finite_f2.sum() < 2 or finite_f1.sum() < 2:
        log_evidence_se = float("nan")
    else:
        m2 = log_f2[finite_f2].max()
        m1 = log_f1[finite_f1].max()
        f2 = np.exp(log_f2[finite_f2] - m2)
        f1 = np.exp(log_f1[finite_f1] - m1)
        rel_var = (f2.var() / (n2 * f2.mean() ** 2)) + (f1.var() / (n1 * f1.mean() ** 2))
        log_evidence_se = float(math.sqrt(max(rel_var, 0.0)))

    return dict(
        log_evidence=float(log_r),
        log_evidence_se=log_evidence_se,
        num_iterations=iters_run,
        converged=converged,
        num_proposal_in_box=int(in_box_prop.sum()),
        posterior_samples=post,
        proposal_mean=mu,
        proposal_cov=cov,
    )


def imf_log_evidence(
    masses,
    model=DEFAULT_MODEL,
    method: str = "laplace",
    **kwargs,
):
    """Dispatcher: ``method="laplace"`` or ``method="bridge"``.

    See :func:`imf_log_evidence_laplace` and :func:`imf_log_evidence_bridge`
    for the per-method kwargs.
    """
    m = method.lower()
    if m == "laplace":
        return imf_log_evidence_laplace(masses, model=model, **kwargs)
    if m == "bridge":
        return imf_log_evidence_bridge(masses, model=model, **kwargs)
    raise ValueError(f"unknown method {method!r}; expected 'laplace' or 'bridge'")
