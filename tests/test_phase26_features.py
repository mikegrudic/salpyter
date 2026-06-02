"""Tests for Phase 2.6 / Phase 3 features: ordered piecewise, return_dict."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import salpyter
from salpyter import piecewise


def _kroupa_model_ordered():
    base = piecewise(
        salpyter.powerlaw_imf,
        salpyter.powerlaw_imf,
        salpyter.powerlaw_imf,
        ordered=True,
    )
    return base.truncate(
        default_logmmin=-2.0,
        default_logmmax=2.0,
        bound_logmmin=(-3.0, -1.0),
        bound_logmmax=(1.0, 3.0),
    )


def _kroupa_true_params():
    return np.array([0.7, -0.3, -1.3, np.log10(0.08), np.log10(0.5), -2.0, 2.0], dtype=float)


def test_piecewise_param_names_dedupe_only_when_duplicated():
    """Duplicated component names get segment-suffixed; unique names pass through."""
    # Three identical powerlaws -> three "slope_<idx>" names.
    p3 = piecewise(salpyter.powerlaw_imf, salpyter.powerlaw_imf, salpyter.powerlaw_imf)
    assert p3.param_names[:3] == ("slope_1", "slope_2", "slope_3")

    # Mixed unique-name components -> no suffixing on uniquely-named params.
    mix = piecewise(salpyter.chabrier_smooth_imf, salpyter.powerlaw_imf)
    # chabrier_smooth has ("logm0", "logsigma", "alpha"); powerlaw has ("slope",).
    # None of these names collide, so all stay clean.
    assert mix.param_names[:4] == ("logm0", "logsigma", "alpha", "slope")


def test_ordered_piecewise_log_jacobian_matches_autodiff():
    """The analytic log_jacobian_fn agrees with the autodiff log|det J|."""
    import jax
    import jax.numpy as jnp

    m = piecewise(
        salpyter.powerlaw_imf,
        salpyter.powerlaw_imf,
        salpyter.powerlaw_imf,
        ordered=True,
    )
    assert m.log_jacobian_fn is not None

    rng = np.random.default_rng(0)
    # Try a few random unconstrained vectors.
    for _ in range(5):
        p_unc = jnp.asarray(rng.standard_normal(m.ndim))
        analytic = float(m.log_jacobian_fn(p_unc))
        J = jax.jacobian(m.from_unconstrained)(p_unc)
        autodiff = float(jnp.log(jnp.abs(jnp.linalg.det(J))))
        np.testing.assert_allclose(analytic, autodiff, atol=1e-10)


def test_ordered_piecewise_round_trip():
    """to_unconstrained then from_unconstrained returns the input."""
    m = piecewise(
        salpyter.powerlaw_imf,
        salpyter.powerlaw_imf,
        salpyter.powerlaw_imf,
        ordered=True,
    )
    assert m.has_reparam
    p_user = np.array([0.7, -0.3, -1.3, -1.1, -0.3])
    import jax.numpy as jnp
    p_unc = np.asarray(m.to_unconstrained(jnp.asarray(p_user)))
    p_back = np.asarray(m.from_unconstrained(jnp.asarray(p_unc)))
    np.testing.assert_allclose(p_user, p_back, atol=1e-10)


def test_ordered_piecewise_samples_are_ordered():
    """Every NUTS sample produced under ordered=True has lb_1 < lb_2."""
    np.random.seed(0)
    kroupa = _kroupa_model_ordered()
    true_p = _kroupa_true_params()
    masses = salpyter.imf_samples(
        2000, kroupa.imf_fn, params=true_p, logmmin=-2.0, logmmax=2.0,
    )
    samples = salpyter.imf_lnprob_samples(
        masses, model=kroupa, p0=true_p + 0.05 * np.random.standard_normal(true_p.shape),
        num_warmup=200, num_samples=400, seed=0,
    )
    # logmbreak_1 is index 3, logmbreak_2 is index 4.
    assert np.all(samples[:, 4] > samples[:, 3]), (
        f"found {(samples[:, 4] <= samples[:, 3]).sum()} samples with break ordering violated"
    )


def test_return_dict_keys_and_shapes():
    np.random.seed(0)
    model = "chabrier_smooth"
    masses = salpyter.imf_samples(500, model)
    out = salpyter.imf_lnprob_samples(
        masses, model=model, num_warmup=100, num_samples=200,
        seed=0, return_dict=True,
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {"logm0", "logsigma", "alpha"}
    for v in out.values():
        assert v.shape == (200,)


def test_return_dict_for_piecewise():
    np.random.seed(0)
    kroupa = _kroupa_model_ordered()
    true_p = _kroupa_true_params()
    masses = salpyter.imf_samples(
        2000, kroupa.imf_fn, params=true_p, logmmin=-2.0, logmmax=2.0,
    )
    out = salpyter.imf_lnprob_samples(
        masses, model=kroupa, p0=true_p,
        num_warmup=100, num_samples=200, seed=0, return_dict=True,
    )
    # With the auto-suffix in piecewise, the three "slope" entries become
    # slope_1, slope_2, slope_3 — no dedupe collisions. All 7 params come back
    # under distinct keys.
    expected_keys = {
        "slope_1", "slope_2", "slope_3",
        "logmbreak_1", "logmbreak_2",
        "logmmin", "logmmax",
    }
    assert isinstance(out, dict)
    assert set(out.keys()) == expected_keys
    for v in out.values():
        assert v.shape == (200,)
