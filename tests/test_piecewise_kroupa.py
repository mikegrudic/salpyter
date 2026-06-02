"""Kroupa-from-piecewise recovery test.

Builds the Kroupa (2001) three-segment power-law IMF as a composition of
three :class:`salpyter.IMFModel` ``powerlaw`` segments via
:func:`salpyter.piecewise`, applied with hard mass cutoffs via
:func:`salpyter.IMFModel.truncate`. Samples synthetic data from this model,
runs NUTS, and checks that the posterior medians recover the input
parameters within reasoned tolerances.

This is the smallest end-to-end test that exercises the new abstraction:
  * ``piecewise`` with N=3 segments and free break points
  * ``.truncate()`` to add free low- and high-mass cutoffs
  * Sampling and fitting with a freshly-constructed (un-registered)
    ``IMFModel`` passed directly into ``imf_lnprob_samples``.

Recovery tolerances
-------------------
At N=10000 stars drawn from the model with cutoffs at log10(M) in [-2, 2]:
  * slopes (3 of them) — well-constrained per segment; ~0.05–0.10 std error
  * break points (2 of them) — sharply pinned by the slope discontinuity,
    ~0.03–0.05 std error
  * cutoffs (logmmin/logmmax) — pinned to data extrema, NOT to the input
    cutoff values; check the *upper edge* of the logmmin posterior and
    *lower edge* of logmmax posterior instead, same posterior story as the
    hard-bounds chabrier variant we already tested.
"""

import os

# Force CPU before any JAX import so this test plays nice with multi-test runs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import salpyter
from salpyter import IMFModel, piecewise

# True Kroupa parameters in dN/d(log10 m) units. dN/dm Kroupa slopes are
# (0.3, 1.3, 2.3); converting to dN/dlogm by adding 1 - α gives (0.7, -0.3, -1.3).
KROUPA_SLOPES = (0.7, -0.3, -1.3)
KROUPA_BREAKS = (np.log10(0.08), np.log10(0.5))   # (-1.097, -0.301)
KROUPA_CUTOFFS = (-2.0, 2.0)                       # 0.01 Msun to 100 Msun
N_SAMPLES = 10000


def _kroupa_model() -> IMFModel:
    """Build the Kroupa model = piecewise(3x powerlaw).truncate()."""
    base = piecewise(salpyter._powerlaw_model, salpyter._powerlaw_model, salpyter._powerlaw_model)
    return base.truncate(
        default_logmmin=-2.0,
        default_logmmax=2.0,
        bound_logmmin=(-3.0, -1.0),
        bound_logmmax=(1.0, 3.0),
    )


def _kroupa_true_params() -> np.ndarray:
    return np.array(
        list(KROUPA_SLOPES) + list(KROUPA_BREAKS) + list(KROUPA_CUTOFFS),
        dtype=float,
    )


def test_kroupa_param_names_and_dim():
    """Sanity check that the composed model exposes the expected metadata."""
    m = _kroupa_model()
    expected_names = (
        "slope", "slope", "slope",         # one per segment
        "logmbreak_1", "logmbreak_2",      # from piecewise
        "logmmin", "logmmax",              # from truncate
    )
    assert m.param_names == expected_names
    assert m.ndim == 7
    assert len(m.default_params) == 7
    assert len(m.default_bounds) == 7


def test_kroupa_piecewise_recovers_input_params():
    np.random.seed(0)
    kroupa = _kroupa_model()
    true_params = _kroupa_true_params()

    # Sample synthetic masses from the Kroupa model. The rejection sampler
    # only needs a JAX-callable; pass the composed IMFModel directly via its
    # imf_fn attribute.
    masses = salpyter.imf_samples(
        N_SAMPLES,
        kroupa.imf_fn,
        params=true_params,
        logmmin=KROUPA_CUTOFFS[0],
        logmmax=KROUPA_CUTOFFS[1],
    )

    # MAP from the model's defaults won't put us in the right basin (slopes
    # all default to -1.3, breaks default to even spacing). Hand-bootstrap
    # with the truth + small perturbation: this test is about whether NUTS
    # explores the posterior correctly, not about whether the MAP optimizer
    # finds the global mode of a 7-D piecewise problem from a cold start.
    rng = np.random.default_rng(1)
    p0 = true_params + 0.05 * rng.standard_normal(true_params.shape)

    samples = salpyter.imf_lnprob_samples(
        masses,
        model=kroupa,
        p0=p0,
        num_warmup=800,
        num_samples=2000,
        seed=0,
    )

    median = np.median(samples, axis=0)
    diff = np.abs(median - true_params)

    # Per-parameter tolerances reasoned from the expected posterior width
    # at N=10000:
    #   slopes:   1/sqrt(N_segment) ~ 0.03 each; segments have unequal counts
    #             so the high-mass slope has the worst constraint (~0.05)
    #             ⇒ tol 0.15 (3x headroom)
    #   breaks:   the slope discontinuity makes the break sharp; ~0.05 std
    #             ⇒ tol 0.15
    #   cutoffs:  pinned to data extrema, not to the true cutoff values.
    #             Mark as inf and check the relevant percentile instead.
    tol = np.array([0.15, 0.15, 0.15, 0.15, 0.15, np.inf, np.inf])
    checked = np.isfinite(tol)
    assert np.all(diff[checked] < tol[checked]), (
        f"Kroupa: posterior medians did not recover input params within tolerance.\n"
        f"  param_names: {kroupa.param_names}\n"
        f"  true:        {true_params}\n"
        f"  median:      {median}\n"
        f"  |diff|:      {diff}\n"
        f"  tol:         {tol}"
    )

    # The cutoff posteriors are pinned to the data extrema, not to the
    # nominal input cutoffs. Check that they are sharply pinned just outside
    # the data range — same diagnostic we used for chabrier_smooth_bounds.
    data_logm = np.log10(masses)
    data_min, data_max = float(data_logm.min()), float(data_logm.max())

    logmmin_post = samples[:, 5]
    logmmax_post = samples[:, 6]
    upper_logmmin = np.percentile(logmmin_post, 99)
    lower_logmmax = np.percentile(logmmax_post, 1)

    # 99th percentile of logmmin should be within 0.1 dex below data_min.
    assert upper_logmmin <= data_min + 1e-6, (
        f"logmmin 99% percentile {upper_logmmin:.4f} above data_min {data_min:.4f}"
    )
    assert data_min - upper_logmmin < 0.1, (
        f"logmmin posterior not pinned at the data cut: "
        f"data_min={data_min:.4f}, 99% percentile={upper_logmmin:.4f}"
    )
    assert lower_logmmax >= data_max - 1e-6, (
        f"logmmax 1% percentile {lower_logmmax:.4f} below data_max {data_max:.4f}"
    )
    assert lower_logmmax - data_max < 0.1, (
        f"logmmax posterior not pinned at the data cut: "
        f"data_max={data_max:.4f}, 1% percentile={lower_logmmax:.4f}"
    )
