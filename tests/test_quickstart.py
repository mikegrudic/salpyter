"""Quickstart pytest for the jax-branch salpyter.

Draws masses from the default chabrier_smooth IMF, fits with the NUTS
sampler, and checks the posterior median recovers the input parameters.
Tolerances are sized to the expected posterior width at N=1000 with
~3-5x headroom for sampler/seed noise.
"""

import numpy as np
import pytest

import salpyter
from salpyter.default_imf_params import DEFAULT_IMF_PARAMS


@pytest.mark.parametrize("model", salpyter.IMF_LIST)
def test_quickstart_recovers_input_params(model):
    np.random.seed(42)
    true_params = np.array(DEFAULT_IMF_PARAMS[model], dtype=float)

    # N=5000 is needed because at N=1000 the alpha posterior has a long flat
    # tail (the IMF degenerates to a pure lognormal as alpha → -inf since
    # logmbreak crosses out of the data's mass range), and the median wanders
    # in that tail. At N=5000 the high-mass tail has enough stars to pin
    # alpha down near the truth.
    masses = salpyter.imf_samples(5000, model)
    sol = salpyter.imf_mostlikely_params(masses, model)

    num_samples = 1000
    samples = salpyter.imf_lnprob_samples(
        masses,
        model,
        p0=sol.x,
        num_warmup=500,
        num_samples=num_samples,
        seed=0,
    )

    assert samples.shape == (num_samples, len(true_params)), (
        f"unexpected sample shape {samples.shape}"
    )

    median = np.median(samples, axis=0)
    diff = np.abs(median - true_params)
    # Same tolerances as the master-branch quickstart test for chabrier_smooth.
    tol = np.array([0.05, 0.1, 0.2])
    assert np.all(diff < tol), (
        f"{model}: posterior medians did not recover input params within tolerance.\n"
        f"  true:   {true_params}\n"
        f"  median: {median}\n"
        f"  |diff|: {diff}\n"
        f"  tol:    {tol}"
    )
