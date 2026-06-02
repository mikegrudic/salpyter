"""Quickstart pytest for the jax-branch salpyter.

Parameterized over ``salpyter.IMF_LIST``. Each model: draw masses from its
defaults, fit via NUTS, check the posterior median recovers the input
parameters within reasoned per-parameter tolerances.

Per-model notes
---------------
- ``chabrier_smooth``: N=5000 needed because at N=1000 the alpha posterior
  has a long flat tail (IMF degenerates to pure lognormal as
  ``alpha → -∞`` once logmbreak crosses out of the data's mass range).
- ``chabrier``: N=10000 to tighten the wider 4-param posterior (the default
  parameters do *not* satisfy the smooth-break condition, giving a
  discontinuity at logmbreak that broadens the joint posterior).
- ``chabrier_smooth_bounds``: only the 3 base chabrier_smooth params are
  checked; the sampled cutoffs ``logmmin`` and ``logmmax`` are unidentifiable
  (any value below ``min(data)`` / above ``max(data)`` gives the same
  likelihood once the support contains the data), so they're marked ``inf``.
"""

import numpy as np
import pytest

import salpyter
from salpyter.default_imf_params import DEFAULT_IMF_PARAMS


_CONFIG_BY_MODEL = {
    "chabrier_smooth": dict(N=5000, tol=[0.05, 0.1, 0.2]),
    "chabrier": dict(N=10000, tol=[0.05, 0.1, 0.2, 0.3]),
    "chabrier_smooth_bounds": dict(N=5000, tol=[0.05, 0.1, 0.2, np.inf, np.inf]),
    # exp_bounds: Schechter cutoffs are weakly constrained because of the
    # base shape <-> cutoff degeneracy (the cutoff scale trades off with the
    # natural lognormal/powerlaw extent). Same flagging strategy as the hard-
    # bounds variant: check shape, skip cutoff params.
    "chabrier_smooth_exp_bounds": dict(
        N=5000, tol=[0.05, 0.1, 0.3, np.inf, np.inf],
    ),
    "chabrier_exp_bounds": dict(
        N=10000, tol=[0.05, 0.1, 0.3, 0.4, np.inf, np.inf],
    ),
}


@pytest.mark.parametrize("model", salpyter.IMF_LIST)
def test_quickstart_recovers_input_params(model):
    cfg = _CONFIG_BY_MODEL[model]
    tol = np.array(cfg["tol"], dtype=float)

    np.random.seed(42)
    true_params = np.array(DEFAULT_IMF_PARAMS[model], dtype=float)
    assert tol.shape == true_params.shape, (
        f"{model}: tol shape {tol.shape} != params shape {true_params.shape}"
    )

    masses = salpyter.imf_samples(cfg["N"], model)
    sol = salpyter.imf_mostlikely_params(masses, model)

    num_samples = 1000
    samples = salpyter.imf_lnprob_samples(
        masses, model, p0=sol.x, num_warmup=500, num_samples=num_samples, seed=0,
    )

    assert samples.shape == (num_samples, len(true_params)), (
        f"unexpected sample shape {samples.shape}"
    )

    median = np.median(samples, axis=0)
    diff = np.abs(median - true_params)
    checked = np.isfinite(tol)

    assert np.all(diff[checked] < tol[checked]), (
        f"{model}: posterior medians did not recover input params within tolerance.\n"
        f"  true:   {true_params}\n"
        f"  median: {median}\n"
        f"  |diff|: {diff}\n"
        f"  tol:    {tol}"
    )
