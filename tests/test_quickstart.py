"""Pytest mirroring the README quickstart: draw masses from a known IMF, fit
them, and check the posterior recovers the input parameters.

Parameterized over salpyter.IMF_LIST. Per-model configuration controls the
sample size N and the per-parameter tolerance. ``np.inf`` tolerance means
"not checked" (param is unidentifiable from the data).

Per-model notes
---------------
- powerlaw: skipped (upstream bug — ``imf_rejection_samples`` calls the IMF
  with the function default ``logmmax=None``, which crashes ``10**None``).
- ``*_bounds``: skipped (``imf_samples`` has no DEFAULT_IMF_PARAMS entry for
  the bounded variants → KeyError).
- chabrier: 4-param posterior has a wider (logm0, logsigma, alpha, logmbreak)
  joint than the smooth 3-param version, since the chabrier defaults have a
  discontinuity at logmbreak=0 (do not satisfy the smooth-break condition).
  Uses N=10000 so each marginal posterior is tight enough that the original
  tolerances catch a broken fit instead of normal posterior breadth.
- chabrier_smooth_lognormal and chabrier_smooth_cutoff_lognormal: the
  high-mass tail can be explained equally well by the chabrier_smooth power
  law (alpha) or the additional lognormal bump component. At quickstart scale
  these are not separately identifiable, so alpha and the bump/cutoff params
  are not checked.
"""

import numpy as np
import pytest
import salpyter
from salpyter.default_imf_params import DEFAULT_IMF_PARAMS


# Per-model config: how many samples to draw, and per-param tolerance.
# np.inf tolerance = unidentifiable param, not checked.
_CONFIG_BY_MODEL = {
    "chabrier_smooth": dict(N=1000, tol=[0.05, 0.1, 0.2]),
    "chabrier": dict(N=10000, tol=[0.05, 0.1, 0.2, 0.3]),
    "chabrier_smooth_lognormal": dict(
        N=1000,
        tol=[0.05, 0.1, np.inf, np.inf, np.inf, np.inf],
    ),
    "chabrier_smooth_cutoff_lognormal": dict(
        N=1000,
        tol=[0.05, 0.1, np.inf, np.inf, np.inf, np.inf, np.inf],
    ),
}


@pytest.mark.parametrize("model", sorted(salpyter.IMF_LIST))
def test_quickstart_recovers_input_params(model):
    if model.endswith("_bounds"):
        pytest.skip(
            "imf_samples cannot draw from _bounds models (no DEFAULT_IMF_PARAMS entry)"
        )
    if model == "powerlaw":
        pytest.skip(
            "imf_rejection_samples crashes for powerlaw: powerlaw_imf default "
            "logmmax=None propagates into 10**None"
        )
    if model not in _CONFIG_BY_MODEL:
        pytest.skip(f"no test config for {model!r}")

    cfg = _CONFIG_BY_MODEL[model]
    tol = np.array(cfg["tol"], dtype=float)

    np.random.seed(42)
    true_params = np.array(DEFAULT_IMF_PARAMS[model], dtype=float)
    assert tol.shape == true_params.shape, (
        f"{model}: tol shape {tol.shape} != params shape {true_params.shape}"
    )

    masses = salpyter.imf_samples(cfg["N"], model)

    p0 = salpyter.imf_mostlikely_params(masses, model).x
    samples = salpyter.imf_lnprob_samples(masses, model, p0, chainlength=3000)

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
