"""Default parameters and bounds for IMF models (jax branch, MVP scope).

Only ``chabrier_smooth`` is implemented. Bounds are only used for the L-BFGS-B
MAP estimator; the NUTS sampler operates in unconstrained coordinates.
"""

import numpy as np

DEFAULT_MODEL = "chabrier_smooth"
DEFAULT_LOGMMIN = -3
DEFAULT_LOGMMAX = 2

# chabrier_smooth: [log10(m_peak), log(sigma), high-mass slope]
CHABRIER_SMOOTH_DEFAULT_PARAMS = [
    float(np.log10(0.25)),
    float(np.log(0.55)),
    -1.3,
]

DEFAULT_IMF_PARAMS = {
    "chabrier_smooth": CHABRIER_SMOOTH_DEFAULT_PARAMS,
}

DEFAULT_IMF_PARAMS_BOUNDS = {
    "chabrier_smooth": [[-2.0, 2.0], [-2.0, 2.0], [-10.0, 2.0]],
}


def imf_default_params(model=DEFAULT_MODEL):
    return list(DEFAULT_IMF_PARAMS[model])


def imf_default_bounds(model=DEFAULT_MODEL):
    return [list(b) for b in DEFAULT_IMF_PARAMS_BOUNDS[model]]
