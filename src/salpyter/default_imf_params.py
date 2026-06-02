"""Default parameters and bounds for IMF models (jax branch).

Bounds are used both for the L-BFGS-B MAP estimator and for the NUTS sampler
(as a smooth quadratic barrier — see ``sampling.imf_lnprob_samples``).
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
# chabrier: chabrier_smooth + free logmbreak (default = log10(1 Msun))
CHABRIER_DEFAULT_PARAMS = CHABRIER_SMOOTH_DEFAULT_PARAMS + [0.0]

DEFAULT_IMF_PARAMS = {
    "chabrier_smooth": CHABRIER_SMOOTH_DEFAULT_PARAMS,
    "chabrier": CHABRIER_DEFAULT_PARAMS,
    # chabrier_smooth_bounds: smooth params + [logmmin, logmmax]. The bounds
    # default to the imf_samples mass range so the rejection sampler draws
    # cleanly within the model's support.
    "chabrier_smooth_bounds": (
        CHABRIER_SMOOTH_DEFAULT_PARAMS + [float(DEFAULT_LOGMMIN), float(DEFAULT_LOGMMAX)]
    ),
    # exp_bounds variants: same parameterization as bounds variants but with
    # smooth Schechter cutoffs instead of hard limits. Defaults set the cutoff
    # masses to the imf_samples sampling range, so the cutoffs are exp(-1)
    # at the edges of where the rejection sampler draws.
    "chabrier_smooth_exp_bounds": (
        CHABRIER_SMOOTH_DEFAULT_PARAMS + [float(DEFAULT_LOGMMIN), float(DEFAULT_LOGMMAX)]
    ),
    "chabrier_exp_bounds": (
        CHABRIER_DEFAULT_PARAMS + [float(DEFAULT_LOGMMIN), float(DEFAULT_LOGMMAX)]
    ),
    "powerlaw": [-1.3],
}

DEFAULT_IMF_PARAMS_BOUNDS = {
    "chabrier_smooth": [[-2.0, 2.0], [-2.0, 2.0], [-10.0, 2.0]],
    # chabrier adds logmbreak bounds.
    "chabrier": [[-2.0, 2.0], [-2.0, 2.0], [-10.0, 2.0], [-3.0, 3.0]],
    # chabrier_smooth_bounds adds prior bounds on the sampled IMF support.
    "chabrier_smooth_bounds": [
        [-2.0, 2.0], [-2.0, 2.0], [-10.0, 2.0], [-4.0, 4.0], [-4.0, 4.0],
    ],
    # exp_bounds variants: same prior boxes as the hard-bounds versions.
    "chabrier_smooth_exp_bounds": [
        [-2.0, 2.0], [-2.0, 2.0], [-10.0, 2.0], [-4.0, 4.0], [-4.0, 4.0],
    ],
    "chabrier_exp_bounds": [
        [-2.0, 2.0], [-2.0, 2.0], [-10.0, 2.0], [-3.0, 3.0], [-4.0, 4.0], [-4.0, 4.0],
    ],
    "powerlaw": [[-10.0, 5.0]],
}


def imf_default_params(model=DEFAULT_MODEL):
    return list(DEFAULT_IMF_PARAMS[model])


def imf_default_bounds(model=DEFAULT_MODEL):
    return [list(b) for b in DEFAULT_IMF_PARAMS_BOUNDS[model]]
