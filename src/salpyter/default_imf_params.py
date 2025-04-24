"""Definitions of default parameters for various IMFs"""

import numpy as np

DEFAULT_MODEL = "chabrier"
DEFAULT_LOGMMIN = -2
DEFAULT_LOGMMAX = 4

# Chabrier family of IMFs
CHABRIER_DEFAULT_PARAMS = [np.log10(0.25), np.log(0.55), -1.3, 0.0]  # log mc, log sigma, high-mass slope, log m_break
CHABRIER_SMOOTH_DEFAULT_PARAMS = [np.log10(0.25), np.log(0.55), -1.3]  # log mc, log sigma, high-mass slope

DEFAULT_IMF_PARAMS = {
    "powerlaw": [-1.3],
    "powerlaw_mmax": [-1.3, 4],
    "chabrier": CHABRIER_DEFAULT_PARAMS,
    "chabrier_smooth": CHABRIER_SMOOTH_DEFAULT_PARAMS,
    "chabrier_smooth_lognormal": [np.log10(0.25), np.log(0.55), -1.0, -1.0, 3, 0.0],
    "chabrier_smooth_cutoff_lognormal": [np.log10(0.25), np.log(0.55), -1.0, 2, -1.0, 3, 0.0],
}


DEFAULT_IMF_PARAMS_BOUNDS = {
    # log mc, log sigma, high-mass slope
    "chabrier_smooth": [[-2, 2], [-2, 2], [-10, 2]],
    "powerlaw": [[-10, 2]],
    "powerlaw_mmax": [[-10, 2], [-2, 4]],
}

p0 = DEFAULT_IMF_PARAMS_BOUNDS["chabrier_smooth"]
# + log m_break
DEFAULT_IMF_PARAMS_BOUNDS["chabrier"] = p0 + [[-3, 3]]
# + fraction in lognormal part, log mc of lognormal part, log sigma of lognormal part
DEFAULT_IMF_PARAMS_BOUNDS["chabrier_smooth_lognormal"] = p0 + [[-10, 6], [-1, 4], [-2, 1]]
# + high-mass cutoff of lognormal part, fraction in lognormal part, log mc of lognormal part, log sigma of lognormal part
DEFAULT_IMF_PARAMS_BOUNDS["chabrier_smooth_cutoff_lognormal"] = p0 + [[0, 4], [-10, 6], [-1, 4], [-2, 1]]


def imf_default_params(model=DEFAULT_MODEL):
    """Convenience method to access default IMF parameters"""

    if "_bounds" in model:  # 'imf + upper and lower bounds' model
        params = DEFAULT_IMF_PARAMS[model.replace("_bounds", "")] + [DEFAULT_LOGMMIN, DEFAULT_LOGMMAX]
        #
    else:
        params = DEFAULT_IMF_PARAMS[model]
    return params


def imf_default_bounds(model=DEFAULT_MODEL):
    """Convenience method to access default IMF parameter bounds"""

    if "_bounds" in model:  # 'imf + upper and lower bounds' model
        bounds = DEFAULT_IMF_PARAMS_BOUNDS[model.replace("_bounds", "")] + [[-4, 4], [-4, 4]]
        # print(bounds)
    else:
        bounds = DEFAULT_IMF_PARAMS_BOUNDS[model]
    return bounds
