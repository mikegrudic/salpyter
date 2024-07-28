"""Routine for generating samples from a certain IMF"""

from .imfs import *


def imf_samples(num_samples, imf, params=None, logmmin=-3, logmmax=2):
    """Uses rejection sampling to sample from an IMF"""

    if isinstance(imf, str):
        imf_function = get_imf_function(imf)
        if params is None:
            params = DEFAULT_IMF_PARAMS[imf]
    else:
        imf_function = imf
        if params is None:
            params = DEFAULT_IMF_PARAMS[imf.__name__]

    x = np.random.rand(num_samples)
    logm = logmmin + (logmmax - logmmin) * x
    imf_val = imf_function(logm, params)
    imf_max = imf_val.max()
    y = imf_max * np.random.rand(num_samples)
    return 10 ** logm[y < imf_val]


def get_imf_function(imf_name: str):
    """Returns the IMF function by name"""
    if not "_imf" in imf_name:
        imf_name = imf_name + "_imf"
    return globals()[imf_name]
