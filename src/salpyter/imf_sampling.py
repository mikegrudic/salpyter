"""Routine for generating samples from a certain IMF"""

from .imfs import *
from .get_imf_function import get_imf_function
from collections.abc import Callable
import numpy as np


def imf_samples(num_samples, imf, params=None, logmmin=-3, logmmax=2):
    """
    Sample from an IMF

    Parameters
    ----------
    num_samples: int
        Number of samples desired
    imf: function or string
        name or function implementing the desired IMF
    params: array_like, optional
        IMF parameters; if left blank will use implemented default parameters
    logmmin: float, optional
        log of low-mass cutoff
    logmmax: float, optional
        log of high-mass cutoff

    Returns
    -------
    sample: array_like
        Shape (num_samples,) array of samples from specified IMF
    """
    return imf_rejection_samples(num_samples, imf, params, logmmin, logmmax)


def imf_rejection_samples(num_samples: int, imf: Callable | str, params=None, logmmin=-3, logmmax=2):
    """
    Uses rejection sampling to sample from an IMF

    Parameters
    ----------
    num_samples: int
        Number of samples desired
    imf: function or string
        name or function implementing the desired IMF
    params: array_like, optional
        IMF parameters; if left blank will use implemented default parameters
    logmmin: float, optional
        log of low-mass cutoff
    logmmax: float, optional
        log of high-mass cutoff

    Returns
    -------
    sample: array_like
        Shape (num_samples,) array of samples from specified IMF
    """

    if isinstance(imf, str):
        imf_function = get_imf_function(imf)
        if params is None:
            params = DEFAULT_IMF_PARAMS[imf]
    else:
        imf_function = imf
        if params is None:
            params = DEFAULT_IMF_PARAMS[imf.__name__]

    samples = np.empty(0)
    N = num_samples
    while len(samples) < num_samples:
        print(N, len(samples))
        x = np.random.rand(N)
        logm = logmmin + (logmmax - logmmin) * x
        imf_val = imf_function(logm, params)
        imf_max = imf_val.max()
        y = imf_max * np.random.rand(N)
        samples = 10 ** logm[y < imf_val]
        N *= 2
    return samples[:num_samples]


# def get_imf_function(imf_name: str):
#     """Returns the IMF function by name"""
#     if not "_imf" in imf_name:
#         imf_name = imf_name + "_imf"
#     return globals()[imf_name]
