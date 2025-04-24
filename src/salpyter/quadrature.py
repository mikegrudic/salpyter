# @njit(fastmath=True, error_model="numpy")
import numpy as np
from scipy.special import roots_hermite, roots_legendre

MU = 2.0
ROOTS = np.array(
    [
        -0.9815606342467192,
        -0.9041172563704749,
        -0.7699026741943047,
        -0.5873179542866175,
        -0.3678314989981801,
        -0.125233408511469,
        0.125233408511469,
        0.3678314989981801,
        0.5873179542866175,
        0.7699026741943047,
        0.9041172563704749,
        0.9815606342467192,
    ]
)

WEIGHTS = np.array(
    [
        0.0471753363865132,
        0.1069393259953178,
        0.1600783285433461,
        0.2031674267230657,
        0.2334925365383547,
        0.2491470458134026,
        0.2491470458134026,
        0.2334925365383547,
        0.2031674267230657,
        0.1600783285433461,
        0.1069393259953178,
        0.0471753363865132,
    ]
)


def gaussian_quadrature(f, a, b):
    """Gaussian quadrature for Planck function integral from a to b."""

    a, b = min(a, b), max(a, b)
    if a == b:
        return 0.0
    if np.isfinite(a) and np.isfinite(b):
        x = 0.5 * (b - a) * (ROOTS + 1.0) + a
        funcval = f(x)
    elif np.isinf(a) and np.isinf(b):  # integrate over y = x/(1+x^2)
        # a, b = -1, 1
        # y = 0.5 * (b - a) * (ROOTS + 1.0) + a
        # x = y / np.sqrt(1 - y * y)
        # funcval = f(x) * (1 + x * x) ** 1.5
        roots, weights = roots_hermite(10)
        # x = 0.5 * (b - a) * (ROOTS + 1.0) +
        return np.sum(f(roots) * np.exp(roots * roots) * weights)

    integral = np.sum(funcval * WEIGHTS * (b - a) / MU)

    # for i, w in enumerate(WEIGHTS):
    # x = 0.5 * (b - a) * (ROOTS[i] + 1.0) + a
    # xpow = 1.0
    # for _ in range(p):
    # xpow *= x
    # integral += xpow / np.expm1(x) * w
    return integral
