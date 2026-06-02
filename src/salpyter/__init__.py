"""salpyter — differentiable IMF likelihoods and HMC posterior sampling.

This is the ``jax`` branch implementation. The IMF math is written in JAX so
log-likelihoods are differentiable; posterior sampling uses NUTS via blackjax,
which exploits gradients to sample much more efficiently than the emcee
ensemble sampler on the master branch.

MVP scope: ``chabrier_smooth`` only. Other models from the emcee branch
(``chabrier``, ``chabrier_smooth_bounds``, the lognormal extensions) can be
added by writing their IMF in JAX and registering them in :data:`IMF_LIST`
and :func:`get_imf_function`.
"""

# JAX defaults to float32; the small IMF values produced by the lognormal tail
# underflow noticeably at float32, so enable float64 globally before any other
# JAX import happens.
import jax as _jax

_jax.config.update("jax_enable_x64", True)

from .default_imf_params import (  # noqa: E402
    CHABRIER_DEFAULT_PARAMS,
    CHABRIER_SMOOTH_DEFAULT_PARAMS,
    DEFAULT_IMF_PARAMS,
    DEFAULT_IMF_PARAMS_BOUNDS,
    DEFAULT_LOGMMAX,
    DEFAULT_LOGMMIN,
    DEFAULT_MODEL,
    imf_default_bounds,
    imf_default_params,
)
from .imfs import (  # noqa: E402
    chabrier_exp_bounds_imf,
    chabrier_imf,
    chabrier_smooth_bounds_imf,
    chabrier_smooth_exp_bounds_imf,
    chabrier_smooth_imf,
)
from .likelihood import _MODEL_TO_FUNC, imf_lnprob, imf_mostlikely_params  # noqa: E402
from .sampling import imf_lnprob_samples, imf_samples  # noqa: E402

IMF_LIST = list(_MODEL_TO_FUNC.keys())


def get_imf_function(model: str):
    """Return the JAX-callable IMF function for ``model``."""
    fn = _MODEL_TO_FUNC.get(model.lower())
    if fn is None:
        raise NotImplementedError(
            f"jax salpyter supports {sorted(_MODEL_TO_FUNC)}; got {model!r}"
        )
    return fn


__all__ = [
    "CHABRIER_DEFAULT_PARAMS",
    "CHABRIER_SMOOTH_DEFAULT_PARAMS",
    "DEFAULT_IMF_PARAMS",
    "DEFAULT_IMF_PARAMS_BOUNDS",
    "DEFAULT_LOGMMAX",
    "DEFAULT_LOGMMIN",
    "DEFAULT_MODEL",
    "IMF_LIST",
    "chabrier_exp_bounds_imf",
    "chabrier_imf",
    "chabrier_smooth_bounds_imf",
    "chabrier_smooth_exp_bounds_imf",
    "chabrier_smooth_imf",
    "get_imf_function",
    "imf_default_bounds",
    "imf_default_params",
    "imf_lnprob",
    "imf_lnprob_samples",
    "imf_mostlikely_params",
    "imf_samples",
]
