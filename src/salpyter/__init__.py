"""salpyter — differentiable IMF likelihoods and HMC posterior sampling.

This is the ``jax`` branch implementation. The IMF math is written in JAX so
log-likelihoods are differentiable; posterior sampling uses NUTS via blackjax.

Public surface
--------------
* Legacy string-based API (mirrors master): ``imf_lnprob_samples(...,
  model="chabrier_smooth")``, ``get_imf_function("chabrier_smooth")``,
  ``IMF_LIST``, etc. Unchanged.
* Object-based API (new): pass an :class:`~salpyter.model.IMFModel` directly
  for ``model=``. Compose new models with ``+``, ``*``, ``.truncate()`` or
  :func:`~salpyter.model.piecewise`.
"""

# JAX defaults to float32; the small IMF values produced by the lognormal tail
# underflow noticeably at float32, so enable float64 globally before any other
# JAX import happens.
import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Order matters: model/registry must populate before likelihood/sampling import.
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
    chabrier_smooth_bounds_imf,
    chabrier_smooth_exp_bounds_imf,
    chabrier_smooth_imf,
    lognormal_imf,
    powerlaw_imf,
)
from .model import (  # noqa: E402
    Cutoff,
    IMFModel,
    all_models,
    imf_model,
    piecewise,
    register,
)
# Importing registry has the side effect of registering all base + legacy
# models in the global IMFModel registry.
from . import registry  # noqa: E402, F401
from .registry import (  # noqa: E402
    chabrier as _chabrier_model,
    chabrier_exp_bounds as _chabrier_exp_bounds_model,
    chabrier_smooth as _chabrier_smooth_model,
    chabrier_smooth_bounds as _chabrier_smooth_bounds_model,
    chabrier_smooth_exp_bounds as _chabrier_smooth_exp_bounds_model,
    kroupa,
    powerlaw as _powerlaw_model,
    salpeter,
    scalo,
    schechter,
)
from .likelihood import imf_lnprob, imf_log_slope, imf_mostlikely_params  # noqa: E402
from .sampling import imf_lnprob_samples, imf_samples  # noqa: E402
from .evidence import (  # noqa: E402
    imf_log_evidence,
    imf_log_evidence_bridge,
    imf_log_evidence_laplace,
)


def get_imf_function(model: str):
    """Return the JAX-callable IMF function for ``model``.

    Goes through :func:`salpyter.likelihood._resolve_model`, which knows
    about the auto-``_bounds`` convention (so ``"kroupa_bounds"`` etc. work
    even though they aren't hand-registered).
    """
    from .likelihood import _resolve_model
    return _resolve_model(model).imf_fn


def IMF_LIST():
    """List of all registered model names (snapshot)."""
    from .model import _REGISTRY
    return list(_REGISTRY.keys())


from .model import _REGISTRY as _MODEL_REGISTRY  # noqa: E402
# Compatibility alias: salpyter.IMF_LIST used to be a list, not a function.
IMF_LIST = list(_MODEL_REGISTRY.keys())


__all__ = [
    # Constants
    "CHABRIER_DEFAULT_PARAMS",
    "CHABRIER_SMOOTH_DEFAULT_PARAMS",
    "DEFAULT_IMF_PARAMS",
    "DEFAULT_IMF_PARAMS_BOUNDS",
    "DEFAULT_LOGMMAX",
    "DEFAULT_LOGMMIN",
    "DEFAULT_MODEL",
    "IMF_LIST",
    # Legacy IMF callables (kept for direct use)
    "chabrier_exp_bounds_imf",
    "chabrier_smooth_bounds_imf",
    "chabrier_smooth_exp_bounds_imf",
    "chabrier_smooth_imf",
    "lognormal_imf",
    "powerlaw_imf",
    # New object-based API
    "Cutoff",
    "IMFModel",
    "all_models",
    "imf_model",
    "kroupa",
    "piecewise",
    "register",
    "salpeter",
    "scalo",
    "schechter",
    # Functions
    "get_imf_function",
    "imf_default_bounds",
    "imf_default_params",
    "imf_lnprob",
    "imf_lnprob_samples",
    "imf_log_evidence",
    "imf_log_evidence_bridge",
    "imf_log_evidence_laplace",
    "imf_log_slope",
    "imf_mostlikely_params",
    "imf_samples",
]
