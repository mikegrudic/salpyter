"""Wire the existing JAX IMF functions and the schechter cutoff into the
new :class:`~salpyter.model.IMFModel`/:class:`~salpyter.model.Cutoff` registry.

This is the bridge between the legacy "function + dict" layout in ``imfs.py``
and the new abstraction in ``model.py``. Eventually each ``*_imf`` function
can grow its own ``@imf_model(...)`` decorator and this file can be deleted.
For now, keep the math in one place and the metadata in another.
"""

from . import imfs
from .default_imf_params import (
    DEFAULT_IMF_PARAMS,
    DEFAULT_IMF_PARAMS_BOUNDS,
)
from .model import Cutoff, IMFModel, register


def _make(name: str, imf_fn) -> IMFModel:
    return IMFModel(
        name=name,
        imf_fn=imf_fn,
        param_names=_PARAM_NAMES[name],
        default_params=tuple(DEFAULT_IMF_PARAMS[name]),
        default_bounds=tuple(tuple(b) for b in DEFAULT_IMF_PARAMS_BOUNDS[name]),
    )


# Parameter names for the existing (legacy) models. The JAX functions use
# positional indexing today; these names mirror what the function docstrings
# already document.
_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "chabrier_smooth": ("logm0", "logsigma", "alpha"),
    "chabrier": ("logm0", "logsigma", "alpha", "logmbreak"),
    "chabrier_smooth_bounds": ("logm0", "logsigma", "alpha", "logmmin", "logmmax"),
    "chabrier_smooth_exp_bounds": ("logm0", "logsigma", "alpha", "logmmin", "logmmax"),
    "chabrier_exp_bounds": ("logm0", "logsigma", "alpha", "logmbreak", "logmmin", "logmmax"),
    "powerlaw": ("slope",),
}


# Legacy models — same dispatch as `_MODEL_TO_FUNC` exposes today.
chabrier_smooth = register(_make("chabrier_smooth", imfs.chabrier_smooth_imf))
chabrier = register(_make("chabrier", imfs.chabrier_imf))
chabrier_smooth_bounds = register(_make("chabrier_smooth_bounds", imfs.chabrier_smooth_bounds_imf))
chabrier_smooth_exp_bounds = register(
    _make("chabrier_smooth_exp_bounds", imfs.chabrier_smooth_exp_bounds_imf)
)
chabrier_exp_bounds = register(_make("chabrier_exp_bounds", imfs.chabrier_exp_bounds_imf))


# New base model.
DEFAULT_IMF_PARAMS["powerlaw"] = [-1.3]
DEFAULT_IMF_PARAMS_BOUNDS["powerlaw"] = [[-10.0, 5.0]]
powerlaw = register(_make("powerlaw", imfs.powerlaw_imf))


# Schechter cutoff — the multiplicand for `model * schechter`.
schechter = Cutoff(
    name="schechter",
    cutoff_fn=imfs._schechter_cutoff_fn,
    param_names=("logmmin", "logmmax"),
    default_params=(-3.0, 2.0),
    default_bounds=((-4.0, 4.0), (-4.0, 4.0)),
    support_hint=imfs._schechter_support_hint,
)
