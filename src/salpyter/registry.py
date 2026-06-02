"""Standalone :class:`~salpyter.model.Cutoff` declarations.

The IMF models themselves are now registered inline in ``imfs.py`` via
``@imf_model`` decorators. This module is left only for ``Cutoff`` instances
(which are not models and so cannot use the same decorator).
"""

from . import imfs
from .model import Cutoff

# Schechter cutoff for ``model * schechter`` composition.
schechter = Cutoff(
    name="schechter",
    cutoff_fn=imfs._schechter_cutoff_fn,
    param_names=("logmmin", "logmmax"),
    default_params=(-3.0, 2.0),
    default_bounds=((-4.0, 4.0), (-4.0, 4.0)),
    support_hint=imfs._schechter_support_hint,
)


# Backward-compat aliases — some downstream code (and the previous version of
# ``__init__.py``) imported the IMFModel instances from registry.py under their
# bare names. Re-export them from the registry to preserve those imports.
from .model import _REGISTRY  # noqa: E402

chabrier_smooth = _REGISTRY["chabrier_smooth"]
chabrier = _REGISTRY["chabrier"]
chabrier_smooth_bounds = _REGISTRY["chabrier_smooth_bounds"]
chabrier_smooth_exp_bounds = _REGISTRY["chabrier_smooth_exp_bounds"]
chabrier_exp_bounds = _REGISTRY["chabrier_exp_bounds"]
powerlaw = _REGISTRY["powerlaw"]
