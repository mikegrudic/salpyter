"""Standalone :class:`~salpyter.model.Cutoff` declarations.

The IMF models themselves are now registered inline in ``imfs.py`` via
``@imf_model`` decorators. This module is left only for ``Cutoff`` instances
(which are not models and so cannot use the same decorator).
"""

from . import imfs
from .model import Cutoff, piecewise, register

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


# ---------------------------------------------------------------------------
# Composed models registered under conventional astronomical names.
# ---------------------------------------------------------------------------

# Salpeter (1955): a single power-law segment. ``salpeter`` is just the
# already-registered ``powerlaw`` model under a more conventional name. The
# IMF math is identical; the default slope of -1.3 (in dN/d(log10 m) units)
# corresponds to dN/dm ∝ m^-2.3, slightly shallower than the canonical
# Salpeter -2.35 but inside the bounds box so the MAP/NUTS can recover it.
salpeter = register(powerlaw, "salpeter")

# Scalo (1986)-style two-segment broken power law: a low-mass and a high-mass
# power-law joined at a free break. Three parameters total:
# ("slope_1", "slope_2", "logmbreak_1"). Use ``.truncate()`` for the bounded
# variant or ``* schechter`` for a smooth high-mass cutoff.
scalo = register(piecewise(powerlaw, powerlaw), "scalo")

# Kroupa (2001) IMF as three free-slope, free-break power-law segments. The
# composition is C0 continuous at each break (cascading scales in piecewise);
# breaks are sampled via the delta reparameterization so they are always
# ordered. Seven parameters: three slopes + two free breaks. The bounded
# variant is just ``kroupa.truncate()``.
kroupa = register(
    piecewise(powerlaw, powerlaw, powerlaw),
    "kroupa",
)
