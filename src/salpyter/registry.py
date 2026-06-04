"""Standalone :class:`~salpyter.model.Cutoff` declarations and composed models.

The base IMF shapes (``lognormal``, ``powerlaw``, ``chabrier_smooth``, etc.) are
registered inline in ``imfs.py`` via ``@imf_model`` decorators. This module
constructs the conventional astronomical names (Chabrier, Salpeter, Kroupa,
Scalo) as compositions of those base shapes, plus the ``schechter`` ``Cutoff``
that can't use the model decorator.
"""

import dataclasses

from . import imfs
from .default_imf_params import DEFAULT_IMF_PARAMS, DEFAULT_IMF_PARAMS_BOUNDS
from .model import Cutoff, _REGISTRY, piecewise, register

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

chabrier_smooth = _REGISTRY["chabrier_smooth"]
chabrier_smooth_bounds = _REGISTRY["chabrier_smooth_bounds"]
chabrier_smooth_exp_bounds = _REGISTRY["chabrier_smooth_exp_bounds"]
chabrier_exp_bounds = _REGISTRY["chabrier_exp_bounds"]
powerlaw = _REGISTRY["powerlaw"]
lognormal = _REGISTRY["lognormal"]


# ---------------------------------------------------------------------------
# Composed models registered under conventional astronomical names.
# ---------------------------------------------------------------------------

# Chabrier (2003/2005): lognormal at low mass joined to a single power-law at
# high mass with C0 continuity at a free break. Expressed as
# ``piecewise(lognormal, powerlaw)`` — the cascading-scale rule inside
# ``piecewise`` enforces the same value-matching at ``logmbreak_1`` that the
# old hand-implemented ``chabrier_imf`` did with ``normal_at_break * (m/mbreak)^alpha``.
# Bootstrap, default params, and default bounds are pinned to the values used
# by the previous hand-implemented model so quickstart tests and the
# IMF_analysis_jax.py driver behave identically (param order is also unchanged:
# ``(logm0, logsigma, slope, logmbreak_1)`` aligns positionally with the old
# ``(logm0, logsigma, alpha, logmbreak)``).
chabrier = register(
    dataclasses.replace(
        piecewise(lognormal, powerlaw),
        name="chabrier",
        bootstrap_fn=imfs._bootstrap_chabrier,
        default_params=tuple(DEFAULT_IMF_PARAMS["chabrier"]),
        default_bounds=tuple(tuple(b) for b in DEFAULT_IMF_PARAMS_BOUNDS["chabrier"]),
    ),
    "chabrier",
)

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
