"""``IMFModel`` and ``Cutoff`` — the user-facing abstraction for declaring IMF
models and composing them via algebra (mixture, cutoff, truncation, piecewise).

Concepts
--------
* :class:`IMFModel` wraps a pure-JAX ``imf_fn(logm, params, logmmin, logmmax)``
  plus metadata (parameter names, defaults, bounds, optional bootstrap).
  An ``IMFModel`` is itself callable, so it slots into any place that expects
  the bare JAX IMF function.

* :class:`Cutoff` wraps a multiplier-in-``[0, 1]`` function plus its own
  metadata. Used as the right operand of ``IMFModel.__mul__`` to bolt a
  smooth cutoff onto an existing model.

* Operators on ``IMFModel``:
    - ``a + b``         — mixture of two models with one extra logit-weight parameter.
    - ``a * cutoff``    — apply a smooth cutoff and re-normalize via trapz.
    - ``a.truncate()``  — apply a hard low/high mass cutoff (two new params)
                          and re-normalize via trapz.

* :func:`piecewise` — join N >= 2 models on consecutive mass segments with
  automatic C0 continuity at each break. Breaks may be fixed or free
  parameters.

Every operator returns a new ``IMFModel`` whose ``imf_fn`` is still pure JAX,
so ``jax.jit``, ``jax.grad`` and ``jax.vmap`` flow through compositions of
any depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

# Internal grid resolution for trapz renormalization in derived models.
_DERIVED_NGRID = 501
_DERIVED_MARGIN = 5.0


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, "IMFModel"] = {}


def register(model: "IMFModel", name: Optional[str] = None) -> "IMFModel":
    """Add a model to the global registry under ``name`` (or ``model.name``).

    When ``name`` is provided and differs from ``model.name``, the model is
    rebuilt with the new name so the registry key and ``model.name`` stay in
    sync (the IMFModel dataclass is frozen).
    """
    if name is not None and name != model.name:
        import dataclasses
        model = dataclasses.replace(model, name=name)
    key = model.name
    if key in _REGISTRY:
        raise ValueError(f"model {key!r} is already registered")
    _REGISTRY[key] = model
    return model


def all_models() -> dict[str, "IMFModel"]:
    """Return the registry as a name -> IMFModel dict."""
    return dict(_REGISTRY)


# --------------------------------------------------------------------------- #
# IMFModel                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IMFModel:
    """A named IMF model: pure JAX function + parameter metadata.

    Parameters
    ----------
    from_unconstrained : callable, optional
        ``(p_unconstrained) -> p_user`` that maps the parameter vector NUTS
        samples in (the "unconstrained" space) to the user-facing
        representation that ``param_names`` describes. Default identity.
    to_unconstrained : callable, optional
        Inverse of ``from_unconstrained`` — used to convert a user-supplied
        ``p0`` into the sampler's coordinate system. Default identity.

    These hooks let composite models (notably :func:`piecewise` with
    ``ordered=True``) sample in a reparameterized space (e.g. deltas
    instead of absolute breaks) while keeping ``param_names`` and the
    posterior samples in their natural user-facing scale.
    """

    name: str
    imf_fn: Callable[..., jnp.ndarray]
    param_names: tuple[str, ...]
    default_params: tuple[float, ...]
    default_bounds: tuple[tuple[float, float], ...]
    bootstrap_fn: Optional[Callable[..., list[float]]] = field(default=None, repr=False)
    from_unconstrained: Optional[Callable[..., jnp.ndarray]] = field(default=None, repr=False)
    to_unconstrained: Optional[Callable[..., jnp.ndarray]] = field(default=None, repr=False)
    # log |det J| of from_unconstrained at p_unc, evaluated in JAX.
    # Required for unbiased NUTS sampling when from_unconstrained is non-identity:
    # target density in unconstrained space is p(f(phi)) * |det df/dphi|.
    log_jacobian_fn: Optional[Callable[..., jnp.ndarray]] = field(default=None, repr=False)

    def __post_init__(self):
        n = len(self.param_names)
        if len(self.default_params) != n or len(self.default_bounds) != n:
            raise ValueError(
                f"{self.name!r}: param_names ({n}), default_params "
                f"({len(self.default_params)}), default_bounds "
                f"({len(self.default_bounds)}) must have the same length"
            )

    @property
    def ndim(self) -> int:
        return len(self.param_names)

    @property
    def has_reparam(self) -> bool:
        return self.from_unconstrained is not None and self.to_unconstrained is not None

    def __call__(self, logm, params, logmmin=-jnp.inf, logmmax=4.0):
        return self.imf_fn(logm, params, logmmin, logmmax)

    def bootstrap(self, masses, logmmin=None, logmmax=None) -> list[float]:
        """Return a starting parameter vector for the MAP optimizer."""
        if self.bootstrap_fn is not None:
            return self.bootstrap_fn(masses, logmmin, logmmax)
        return list(self.default_params)

    # --------------------------------------------------------------------- #
    # Operator algebra                                                      #
    # --------------------------------------------------------------------- #

    def __add__(self, other: "IMFModel") -> "IMFModel":
        """Mixture of two models with one extra logit-weight parameter."""
        if not isinstance(other, IMFModel):
            return NotImplemented
        nA, nB = self.ndim, other.ndim

        def imf_fn(logm, p, logmmin=-jnp.inf, logmmax=4.0):
            pA = jax.lax.dynamic_slice(p, (0,), (nA,))
            pB = jax.lax.dynamic_slice(p, (nA,), (nB,))
            w = jax.nn.sigmoid(p[nA + nB])
            return w * self.imf_fn(logm, pA, logmmin, logmmax) + (1 - w) * other.imf_fn(logm, pB, logmmin, logmmax)

        return IMFModel(
            name=f"({self.name}+{other.name})",
            imf_fn=imf_fn,
            param_names=self.param_names + other.param_names + ("logit_w",),
            default_params=self.default_params + other.default_params + (0.0,),
            default_bounds=self.default_bounds + other.default_bounds + ((-5.0, 5.0),),
        )

    def __mul__(self, cutoff: "Cutoff") -> "IMFModel":
        """Apply a smooth cutoff and re-normalize via trapz."""
        if not isinstance(cutoff, Cutoff):
            return NotImplemented
        nA, nB = self.ndim, cutoff.ndim

        def imf_fn(logm, p, logmmin=-jnp.inf, logmmax=4.0):
            pA = jax.lax.dynamic_slice(p, (0,), (nA,))
            pC = jax.lax.dynamic_slice(p, (nA,), (nB,))
            if cutoff.support_hint is not None:
                lo, hi = cutoff.support_hint(pC)
            else:
                lo, hi = logmmin - _DERIVED_MARGIN, logmmax + _DERIVED_MARGIN
            grid = jnp.linspace(lo, hi, _DERIVED_NGRID)
            base_at_grid = self.imf_fn(grid, pA, logmmin, logmmax)
            cutoff_at_grid = cutoff.cutoff_fn(grid, pC)
            norm = jnp.trapezoid(base_at_grid * cutoff_at_grid, grid)
            return (
                self.imf_fn(logm, pA, logmmin, logmmax)
                * cutoff.cutoff_fn(logm, pC)
                / norm
            )

        return IMFModel(
            name=f"{self.name}*{cutoff.name}",
            imf_fn=imf_fn,
            param_names=self.param_names + cutoff.param_names,
            default_params=self.default_params + cutoff.default_params,
            default_bounds=self.default_bounds + cutoff.default_bounds,
        )

    def truncate(
        self,
        default_logmmin: float = -3.0,
        default_logmmax: float = 2.0,
        bound_logmmin: tuple[float, float] = (-4.0, 4.0),
        bound_logmmax: tuple[float, float] = (-4.0, 4.0),
    ) -> "IMFModel":
        """Apply hard ``[logmmin, logmmax]`` cutoffs as two new free parameters.

        Inside the cutoffs the IMF is the base model evaluated with those
        cutoffs as its normalization range; outside it is set to zero. The
        result is re-normalized over ``[logmmin_p, logmmax_p]`` via trapz so
        small analytic-norm drifts from the base model don't propagate.
        """
        n_base = self.ndim

        def imf_fn(logm, p, logmmin=-jnp.inf, logmmax=4.0):
            base_p = jax.lax.dynamic_slice(p, (0,), (n_base,))
            logmmin_p = p[n_base]
            logmmax_p = p[n_base + 1]

            base_at_data = self.imf_fn(logm, base_p, logmmin_p, logmmax_p)
            inside = (logm >= logmmin_p) & (logm <= logmmax_p)
            masked = jnp.where(inside, base_at_data, 0.0)

            grid = jnp.linspace(logmmin_p, logmmax_p, _DERIVED_NGRID)
            base_at_grid = self.imf_fn(grid, base_p, logmmin_p, logmmax_p)
            norm = jnp.trapezoid(base_at_grid, grid)

            return masked / norm

        return IMFModel(
            name=f"{self.name}.truncate()",
            imf_fn=imf_fn,
            param_names=self.param_names + ("logmmin", "logmmax"),
            default_params=self.default_params + (default_logmmin, default_logmmax),
            default_bounds=self.default_bounds + (bound_logmmin, bound_logmmax),
        )


# --------------------------------------------------------------------------- #
# Cutoff                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cutoff:
    """A smooth multiplier-in-``[0, 1]`` to bolt onto an :class:`IMFModel`."""

    name: str
    cutoff_fn: Callable[..., jnp.ndarray]   # (logm, params) -> multiplier
    param_names: tuple[str, ...]
    default_params: tuple[float, ...]
    default_bounds: tuple[tuple[float, float], ...]
    # Optional hint: (logm_lo, logm_hi) span of the trapz integration grid.
    support_hint: Optional[Callable[..., tuple[float, float]]] = field(default=None, repr=False)

    @property
    def ndim(self) -> int:
        return len(self.param_names)

    def __call__(self, logm, params):
        return self.cutoff_fn(logm, params)


# --------------------------------------------------------------------------- #
# Piecewise                                                                   #
# --------------------------------------------------------------------------- #


def piecewise(
    *models: IMFModel,
    breaks: Optional[Sequence[Optional[float]]] = None,
    ordered: bool = True,
) -> IMFModel:
    """Build an N-segment piecewise IMF with C0 continuity enforced at breaks.

    Parameters
    ----------
    *models : IMFModel
        N >= 2 component models, listed low-mass to high-mass.
    breaks : sequence of length N-1, optional
        Each entry is either a float (fixed break point in log10 mass) or
        ``None`` (becomes a free parameter named ``"logmbreak_<i>"``).
        Default ``None`` means all breaks are free.
    ordered : bool, default True
        When True, free breaks are internally reparameterized as deltas
        (the i-th free break is ``lb_{i-1} + exp(raw_i)`` for i >= 1, so
        breaks are strictly increasing by construction). NUTS samples the
        deltas; the returned model still exposes the natural
        ``logmbreak_i`` names and posterior samples are transformed back
        into that space transparently. When False, free breaks are sampled
        directly — ordering must be enforced manually via the prior
        bounds, and label-swap degeneracy is possible.

    Notes
    -----
    Continuity is enforced by cascading multiplicative scales::

        s_0 = 1
        s_i = s_{i-1} * model_{i-1}(lb_{i-1}) / model_i(lb_{i-1})

    The resulting piecewise function is then trapz-normalized over
    ``[logmmin, logmmax]``.
    """
    if len(models) < 2:
        raise ValueError("piecewise needs at least 2 models")
    N = len(models)
    n_breaks = N - 1
    if breaks is None:
        breaks = [None] * n_breaks
    breaks = list(breaks)
    if len(breaks) != n_breaks:
        raise ValueError(f"piecewise of {N} models needs {n_breaks} breaks; got {len(breaks)}")

    # Indices into the free-break tail of the parameter vector.
    free_break_positions = [i for i, b in enumerate(breaks) if b is None]
    n_free_breaks = len(free_break_positions)

    model_sizes = tuple(m.ndim for m in models)
    n_model_params = sum(model_sizes)

    def _resolve_breaks(p):
        """Return a (N-1,) array of break values, mixing fixed and free."""
        free_idx = 0
        vals = []
        for i in range(n_breaks):
            if breaks[i] is None:
                vals.append(p[n_model_params + free_idx])
                free_idx += 1
            else:
                vals.append(jnp.asarray(breaks[i], dtype=p.dtype))
        return jnp.stack(vals)

    def _slice_model_params(p):
        offsets = []
        offset = 0
        for s in model_sizes:
            offsets.append((offset, s))
            offset += s
        return [jax.lax.dynamic_slice(p, (lo,), (s,)) for lo, s in offsets]

    def imf_fn(logm, p, logmmin=-jnp.inf, logmmax=4.0):
        model_params = _slice_model_params(p)
        breaks_arr = _resolve_breaks(p)

        # Cascading C0 scales.
        scales = [jnp.array(1.0, dtype=p.dtype)]
        for i in range(1, N):
            lb = breaks_arr[i - 1][None]
            lo_at = models[i - 1].imf_fn(lb, model_params[i - 1], logmmin, logmmax)[0]
            hi_at = models[i    ].imf_fn(lb, model_params[i    ], logmmin, logmmax)[0]
            scales.append(scales[-1] * lo_at / hi_at)
        scales_arr = jnp.stack(scales)

        def piecewise_at(lm_arr):
            # Stack all N model evaluations: (N, len(lm))
            stack = jnp.stack(
                [
                    models[i].imf_fn(lm_arr, model_params[i], logmmin, logmmax)
                    for i in range(N)
                ]
            )
            seg = jnp.searchsorted(breaks_arr, lm_arr, side="right")  # (len(lm),)
            idx = jnp.arange(lm_arr.shape[0])
            return scales_arr[seg] * stack[seg, idx]

        grid = jnp.linspace(logmmin, logmmax, _DERIVED_NGRID)
        norm = jnp.trapezoid(piecewise_at(grid), grid)
        return piecewise_at(jnp.asarray(logm)) / norm

    # Parameter metadata. Free breaks are always exposed as logmbreak_i in
    # param_names regardless of `ordered` — the reparameterization (when
    # ordered=True) happens transparently via from/to_unconstrained.
    param_names: list[str] = []
    default_params: list[float] = []
    default_bounds: list[tuple[float, float]] = []
    for m in models:
        param_names.extend(m.param_names)
        default_params.extend(m.default_params)
        default_bounds.extend(m.default_bounds)
    for i, b in enumerate(breaks):
        if b is None:
            param_names.append(f"logmbreak_{i + 1}")
            # Default break placement: spread evenly between -2 and 2 in log10 mass.
            default_params.append(float(-2 + 4 * (i + 1) / N))
            default_bounds.append((-4.0, 4.0))

    # Delta reparameterization for ordered free breaks.
    #
    # The IMF function above always operates on the *user-facing* parameter
    # vector (ordered logmbreak_i values). When `ordered=True`, NUTS samples in
    # a different ("unconstrained") space where the trailing free-break entries
    # are interpreted as (logmbreak_1, log_delta_1, log_delta_2, ...), and a
    # cumulative-exp converts them back to (logmbreak_1, logmbreak_2, ...)
    # before the prior box and the imf_fn see them. By construction the deltas
    # are positive (exp), so breaks are strictly increasing — no label-swap
    # degeneracy, no need to constrain ordering via the prior bounds.
    from_unc = None
    to_unc = None
    log_jac_fn = None
    if ordered and n_free_breaks >= 2:
        anchor_idx = n_model_params  # index of logmbreak_1 (absolute)

        def from_unconstrained(p_unc):
            """(.., lb_1, log_d_1, log_d_2, ..) -> (.., lb_1, lb_2, lb_3, ..)."""
            p_unc = jnp.asarray(p_unc)
            head = p_unc[:anchor_idx]
            anchor = p_unc[anchor_idx]
            raw_deltas = p_unc[anchor_idx + 1 : anchor_idx + n_free_breaks]
            deltas = jnp.exp(raw_deltas)
            breaks_user = anchor + jnp.concatenate([jnp.zeros(1), jnp.cumsum(deltas)])
            return jnp.concatenate([head, breaks_user])

        def to_unconstrained(p_user):
            """(.., lb_1, lb_2, lb_3, ..) -> (.., lb_1, log_d_1, log_d_2, ..)."""
            p_user = jnp.asarray(p_user)
            head = p_user[:anchor_idx]
            anchor = p_user[anchor_idx]
            breaks_tail = p_user[anchor_idx + 1 : anchor_idx + n_free_breaks]
            diffs = breaks_tail - jnp.concatenate([anchor[None], breaks_tail[:-1]])
            raw_deltas = jnp.log(jnp.maximum(diffs, 1e-300))
            return jnp.concatenate([head, anchor[None], raw_deltas])

        def log_jacobian(p_unc):
            """log |det df/dp_unc| for the delta reparameterization.

            The Jacobian of (lb_1, log_d_1, log_d_2, ...) -> (lb_1, lb_2,
            lb_3, ...) is lower triangular with diagonal entries
            (1, exp(log_d_1), exp(log_d_2), ...) — so the determinant is
            exp(sum(log_d_i)) and the log determinant is sum(log_d_i).
            All other coordinates (the head and anchor) are identity, so
            they contribute 0 to the log determinant.
            """
            p_unc = jnp.asarray(p_unc)
            raw_deltas = p_unc[anchor_idx + 1 : anchor_idx + n_free_breaks]
            return jnp.sum(raw_deltas)

        from_unc = from_unconstrained
        to_unc = to_unconstrained
        log_jac_fn = log_jacobian

    return IMFModel(
        name=f"piecewise({','.join(m.name for m in models)})",
        imf_fn=imf_fn,
        param_names=tuple(param_names),
        default_params=tuple(default_params),
        default_bounds=tuple(default_bounds),
        from_unconstrained=from_unc,
        to_unconstrained=to_unc,
        log_jacobian_fn=log_jac_fn,
    )


# --------------------------------------------------------------------------- #
# Decorator                                                                   #
# --------------------------------------------------------------------------- #


def imf_model(
    *,
    name: str,
    param_names: Sequence[str],
    default_params: Sequence[float],
    default_bounds: Sequence[tuple[float, float]],
    bootstrap_fn: Optional[Callable[..., list[float]]] = None,
) -> Callable[[Callable], IMFModel]:
    """Decorator that wraps a JAX IMF function into a registered :class:`IMFModel`.

    The decorated name is bound to the IMFModel (not the original function);
    calls like ``chabrier_smooth_imf(logm, params)`` still work because
    ``IMFModel.__call__`` forwards to ``imf_fn``. The original function's
    docstring is preserved on the returned model so ``help(...)`` and
    Sphinx pick it up.
    """

    def wrap(fn: Callable) -> IMFModel:
        model = IMFModel(
            name=name,
            imf_fn=fn,
            param_names=tuple(param_names),
            default_params=tuple(default_params),
            default_bounds=tuple(default_bounds),
            bootstrap_fn=bootstrap_fn,
        )
        register(model)
        # Preserve the original function's docstring on the returned IMFModel
        # (frozen dataclass → use object.__setattr__).
        if fn.__doc__:
            object.__setattr__(model, "__doc__", fn.__doc__)
        return model

    return wrap
