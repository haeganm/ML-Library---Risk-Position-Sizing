"""ctypes binding to the mlrisk C library.

Everything public in this package goes through here. Two rules hold for
every function in it:

* Output buffers are always allocated on this side and never taken from the
  caller, so the ``restrict`` non-aliasing contract on every C output
  parameter cannot be violated by a numpy view.
* Inputs are converted to C-contiguous float64 before the pointer is taken,
  so a strided view, a Python list, a masked array or a pandas Series is
  never read as if it were a packed array of doubles.
"""

from __future__ import annotations

import ctypes
import pathlib
from typing import Any

import numpy as np

try:  # pandas is optional; it is only used to hand a Series back a Series
    import pandas as _pd
except ImportError:  # pragma: no cover - exercised by environments without pandas
    _pd = None

__all__ = [
    "DomainError",
    "Garch",
    "LinModel",
    "Split",
    "c_version",
    "check",
    "lib",
]

# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class DomainError(ValueError):
    """The arguments were well formed but the computation has no answer.

    Raised for a singular design matrix, a sample with zero or unrepresentable
    variance, and recursions that overflow. Mirrors ``MLR_EDOMAIN``.
    """


_OK, _EINVAL, _ENOMEM, _EBOUNDS, _EDOMAIN = range(5)

_MESSAGES = {
    _EINVAL: "invalid argument",
    _ENOMEM: "allocation failed",
    _EBOUNDS: "output capacity too small",
    _EDOMAIN: "domain error",
}


def check(status: int, what: str, detail: str = "", domain: str = "") -> None:
    """Turn an ``mlr_status`` into the matching Python exception.

    ``detail`` describes the preconditions an invalid argument violated;
    ``domain`` describes why a well-formed computation had no answer. They
    are different sentences, and a domain error must not repeat the
    preconditions, which by then the caller has met.
    """
    if status == _OK:
        return
    message = f"{what}: {_MESSAGES.get(status, f'unknown status {status}')}"
    if status == _EDOMAIN:
        if domain:
            message = f"{message} ({domain})"
        raise DomainError(message)
    if detail:
        message = f"{message} ({detail})"
    if status == _ENOMEM:
        raise MemoryError(message)
    if status == _EINVAL:
        raise ValueError(message)
    raise RuntimeError(message)  # EBOUNDS never escapes: we size every buffer


# --------------------------------------------------------------------------
# Structures, mirroring include/mlrisk/*.h field for field
# --------------------------------------------------------------------------


class Garch(ctypes.Structure):
    _fields_ = [
        ("omega", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("sigma2_next", ctypes.c_double),
        ("loglik", ctypes.c_double),
        ("converged", ctypes.c_int),
        ("backcast", ctypes.c_double),
    ]


class Split(ctypes.Structure):
    _fields_ = [
        ("train_start", ctypes.c_size_t),
        ("train_end", ctypes.c_size_t),
        ("test_start", ctypes.c_size_t),
        ("test_end", ctypes.c_size_t),
        ("train_post_start", ctypes.c_size_t),
        ("train_post_end", ctypes.c_size_t),
    ]


class LinModel(ctypes.Structure):
    _fields_ = [
        ("d", ctypes.c_size_t),
        ("w", ctypes.POINTER(ctypes.c_double)),
        ("b", ctypes.c_double),
        ("ridge", ctypes.c_double),
        ("fitted", ctypes.c_int),
    ]


_D = ctypes.POINTER(ctypes.c_double)
_SIZE = ctypes.c_size_t

_SIGNATURES: dict[str, tuple[list[Any], Any]] = {
    # rolling.h
    "mlr_rolling_mean": ([_D, _SIZE, _SIZE, _D], ctypes.c_int),
    "mlr_rolling_std": ([_D, _SIZE, _SIZE, _D], ctypes.c_int),
    "mlr_ewma_vol": ([_D, _SIZE, ctypes.c_double, _D], ctypes.c_int),
    # vol.h
    "mlr_garch_fit": ([_D, _SIZE, ctypes.POINTER(Garch)], ctypes.c_int),
    "mlr_garch_filter": ([ctypes.POINTER(Garch), _D, _SIZE, _D], ctypes.c_int),
    "mlr_garch_filter_from": (
        [ctypes.POINTER(Garch), ctypes.c_double, _D, _SIZE, _D],
        ctypes.c_int,
    ),
    "mlr_garch_forecast": ([ctypes.POINTER(Garch), _SIZE, _D], ctypes.c_int),
    "mlr_parkinson_vol": ([_D, _D, _SIZE, _D], ctypes.c_int),
    "mlr_garman_klass_vol": ([_D, _D, _D, _D, _SIZE, _D], ctypes.c_int),
    # sizing.h
    "mlr_vol_target_position": (
        [_D, ctypes.c_double, ctypes.c_double, _D, ctypes.c_double, _SIZE, _D],
        ctypes.c_int,
    ),
    "mlr_kelly_fraction": ([_D, _SIZE, ctypes.c_double, _D], ctypes.c_int),
    "mlr_drawdown_scale": ([_D, _SIZE, ctypes.c_double, _D], ctypes.c_int),
    # split.h
    "mlr_walk_forward_splits": (
        [_SIZE] * 6 + [ctypes.c_int, ctypes.POINTER(Split), _SIZE, ctypes.POINTER(_SIZE)],
        ctypes.c_int,
    ),
    # linreg.h
    "mlr_lin_model_init": ([ctypes.POINTER(LinModel), _SIZE], ctypes.c_int),
    "mlr_lin_model_free": ([ctypes.POINTER(LinModel)], None),
    "mlr_linreg_fit": (
        [_D, _D, _SIZE, _SIZE, ctypes.c_double, ctypes.POINTER(LinModel)],
        ctypes.c_int,
    ),
    "mlr_linreg_predict": (
        [_D, _SIZE, _SIZE, ctypes.POINTER(LinModel), _D],
        ctypes.c_int,
    ),
    # version.h
    "mlr_version": ([], ctypes.c_char_p),
    "mlr_version_number": ([], ctypes.c_int),
}


def _load() -> ctypes.CDLL:
    """Load the shared library that ships inside this package."""
    here = pathlib.Path(__file__).resolve().parent
    matches = sorted(
        path
        for pattern in ("walkforward_native.*", "libwalkforward_native.*")
        for path in here.glob(pattern)
        if path.suffix in {".so", ".dylib", ".dll", ".pyd"}
    )
    if not matches:
        raise ImportError(
            f"the walkforward native library is missing from {here}. "
            "Reinstall the package, or build it from a checkout with "
            "`pip install .` in the repository root."
        )
    library = ctypes.CDLL(str(matches[0]))
    for name, (argtypes, restype) in _SIGNATURES.items():
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = restype
    return library


lib = _load()


def c_version() -> str:
    """Version reported by the loaded shared library itself."""
    return lib.mlr_version().decode()


# --------------------------------------------------------------------------
# Array plumbing
# --------------------------------------------------------------------------


def as_input(values: Any, name: str, *, ndim: int = 1) -> np.ndarray:
    """Return ``values`` as a C-contiguous float64 array of the given rank.

    Masked entries become NaN, which every function here treats as missing.
    The rank and the kind of the data are checked before the conversion,
    because ``np.ascontiguousarray`` would turn a scalar into a one-element
    series and a datetime into nanoseconds without a word.
    """
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    if hasattr(values, "toarray") and not isinstance(values, np.ndarray):
        raise TypeError(
            f"{name} must be a dense array; sparse input is not supported, call .toarray() first"
        )
    raw = np.asarray(values)
    if raw.ndim != ndim:
        what = "a scalar" if raw.ndim == 0 else f"{raw.ndim} dimensions"
        raise ValueError(f"{name} must be {ndim}-dimensional, got {what}")
    if raw.dtype.kind not in "biufO":
        raise TypeError(
            f"{name} must be numeric, got dtype {raw.dtype}; convert it to float first"
        )
    # Convert from the original object, not from `raw`: a pandas nullable
    # column turns its NA into NaN only when asked for float64 directly.
    try:
        array = np.ascontiguousarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be numeric; an element could not be read as a float ({error})"
        ) from None
    if ndim == 2 and array.shape[1] == 0:
        raise ValueError(f"{name} has no features (shape {array.shape})")
    if array.size == 0:
        raise ValueError(f"{name} is empty; every estimator needs at least one observation")
    return array


def ptr(array: np.ndarray) -> Any:
    """Pointer to the first element. The array must already be contiguous."""
    return array.ctypes.data_as(_D)


def out_like(n: int) -> np.ndarray:
    """A fresh writable output buffer, never shared with any input."""
    return np.empty(n, dtype=np.float64)


def same_length(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"{name_a} and {name_b} must be the same length, "
            f"got {a.shape[0]} and {b.shape[0]}"
        )


def same_index(name_a: str, a: Any, name_b: str, b: Any) -> None:
    """Refuse two pandas Series whose indexes differ.

    Paired inputs are read positionally. Two Series of the same length on
    different indexes would be paired by position and the result stamped with
    the first index, which is exactly how a shifted series ends up sized
    against the wrong bar. Use ``lag`` to realign instead of slicing.
    """
    if _pd is None or not (isinstance(a, _pd.Series) and isinstance(b, _pd.Series)):
        return
    if not a.index.equals(b.index):
        raise ValueError(
            f"{name_a} and {name_b} are pandas Series on different indexes and "
            "would be paired by position. Align them first; lag() keeps the index."
        )


def like(values: np.ndarray, template: Any) -> Any:
    """Give a pandas input its index back.

    Realigning a bare array by hand is how lookahead creeps into a pandas
    workflow, so a Series in means a Series out, on the same index.
    """
    if _pd is not None and isinstance(template, _pd.Series):
        return _pd.Series(values, index=template.index, name=template.name)
    return values


def as_count(value: Any, name: str, *, minimum: int = 0) -> int:
    """Validate a count before it becomes a ``size_t``.

    Python integers are unbounded and negative values wrap when converted, so
    the range check has to happen here rather than in C.
    """
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if count != value:
        raise TypeError(f"{name} must be a whole number, got {value!r}")
    if count < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {count}")
    if count > 2**63 - 1:
        raise ValueError(f"{name} is too large: {count}")
    return count
