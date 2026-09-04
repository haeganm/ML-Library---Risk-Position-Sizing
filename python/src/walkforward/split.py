"""Purged walk-forward splits.

Training data always precedes test data. Purging additionally drops the
training observations whose *labels* reach into the test window, which is the
leak that ordinary time-series splitting misses: a model predicting the next
21-day return has labels 21 periods wide, so the twenty training rows before
the test window were partly scored on test-window prices.

The rule is ``purge = label_horizon - 1``, and this module asks for the label
horizon rather than the purge count so it cannot be got wrong. With
one-period labels nothing is purged and the split reduces to plain
walk-forward.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np

from ._core import Split, as_count, check, lib

try:  # sklearn is optional: the splitter is duck-typed for it either way
    from sklearn.model_selection import BaseCrossValidator as _BaseCrossValidator
except ImportError:  # pragma: no cover - exercised where sklearn is absent

    class _BaseCrossValidator:  # type: ignore[no-redef]
        pass


__all__ = ["PurgedWalkForward", "WalkForwardSplit", "walk_forward_splits"]


class WalkForwardSplit(NamedTuple):
    """One split. All ranges are half-open, ``[start, end)``."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_post_start: int
    train_post_end: int

    @property
    def train(self) -> slice:
        """Slice of the pre-test training window, ready for ``X[split.train]``."""
        return slice(self.train_start, self.train_end)

    @property
    def test(self) -> slice:
        """Slice of the test window."""
        return slice(self.test_start, self.test_end)

    @property
    def train_post(self) -> slice:
        """Slice of the post-test training window. Empty unless asked for."""
        return slice(self.train_post_start, self.train_post_end)

    @property
    def purge(self) -> int:
        """Observations dropped between the training and test windows."""
        return self.test_start - self.train_end


def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    label_horizon: int = 1,
    purge: int | None = None,
    embargo: int = 0,
    include_post_train: bool = False,
) -> list[WalkForwardSplit]:
    """Generate purged walk-forward splits over ``n`` observations.

    Parameters
    ----------
    n
        Number of observations.
    train_size
        Length of each training window.
    test_size
        Length of each test window, which follows the training window.
    step
        Distance between consecutive splits. Defaults to ``test_size``, giving
        test windows that tile the sample without overlapping.
    label_horizon
        Number of periods each label spans. A label at ``i`` built from
        ``[i, i + label_horizon)`` overlaps the test window for the last
        ``label_horizon - 1`` training observations, and those are purged.
        Leave at 1 for labels that span a single period.
    purge
        Overrides ``label_horizon`` when you want to set the purge directly.
    embargo
        Observations skipped after the test window before post-test training
        data. Only has an effect together with ``include_post_train``: a pure
        walk-forward has no training data after the test window to embargo.
    include_post_train
        Expose a post-test training segment running to the end of the sample.
        That segment is future data relative to the test window *and* contains
        the test windows of every later split, so using it for anything the
        evaluation depends on invalidates this split and all the ones after
        it. It exists for purged cross-validation over a fixed sample, not for
        walk-forward.

    Returns
    -------
    A list of :class:`WalkForwardSplit`, empty when the sample is too short
    for even one split.
    """
    total = as_count(n, "n")
    train = as_count(train_size, "train_size", minimum=1)
    test = as_count(test_size, "test_size", minimum=1)
    stride = as_count(test if step is None else step, "step", minimum=1)
    horizon = as_count(label_horizon, "label_horizon", minimum=1)
    drop = horizon - 1 if purge is None else as_count(purge, "purge")
    gap = as_count(embargo, "embargo")

    if drop >= train:
        raise ValueError(
            f"purge ({drop}) must be smaller than train_size ({train}); a label "
            f"horizon of {horizon} purges {drop} observations and leaves nothing "
            "to train on"
        )

    count = ctypes.c_size_t()
    args = (total, train, test, stride, drop, gap, int(bool(include_post_train)))
    check(
        lib.mlr_walk_forward_splits(*args, None, 0, ctypes.byref(count)),
        "walk_forward_splits",
    )
    if count.value == 0:
        return []
    buffer = (Split * count.value)()
    check(
        lib.mlr_walk_forward_splits(*args, buffer, count.value, ctypes.byref(count)),
        "walk_forward_splits",
    )
    return [
        WalkForwardSplit(
            s.train_start,
            s.train_end,
            s.test_start,
            s.test_end,
            s.train_post_start,
            s.train_post_end,
        )
        for s in buffer[: count.value]
    ]


class PurgedWalkForward(_BaseCrossValidator):
    """Walk-forward cross-validator with purging, for scikit-learn.

    Drops into anything that takes a ``cv`` object::

        from sklearn.model_selection import cross_val_score
        from walkforward import PurgedWalkForward

        cv = PurgedWalkForward(train_size=756, test_size=252, label_horizon=21)
        scores = cross_val_score(model, X, y, cv=cv)

    Every training window ends at least ``label_horizon - 1`` observations
    before its test window begins, so no training label was computed from a
    period inside the test window. Rows are assumed to be in time order.

    Parameters
    ----------
    train_size
        Length of each training window.
    test_size
        Length of each test window.
    step
        Distance between consecutive splits. Defaults to ``test_size``, so the
        test windows tile the sample without overlapping and every observation
        outside the first training window is tested exactly once.
    label_horizon
        Periods each label spans, which sets the purge to
        ``label_horizon - 1``. For a target built as an ``h``-period forward
        return, pass ``h``.
    purge
        Set the purge directly, overriding ``label_horizon``.

    Notes
    -----
    There is no ``embargo`` parameter. An embargo protects training data that
    sits *after* a test window, and a walk-forward never trains on anything
    after the window it is testing. Use :func:`walk_forward_splits` with
    ``include_post_train`` if you want that variant, and read its warning
    first.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step: int | None = None,
        label_horizon: int = 1,
        purge: int | None = None,
    ) -> None:
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
        self.label_horizon = label_horizon
        self.purge = purge

    def _splits(self, n: int) -> list[WalkForwardSplit]:
        return walk_forward_splits(
            n,
            train_size=self.train_size,
            test_size=self.test_size,
            step=self.step,
            label_horizon=self.label_horizon,
            purge=self.purge,
        )

    @staticmethod
    def _n_samples(X: Any, y: Any = None) -> int:
        source = X if X is not None else y
        if source is None:
            raise ValueError("PurgedWalkForward needs X or y to know the sample length")
        try:
            return len(source)
        except TypeError:
            return int(source.shape[0])

    def split(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_index, test_index)`` pairs of positional indices."""
        del groups  # accepted for the scikit-learn interface, unused
        n = self._n_samples(X, y)
        for split in self._splits(n):
            yield (
                np.arange(split.train_start, split.train_end),
                np.arange(split.test_start, split.test_end),
            )

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of splits this configuration produces for the given sample."""
        del groups
        return len(self._splits(self._n_samples(X, y)))

    def __repr__(self) -> str:
        parts = [f"train_size={self.train_size}", f"test_size={self.test_size}"]
        if self.step is not None:
            parts.append(f"step={self.step}")
        if self.purge is not None:
            parts.append(f"purge={self.purge}")
        elif self.label_horizon != 1:
            parts.append(f"label_horizon={self.label_horizon}")
        return f"PurgedWalkForward({', '.join(parts)})"
