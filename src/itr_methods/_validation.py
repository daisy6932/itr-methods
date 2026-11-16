"""Shared validation for the public treatment-policy estimators."""

from numbers import Integral, Real
import warnings

import numpy as np


def treatment_labels(values, n):
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError("treatment must be a one-dimensional array aligned with X")
    if all(isinstance(value, str) and value for value in raw):
        return np.asarray(raw, dtype=str)
    if all(
        not isinstance(value, (bool, np.bool_))
        and (isinstance(value, Integral)
             or (isinstance(value, Real) and np.isfinite(value) and value == int(value)))
        for value in raw
    ):
        try:
            return np.asarray(raw, dtype=np.int64)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Integer treatment labels must fit in int64") from exc
    if any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError("Use nonempty strings or finite integer treatment labels; boolean labels are unsupported")
    if any(isinstance(value, str) and not value for value in raw):
        raise ValueError("Treatment labels must be nonempty strings or finite integers; found an empty string")
    raise ValueError("Use nonempty strings or finite integer treatment labels")


def feature_names(X):
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = list(columns)
    string_names = [isinstance(name, str) for name in names]
    if not any(string_names):
        return None
    if not all(string_names):
        raise ValueError("Feature names must be all strings or all nonstrings")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique")
    return np.asarray(names, dtype=object)


def check_feature_names(X, expected):
    actual = feature_names(X)
    if expected is not None:
        if actual is None:
            warnings.warn("Features have no names; preserve the training column order", UserWarning, stacklevel=3)
        elif not np.array_equal(actual, expected):
            raise ValueError("Feature names/order differ from training")
