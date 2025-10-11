import os
from dataclasses import dataclass, astuple
from typing import Union

import numpy as np

FilePath = Union[str, os.PathLike]
DirectoryPath = Union[str, os.PathLike]


@dataclass(frozen=True)
class ArrayComparisonMixin:
    """Allows equality comparison between dataclasses containing fields with numpy arrays.

    For this mixin to work the dataclass inheriting the mixin must have the
    `eq=False` keyword option, otherwise the mixin equality method will be overwritten
    with the standard dataclass equality method.
    """

    def __eq__(self, other) -> bool:
        """checks if two dataclasses which hold numpy arrays are equal"""
        if self is other:
            return True
        if self.__class__ is not other.__class__:
            return NotImplemented
        t1 = astuple(self)
        t2 = astuple(other)
        return all(self.__array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))

    @staticmethod
    def __array_safe_eq(a, b) -> bool:
        """Check if a and b are equal, even if they are numpy arrays"""
        if a is b:
            return True
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a.shape == b.shape and np.allclose(a, b)
        try:
            return a == b
        except TypeError:
            return NotImplemented
