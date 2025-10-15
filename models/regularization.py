from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class RegE(Enum):
    L2 = 1
    MOCK = 2


class Regularization(ABC):

    @abstractmethod
    def reg(self, **kwargs):
        raise NotImplementedError("'reg' is not implemented !")

    @abstractmethod
    def deriv(self, **kwargs):
        raise NotImplementedError("'reg' is not implemented !")


class L2Regularization(Regularization):

    def reg(self, **kwargs):
        w = kwargs.get('w', 0)
        return np.dot(w, w)

    def deriv(self, **kwargs):
        w = kwargs.get('w', 0)
        return 2 * w


class MockRegularization(Regularization):

    def reg(self, **kwargs):
        return 0

    def deriv(self, **kwargs):
        return 0


def get_reg(key):
    match key:
        case RegE.L2:
            return L2Regularization()
        case _:
            return MockRegularization()
