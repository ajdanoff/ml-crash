import math
import pdb
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from sklearn.metrics import log_loss

import numpy as np


class LossE(Enum):
    MSE = 1
    LOG_LOSS = 2
    NN_MSE_LOSS = 3


class Error(ABC):

    @abstractmethod
    def error(self, y, pred):
        raise NotImplementedError("'error' is not implemented !")

    @abstractmethod
    def deriv(self, y, pred):
        """
        y, pred
        """
        raise NotImplementedError("'deriv' is not implemented")


class SError(Error):

    def error(self, y, pred):
        """
        y, pred
        """
        return np.square(pred-y)

    def deriv(self, y, pred):
        """
        y, pred
        """
        return pred - y


class LOGError(SError):

    def error(self, y, pred):
        """
        y, pred
        """
        return y * np.log(pred) + (1 - y) * np.log(1 - pred)


class Loss(ABC):
    _error: Error

    def __init__(self, error):
        self._error = error

    @property
    def error(self):
        return self._error

    @abstractmethod
    def loss(self, **kwargs):
        """
        w, b, x, y, pred
        """
        raise NotImplementedError("'loss' is not implemented !")

    @abstractmethod
    def deriv(self, **kwargs):
        """
        w, b, x, y, pred
        """
        raise NotImplementedError("'deriv' is not implemented !")

    @staticmethod
    def xtr_args(kwargs: dict[str, Any]) -> np.floating[Any] | np.complexfloating[Any, Any] | Any:
        return kwargs.get("w"), kwargs.get("b"), kwargs.get("x"), kwargs.get("y"), kwargs.get("pred_")


class MSELoss(Loss):

    def __init__(self):
        super().__init__(SError())

    def mse(self, w, b, x, y, pred):
        return np.mean(self.error.error(y, pred(w, b, x)))

    def loss(self, **kwargs):
        """
        w, b, x, y, pred
        """
        w, b, x, y, pred = self.xtr_args(kwargs)
        return self.mse(w, b, x, y, pred)

    def deriv(self, **kwargs):
        w, b, x, y, pred = self.xtr_args(kwargs)
        derror = self.error.deriv(y, pred(w, b, x))
        wd = np.dot(derror, 2 * x) / len(x)
        bd = np.mean(derror * 2)
        return wd, bd


class LogLoss(Loss):

    def __init__(self):
        super().__init__(LOGError())

    def deriv(self, **kwargs):
        """
        w, b, x, y, pred
        """
        # pdb.set_trace()
        w, b, x, y, pred = self.xtr_args(kwargs)
        y_pred = pred(w, b, x)
        derror = self.error.deriv(y, y_pred)
        wd = np.dot(x.T, derror) / len(y)
        bd = np.mean(derror)
        return wd, bd

    def loss(self, **kwargs):
        """
        w, b, x, y, pred
        """
        # pdb.set_trace()
        w, b, x, y, pred = self.xtr_args(kwargs)
        y_pred = pred(w, b, x)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(self.error.error(y, y_pred))
        return loss


class NNLoss(Loss, ABC):

    @abstractmethod
    def loss(self, **kwargs):
        """
        y, pred
        """
        raise NotImplementedError("'loss' is not implemented !")

    @abstractmethod
    def deriv(self, **kwargs):
        """
        error_deriv, act_func, x, w, y, learning_rate
        """
        raise NotImplementedError("'deriv' is not implemented")


class NNMSELoss(NNLoss):

    def __init__(self):
        super().__init__(SError())

    @staticmethod
    def xtr_args_nn(kwargs: dict[str, Any]):
        return kwargs.get("error_deriv"), kwargs.get("act_func"), kwargs.get("x"), kwargs.get("w"), kwargs.get("y"), kwargs.get("learning_rate")

    def loss(self, **kwargs):
        """
        y, pred
        """
        # pdb.set_trace()
        y = kwargs.get("y")
        pred = kwargs.get("pred")
        return np.mean(self.error.error(y, pred))

    def deriv(self, **kwargs):
        """
        error_deriv, act_func, x, w, y, learning_rate
        """
        # pdb.set_trace()
        error_deriv, act_func, x, w, y, learning_rate = self.xtr_args_nn(kwargs)
        delta = error_deriv * act_func.deriv_(y)
        dw = -np.dot(x.transpose(), delta) * learning_rate
        db = -np.sum(delta, keepdims=True, axis=0) * learning_rate
        return np.dot(delta, w.transpose()), dw, db
        y = kwargs.get("y")
        pred = kwargs.get("pred")
        return pred - y


def get_loss(key):
    match key:
        case LossE.MSE:
            return MSELoss()
        case LossE.LOG_LOSS:
            return LogLoss()
        case LossE.NN_MSE_LOSS:
            return NNMSELoss()
        case _:
            return MSELoss()
