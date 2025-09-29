import math
import pdb
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd
import pytest

"""
pounds  miles per gallon
3.5 	18
3.69 	15
3.44 	18
3.43 	16
4.34 	15
4.42 	14
2.37 	24
"""

class LossE(Enum):
    MSE = 1
    LOG_LOSS = 2

class Loss(ABC):

    @abstractmethod
    def loss(self, w, b, x, y, pred):
        raise NotImplementedError("'loss' is not implemented !")

    @abstractmethod
    def deriv(self, w, b, x, y, pred):
        raise NotImplementedError("'deriv' is not implemented !")


class MSELoss(Loss):

    @classmethod
    def mse(cls, w, b, x, y, pred):
        return np.mean(np.square(np.array(y) - pred(w, b, np.array(x))))

    def loss(self, w, b, x, y, pred):
        return self.mse(w, b, x, y, pred)

    def deriv(self, w, b, x, y, pred):
        diff = pred(w, b, np.array(x)) - np.array(y)
        wd = np.dot(diff, 2 * np.array(x)) / len(x)
        bd = np.mean(diff * 2)
        return wd, bd


class LogLoss(Loss):

    @classmethod
    def sigmoid(cls, w, b, x, pred):
        return 1 / (1 + math.e ** (-pred(w, b, np.array(x))))

    def deriv(self, w, b, x, y, pred):
        return self.sigmoid(w, b, x, pred) - y

    def loss(self, w, b, x, y, pred):
        return np.dot(-y, np.log(pred)) - np.dot(1 - y, np.log(1 - pred(w, b, np.array(x))))


def get_loss(key):
    match key:
        case LossE.MSE:
            return MSELoss()
        case _:
            return MSELoss()


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


class LinearModel:
    _w: np.ndarray | float
    _b: float
    _learning_rate: float
    _epochs: int
    _num_features: int
    _error: float
    _max_num_iterations: int
    _reg_obj: Regularization
    _loss_obj: Loss

    def __init__(self, epochs: int, num_features: int, learning_rate: float = 0.001, error: float = 1e-5,
                 max_num_iterations: int = 1000, loss_key=MSELoss, reg_key=RegE.L2):
        if num_features > 1:
            self._w = np.zeros(num_features)
            # self._b = np.zeros(num_features)
        else:
            self._w = 0.0
        self._b = 0.0
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._error = error
        self._max_num_iterations = max_num_iterations
        self._reg_obj = get_reg(reg_key)
        self._loss_obj = get_loss(loss_key)

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def epochs(self):
        return self._epochs

    @property
    def error(self):
        return self._error

    @property
    def max_num_iterations(self):
        return self._max_num_iterations

    @property
    def reg_obj(self):
        return self._reg_obj

    @reg_obj.setter
    def reg_obj(self, reg):
        self._reg_obj = reg

    @property
    def loss_obj(self):
        return self._loss_obj

    @loss_obj.setter
    def loss_obj(self, loss_obj):
        self._loss_obj = loss_obj

    @classmethod
    def _pred(cls, w, b, x):
        # x = np.array(x).transpose()
        # pdb.set_trace()
        return np.dot(w, x.transpose()) + b

    def pred(self, x):
        # x = np.array(x).transpose()
        return np.dot(self.w, x.transpose()) + self.b

    def _loss(self, w, b, x, y):
        return self.loss_obj.loss(w, b, x, y, self._pred)

    def loss(self, x, y):
        return self._loss(self.w, self.b, x, y)

    def _objective(self, w, b, x, y, l=0.0):
        r = (l * self.reg_obj.reg(w=w)) if l and self.reg_obj else 0
        return self._loss(w, b, x, y) + r

    def deriv(self, w, b, x, y, l=0.0):
        """
        """
        # pdb.set_trace()
        rd = l * self.reg_obj.deriv(w=w) if l and self.reg_obj else 0
        dwd, dbd = self.loss_obj.deriv(w, b, x, y, self._pred)
        wd = dwd + rd
        bd = dbd
        return wd, bd

    def grad_desc(self, x, y, l=0.0):
        """
        Can use a regularization and early stopping
        """
        # pdb.set_trace()
        w = self._w
        b = self._b
        mse = self._objective(w, b, x, y, l)
        dmse = 1000 * self.error
        it = 0
        while abs(dmse) > self._error and it < self.max_num_iterations:
            # print("gradient descent: it:  %s, MSE:     %s" %(it, mse))
            # print(self.w, self.b)
            wd, bd = self.deriv(w, b, x, y, l)
            w -= self.learning_rate * wd
            b -= self.learning_rate * bd
            dmse = mse
            mse = self._objective(w, b, x, y, l)
            dmse -= mse
            it += 1
        return w, b

    def train(self, x, y, l=0.0):
        # pdb.set_trace()
        for e in range(1, self.epochs + 1):
            w, b = self.grad_desc(x, y, l)
            self._w, self._b = w, b
            print("epochs:  %s, LOSS:     %s" % (e, self.loss(x, y)))
        return self.w, self.b


class MiniBatchModel(LinearModel):
    _batch: int
    _epochs: int

    def __init__(self, batch: int, epochs: int, num_features: int, learning_rate: float = 0.001, error: float = 1e-5,
                 max_num_iterations: int = 1000, loss_key=MSELoss, reg_key=RegE.L2):
        LinearModel.__init__(self, epochs, num_features, learning_rate, error, max_num_iterations, loss_key, reg_key)
        self._batch = batch
        self._epochs = epochs

    @property
    def batch(self):
        return self._batch

    @property
    def epochs(self):
        return self._epochs

    def mini_batch_grad_desc(self, x, y, l=0.0):
        # pdb.set_trace()
        dataset_len = len(x)
        num_iterations = math.ceil(len(x) / self.batch)
        w, b = self.w, self.b
        for i in range(num_iterations):
            if (i + 1) * self.batch < dataset_len:
                x_batch = x[i * self.batch:(i + 1) * self.batch]
                y_batch = y[i * self.batch:(i + 1) * self.batch]
            else:
                x_batch = x[i * self.batch:]
                y_batch = y[i * self.batch:]
            # print(x_batch, y_batch)
            w, b = self.grad_desc(x_batch, y_batch, l)
            self._w, self._b = w, b
        return self.w, self.b

    def train(self, x, y, l=0.0):
        # pdb.set_trace()
        for e in range(1, self.epochs + 1):
            w, b = self.mini_batch_grad_desc(x, y, l)
            print("epochs:  %s, LOSS:     %s" % (e, self.loss(x, y)))
        return self.w, self.b


@pytest.fixture
def auto_dataset():
    auto_dataset = pd.read_csv("./data/auto-mpg.xls")
    auto_dataset['weight'] = auto_dataset['weight'] / 100
    yield auto_dataset


def test_linear_model(auto_dataset):
    lm = LinearModel(5, 2, learning_rate=0.00001)
    mbm = MiniBatchModel(10, 5, 2, learning_rate=0.00001)
    print(lm.train(auto_dataset[['weight', 'acceleration']], auto_dataset['mpg'], 0.0001))
    print(mbm.train(auto_dataset[['weight', 'acceleration']], auto_dataset['mpg'], 0.0001))
    lm_pred_acc = 1.0 - abs(lm.pred(auto_dataset[['weight', 'acceleration']].iloc[0]) - auto_dataset['mpg'].iloc[0]) / \
                  auto_dataset['mpg'].iloc[0]
    mbm_pred_acc = 1.0 - abs(mbm.pred(auto_dataset[['weight', 'acceleration']].iloc[0]) - auto_dataset['mpg'].iloc[0]) / \
                   auto_dataset['mpg'].iloc[0]
    print("linear model prediction accuracy: %s" % lm_pred_acc)
    print("minibatch model prediction accuracy: %s" % mbm_pred_acc)
    # print(mbm.pred(auto_dataset[['weight', 'acceleration']].iloc[0]))
    # print(mbm.pred(auto_dataset[['weight', 'acceleration']].iloc[10]))
