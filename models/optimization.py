import math
import pdb
from abc import ABC, abstractmethod
from enum import Enum
from typing import AnyStr, Any

from models.loss import Loss, LossE, get_loss
from models.nodes import Node, RealNode, NeuralNetwork
from models.regularization import Regularization, RegE, get_reg


class OptimizersE(Enum):
    GDSC = 1
    MBGDSC = 2
    NNGDSC = 3


class Optimizer(ABC):

    _learning_rate: float
    _error: float
    _max_num_iterations: int
    _reg_obj: Regularization
    _loss_obj: Loss
    _reg_key: RegE
    _loss_key: LossE
    _batch: int

    def __init__(self, **kwargs):
        """
        loss_key=LossE.MSE,
                 reg_key=RegE.L2, learning_rate: float = 0.001, error: float = 1e-5, max_num_iterations: int = 1000
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rage):
        self._learning_rate = learning_rage

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error):
        self._error = error

    @property
    def max_num_iterations(self):
        return self._max_num_iterations

    @max_num_iterations.setter
    def max_num_iterations(self, max_num_iterations):
        self._max_num_iterations = max_num_iterations

    @property
    def reg_obj(self):
        return self._reg_obj

    @reg_obj.setter
    def reg_obj(self, reg_obj):
        self._reg_obj = reg_obj

    @property
    def loss_obj(self):
        return self._loss_obj

    @loss_obj.setter
    def loss_obj(self, loss_obj):
        self._loss_obj = loss_obj

    @property
    def loss_key(self):
        return self._loss_key

    @loss_key.setter
    def loss_key(self, loss_key):
        self._loss_key = loss_key
        self.loss_obj = get_loss(loss_key)

    @property
    def reg_key(self):
        return self._reg_key

    @reg_key.setter
    def reg_key(self, reg_key):
        self._reg_key = reg_key
        self.reg_obj = get_reg(reg_key)

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch):
        self._batch = batch

    @abstractmethod
    def loss(self, x, y):
        raise NotImplementedError("'loss' is not implemented !")

    @abstractmethod
    def train(self, x, y, l):
        raise NotImplementedError("'train' is not implemented !")

    @abstractmethod
    def objective(self, x, y, l=0.0):
        raise NotImplementedError("'objective' is not implemented !")

    @abstractmethod
    def deriv(self, x, y, l=0.0):
        raise NotImplementedError("'deriv' is not implemented !")

    def _loss(self, **kwargs):
        loss = kwargs.get("loss")
        return loss(**kwargs)

    def _objective(self, l=0.0, **kwargs):
        """
        w, b, x, y, pred_, loss
        """
        w = kwargs.get("w")
        r = (l * self.reg_obj.reg(w=w)) if l and self.reg_obj else 0
        return self._loss(**kwargs) + r


class GDsc(Optimizer):

    _node: RealNode

    def __init__(self, node: RealNode, **kwargs):
        """
        batch: int, loss_key=LossE.MSE, reg_key=RegE.L2, learning_rate: float = 0.001,
                 error: float = 1e-5, max_num_iterations: int = 1000
        """
        super().__init__(**kwargs)
        self._node = node

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node):
        self._node = node

    #def _loss(self, w, b, x, y):
    #    return self.loss_obj.loss(w, b, x, y, self.node.pred_)

    def loss(self, x, y):
        return self._loss(w=self.node.get_w(), b=self.node.get_b(), x=x, y=y, pred_=self.node.pred_, loss=self.loss_obj.loss)

    def objective(self, x, y, l=0.0):
        return self._objective(l, w=self.node.get_w(), b=self.node.get_b(), x=x, y=y, pred_=self.node.pred_, loss=self.loss_obj.loss)

    def _deriv(self, l=0.0, **kwargs):
        """
         w, b, x, y, pred
        """
        # pdb.set_trace()
        w = kwargs.get("w")
        rd = l * self.reg_obj.deriv(w=w) if l and self.reg_obj else 0
        dwd, dbd = self.loss_obj.deriv(**kwargs)
        wd = dwd + rd
        bd = dbd
        return wd, bd

    def deriv(self, x, y, l=0.0):
        return self._deriv(l, w=self.node.get_w(), b=self.node.get_b(), x=x, y=y, pred_=self.node.pred_)

    def train(self, x, y, l=0.0):
        """
        Can use a regularization and early stopping
        """
        # pdb.set_trace()
        w = self.node.get_w()
        b = self.node.get_b()
        mse = self._objective(l, w=w, b=b, x=x, y=y, pred_=self.node.pred_, loss=self.loss_obj.loss)
        dmse = 1000 * self.error
        it = 0
        while abs(dmse) > self.error and it < self.max_num_iterations:
            # print("gradient descent: it:  %s, MSE:     %s" %(it, mse))
            # print(self.w, self.b)
            wd, bd = self._deriv(l, w=w, b=b, x=x, y=y, pred_=self.node.pred_)
            w -= self.learning_rate * wd
            b -= self.learning_rate * bd
            dmse = mse
            mse = self._objective(l, w=w, b=b, x=x, y=y, pred_=self.node.pred_, loss=self.loss_obj.loss)
            dmse -= mse
            it += 1
        return w, b


class MBGDsc(GDsc):

    def train(self, x, y, l=0.0):
        # pdb.set_trace()
        dataset_len = len(x)
        num_iterations = math.ceil(len(x) / self.batch)
        w, b = self.node.get_w(), self.node.get_b()
        for i in range(num_iterations):
            if (i + 1) * self.batch < dataset_len:
                x_batch = x[i * self.batch:(i + 1) * self.batch]
                y_batch = y[i * self.batch:(i + 1) * self.batch]
            else:
                x_batch = x[i * self.batch:]
                y_batch = y[i * self.batch:]
            # print(x_batch, y_batch)
            w, b = super().train(x_batch, y_batch, l)
            self.node.set_w(w)
            self.node.set_b(b)
        return self.node.get_w(), self.node.get_b()


class NNGDsc(Optimizer):


    _net: NeuralNetwork

    def __init__(self, net: NeuralNetwork, **kwargs):
        """
        batch: int, loss_key=LossE.MSE, reg_key=RegE.L2, learning_rate: float = 0.001,
                 error: float = 1e-5, max_num_iterations: int = 1000
        """
        super().__init__(**kwargs)
        self._net = net

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, net):
        self._net = net

    def loss(self, x, y):
        pass

    def objective(self, x, y, l=0.0):
        pass

    def deriv(self, x, y, l=0.0):
        pass

    def train(self, x, y, l=0.0):
        """
        Can use a regularization and early stopping
        loss = self.loss
        pred = self.forward(x)
        pred_loss = loss.loss(y=y, pred=pred)
        print("initial loss: %f" % pred_loss)
        for epoch in range(self.epochs):
            deriv = loss.error.deriv(y=y, pred=pred)
            dw, db = self.backward(deriv, self.learning_rate)
            self.update_w(dw)
            self.update_b(db)
            pred = self.forward(x)
            pred_loss = loss.loss(y=y, pred=pred)
            print("epoch: %d, loss: %f" % (epoch, pred_loss))
        """
        # pdb.set_trace()
        pred = self.net.forward(x)
        # check have the same dims
        if y.shape != pred.shape:
            y = y.reshape(pred.shape)
        pred_loss = self._objective(w=self.net.get_w(), y=y, pred=pred, loss=self.loss_obj.loss)
        # print("initial loss: %f" % pred_loss)
        dmse = 1000 * self.error
        it = 0
        while abs(dmse) > self.error and it < self.max_num_iterations:
            # print("gradient descent: it:  %s, MSE:     %s" %(it, mse))
            # print(self.w, self.b)
            deriv = self.loss_obj.error.deriv(y=y, pred=pred)
            dw, db = self.net.backward(deriv, self.learning_rate)
            self.net.update_w(dw)
            self.net.update_b(db)
            pred = self.net.forward(x)
            dmse = pred_loss
            pred_loss = self._objective(w=self.net.get_w(), y=y, pred=pred, loss=self.loss_obj.loss)
            dmse -= pred_loss
            it += 1
        return self.net.get_w(), self.net.get_b()



def get_opt(node: RealNode | NeuralNetwork, opt_key: OptimizersE, **kwargs):
    match opt_key:
        case OptimizersE.GDSC:
            return GDsc(node, **kwargs)
        case OptimizersE.MBGDSC:
            return MBGDsc(node, **kwargs)
        case OptimizersE.NNGDSC:
            return NNGDsc(node, **kwargs)
        case _:
            return GDsc(node, **kwargs)