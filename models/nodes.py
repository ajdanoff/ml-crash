import math
import pdb
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np

from models.loss import NNLoss


class NodesE(Enum):

    LINEAR_NODE = 1
    ACTIV_NODE = 2


class Node(ABC):
    """
    Abstract node class
    """

    @abstractmethod
    def pred_(self, w, b, x):
        raise NotImplementedError("'_pred' is not implemented !")

    @abstractmethod
    def pred(self, x):
        raise NotImplementedError("'_pred' is not implemented !")


class LinearNode:
    """
    Linear Node class
    """
    _w: np.ndarray | float
    _b: float
    _num_features: int

    def __init__(self, num_features: int):
        self._w = self.init_random(-1, 1, num_features)
        self._b = self.init_random(-1, 1)

    @classmethod
    def init_random(cls, a, b, size: int = 1):
        rng = np.random.default_rng()
        if size == 1:
            return (b - a)*rng.random() + a
        return (b - a)*rng.random(size) + a

    def pred_(self, w, b, x):
        # x = np.array(x).transpose()
        return np.dot(x, w) + b

    def pred(self, x):
        # x = np.array(x).transpose()
        return np.dot(x, self.get_w()) + self.get_b()

    def get_b(self):
        return self._b

    def get_w(self):
        return self._w

    def set_w(self, w):
        self._w = w

    def set_b(self, b):
        self._b = b

    @property
    def num_features(self):
        return self._num_features


class ActFunction(ABC):

    @abstractmethod
    def act_function(self, y):
        raise NotImplementedError("'act_function' is not implemented !")

    @abstractmethod
    def deriv_(self, y):
        raise NotImplementedError("'deriv_' is not implemented !")

    def activate(self, y):
        return self.act_function(y)


class SigmActFunction(ActFunction):

    @classmethod
    def sigmoid(cls, z):
        return 1 / (1 + np.exp(-z))

    def act_function(self, y):
        return self.sigmoid(y)

    def deriv_(self, y):
        return y * ( 1 - y )


class TanhActFunction(ActFunction):

    def deriv_(self, y):
        pass

    def act_function(self, y):
        return np.tanh(y)


class ReLUActFunction(ActFunction):

    def deriv_(self, y):
        grad = np.where(y > 0, 1, 0)
        return grad

    def act_function(self, y):
        return np.maximum(0, y)

class SoftmaxActFunction(ActFunction):

    def deriv_(self, y):
        pass

    def act_function(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class ActFuncsE(Enum):
    SIGM = 1
    TANH = 2
    RELU = 3
    SOFTMAX = 4

def get_act_func(func_key: ActFuncsE):
    match func_key:
        case ActFuncsE.SIGM:
            return SigmActFunction()
        case ActFuncsE.TANH:
            return TanhActFunction()
        case ActFuncsE.RELU:
            return ReLUActFunction()
        case ActFuncsE.SOFTMAX:
            return SoftmaxActFunction()
        case _:
            return SigmActFunction()


class ActNode(LinearNode):
    """
    Activation Node with a sigmoid function
    """

    _act_func: ActFunction

    def __init__(self, num_features: int, func_key: ActFuncsE = ActFuncsE.SIGM):
        super().__init__(num_features)
        self._act_func = get_act_func(func_key)

    @property
    def act_func(self):
        return self._act_func

    def pred_(self, w, b, x):
        return self.act_func.activate(super().pred_(w, b, x))


class ImNode(ActNode):

    _inp: Any
    _outp: Any

    def __init__(self, num_features: int, func_key: ActFuncsE = ActFuncsE.SIGM):
        super().__init__(num_features, func_key)
        self._inp = None
        self._outp = None

    @property
    def inp(self):
        return self._inp

    @inp.setter
    def inp(self, inp):
        self._inp = inp

    @property
    def outp(self):
        return self._outp

    @outp.setter
    def outp(self, outp):
        self._outp = outp

    def forward(self, x):
        self.inp = x
        return self.pred(x)

    def update_w(self, dw):
        # pdb.set_trace()
        self.set_w(self.get_w() + dw)

    def update_b(self, db):
        self.set_b(self.get_b() + db)

    def backward(self, y):
        d = (y - self.get_b())
        outp = d.reshape(len(d), 1)*1/self.get_w()
        self.outp = outp
        return outp


class RealNode(Node):
    """
    Real Node: encapsulates linear or activation.
    """

    def __init__(self, node_key: NodesE, num_features):
        self._node = self.get_node(node_key, num_features)

    @property
    def node(self):
        return self._node

    def pred_(self, w, b, x):
        return self.node.pred_(w, b, x)

    def pred(self, x):
        return self.node.pred(x)

    def get_w(self):
        return self.node.get_w()

    def get_b(self):
        return self.node.get_b()

    def set_w(self, w):
        self.node.set_w(w)

    def set_b(self, b):
        self.node.set_b(b)

    @classmethod
    def get_node(cls, node_key: NodesE, num_features: int):
        match node_key:
            case NodesE.LINEAR_NODE:
                return LinearNode(num_features)
            case NodesE.ACTIV_NODE:
                return ActNode(num_features)
            case _:
                return LinearNode(num_features)


class NNLayer(ABC):
    _loss: NNLoss

    def __init__(self, loss: NNLoss):
        self._loss = loss

    @property
    def loss(self):
        return self._loss

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("'forward' is not implemented !")

    @abstractmethod
    def backward(self, error, learning_rate: float):
        raise NotImplementedError("'backward' is not implemented !")

    @abstractmethod
    def print_params(self):
        raise NotImplementedError("'print_params' is not implemented !")

    @classmethod
    def print_node(cls, node: ImNode):
        print("node: %s, weights: %s, bias: %s" % (node, node.get_w(), node.get_b()))

    @abstractmethod
    def update_w(self, dw):
        raise NotImplementedError("'update_w' is not implemented !")

    @abstractmethod
    def update_b(self, db):
        raise NotImplementedError("'update_b' is not implemented !")

    @abstractmethod
    def get_w(self):
        raise NotImplementedError("'get_w' is not implemented !")

    @abstractmethod
    def get_b(self):
        raise NotImplementedError("'get_b' is not implemented !")


class NeuralNetwork(NNLayer):

    @abstractmethod
    def train(self, x, y, l=0.0):
        raise NotImplementedError("'train' is not implemented !")

    @abstractmethod
    def pred(self, x):
        raise NotImplementedError("'pred' is not implemented !")
