import pdb
from typing import Any

import numpy as np
import pandas as pd
import pytest

from models.classifier import LogRegressionClassifier
from models.loss import LossE, get_loss, NNLoss
from models.model_stats import DataStats
from models.nodes import ImNode, ActFuncsE, get_act_func, ActFunction, NNLayer, NeuralNetwork
from models.optimization import get_opt, OptimizersE, Optimizer
from models.regularization import RegE


class HiddenLayer(NNLayer):
    
    _nodes: list[ImNode]
    _act_func: ActFunction
    _inp: Any
    _outp: Any

    def __init__(self, num_nodes: int, num_features: int, func_key: ActFuncsE = ActFuncsE.SIGM, loss_key: LossE = LossE.NN_MSE_LOSS):
        super().__init__(get_loss(loss_key))
        self._nodes = [ImNode(num_features, func_key=func_key) for _ in range(num_nodes)]
        self._act_func = get_act_func(func_key)
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

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def act_func(self):
        return self._act_func

    def get_w(self):
        w = [node.get_w() for node in self.nodes]
        w = np.stack(w, axis=1)
        return w

    def get_b(self):
        b = [node.get_b() for node in self.nodes]
        return b

    def forward(self, x):
        # pdb.set_trace()
        self.inp = x
        self.outp = self.act_func.activate(np.dot(x, self.get_w()) + self.get_b())
        return self.outp

    def backward(self, error_deriv, learning_rate: float):
        return self.loss.deriv(error_deriv=error_deriv, act_func=self.act_func, x=self.inp, w=self.get_w(), y=self.outp, learning_rate=learning_rate)
        #delta = error * self.act_func.deriv_(self.outp)
        #dw = -np.dot(self.inp.transpose(), delta) * learning_rate
        #db = -np.sum(delta, keepdims=True, axis=0)*learning_rate
        #return np.dot(delta, self.get_w().transpose()), dw, db

    def print_params(self):
        for node in self.nodes:
            self.print_node(node)

    def update_w(self, dw):
        # pdb.set_trace()
        # print(dw)
        # print(self.nodes)
        for i in range(dw.shape[1]):
            self.nodes[i].update_w(dw[:,i])

    def update_b(self, db):
        # pdb.set_trace()
        for i in range(len(self.nodes)):
            self.nodes[i].update_b(db[0][i])


class OutputLayer(HiddenLayer):

    def __init__(self, num_nodes: int, num_features: int, func_key: ActFuncsE = ActFuncsE.SIGM, loss_key: LossE = LossE.NN_MSE_LOSS):
        super().__init__(num_nodes, num_features, func_key, loss_key)


class FFNN(NeuralNetwork):

    _layers: list[NNLayer]
    _learning_rate: float
    _epochs: int
    _optimizer: Optimizer

    def __init__(self, layers: list[NNLayer], epochs=10, opt_key: OptimizersE = OptimizersE.NNGDSC, loss_key=LossE.NN_MSE_LOSS,
                 reg_key=RegE.L2, learning_rate: float = 0.001, error: float = 1e-5, max_num_iterations: int = 1000):
        super().__init__(get_loss(loss_key))
        self._layers = layers
        self._epochs = epochs
        self._optimizer = get_opt(self, opt_key, loss_key=loss_key, reg_key=reg_key, learning_rate=learning_rate, error=error, max_num_iterations=max_num_iterations)
        # self._learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def epochs(self):
        return self._epochs

    @property
    def layers(self):
        return self._layers

    @property
    def optimizer(self):
        return self._optimizer

    def get_w(self):
        w = [node.get_w() for node in self.layers]
        return w

    def get_b(self):
        b = [node.get_b() for node in self.layers]
        return b

    def update_w(self, dw):
        for i in range(len(self.layers)):
            self.layers[i].update_w(dw[i])

    def update_b(self, db):
        for i in range(len(self.layers)):
            self.layers[i].update_w(db[i])

    def train(self, x, y, l=0.0):
        """loss = self.loss
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
        pred = self.forward(x)
        pred_loss = self.loss.loss(y=y, pred=pred)
        print("initial loss: %f" % pred_loss)
        for epoch in range(self.epochs):
            # update of w,b is inside optimizer
            _, _ = self.optimizer.train(x, y, l)
            pred = self.forward(x)
            pred_loss = self.loss.loss(y=y, pred=pred)
            print("epoch: %d, loss: %f" % (epoch, pred_loss))

    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def backward(self, error, learning_rate: float):
        dwl = []
        dbl = []
        for layer in self._layers[::-1]:
            error, dw, db = layer.backward(error, learning_rate)
            dwl.insert(0, dw)
            dbl.insert(0, db)
        return dwl, dbl

    def print_params(self):
        for layer in self._layers:
            layer.print_params()

    def pred(self, x):
        return self.forward(x)


def test_forward_nn():
    #pdb.set_trace()
    hl1 = HiddenLayer(4, 3)
    hl1.print_params()
    hl2 = HiddenLayer(4, 4)
    hl2.print_params()
    ol = OutputLayer(1, 4)
    ol.print_params()
    x = np.array([[0, 0, 0],
         [1, 1, 1],
         [0, 0, 1],
         [0, 1, 0]])
    y = np.array([[0], [0], [1], [1]])
    print(hl1.forward(x))
    print(hl2.forward(hl1.forward(x)))
    pred = ol.forward(hl2.forward(hl1.forward(x)))
    print("pred: %s" % pred)
    error = ol.loss.loss(y=y, pred=pred)
    print("error: %s" % error)
    deriv = ol.loss.error.deriv(y=y, pred=pred)
    deriv1, dw, db = ol.backward(deriv, learning_rate=0.001)
    ol.update_w(dw)
    ol.update_b(db)
    deriv2, dw, db = hl2.backward(deriv1, learning_rate=0.001)
    hl2.update_w(dw)
    hl2.update_b(db)
    _, dw, db = hl1.backward(deriv2, learning_rate=0.001)
    hl1.update_w(dw)
    hl1.update_b(db)
    pred = ol.forward(hl2.forward(hl1.forward(x)))
    print("pred: %s" % pred)
    error = ol.loss.loss(y=y, pred=pred)
    print("error: %s" % error)
    layers = [
        hl1,
        hl2,
        ol
    ]
    ffnn = FFNN(layers)
    ffnn.print_params()
    y_pred = ffnn.forward(x)
    error = ol.loss.loss(y=y, pred=y_pred)
    deriv = ol.loss.error.deriv(y=y, pred=y_pred)
    dw, db = ffnn.backward(deriv, 0.001)
    ffnn.update_w(dw)
    ffnn.update_b(db)
    print("error: %s" % error)


def test_train_nn():
    x = np.array([[0, 0],
                  [1, 1],
                  [0, 1],
                  [1, 0]])
    y = np.array([[0], [0], [1], [1]])
    hl1 = HiddenLayer(4, 2)
    hl2 = HiddenLayer(4, 4)
    ol = OutputLayer(1, 4)
    layers = [
        hl1,
        hl2,
        ol
    ]
    ffnn = FFNN(layers, epochs=2, learning_rate=1.0, error=1e-10)
    ffnn.print_params()
    ffnn.train(x, y, l=0.001)
    print(ffnn.forward([1, 1]))

@pytest.fixture
def emails_dataset_input_features():
    # return ['the', 'ect']
    return ['the', 'to', 'ect', 'and', 'for', 'of']

@pytest.fixture
def emails_dataset(emails_dataset_input_features):
    # pdb.set_trace()
    emails_dataset = pd.read_csv("./data/emails.csv")
    ds = DataStats(emails_dataset.sample(n=500), label_cols=["Prediction"])
    # add second-order features
    extended_features = list(emails_dataset_input_features)
    # extended_features = ds.poly(emails_dataset_input_features, 2)
    # extended_features = ds.corr(emails_dataset_input_features)
    # extended_features = ds.cross(emails_dataset_input_features)
    rnd_cols_df = emails_dataset.sample(n=16, axis='columns')
    yield ds.split("Prediction"), rnd_cols_df.columns.to_list()

def test_emails_dataset_nn(emails_dataset):
    # pdb.set_trace()
    split, extended_features = emails_dataset
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = split
    labels = [train_labels, validation_labels, test_labels]
    for i in range(len(labels)):
        labels[i] = labels[i].reshape((labels[i].shape[0], 1))
    hl1 = HiddenLayer(64, num_features=len(extended_features))
    hl2 = HiddenLayer(32, 64)
    hl3 = HiddenLayer(16, 32)
    ol = OutputLayer(1, 16)
    layers = [
        hl1,
        hl2,
        hl3,
        ol
    ]
    ffnn = FFNN(layers, epochs=30, learning_rate=0.0001, error=1e-15, max_num_iterations=5000)
    # ffnn.print_params()
    lrc = LogRegressionClassifier(ffnn, 0.35)
    ffnn.train(train_features[extended_features], labels[0], 0.01)
    pred = ffnn.pred(validation_features[extended_features])
    y = labels[1]
    accuracy = lrc.accuracy(y, pred)
    recall = lrc.recall(y, pred)
    precision = lrc.precision(y, pred)
    f1 = lrc.f1(y, pred)
    print(
        "logistic regression model (validation) prediction accuracy: %s, recall: %s, precision: %s, f1: %s" % (accuracy,
                                                                                                               recall,
                                                                                                               precision,
                                                                                                               f1))
    pred = ffnn.pred(test_features[extended_features])
    y = labels[2]
    accuracy = lrc.accuracy(y, pred)
    recall = lrc.recall(y, pred)
    precision = lrc.precision(y, pred)
    f1 = lrc.f1(y, pred)
    print(
        "logistic regression model (test) prediction accuracy: %s, recall: %s, precision: %s, f1: %s" % (accuracy,
                                                                                                         recall,
                                                                                                         precision,
                                                                                                         f1))
