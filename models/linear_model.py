import math
import pdb

import pandas as pd
import pytest

from models.loss import Loss, MSELoss, get_loss, LossE
from models.nodes import LinearNode, NodesE, Node, RealNode
from models.optimization import Optimizer, get_opt, OptimizersE
from models.regularization import RegE, Regularization, get_reg

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


class LinearModel(RealNode):  # (LinearNode):
    _epochs: int
    _optimizer: Optimizer

    def __init__(self, epochs: int, num_features: int, node_key: NodesE = NodesE.LINEAR_NODE, opt_key: OptimizersE = OptimizersE.GDSC,
                 learning_rate: float = 0.001, error: float = 1e-5, max_num_iterations: int = 1000, loss_key=LossE.MSE,
                 reg_key=RegE.L2):
        super().__init__(node_key, num_features)
        self._optimizer = get_opt(self, opt_key, loss_key=loss_key, reg_key=reg_key, learning_rate=learning_rate, error=error, max_num_iterations=max_num_iterations)
        self._epochs = epochs

    @property
    def epochs(self):
        return self._epochs

    @property
    def optimizer(self):
        return self._optimizer

    def train(self, x, y, l=0.0):
        # pdb.set_trace()
        for e in range(1, self.epochs + 1):
            w, b = self.optimizer.train(x, y, l)
            self.set_w(w)
            self.set_b(b)
            print("epochs:  %s, LOSS:     %s" % (e, self.optimizer.loss(x, y)))
        return self.get_w(), self.get_b()


class MiniBatchModel(LinearModel):
    _batch: int

    def __init__(self, batch: int, epochs: int, num_features: int, node_key: NodesE = NodesE.LINEAR_NODE, opt_key: OptimizersE = OptimizersE.MBGDSC,
                 learning_rate: float = 0.001, error: float = 1e-5,
                 max_num_iterations: int = 1000, loss_key=MSELoss, reg_key=RegE.L2):
        LinearModel.__init__(self,
                             epochs=epochs,
                             num_features=num_features,
                             node_key=node_key,
                             opt_key=opt_key,
                             learning_rate=learning_rate,
                             error=error, max_num_iterations=max_num_iterations, loss_key=loss_key, reg_key=reg_key)
        self.optimizer.batch = batch
        self._batch = batch

    @property
    def batch(self):
        return self._batch

    def train(self, x, y, l=0.0):
        # pdb.set_trace()
        for e in range(1, self.epochs + 1):
            w, b = self.optimizer.train(x, y, l)
            print("epochs:  %s, LOSS:     %s" % (e, self.optimizer.loss(x, y)))
        return self.get_w(), self.get_b()


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
