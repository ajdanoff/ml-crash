import math
import pdb

import numpy as np
import pandas as pd
import pytest

from models.linear_model import LinearModel, MiniBatchModel
from models.nodes import NodesE
from models.optimization import OptimizersE
from models.regularization import RegE
from models.loss import LossE


class LogisticRegression(LinearModel):
    """
    """

    def __init__(self, epochs: int, num_features: int, node_key: NodesE = NodesE.ACTIV_NODE, opt_key: OptimizersE = OptimizersE.GDSC,
                 learning_rate: float = 0.001, error: float = 1e-5,
                 max_num_iterations: int = 1000, loss_key: LossE = LossE.LOG_LOSS, reg_key: RegE = RegE.L2):

        super().__init__(epochs, num_features, node_key, opt_key, learning_rate, error, max_num_iterations, loss_key, reg_key)


class MiniBatchLogisticRegression(MiniBatchModel):
    """

    """

    def __init__(self, batch: int, epochs: int, num_features: int, node_key: NodesE = NodesE.ACTIV_NODE, opt_key: OptimizersE = OptimizersE.MBGDSC,
                 learning_rate: float = 0.001, error: float = 1e-5,
                 max_num_iterations: int = 1000, loss_key: LossE = LossE.LOG_LOSS, reg_key: RegE = RegE.L2):
        super().__init__(batch, epochs, num_features, node_key, opt_key, learning_rate, error, max_num_iterations, loss_key, reg_key)


@pytest.fixture
def emails_dataset():
    emails_dataset = pd.read_csv("./data/emails.csv")
    yield emails_dataset

def test_logistic_regression(emails_dataset):
    lr = LogisticRegression(epochs=10, num_features=6, learning_rate=0.001)
    print(lr.train(emails_dataset[['the', 'to', 'ect', 'and', 'for', 'of']], emails_dataset['Prediction'], 0.001))
    pred = lr.pred(emails_dataset[['the', 'to', 'ect', 'and', 'for', 'of']])
    y = emails_dataset['Prediction']
    threshold = 0.35
    a = pred[y==1]
    tp = a[a>threshold]
    b = pred[y == 0]
    tn = b[b<=threshold]
    c = y[pred > threshold]
    fp = c[c == 0]
    d = y[pred <= threshold]
    fn = d[d==1]
    accuracy = (len(tp) + len(tn)) / (len(tp) + len(tn) + len(fp) + len(fn))
    print("linear model prediction accuracy: %s" % accuracy)