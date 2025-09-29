from abc import ABC, abstractmethod
from typing import Any

import keras
import pandas as pd
import pytest

from models.linear_model import LinearModel
from models.logistic_regression import LogisticRegression, MiniBatchLogisticRegression


class Classifier(ABC):

    @abstractmethod
    def train(self, x:Any, y:Any, l:float):
        raise NotImplementedError("'train' is not implemented !")

    @abstractmethod
    def pred(self, x: Any):
        raise NotImplementedError("'pred' is not implemented !")

    @abstractmethod
    def tp(self, y, pred):
        raise NotImplementedError("'tp' is not implemented !")

    @abstractmethod
    def fp(self, y, pred):
        raise NotImplementedError("'fp' is not implemented !")

    @abstractmethod
    def tn(self, y, pred):
        raise NotImplementedError("'tn' is not implemented !")

    @abstractmethod
    def fn(self, y, pred):
        raise NotImplementedError("'fn' is not implemented !")

    def accuracy(self, y, pred):
        return (self.tp(y, pred) + self.tn(y, pred))/(self.tp(y, pred) + self.tn(y, pred) + self.fp(y, pred) + self.fn(y, pred))

    def recall(self, y, pred):
        return self.tp(y, pred)/(self.tp(y, pred) + self.fn(y, pred))

    def fpr(self, y, pred):
        return self.fp(y, pred)/(self.fp(y, pred) + self.tn(y, pred))

    def precision(self, y, pred):
        return self.tp(y, pred)/(self.tp(y, pred) + self.fp(y, pred))

    def f1(self, y, pred):
        return 2 * self.precision(y, pred) * self.recall(y, pred) / (self.precision(y, pred) + self.recall(y, pred))


class LogRegressionClassifier(Classifier):

    def __init__(self, model: LinearModel, threshold: float):
        self._model = model
        self._threshold = threshold

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    def train(self, x: Any, y: Any, l: float):
        self.model.train(x, y, l)

    def pred(self, x: Any):
        return self.model.pred(x)

    def tp(self, y, pred):
        a = pred[y == 1]
        return len( a[a > self.threshold] )

    def fp(self, y, pred):
        c = y[pred > self.threshold]
        return len(c[c == 0])

    def tn(self, y, pred):
        b = pred[y == 0]
        return len(b[b <= self.threshold])

    def fn(self, y, pred):
        d = y[pred <= self.threshold]
        return len( d[d == 1] )

@pytest.fixture
def emails_dataset():
    emails_dataset = pd.read_csv("./data/emails.csv")
    yield emails_dataset

@pytest.fixture
def rice_dataset():
    rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

    # @title
    # Read and provide statistics on the dataset.
    rice_dataset = rice_dataset_raw[[
        'Area',
        'Perimeter',
        'Major_Axis_Length',
        'Minor_Axis_Length',
        'Eccentricity',
        'Convex_Area',
        'Extent',
        'Class',
    ]]

    # Calculate the Z-scores of each numerical column in the raw data and write
    # them into a new DataFrame named df_norm.

    feature_mean = rice_dataset.mean(numeric_only=True)
    feature_std = rice_dataset.std(numeric_only=True)
    numerical_features = rice_dataset.select_dtypes('number').columns
    normalized_dataset = (
                                 rice_dataset[numerical_features] - feature_mean
                         ) / feature_std

    # Copy the class to the new dataframe
    normalized_dataset['Class'] = rice_dataset['Class']

    keras.utils.set_random_seed(42)

    # Create a column setting the Cammeo label to '1' and the Osmancik label to '0'
    # then show 10 randomly selected rows.
    normalized_dataset['Class_Bool'] = (
        # Returns true if class is Cammeo, and false if class is Osmancik
            normalized_dataset['Class'] == 'Cammeo'
    ).astype(int)
    normalized_dataset.sample(10)

    # Create indices at the 80th and 90th percentiles
    number_samples = len(normalized_dataset)
    index_80th = round(number_samples * 0.8)
    index_90th = index_80th + round(number_samples * 0.1)

    # Randomize order and split into train, validation, and test with a .8, .1, .1 split
    shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
    train_data = shuffled_dataset.iloc[0:index_80th]
    validation_data = shuffled_dataset.iloc[index_80th:index_90th]
    test_data = shuffled_dataset.iloc[index_90th:]

    label_columns = ['Class', 'Class_Bool']

    train_features = train_data.drop(columns=label_columns)
    train_labels = train_data['Class_Bool'].to_numpy()
    validation_features = validation_data.drop(columns=label_columns)
    validation_labels = validation_data['Class_Bool'].to_numpy()
    test_features = test_data.drop(columns=label_columns)
    test_labels = test_data['Class_Bool'].to_numpy()
    return train_features, train_labels, validation_features, validation_labels, test_features, test_labels

def test_emails_dataset(emails_dataset):
    lr = LogisticRegression(epochs=60, num_features=6, learning_rate=0.001, error=1e-10, max_num_iterations=1000)
    lrc = LogRegressionClassifier(lr, 0.35)
    input_features = ['the', 'to', 'ect', 'and', 'for', 'of']
    lrc.train(emails_dataset[input_features], emails_dataset['Prediction'], 0.001)
    pred = lr.pred(emails_dataset[input_features])
    y = emails_dataset['Prediction']
    accuracy = lrc.accuracy(y, pred)
    recall = lrc.recall(y, pred)
    precision = lrc.precision(y, pred)
    f1 = lrc.f1(y, pred)
    print("linear model prediction accuracy: %s, recall: %s, precision: %s, f1: %s" % (accuracy, recall, precision, f1))

def test_rice_dataset(rice_dataset):
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = rice_dataset
    lr = MiniBatchLogisticRegression(batch = 100, epochs=60, num_features=3, learning_rate=0.001, error=1e-10, max_num_iterations=1000)
    lrc = LogRegressionClassifier(lr, 0.35)
    input_features = [
        'Eccentricity',
        'Major_Axis_Length',
        'Area',
    ]
    lrc.train(train_features[input_features], train_labels, 0.001)
    pred = lr.pred(validation_features[input_features])
    y = validation_labels
    accuracy = lrc.accuracy(y, pred)
    recall = lrc.recall(y, pred)
    precision = lrc.precision(y, pred)
    f1 = lrc.f1(y, pred)
    print("linear model prediction accuracy for validation data: %s, recall: %s, precision: %s, f1: %s" % (accuracy, recall, precision, f1))
    pred = lr.pred(test_features[input_features])
    y = test_labels
    accuracy = lrc.accuracy(y, pred)
    recall = lrc.recall(y, pred)
    precision = lrc.precision(y, pred)
    f1 = lrc.f1(y, pred)
    print("linear model prediction accuracy for testing data: %s, recall: %s, precision: %s, f1: %s" % (accuracy,
                                                                                                           recall,
                                                                                                           precision,
                                                                                                           f1))