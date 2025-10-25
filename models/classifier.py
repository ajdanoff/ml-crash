import pdb
from abc import ABC, abstractmethod
from typing import Any
import keras
import pandas as pd
import pytest

from models.linear_model import LinearModel
from models.logistic_regression import LogisticRegression, MiniBatchLogisticRegression
from models.model_stats import DataStats, ScalingE
from models.nodes import NeuralNetwork

class Classifier(ABC):
    """
    Abstract base class for classifiers.
    Defines the interface for training, prediction, and evaluation metrics.
    """

    @abstractmethod
    def train(self, x: Any, y: Any, l: float):
        """
        Train the model with features x, labels y, and learning rate l.
        """
        raise NotImplementedError("'train' is not implemented!")

    @abstractmethod
    def pred(self, x: Any):
        """
        Predict labels for the given features x.
        """
        raise NotImplementedError("'pred' is not implemented!")

    @abstractmethod
    def tp(self, y, pred):
        """
        Calculate true positives given true labels y and predictions pred.
        """
        raise NotImplementedError("'tp' is not implemented!")

    @abstractmethod
    def fp(self, y, pred):
        """
        Calculate false positives given true labels y and predictions pred.
        """
        raise NotImplementedError("'fp' is not implemented!")

    @abstractmethod
    def tn(self, y, pred):
        """
        Calculate true negatives given true labels y and predictions pred.
        """
        raise NotImplementedError("'tn' is not implemented!")

    @abstractmethod
    def fn(self, y, pred):
        """
        Calculate false negatives given true labels y and predictions pred.
        """
        raise NotImplementedError("'fn' is not implemented!")

    def accuracy(self, y, pred):
        """
        Calculate accuracy metric based on true positives, true negatives,
        false positives, and false negatives.
        """
        return (self.tp(y, pred) + self.tn(y, pred)) / (self.tp(y, pred) + self.tn(y, pred) + self.fp(y, pred) + self.fn(y, pred))

    def recall(self, y, pred):
        """
        Calculate recall metric: TP / (TP + FN)
        """
        return self.tp(y, pred) / (self.tp(y, pred) + self.fn(y, pred))

    def fpr(self, y, pred):
        """
        Calculate false positive rate: FP / (FP + TN)
        """
        return self.fp(y, pred) / (self.fp(y, pred) + self.tn(y, pred))

    def precision(self, y, pred):
        """
        Calculate precision metric: TP / (TP + FP)
        """
        return self.tp(y, pred) / (self.tp(y, pred) + self.fp(y, pred))

    def f1(self, y, pred):
        """
        Calculate F1 score: harmonic mean of precision and recall.
        """
        return 2 * self.precision(y, pred) * self.recall(y, pred) / (self.precision(y, pred) + self.recall(y, pred))


class LogRegressionClassifier(Classifier):
    """
    Logistic Regression classifier wrapper supporting threshold-based
    classification and metric calculations.
    """

    def __init__(self, model: LinearModel | LogisticRegression | NeuralNetwork, threshold: float):
        """
        Initialize with a base model and a threshold for classification.

        Args:
            model: Underlying model implementing train and pred methods.
            threshold: Threshold above which prediction is positive.
        """
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
        """
        Train the model with features x, labels y, and learning rate l.
        """
        self.model.train(x, y, l)

    def pred(self, x: Any):
        """
        Predict probabilities for input features x.
        """
        return self.model.pred(x)

    def tp(self, y, pred):
        """
        Calculate true positives.
        """
        positive_preds = pred[y == 1]
        return len(positive_preds[positive_preds > self.threshold])

    def fp(self, y, pred):
        """
        Calculate false positives.
        """
        pred_above_threshold = y[pred > self.threshold]
        return len(pred_above_threshold[pred_above_threshold == 0])

    def tn(self, y, pred):
        """
        Calculate true negatives.
        """
        negative_preds = pred[y == 0]
        return len(negative_preds[negative_preds <= self.threshold])

    def fn(self, y, pred):
        """
        Calculate false negatives.
        """
        pred_below_threshold = y[pred <= self.threshold]
        return len(pred_below_threshold[pred_below_threshold == 1])


@pytest.fixture
def emails_dataset_input_features():
    """
    Return input features to use for the emails dataset.
    """
    return ['the', 'to', 'ect', 'and', 'for', 'of']


@pytest.fixture
def emails_dataset(emails_dataset_input_features):
    """
    Load emails dataset, compute statistics, and prepare train/test splits.
    Yields dataset splits and extended features.
    """
    emails_dataset = pd.read_csv("./data/emails.csv")
    ds = DataStats(emails_dataset, label_cols=["Prediction"])

    extended_features = list(emails_dataset_input_features)
    # Uncomment to add advanced features:
    # extended_features = ds.poly(emails_dataset_input_features, 2)
    # extended_features = ds.corr(emails_dataset_input_features)
    # extended_features = ds.cross(emails_dataset_input_features)

    yield ds.split("Prediction"), extended_features


@pytest.fixture
def rice_dataset():
    """
    Load rice dataset from public URL, normalize numerical features,
    encode class labels, shuffle, and split into train/val/test.
    Returns features and labels for each split.
    """
    rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

    # Selecting relevant features
    rice_dataset = rice_dataset_raw[
        ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
         'Eccentricity', 'Convex_Area', 'Extent', 'Class']]

    # Compute mean and std of numerical columns for normalization
    feature_mean = rice_dataset.mean(numeric_only=True)
    feature_std = rice_dataset.std(numeric_only=True)
    numerical_features = rice_dataset.select_dtypes('number').columns

    # Normalize numerical features
    normalized_dataset = (rice_dataset[numerical_features] - feature_mean) / feature_std

    # Copy class labels to normalized DataFrame
    normalized_dataset['Class'] = rice_dataset['Class']

    # Set random seed for reproducibility
    keras.utils.set_random_seed(42)

    # Binary class label: Cammeo = 1, Osmancik = 0
    normalized_dataset['Class_Bool'] = (normalized_dataset['Class'] == 'Cammeo').astype(int)

    # Shuffle dataset and split indices
    number_samples = len(normalized_dataset)
    index_80th = round(number_samples * 0.8)
    index_90th = index_80th + round(number_samples * 0.1)

    shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
    train_data = shuffled_dataset.iloc[0:index_80th]
    validation_data = shuffled_dataset.iloc[index_80th:index_90th]
    test_data = shuffled_dataset.iloc[index_90th:]

    label_columns = ['Class', 'Class_Bool']

    # Split features and labels
    train_features = train_data.drop(columns=label_columns)
    train_labels = train_data['Class_Bool'].to_numpy()
    validation_features = validation_data.drop(columns=label_columns)
    validation_labels = validation_data['Class_Bool'].to_numpy()
    test_features = test_data.drop(columns=label_columns)
    test_labels = test_data['Class_Bool'].to_numpy()

    return train_features, train_labels, validation_features, validation_labels, test_features, test_labels


def test_emails_dataset(emails_dataset):
    """
    Test logistic regression on emails dataset, printing accuracy, recall,
    precision, and F1 score on validation and test sets.
    """
    split, extended_features = emails_dataset
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = split

    lr = LogisticRegression(epochs=50, num_features=len(extended_features), learning_rate=0.001, error=1e-15, max_num_iterations=5000)
    lrc = LogRegressionClassifier(lr, 0.35)

    # Train logistic regression classifier
    lrc.train(train_features[extended_features], train_labels, 0.001)

    # Validation evaluation
    pred = lr.pred(validation_features[extended_features])
    y = validation_labels
    accuracy = lrc.accuracy(y, pred)
    recall = lrc.recall(y, pred)
    precision = lrc.precision(y, pred)
    f1 = lrc.f1(y, pred)
    print(f"logistic regression model (validation) prediction accuracy: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}")

    # Test evaluation
    pred = lr.pred(test_features[extended_features])
    y = test_labels
    accuracy = lrc.accuracy(y, pred)
    recall = lrc.recall(y, pred)
    precision = lrc.precision(y, pred)
    f1 = lrc.f1(y, pred)
    print(f"logistic regression model (test) prediction accuracy: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}")


def test_rice_dataset(rice_dataset):
    """
    Test mini-batch logistic regression on rice dataset,
    printing accuracy, recall, precision, and F1 score on validation and test sets.
    """
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = rice_dataset

    lr = MiniBatchLogisticRegression(batch=100, epochs=60, num_features=3, learning_rate=0.001, error=1e-10, max_num_iterations=1000)
    lrc = LogRegressionClassifier(lr, 0.35)

    input_features = [
        'Eccentricity',
        'Major_Axis_Length',
        'Area',
    ]

    # Train classifier with selected features
    lrc.train(train_features[input_features], train_labels, 0.001)

    # Validation evaluation
    pred = lr.pred(validation_features[input_features])
    y = validation_labels
    accuracy = lrc.accuracy(y, pred)
    recall = lrc.recall(y, pred)
    precision = lrc.precision(y, pred)
    f1 = lrc.f1(y, pred)
    print(f"linear model prediction accuracy for validation data: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}")

    # Test evaluation
    pred = lr.pred(test_features[input_features])
    y = test_labels
    accuracy = lrc.accuracy(y, pred)
    recall = lrc.recall(y, pred)
    precision = lrc.precision(y, pred)
    f1 = lrc.f1(y, pred)
    print(f"linear model prediction accuracy for testing data: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}")
