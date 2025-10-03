import pdb
from enum import Enum
from typing import Literal, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


class ScalingE(Enum):

    LINEAR_NORM = 1
    Z_SCORE_NORM = 2
    LOG_NORM = 3
    CLIP_NORM = 4


class DataStats:
    _df: pd.DataFrame
    _normalied_df: pd.DataFrame
    _number_samples: int
    _label_cols: list

    def __init__(self, df: pd.DataFrame, scaling: ScalingE = ScalingE.Z_SCORE_NORM, feature_cols: list | None = None, label_cols: list | None = None):
        # pdb.set_trace()
        self._df = df
        self.feature_mean = self.df.mean(numeric_only=True)
        self.feature_std = self.df.std(numeric_only=True)
        if feature_cols is not None:
            self.numerical_features = pd.Index(feature_cols)
        else:
            self.numerical_features = self.df.select_dtypes('number').columns
        self._label_cols = label_cols
        self.feature_min = self.df.min(numeric_only=True)
        self.feature_max = self.df.max(numeric_only=True)
        if self._label_cols is not None:
            self.feature_mean = self.feature_mean.drop(self._label_cols, errors="ignore")
            self.feature_std = self.feature_std.drop(self._label_cols, errors="ignore")
            self.feature_min = self.feature_min.drop(self._label_cols, errors="ignore")
            self.feature_max = self.feature_max.drop(self._label_cols, errors="ignore")
            self.numerical_features = self.numerical_features.drop(self._label_cols, errors="ignore")
        self._normalized_df = self.scale(scaling)
        self._normalized_df[self._label_cols] = self.df[self._label_cols]
        self._number_samples = len(self._normalized_df)


    @property
    def df(self):
        return self._df

    @property
    def normalized_df(self):
        return self._normalized_df

    @property
    def number_samples(self):
        return self._number_samples

    @property
    def label_cols(self):
        return self._label_cols

    def linear_scaling(self):
        return (self.df[self.numerical_features] - self.feature_min) / (self.feature_max - self.feature_min)

    def z_score_scaling(self):
        # pdb.set_trace()
        return (
                self.df[self.numerical_features] - self.feature_mean
        ) / self.feature_std

    def log_scaling(self):
        return np.log(self.df[self.numerical_features])

    @classmethod
    def binning(cls, df, col, bins: int | list , labels: list | None = None):
        if labels is None:
            return pd.cut(df[col], bins=bins, include_lowest=True)
        else:
            return pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)

    @classmethod
    def qbinning(cls, df, col, quantiles: int | list , labels: list | None = None, duplicates: Literal["raise", "drop"] = "raise"):
        if labels is None:
            return pd.qcut(df[col], q=quantiles, duplicates = duplicates)
        else:
            return pd.qcut(df[col], q=quantiles, labels=labels, duplicates = duplicates)


    @classmethod
    def clip(cls, df, min_threshold: int | None = None, max_threshold: int | None = None):
        numerical_features = df.select_dtypes('number').columns
        for col in numerical_features:
            df.loc[df[col] > max_threshold] = max_threshold
            df.loc[df[col] < min_threshold] = min_threshold
        return df

    def poly(self, cols: list[str], order: int):
        extended_cols = list(cols)
        for col in cols:
            col_2 = f"{col}^{order}"
            self.normalized_df[col_2] = self.normalized_df[col] ** order
            extended_cols.append(col_2)
        return extended_cols

    def corr(self, cols: list[str]):
        extended_cols = list(cols)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                col_corr = f"{cols[i]}_{cols[j]}"
                self.normalized_df[col_corr] = self.normalized_df[cols[i]].corr(self.normalized_df[cols[j]])
                extended_cols.append(col_corr)
        return extended_cols



    def split(self, label: str, q1: float = 0.8, q2: float = 0.1):
        index_80th = round(self.number_samples * q1)
        index_90th = index_80th + round(self.number_samples * q2)

        # Randomize order and split into train, validation, and test with a .8, .1, .1 split
        shuffled_dataset = self.normalized_df.sample(frac=1, random_state=100)
        train_data = shuffled_dataset.iloc[0:index_80th]
        validation_data = shuffled_dataset.iloc[index_80th:index_90th]
        test_data = shuffled_dataset.iloc[index_90th:]

        # Show the first five rows of the last split
        # print(test_data.head())

        # label_columns = ['Class', 'Class_Bool']

        train_features = train_data.drop(columns=self.label_cols)
        train_labels = train_data[label].to_numpy()
        validation_features = validation_data.drop(columns=self.label_cols)
        validation_labels = validation_data[label].to_numpy()
        test_features = test_data.drop(columns=self.label_cols)
        test_labels = test_data[label].to_numpy()
        return train_features, train_labels, validation_features, validation_labels, test_features, test_labels

    def scale(self, scaling: ScalingE):
        match scaling:
            case ScalingE.LINEAR_NORM:
                return self.linear_scaling()
            case ScalingE.Z_SCORE_NORM:
                return self.z_score_scaling()
            case ScalingE.LOG_NORM:
                return self.log_scaling()
            case _:
                return self.z_score_scaling()

    def detect_outliers(self):
        outliers = []
        for col in self.numerical_features:
            mstd = self.df[col].mean() / self.df[col].std()
            d75max = self.df[col].max() - self.df[col].quantile(0.75)
            d25min = self.df[col].quantile(0.25) - self.df[col].min()
            rd75maxd25min = d75max / d25min
            if mstd > 0.77 and rd75maxd25min < 0.1:
                outliers.append(col)
        return outliers

    def plot_the_dataset(self, feature, label, number_of_points_to_plot):
        """Plot N random points of the dataset."""

        # Label the axes.
        plt.xlabel(feature)
        plt.ylabel(label)

        # Create a scatter plot from n random points of the dataset.
        random_examples = self.df.sample(n=number_of_points_to_plot)
        plt.scatter(random_examples[feature], random_examples[label])

        # Render the scatter plot.
        plt.show()

    def plot_a_contiguous_portion_of_dataset(self, feature, label, start, end):
        """Plot the data points from start to end."""

        # Label the axes.
        plt.xlabel(feature)
        plt.ylabel(label)

        # Create a scatter plot.
        plt.scatter(self.df[feature][start:end], self.df[label][start:end])

        # Render the scatter plot.
        plt.show()
