import itertools
import pdb
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import pytest

import string
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Token:

    def __init__(self, code: int, freq: int = 0):
        self.code = code
        self.freq = freq

class Vectorizer:
    """

    """
    def __init__(self, max_voc_size: int = 20000, language:str = "english"):
        self.vocabulary = {"": Token(0), "[UNK]": Token(1)}
        self.max_voc_size = max_voc_size
        self.stemmer = SnowballStemmer(language)
        self.language = language
        self.stop_words = set(stopwords.words(self.language))

    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    def tokenize(self, text: str):
        return word_tokenize(text, self.language)

    def make_vocabulary(self, dataset, stemming: bool = False ):
        t = 0
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if stemming:
                    token = self.stemmer.stem(token)
                if token not in self.stop_words:
                    if token not in self.vocabulary:
                        self.vocabulary[token] = Token(t)
                        t += 1
                    else:
                        self.vocabulary[token].freq += 1
            if len(self.vocabulary) > self.max_voc_size:
                self.trunc_vocab()
        self.inverse_vocabulary = dict((v.code, k) for k, v in self.vocabulary.items())

    def trunc_vocab(self):
        sorted_dict = dict(sorted(self.vocabulary.items(), key=lambda item: item[1].freq, reverse=True))
        trunc_sorted_dict = dict(itertools.islice(sorted_dict.items(), 0, self.max_voc_size))
        keys = list(trunc_sorted_dict.keys())
        self.vocabulary = {"": Token(0), "[UNK]": Token(1)}
        self.vocabulary.update(dict((keys[i], Token(i+2)) for i in range(len(keys))))
        print("vocabulary truncated: %s" %dict(itertools.islice(self.vocabulary.items(), 0, 10)))

    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, self.vocabulary["[UNK]"]).code for token in tokens]

    def multi_hot_encode(self, codes: Any):
        vec = np.zeros(len(self.vocabulary) + 2)
        for code in codes:
            vec[code] = 1
        return vec

    def multi_hot_decode(self, mlh_codes: Any):
        codes = [i for i in range(len(mlh_codes)) if mlh_codes[i] == 1]
        return codes

    def decode(self, int_sequence):
        return " ".join(
            self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence
        )

    def print_vocabulary(self):
        for k, v in self.vocabulary.items():
            print("%s: %s" %(k, v))

dataset = [
    """
    “Many Karate teachers teach a watered down style – no hip action and no depth of punching – so it is easy to say that these teachers have no depth to their knowledge.
     You are what your teacher is, and if he knows a lot, you should be able to demonstrate this knowledge.”
    """,
    """
    “Karate has no philosophy. 
    Some people think that the tradition of Karate came from Buddhism and Karate has a connection with the absolute, space and universe, but I don’t believe in that.
    My philosophy is to knock my opponent out, due to the use of only one technique. One finishing blow!”
    """,
    """
    “In the past, it was expected that about three years were required to learn a single kata, and usually even an expert of considerable skill would only know three,
     or at most five, kata.” 
    """,
    """
    “To all those whose progress remains hampered by ego-related distractions, 
    let humility – the spiritual cornerstone upon which Karate rests – serve to remind one to place virtue before vice, values before vanity and principles before personalities.” 
    """,
    """
    “Once a kata has been learned, it must be practiced repeatedly until it can be applied in an emergency, for knowledge of just the sequence of a form in Karate is useless.”
    """
]

@pytest.mark.parametrize("s", [dataset])
def test_tokenization(s):
    vectorizer = Vectorizer()
    vectorizer.make_vocabulary(dataset, stemming=True)
    vectorizer.print_vocabulary()
    test_sentence = "I demonstrate a kata."
    encoded_sentence = vectorizer.encode(test_sentence)
    print(encoded_sentence)
    mlh_encoded = vectorizer.multi_hot_encode(encoded_sentence)
    print(mlh_encoded)
    mlh_decoded = vectorizer.multi_hot_decode(mlh_encoded)
    print(mlh_decoded)
    decoded_sentence = vectorizer.decode(mlh_decoded)
    print(decoded_sentence)

@pytest.fixture
def fine_tuning():
    pdb.set_trace()
    train = pd.read_csv("./data/llm_class_fine_tuning/train.csv").sample(n=1000)
    # train['model_a'] = train['model_a'].apply()
    test = pd.read_csv("./data/llm_class_fine_tuning/test.csv")
    train, _ = apply_vectorization(train, ['model_a', 'model_b'])
    train, vec = apply_vectorization(train, ['prompt', 'response_a', 'response_b'], stemming=True)
    test = apply_vectorization(test, ['prompt', 'response_a', 'response_b'], vec, stemming=True)
    yield train, test

def apply_vectorization(data_set, cols, vectorizer: Vectorizer | None = None, stemming: bool = False):
    if vectorizer is None:
        vectorizer = Vectorizer()
    for col in cols:
        vectorizer.make_vocabulary(data_set[col], stemming)
        col_enc = data_set[col].apply(vectorizer.encode)
        data_set[col] = col_enc.apply(vectorizer.multi_hot_encode)
    return data_set, vectorizer

def test_tokenization_fine_tuning(fine_tuning):
    train, test = fine_tuning
    pdb.set_trace()
    print(train)

from tensorflow.keras.layers import TextVectorization
text_vectorization = TextVectorization(
    output_mode="int"
)
import tensorflow as tf

def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace(
        lowercase_string, f"[{re.escape(string.punctuation)}]", ""
    )

def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)

@pytest.mark.parametrize("s", [dataset])
def test_vectorization_keras(s):
    text_vectorization = TextVectorization(
        output_mode = "int",
        standardize = custom_standardization_fn,
        split=custom_split_fn
    )
    text_vectorization.adapt(s)
    vocabulary = text_vectorization.get_vocabulary()
    test_sentence = "I write, rewrite, and still rewrite again"
    encoded_sentence = text_vectorization(test_sentence)
    print(encoded_sentence)
    inverse_vocab = dict(enumerate(vocabulary))
    decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
    print(decoded_sentence)

import os, pathlib, shutil, random

def test_prepare_val_set_imdb():
    base_dir = pathlib.Path("data/aclImdb")
    val_dir = base_dir / "val"
    train_dir = base_dir / "train"
    for category in ("neg", "pos"):
        os.makedirs(val_dir / category)
        files = os.listdir(train_dir / category)
        random.Random(1337).shuffle(files)
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname,
                        val_dir / category / fname)

from tensorflow import keras
batch_size = 32

@pytest.fixture()
def create_datasets():
    train_ds = keras.utils.text_dataset_from_directory(
        "data/aclImdb/train", batch_size=batch_size
    )
    val_ds = keras.utils.text_dataset_from_directory(
        "data/aclImdb/val", batch_size=batch_size
    )
    test_ds = keras.utils.text_dataset_from_directory(
        "data/aclImdb/test", batch_size=batch_size
    )
    yield train_ds, val_ds, test_ds

@pytest.fixture()
def map_multihot(create_datasets):
    train_ds, val_ds, test_ds = create_datasets
    text_vectorization = TextVectorization(
        max_tokens = 20000,
        output_mode = "multi_hot",
    )
    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)

    binary_1gram_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4
    )
    binary_1gram_val_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4
    )
    binary_1gram_test_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4
    )
    return binary_1gram_train_ds, binary_1gram_val_ds, binary_1gram_test_ds

def test_inputs(map_multihot):
    train_ds, val_ds, test_ds = map_multihot
    for inputs, targets in train_ds:
        print("inputs.shape: ", inputs.shape)
        print("inputs.dtype: ", inputs.dtype)
        print("targets.shape: ", targets.shape)
        print("targets.dtype: ", targets.dtype)
        print("inputs[0]:", inputs[0])
        print("targets[0]:", targets[0])
        break

from tensorflow import keras
from tensorflow.keras import layers

def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens, ))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def test_bin_unigram_aclimdb(map_multihot):
    binary_1gram_train_ds, binary_1gram_val_ds, binary_1gram_test_ds = map_multihot
    model = get_model()
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint("binary_1gram.keras",
                                        save_best_only=True
                                        )
    ]
    model.fit(binary_1gram_train_ds.cache(),
              validation_data=binary_1gram_val_ds.cache(),
              epochs=10,
              callbacks=callbacks
              )
    model = keras.models.load_model("binary_1gram.keras")
    print(f"Test acc: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")