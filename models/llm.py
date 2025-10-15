import re

import pytest

import string

class Vectorizer:
    """

    """
    def __init__(self):
        self.vocabulary = {"": 0, "[UNK]": 1}

    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    def tokenize(self, text: str):
        return re.findall(r'\b\w+\b', text)

    def make_vocabulary(self, dataset):
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = dict((v, k) for k, v in self.vocabulary.items())

    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return " ".join(
            self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence
        )

dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
    "On sweet plum blossoms",
    "The sun rises suddenly.",
    "Look, a mountain path I"
]

@pytest.mark.parametrize("s", [dataset])
def test_tokenization(s):
    vectorizer = Vectorizer()
    vectorizer.make_vocabulary(dataset)
    test_sentence = "I write, rewrite, and still rewrite again"
    encoded_sentence = vectorizer.encode(test_sentence)
    print(encoded_sentence)
    decoded_sentence = vectorizer.decode(encoded_sentence)
    print(decoded_sentence)

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