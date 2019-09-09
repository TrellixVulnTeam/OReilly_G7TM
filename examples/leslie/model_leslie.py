"""Train a single layer neural network on MNIST with two classification
outputs. The first output and loss function is the 10-class classification
that is normally used to train MNIST. The second output and loss function is
a binary classification target that predicts if the input digit is >= 5.
"""

import logging
from typing import Any, Dict, Tuple

import keras
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D 
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Sequential 
from keras.optimizers import SGD
from keras.utils.data_utils import Sequence
from mnist_utils import (
    DATA_SET_SIZE,
    download,
    extract_data,
    extract_labels,
    IMAGE_SIZE,
    NUM_CHANNELS,
    NUM_CLASSES,
    INPUT_SHAPE
)

import pedl
from pedl.frameworks.keras import KerasTrial

class MultiTaskMNistTrial(KerasTrial):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams)
        self._batch_size = hparams.get("batch_size", 1)
        self._dropout = hparams.get("dropout", 0.5)
        self._lr = hparams.get("lr", 0.001)
        self._momentum = hparams.get("momentum", 0)

    def build_model(self, hparams: Dict[str, Any]): 
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self._dropout))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation="softmax"))
        return model

    def batch_size(self) -> int:
        return self._batch_size

    def optimizer(self) -> keras.optimizers.Optimizer:
        return SGD(lr=self._lr, momentum=self._momentum)

    def loss(self) -> dict:
        return categorical_crossentropy

    def training_metrics(self) -> dict:
        return { "accuracy": categorical_accuracy } 

    def validation_metrics(self) -> dict:
        return { "accuracy": categorical_accuracy } 


class MultiTaskMNISTSequence(Sequence):  # type: ignore
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        assert len(X) == len(y)
        self._X = X
        self._y = y
        self._batch_size = batch_size

    def __len__(self) -> int:
        return len(self._X) // self._batch_size

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        start = index * self._batch_size
        end = (index + 1) * self._batch_size
        digit_labels = keras.utils.to_categorical(self._y[start:end], NUM_CLASSES)

        return (self._X[start:end], digit_labels)


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[MultiTaskMNISTSequence, MultiTaskMNISTSequence]:
    data_config = experiment_config["data"]
    base_url = data_config["base_url"]
    training_data_file = data_config["training_data"]
    training_labels_file = data_config["training_labels"]
    validation_set_size = data_config["validation_set_size"]
    
    dataset_size=data_config.get("dataset_size", DATA_SET_SIZE)
    batch_size = hparams.get("batch_size", 1) 


    if not pedl.is_valid_url(base_url):
        raise ValueError("Invalid base_url: {}".format(base_url))

    assert dataset_size > validation_set_size
    training_set_size = dataset_size - validation_set_size

    tmp_data_file = download(base_url, training_data_file)
    tmp_labels_file = download(base_url, training_labels_file)

    train_data = extract_data(tmp_data_file, dataset_size)
    train_labels = extract_labels(tmp_labels_file, dataset_size)

    # Shuffle the data.
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # Generate the training and validation sets
    validation_loader = MultiTaskMNISTSequence(
        train_data[:validation_set_size, ...], train_labels[:validation_set_size], batch_size
    )
    
    training_loader = MultiTaskMNISTSequence(
        train_data[validation_set_size:, ...], train_labels[validation_set_size:], batch_size
    )

    assert len(training_loader) == training_set_size // batch_size
    assert len(validation_loader) == validation_set_size // batch_size

    logging.info(
        "Extracted MNIST data: {} training batches, {} validation batches".format(
            training_set_size, validation_set_size
        )
    )

    return training_loader, validation_loader
