import os

import numpy as np


def create_output_dir(dir_path):
    """
    Creates the model's directory if it doesn't exists
    :param dir_path: directory path
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def to_categorical(vector, num_classes=10):
    vector = np.array(vector, dtype='int')
    input_shape = vector.shape

    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    vector = vector.ravel()
    rows = vector.shape[0]

    categorical = np.zeros((rows, num_classes), dtype="float32")
    categorical[np.arange(rows), vector] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical
