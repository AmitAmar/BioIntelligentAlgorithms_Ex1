import os
import numpy as np
import pandas as pd


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


def from_categorical(vector: np.ndarray) -> int:
    try:
        return vector.tolist()[0].index(1)
    except ValueError:
        import ipdb; ipdb.set_trace()


def normalize_data(data):
    new_data = list()
    for i in range(len(data)):
        record = data[i]
        record = (record - record.mean()) / record.std()
        new_data.append(record)
    return new_data


def load_dataset(dataset_path: str):
    df = pd.read_csv(dataset_path, header=None)

    data = df.drop(0, axis=1).to_numpy()
    data = [x.reshape(1, len(data[0])) for x in data]
    data = normalize_data(data)

    try:
        tags = to_categorical(df[0].to_numpy() - 1)
    except ValueError:
        tags = to_categorical(np.array([0] * len(data)))

    return data, tags
