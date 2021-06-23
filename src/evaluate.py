import sys
import pandas as pd

from ann import ANN, Layer
from activations_functions import Relu
from utils import load_dataset


def main():
    if len(sys.argv) != 3:
        print(f"USAGE {sys.argv[0]} <model_path> <dataset_path>")
        return

    model_path = sys.argv[1]
    dataset_path = sys.argv[2]

    ann = ANN.load(model_path)
    data, tags = load_dataset(dataset_path)

    print(ann.evaluate(data, tags))


if __name__ == "__main__":
    main()
