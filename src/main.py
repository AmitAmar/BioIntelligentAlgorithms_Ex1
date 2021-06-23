from activations_functions import ActivationFunction, Sigmoid, Relu, Softmax
import numpy as np
import os
from utils import load_dataset, create_output_dir
from ann import ANN

TRAIN_CSV_PATH = r"../data/train.csv"
VALIDATE_CSV_PATH = r"../data/validate.csv"
MODELS_DIR = "../models/"

EPOCHS = 100
START_EPOCH = 0


def main():
    # Fixes numpy's random seed
    np.random.seed(0)

    print(f"Loading train and validate datasets...")
    train_data, train_tags = load_dataset(TRAIN_CSV_PATH)
    validate_data, validate_tags = load_dataset(VALIDATE_CSV_PATH)
    print(f"Loaded datasets successfully")

    ann = ANN()
    ann.add_layer(number_of_neurons=256, activation_function=Relu, input_dim=3072)
    ann.add_layer(number_of_neurons=128, activation_function=Relu)
    ann.add_layer(number_of_neurons=10, activation_function=Softmax)

    create_output_dir(MODELS_DIR)

    print(f"Starting the ANN train process...")
    for i in range(START_EPOCH, EPOCHS + START_EPOCH):
        ann.train(train_data, train_tags, alpha=0.0005, epochs=1, noise_factor=0.8)
        acc_train = ann.evaluate(train_data, train_tags)
        acc_validate = ann.evaluate(validate_data, validate_tags)

        model_file_name = f"{i}_{acc_train * 100:.3f}_{acc_validate * 100:.3f}" + ANN.EXTENSION
        print(f"Epoch: {i}, Train accuracy: {acc_train * 100:.3f}, Validate accuracy: {acc_validate * 100:.3f}")
        ann.save(os.path.join(MODELS_DIR, model_file_name))


if __name__ == "__main__":
    main()
