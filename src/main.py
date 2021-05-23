from os.path import isdir
from activations_functions import ActivationFunction, Sigmoid, Relu, Softmax
import numpy as np
import pandas as pd
import keras
import os
import utils
from ann import ANN


TRAIN_CSV_PATH = r"../data/train.csv"
VALIDATE_CSV_PATH = r"../data/validate.csv"

EPOCHS = 100
MODELS_DIR = "../models/"


def load_data(train_csv_path: str, validate_csv_path: str):
   df_train = pd.read_csv(train_csv_path, header=None)
   df_validate = pd.read_csv(validate_csv_path, header=None)

   train_data = df_train.drop(0, axis=1).to_numpy()
   train_data = [x.reshape(1, len(train_data[0])) for x in train_data]
   train_tags = keras.utils.to_categorical(df_train[0].to_numpy() - 1)

   validate_data = df_validate.drop(0, axis=1).to_numpy()
   validate_data = [x.reshape(1, len(validate_data[0])) for x in validate_data]
   validate_tags = keras.utils.to_categorical(df_validate[0].to_numpy() - 1)

   return train_data, train_tags, validate_data, validate_tags


def main():
    # Fixes numpy's random seed
    np.random.seed(0)

    start_epoch = 0
    ann = ANN()
    ann.add_layer(number_of_neurons=400, activation_function=Sigmoid, input_dim=3072)
    ann.add_layer(number_of_neurons=400, activation_function=Sigmoid)
    ann.add_layer(number_of_neurons=10, activation_function=Sigmoid)
    print(ann)

    train_data, train_tags, validate_data, validate_tags = load_data(TRAIN_CSV_PATH, VALIDATE_CSV_PATH)

    # Creates a new ANN to be trained with the data
    #start_epoch = 0
    #ann = ANN(input_dim=3072, output_dim=10, hidden_layers=2, hidden_layer_length=400, activation_function=Relu)

    #last_model_path = os.path.join(MODELS_DIR, "99_64.67500000000001_27.0.ann")
    #start_epoch = 100
    #ann = ANN.load(last_model_path)
    #print(f"Loaded model from path: \"{last_model_path}\", starting with epoch: {start_epoch}")
    #print(ann)

    utils.create_output_dir(MODELS_DIR)

    # Trains the ANN with the dataset, save the ANN to a file after each epoch
    for i in range(start_epoch, EPOCHS + start_epoch):
        ann.train(train_data, train_tags, alpha=0.001, epochs=1)
        acc_train = ann.evaluate(train_data, train_tags)
        acc_validate = ann.evaluate(validate_data, validate_tags)

        model_file_name = f"{i}_{acc_train * 100}_{acc_validate * 100}" + ANN.EXTENSION
        ann.save(os.path.join(MODELS_DIR, model_file_name))


if __name__ == "__main__":
    main()
