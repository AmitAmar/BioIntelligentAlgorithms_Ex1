import numpy as np
import pandas as pd
import keras
from ann import ANN


TRAIN_CSV_PATH = r"../data/train.csv"
VALIDATE_CSV_PATH = r"../data/validate.csv"


def load_data(train_csv_path: str, validate_csv_path: str) -> list:
   df_train = pd.read_csv(train_csv_path, header=None)
   df_validate = pd.read_csv(validate_csv_path, header=None)

   train_data = df_train.drop(0, axis=1).to_numpy()
   train_data = [x.reshape(1, len(train_data[0])) for x in train_data]
   train_tags = keras.utils.to_categorical(df_train[0].to_numpy() - 1)

   validate_data = df_validate.drop(0, axis=1).to_numpy().transpose()
   validate_data = [x.reshape(1, len(validate_data[0])) for x in validate_data]
   validate_tags = keras.utils.to_categorical(df_validate[0].to_numpy() - 1)

   return train_data, train_tags, validate_data, validate_tags

def main():
    # Fixes numpy's random seed
    np.random.seed(0)

    train_data, train_tags, validate_data, validate_tags = load_data(TRAIN_CSV_PATH, VALIDATE_CSV_PATH)

    # Creates a new ANN to be trained with the data
    ann = ANN(input_dim=3072, output_dim=10, hidden_layers=2, hidden_layer_length=400)
    
    # Trains the ANN with the dataset
    ann.train(train_data, train_tags, alpha=0.1, epochs=1)


if __name__ == "__main__":
    main()
