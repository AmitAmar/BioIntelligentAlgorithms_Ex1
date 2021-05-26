from os import name
import numpy as np
import random
import pickle
from collections import namedtuple
from activations_functions import *
import math

Layer = namedtuple("Layer", ["activation_function", "weights"])


class ANN(object):
    EXTENSION = '.ann'

    def __init__(self):
        self.layers = list()

    def add_layer(self, number_of_neurons, activation_function, input_dim=None):

        if len(self.layers) == 0 and input_dim is None:
            raise ValueError("Input dim must be provided for the first layer!")
        
        prev_layer_size = input_dim
        if len(self.layers) > 0:
            prev_layer_size = self.layers[-1].weights.shape[1]
        
        self.layers.append(Layer(activation_function=activation_function,
                                 weights=activation_function.rand_distribution((prev_layer_size, number_of_neurons))))

    def feed_forward(self, x):
        layers_outputs = list()

        # Append the input layer to the given outputs
        layers_outputs.append(x)

        # Feed forward the output of each layer with its weights and activation function
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layers_outputs.append(layer.activation_function.activation_function(layers_outputs[i].dot(layer.weights)))
        
        return layers_outputs

    def back_propagation(self, x, y, alpha):
        layers_output = self.feed_forward(x)
        layers_error = [0] * len(self.layers)

        output_error = ANN.loss(layers_output[-1], y)
        layers_error[-1] = output_error

        # len() - 1 is the already calculated output layer
        # The first output in layers output is the input, so the ith layer output is in the i+1 index
        for i in range(len(self.layers) - 2, -1, -1):
            curr_error = np.multiply(
                self.layers[i + 1].weights.dot(
                    layers_error[i + 1].transpose()).transpose(),
                    np.multiply(layers_output[i + 1],
                                self.layers[i].activation_function.derivative_function(layers_output[i + 1]))
            )
            layers_error[i] = curr_error
        
        for i in range(len(self.layers)):
            adj = layers_output[i].transpose().dot(layers_error[i])
            self.layers[i] = Layer(
                activation_function=self.layers[i].activation_function,
                weights=(self.layers[i].weights - (alpha * adj)))

    @staticmethod
    def noise(x, noise_factor=0.01):
        number_of_noises = int(x.shape[1] * noise_factor)
        a = np.array([0] * number_of_noises + [1] * (x.shape[1] - number_of_noises))
        np.random.shuffle(a)
        return x * a

    def train(self, x, y, alpha=0.01, epochs=10):
        for j in range(epochs):
            for i in range(len(x)):
                self.back_propagation(ANN.noise(x[i]), y[i], alpha)

    def predict(self, x):
        output = self.feed_forward(x)[-1]
        return np.int8(output == output.max())

    def evaluate(self, validate_data, validate_tags):
        predictions = [self.predict(record) for record in validate_data]
        correct = len([x for x,y in zip(predictions, validate_tags) if (x==y).all()])
        return correct / len(validate_tags)

    def save(self, path: str):
        with open(path, "wb") as file_:
            pickle.dump(self, file_)

    @staticmethod
    def load(path):
        with open(path, "rb") as file_:
            return pickle.load(file_)

    @staticmethod
    def loss(output, expected_output):
        return output - expected_output

    def __str__(self):
        return str(self.__dict__)
