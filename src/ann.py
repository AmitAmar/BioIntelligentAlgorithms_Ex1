import numpy as np
import random
from activations_functions import *


class ANN(object):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_layer_length):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_layer_length = hidden_layer_length

        self.init_weights()

    def init_weights(self):
        self.layers = list()

        # In case that there are no hidden layers, connect the input and the output layer
        if self.hidden_layers == 0:
            self.layers.append(np.random.randn(self.input_dim, self.output_dim))
            return
        else:
            self.layers.append(np.random.randn(self.input_dim, self.hidden_layer_length))

        for i in range(self.hidden_layers - 1):
            self.layers.append(np.random.randn(self.hidden_layer_length, self.hidden_layer_length))
        
        self.layers.append(np.random.randn(self.layers[-1].shape[1], self.output_dim))

        # TODO: Remove this print, its here for debug purposes
        for i in range(len(self.layers)):
            print(f"Layer {i} shape: {self.layers[i].shape}")

    def feed_forward(self, x):
        layers_outputs = list()

        # Append the input layer to the given outputs
        layers_outputs.append(x)

        # Feed forward the output of each layer with its weights and activation function
        for i in range(len(self.layers)):
            layers_outputs.append(sigmoid(layers_outputs[i].dot(self.layers[i])))
        
        return layers_outputs

    def back_propagation(self, x, y, alpha):
        layers_output = self.feed_forward(x)
        layers_error = [0] * len(self.layers)

        output_error = (layers_output[-1] - y)
        layers_error[-1] = output_error

        # len() - 1 is the already calculated output layer
        # The first output in layers output is the input, so the ith layer output is in the i+1 index
        for i in range(len(self.layers) - 2, -1, -1):
            curr_error = np.multiply(
                self.layers[i + 1].dot(layers_error[i + 1].transpose()).transpose(),
                np.multiply(layers_output[i + 1], 1 - layers_output[i + 1])
            )
            layers_error[i] = curr_error
        
        for i in range(len(self.layers)):
            adj = layers_output[i].transpose().dot(layers_error[i])
            self.layers[i] = self.layers[i] - (alpha * adj)

    def train(self, x, y, alpha=0.01, epochs=10, batch_size=8000):
        for j in range(epochs):
            batch = random.choices(list(zip(x,y)), k=batch_size)
            for i in range(len(batch)):
                self.back_propagation(batch[i][0], batch[i][1], alpha)
            print("epochs:", j + 1, "======== acc:", self.evaluate(x, y) * 100)

    def predict(self, x):
        output = self.feed_forward(x)[-1]
        return np.int8(output == output.max())

    def evaluate(self, validate_data, validate_tags):
        predictions = [self.predict(record) for record in validate_data]
        correct = len([x for x,y in zip(predictions, validate_tags) if (x==y).all()])
        return correct/ len(validate_tags)


    @staticmethod
    def loss(output, expected_output):
        s = np.square(output - expected_output)
        s = np.sum(s) / len(expected_output)
        return s

    def __str__(self):
        return str(f"Input weights: {self.input_layer_weights}\nHidden weights:{self.hidden_layers_weights}\nOutput weights:{self.output_layer_weights}")
        #return f"input shape: {self.input_layer_weights.shape}, hidden shape: {self.hidden_layers_weights.shape}, output shape: {self.output_layer_weights.shape}"
