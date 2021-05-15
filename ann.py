import numpy as np


class ANN(object):

    def __init__(self, input_dim, output_dim, hidden_layers, hidden_layer_length):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_layer_length = hidden_layer_length

        self.init_weights()

    @staticmethod
    def relu(x):
        return max(0, x)
    
    @staticmethod
    def sigmoid(x):
        return(1/(1 + np.exp(-x)))

    def init_weights(self):
        self.input_layer_weights = np.random.rand(self.input_dim, self.hidden_layer_length)
        self.output_layer_weights = np.random.rand(self.hidden_layer_length, self.output_dim)
        self.hidden_layers_weights = np.random.rand(self.hidden_layers - 1, self.hidden_layer_length, self.hidden_layer_length)

    def feed_forward(self, x):
        
        layers_outputs = list()

        # Handle the input layer
        input_layer_output = x.dot(self.input_layer_weights)
        input_layer_output = ANN.sigmoid(input_layer_output)
        layers_outputs.append(input_layer_output)

        curr_input = input_layer_output

        # Iterate through the hidden layers and feed forward
        for i in range(len(self.hidden_layers_weights)):
            curr_output = curr_input.dot(self.hidden_layers_weights[i])
            curr_output = ANN.sigmoid(curr_output)

            layers_outputs.append(curr_output)

            curr_input = curr_output
        
        # Handle the output layer
        output = curr_input.dot(self.output_layer_weights)
        output = ANN.sigmoid(output)

        layers_outputs.append(output)

        return layers_outputs

    def loss(output, expected_output):
        s = np.square(output - expected_output)
        s = np.sum(s) / len(expected_output)
        return s

    def back_propogation(x, y, alpha):
        pass

    def train(self, x, y, alpha=0.01, epochs=10):
        pass

    def __str__(self):
        return str(f"Input weights: {self.input_layer_weights}\nHidden weights:{self.hidden_layers_weights}\nOutput weights:{self.output_layer_weights}")
        #return f"input shape: {self.input_layer_weights.shape}, hidden shape: {self.hidden_layers_weights.shape}, output shape: {self.output_layer_weights.shape}"
