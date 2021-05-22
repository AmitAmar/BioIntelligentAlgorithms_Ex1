import numpy as np

class ActivationFunction(object):

    @staticmethod
    def activation_function(x):
        raise NotImplementedError()
    
    @staticmethod
    def derivative_function(x):
        raise NotImplementedError()

class Sigmoid(ActivationFunction):

    @staticmethod
    def activation_function(x):
        return (1 / (1 + np.exp(-x)))

    @staticmethod
    def derivative_function(x):
        return 1 - x

class Relu(ActivationFunction):

    @staticmethod
    def activation_function(x):
        return np.maximum(x, 0)

    @staticmethod
    def derivative_function(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
