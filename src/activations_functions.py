import numpy as np

np.random.seed(0)


class ActivationFunction(object):

    @staticmethod
    def rand_distribution(shape):
        return np.random.randn(*shape) * np.sqrt(1. / shape[0])

    @staticmethod
    def activation_function(x):
        raise NotImplementedError()

    @staticmethod
    def derivative_function(x):
        raise NotImplementedError()


class Sigmoid(ActivationFunction):

    @staticmethod
    def activation_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative_function(x):
        return 1 - x


class Relu(ActivationFunction):

    @staticmethod
    def activation_function(x):
        return np.maximum(x, 0)

    @staticmethod
    def derivative_function(x):
        return np.where(x <= 0, 0, 1)


class Softmax(ActivationFunction):

    @staticmethod
    def activation_function(x):
        exps = np.exp(x)
        return exps / exps.sum()

    @staticmethod
    def derivative_function(x):
        raise NotImplementedError("Using Softmax in an intermediate layer isn't supported yet")
