import numpy as np


class ActivationFunction(object):

    @staticmethod
    def rand_distribution(shape):
        raise NotImplementedError()

    @staticmethod
    def activation_function(x):
        raise NotImplementedError()
    
    @staticmethod
    def derivative_function(x):
        raise NotImplementedError()


class Sigmoid(ActivationFunction):

    @staticmethod
    def rand_distribution(shape):
        return np.random.randn(*shape)

    @staticmethod
    def activation_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative_function(x):
        return 1 - x


class Relu(ActivationFunction):

    MIN_INIT_WEIGHT = -0.01
    MAX_INIT_WEIGHT = 0.01

    @staticmethod
    def rand_distribution(shape):
        return np.random.uniform(Relu.MIN_INIT_WEIGHT, Relu.MAX_INIT_WEIGHT, shape)

    @staticmethod
    def activation_function(x):
        return np.maximum(x, 0)

    @staticmethod
    def derivative_function(x):
        return np.where(x <= 0, 0, 1)


class Softmax(ActivationFunction):

    MIN_INIT_WEIGHT = -0.01
    MAX_INIT_WEIGHT = 0.01

    # @staticmethod
    # def rand_distribution(shape):
    #     return np.random.uniform(Softmax.MIN_INIT_WEIGHT, Softmax.MAX_INIT_WEIGHT, shape)

    @staticmethod
    def rand_distribution(shape):
        return np.random.randn(*shape)

    @staticmethod
    def activation_function(x):
        exps = np.exp(x)
        return exps / exps.sum()

    @staticmethod
    def derivative_function(x):
        raise NotImplementedError("Using Softmax in an intermediate layer isn't supported yet")
