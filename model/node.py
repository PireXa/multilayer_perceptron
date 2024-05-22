import numpy as np


def he_initialization(input_size):
    """Initialize weights using He initialization."""
    # Scale factor for He initialization
    scale = np.sqrt(2.0 / input_size)
    # Draw weights from a normal distribution with mean 0 and standard deviation scale
    numbers = np.random.randn(input_size)
    #return numbers * scale
    # return np.random.normal(0, scale, input_size)
    return numbers * scale
    #return np.random.normal(0, 2, input_size)

def random_seeded(input_size):
    np.random.seed(2)
    return np.random.randn(input_size)

class Node:
    """
    Node class valid for input, hidden and output nodes
    """

    def __init__(self, shape: int | tuple, activation, input_size):
        self.output = []
        if isinstance(shape, tuple):
            self.weights = np.ones(1)
            self.weight_delta = np.zeros(1)
            self.delta = np.zeros(1)
        else:
            self.weights = he_initialization(input_size)
            self.weight_delta = np.zeros(input_size)
            #self.weights = random_seeded(input_size)
            self.delta = np.zeros(shape)
        self.activation_function = activation
        self.bias = 0
        #print(self.weights)

    def activation(self, x):
        """
        Activation function
        """
        # print(self.activation_function)
        if self.activation_function == 'relu':
            # print('x da ativacao', x)
            return np.maximum(0, x)
            # return np.where(x <= 0, 0, x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'softmax':
            print('x', x)
            exps = np.exp(x - np.max(x))
            # return exps / np.sum(exps, axis=0)
            # exps = np.exp(x)
            return exps / np.sum(exps, axis=0)
        else:
            raise ValueError('Activation function not supported')

    def activation_derivative(self, x):
        """
        Derivative of the activation function
        """
        # x = np.array(X)
        if self.activation_function == 'relu':
            # return np.where(x <= 0, 0, 1)
            return np.array([1 if e > 0 else 0 for e in x])
        elif self.activation_function == 'sigmoid':
            return x * (1 - x)
        elif self.activation_function == 'softmax':
            return x * (1 - x)
        else:
            raise ValueError('Activation function not supported')

    def __str__(self):
        return (f'--------------------Node--------------------'
                f'\nWeights: {self.weights}\nActivation function: {self.activation_function}')

