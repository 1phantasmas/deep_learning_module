import numpy as np
from collections import defaultdict


class LayerStructure:

    def __init__(self, layer_structure, activation):
        # layer structure should look like
        # [ # of nodes for 1st layer(input size), # of nodes for 2nd layer ... , # of nodes for L-th layer]
        # so len(layer_structure) should be giving number of layers
        self.number_of_layers = len(layer_structure)
        self.weights = {i: np.random.rand(element, layer_structure[i + 1])
                        for i, element in enumerate(layer_structure[:-1])}
        self.activation = activation

    # noinspection PyMethodMayBeStatic
    def activation(self, activation):
        def sigmoid(x):
            return 1 / (1 + np.exp(x))

        def relu(x):
            return max(0, x)

        def tanh(x):
            return np.tanh(x)

        return {sigmoid: sigmoid, relu: relu, tanh: tanh}[activation]

    def train(self):
        pass

    def optimize(self):
        pass