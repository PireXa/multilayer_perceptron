from node import Node


class Layer:
    """
    Layer class valid for input, hidden and output layers
    """

    def __init__(self, shape: tuple | int, activation: str, input_size):
        if isinstance(shape, tuple):
            self.nodes = [Node(shape, activation, input_size) for _ in range(shape[1])]
            self.shape = shape[1]
        else:
            self.nodes = [Node(shape, activation, input_size) for _ in range(shape)]
            self.shape = shape

    def __str__(self):
        # Print each one of the nodes of the layer
        return ('\n====================================LAYER====================================\n'
                + '\n'.join([str(node) for node in self.nodes]))

