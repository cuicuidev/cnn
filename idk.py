import numpy as np

class Node:
    
    def __init__(self):
        self.value: float = 0

class Neuron(Node):

    def __init__(self):
        super().__init__()
        self.bias: float = np.random.randn()

class Activation(Node):

    def __init__(self):
        super().__init__()

class Edge:

    def __init__(self, starting_node: Node, target_node: Node):
        self.starting_node = starting_node
        self.target_node = target_node
        self.weight: float = 0

class Layer:

    def __init__(self):
        self.type: str = 'basic'
        self.n = 0
        self.nodes: list[Node] = []

    def init_params(self):
        pass
class NeuronLayer(Layer):

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.type: str = 'neuron'

    def init_params(self):
        for _ in range(self.n):
            self.nodes.append(Neuron())

class Dense(NeuronLayer):
    
    def __init__(self, n):
        super().__init__(n = n)

class Input(NeuronLayer):
    def __init__(self, n):
        super().__init__(n = n)

class ActivationLayer(Layer):

    def __init__(self):
        super().__init__()
        self.type: str = 'activation'
    
    def init_params(self):
        for _ in range(self.n):
            self.nodes.append(Activation())

    def activate(self):
        pass

class ReLU(ActivationLayer):
    
    def __init__(self):
        super().__init__()

    def activate(self):
        pass

class Softmax(ActivationLayer):
    def __init__(self):
        super().__init__()

    def activate(self):
        pass

class Model:

    def __init__(self):
        self.layers: list[Layer] = []
        self.edges: list[Edge] = []
    
    def add(self, layer: Layer):
        if layer.type == 'activation':
            layer.n = self.layers[-1].n
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        for layer in self.layers:
            layer.init_params()

        self._init_edges()
    
    def _init_edges(self):

        starting_nodes_layers: list[Layer] = []
        target_nodes_layers: list[Layer] = []

        for idx, layer in enumerate(self.layers):
            if idx != len(self.layers) - 1:
                starting_nodes_layers.append(layer)
            if idx != 0:
                target_nodes_layers.append(layer)
        
        for starting_nodes_layer, target_nodes_layer in zip(starting_nodes_layers, target_nodes_layers):

            if starting_nodes_layer.type == 'neuron':
                if target_nodes_layer.type == 'neuron':

                    for starting_node in starting_nodes_layer.nodes:
                        for target_node in target_nodes_layer.nodes:
                            self.edges.append(Edge(starting_node = starting_node, target_node = target_node))

                if target_nodes_layer.type == 'activation':

                    for starting_node, target_node in zip(starting_nodes_layer.nodes, target_nodes_layer.nodes):
                        self.edges.append(Edge(starting_node = starting_node, target_node = target_node))


            if starting_nodes_layer.type == 'activation':
                if target_nodes_layer.type == 'neuron':

                    for starting_node in starting_nodes_layer.nodes:
                        for target_node in target_nodes_layer.nodes:
                            self.edges.append(Edge(starting_node = starting_node, target_node = target_node))

                if target_nodes_layer.type == 'activation':

                    for starting_node, target_node in zip(starting_nodes_layer.nodes, target_nodes_layer.nodes):
                        self.edges.append(Edge(starting_node = starting_node, target_node = target_node))

    def _forward(self, X_train):
        pass

    def _back(self, y_train):
        pass

    def fit(self, X_train, y_train, epochs, batch_size, X_val = None, y_val = None):
        pass

    def predict(self, X_test):
        pass

    def evaluate(self, X_test, y_test):
        pass


model = Model()
model.add(Input(n = 784))
model.add(ReLU())
model.add(Dense(n = 128))
model.add(ReLU())
model.add(Dense(n = 32))
model.add(ReLU())
model.add(Dense(n = 10))
model.add(Softmax())

model.compile(loss = 'categorical_crossentropy', optimizer = 'gradient')