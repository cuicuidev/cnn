import random

class Neuron:

    def __init__(self, bias: float):
        self.bias = bias
        self.value = 0

class Link:

    def __init__(self, weight: float):
        self.weight = weight

class Layer:
    pass

class Dense(Layer):

    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.neurons = [Neuron(bias = random.random()) for x in range(n_neurons)]

class Input(Layer):

    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.neurons = [Neuron(bias = 0) for x in range(n_neurons)]
    
    def feed_data(self, data):
        if len(data) != self.n_neurons:
            raise Exception(f'Input size of `{len(data)}` does not match specified size of `{self.n_neurons}`')
        for neuron, value in zip(self.neurons, data):
            neuron.value = value

def propagate_to_neuron(input_, weights):
    result = 0
    for node, weight in zip(input_, weights):
        result += node * weight
    return result

def generate_links(first_layer: Layer, second_layer: Layer):
    link_matrix = []
    for neuron2 in second_layer.neurons:
        link_list = []
        for neuron1 in first_layer.neurons:
            link = Link(weight = random.random())
            link_list.append(link)
        link_matrix.append(link_list)
    return link_matrix

def propagate_to_layer(first_layer: Layer, second_layer: Layer, links: Link, activation):
    z = []
    for link_row in links:
        value = propagate_to_neuron([x.value for x in first_layer.neurons], [x.weight for x in link_row])
        z.append(value)
    z = activation(z)

    for neuron2, value in zip(second_layer.neurons, z):
        neuron2.value = value

def forward_propagation():
    pass

def relu(z):
    return [max([0, value]) for value in z]

def softmax(z):
    e_z = [exp_val - max(z) for exp_val in z]
    e_z = [pow(2.71828, exp_val) for exp_val in e_z]
    sum_e_z = sum(e_z)
    softmax_probs = [exp_val / sum_e_z for exp_val in e_z]
    return softmax_probs

if __name__ == '__main__':

    input_ = [random.random() for _ in range(784)]# [0.4, 0.6, 1, 0.1, -1, -0.33, 0, 0]
    input_layer = Input(n_neurons = 28*28)
    dense_1 = Dense(n_neurons = 128)
    dense_2 = Dense(n_neurons = 128)
    output = Dense(n_neurons = 10)

    input_layer.feed_data(input_)

    links_1 = generate_links(first_layer=input_layer, second_layer=dense_1)
    links_2 = generate_links(first_layer=dense_1, second_layer=dense_2)
    links_3 = generate_links(first_layer=dense_2, second_layer=output)

    propagate_to_layer(first_layer=input_layer, second_layer=dense_1, links = links_1, activation = relu)
    propagate_to_layer(first_layer=dense_1, second_layer=dense_2, links = links_2, activation = relu)
    propagate_to_layer(first_layer=dense_2, second_layer=output, links = links_3, activation = softmax)

    result = [x.value for x in output.neurons]
    print(result)