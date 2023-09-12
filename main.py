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

input_ = [0.4, 0.6, 1, 0.1, -1, -0.33, 0, 0]
input_layer = Dense(n_neurons = 8)
dense_1 = Dense(n_neurons = 128)
dense_2 = Dense(n_neurons = 128)
output = Dense(n_neurons = 10)

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

def propagate_to_layer(first_layer: Layer, second_layer: Layer, links: Link):
    for neuron2, link_row in zip(second_layer.neurons, links):
        value = propagate_to_neuron([x.value for x in first_layer.neurons], [x.weight for x in link_row])

        neuron2.value = value

def forward_propagation():
    pass

if __name__ == '__main__':

    for val, neur in zip(input_, input_layer.neurons):
        neur.value = val

    links_1 = generate_links(first_layer=input_layer, second_layer=dense_1)
    links_2 = generate_links(first_layer=dense_1, second_layer=dense_2)
    links_3 = generate_links(first_layer=dense_2, second_layer=output)

    propagate_to_layer(first_layer=input_layer, second_layer=dense_1, links = links_1)
    propagate_to_layer(first_layer=dense_1, second_layer=dense_2, links = links_2)
    propagate_to_layer(first_layer=dense_2, second_layer=output, links = links_3)

    result = [x.value for x in output.neurons]
    print(result)