import numpy as np
import matplotlib.pyplot as plt

class JanniksNN:

    def __init__(self, input_size=5, hidden_layers=(3,), output_size=1, neuron_fct='relu'):
        """

        :param hidden_layers: tuple with sizes of hidden layers
        :param input_size: size of input array
        :param output_size: size of output
        """

        self.layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = []
        self.weights = []
        self.neuron_fct = neuron_fct

    def initialize(self):
        self.initialize_weights()
        self.initialize_biases()

    def initialize_weights(self):
        """
        This function initializes the weights. The weights will be randomly sampled as described here:
        https://hackernoon.com/how-to-initialize-weights-in-a-neural-net-so-it-performs-well-3e9302d4490f
        (i.e. standard gaussian distribution and multiplication of weights with sqrt(2/len(input_layer)))
        :return:
        """

        self.weights = []
        if len(self.layers) == 0:  # -> no hidden layers
            self.weights.append(np.random.normal(size=(self.output_size, self.input_size))*((2/self.input_size)**0.5))
            return

        # add matrix with weights between input layer and first hidden layer
        self.weights.append(np.random.normal(size=(self.layers[0], self.input_size))*((2/self.input_size)**0.5))

        # add matrices with weights between hidden layers
        for i in range(0, len(self.layers)-1):
            self.weights.append(np.random.normal(size=(self.layers[i+1], self.layers[i]))*((2/self.layers[i])**0.5))

        # add matrix with weights between last hidden layer and output layer
        self.weights.append(np.random.normal(size=(self.output_size, self.layers[-1]))*((2/self.layers[-1])**0.5))

    def initialize_biases(self):
        """
        I have now idea about which distributions of biases is sensible so I am sampling from standard gaussian
        :return:
        """
        self.neurons = []
        # add biases for neurons in hidden layers
        for layer_size in self.layers:
            self.neurons.append(np.random.normal(size=layer_size))
        # add biases for neurons in output layer
        self.neurons.append(np.random.normal(size=self.output_size))

    def __repr__(self):
        if len(self.layers) == 0:
            return f'Layer structure: {self.input_size} -> {self.output_size}'
        else:
            out = f'Layer structure: {self.input_size} -> ['
            for layer in self.layers:
                out = out + f'{layer} -> '
            return out[0:-4] + f'] -> {self.output_size}'

    def predict(self, input_vector):
        """
        This function takes an input array of size self.input_size and calculates the output by applying the ordinary
        neural network logic: The activation of a neuron is given by sigma( sum: w_i * a_i + bias) where w_i are the
        weights of the connections from the previous layer to the respective neuron
        :param input:
        :return:
        """
        for i, weights in enumerate(self.weights):
            mat = np.concatenate([weights, np.array([self.neurons[i]]).T], axis=1)  # add bias as column to weight mat
            input_vector = np.concatenate([input_vector, [1]])  # add 1 to input vector as factor for bias
            input_vector = mat.dot(input_vector)  # calculate weighted sum with bias
            input_vector = self.activate(input_vector)  # calculate activation of neurons

        return input_vector

    def activate(self, input_vector):
        """
        This function calculates the activation of a neuron given the weighted sum with bias by applying the neuron_fct
        :param input_vector:
        :return:
        """

        if self.neuron_fct == 'relu':
            input_vector[input_vector < 0] = 0
            return input_vector

    def visualize(self):
        """
        Make visualization of the network using matplotlib.pyplot
        :return:
        """

        fig, ax = plt.subplots()
        # draw input layer neurons:
        ax.scatter(x=np.zeros(self.input_size),
                   y=np.linspace(-(self.input_size-1)/2, (self.input_size-1)/2, self.input_size),
                   facecolor='black')
        # draw hidden layer neurons:
        for i, layer in enumerate(self.neurons):
            sizes = abs(layer)/(abs(layer).max())*100
            colors = []
            for bias in layer:
                if bias >= 0:
                    colors.append('g')
                else:
                    colors.append('r')
            ax.scatter(x=np.ones(len(layer))*(i+1)*10,
                       y=np.linspace(-(len(layer)-1)/2, (len(layer)-1)/2, len(layer)),
                       s=sizes,
                       c=colors)
        # draw edges:
        for i, weights in enumerate(self.weights):
            for i_x in range(0, weights.shape[0]):
                for i_y in range(0, weights.shape[1]):
                    x = [i*10, (i+1)*10]
                    y = [-(weights.shape[1] - 1) / 2 + i_y, -(weights.shape[0] - 1) / 2 + i_x]
                    if weights[i_x, i_y] >= 0:
                        color = 'g'
                    else:
                        color = 'r'
                    ax.plot(x, y, color=color, linewidth=weights[i_x, i_y])

    def zip(self):
        """
        This function creates a one dimensional array with all weights and biases from the neural network.
        This array represents the dna of the network and can be crossed with other dna or mutated.
        The function unzip can be used to initialize the network with the data given by a dna array
        :return: dna np.array
        """

        # The first entries of the dna encode the weights between the input layer and the first hidden layer, the next
        # entries of the dna encode the biases of the first hidden layer, the next entries encode the weights between
        # the first hidden layer and the next hidden layer and so on... the last entries of dna encode the biases of the
        # output layer
        dna = np.array([])
        for idx, weights in enumerate(self.weights):
            dna = np.concatenate([dna, weights.reshape((weights.size,)), self.neurons[idx]])
        return dna

    def unzip(self, dna):
        """
        This function initializes the weights and biases of self with the values from dna
        :param dna:
        :return:
        """

        # delete current weights and biases
        self.weights = []
        self.neurons = []

        if len(self.layers) == 0:  # no hidden layers -> only one weight matrix and one bias vector
            num_el = self.input_size*self.output_size  # number of values needed for weight matrix
            self.weights.append(dna[0:num_el].reshape((self.output_size, self.input_size)))
            dna = dna[num_el:]  # reduce dna to remaining values
            num_el = self.output_size  # number of elements needed for bias vector
            self.neurons.append(dna[0:num_el])
            dna = dna[num_el:]
            if dna.size > 0:
                raise ValueError('DNA was to long for given net structure')
            return

        # unzip weights between input and first hidden layser and biases of first hidden layer
        num_el = self.input_size * self.layers[0]  # number of values needed for weight matrix
        self.weights.append(dna[0:num_el].reshape((self.layers[0], self.input_size)))
        dna = dna[num_el:]  # reduce dna to remaining values
        num_el = self.layers[0]  # number of elements needed for bias vector
        self.neurons.append(dna[0:num_el])
        dna = dna[num_el:]

        # unzip weights and biases between hidden layers
        for i in range(0, len(self.layers)-1):
            num_el = self.layers[i] * self.layers[i+1]  # number of values needed for weight matrix
            self.weights.append(dna[0:num_el].reshape((self.layers[i+1], self.layers[i])))
            dna = dna[num_el:]  # reduce dna to remaining values
            num_el = self.layers[i+1]  # number of elements needed for bias vector
            self.neurons.append(dna[0:num_el])
            dna = dna[num_el:]

        # unzip weights between last hidden layer and output and biases of output
        num_el = self.layers[-1] * self.output_size  # number of values needed for weight matrix
        self.weights.append(dna[0:num_el].reshape((self.output_size, self.layers[-1])))
        dna = dna[num_el:]  # reduce dna to remaining values
        num_el = self.output_size # number of elements needed for bias vector
        self.neurons.append(dna[0:num_el])
        dna = dna[num_el:]

        if dna.size > 0:
            raise ValueError('DNA was to long for given net structure')

        # TODO: This is a little inconvinient because I have to consider the three spacial cases of weights between
        # input and hidden, hidden and hidden, and hidden and output. To solve this I could encode the size of input and
        # output in the self.layer tuple. If I do that I have to change other functions as well (at least the
        # initialization function) so I'll keep it how it is...







