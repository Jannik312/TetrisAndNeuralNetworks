import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tetris import Tetris
from abc import ABC, abstractmethod
import pickle


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
        self.fitness = None

    def initialize(self):
        self.initialize_weights()
        self.initialize_biases()
        return self

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

        return self
        # TODO: This is a little inconvinient because I have to consider the three spacial cases of weights between
        # input and hidden, hidden and hidden, and hidden and output. To solve this I could encode the size of input and
        # output in the self.layer tuple. If I do that I have to change other functions as well (at least the
        # initialization function) so I'll keep it how it is...


class Population(ABC):

    def __init__(self, size=50):
        self.individuals = pd.DataFrame(index=range(0, size), columns=['Individual', 'Fitness'])
        self.generation = 0
        self.size = size
        self.fitness_history = []

    def generate_first_generation(self, input_size, hidden_layers, output_size, neuron_fct):
        """
        This function creates self.size many JanniksNNs with required structure and initializes them randomly
        :param input_size:
        :param hidden_layers:
        :param output_size:
        :param neuron_fct:
        :return:
        """
        for i in range(0, self.size):
            self.individuals.loc[i, 'Individual'] = \
                JanniksNN(input_size, hidden_layers, output_size, neuron_fct).initialize()

        return self

    def __repr__(self):
        return self.individuals.__repr__()

    @abstractmethod
    def fitness_function(self, neural_network):
        pass

    def evaluate_all_fitness(self, skip=True):
        """
        This function calculates the fitness of all individuals. If skip==True the function will skip the calculation
        for individuals which already have a fitness value
        :param skip:
        :return:
        """

        for idx in range(0, self.size):
            if not skip or self.individuals['Fitness'][idx] is np.nan:
                self.individuals['Fitness'][idx] = self.fitness_function(self.individuals['Individual'][idx])

    def next_generation(self, survivors=5, chance_of_mutation=0.01):
        """
        This function generates the next generation of NNs. The survivors-fittest NNs will stay alive. Further
        self.size - survivors new individuals will be generated. This is done by crossing the weights and biases of two
        individuals. The probability that one individual is selected for reproduction is proportional to its fitness
        rank (Motivated by this paper https://www.researchgate.net/publication/259461147_Selection_Methods_for_Genetic_A
        lgorithms). Which of the two individuals hands down a specific weight or bias is determined by a coin toss.
        After the reproduction the new individual is mutated.

        :param survivors: int determining the number of survivors
        :param chance_of_mutation: chance that a weight or bias from new individual gets replaced by new random value
        :return:
        """

        self.sort_by_fitness()
        # add fitness of current generation to self.fitness_history:
        self.fitness_history.append(self.individuals['Fitness'])
        # assign fittest individuals to next generation
        new_individuals = self.individuals.iloc[0:survivors, :]
        # calculate probability of individual getting selected for reproduction:
        p = np.array(self.individuals.index)
        p = self.size - p
        p = p/p.sum()
        for i in range(survivors, self.size):
            [first_parent, second_parent] = np.random.choice(self.individuals['Individual'], p=p, size=2)
            first_parent = first_parent.zip()
            second_parent = second_parent.zip()
            # set new individual as first parent and change weights and biases to values from second parent randomly
            new_individual = first_parent
            choice = np.random.choice([True, False], size=len(first_parent))
            new_individual[choice] = second_parent[choice]
            # mutate:
            mutation_array = JanniksNN(input_size=self.individuals['Individual'][0].input_size,
                                       hidden_layers=self.individuals['Individual'][0].layers,
                                       output_size=self.individuals['Individual'][0].output_size).initialize().zip()
            choice = np.random.choice([True, False], size=len(first_parent),
                                      p=[chance_of_mutation, 1-chance_of_mutation])
            new_individual[choice] = mutation_array[choice]
            new_individual = JanniksNN(input_size=self.individuals['Individual'][0].input_size,
                                       hidden_layers=self.individuals['Individual'][0].layers,
                                       output_size=self.individuals['Individual'][0].output_size).unzip(new_individual)
            new_individuals = new_individuals.append(pd.DataFrame(index=[i], columns=['Individual', 'Fitness'],
                                                                  data=[[new_individual, np.nan]]))

        self.individuals = new_individuals
        self.generation += 1

    def sort_by_fitness(self):
        self.individuals = self.individuals.sort_values(by='Fitness', ascending=False)
        self.individuals.index = range(0, self.size)
        return self

    def visualize_history(self):
        max_fitness_of_each_gen = np.array([])
        mean_fitness_of_each_gen = np.array([])
        min_fitness_of_each_gen = np.array([])
        std_of_fitness_of_each_gen = np.array([])
        for fitness_of_gen in self.fitness_history:
            max_fitness_of_each_gen = np.append(max_fitness_of_each_gen, max(fitness_of_gen))
            mean_fitness_of_each_gen = np.append(mean_fitness_of_each_gen, np.array(fitness_of_gen).mean())
            min_fitness_of_each_gen = np.append(min_fitness_of_each_gen, min(fitness_of_gen))
            std_of_fitness_of_each_gen = np.append(std_of_fitness_of_each_gen, np.array(fitness_of_gen).std())

        fig, ax = plt.subplots()
        x = range(0, self.generation)
        shades = 50
        for i in np.linspace(0, 2, num=shades):
            ax.fill_between(x, mean_fitness_of_each_gen + i*std_of_fitness_of_each_gen,
                            mean_fitness_of_each_gen - i*std_of_fitness_of_each_gen,
                            alpha=1/(2*shades), facecolor='blue')
        ax.plot(x, max_fitness_of_each_gen, color='red')
        ax.plot(x, min_fitness_of_each_gen, color='green')


class TetrisPopulation(Population):

    def generate_first_generation(self, input_size=23, hidden_layers=(20,), output_size=4, neuron_fct='relu'):
        return super().generate_first_generation(input_size=input_size, hidden_layers=hidden_layers,
                                                 output_size=output_size, neuron_fct=neuron_fct)

    def fitness_function(self, neural_network, number_of_games=5):
        """
        This function takes a neuronal network and lets it play tetris to calculate a score. This is done multiple times
        until the average score is significant for the fitness of the nn.
        :param neural_network:
        :param number_of_games:
        :return:
        """

        scores = []
        for i in range(0, number_of_games):
            # start new game:
            game = Tetris()
            # play until game over
            while not game.game_over:
                input_vector = self.generate_input_vector(game)
                nn_command = neural_network.predict(input_vector)
                game.make_nn_move(nn_command)

            scores.append(game.score)

        return np.array(scores).mean()

    @staticmethod
    def generate_input_vector(game):
        # this vector specifies how high each column is filled in %:
        board_filling = (20 - np.argmax(game.board, axis=0))/20
        # this vector specifies which type the current tile has using 'one hot encoding'
        tile_type = (np.array(['T', 'I', 'O', 'L1', 'L2', 'S1', 'S2']) == game.tile.type) * 1.0
        # this vector specifies the orientation of the tile using 'one hot encoding'
        tile_rotation = (np.array([0, 1, 2, 3]) == game.tile.orientation) * 1.0
        # this vector specifies the center of tile by relative position in x and y direction
        tile_positon = game.tile.center / np.array([20, 10])
        # together all those vectors create the vector that will be the input of the neural net.
        input_vector = np.concatenate([board_filling, tile_type, tile_rotation, tile_positon])
        return input_vector


def train_tetris_NN(pop=None, max_iter=None):

    if pop is None:
        pop = TetrisPopulation(size=100).generate_first_generation(hidden_layers=(100, 100))

    i = 1
    while True:
        if max_iter is not None and i > max_iter:
            break
        pop.evaluate_all_fitness(skip=False)
        pop.next_generation(chance_of_mutation=0.05, survivors=5)
        best_nn = pickle.load(open("Best_NN.pickle", "rb"))
        print(f'Best Fitness of generateion {pop.generation}: {pop.individuals["Fitness"][0]}')
        if best_nn.fitness < pop.individuals['Fitness'][0]:
            new_best_nn = pop.individuals['Individual'][0]
            new_best_nn.fitness = pop.individuals['Fitness'][0]
            pickle.dump(new_best_nn, open("Best_NN.pickle", "wb"))
        print(f'Best Fitness overall: {best_nn.fitness}')
        i += 1

    return pop








