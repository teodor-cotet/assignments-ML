import numpy
import random
from PIL import Image

class HopfieldNetwork:
    CHAR_TO_INT = { "_": -1, "X": 1 }
    INT_TO_CHAR = { -1: "_", 1: "X" }

    # Initalize a Hopfield Network with N neurons
    def __init__(self, neurons_no):
        self.neurons_no = neurons_no
        self.state = numpy.ones((self.neurons_no), dtype=int)
        self.weights = numpy.zeros((self.neurons_no, self.neurons_no))

# ------------------------------------------------------------------------------

    # Learn some patterns
    def learn_patterns(self, patterns, learning_rate):
        # TASK 1
        #nr_patterns = len(patterns)
        #s = numpy.zeros((nr_patterns, self.neurons_no))
        #for i in range(nr_patterns):
        #    for j in range(self.neurons_no):
        #        s[i][j] = self.CHAR_TO_INT[patterns[i][j]]
        #for i in range(nr_patterns):
        #    t = s[i].reshape( (1, self.neurons_no) )
        #    res = numpy.transpose(t) * t
        #    self.weights = self.weights  + res - nr_patterns * numpy.identity(self.neurons_no)
        for i in range(self.neurons_no):
            for j in range(i + 1, self.neurons_no):
                for k in range(len(patterns)):
                    self.weights[i][j] += HopfieldNetwork.CHAR_TO_INT[patterns[k][i]] * HopfieldNetwork.CHAR_TO_INT[patterns[k][j]]
                    self.weights[j][i] = self.weights[i][j]
        self.weights /= len(patterns)
        # print(self.weights)
    # Compute the energy of the current configuration

    def unlearn(self, patterns, learning_rate):
        # TASK 1
        #nr_patterns = len(patterns)
        #s = numpy.zeros((nr_patterns, self.neurons_no))
        #for i in range(nr_patterns):
        #    for j in range(self.neurons_no):
        #        s[i][j] = self.CHAR_TO_INT[patterns[i][j]]
        #for i in range(nr_patterns):
        #    t = s[i].reshape( (1, self.neurons_no) )
        #    res = numpy.transpose(t) * t
        #    self.weights = self.weights  + res - nr_patterns * numpy.identity(self.neurons_no)
        w = numpy.zeros((self.neurons_no, self.neurons_no))

        for i in range(self.neurons_no):
            for j in range(i + 1, self.neurons_no):
                for k in range(len(patterns)):
                    w[i][j] += HopfieldNetwork.CHAR_TO_INT[patterns[k][i]] * HopfieldNetwork.CHAR_TO_INT[patterns[k][j]]
                    w[j][i] = self.weights[i][j]
            w[i][j] *= learning_rate
        w /= len(patterns)
        return w

    def energy(self):
        # TASK 1:
        en = .0
        nn = self.neurons_no

        for i in range(nn):
            for j in range(nn):
                en +=  self.state[i] * self.weights[i][j] * self.state[j]
        en = en * (-0.5)
        return en

    # Update a single random neuron
    def single_update(self):
        # TASK 1
        neuro = random.randint(0, self.neurons_no - 1)
        n = self.neurons_no
        s = 0.0
        for i in range(n):
            s += self.weights[neuro][i] * self.state[i]

        if s < 0:
            self.state[neuro] = -1
        else:
            self.state[neuro] = 1

    # Check if energy is minimal
    def is_energy_minimal(self):
        
        nn = self.neurons_no
        for i in range(nn):
            s = .0
            for j in range(nn):
                s += self.weights[i][j] * self.state[j]

            res_s = -1 if s < 0 else 1

            if res_s != self.state[i]:
                return False

        return True

    # --------------------------------------------------------------------------

    # Approximate the distribution of final states
    # starting from @samples_no random states.
    def get_final_states_distribution(self, samples_no=1000):
        # TASK 3
        final_states = {}

        for _ in range(samples_no):
            self.random_reset()

            while not self.is_energy_minimal():
                self.single_update()

            if HopfieldNetwork.state_to_string(self.state) not in final_states.keys():
                final_states[HopfieldNetwork.state_to_string(self.state)] = 1
            else:
                final_states[HopfieldNetwork.state_to_string(self.state)] += 1

        for k in final_states:
            final_states[k] = 1.0 * final_states[k] /  samples_no

        return final_states

    # -------------------------------------------------------------------------


    # Unlearn some patterns
    def unlearn_patterns(self, patterns, learning_rate):
        # TASK BONUS

        final_states = {}
        print(patterns)

        for _ in range(samples_no):
            self.random_reset()

            while not self.is_energy_minimal():
                self.single_update()

            if HopfieldNetwork.state_to_string(self.state) not in patterns:
                print('modify states')
                ww = self.unlearn(patterns, learning_rate)
                self.weights -= ww

    # -------------------------------------------------------------------------


    # Get the pattern of the state as string
    def get_pattern(self):
        return "".join([HopfieldNetwork.INT_TO_CHAR[n] for n in self.state])

    # Reset the state of the Hopfield Network to a given pattern
    def reset(self, pattern):
        assert(len(pattern) == self.neurons_no)
        for i in range(self.neurons_no):
            self.state[i] = HopfieldNetwork.CHAR_TO_INT[pattern[i]]

    # Reset the state of the Hopfield Network to a random pattern
    def random_reset(self):
        for i in range(self.neurons_no):
            self.state[i] = 1 - 2* numpy.random.randint(0, 2)

    def to_string(self):
        return HopfieldNetwork.state_to_string(self.state)

    @staticmethod
    def state_to_string(state):
        return "".join([HopfieldNetwork.INT_TO_CHAR[c] for c in state])

    @staticmethod
    def state_from_string(str_state):
        return numpy.array([HopfieldNetwork.CHAR_TO_INT[c] for c in str_state])

    # display the current state of the HopfieldNetwork
    def display_as_matrix(self, rows_no, cols_no):
        assert(rows_no * cols_no == self.neurons_no)
        HopfieldNetwork.display_state_as_matrix(self.state, rows_no, cols_no)

    # display the current state of the HopfieldNetwork
    def display_as_image(self, rows_no, cols_no):
        assert(rows_no * cols_no == self.neurons_no)
        pixels = [1 if s > 0 else 0 for s in self.state]
        img = Image.new('1', (rows_no, cols_no))
        img.putdata(pixels)
        img.show()

    @staticmethod
    def display_state_as_matrix(state, rows_no, cols_no):
        assert(state.size == rows_no * cols_no)
        print("")
        for i in range(rows_no):
            print("".join([HopfieldNetwork.INT_TO_CHAR[state[i*cols_no+j]]
                           for j in range(cols_no)]))
        print("")
