import numpy as np
from data_loader import load_mnist
from transfer_functions import identity, logistic, hyperbolic_tangent
from neuron import Neuron

class CascadeCorrelation:
    def __init__(self, input_size, output_layer_info):
        # input_size = nr of pixels in an image
        # output_layer_info = (nr output neurons, function)
        self.output_units = []
        self.no_output_units = output_layer_info[0]
        self.no_input_units = input_size
        # for now connect the outputs only to the input pixels
        for i in range(self.no_output_units):
            self.output_units.append(Neuron(input_size, output_layer_info[1]))

        self.hidden_units = []
        self.no_hidden_units = 0
        self.candidate = None

    def add_hidden_unit(self, f, w, b):

        neuron = Neuron(self.no_input_units + self.no_hidden_units, f)
        # set the input weights
        for i in range(neuron.inputs_no):
            neuron.weights[i] = w[i]
        neuron.bias = b
        # neuron.bias = 0
        # actualize ouput units to support the new hidden unit
        for neu in self.output_units:
            neu.inputs_no += 1
            neu.weights = np.append(neu.weights, 0)
            neu.g_weights =  np.append(neu.g_weights, 0)

        self.no_hidden_units += 1
        self.hidden_units.append(neuron)


    def construct_input_hidden(self, inputs):
        (n, m) = inputs.shape
        output_hidden_units = [ x.output for x in self.hidden_units ]
        inp = np.array( [ x[0] for x in inputs ] )
        inp = np.append(inp, output_hidden_units)
        inp = inp.reshape((n + len(self.hidden_units), 1))
        return inp

    def forward_outputs(self, inputs):
        # inputs = image input (700 x1)
        inp = self.construct_input_hidden(inputs)
        for o in self.output_units:
            o.forward(inp)

        return self.output_units

    def forward_candidate(self, inputs):
        pass

    def backward_outputs(self, inputs, output_neurons):
        # construct inputs format, same for all output neurons
        inp = self.construct_input_hidden(inputs)

        for x in output_neurons:
            x.backward(inp, x.output, x.a)

        # for layer_no in range(len(self.layers)-1, 0, -1):
        #     crt_layer = self.layers[layer_no]
        #     prev_layer = self.layers[layer_no - 1]
        #     crt_error = crt_layer.backward(prev_layer.outputs, crt_error)
        # self.layers[0].backward(inputs, crt_error)

    def update_parameters(self, learning_rate):
        for u in self.output_units:
            u.update_parameters(learning_rate)

    def get_trained_candidate_unit(self, inputs_p, labels_p, learning_rate):

        no_p = len(inputs_p)
        no_in = self.no_hidden_units + self.no_input_units
        
        cand = Neuron(no_in, hyperbolic_tangent)

        residual_errors = np.zeros(self.no_output_units)
        delta = np.zeros(self.no_output_units)
        sgn_delta = np.zeros(self.no_output_units)
        avg_err = np.zeros(self.no_output_units)

        for p in range(no_p):
            inp = self.construct_input_hidden(inputs_p[p])
            self.forward_outputs(inputs_p[p])          
            
            for i in range(self.no_output_units):
                residual_errors[i] = self.output_units[i].output
                residual_errors[i] += 1
            
            residual_errors[labels_p[p]] -= 2
            # delta = correlation between residual error and output unit at o

            for i in range(self.no_output_units):
                avg_err[i] += residual_errors[i]

        for i in range(self.no_output_units):
            avg_err[i] /= no_p

        corr = None
        corr_prev = None

        while corr == None or corr_prev == None or corr - corr_prev > 1:
            #print(corr)

            delta_p = 0
           
            for p in range(no_p):
                inp = self.construct_input_hidden(inputs_p[p])
                cand.forward(inp)
                self.forward_outputs(inputs_p[p])

                for i in range(self.no_output_units):
                    residual_errors[i] = self.output_units[i].output
                    residual_errors[i] += 1
                residual_errors[labels_p[p]] -= 2

                for i in range(self.no_output_units):
                    delta[i] = residual_errors[i] * cand.output
                    if delta[i] < 0:
                        sgn_delta[i] = -1
                    else:
                        sgn_delta[i] = 1

                for i in range(self.no_output_units):
                    delta_p += (residual_errors[i] - avg_err[i]) * sgn_delta[i] * cand.f(cand.a, True)
                    #print((residual_errors[i] - avg_err[i]) * sgn_delta[i] * cand.f(cand.a, True))

            # for each input weight (wither pixel or other hidden input)
            #print(delta_p)
            for i in range(no_in):
                cand.g_weights[i] = delta_p * inp[i]
            cand.g_bias = 0
            cand.update_parameters(learning_rate)

            # compute new correlation

            corr_prev = corr
            corr = 0
            for o in range(self.no_output_units):
                for p in range(no_p):
                    inp = self.construct_input_hidden(inputs_p[p])
                    self.forward_outputs(inputs_p[p])
                    cand.forward(inp)
                    
                    res_error = self.output_units[o].output
                    if o == labels_p[p]:
                        res_error -= 1
                    else:
                        res_error += 1

                    corr += abs(cand.output * (res_error - avg_err[o]))
            print(corr)
        return cand
    # def to_string(self):
    #     return " -> ".join(map(lambda l: l.to_string(), self.layers))

if __name__ == "__main__":
    data = load_mnist()
    input_size = data["train_imgs"][0].size

    cc = CascadeCorrelation(input_size, (10, identity))
    cc.forward_outputs(data["train_imgs"][0])