# Tudor Berariu, 2016

import numpy as np

class Layer:

    def __init__(self, inputs_no, outputs_no, transfer_function):
        # Number of inputs, number of outputs, and the transfer function
        self.inputs_no = inputs_no
        self.outputs_no = outputs_no
        self.f = transfer_function

        # Layer's parameters
        self.weights = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_no + self.inputs_no)),
            (self.outputs_no, self.inputs_no)
        )
        self.biases = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_no + self.inputs_no)),
            (self.outputs_no, 1)
        )

        # Computed values
        self.a = np.zeros((self.outputs_no, 1))
        self.outputs = np.zeros((self.outputs_no, 1))

        # Gradients
        self.g_weights = np.zeros((self.outputs_no, self.inputs_no))
        self.g_biases = np.zeros((self.outputs_no, 1))


    def forward(self, inputs):
        assert(inputs.shape == (self.inputs_no, 1))

        # TODO (2.a)
        # -> compute self.a and self.outputs
        #n, m = inputs.shape # n lines, m columns
        #out_no, mm = self.a.shape

        # for j in range(out_no):
        #     self.a[j] = 0
        #     for i in range(n):
        #         self.a[j][0] += self.weights[j][i] * inputs[i][0] + self.biases[j][0]
        self.a = np.matmul(self.weights, inputs) + self.biases
        self.outputs = self.f(self.a)
        return self.outputs

    def backward(self, inputs, output_errors):
        assert(output_errors.shape == (self.outputs_no, 1))

        # TODO (2.b.i)
        # Compute the gradients w.r.t. the bias terms (self.g_biases)
        self.g_biases = output_errors

        # TODO (2.b.ii)
        # Compute the gradients w.r.t. the weights (self.g_weights)
        self.g_weights =  inputs.dot(self.g_biases.T).T
        #  self.g_weights =  a.dot(output_errors.T).T
        # TODO (2.b.iii)
        # Compute and return the gradients w.r.t the inputs of this layer
        #gradient_inputs = sel.dot(self.weights)
        #outputerror * weights
        # g_biases * weights
        #self.outputs =  np.transpose(np.matmul(np.transpose(self.g_biases), self.weights)) * self.f(inputs, True)
        return np.transpose(np.matmul(np.transpose(self.g_biases), self.weights)) * self.f(inputs, True)

    def to_string(self):
        return "(%s -> %s)" % (self.inputs_no, self.outputs_no)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from util import close_enough

def test_linear_layer():
    from transfer_functions import hyperbolic_tangent

    l = Layer(4, 5, hyperbolic_tangent)
    l.weights = np.array([[ 0.00828426,  0.35835909, -0.26848058,  0.37474081],
                          [ 0.17125137, -0.10246062,  0.301141  , -0.02042449],
                          [ 0.3111425 , -0.04866925, -0.04644496,  0.05068646],
                          [-0.36114934,  0.40810522, -0.18082862,  0.01905515],
                          [ 0.06907316, -0.1069273 , -0.35200473, -0.29067378]])
    l.biases = np.array([[-0.4146], [0.0982], [-0.3392] , [0.4674], [0.0317]])

    x = np.array([[0.123], [-0.124], [0.231], [-0.400]])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([[-0.58493574],
                       [ 0.20668163],
                       [-0.31483002],
                       [ 0.31219906],
                       [ 0.08818176]])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")

    output_err = np.array([[.001], [.001], [.99], [.001], [.001]])

    print("Testing backward computation...")

    g = l.backward(x, output_err)

    print("    i. testing gradients w.r.t. the bias terms...")
    gbias_target = np.array(
        [[ 0.001],
         [ 0.001],
         [ 0.99 ],
         [ 0.001],
         [ 0.001]]
    )

    assert (l.g_biases.shape == gbias_target.shape), "Wrong size"
    assert close_enough(l.g_biases, gbias_target), "Wrong values"
    print("     OK")

    print("   ii. testing gradients w.r.t. the weights...")
    gweights_target = np.array(
        [[1.23000000e-04, -1.24000000e-04, 2.31000000e-04, -4.00000000e-04],
         [1.23000000e-04, -1.24000000e-04, 2.31000000e-04, -4.00000000e-04],
         [1.21770000e-01, -1.22760000e-01, 2.28690000e-01, -3.96000000e-01],
         [1.23000000e-04, -1.24000000e-04, 2.31000000e-04, -4.00000000e-04],
         [1.23000000e-04, -1.24000000e-04, 2.31000000e-04, -4.00000000e-04]]
    )

    assert (l.g_weights.shape == gweights_target.shape), "Wrong size"
    assert close_enough(l.g_weights, gweights_target), "Wrong values"
    print("     OK")


    print("  iii. testing gradients w.r.t. the inputs...")
    in_target = np.array(
        [[ 0.30326003], [-0.04689319], [-0.04400043], [ 0.04222033]]
    )

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in input gradients"
    print("     OK")

    print("Backward computation implemented ok!")


if __name__ == "__main__":
    test_linear_layer()
