import numpy as np

from layer_interface import LayerInterface

class MaxPoolingLayer(LayerInterface):

    def __init__(self, stride):
        # Dimensions: stride
        self.stride = stride

        # indexes of max activations

    def forward(self, inputs):

        # TODO 3
        f = lambda x : (x % self.stride == 0)
        d, l, c = inputs.shape
        lines = list(filter(f, range(l)))
        cols = list(filter(f, range(c)))
        self.outputs = np.zeros((d, len(lines), len(cols)))
        self.switches = np.zeros((d, len(lines), len(cols), 2))

        for k in range(d):
            for i in lines:
                for j in cols:
                    res = -np.inf
                    p1 = None
                    p2 = None
                    pos1 = i//self.stride
                    pos2 = j//self.stride
                    for a in range(self.stride):
                        for b in range(self.stride):
                            if res < inputs[k][i + a][j + b]:
                                p1 = i + a
                                p2 = j + b
                                res = max(res, inputs[k][i + a][j + b])
                    
                    self.switches[k][pos1][pos2][0] = p1
                    self.switches[k][pos1][pos2][1] = p2
                    self.outputs[k][pos1][pos2] = res

        self.outputs = np.array(self.outputs)

        return self.outputs

    def backward(self, inputs, output_errors):

        # TODO 3
        f = lambda x : (x % self.stride == 0)

        d, l, c = inputs.shape
        lines = list(filter(f, range(l)))
        cols = list(filter(f, range(c)))
        result = np.zeros((d, l, c))

        for k in range(d):
            for i in lines:
                for j in cols:
                    # maximum = -np.inf
                    # pos = (0, 0)
                    # for a in range(self.stride):
                    #     for b in range(self.stride):
                    #         if maximum < inputs[k][i + a][j + b]:
                    #             maximum = inputs[k][i + a][j + b]
                    #             pos = (i + a, j + b)
                    pos = (int(self.switches[k][i//self.stride][j//self.stride][0]), int(self.switches[k][i//self.stride][j//self.stride][1]))
                    result[k][pos[0]][pos[1]] = output_errors[k][pos[0]//self.stride][pos[1] // self.stride]

        return np.array(result)

    def to_string(self):
        return "[MP (%s x %s)]" % (self.stride, self.stride)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from util import close_enough

def test_max_pooling_layer():

    l = MaxPoolingLayer(2)

    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]],
                  [[9, 10, 11, 12], [13, 14, 15, 16]],
                  [[17, 18, 19, 20], [21, 22, 23, 24]]])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([[[6, 8]],
                       [[14, 16]],
                       [[22, 24]]])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")


    output_err = output

    print("Testing backward computation...")

    g = l.backward(x, output_err)
    print(g)


    print("Testing gradients")
    in_target = np.array([[[0, 0, 0, 0], [0, 6, 0, 8]],
                          [[0, 0, 0, 0], [0, 14, 0, 16]],
                          [[0, 0, 0, 0], [0, 22, 0, 24]]])

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in gradients"
    print("     OK")

    print("Backward computation implemented ok!")


if __name__ == "__main__":
    test_max_pooling_layer()
