import numpy as np

from layer_interface import LayerInterface

class LinearizeLayer(LayerInterface):

    def __init__(self, depth, height, width):
        # Dimensions: depth, height, width
        self.depth = depth
        self.height = height
        self.width = width


    def forward(self, inputs):
        assert(inputs.shape == (self.depth, self.height, self.width))

        # TODO 1
        # Reshape inputs- transform volume to column
        (M, H, W) = inputs.shape
        self.outputs = np.zeros( (M * H * W, 1))
        for m in range(M):
            for h in range(H):
                for w in range(W):
                    self.outputs[m * H * W + h * W + w ][0] = inputs[m][h][w]
        return self.outputs

    def backward(self, inputs, output_errors):
        # unused argument - inputs
        assert(output_errors.shape == (self.depth * self.height * self.width, 1))

        # TODO 1
        # Reshape gradients - transform column to volume
        (M, H, W) = inputs.shape
        for m in range(M):
            for h in range(H):
                for w in range(W):
                    inputs[m][h][w] = output_errors[m * H * W + h * W + w ][0]
        return inputs

    def to_string(self):
        return "[Lin ((%s, %s, %s) -> %s)]" % (self.depth, self.height, self.width, self.depth * self.height * self.width)


class LinearizeLayerReverse(LayerInterface):

    def __init__(self, depth, height, width):
        # Dimensions: depth, height, width
        self.depth = depth
        self.height = height
        self.width = width


    def forward(self, inputs):
        assert(inputs.shape == (self.depth * self.height * self.width, 1))

        # TODO 1
        # Reshape inputs - transform column to volume
        M = self.depth
        H = self.height
        W = self.width
        self.outputs = np.zeros((M, H, W))
        for m in range(M):
            for h in range(H):
                for w in range(W):
                    self.outputs[m][h][w] = inputs[m * H * W + h * W + w ][0]

        return self.outputs

    def backward(self, inputs, output_errors):
        # unused argument - inputs
        assert(output_errors.shape == (self.depth, self.height, self.width))

        # TODO 1
        # Reshape gradients - transform volume to column
        (M, H, W) = output_errors.shape
        for m in range(M):
            for h in range(H):
                for w in range(W):
                    inputs[m * H * W + h * W + w ][0] = output_errors[m][h][w]
        return inputs

    def to_string(self):
        return "[Lin (%s -> (%s, %s, %s))]" % (self.depth * self.height * self.width, self.depth, self.height, self.width)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from util import close_enough

def test_linearize_layer():

    l = LinearizeLayer(2, 3, 4)

    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24]])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")

    output_err = output

    print("Testing backward computation...")

    g = l.backward(x, output_err)

    print("Testing gradients")
    in_target = x

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in gradients"
    print("     OK")

    print("Backward computation implemented ok!")


def test_linearize_layer_reverse():

    l = LinearizeLayerReverse(2, 3, 4)

    x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24]])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")

    output_err = output

    print("Testing backward computation...")

    g = l.backward(x, output_err)

    print("Testing gradients")
    in_target = x

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in gradients"
    print("     OK")

    print("Backward computation implemented ok!")


if __name__ == "__main__":
    test_linearize_layer()
    test_linearize_layer_reverse()