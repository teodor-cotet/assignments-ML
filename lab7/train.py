# Tudor Berariu, 2015
import numpy as np                                  # Needed to work with arrays
from argparse import ArgumentParser

import matplotlib
matplotlib.use('TkAgg')
import pylab

from data_loader import load_mnist
from feed_forward import FeedForward
from transfer_functions import identity, logistic, hyperbolic_tangent

def eval_nn(nn, imgs, labels, maximum = 0):
    # TODO (4.b)


    # Compute the confusion matrix
    confusion_matrix = np.zeros((10, 10))
    correct_no = 0
    total = np.zeros((10, 1))
    how_many = imgs.shape[0] if maximum == 0 else maximum
    for i in range(imgs.shape[0])[:how_many]:
        y = np.argmax(nn.forward(imgs[i]))
        t = labels[i]
        if y == t:
            correct_no += 1
        confusion_matrix[t][y] += 1
        total[t] += 1

    for i in range(10):
        for j in range(10):
            confusion_matrix[i][j] /= 1.0 * total[i]

    return float(correct_no) / float(how_many), confusion_matrix

def train_nn(nn, data, args):
    pylab.ion()
    cnt = 0
    for i in np.random.permutation(data["train_no"]):

        cnt += 1

        # TODO (4.a)

        error = nn.forward(data["train_imgs"][i])
        error[data["train_labels"][i]] -= 1
        nn.backward(data["train_imgs"][i], error)
        nn.update_parameters(args.learning_rate)

        # Evaluate the network
        if cnt % args.eval_every == 0:
            test_acc, test_cm = \
                eval_nn(nn, data["test_imgs"], data["test_labels"], 600)
            train_acc, train_cm = \
                eval_nn(nn, data["train_imgs"], data["train_labels"], 600)
            print("Train acc: %2.6f ; Test acc: %2.6f" % (train_acc, test_acc))
            pylab.imshow(test_cm)
            pylab.draw()

            matplotlib.pyplot.pause(0.001)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type = float, default = 0.001,
                        help="Learning rate")
    parser.add_argument("--eval_every", type = int, default = 50,
                        help="Learning rate")
    args = parser.parse_args()


    mnist = load_mnist()
    input_size = mnist["train_imgs"][0].size

    nn = FeedForward(input_size, [(300, logistic), (10, identity)])
    print(nn.to_string())

    train_nn(nn, mnist, args)
