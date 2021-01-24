
import numpy as np                                  # Needed to work with arrays
from argparse import ArgumentParser

import matplotlib
matplotlib.use('TkAgg')
import pylab

from data_loader import load_mnist
from cascade_correlation import CascadeCorrelation
from transfer_functions import identity, logistic, hyperbolic_tangent

def eval_nn(nn, imgs, labels, maximum = 0):
    # Compute the confusion matrix
    confusion_matrix = np.zeros((10, 10))
    correct_no = 0
    total = np.zeros((10, 1))
    how_many = imgs.shape[0] if maximum == 0 else maximum
    for i in range(imgs.shape[0])[:how_many]:
        y = np.argmax([ x.output for x in nn.forward_outputs(imgs[i]) ])
        t = labels[i]
        if y == t:
            correct_no += 1
        confusion_matrix[t][y] += 1
        total[t] += 1

    for i in range(10):
        for j in range(10):
            confusion_matrix[i][j] /= 1.0 * total[i]

    return float(correct_no) / float(how_many), confusion_matrix

def train_nn(cc, data, args):
    pylab.ion()
    cnt = 0
    many = 1500
    neurons_to_add = 100
    data["train_no"] = many
    for j in np.random.permutation(neurons_to_add):

        cnt += 1
        #cc.add_hidden_unit(identity, np.random.normal(0, 0.1, cc.no_input_units + cc.no_hidden_units), 1)
        new_hidden = cc.get_trained_candidate_unit(data["train_imgs"][:many], data["train_labels"][:many], args.learning_rate)
        cc.add_hidden_unit(new_hidden.f, new_hidden.weights, 0)
        # TODO (4.a)
        train_prev = None
        train_curr = None
        count_train_episodes = 0
        while  train_prev == None or train_curr == None or train_curr - train_prev > 0 or count_train_episodes < 60:

        	i = np.random.permutation(data["train_no"])[0]
	        output_neurons = cc.forward_outputs(data["train_imgs"][i])
	        output_neurons[data["train_labels"][i]].output -= 5
	        err = [x.output for x in output_neurons]
	        #print(error_v)
	        # inputs = pixels + hidden_units
	        inputs = data["train_imgs"][i]

	        cc.backward_outputs(inputs, output_neurons)
	        cc.update_parameters(args.learning_rate)
	        count_train_episodes += 1
	        if count_train_episodes % args.eval_every == 0:
	            test_acc, test_cm = \
	                eval_nn(cc, data["test_imgs"], data["test_labels"], many)
	            train_acc, train_cm = \
	                eval_nn(cc, data["train_imgs"], data["train_labels"], many)
	            print("Train acc: %2.6f ; Test acc: %2.6f" % (train_acc, test_acc))

	            train_prev = train_curr
	            train_curr = train_acc
	            
	            pylab.imshow(test_cm)
	            pylab.draw()

	            matplotlib.pyplot.pause(0.001)



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type = float, default = 0.001,
                        help="Learning rate")
    parser.add_argument("--eval_every", type = int, default = 10,
                        help="Learning rate")
    args = parser.parse_args()

    data = load_mnist()
    input_size = data["train_imgs"][0].size

    cc = CascadeCorrelation(input_size, (10, identity))
    #cc.add_hidden_unit(identity, np.random.normal(0, 0.1, cc.no_input_units + cc.no_hidden_units), 1)
    #cc.forward_outputs(data["train_imgs"][0])
    train_nn(cc, data, args)