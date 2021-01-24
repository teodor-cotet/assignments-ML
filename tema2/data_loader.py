import mnist_reader
import numpy as np

def preprocess(train_imgs, test_imgs):
    avg = np.mean(train_imgs)
    dev = np.std(train_imgs)

    train_imgs -= avg
    train_imgs /=  dev
    test_imgs -= avg
    test_imgs /= dev

def load_mnist():
    #train_imgs, train_labels = mnist_data.load_training()
    #test_imgs, test_labels = mnist_data.load_testing()

    train_imgs, train_labels = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
    test_imgs, test_labels = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='t10k')

    data = {}
    data["train_imgs"] = np.array(train_imgs, dtype="f").reshape(60000, 784, 1)
    data["test_imgs"] = np.array(test_imgs, dtype="f").reshape(10000, 784, 1)
    data["train_labels"] = np.array(train_labels)
    data["test_labels"] = np.array(test_labels)
    (data["train_imgs"][0])
    preprocess(data["train_imgs"], data["test_imgs"])

    data["train_no"] = 60000
    data["test_no"] = 10000

    return data

if __name__ == "__main__":
    data = load_mnist()                                       # load the dataset
    
