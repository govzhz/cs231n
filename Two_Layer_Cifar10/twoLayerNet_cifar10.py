import numpy as np
from data_utils import load_CIFAR10
from twoLayerNet import TwoLayerNet
import matplotlib.pyplot as plt


def pre_dataset(path):
    X_train, y_train, X_test, y_test = load_CIFAR10(path)

    num_train = 49000
    num_val = 1000

    mask = range(num_train, num_train + num_val)
    X_val = X_train[mask]
    y_val = y_train[mask]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    """
    print('Train data shape: {}'.format(X_train.shape))
    print('Train labels shape: {}'.format(y_train.shape))
    print('Validation data shape: {}'.format(X_val.shape))
    print('Validation labels shape: {}'.format(y_val.shape))
    print('Test data shape: {}'.format(X_test.shape))
    print('Test labels shape: {}'.format(y_test.shape))
    """
    return X_train, y_train, X_test, y_test, X_val, y_val

def auto_get_parameters(X_train, y_train, X_val, y_val):
    "hidden layer size, learning rate, numer of training epochs, and regularization strength"
    input_size = 32 * 32 * 3
    hidden_size = 100
    num_classes = 10
    learning_rate = [5e-4, 7e-4]
    regularization_strengths = [0.01, 0.05, 0.1, 0.5, 0.7]
    learning_rate_decay = [0.95]
    iterations_per_epoch = [200, 300, 400]
    num_iters = 2000
    batch_size = 200
    best_net = None
    best_acc = -1
    best_parameters = None
    results = {}

    for lr in learning_rate:
        for rs in regularization_strengths:
            for lrd in learning_rate_decay:
                for ipe in iterations_per_epoch:
                    net = TwoLayerNet(input_size, hidden_size, num_classes, 1e-4)
                    net.train(X_train, y_train, X_val, y_val, rs, lr, lrd, num_iters, batch_size, ipe, True)
                    val_acc = np.mean(net.predict(X_val) == y_val)
                    results[(lr, rs, lrd, ipe)] = val_acc
                    
                    if best_acc < val_acc:
                        best_acc = val_acc
                        best_net = net
                        best_parameters = (lr, rs, lrd, ipe)

    for lr, reg, lrd, ipe in sorted(results):  
        val_accuracy = results[(lr, reg, lrd, ipe)]  
        print('lr %e, reg %e, lrd %e, ipe %e,val accuracy: %f' % (  
                    lr, reg, lrd, ipe, val_accuracy))
    print('best validation accuracy achieved during cross-validation: {}, which parameters is {}'.format(best_acc, best_parameters))
    
    return best_net

def visualize_loss(loss_history):
    # plot the loss history
    plt.plot(loss_history)
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.show()

def test_true():
    # acc about 29%
    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes, 1e-4)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
                0.5, 1e-4, 0.95, 1000, 200, True)

    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()
    print( 'Validation accuracy: {}'.format(val_acc))


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_val, y_val = pre_dataset('datasets/cifar-10-batches-py')
    # test_true()
    best_net = auto_get_parameters(X_train, y_train, X_val, y_val)
    test_acc = np.mean(best_net.predict(X_test) == y_test)
    print('Test accuracy: {}'.format(test_acc))

