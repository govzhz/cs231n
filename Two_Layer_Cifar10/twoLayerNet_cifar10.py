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
    "hidden layer size, learning rate, decay of learning rate, iterations_per_Annealing the learning rate, numer of training epochs, and regularization strength"
    best_net = None
    best_acc = -1
    best_parameters = None
    results = {}
    # params
    input_size = 32 * 32 * 3
    hidden_size = 80
    num_classes = 10
    learning_rate = [8e-4, 9e-4]
    reg = [0.01, 0.1]
    learning_rate_decay = [0.95, 0.97, 0.99]
    iterations_per_lr_annealing = [400, 500]  # annealing learning rate per 200 iters
    num_epoches = [15]  # a epoch is (number_of_train / batch_size)
    batch_size = [250]  # mini batch

    figure_index = 0
    for lr in learning_rate:
        for rs in reg:
            for lrd in learning_rate_decay:
                for ipla in iterations_per_lr_annealing:
                    for ne in num_epoches:
                        for bs in batch_size:
                            figure_index += 1
                            print('current params: %s, %s, %s, %s, %s, %s' % (lr, rs, lrd, ipla, ne, bs))
                            net, val_acc = visualize_net(figure_index, input_size, hidden_size, num_classes, lr, rs, lrd, ipla, ne, bs)
                            results[(lr, rs, lrd, ipla, ne, bs)] = val_acc
                            
                            if best_acc < val_acc:
                                best_acc = val_acc
                                best_net = net
                                best_parameters = (lr, rs, lrd, ipla, ne, bs)

    for lr, rs, lrd, ipla, ne, bs in sorted(results):  
        val_accuracy = results[(lr, rs, lrd, ipla, ne, bs)]  
        print('lr %e, reg %e, lrd %e, ipla %e, ne %e, bs %e val accuracy: %f' % (  
                    lr, rs, lrd, ipla, ne, bs, val_accuracy))
    print('best validation accuracy achieved during cross-validation: {}, which parameters is {}'.format(best_acc, best_parameters))
    plt.show()
    return best_net

def visualize_net(figure_index, input_size, hidden_size, num_classes, learning_rate, reg, 
        learning_rate_decay, iterations_per_lr_annealing, num_epoches, batch_size):
    # Create a network
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val, 
                reg, learning_rate, learning_rate_decay, 
                iterations_per_lr_annealing, num_epoches, 
                batch_size, False)
    
    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()
    
    # Plot the loss function and train / validation accuracies
    params = {'lr': learning_rate, 'reg': reg, 'lrd': learning_rate_decay, 
                'lrpla': iterations_per_lr_annealing, 'ne': num_epoches, 
                'bs': batch_size, 'val_acc': ("%.2f" % val_acc)}
    plt.figure(figure_index)
    plt.figtext(0.04, 0.95, params, color='green')
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')

    return net, val_acc


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_val, y_val = pre_dataset('datasets/cifar-10-batches-py')
    best_net = auto_get_parameters(X_train, y_train, X_val, y_val)
    test_acc = np.mean(best_net.predict(X_test) == y_test)
    print('Test accuracy: {}'.format(test_acc))

