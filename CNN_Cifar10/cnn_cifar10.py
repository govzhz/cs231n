from solver import Solver
from cnn import CNN
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10

def auto_get_params(data):
    N = data['X_train'].shape[0]
    num_train = np.random.choice(N, 2000, replace=True)
    small_data = {
        'X_train': data['X_train'][num_train, :],
        'y_train': data['y_train'][num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val']
    }

    best_acc = -1
    best_params = None
    results = {}

    learning_rate = [0.001, 0.002]
    lr_decay = [0.99, 0.97]
    weight_scales = [0.001, 0.002]
    reg = [0.001, 0.005]
    for lr in learning_rate:
        for ld in lr_decay:
            for ws in weight_scales:
                for rs in reg:
                    print(lr, ld, ws, rs)
                    model = CNN(weight_scales=ws, reg=rs, use_batchnorm=True)

                    solver = Solver(model, small_data, 
                                    optim_config={
                                        'learning_rate': lr
                                    },
                                    batch_size=30,
                                    iters_per_ann=400,
                                    num_epochs=1,
                                    update_rule='adam',
                                    print_every=20,
                                    verbose=True,
                                    lr_decay = ld)
    
                    solver.train()
                    val_acc = solver.check_accuracy(data['X_test'], data['y_test'])
                    results[(lr, ld, ws, rs)] = val_acc
                    if best_acc < val_acc:
                        best_acc = val_acc
                        best_params = (lr, ld, ws, rs)

    for lr, ld, ws, rs in sorted(results):  
        val_accuracy = results[(lr, ld, ws, rs)]  
        print('lr %s, ld %s, ws %s, rs %s, val accuracy: %f' % (  
                    lr, ld, ws, rs, val_accuracy))
    print('best validation accuracy achieved during cross-validation: {}, which parameters is {}'.format(best_acc, best_params))
    return best_params

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

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    """
    print('Train data shape: {}'.format(X_train.shape))
    print('Train labels shape: {}'.format(y_train.shape))
    print('Validation data shape: {}'.format(X_val.shape))
    print('Validation labels shape: {}'.format(y_val.shape))
    print('Test data shape: {}'.format(X_test.shape))
    print('Test labels shape: {}'.format(y_test.shape))
    """
    
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_val': X_val,
        'y_val': y_val
    }
    return data

def compare_batchnorm(data):
    num_train = 100
    small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
    }

    model = CNN(weight_scales=0.001, reg=0.001, use_batchnorm=False)

    solver = Solver(model, data, 
                    optim_config={
                        'learning_rate': 1e-3
                    },
                    batch_size=60,
                    iters_per_ann=400,
                    num_epochs=1,
                    update_rule='adam',
                    print_every=20,
                    verbose=True,
                    lr_decay = 1)
    
    solver.train()

    
    bn_model = CNN(weight_scales=0.001, reg=0.001, use_batchnorm=True)

    bn_solver = Solver(bn_model, data, 
                    optim_config={
                        'learning_rate': 1e-3
                    },
                    batch_size=60,
                    iters_per_ann=400,
                    num_epochs=1,
                    update_rule='adam',
                    print_every=20,
                    verbose=True,
                    lr_decay = 1)

    bn_solver.train()
    

    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 1)
    plt.plot(solver.loss_history, 'o', label='baseline')
    plt.plot(bn_solver.loss_history, 'o', label='batchnorm')

    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label='baseline')
    plt.plot(bn_solver.train_acc_history, '-o', label='batchnorm')

    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o', label='baseline')
    plt.plot(bn_solver.val_acc_history, '-o', label='batchnorm')
    
    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(15, 15)
    plt.show()

def train_model(data):
    print('Train model ---------------------------------------------')
    model = CNN(weight_scales=0.002, reg=0.001, use_batchnorm=True)

    solver = Solver(model, data, 
                    optim_config={
                        'learning_rate': 1e-3
                    },
                    batch_size=80,
                    iters_per_ann=400,
                    num_epochs=10,
                    update_rule='adam',
                    print_every=20,
                    verbose=True,
                    lr_decay = 0.97)
    
    solver.train()

    solver.visualization_model()

    test_acc = solver.check_accuracy(data['X_test'], data['y_test'])
    print('Test acc: {}'.format(test_acc))


if __name__ == '__main__':
    data = pre_dataset('datasets/cifar-10-batches-py')
    # compare_batchnorm(data)
    train_model(data)
    # auto_get_params(data)
