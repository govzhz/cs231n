from fullyConnectedNet import FullyConnectedNet
from solver import Solver
from data_utils import load_CIFAR10
import numpy as np
import matplotlib.pyplot as plt

def auto_get_params(data):
    input_dims = 32 * 32 * 3
    hidden_dims = [100, 100, 100, 100, 100]
    num_classes = 10
    N = data['X_train'].shape[0]
    num_train = np.random.choice(N, 5000, replace=True)
    small_data = {
        'X_train': data['X_train'][num_train, :],
        'y_train': data['y_train'][num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val']
    }
    weight_scale = [2e-2]
    batch_size = [250]
    iters_per_ann = [400]
    num_epochs = 10
    update_rule = 'adam'
    lr_decay = [0.95]
    dropout = [0.9]
    learning_rate = [9e-4]

    best_params = None
    best_acc = -1
    results = {}

    for ws in weight_scale:
        for bs in batch_size:
            for ipa in iters_per_ann:
                for ld in lr_decay:
                    for dp in dropout:
                        for lr in learning_rate:
                            print(ws, bs, ipa, ld, dp, lr)
                            model = FullyConnectedNet(input_dims, hidden_dims, num_classes, weight_scale=ws, use_batchnorm=True, dropout=dp)

                            solver = Solver(model, small_data, 
                                            optim_config={
                                                'learning_rate': lr
                                            },
                                            batch_size=bs,
                                            iters_per_ann=ipa,
                                            num_epochs=num_epochs,
                                            update_rule=update_rule,
                                            print_every=200,
                                            verbose=True,
                                            lr_decay = ld)
                            
                            solver.train()

                            val_acc = solver.check_accuracy(small_data['X_val'], small_data['y_val'])
                            results[(ws, bs, ipa, ld, dp, lr)] = val_acc
                            if val_acc > best_acc:
                                best_acc = val_acc
                                best_params = (ws, bs, ipa, ld, dp, lr)
                                best_model = model
        
    for ws, bs, ipa, ld, dp, lr in sorted(results):  
        val_accuracy = results[(ws, bs, ipa, ld, dp, lr)]  
        print('ws %s, bs %s, ipa %s, id %s, dp %s, lr %s val accuracy: %f' % (  
                    ws, bs, ipa, ld, dp, lr, val_accuracy))
    print('best validation accuracy achieved during cross-validation: {}, which parameters is {}'.format(best_acc, best_params))
    return best_params

def create_best_model(data, best_parmas):
    ws, bs, ipa, ld, dp, lr = best_parmas

    input_dims = 32 * 32 * 3
    hidden_dims = [100, 100, 100, 100, 100]
    num_classes = 10
    num_epochs = 10
    update_rule = 'adam'

    model = FullyConnectedNet(input_dims, hidden_dims, num_classes, weight_scale=ws, use_batchnorm=True, dropout=dp)

    solver = Solver(model, data, 
                    optim_config={
                        'learning_rate': lr
                    },
                    batch_size=bs,
                    iters_per_ann=ipa,
                    num_epochs=num_epochs,
                    update_rule=update_rule,
                    print_every=200,
                    verbose=True,
                    lr_decay = ld)
    
    solver.train()

    solver.visualization_model()

    test_acc = solver.check_accuracy(data['X_test'], data['y_test'])
    print('Test accuracy: {}'.format(test_acc))

def compare_batchnorm(data):
    # Try training a very deep net with batchnorm
    input_dims = 32 * 32 * 3
    hidden_dims = [100, 100, 100, 100, 100]
    num_classes = 10
    num_train = 1000
    small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
    }

    weight_scale = 2e-2
    bn_model = FullyConnectedNet(input_dims, hidden_dims, num_classes, weight_scale=weight_scale, use_batchnorm=True)
    model = FullyConnectedNet(input_dims, hidden_dims, num_classes, weight_scale=weight_scale, use_batchnorm=False)

    bn_solver = Solver(bn_model, small_data, 
                optim_config={
                    'learning_rate': 1e-3
                },
                batch_size=50,
                iters_per_ann=400,
                num_epochs=10,
                update_rule='adam',
                print_every=200,
                verbose=True,
                lr_decay = 1)
    bn_solver.train()

    solver = Solver(model, small_data, 
                optim_config={
                    'learning_rate': 1e-3
                },
                batch_size=50,
                iters_per_ann=400,
                num_epochs=10,
                update_rule='adam',
                print_every=200,
                verbose=True,
                lr_decay = 1)

    solver.train()

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

def compare_dropout(data):
    input_dims = 32 * 32 * 3
    hidden_dims = [500]
    num_classes = 10
    weight_scale = 1e-2
    num_train = 500
    small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
    }

    solvers = {}
    dropout_choices = [0, 0.75]
    for dropout in dropout_choices:
        model = FullyConnectedNet(input_dims, hidden_dims, num_classes, weight_scale=weight_scale, use_batchnorm=False, dropout=dropout)
        print(dropout)

        solver = Solver(model, small_data, 
                optim_config={
                    'learning_rate': 5e-4
                },
                batch_size=100,
                iters_per_ann=400,
                num_epochs=25,
                update_rule='adam',
                print_every=100,
                verbose=True,
                lr_decay = 1)

        solver.train()
        solvers[dropout] = solver

    train_accs = []
    val_accs = []
    for dropout in dropout_choices:
        solver = solvers[dropout]
        train_accs.append(solver.train_acc_history[-1])
        val_accs.append(solver.val_acc_history[-1])

    plt.subplot(3, 1, 1)
    for dropout in dropout_choices:
        plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
    plt.title('Train accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(ncol=2, loc='lower right')
    
    plt.subplot(3, 1, 2)
    for dropout in dropout_choices:
        plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
    plt.title('Val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(ncol=2, loc='lower right')

    plt.gcf().set_size_inches(15, 15)
    plt.show()

def compare_optims(data):
    num_train = 4000
    small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
    }

    solvers = {}
    input_dims = 32 * 32 * 3
    hidden_dims = [100, 100, 100, 100, 100]
    num_classes = 10
    weight_scale = 5e-2
    learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3, 'sgd': 1e-2, 'sgd_momentum': 1e-2}
    reg = 0.0
    update_rule = ['rmsprop', 'sgd_momentum', 'sgd', 'adam']

    for uo in update_rule:
        model = FullyConnectedNet(input_dims, hidden_dims, num_classes, weight_scale, reg)

        solver = Solver(model, data, 
                optim_config={
                    'learning_rate': learning_rates[uo]
                },
                batch_size=100,
                iters_per_ann=400,
                num_epochs=5,
                update_rule=uo,
                print_every=100,
                verbose=True,
                lr_decay = 1)

        solver.train()
        solvers[uo] = solver
    
    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

    for update_rule, solver in solvers.items():
        plt.subplot(3, 1, 1)
        plt.plot(solver.loss_history, 'o', label=update_rule)
        
        plt.subplot(3, 1, 2)
        plt.plot(solver.train_acc_history, '-o', label=update_rule)

        plt.subplot(3, 1, 3)
        plt.plot(solver.val_acc_history, '-o', label=update_rule)
    
    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(15, 15)
    plt.show()

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
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_val': X_val,
        'y_val': y_val
    }
    return data

if __name__ == '__main__':
    data = pre_dataset('datasets/cifar-10-batches-py')
    # compare_optims(data)
    # compare_batchnorm(data)
    # compare_dropout(data)
    best_params = auto_get_params(data)
    create_best_model(data, best_params)