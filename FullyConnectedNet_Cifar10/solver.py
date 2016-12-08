from fullyConnectedNet import FullyConnectedNet
import numpy as np
import matplotlib.pyplot as plt
import optim

class Solver(object):
    def __init__(self, model, data, **kwargs):
        self.model = model

        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.batch_size = kwargs.pop('batch_size', 100)
        self.iters_per_ann = kwargs.pop('iters_per_ann', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.verbose = kwargs.pop('verbose', True)
        self.print_every = kwargs.pop('print_every', 10)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)

        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
        
        
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
    
    def _step(self):
        N = self.X_train.shape[0]

        sample_index = np.random.choice(N, self.batch_size, replace=True)
        X_batch = self.X_train[sample_index, :]
        y_batch = self.y_train[sample_index]

        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p], self.optim_configs[p] = next_w, next_config
    
    def check_accuracy(self, X, y, num_samples=None):
        N = X.shape[0]
        if num_samples is not None:
            mask = np.random.choice(N, num_samples, replace=True)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batchsize = int(N / self.batch_size)
        if N % self.batch_size != 0:
            num_batchsize += 1
			
        y_pred = []
        for i in range(num_batchsize):
            start = int(i * self.batch_size)
            end = int((i + 1) * self.batch_size)
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = (y_pred == y).mean()
        return acc


    def train(self):
        N = self.X_train.shape[0]

        iters_per_epochs = max(1, N / self.batch_size)
        num_iters = int(iters_per_epochs * self.num_epochs) 
        epoch = 0
        for it in range(num_iters):
            self._step()

            if (it + 1) % self.iters_per_ann == 0:
                for c in self.optim_configs:
                    self.optim_configs[c]['learning_rate'] *= self.lr_decay

            if self.verbose and it % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (it + 1, num_iters, self.loss_history[-1]))

            if self.verbose and (it % iters_per_epochs or it == (num_iters - 1))  == 0:
                epoch += 1
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)
                self.train_acc_history.append(train_acc)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.val_acc_history.append(val_acc)
                print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (epoch, self.num_epochs, train_acc, val_acc))
    
    
    def visualization_model(self):
        plt.subplot(2, 1, 1)
        plt.title('Training loss')
        plt.plot(self.loss_history, 'o')
        plt.xlabel('Iteration')

        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.plot(self.train_acc_history, '-o', label='train')
        plt.plot(self.val_acc_history, '-o', label='val')
        plt.plot([0.5] * len(self.val_acc_history), 'k--')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.gcf().set_size_inches(15, 12)
        plt.show()