import numpy as np

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, num_classes, std=1e-4):
        """
        Weights are initialized to small random values and biases are initialized to zero.
        """
        self.parameters = {}
        self.parameters['W1'] = std * np.random.randn(hidden_size, input_size)
        self.parameters['b1'] = np.zeros(hidden_size)
        self.parameters['W2'] = std * np.random.randn(num_classes, hidden_size)
        self.parameters['b2'] = np.zeros(num_classes)
    
    def loss(self, X, y, reg):
        """
        X: N * D 
        y: N * 1
        """
        W1 = self.parameters['W1']  # H * D 
        b1 = self.parameters['b1']  # H * 1
        W2 = self.parameters['W2']  # C * H 
        b2 = self.parameters['b2']  # C * 1
        num_examples = X.shape[0]  

        # Compute the forward pass
        Relu = lambda x: np.maximum(0, x)
        z1 = X.dot(W1.T) + b1  # N * H
        a1 = Relu(z1)
        z2 = a1.dot(W2.T) + b2  # N * C 
        scores = z2 

        if y is None:
            return scores
        
        # Compute the loss
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        pro_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        ground_true = np.zeros(scores.shape)
        ground_true[range(num_examples), y] = 1
        loss = -np.sum(ground_true * np.log(pro_scores)) / num_examples + 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # Backward pass: compute gradients
        grads = {}
        # Compute the gradient of z2 (scores)
        dz2 = -(ground_true - pro_scores) / num_examples  # N * C 
        # Backprop into W2, b2 and a1
        dW2 = dz2.T.dot(a1)  # C * H 
        db2 = np.sum(dz2, axis=0)  # 1 * C 
        da1 = dz2.dot(W2)  # N * H
        # Backprop into z1
        dz1 = da1
        dz1[a1 <= 0] = 0  # N * H 
        # Backprop into W1, b1
        dW1 = dz1.T.dot(X)  # H * D 
        db1 = np.sum(dz1, axis=0) # 1 * H

        # add the regularization
        grads['W1'] = dW1 + reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + reg * W2
        grads['b2'] = db2

        return loss, grads

    def train(self, X, y, X_val, y_val, reg, learning_rate, 
                learning_rate_decay, iterations_per_lr_annealing, 
                num_epoches, batch_size, verbose):
        num_examples = X.shape[0]
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        iterations_per_epoch = max(num_examples / batch_size, 1)
        num_iters = int(num_epoches * iterations_per_epoch)

        for i in range(num_iters):
            # mini batch
            sample_index = np.random.choice(num_examples, batch_size, replace=True)
            X_batch = X[sample_index, :]
            y_batch = y[sample_index]
            
            # cal loss
            loss, grads = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.parameters['W1'] -= learning_rate * grads['W1']
            self.parameters['b1'] -= learning_rate * grads['b1']
            self.parameters['W2'] -= learning_rate * grads['W2']
            self.parameters['b2'] -= learning_rate * grads['b2']

            if verbose and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, num_iters, loss))

            if i % iterations_per_epoch == 0:
                train_acc_history.append(np.mean(self.predict(X_batch) == y_batch))
                val_acc_history.append(np.mean(self.predict(X_val) == y_val))
            
            if i % iterations_per_lr_annealing == 0:
                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }
    
    def predict(self, X):
        # Compute the forward pass
        Relu = lambda x: np.maximum(0, x)
        z1 = X.dot(self.parameters['W1'].T) + self.parameters['b1']
        a1 = Relu(z1)
        z2 = a1.dot(self.parameters['W2'].T) + self.parameters['b2']
        score = z2

        y_pred = np.argmax(score, axis=1)
        return y_pred
        

