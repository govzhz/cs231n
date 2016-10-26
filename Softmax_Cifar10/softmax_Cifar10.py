import numpy as np
from data_utils import load_CIFAR10
from softmax import Softmax
import matplotlib.pyplot as plt


def pre_dataset():
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    VisualizeImage(X_train, y_train)

    num_train = 49000
    num_val = 1000

    mask = range(num_train, num_train + num_val)
    X_val = X_train[mask]
    y_val = y_train[mask]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # add a parameter for W
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

    return X_train, y_train, X_test, y_test, X_val, y_val


def auto_get_parameters(X_train, y_train, X_val, y_val):
    learning_rates = [1e-7, 5e-5]
    regularization_strengths = [5e4, 1e5]

    best_val = -1
    best_parameters = None

    for i in learning_rates:
        for j in regularization_strengths:
            softmax = Softmax()
            softmax.train(X_train, y_train, j, i, 200, 1500, True)
            y_pred = softmax.predict(X_val)
            acc = np.mean(y_pred == y_val)
            if acc > best_val:
                best_val = acc
                best_parameters = (i, j)

    print('OK! Have been identified parameter! Best validation accuracy achieved during cross-validation: %f' % best_val)
    return best_parameters


def get_softmax_model(parameters, X, y):
    softmax = Softmax()
    loss_history = softmax.train(X, y, parameters[1], parameters[0], 200, 1500, True)
    VisualizeLoss(loss_history)
    VisualizeW(softmax)
    return softmax


def VisualizeLoss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()


def VisualizeImage(X_train, y_train):
    """可视化数据集

    :param X_train: 训练集
    :param y_train: 训练标签
    :return:
    """
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 8
    for y, cls in enumerate(classes):
        # 得到该标签训练样本下标索引
        idxs = np.flatnonzero(y_train == y)
        # 从某一分类的下标中随机选择8个图像（replace设为False确保不会选择到同一个图像）
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        # 将每个分类的8个图像显示出来
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            # 创建子图像
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            # 增加标题
            if i == 0:
                plt.title(cls)
    plt.show()


def VisualizeW(softmax):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    # Visualize the learned weights for each class
    w = softmax.W[:, :-1]  # strip out the bias
    w = w.reshape(10, 32, 32, 3)

    w_min, w_max = np.min(w), np.max(w)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[i, :, :, :].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_val, y_val = pre_dataset()
    best_parameters = auto_get_parameters(X_train, y_train, X_val, y_val)
    softmax = get_softmax_model(best_parameters, X_train, y_train)
    y_pred = softmax.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print('Accuracy achieved during cross-validation: %f' % acc)