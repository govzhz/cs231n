import numpy as np


class KNearestNeighbor(object):
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X_train, y_train):
        """KNN无需训练

        """
        self.X_train = X_train
        self.y_train = y_train

    def compute_distances(self, X_test):
        """计算测试集和每个训练集的欧氏距离

        向量化实现需转化公式后实现（单个循环不需要）
        :param X_test: 测试集 numpy.ndarray
        :return: 测试集与训练集的欧氏距离数组 numpy.ndarray
        """
        dists = np.zeros((X_test.shape[0], self.X_train.shape[0]))

        value_2xy = np.multiply(X_test.dot(self.X_train.T), -2)
        value_x2 = np.sum(np.square(X_test), axis=1, keepdims=True)
        value_y2 = np.sum(np.square(self.X_train), axis=1)
        dists = value_2xy + value_x2 + value_y2
        return dists

    def predict_label(self, dists, k):
        """选择前K个距离最近的标签，从这些标签中选择个数最多的作为预测分类

        :param dists: 欧氏距离
        :param k: 前K个分类
        :return: 预测分类（向量）
        """
        y_pred = np.zeros(dists.shape[0])

        for i in range(dists.shape[0]):
            # 取前K个标签
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            # 取K个标签中个数最多的标签
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

    def predict(self, X_test, k):
        dists = self.compute_distances(X_test)
        y_pred = self.predict_label(dists, k)
        return y_pred
