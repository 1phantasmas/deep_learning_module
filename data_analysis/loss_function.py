import numpy as np
from math import log
from .optimizer import *
from collections import defaultdict


class LossFunction:
    def __init__(self, feature, target, beta):
        self.feature = feature
        self.target = target
        self.beta = beta

    def loss(self, x_i, y_i, beta):
        pass

    def loss_grad(self, x_i, y_i, beta):
        pass

    def full_loss(self, beta):
        pass

    def accuracy(self, beta):
        pass


class MeanSquaredError(LossFunction):
    def loss(self, x_i, y_i, beta):
        return (beta[0] + sum(x_i * beta[1:]) - y_i) ** 2

    def loss_grad(self, x_i, y_i, beta):
        return 2 * np.array(list([1] + list(x_i))) * (beta[0] + sum(x_i * beta[1:]) - y_i)

    def full_loss(self, beta):
        return sum(self.loss(x_i, y_i, beta) for x_i, y_i in zip(self.feature, self.target))

    def accuracy(self, beta):
        print("for Linear Regression, we see R-squared")
        SSR_2 = self.full_loss(beta)
        SST_2 = sum((self.target - (sum(self.target) / self.target.shape)) ** 2)
        print(SST_2)
        print(1 - (SSR_2 / SST_2))
        return 1 - SSR_2 / SST_2


class LogLikelihood(LossFunction):

    # noinspection PyMethodMayBeStatic
    def sigmoid(self, x_i, beta):
        prevent_overflow = np.clip(sum(x_i * beta[1:]) + beta[0], -500, 500)
        return 1 / (1 + (np.exp(-prevent_overflow)))

    # Negative Log-Likelihood
    # noinspection PyMethodMayBeStatic
    def loss(self, x_i, y_i, beta):
        if y_i == 1:
            return -log(self.sigmoid(x_i, beta) + 1e-7)
        if y_i == 0:
            return -log(1 - self.sigmoid(x_i, beta) + 1e-7)

    # noinspection PyMethodMayBeStatic
    def loss_grad(self, x_i, y_i, beta):
        def sigmoid_prime(x_i_, beta_):
            return np.array(list([1] + list(x_i_))) * self.sigmoid(x_i_, beta_) * (1 - self.sigmoid(x_i_, beta_))

        if y_i == 1:
            return -sigmoid_prime(x_i, beta) / (self.sigmoid(x_i, beta) + 1e-7)
        if y_i == 0:
            return sigmoid_prime(x_i, beta) / (1 - self.sigmoid(x_i, beta) + 1e-7)

    def accuracy(self, beta):
        count = 0
        for x_i, y_i in zip(self.feature, self.target):
            if (self.sigmoid(x_i, beta) > 0.5) & (y_i == 1): count += 1
            if (self.sigmoid(x_i, beta) < 0.5) & (y_i == 0): count += 1
        print(count)
        return count / np.shape(self.feature)[0] * 100


class CrossEntropy(LossFunction):

    def __int__(self, feature, target, beta):
        super().__init__(feature, target, beta)
        # self.target_dict = {i: element for i, element in enumerate(set(self.target))}
        self.target_dict = dict.fromkeys(set(self.target))
        for i, key in enumerate(self.target_dict):
            self.target_dict[key] = [1 if _ == i else 0 for _ in range(self.k)]

        self.k = len(self.target_dict)
        self.beta_set = defaultdict()
        for i in range(self.k):
            self.beta_set[i] = np.random.randn(np.shape(self.feature)[1] + 1)

    #@loss.abstractmethod
    def loss(self, y, t):
        # x_i : unprepared probability
        # y_i ; one-hot encoded probability.
        return -np.sum(y * np.log(t + 1e-7))

    def loss_grad(self, x_i, y_i, beta):
        pass

    def optimize(self, lr):
        min_theta, min_value = None, float('inf')
        alpha, theta = lr, self.beta_set
        iterations_without_improvement = 0

    def accuracy(self, beta):
        pass

