import numpy as np
import random
from math import log


def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]


def numerical_grad(f, x, h=1e-4):
    return f(x+h) - f(x-h) / (2*h)


class SGD:

    def __init__(self, lr=0.001, mini_batch_size=1, iterations=100):
        self.lr = lr
        self.iter = iterations
        self.iterations_without_improvement = 0
        self.mini_batch_size = mini_batch_size

    def update(self, params, grads):
        params -= self.lr * grads

    def optimize(self, f, df, x, y, params):
        data = list(zip(x, y))
        min_theta, min_value = None, float('inf')
        alpha, theta = self.lr, params
        iterations_without_improvement = 0

        while iterations_without_improvement < self.iter:
            # full value
            value = sum(f(x_i, y_i, theta) for x_i, y_i in zip(x, y))

            if value < min_value:
                min_theta, min_value = theta, value
                iterations_without_improvement = 0
                alpha = self.lr
            else:
                iterations_without_improvement += 1
                alpha *= .9

            # iterate all datas(move theta along gradient 500 times)
            for x_i, y_i in in_random_order(data):
                self.update(params, df(np.array(x_i), np.array(y_i), params))

        return min_theta

    def change_settings(self, lr, mini_batch_size, iterations):
        self.lr = lr
        self.iter = iterations
        self.mini_batch_size = mini_batch_size


class Momentum(SGD):

    def __init__(self, lr=0.01, mini_batch_size=1, iterations=100, momentum=0.9):
        super().__init__(lr, mini_batch_size, iterations)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)

        self.v = self.momentum * self.v - self.lr * grads
        params += self.v

    #@SGD.change_settings.abstractmethod
    def change_settings(self, lr, mini_batch_size, iterations, momentum):
        super().change_settings(lr, mini_batch_size, iterations)
        self.momentum = momentum


class Nesterov(Momentum):

    def update(self, params, grads):
        super().update(params, grads)
        params -= self.v
        params += (self.momentum * self.momentum) * self.v - (1 + self.momentum) * self.lr * grads


class AdaGrad(SGD):

    def __init__(self, lr=0.01, mini_batch_size=1, iterations=100):
        super().__init__(lr, mini_batch_size, iterations)
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = np.zeros_like(params)

        self.h += grads * grads
        params -= self.lr * grads / (np.sqrt(self.h) + 1e-7)


class Adam(SGD):

    def __init__(self,  lr=0.01, mini_batch_size=1, iterations=100, beta1=0.9, beta2=0.999):
        super().__init__(lr, mini_batch_size, iterations)
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter_adam = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.iter_adam += 1
        lr_t = self.lr * np.sqrt(1. - self.beta2 ** self.iter_adam) / (1. - self.beta1 ** self.iter)

        self.m += (1 - self.beta1) * (grads - self.m)
        self.v += (1 - self.beta2) * (grads * grads - self.v)

        params -= lr_t * self.m / (np.sqrt(self.v) + 1e-7)

    def change_settings(self, lr, mini_batch_size, iterations, beta1, beta2):
        super().change_settings(lr, mini_batch_size, iterations)
        self.beta1 = beta1
        self.beta2 = beta2