# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from optimizer import *
import time
from math import exp


def f(func):
    def f1(x, y):
        return x ** 2 / 20.0 + y ** 2

    def df1(x, y):
        return x / 10, 2 * y

    def f2(x, y):
        return 3 * x * np.exp(-x * x / 20 - y * y / 20)

    def df2(x, y):
        return -0.3 * (x ** 2 - 10) * exp(-0.05 * (x ** 2 + y ** 2)), \
               -0.3 * x * y * exp(-0.05 * (x ** 2 + y ** 2))

    def f3(x, y):
        return x * y * np.exp(-x * x / 20 - y * y / 20)

    def df3(x, y):
        return -0.1 * (x ** 2 - 10) * y * np.exp(-0.05 * (x ** 2 + y ** 2)), \
               -0.1 * x * (y ** 2 - 10) * np.exp(-0.05 * (x ** 2 + y ** 2))

    return {1: (f1, df1, (-7., 2.), (-10, 10, -10, 10)),
            2: (f2, df2, (-1., 2.), (-10, 10, -10, 10)),
            3: (f3, df3, (-1., 2.), (-10, 10, -10, 10))}[func]


def o(opt):
    if opt is 6:
        optimizers = OrderedDict()
        optimizers["SGD"] = SGD(lr=1)
        optimizers["Momentum"] = Momentum(lr=0.01)
        optimizers["AdaGrad"] = AdaGrad(lr=0.1)
        optimizers["Adam"] = Adam(lr=0.001)
        optimizers["Nesterov"] = Nesterov(lr=0.01)
        return optimizers

    else:
        return {1: SGD, 2: Momentum, 3: AdaGrad, 4: Adam, 5: Nesterov}[opt]()


def plot(functions, optimizer):
    f, df, init_pos = functions[:3]
    params = np.array(list(init_pos))
    x_history, y_history = [], []

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    x_lim_left, x_lim_right, y_lim_left, y_lim_right = functions[-1]

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    mask = Z > 7
    Z[mask] = 0

    if type(optimizer) == dict:
        pass
    else:
        min_pos, min_value = None, float('inf')
        iterations_without_movement = 0
        trial = 0
        while iterations_without_movement < 130:
            x_history.append(params[0])
            y_history.append(params[1])
            value = f(params[0], params[1])

            if value < min_value:
                min_pos, min_value = params, value
                iterations_without_movement = 0
                optimizer.lr = 1

            else:
                iterations_without_movement += 1
                optimizer.lr *= .9

            optimizer.update(params, np.array(df(params[0], params[1])))
            trial += 1

        plt.xlim(x_lim_left, x_lim_right)
        plt.ylim(y_lim_left, y_lim_right)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.plot(x_history, y_history, 'o-', color="red")
        plt.contour(X, Y, Z)

        plt.show()
        print(params)
        print(trial)


if __name__ == "__main__":
    print("Visualization of various SGD methods convergence with some functions")
    time.sleep(.5)
    print("Some options that you can choose are the followings")

    while True:
        print(" * Functions")
        print("\t 1. f(x, y) = x**2/20 + y**2")
        print("\t 2. f(x, y) = x * exp(-x**2 -y**2)")
        print("\t 3. f(x, y) = x * y * exp(-x**2/20 - y**2 / 20")
        func_ = int(input("Choose a function : "))
        print("You've chose " +
              {1: "f(x, y) = x**2/20 + y**2 ",
               2: "f(x, y) = x * exp(-x**2 -y**2)",
               3: "f(x, y) = x * y * exp(-x**2/20 - y**2 / 20"}[func_])

        print("\n")
        time.sleep(1)

        print(" * Optimizers")
        print("\t 1. Normal Stochastic Gradient Descent(we've done in class")
        print("\t 2. Momentum SGD")
        print("\t 3. AdaGrad")
        print("\t 4. Adam(combination of Momentum SGD and AdaGrad)")
        print("\t 5. Nesterov")
        print("\t 6. See them all")
        print("\t there are no options of choosing multiple optimizers")
        print("\t if you choose between 1 - 4, you will see how they converge time by time")
        opt_ = int(input("Choose a optimizer : "))
        if opt_ == 6:
            pass
        else:
            # 1 by 1
            pass

        print("You've chose " +
              {1: "Normal Stochastic Gradient Descent(we've done in class",
               2: "Momentum SGD",
               3: "AdaGrad",
               4: "Adam(combination of Momentum SGD and AdaGrad",
               5: "Nesterov",
               6: "See them all"}[opt_])

        func = f(func_)
        opt = o(opt_)
        plot(func, opt)

        end = int(input("Continue? \n More = 1, No = 2 : "))
        if end == 1: pass
        if end == 2: break
