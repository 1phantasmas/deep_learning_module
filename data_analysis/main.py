from loss_function import *
from optimizer import *
import csv
from collections import defaultdict


class Model:
    def __init__(self, feature, target, loss_function):
        self.feature = np.array(feature)
        self.target = np.array(target)
        self.beta = np.random.randn(np.shape(self.feature)[1] + 1)
        self.loss_function = loss_function(self.feature, self.target, self.beta)
        self.optimizer = self.set_optimizer()

    # noinspection PyMethodMayBeStatic
    def set_optimizer(self):
        print("There are 4 options of optimizers")
        print("1. Normal Stochastic Gradient")
        print("2. Momentum SGD")
        print("3. AdaGrad SGD")
        print("4. Adam(combination of Momentum SGD and AdaGrad)")
        print("5. Nesterov")
        optimizer_ = int(input("Type in the number of what to optimize : "))
        optimizer = {1: SGD, 2: Momentum, 3: AdaGrad, 4: Adam, 5: Nesterov}[optimizer_]()
        return optimizer

    @staticmethod
    def how_to_use():
        print()

    def batch_normalize(self):
        mean, var = np.zeros_like(self.feature[0]), np.zeros_like(self.feature[0])
        for x_i in self.feature:
            mean += x_i
        mean /= self.feature.shape[0]

        for x_i in self.feature:
            var += (x_i - mean) ** 2
        var /= self.feature.shape[0]

        self.feature = (self.feature - mean) / np.sqrt(var + 1e-7)
        # self.target = (self.target - (sum(self.target) / self.target.shape)) / np.sqrt()

    def train(self):
        if self.loss_function == CrossEntropy:
            self.loss_function.optimize()
        else:
            self.loss_function.accuracy(self.beta)
            self.beta = self.optimizer.optimize(self.loss_function.loss, self.loss_function.loss_grad,
                                                self.feature, self.target, self.beta)
            self.loss_function.accuracy(self.beta)


if __name__ == "__main__":
    # diabetes
    diabetes_csv = csv.reader(open("C:/Users/J/Dropbox/Doodle/final_project/datasets/diabetes.csv"))
    header_diabetes = diabetes_csv.__next__()

    diabetes_raw = list(d for d in diabetes_csv)
    diabetes_feature = [[float(di) for di in d[:-1]] for d in diabetes_raw]
    diabetes_target = list([int(d[-1]) for d in diabetes_raw])

    # boston house set
    boston_ = csv.reader(open("C:/Users/J/Dropbox/Doodle/final_project/datasets/boston.csv"))
    header_boston = boston_.__next__()

    boston_raw = list(d for d in boston_)
    boston_feature = [[float(di) for di in d[:-1]] for d in boston_raw]
    boston_target = list([float(d[-1]) for d in boston_raw])

    model = Model(boston_feature, boston_target, LogLikelihood)
    # print(model.optimizer)
    # print(model.beta)
    model.train()


