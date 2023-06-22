import numpy as np


class Sgd:

    def __init__(self, learning_rate: float):

        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):

        new_weight_tensor = weight_tensor - (self.learning_rate * gradient_tensor)

        return new_weight_tensor


class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):

        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        momentum = (self.momentum_rate * self.momentum) - (self.learning_rate * gradient_tensor)
        self.momentum = momentum

        return weight_tensor + momentum


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k = 0
        self.r_k = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v_k = self.mu * self.v_k + (1 - self.mu) * gradient_tensor
        self.r_k = self.rho * self.r_k + (1 - self.rho) * gradient_tensor * gradient_tensor
        v_hat = self.v_k / (1 - self.mu ** self.k)   # Bias correction
        r_hat = self.r_k / (1 - self.rho ** self.k)  # Bias correction
        self.k += 1
        return weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps))
