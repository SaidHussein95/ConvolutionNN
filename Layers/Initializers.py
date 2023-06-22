import numpy as np


class Constant:

    def __init__(self, constant=0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = self.constant * np.ones(weights_shape)
        return initialized_tensor


class UniformRandom:

    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = np.random.random(weights_shape)
        return initialized_tensor


class Xavier:

    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        initialized_tensor = np.random.normal(loc=0, scale=sigma, size=weights_shape)
        return initialized_tensor


class He:

    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)  # only depends on the previous layer
        initialized_tensor = np.random.normal(loc=0, scale=sigma, size=weights_shape)
        return initialized_tensor
