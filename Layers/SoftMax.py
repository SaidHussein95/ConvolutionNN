from Layers import Base
import numpy as np


class SoftMax(Base.BaseLayer):

    def __init__(self):
        super().__init__()

        self.input_buffer = None

        return

    def forward(self, input_tensor):
        x_max = np.amax(input_tensor)
        input_tensor -= x_max

        e_function = lambda x: np.exp(x)
        expon = e_function(input_tensor)

        sums = expon.sum(axis=1)
        sums = np.reshape(sums, (expon.shape[0], 1))
        sums = np.tile(sums, (1, expon.shape[1]))

        next_input_tensor = expon / sums

        self.input_buffer = next_input_tensor

        return next_input_tensor

    def backward(self, error_tensor):
        sums = error_tensor * self.input_buffer
        sums = sums.sum(axis=1)
        sums = np.reshape(sums, (sums.shape[0], 1))
        sums = np.tile(sums, (1, error_tensor.shape[1]))

        prev_error_tensor = self.input_buffer * (error_tensor - sums)

        return prev_error_tensor
