import numpy as np
from Layers.Base import *


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor_shape = input_tensor.shape
        reshape = np.zeros((self.input_tensor_shape[0], np.prod(self.input_tensor_shape[1:])))
        for i in range(self.input_tensor_shape[0]):
            reshape[i] = input_tensor[i].flatten()

        return reshape

    def backward(self, error_tensor):

        return error_tensor.reshape(self.input_tensor_shape)

