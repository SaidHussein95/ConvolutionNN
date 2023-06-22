from Layers import Base
import numpy as np


class ReLU(Base.BaseLayer):

    def __init__(self):
        super().__init__()

        self.input_buffer = None

        return

    def forward(self, input_tensor):
        relu = lambda x: np.maximum(x, 0)

        next_input_tensor = relu(input_tensor)

        self.input_buffer = input_tensor

        return next_input_tensor

    def backward(self, error_tensor):
        sign = np.maximum(np.sign(self.input_buffer), 0)

        prev_error_tensor = sign * error_tensor

        return prev_error_tensor
