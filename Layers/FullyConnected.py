from Layers import Base
import numpy as np


class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.input_size = input_size + 1  # adding bias
        self.output_size = output_size

        self.trainable = True
        self.weights = np.random.rand(self.input_size, self.output_size)

        self.input_buffer = None
        self.error_buffer = None

        self._optimizer = None
        self._gradient_weights = None

        return

    def initialize(self, weights_initializer, bias_initializer):

        weights = weights_initializer.initialize((self.input_size-1, self.output_size),
                                                 fan_in=self.input_size-1,
                                                 fan_out=self.output_size)

        bias = bias_initializer.initialize((1, self.output_size),
                                           fan_in=1,
                                           fan_out=self.output_size)

        self.weights = np.concatenate((weights, bias))

        return

    def forward(self, input_tensor):
        input_tensor = np.append(input_tensor, np.ones((input_tensor.shape[0], 1)), axis=1)
        self.input_buffer = input_tensor

        next_input_tensor = np.matmul(input_tensor, self.weights)

        return next_input_tensor

    def backward(self, error_tensor):
        self.error_buffer = error_tensor

        prev_error_tensor = np.matmul(error_tensor, np.transpose(self.weights))
        prev_error_tensor = np.delete(prev_error_tensor, prev_error_tensor.shape[1] - 1, axis=1)

        gradient_tensor = np.matmul(np.transpose(self.input_buffer), error_tensor)
        gradient_weights = gradient_tensor
        self.gradient_weights = gradient_weights

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)

        return prev_error_tensor

    # optimizer getter
    @property
    def optimizer(self):
        return self._optimizer

    # optimizer setter
    @optimizer.setter
    def optimizer(self, learning_rate):
        self._optimizer = learning_rate
        return

    # gradient weights getter
    @property
    def gradient_weights(self):
        return self._gradient_weights

    # gradient weights setter
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
        return
