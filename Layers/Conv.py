from scipy import signal
from Layers import Base
import numpy as np
import copy


class Conv(Base.BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels: int):
        super().__init__()

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.trainable = True

        self.weights = np.ones((num_kernels,) + convolution_shape)  # Concatenate
        self.bias = np.ones(num_kernels)

        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer_weights = None
        self._optimizer_bias = None

        self.input_buffer = None
        self.error_buffer = None

    def initialize(self, weights_initializer, bias_initializer):  # uniformly between 0 and 1

        weights = weights_initializer.initialize(self.weights.shape,
                                                 fan_in=np.prod(self.convolution_shape),  # number of input channels
                                                 fan_out=self.num_kernels * np.prod(self.convolution_shape[1:]))
        # channels of output = num_kernels, number of output channels
        bias = bias_initializer.initialize(self.bias.shape, fan_in=1, fan_out=1)

        self.weights = weights
        self.bias = bias

    def forward(self, input_tensor):
        self.input_buffer = input_tensor

        # calculate correct shape for the output tensor
        batch_size = input_tensor.shape[0]

        output_size_y = int(input_tensor.shape[-2] / self.stride_shape[0])
        if (input_tensor.shape[-2] % output_size_y) != 0:
            output_size_y += 1

        output_size_x = int(input_tensor.shape[-1] / self.stride_shape[-1])
        if (input_tensor.shape[-1] % output_size_x) != 0:
            output_size_x += 1

        if len(self.stride_shape) == 1:  # 1D
            next_input_tensor = np.array(np.ones((batch_size, self.num_kernels, output_size_x)))
        else:  # 2D
            next_input_tensor = np.array(np.ones((batch_size, self.num_kernels, output_size_y, output_size_x)))

        # get all kernels in a list
        kernels = self.num_kernels * [np.array(0)]
        for k in range(0, self.num_kernels):
            kernels[k] = np.array(self.weights[k])

        # apply correlation
        for i in range(0, batch_size):

            for k in range(0, self.num_kernels):

                raw_conv = signal.correlate(input_tensor[i], kernels[k], mode='same')  # same output size
                # calculates the index corresponding to the central element
                raw_conv = raw_conv[int(raw_conv.shape[0] / 2)]
                raw_conv += self.bias[k]

                if len(self.stride_shape) == 1:  # 1D
                    stride_conv = raw_conv[0::self.stride_shape[0]]
                else:  # 2D
                    stride_conv = raw_conv[0::self.stride_shape[0], 0::self.stride_shape[1]]

                next_input_tensor[i][k] = stride_conv

        return next_input_tensor

    def backward(self, error_tensor):
        self.error_buffer = error_tensor
        # calculate gradient with respect to lower layers

        # create new kernels for backward pass
        num_channels = self.convolution_shape[0]
        backward_kernels = num_channels * [np.array(0)]
        for s in range(0, num_channels):  # num_channels of input = num_kernels in backward
            kernel_buffer = self.num_kernels * [np.array(0)]
            for k in range(0, self.num_kernels):
                kernel_buffer[k] = np.array(self.weights[k][s])
            backward_kernels[s] = np.array(kernel_buffer)
        backward_kernels = np.array(backward_kernels)

        # UpSample the error tensor if we have stride
        if len(self.stride_shape) == 2:
            padded_error = np.zeros((error_tensor.shape[0], # batch
                                     error_tensor.shape[1],  # channel
                                     self.input_buffer.shape[2],   # x
                                     self.input_buffer.shape[3]))  # y
            for y in range(0, error_tensor.shape[2]):
                for x in range(0, error_tensor.shape[3]):
                    padded_error[:, :, y * self.stride_shape[0], x * self.stride_shape[1]] = error_tensor[:, :, y, x]
        else:
            padded_error = np.zeros((error_tensor.shape[0],
                                     error_tensor.shape[1],
                                     self.input_buffer.shape[2]))
            for y in range(0, error_tensor.shape[2]):
                padded_error[:, :, y * self.stride_shape[0]] = error_tensor[:, :, y]

        # calculate gradient with respect to the bias
        if len(self.stride_shape) == 2:
            gradient_bias = np.sum(padded_error, axis=(0, 2, 3))
        else:
            gradient_bias = np.sum(padded_error, axis=(0, 2))
        self.gradient_bias = gradient_bias

        # apply convolution to the (UpSampled) error tensor
        batch_size = error_tensor.shape[0]
        num_channels = error_tensor.shape[1]
        prev_error_tensor = np.zeros_like(self.input_buffer)

        for i in range(0, batch_size):
            for k in range(0, len(backward_kernels)):
                convolution_output = np.zeros_like(self.input_buffer[0, 0])
                for c in range(0, num_channels):
                    convolution_output += signal.convolve(padded_error[i][c], backward_kernels[k][c], mode='same')
                prev_error_tensor[i][k] = convolution_output

        # calculate gradient with respect to the weights

        # calculate padding dimensions
        # height
        pad_height_before = int(self.weights.shape[2] / 2)
        if self.weights.shape[2] % 2 == 0:  # kernel size is even
            pad_height_after = pad_height_before - 1  # asymmetric padding
        else:
            pad_height_after = pad_height_before

        # Width
        pad_width_before, pad_width_after = 0, 0
        if len(self.stride_shape) == 2:  # 2D stride
            pad_width_before = int(self.weights.shape[3] / 2)
            if self.weights.shape[3] % 2 == 0:  # for even width
                pad_width_after = pad_width_before - 1  # asymmetric padding
            else:  # 1D stride
                pad_width_after = pad_width_before

        # apply correlation between input and error tensor
        gradient_weights = np.zeros_like(self.weights)

        for i in range(0, self.input_buffer.shape[0]):  # batch size

            for k in range(0, padded_error.shape[1]):
                gradient_kernel = np.zeros(self.convolution_shape)

                for s in range(self.weights.shape[1]):

                    if len(self.stride_shape) == 2:
                        padded_input = np.pad(self.input_buffer[i][s],
                                              pad_width=((pad_height_before, pad_height_after),
                                                         (pad_width_before, pad_width_after)),
                                              mode='constant')
                    else:
                        padded_input = np.pad(self.input_buffer[i][s],
                                              pad_width=(pad_height_before, pad_height_after),
                                              mode='constant')

                    gradient_kernel[s] = signal.correlate(padded_input, padded_error[i][k], mode='valid')  # no padding

                gradient_weights[k] += gradient_kernel

        self._gradient_weights = gradient_weights

        # update weights and bias
        if self.optimizer is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return prev_error_tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
        return

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias
        return

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_weights = optimizer
        self._optimizer_bias = copy.deepcopy(optimizer)
