from Layers import Base
import numpy as np


class Pooling(Base.BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_buffer = None
        self.locations = None

    def forward(self, input_tensor):
        self.input_buffer = input_tensor

        stride_y = self.stride_shape[0]
        stride_x = self.stride_shape[1]

        pooling_y = self.pooling_shape[0]
        pooling_x = self.pooling_shape[1]

        batch_size = input_tensor.shape[0]
        num_channels = input_tensor.shape[1]

        output_size_y = int((input_tensor.shape[2]) / stride_y)  # check weather kernel size = stride
        if (input_tensor.shape[2] - stride_y * (output_size_y - 1)) < pooling_y:
            output_size_y -= 1

        output_size_x = int((input_tensor.shape[3]) / stride_x)
        if (input_tensor.shape[3] - stride_x * (output_size_x - 1)) < pooling_x:
            output_size_x -= 1

        next_input_tensor = np.array(np.ones((batch_size, num_channels, output_size_y, output_size_x)))
        locations = np.array(np.ones((batch_size, num_channels, output_size_y, output_size_x, 2)))

        for i in range(0, batch_size):
            for c in range(0, num_channels):
                for y in range(0, next_input_tensor.shape[2]):
                    for x in range(0, next_input_tensor.shape[3]):
                        y_start = y * stride_y
                        y_end = y_start + pooling_y

                        x_start = x * stride_x
                        x_end = x_start + pooling_x

                        pooling_input = input_tensor[i, c, y_start:y_end, x_start:x_end]

                        maximum = np.amax(pooling_input)
                        next_input_tensor[i][c][y][x] = maximum

                        row_indices, column_indices = np.where(pooling_input == maximum)
                        locations[i][c][y][x] = np.array([y_start + row_indices[0], x_start + column_indices[0]])

        self.locations = locations

        return next_input_tensor

    def backward(self, error_tensor):

        prev_error_tensor = np.zeros_like(self.input_buffer)

        for i in range(0, error_tensor.shape[0]):
            for c in range(0, error_tensor.shape[1]):
                for y in range(0, error_tensor.shape[2]):
                    for x in range(0, error_tensor.shape[3]):

                        y_index = int(self.locations[i][c][y][x][0])
                        x_index = int(self.locations[i][c][y][x][1])

                        prev_error_tensor[i][c][y_index][x_index] += error_tensor[i][c][y][x]

        return prev_error_tensor
