import numpy as np


class CrossEntropyLoss:

    def __init__(self):

        self.epsilon = np.finfo(float).eps
        self.y_head = None

        return

    def forward(self, prediction_tensor, label_tensor):
        data = prediction_tensor * label_tensor

        data = data + (self.epsilon * label_tensor)

        data = np.sum(data, axis=1)

        log_function = lambda x: -1 * np.log(x)
        log_data = log_function(data)

        end_data = np.sum(log_data)

        self.y_head = prediction_tensor

        return end_data

    def backward(self, label_tensor):

        # prev_error_tensor = -1 * label_tensor / (self.y_head + self.epsilon)
        prev_error_tensor = -1 * label_tensor / self.y_head

        return prev_error_tensor
