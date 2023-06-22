from Layers import Base
import numpy as np
import copy


class NeuralNetwork(Base.BaseLayer):

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        super().__init__()

        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_buffer = None

        return

    def forward(self):

        input_tensor, self.label_buffer = self.data_layer.next()

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        output = self.loss_layer.forward(input_tensor, self.label_buffer)

        return output

    def backward(self):

        error_tensor = self.loss_layer.backward(self.label_buffer)

        for layer in np.flip(self.layers):
            error_tensor = layer.backward(error_tensor)

        return

    def append_layer(self, layer):

        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            deep_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = deep_copy

        self.layers.append(layer)

        return

    def train(self, iterations):

        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

        return

    def test(self, input_tensor):

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        results = input_tensor

        return results
