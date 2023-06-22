# ConvolutionNN
This repository contains a Python framework for building and training fully connected neural networks, with extensions for convolutional neural networks (CNNs). The framework includes implementation of initialization schemes, advanced optimizers, and key layers such as convolutional and max-pooling layers.
# Install the required dependencies:
pip install numpy 
from Base import Base Layer
# Initializers
The Initializers.py module in the Layers folder contains implementation of four initialization schemes: Constant, UniformRandom, Xavier, and He. Each initializer provides the initialize(weights_shape, fan_in, fan_out) method for initializing tensors.

# Advanced Optimizers
The Optimizers.py module in the Optimization folder includes implementation of two advanced optimizers: SgdWithMomentum and Adam. These optimizers provide the calculate_update(weight_tensor, gradient_tensor) method for updating weights based on gradients.

# Flatten Layer
The Flatten.py module in the Layers folder implements the Flatten layer, which reshapes multi-dimensional input to a one-dimensional feature vector. It provides the forward(input_tensor) and backward(error_tensor) methods.

# Convolutional Layer
The Conv.py module in the Layers folder contains the implementation of the Conv layer, which performs convolution operations. It supports both 1D and 2D convolutions and provides the forward(input_tensor) and backward(error_tensor) methods.

# Pooling Layer
The Pooling.py module in the Layers folder implements the Pooling layer, specifically the max-pooling operation. It provides the forward(input_tensor) and backward(error_tensor) methods.

# Contributing
Contributions to this framework are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue.
