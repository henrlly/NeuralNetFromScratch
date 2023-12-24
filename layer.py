import numpy as np


class LinearLayer:
    """
    Linear layer with weights and optional biases.

    Attributes:
    - input_size (int): The size of the input features.
    - output_size (int): The size of the output features.
    - w (ndarray): The weight matrix of shape (output_size, input_size).
    - b (ndarray): The bias vector of shape (output_size).
    - x (ndarray): The input data of shape (batch_size, input_size).
    - grad_w (ndarray): The gradient of the loss with respect to the weights,
                        of shape (input_size, output_size).
    - grad_b (ndarray): The gradient of the loss with respect to the biases,
                        of shape (output_size).
    - biases (bool): Whether to include biases in the layer.

    Methods:
    - __init__(input_size, output_size, biases=True): Initializes a linear layer
                                                      with random weights and biases (optional).
    - forward(x, mode="train"): Performs forward propagation through the linear layer.
    - backward(grad): Performs backward propagation through the linear layer.
    - update(lr): Updates the weights and biases of the linear layer using gradient descent.
    """

    def __init__(self, input_size, output_size, biases=True):
        """
        Initializes a linear layer with random weights and biases (optional).

        Args:
        - input_size (int): The size of the input features.
        - output_size (int): The size of the output features.
        - biases (bool, optional): Whether to include biases in the layer. Defaults to True.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.w = np.random.rand(output_size, input_size) - 0.5
        self.b = np.random.rand(output_size) - 0.5
        self.x = None
        self.grad_w = None
        self.grad_b = None
        self.biases = biases

    def forward(self, x, mode="train"):
        """
        Performs forward propagation through the linear layer.

        Args:
        - x (ndarray): The input data of shape (batch_size, input_size).
        - mode (str, optional): The mode of operation. Defaults to "train".

        Returns:
        - ndarray: The output of the linear layer of shape (batch_size, output_size).
        """
        self.x = x

        # f(x) = x . Weights.T + Biases
        return x.dot(self.w.T) + (self.b if self.biases else 0)

    def backward(self, grad):
        """
        Performs backward propagation through the linear layer.

        Args:
        - grad (ndarray): The gradient of the loss with respect to the output of the linear layer,
                            of shape (batch_size, output_size).

        Returns:
        - ndarray: The gradient of the loss with respect to the input of the linear layer,
                     of shape (batch_size, input_size).
        """

        # Calculate gradients for weights
        # Divide by batch size to stabalise gradients
        self.grad_w = grad.T.dot(self.x) / grad.shape[0]

        if self.biases:
            # Calculate gradients for biases
            # Divide by batch size to stabalise gradients
            self.grad_b = np.sum(grad, axis=0) / grad.shape[0]

        return grad.dot(self.w)

    def update(self, lr):
        """
        Updates the weights and biases of the linear layer using gradient descent.

        Args:
        - lr (float): The learning rate.
        """
        self.w -= lr * self.grad_w
        if self.biases:
            self.b -= lr * self.grad_b


class DropoutLayer:
    """
    A class representing a dropout layer in a neural network.

    Attributes:
    - p (float): The probability of dropping out a neuron.
    - mask (ndarray): The mask used during forward propagation to drop out neurons.

    Methods:
    - forward(x, mode="train"): Performs forward propagation through the dropout layer.
    - backward(grad): Performs backward propagation through the dropout layer.
    - update(lr): Updates the parameters of the dropout layer.
    """

    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, x, mode="train"):
        """
        Performs forward propagation through the dropout layer.

        Args:
        - x (ndarray): The input to the dropout layer.
        - mode (str): The mode of operation. Default is "train".

        Returns:
        - ndarray: The output of the dropout layer.
        """
        if mode == "train":
            self.mask = np.random.rand(*x.shape) > self.p

            # Zero out neurons with probability p
            return x * self.mask
        else:
            return x

    def backward(self, grad):
        """
        Performs backward propagation through the dropout layer.

        Args:
        - grad (ndarray): The gradient of the loss with respect to the output of the dropout layer.

        Returns:
        - ndarray: The gradient of the loss with respect to the input of the dropout layer.
        """

        # Only propagate gradients whose neurons were not zeroed out
        return grad * self.mask

    def update(self, lr):
        """
        Updates the parameters of the dropout layer.

        Args:
        - lr (float): The learning rate.
        """
        pass


class ReLULayer:
    """
    ReLU (Rectified Linear Unit) activation function.
    """

    def __init__(self):
        self.x = None

    def forward(self, x, mode="train"):
        """
        Performs forward pass through the ReLU layer.

        Args:
        - x: Input data.
        - mode: Mode of operation ("train" or "test").

        Returns:
        - Output of the ReLU layer.
        """
        self.x = x

        # Zero out negative neurons
        return np.maximum(x, 0)

    def backward(self, grad):
        """
        Performs backward pass through the ReLU layer.

        Args:
        - grad: Gradient of the loss with respect to the output of the ReLU layer.

        Returns:
        - Gradient of the loss with respect to the input of the ReLU layer.
        """

        # Only propagate gradients whose neurons were not zeroed out
        return grad * (self.x > 0)

    def update(self, lr):
        """
        Updates the parameters of the ReLU layer.

        Args:
        - lr: Learning rate.
        """
        pass


class SoftmaxLayer:
    """
    Softmax activation function.
    """

    def forward(self, x, mode="train"):
        """
        Performs the forward pass of the softmax layer.

        Args:
        - x (ndarray): Input data.
        - mode (str): Mode of operation. Default is "train".

        Returns:
        - ndarray: Output of the softmax layer.
        """
        # Stabilize exp (by preventing very large or very small x)
        e_x = np.exp(x - np.max(x))
        return e_x / np.repeat(e_x.sum(axis=1), x.shape[1]).reshape(x.shape)

    def backward(self, grad):
        """
        Performs the backward pass of the softmax layer.

        Args:
        - grad (ndarray): Gradient of the loss with respect to the output of the softmax layer.

        Returns:
        - ndarray: Gradient of the loss with respect to the input of the softmax layer.
        """
        return grad

    def update(self, lr):
        """
        Updates the parameters of the softmax layer.

        Args:
        - lr (float): Learning rate.
        """
        pass


class TanhLayer:
    """
    Tanh activation function.
    """

    def __init__(self):
        self.x = None

    def forward(self, x, mode="train"):
        """
        Performs the forward pass of the TanhLayer.

        Args:
        - x: Input tensor.
        - mode: Mode of operation (default is "train").

        Returns:
        - ndarray: Output tensor after applying the tanh activation function.
        """
        self.x = x
        return np.tanh(x)

    def backward(self, grad):
        """
        Performs the backward pass of the TanhLayer.

        Args:
        - grad: Gradient tensor.

        Returns:
        - ndarray: Gradient tensor after backpropagation through the tanh activation function.
        """
        return grad * (1 - np.tanh(self.x) ** 2)

    def update(self, lr):
        """
        Updates the parameters of the TanhLayer.

        Args:
        - lr: Learning rate.
        """
        pass


class SigmoidLayer:
    """
    Sigmoid activation function.
    """

    def __init__(self):
        self.x = None

    def forward(self, x, mode="train"):
        """
        Performs the forward pass of the sigmoid layer.

        Args:
        - x: The input to the layer.
        - mode: The mode of operation (default is "train").

        Returns:
        - ndarray: The output of the sigmoid layer.
        """
        return 1 / (1 + np.exp(-x))

    def backward(self, grad):
        """
        Performs the backward pass of the sigmoid layer.

        Args:
        - grad: The gradient of the loss with respect to the output of the sigmoid layer.

        Returns:
        - ndarray: The gradient of the loss with respect to the input of the sigmoid layer.
        """
        return grad

    def update(self, lr):
        """
        Updates the parameters of the sigmoid layer.

        Args:
        - lr: The learning rate for the update.
        """
        pass


class BatchNorm1DLayer:
    """
    Batch Normalization 1D Layer.

    Args:
    - input_dim (int): The dimension of the input.
    - eps (float, optional): A small value added to the variance to avoid division by zero.
                             Defaults to 1e-5.
    - momentum (float, optional): The momentum for updating the running mean and variance.
                                  Defaults to 0.1.
    - running (bool, optional): Whether to use the running mean and variance during training.
                                Defaults to True.
    """

    def __init__(self, input_dim, eps=1e-5, momentum=0.1, running=True):
        self.input_dim = input_dim
        self.eps = eps
        self.momentum = momentum
        self.running = running

        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        self.gamma = np.ones(input_dim)
        self.beta = np.zeros(input_dim)

        self.x_norm = self.x = self.dgamma = self.dbeta = None
        self.mean_norm = self.std_dev = self.mean = self.var = None

    def forward(self, x, mode="train"):
        """
        Forward pass of the Batch Normalization 1D Layer.

        Args:
        - x (ndarray): The input array.
        - mode (str, optional): The mode of operation. Can be "train" or "test".
                                Defaults to "train".

        Returns:
        - ndarray: The output array after applying batch normalization.
        """
        if mode == "train":
            self.x = x
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)

            if self.running:
                # Update running mean and var
                self.running_mean = (
                    self.momentum * self.running_mean + (1 - self.momentum) * self.mean
                )
                self.running_var = (
                    self.momentum * self.running_var + (1 - self.momentum) * self.var
                )

            self.std_dev = np.sqrt(self.var + self.eps)
            self.mean_norm = x - self.mean

            # Normalize
            self.x_norm = self.mean_norm / self.std_dev
            x_norm = self.x_norm

        else:
            if self.running:
                # Use running var and mean
                var = self.running_var
                mean = self.running_mean

            else:
                # Calculate batch's var and mean
                var = np.var(x, axis=0)
                mean = np.mean(x, axis=0)

            x_norm = (x - mean) / np.sqrt(var + self.eps)

        out = self.gamma * x_norm + self.beta

        return out

    def backward(self, grad):
        """
        Backward pass of the Batch Normalization 1D Layer.

        Args:
        - grad (ndarray): The gradient array.

        Returns:
        - ndarray: The gradient array after backpropagation.
        """
        batch_size = grad.shape[0]

        self.dbeta = np.sum(grad, axis=0)
        # out_unbeta = out + beta
        # out = out_norm * gamma + beta
        # -> d_out / d_beta = 1

        self.dgamma = np.sum(grad * self.x_norm, axis=0)
        # out_unbeta = x_norm * gamma
        # -> d_out_unbeta / d_gamma = x_norm

        grad_norm = grad * self.gamma
        # out_unbeta = x_norm * gamma
        # -> d_out_unbeta / d_x_norm = gamma

        grad_mean_norm_1 = grad_norm / self.std_dev
        # var_norm = 1 / x_std_dev
        # x_norm = x_mean_norm * x_var_norm
        # -> d_x_norm / d_x_mean_norm = x_var_norm = 1 / x_std_dev

        grad_var_norm = np.sum(grad_norm * self.mean_norm, axis=0)
        # x_mean_norm = x - x_mean
        # x_norm = x_mean_norm * x_var_norm
        # -> d_out_norm / d_x_var_norm = x_mean_norm

        grad_std_dev = -grad_var_norm / self.var
        # 1 / x_std_dev = x_var_norm
        # -> d_out_var_norm / d_x_std_dev = -1 / (x_std_dev ** 2) = -1 / x_var

        grad_var = grad_std_dev * 0.5 / self.std_dev
        # x_std_dev = (x_var + eps) ** 0.5
        # -> d_out_std_dev / d_x_var = 0.5 * (1 / ((x_var + eps) ** 0.5)) = 0.5 / x_std_dev

        grad_sq = grad_var / batch_size
        # x_var = sum(x_sq) / batch_size
        # -> d_out_var / d_x_sq = 1 / batch_size

        grad_mean_norm_2 = grad_sq * 2 * self.mean_norm
        # x_sq = x_mean_norm ** 2
        # -> d_out_sq / d_x_mean_norm = 2 * x_mean_norm

        grad_mean_norm = (
            grad_mean_norm_1 + grad_mean_norm_2
        )  # 2 gradients accumulate into mean_norm

        grad_x_1 = grad_mean_norm
        # x_mean_norm = x - x_mean
        # -> d_out_mean_norm / d_x = 1

        grad_mean = -np.sum(grad_mean_norm, axis=0)
        # -> d_out_mean_norm / d_x_mean = -1

        grad_x_2 = grad_mean / batch_size
        # x_mean = sum(x) / batch_size
        # -> d_out_mean/ d_x = 1 / batch_size

        return grad_x_1 + grad_x_2  # 2 gradients accumulate into grad_x

    def update(self, lr):
        """
        Update the parameters of the Batch Normalization 1D Layer.

        Args:
        - lr (float): The learning rate.
        """
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta
