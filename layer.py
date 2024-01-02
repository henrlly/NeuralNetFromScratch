import numpy as np


class Layer:
    """
    Base class for a layer in a neural network.
    """

    def __init__(self):
        """
        Initialises a layer. Sets trainable to False and x (input) to None.
        """
        self.x = None
        self.trainable = False

    def forward(self, x, mode="train"):
        """
        Performs a forward pass of the layer.

        Args:
        - x (ndarray): The input to the layer.
        - mode (str, optional): The mode of the layer. Defaults to "train".

        Returns:
        - ndarray: The output of the layer.
        """
        raise NotImplementedError

    def backward(self, grad):
        """
        Performs a backward pass of the layer.

        Args:
        - grad (ndarray): The gradient of the loss with respect to the output of the layer.

        Returns:
        - ndarray: The gradient of the loss with respect to the input of the layer.
        """
        raise NotImplementedError

    def get_grads(self):
        """
        Returns the gradients of the layer.

        Returns:
        - list[ndarray]: The gradients of the layer.
        """
        return []

    def get_params(self):
        """
        Returns the parameters of the layer.

        Returns:
        - list[ndarray]: The parameters of the layer.
        """
        return []

    def is_trainable(self):
        """
        Returns whether the linear layer is trainable.

        Returns:
        - bool: Whether the linear layer is trainable.
        """
        return self.trainable


class LinearLayer(Layer):
    """
    Linear layer with weights and optional biases.
    """

    def __init__(self, input_size, output_size, biases=True):
        """
        Initialises a linear layer with random weights and optional biases.
        Trainable by default.

        Args:
        - input_size (int): The size of the input features.
        - output_size (int): The size of the output features.
        - biases (bool, optional): Whether to include biases in the layer. Defaults to True.
        """
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        # Initialise weights and biases with from uniform distribution over [-0.5, 0.5).
        # High entropy initialisation to "break symmetry".
        # If two units the same initial parameters,
        # then a deterministic learning algorithm applied to a deterministic cost and model
        # will constantly update both of these units in the same way.
        self.w = np.random.rand(output_size, input_size) - 0.5

        # Initialise biases to constant zero (heuristic)
        self.b = np.zeros(output_size)
        self.grad_w = None
        self.grad_b = None
        self.biases = biases

    def forward(self, x, mode="train"):
        self.x = x

        # f(x) = x . Weights.T + Biases
        return x.dot(self.w.T) + (self.b if self.biases else 0)

    def backward(self, grad):
        # Calculate gradients for weights
        # Divide by batch size to stabalise gradients
        self.grad_w = grad.T.dot(self.x) / grad.shape[0]

        if self.biases:
            # Calculate gradients for biases
            # Divide by batch size to stabalise gradients
            self.grad_b = np.sum(grad, axis=0) / grad.shape[0]

        return grad.dot(self.w)

    def get_grads(self):
        if self.biases:
            return [self.grad_w, self.grad_b]
        else:
            return [self.grad_w]

    def get_params(self):
        if self.biases:
            return [self.w, self.b]
        else:
            return [self.w]


class DropoutLayer(Layer):
    """
    Dropout layer.
    Dropout masks neurons with probability p, creating an (exponentially large) ensemble of sub-networks.
    """

    def __init__(self, p):
        """
        Initialises a dropout layer.

        Args:
        - p (float): The probability of dropping out a neuron.
        """
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x, mode="train"):
        if mode == "train":
            # Random sample from uniform distribution over [0, 1)
            self.mask = np.random.rand(*x.shape) > self.p

            # Zero out neurons with probability p
            return x * self.mask
        else:
            # In test mode, output is multiplied by 1 - p (retention probability)
            # so expected output is the same as the actual output
            # See: Weight scaling inference rule
            return x * (1 - self.p)

    def backward(self, grad):
        # Only propagate gradients whose neurons were not zeroed out
        return grad * self.mask


class ReLULayer(Layer):
    """
    ReLU (Rectified Linear Unit) activation function.
    """

    def forward(self, x, mode="train"):
        self.x = x

        # Zero out negative neurons
        return np.maximum(x, 0)

    def backward(self, grad):
        # Only propagate gradients whose neurons were not zeroed out
        return grad * (self.x > 0)


class SoftmaxLayer(Layer):
    """
    Softmax activation function.
    """

    def forward(self, x, mode="train"):
        # Stabilize exp (by preventing very large or very small x)
        e_x = np.exp(x - np.max(x))
        return e_x / np.repeat(e_x.sum(axis=1), x.shape[1]).reshape(x.shape)

    def backward(self, grad):
        return grad


class TanhLayer(Layer):
    """
    Tanh activation function.
    """

    def forward(self, x, mode="train"):
        self.x = x
        return np.tanh(x)

    def backward(self, grad):
        return grad * (1 - np.tanh(self.x) ** 2)


class SigmoidLayer(Layer):
    """
    Sigmoid activation function.
    """

    def forward(self, x, mode="train"):
        return 1 / (1 + np.exp(-x))

    def backward(self, grad):
        return grad


class BatchNorm1DLayer(Layer):
    """
    Batch Normalization 1D Layer.
    """

    def __init__(self, input_size, eps=1e-5, momentum=0.1, running=True):
        """
        Initialises a Batch Normalization 1D Layer.
        Trainable by default.

        Args:
        - input_size (int): The size of the input features.
        - eps (float, optional): The epsilon value. Defaults to 1e-5.
        - momentum (float, optional): The momentum value. Defaults to 0.1.
        - running (bool, optional): Whether to use running mean and var. Defaults to True.
        """
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.eps = eps
        self.momentum = momentum
        self.running = running

        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)

        self.x_norm = self.x = self.dgamma = self.dbeta = None
        self.mean_norm = self.std_dev = self.mean = self.var = None

    def forward(self, x, mode="train"):
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

    def get_grads(self):
        return [self.dgamma, self.dbeta]

    def get_params(self):
        return [self.gamma, self.beta]
