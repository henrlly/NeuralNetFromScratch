import numpy as np


class Loss:
    """
    Base class for a loss function.
    """

    def __init__(self):
        """
        Initialises the loss function.
        """
        self.name = "Loss"

    def get_name(self):
        """
        Returns the name of the loss function.

        Returns:
        - str: The name of the loss function.
        """

        return self.name

    def get_loss(self, y_pred, y_res):
        """
        Calculates the loss.

        Args:
        - y_pred (ndarray): The predicted values.
        - y_res (ndarray): The true values.

        Returns:
        - float: The calculated loss.
        """

        raise NotImplementedError

    def get_grad(self, y_pred, y_res):
        """
        Calculates the gradient of the loss.

        Args:
        - y_pred (ndarray): The predicted values.
        - y_res (ndarray): The true values.

        Returns:
        - ndarray: The calculated gradient.
        """

        raise NotImplementedError


class CrossEntropy(Loss):
    """
    Cross Entropy loss function.
    """

    def __init__(self, min_val=1e-9):
        """
        Initialises the CrossEntropy loss function.

        Args:
        - min_val (float, optional): Minimum value to prevent log(<=0). Defaults to 1e-9.
        """
        super().__init__()
        self.min_val = min_val
        self.name = "Cross Entropy Loss"

    def get_loss(self, y_pred, y_res):
        return (
            -np.sum(y_res * np.log(np.maximum(y_pred, self.min_val))) / y_pred.shape[0]
        )

    def get_grad(self, y_pred, y_res):
        return y_pred - y_res


class MSE(Loss):
    """
    Mean Squared Error loss function.
    """

    def __init__(self):
        """
        Initialises the MSE loss function.
        """
        super().__init__()
        self.name = "Mean Squared Error Loss"

    def get_loss(self, y_pred, y_res):
        # Average over batch size and number of output neurons
        return np.sum((y_pred - y_res) ** 2) / (2 * np.prod(y_pred.shape))

    def get_grad(self, y_pred, y_res):
        # Average over number of output neurons
        return (y_pred - y_res) / np.prod(y_pred.shape[1:])
