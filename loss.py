import numpy as np


class CrossEntropy:
    """
    A class that represents the Cross Entropy loss function.

    Attributes:
    - min_val (float): Minimum value to prevent log(<=0).
    - set_mean (bool): Whether to divide the loss by batch_size.
    - name (str): The name of the loss function.

    Methods:
    - __init__(min_val=1e-9, set_mean=True): Initializes the CrossEntropy loss function.
    - get_name(): Returns the name of the loss function.
    - get_loss(y_pred, y_res): Calculates the CrossEntropy loss.
    - get_grad(y_pred, y_res): Calculates the gradient of the CrossEntropy loss.
    """

    def __init__(self, min_val=1e-9, set_mean=True):
        """
        Initializes the CrossEntropy loss function.

        Args:
        - min_val (float, optional): Minimum value to prevent log(<=0). Defaults to 1e-9.
        - set_mean (bool, optional): Whether to divide the loss by batch_size. Defaults to True.
        """

        self.min_val = min_val
        self.set_mean = set_mean
        self.name = "Cross Entropy Loss"

    def get_name(self):
        """
        Returns the name of the loss function.

        Returns:
        - str: The name of the loss function.
        """

        return self.name

    def get_loss(self, y_pred, y_res):
        """
        Calculates the CrossEntropy loss.

        Args:
        - y_pred (ndarray): The predicted values.
        - y_res (ndarray): The true values.

        Returns:
            float: The calculated loss.
        """

        return -np.sum(y_res * np.log(np.maximum(y_pred, self.min_val))) / (
            y_pred.shape[0] if self.set_mean else 1
        )

    def get_grad(self, y_pred, y_res):
        """
        Calculates the gradient of the CrossEntropy loss.

        Args:
        - y_pred (ndarray): The predicted values.
        - y_res (ndarray): The true values.

        Returns:
        - ndarray: The calculated gradient.
        """

        return y_pred - y_res
