import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import layer as ly


class NeuralNetwork:
    """
    A class representing a neural network.

    Attributes:
    - layers (list): List of layers in the neural network.
    - input_size (int): Size of the input layer.
    - output_size (int): Size of the output layer.
    - loss_fn (object): Loss function used for training the network.
    - loss_ls (list): List to store the training loss per epoch.
    - acc_ls (list): List to store the accuracy per epoch.

    Methods:
    - __init__(layers, input_size, output_size, loss_fn): Initializes the NeuralNetwork class.
    - forward(x, mode="train"): Forward propagate across each layer.
    - backward(grad): Backpropagate across each layer in reverse order.
    - update(lr): Update the learning rate of each layer.
    - train(x, y, lr, epochs, batch_size, x_test, y_test, lr_decay=1, epoch_delay=50): Train the neural network using minibatched SGD and an optional learning rate decay.
    - accuracy(x, y): Calculate the accuracy of the model given input x and labels y.
    - predict(x): Predict the class index for input x.
    - plot_loss(): Plot a loss per epoch line graph.
    - plot_acc(): Plot an accuracy per epoch line graph.
    - test_pred(index, x, y): Plot the image at index in the test set and print the predicted label.
    - save(filename): Save the model into a Pickle file.
    """

    def __init__(
        self, layers, input_size, output_size, loss_fn, norm=None, norm_alpha=1e-5
    ):
        """
        Initializes a neural network object.

        Args:
        - layers (list): List of integers representing the number of neurons in each layer.
        - input_size (int): Number of input features.
        - output_size (int): Number of output classes.
        - loss_fn (function): Loss function used for training the network.
        """

        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.loss_fn = loss_fn
        self.norm = norm
        self.norm_alpha = norm_alpha
        self.loss_ls = []
        self.acc_ls = []

    def forward(self, x, mode="train"):
        """
        Forward propagate across each layer.

        Args:
        - x (ndarray): Input data.
        - mode (str): Mode of operation. Default is "train".
        """

        for layer in self.layers:
            x = layer.forward(x, mode)
        return x

    def backward(self, grad):
        """
        Backpropagate across each layer in reverse order.

        Args:
        - grad (ndarray): Gradient of the loss function.
        """

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        """
        Update the learning rate of each layer.

        Args:
        - lr (float): Learning rate.
        """

        for layer in self.layers:
            if isinstance(layer, ly.LinearLayer):
                # Additional weight decay term if linear layer
                layer.update(lr, norm=self.norm, norm_alpha=self.norm_alpha)
            else:
                layer.update(lr)

    def train(self, x, y, lr, epochs, batch_size, x_test, y_test, lr_decay=1):
        """
        Train the neural network with minibatched SGD and an optional learning rate decay.
        Display epoch, lr, training loss, and test data accuracy with a tqdm progress bar.

        Args:
        - x (ndarray): Training data.
        - y (ndarray): Training labels.
        - lr (float): Initial earning rate.
        - epochs (int): Number of training epochs.
        - batch_size (int): Size of each minibatch.
        - x_test (ndarray): Test data.
        - y_test (ndarray): Test labels.
        - lr_decay (float): Learning rate decay factor. Default is 1. Exponential decay.
        """

        # Record training loss and accuracy per epoch
        self.loss_ls = []
        self.acc_ls = []

        # Progress bar for tracking minibatches
        pbar = tqdm(range(epochs))

        x_len = x.shape[0]  # No. of training examples

        for epoch in pbar:
            # Shuffle x and y
            indexes = np.arange(x_len)
            np.random.shuffle(indexes)
            x = x[indexes]
            y = y[indexes]

            # Progress bar for tracking minibatches
            # pbar = tqdm(range(x_len // batch_size))

            # One entire passthrough of data
            for i in range(x_len // batch_size):
                # Select batch's features and labels
                x_batch = x[
                    i * batch_size : (i + 1) * batch_size
                ]  # (batch_size, input_size)
                y_batch = y[
                    i * batch_size : (i + 1) * batch_size
                ]  # (batch_size, output_size)

                # Forward propagate, set mode to train to accumulate gradients
                y_pred = self.forward(x_batch, mode="train")

                # Calculate training loss
                loss = self.loss_fn.get_loss(y_pred, y_batch)

                # Backprop loss
                self.backward(self.loss_fn.get_grad(y_pred, y_batch))

                # Calculate test accuracy
                acc = self.accuracy(x_test, y_test)

                # Store loss and accuracy (to plot later)
                self.loss_ls.append(loss)
                self.acc_ls.append(acc)

                # Calculate new learning rate (according to exponential decay schedule)
                new_lr = (lr_decay**epoch) * lr

                # Update layers' learning rate with new learning rate
                self.update(new_lr)

                # Update Epoch, Loss and Accuracy in progress bar description
                pbar.set_description(
                    f"Epoch: {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f} LR: {new_lr:.4f}"
                )

        pbar.close()

    def accuracy(self, x, y):
        """
        Calculate the accuracy (0.0 - 1.0) of the model given input x and labels y.

        Args:
        - x (ndarray): Input data.
        - y (ndarray): Labels.

        Returns:
        - float: Accuracy of the model.
        """

        y_pred = self.forward(x, mode="test")
        return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / y.shape[0]

    def predict(self, x):
        """
        Predict the class index for input x.

        Args:
        - x (ndarray): Input data.

        Returns:
        - ndarray: Predicted class index.
        """

        y_pred = self.forward(x, mode="test")
        return np.argmax(y_pred, axis=1)

    def plot_loss(self):
        """
        Plot a loss per epoch line graph.
        """

        plt.plot(self.loss_ls)
        plt.title(f"{self.loss_fn.get_name()} per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(self.loss_fn.get_name())
        plt.show()

    def plot_acc(self):
        """
        Plot an accuracy per epoch line graph.
        """

        plt.plot(self.acc_ls)
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()

    def test_pred(self, index, x, y):
        """
        Plot the image at index in the test set and print the predicted label.

        Args:
        - index (int): Index of the image in the test set.
        - x (ndarray): Test data.
        - y (ndarray): Test labels.
        """

        y_pred = self.forward(x[None, index, :], mode="test")

        print("Predicted:", np.argmax(y_pred, 1)[0])
        print("Actual:", np.argmax(y[index, :]))

        plt.figure()
        plt.imshow(x[None, index, :].reshape((28, 28)), cmap="gray")

    def save(self, filename):
        """
        Save the model into a Pickle file.

        Args:
        - filename (str): Name of the file to save the model.
        """

        with open(filename, "wb+") as f:
            pickle.dump(self, f)


def score_ensemble_mean(models, x, y):
    """
    Scores the ensemble of models with soft voting (highest mean probability)

    Args:
    - models (list): A list of models to be used in the ensemble.
    - x (ndarray): Input data for prediction.
    - y (ndarray): True labels for the input data.

    Returns:
    - float: The accuracy score of the ensemble of models.
    """
    # Create empty y_pred
    y_pred = np.zeros((x.shape[0], models[0].output_size))

    for model in models:
        # Add probabilities prediction from each model
        y_pred += model.forward(x, mode="test")

    # Sum up correct predictions and divide by no. of predictions
    return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / y.shape[0]


def score_ensemble_mode(models, x, y):
    """
    Scores of the ensemble of models with hard voting (most votes)

    Args:
    - models (list): A list of models to be used in the ensemble.
    - x (ndarray): Input data for prediction.
    - y (ndarray): True labels for the input data.

    Returns:
    - float: The accuracy score of the ensemble of models.
    """

    # Create empty y_pred
    y_pred = np.zeros((x.shape[0], models[0].output_size))

    for model in models:
        # Add one-hot encoded prediction from each model
        y_pred += np.eye(y.shape[1])[np.argmax(model.forward(x, mode="test"), axis=1)]

    # Sum up correct predictions and divide by no. of predictions
    return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / y.shape[0]


def load_model(filename):
    """
    Load model from Pickle file

    Args:
    - filename (str): The path to the Pickle file containing the model.

    Returns:
    - object: The loaded model object.
    """

    with open(filename, "rb") as f:
        return pickle.load(f)
