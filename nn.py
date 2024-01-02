import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy


class NeuralNetwork:
    """
    A class representing a neural network.
    """

    def __init__(
        self,
        layers,
        input_size,
        output_size,
    ):
        """
        Initialises a neural network object.

        Args:
        - layers (list): List of Layers.
        - input_size (int): Number of input features.
        - output_size (int): Number of output classes.
        """

        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.loss_fn = None
        self.norm = None
        self.norm_alpha = None
        self.loss_ls = []
        self.acc_ls = []
        self.val_loss_ls = []
        self.val_acc_ls = []
        self.optimiser = []
        self.lr = None

    def forward(self, x, mode="train"):
        """
        Forward propagate across each layer.

        Args:
        - x (ndarray): Input data.
        - mode (str): Mode of operation. Default is "train".
        """
        i = 0
        for layer in self.layers:
            if layer.is_trainable() and mode == "train":
                # Interim update params (for Nesterov momentum)
                self.optimiser[i].train_update()

            x = layer.forward(x, mode)

            if layer.is_trainable() and mode == "train":
                # Interim unupdate params (for Nesterov momentum)
                self.optimiser[i].train_unupdate()
                i += 1
        return x

    def backward(self, grad):
        """
        Backpropagate across each layer in reverse order.

        Args:
        - grad (ndarray): Gradient of the loss function.
        """

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self):
        """
        Update the learning rate of each layer.

        Args:
        - lr (float): Learning rate.
        """
        i = 0
        for layer in self.layers:
            if layer.is_trainable():
                self.optimiser[i].update(layer.get_grads())
                i += 1

    def init_optimiser(self, optimiser):
        """
        Initialise the optimiser for the neural network.

        Args:
        - optimiser (object): Optimiser object.
        """
        self.optimiser = []
        self.lr = optimiser.get_lr()
        for layer in self.layers:
            if layer.is_trainable():
                # Create deepcopy of optimiser for each trainable layer
                optim = deepcopy(optimiser)
                # Set param for each optimiser
                optim.set_params(layer.get_params())
                self.optimiser.append(optim)

    def train(
        self,
        x,
        y,
        epochs,
        batch_size,
        x_val,
        y_val,
        early_stopping=-1,
        loss_fn=None,
        norm=None,
        norm_alpha=0.0,
        optimiser=None,
    ):
        """
        Train the neural network with minibatched SGD and an optional learning rate decay.
        Display epoch, training loss and accuracy, and validation loss and accuracy with a tqdm progress bar.

        Args:
        - x (ndarray): Input data for training.
        - y (ndarray): Target data for training.
        - epochs (int): Number of training epochs.
        - batch_size (int): Size of each training batch.
        - x_val (ndarray): Input data for validation.
        - y_val (ndarray): Target data for validation.
        - early_stopping (int, optional): Number of epochs to wait for improvement in validation loss before early stopping. Defaults to -1.
        - loss_fn (LossFunction, optional): Loss function to calculate the training loss. Defaults to None.
        - norm (Normalization, optional): Weight normalization method for linear layer. Defaults to None.
        - norm_alpha (float, optional): Alpha value for weight normalization. Defaults to 0.0.
        - optimiser (Optimiser, optional): Optimiser for updating weights. Defaults to None.
        """
        # Check if there's validation data
        has_val = x_val is not None and y_val is not None

        # Set early stopping to -1 if no validation data
        early_stopping = -1 if not has_val else early_stopping

        # Initialise optimiser
        self.init_optimiser(optimiser)

        # Initialise loss function
        self.loss_fn = loss_fn

        # Initialise weight normalisation (for linear layer)
        self.norm = norm
        self.norm_alpha = norm_alpha

        # Record training loss and accuracy per epoch
        self.loss_ls = []
        self.acc_ls = []
        self.val_loss_ls = []
        self.val_acc_ls = []

        # Min validation loss
        min_loss = np.inf

        # Count rounds for early stopping
        stopping_rounds = 0

        # Progress bar for tracking minibatches
        pbar = tqdm(range(epochs))

        x_len = x.shape[0]  # No. of training examples

        for epoch in pbar:
            # Break if early stopping
            if early_stopping != -1 and stopping_rounds > early_stopping:
                break

            # Shuffle x and y
            indexes = np.arange(x_len)
            np.random.shuffle(indexes)
            x = x[indexes]
            y = y[indexes]

            # Track total validation loss per epoch
            total_val_loss = 0

            # One entire passthrough of data - one epoch
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

                # Backpropagate loss
                self.backward(self.loss_fn.get_grad(y_pred, y_batch))

                # Update weights
                self.update()

                # Calculate training accuracy
                acc = self.accuracy(y_pred, y_batch)

                # Store training loss and accuracy (to plot later)
                self.loss_ls.append(loss)
                self.acc_ls.append(acc)

                if has_val:
                    # Get validation predictions
                    val_pred = self.forward(x_val, mode="test")

                    # Calculate validation loss
                    val_loss = self.loss_fn.get_loss(val_pred, y_val)

                    # Calculate validation accuracy
                    val_acc = self.accuracy(val_pred, y_val)

                    # Store validation loss and accuracy (to plot later)
                    self.val_loss_ls.append(val_loss)
                    self.val_acc_ls.append(val_acc)

                    # Add val loss to total val loss per epoch
                    total_val_loss += val_loss

                # Update Epoch, Loss and Accuracy in progress bar description
                pbar.set_description(
                    f"Epoch: {epoch+1}, Train Loss: {loss:.4f}, Train Acc: {acc:.4f}{f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if has_val else ""}"
                )

            # Early stopping
            if early_stopping != -1:
                if total_val_loss < min_loss:
                    min_loss = total_val_loss
                else:
                    stopping_rounds += 1

        pbar.close()

    def accuracy(self, y_pred, y):
        """
        Calculates the accuracy of the predicted values.

        Parameters:
        - y_pred (ndarray): The predicted values.
        - y (ndarray): The true values.

        Returns:
        - float: The accuracy of the predictions.
        """
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

        plt.plot(self.loss_ls, label="Training Loss")
        plt.plot(self.val_loss_ls, label="Validation Loss")
        plt.title(f"{self.loss_fn.get_name()} per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(self.loss_fn.get_name())
        plt.legend()
        plt.show()

    def plot_acc(self):
        """
        Plot an accuracy per epoch line graph.
        """

        plt.plot(self.acc_ls, label="Training Accuracy")
        plt.plot(self.val_acc_ls, label="Validation Accuracy")
        plt.title("Validation Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
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
