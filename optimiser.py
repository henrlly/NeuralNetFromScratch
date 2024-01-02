import numpy as np


class Optimiser:
    """
    Base class for an optimiser.
    """

    def __init__(self, lr, norm=None, norm_alpha=1e-5):
        """
        Initialises the optimiser.
        Sets the learning rate and initialises iteration count to 0.

        Args:
        - lr (float): The learning rate.
        - norm (str): The type of regularisation to use. "l1", "l2" or None.
        - norm_alpha (float): The regularisation strength.
        """
        self.lr = lr
        self.iteration = 0
        self.vars = []
        self.params = None
        self.norm = norm
        self.norm_alpha = norm_alpha

    def norm_update(self):
        """
        Updates the parameters with regularisation.
        """
        # Regularise each weight
        for param in self.params:
            if self.norm == "l1":
                # Regularisation term = abs(weight)
                # Gradient = sign(weight) or weight / abs(weight)
                # Results in sparse weights
                param -= self.lr * self.norm_alpha * np.sign(param)

            elif self.norm == "l2":
                # Regularisation term = 1/2 * (weight ** 2)
                # Gradient = weight
                # Reduces magnitude of weights with large magnitude
                param -= self.lr * self.norm_alpha * param

    def set_params(self, params):
        """
        Sets the parameters to be optimised.

        Args:
        - params (list[ndarray]): List of parameters to be optimised.
        """
        # Initialise variables with zeros and with same shape as the parameters
        for p in params:
            for var in self.vars:
                var.append(np.zeros(p.shape))

        # Save parameters
        self.params = params

        # Free memory
        del self.vars

    def get_lr(self):
        """
        Returns the learning rate.
        """
        return self.lr

    def train_update(self):
        """
        Updates the parameters before each training iteration.
        For Nesterov momentum's interim update.
        """

    def train_unupdate(self):
        """
        Reverts the parameters after each training iteration.
        For Nesterov momentum's interim update.
        """

    def update(self, grad):
        """
        Updates the parameters.

        Args:
        - grad (list[ndarray]): List of gradients.
        """
        raise NotImplementedError


class SGD(Optimiser):
    """
    Stochastic Gradient Descent optimiser.
    """

    def __init__(
        self,
        lr,
        momentum=0.0,
        nesterov=False,
        lr_decay_iteration=0,  # Linearly decay until this interation
        lr_decay_min=0.0,  # Minimum learning rate to decay until
        norm=None,
        norm_alpha=1e-5,
    ):
        super().__init__(lr, norm, norm_alpha)
        self.original_lr = lr
        self.lr_decay_iteration = lr_decay_iteration
        self.lr_decay_min = lr_decay_min
        self.momentum = momentum
        self.v = []  # Velocity
        self.vars = [self.v]
        self.nesterov = nesterov

    def update(self, grad):
        # Decoupled weight decay
        if self.norm is not None:
            self.norm_update()

        for i, param in enumerate(self.params):
            if self.momentum > 0:
                # Compute velocity update
                self.v[i] = self.momentum * self.v[i] - self.lr * grad[i]
                param += self.v[i]
            else:
                # Update parameters
                param -= self.lr * grad[i]

        # Linearly decay learning rate until lr_decay_iteration
        if self.lr_decay_iteration > 0 and self.iteration < self.lr_decay_iteration:
            self.lr_decay()

            # Update iteration count
            self.iteration += 1

    def lr_decay(self):
        """
        Linearly decays the learning rate.
        """
        # Linearly decay learning rate
        self.lr = (1 - self.iteration / self.lr_decay_iteration) * self.original_lr + (
            self.iteration / self.lr_decay_iteration
        ) * self.lr_decay_min

    def train_update(self):
        if self.nesterov and self.momentum > 0:
            for i, param in enumerate(self.params):
                param += self.momentum * self.v[i]

    def train_unupdate(self):
        if self.nesterov and self.momentum > 0:
            for i, param in enumerate(self.params):
                param -= self.momentum * self.v[i]


class AdaGrad(Optimiser):
    """
    AdaGrad optimiser.
    """

    def __init__(self, lr, epsilon=1e-8, norm=None, norm_alpha=1e-5):
        super().__init__(lr, norm, norm_alpha)
        self.epsilon = epsilon
        self.r = []  # Accumulated squared gradient
        self.vars = [self.r]

    def update(self, grad):
        # Decoupled weight decay
        if self.norm is not None:
            self.norm_update()

        for i, param in enumerate(self.params):
            # Accumulate squared gradient
            self.r[i] += grad[i] ** 2
            # Update parameters
            param -= self.lr * grad[i] / (np.sqrt(self.r[i]) + self.epsilon)


class RMSProp(Optimiser):
    """
    RMSProp optimiser.
    """

    def __init__(
        self,
        lr,
        decay_rate=0.9,
        epsilon=1e-8,
        nesterov_momentum=0.0,
        norm=None,
        norm_alpha=1e-5,
    ):
        super().__init__(lr, norm, norm_alpha)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.r = []  # Accumulated squared gradient
        self.v = []  # Velocity
        self.vars = [self.r, self.v]
        self.nesterov_momentum = nesterov_momentum

    def update(self, grad):
        # Decoupled weight decay
        if self.norm is not None:
            self.norm_update()

        for i, param in enumerate(self.params):
            # Accumulate squared gradient
            self.r[i] = (
                self.decay_rate * self.r[i] + (1 - self.decay_rate) * grad[i] ** 2
            )

            if self.nesterov_momentum > 0:
                # Compute velocity update
                self.v[i] = self.nesterov_momentum * self.v[i] - self.lr * grad[i] / (
                    np.sqrt(self.r[i]) + self.epsilon
                )
                param += self.v[i]
            else:
                # Update parameters
                param -= self.lr * grad[i] / (np.sqrt(self.r[i]) + self.epsilon)

    def train_update(self):
        if self.nesterov_momentum > 0:
            for i, param in enumerate(self.params):
                param += self.nesterov_momentum * self.v[i]

    def train_unupdate(self):
        if self.nesterov_momentum > 0:
            for i, param in enumerate(self.params):
                param -= self.nesterov_momentum * self.v[i]


class Adam(Optimiser):
    """
    Adam optimiser.
    """

    def __init__(
        self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, norm=None, norm_alpha=1e-5
    ):
        super().__init__(lr, norm, norm_alpha)
        self.beta1 = beta1  # Exponential decay rate for the first moment estimates
        self.beta2 = beta2  # Exponential decay rate for the second moment estimates
        self.epsilon = epsilon
        self.s = []  # First moment estimate
        self.r = []  # Second moment estimate
        self.vars = [self.s, self.r]

    def update(self, grad):
        # Decoupled weight decay
        if self.norm is not None:
            self.norm_update()

        # Update iteration count
        self.iteration += 1

        for i, param in enumerate(self.params):
            # Update biased first moment estimate
            self.s[i] = self.beta1 * self.s[i] + (1 - self.beta1) * grad[i]
            # Update biased second moment estimate
            self.r[i] = self.beta2 * self.r[i] + (1 - self.beta2) * grad[i] ** 2
            # Compute bias-corrected first moment estimate
            s_hat = self.s[i] / (1 - self.beta1 ** (self.iteration))
            # Compute bias-corrected second moment estimate
            r_hat = self.r[i] / (1 - self.beta2 ** (self.iteration))
            # Update parameters
            param -= self.lr * s_hat / (np.sqrt(r_hat) + self.epsilon)
