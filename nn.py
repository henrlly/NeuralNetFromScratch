import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

class LinearLayer:
    def __init__(self, input_size, output_size, biases=True):
        self.input_size = input_size
        self.output_size = output_size
        self.w = np.random.rand(output_size, input_size) - 0.5
        self.b = np.random.rand(output_size, 1) - 0.5
        self.x = None
        self.grad_w = None
        self.grad_b = None
        self.biases = biases

    def forward(self, x):
        self.x = x
        return self.w.dot(x) + self.b
    
    def backward(self, grad):
        self.grad_w = grad.dot(self.x.T) / grad.shape[1]
        if self.biases:
            self.grad_b = np.sum(grad) / grad.shape[1]
        return self.w.T.dot(grad) # Next layer error
    
    def update(self, lr):
        self.w -= lr * self.grad_w
        if self.biases:
            self.b -= lr * self.grad_b

class DropoutLayer:
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, x):
        self.mask = np.random.rand(*x.shape) > self.p
        return x * self.mask
    
    def backward(self, grad):
        return grad * self.mask
    
    def update(self, lr):
        pass
    
class ReLuLayer:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)
    
    def backward(self, grad):
        return grad * (self.x > 0)
    
    def update(self, lr):
        pass
    
class SoftmaxLayer:
    def forward(self, x):
        return np.exp(x) / sum(np.exp(x))
    
    def backward(self, grad):
        return grad
    
    def update(self, lr):
        pass

class TanhLayer:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.tanh(x)
    
    def backward(self, grad):
        return grad * (1 - np.tanh(self.x) ** 2)
    
    def update(self, lr):
        pass

class SigmoidLayer:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))
    
    def backward(self, grad):
        return grad * (1 - 1 / (1 + np.exp(-self.x)))
    
    def update(self, lr):
        pass

class NeuralNetwork:
    def __init__(self, layers, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.loss_ls =[]
        self.acc_ls = []


    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def loss(self, y_pred, y_res):
        y_pred = np.maximum(y_pred, 1e-9)
        return np.sum(-y_res * np.log(y_pred)) / y_pred.shape[1]
    
    def grad(self, y_pred, y_res):
        return -y_res * np.log(y_pred)

    def train(self, x, y, lr, epochs, batch_size, x_test, y_test):
        self.loss_ls =[]
        self.acc_ls = []
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            x_batch = x[:batch_size]
            y_batch = y[:batch_size]
            y_pred = self.forward(x_batch.T)
            y_res = y_batch.T
            loss = self.loss(y_pred, y_res)
            self.backward(y_pred - y_res)
            self.update(lr)
            x = np.roll(x, batch_size, axis=0)
            y = np.roll(y, batch_size, axis=0)
            
            acc = self.accuracy(x_test, y_test)
            self.loss_ls.append(loss)
            self.acc_ls.append(acc)
            # if epoch and epoch % 50 == 0: # learning rate decay
            #     lr /= 2

            pbar.set_description('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, acc))
        pbar.close()
    
    def accuracy(self, x, y):
        y_pred = self.forward(x.T)
        y_res = y.T
        return np.sum(np.argmax(y_pred, axis=0) == np.argmax(y_res, axis=0)) / y.shape[0]
    
    def predict(self, x):
        y_pred = self.forward(x.T)
        return np.argmax(y_pred, axis=0)
    
    def plot_loss(self):
        plt.plot(self.loss_ls)
        plt.show()

    def plot_acc(self):
        plt.plot(self.acc_ls)
        plt.show()

    def test_pred(self, index, x):
        '''
        Plot the image at index in the test set and print the predicted label
        '''
        y_pred = self.forward(x.T[:, index, None])
        y_res = np.argmax(y_pred, 0)

        print(y_res)
        plt.figure()
        plt.imshow(x.T[:, index, None].reshape((28, 28)), cmap='gray')

    def save(self, filename):
        with open(filename, 'wb+') as f:
            pickle.dump(self, f)
        

def score_ensemble_mean(models, x, y):
    y_pred = np.zeros((models[0].output_size, x.shape[0]))
    for model in models:
        y_pred += model.forward(x.T)
    return np.sum(np.argmax(y_pred, axis=0) == np.argmax(y.T, axis=0)) / y.shape[0]

def score_ensemble_mode(models, x, y):
    y_pred = np.zeros((models[0].output_size, x.shape[0]))
    for model in models:
        y_pred += one_hot_encode(np.argmax(model.forward(x.T), axis=0)).T
    return np.sum(np.argmax(y_pred, axis=0) == np.argmax(y.T, axis=0)) / y.shape[0]

def one_hot_encode(preds):
    encoding = np.zeros((len(preds), 10))
    for i, val in enumerate(preds):
        encoding[i, val] = 1
    return encoding

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

