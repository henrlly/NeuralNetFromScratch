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

    def forward(self, x, mode='train'):
        self.x = x
        return self.w.dot(x) + self.b
    
    def backward(self, grad):
        self.grad_w = grad.dot(self.x.T) / grad.shape[1]
        if self.biases:
            self.grad_b = np.sum(grad, axis=1).reshape(self.output_size, 1) / grad.shape[1]
        return self.w.T.dot(grad) # Next layer error
    
    def update(self, lr):
        self.w -= lr * self.grad_w
        if self.biases:
            self.b -= lr * self.grad_b

class DropoutLayer:
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, x, mode='train'):
        if mode == 'train':
            self.mask = np.random.rand(*x.shape) > self.p
            return x * self.mask
        else:
            return x
    
    def backward(self, grad):
        return grad * self.mask
    
    def update(self, lr):
        pass
    
class ReLuLayer:
    def __init__(self):
        self.x = None

    def forward(self, x, mode='train'):
        self.x = x
        return np.maximum(x, 0)
    
    def backward(self, grad):
        return grad * (self.x > 0)
    
    def update(self, lr):
        pass
    
class SoftmaxLayer:
    def forward(self, x, mode='train'):
        return np.exp(x) / sum(np.exp(x))
    
    def backward(self, grad):
        return grad
    
    def update(self, lr):
        pass

class TanhLayer:
    def __init__(self):
        self.x = None

    def forward(self, x, mode='train'):
        self.x = x
        return np.tanh(x)
    
    def backward(self, grad):
        return grad * (1 - np.tanh(self.x) ** 2)
    
    def update(self, lr):
        pass

class SigmoidLayer:
    def __init__(self):
        self.x = None

    def forward(self, x, mode='train'):
        return  1 / (1 + np.exp(-x))
    
    def backward(self, grad):
        return grad
    
    def update(self, lr):
        pass

class BatchNorm1DLayer:
    def __init__(self, input_dim, eps=1e-5, momentum=0.9):
        self.input_dim = input_dim
        self.eps = eps
        self.momentum = momentum

        self.running_mean = np.zeros((input_dim, 1))
        self.running_var = np.ones((input_dim, 1))
        self.gamma = np.ones((input_dim, 1))
        self.beta = np.zeros((input_dim, 1))

    def forward(self, x, mode='train'):
        self.batch_size = x.shape[1]
        if mode == 'train':
            self.mean = np.mean(x, axis=1).reshape(self.input_dim, 1)
            self.var = np.var(x, axis=1).reshape(self.input_dim, 1)
            self.x = x

            # update running mean and var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            # normalize
            self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)

        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.gamma * self.x_norm + self.beta

        return out

    def backward(self, grad):
        dx_norm = grad * self.gamma
        std = np.sqrt(self.var + self.eps)

        self.dgamma = np.sum(grad * self.x_norm, axis=1).reshape(self.input_dim, 1)
        self.dbeta = np.sum(grad, axis=1).reshape(self.input_dim, 1)

        dx =  std * (self.batch_size * dx_norm - 
                      np.sum(dx_norm, axis=1).reshape(self.input_dim, 1) - 
                      self.x_norm * np.sum(dx_norm * self.x_norm, axis=1).reshape(self.input_dim, 1)) / self.batch_size
        return dx

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta
    
class BatchNorm2DLayer:
    def __init__(self, input_dim, eps=1e-5, momentum=0.9):
        self.input_dim = input_dim
        self.eps = eps
        self.momentum = momentum

        self.running_mean = np.zeros((input_dim, 1))
        self.running_var = np.ones((input_dim, 1))
        self.gamma = np.ones((input_dim, 1))
        self.beta = np.zeros((input_dim, 1))

    def forward(self, x, mode='train'):
        self.batch_size = x.shape[0]
        if mode == 'train':
            self.mean = np.mean(x, axis=(0, 2, 3)).reshape(self.input_dim, 1)
            self.var = np.var(x, axis=(0, 2, 3)).reshape(self.input_dim, 1)
            self.x = x

            # update running mean and var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            # normalize
            self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)

        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.gamma * self.x_norm + self.beta

        return out

    def backward(self, grad):
        dx_norm = grad * self.gamma
        std = np.sqrt(self.var + self.eps)

        self.dgamma = np.sum(grad * self.x_norm, axis=(0, 2, 3)).reshape(self.input_dim, 1)
        self.dbeta = np.sum(grad, axis=(0, 2, 3)).reshape(self.input_dim, 1)

        dx =  std * (self.batch_size * dx_norm - 
                      np.sum(dx_norm, axis=(0, 2, 3)).reshape(self.input_dim, 1) - 
                      self.x_norm * np.sum(dx_norm * self.x_norm, axis=(0, 2, 3)).reshape(self.input_dim, 1)) / self.batch_size
        return dx

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta



class Conv2DLayer:
    def __init__(self, input_size, output_size, kernel_size, stride, padding, biases=True):
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.biases = biases
        
        self.w = np.random.rand(output_size, input_size, kernel_size, kernel_size) - 0.5
        self.b = np.random.rand(output_size, 1) - 0.5
        self.x = None
        self.grad_w = None
        self.grad_b = None

    def forward(self, x, mode='train'):
        self.x = x
        self.batch_size = x.shape[0]
        
        self.x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        
        self.output_size = int((self.x_padded.shape[2] - self.kernel_size) / self.stride + 1)
        
        out = np.zeros((self.batch_size, self.output_size, self.output_size, self.output_size))
        
        for i in range(self.output_size):
            for j in range(self.output_size):
                out[:, i, j, :] = np.sum(self.x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, np.newaxis] * self.w[np.newaxis, :, :, :, :], axis=(2, 3, 4))
        
        if self.biases:
            out += self.b.T
        
        return out
    
    def backward(self, grad):
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        
        grad_padded = np.zeros_like(self.x_padded)
        
        for i in range(self.output_size):
            for j in range(self.output_size):
                self.grad_w += np.sum(grad[:, i, j, :, np.newaxis, np.newaxis, np.newaxis] * self.x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, np.newaxis], axis=0)
                grad_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += np.sum(grad[:, i, j, :, np.newaxis, np.newaxis, np.newaxis] * self.w[np.newaxis, :, :, :, :], axis=1)
        
        if self.biases:
            self.grad_b = np.sum(grad, axis=(0, 1, 2))[:, np.newaxis]
        
        return grad_padded[:, :, self.padding:self.padding+self.x.shape[2], self.padding:self.padding+self.x.shape[3]]
    
    def update(self, lr):
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b

class Conv1DLayer:
    def __init__(self, input_size, output_size, kernel_size, stride, padding, biases=True):
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.biases = biases
        
        self.w = np.random.rand(output_size, input_size, kernel_size) - 0.5
        self.b = np.random.rand(output_size, 1) - 0.5
        self.x = None
        self.grad_w = None
        self.grad_b = None

    def forward(self, x, mode='train'):
        self.x = x
        self.batch_size = x.shape[0]
        
        self.x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)), 'constant')
        
        self.output_size = int((self.x_padded.shape[2] - self.kernel_size) / self.stride + 1)
        
        out = np.zeros((self.batch_size, self.output_size, self.output_size))
        
        for i in range(self.output_size):
            out[:, i, :] = np.sum(self.x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, np.newaxis] * self.w[np.newaxis, :, :, np.newaxis], axis=(2, 3))
        
        if self.biases:
            out += self.b.T
        
        return out
    
    def backward(self, grad):
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        
        grad_padded = np.zeros_like(self.x_padded)
        
        for i in range(self.output_size):
            self.grad_w += np.sum(grad[:, i, :, np.newaxis, np.newaxis] * self.x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, np.newaxis], axis=0)
            grad_padded[:, :, i*self.stride:i*self.stride+self.kernel_size] += np.sum(grad[:, i, :, np.newaxis, np.newaxis] * self.w[np.newaxis, :, :, np.newaxis], axis=1)
        
        if self.biases:
            self.grad_b = np.sum(grad, axis=(0, 1))[:, np.newaxis]
        
        return grad_padded[:, :, self.padding:self.padding+self.x.shape[2]]
    
    def update(self, lr):
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b

class MaxPool2DLayer:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x = None
        self.indices = None

    def forward(self, x, mode='train'):
        self.x = x
        self.batch_size = x.shape[0]
        self.input_size = x.shape[1]
        self.output_size = int((self.input_size - self.kernel_size) / self.stride + 1)
        
        out = np.zeros((self.batch_size, self.output_size, self.output_size, self.input_size))
        self.indices = np.zeros((self.batch_size, self.output_size, self.output_size, self.input_size), dtype=np.int32)
        
        for i in range(self.output_size):
            for j in range(self.output_size):
                out[:, i, j, :] = np.max(self.x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size], axis=(2, 3))
                self.indices[:, i, j, :] = np.argmax(self.x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size], axis=(2, 3))
        
        return out
    
    def backward(self, grad):
        grad_padded = np.zeros_like(self.x)
        
        for i in range(self.output_size):
            for j in range(self.output_size):
                grad_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += grad[:, i, j, :, np.newaxis, np.newaxis, np.newaxis] * (self.indices[:, i, j, :, np.newaxis, np.newaxis, np.newaxis] == np.arange(self.kernel_size ** 2)[np.newaxis, np.newaxis, np.newaxis, :])
        
        return grad_padded
    
class MaxPool1DLayer:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x = None
        self.indices = None

    def forward(self, x, mode='train'):
        self.x = x
        self.batch_size = x.shape[0]
        self.input_size = x.shape[1]
        self.output_size = int((self.input_size - self.kernel_size) / self.stride + 1)
        
        out = np.zeros((self.batch_size, self.output_size, self.input_size))
        self.indices = np.zeros((self.batch_size, self.output_size, self.input_size), dtype=np.int32)
        
        for i in range(self.output_size):
            out[:, i, :] = np.max(self.x[:, i*self.stride:i*self.stride+self.kernel_size, :], axis=2)
            self.indices[:, i, :] = np.argmax(self.x[:, i*self.stride:i*self.stride+self.kernel_size, :], axis=2)
        
        return out
    
    def backward(self, grad):
        grad_padded = np.zeros_like(self.x)
        
        for i in range(self.output_size):
            grad_padded[:, i*self.stride:i*self.stride+self.kernel_size, :] += grad[:, i, :, np.newaxis] * (self.indices[:, i, :, np.newaxis] == np.arange(self.kernel_size)[np.newaxis, np.newaxis, :])
        
        return grad_padded

class NeuralNetwork:
    def __init__(self, layers, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.loss_ls =[]
        self.acc_ls = []


    def forward(self, x, mode='train'):
        for layer in self.layers:
            x = layer.forward(x, mode)
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

    def train(self, x, y, lr, epochs, batch_size, x_test, y_test, lr_decay=1, epoch_delay=50):
        self.loss_ls =[]
        self.acc_ls = []
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            x_batch = x[:batch_size]
            y_batch = y[:batch_size]
            y_pred = self.forward(x_batch.T, mode='train')
            y_res = y_batch.T
            loss = self.loss(y_pred, y_res)
            self.backward(y_pred - y_res)
            self.update(lr)
            x = np.roll(x, batch_size, axis=0)
            y = np.roll(y, batch_size, axis=0)
            
            acc = self.accuracy(x_test, y_test)
            self.loss_ls.append(loss)
            self.acc_ls.append(acc)

            if epoch and epoch % epoch_delay == 0: # learning rate decay
                lr /= lr_decay

            pbar.set_description('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, acc))
        pbar.close()
    
    def accuracy(self, x, y):
        y_pred = self.forward(x.T, mode='test')
        y_res = y.T
        return np.sum(np.argmax(y_pred, axis=0) == np.argmax(y_res, axis=0)) / y.shape[0]
    
    def predict(self, x):
        y_pred = self.forward(x.T, mode='test')
        return np.argmax(y_pred, axis=0)
    
    def plot_loss(self):
        plt.plot(self.loss_ls)
        plt.title('Cross Entropy Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.show()

    def plot_acc(self):
        plt.plot(self.acc_ls)
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def test_pred(self, index, x, y):
        '''
        Plot the image at index in the test set and print the predicted label
        '''
        y_pred = self.forward(x.T[:, index, None], mode='test')
        y_res = np.argmax(y_pred, 0)

        print('Predicted:', y_res[0])
        print('Actual:', np.argmax(y.T[:, index, None], 0)[0])
        plt.figure()
        plt.imshow(x.T[:, index, None].reshape((28, 28)), cmap='gray')

    def save(self, filename):
        with open(filename, 'wb+') as f:
            pickle.dump(self, f)
        

def score_ensemble_mean(models, x, y):
    y_pred = np.zeros((models[0].output_size, x.shape[0]))
    for model in models:
        y_pred += model.forward(x.T, mode='test')
    return np.sum(np.argmax(y_pred, axis=0) == np.argmax(y.T, axis=0)) / y.shape[0]

def score_ensemble_mode(models, x, y):
    y_pred = np.zeros((models[0].output_size, x.shape[0]))
    for model in models:
        y_pred += one_hot_encode(np.argmax(model.forward(x.T, mode='test'), axis=0)).T
    return np.sum(np.argmax(y_pred, axis=0) == np.argmax(y.T, axis=0)) / y.shape[0]

def one_hot_encode(preds):
    encoding = np.zeros((len(preds), 10))
    for i, val in enumerate(preds):
        encoding[i, val] = 1
    return encoding

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

