# NeuralNetFromScratch

Annotated neural net from scratch with Numpy

## Features

- Linear (fully connected) layer
- ReLU
- Tanh
- Sigmoid
- Softmax
- Dropout
- Batch normalisation
- Learning rate decay (exponential)
- Ensemble models
- Save and load models

## Results

- MNIST dataset
  - Ensemble of 10 fully connected networks
    - Trained with SGD over 128 epoches, batch size of 1000 per epoch
    - **96.85%** accuracy on test set
