# NeuralNetFromScratch

Annotated neural net from scratch with Numpy

## Features

- Layers
  - Linear
  - ReLU
  - Tanh
  - Sigmoid
  - Softmax
  - Dropout
  - Batch normalisation
- Weight decay (L1 and L2 normalisation)
- Learning rate decay (exponential)
- Ensemble models
- Save and load models

## Results

- MNIST dataset
  - Ensemble of 10 fully connected networks
    - Trained with SGD over 128 epoches, batch size of 1000 per epoch
    - **96.93%** accuracy on test set
