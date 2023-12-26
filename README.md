# NeuralNetFromScratch

Annotated neural net from scratch with NumPy

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
- Early stopping
- Inference and (hard and soft) scoring with an ensemble of models
- Save and load models

## Results

- MNIST dataset
  - Ensemble of 10 fully connected networks
    - Trained with SGD over 128 epoches, batch size of 1000 per epoch
    - **96.97%** accuracy on test set
