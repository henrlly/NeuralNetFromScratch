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
- Loss functions
  - Cross entropy
  - Mean squared error
- Optimisers
  - Stochastic gradient descent (SGD)
    - Linear LR decay
    - Momentum
    - Nesterov momentum
  - AdaGrad
  - RMSProp
  - Adam
- Decoupled weight decay (L1 and L2 regularisation)
- Early stopping
- Inference and (hard and soft) scoring with an ensemble of models
- Save and load models

---

Example of neural net trained on MNIST dataset classification in `classifier.ipynb`

---

This was created while reading the [Deep Learning Book](https://www.deeplearningbook.org).
