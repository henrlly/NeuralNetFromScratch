# NeuralNetFromScratch
 Neural net implemented from scratch using only Numpy

## Features
 - Linear layer w/wo biases
 - ReLu
 - Tanh
 - Sigmoid
 - Softmax
 - Dropout
 - Ensemble models
 - Save and load models

## Results
 - MNIST dataset
    - Ensemble of 10 model (no CNN)
    - Network structure: [784, 100, 10]
    - **96.2%** accuracy on test set

## To use
Run `run.ipynb`

Alternatively, `import nn` in your `.py`/`.ipynb` file and call classes/functions from there

## Potential improvements
 - [ ] Convolutional layer 
 - [ ] L2 and L1 regularisation

### Dependencies
curl, tqdm, matplotlib, numpy
