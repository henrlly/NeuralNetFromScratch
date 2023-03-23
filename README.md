# NeuralNetFromScratch
 Neural net implemented from scratch using only Numpy

## Features
 - Linear (fully connected) layer w/wo biases
 - ReLU
 - Tanh
 - Sigmoid
 - Softmax
 - Dropout
 - Batch normalisation (1D and 2D)
 - Convolutional layer (1D and 2D) w/wo biases
 - Max pooling (1D and 2D)
 - Learning rate decay
 - Ensemble models
 - Save and load models

## Results
 - MNIST dataset
    - Ensemble of 10 fully connected networks (no CNN)
        - Trained with SGD over 800 epoches, batch size of 1000 per epoch
        - **97.21%** accuracy on test set

## To use
Run `run.ipynb`

Alternatively, `import nn` in your `.py`/`.ipynb` file and call classes/functions from there

## Potential improvements
 - [ ] L2 and L1 regularisation
 - [ ] Residual connections
 - [ ] Attention mechanisms
 - [ ] RNNs
 - [ ] Transformers
 - [x] Convolutional layer 

### Dependencies
curl, tqdm, matplotlib, numpy
