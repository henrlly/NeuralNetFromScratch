# NeuralNetFromScratch
 Neural net implemented from scratch using only Numpy

## Features
 - Linear (fully connected) layer w/wo biases
 - ReLu
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
        - **96.95%** accuracy on test set

## To use
Run `run.ipynb`

Alternatively, `import nn` in your `.py`/`.ipynb` file and call classes/functions from there

## Potential improvements
 - [x] Convolutional layer 
 - [ ] L2 and L1 regularisation

### Dependencies
curl, tqdm, matplotlib, numpy
