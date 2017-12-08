'''
    Brandon Wood
    12/7/17
    Test Driving a Neural Network
'''
import numpy as np
import math
import random
import time

class nn1():
    '''Simple backprop neural network class with randomly initialized weights.
    '''

    # Class Methods

    def __init__(self, layer_size):

        # Meta Parameters
        self.trainingRate = .5
        self.normalScale = 1
        self.maxSteps = 100000
        self.minErr = .00005

        # Layer Info
        self.layerCount = len(layer_size) - 1
        self.shape = layer_size

        # Intermediate Values
        self._layerInput = []
        self._layerOutput = []

        # Weights
        self.weights = []
        self.transform_shape()
        self.init_weights()

    def transform_shape(self):
        transformed = []
        for i in range(len(self.shape)-1):
            transformed.append([self.shape[i], self.shape[i + 1]])
        self.transformed = transformed

    def init_weights(self):
        for x, y in self.transformed:
            self.weights.append(np.random.normal(scale  = self.normalScale, size = (y, x + 1))) # Add 1 for biases

    # Transfer Functions

    def sigmoid(self, x):
        return(1/(1 + np.exp(-1*x)))

    def sigmoidPrime(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    # Feeds

    def feedForward(self, inputs):
        '''Runs Network forwards. Expected input should be MxN Numpy Array (Each ROW is a sample)'''

        # Grab the number of inputs for creaing biases
        samples = inputs.shape[0]

        # Clear out last values
        self._layerInput = []
        self._layerOutput = []

        # Feed
        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[index].dot(np.vstack([inputs.T, np.ones([1, samples])]))
            else:
                layerInput =  self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, samples])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sigmoid(layerInput))

    def backpropagate(self, inputs, targets):
        '''
        Trains the network on a set of sample inputs and their target outputs.

        Both the inputs and target outputs should follow the same structure
        as the inputs to the feedForward method.
        '''

        # Useful values
        samples = inputs.shape[0]
        delta = []
        # Feed Forwards
        self.feedForward(inputs)

        # Backpropagate
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                output_delta = self._layerOutput[index] - targets.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.sigmoidPrime(self._layerOutput[index]))
            else:
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * self.sigmoidPrime(self._layerOutput[index]))

        # Compute Weight Deltas
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index
            if index == 0:
                layerOutput = np.vstack([inputs.T, np.ones([1, samples])])
            else:
                layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            weightDelta = np.sum(layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0), axis = 0)

            self.weights[index] -= self.trainingRate*weightDelta
        return error

    # Utilities

    def print_shape(self):
        print([x for x in self.transformed])
        print()
        print([x.shape for x in self.weights])
        print()
        print([x.shape for x in self._layerOutput])
        print()
        print([x.shape for x in self._delta])

    def learn(self, inputs, targets):

        for i in range(self.maxSteps - 1):
            e = n.backpropagate(inputs, targets)
            if e < self.minErr:
                print("Mininum Error Achieved at:", i)
                break
            if i%1000 == 0:
                print(e)
                print(i)
                print(n._layerOutput[-1])
                print()

if __name__ == '__main__':
    n = nn1((2, 2, 1))
    inputs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    targets = np.array([[.95], [.95], [0.05], [0.05]])
    n.learn(inputs, targets)
