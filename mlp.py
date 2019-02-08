#! /usr/bin/env python3
import numpy as np
import math
import sys
import time
import pickle

# def softmax(x, c = 1):
#     """Compute softmax values for each sets of scores in x."""
#     print(x.shape,x)
#     e_x = np.exp(c * x - np.max(c*x))
#     print(e_x.shape,e_x)
#     return e_x / e_x.sum(axis=1)

def softmax(v, c=1):
    # print(v)
    e_x = np.exp(c*v - np.amax(c*v, axis=0))
    # print(e_x / e_x.sum(axis=0))
    return e_x / e_x.sum(axis=0)


def cross_entropy(p,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss


def onehot(Y, m):
    ret = np.zeros((Y.shape[0],m))
    for i in range(Y.shape[0]):
        ret[i,Y[i]]=1.
    return ret.astype(int)

class Activation_function:
    def __str__(self):
        return "Activation function"
    def __call__(self, x):
        raise NotImplementedError()
    def derivative(self, x):
        raise NotImplementedError()

class Identity(Activation_function):
    def __str__(self):
        return "identity"
    def __call__(self, x):
        return x
        def derivative(self, x):
            return 1.

class Sigmoid(Activation_function):
    def __str__(self):
        return "sigmoid"
    def __call__(self, x):
        return 1./(np.exp(-x)+1.)
    def derivative(self, x):
        return self(x)*(1. - self(x))

class ReLU(Activation_function):
    def __str__(self):
        return "ReLU"
    def __call__(self, x):
        return x + (x > 0)
    def derivative(self, x):
        return 1. * (x > 0)

class Tanh(Activation_function):
    def __str__(self):
        return "tanh"
    def __call__(self, x):
        tmp1 = np.exp(x)
        tmp2 = np.exp(-x)
        return (tmp1  - tmp2) / (tmp1 + tmp2)
    def derivative(self, x):
        return 1. - self(x)**2


class MLP_2L:
    """ A simple implementation of a multi layers perceptron with two hidden layers. """
    def __init__(self, input_size, n1, n2, output_size, c=1, init="normal",\
                    activation="identity", l1=0, l2=0):
        """ Initialises a two layers MLP.

            Parameters
            ----------

            input_size : int
                Input dimension

            n1 : int
                First layer dimension

            n2 : int
                Second layer dimension

            output_size : int
                Output dimension

            c : float
                Scalar, the output layer is softmax(c * oa), with oa the preactivation.
                By default c = 1. c must be strictly greater than 0.

            init : String
                Defines the weights initialization process. init values must be "normal",
                "zeros" or "glorot". By default init = "normal"

            activation : String
                Defines the activation function used on the two layers. activation must be
                "identity", "sigmoid", "relu" or tanh. By default activation="identity"

            l1 : float
                weigth of the L1 regularizer. Must be positive. By default l1 = 0
                : no L1 regularizer

            l2 : float
                weigth of the L2 regularizer. Must be positive. By default l2 = 0
                : no L2 regularizer
        """

        n_parameter = n1 * (input_size + 1) + n2 * (n1 + 1) + output_size * (n2 + 1)
        print("Input dimension {:d}\tLayer 1 dimension {:d}\tLayer 2 dimension {:d}\tOutput dimension {:d}\t Initilization method {:s}\tActivation function {:s}".format(input_size, n1, n2, output_size, init, str(activation)))
        print("Total number of parameters : {:d}".format(n_parameter))

        self.input_size = input_size
        self.n1 = n1
        self.n2 = n2
        self.output_size = output_size
        self.c = c
        self.l1 = l1
        self.l2 =l2

        if activation == "identity":
            self.activation = Identity()
        elif activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation=="relu":
            self.activation = ReLU()
        elif activation=="tanh":
            self.activation = Tanh()
        else :
            raise ValueError("activation must be 'identity', 'sigmoid', 'relu' or 'tanh'")


        if init == "normal":
            self.w1 = np.random.normal(size=(n1, input_size))
            self.w2 = np.random.normal(size=(n2, n1))
            self.w3 = np.random.normal(size=(output_size, n2))
        elif init == "zeros":
            self.w1 = np.zeros(shape=(n1, input_size))
            self.w2 = np.zeros(shape=(n2, n1))
            self.w3 = np.zeros(shape=(output_size, n2))
        elif init == "glorot":
            d1 = math.sqrt(6 / (input_size+n1))
            d2 = math.sqrt(6 / (n1 + n2))
            d3 = math.sqrt(6 / (n2 + output_size))
            self.w1 = np.random.uniform(-d1, d1, size=(n1, input_size))
            self.w2 = np.random.uniform(-d2, d2,size=(n2, n1))
            self.w3 = np.random.uniform(-d3, d3,size=(output_size, n2))
        elif init=="load":
            pass
        else:
            raise ValueError("init must be 'normal', 'zeros' or 'glorot'")

        self.b1 = np.zeros((n1, 1))
        self.b2 = np.zeros((n2, 1))
        self.b3 = np.zeros((output_size, 1))

    def fprop(self, X):
        self.X=X
        self.a1 = np.dot(self.w1, X.T ) + self.b1
        self.h1 = self.activation(self.a1)

        self.a2 = np.dot(self.w2, self.h1 ) + self.b2
        self.h2 = self.activation(self.a2)

        self.oa = np.dot(self.w3, self.h2 ) + self.b3
        self.os = softmax(self.oa, self.c)

    def bprop(self, Y, learning_rate):
        """ We assume fprop has been run before bprop
            This function calculate the gradient and update the weights after
        """

        batch_size = Y.size

        #compute the gradient

        #output layer
        self.grad_oa = self.os - onehot(Y,self.output_size).T
        self.grad_w3 = np.dot(self.grad_oa, self.h2.T) #without regularizer
        self.grad_w3 += self.l1 * self.w3 / ( np.abs(self.w3) + (self.w3 == 0).astype(int) ) #add l1 regularizer
        self.grad_w3 += 2 * self.l2 * self.w3 # add l2 regularizer
        self.grad_b3 = self.grad_oa #without regularizer
        self.grad_b3 += self.l1 * self.b3 / ( np.abs(self.b3) + (self.b3 == 0).astype(int) ) #add l1 regularizer
        self.grad_b3 +=  2 * self.l2 * self.b3 #add l2 regularizer

        #2nd layer
        self.grad_h2 = np.dot(self.w3.T, self.grad_oa)
        self.grad_a2 = self.grad_h2 * self.activation.derivative(self.a2)
        self.grad_w2 = np.dot(self.grad_a2, self.h1.T) + self.l1 * self.w2 / ( np.abs(self.w2) + (self.w2 == 0).astype(int) ) + 2 * self.l2 * self.w2
        self.grad_b2 = self.grad_h2 + self.l1 * self.b2 / ( np.abs(self.b2) + (self.b2 == 0).astype(int) ) + 2 * self.l2 * self.b2

        #1st layer
        self.grad_h1 = np.dot(self.w2.T, self.grad_h2)
        self.grad_a1 = self.grad_h1 * self.activation.derivative(self.a1)
        self.grad_w1 = np.dot(self.grad_a1, self.X) + self.l1 * self.w1 / ( np.abs(self.w1) + (self.w1 == 0).astype(int) ) + 2 * self.l2 * self.w1
        self.grad_b1 = self.grad_h1 + self.l1 * self.b1 / ( np.abs(self.b1) + (self.b1 == 0).astype(int) ) + 2 * self.l2 * self.b1

        #weight update
        self.w3 -= learning_rate * self.grad_w3 / batch_size
        self.b3 -= learning_rate * self.grad_b3.sum(axis=1).reshape((-1,1)) / batch_size

        self.w2 -= learning_rate * self.grad_w2 / batch_size
        self.b2 -= learning_rate * self.grad_b2.sum(axis=1).reshape((-1,1)) / batch_size

        self.w1 -= learning_rate * self.grad_w1 / batch_size
        self.b1 -= learning_rate * self.grad_b1.sum(axis=1).reshape((-1,1)) / batch_size


    def fit(self,X, Y, epochs, batch_size, learning_rate):
        train_time = time.time()
        for epoch in range(epochs):
            epoch_time = time.time()
            pred = 0
            for i in range(math.ceil(X.shape[0]/batch_size)):
                i_min = i * batch_size
                i_max = min( (i+1) * batch_size, X.shape[0] )

                self.fprop(X[i_min:i_max] )
                self.bprop(Y[i_min:i_max], learning_rate)

                pred += np.sum( (np.argmax(self.os,axis=0) == Y[i_min:i_max] ).astype(int) )
                print("Epoch {:d}/{:d}\tExamples {:d}/{:d}\tAccuracy {:.3f}\tEpoch time {:.2f}s\tTraining time {:.2f}s".format(epoch+1, epochs, i_max, X.shape[0], pred/i_max, time.time() - epoch_time, time.time() - train_time), end='\r')
            print("Epoch {:d}/{:d}\tExamples {:d}/{:d}\tAccuracy {:.3f}\tEpoch time {:.2f}s\tTraining time {:.2f}s".format(epoch+1, epochs, i_max, X.shape[0], self.evaluate(X,Y) , time.time() - epoch_time, time.time() - train_time))

            if np.isnan(self.w1).any() :
                sys.exit("ERROR : The parameters contain NaNs. Use a smaller learning rate.")
        print("Total training time {:.2f}s".format(time.time() - train_time))

    def predict(self,X):
        self.fprop(X)
        return np.argmax(self.os,axis=0)

    def evaluate(self, X, Y):
        pred = self.predict(X)
        return np.sum((pred==Y).astype(int))/Y.size

    def save(self,filename):
        tmp = dict()
        tmp["input_size"] = self.input_size
        tmp["n1"] = self.n1
        tmp["n2"] = self.n2
        tmp["output_size"] = self.output_size
        tmp["w1"] = self.w1
        tmp["b1"] = self.b1
        tmp["w2"] = self.w2
        tmp["b2"] = self.b2
        tmp["w3"] = self.w3
        tmp["b3"] = self.b3

        tmp["l1"] = self.l1
        tmp["l2"] = self.l2

        tmp["c"] = self.c

        tmp["activation"] = str(self.activation)

        with open(filename, 'wb') as handle:
            pickle.dump(tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(filename):
        with open(filename, 'rb') as handle:
            dic = pickle.load(handle)
        tmp = MLP_2L(dic["input_size"], dic["n1"], dic["n2"], dic["output_size"], init="load", activation=dic["activation"], c=dic["c"], l1=dic["l1"], l2=dic["l2"])

        tmp.w1 = dic["w1"]
        tmp.b1 = dic["b1"]
        tmp.w2 = dic["w2"]
        tmp.b2 = dic["b2"]
        tmp.w3 = dic["w3"]
        tmp.b3 = dic["b3"]
        return tmp
