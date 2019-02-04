#! /usr/bin/env python3
import numpy as np
import math
import sys

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


class MLP:
    def __init__(self, input_size, n1, n2, output_size, c=1, init="normal"):
        n_parameter = n1 * (input_size + 1) + n2 * (n1 + 1) + output_size * (n2 + 1)
        print("Number of parameters : "+str(n_parameter))
        self.output_size = output_size
        if init == "normal":
            self.W1 = np.random.normal(size=(n1, input_size))
            self.W2 = np.random.normal(size=(n2, n1))
            self.W3 = np.random.normal(size=(output_size, n2))
        elif init == "zeros":
            self.W1 = np.zeros(shape=(n1, input_size))
            self.W2 = np.zeros(shape=(n2, n1))
            self.W3 = np.zeros(shape=(output_size, n2))
        elif init == "glorot":
            d1 = math.sqrt(6 / (input_size+n1))
            d2 = math.sqrt(6 / (n1 + n2))
            d3 = math.sqrt(6 / (n2 + output_size))
            self.W1 = np.random.uniform(-d1, d1, size=(n1, input_size))
            self.W2 = np.random.uniform(-d2, d2,size=(n2, n1))
            self.W3 = np.random.uniform(-d3, d3,size=(output_size, n2))

        self.b1 = np.zeros((n1, 1))
        self.b2 = np.zeros((n2, 1))
        self.b3 = np.zeros((output_size, 1))

        self.c = c

    def fprop(self, X):
        self.X=X
        self.h1 = np.dot(self.W1, X.T ) + self.b1
        self.h2 = np.dot(self.W2, self.h1 ) + self.b2

        self.oa = np.dot(self.W3, self.h2 ) + self.b3

        self.os = softmax(self.oa, self.c)

    def bprop(self, Y, learning_rate):
        # L = cross_entropy(self.os, Y)
        self.grad_oa = self.os - onehot(Y,self.output_size).T
        self.grad_w3 = np.dot(self.grad_oa, self.h2.T)
        self.grad_b3 = self.grad_oa
        self.W3 -= learning_rate * self.grad_w3
        self.b3 -= learning_rate * self.grad_b3.sum(axis=1).reshape((-1,1)) / self.grad_b3.shape[1]

        self.grad_h2 = np.dot(self.W3.T, self.grad_oa)
        self.grad_w2 = np.dot(self.grad_h2, self.h1.T)
        self.grad_b2 = self.grad_h2
        self.W2 -= learning_rate * self.grad_w2
        self.b2 -= learning_rate * self.grad_b2.sum(axis=1).reshape((-1,1)) / self.grad_b2.shape[1]

        self.grad_h1 = np.dot(self.W2.T, self.grad_h2)
        self.grad_w1 = np.dot(self.grad_h1, self.X)
        self.grad_b1 = self.grad_h1
        self.W1 -= learning_rate * self.grad_w1
        self.b1 -= learning_rate * self.grad_b1.sum(axis=1).reshape((-1,1)) / self.grad_b2.shape[1]

    def train(self,X, Y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            pred = 0
            for i in range(math.ceil(X.shape[0]/batch_size)):
                i_min = i * batch_size
                i_max = min( (i+1) * batch_size, X.shape[0] )

                self.fprop(X[i_min:i_max] )
                self.bprop(Y[i_min:i_max], learning_rate)

                pred += np.sum( (np.argmax(self.os,axis=0) == Y[i_min:i_max] ).astype(int) )

                print("Epoch "+str(epoch+1)+"/"+str(epochs)+"\tExamples "+str(i_max)+"/"+
                    '{:.3f}'.format(X.shape[0])+"\tRealtime accuracy :"+'{:.3f}'.format(pred/i_max),end='\r')
            # print("Epoch "+str(epoch+1)+"/"+str(epochs)+"\tFinal accuracy :"+str(self.evaluate(X,Y)))
            print("Epoch "+str(epoch+1)+"/"+str(epochs)+"\tExamples "+str(i_max)+"/"+
                str(X.shape[0])+"\tRealtime accuracy :"+'{:.3f}'.format(pred/i_max)+"\tFinal accuracy :"+'{:.3f}'.format(self.evaluate(X,Y)) )
            if np.isnan(self.W1).any() :
                sys.exit("ERROR : The parameters contain NaNs. Use a smaller learning rate.")

    def predict(self,X):
        self.fprop(X)
        return np.argmax(self.os,axis=0)

    def evaluate(self, X, Y):
        pred = self.predict(X)
        return np.sum((pred==Y).astype(int))/Y.size
