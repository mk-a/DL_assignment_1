#! /usr/bin/env python3
import numpy as np
import math
import sys
import time
import pickle

def softmax(v):
    # print(v)
    e_x = np.exp(v - np.amax(v, axis=0))
    # print(e_x / e_x.sum(axis=0))
    return e_x / e_x.sum(axis=0)


def loss(os,Y):
    """ return the categorical cross entropy loss over f(X),Y
        os is the output of the neural net
        Y is the expected output
    """
    loss = 0
    for i in range(Y.size):
        loss -= math.log(os[Y[i],i])
    return loss/Y.size

def categorical_cross_entropy(pred, expect):
    """ return the categorical cross entropy between the expected classes and the predictions

        pred : 2 dimensional array of the predicted probabilities
            pred.shape must be : (number of samples, number of classes).
            You must transpose the output of the NN before compute the categorical cross entropy !!

        expect : 2 dimensional array of the onehot encoding of the classes
            Y.shape must be : (number of samples, number of classes).
    """

    return -np.sum( np.log( (pred*expect)[ (pred*expect) > 0]) ) / pred.shape[0]

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
    def __init__(self, input_size, n1, n2, output_size, init="normal",\
                    activation="identity", l1=0, l2=0, verbose=True):
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
        if verbose :
            print("Input dimension {:d}\tLayer 1 dimension {:d}\tLayer 2 dimension {:d}\tOutput dimension {:d}\t Initilization method {:s}\tActivation function {:s}".format(input_size, n1, n2, output_size, init, str(activation)))
            print("Total number of parameters : {:d}".format(n_parameter))

        self.input_size = input_size
        self.n1 = n1
        self.n2 = n2
        self.output_size = output_size
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
        self.X=X #temporary save the input value
        self.a1 = np.dot(self.w1, X.T ) + self.b1
        self.h1 = self.activation(self.a1)

        self.a2 = np.dot(self.w2, self.h1 ) + self.b2
        self.h2 = self.activation(self.a2)

        self.oa = np.dot(self.w3, self.h2 ) + self.b3
        self.os = softmax(self.oa)
        return self.os
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


    def fit(self,X, Y, epochs, batch_size, learning_rate, validation_data=None, verbose=True, previous=None):
        """ Train the model for a given number of epochs

            return a dictionary with accuracy and loss on the training and if present the validation set

            Parameters
            --------
            X : two dimensional array
                The training data

            Y : one dimensional array
                The training targets.

            epochs : int
                Number of epochs on which the model is trained

            batch_size : int

            learning_rate : float

            validation_data : tuple of two arrays
                If not none the accuracy will be calculated on it. The model will not be train on this data
                first element must by a 2 dimensional array
                second element must be a one dimensional array

            verbose : boolean
                allow the function to prinf informations about the traning

            previous : dictionary
                dictionnary returned by a previous training. It allows to add the new loss and accuracy to the tail of the lists
        """
        if validation_data != None:
            X_valid , Y_valid = validation_data

        if previous == None:
            prev_epoch = 0 # number of epoch previously done, here it is zero
            t_acc_list = []
            t_loss_list = []
            epoch_list = []
            if validation_data != None:
                v_acc_list = []
                v_loss_list = []
        else:
            prev_epoch = previous["epoch"][-1] # number of epoch previously done
            t_acc_list = previous["train_acc"]
            t_loss_list = previous["train_loss"]
            epoch_list = previous["epoch"]
            if validation_data != None:
                v_acc_list = previous["valid_acc"]
                v_loss_list = previous["valid_loss"]

        if verbose:
            if validation_data == None:
                print("Train on {:d} samples\n".format(Y.size))
            else :
                print("Train on {:d} samples\tEvaluate on {:d}\n".format(Y.size, Y_valid.size ))

        train_time = time.time()

        for epoch in range(epochs):
            if verbose:
                print("Epoch {:d}/{:d}\t\tTotal training time {:.1f}s".format(epoch+1, epochs, time.time() - train_time ))
            epoch_time = time.time()
            pred = 0 # counter of well predicted samples
            loss = 0 # loss over the training set, calculated during the epoch
            for i in range(math.ceil(X.shape[0]/float(batch_size) )):
                i_min = i * batch_size
                i_max = min( (i+1) * batch_size, X.shape[0] )

                os = self.fprop(X[i_min:i_max] )
                self.bprop(Y[i_min:i_max], learning_rate)
                loss =  (i_min * loss + (i_max - i_min) * self.loss(X[i_min:i_max], Y[i_min:i_max]) ) / i_max
                pred += np.sum( (np.argmax(self.os,axis=0) == Y[i_min:i_max] ).astype(int) )
                if verbose:
                    print("\tSamples {:d}/{:d}\tEpoch time {:.2f}s\tAccuracy {:.3f}\tLoss {:.3f}".format(i_max, X.shape[0],time.time() - epoch_time, pred/i_max, loss), end='\r')

            if np.isnan(self.w1).any() :
                #This stops the program in case of exploding gradient
                sys.exit("ERROR : The parameters contain NaNs. Use a smaller learning rate.")

            t_acc = self.evaluate(X, Y)
            t_loss = self.loss(X, Y)
            epoch_list.append(epoch + 1 + prev_epoch)
            t_acc_list.append(t_acc)
            t_loss_list.append(t_loss)
            if validation_data != None:
                v_acc = self.evaluate(X_valid, Y_valid)
                v_loss = self.loss(X_valid, Y_valid)
                v_acc_list.append(v_acc)
                v_loss_list.append(v_loss)

            if verbose:
                if validation_data == None:
                    print("\tSamples {:d}/{:d}\tEpoch time {:.2f}s\tAccuracy {:.3f}\tLoss {:.3f}".format(i_max, X.shape[0],time.time() - epoch_time, t_acc, t_loss ))
                else:
                    print("\tSamples {:d}/{:d}\tEpoch time {:.2f}s\tAccuracy {:.3f}\tLoss {:.3f}\tValid accuracy {:.3f}\t Valid loss {:.3f}".format(i_max, X.shape[0],time.time() - epoch_time, t_acc, t_loss, v_acc, v_loss ) )

        if verbose :
            print("\nTotal training time {:.2f}s".format(time.time() - train_time))

        if validation_data == None:
            ret = {"epoch" : epoch_list, "train_acc" : t_acc_list, "train_loss" : t_loss_list}
        else :
            ret = {"epoch" : epoch_list, "train_acc" : t_acc_list, "train_loss" : t_loss_list, "valid_acc" : v_acc_list, "valid_loss" : v_loss_list}

        return ret

    def loss(self, X, Y):
        """ Compute the loss over the input X and the expected classes Y
        """
        l1 = np.sum(np.absolute(self.w1)) + np.sum(np.absolute(self.b1)) + np.sum(np.absolute(self.w2)) + np.sum(np.absolute(self.b2)) + np.sum(np.absolute(self.w3)) + np.sum(np.absolute(self.b3))
        l2 = np.sum(np.power(self.w1,2)) + np.sum(np.power(self.b1,2)) + np.sum(np.power(self.w2,2)) + np.sum(np.power(self.b2,2)) + np.sum(np.power(self.w3,2)) + np.sum(np.power(self.b3, 2))
        return categorical_cross_entropy(self.fprop(X).T, onehot(Y,self.output_size)) + self.l1 * l1 + self.l2 * l2

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

        tmp["activation"] = str(self.activation)

        with open(filename, 'wb') as handle:
            pickle.dump(tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(filename):
        with open(filename, 'rb') as handle:
            dic = pickle.load(handle)
        tmp = MLP_2L(dic["input_size"], dic["n1"], dic["n2"], dic["output_size"], init="load", activation=dic["activation"], l1=dic["l1"], l2=dic["l2"])

        tmp.w1 = dic["w1"]
        tmp.b1 = dic["b1"]
        tmp.w2 = dic["w2"]
        tmp.b2 = dic["b2"]
        tmp.w3 = dic["w3"]
        tmp.b3 = dic["b3"]
        return tmp

    def check_grad_w2(self, x, y, index, epsilon):
        """ return the absolute difference of gradient for the index value of W2 with
            the finite difference.
            x must be a two dimensional array with the first dimension equals 1
            y must be a one dimensional array with only one element.
        """
        if index >= self.w1.size:
            raise ValueError("index out of bounds")
        self.fprop(x)
        self.bprop(y, 0) #learning rate = 0 so that the weights are not updated

        i = math.floor(index / self.w2.shape[1])
        j = index % self.w2.shape[1]
        #this will allow us to access to the index value of w2 with w2[i,j]

        print("initial value {:.5f}".format(self.w2[i,j]))
        grad = self.grad_w2[i,j] # store the value of the grad
        self.w2[i,j] += epsilon # change the value of the index value of w2
        print("- epsilon {:.5f}".format(self.w2[i,j]))
        l1 = loss(self.fprop(x),y)
        self.w2[i,j] -= 2 * epsilon # rechange the index value of w2
        print("+ epsilon {:.5f}".format(self.w2[i,j]))
        l2 = loss(self.fprop(x),y)

        self.w2[i,j] += epsilon # reset to the initial value
        print("initial value after {:.5f}".format(self.w2[i,j]))
        print("l1 = {:.5f}\tl2 = {:.5f}".format(l1,l2))
        print("grad={:.5f}\tfinite={:.5f}\tdiff={:.10f}".format(grad, (l1 - l2) / (2* epsilon), (l1 - l2) / (2* epsilon)- grad))
        return abs( ( (l1 - l2) / (2.* epsilon) ) - grad)
