#!/usr/bin/env python
from pdb import set_trace as debug
import numpy as np
from numpy import array, zeros, exp
from numpy.linalg import norm
import pickle
import sys
from functools import partial
import bhatta as b
import mnist
with open('data.dat','r') as data_f:
    data_list = pickle.load(data_f) 

np.set_printoptions(threshold='nan')

train_proportion = .1
n_data = len(data_list)
dividing_pt = int(n_data * train_proportion)

train_data = data_list[0:dividing_pt]
test_data  = data_list[dividing_pt:n_data]

def print2(str):
    sys.stderr.write(str + "\n")
    print str


def linear_perceptron_iter((data,label),weights, learn=1):
    predicted = (np.dot(data,weights) > 0)
    if learn and (predicted != label):
        if label == True:
            weights += data
        else:
            weights -= data
    return predicted == label

def learning_kernel_perceptron(training_set, kernel):
    weights = np.zeros(len(training_set))
    for t in xrange(len(training_set)):
        (x_t, label_t) = training_set[t]
        
        kernel_t = lambda D: kernel(D[0],x_t)
        kernel_values = array(map(kernel_t,training_set[0:t]))

        array([kernel(D[0],x_t) for D in training_set[0:t]])
        
        weight_values = weights[0:t]

        prediction = sum(kernel_values * weight_values) > 0

        if prediction != label_t:
            if prediction == 0:
                weights[t] = 1
            else:
                weights[t] = -1
    return weights

def test_kernel_perceptron(training_set, weights, D, kernel):
    (x_t, label) = D
    kernel_t = lambda D: kernel(D[0],x_t)
    kernel_values = array(map(kernel_t, training_set))
    prediction = sum(kernel_values * weights) > 0
    return prediction == label

def perceptron_kernel(kernel):
    weights = learning_kernel_perceptron(train_data, kernel)
    tkp = lambda D: test_kernel_perceptron(train_data, weights, D, kernel)
    results = map(tkp,test_data)
    correct = sum(results)
    errors = len(results) - correct

    rate = float(errors) / len(results)
    #print str(weights)
    #print str(results)
    return rate

def gaussian_rbf(x1, x2, sigma=1):
    return exp(-norm(x1-x2)**2/(2*sigma*sigma))

def poly_kernel(x1,x2,p):
    return (1 + np.dot(x1,x2))**p



class BhattaVectorWrapper:
    def __init__(self, kernel, eta, r):
        self.bhatta_fn = b.HashBhatta(kernel, eta, r).bhatta
        self.vim = {}

    def bhatta(self, v1, v2):
        X1 = self.get_vectorized(v1)
        X2 = self.get_vectorized(v2)
        return self.bhatta_fn(X1,X2)

    def get_vectorized(self, v):
        hashval = b.array_hash(v)
        try:
            vim = self.vim[hashval]
        except KeyError:
            vim = mnist.vectorize_images(np.matrix(v))[0]
            self.vim[hashval] = vim
        return vim




# data1 = train_data[5][0]
# data2 = train_data[6][0]

# for (fn, text) in fn_list:
#     print text
#     print "     " + str(fn(data1,data2))

def main(): 
    fn_list = []#[(np.dot,'dot product')]

    # sigmas = list(np.linspace(4, 6,3))

    # for sigma in sigmas:
    #     new_fn = partial(gaussian_rbf, sigma=sigma)
    #     infotext = "RBF sigma = " + str(sigma)
    #     fn_list.append((new_fn,infotext))

    # polys = [3,5]
    # for poly in polys:
    #     new_fn = partial(poly_kernel, p=poly)
    #     infotext = "Poly p = " + str(poly)
    #     fn_list.append((new_fn, infotext))

    gsk = [.5, 1, 5]
    for gauss in gsk:
        kernel = b.gaussk(gauss)
        eta = .1
        r = 15
        bhatta_k = BhattaVectorWrapper(kernel, eta, r).bhatta
        infotext = "bhatta w/ gaussk {}".format(gauss)
        fn_list.append((bhatta_k, infotext))
    for (fn, descriptor) in fn_list:
        print2("Kernel " + descriptor + ": Error rate: " + str(perceptron_kernel(fn)))

if __name__ == '__main__':
    main()


# for sigma in sigmas:
#     new_fn = lambda x1, x2: gaussian_rbf(x1,x2,sigma)
#     infotext = "RBF sigma = " + str(sigma)
#     fn_list.append((new_fn,infotext))