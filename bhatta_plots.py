#!/usr/bin/env python
import numpy as np
from numpy import random
from numpy.matlib import *
from numpy.linalg import *
import math
import matplotlib.pyplot as plt
from pdb import set_trace as debug
import bhatta
import os
import cPickle


def vim2xyz(vimg):
    vimg = np.array(vimg)
    x = vimg[:,0]
    y = vimg[:,1]
    z = vimg[:,2]
    #x = np.array(x)
    #y = np.array(y)
    #z = np.array(z)
    return (x,y,z)

def plotvim(vimg):
    (x,y,z) = vim2xyz(vimg)
    plt.scatter(x,y,c=-z,s=120, marker='s')
    plt.gray()
    plt.show()

def plot_mnist_BM():
    with open('vim_BM.pkl', 'r') as f:
        BM = cPickle.load(f)

    with open('vectorized_mnist.pkl', 'r') as f:
        (vim, labels) = cPickle.load(f)

    for i in xrange(20):
        for j in xrange(i):
            k = BM[i,j]
            li = labels[i]
            lj = labels[j]
            title = "{}:{},{}:{} k = {:.2f}".format(i,li,j,li,k)
            D1 = vim[i]
            D2 = vim[j]
            savefile = "vim_{:2d}-{:2d}".format(i,j)
            plot2data(D1,D2,title,savefile=savefile)

def plot2data(D1, D2, title_text, xrnge=(0,28), yrnge=(0,28), savefile=0):
    #D1: (n1 x 3) matrix
    (x1, y1, z1) = vim2xyz(D1)
    (x2, y2, z2) = vim2xyz(D2)
    plt.scatter(x1,y1,c='r',marker='s',s=z1*100)
    plt.scatter(x2,y2,c='b',marker='o',s=z2*100)
    plt.axis = (xrnge, yrnge)

    plt.title(title_text)
    #plt.text(kval_x, kval_y, kvals)
    if savefile == 0:
        plt.show()
    else:
        plt.savefig("./Plots/" + savefile + ".pdf")
        print "Saved " + savefile




def plot_distribution_suite(size=100, verbose=False):
    dists = []
    d0 = mvnorm([0,0], [1, -.5, -.5, -1], size)
    d1 = mvnorm([1,1], [1, .2, .2, -1], size)
    d2 = mvnorm([0,0], [1, -.5, -.5,.3], size)
    d3 = mvnorm([-1,1], [1, 0, -1, -1], size)
    d4 = mvnorm([0,0], [-1.5, 2, 2, 3], size)
    d5 = uniform([-1,1,-1,1], size)
    d6 = uniform([-2,2,-1,1], size)
    d7 = uniform([-3,3,-3,3], size)
    dists = [d0, d1, d2, d3, d4, d5, d6, d7]

    ensure('./Plots')

    itr = len(dists)
    n_comparisons = itr * (itr-1) / 2
    n = 0
    for i in xrange(itr):
        for j in xrange(i):
            n += 1
            title_text = "Comparison {}".format(n)
            plot_kernels(dists[i], dists[j], title_text, "%d" % n, verbose=verbose)
            print "Finished comparison %d of %d" % (n, n_comparisons)


def plot_kernels((X1, gen1), (X2, gen2), 
    title_text="Bhattacharrya Kernel",savefile=0, verbose=False):
    """Plot the two datasets and report kernel values for several parameters"""

    sigmas = [.5, 1, 2]
    etas   = [.01, .1, .5]
    (xmin, xmax) = (-4,4)
    (ymin, ymax) = (-4,4)
    x_range = (xmax-xmin)
    y_range = (ymax-ymin)

    (n1, d1) = X1.shape
    (n2, d2) = X2.shape
    assert d1 == d2 == 2
    X1 = matrix(X1)
    X2 = matrix(X2)
    X1x = X1[:,0]
    X1y = X1[:,1]
    X2x = X2[:,0]
    X2y = X2[:,1]
    #plt.figure()

    #nk_b = nk_bhatta(X1, X2, 0)
    #kvals = "Non-kernelized: " + "{:.2f}\n".format(nk_b).lstrip('0')
    table_vals = []
    for eta in etas:
        sig_vals = []
        for sigma in sigmas:
            if verbose: print "Evaluating {:s} for e={:.2f} s={:.2f}"\
                .format(title_text, eta, sigma)
            kappa = bhatta.eig_bhatta(X1, X2, bhatta.gaussk(sigma), eta, 25)
            sig_vals.append(kappa)
        table_vals.append(sig_vals)

    plt.figure()

    plt.plot(X1x, X1y, 'ro', X2x, X2y, 'bo')
    plt.axis([xmin, xmax, ymin, ymax])

    col_labels = [r'$\sigma = %.2f$' % sig for sig in sigmas]
    row_labels = [r'$\eta = %.2f$'   % eta for eta in etas]

    #kval_x, kval_y = xmin + x_range * .8, ymin + y_range * .7
    gen1_x, gen1_y = xmin + x_range * .02, ymin + y_range * .8
    gen2_x, gen2_y = xmin + x_range * .02, ymin + y_range * .6

    table_text = []
    for row in table_vals:
        newtext = []
        for val in row:
            newtext.append('{:.2f}'.format(val).lstrip('0'))
        table_text.append(newtext)
    n_rows = len(etas)
    n_cols = len(sigmas)

    the_table = plt.table(cellText=table_text,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  colWidths = [.1]*n_cols,
                  loc='upper right')

    plt.text(gen1_x, gen1_y, gen1, color="red")
    plt.text(gen2_x, gen2_y, gen2, color="blue")
    plt.title(title_text)
    #plt.text(kval_x, kval_y, kvals)
    if savefile == 0:
        plt.show()
    else:
        plt.savefig("./Plots/" + savefile + ".pdf")


def mvnorm(mean=[0,0], cov=[0,.5,.5,-.7], size=40):
    (a,b,c,d) = cov
    data = random.multivariate_normal(mean, [[a,b],[c,d]], size)
    title = "Multivariate normal, n = {:d}\n".format(size)
    row1 = "mean = {}\n cov = [{}, {}; {}, {}]\n".format(mean, a, b, c, d)
    return data, title+row1

def uniform(bounds = [-1, 1, -1, 1], size=40):
    xmin, xmax, ymin, ymax = bounds
    data = zeros((size,2))
    data[:,0] = matrix(random.uniform(xmin, xmax, size)).T
    data[:,1] = matrix(random.uniform(ymin, ymax, size)).T
    title = "Uniform distribution, n = {}\n".format(size)
    row1 = "x: ({}, {}) y: ({}, {})\n".format(*bounds)
    return data, title+row1

def viz(X1):
    X1x = X1[:,0]
    X1y = X1[:,1]
    plt.plot(X1x, X1y, 'ro')
    plt.axis([-1,6,-1,6])
    plt.show()

def kerneltext(sigma, eta, bhatta_value):
    btext = "{:.2f}\n".format(bhatta_value).lstrip('0')
    text = r'$\sigma = {:.2f} \eta = {:.2f} \kappa = $'.format(sigma, eta)
    return text + btext

def ensure(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def main():
    plot_mnist_BM()

if __name__ == '__main__':
    main()


# def plot2data(D1, D2, title, row_names, col_names, table_vals, xrnge=(0,28), yrnge=(0,28), savefile=0):
#     #D1: (n1 x 3) matrix
#     (x1, y1, z1) = vim2xyz(D1)
#     (x2, y2, z2) = vim2xyz(D2)
#     plt.scatter(x1,y1,c='r',marker='s',size=150)
#     plt.scatter(x2,y2,c='b',marker='o',size=80)
#     plt.axis = (xrnge, yrnge)

#     n_rows, n_cols = table_vals.shape

#     table_text = []
#     for row in xrange(n_rows):
#         newtext = []
#         for col in xrange(n_cols):
#             val = table_vals[row,col]
#             newtext.append('{:.2f}'.format(val).lstrip('0'))
#         table_text.append(newtext)

#     the_table = plt.table(cellText=table_text,
#                   rowLabels=row_names,
#                   colLabels=col_names,
#                   colWidths = [.1]*n_cols,
#                   loc='upper right')
#         plt.title(title_text)
#     #plt.text(kval_x, kval_y, kvals)
#     if savefile == 0:
#         plt.show()
#     else:
#         plt.savefig("./Plots/" + savefile + ".pdf")
