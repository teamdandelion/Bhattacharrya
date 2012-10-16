#!/usr/bin/env python
# Some test functions to determine if the kernel or different components of it are working properly.
import numpy as np
import numpy.matlib as mat
import numpy.linalg as la
import math
import bhatta_poly as poly
from scipy.sparse.linalg import eigsh # Lanzcos algorithm
import time
from bhatta import *

def test_suite_1():
    n1 = 100
    n2 = 100
    n = n1+n2
    d = 5
    eta = .1
    degree = 3
    iterations = 1
    results = mat.zeros((8,5)) 
    times = mat.zeros((1,5))
    sigma = 2
    # 1st col is non-kernelized
    # 2nd col is poly-kernel 

    for itr in xrange(iterations):
        X = mat.randn(n1,d)
        Phi_X = poly.phi(X, degree)

        D0 = X + mat.rand(n2,d) / 1000
        # Verify identity K(X,X) = 1
        D1 = mat.randn(n2,d) 
        # How does kernel perform iid data
        D2 = mat.rand(n2,d)
         # Uniform rather than normal distribution
        D3 = mat.randn(n2,d) * 2 + 2
        # Linear transformation
        D4 = mat.power(mat.randn(n2,d) + 1 ,3) 
        #Non-linear transformation
        D5 = mat.power(X+1,3) 
        #non-linear transformation of the D0 dataset;
        D6 = mat.rand(n2,d)/100 + mat.eye(n2,d) 
        #Totally different data - should have low similarity
        D7 = mat.rand(n2,d)/100 + mat.eye(n2,d)*5 
        # Scaled version of D7

        Data = [D0, D1, D2, D3, D4, D5, D6, D7]


        for idx in xrange(8):
            D = Data[idx]
            start = time.time()
            results[idx, 0] += nk_bhatta(X, D, 0)
            nk = time.time()
             emp = time.time()
            results[idx, 1] += Bhattacharrya(X,D,gaussk(sigma),eta,5)
            e5 = time.time()
            results[idx, 2] += Bhattacharrya(X,D,gaussk(sigma),eta,15)
            e15 = time.time()
            results[idx, 3] += Bhattacharrya(X,D,gaussk(sigma),eta,25)
            e25 = time.time()
            nktime = nk-start
            emptime = emp-nk
            e5time = e5-emp
            e15time = e15-e5
            e25time = e25-e15
            print "nk: {:.1f}, emp: {:.1f}, e5: {:.1f}, e15: {:.1f}, e25: {:.1f}".format(nktime, emptime, e5time, e15time, e25time)
            times[0,0]+= nktime
            times[0,4]+= emptime
            times[0,1]+= e5time
            times[0,2]+= e15time
            times[0,3]+= e25time
    results /= iterations
    print results
    print times

    return results

def verify_kernel_matrix():
    n1 = 10
    n2 = 10
    n = n1+n2
    d = 5
    degree = 3
    X = randn(n,d)
    Phi = poly.phi(X, degree)
    (K, Kuc, Kc) = kernel_matrix(X, polyk(degree), n1, n2)
    P1 = Phi[0:n1,:]
    P2 = Phi[n1:n,:]

    mu1 = mat.sum(P1,0) / n1
    mu2 = mat.sum(P2,0) / n2
    P1c = P1 - mat.tile(mu1, (n1,1))
    P2c = P2 - mat.tile(mu2, (n2,1))
    Pc = bmat('P1c; P2c')

    KP = mat.zeros((n,n))
    for i in xrange(n):
        for j in xrange(i+1):
            KP[i,j] = dotp(Phi[i,:], Phi[j,:])
            KP[j,i] = KP[i,j]

    KucP = mat.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            KucP[i,j] = dotp(Phi[i,:], Pc[j,:])

    KcP = mat.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            KcP[i,j] = dotp(Pc[i,:], Pc[j,:])
            #KcP[j,i] = KcP[i,j]

    #debug()
    print "Div1: " + str(sum(abs(K-KP)))
    print "Div2: " + str(sum(abs(Kuc-KucP)))
    print "Div3: " + str(sum(abs(Kc-KcP)))

def main():
    test_suite_1()


if __name__ == '__main__':
    main()

