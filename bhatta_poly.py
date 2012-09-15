#!/usr/bin/env python

from numpy.matlib import *
from numpy.linalg import *
import math
from pdb import set_trace as debug

def dotp(x,y):
    return float(inner(x,y))

def nk_bhatta(X1, X2, eta):
    # Make sure X1, X2 are matrix types
    (n1, d1) = X1.shape
    (n2, d ) = X2.shape
    assert d1 == d
    mu1 = sum(X1,0) / n1
    mu2 = sum(X2,0) / n2
    X1c = X1 - tile(mu1, (n1,1))
    X2c = X2 - tile(mu2, (n2,1))
    Eta = eye(d) * eta
    S1 = Eta + X1c.T * X1c / n1
    S2 = Eta + X2c.T * X2c / n2

    invS1 = inv(S1)
    invS2 = inv(S2)

    mu3 = .5 * (invS1 * mu1.T + invS2 * mu2.T).T
    S3  = 2  * inv(invS1 + invS2)

    dt1 = det(S1) ** -.25
    dt2 = det(S2) ** -.25
    dt3 = det(S3) ** .5
    dterm = dt1 * dt2 * dt3

    e1 = -.25 * mu1 * invS1 * mu1.T
    e2 = -.25 * mu2 * invS2 * mu2.T
    e3 = .5   * mu3 * S3    * mu3.T

    eterm = exp(e1 + e2 + e3)

    return (dterm * eterm)

def poly_phi(X, deg):
    """Maps the matrix X into the higher-dimensional matrix Phi which corresponds to using the polynomial kernel with degree deg"""
    (r, c) = X.shape
    compositions = list(composition_generator(deg,c))
    # Every possible length-c list which sums to deg
    # [1,2,0,3] corresponds to x1 * x2^2 * x4^3 etc
    phi_dimensionality = len(compositions)
    Phi = ones((r, phi_dimensionality))
    for i in xrange(phi_dimensionality):
        comp = compositions[i]
        for d in xrange(c):
            new_col = power(X[:,d], comp[d])
            Phi[:,i] = multiply(Phi[:,i], new_col)
        m = multi_coefficient(deg, comp)
        Phi[:,i] *= m ** .5
    return Phi

def test_poly_phi(n, d, deg):
    X = randn(n,d)
    Phi = poly_phi(X,deg)
    polydeg = lambda x,y: dotp(x,y)**deg
    K  = kernel_matrix(X, polydeg)
    Kp = kernel_matrix(Phi, dotp)
    return sum(abs(K-Kp))


def kernel_matrix(X, kernel):
    (n, d) = X.shape
    K = zeros((n,n))
    for i in xrange(n):
        for j in xrange(i+1):
            K[i,j] = kernel(X[i,:], X[j,:])
            K[j,i] = K[i,j]
    return K

def product(list):
    p = 1
    for i in list:
        p *= i
    return p

def multi_coefficient(n, comp):
    return math.factorial(n) / product(map(math.factorial, comp))




def composition_generator(n,r):
    if r == 1:
        yield [n]
    else:
        for x in xrange(n+1):
            for combo in composition_generator(n-x, r-1):
                yield [x] + combo



def val(n,r):
    return len(list(composition_generator(n,r)))

def display(n,r):
    for combo in composition_generator(n,r):
        print combo


def test(n1, d):
    X1 = randn(n1,d)
    X2 = X1 + rand(n1,d) / 10000
    print nk_bhatta(X1, X2 ,.01)

def main():
    print "hellO"
    test(10, 5)

if __name__ == '__main__':
    main()
