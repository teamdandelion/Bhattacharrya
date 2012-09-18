from numpy.matlib import *
from numpy.linalg import *
import math
import bhatta
from pdb import set_trace as debug

def dotp(x,y):
    return float(inner(x,y))

def phi(X, deg):
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
    Phi = phi(X,deg)
    polydeg = lambda x,y: dotp(x,y)**deg
    (K , _, _) = bhatta.kernel_matrix(X, polydeg)
    (Kp, _, _) = bhatta.kernel_matrix(Phi, dotp)
    return sum(abs(K-Kp))

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
