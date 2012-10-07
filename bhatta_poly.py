#!/usr/bin/env python
from numpy.matlib import *
from numpy.linalg import *
import math
#import bhatta
from scipy.sparse.linalg import eigsh # Lanzcos algorithm
from bhatta import *
import bhatta
#from pdb import set_trace as debug

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


def eig_poly(X1, X2, deg, eta, r):
    # Not tested in the slightest ... probably all broken

    # Make Kc1, Kc2
    # U1.T * S1 * U1
    kernel = bhatta.polyk(deg)
    (n1, d1) = X1.shape
    (n2, d) = X2.shape
    assert d1==d
    n = n1+n2
    X = bmat("X1;X2")

    #Let's get this shit forreal now:
    Xp = phi(X,deg)
    (n, pd) = Xp.shape
    X1p = Xp[0:n1, :]
    X2p = Xp[n1:n, :]
    mu1p = sum(X1p,0) / n1
    mu2p = sum(X2p,0) / n2
    X1cp = X1p - tile(mu1p, (n1,1))
    X2cp = X2p - tile(mu2p, (n2,1))
    Xcp = bmat('X1cp; X2cp')
    Etap = eye(pd) * eta
    S1p  = Etap + X1cp.T * X1cp / n1
    S2p  = Etap + X2cp.T * X2cp / n2

    mu3p = .5 * (S1p.I * mu1p.T + S2p.I * mu2p.T).T
    S3p  = 2  * (S1p.I + S2p.I).I

    dt1p = det(S1p) ** -.25
    dt2p = det(S2p) ** -.25
    dt3p = det(S3p) ** .5
    dterm = dt1p * dt2p * dt3p

    e1p = -.25 * mu1p * S1p.I * mu1p.T
    e2p = -.25 * mu2p * S2p.I * mu2p.T
    e3p = .5   * mu3p * S3p   * mu3p.T

    eterm = math.exp(e1p + e2p + e3p)

    (Lam1p, Alpha1p) = eigsh(S1p, r)
    (Lam2p, Alpha2p) = eigsh(S2p, r)
    Alpha1p = matrix(Alpha1p)
    Alpha2p = matrix(Alpha2p)



    (K, Kuc, Kc) = bhatta.kernel_matrix(X, kernel, n1, n2)
    Kc1 = Kc[0:n1, 0:n1]
    Kc2 = Kc[n1:n, n1:n]

    (Lam1, Alpha1) = eigsh(Kc1, r)
    (Lam2, Alpha2) = eigsh(Kc2, r)
    Alpha1 = matrix(Alpha1)
    Alpha2 = matrix(Alpha2)
    Lam1 = Lam1 / n1
    Lam2 = Lam2 / n2
    Beta1 = zeros((n,r))
    Beta2 = zeros((n,r))

    for i in xrange(r):
        Beta1[0:n1, i] = Alpha1[:,i] / (n1 * Lam1[i])**.5
        Beta2[n1:n, i] = Alpha2[:,i] / (n2 * Lam2[i])**.5


    #Eta = eye((gamma, gamma)) * eta
    Beta = bmat('Beta1, Beta2')
    Gamma = eig_ortho(Kc, Beta)
    Omega = Beta * Gamma
    V = Xcp.T * Beta #RKHS representation of eigenvectors
    #V[0:r] is orthonormal, V[r:2r] is orthonormal
    W = Xcp.T * Omega # RKHS representation of orthogonalized eigenvectors

    mu1_v = sum(Kuc[0:n1, :] * Beta1, 0) / n1
    mu2_v = sum(Kuc[n1:n, :] * Beta2, 0) / n2
    mu1_w = sum(Kuc[0:n1, :] * Omega, 0) / n1
    mu2_w = sum(Kuc[n1:n, :] * Omega, 0) / n2

    Eta_v = eta * eye(r)
    Eta_w = eta * eye(2*r)
    S1_v = makediag(Lam1) + Eta_v
    S2_v = makediag(Lam2) + Eta_v

    #d1 = (product(Lam1 + eta) * eta ** n2) ** -.25
    #d2 = product(Lam2 + eta) ** -.25


    

    S1_w = Omega.T * Kc[:,0:n1] * Kc[0:n1,:] * Omega / n1
    S2_w = Omega.T * Kc[:,n1:n] * Kc[n1:n,:] * Omega / n2
    S1_w += Eta_w
    S2_w += Eta_w

    mu3_w = .5 * (S1_w.I * mu1_w.T + S2_w.I * mu2_w.T).T
    S3_w = 2 * (S1_w.I + S2_w.I).I

    d1 = det(S1_w) ** -.25
    d2 = det(S2_w) ** -.25

    e1 = exp(-mu1_w * S1_w.I * mu1_w.T / 4)
    e2 = exp(-mu2_w * S2_w.I * mu2_w.T / 4)
    d3 = det(S3_w) ** .5
    e3 = exp(mu3_w * S3_w * mu3_w.T / 2)

    try:
        assert not(isnan(d1*d2*d3*e1*e2*e3))
        rval = float(d1*d2*d3*e1*e2*e3)
    except AssertionError:
        rval = -1

    ebhatta = eterm*dterm
    #print "Explicit: {} Eigen: {}".format(ebhatta,rval)
    #print "Diff: {} Ratio: {}".format(ebhatta-rval, ebhatta/rval)
    #print "---"

    return (rval)


def main():
    X1 = randn(20,6)
    X2 = randn(20,6)
    deg = 3
    (explicit, eigen) = eig_poly(X1, X2, deg, .1, 19)
    print explicit, eigen

if __name__ == '__main__':
    main()
