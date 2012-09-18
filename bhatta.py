#!/usr/bin/env python

from numpy.matlib import *
from numpy.linalg import *
import math
import bhatta_poly as poly
from pdb import set_trace as debug

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

    #invS1 = inv(S1)
    #invS2 = inv(S2)

    mu3 = .5 * (S1.I * mu1.T + S2.I * mu2.T).T
    S3  = 2  * (S1.I + S2.I).I

    dt1 = det(S1) ** -.25
    dt2 = det(S2) ** -.25
    dt3 = det(S3) ** .5
    dterm = dt1 * dt2 * dt3

    e1 = -.25 * mu1 * S1.I * mu1.T
    e2 = -.25 * mu2 * S2.I * mu2.T
    e3 = .5   * mu3 * S3    * mu3.T

    eterm = math.exp(e1 + e2 + e3)

    return (dterm * eterm)

def GS_basis(Kc, n1, n2):
    n = n1 + n2
    G = eye(n,n)
    G = delete(G,n-1,1)
    G = delete(G,n1-1,1)
    for ell in xrange(n-2):
        for i in xrange(ell):
            print (ell+1, i+1)
            debug()
            G[:,ell] -= (G.T*Kc*G)[ell,i] * G[:,i]

        cf = (G.T*Kc*G)[ell,ell]
        assert cf >= 0
        G[:,ell] /= cf ** .5
    GKG = G.T*Kc*G
    #debug()
    #print "Divergence: " + str(sum(abs(GKG - eye(n-2,n-2))))
    return G

def empirical_bhatta(X1, X2, kernel, eta):
    (n1, d1) = X1.shape
    (n2, d ) = X2.shape
    assert d1 == d
    n = n1 + n2
    X = bmat('X1;X2')
    (K, Kuc, Kc) = kernel_matrix(X, kernel, n1, n2)
    G = GS_basis(Kc, n1, n2)

    mu1 = sum(Kuc[0:n1,:] * G,0) / n1
    mu2 = sum(Kuc[n1:n,:] * G,0) / n2
    Eta = eye((n-2)) * eta
    S1 = Eta + G.T * Kc[:,0:n1] * Kc[0:n1, :] * G / n1
    S2 = Eta + G.T * Kc[:,n1:n] * Kc[n1:n, :] * G / n2

    mu3 = (S1.I * mu1.T + S2.I * mu2.T).T * .5
    S3 = 2 * (S1.I + S2.I).I

    d1 = det(S1) ** -.25
    d2 = det(S2) ** -.25
    d3 = det(S3) ** .5
    dterm = d1*d2*d3

    e1 = mu1 * S1.I * mu1.T * -.25
    e2 = mu2 * S2.I * mu2.T * -.25
    e3 = mu3 * S3   * mu3.T * .5
    eterm = exp(e1 + e2 + e3)

    return dterm * eterm

def kernel_matrix(X, kernel, n1, n2):
    (n, d) = X.shape
    assert n == n1 + n2

    K = zeros((n,n))
    for i in xrange(n):
        for j in xrange(i+1):
            K[i,j] = kernel(X[i,:], X[j,:])
            K[j,i] = K[i,j]

    U1 = sum(K[0:n1,:],0) / n1
    U2 = sum(K[n1:n,:],0) / n2
    U1m = tile(U1, (n1,1))
    U2m = tile(U2, (n2,1))
    U = bmat('U1m; U2m')
    m1m1 = sum(K[0:n1, 0:n1]) / (n1*n1)
    m1m2 = sum(K[0:n1, n1:n]) / (n1*n2)
    m2m2 = sum(K[n1:n, n1:n]) / (n2*n2) 
    mumu = zeros((n,n))
    mumu[0:n1, 0:n1] = m1m1
    mumu[0:n1, n1:n] = m1m2
    mumu[n1:n, 0:n1] = m1m2
    mumu[n1:n, n1:n] = m2m2
    Kcu = K - U
    Kuc = Kcu.T
    N = ones((n,n))/n
    Kc = K - U - U.T + mumu
    return (K, Kuc, Kc)

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

    mu1 = sum(P1,0) / n1
    mu2 = sum(P2,0) / n2
    P1c = P1 - tile(mu1, (n1,1))
    P2c = P2 - tile(mu2, (n2,1))
    Pc = bmat('P1c; P2c')

    KP = zeros((n,n))
    for i in xrange(n):
        for j in xrange(i+1):
            KP[i,j] = dotp(Phi[i,:], Phi[j,:])
            KP[j,i] = KP[i,j]

    KucP = zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            KucP[i,j] = dotp(Phi[i,:], Pc[j,:])

    KcP = zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            KcP[i,j] = dotp(Pc[i,:], Pc[j,:])
            #KcP[j,i] = KcP[i,j]

    #debug()
    print "Div1: " + str(sum(abs(K-KP)))
    print "Div2: " + str(sum(abs(Kuc-KucP)))
    print "Div3: " + str(sum(abs(Kc-KcP)))

def test_suite_1():
    n1 = 15
    n2 = 15
    n = n1+n2
    d = 5
    eta = 1
    degree = 3
    iterations = 10
    results = zeros((8,3)) 
    # 1st col is non-kernelized
    # 2nd col is poly-kernel 

    for itr in xrange(iterations):
        X = randn(n1,d)
        Phi_X = poly.phi(X, degree)

        D0 = X + rand(n2,d) / 1000
        # Verify identity K(X,X) = 1
        D1 = randn(n2,d) 
        # How does kernel perform iid data
        D2 = rand(n2,d)
         # Uniform rather than normal distribution
        D3 = randn(n2,d) * 2 + 2
        # Linear transformation
        D4 = power(randn(n2,d) + 1 ,3) 
        #Non-linear transformation
        D5 = power(X+1,3) 
        #non-linear transformation of the D0 dataset;
        D6 = rand(n2,d)/100 + eye(n2,d) 
        #Totally different data - should have low similarity
        D7 = rand(n2,d)/100 + eye(n2,d)*5 
        # Scaled version of D7

        Data = [D0, D1, D2, D3, D4, D5, D6, D7]

        for idx in xrange(8):
            D = Data[idx]
            results[idx, 0] += nk_bhatta(X, D, 0)
            Phi_D = poly.phi(D, degree)
            results[idx, 1] += nk_bhatta(Phi_X, Phi_D, eta)
            results[idx, 2] += empirical_bhatta(X, D, polyk(degree), eta)
    results /= iterations

    return results

def dotp(x,y):
    return float(inner(x,y))

def polyk(degree):
    return lambda x,y: dotp(x,y)**degree 

def main():
    res = test_suite_1()
    print res

if __name__ == '__main__':
    main()

