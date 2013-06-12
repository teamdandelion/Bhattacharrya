#!/usr/bin/env python
import numpy as np
import numpy.matlib as mat
import numpy.linalg as la
import math, hashlib, time, cPickle
import bhatta_poly as poly
from scipy.sparse.linalg import eigsh # Lanzcos algorithm

def Bhattacharrya_Kernel_Matrix(Data, kernel, eta, r):
    """Efficiently computes the Bhattacharrya kernel matrix for several datasets. M[i,j] = K(D[i], D[j])
    If all (or a large portion) of the matrix will be used, this is more efficient than calling the bhattacharrya kernel on its own.
    This function is a wrapper to the Bhatta_Manager class."""
    manager = Bhatta_Manager(Data, kernel, eta, r)
    return manager.bhatta_matrix()


def Bhattacharrya(kernel, r, eta):
    """Returns a hashed Bhattacharrya function with the kernel, eta, and r as given.
    Description:
    The Bhattacharrya function is a kernel between sets of vectors X1 and X2. The algorithm maps the two sets of vectors to a feature space using a base kernel, fits normal distributions to the data in that feature space, and then calculates the Bhattacharrya overlap between those probability distributions. For details, see the paper A Kernel Between Sets of Vectors by R. Kondor and T. Jebara (2003): 
        http://www1.cs.columbia.edu/~jebara/papers/KonJeb03.pdf

    The datasets X1 and X2 are represented as (n1 x d) and (n2 x d) matrices respectively. Each row corresponds to a data vector.

    Parameters:
    kernel: a base kernel used to map the data sets to feature space. I recommend a gaussian kernel 'gaussk(sigma)'
    r: the number of eigenvectors used to reconstruct the covariance matrix. I recommend between 5 and 20.
    eta: a regularization parameter used for 'smoothing' 

    Returns: A kernel function k(X1,X2). Computes the Bhattacharrya kernel between the two datasets."""
    MemoziedBhatta = MemoizedBhattaClass(kernel, r, eta)
    return MemoziedBhatta.bhatta

class MemoizedBhattaClass:
    """Handles Bhattacharrya kernel evaluations efficiently with a simple interface"""
    def __init__(self, kernel, r, eta):
        self.kernel = kernel
        self.eta = eta
        self.r = r 
        self.data = {}

    def bhatta(self, X1, X2):
        D1 = self.get_data(X1)
        D2 = self.get_data(X2)
        return BhattaFromDataset(D1,D2, self.eta)

    def get_data(self, X):
        hashval = array_hash(X)
        try:
            D = self.data[hashval]
        except KeyError:
            D = Dataset(X, self.kernel, self.r)
            self.data[hashval] = D
        return D


class Bhatta_Manager:
    """The goal of the Bhatta_Manager class is to handle a large number of Bhattacharrya kernel evaluations efficiently.
    It takes a list of datasets (D), where each dataset is expressed as a (n_i x d) matrix where each row is a data vector
    It precomputes the eigenvector decomposition of each dataset's covariance matrix, etc
    Parameters: eta = regularization parameter (try .1 to 1), r = number of eigenvectors (must be less than min(n_i))"""
    def __init__(self, Data, kernel, eta, r):
        # Data = [X1, X2, X3]...
        # kernel: positive-semidefinite kernel
        # r: number of eigenvectors to use, r < min(n1, n2...)
        # etas: [eta1, eta2, eta3] normalization factors
        self.kernel = kernel
        self.r = r
        self.eta = eta
        self.datasets = map(Dataset, Data)

    def bhatta_matrix(self):
        """Returns a matrix of Bhattacharrya kernel evaluations between datasets"""
        n = len(self.datasets)
        BM = mat.eye(n)
        for i in xrange(n):
            for j in xrange(i):
                BM[i,j] = self.bhatta(i,j)
                BM[j,i] = BM[i,j]
        return BM

    def bhatta(self,i,j):
        eta = self.eta
        D1 = self.datasets[i]
        D2 = self.datasets[j]
        Beta1 = D1.Beta
        Beta2 = D2.Beta
        (n1, r) = Beta1.shape
        (n2, r) = Beta2.shape
        n = n1 + n2
        Beta = mat.zeros((n,2*r))
        Beta[0:n1,0:r] = Beta1
        Beta[n1:n,r:2*r] = Beta2
        (K, Kuc, Kc) = self.kernel_supermatrix(i,j)
        Omega = eig_ortho(Kc, Beta)
        mu1 = mat.sum(Kuc[0:n1, :] * Omega, 0) / n1
        mu2 = mat.sum(Kuc[n1:n, :] * Omega, 0) / n2

        S1 = Omega.T * Kc[:,0:n1] * Kc[0:n1,:] * Omega / n1
        S2 = Omega.T * Kc[:,n1:n] * Kc[n1:n,:] * Omega / n2
        Eta= eta * mat.eye(2*r)    
        S1 += Eta
        S2 += Eta

        mu3 = .5 * (S1.I * mu1.T + S2.I * mu2.T).T
        S3 = 2 * (S1.I + S2.I).I

        d1 = la.det(S1) ** -.25
        d2 = la.det(S2) ** -.25
        d3 = la.det(S3) ** .5

        e1 = exp(-mu1 * S1.I * mu1.T / 4)
        e2 = exp(-mu2 * S2.I * mu2.T / 4)
        e3 = exp(mu3 * S3 * mu3.T / 2)

        dterm = d1*d2*d3
        eterm = e1*e2*e3
        rval = dterm*eterm

        if math.isnan(rval):
            rval = -1
            print "Warning: Kernel failed on datasets ({},{})".format(i,j)
        return rval


    def kernel_supermatrix(self, i, j):
        kernel = self.kernel
        D1 = self.datasets[i]
        D2 = self.datasets[j]
        X1 = D1.X
        X2 = D2.X
        (n1, d) = X1.shape
        (n2, d) = X2.shape
        n = n1 + n2
        X = mat.bmat('X1; X2')
        K1 = D1.K
        K2 = D2.K
        K = mat.zeros((n,n))
        K[0:n1, 0:n1] = K1
        K[n1:n, n1:n] = K2
        for i in xrange(n1):
            for j in xrange(n1, n):
                K[i,j] = kernel(X[i,:], X[j,:])
                K[j,i] = K[i,j]

        # Inelegant - improve later
        U1 = mat.sum(K[0:n1,:],0) / n1
        U2 = mat.sum(K[n1:n,:],0) / n2
        U1m = mat.tile(U1, (n1,1))
        U2m = mat.tile(U2, (n2,1))
        U = mat.bmat('U1m; U2m')
        m1m1 = mat.sum(K[0:n1, 0:n1]) / (n1*n1)
        m1m2 = mat.sum(K[0:n1, n1:n]) / (n1*n2)
        m2m2 = mat.sum(K[n1:n, n1:n]) / (n2*n2) 
        mumu = mat.zeros((n,n))
        mumu[0:n1, 0:n1] = m1m1
        mumu[0:n1, n1:n] = m1m2
        mumu[n1:n, 0:n1] = m1m2
        mumu[n1:n, n1:n] = m2m2
        Kcu = K - U
        Kuc = Kcu.T
        N = mat.ones((n,n))/n
        Kc = K - U - U.T + mumu
        return (K, Kuc, Kc)

class Dataset:
    """Manage bhattacharrya evaluations of a given dataset
    Contains original data, kernel, and (most importantly) eigenvector decomposition"""
    def __init__(self, X, kernel, r):
        self.X = X
        self.kernel = kernel
        self.r = r
        self.K, self.Kc = self.kernel_submatrix() # Kernel matrix and centered kernel matrix
        self.Beta = self.gen_beta() # Eigenvector decomposition

    def kernel_submatrix(self):
        X = self.X
        (n,d) = X.shape
        K = mat.zeros((n,n))
        for i in xrange(n):
            for j in xrange(i+1):
                K[i,j] = self.kernel(X[i,:], X[j,:])
                K[j,i] = K[i,j]

        Ki = mat.sum(K,1) / n
        k  = mat.sum(K) / (n*n)
        Kc = K - Ki * mat.ones((1,n)) - mat.ones((n,1)) * Ki.T + k * mat.ones((n,n))
        return (K, Kc)

    def gen_beta(self):
        (n,n_) = self.Kc.shape
        (Lam, Alpha) = eigsh(self.Kc, self.r)
        Alpha = mat.matrix(Alpha) 
        Lam = Lam / n # Eigenvalues
        def greater_than_zero(x): return x>0
        assert all(map(greater_than_zero,Lam))

        Beta = mat.zeros((n,self.r))
        for i in xrange(self.r):
            Beta[:, i] = Alpha[:,i] / (n * Lam[i])**.5

        return Beta


def BhattaFromDataset(D1, D2, eta):
    assert D1.r == D2.r
    kernel = D1.kernel
    Beta1 = D1.Beta
    Beta2 = D2.Beta
    (n1, r) = Beta1.shape
    (n2, r) = Beta2.shape
    n = n1 + n2
    Beta = mat.zeros((n,2*r))
    Beta[0:n1,0:r] = Beta1
    Beta[n1:n,r:2*r] = Beta2
    (K, Kuc, Kc) = kernel_supermatrix(D1,D2)
    Omega = eig_ortho(Kc, Beta)
    mu1 = mat.sum(Kuc[0:n1, :] * Omega, 0) / n1
    mu2 = mat.sum(Kuc[n1:n, :] * Omega, 0) / n2

    S1 = Omega.T * Kc[:,0:n1] * Kc[0:n1,:] * Omega / n1
    S2 = Omega.T * Kc[:,n1:n] * Kc[n1:n,:] * Omega / n2
    Eta= eta * mat.eye(2*r)    
    S1 += Eta
    S2 += Eta

    mu3 = .5 * (S1.I * mu1.T + S2.I * mu2.T).T
    S3 = 2 * (S1.I + S2.I).I

    d1 = la.det(S1) ** -.25
    d2 = la.det(S2) ** -.25
    d3 = la.det(S3) ** .5

    e1 = math.exp(-mu1 * S1.I * mu1.T / 4)
    e2 = math.exp(-mu2 * S2.I * mu2.T / 4)
    e3 = math.exp(mu3 * S3 * mu3.T / 2)

    dterm = d1*d2*d3
    eterm = e1*e2*e3
    rval = dterm*eterm

    if math.isnan(rval):
        rval = -1
        print "Warning: Kernel failed on datasets ({},{})".format(i,j)
    return rval

def Bhattacharrya(X1, X2, kernel, eta=.1, r=0):
    """Compute the Bhattacharrya kernel between datasets X1, X2
    X1 = (n1 x d) matrix, X2 = (n2 x d) matrix. Each row represents a data vector.
    eta is a regularization parameter, default value .1
    r is the number of eigenvectors to use. Default value 0 will try to guess a good parameter based on the data. If the parameter is -1, then it will compute the empirical covariance matrices rather than the eigenvector approach.
    Using the eigenvectors has better stability and is generally much more efficient, so that is recommended. However, if n is very small (n<10) then using the empirical approach may be more appropriate (untested)"""
    X1 = np.matrix(X1)
    X2 = np.matrix(X2)
    (n1, d) = X1.shape
    (n2, d_) = X2.shape
    assert d == d_
    n = n1+n2
    if r == 0:
        r = min(n1,n2) -3
        r = max(r, 2)
        assert r>0

    D1 = Dataset(X1, kernel, r)
    D2 = Dataset(X2, kernel, r)
    return BhattaFromDataset(D1,D2)


def kernel_supermatrix(D1, D2):
    kernel = D1.kernel
    X1 = D1.X
    X2 = D2.X
    (n1, d) = X1.shape
    (n2, d) = X2.shape
    n = n1 + n2
    X = mat.bmat('X1; X2')
    K1 = D1.K
    K2 = D2.K
    K = mat.zeros((n,n))
    K[0:n1, 0:n1] = K1
    K[n1:n, n1:n] = K2
    for i in xrange(n1):
        for j in xrange(n1, n):
            K[i,j] = kernel(X[i,:], X[j,:])
            K[j,i] = K[i,j]

    # Inelegant - improve later
    U1 = mat.sum(K[0:n1,:],0) / n1
    U2 = mat.sum(K[n1:n,:],0) / n2
    U1m = mat.tile(U1, (n1,1))
    U2m = mat.tile(U2, (n2,1))
    U = mat.bmat('U1m; U2m')
    m1m1 = mat.sum(K[0:n1, 0:n1]) / (n1*n1)
    m1m2 = mat.sum(K[0:n1, n1:n]) / (n1*n2)
    m2m2 = mat.sum(K[n1:n, n1:n]) / (n2*n2) 
    mumu = mat.zeros((n,n))
    mumu[0:n1, 0:n1] = m1m1
    mumu[0:n1, n1:n] = m1m2
    mumu[n1:n, 0:n1] = m1m2
    mumu[n1:n, n1:n] = m2m2
    Kcu = K - U
    Kuc = Kcu.T
    N = mat.ones((n,n))/n
    Kc = K - U - U.T + mumu
    return (K, Kuc, Kc)


def empirical_bhatta(X1, X2, kernel, eta, verbose=False):
    (n1, d) = X1.shape
    (n2, d_) = X2.shape
    assert (d == d_)
    try:
        (Kc, G, mu1, mu2) = prepare_bhatta(X1, X2, kernel, eta, verbose=verbose)
        (S1, S2) = e_covariance(n1, n2, Kc, G, mu1, mu2, eta)
        return bhatta_composition(mu1, mu2, S1, S2)
    except AssertionError:
        return -1

def nk_bhatta(X1, X2, eta):
    # Returns the non-kernelized Bhattacharrya
    #I.e. fits normal distributions in input space and calculates Bhattacharrya overlap between them
    (n1, d1) = X1.shape
    (n2, d ) = X2.shape
    assert d1 == d
    mu1 = mat.sum(X1,0) / n1
    mu2 = mat.sum(X2,0) / n2
    X1c = X1 - mat.tile(mu1, (n1,1))
    X2c = X2 - mat.tile(mu2, (n2,1))
    Eta = mat.eye(d) * eta
    S1 = X1c.T * X1c / n1 + Eta
    S2 = X2c.T * X2c / n2 + Eta

    mu3 = .5 * (S1.I * mu1.T + S2.I * mu2.T).T
    S3  = 2  * (S1.I + S2.I).I

    d1 = la.det(S1) ** -.25
    d2 = la.det(S2) ** -.25
    d3 = la.det(S3) ** .5
    dterm = d1 * d2 * d3

    e1 = -.25 * mu1 * S1.I * mu1.T
    e2 = -.25 * mu2 * S2.I * mu2.T
    e3 = .5   * mu3 * S3   * mu3.T

    eterm = math.exp(e1 + e2 + e3)

    return float(dterm * eterm)

def GS_basis(Kc, verbose=False):
    (n, n_) = Kc.shape
    assert n == n_
    if verbose: print "Beginning GS process. n = {}".format(n)
    G = mat.eye(n,n)
    deleted = 0
    for ell in xrange(n):
        if verbose: 
            if ell%10 == 0: print "ell = {}".format(ell)
        ell -= deleted
        for i in xrange(ell):
            G[:,ell] -= (G.T * Kc * G)[ell,i] * G[:,i]
        cf = (G.T*Kc*G)[ell,ell]
        if cf < (10**-8):
            if verbose: print "Deleting column " + str(ell)
            G = mat.delete(G, ell, 1)
            deleted += 1
        else:
            G[:,ell] /= cf ** .5
    GKG = G.T*Kc*G
    (dummy, g_dim) = G.shape
    assert not mat.isnan(G).any()
    if verbose: 
        print "GKG Divergence: {:e}".format(sum(abs(GKG - mat.eye(g_dim,g_dim))))
    return G



def eig_ortho(Kc, Beta):
    """Takes Beta, a matrix with coefficients for eigenvectors of two datasets. This function returns Omega, which has coefficients for the 'combined' orthonormal eigenvector set.
    Specifically, Beta[:,0:r] represents the coefficients for an orthonormal set of eigenvectors, and Beta[:,r:2r] represents the coefficients for another orthonormal set of eigenvectors. Omega has the coefficients for the combined orthonormal basis
    Omega[:,0:r] = Beta[:,0:r] since those eigenvectors were already orthonormal. Omega[:,r:2*r] is the second set of eigenvectors, orthogonalized relative to the first set"""
    (n, rr) = Beta.shape
    # rr literally stands for '2r'
    (n_, n__) = Kc.shape
    assert n == n_ == n__
    Gamma = mat.eye(rr)
    r = rr/2
    for i in xrange(r, rr): # Skips the first r columns since they're already orthonormal
        for j in xrange(i): # Make sure the vector is orthogonal to each previous vector
            Omega = Beta * Gamma
            g = float(Omega.T[i,:] * Kc * Omega[:,j])
            Gamma[:,i] -= g * Gamma[:,j]
        Omega = Beta * Gamma
        nrm = float(Omega.T[i,:] * Kc * Omega[:,i])
        assert nrm > 0
        nrm **=.5
        Gamma[:,i] /= nrm
    return Beta * Gamma

def prepare_bhatta(X1, X2, kernel, eta, verbose=False):
    (n1, d1) = X1.shape
    (n2, d ) = X2.shape
    assert d1 == d
    n = n1 + n2
    X = mat.bmat('X1;X2')
    (K, Kuc, Kc) = kernel_matrix(X, kernel, n1, n2)
    G = GS_basis(Kc, verbose)
    (null, g_dim) = G.shape

    mu1 = mat.sum(Kuc[0:n1,:] * G,0) / n1
    mu2 = mat.sum(Kuc[n1:n,:] * G,0) / n2

    return (Kc, G, mu1, mu2)

def e_covariance(n1, n2, Kc, G, mu1, mu2, eta):
    (n, n_) = Kc.shape    
    assert n == n_ == n1+n2
    (n_, gamma) = G.shape

    Eta = mat.eye((gamma)) * eta
    S1 = Eta + G.T * Kc[:,0:n1] * Kc[0:n1, :] * G / n1
    S2 = Eta + G.T * Kc[:,n1:n] * Kc[n1:n, :] * G / n2

    return (S1, S2)

def bhatta_composition(mu1, mu2, S1, S2):

    mu3 = (S1.I * mu1.T + S2.I * mu2.T).T * .5
    S3 = 2 * (S1.I + S2.I).I

    d1 = abs(la.det(S1)) ** -.25
    d2 = abs(la.det(S2)) ** -.25
    d3 = abs(la.det(S3)) ** .5
    dterm = d1*d2*d3

    e1 = mu1 * S1.I * mu1.T * -.25
    e2 = mu2 * S2.I * mu2.T * -.25
    e3 = mu3 * S3   * mu3.T * .5
    eterm = math.exp(e1 + e2 + e3)

    assert not(math.isnan(dterm*eterm))
    
    return float(dterm * eterm)

def pca_covariance(n1, n2, Kc, G, mu1, mu2, eta, r):
    # Completely untested
    Kc1 = Kc[0:n1, 0:n1]
    Kc2 = Kc[n1:n, n1:n]
    (Lam1, Alpha1) = eigsh(Kc1, r)
    (Lam2, Alpha2) = eigsh(Kc2, r)
    Alpha1 = mat.matrix(Alpha1)
    Alpha2 = mat.matrix(Alpha2)
    Lam1   = mat.matrix(Lam1 / n1)
    Lam2   = mat.matrix(Lam2 / n2)
    Beta1  = mat.zeros(n,r)
    Beta2  = mat.zeros(n,r)

    for i in xrange(r):
        Beta1[0:n1, i] = Alpha1[:,i] / (n1 * Lam1[i])
        Beta2[n1:n, i] = Alpha2[:,i] / (n2 * Lam2[i])

    Eta = mat.eye((gamma, gamma)) * eta # consider moving this to prep
    S1 = (G.T*Kc*Beta1) * diag(Lam1) * (G.T*Kc*Beta1).T
    S2 = (G.T*Kc*Beta2) * diag(Lam2) * (G.T*Kc*Beta2).T
    S1 += Eta
    S2 += Eta

    return (S1, S2)


def makediag(M):
    (c,) = M.shape
    D = mat.zeros((c,c))
    for i in xrange(c):
        D[i,i] = M[i]
    return D

def eig_bhatta(X1, X2, kernel, eta, r):
    # Tested. Verified:
    # Poly-kernel RKHS representations of all objects are roughly equal to eigenbasis representations (slight differences for S3)
    # Correctness for X1 ~= X2
    # Close results to empirical bhatta in test_suite_1
    # Remaining issues: Eigendecomposition of centered kernel matrices
    # occasionally produces negative-value eigenvalues
    (n1, d1) = X1.shape
    (n2, d2) = X2.shape
    assert d1==d2
    n = n1+n2
    X = mat.bmat("X1;X2")
    (K, Kuc, Kc) = kernel_matrix(X, kernel, n1, n2)
    Kc1 = Kc[0:n1, 0:n1]
    Kc2 = Kc[n1:n, n1:n]

    (Lam1, Alpha1) = eigsh(Kc1, r)
    (Lam2, Alpha2) = eigsh(Kc2, r)
    Alpha1 = matrix(Alpha1)
    Alpha2 = matrix(Alpha2)
    Lam1 = Lam1 / n1
    Lam2 = Lam2 / n2
    Beta1 = mat.zeros((n,r))
    Beta2 = mat.zeros((n,r))

    for i in xrange(r):
        Beta1[0:n1, i] = Alpha1[:,i] / (n1 * Lam1[i])**.5
        Beta2[n1:n, i] = Alpha2[:,i] / (n2 * Lam2[i])**.5


    #Eta = mat.eye((gamma, gamma)) * eta
    Beta = mat.bmat('Beta1, Beta2')
    assert not(any(math.isnan(Beta)))
    Omega = eig_ortho(Kc, Beta)
    mu1_w = mat.sum(Kuc[0:n1, :] * Omega, 0) / n1
    mu2_w = mat.sum(Kuc[n1:n, :] * Omega, 0) / n2

    Eta_w = eta * mat.eye(2*r)    

    S1_w = Omega.T * Kc[:,0:n1] * Kc[0:n1,:] * Omega / n1
    S2_w = Omega.T * Kc[:,n1:n] * Kc[n1:n,:] * Omega / n2
    S1_w += Eta_w
    S2_w += Eta_w

    mu3_w = .5 * (S1_w.I * mu1_w.T + S2_w.I * mu2_w.T).T
    S3_w = 2 * (S1_w.I + S2_w.I).I

    d1 = la.det(S1_w) ** -.25
    d2 = la.det(S2_w) ** -.25

    e1 = exp(-mu1_w * S1_w.I * mu1_w.T / 4)
    e2 = exp(-mu2_w * S2_w.I * mu2_w.T / 4)
    d3 = la.det(S3_w) ** .5
    e3 = exp(mu3_w * S3_w * mu3_w.T / 2)

    dterm = d1*d2*d3
    eterm = e1*e2*e3
    rval = float(dterm*eterm)

    if math.isnan(rval):
        rval = -1

    return rval

def kernel_matrix(X, kernel, n1, n2):
    (n, d) = X.shape
    assert n == n1 + n2

    K = mat.zeros((n,n))
    for i in xrange(n):
        for j in xrange(i+1):
            K[i,j] = kernel(X[i,:], X[j,:])
            K[j,i] = K[i,j]

    U1 = mat.sum(K[0:n1,:],0) / n1
    U2 = mat.sum(K[n1:n,:],0) / n2
    U1m = mat.tile(U1, (n1,1))
    U2m = mat.tile(U2, (n2,1))
    U = mat.bmat('U1m; U2m')
    m1m1 = mat.sum(K[0:n1, 0:n1]) / (n1*n1)
    m1m2 = mat.sum(K[0:n1, n1:n]) / (n1*n2)
    m2m2 = mat.sum(K[n1:n, n1:n]) / (n2*n2) 
    mumu = mat.zeros((n,n))
    mumu[0:n1, 0:n1] = m1m1
    mumu[0:n1, n1:n] = m1m2
    mumu[n1:n, 0:n1] = m1m2
    mumu[n1:n, n1:n] = m2m2
    Kcu = K - U
    Kuc = Kcu.T
    N = mat.ones((n,n))/n
    Kc = K - U - U.T + mumu
    return (K, Kuc, Kc)

def dotp(x,y):
    return float(inner(x,y))

def polyk(degree):
    return lambda x,y: dotp(x,y)**degree 

def gaussk(sigma):
    return lambda X,Y: gaussian_kernel(X,Y,sigma)

def gaussian_kernel(X,Y,sigma):
    return math.exp( -la.norm(X-Y)**2 / (2 * sigma**2))

def array_hash(A):
    return hashlib.sha1(A).hexdigest()

