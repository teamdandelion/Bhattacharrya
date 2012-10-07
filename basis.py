# Scratch space for working with change of basis transformations
from numpy.matlib import *
import copy
#from pdb import set_trace as debug

def cdot(a,b):
    return float(a.T*b)


def decompose(V):
    (r,c) = V.shape
    W = V.copy()
    G = eye(c)
    for i in xrange(c):
        for j in xrange(i):
            cf = cdot(W[:,i], W[:,i])
            G[:,i] -= cf * G[:,j]
            W[:,i] -= cf * W[:,j]
        nf = cdot(W[:,i], W[:,i]) ** (.5)
        G[:,i] /= nf
        W[:,i] /= nf

    return W,G



def orthogonalize(M):
    (r,c) = M.shape
    O = M.copy()
    for i in xrange(c):
        V = O[:,i]
        for j in xrange(i):
            U = O[:,j]
            #debug()
            V -= U * cdot(V,U) / cdot(U,U)

    return O

def is_normal(M):
    (r,c) = M.shape
    try:
        for i in xrange(c):
            assert cdot(M[:,i],M[:,i]) - 1 < .000001
        return 1
    except AssertionError:
        return 0

def is_ortho(M):
    (r,c) = M.shape
    try:
        for i in xrange(c):
            for j in xrange(i):
                assert cdot(M[:,i],M[:,j]) < .000001
        return 1
    except AssertionError:
        return 0

def is_on(M):
    return is_ortho(M) and is_normal(M)


A = mat("1., 0., 0.; 1., 1., 0.; 0., 0., 1.")
B = orthogonalize(A)

def x_ortho(B):
    debug
    C = zeros((4,2))
    C[2,0] = 1
    C[3,1] = 1    
    B3 = B[:,2]
    B4 = B[:,3]

    for i in xrange(2):
        oB = B[:,i]
        C[i,0] = -cdot(B3,oB) / cdot(oB,oB)
        B3 += C[i,0] * oB

    for i in xrange(3):
        oB = B[:,i]
        C[i,1] = -cdot(B4,oB) / cdot(oB,oB)
        B4 += C[i,0] * oB
    return C

def ort(v, ov):
    return v - ov * cdot(v,ov) / cdot(ov,ov)


def cort(v, ov):
    v -= ov * cdot(v,ov) / cdot(ov,ov)
    return cdot(v,ov) / cdot(ov,ov)

def orthol(v, vlist):
    for ov in vlist:
        v = ort(v,ov)
    return v

def orthom(v, vmat):
    (r,c) = vmat.shape
    for i in xrange(c):
        ov = vmat[:,i]
        v = ort(v,ov)
    return v