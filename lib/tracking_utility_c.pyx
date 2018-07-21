import math
import numpy as np
from numpy import *

# cython
cimport numpy as np

def get_uDp(np.ndarray[double, ndim=3] x):
    #assume x = array([[[x1,y1],[x2,y2],[x3,y3]],[],...])
    #return array([[u,v,w],[],...])
    
    cdef int i
    cdef np.ndarray[double, ndim=1] q
    cdef np.ndarray[double, ndim=1] cxe, cye, re, u, D, phi, ATan0, ATan1, cos_phi, sin_phi, ax, ay, sumxy
    cdef np.ndarray[double, ndim=2] sumx, sumx2, G, T
    cdef np.ndarray[double, ndim=3] F, invF
    #cdef np.ndarray[np.unit8_t, ndim=1, cast=True] con_lv0, con_lv1
    #cdef np.ndarray[np.unit8_t, ndim=2, cast=True] condition
    cdef list con_lv0, con_lv1
    cdef list condition

    if x.shape[2] != 2:
        return np.zeros(3)

    sumx = np.sum(x,axis=1)
    sumx2 = np.sum(x**2,axis=1)
    sumxy = np.sum(x[:,:,0]*x[:,:,1],axis=1)

    F = np.array([np.array([sumx2[:,0],sumxy,sumx[:,0]]),
                  np.array([sumxy,sumx2[:,1],sumx[:,1]]),
                  np.array([sumx[:,0], sumx[:,1], np.ones(x.shape[0])*x.shape[1]])]).transpose()
    G = np.array([-np.sum(x[:,:,0]**3 + x[:,:,0] * (x[:,:,1]**2),axis=1),
                -np.sum((x[:,:,0]**2) * x[:,:,1] + x[:,:,1]**3,axis=1),
                -np.sum(x[:,:,0]**2 + x[:,:,1]**2,axis=1)]).transpose()

    invF = np.linalg.inv(F)
    T = np.zeros((x.shape[0],3))
    for i in range(3):
        T[:,i] = np.sum(invF[:,i]*G,axis=1)


    cxe = -T[:,0]/2
    cye = -T[:,1]/2
    re = np.sqrt(cxe**2 + cye**2 - T[:,2])
    

    u = 1./re
    D = re - np.sqrt(cxe**2 + cye**2) 
    phi = np.arctan2(-cxe,cye)
    phi[phi<0] += math.pi

    ATan0 = np.arctan((x[:,0,1]-cye)/(x[:,0,0]-cxe)) 
    ATan1 = np.arctan((x[:,1,1]-cye)/(x[:,1,0]-cxe)) 

    con_lv0 = [x[:,0,0] < cxe, 
               ~(x[:,0,0] < cxe) & (ATan0 < 0), 
               ~(x[:,0,0] < cxe) & ~(ATan0 < 0)]
    con_lv1 = [x[:,1,0] < cxe,
               ~(x[:,1,0] < cxe) & (ATan1 < 0),
               ~(x[:,1,0] < cxe)]
    ATan0[con_lv0[0]] += math.pi
    ATan1[con_lv0[0] & con_lv1[0]] += math.pi
    ATan1[con_lv0[0] & con_lv1[1]] += 2*math.pi
    ATan0[con_lv0[1]] += 2*math.pi
    ATan1[con_lv0[1] & con_lv1[0]] += math.pi
    ATan1[con_lv0[1] & con_lv1[2]] += 2*math.pi
    ATan1[con_lv0[2] & con_lv1[0]] += math.pi
    ATan1[con_lv0[2] & con_lv1[1]] += 2*math.pi
    ATan0[con_lv0[2] & con_lv1[1]] += 2*math.pi

    q = np.ones(x.shape[0])
    q[ATan1>ATan0] = -1
    
    u *= q
    D *= q
    #check
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    ax = -q * (1./u - D)*sin_phi
    ay = q * (1./u - D)*cos_phi
    #print(ax, ay)
    condition = [(phi>math.pi) & ((ax - cxe)**2 + (ay - cye)**2 >1.e1), (phi<math.pi) & ((ax - cxe)**2 + (ay - cye)**2 >1.e1)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi

    condition = [(q<0) & (phi>math.pi), (q<0) & ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi
    #eta *= q

    return (u, D, phi)

