import math
import numpy as np
from numpy import *
import detector as det
import pandas as pd
#from scipy import optimize

def fit_circle(x,y):
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix,iy) in zip(x,y)])

    F = np.array([[sumx2,sumxy,sumx],
                  [sumxy,sumy2,sumy],
                  [sumx, sumy, len(x)]])
    G = np.array([[-sum([ix ** 3 + ix*iy **2 for (ix,iy) in zip(x,y)])],
                  [-sum([ix **2 *iy + iy **3 for (ix,iy) in zip(x,y)])],
                  [-sum([ix ** 2 + iy **2 for (ix,iy) in zip(x,y)])]])

    T = np.linalg.inv(F).dot(G)

    cxe = float(T[0]/-2)
    cye = float(T[1]/-2)
    re = math.sqrt(cxe**2 + cye**2 - T[2])

    return (cxe,cye,re)

#def get_1uDp(x,y):
    #assume x = array([x1, x2, x3]) and y = array([y1, y2, y3]). return (u, D, phi)
#    if(len(x)!=3 or len(y)!=3):
#        return (0,0,0)
#    r2 = x*x + y*y
#    r2_dif = [r2[1] - r2[2],r2[2] - r2[0], r2[0] - r2[1]]
#    deno = sum(x * r2_dif)
    #protect from zero divide
#    if deno == 0:
#        deno = 1.e-20
#    phi = math.atan(sum(y * r2_dif)/deno)
#    if phi < 0:
#        phi += math.pi

#    rd = -r2_dif[2]/2.0/(math.sin(phi)*(x[0]-x[1])-math.cos(phi)*(y[0]-y[1]))
#    if rd < 0:
#        rd = -rd
#        phi += math.pi

#    rr = r2[0] + 2.0*rd*(math.sin(phi)*x[0]-math.cos(phi)*y[0])+rd**2
#    u = 1.0/math.sqrt(rr)
#    D = math.sqrt(rr) - rd

#    deno = x[0]+rd*math.sin(phi)
    #protect from zero divide
#    if deno == 0:
#        deno = 1.e-20
#    ATan0 = math.atan((y[0]-rd*math.cos(phi))/deno) 

#    deno = x[1]+rd*math.sin(phi)
    #protect from zero divide
#    if deno == 0:
#        deno = 1.e-20
#    ATan1 = math.atan((y[1]-rd*math.cos(phi))/deno)
#    if x[0] < -rd*math.sin(phi):
#        ATan0 += math.pi
#        if x[1] < -rd*math.sin(phi):
#            ATan1 += math.pi
#        elif ATan1 < 0:
#            ATan1 += 2*math.pi
#    elif ATan0 < 0:
#        ATan0 += 2*math.pi
#        if x[1] < -rd*math.sin(phi):
#            ATan1 += math.pi
#        else:
#            ATan1 += 2*math.pi
#    else:
#        if x[1] < -rd*math.sin(phi):
#            ATan1 += math.pi
#        elif ATan1 < 0:
#            ATan1 += 2*math.pi
#            ATan0 += 2*math.pi

#    q = 1
#    if ATan1 < ATan0:
#        q = -1
    
    #z is not used
    #ASin1 = calcurate_asin(math.arcsin((y[1] - rd*math.cos(phi))*u), q, x[1], -rd*math.sin(phi))
    #ASin0 = calcurate_asin(math.arcsin((y[0] - rd*math.cos(phi))*u), q, x[0], -rd*math.sin(phi))
    #eta = q * sqrt(rr) * (ASin1 - ASin0) / (z[1] - z[0])
    #nanika = z[0] - q * math.sqrt(rr) * (ASin0 - phi + math.pi/2)/eta
    
#    u *= q
#    D *= q
#    if q < 0:
#        if phi > math.pi:
#            phi -= math.pi
#        else:
#            phi += math.pi
    #eta *= q

#    return (u, D, phi)

def get_uDp(x):
    #assume x = array([[[x1,y1],[x2,y2],[x3,y3]],[],...])
    #return array([[u,v,w],[],...])
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
    #print(cxe, cye, re)

    #r2 = x[:,:,0]**2 + x[:,:,1]**2
    #deno = 2.*((x[:,0,0] - x[:,1,0])*(x[:,1,1]-x[:,2,1]) - (x[:,1,0] - x[:,2,0])*(x[:,0,1] - x[:,1,1]))
    #deno[deno==0] = 1.e-20
    #cxe = (x[:,1,1] - x[:,2,1])*(r2[:,0] - r2[:,1]) - (x[:,0,1] - x[:,1,1])*(r2[:,1] - r2[:,2])
    #cxe /= deno
    #cye = (x[:,1,0] - x[:,2,0])*(r2[:,0] - r2[:,1]) - (x[:,0,0] - x[:,1,0])*(r2[:,1] - r2[:,2])
    #cye /= -deno
    #ri = np.sqrt((x[:,:,0] - cxe[:,newaxis])**2 + (x[:,:,1] - cye[:,newaxis])**2)
    #re = np.mean(ri,axis=1)
    #print(cxe, cye, re)
    

    u = 1./re
    D = re - np.sqrt(cxe**2 + cye**2) 
    phi = np.arctan2(-cxe,cye)
    phi[phi<0] += math.pi
    #print(cxe,cye,re)

    #r2 = (x*x).sum(axis=2)
    #r2_dif = np.array([r2[:,1] - r2[:,2],r2[:,2] - r2[:,0], r2[:,0] - r2[:,1]]).transpose()
    #phi = np.arctan((x[:,:,1] * r2_dif).sum(axis=1)/(x[:,:,0] * r2_dif).sum(axis=1))
    #phi[phi<0] += math.pi

    #rd = -r2_dif[:,2]/2.0/(np.sin(phi)*(x[:,0,0]-x[:,1,0])-np.cos(phi)*(x[:,0,1]-x[:,1,1]))
    #phi[rd<0] += math.pi
    #rd[rd<0] *= -1

    #rr = r2[:,0] + 2.0*rd*(np.sin(phi)*x[:,0,0]-np.cos(phi)*x[:,0,1])+rd**2
    #u = 1.0/np.sqrt(rr)
    #D = np.sqrt(rr) - rd

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
    

    #z is not used
    #ASin1 = calcurate_asin(math.arcsin((y[1] - rd*math.cos(phi))*u), q, x[1], -rd*math.sin(phi))
    #ASin0 = calcurate_asin(math.arcsin((y[0] - rd*math.cos(phi))*u), q, x[0], -rd*math.sin(phi))
    #eta = q * sqrt(rr) * (ASin1 - ASin0) / (z[1] - z[0])
    #nanika = z[0] - q * math.sqrt(rr) * (ASin0 - phi + math.pi/2)/eta
    
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

def get_jacobian(state,module_trans):
    ndata = state.shape[0]
    ndim_state = state.shape[1]
    ndim_obsrv = 1
    
    cx = module_trans['cx']
    cy = module_trans['cy']
    cos_theta = module_trans['cos_theta']
    sin_theta = module_trans['sin_theta']
    #protect from 0 divide
    state[:,0][state[:,0]==0] = 1e-20

    #define parameters
    q = np.ones(ndata)
    q[state[:,0]<0] = -1
    phi = state[:,2]
    condition = [(q<0) & (phi>math.pi), (q<0) & ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    ax = -q * (1./state[:,0] - state[:,1])*sin_phi
    ay = q * (1./state[:,0] - state[:,1])*cos_phi
    root = 1./state[:,0]/state[:,0] - ((cx-ax)*sin_theta - (cy-ay)*cos_theta)**2
    root[root>0] = np.sqrt(root[root>0])
    root[root<0] = 1.e-20
    cos_phi[phi>math.pi] = np.cos(phi[phi>math.pi]-math.pi)
    sin_phi[phi>math.pi] = np.sin(phi[phi>math.pi]-math.pi)

    #calculate jacobian
    jacobian = np.zeros((ndata,ndim_obsrv,ndim_state))
    jacobian[:,0,0] = (sin_phi*cos_theta - cos_phi*sin_theta)/state[:,0]/state[:,0]
    tmp = (sin_phi*sin_theta - cos_phi*sin_theta)*((cx-ax)*sin_theta - (cy-ay)*cos_theta) - 1./state[:,0]
    tmp /= root*state[:,0]*state[:,0]
    jacobian[:,0,0][q<0] -= tmp[q<0]
    jacobian[:,0,0][q>0] += tmp[q>0]
    #print(jacobian[:,0,0])

    jacobian[:,0,1] = sin_phi*cos_theta - cos_phi*sin_theta
    #print(jacobian[:,0,1])
    tmp = (sin_phi*sin_theta + cos_phi*cos_theta)*((cx-ax)*sin_theta - (cy-ay)*cos_theta)/root 
    #print(tmp)
    jacobian[:,0,1][q<0] -= tmp[q<0]
    jacobian[:,0,1][q>0] += tmp[q>0]
    #print(jacobian[:,0,1])

    jacobian[:,0,2] = - (1./state[:,0] - state[:,1])*(cos_phi*cos_theta + sin_phi*sin_theta)
    #print(jacobian[:,0,2])
    tmp = (1./state[:,0] - state[:,1])*(cos_phi*sin_theta - sin_phi*cos_theta)*((cx-ax)*sin_theta - (cy-ay)*cos_theta)/root 
    #print(tmp)
    jacobian[:,0,2][q<0] += tmp[q<0]
    jacobian[:,0,2][q>0] -= tmp[q>0]
    #print(jacobian[:,0,2])

    #not use z info
    #jacobian[:,0,3] = 0
    #jacobian[:,0,4] = 0

    return jacobian

def get_jacobian_numerically(state,module_trans):
    eps = 1.e-10

    jacobian = np.zeros((state.shape[0],1,state.shape[1]))
    obsrv = get_predict_obsrv(state,module_trans)
    for i in range(state.shape[1]):
        state_tmp = state
        state_tmp[:,i] += state[:,i]*eps
        jacobian[:,0,i] = get_predict_obsrv(state_tmp,module_trans)[:,0] - obsrv[:,0]
        jacobian[:,0,i] /= state[:,i]*eps

    return jacobian

def get_predict_obsrv(state,module_trans):
    ndata = state.shape[0]
    ndim_state = state.shape[1]
    ndim_obsrv = 1
    
    cx = module_trans['cx']
    cy = module_trans['cy']
    cos_theta = module_trans['cos_theta']
    sin_theta = module_trans['sin_theta']
    #protect from 0 divide
    state[:,0][state[:,0]==0] = 1e-20
    #define parameters
    q = np.ones(ndata)
    q[state[:,0]<0] = -1
    phi = state[:,2]
    condition = [(q<0) & (phi>math.pi), (q<0) & ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    ax = -q * (1./state[:,0] - state[:,1])*sin_phi
    ay = q * (1./state[:,0] - state[:,1])*cos_phi
    #print('cos ',cos_phi, 'sin ',sin_phi, 'ax ',ax,'ay ',ay)
    root = 1./state[:,0]/state[:,0] - ((cx-ax)*sin_theta - (cy-ay)*cos_theta)**2
    root[root>0] = np.sqrt(root[root>0])
    root[root<0] = 1.e-20
    #cos_phi[phi>math.pi] = np.cos(phi[phi>math.pi]-math.pi)
    #sin_phi[phi>math.pi] = np.sin(phi[phi>math.pi]-math.pi)

    #calculate hit position in local
    #print('root ',root)
    predict_obsrv = np.zeros((ndata,ndim_obsrv))
    predict_obsrv[:,0] = -(cx-ax)*cos_theta - (cy-ay)*sin_theta
    #print('cos ',cos_theta, 'sin ',sin_theta, 'cx ',cx,'cy ',cy)
    #print('obsrv ', predict_obsrv[:,0])
    #print(q)
    predict_obsrv[:,0][q<0] -= root[q<0]
    predict_obsrv[:,0][q>0] += root[q>0]

    return predict_obsrv

def invert_state(state):
    ndata = state.shape[0]
    ndim_state = state.shape[1]
    q = np.ones(ndata)
    q[state[:,0]<0] = -1
    phi = state[:,2]
    condition = [(q<0) & (phi>math.pi), (q<0) & ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi
    condition = [(q>0) & (phi>math.pi), (q>0) & ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi
    state[:,0] *= -1
    state[:,1] *= -1
    state[:,2] = phi

    return state

def rotate_state(state):
    phi = state[:,2]
    condition = [phi>math.pi, ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi
    state[:,2] = phi

    return state

def get_1uDp(x,y):
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix,iy) in zip(x,y)])

    F = np.array([[sumx2,sumxy,sumx],
                  [sumxy,sumy2,sumy],
                  [sumx, sumy, len(x)]])
    G = np.array([[-sum([ix ** 3 + ix*iy **2 for (ix,iy) in zip(x,y)])],
                  [-sum([ix **2 *iy + iy **3 for (ix,iy) in zip(x,y)])],
                  [-sum([ix ** 2 + iy **2 for (ix,iy) in zip(x,y)])]])

    T = np.linalg.inv(F).dot(G)

    cxe = float(T[0]/-2)
    cye = float(T[1]/-2)
    re = math.sqrt(cxe**2 + cye**2 - T[2])

    #if re < 200:
        #assume passing (0,0)
    #    xp = np.append(x,0)
    #    yp = np.append(y,0)
    #    cxe, cye, re = fit_circle(xp, yp)

    #print(cxe, cye, re)

    u = 1./re
    D = re - np.sqrt(cxe**2 + cye**2)
    #if D**2 > re**2:
    #    D -= 2*re
    phi = np.arctan2(-cxe,cye)
    if phi < 0:
        phi += math.pi

    deno = x[0]-cxe
    #protect from zero divide
    if deno == 0:
        deno = 1.e-20
    ATan0 = math.atan((y[0]-cye)/deno) 

    deno = x[2]-cxe
    #protect from zero divide
    if deno == 0:
        deno = 1.e-20
    ATan1 = math.atan((y[2]-cye)/deno)
    if x[0] < cxe:
        ATan0 += math.pi
        if x[2] < cxe:
            ATan1 += math.pi
        elif ATan1 < 0:
            ATan1 += 2*math.pi
    elif ATan0 < 0:
        ATan0 += 2*math.pi
        if x[2] < cxe:
            ATan1 += math.pi
        else:
            ATan1 += 2*math.pi
    else:
        if x[2] < cxe:
            ATan1 += math.pi
        elif ATan1 < 0:
            ATan1 += 2*math.pi
            ATan0 += 2*math.pi

    q = 1
    if ATan1 > ATan0:
        q = -1

    u *= q
    D *= q

    #check
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    ax = -q * (1./u - D)*sin_phi
    ay = q * (1./u - D)*cos_phi
    if (ax - cxe)**2 + (ay - cye)**2 >1.e2:
        if phi > math.pi:
            phi -= math.pi
        else:
            phi += math.pi

    
    if q < 0:
        if phi > math.pi:
            phi -= math.pi
        else:
            phi += math.pi

    return (u, D, phi)


class Tracking:
    def __init__(self,nhit_max):
        self.df = pd.DataFrame()
        self.nhit_max = nhit_max

    def get_track_id(self):
        return self.df.track_id.values

    def get_hit_id(self):
        hit_id = np.zeros((ndata,nhit_max))
        for ihit in range(self.nhit_max):
            hit_id[ihit] = self['hit_id_' + str(ihit)].transpose()
        return hit_id

    def get_state(self):
        return np.array([self.df.u.values,self.df.D.values,self.df.phi.values]).transpose()

    def get_state_cov(self):
        cov_u = np.array([self.df.cov_uu.values,self.df.cov_Du.values,self.df.cov_pu.values])
        cov_D = np.array([self.df.cov_uD.values,self.df.cov_DD.values,self.df.cov_pD.values])
        cov_p = np.array([self.df.cov_up.values,self.df.cov_Dp.values,self.df.cov_pp.values])
        return np.array([cov_u,cov_D,cov_p]).transpose()
  
    def get_sum_chi2(self):
        return self.df.sum_chi2.values

    def set_track_id(self,track_id):
        self.df['track_id'] = track_id
   
    def set_hit_id(self,hit_id):
        for ihit in range(self.nhit_max):
            self.df['hit_id_' + str(ihit)] = hit_id[:,ihit]
   
    def set_state(self,state):
        self.df['u'] = state[:,0]
        self.df['D'] = state[:,1]
        self.df['phi'] = state[:,2]

    def set_state_cov(self,state_cov):
        self.df['cov_uu'] = state_cov[:,0,0]
        self.df['cov_uD'] = state_cov[:,0,1]
        self.df['cov_up'] = state_cov[:,0,2]
        self.df['cov_Du'] = state_cov[:,1,0]
        self.df['cov_DD'] = state_cov[:,1,1]
        self.df['cov_Dp'] = state_cov[:,1,2]
        self.df['cov_pu'] = state_cov[:,2,0]
        self.df['cov_pD'] = state_cov[:,2,1]
        self.df['cov_pp'] = state_cov[:,2,2]

    def set_sum_chi2(self,sum_chi2):
        self.df['sum_chi2'] = sum_chi2

    def add_track_id(self,track_id):
        np.append(self.df.track_id.values,track_id)

    def add_hit_id(self,hit_id):
        np.append(self.df.hit_id.values,hit_id)

    def add_state(self,state):
        np.append(self.df.state.values,state)

    def add_state_cov(self,state_cov):
        np.append(self.df.state_cov.values,state_cov)


    def add_sum_chi2(self,sum_chi2):
        np.append(self.df.sum_chi2.values,sum_chi2)

    def cut_by_chi2(self,limit):
        cut = self.df.sum_chi2 > limit
        track_id = np.zeros(len(self.df.track_id))
        hit_id = np.zeros((self.get_hit_id().shape[0],self.get_hit_id().shape[1]))
        state = np.zeros((self.get_state().shape[0],self.get_state().shape[1]))
        state_cov = np.zeros((self.get_state_cov().shape[0],self.get_state_cov().shape[1],self.get_state_cov().shape[2]))
        sum_chi2 = np.zeros(len(self.df.sum_chi2))
        n = 0
        for i in range(len(cut)):
            if cut[i]:
                continue
            track_id[n] = self.df.track_id[i]
            hit_id[n] = self.get_hit_id()[i]
            state[n] = self.get_state()[i]
            state_cov[n] = self.get_state_cov()[i]
            sum_chi2[n] = self.sum_chi2[i]
            n += 1
        self.df.track_id.values = track_id[0:n]
        self.set_hit_id(hit_id[0:n,:])
        self.set_state(state[0:n,:])
        self.set_state_cov(state_cov[0:n,:,:])
        self.df.sum_chi2.values = sum_chi2[0:n]
    
        print('removed ', len(cut) - n , 'track candidates')

