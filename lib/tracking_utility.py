import math
import numpy as np
import detector as det

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

def get_1uDp(x,y):
    #assume x = array([x1, x2, x3]) and y = array([y1, y2, y3]). return (u, D, phi)
    if(len(x)!=3 or len(y)!=3):
        return (0,0,0)
    r2 = x*x + y*y
    r2_dif = [r2[1] - r2[2],r2[2] - r2[0], r2[0] - r2[1]]
    phi = math.atan(sum(y * r2_dif)/sum(x *r2_dif))
    if phi < 0:
        phi += math.pi

    rd = -r2_dif[2]/2.0/(math.sin(phi)*(x[0]-x[1])-math.cos(phi)*(y[0]-y[1]))
    if rd < 0:
        rd = -rd
        phi += math.pi

    rr = r2[0] + 2.0*rd*(math.sin(phi)*x[0]-math.cos(phi)*y[0])+rd**2
    u = 1.0/math.sqrt(rr)
    D = math.sqrt(rr) - rd

    ATan0 = math.atan((y[0]-rd*math.cos(phi))/(x[0]+rd*math.sin(phi))) 
    ATan1 = math.atan((y[1]-rd*math.cos(phi))/(x[1]+rd*math.sin(phi))) 
    if x[0] < -rd*math.sin(phi):
        ATan0 += math.pi
        if x[1] < -rd*math.sin(phi):
            ATan1 += math.pi
        elif ATan1 < 0:
            ATan1 += 2*math.pi
    elif ATan0 < 0:
        ATan0 += 2*math.pi
        if x[1] < -rd*math.sin(phi):
            ATan1 += math.pi
        else:
            ATan1 += 2*math.pi
    else:
        if x[1] < -rd*math.sin(phi):
            ATan1 += math.pi
        elif ATan1 < 0:
            ATan1 += 2*math.sin(phi)
            ATan0 += 2*math.sin(phi)

    q = 1
    if ATan1 > ATan0:
        q = -1
    
    #z is not used
    #ASin1 = calcurate_asin(math.arcsin((y[1] - rd*math.cos(phi))*u), q, x[1], -rd*math.sin(phi))
    #ASin0 = calcurate_asin(math.arcsin((y[0] - rd*math.cos(phi))*u), q, x[0], -rd*math.sin(phi))
    #eta = q * sqrt(rr) * (ASin1 - ASin0) / (z[1] - z[0])
    #nanika = z[0] - q * math.sqrt(rr) * (ASin0 - phi + math.pi/2)/eta
    
    u *= q
    D *= q
    if q > 0:
        if phi > math.pi:
            phi -= math.pi
        else:
            phi += math.pi
    #eta *= q

    return (u, D, phi)

def get_uDp(x):
    #assume x = array([[[x1,y1],[x2,y2],[x3,y3]],[],...])
    #return array([[u,v,w],[],...])
    if (x.shape[1] !=3) or (x.shape[2] != 2):
        return np.zeros(3)
    r2 = (x*x).sum(axis=2)
    r2_dif = np.array([r2[:,1] - r2[:,2],r2[:,2] - r2[:,0], r2[:,0] - r2[:,1]]).transpose()
    phi = np.arctan((x[:,:,1] * r2_dif).sum(axis=1)/(x[:,:,0] * r2_dif).sum(axis=1))
    phi[phi<0] += math.pi

    rd = -r2_dif[:,2]/2.0/(np.sin(phi)*(x[:,0,0]-x[:,1,0])-np.cos(phi)*(x[:,0,1]-x[:,1,1]))
    phi[rd<0] += math.pi
    rd[rd<0] *= -1

    rr = r2[:,0] + 2.0*rd*(np.sin(phi)*x[:,0,0]-np.cos(phi)*x[:,0,1])+rd**2
    u = 1.0/np.sqrt(rr)
    D = np.sqrt(rr) - rd

    ATan0 = np.arctan((x[:,0,1]-rd*np.cos(phi))/(x[:,0,0]+rd*np.sin(phi))) 
    ATan1 = np.arctan((x[:,1,1]-rd*np.cos(phi))/(x[:,1,0]+rd*np.sin(phi))) 

    con_lv0 = [x[:,0,0] < -rd*np.sin(phi), 
               ~(x[:,0,0] < -rd*np.sin(phi)) & (ATan0 < 0), 
               ~(x[:,0,0] < -rd*np.sin(phi)) & ~(ATan0 < 0)]
    con_lv1 = [x[:,1,0] < -rd*np.sin(phi),
               ~(x[:,1,0] < -rd*np.sin(phi)) & (ATan1 < 0),
               ~(x[:,1,0] < -rd*np.sin(phi))]
    ATan0[con_lv0[0]] += math.pi
    ATan1[con_lv0[0] & con_lv1[0]] += math.pi
    ATan1[con_lv0[0] & con_lv1[1]] += 2*math.pi
    ATan0[con_lv0[1]] += 2*math.pi
    ATan1[con_lv0[1] & con_lv1[0]] += math.pi
    ATan1[con_lv0[1] & con_lv1[2]] += 2*math.pi
    ATan1[con_lv0[2] & con_lv1[0]] += math.pi
    ATan1[con_lv0[2] & con_lv1[1]] += 2*math.pi
    ATan0[con_lv0[2] & con_lv1[1]] += 2*math.pi
    #if x[:,0,0] < -rd*np.sin(phi):
    #    ATan0 += math.pi
    #    if x[:,1,0] < -rd*np.sin(phi):
    #        ATan1 += math.pi
    #    elif ATan1 < 0:
    #        ATan1 += 2*math.pi
    #elif ATan0 < 0:
    #    ATan0 += 2*math.pi
    #    if x[:,1,0] < -rd*np.sin(phi):
    #        ATan1 += math.pi
    #    else:
    #        ATan1 += 2*math.pi
    #else:
    #    if x[:,1,0] < -rd*np.sin(phi):
    #        ATan1 += math.pi
    #    elif ATan1 < 0:
    #        ATan1 += 2*math.pi
    #        ATan0 += 2*math.pi

    q = np.ones(x.shape[0])
    q[ATan1>ATan0] = -1
    #if ATan1 > ATan0:
    #    q = -1
    
    #z is not used
    #ASin1 = calcurate_asin(math.arcsin((y[1] - rd*math.cos(phi))*u), q, x[1], -rd*math.sin(phi))
    #ASin0 = calcurate_asin(math.arcsin((y[0] - rd*math.cos(phi))*u), q, x[0], -rd*math.sin(phi))
    #eta = q * sqrt(rr) * (ASin1 - ASin0) / (z[1] - z[0])
    #nanika = z[0] - q * math.sqrt(rr) * (ASin0 - phi + math.pi/2)/eta
    
    u *= q
    D *= q

    condition = [(q>0) & (phi>math.pi), (q>0) & ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi
    #eta *= q

    return (u, D, phi)

def get_jacobian(state,module_trans):
    ndata = state.shape[0]
    ndim_state = state.shape[1]
    ndim_obsrv = 1
    
    cx = module_trans["cx"]
    cy = module_trans["cy"]
    cos_theta = module_trans["cos_theta"]
    sin_theta = module_trans["sin_theta"]
    #protect from 0 divide
    state[:,0][state[:,0]==0] = 1e-12

    #define parameters
    q = np.ones(ndata)
    q[state[:,0]<0] = -1
    phi = state[:,2]
    condition = [(q>0) & (phi>math.pi), (q>0) & ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    ax = -q * (1./state[:,0] - state[:,1])*sin_phi
    ay = q * (1./state[:,0] - state[:,1])*cos_phi
    root = 1./state[:,0]/state[:,0] - ((cx-ax)*sin_theta - (cy-ay)*cos_theta)**2
    root[root>0] = np.sqrt(root[root>0])
    root[root<0] = 1.e-9
    cos_phi[phi>math.pi] = np.cos(phi[phi>math.pi]-math.pi)
    sin_phi[phi>math.pi] = np.sin(phi[phi>math.pi]-math.pi)

    #calculate jacobian
    jacobian = np.zeros((ndata,ndim_obsrv,ndim_state))
    jacobian[:,0,0] = (sin_phi*cos_theta - cos_phi*sin_theta)/state[:,0]/state[:,0]
    tmp = (sin_phi*sin_theta - cos_phi*sin_theta)*((cx-ax)*sin_theta - (cy-ay)*cos_theta) - 1./state[:,0]
    tmp /= root/state[:,0]/state[:,0]
    jacobian[:,0,0][q<0] -= tmp[q<0]
    jacobian[:,0,0][q>0] += tmp[q>0]

    jacobian[:,0,1] = sin_phi*cos_theta - cos_phi*sin_theta
    tmp = (sin_phi*sin_theta + cos_phi*cos_theta)*((cx-ax)*sin_theta - (cy-ay)*cos_theta)/root 
    jacobian[:,0,1][q<0] -= tmp[q<0]
    jacobian[:,0,1][q>0] += tmp[q>0]

    jacobian[:,0,2] = - (1./state[:,0] - state[:,1])*(cos_phi*cos_theta + sin_phi*sin_theta)
    tmp = (1./state[:,0] - state[:,1])*(cos_phi*sin_theta - sin_phi*cos_theta)*((cx-ax)*sin_theta - (cy-ay)*cos_theta)/root 
    jacobian[:,0,2][q<0] += tmp[q<0]
    jacobian[:,0,2][q>0] -= tmp[q>0]

    #not use z info
    #jacobian[:,0,3] = 0
    #jacobian[:,0,4] = 0

    return jacobian

def get_predict_obsrv(state,module_trans):
    ndata = state.shape[0]
    ndim_state = state.shape[1]
    ndim_obsrv = 1
    
    cx = module_trans["cx"]
    cy = module_trans["cy"]
    cos_theta = module_trans["cos_theta"]
    sin_theta = module_trans["sin_theta"]
    #protect from 0 divide
    state[:,0][state[:,0]==0] = 1e-12
    #define parameters
    q = np.ones(ndata)
    q[state[:,0]<0] = -1
    phi = state[:,2]
    condition = [(q>0) & (phi>math.pi), (q>0) & ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    phi[condition[1]] += math.pi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    ax = -q * (1./state[:,0] - state[:,1])*sin_phi
    ay = q * (1./state[:,0] - state[:,1])*cos_phi
    #print("cos ",cos_phi, "sin ",sin_phi, "ax ",ax,"ay ",ay)
    root = 1./state[:,0]**2 - ((cx-ax)*sin_theta - (cy-ay)*cos_theta)**2
    root[root>0] = np.sqrt(root[root>0])
    root[root<0] = 1.e-9
    #cos_phi[phi>math.pi] = np.cos(phi[phi>math.pi]-math.pi)
    #sin_phi[phi>math.pi] = np.sin(phi[phi>math.pi]-math.pi)

    #calculate hit position in local
    #print("root ",root)
    predict_obsrv = np.zeros((ndata,ndim_obsrv))
    predict_obsrv[:,0] = -(cx-ax)*cos_theta - (cy-ay)*sin_theta
    #print("cos ",cos_theta, "sin ",sin_theta, "cx ",cx,"cy ",cy)
    #print("obsrv ", predict_obsrv[:,0])
    predict_obsrv[:,0][q<0] -= root[q<0]
    predict_obsrv[:,0][q>0] += root[q>0]
    #+-??

    return predict_obsrv

class Tracking:
#    def __int__(self):

    def get_track_id(self):
        return self.track_id

    def get_hit_id(self):
        return self.hit_id

    def get_state(self):
        return self.state

    def get_state_cov(self):
        return self.state_cov
  
    def get_sum_chi2(self):
        return self.sum_chi2

    def set_track_id(self,track_id):
        self.track_id = track_id
   
    def set_hit_id(self,hit_id):
        self.hit_id = hit_id
   
    def set_state(self,state):
        self.state = state

    def set_state_cov(self,state_cov):
        self.state_cov = state_cov

    def set_sum_chi2(self,sum_chi2):
        self.sum_chi2 = sum_chi2

    def add_track_id(self,track_id):
        np.append(self.track_id,track_id)

    def add_hit_id(self,hit_id):
        np.append(self.hit_id,hit_id)

    def add_state(self,state):
        np.append(self.state,state)

    def add_state_cov(self,state_cov):
        np.append(self.state_cov,state_cov)


    def add_sum_chi2(self,sum_chi2):
        np.append(self.sum_chi2,sum_chi2)

    def cut_by_chi2(self,limit):
        cut = self.sum_chi2 > limit
        track_id = np.zeros(self.track_id.shape[0])
        hit_id = np.zeros((self.hit_id.shape[0],self.hit_id.shape[1]))
        state = np.zeros((self.state.shape[0],self.state.shape[1]))
        state_cov = np.zeros((self.state_cov.shape[0],self.state_cov.shape[1],self.state_cov.shape[2]))
        sum_chi2 = np.zeros(self.sum_chi2.shape[0])
        n = 0
        for i in range(len(cut)):
            if cut[i]:
                continue
            track_id[n] = self.track_id[i]
            hit_id[n] = self.hit_id[i]
            state[n] = self.state[i]
            state_cov[n] = self.state_cov[i]
            sum_chi2[n] = self.sum_chi2[i]
            n += 1
        self.track_id = track_id[0:n]
        self.hit_id = hit_id[0:n,:]
        self.state = state[0:n,:]
        self.state_cov = state_cov[0:n,:,:]
        self.sum_chi2 = sum_chi2[0:n]
    
        print("removed ", len(cut) - n , "track candidates")

