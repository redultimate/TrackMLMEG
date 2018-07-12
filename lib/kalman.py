import numpy as np
import math
import detector as det
import tracking_utility as tr_ut

class KalmanFilter:
#    def __init__(self):
    
    def predict(self,tracks,obsrv,module_trans):
        ndata = tracks.get_state().shape[0]
        ndim = tracks.get_state().shape[1]
        state_trans = np.matlib.repmat(np.identity(ndim),ndata,1).reshape(ndata,ndim,ndim)
        state_trans_control= np.zeros((ndata,ndim))
        self.predict_state = np.zeros((ndata,ndim)) 
        self.predict_cov = np.zeros((ndata,ndim,ndim)) 
        PA = np.zeros((ndata,ndim,ndim)) 
        for i in range(ndim):
            #state prediction
            self.predict_state[:,i] = np.sum(state_trans[:,i] * tracks.get_state(), axis=1) + state_trans_control[:,i]
            for j in range(ndim):
                PA[:,i,j] = np.sum(tracks.get_state_cov()[:,i] * state_trans[:,j], axis=1)

        for i in range(ndim):
            for j in range(ndim):
                self.predict_cov[:,i,j] = np.sum(state_trans[:,i] * PA[:,:,j],axis=1)

        self.predict_cov += tracks.get_state_cov()
        self.predict_obsrv = tr_ut.get_predict_obsrv(self.predict_state,module_trans)[:,0]
        #print(obsrv, " - ", self.predict_obsrv)
        chi = obsrv - self.predict_obsrv

        return chi
        
    def filter(self,tracks,obsrv,obsrv_cov,module_trans):
        ndata = tracks.get_state().shape[0]
        ndim_state = tracks.get_state().shape[1]
        ndim_obsrv = 1

        q = np.ones(ndata)
        q[tracks.get_state()[:,0]<0] = -1
        phi = tracks.get_state()[:,2]
        tracks.get_state()[:,2][(q<0) & (phi<math.pi)] += math.pi
        tracks.get_state()[:,2][(q<0) & ~(phi<math.pi)] -= math.pi
        
        G = np.zeros((ndata,ndim_state,ndim_obsrv))#kalman gain
        #C = np.zeros((ndata,ndim_obsrv,ndim_state))
        C = tr_ut.get_jacobian(self.predict_state,module_trans)
        if C.shape[1] != ndim_obsrv:
            print("dimension of obsrvation is expected to be ", C.shape[1])
            return
        PC = np.zeros((ndata,ndim_state,ndim_obsrv))
        CPC = np.zeros((ndata,ndim_obsrv,ndim_obsrv))
        GC = np.zeros((ndata,ndim_state,ndim_state))
        GCP = np.zeros((ndata,ndim_state,ndim_state))
        for i in range(ndim_state):
            for j in range(ndim_obsrv):
                PC[:,i,j] = np.sum(self.predict_cov[:,i] * C[:,j],axis=1)

        for i in range(ndim_obsrv):
            for j in range(ndim_obsrv):
                CPC[:,i,j] = np.sum(C[:,i] * PC[:,:,j],axis=1)
        CPC[:,0,0] += obsrv_cov[:,0] 
        #only for ndim_obsrv=1
        
        for i in range(ndim_state):
            for j in range(ndim_obsrv):
                G[:,i,j] = np.sum(PC[:,i] * np.linalg.inv(CPC)[:,:,j],axis=1)

        for i in range(ndim_state):
            for j in range(ndim_state):
                GC[:,i,j] = np.sum(G[:,i] * C[:,:,j],axis=1)
        
        for i in range(ndim_state):
            for j in range(ndim_state):
                GCP[:,i,j] = np.sum(G[:,i] * self.predict_cov[:,:,j],axis=1)  
        tracks.set_state_cov(self.predict_cov - GCP)
        #set lower limit for sigma**2
        for i in range(ndim_state):
            tracks.get_state_cov()[:,i,i][tracks.get_state_cov()[:,i,i]<1.e-20] = 1.e-20

        #copy state vector
        tracks.set_state(self.predict_state)
        tracks.get_state()[:,2] = phi
        tracks.get_state()[:,2][tracks.get_state()[:,2]>math.pi] -= math.pi
        
        chi = obsrv - self.predict_obsrv
        #print("obsrv : " , obsrv)
        #print("predict : ", self.predict_obsrv)
        #print("chi ", chi)
        #print("before " , tracks.get_state())
        for i in range(ndim_state):
            #print(G[:,i,0])
            #print(G[:,i,0]*chi)
            tracks.get_state()[:,i] += G[:,i,0] * chi
            #assume only u observation
        #print("after " , tracks.get_state())

        q = np.ones(ndata)
        q[tracks.get_state()[:,0]<0] = -1
        tracks.get_state()[:,2][phi>math.pi] += math.pi
        phi = tracks.get_state()[:,2]
        tracks.get_state()[:,2][(q<0) & (phi<math.pi)] += math.pi
        tracks.get_state()[:,2][(q<0) & ~(phi<math.pi)] -= math.pi

        self.predict_obsrv = tr_ut.get_predict_obsrv(self.predict_state,module_trans)[:,0]
        chi = obsrv - self.predict_obsrv

        tracks.set_sum_chi2(tracks.get_sum_chi2() + chi*chi)

        return chi 

