import numpy as np
import math
import detector as det
import tracking_utility as tr_ut
import pandas as pd

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
        chi = obsrv - self.predict_obsrv

        #print('predict: ',self.predict_obsrv, chi)
        if 1:
            tmp_predict_state = np.zeros((ndata,ndim)) 
            invert_state = tr_ut.invert_state(tracks.get_state())
            for i in range(ndim):
                #state prediction
                tmp_predict_state[:,i] = np.sum(state_trans[:,i] * invert_state, axis=1) + state_trans_control[:,i]
            tmp_predict_obsrv = tr_ut.get_predict_obsrv(tmp_predict_state,module_trans)[:,0]
            invert = chi**2 > (obsrv - tmp_predict_obsrv)**2
            self.predict_obsrv[invert] = tmp_predict_obsrv[invert]
            self.predict_state[invert] = tmp_predict_state[invert]
            chi = obsrv - self.predict_obsrv

            #rotate_state = tr_ut.rotate_state(tracks.get_state())
            #for i in range(ndim):
                #state prediction
            #    tmp_predict_state[:,i] = np.sum(state_trans[:,i] * rotate_state, axis=1) + state_trans_control[:,i]
            #tmp_predict_obsrv = tr_ut.get_predict_obsrv(tmp_predict_state,module_trans)[:,0]
            #rotate = chi**2 > (obsrv - tmp_predict_obsrv)**2
            #self.predict_obsrv[rotate] = tmp_predict_obsrv[rotate]
            #self.predict_state[rotate] = tmp_predict_state[rotate]

        tracks.set_state(self.predict_state)
        chi = obsrv - self.predict_obsrv
        #print('obsrv: ',obsrv)
        #print('predict: ',self.predict_obsrv, chi)
        tracks.set_sum_chi2(tracks.df.sum_chi2.values + chi*chi)

        return chi
        
    def filter(self,tracks,obsrv,obsrv_cov,module_trans):
        ndata = tracks.get_state().shape[0]
        ndim_state = tracks.get_state().shape[1]
        ndim_obsrv = 1

        q = np.ones(ndata)
        q[tracks.get_state()[:,0]<0] = -1
        phi = tracks.get_state()[:,2]
        condition = [(q<0) & (phi<math.pi), (q<0) & ~(phi<math.pi)]
        tracks.get_state()[:,2][condition[0]] += math.pi
        tracks.get_state()[:,2][condition[1]] -= math.pi
        
        G = np.zeros((ndata,ndim_state,ndim_obsrv))#kalman gain
        #C = np.zeros((ndata,ndim_obsrv,ndim_state))
        #C = tr_ut.get_jacobian_numerically(tracks.get_state(),module_trans)
        #print("jacobian ", C)
        C = tr_ut.get_jacobian(tracks.get_state(),module_trans)
        if C.shape[1] != ndim_obsrv:
            print('dimension of obsrvation is expected to be ', C.shape[1])
            return
        #print("jacobian ", C)
        PC = np.zeros((ndata,ndim_state,ndim_obsrv))
        CPC = np.zeros((ndata,ndim_obsrv,ndim_obsrv))
        GC = np.zeros((ndata,ndim_state,ndim_state))
        GCP = np.zeros((ndata,ndim_state,ndim_state))
        for i in range(ndim_state):
            for j in range(ndim_obsrv):
                PC[:,i,j] = np.sum(self.predict_cov[:,i] * C[:,j],axis=1)
        #print("PC : ",PC)

        for i in range(ndim_obsrv):
            for j in range(ndim_obsrv):
                CPC[:,i,j] = np.sum(C[:,i] * PC[:,:,j],axis=1)
        CPC[:,0,0] += obsrv_cov[:,0] 
        #only for ndim_obsrv=1
        #print("CPC : ",CPC)
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
        #tracks.set_state(self.predict_state)
        #tracks.get_state()[:,2] = phi
        #tracks.get_state()[:,2][tracks.get_state()[:,2]>math.pi] -= math.pi
        
        chi = obsrv - self.predict_obsrv
        #print('obsrv : ' , obsrv)
        #print('predict : ', self.predict_obsrv)
        #print('chi ', chi)
        #print('before ' , tracks.get_state())
        state_tmp = tracks.get_state()
        for i in range(ndim_state):
            #print(G[:,i,0]*chi)
            state_tmp[:,i] += G[:,i,0] * chi
            #assume only u observation
        tracks.set_state(state_tmp)
        #print('after ' , tracks.get_state())

        q = np.ones(ndata)
        q[tracks.get_state()[:,0]<0] = -1
        tracks.get_state()[:,2][phi>math.pi] += math.pi
        phi = tracks.get_state()[:,2]
        condition = [(q<0) & (phi<math.pi), (q<0) & ~(phi<math.pi)]
        tracks.get_state()[:,2][condition[0]] += math.pi
        tracks.get_state()[:,2][condition[1]] -= math.pi

        self.predict_obsrv = tr_ut.get_predict_obsrv(self.predict_state,module_trans)[:,0]
        chi = obsrv - self.predict_obsrv

        if 0:
            tmp_predict_state = np.zeros((ndata,ndim_state)) 
            tmp_predict_cov = np.zeros((ndata,ndim_state,ndim_state)) 
            invert_state = tr_ut.invert_state(tracks.get_state())
            tmp_predict_obsrv = tr_ut.get_predict_obsrv(invert_state,module_trans)[:,0]
            invert = chi**2 > (obsrv - tmp_predict_obsrv)**2
            self.predict_obsrv[invert] = tmp_predict_obsrv[invert]

        tracks.set_sum_chi2(tracks.df.sum_chi2.values + chi*chi)

        return chi 

    def clear(self):
        del self.predict_state
        del self.predict_cov
        del self.predict_obsrv
