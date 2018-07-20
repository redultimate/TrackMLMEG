import numpy as np
import pandas as pd
import pickle as pkl
import time

import sys
sys.path.append("../lib/")

# meg library
from tracking_utility import *

# see how_to_get_seed_candidates
# if debug = 1, only 2 hit combinations in each layer will be processed

# cython
cimport numpy as np

cdef int nhits1st, id1st, id1st0, nhits2nd, id2nd, debug, mid1st, mid2nd, true_particle_id
cdef double thre
cdef double t1, t2, t3, t4, t5
cdef list mid_lookup_17_4_to_17_2
cdef list mid_lookup_17_2_to_13_8
#cdef char* path_to_tables

class SeedFinder(object):

    def __init__(self, double thre, int debug):
        
        self.thre = thre
        self.debug = debug

    def get_candidates(self, np.ndarray[double, ndim=2] hits, path_to_tables):
        cdef np.ndarray[double, ndim=1] u, D, p
        #cdef np.ndarray[double, ndim=2] hits
        cdef np.ndarray[double, ndim=2] hits_17_4, hits_17_2, hits_13_8, hits_17_4_selected, hits_17_2_selected, hits_13_8_selected
        cdef np.ndarray[np.uint8_t, ndim=1, cast=True] mask_17_4, mask_17_2, mask_13_8, mask2nd, mask3rd, mask_cut
        cdef np.ndarray[double, ndim=2] aXYmodule3o, aXTmodule1_2o
        cdef np.ndarray[double, ndim=1] aHitIDmodule3o 
        cdef np.ndarray[double, ndim=2] aHitIDmodule1_2o 
        cdef np.ndarray[double, ndim=2] HitIDmodule1_2_3n 
        cdef np.ndarray[double, ndim=3] aXYmodule3, aXYmodule1_2, aXYmodule2_3, XYmodule2_3, XYmodule1_2_3
        cdef np.ndarray[double, ndim=3] aHitIDmodule3, aHitIDmodule1_2, aHitIDmodule2_3, HitIDmodule2_3, HitIDmodule1_2_3
        cdef np.ndarray[double, ndim=2] track_parameter, track_parameters
        t1 = time.time()
        #hits = hits_df.loc[:, ['hit_id','x','y', 'volume_id', 'layer_id', 'module_id', 'particle_id', 'weight']].values 
        mask_17_4 = ((hits[:, 3] == 17) & (hits[:, 4] == 4))
        mask_17_2 = ((hits[:, 3] == 17) & (hits[:, 4] == 2))
        mask_13_8 = ((hits[:, 3] == 13) & (hits[:, 4] == 8))
        hits_17_4 = hits[mask_17_4]
        hits_17_2 = hits[mask_17_2]
        hits_13_8 = hits[mask_13_8]
        
        # read lookup tables
        f = open("%s/mid_lookup_17_4_to_17_2.pkl" % path_to_tables,"rb")
        mid_lookup_17_4_to_17_2 = pkl.load(f)
        f = open("%s/mid_lookup_17_2_to_13_8.pkl" % path_to_tables,"rb")
        mid_lookup_17_2_to_13_8 = pkl.load(f)

        #if self.debug:
        #nhits1st = self.debug
        #else:
        nhits1st = len(hits_17_4)

        # index in the 1st layer, all the hits in the 1st layer
        for id1st in range(nhits1st):
        #for id1st0 in range(nhits1st):
            #id1st = int(2600+id1st0)
            print(id1st)
            # module id in the 1st outer layer (volume17, layer4 in this case)
            mid1st = int(hits_17_4[id1st, 5])
            true_particle_id = hits_17_4[id1st, 6]
            #if self.debug:
            #    print(id1st, "th hit in the 1st layer")
            #    print("module ", mid1st, "in the 1st layer")
            #    print("(1st layer) [x, y] = ", hits_17_4.loc[:, ['x', 'y']].values[id1st])
            #    print("(1st layer) hitid = ", hits_17_4.loc[:, ['hit_id']].values[id1st])
            #print("true particle_id: ", true_particle_id)
            # get module id in the 2nd layer which corresponds to the module in the 1st layer according to the look-up table
            # mid1st-1 because module id is start from 1 not 0
            mask2nd = np.isin(hits_17_2[:, 5], mid_lookup_17_4_to_17_2[mid1st-1])
            hits_17_2_selected = hits_17_2[mask2nd]

            #if self.debug:
            #nhits2nd = self.debug
            #else:
            nhits2nd = len(hits_17_2_selected)
            # loop for 2nd layer (volume17, layer2 in this case)
            for id2nd in range(nhits2nd):
                mid2nd = int(hits_17_2_selected[id2nd, 5])

                #if self.debug:
                #    print("------------------------------")
                #    print(id2nd, "th hit in the 2nd layer")
                #print("module ", mid2nd, "in the 2nd layer")
                #    print(hits_17_2_selected.loc[:, ['x', 'y']].values[id2nd])
                #    print(hits_17_2_selected.loc[:, ['hit_id']].values[id2nd])

                # get module id in the 3rd layer which corresponds to the module in the 2nd layer according to the look-up table
                # mid2nd-1 because module id is start from 1 not 0
                mask3rd = np.isin(hits_13_8[:, 5], mid_lookup_17_2_to_13_8[mid2nd-1])
                hits_13_8_selected = hits_13_8[mask3rd]
                # get (x,y) of the selected hits in 3rd layer
                aXYmodule3o = hits_13_8_selected[:, [1, 2]]
                aHitIDmodule3o = hits_13_8_selected[:, 0]
                # reshape to concat the array to 1st and 2nd layer
                aXYmodule3 = aXYmodule3o.reshape(aXYmodule3o.shape[0],1,2)
                aHitIDmodule3 = aHitIDmodule3o.reshape(aHitIDmodule3o.shape[0],1,1)

                #if self.debug:
                #    print(aXYmodule3.shape)
                #    print(aHitIDmodule3.shape)

                # get (x,y) of the specific hit in the 1st and 2nd layer
                aXYmodule1_2o = np.array([hits_17_4[id1st, [1, 2]], hits_17_2_selected[id2nd, [1, 2]]])
                aHitIDmodule1_2o = np.array([hits_17_4[id1st, [0]], hits_17_2_selected[id2nd, [0]]])

                #if self.debug:
                #    print(aXYmodule1_2.shape)
                #    print(aXYmodule1_2)
                #    print(aHitIDmodule1_2.shape)
                #    print(aHitIDmodule1_2)

                # tile to concat the array to 3rd layer
                aXYmodule1_2 = np.tile(aXYmodule1_2o, [aXYmodule3.shape[0],1,1])
                aHitIDmodule1_2 = np.tile(aHitIDmodule1_2o, [aHitIDmodule3.shape[0],1,1])

                #if self.debug:
                #    print(aXYmodule1_2.shape)
                #    print(aHitIDmodule1_2.shape)

                # concat the arrays to calculate track parameters at the same time

                aXYmodule2_3 = np.r_['1', aXYmodule1_2, aXYmodule3]
                aHitIDmodule2_3 = np.r_['1', aHitIDmodule1_2, aHitIDmodule3]

                #if self.debug:

                if id2nd == 0:
                    XYmodule2_3 = aXYmodule2_3
                    HitIDmodule2_3 = aHitIDmodule2_3
                else:
                    XYmodule2_3 =  np.concatenate((XYmodule2_3, aXYmodule2_3), axis=0)
                    HitIDmodule2_3 =  np.concatenate((HitIDmodule2_3, aHitIDmodule2_3), axis=0)

            #if self.debug:
            #    print(XYmodule2_3.shape[0], " hit combinations are selected")
            #    print(HitIDmodule2_3.shape[0], " hit combinations are selected")
            #true_track = hits[hits['particle_id']==true_particle_id]
            #true_track
            #mask_17_4_true = ((true_track['volume_id'].values == 17) & (true_track['layer_id'].values == 4))
            #mask_17_2_true = ((true_track['volume_id'].values == 17) & (true_track['layer_id'].values == 2))
            #mask_13_8_true = ((true_track['volume_id'].values == 13) & (true_track['layer_id'].values == 8))
            #true_combination = np.array([[true_track[mask_17_4_true].loc[:, ['x', 'y']].values[0], true_track[mask_17_2_true].loc[:, ['x', 'y']].values[0], true_track[mask_13_8_true].loc[:, ['x', 'y']].values[0]]])
            #true_combination

            u, D, p = get_uDp(XYmodule2_3)
            mask_cut = abs(D) < self.thre
            
            track_parameter = np.r_['1, 2, 0', u[mask_cut], D[mask_cut]]
            track_parameter = np.r_['1, 2, 0', track_parameter, p[mask_cut]]
            
            
            if id1st == 0:
                HitIDmodule1_2_3 = HitIDmodule2_3[mask_cut]
                track_parameters = track_parameter

            else:
                HitIDmodule1_2_3 = np.concatenate((HitIDmodule1_2_3, HitIDmodule2_3[mask_cut]), axis=0)
                track_parameters = np.concatenate((track_parameters, track_parameter), axis=0)
            
            """
            if id1st == 2600:
                XYmodule1_2_3 = XYmodule2_3
                HitIDmodule1_2_3 = HitIDmodule2_3

            else:
                XYmodule1_2_3 = np.concatenate((XYmodule1_2_3, XYmodule2_3), axis=0)
                HitIDmodule1_2_3 = np.concatenate((HitIDmodule1_2_3, HitIDmodule2_3), axis=0)
            """

        """
        t2 = time.time()
        u, D, p = get_uDp(XYmodule1_2_3)
        t3 = time.time()
        mask_cut = abs(D) < self.thre
        t4 = time.time()
        track_parameter = np.r_['1, 2, 0', u[mask_cut], D[mask_cut]]
        track_parameters = np.r_['1, 2, 0', track_parameter, p[mask_cut]]
        """
        #print(track_parameters.shape[0], " hit combinations pass D cut")
        #print(HitIDmodule1_2_3[mask_cut].shape[0], " hit combinations pass D cut")

        #print(track_parameters.shape[0], track_parameters.shape[1])
        #print(track_parameters)
        HitIDmodule1_2_3n = HitIDmodule1_2_3.reshape(HitIDmodule1_2_3.shape[0], 3)
        #print(HitIDmodule1_2_3n[mask_cut].shape)
        #print(HitIDmodule1_2_3n[mask_cut])
        t5 = time.time()
        #print(t1-t1, t2-t1, t3-t1, t4-t1, t5-t1)
        print(t1-t1, t5-t1)
        #return HitIDmodule1_2_3n[mask_cut], track_parameters
        
        print(track_parameters.shape[0], " hit combinations pass D cut")
        print(HitIDmodule1_2_3n.shape[0], " hit combinations pass D cut")
        return HitIDmodule1_2_3n, track_parameters
    
