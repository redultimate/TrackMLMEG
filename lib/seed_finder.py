import numpy as np
import pandas as pd

# if debug = 1, only 2 hit combinations in each layer will be processed

mask_17_4 = ((hits['volume_id'].values == 17) & (hits['layer_id'].values == 4))
mask_17_2 = ((hits['volume_id'].values == 17) & (hits['layer_id'].values == 2))
mask_13_8 = ((hits['volume_id'].values == 13) & (hits['layer_id'].values == 8))
hits_17_4 = hits[mask_17_4]
hits_17_2 = hits[mask_17_2]
hits_13_8 = hits[mask_13_8]

class seed_finder(object):

    def __init__(self, thre, debug):
        
        self.thre = thre
        self.debug = debug

    def get_candidates(self, hits):

        if self.debug:
            nhits1st = 2
        else:
            nhits1st = len(hits_17_4)

        #index in the 1st layer, all the hits in the 1st layer
        for id1st in range(nhits1st):

            #module id in the 1st outer layer (volume17, layer4 in this case)
            mid1st = hits_17_4['module_id'].values[id1st]
            if self.debug:
                print(id1st, "th hit in the 1st layer")
                print("module ", mid1st, "in the 1st layer")
                print("(1st layer) [x, y] = ", hits_17_4.loc[:, ['x', 'y']].values[id1st])
                print("(1st layer) hitid = ", hits_17_4.loc[:, ['hit_id']].values[id1st])

            true_particle_id = hits_17_4['particle_id'].values[id1st]
            print("true particle_id: ", true_particle_id)
            #get module id in the 2nd layer which corresponds to the module in the 1st layer according to the look-up table
            #mid1st-1 because module id is start from 1 not 0
            mask2nd= hits_17_2['module_id'].isin(mid_lookup_17_4_to_17_2[mid1st-1])
            hits_17_2_selected = hits_17_2[mask2nd]

            if self.debug:
                nhits2nd= 2
            else:
                nhits2nd = len(hits_17_2_selected)
            #loop for 2nd layer (volume17, layer2 in this case)
            for id2nd in range(nhits2nd):
                mid2nd = hits_17_2_selected['module_id'].values[id2nd]
                if self.debug:
                    print("------------------------------")
                    print(id2nd, "th hit in the 2nd layer")
                    print("module ", mid2nd, "in the 2nd layer")
                    print(hits_17_2_selected.loc[:, ['x', 'y']].values[id2nd])
                    print(hits_17_2_selected.loc[:, ['hit_id']].values[id2nd])

                #get module id in the 3rd layer which corresponds to the module in the 2nd layer according to the look-up table
                # mid2nd-1 because module id is start from 1 not 0
                mask3rd = hits_13_8['module_id'].isin(mid_lookup_17_2_to_13_8[mid2nd-1])
                hits_13_8_selected = hits_13_8[mask3rd]

                #get (x,y) of the selected hits in 3rd layer
                aXYmodule3= np.array(hits_13_8_selected.loc[:, ['x', 'y']])
                aHitIDmodule3 = np.array(hits_13_8_selected.loc[:, ['hit_id']])
                #reshape to concat the array to 1st and 2nd layer
                aXYmodule3 = aXYmodule3.reshape(aXYmodule3.shape[0],1,2)
                aHitIDmodule3 = aHitIDmodule3.reshape(aHitIDmodule3.shape[0],1,1)

                if self.debug:
                    print(aXYmodule3.shape)
                    print(aHitIDmodule3.shape)

                #get (x,y) of the specific hit in the 1st and 2nd layer
                aXYmodule1_2 = np.array([hits_17_4.loc[:, ['x', 'y']].values[id1st],hits_17_2_selected.loc[:, ['x', 'y']].values[id2nd]])
                aHitIDmodule1_2 = np.array([hits_17_4.loc[:, ['hit_id']].values[id1st],hits_17_2_selected.loc[:, ['hit_id']].values[id2nd]])

                if self.debug:
                    print(aXYmodule1_2.shape)
                    print(aXYmodule1_2)
                    print(aHitIDmodule1_2.shape)
                    print(aHitIDmodule1_2)

                #tile to concat the array to 3rd layer
                aXYmodule1_2 = np.tile(aXYmodule1_2, [aXYmodule3.shape[0],1,1])
                aHitIDmodule1_2 = np.tile(aHitIDmodule1_2, [aHitIDmodule3.shape[0],1,1])

                if self.debug:
                    print(aXYmodule1_2.shape)
                    print(aHitIDmodule1_2.shape)

                #concat the arrays to calculate track parameters at the same time
                aXYmodule2_3 = np.r_['1', aXYmodule1_2, aXYmodule3]
                aHitIDmodule2_3 = np.r_['1', aHitIDmodule1_2, aHitIDmodule3]

                if self.debug:
                    print(aXYmodule2_3.shape)
                    print(aHitIDmodule2_3.shape)

                if id2nd == 0:
                    XYmodule2_3 = aXYmodule2_3
                    HitIDmodule2_3 = aHitIDmodule2_3
                else:
                    XYmodule2_3 =  np.concatenate((XYmodule2_3, aXYmodule2_3), axis=0)
                    HitIDmodule2_3 =  np.concatenate((HitIDmodule2_3, aHitIDmodule2_3), axis=0)

            print(XYmodule2_3.shape[0], " hit combinations are selected")
            print(HitIDmodule2_3.shape[0], " hit combinations are selected")
            #true_track = hits[hits['particle_id']==true_particle_id]
            #true_track
            #mask_17_4_true = ((true_track['volume_id'].values == 17) & (true_track['layer_id'].values == 4))
            #mask_17_2_true = ((true_track['volume_id'].values == 17) & (true_track['layer_id'].values == 2))
            #mask_13_8_true = ((true_track['volume_id'].values == 13) & (true_track['layer_id'].values == 8))
            #true_combination = np.array([[true_track[mask_17_4_true].loc[:, ['x', 'y']].values[0], true_track[mask_17_2_true].loc[:, ['x', 'y']].values[0], true_track[mask_13_8_true].loc[:, ['x', 'y']].values[0]]])
            #true_combination

            if id1st == 0:
                XYmodule1_2_3 = XYmodule2_3
                HitIDmodule1_2_3 = HitIDmodule2_3

            else:
                XYmodule1_2_3 = np.concatenate((XYmodule1_2_3, XYmodule2_3), axis=0)
                HitIDmodule1_2_3 = np.concatenate((HitIDmodule1_2_3, HitIDmodule2_3), axis=0)

        u, D, p = get_uDp(XYmodule1_2_3)
        mask_cut = abs(D) < self.thre
        track_parameters = np.r_['1, 2, 0', u[mask_cut], D[mask_cut]]
        track_parameters = np.r_['1, 2, 0', track_parameters, p[mask_cut]]
        print(track_parameters.shape[0], " hit combinations pass D cut")
        print(HitIDmodule1_2_3[mask_cut].shape[0], " hit combinations pass D cut")

        print(track_parameters.shape)
        print(track_parameters)

        HitIDmodule1_2_3 = HitIDmodule1_2_3.reshape(HitIDmodule1_2_3.shape[0], 3)
        print(HitIDmodule1_2_3[mask_cut].shape)
        print(HitIDmodule1_2_3[mask_cut])

        return HitIDmodule1_2_3[mask_cut], track_parameters
