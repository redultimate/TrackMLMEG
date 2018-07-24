"""
Estimate track by DB scan.
predict(self, hits, eps, min_samples, volume_ids, outer_volume):
hits: hits array
eps, min_samples : Parameter for DB scan
volume_ids : volume_id array to be used
Outer_volume : Most-outer Volume_id. Reconstructed track which contains hits in this volume is treated as correct track.

Output:
hits array. Tracking result from this step is combined with result of previous step.
"""
import sys
sys.path.append("../lib/")

# meg library
import db_scan as ds
import numpy as np
import pandas as pd

class Tracker(object):
    #_____________________________________________________________
    def __init__(self):
        return
    #_____________________________________________________________
    def predict(self, hits, eps, min_samples, volume_ids, outer_volume):
        #________Def first trarck ID________
        first_track_id = hits['track_id'].max() + 1

        #________Prepare hit list________
        ## Hits "in given volum" && "not already reconstructed" is used
        hits_this = pd.DataFrame()
        for volume_id in volume_ids:
            hits_this = pd.concat([hits_this, hits[hits['volume_id'] == volume_id]])
        hits_this = hits_this[hits_this['track_reconstructed'] == False]

        #________Perform DBscan________
        dscl = ds.Clusterer(eps, min_samples)
        hits_this = dscl.predict(hits_this)

        #________Select track to be used________
        ## Track which contains hits in volume 7 is used
        aa = hits_this.groupby('track_id')
        tracks = aa.apply(lambda x: np.isin(outer_volume, x['volume_id'])).reset_index()
        tracks.columns = ['track_id', 'ContainsVol']
        tracks['track_reconstructed'] = np.logical_and(tracks['ContainsVol'], tracks['track_id'] >= 0)

        hits_this = pd.merge(hits_this, tracks[['track_id', 'track_reconstructed']], how='left', on='track_id')
        hits_this = hits_this.drop('track_reconstructed_x', axis=1)
        hits_this = hits_this.rename(columns={'track_reconstructed_y': 'track_reconstructed'})
        hits_this.loc[hits_this['track_id'] >= 0, 'track_id'] += first_track_id

        #________Merge result________
        hits = self.merge_result(hits, hits_this)

        #________Print result________
        print('________________________')
        print('Volume: ', volume_ids)
        print('Reco Ntrack: ', len(tracks[tracks['track_reconstructed']==True]))
        print('Nhit,       total,       clustered,    clustered(%)')
        for volume_id in volume_ids:
            aaa = hits_this[hits_this['volume_id'] == volume_id]
            print('vol  %d     : %d,        %d,       %.1lf'%(volume_id, len(aaa), len(aaa[aaa['track_reconstructed'] == True]), len(aaa[aaa['track_reconstructed'] == True]) / len(aaa) * 100))
        aaa = hits_this
        print('Sum        : %d,        %d,      %.1lf'%(len(aaa), len(aaa[aaa['track_reconstructed'] == True]), len(aaa[aaa['track_reconstructed'] == True]) / len(aaa) * 100))
        print('')
        aaa = hits
        print('All vol    : %d,        %d,      %.1lf'%(len(aaa), len(aaa[aaa['track_reconstructed'] == True]), len(aaa[aaa['track_reconstructed'] == True]) / len(aaa) * 100))
        return hits

    #_____________________________________________________________
    def merge_result(self, hits_x, hits_y):
        #________Merge result________
        ## Combine results from different steps.
        hits_x = pd.merge(hits_x, hits_y[['hit_id', 'track_id', 'track_reconstructed']], how='left', on='hit_id')
        hits_x['track_id'] = -1
        hits_x.loc[hits_x['track_reconstructed_x']==True, 'track_id'] = hits_x.loc[hits_x['track_reconstructed_x']==True, 'track_id_x']
        hits_x.loc[hits_x['track_reconstructed_y']==True, 'track_id'] = hits_x.loc[hits_x['track_reconstructed_y']==True, 'track_id_y']
        hits_x['track_reconstructed'] = (hits_x['track_id'] >= 0)
        hits_x = hits_x.drop(['track_id_x', 'track_id_y', 'track_reconstructed_x', 'track_reconstructed_y'], axis=1)
        return hits_x