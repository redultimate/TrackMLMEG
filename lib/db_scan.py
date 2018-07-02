"""
Simple clustering by DBScan.
Assuming linear track from origin.
Needs 'hit_theta', 'hit_phi', 'hit_z2'.

Possible Update:
# Move normalization factor to input parameter.
# Optimize normalizzation from true track.
"""

from sklearn import cluster

class Clusterer(object):
    #_____________________________________________________________
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        return
    #_____________________________________________________________
    def predict(self, hits):
       # Define DB scan
       dbscan = cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean', algorithm='auto', n_jobs=-1)

       # Normalize
       hits['hit_theta'] /= 25
       hits['hit_phi'] /= 2.5
       hits['hit_z2'] /= 1.

       # Predict
       hits['track_id'] = dbscan.fit_predict(hits[['hit_theta', 'hit_phi', 'hit_z2']].values)
       return hits