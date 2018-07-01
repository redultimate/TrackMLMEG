#Methods to calc and record score for each hit, track.
#Bassed on trackml.score

import numpy as np
import pandas as pd
from trackml.score import _analyze_tracks

class ScoreCalculator:
   #_____________________________________________________________
   def __init__(self):
      return
   #_____________________________________________________________
   def calc_score(hits):
      # Check rec w/ truth and fill score for each tracks
      tracks = _analyze_tracks(hits, hits)
      tracks['track_purity_rec'] = np.true_divide(tracks['major_nhits'], tracks['nhits'])
      tracks['track_purity_maj'] = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
      tracks['track_good_track'] = np.logical_and(tracks['track_purity_rec'] > 0.5, tracks['track_purity_maj'] > 0.5)
      # print(tracks.columns)

      # Dummy data for track_id -1.
      tracks =pd.concat([tracks, pd.DataFrame([[-1,0,0,0,0,0,0,0,False]], columns=tracks.columns)])
      hits = pd.merge(hits, tracks[['track_id', 'track_purity_maj', 'track_purity_rec', 'track_good_track', 'major_particle_id']], how='left', on='track_id')

      # Fill score for each hit
      hits['hit_rec_ok'] = (hits['major_particle_id'] == hits['particle_id'])
      hits['hit_score'] = hits['weight'] * hits['hit_rec_ok'] * hits['track_good_track']
      hits['hit_score_loss'] = hits['weight'] - hits['hit_score']

      return hits