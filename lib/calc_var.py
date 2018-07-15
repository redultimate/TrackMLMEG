"""
Usage:
hits = (cv.MeasCalculator()).calc_all(hits)
hits = (cv.MCCalculator()).calc_all(hits)
"""

import numpy as np

class MeasCalculator:
   #_____________________________________________________________
   def __init__(self):
      return
   #_____________________________________________________________
   def calc_all(self, hits):
      hits = self.calc_cylindrical(hits)
      hits = self.calc_xyz2(hits)
      return hits
   #_____________________________________________________________
   def calc_cylindrical(self, hits):
      hits['hit_r'] = np.sqrt(hits['x'] **2+ hits['y']**2)
      hits['hit_theta'] = np.degrees(np.arctan2(hits['x'], hits['y']))
      hits['hit_phi'] = np.degrees(np.arctan2(hits['hit_r'], hits['z']))
      return hits
   #_____________________________________________________________
   def calc_xyz2(self, hits):
      hits['hit_x2'] = hits['x']/ np.sqrt(hits['x']**2 + hits['y']**2 + hits['z']**2)
      hits['hit_y2'] = hits['y']/ np.sqrt(hits['x']**2 + hits['y']**2 + hits['z']**2)
      hits['hit_z2'] = hits['z']/ np.sqrt(hits['x']**2 + hits['y']**2)
      return hits

class MCCalculator:
   #_____________________________________________________________
   def __init__(self):
      return
   #_____________________________________________________________
   def calc_all(self, hits):
      hits = self.calc_cylindrical(hits)
      hits = self.calc_momentum(hits)
      hits = self.calc_trackweight(hits)
      return hits
   #_____________________________________________________________
   def calc_cylindrical(self, hits):
      hits['track_vr'] = np.sqrt(hits['vx'] **2+ hits['vy']**2)
      hits['track_vtheta'] = np.degrees(np.arctan2(hits['vx'], hits['vy']))
      hits['track_vphi'] = np.degrees(np.arctan2(hits['track_vr'], hits['vz']))
      return hits

   #_____________________________________________________________
   def calc_momentum(self, hits):
      hits['hit_tpt'] = np.sqrt(hits['tpx']**2 + hits['tpy']**2)
      return hits

   #_____________________________________________________________
   def calc_trackweight(self, hits):
      trackweights = hits.groupby('particle_id').sum()['weight']
      hits['track_weight'] = trackweights[hits['particle_id']].reset_index()['weight']
      return hits