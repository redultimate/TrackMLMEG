# Input data Handeler
# Usage:
"""
Input data handler
Usage:
dh = rd.DataHandler()
train, test, det = dh.get_event_list(your_file_path)
for event in train:
   hits = dh.read_all(event)
   # To read only meas
   # hits = dh.read_meas(event)
"""

import pandas as pd
import numpy as np
from trackml.dataset import load_event
import glob, os

class DataHandler:
   #_____________________________________________________________
   def __init__(self):
      return
   #_____________________________________________________________
   def get_event_list(self, filepath):
    # List up input files
    os.chdir(filepath)

    train = np.unique([p.split('-')[0] for p in sorted(glob.glob('train_100_events/**'))])
    test = np.unique([p.split('-')[0] for p in sorted(glob.glob('test/**'))])

    det = pd.read_csv('detectors.csv')
    print('Nevent: train %d, test %d'%(len(train), len(test)))
    # sub = pd.read_csv('sample_submission.csv')
    # print(len(train), len(test), len(det), len(sub))
    return train, test, det

   #_____________________________________________________________
   def read_all(self, event):
      # Read both meas and MC files.
      hits = self.read_meas(event)
      mchits = self.read_mc(event)
      hits = pd.merge(hits, mchits, how='left', on='hit_id')
      return hits

   #_____________________________________________________________
   def read_meas(self, event):
      # Read meas files.
      hits, cells = load_event(event, parts=['hits', 'cells'])

      cells = cells.groupby(by=['hit_id'])['value'].agg(['count', 'sum']).reset_index()
      cells.columns = ['hit_id', 'hit_ncell', 'hit_edep']
      hits = pd.merge(hits, cells, how='left', on='hit_id')

      return hits

   #_____________________________________________________________
   def read_mc(self, event):
      # Read MC files.
      mchits, mctracks = load_event(event, parts=['truth', 'particles'])
      print(len(mctracks))

      # Fill dummy data for non-particle hits.
      mctracks =pd.concat([mctracks, pd.DataFrame([[0,0,0,0,0,0,0,0,0]], columns=mctracks.columns)])

      mchits = pd.merge(mchits, mctracks, how='left', on='particle_id')
      return mchits