import sys

sys.path.append("../lib/")#path to libraly directory

import read_data as rd
import calc_var as cv
import calc_score as cs
import db_tracker as dt

import sys
import numpy as np
from trackml.score import score_event
# import matplotlib.pyplot as plt

PAUSE_EACH_EVENT = True
# PAUSE_EACH_EVENT = False

#_____________________________________________________________
def main():

    train, test, det = BeginOfRun()
    for event in train:
        Event(event)
    EndOfRun()
    return

#_____________________________________________________________
def BeginOfRun():
    #________List up input files________
    Input_PATH = '/home/ec2-user/pr/kaggle/trackml/'
    dh = rd.DataHandler()
    train, test, det = dh.get_event_list(Input_PATH)

    return train, test, det

#_____________________________________________________________
def Event(event):
   print('Event ID: %d'%int(event[-9:]))
   dh = rd.DataHandler()
   hits = dh.read_all(event)

   #________Calc valiables________
   hits = (cv.MeasCalculator()).calc_all(hits)
   hits = (cv.MCCalculator()).calc_all(hits)

   #________Sorrt & selection________
   print('Nhits : %d, total weight: %.3lf'%(len(hits), hits.sum()['weight']))
   hits = hits[~np.logical_and(np.logical_or(hits['volume_id'] == 7, hits['volume_id'] == 9), hits['hit_ncell'] >= 4)]
   print('Nhits : %d, total weight: %.3lf'%(len(hits), hits.sum()['weight']))

   hits = hits.sort_values(by=['track_weight', 'particle_id','hit_r'], ascending=False).reset_index(drop=True)

   #________Init track id________
   hits['track_id'] = -1
   hits['track_reconstructed'] = False

   #________Print________
   if (not len(hits)) :
      return;
   print('Nhits : %d'%len(hits))

   #________Def DBscan________
   dttr = dt.Tracker()

   hits = dttr.predict(hits, 0.06, 3, [7,8], 7)
   print('Score: %lf'%score_event(hits, hits))

   hits = dttr.predict(hits, 0.06, 3, [8,9], 9)
   print('Score: %lf'%score_event(hits, hits))

   hits = dttr.predict(hits, 0.06, 3, [12, 13, 7, 8], 12)
   print('Score: %lf'%score_event(hits, hits))

   hits = dttr.predict(hits, 0.06, 3, [14, 13, 9, 8], 14)
   print('Score: %lf'%score_event(hits, hits))

   hits = dttr.predict(hits, 0.06, 3, [16, 12, 13, 8], 16)
   print('Score: %lf'%score_event(hits, hits))

   hits = dttr.predict(hits, 0.06, 3, [18, 14, 13, 8], 18)
   print('Score: %lf'%score_event(hits, hits))

   hits = dttr.predict(hits, 0.06, 3, [17, 13, 8], 17)
   print('Score: %lf'%score_event(hits, hits))

   #________Check_Score________
   hits = (cs.ScoreCalculator).calc_score(hits)


   #________Print_Result________
   # print(hits.head(49)[['particle_id', 'volume_id', 'track_id', 'track_purity_rec', 'track_purity_maj', 'track_reconstructed']])
   print(hits['hit_score'].sum())
   print('Score: %lf'%score_event(hits, hits))

   Pause()

   return

#_____________________________________________________________
def EndOfRun():
    return

#_____________________________________________________________
def Pause():
        sys.stderr.write('[Read]\tstop.\tPress "q" to quit. else proceed. >')
        ans = input('> ')
        if ans in ['q', 'Q']:
            sys.exit(-1)
        else :
            return

#_____________________________________________________________
if __name__ == '__main__':
    main()