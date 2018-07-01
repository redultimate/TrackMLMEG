# based on
# https://www.kaggle.com/mikhailhushchyn/hough-transform
# possible updates
## off-center tracks

import numpy as np
import pandas as pd

import sys
sys.path.append("../lib/")

from hough_trans_utility import *

class Clusterer(object):

    def __init__(self, N_bins_r0inv, N_bins_gamma, N_theta, min_hits):

        self.N_bins_r0inv = N_bins_r0inv
        self.N_bins_gamma = N_bins_gamma
        self.N_theta = N_theta
        self.min_hits = min_hits

    def predict(self, hits):

        tracks = []

        hough_matrix = create_hough_matrix(hits)
        for theta in np.linspace(-np.pi, np.pi, self.N_theta):
            slice_tracks, hough_matrix = one_slice(hough_matrix, theta, self.N_bins_r0inv, self.N_bins_gamma, self.min_hits)
            tracks += list(slice_tracks)

        labels = np.zeros(len(hits))
        used = np.zeros(len(hits))
        track_id = 0
        for atrack in tracks:
            u_track = atrack[used[atrack] == 0]
            if len(u_track) >= self.min_hits:
                labels[u_track] = track_id
                used[u_track] = 1
                track_id += 1

        return labels

