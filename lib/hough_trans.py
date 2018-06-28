# based on
# https://www.kaggle.com/mikhailhushchyn/hough-transform
# possible updates
## off-center tracks

import numpy as np
import pandas as pd

def transform_cartesian_to_cylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    z = z
    
    return r, phi, z

def create_hough_matrix(hits):
    hough_matrix = hits[['hit_id', 'x', 'y', 'z']].copy()
    hough_matrix.columns = ['HitID', 'X', 'Y', 'Z']

    r, phi, z = transform_cartesian_to_cylindrical(hough_matrix['X'].values,
                                     hough_matrix['Y'].values,
                                     hough_matrix['Z'].values)

    hough_matrix['R'] = r
    hough_matrix['Phi'] = phi
    
    return hough_matrix

def add_theta(hough_matrix, theta):
    hough_matrix['Theta'] = theta
    return hough_matrix

def add_r0_inv(hough_matrix):
    hough_matrix['R0Inv'] = (2. * np.cos(hough_matrix['Phi'] - hough_matrix['Theta']) / hough_matrix['R']).values
    return hough_matrix

def add_gamma(hough_matrix):
    hough_matrix['Gamma'] = hough_matrix['Z']/hough_matrix['R']
    return hough_matrix

def digitize_column(hough_matrix, col, N, min_val=None, max_val=None):
    x = hough_matrix[col].values
    if min_val is not None and max_val is not None:
        bins = np.linspace(min_val, max_val, N)
    else:
        bins = np.linspace(x.min(), x.max(), N)
    bin_ids = np.digitize(x, bins)
    hough_matrix[col+'Digi'] = bin_ids

    return hough_matrix

def combine_digi(hough_matrix, columns):    
    hough_matrix['ComboDigi'] = np.zeros(len(hough_matrix))
    
    for i_col, acol in enumerate(columns):
        digi = hough_matrix[acol]
        hough_matrix['ComboDigi'] += digi * 10**(i_col * 5)
    
    return hough_matrix

def count_combo_digi(hough_matrix):
    unique, indeces, counts = np.unique(hough_matrix['ComboDigi'].values,
                                     return_counts=True, return_inverse=True)
    hough_matrix['ComboDigiCounts'] = counts[indeces]

    return hough_matrix

def out_of_border_counters_to_zero(hough_matrix, col, N):
    hough_matrix['ComboDigiCounts'] *= (hough_matrix[col].values != 0) * (hough_matrix[col].values != N)
    return hough_matrix

def one_slice(hough_matrix, theta, N_bins_r0inv, N_bins_gamma, min_hits):
    tracks = []

    hough_matrix = add_theta(hough_matrix, theta)
    hough_matrix = add_r0_inv(hough_matrix)
    hough_matrix = add_gamma(hough_matrix)

    hough_matrix = digitize_column(hough_matrix, 'R0Inv', N_bins_r0inv, -0.02, 0.02) # Tune it.
    hough_matrix = digitize_column(hough_matrix, 'Gamma', N_bins_gamma, -50, 50) # Tune it.

    hough_matrix = combine_digi(hough_matrix, ['R0InvDigi', 'GammaDigi'])
    hough_matrix = count_combo_digi(hough_matrix)

    hough_matrix = out_of_border_counters_to_zero(hough_matrix, 'R0InvDigi', N_bins_r0inv)
    hough_matrix = out_of_border_counters_to_zero(hough_matrix, 'GammaDigi', N_bins_gamma)

    counts = hough_matrix.ComboDigiCounts.values
    bins = hough_matrix.ComboDigi.values
    hit_ids = np.arange(len(hough_matrix))
    for abin in np.unique(bins[counts >= min_hits]):
        atrack = hit_ids[(bins == abin)]
        tracks.append(atrack)

    return tracks, hough_matrix
