import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class CellCalculator:
    def __init__(self,cells):
        self.cells = cells

    def get_hit_edep(self):
        edeps = []
        for i in range(len(self.cells.hit_id.unique())):
            edep = self.cells.value[self.cells.hit_id==i]
            edep_sum = sum(edep)
            edeps = np.append(edeps,edep_sum)

        return edeps

class ParticleCalculator:
    def __init__(self,particles):
        self.particles = particles

    def get_momentum(self):
        px = self.particles.px.values
        py = self.particles.py.values
        pz = self.particles.pz.values
        p = np.sqrt(px*px + py*py + pz*pz)
        self.particles["p"] = p

        return p	

class TruthCalculator:
    def __init__(self,truth):
        self.truth = truth

    def get_particle_weight(self):
        pw = [0 for i in range(len(self.truth))]
        indices = self.truth.particle_id.unique()
        for i in tqdm(range(len(indices))):
            pid = indices[i]
            if pid == 0:
                continue
            HitOfParticle = self.truth[self.truth.particle_id == pid]
            w_sum = np.sum(HitOfParticle.weight.values)
            theParticle_list = self.truth.particle_id.values
            for j in range(len(pw)):
                if theParticle_list[j] == pid:
                    pw[j] = w_sum
        
        self.truth["particle_weight"] = pw

