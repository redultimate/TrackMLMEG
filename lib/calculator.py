import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler

class CellCalculator:
	def get_hit_edep(self,cells):
		edeps = []
		for i in range(len(cells.hit_id.unique())):
			edep = cells.value[cells.hit_id==i]
			edep_sum = sum(edep)
			edeps = np.append(edeps,edep_sum)

		return edeps

class ParticleCalculator:
	def get_momentum(self,particles):
		px = particles.px.values
		py = particles.py.values
		pz = particles.pz.values
		p = np.sqrt(px*px + py*py + pz*pz)

		return p	
