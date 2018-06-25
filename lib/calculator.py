import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler

class CalculatorCells:
	def sum_edep(cells):
		n = len(hits.hit_id.values)
		edeps = []
		for i in range(len(cells.hit_id.unique())):
			edep = cells.value[cells.hit_id==i]
			edep_sum = sum(edep)
			edeps = np.append(edeps,edep_sum)
            
		return edeps

class CalculatorParticles:
	def momentum(particles):
		px = particles.px.values
		py = particles.py.values
		pz = particles.pz.values
		p = np.sqrt(px*px + py*py + pz*pz)

		particles['p'] = p

		return p	
