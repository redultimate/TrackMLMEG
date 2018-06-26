from calculate import CalculatorCells as CalCells

class Converter:
	def particles2truth(self, particles, truth):
		assembly = truth

		particle_id = particles.particle_id.values
		vx = [0 for i in range(len(truth.hit_id.values))]
		vy = [0 for i in range(len(truth.hit_id.values))]
		vz = [0 for i in range(len(truth.hit_id.values))]
		px = [0 for i in range(len(truth.hit_id.values))]
		py = [0 for i in range(len(truth.hit_id.values))]
		pz = [0 for i in range(len(truth.hit_id.values))]
		q = [0 for i in range(len(truth.hit_id.values))]
		nhits = [0 for i in range(len(truth.hit_id.values))]

		for i in range(len(particle_id)):
			hit_id = assembly.hit_id[assembly.particle_id == particle_id[i]]
			if len(hit_id) == 0:
				continue
        
			vx2   = particles.vx[particles.particle_id == particle_id[i]]
			vy2   = particles.vy[particles.particle_id == particle_id[i]]
			vz2   = particles.vz[particles.particle_id == particle_id[i]]
			px2   = particles.px[particles.particle_id == particle_id[i]]
			py2   = particles.py[particles.particle_id == particle_id[i]]
			pz2   = particles.pz[particles.particle_id == particle_id[i]]
			q2    = particles.q[particles.particle_id == particle_id[i]]
			nhits2 = particles.nhits[particles.particle_id == particle_id[i]]
        
			for j in range(len(hit_id)):
				vx[hit_id.values[j]-1] = vx2.values[0]
				vy[hit_id.values[j]-1] = vy2.values[0]
				vz[hit_id.values[j]-1] = vz2.values[0]
				px[hit_id.values[j]-1] = px2.values[0]
				py[hit_id.values[j]-1] = py2.values[0]
				pz[hit_id.values[j]-1] = pz2.values[0]
				q[hit_id.values[j]-1] = q2.values[0]
				nhits[hit_id.values[j]-1] = nhits2.values[0]
        
			assembly['vx'] = vx
			assembly['vy'] = vy
			assembly['vz'] = vz
			assembly['px'] = px
			assembly['py'] = py
			assembly['pz'] = pz
			assembly['q'] = q
			assembly['nhits'] = nhits
        
		return assembly

	def truth2hits(self,truth,hits):
		assembly = hits
		keys = truth.keys()
		for i in range(len(keys)):
			assembly[keys[i]] = truth[keys[i]].values
    
		return assembly

	def cells2hits(self,cells,hits):
		assembly = hits
		total_edep = CalCells.sum_edep(cells)
		assembly['total_edep'] = total_edep

		return assembly

	def AssembleAll(self,hits,cells,particles,truth):
		assembly = self.cells2hits(cells,hits)
		assembly2 = self.particles2truth(particles,truth)

		keys = assembly2.keys();
		for i in range(len(keys)):
			assembly[keys[i]] = assembly2[keys[i]].values

		return assembly
	
	
