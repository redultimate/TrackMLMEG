from calculator import CalculatorCells as CalCells

class Converter:
	def particles2truth(particles, truth):
		particle_id = particles.particle_id.values
		vx = [0 for i in range(len(truth.hit_id.values))]
		vy = [0 for i in range(len(truth.hit_id.values))]
		vz = [0 for i in range(len(truth.hit_id.values))]
		px = [0 for i in range(len(truth.hit_id.values))]
		py = [0 for i in range(len(truth.hit_id.values))]
		pz = [0 for i in range(len(truth.hit_id.values))]
		q = [0 for i in range(len(truth.hit_id.values))]
		nhit = [0 for i in range(len(truth.hit_id.values))]

		for i in range(len(particle_id)):
			hit_id = truth.hit_id[truth.particle_id == particle_id[i]]
			if len(hit_id) == 0:
				continue
        
			vx2   = particles.vx[particles.particle_id == particle_id[i]]
			vy2   = particles.vy[particles.particle_id == particle_id[i]]
			vz2   = particles.vz[particles.particle_id == particle_id[i]]
			px2   = particles.px[particles.particle_id == particle_id[i]]
			py2   = particles.py[particles.particle_id == particle_id[i]]
			pz2   = particles.pz[particles.particle_id == particle_id[i]]
			q2    = particles.q[particles.particle_id == particle_id[i]]
			nhit2 = particles.nhit[particles.particle_id == particle_id[i]]
        
			for j in range(len(hit_id)):
				vx[hit_id.values[j]-1] = vx2.values[0]
				vy[hit_id.values[j]-1] = vy2.values[0]
				vz[hit_id.values[j]-1] = vz2.values[0]
				px[hit_id.values[j]-1] = px2.values[0]
				py[hit_id.values[j]-1] = py2.values[0]
				pz[hit_id.values[j]-1] = pz2.values[0]
				q[hit_id.values[j]-1] = q2.values[0]
				nhit[hit_id.values[j]-1] = nhit2.values[0]
        
			truth['vx'] = vx
			truth['vy'] = vy
			truth['vz'] = vz
			truth['px'] = px
			truth['py'] = py
			truth['pz'] = pz
			truth['q'] = q
			truth['nhit'] = nhit
        
		return 

	def truth2hits(truth,hits):
		hits['particle_id'] = truth.particle_id.values
		hits['tx'] = truth.tx.values
		hits['ty'] = truth.ty.values
		hits['tz'] = truth.tz.values
		hits['tpx'] = truth.tpx.values
		hits['tpy'] = truth.tpy.values
		hits['tpz'] = truth.tpz.values
		hits['weight'] = truth.weight.values
    
		return

	def cells2hits(cells,hits):
		total_edep = CalCells.sum_edep(cells)
		hits['total_edep'] = total_edep

		return

		
