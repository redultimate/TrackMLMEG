from calculator import CellCalculator as CellCal
from tqdm import tqdm

class Converter:
    def __init__(self,hits,cells,particles,truth):
        self.hits = hits
        self.cells = cells
        self.particles = particles
        self.truth = truth

    def convert_particles2truth(self):
        assembly = self.truth

        particle_id = self.particles.particle_id.values
        vx = [0 for i in range(len(self.truth.hit_id.values))]
        vy = [0 for i in range(len(self.truth.hit_id.values))]
        vz = [0 for i in range(len(self.truth.hit_id.values))]
        px = [0 for i in range(len(self.truth.hit_id.values))]
        py = [0 for i in range(len(self.truth.hit_id.values))]
        pz = [0 for i in range(len(self.truth.hit_id.values))]
        q = [0 for i in range(len(self.truth.hit_id.values))]
        nhits = [0 for i in range(len(self.truth.hit_id.values))]

        for i in tqdm(range(len(particle_id))):
            hit_id = assembly.hit_id[assembly.particle_id == particle_id[i]]
            if len(hit_id) == 0:
                continue
       
            vx2   = self.particles.vx[self.particles.particle_id == particle_id[i]]
            vy2   = self.particles.vy[self.particles.particle_id == particle_id[i]]
            vz2   = self.particles.vz[self.particles.particle_id == particle_id[i]]
            px2   = self.particles.px[self.particles.particle_id == particle_id[i]]
            py2   = self.particles.py[self.particles.particle_id == particle_id[i]]
            pz2   = self.particles.pz[self.particles.particle_id == particle_id[i]]
            q2    = self.particles.q[self.particles.particle_id == particle_id[i]]
            nhits2 = self.particles.nhits[self.particles.particle_id == particle_id[i]]
       
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

    def convert_truth2hits(self):
        assembly = self.hits
        keys = self.truth.keys()
        for i in range(len(keys)):
            assembly[keys[i]] = self.truth[keys[i]].values
    
        return assembly

    def convert_cells2hits(self):
        cellcal = CellCal(self.cells)
        assembly = self.hits
        hit_edep = cellcal.get_hit_edep()
        assembly['hit_edep'] = hit_edep

        return assembly

    def assemble_all(self):
        assembly = self.convert_cells2hits()
        assembly2 = self.convert_particles2truth()

        keys = assembly2.keys();
        for i in tqdm(range(len(keys))):
            assembly[keys[i]] = assembly2[keys[i]].values

        return assembly
	
	
