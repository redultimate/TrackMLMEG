# based on
# https://www.kaggle.com/wesamelshamy/trackml-problem-explanation-and-data-exploration

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plots2D:
    def __init__(self):
        self.hits = None

    def get_XY(self, hits):
        self.hits = hits
        g = sns.jointplot(self.hits.x, self.hits.y, s=1, size=12)
        g.ax_joint.cla()
        plt.sca(g.ax_joint)

        volumes = self.hits.volume_id.unique()
        for volume in volumes:
            v = self.hits[self.hits.volume_id == volume]
            plt.scatter(v.x, v.y, s=2, label='volume {}'.format(volume))

        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.legend()
        plt.show()

    def get_YZ(self, hits):
        self.hits = hits
        g = sns.jointplot(self.hits.z, self.hits.y, s=1, size=12)
        g.ax_joint.cla()
        plt.sca(g.ax_joint)

        volumes = self.hits.volume_id.unique()
        for volume in volumes:
            v = self.hits[self.hits.volume_id == volume]
            plt.scatter(v.z, v.y, s=3, label='volume {}'.format(volume))

        plt.xlabel('Z (mm)')
        plt.ylabel('Y (mm)')
        plt.legend()
        plt.show()

class Plots3D:
    def __init__(self):
        self.hits = None

    def get_XYZ(self, hits):
        self.hits = hits
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        volumes = self.hits.volume_id.unique()
        for volume in volumes:
            v = self.hits[self.hits.volume_id == volume]
            ax.scatter(v.z, v.x, v.y, s=1, label='volume {}'.format(volume), alpha=0.5)
        ax.set_title('SHit Locations')
        ax.set_xlabel('Z (millimeters)')
        ax.set_ylabel('X (millimeters)')
        ax.set_zlabel('Y (millimeters)')
        plt.show()
	
    def get_track(self, assembly, particles_sample):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        particle_id = particles_sample.particle_id.values

        for i in range(len(particle_id)):
            hit_id = assembly.hit_id[assembly.particle_id == particle_id[i]]
            x = assembly.x[assembly.particle_id == particle_id[i]]
            y = assembly.y[assembly.particle_id == particle_id[i]]
            z = assembly.z[assembly.particle_id == particle_id[i]]
        
            ax.scatter(z, x, y, s=5, label='particle {}'.format(particle_id[i]), alpha=1)
    
            print('particle ', particle_id[i], ' nhits = ', particles_sample.nhits.values[i])
 
    
        ax.set_title('track')
        ax.set_xlabel('z (millimeters)')
        ax.set_ylabel('x (millimeters)')
        ax.set_zlabel('y (millimeters)')
        plt.show()

