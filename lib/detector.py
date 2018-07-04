import numpy as np

class Detector:
    def __init__(self, detectors):
        self.detectors = detectors

    def get_1local(self,hit):
        #assume only one hit. return arra([u,v,w])
        theModule = self.detectors[(self.detectors.volume_id == hit.volume_id.values[0])
                                 & (self.detectors.layer_id == hit.layer_id.values[0]) 
                                 & (self.detectors.module_id == hit.module_id.values[0])]
        trans = np.array([theModule.cx.values[0],theModule.cy.values[0],theModule.cz.values[0]])
        rot = np.array([[theModule.rot_xu.values[0],theModule.rot_xv.values[0],theModule.rot_xw.values[0]],
                        [theModule.rot_yu.values[0],theModule.rot_yv.values[0],theModule.rot_yw.values[0]],
                        [theModule.rot_zu.values[0],theModule.rot_zv.values[0],theModule.rot_zw.values[0]]])
        inv_rot = np.linalg.inv(rot)
        pos_xyz = np.array([hit.x.values[0], hit.y.values[0],hit.z.values[0]])
        pos_uvw = np.dot(inv_rot, pos_xyz - trans)

        return pos_uvw

    def get_local(self,hits):
        #assume hits of arbitraly size. return list[array([u,v,w]),array([u,v,w]),...]
        module_list = np.array([hits.volume_id.values,hits.layer_id.values, hits.module_id.values])
        pos_xyz = np.array([hits.x.values, hits.y.values,hits.z.values])
        pos_uvw = []
        for i in range(len(hits)):
            theModule = self.detectors[(self.detectors.volume_id == module_list[0,i]) & 
                                       (self.detectors.layer_id == module_list[1,i]) &
                                       (self.detectors.module_id == module_list[2,i])]
            trans = np.array([theModule.cx.values[0],theModule.cy.values[0],theModule.cz.values[0]])
            rot = np.array([[theModule.rot_xu.values[0],theModule.rot_xv.values[0],theModule.rot_xw.values[0]],
                           [theModule.rot_yu.values[0],theModule.rot_yv.values[0],theModule.rot_yw.values[0]],
                           [theModule.rot_zu.values[0],theModule.rot_zv.values[0],theModule.rot_zw.values[0]]])
            inv_rot = np.linalg.inv(rot)
            pos_uvw.append(np.dot(inv_rot, pos_xyz[:,i] - trans))

        return pos_uvw
