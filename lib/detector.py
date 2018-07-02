import numpy as np

class Detector:
    def __init__(self, detectors):
        self.detectors = detectors

    def get_local(self,hit):
        #assume only one hit
        theModule = detectors[detectors.volume_id == hit.volume_id and detectors.layer_id == hit.layer_id and detectors.module_id == hit.module_id]
        trans = np.array([theModule.cx.values[0],theModule.cy.values[0],theModule.cz.values[0]])
        rot = np.array([theModule.rot_xu.values[0],theModule.rot_xv.values[0],theModule.rot_xw.values[0]],
                       [theModule.rot_yu.values[0],theModule.rot_yv.values[0],theModule.rot_yw.values[0]],
                       [theModule.rot_zu.values[0],theModule.rot_zv.values[0],theModule.rot_zw.values[0]])
        inv_rot = np.linalg.inv(rot)
        pos_xyz = np.array([hit.x.values[0], hit.y.values[0],hit.z.values[0]])
        pos_uvw = np.dot(rot_inv, pos_xyz - trans)

        return pos_uvw

