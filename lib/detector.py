import numpy as np

class Detector:
    def __init__(self, detectors):
        self.detectors = detectors

    def get_detectors(self):
        return self.detectors

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
        #assume hits of arbitraly size. return array([[u,v,w],[u,v,w],...])
        module_list = np.array([hits.volume_id.values,hits.layer_id.values, hits.module_id.values]).transpose()
        pos_xyz = np.array([hits.x.values, hits.y.values,hits.z.values]).transpose()
        pos_uvw = np.zeros((len(hits),pos_xyz.shape[1]))
        for i in range(len(hits)):
            theModule = self.detectors[(self.detectors.volume_id == module_list[i,0]) & 
                                       (self.detectors.layer_id == module_list[i,1]) &
                                       (self.detectors.module_id == module_list[i,2])]
            trans = np.array([theModule.cx.values[0],theModule.cy.values[0],theModule.cz.values[0]])
            rot = np.array([[theModule.rot_xu.values[0],theModule.rot_xv.values[0],theModule.rot_xw.values[0]],
                           [theModule.rot_yu.values[0],theModule.rot_yv.values[0],theModule.rot_yw.values[0]],
                           [theModule.rot_zu.values[0],theModule.rot_zv.values[0],theModule.rot_zw.values[0]]])
            inv_rot = np.linalg.inv(rot)
            pos_uvw[i] = np.dot(inv_rot, pos_xyz[i] - trans)

        return pos_uvw

    def get_local_array(self,pos_xyz,module_list):
        #assume global xyz and module list of arbitraly size. return array([[u,v,w],[u,v,w],...])
        pos_uvw = np.zeros((len(pos_xyz),pos_xyz.shape[1]))
        for i in range(len(pos_xyz)):
            theModule = self.detectors[(self.detectors.volume_id == module_list[i,0]) & 
                                       (self.detectors.layer_id == module_list[i,1]) &
                                       (self.detectors.module_id == module_list[i,2])]
            trans = np.array([theModule.cx.values[0],theModule.cy.values[0],theModule.cz.values[0]])
            rot = np.array([[theModule.rot_xu.values[0],theModule.rot_xv.values[0],theModule.rot_xw.values[0]],
                           [theModule.rot_yu.values[0],theModule.rot_yv.values[0],theModule.rot_yw.values[0]],
                           [theModule.rot_zu.values[0],theModule.rot_zv.values[0],theModule.rot_zw.values[0]]])
            inv_rot = np.linalg.inv(rot)
            pos_uvw[i] = np.dot(inv_rot, pos_xyz[i] - trans)

        return pos_uvw

    def get_module_trans(self,module_list):
        ndata = len(module_list)
        cx = np.zeros(ndata)
        cy = np.zeros(ndata)
        cos_theta = np.zeros(ndata)
        sin_theta = np.zeros(ndata)
        for i in range(ndata):
            theModule = self.detectors[(self.detectors.volume_id == module_list[i,0]) &
                                       (self.detectors.layer_id == module_list[i,1]) &
                                       (self.detectors.module_id == module_list[i,2])]
            #module's trans
            cx[i] = theModule.cx.values[0]
            cy[i] = theModule.cy.values[0]
            #cz = theModule.cz.values[0]#not use z info
            #module's slope
            cos_theta[i] = theModule.rot_xu.values[0]
            sin_theta[i] = theModule.rot_xw.values[0]

        module_trans = {"cx":cx,"cy":cy,"cos_theta":cos_theta,"sin_theta":sin_theta}

        return module_trans
