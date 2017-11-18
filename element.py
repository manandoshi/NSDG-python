import numpy as np
from directions import opp


class element(object):
    def __init__(self, x, dx, y, dy, properties={}, system=None, num_samples_x=2, num_samples_y=2):
        self.x_min = x
        self.y_min = y
        self.x_max = x + dx
        self.y_max = y + dy

        self.dx, self.dy = dx, dy

        self.properties = properties
        self.system = system

        self.x_comp = self.system.x_nodes
        self.y_comp = self.system.y_nodes

        

        self.x_real = self.x_min + (self.x_comp + 1) * dx / 2
        self.y_real = self.y_min + (self.y_comp + 1) * dy / 2

        self.properties["x"][:], self.properties["y"][:] = np.meshgrid(
            self.x_real, self.y_real)

        self.x_sample = self.x_min + (self.system.x_sample + 1) * dx / 2
        self.y_sample = self.y_min + (self.system.y_sample + 1) * dy / 2

        self.properties["x_sample"][:],self.properties["y_sample"][:] = np.meshgrid(
                self.x_sample, self.y_sample)

        self.neighbor = {}
        self.boundaries = {}
        self.neighbor_boundaries = {}

    def add_property(self, p, val):
        self.properties[p] = val
        self.set_self_boundary(p)
        self.neighbor_boundaries[p] = self.boundaries[p].copy()

    def set_self_boundary(self, p):
        prop = self.properties[p]
        self.boundaries[p] = {'N': prop[-1, :],
                              'S': prop[0, :],
                              'W': prop[:, 0],
                              'E': prop[:, -1]}

    def setNeighbor(self, neighbors=None, typ=None, element=None):
        if neighbors is not None:
            self.neighbor.update(neighbors)
        else:
            self.neighbor[typ] = element

    def setNeighborBoundary(self, p, val=None, typ='N'):
        if val is None:
            for n_typ, neighbor in self.neighbor.iteritems():
                self.neighbor_boundaries[p][n_typ] = neighbor.boundaries[p][opp(
                    n_typ)]

        else:
            self.neighbor_boundaries[p][typ] = val

    def boundaryCondition(self, prop, typ, func):
        self.neighbor_boundaries[prop][typ][:] = func(
            self.boundaries["x"][typ][:], self.boundaries["y"][typ][:])
        pass

    def ddx(self, outVar, var, fluxType, fluxVar, Dx, Fx):
        prop = self.properties[var]
        outProp = self.properties[outVar]
        outProp[:] = Dx.dot(prop.ravel()).reshape(outProp.shape)
        self.set_self_boundary(outVar)

        if fluxType == "centered":
            self.boundaries[outVar]['W'][:] += Fx.dot(0.5 * (self.boundaries[var]['W']
                                                             + self.neighbor_boundaries[var]['W']))

            self.boundaries[outVar]['E'][:] += -1*Fx.dot(0.5 * (self.boundaries[var]['E']
                                                         + self.neighbor_boundaries[var]['E']))
        elif fluxType == "rusanov":
            mask_in = (self.boundaries[fluxVar]['W'] > 0).astype(int)
            mask_out= (self.boundaries[fluxVar]['E'] > 0).astype(int)

            self.boundaries[outVar]['W'][:] += Fx.dot((1-mask_in) * (self.boundaries[var]['W'])
                                                             + self.neighbor_boundaries[var]['W']*mask_in)

            self.boundaries[outVar]['E'][:] += -1*Fx.dot(mask_out * self.boundaries[var]['E']
                                                         + (1-mask_out)*(self.neighbor_boundaries[var]['E']))

        outProp[:] = -1*self.system.M_inv.dot(outProp.ravel()).reshape(outProp.shape)

    def ddy(self, outVar, var, fluxType, fluxVar, Dy, Fy):
#        import pdb; pdb.set_trace()
        prop        = self.properties[var]
        outProp     = self.properties[outVar]
        outProp[:]  = Dy.dot(prop.ravel()).reshape(outProp.shape)

        self.set_self_boundary(outVar)

        if fluxType == "centered":
            self.boundaries[outVar]['S'][:] += Fy.dot(0.5 * (self.boundaries[var]['S']
                                                             + self.neighbor_boundaries[var]['S']))

            self.boundaries[outVar]['N'][:] += -1*Fy.dot(0.5 * (self.boundaries[var]['N']
                                                         + self.neighbor_boundaries[var]['N']))
        elif fluxType == "rusanov":
            mask_in = (self.boundaries[fluxVar]['S'] > 0).astype(int)
            mask_out= (self.boundaries[fluxVar]['N'] > 0).astype(int)

            self.boundaries[outVar]['S'][:] += Fy.dot((1-mask_in) * (self.boundaries[var]['S'])
                                                             + self.neighbor_boundaries[var]['S']*mask_in)

            self.boundaries[outVar]['N'][:] += -1*Fy.dot(mask_out * self.boundaries[var]['N']
                                                         + (1-mask_out)*(self.neighbor_boundaries[var]['N']))

        outProp[:] = -1*self.system.M_inv.dot(outProp.ravel()).reshape(outProp.shape)

    def computeSample(self,p):
        self.properties[p+'_sample'][:] = \
        self.system.interp.dot(self.properties[p].ravel()).reshape(self.properties['x_sample'].shape)
