import numpy as np


def opp(direction):
    if direction = 'N':
        return "S"
    elif direction == 'S':
        return "N"
    elif direction == 'E':
        return "W"
    elif direction == 'W':
        return "S"
    else:
        assert False

class element(object):
    def __init__(self, x, y, dx, dy, system):
        self.x_min = x
        self.y_min = y
        self.x_max = x+dx
        self.y_max = y+dy
        
        self.dx, self.dy = dx,dy

        self.property = {}
        self.system = system

        self.x_comp = system.x_nodes
        self.y_comp = system.y_nodes

        self.x_real = self.x_min + (self.x_comp + 1)*dx/2
        self.y_real = self.y_min + (self.y_comp + 1)*dy/2

        self.mesh = np.meshgrid(self.x_real,self.y_real)

        self.neighbor = {}
        self.boundary = {}
        self.neighbor_boundary = {}

        self.addProp("x", lambda m: m[0]}
        self.addProp("y", lambda m: m[1]}


    def addProp(self, p, init = None):
        if init is None:
            self.property[p] = np.zeros([self.system.mx,self.system.my])
        else:
            self.property[p] = init(self.mesh)

        self.setBoundary(p)

    def setBoundary(self, p):
        prop = self.property[p]
        self.boundary[p] = {'N':prop[-1, :],
                            'S':prop[0 , :],
                            'W':prop[: , 0],
                            'E':prop[: ,-1]}

    def setNeighbor(self, neighbors = None, typ = None, element = None):
        if neighbors is not None:
            self.neighbor.update(neighbors)
        else:
            self.neighbor[typ] = element


    def setNeighborBoundary(self, p, neighbors = None):
        self.neighbor_boundary[p] = {}
        for typ, neighbor in self.neighbor:
            self.neighbor_boundary[p][typ] = neighbor.boundary[opp(typ)]

    def boundaryCondition(self,prop,typ,func):
        self.neighbor_boundary[prop][typ][:] = func(self.boundary["x"][typ][:], self.boundary["y"][typ][:])
        pass


    def ddx(self,prop, outProp,flux="centered",fluxVar=""):
        prop = self.property[prop]
        if flux == "centered":
            self.property[outProp] = system.Dx.dot(prop.ravel()).reshape(prop.shape)
            self.setBoundary(p)
            self.setNeighborBoundary(p)
