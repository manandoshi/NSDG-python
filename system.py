import lobatto
import interpolation
import matrix_generator
import numpy as np
from element import element


class system(object):
    def __init__(   self,
                    xmin=-1.0, xmax=1.0,
                    ymin=-1.0, ymax=1.0,
                    num_el_x=3, num_el_y=3,
                    order_x=10, order_y=10,
                    exact = True):
        
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.nx, self.ny     = num_el_x, num_el_y
        self.mx, self.my     = order_x, order_y
        self.dx, self.dy     = (xmax-xmin)/(self.nx), (ymax-ymin)/(self.ny)
        self.properties = {}
        self.boundaries = {}

        self.generate_nodes()

        self.properties["x"], self.properties["y"]  = np.meshgrid(np.zeros([self.nx * self.mx]), 
                                                                  np.zeros([self.ny * self.my]))
        self.create_elements()
        self.generate_matrices(exact)
    
        return

    def create_elements(self):
        self.x_elements, self.y_elements = np.meshgrid(np.linspace(self.xmin,self.xmax, self.nx + 1)[:-1], 
                                                       np.linspace(self.ymin,self.ymax, self.ny + 1)[:-1])
        self.elements = self.x_elements.tolist()
        for i in xrange(self.nx):
            for j in xrange(self.ny):
                
                prop = {}
                prop["x"] = self.properties["x"][j*self.my:(j+1)*self.my,
                                                 i*self.mx:(i+1)*self.mx]

                prop["y"] = self.properties["y"][j*self.my:(j+1)*self.my,
                                                  i*self.mx:(i+1)*self.mx]

                self.elements[j][i] = element(self.x_elements[j][i], self.dx,
                                              self.y_elements[j][i], self.dy, 
                                              properties=prop, system=self)

        for p in ["x","y"]:
            prop = self.properties[p]
            self.boundaries[p] = {'N':prop[-1, :],
                                  'S':prop[0 , :],
                                  'W':prop[: , 0],
                                  'E':prop[: ,-1]}
        for i in xrange(self.nx):
            for j in xrange(self.ny):
                if i != self.nx-1:
                    self.elements[j][i].setNeighbor(typ='E', element=self.elements[j][i+1])
                    self.elements[j][i+1].setNeighbor(typ='W', element=self.elements[j][i])
                if j != self.ny-1:
                    self.elements[j][i].setNeighbor(typ='N', element=self.elements[j+1][i])
                    self.elements[j+1][i].setNeighbor(typ='S', element=self.elements[j][i])

    def add_property(self, p, func, arg_params, copy_to_elements=True):
        sub_dict = {arg_param: self.properties[arg_param] for arg_param in arg_params}
        self.properties[p] = func(sub_dict)
        self.boundaries[p] = {}
        if copy_to_elements:
            self.copy_property_to_elements(p)

        for i in xrange(self.nx):
            for j in xrange(self.ny):
                self.elements[j][i].setNeighborBoundary(p)

    def set_boundary(self, p, arg_params=[], func=None, typs=["N", "E", "W", "S"], typ=None):
        if typ is not None:
            typs = [typ]

        for typ in typs:
            sub_dict = {arg_param: self.boundaries[arg_param][typ] for arg_param in arg_params}
            self.boundaries[p][typ] = func(sub_dict)

            if typ == 'N':
                for i in xrange(self.nx):
                    self.elements[-1][i].setNeighborBoundary(p, typ=typ, 
                            val= self.boundaries[p][typ][i*self.mx:(i+1)*self.mx])
            if typ == 'S':
                for i in xrange(self.nx):
                    self.elements[0][i].setNeighborBoundary(p, typ=typ, 
                            val= self.boundaries[p][typ][i*self.mx:(i+1)*self.mx])
            if typ == 'E':
                for j in xrange(self.ny):
                    self.elements[j][-1].setNeighborBoundary(p, typ=typ, 
                            val= self.boundaries[p][typ][j*self.mx:(j+1)*self.mx])
            if typ == 'W':
                for j in xrange(self.ny):
                    self.elements[j][0].setNeighborBoundary(p, typ=typ, 
                            val= self.boundaries[p][typ][j*self.mx:(j+1)*self.mx])


    def copy_property_to_elements(self, p):
        for i in xrange(self.nx):
            for j in xrange(self.ny):
                self.elements[j][i].add_property(p,self.properties[p][j*self.my:(j+1)*self.my,
                                                                    i*self.mx:(i+1)*self.mx])
    def generate_nodes(self):
        self.x_nodes, self.x_weights = lobatto.compute_nodes(self.mx)
        self.y_nodes, self.y_weights = lobatto.compute_nodes(self.my)

        self.lobatto_mesh = np.meshgrid(self.x_nodes, self.y_nodes)
    
    def generate_matrices(self, exact=True):
        self.M      = matrix_generator.massMatrix_2D(self.x_nodes, self.y_nodes, self.x_weights, self.y_weights, exact)
        self.M_inv  = np.linalg.inv(self.M)
        self.Dx, self.Dy   = matrix_generator.derMatrix_2D(self.x_nodes, self.y_nodes, self.x_weights, self.y_weights, exact)
        self.Fx, self.Fy   = matrix_generator.fluxMatrix(self.x_nodes, self.y_nodes, self.x_weights, self.y_weights, exact)

    def set_property(self,prop,func=None,val=None):
        if func is not None:
            self.properties[prop] = func(self.properties["x"], self.properties["y"])
