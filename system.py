import lobatto
import interpolation
import matrix_generator
import numpy as np
from element import element

class system(object):
    def __init__(   self, xmin=-1.0, xmax=1.0, 
                    ymin=-1.0, ymax=1.0, num_el_x=3, 
                    num_el_y=3, order_x=10, order_y=10, 
                    exact = True):
        
        self.xmin, self.xmax = xmin, xmax
        self.nx,  self.ny    = num_el_x, num_el_y
        self.mx, self.my     = order_x, order_y
        self.dx, self.dy     = (xmax-xmin)/(self.nx-1), (ymax-ymin)/(self.ny-1)

        self.x_nodes, self.x_weights = lobatto.compute_nodes(self.mx)
        self.y_nodes, self.y_weights = lobatto.compute_nodes(self.my)

        self.mesh = np.meshgrid(self.x_nodes, self.y_nodes)
    
        self.elements = [[element(x,y,self.dx,self.dy, system=self)\
                            for y in np.linspace(ymin,ymax,self.ny)] \
                            for x in np.linspace(xmin,xmax,self.nx)]

        self.M      = matrix_generator.massMatrix_2D(self.x_nodes, self.y_nodes, self.x_weights, self.y_weights, exact)
        self.M_inv  = np.linalg.inv(self.M)
        self.Dx, self.Dy   = matrix_generator.derMatrix_2D(self.x_nodes, self.y_nodes, self.x_weights, self.y_weights, exact)
        self.Fx, self.Fy   = matrix_generator.fluxMatrix(self.x_nodes, self.y_nodes, self.x_weights, self.y_weights, exact)


        for i in xrange(self.nx):
            for j in xrange(self.ny):
                if i > 0:
                    self.elements[i][j].setNeighbor(typ="W", element=self.elements[i-1][j])
                    self.elements[i-1][j].setNeighbor(typ="E", element=self.elements[i][j])
                if j > 0:
                    self.elements[i][j].setNeighbor(typ="S", element=self.elements[i][j-1])
                    self.elements[i][j-1].setNeighbor(typ="N", element=self.elements[i][j])
