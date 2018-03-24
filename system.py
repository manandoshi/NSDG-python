import lobatto
import interpolation
import matrix_generator
import numpy as np
from element import element
from utilities import split, edge, opp


class system(object):
    def __init__(self,
                 xmin=-1.0, xmax=1.0,
                 ymin=-1.0, ymax=1.0,
                 num_el_x=3, num_el_y=3,
                 order_x=10, order_y=10,
                 exact=True,
                 num_samples_x=20, num_samples_y=20):

        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.nx, self.ny = num_el_x, num_el_y
        self.mx, self.my = order_x, order_y
        self.dx, self.dy = (xmax - xmin) / (self.nx), (ymax - ymin) / (self.ny)
        self.num_samples_x, self.num_samples_y = num_samples_x, num_samples_y

        self.x_sample = np.linspace(-1, 1, num_samples_x)
        self.y_sample = np.linspace(-1, 1, num_samples_y)

        self.properties = {}
        self.boundaries = {}

        self.generate_nodes()

        self.properties["x"], self.properties["y"] = np.meshgrid(np.zeros([self.nx * self.mx]),
                                                                 np.zeros([self.ny * self.my]))

        self.properties["x_sample"], self.properties["y_sample"] = np.meshgrid(np.zeros([self.nx * self.num_samples_x]),
                                                                               np.zeros([self.ny * self.num_samples_y]))
        self.create_elements()
        self.generate_matrices(exact)

        return

    def create_elements(self):
        self.x_elements, self.y_elements = np.meshgrid(np.linspace(self.xmin, self.xmax, self.nx + 1)[:-1],
                                                       np.linspace(self.ymin, self.ymax, self.ny + 1)[:-1])
        self.elements = self.x_elements.tolist()
        for i in xrange(self.nx):
            for j in xrange(self.ny):

                prop = {}
                prop["x"] = self.properties["x"][j * self.my:(j + 1) * self.my,
                                                 i * self.mx:(i + 1) * self.mx]

                prop["y"] = self.properties["y"][j * self.my:(j + 1) * self.my,
                                                 i * self.mx:(i + 1) * self.mx]

                prop["x_sample"] = self.properties["x_sample"][j * self.num_samples_y:(j + 1) * self.num_samples_y,
                                                               i * self.num_samples_x:(i + 1) * self.num_samples_x]

                prop["y_sample"] = self.properties["y_sample"][j * self.num_samples_y:(j + 1) * self.num_samples_y,
                                                               i * self.num_samples_x:(i + 1) * self.num_samples_x]

                self.elements[j][i] = element(self.x_elements[j][i], self.dx,
                                              self.y_elements[j][i], self.dy,
                                              properties=prop, system=self, num_samples_x=self.num_samples_x, num_samples_y=self.num_samples_y)

        for p in ["x", "y"]:
            prop = self.properties[p]
            self.boundaries[p] = {'N': prop[-1, :],
                                  'S': prop[0, :],
                                  'W': prop[:, 0],
                                  'E': prop[:, -1]}
        for i in xrange(self.nx):
            for j in xrange(self.ny):
                if i != self.nx - 1:
                    self.elements[j][i].setNeighbor(
                        typ='E', element=self.elements[j][i + 1])
                    self.elements[j][i +
                                     1].setNeighbor(typ='W', element=self.elements[j][i])
                if j != self.ny - 1:
                    self.elements[j][i].setNeighbor(
                        typ='N', element=self.elements[j + 1][i])
                    self.elements[j +
                                  1][i].setNeighbor(typ='S', element=self.elements[j][i])

    def add_property(self, p, func=None, arg_params=[], copy_to_elements=True, sample=False):
        sub_dict = {arg_param: self.properties[arg_param]
                    for arg_param in arg_params}
        if func is None:
            self.properties[p] = np.zeros_like(self.properties["x"])
        else:
            self.properties[p] = func(sub_dict)

        prop = self.properties[p]

        if sample:
            self.properties[p +
                            '_sample'] = np.zeros_like(self.properties["x_sample"])

        self.boundaries[p] = {'N': prop[-1, :],
                              'S': prop[0, :],
                              'W': prop[:, 0],
                              'E': prop[:, -1]}
        if copy_to_elements:
            self.copy_property_to_elements(p, sample)

        for i in xrange(self.nx):
            for j in xrange(self.ny):
                self.elements[j][i].setNeighborBoundary(p)

    def update_property(self, p, func=None, arg_params=[]):
        sub_dict = {arg_param: self.properties[arg_param]
                    for arg_param in arg_params}
        if func is None:
            self.properties[p][:] = np.zeros_like(self.properties["x"])
        else:
            self.properties[p][:] = func(sub_dict)

    def set_boundaries_multivar(self, boundaries):
        for p, boundary_p in boundaries.iteritems():
            self.set_boundaries(p, boundary_p)

    def set_boundaries(self, p, boundaries):
        for direction, boundary in boundaries.iteritems():
            self.set_boundary(p, direction, boundary)

    def set_boundary(self, p, direction, boundary):
        if boundary["type"] == 'dirichlet':
            # Computing the BC give fn
            sub_dict = {arg: self.boundaries[arg][direction]
                        for arg in boundary['args']}
            self.boundaries[p][direction] = boundary['val'](sub_dict)

            # Setting BC for elements
            for element, boundary_val in zip(edge(self.elements, direction),
                                             split(direction, self.boundaries[p][direction], self.nx, self.ny)):
                element.setNeighborBoundary(p, typ=direction, val=boundary_val)

        if boundary["type"] == 'periodic':
            for element_dir, element_opp in zip(edge(self.elements, direction),
                                                edge(self.elements, opp(direction))):
                element_dir.setNeighborBoundary(p, typ=direction,
                                                val=element_opp.boundaries[p][opp(direction)])

    def copy_property_to_elements(self, p, sample):
        for i in xrange(self.nx):
            for j in xrange(self.ny):
                self.elements[j][i].add_property(p, self.properties[p][j * self.my:(j + 1) * self.my,
                                                                       i * self.mx:(i + 1) * self.mx])

        if sample:
            for i in xrange(self.nx):
                for j in xrange(self.ny):
                    self.elements[j][i].add_property(p + '_sample', self.properties[p + '_sample'][j * self.num_samples_y:(j + 1) * self.num_samples_y,
                                                                                                   i * self.num_samples_x:(i + 1) * self.num_samples_x])

    def generate_nodes(self):
        self.x_nodes, self.x_weights = lobatto.compute_nodes(self.mx)
        self.y_nodes, self.y_weights = lobatto.compute_nodes(self.my)

        self.interp = interpolation.compute_2Dmatrix(
            self.x_nodes, self.y_nodes, self.x_sample, self.y_sample)

        self.lobatto_mesh = np.meshgrid(self.x_nodes, self.y_nodes)

    def generate_matrices(self, exact=True):
        self.M = self.dx * self.dy * 0.25 * matrix_generator.massMatrix_2D(self.x_nodes, self.y_nodes,
                                                                           self.x_weights, self.y_weights, exact)
        self.M_inv = np.linalg.inv(self.M)
        self.Dx, self.Dy = matrix_generator.derMatrix_2D(self.x_nodes, self.y_nodes,
                                                         self.x_weights, self.y_weights, exact)
        self.Dx = self.Dx * self.dy / 2
        self.Dy = self.Dy * self.dx / 2
        self.Fy, self.Fx = matrix_generator.fluxMatrix(self.x_nodes, self.y_nodes,
                                                       self.x_weights, self.y_weights, exact)

        self.Fx = self.Fx * self.dy / 2
        self.Fy = self.Fy * self.dx / 2

    def set_property(self, prop, func=None, val=None):
        if func is not None:
            self.properties[prop] = func(
                self.properties["x"], self.properties["y"])

    def ddx(self, outVar, var, fluxType="centered", fluxVar="u"):
        if outVar not in self.properties:
            self.add_property(outVar)
        for j in xrange(self.ny):
            for i in xrange(self.nx):
                self.elements[j][i].ddx(
                    outVar, var, fluxType, fluxVar, self.Dx, self.Fx)

    def ddy(self, outVar, var, fluxType="centered", fluxVar="v"):
        if outVar not in self.properties:
            self.add_property(outVar)
        for j in xrange(self.ny):
            for i in xrange(self.nx):
                self.elements[j][i].ddy(
                    outVar, var, fluxType, fluxVar, self.Dy, self.Fy)

    def computeSample(self, p):
        for j in xrange(self.ny):
            for i in xrange(self.nx):
                self.elements[j][i].computeSample(p)
        return self.properties[p + '_sample']
