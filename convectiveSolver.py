import numpy as np
from system import system


class convectiveSolver(object):
    def __init__(self,
                 xmin=-1.0, xmax=1.0, nx=3, mx=4,
                 ymin=-1.0, ymax=1.0, ny=3, my=4,
                 boundaries=None, init=None):

        self.system = system(xmin, xmax, ymin, ymax, nx, ny, mx, my, True)

        for p in ["u", "v", "T"]:
            self.system.add_property(
                p, func=init[p]["val"], arg_params=init[p]["args"])

        self.system.set_boundaries_multivar(boundaries)

        self.system.add_property(
            "uT", func=self.product, arg_params=["u", "T"])
        self.system.add_property(
            "vT", func=self.product, arg_params=["v", "T"])

        boundaries = {}
        for var in ['uT', 'vT']:
            boundaries[var] = {}
            for direction in ['N', 'E', 'W', 'S']:
                boundaries[var][direction] = {
                    'type': 'dirichlet', 'val': self.product, 'args': list(var)}
        self.system.set_boundaries_multivar(boundaries)
        #self.system.setBoundary("uT", func=self.product, arg_params=["u","T"], typs=["N","E","W","S"])
        #self.system.setBoundary("vT", func=self.product, arg_params=["v","T"], typs=["N","E","W","S"])

    def product(self, args):
        product = 1.0
        for key in args:
            product = product * args[key]
        return product

    def solve(self, t, dt):
        pass
