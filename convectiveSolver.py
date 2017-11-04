import numpy as np
from system import system


class convectiveSolver(object):
    def __init__(self,
                 xmin=-1.0, xmax=1.0, nx=3, mx=4,
                 ymin=-1.0, ymax=1.0, ny=3, my=4,
                 boundaries=None, init=None, k=0.1,
                 Th=10, dt=0.1):

        self.system = system(xmin, xmax, ymin, ymax, nx, ny, mx, my, True)
        self.k = k
        self.dt, self.Th = dt, Th

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

    def product(self, args):
        product = 1.0
        for key in args:
            product = product * args[key]
        return product
    
    def diffusion(args):
        Tx, Ty = args['d2Tdx2'], args['d2Tdy2']
        return -1*self.k*(Tx+Ty)

    def euler_step(args, dt):
        d = args['dTdt']
        t = args['T']


    def solve(self, t, dt):
        for step in int(t/dt):
            self.s.ddx('dTdx','T')
            self.s.ddy('dTdy','T')

            self.s.ddx('d2Tdx2','dTdx')
            self.s.ddy('d2Tdy2','dTdy')

            self.s.add_property('dTdt')

        pass
