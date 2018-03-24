import numpy as np
from system import system


class convectiveSolver(object):
    def __init__(self,
                 xmin=-1.0, xmax=1.0, nx=3, mx=4,
                 ymin=-1.0, ymax=1.0, ny=3, my=4,
                 boundaries=None, init=None, 
                 alpha=0.1, nu=0.1,
                 sample_x=30, sample_y=30,
                 Th=10, dt=0.1, exact=True, adv_flux="rusanov"):

        self.system = system(xmin, xmax, ymin, ymax, nx, ny, mx, my, exact=exact, num_samples_x=sample_x, num_samples_y=sample_y)
        self.dt, self.Th = dt, Th
        self.k = {'T':alpha, 'u':nu, 'v':nu }
        self.fluxType = adv_flux

        for p in ["u", "v", "T","P"]:
            self.system.add_property(
                p, func=init[p]["val"], arg_params=init[p]["args"], sample=True)

        self.system.set_boundaries_multivar(boundaries)

        boundaries = {}

        for var in ['uT', 'vT', 'uu', 'vv']:
            self.system.add_property(
                var, func=self.product, arg_params=list(var))
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
    
    def advectionDiffusion(self,args,p):
        return self.diffusion(args,p) - self.advection(args,p)

    def diffusion(self, args, p):
        px, py = args['d2'+p+'dx2'], args['d2'+p+'dy2']
        return self.k[p]*(px+py)

    def advection(self, args, p):
        px, py = args['du'+p+'dx'],args['dv'+p+'dy']
        return px+py

    def euler_step(self, args):
        d = args['dpdt']
        t = args['T']
        return t + d*self.dt

    def compute_terms(self, p):
        #Advection terms
        self.system.add_property('u'+p, func=self.product, arg_params=['u',p])
        self.system.add_property('v'+p, func=self.product, arg_params=['v',p])
        boundaries = {}
        for var in ['u'+p, 'v'+p]:
            boundaries[var] = {}
            for direction in ['N', 'E', 'W', 'S']:
                boundaries[var][direction] = {
                    'type': 'dirichlet', 'val': self.product, 'args': list(var)}
        self.system.set_boundaries_multivar(boundaries)
        self.system.ddx('du'+p+'dx','u'+p, fluxType=self.fluxType, fluxVar='u')
        self.system.ddy('dv'+p+'dy','v'+p, fluxType=self.fluxType, fluxVar='v')

        #Diffussion terms
        self.system.ddx('d'+p+'dx',p)
        self.system.ddy('d'+p+'dy',p)
        
        self.system.ddx('d2'+p+'dx2','d'+p+'dx')
        self.system.ddy('d2'+p+'dy2','d'+p+'dy')

        self.system.add_property('d'+p+'dt', func=lambda args:self.advectionDiffusion(args, p), arg_params=['d2'+p+'dx2', 'd2'+p+'dy2','du'+p+'dx','dv'+p+'dy'])

    def RK2_step(self):
        #step1
        self.compute_terms('T')
        dTdt   = self.system.properties['dTdt']
        T      = self.system.properties['T']
        T_temp = T.copy()
        T[:]   = T + 0.5*self.dt*dTdt

        self.compute_terms('T')
        dTdt   = self.system.properties['dTdt']
        T[:]   = T_temp + self.dt*dTdt

        del T_temp
        
    def solve(self, Th=10.0, dt=0.10):
        
        self.Th = Th
        self.dt = dt
        for step in xrange(int(Th/dt)):
            self.RK2_step()
        pass
