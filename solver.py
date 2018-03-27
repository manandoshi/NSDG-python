import numpy as np
from system import system


class Solver(object):
    def __init__(self,
                 xmin=-1.0, xmax=1.0, nx=3, mx=4,
                 ymin=-1.0, ymax=1.0, ny=3, my=4,
                 boundaries=None, init=None, 
                 alpha=0.1, nu=0.1, rho=1.0,
                 sample_x=30, sample_y=30,
                 Th=10, dt=0.1, exact=True, adv_flux="rusanov"):

        #Initializing "system"
        self.system = system(xmin, xmax, ymin, ymax, nx, ny, mx, my, exact=exact, num_samples_x=sample_x, num_samples_y=sample_y)
        self.dt, self.Th = dt, Th
        self.k = {'T':alpha, 'u':nu, 'v':nu }
        self.rho = rho

        self.fluxType = adv_flux

        #Adding main variables to system
        for p in ["u", "v", "T","P"]:
            self.system.add_property(
                p, func=init[p]["val"], arg_params=init[p]["args"], sample=True)

        self.system.set_boundaries_multivar(boundaries)

        boundaries = {}

        #adding derived variables to system
        for var in ['uT', 'vT', 'uu', 'vv']:
            self.system.add_property(
                var, func=self.product, arg_params=list(var))
            boundaries[var] = {}
            for direction in ['N', 'E', 'W', 'S']:
                boundaries[var][direction] = {
                    'type': 'dirichlet', 'val': self.product, 'args': list(var)}
        self.system.set_boundaries_multivar(boundaries)
        self.gen_inv_lap(self,nx,ny,mx,boundaries['P'])

    #Generate inverse laplace matrix with boundaries for Pressure poisson
    def gen_inv_lap(self,nx,ny,n,P_boundary):

        lap = self.system.lap
        derx, dery = self.system.global_derx, self.system.global_dery

        lap[:n*nx] = 0
        lap[:n*nx,:n*nx] = np.eye(n*nx) if P_boundary['S']['type'] == 'dirichlet' else dery[:n*nx,:n*nx]
        lap[-n*nx:] = 0
        lap[-n*nx:,-n*nx:] = np.eye(n*nx) if P_boundary['N']['type'] == 'dirichlet' else dery[-n*nx:,-n*nx:]

        lap[::n*nx] = 0
        lap[::n*nx,::n*nx] = np.eye(n*ny) if P_boundary['W']['type'] == 'dirichlet' else derx[::n*nx,::n*nx]
        lap[n*nx-1::n*nx] = 0
        lap[n*nx-1::n*nx,n*nx-1::n*nx] = np.eye(n*ny) if P_boundary['E']['type'] == 'dirichlet' else derx[n*nx-1::n*nx,n*nx-1::n*nx]

        self.inv_lap = np.linalg.pinv(lap)

    def P_solve(self):
        rhs = -1*self.rho*(self.system.properties['dudx']**2 + self.system.properties['dvdy']**2 + 2*self.system.properties['dudy']*self.properties['dvdx'])
        
        rhs[:n*nx]          = 0.0  #S
        rhs[-n*nx:]         = 0.0 #N
        rhs[::n*nx]         = 0.0 #W
        rhs[n*nx-1::n*nx]   = 0.0 #E
        
        self.system.properties['P'][:] = self.inv_lap.dot(rhs)

    #Utility fn to compute product of two properties
    def product(self, args):
        product = 1.0
        for key in args:
            product = product * args[key]
        return product
    
    def compute_dpdt(self,args,p):
        if p == 'u':
            return self.diffusion(args,p) - self.advection(args,p) - (1.0/self.rho)*selp.properties('dPdx')

        if p == 'v':
            return self.diffusion(args,p) - self.advection(args,p) - (1.0/self.rho)*selp.properties('dPdy')
        else:
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

    #Function to compute required terms used in advection, diffussion and pressure term computation
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

        #Pressure_term
        if p == 'u':
            self.system.ddx('dPdx','P')
        elif p == 'v':
            self.system.ddy('dPdy','P')
        
        #This shouldn't be here
        self.system.add_property('d'+p+'dt', func=lambda args:self.compute_dpdt(args, p), arg_params=['d2'+p+'dx2', 'd2'+p+'dy2','du'+p+'dx','dv'+p+'dy'])


    def RK2_step(self):
        #step1
        self.compute_terms('u')
        dudt   = self.system.properties['dudt']
        u      = self.system.properties['u']
        u_temp = T.copy()
        u[:]   = T + 0.5*self.dt*dudt

        self.compute_terms('v')
        dvdt   = self.system.properties['dvdt']
        v      = self.system.properties['v']
        v_temp = T.copy()
        v[:]   = T + 0.5*self.dt*dvdt

        #step2
        self.compute_terms('u')
        dudt   = self.system.properties['dudt']
        u[:]   = u_temp + self.dt*dudt

        self.compute_terms('v')
        dvdt   = self.system.properties['dvdt']
        v[:]   = v_temp + self.dt*dvdt

        del u_temp
        del v_temp
        
    def solve(self, Th=10.0, dt=0.10):

        self.Th = Th
        self.dt = dt
        u_temp = self.system.properties['u']
        v_temp = self.system.properties['v']
        P_temp = self.system.properties['P']
        for step in xrange(int(Th/dt)):
            while True:
                self.RK2_step()
                self.P_solve()
                if np.max(abs(self.system.properties['P']-P_temp)) < 1e-2:
                    break
                P_temp = self.system.properties['P']
                self.system.properties['u'][:] = u_temp
                self.system.properties['v'][:] = v_temp
        pass
