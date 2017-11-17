A framework for incompressible Navier-Stokes using Discontinuous Galerkin written in python.

Exaxmple:
```python
import system
import convectiveSolver
import numpy as np
import matplotlib.pyplot as plt
from utilities import *

def one(args, factor=1.0):
    x,y = args["x"], args["y"]
    return factor*np.ones_like(x);

def zero(args):
    x,y = args["x"], args["y"]
    return np.zeros_like(x)

def sin_y(args, factor=1.0):
    x,y = args["x"], args["y"]
    return factor*np.sin(np.pi*y);

def gaussian(args):
    x,y = args["x"], args["y"]
    return (1/(0.1))*np.exp(-(x**2+y**2)/(0.4*0.1))

def exact(x,y,t):
    return (1/(0.1+t))*np.exp(-(x**2+y**2)/(0.4*(0.1+t)))

boundary_u = {'N':{'type':'dirichlet','val':zero, 'args':["x","y"]},
              'E':{'type':'dirichlet','val':zero, 'args':["x","y"]},
              'W':{'type':'dirichlet','val':zero, 'args':["x","y"]}, 
              'S':{'type':'dirichlet','val':zero, 'args':["x","y"]}}

boundary_v = {'N':{'type':'dirichlet' ,'val':zero, 'args':["x","y"]},
              'E':{'type':'dirichlet' ,'val':zero, 'args':["x","y"]},
              'W':{'type':'dirichlet' ,'val':zero, 'args':["x","y"]}, 
              'S':{'type':'dirichlet' ,'val':zero, 'args':["x","y"]}}

boundary_T = {'N':{'type':'dirichlet' ,'val':zero, 'args':["x","y"]},
              'E':{'type':'dirichlet' ,'val':zero, 'args':["x","y"]},
              'W':{'type':'dirichlet' ,'val':zero, 'args':["x","y"]}, 
              'S':{'type':'dirichlet' ,'val':zero, 'args':["x","y"]}}
boundaries = {'u':boundary_u,
              'v':boundary_v,
              'T':boundary_T,}
init = {'u':{'val':zero,    'args':["x","y"]},
        'v':{'val':zero,    'args':["x","y"]},
        'T':{'val':gaussian,'args':["x","y"]}}

s = convectiveSolver.convectiveSolver(init=init, boundaries=boundaries, mx=10, my=10, nx=5, ny=5, alpha=0.1, exact=True)
s.solve(dt=1e-4, Th=0.2)

sys = s.system
sys.computeSample('T')
x   = sys.properties["x_sample"]
y   = sys.properties["y_sample"]
T   = sys.properties["T_sample"]
T_exact = exact(x,y,0.2)
T_init = exact(x,y,0)
fig, axs = plt.subplots(2,2,figsize=(20,20))
axs[0][0].contourf(x,y,T,200);
axs[0][1].contourf(x,y,T_exact,200);
axs[1][0].contourf(x,y,T_init,200);
c = axs[1][1].contourf(x,y,(T-T_exact),200)
plt.colorbar(c)
plt.show()
```
