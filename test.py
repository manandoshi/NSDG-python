
# coding: utf-8

# In[1]:

import system
import convectiveSolver
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
reload(convectiveSolver); reload(system)


# In[2]:

def one(args):
    x,y = args["x"], args["y"]
    return np.ones_like(x);

def zero(args):
    x,y = args["x"], args["y"]
    return np.zeros_like(x)

def gaussian(args):
    x,y = args["x"], args["y"]
    return np.exp(-np.sqrt(x**2+y**2)/4)
    

boundary_u = {'N':{'type':'dirichlet','val':zero, 'args':["x","y"]},
              'E':{'type':'dirichlet','val':zero, 'args':["x","y"]},
              'W':{'type':'dirichlet','val':zero, 'args':["x","y"]}, 
              'S':{'type':'dirichlet','val':zero, 'args':["x","y"]}}

boundary_v = {'N':{'type':'dirichlet','val':zero, 'args':["x","y"]},
              'E':{'type':'dirichlet','val':zero, 'args':["x","y"]},
              'W':{'type':'dirichlet','val':zero, 'args':["x","y"]}, 
              'S':{'type':'dirichlet','val':zero, 'args':["x","y"]}}

boundary_T = {'N':{'type':'dirichlet','val':zero, 'args':["x","y"]},
              'E':{'type':'dirichlet','val':zero, 'args':["x","y"]},
              'W':{'type':'dirichlet','val':one , 'args':["x","y"]}, 
              'S':{'type':'dirichlet','val':zero, 'args':["x","y"]}}
boundaries = {'u':boundary_u,
              'v':boundary_v,
              'T':boundary_T,}
init = {'u':{'val':zero, 'args':["x","y"]},
        'v':{'val':zero, 'args':["x","y"]},
        'T':{'val':gaussian, 'args':["x","y"]}}


# In[3]:

s = convectiveSolver.convectiveSolver(init=init, boundaries=boundaries, mx=10, my=10, nx=10, ny=15)


# In[4]:

#sys = s.system
#x   = sys.properties["x"]
#y   = sys.properties["y"]
#plt.figure(figsize=(10,10));plt.plot(x,y,'rx');plt.show()
#print sys.properties['T']


# In[5]:

s.solve()


# In[ ]:



