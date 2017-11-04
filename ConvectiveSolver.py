import system
import numpy as np


def q(mesh):
    return np.exp((mesh[0]**2 + mesh[1]**2) / 16)


def u(mesh):
    return 0


s = system(num_el_x=6, num_el_y=4)
