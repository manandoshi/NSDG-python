from scipy.special import legendre
from scipy.optimize import newton
import numpy as np


def compute_nodes(m):
    leg = legendre(m - 1)
    p = leg.deriv()
    p_dash = p.deriv()
    p_2dash = p_dash.deriv()
    roots = -1 * np.cos(np.pi * np.linspace(0, 1, m))

    for i, root in enumerate(roots[1:-1]):
        root = newton(p, root, fprime=p_dash, fprime2=p_2dash, maxiter=100)
        roots[i + 1] = root

    weights = 2 / (m * (m - 1) * leg(roots)**2)
    return roots, weights
