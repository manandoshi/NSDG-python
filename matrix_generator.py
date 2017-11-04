import numpy as np
import interpolation
import lobatto


def massMatrix(nodes, weights, exact=False):
    if exact:
        n = weights.size
        integration_nodes, integration_weights = lobatto.compute_nodes(n + 1)
        mat = interpolation.compute_matrix(nodes, integration_nodes)
        return mat.T.dot(mat * np.reshape(integration_weights, [-1, 1]))
    else:
        return np.diag(weights)


def massMatrix_2D(nodes_x, nodes_y, weights_x, weights_y, exact=False, exact_x=False, exact_y=False):
    nx = weights_x.size
    ny = weights_y.size
    mx = massMatrix(nodes_x, weights_x, exact or exact_x)
    my = massMatrix(nodes_y, weights_y, exact or exact_y)
    mx = np.concatenate([mx for i in xrange(ny)], 0)
    mx = np.concatenate([mx for i in xrange(ny)], 1)
    my = np.repeat(my, nx, 0)
    my = np.repeat(my, nx, 1)
    return mx * my


def derMatrix(nodes, weights):
    n = nodes.size
    mat = np.zeros([n, n])
    for i, node in enumerate(nodes):
        mat[i, :] = interpolation.dlagrangian(i, nodes) * weights[i]
    return mat


def derMatrix_2D(nodes_x, nodes_y, weights_x, weights_y, exact=True):
    nx = weights_x.size
    ny = weights_y.size

    dx = derMatrix(nodes_x, weights_x)
    mx = massMatrix(nodes_x, weights_x, exact)

    dy = derMatrix(nodes_y, weights_y)
    my = massMatrix(nodes_y, weights_y, exact)

    dx = np.concatenate([dx for i in xrange(ny)], 0)
    dx = np.concatenate([dx for i in xrange(ny)], 1)

    mx = np.concatenate([mx for i in xrange(ny)], 0)
    mx = np.concatenate([mx for i in xrange(ny)], 1)

    dy = np.repeat(dy, nx, 0)
    dy = np.repeat(dy, nx, 1)

    my = np.repeat(my, nx, 0)
    my = np.repeat(my, nx, 1)

    der_x = dx * my
    der_y = mx * dy

    return der_x.T, der_y.T


def fluxMatrix(nodes_x, nodes_y, weights_x, weights_y, exact=True):
    return massMatrix(nodes_x, weights_x, exact), massMatrix(nodes_y, weights_y, exact)
